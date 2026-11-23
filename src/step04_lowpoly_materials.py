import bpy

from utils import scene_utils, texture_utils


def _duplicate_hierarchy(root_obj, new_name: str):
    scene_utils.select_hierarchy(root_obj, include_root=True)
    bpy.ops.object.duplicate(linked=False)

    dup_roots = [o for o in bpy.context.selected_objects if o.parent == root_obj.parent]
    dup_root = dup_roots[0] if dup_roots else bpy.context.active_object
    dup_root.name = new_name
    dup_root["is_lowpoly"] = True

    for obj in scene_utils.iter_hierarchy(dup_root):
        obj["is_lowpoly"] = True
        if obj.type == "MESH" and obj.data and obj.data.users > 1:
            obj.data = obj.data.copy()
    return dup_root


def _apply_lowpoly_decimate(obj, decimate_ratio: float, clear_shapekeys: bool = True):
    if not (0.05 <= decimate_ratio < 0.99) or len(obj.data.polygons) <= 12:
        return

    # 若存在形态键（Shape Keys），先在低模副本上清理形态键再减面（不影响原高模）
    if obj.data.shape_keys is not None:
        if clear_shapekeys:
            obj.active_shape_key_index = 0
            obj.shape_key_clear()
        else:
            print(f"  [跳过减面] {obj.name} 含有形态键且设置了保留形态键")
            return

    mod = obj.modifiers.new(name="__LowPoly_Decimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = decimate_ratio

    # 移动到修改器列表首位（在 Armature 蒙皮之前应用），避免顺序警告
    if len(obj.modifiers) > 1:
        obj.modifiers.move(len(obj.modifiers) - 1, 0)

    scene_utils.ensure_object_visible_and_linked(obj)
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def run(config: dict, root_obj: bpy.types.Object, cache_dir: str):
    print("[Step 04] 生成低模及低模材质...")
    low_tex_size = int(config.get("lowpoly_tex_size", 512))
    decimate_ratio = float(config.get("lowpoly_decimate_ratio", 0.6))
    clear_shapekeys = bool(config.get("lowpoly_clear_shapekeys", True))

    low_root = _duplicate_hierarchy(root_obj, f"{root_obj.name}_LowPoly")
    low_mat_cache = {}

    for obj in scene_utils.iter_hierarchy(low_root):
        if obj.type != "MESH":
            continue

        # 1. 自动生成简化低模材质（贴图降采样 + 轻量 PBR 参数 + ALPHA_CLIP）
        for slot in obj.material_slots:
            src_mat = slot.material
            if src_mat is None:
                continue

            if src_mat.name in low_mat_cache:
                slot.material = low_mat_cache[src_mat.name]
                continue

            low_mat = src_mat.copy()
            low_mat.name = f"{src_mat.name}_Low"
            low_mat.use_nodes = True
            nt = low_mat.node_tree

            base_img = None
            base_color_val = (0.8, 0.8, 0.8, 1.0)
            has_alpha_link = False
            for node in nt.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    base_img = node.image
                elif node.type == "BSDF_PRINCIPLED":
                    base_color_val = tuple(node.inputs["Base Color"].default_value)
                    has_alpha_link = node.inputs["Alpha"].is_linked

            nt.nodes.clear()
            out_node = nt.nodes.new("ShaderNodeOutputMaterial")
            out_node.location = (340, 0)

            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            scene_utils.set_principled_input(bsdf, ["Roughness"], 0.9)
            scene_utils.set_principled_input(bsdf, ["Metallic"], 0.0)
            scene_utils.set_principled_input(
                bsdf, ["Specular IOR Level", "Specular"], 0.1
            )
            nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

            if base_img is not None:
                low_img = texture_utils.bake_tint_and_convert_png(
                    base_img,
                    tint_rgba=(1.0, 1.0, 1.0, 1.0),
                    material_alpha=1.0,
                    cache_dir=cache_dir,
                    max_size=low_tex_size,
                    suffix="_low",
                )
                tex_node = nt.nodes.new("ShaderNodeTexImage")
                tex_node.location = (-360, 0)
                tex_node.image = low_img
                nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
                if has_alpha_link:
                    math_node = nt.nodes.new("ShaderNodeMath")
                    math_node.operation = "GREATER_THAN"
                    math_node.location = (-130, -140)
                    math_node.inputs[1].default_value = 0.5
                    nt.links.new(tex_node.outputs["Alpha"], math_node.inputs[0])
                    nt.links.new(math_node.outputs["Value"], bsdf.inputs["Alpha"])
            else:
                scene_utils.set_principled_input(bsdf, ["Base Color"], base_color_val)

            if hasattr(low_mat, "blend_method") and low_mat.blend_method == "BLEND":
                low_mat.blend_method = "CLIP"
                low_mat.alpha_threshold = 0.5
            if hasattr(low_mat, "surface_render_method"):
                low_mat.surface_render_method = "DITHERED"

            low_mat_cache[src_mat.name] = low_mat
            slot.material = low_mat

        # 2. 低模减面（自动清理副本形态键后置顶应用）
        _apply_lowpoly_decimate(obj, decimate_ratio, clear_shapekeys=clear_shapekeys)

    print(
        f"  [完成] 低模节点: {low_root.name}，生成 {len(low_mat_cache)} 个低模材质 (贴图上限 {low_tex_size}px)"
    )
    return low_root
