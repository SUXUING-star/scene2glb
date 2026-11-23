import math

import bmesh
import bpy

from utils import scene_utils, texture_utils

NON_COLLIDER_KEYWORDS = (
    "sunlight",
    "sun_ray",
    "sunray",
    "godray",
    "god_ray",
    "lightshaft",
    "light_shaft",
    "lightbeam",
    "light_beam",
    "sunbeam",
    "sun_beam",
    "flare",
    "halo",
    "glow",
    "bloom",
    "fog",
    "mist",
    "cloud",
    "smoke",
    "dust",
    "spark",
    "particle",
    "vfx",
    "skybox",
    "skydome",
    "sky_dome",
    "aura",
)


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


def _is_non_collider_material(mat: bpy.types.Material, obj_name: str) -> bool:
    if mat is None:
        return True

    combined_name = f"{obj_name} {mat.name}".lower()
    if any(k in combined_name for k in NON_COLLIDER_KEYWORDS):
        return True

    if getattr(mat, "blend_method", "OPAQUE") == "BLEND":
        return True

    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                em_str_inp = node.inputs.get("Emission Strength")
                em_str = float(em_str_inp.default_value) if em_str_inp else 0.0
                if em_str > 0.01:
                    bc_inp = node.inputs.get("Base Color")
                    has_base_tex = bool(bc_inp and bc_inp.is_linked)
                    if not has_base_tex and bc_inp:
                        c = bc_inp.default_value
                        if max(c[0], c[1], c[2]) < 0.05:
                            return True
                    if any(
                        k in combined_name
                        for k in ("sun", "ray", "beam", "light", "fx", "eff")
                    ):
                        return True
                break

    return False


def _strip_non_collider_geometry(obj: bpy.types.Object) -> tuple[bool, bool]:
    """
    返回值: (should_delete_object, stripped_any_faces)
    """
    if obj.type != "MESH" or not obj.data or len(obj.data.polygons) == 0:
        return True, False

    if len(obj.material_slots) == 0:
        name_low = obj.name.lower()
        return any(k in name_low for k in NON_COLLIDER_KEYWORDS), False

    non_collider_slots = {
        idx
        for idx, slot in enumerate(obj.material_slots)
        if _is_non_collider_material(slot.material, obj.name)
    }

    if not non_collider_slots:
        return False, False

    if len(non_collider_slots) >= len(obj.material_slots):
        return True, True

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    faces_to_del = [f for f in bm.faces if f.material_index in non_collider_slots]
    if faces_to_del:
        bmesh.ops.delete(bm, geom=faces_to_del, context="FACES")
        bm.faces.ensure_lookup_table()

    if len(bm.faces) == 0:
        bm.free()
        return True, True

    idx_map = {}
    new_i = 0
    for old_i in range(len(mesh.materials)):
        if old_i not in non_collider_slots:
            idx_map[old_i] = new_i
            new_i += 1
    for f in bm.faces:
        f.material_index = idx_map.get(f.material_index, 0)

    bm.to_mesh(mesh)
    bm.free()

    for idx in sorted(non_collider_slots, reverse=True):
        mesh.materials.pop(index=idx)

    mesh.update()
    return False, True


def _collect_protected_verts(mesh: bpy.types.Mesh, sharp_angle_deg: float = 15.0):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    bm.normal_update()

    protect_indices = set()
    sharp_rad = math.radians(sharp_angle_deg)
    for e in bm.edges:
        if e.is_boundary or len(e.link_faces) != 2:
            protect_indices.add(e.verts[0].index)
            protect_indices.add(e.verts[1].index)
        else:
            f0, f1 = e.link_faces[0], e.link_faces[1]
            if f0.normal.length > 0 and f1.normal.length > 0:
                if f0.normal.angle(f1.normal) > sharp_rad:
                    protect_indices.add(e.verts[0].index)
                    protect_indices.add(e.verts[1].index)
    bm.free()
    return list(protect_indices)


def _apply_lowpoly_decimate(
    obj,
    decimate_ratio: float,
    clear_shapekeys: bool = True,
    collapse_min_faces: int = 1000,
    sharp_angle_deg: float = 15.0,
):
    if (
        not (0.05 <= decimate_ratio < 0.99)
        or len(obj.data.polygons) <= collapse_min_faces
    ):
        return

    if obj.data.shape_keys is not None:
        if clear_shapekeys:
            obj.active_shape_key_index = 0
            obj.shape_key_clear()
        else:
            print(f"  [跳过减面] {obj.name} 含有形态键且设置了保留形态键")
            return

    protect_indices = _collect_protected_verts(
        obj.data, sharp_angle_deg=sharp_angle_deg
    )
    total_verts = len(obj.data.vertices)
    if total_verts == 0 or len(protect_indices) >= total_verts * 0.85:
        return

    vg = None
    if protect_indices:
        vg = obj.vertex_groups.new(name="__LowPoly_Protect")
        vg.add(protect_indices, 1.0, "REPLACE")

    mod = obj.modifiers.new(name="__LowPoly_Decimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = decimate_ratio
    mod.use_collapse_triangulate = True
    if vg is not None and hasattr(mod, "vertex_group"):
        mod.vertex_group = vg.name
        mod.invert_vertex_group = True
        if hasattr(mod, "vertex_group_factor"):
            mod.vertex_group_factor = 1.0

    if len(obj.modifiers) > 1:
        obj.modifiers.move(len(obj.modifiers) - 1, 0)

    scene_utils.ensure_object_visible_and_linked(obj)
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_apply(modifier=mod.name)

    if vg is not None and vg.name in obj.vertex_groups:
        obj.vertex_groups.remove(vg)


def run(config: dict, root_obj: bpy.types.Object, cache_dir: str):
    print("[Step 04] 生成低模及低模材质...")
    low_tex_size = int(config.get("lowpoly_tex_size", 512))
    decimate_ratio = float(config.get("lowpoly_decimate_ratio", 0.6))
    clear_shapekeys = bool(config.get("lowpoly_clear_shapekeys", True))
    collapse_min_faces = int(config.get("lowpoly_collapse_min_faces", 1000))
    sharp_angle_deg = float(config.get("lowpoly_protect_angle", 15.0))
    strip_non_collider = bool(config.get("lowpoly_strip_non_collider", True))

    low_root = _duplicate_hierarchy(root_obj, f"{root_obj.name}_LowPoly")
    low_mat_cache = {}
    objects_to_delete = []
    stripped_face_objs = 0

    for obj in list(scene_utils.iter_hierarchy(low_root)):
        if obj.type != "MESH":
            continue

        # 0. 剔除绘制阳光/光束/雾效/天空盒/BLEND半透明等非碰撞几何体
        if strip_non_collider:
            should_del, stripped_faces = _strip_non_collider_geometry(obj)
            if should_del:
                objects_to_delete.append(obj)
                continue
            if stripped_faces:
                stripped_face_objs += 1

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
                if node.type == "BSDF_PRINCIPLED":
                    bc_inp = node.inputs.get("Base Color")
                    if bc_inp:
                        base_color_val = tuple(bc_inp.default_value)
                        if (
                            bc_inp.is_linked
                            and bc_inp.links[0].from_node.type == "TEX_IMAGE"
                        ):
                            base_img = bc_inp.links[0].from_node.image
                    alpha_inp = node.inputs.get("Alpha")
                    has_alpha_link = bool(alpha_inp and alpha_inp.is_linked)
                    break

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

        # 2. 低模减面（锁定边界与折角顶点，只对高面数网格做 COLLAPSE）
        _apply_lowpoly_decimate(
            obj,
            decimate_ratio=decimate_ratio,
            clear_shapekeys=clear_shapekeys,
            collapse_min_faces=collapse_min_faces,
            sharp_angle_deg=sharp_angle_deg,
        )

    for obj in objects_to_delete:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    if objects_to_delete or stripped_face_objs:
        print(
            f"  [碰撞过滤] 已剔除 {len(objects_to_delete)} 个非碰撞光效/透明物体，清理 {stripped_face_objs} 个物体的非碰撞面片"
        )

    print(
        f"  [完成] 低模节点: {low_root.name}，生成 {len(low_mat_cache)} 个低模材质 (贴图上限 {low_tex_size}px)"
    )
    return low_root
