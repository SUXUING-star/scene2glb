import bpy

from utils import bake_utils, mmd_utils, scene_utils, texture_utils

DOUBLE_SIDED_KEYWORDS = (
    "hair",
    "kami",
    "skirt",
    "ribbon",
    "cloth",
    "dress",
    "cape",
    "lace",
    "frill",
    "leaf",
    "leaves",
    "feather",
    "wing",
    "sleeve",
    "collar",
    "coat",
    "scarf",
    "veil",
    "brow",
    "lash",
    "shadow",
    "flower",
    "petal",
)
SINGLE_SIDED_KEYWORDS = (
    "skin",
    "body",
    "face",
    "head",
    "eye",
    "iris",
    "pupil",
    "shoe",
    "boot",
    "sock",
    "metal",
    "armor",
    "weapon",
    "prop",
)


def _decide_backface_culling(
    mat_name: str, mmd_double_sided: bool, alpha_mode: str
) -> bool:
    low = mat_name.lower()
    if any(k in low for k in SINGLE_SIDED_KEYWORDS) and alpha_mode == "OPAQUE":
        return True
    if any(k in low for k in DOUBLE_SIDED_KEYWORDS) or alpha_mode != "OPAQUE":
        return False
    return not mmd_double_sided


def _apply_alpha_settings(mat, alpha_mode: str, cutoff: float):
    if hasattr(mat, "blend_method"):
        mat.blend_method = alpha_mode
    if hasattr(mat, "alpha_threshold"):
        mat.alpha_threshold = cutoff
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = (
            "CLIP"
            if alpha_mode == "CLIP"
            else ("NONE" if alpha_mode == "BLEND" else "OPAQUE")
        )
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED" if alpha_mode == "BLEND" else "DITHERED"


def _rebuild_standard_principled(
    mat, info, base_img, normal_img, emission_img, alpha_mode, double_sided, cutoff
):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (420, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (80, 0)
    scene_utils.set_principled_input(bsdf, ["Roughness"], info["roughness"])
    scene_utils.set_principled_input(bsdf, ["Metallic"], info["metallic"])
    scene_utils.set_principled_input(bsdf, ["Specular IOR Level", "Specular"], 0.25)
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    tint = info["tint_color"]
    if base_img is not None:
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.location = (-400, 120)
        tex_node.image = base_img
        nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

        if alpha_mode == "CLIP":
            math_node = nt.nodes.new("ShaderNodeMath")
            math_node.operation = "GREATER_THAN"
            math_node.location = (-130, -80)
            math_node.inputs[1].default_value = cutoff
            nt.links.new(tex_node.outputs["Alpha"], math_node.inputs[0])
            nt.links.new(math_node.outputs["Value"], bsdf.inputs["Alpha"])
        elif alpha_mode == "BLEND":
            nt.links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
    else:
        scene_utils.set_principled_input(
            bsdf, ["Base Color"], (tint[0], tint[1], tint[2], 1.0)
        )
        if alpha_mode == "BLEND":
            scene_utils.set_principled_input(bsdf, ["Alpha"], info["alpha"])

    if info["emission_strength"] > 0.001:
        scene_utils.set_principled_input(
            bsdf, ["Emission Strength"], min(info["emission_strength"], 10.0)
        )
        if emission_img is not None:
            em_node = nt.nodes.new("ShaderNodeTexImage")
            em_node.location = (-400, -180)
            em_node.image = emission_img
            em_sock = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
            if em_sock:
                nt.links.new(em_node.outputs["Color"], em_sock)
        else:
            scene_utils.set_principled_input(
                bsdf, ["Emission Color", "Emission"], info["emission_color"]
            )

    if normal_img is not None:
        try:
            normal_img.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        n_tex = nt.nodes.new("ShaderNodeTexImage")
        n_tex.location = (-420, -420)
        n_tex.image = normal_img
        n_map = nt.nodes.new("ShaderNodeNormalMap")
        n_map.location = (-160, -420)
        nt.links.new(n_tex.outputs["Color"], n_map.inputs["Color"])
        nt.links.new(n_map.outputs["Normal"], bsdf.inputs["Normal"])

    mat.use_backface_culling = not double_sided
    _apply_alpha_settings(mat, alpha_mode, cutoff)


def run(config: dict, root_obj: bpy.types.Object, cache_dir: str):
    print("[Step 03] 解析贴图并对实时/程序化材质执行 Cycles Bake...")
    blend_dir = scene_utils.get_blend_dir()
    texture_utils.resolve_packed_or_external_images(blend_dir, cache_dir)

    max_tex_size = int(config.get("max_tex_size", 2048))
    bake_res = int(config.get("bake_tex_size", 1024))
    bake_mode = str(
        config.get("bake_mode", "auto")
    ).lower()  # "auto" | "all" | "combined"

    meshes = [o for o in scene_utils.iter_hierarchy(root_obj) if o.type == "MESH"]
    has_armature = any(
        o.type == "ARMATURE" for o in scene_utils.iter_hierarchy(root_obj)
    )

    # 静态场景烘焙前先应用 Mirror/Array/GeometryNodes 等修改器，确保 UV 与几何体对应
    if not has_armature:
        for obj in meshes:
            if len(obj.modifiers) > 0 and obj.data.shape_keys is None:
                scene_utils.ensure_object_visible_and_linked(obj)
                with bpy.context.temp_override(object=obj, active_object=obj):
                    for mod in list(obj.modifiers):
                        try:
                            bpy.ops.object.modifier_apply(modifier=mod.name)
                        except Exception:
                            pass
                if obj.animation_data and obj.animation_data.drivers:
                    for drv in list(obj.animation_data.drivers):
                        if "modifiers[" in drv.data_path:
                            obj.animation_data.drivers.remove(drv)

    baked_mat_names = set()
    processed_mats = {}
    dedup_map = {}

    for obj in meshes:
        mats_to_bake = []
        for slot in obj.material_slots:
            m = slot.material
            if m and m.name not in processed_mats and m.name not in baked_mat_names:
                if bake_utils.needs_cycles_bake(m, bake_mode=bake_mode):
                    mats_to_bake.append(m)

        # 去重保持顺序
        mats_to_bake = list(dict.fromkeys(mats_to_bake))
        bake_results = {}
        if mats_to_bake:
            print(
                f"  [Cycles Bake] {obj.name}: 正在烘焙 {len(mats_to_bake)} 个实时材质 ({bake_res}px)..."
            )
            bake_results = bake_utils.bake_object_materials(
                obj,
                target_mats=mats_to_bake,
                cache_dir=cache_dir,
                res=bake_res,
                bake_mode=bake_mode,
            )
            for m in mats_to_bake:
                baked_mat_names.add(m.name)

        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue

            if mat.name in processed_mats:
                slot.material = processed_mats[mat.name]
                continue

            info = mmd_utils.extract_mmd_material_info(mat)

            if mat.name in bake_results:
                bres = bake_results[mat.name]
                final_img = bres["base_img"]
                emission_img = bres["emit_img"] or (
                    final_img if bres["is_pure_emit"] else None
                )
                if bres["has_emission"] and info["emission_strength"] <= 0.001:
                    info["emission_strength"] = 1.0
                normal_img = None
            else:
                raw_img = info["base_image"]
                alpha_mask_img = info["alpha_image"]
                tint = info["tint_color"]
                alpha_val = info["alpha"]

                final_img = None
                if raw_img is not None:
                    final_img = texture_utils.bake_tint_and_convert_png(
                        raw_img,
                        tint_rgba=tint,
                        material_alpha=alpha_val,
                        cache_dir=cache_dir,
                        max_size=max_tex_size,
                        alpha_mask_img=alpha_mask_img,
                    )
                elif alpha_mask_img is not None:
                    final_img = texture_utils.bake_tint_and_convert_png(
                        alpha_mask_img,
                        tint_rgba=tint,
                        material_alpha=alpha_val,
                        cache_dir=cache_dir,
                        max_size=max_tex_size,
                        alpha_mask_img=alpha_mask_img,
                    )

                normal_img = None
                if info["normal_image"] is not None:
                    normal_img = texture_utils.bake_tint_and_convert_png(
                        info["normal_image"],
                        tint_rgba=(1.0, 1.0, 1.0, 1.0),
                        material_alpha=1.0,
                        cache_dir=cache_dir,
                        max_size=max_tex_size,
                        suffix="_nrm",
                    )

                emission_img = None
                if info["emission_image"] is not None:
                    emission_img = (
                        final_img
                        if info["emission_image"] == raw_img and final_img is not None
                        else texture_utils.bake_tint_and_convert_png(
                            info["emission_image"],
                            tint_rgba=(1.0, 1.0, 1.0, 1.0),
                            material_alpha=1.0,
                            cache_dir=cache_dir,
                            max_size=max_tex_size,
                            suffix="_em",
                        )
                    )

            alpha_mode, cutoff = texture_utils.classify_alpha_mode(
                final_img, info["alpha"]
            )
            use_culling = _decide_backface_culling(
                mat.name, info["double_sided"], alpha_mode
            )
            double_sided = not use_culling

            tint = info["tint_color"]
            img_key = (
                final_img.name
                if final_img
                else f"COLOR_{round(tint[0], 3)}_{round(tint[1], 3)}_{round(tint[2], 3)}"
            )
            sig = (
                img_key,
                normal_img.name if normal_img else "",
                round(info["emission_strength"], 2),
                alpha_mode,
                double_sided,
                round(info["alpha"], 2),
            )

            if sig in dedup_map:
                canonical = dedup_map[sig]
                slot.material = canonical
                processed_mats[mat.name] = canonical
                continue

            mat.name = scene_utils.sanitize_english_name(
                mat.name, f"Mat_{len(dedup_map):02d}"
            )
            _rebuild_standard_principled(
                mat=mat,
                info=info,
                base_img=final_img,
                normal_img=normal_img,
                emission_img=emission_img,
                alpha_mode=alpha_mode,
                double_sided=double_sided,
                cutoff=cutoff,
            )
            dedup_map[sig] = mat
            processed_mats[mat.name] = mat

    print(f"  [完成] 材质整合与烘焙完成，共保留 {len(dedup_map)} 个标准 PBR 材质")
