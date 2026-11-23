import bpy

from utils import mmd_utils, scene_utils, texture_utils

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
    mat, base_img, tint_rgba, alpha_val, alpha_mode, double_sided, cutoff
):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (60, 0)
    scene_utils.set_principled_input(bsdf, ["Roughness"], 0.75)
    scene_utils.set_principled_input(bsdf, ["Metallic"], 0.0)
    scene_utils.set_principled_input(bsdf, ["Specular IOR Level", "Specular"], 0.25)

    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    if base_img is not None:
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.location = (-380, 0)
        tex_node.image = base_img
        nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

        if alpha_mode == "CLIP":
            # 使用 GREATER_THAN 节点，确保 Blender 4.2+/5.x 稳定导出 glTF MASK 模式
            math_node = nt.nodes.new("ShaderNodeMath")
            math_node.operation = "GREATER_THAN"
            math_node.location = (-130, -140)
            math_node.inputs[1].default_value = cutoff
            nt.links.new(tex_node.outputs["Alpha"], math_node.inputs[0])
            nt.links.new(math_node.outputs["Value"], bsdf.inputs["Alpha"])
        elif alpha_mode == "BLEND":
            nt.links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
    else:
        scene_utils.set_principled_input(
            bsdf, ["Base Color"], (tint_rgba[0], tint_rgba[1], tint_rgba[2], 1.0)
        )
        if alpha_mode == "BLEND":
            scene_utils.set_principled_input(bsdf, ["Alpha"], alpha_val)

    mat.use_backface_culling = not double_sided
    _apply_alpha_settings(mat, alpha_mode, cutoff)


def run(config: dict, root_obj: bpy.types.Object, cache_dir: str):
    print("[Step 03] 自动扫描贴图并整合为标准 glTF Principled BSDF 材质...")
    blend_dir = scene_utils.get_blend_dir()
    texture_utils.relink_missing_textures(blend_dir)

    max_tex_size = int(config.get("max_tex_size", 2048))
    meshes = [o for o in scene_utils.iter_hierarchy(root_obj) if o.type == "MESH"]
    processed_mats = {}
    dedup_map = {}

    for obj in meshes:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue

            if mat.name in processed_mats:
                slot.material = processed_mats[mat.name]
                continue

            info = mmd_utils.extract_mmd_material_info(mat)
            raw_img = info["base_image"]
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
                )

            alpha_mode, cutoff = texture_utils.classify_alpha_mode(final_img, alpha_val)
            use_culling = _decide_backface_culling(
                mat.name, info["double_sided"], alpha_mode
            )
            double_sided = not use_culling

            img_key = (
                final_img.name
                if final_img
                else f"COLOR_{round(tint[0], 3)}_{round(tint[1], 3)}_{round(tint[2], 3)}"
            )
            sig = (img_key, alpha_mode, double_sided, round(alpha_val, 2))

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
                base_img=final_img,
                tint_rgba=tint,
                alpha_val=alpha_val,
                alpha_mode=alpha_mode,
                double_sided=double_sided,
                cutoff=cutoff,
            )
            dedup_map[sig] = mat
            processed_mats[mat.name] = mat

    print(f"  [完成] 材质整合完成，共保留 {len(dedup_map)} 个去重后的标准 PBR 材质")
