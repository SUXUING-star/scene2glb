import os

import bpy

from utils import scene_utils


def _select_multiple_hierarchies(roots):
    scene_utils.ensure_object_mode()
    scene_utils.enable_all_collections_in_view_layer()
    bpy.ops.object.select_all(action="DESELECT")

    for r in roots:
        if r is None:
            continue
        for obj in scene_utils.iter_hierarchy(r):
            scene_utils.ensure_object_visible_and_linked(obj)
            if obj.name in bpy.context.view_layer.objects:
                obj.select_set(True)

    if roots and roots[0] and roots[0].name in bpy.context.view_layer.objects:
        bpy.context.view_layer.objects.active = roots[0]


def _export_selected_to_glb(roots, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    _select_multiple_hierarchies(roots)

    selected_meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not selected_meshes:
        raise RuntimeError(f"导出前校验失败：没有选中的 Mesh，终止导出！({filepath})")

    has_armature = any(o.type == "ARMATURE" for o in bpy.context.selected_objects)

    desired_kwargs = {
        "filepath": filepath,
        "export_format": "GLB",
        "use_selection": True,
        "export_yup": True,
        "export_apply": not has_armature,
        "export_texcoords": True,
        "export_normals": True,
        "export_materials": "EXPORT",
        "export_image_format": "AUTO",
        "export_skins": has_armature,
        "export_animations": has_armature,
        "export_def_bones": True,
        "export_extras": True,
    }

    valid_props = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    safe_kwargs = {k: v for k, v in desired_kwargs.items() if k in valid_props}

    bpy.ops.export_scene.gltf(**safe_kwargs)
    size_kb = os.path.getsize(filepath) / 1024.0 if os.path.exists(filepath) else 0.0
    print(
        f"  [导出成功] {len(selected_meshes)} 个网格 ({size_kb:.1f} KB) -> {filepath}"
    )


def run(
    config: dict,
    root_obj: bpy.types.Object,
    low_root: bpy.types.Object,
    output_dir: str,
):
    print("[Step 05] 导出标准 GLB...")
    os.makedirs(output_dir, exist_ok=True)

    pack_lowpoly = bool(config.get("pack_lowpoly", True))

    if pack_lowpoly:
        # 打包进同一个 GLB 文件（不拆成两个模型）
        packed_glb = os.path.join(output_dir, f"{root_obj.name}.glb")
        print(f"  模式: pack_lowpoly=True -> 标准模型与低模打包至同一文件")
        _export_selected_to_glb([root_obj, low_root], packed_glb)
    else:
        # 分别导出两个 GLB 文件
        print(f"  模式: pack_lowpoly=False -> 分别导出标准 GLB 与低模 GLB")
        main_glb = os.path.join(output_dir, f"{root_obj.name}.glb")
        low_glb = os.path.join(output_dir, f"{root_obj.name}_lowpoly.glb")
        _export_selected_to_glb([root_obj], main_glb)
        _export_selected_to_glb([low_root], low_glb)

    if low_root is not None and not config.get("keep_lowpoly_in_scene", False):
        scene_utils.delete_object_tree(low_root)

    scene_utils.purge_unused_data()
