import os

import bpy

from utils import mmd_utils, scene_utils


def run(config: dict) -> bpy.types.Object:
    print("[Step 01] 准备场景、打包多个模型并标准化英文根节点...")
    scene_utils.ensure_object_mode()
    scene_utils.enable_all_collections_in_view_layer()

    jp_name = (config.get("jp_name") or "").strip()
    en_name = (config.get("en_name") or "").strip()

    if not en_name:
        if bpy.data.filepath:
            base = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
            en_name = scene_utils.sanitize_english_name(base, "SceneRoot")
        else:
            en_name = "SceneRoot"
    else:
        en_name = scene_utils.sanitize_english_name(en_name, "CharacterRoot")

    if jp_name:
        for obj in bpy.data.objects:
            if jp_name in obj.name:
                obj.name = obj.name.replace(jp_name, en_name)

    # 先确保所有 MESH / ARMATURE 在当前场景中可用
    for obj in list(bpy.data.objects):
        if obj.type in {"MESH", "ARMATURE"}:
            scene_utils.ensure_object_visible_and_linked(obj)

    # 查找或新建唯一英文根节点
    master_root = bpy.data.objects.get(en_name)
    if master_root is not None and master_root.type != "EMPTY":
        # 如果已存在同名非空物体，先给旧物体让出名字
        master_root.name = f"{en_name}_Obj"
        master_root = None

    if master_root is None:
        master_root = bpy.data.objects.new(en_name, None)
        master_root.empty_display_type = "PLAIN_AXES"
        bpy.context.scene.collection.objects.link(master_root)
    scene_utils.ensure_object_visible_and_linked(master_root)

    # 1. 处理所有 MMD (PMX) 根节点
    mmd_roots = mmd_utils.find_mmd_roots()
    if mmd_roots:
        print(f"  检测到 {len(mmd_roots)} 个 PMX(MMD) 根节点，正在清理物理并打包...")
        for idx, mmd_root in enumerate(mmd_roots):
            mmd_utils.cleanup_mmd_physics_and_dummies(mmd_root)
            for child in list(mmd_root.children):
                scene_utils.reparent_keep_world(child, master_root)
                if child.type == "ARMATURE":
                    child.name = (
                        f"{en_name}_Armature"
                        if len(mmd_roots) == 1
                        else f"{en_name}_Armature_{idx + 1:02d}"
                    )
            if mmd_root != master_root and mmd_root.name in bpy.data.objects:
                bpy.data.objects.remove(mmd_root, do_unlink=True)

    # 2. 不管是否存在 mmd_roots，把场景中其余所有顶层可渲染对象也一并收入 master_root
    ignore_types = {"CAMERA", "LIGHT", "SPEAKER", "LIGHT_PROBE"}
    top_objects = [
        obj
        for obj in list(bpy.context.scene.objects)
        if obj.parent is None
        and obj != master_root
        and obj.type not in ignore_types
        and getattr(obj, "mmd_type", "NONE") not in mmd_utils.MMD_PHYSICS_TYPES
    ]
    for obj in top_objects:
        scene_utils.reparent_keep_world(obj, master_root)

    # 3. 统一将子级命名为英文并去重
    used_names = {master_root.name}
    mesh_count = 0
    armature_count = 0
    for idx, obj in enumerate(scene_utils.iter_hierarchy(master_root)):
        if obj == master_root:
            continue
        base_en = scene_utils.sanitize_english_name(
            obj.name, f"{en_name}_{obj.type.lower()}_{idx:02d}"
        )
        candidate = base_en
        counter = 1
        while candidate in used_names:
            candidate = f"{base_en}_{counter:02d}"
            counter += 1
        obj.name = candidate
        used_names.add(candidate)

        if obj.type == "MESH" and obj.data:
            obj.data.name = f"{obj.name}_Data"
            mesh_count += 1
        elif obj.type == "ARMATURE":
            armature_count += 1

    print(
        f"  [完成] 统一英文根节点: {master_root.name} (Mesh: {mesh_count}, Armature: {armature_count})"
    )
    if mesh_count == 0:
        raise RuntimeError(
            f"根节点 {master_root.name} 下没有检测到任何 Mesh！当前场景对象数: {len(bpy.data.objects)}"
        )
    return master_root
