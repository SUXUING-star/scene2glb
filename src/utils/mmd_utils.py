import bpy

from . import scene_utils

MMD_PHYSICS_TYPES = {
    "RIGID_BODY",
    "JOINT",
    "RIGID_GRP_EMPTY",
    "JOINT_GRP_EMPTY",
    "TEMPORARY_GRP_EMPTY",
}


def is_mmd_root(obj) -> bool:
    return getattr(obj, "mmd_type", "NONE") == "ROOT"


def find_mmd_roots():
    return [obj for obj in bpy.context.scene.objects if is_mmd_root(obj)]


def cleanup_mmd_physics_and_dummies(root_obj):
    """清理 MMD 导入生成的刚体、关节、临时空物体及无用物理约束"""
    to_remove = []
    for obj in scene_utils.iter_hierarchy(root_obj):
        mmd_type = getattr(obj, "mmd_type", "NONE")
        if mmd_type in MMD_PHYSICS_TYPES:
            to_remove.append(obj)
        elif obj.type == "MESH" and (
            "rigidbodies" in obj.name.lower()
            or "joints" in obj.name.lower()
            or obj.name.startswith("mmd_rigid")
            or obj.name.startswith("mmd_joint")
        ):
            to_remove.append(obj)

    for obj in set(to_remove):
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    # 清理骨骼上的 mmd_tools 刚体驱动约束，防止导出后骨骼被锁或异常偏移
    for obj in scene_utils.iter_hierarchy(root_obj):
        if obj.type == "ARMATURE" and obj.pose:
            for pbone in obj.pose.bones:
                for con in list(pbone.constraints):
                    if "mmd_" in con.name.lower():
                        pbone.constraints.remove(con)


def extract_mmd_material_info(mat):
    """从 mmd_tools 材质节点/属性中提取基础贴图、染色、Alpha 与双面标记"""
    info = {
        "base_image": None,
        "tint_color": (1.0, 1.0, 1.0, 1.0),
        "alpha": 1.0,
        "double_sided": not mat.use_backface_culling,
        "is_edge": mat.name.startswith("mmd_edge"),
    }

    mmd_mat = getattr(mat, "mmd_material", None)
    if mmd_mat:
        dc = mmd_mat.diffuse_color
        info["tint_color"] = (float(dc[0]), float(dc[1]), float(dc[2]), 1.0)
        info["alpha"] = float(getattr(mmd_mat, "alpha", 1.0))
        info["double_sided"] = bool(
            getattr(mmd_mat, "is_double_sided", info["double_sided"])
        )

    if mat.use_nodes and mat.node_tree:
        nodes = mat.node_tree.nodes
        # 优先查找 mmd_base_tex
        tex_node = nodes.get("mmd_base_tex")
        if tex_node and tex_node.type == "TEX_IMAGE" and tex_node.image:
            info["base_image"] = tex_node.image
        else:
            # 从 mmd_shader 或 Principled BSDF 追溯图像
            for node in nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    name_low = (node.name + " " + node.image.name).lower()
                    if (
                        "toon" not in name_low
                        and "sphere" not in name_low
                        and "sph" not in name_low
                    ):
                        info["base_image"] = node.image
                        break
    return info
