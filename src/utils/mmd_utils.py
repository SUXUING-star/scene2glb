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

    for obj in scene_utils.iter_hierarchy(root_obj):
        if obj.type == "ARMATURE" and obj.pose:
            for pbone in obj.pose.bones:
                for con in list(pbone.constraints):
                    if "mmd_" in con.name.lower():
                        pbone.constraints.remove(con)


def _trace_first_image_from_socket(socket, visited=None):
    """沿着节点连线向上追溯（支持穿透 ShaderNodeGroup / Reroute / Mix）找到真正连接的图像"""
    if socket is None or not socket.is_linked:
        return None
    if visited is None:
        visited = set()

    for link in socket.links:
        node = link.from_node
        if node in visited:
            continue
        visited.add(node)

        if node.type == "TEX_IMAGE" and node.image:
            name_low = (node.name + " " + node.image.name).lower()
            if (
                "toon" not in name_low
                and "sphere" not in name_low
                and "sph" not in name_low
            ):
                return node.image

        if node.type == "GROUP" and node.node_tree:
            # 穿透节点组查找输出对应的内部输入
            for gnode in node.node_tree.nodes:
                if gnode.type == "GROUP_OUTPUT":
                    for inp in gnode.inputs:
                        if inp.name == link.from_socket.name and inp.is_linked:
                            res = _trace_first_image_from_socket(inp, visited)
                            if res:
                                return res
            # 兜底：节点组外层输入连线
            for inp in node.inputs:
                res = _trace_first_image_from_socket(inp, visited)
                if res:
                    return res
            # 兜底：节点组内部任意非 toon/sphere 的图像
            for gnode in node.node_tree.nodes:
                if gnode.type == "TEX_IMAGE" and gnode.image:
                    name_low = (gnode.name + " " + gnode.image.name).lower()
                    if (
                        "toon" not in name_low
                        and "sphere" not in name_low
                        and "sph" not in name_low
                    ):
                        return gnode.image

        for inp in node.inputs:
            if inp.type in {"RGBA", "VALUE", "VECTOR", "SHADER"} and inp.is_linked:
                res = _trace_first_image_from_socket(inp, visited)
                if res:
                    return res
    return None


def _find_nodes_recursive(node_tree, visited_trees=None):
    if node_tree is None:
        return
    if visited_trees is None:
        visited_trees = set()
    if node_tree in visited_trees:
        return
    visited_trees.add(node_tree)

    for node in node_tree.nodes:
        yield node
        if node.type == "GROUP" and node.node_tree:
            yield from _find_nodes_recursive(node.node_tree, visited_trees)


def extract_mmd_material_info(mat):
    """
    统一提取 MMD 材质与原生 Blender 材质信息（含 Principled / Emission / NodeGroup）
    """
    info = {
        "base_image": None,
        "alpha_image": None,
        "normal_image": None,
        "emission_image": None,
        "tint_color": (1.0, 1.0, 1.0, 1.0),
        "alpha": 1.0,
        "roughness": 0.75,
        "metallic": 0.0,
        "emission_color": (0.0, 0.0, 0.0, 1.0),
        "emission_strength": 0.0,
        "double_sided": not mat.use_backface_culling,
        "is_edge": mat.name.lower().startswith("mmd_edge"),
    }

    mmd_mat = getattr(mat, "mmd_material", None)
    if mmd_mat:
        dc = mmd_mat.diffuse_color
        info["tint_color"] = (float(dc[0]), float(dc[1]), float(dc[2]), 1.0)
        info["alpha"] = float(getattr(mmd_mat, "alpha", 1.0))
        info["double_sided"] = bool(
            getattr(mmd_mat, "is_double_sided", info["double_sided"])
        )

    if not mat.use_nodes or not mat.node_tree:
        if hasattr(mat, "diffuse_color") and not mmd_mat:
            dc = mat.diffuse_color
            info["tint_color"] = (float(dc[0]), float(dc[1]), float(dc[2]), 1.0)
            info["alpha"] = float(dc[3])
        return info

    nodes = mat.node_tree.nodes

    # 1. 优先识别 mmd_tools 节点
    mmd_tex = nodes.get("mmd_base_tex")
    if mmd_tex and mmd_tex.type == "TEX_IMAGE" and mmd_tex.image:
        info["base_image"] = mmd_tex.image
        return info

    # 2. 原生 Blender 着色器分析（含节点组递归）
    all_nodes = list(_find_nodes_recursive(mat.node_tree))

    for node in all_nodes:
        if node.type == "BSDF_PRINCIPLED":
            bc_inp = node.inputs.get("Base Color")
            if bc_inp:
                if bc_inp.is_linked:
                    img = _trace_first_image_from_socket(bc_inp)
                    if img:
                        info["base_image"] = img
                else:
                    c = bc_inp.default_value
                    info["tint_color"] = (float(c[0]), float(c[1]), float(c[2]), 1.0)

            alpha_inp = node.inputs.get("Alpha")
            if alpha_inp:
                if alpha_inp.is_linked:
                    a_img = _trace_first_image_from_socket(alpha_inp)
                    if a_img:
                        info["alpha_image"] = a_img
                else:
                    info["alpha"] = float(alpha_inp.default_value)

            rough_inp = node.inputs.get("Roughness")
            if rough_inp and not rough_inp.is_linked:
                info["roughness"] = float(rough_inp.default_value)

            metal_inp = node.inputs.get("Metallic")
            if metal_inp and not metal_inp.is_linked:
                info["metallic"] = float(metal_inp.default_value)

            norm_inp = node.inputs.get("Normal")
            if norm_inp and norm_inp.is_linked:
                info["normal_image"] = _trace_first_image_from_socket(norm_inp)

            em_inp = node.inputs.get("Emission Color") or node.inputs.get("Emission")
            em_str_inp = node.inputs.get("Emission Strength")
            if em_inp:
                if em_inp.is_linked:
                    info["emission_image"] = _trace_first_image_from_socket(em_inp)
                    info["emission_strength"] = (
                        float(em_str_inp.default_value) if em_str_inp else 1.0
                    )
                else:
                    ec = em_inp.default_value
                    if ec[0] > 0.01 or ec[1] > 0.01 or ec[2] > 0.01:
                        info["emission_color"] = (
                            float(ec[0]),
                            float(ec[1]),
                            float(ec[2]),
                            1.0,
                        )
                        info["emission_strength"] = (
                            float(em_str_inp.default_value) if em_str_inp else 1.0
                        )
            break

        elif node.type == "EMISSION":
            col_inp = node.inputs.get("Color")
            str_inp = node.inputs.get("Strength")
            if col_inp:
                if col_inp.is_linked:
                    img = _trace_first_image_from_socket(col_inp)
                    info["base_image"] = img
                    info["emission_image"] = img
                else:
                    ec = col_inp.default_value
                    info["tint_color"] = (float(ec[0]), float(ec[1]), float(ec[2]), 1.0)
                    info["emission_color"] = info["tint_color"]
            info["emission_strength"] = float(str_inp.default_value) if str_inp else 1.0

        elif node.type == "BSDF_DIFFUSE":
            col_inp = node.inputs.get("Color")
            if col_inp:
                if col_inp.is_linked and info["base_image"] is None:
                    info["base_image"] = _trace_first_image_from_socket(col_inp)
                elif not col_inp.is_linked:
                    dc = col_inp.default_value
                    info["tint_color"] = (float(dc[0]), float(dc[1]), float(dc[2]), 1.0)

    # 3. 兜底：若还没找到贴图，从节点树中找任意有效贴图
    if info["base_image"] is None:
        for node in all_nodes:
            if node.type == "TEX_IMAGE" and node.image:
                name_low = (node.name + " " + node.image.name).lower()
                if (
                    "toon" not in name_low
                    and "sphere" not in name_low
                    and "sph" not in name_low
                    and "normal" not in name_low
                ):
                    info["base_image"] = node.image
                    break

    return info
