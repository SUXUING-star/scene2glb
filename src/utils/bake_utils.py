import hashlib
import math
import os

import bpy
import numpy as np

from . import scene_utils, texture_utils

# 必须通过 Cycles Bake 才能保留效果的程序化 / 复合节点类型
COMPLEX_BAKE_NODES = {
    "TEX_NOISE",
    "TEX_VORONOI",
    "TEX_WAVE",
    "TEX_MUSGRAVE",
    "TEX_GRADIENT",
    "TEX_CHECKER",
    "TEX_MAGIC",
    "TEX_BRICK",
    "VALTORGB",
    "MIX",
    "MIX_RGB",
    "HUE_SAT",
    "BRIGHTCONTRAST",
    "CURVE_RGB",
    "SHADERTORGB",
    "ATTRIBUTE",
    "VERTEX_COLOR",
    "AMBIENT_OCCLUSION",
    "NEW_GEOMETRY",
}


def _iter_all_nodes(node_tree, visited=None):
    if node_tree is None:
        return
    if visited is None:
        visited = set()
    if node_tree in visited:
        return
    visited.add(node_tree)
    for n in node_tree.nodes:
        yield n
        if n.type == "GROUP" and n.node_tree:
            yield from _iter_all_nodes(n.node_tree, visited)


def uses_position_dependent_coords(mat: bpy.types.Material) -> bool:
    if not mat or not mat.use_nodes or not mat.node_tree:
        return False
    for node in _iter_all_nodes(mat.node_tree):
        if node.type == "TEX_COORD":
            for out_name in ("Generated", "Object", "Window", "Camera", "Reflection"):
                out = node.outputs.get(out_name)
                if out and out.is_linked:
                    return True
        elif node.type in {
            "NEW_GEOMETRY",
            "AMBIENT_OCCLUSION",
            "TEX_NOISE",
            "TEX_VORONOI",
            "TEX_GRADIENT",
        }:
            return True
    return False


def needs_cycles_bake(mat: bpy.types.Material, bake_mode: str = "auto") -> bool:
    """
    智能判断是否需要启动 Cycles Bake:
    - 'none': 绝不烘焙
    - 'all' / 'combined': 强制全部烘焙
    - 'auto': 若属于 mmd_tools 原生材质，不走 Bake（保留底图）；
              否则若使用程序化节点、ColorRamp、混色节点、ShaderToRGB、顶点色或非 UV 坐标时走 Bake。
    """
    if bake_mode in ("none", "false", "off"):
        return False
    if bake_mode in ("all", "combined"):
        return True
    if not mat or not mat.use_nodes or not mat.node_tree:
        return False

    # mmd_tools 自带材质直接走底图提取，避免把 toon 阴影烘死到固有色
    if mat.node_tree.nodes.get("mmd_base_tex") is not None:
        return False

    all_nodes = list(_iter_all_nodes(mat.node_tree))
    for node in all_nodes:
        if node.type in COMPLEX_BAKE_NODES:
            return True

    if uses_position_dependent_coords(mat):
        return True

    # 多个贴图同时存在且无简单映射时，或使用了 MixShader(非纯Transparent)
    tex_nodes = [n for n in all_nodes if n.type == "TEX_IMAGE" and n.image is not None]
    for node in all_nodes:
        if node.type == "MIX_SHADER":
            s1 = node.inputs[1].links[0].from_node if node.inputs[1].is_linked else None
            s2 = node.inputs[2].links[0].from_node if node.inputs[2].is_linked else None
            has_trans = (s1 and s1.type == "BSDF_TRANSPARENT") or (
                s2 and s2.type == "BSDF_TRANSPARENT"
            )
            if not has_trans and len(tex_nodes) > 0:
                return True

    return False


def _has_out_of_bounds_uv(mesh: bpy.types.Mesh) -> bool:
    if len(mesh.uv_layers) == 0:
        return True
    uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
    if not uv_layer or len(uv_layer.data) == 0:
        return True
    for loop_uv in uv_layer.data:
        u, v = loop_uv.uv[0], loop_uv.uv[1]
        if u < -0.02 or u > 1.02 or v < -0.02 or v > 1.02:
            return True
    return False


def _ensure_bakeable_visibility_and_check_eval(obj: bpy.types.Object) -> bool:
    scene_utils.enable_all_collections_in_view_layer()
    for col in obj.users_collection:
        if not getattr(col, "hide_render", False):
            col.hide_viewport = False

    scene_utils.ensure_object_visible_and_linked(obj)
    obj.hide_render = False
    obj.hide_viewport = False

    if obj.name not in bpy.context.view_layer.objects:
        return False

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    poly_count = len(eval_mesh.polygons) if eval_mesh else 0
    eval_obj.to_mesh_clear()
    return poly_count > 0


def ensure_bake_uv(obj: bpy.types.Object, force_new_uv: bool = False):
    """
    确保对象可用于烘焙。
    如果原来已有 UV 且因空间坐标/越界平铺需要重展，则在新建的 `__BakeUV` 上展开，
    并返回 (ok, uv_name_to_use, prev_active_uv_name, prev_render_uv_name)，避免破坏其他未烘焙材质的 UV！
    """
    mesh = obj.data
    if len(mesh.polygons) == 0:
        return False, None, None, None

    prev_active = mesh.uv_layers.active.name if mesh.uv_layers.active else None
    prev_render = next(
        (uv.name for uv in mesh.uv_layers if uv.active_render), prev_active
    )

    if len(mesh.uv_layers) > 0 and not force_new_uv:
        return True, prev_active, prev_active, prev_render

    uv_name = "__BakeUV" if len(mesh.uv_layers) > 0 else "UVMap"
    uv_layer = mesh.uv_layers.get(uv_name)
    if uv_layer is None:
        try:
            uv_layer = mesh.uv_layers.new(name=uv_name)
        except Exception:
            uv_layer = None

    if uv_layer is None:
        if len(mesh.uv_layers) > 0:
            uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
            return True, uv_layer.name, prev_active, prev_render
        return False, None, None, None

    mesh.uv_layers.active = uv_layer
    uv_layer.active_render = True

    scene_utils.ensure_object_visible_and_linked(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    with bpy.context.temp_override(
        active_object=obj,
        selected_objects=[obj],
        selected_editable_objects=[obj],
    ):
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.02)
        except Exception:
            return False, None, prev_active, prev_render
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    return True, uv_layer.name, prev_active, prev_render


def _bypass_shader_to_rgb(node_tree):
    for node in list(node_tree.nodes):
        if node.type == "GROUP" and node.node_tree:
            _bypass_shader_to_rgb(node.node_tree)
        elif node.type == "SHADERTORGB":
            shader_inp = node.inputs.get("Shader")
            col_out = node.outputs.get("Color")
            fallback_col = (1.0, 1.0, 1.0, 1.0)
            src_col_socket = None

            if shader_inp and shader_inp.is_linked:
                up_node = shader_inp.links[0].from_node
                for cname in ("Base Color", "Color"):
                    ci = up_node.inputs.get(cname)
                    if ci:
                        if ci.is_linked:
                            src_col_socket = ci.links[0].from_socket
                        else:
                            fallback_col = tuple(ci.default_value)
                        break

            rgb_node = node_tree.nodes.new("ShaderNodeRGB")
            rgb_node.outputs[0].default_value = fallback_col
            for link in list(col_out.links):
                to_sock = link.to_socket
                node_tree.links.remove(link)
                node_tree.links.new(src_col_socket or rgb_node.outputs[0], to_sock)


def _prepare_albedo_bake_material(mat_copy):
    nt = mat_copy.node_tree
    _bypass_shader_to_rgb(nt)
    for node in _iter_all_nodes(nt):
        if node.type == "BSDF_PRINCIPLED":
            for sname in (
                "Metallic",
                "Transmission",
                "Transmission Weight",
                "Emission Strength",
            ):
                sock = node.inputs.get(sname)
                if sock:
                    for lk in list(sock.links):
                        node.id_data.links.remove(lk)
                    sock.default_value = 0.0
            em_sock = node.inputs.get("Emission Color") or node.inputs.get("Emission")
            if em_sock:
                for lk in list(em_sock.links):
                    node.id_data.links.remove(lk)
                em_sock.default_value = (0.0, 0.0, 0.0, 1.0)


def _rewire_for_alpha_bake(mat_copy) -> bool:
    nt = mat_copy.node_tree
    out_node = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if not out_node or not out_node.inputs["Surface"].is_linked:
        return False

    alpha_source_socket = None
    invert_alpha = False

    surf_node = out_node.inputs["Surface"].links[0].from_node
    if surf_node.type == "MIX_SHADER":
        s1 = (
            surf_node.inputs[1].links[0].from_node
            if surf_node.inputs[1].is_linked
            else None
        )
        s2 = (
            surf_node.inputs[2].links[0].from_node
            if surf_node.inputs[2].is_linked
            else None
        )
        fac_inp = surf_node.inputs[0]
        if s1 and s1.type == "BSDF_TRANSPARENT" and fac_inp.is_linked:
            alpha_source_socket = fac_inp.links[0].from_socket
            invert_alpha = False
        elif s2 and s2.type == "BSDF_TRANSPARENT" and fac_inp.is_linked:
            alpha_source_socket = fac_inp.links[0].from_socket
            invert_alpha = True

    if alpha_source_socket is None:
        for node in nt.nodes:
            if node.type == "BSDF_PRINCIPLED":
                a_inp = node.inputs.get("Alpha")
                if a_inp and a_inp.is_linked:
                    alpha_source_socket = a_inp.links[0].from_socket
                    invert_alpha = False
                    break

    if alpha_source_socket is None:
        return False

    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    if invert_alpha:
        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[0].default_value = 1.0
        nt.links.new(alpha_source_socket, sub.inputs[1])
        nt.links.new(sub.outputs["Value"], em.inputs["Color"])
    else:
        nt.links.new(alpha_source_socket, em.inputs["Color"])

    nt.links.new(em.outputs["Emission"], out_node.inputs["Surface"])
    return True


def _set_active_bake_image_node(mat, img):
    nt = mat.node_tree
    for n in nt.nodes:
        n.select = False
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.name = "__BakeTarget"
    tex_node.image = img
    tex_node.select = True
    nt.nodes.active = tex_node
    return tex_node


def _save_image_file(img: bpy.types.Image, out_path: str, file_format: str = "PNG"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.filepath_raw = out_path
    img.file_format = file_format
    img.save()


def bake_object_materials(
    obj: bpy.types.Object,
    target_mats: list,
    cache_dir: str,
    res: int = 1024,
    bake_mode: str = "auto",
):
    if not target_mats or len(obj.data.polygons) == 0:
        return {}

    if not _ensure_bakeable_visibility_and_check_eval(obj):
        return {}

    need_new_uv = len(obj.data.uv_layers) == 0 or (
        (
            any(uses_position_dependent_coords(m) for m in target_mats)
            or _has_out_of_bounds_uv(obj.data)
        )
    )
    ok, bake_uv_name, prev_active_uv, prev_render_uv = ensure_bake_uv(
        obj, force_new_uv=need_new_uv
    )
    if not ok:
        return {}

    scene = bpy.context.scene
    prev_engine = scene.render.engine
    prev_samples = scene.cycles.samples
    prev_denoise = scene.cycles.use_denoising
    prev_device = scene.cycles.device

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4 if bake_mode == "combined" else 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 2
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        if prefs.compute_device_type != "NONE":
            scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"

    orig_slots = [slot.material for slot in obj.material_slots]
    target_set = set(target_mats)

    bake_images = {}
    temp_mats = []
    alpha_baked_mats = set()
    results = {}

    dummy_img = bpy.data.images.new("__bake_dummy_skip", width=8, height=8, alpha=False)
    dummy_mat = bpy.data.materials.new("__bake_dummy_mat")
    dummy_mat.use_nodes = True
    _set_active_bake_image_node(dummy_mat, dummy_img)
    temp_mats.append(dummy_mat)

    try:
        for mat in target_mats:
            tag = hashlib.md5(
                f"{obj.name}|{mat.name}|{res}".encode("utf-8")
            ).hexdigest()[:8]
            safe = (
                "".join(
                    c if c.isascii() and (c.isalnum() or c in "_-") else "_"
                    for c in mat.name
                ).strip("_")[:24]
                or "mat"
            )

            base_img = bpy.data.images.new(
                f"__bake_{safe}_{tag}_base", width=res, height=res, alpha=True
            )
            emit_img = bpy.data.images.new(
                f"__bake_{safe}_{tag}_emit", width=res, height=res, alpha=False
            )
            alpha_img = bpy.data.images.new(
                f"__bake_{safe}_{tag}_alpha", width=res, height=res, alpha=False
            )
            bake_images[mat] = (base_img, emit_img, alpha_img)

        def _run_bake(bake_type: str, **kwargs):
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            with bpy.context.temp_override(
                active_object=obj,
                selected_objects=[obj],
                selected_editable_objects=[obj],
            ):
                try:
                    bpy.ops.object.bake(
                        type=bake_type, margin=4, use_clear=True, **kwargs
                    )
                except RuntimeError:
                    if scene.cycles.device != "CPU":
                        scene.cycles.device = "CPU"
                        bpy.ops.object.bake(
                            type=bake_type, margin=4, use_clear=True, **kwargs
                        )
                    else:
                        raise

        if bake_mode == "combined":
            for idx, orig_mat in enumerate(orig_slots):
                if orig_mat in target_set:
                    b_mat = orig_mat.copy()
                    temp_mats.append(b_mat)
                    _set_active_bake_image_node(b_mat, bake_images[orig_mat][0])
                    obj.material_slots[idx].material = b_mat
                else:
                    obj.material_slots[idx].material = dummy_mat
            _run_bake("COMBINED")
        else:
            # A) 固有色
            for idx, orig_mat in enumerate(orig_slots):
                if orig_mat in target_set:
                    b_mat = orig_mat.copy()
                    temp_mats.append(b_mat)
                    _prepare_albedo_bake_material(b_mat)
                    _set_active_bake_image_node(b_mat, bake_images[orig_mat][0])
                    obj.material_slots[idx].material = b_mat
                else:
                    obj.material_slots[idx].material = dummy_mat
            _run_bake("DIFFUSE", pass_filter={"COLOR"})

            # B) 自发光
            for idx, orig_mat in enumerate(orig_slots):
                if orig_mat in target_set:
                    b_mat = orig_mat.copy()
                    temp_mats.append(b_mat)
                    _bypass_shader_to_rgb(b_mat.node_tree)
                    _set_active_bake_image_node(b_mat, bake_images[orig_mat][1])
                    obj.material_slots[idx].material = b_mat
                else:
                    obj.material_slots[idx].material = dummy_mat
            _run_bake("EMIT")

            # C) Alpha 透明遮罩
            for idx, orig_mat in enumerate(orig_slots):
                if orig_mat in target_set:
                    b_mat = orig_mat.copy()
                    temp_mats.append(b_mat)
                    if _rewire_for_alpha_bake(b_mat):
                        alpha_baked_mats.add(orig_mat)
                        _set_active_bake_image_node(b_mat, bake_images[orig_mat][2])
                    else:
                        _set_active_bake_image_node(b_mat, dummy_img)
                    obj.material_slots[idx].material = b_mat
                else:
                    obj.material_slots[idx].material = dummy_mat
            if alpha_baked_mats:
                _run_bake("EMIT")

        os.makedirs(cache_dir, exist_ok=True)
        for mat in target_mats:
            base_img, emit_img, alpha_img = bake_images[mat]
            base_rgba = texture_utils.get_image_pixels_rgba(base_img)
            emit_rgba = texture_utils.get_image_pixels_rgba(emit_img)

            is_pure_emit = False
            if base_rgba is not None and emit_rgba is not None:
                if (
                    float(base_rgba[:, :, :3].max()) < 0.02
                    and float(emit_rgba[:, :, :3].max()) > 0.02
                ):
                    base_rgba[:, :, :3] = np.clip(emit_rgba[:, :, :3], 0.0, 1.0)
                    is_pure_emit = True

            if mat in alpha_baked_mats and base_rgba is not None:
                alpha_rgba = texture_utils.get_image_pixels_rgba(alpha_img)
                if alpha_rgba is not None:
                    base_rgba[:, :, 3] = np.clip(alpha_rgba[:, :, 0], 0.0, 1.0)

            if base_rgba is not None:
                base_img.pixels.foreach_set(
                    np.ascontiguousarray(base_rgba, dtype=np.float32).ravel()
                )
                base_img.update()

            safe = (
                "".join(
                    c if c.isascii() and (c.isalnum() or c in "_-") else "_"
                    for c in mat.name
                ).strip("_")[:24]
                or "mat"
            )
            tag = hashlib.md5(
                f"{obj.name}|{mat.name}|{res}".encode("utf-8")
            ).hexdigest()[:8]

            has_alpha = base_rgba is not None and float(base_rgba[:, :, 3].min()) < 0.99
            base_ext = ".png" if has_alpha else ".jpg"
            base_fmt = "PNG" if has_alpha else "JPEG"

            base_path = os.path.join(cache_dir, f"baked_{safe}_{tag}{base_ext}")
            emit_path = os.path.join(cache_dir, f"baked_{safe}_{tag}_em.jpg")

            _save_image_file(base_img, base_path, base_fmt)

            has_emission = False
            if (
                emit_rgba is not None
                and float(emit_rgba[:, :, :3].max()) > 0.02
                and not is_pure_emit
            ):
                has_emission = True
                _save_image_file(emit_img, emit_path, "JPEG")

            for tmp_im in (base_img, emit_img, alpha_img):
                if tmp_im.name in bpy.data.images:
                    bpy.data.images.remove(tmp_im)

            saved_base = bpy.data.images.load(base_path, check_existing=True)
            try:
                saved_base.pack()
            except Exception:
                pass

            saved_emit = None
            if has_emission:
                saved_emit = bpy.data.images.load(emit_path, check_existing=True)
                try:
                    saved_emit.pack()
                except Exception:
                    pass

            results[mat] = {
                "base_img": saved_base,
                "emit_img": saved_emit,
                "has_emission": has_emission or is_pure_emit,
                "is_pure_emit": is_pure_emit,
                "uv_map": bake_uv_name if bake_uv_name != prev_render_uv else None,
            }

    except RuntimeError as e:
        print(f"  [Bake警告] {obj.name} 烘焙异常，回退普通材质解析: {e}")
        for imgs in bake_images.values():
            for tmp_im in imgs:
                if tmp_im.name in bpy.data.images:
                    bpy.data.images.remove(tmp_im)
        return {}

    finally:
        for idx, orig_mat in enumerate(orig_slots):
            if idx < len(obj.material_slots):
                obj.material_slots[idx].material = orig_mat
        for tm in temp_mats:
            if tm.name in bpy.data.materials:
                bpy.data.materials.remove(tm)
        if dummy_img.name in bpy.data.images:
            bpy.data.images.remove(dummy_img)

        # 如果原本存在非 __BakeUV 的 UV 层，恢复为默认激活 UV，避免影响同物体未烘焙材质
        if prev_active_uv and prev_active_uv in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[prev_active_uv]
        if prev_render_uv and prev_render_uv in obj.data.uv_layers:
            obj.data.uv_layers[prev_render_uv].active_render = True

        scene.render.engine = prev_engine
        scene.cycles.samples = prev_samples
        scene.cycles.use_denoising = prev_denoise
        scene.cycles.device = prev_device

    return results