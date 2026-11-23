import hashlib
import math
import os

import bpy
import numpy as np

from . import scene_utils, texture_utils

PROCEDURAL_OR_COMPLEX_NODES = {
    "MIX",
    "MIX_RGB",
    "MIX_SHADER",
    "VALTORGB",  # ColorRamp
    "TEX_NOISE",
    "TEX_VORONOI",
    "TEX_WAVE",
    "TEX_MUSGRAVE",
    "TEX_GRADIENT",
    "TEX_CHECKER",
    "TEX_MAGIC",
    "TEX_BRICK",
    "AMBIENT_OCCLUSION",
    "FRESNEL",
    "LAYER_WEIGHT",
    "ATTRIBUTE",
    "VERTEX_COLOR",
    "SHADERTORGB",
    "MAP_RANGE",
    "MATH",
    "VECT_MATH",
    "HUE_SAT",
    "BRIGHTCONTRAST",
    "GAMMA",
    "RGBTOBW",
    "CURVE_RGB",
    "GROUP",
    "BUMP",
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


def needs_cycles_bake(mat: bpy.types.Material, bake_mode: str = "auto") -> bool:
    if bake_mode in ("all", "combined"):
        return True
    if not mat or not mat.use_nodes or not mat.node_tree:
        return False
    for node in _iter_all_nodes(mat.node_tree):
        if (
            node.type == "GROUP"
            and node.node_tree
            and "mmd_shader" in node.node_tree.name.lower()
        ):
            continue
        if node.type in PROCEDURAL_OR_COMPLEX_NODES:
            return True
    return False


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


def _ensure_bakeable_visibility_and_check_eval(obj: bpy.types.Object) -> bool:
    """确保对象及所在集合渲染可见，且评估后的几何体确实有面"""
    scene_utils.enable_all_collections_in_view_layer()
    for col in obj.users_collection:
        col.hide_viewport = False
        col.hide_render = False

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


def ensure_bake_uv(obj: bpy.types.Object, force_reunwrap: bool = False):
    mesh = obj.data
    if len(mesh.polygons) == 0:
        return
    if len(mesh.uv_layers) > 0 and not force_reunwrap:
        return

    uv_name = "__BakeUV"
    uv_layer = mesh.uv_layers.get(uv_name) or mesh.uv_layers.new(name=uv_name)
    mesh.uv_layers.active = uv_layer
    uv_layer.active_render = True

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.02)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


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
        print(f"  [Bake跳过] {obj.name}: 评估面数为 0 或对象不在当前视图层，跳过烘焙")
        return {}

    scene = bpy.context.scene
    prev_engine = scene.render.engine
    prev_samples = scene.cycles.samples
    prev_denoise = scene.cycles.use_denoising
    prev_device = scene.cycles.device

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16 if bake_mode == "combined" else 4
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 4 if bake_mode == "combined" else 2
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        if prefs.compute_device_type != "NONE":
            scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"

    need_reunwrap = len(obj.data.uv_layers) == 0 or any(
        uses_position_dependent_coords(m) for m in target_mats
    )
    ensure_bake_uv(obj, force_reunwrap=need_reunwrap)

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
                f"{obj.name}|{mat.name}|{res}|{bake_mode}".encode("utf-8")
            ).hexdigest()[:8]
            safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in mat.name)[
                :28
            ]

            base_img = bpy.data.images.new(
                f"__bake_{safe}_{tag}_base", width=res, height=res, alpha=True
            )
            emit_img = bpy.data.images.new(
                f"__bake_{safe}_{tag}_emit", width=res, height=res, alpha=False
            )
            alpha_img = bpy.data.images.new(
                f"__bake_{safe}_{tag}_alpha", width=res, height=res, alpha=False
            )
            bake_images[mat.name] = (base_img, emit_img, alpha_img)

        def _run_bake(bake_type: str, **kwargs):
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            with bpy.context.temp_override(
                active_object=obj,
                selected_objects=[obj],
                selected_editable_objects=[obj],
            ):
                bpy.ops.object.bake(type=bake_type, margin=8, use_clear=True, **kwargs)

        if bake_mode == "combined":
            for idx, orig_mat in enumerate(orig_slots):
                if orig_mat in target_set:
                    b_mat = orig_mat.copy()
                    temp_mats.append(b_mat)
                    _set_active_bake_image_node(b_mat, bake_images[orig_mat.name][0])
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
                    _set_active_bake_image_node(b_mat, bake_images[orig_mat.name][0])
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
                    _set_active_bake_image_node(b_mat, bake_images[orig_mat.name][1])
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
                        alpha_baked_mats.add(orig_mat.name)
                        _set_active_bake_image_node(
                            b_mat, bake_images[orig_mat.name][2]
                        )
                    else:
                        _set_active_bake_image_node(b_mat, dummy_img)
                    obj.material_slots[idx].material = b_mat
                else:
                    obj.material_slots[idx].material = dummy_mat
            if alpha_baked_mats:
                _run_bake("EMIT")

        os.makedirs(cache_dir, exist_ok=True)
        for mat in target_mats:
            base_img, emit_img, alpha_img = bake_images[mat.name]
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

            if mat.name in alpha_baked_mats:
                alpha_rgba = texture_utils.get_image_pixels_rgba(alpha_img)
                if alpha_rgba is not None:
                    base_rgba[:, :, 3] = np.clip(alpha_rgba[:, :, 0], 0.0, 1.0)

            base_img.pixels.foreach_set(base_rgba.ravel())
            base_img.update()

            safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in mat.name)[
                :28
            ]
            tag = hashlib.md5(
                f"{obj.name}|{mat.name}|{res}".encode("utf-8")
            ).hexdigest()[:8]

            has_alpha = float(base_rgba[:, :, 3].min()) < 0.99
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

            results[mat.name] = {
                "base_img": saved_base,
                "emit_img": saved_emit,
                "has_emission": has_emission or is_pure_emit,
                "is_pure_emit": is_pure_emit,
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

        scene.render.engine = prev_engine
        scene.cycles.samples = prev_samples
        scene.cycles.use_denoising = prev_denoise
        scene.cycles.device = prev_device

    return results
