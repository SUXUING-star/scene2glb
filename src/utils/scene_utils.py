import os
import re

import bpy


def get_blend_dir():
    blend_path = bpy.data.filepath
    if blend_path:
        return os.path.dirname(os.path.abspath(blend_path))
    return os.path.abspath(os.getcwd())


def sanitize_english_name(name: str, fallback: str = "ModelRoot") -> str:
    if not name:
        return fallback
    ascii_part = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name).strip("_.-")
    return ascii_part if ascii_part else fallback


def ensure_object_mode():
    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def enable_all_collections_in_view_layer():
    """确保所有集合在当前 ViewLayer 中可见且未排除，防止导出时选不中对象"""

    def _enable(layer_col):
        layer_col.exclude = False
        layer_col.hide_viewport = False
        if layer_col.collection:
            layer_col.collection.hide_viewport = False
            layer_col.collection.hide_render = False
        for child in layer_col.children:
            _enable(child)

    _enable(bpy.context.view_layer.layer_collection)


def ensure_object_visible_and_linked(obj):
    if (
        obj.name not in bpy.context.scene.collection.objects
        and len(obj.users_collection) == 0
    ):
        bpy.context.scene.collection.objects.link(obj)
    obj.hide_viewport = False
    obj.hide_render = False
    try:
        obj.hide_set(False)
    except Exception:
        pass


def select_hierarchy(root_obj, include_root=True):
    ensure_object_mode()
    enable_all_collections_in_view_layer()
    bpy.ops.object.select_all(action="DESELECT")

    def _select(obj):
        ensure_object_visible_and_linked(obj)
        if obj.name in bpy.context.view_layer.objects:
            obj.select_set(True)
        for child in obj.children:
            _select(child)

    if include_root:
        _select(root_obj)
    else:
        for child in root_obj.children:
            _select(child)

    if root_obj.name in bpy.context.view_layer.objects:
        bpy.context.view_layer.objects.active = root_obj


def iter_hierarchy(root_obj):
    yield root_obj
    for child in list(root_obj.children):
        yield from iter_hierarchy(child)


def reparent_keep_world(obj, new_parent):
    if obj == new_parent:
        return
    bpy.context.view_layer.update()
    world_mat = obj.matrix_world.copy()
    obj.parent = new_parent
    obj.matrix_parent_inverse = new_parent.matrix_world.inverted()
    obj.matrix_world = world_mat


def mesh_has_skinning(mesh_obj, arm_obj=None) -> bool:
    if mesh_obj.type != "MESH" or not mesh_obj.vertex_groups:
        return False
    for mod in mesh_obj.modifiers:
        if mod.type == "ARMATURE" and mod.object is not None:
            if arm_obj is not None and mod.object != arm_obj:
                continue
            target_arm = mod.object
            if target_arm.type == "ARMATURE" and target_arm.data:
                bone_names = {b.name for b in target_arm.data.bones}
                if any(vg.name in bone_names for vg in mesh_obj.vertex_groups):
                    return True
    return False


def is_armature_actually_used(arm_obj, hierarchy_objects) -> bool:
    if arm_obj.type != "ARMATURE":
        return False
    if arm_obj.animation_data and arm_obj.animation_data.action:
        return True
    for obj in hierarchy_objects:
        if obj.parent == arm_obj and obj.parent_type == "BONE":
            return True
        if obj.type == "MESH" and mesh_has_skinning(obj, arm_obj):
            return True
    return False


def delete_object_tree(obj):
    to_delete = list(iter_hierarchy(obj))
    for o in reversed(to_delete):
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)


def set_principled_input(bsdf, name_candidates, value):
    for name in name_candidates:
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value
            return True
    return False


def purge_unused_data():
    try:
        for _ in range(3):
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
    except Exception:
        pass
