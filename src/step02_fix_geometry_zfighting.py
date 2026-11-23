import bmesh
import bpy
from mathutils import Vector

from utils import scene_utils


def _remove_mmd_edge_modifiers_and_faces(obj):
    for mod in list(obj.modifiers):
        if "mmd_edge" in mod.name.lower():
            obj.modifiers.remove(mod)
        elif mod.type == "SOLIDIFY" and getattr(mod, "use_flip_normals", False):
            if (
                mod.material_offset > 0
                or "outline" in mod.name.lower()
                or "edge" in mod.name.lower()
            ):
                obj.modifiers.remove(mod)

    edge_slot_indices = set()
    for idx, slot in enumerate(obj.material_slots):
        if slot.material:
            mname = slot.material.name.lower()
            if (
                mname.startswith("mmd_edge")
                or mname.endswith("_edge")
                or mname.endswith("_outline")
            ):
                edge_slot_indices.add(idx)

    if not edge_slot_indices:
        return

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    faces_to_del = [f for f in bm.faces if f.material_index in edge_slot_indices]
    if faces_to_del:
        bmesh.ops.delete(bm, geom=faces_to_del, context="FACES")
        bm.faces.ensure_lookup_table()

    # 重新映射材质槽索引后再从 mesh.materials 直接 pop，避免 bpy.ops 上下文错误
    idx_map = {}
    new_i = 0
    for old_i in range(len(mesh.materials)):
        if old_i not in edge_slot_indices:
            idx_map[old_i] = new_i
            new_i += 1
    for f in bm.faces:
        f.material_index = idx_map.get(f.material_index, 0)

    bm.to_mesh(mesh)
    bm.free()

    for idx in sorted(edge_slot_indices, reverse=True):
        mesh.materials.pop(index=idx)


def _fix_mesh_zfighting(obj, offset_eps=0.00012):
    mesh = obj.data
    if not mesh.polygons:
        return

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    degenerate = [f for f in bm.faces if f.calc_area() < 1e-10]
    if degenerate:
        bmesh.ops.delete(bm, geom=degenerate, context="FACES")
        bm.faces.ensure_lookup_table()

    def face_key(face):
        pts = sorted(
            (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)) for v in face.verts
        )
        return tuple(pts)

    seen_faces = {}
    duplicate_faces = []
    coplanar_opposite_pairs = []

    for f in bm.faces:
        key = face_key(f)
        if key not in seen_faces:
            seen_faces[key] = f
        else:
            prev = seen_faces[key]
            dot = f.normal.dot(prev.normal)
            if dot > 0.9:
                duplicate_faces.append(f)
            elif dot < -0.9:
                coplanar_opposite_pairs.append((prev, f))

    if duplicate_faces:
        bmesh.ops.delete(bm, geom=duplicate_faces, context="FACES")
        bm.faces.ensure_lookup_table()

    if coplanar_opposite_pairs:
        vert_offsets = {}
        for f1, f2 in coplanar_opposite_pairs:
            if not f1.is_valid or not f2.is_valid:
                continue
            for v in f1.verts:
                vert_offsets.setdefault(v, Vector((0.0, 0.0, 0.0)))
                vert_offsets[v] += f1.normal * offset_eps
            for v in f2.verts:
                vert_offsets.setdefault(v, Vector((0.0, 0.0, 0.0)))
                vert_offsets[v] += f2.normal * offset_eps

        for v, delta in vert_offsets.items():
            if delta.length > 0:
                if delta.length > offset_eps * 2:
                    delta = delta.normalized() * (offset_eps * 2)
                v.co += delta

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def run(config: dict, root_obj: bpy.types.Object):
    print("[Step 02] 清理 MMD 描边与修复双面渲染 Z-fighting...")
    scene_utils.ensure_object_mode()
    offset_eps = float(config.get("zfighting_offset", 0.00012))

    meshes = [o for o in scene_utils.iter_hierarchy(root_obj) if o.type == "MESH"]
    for obj in meshes:
        _remove_mmd_edge_modifiers_and_faces(obj)
        _fix_mesh_zfighting(obj, offset_eps=offset_eps)

    print(f"  [完成] 已处理 {len(meshes)} 个网格的几何与 Z-fighting 问题")
