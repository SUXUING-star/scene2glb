import os

import bpy

import step01_prepare_roots
import step02_fix_geometry_zfighting
import step03_consolidate_materials
import step04_lowpoly_materials
import step05_export_glb
from utils import scene_utils


def run(config: dict):
    config = config or {}
    blend_dir = scene_utils.get_blend_dir()
    output_dir = config.get("output_dir") or os.path.join(blend_dir, "output_glb")
    tex_cache_dir = os.path.join(output_dir, "_tex_cache")

    print("=" * 60)
    print("DCC Pipeline: Blender / MMD -> Standard GLB")
    print(f"工作目录: {blend_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # Step 1: 打包多个 PMX / 场景对象并改名英文根节点
    root_obj = step01_prepare_roots.run(config)

    # Step 2: 清理 MMD 描边与修复双面 Z-fighting
    step02_fix_geometry_zfighting.run(config, root_obj)

    # Step 3: 整合贴图并转换为标准 PBR 材质
    step03_consolidate_materials.run(config, root_obj, tex_cache_dir)

    # Step 4: 自动生成低模材质
    low_root = step04_lowpoly_materials.run(config, root_obj, tex_cache_dir)

    # Step 5: 导出标准 GLB（及低模版 *_lowpoly.glb）
    step05_export_glb.run(config, root_obj, low_root, output_dir)

    print("全部流程执行完成！")
