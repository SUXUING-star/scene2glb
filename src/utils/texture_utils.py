import hashlib
import os
import re

import bpy
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds", ".tif", ".tiff", ".webp"}


def scan_texture_directory(search_root: str):
    """
    递归扫描目录下的所有贴图文件，建立精确文件名 / 去后缀名索引
    """
    by_filename = {}
    by_stem = {}
    skip_dirs = {"output_glb", "_tex_cache", ".git", "__pycache__"}

    if not search_root or not os.path.isdir(search_root):
        return by_filename, by_stem

    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            full_path = os.path.abspath(os.path.join(root, fname))
            low_name = fname.lower()
            low_stem = os.path.splitext(low_name)[0]
            by_filename.setdefault(low_name, []).append(full_path)
            by_stem.setdefault(low_stem, []).append(full_path)

    return by_filename, by_stem


def _strip_blender_suffix(name: str) -> str:
    return re.sub(r"\.\d{3}$", "", name)


def relink_missing_textures(search_root: str):
    """
    自动扫描本地贴图文件夹并重连所有丢失/未打包的 Blender Image 数据块
    """
    by_filename, by_stem = scan_texture_directory(search_root)
    if not by_filename:
        print(f"  [贴图扫描] 目录 {search_root} 下未发现外部贴图文件")
        return

    relinked_count = 0
    for img in bpy.data.images:
        if img.source not in {"FILE", "SEQUENCE"}:
            continue

        # 已有打包数据或原路径有效且可读取，直接跳过
        abs_fp = bpy.path.abspath(img.filepath) if img.filepath else ""
        if img.packed_file is not None and img.size[0] > 0:
            continue
        if abs_fp and os.path.isfile(abs_fp):
            if img.size[0] == 0:
                try:
                    img.reload()
                except Exception:
                    pass
            if img.size[0] > 0:
                continue

        # 提取候选文件名（兼容 Windows 反斜杠和 Blender .001 后缀）
        candidates = []
        if img.filepath:
            norm = img.filepath.replace("\\", "/")
            candidates.append(os.path.basename(norm))
        candidates.append(_strip_blender_suffix(img.name))

        matched = None
        for cand in candidates:
            if not cand:
                continue
            low_cand = cand.lower()
            stem_cand = os.path.splitext(low_cand)[0]
            if low_cand in by_filename:
                matched = by_filename[low_cand][0]
                break
            if stem_cand in by_stem:
                matched = by_stem[stem_cand][0]
                break

        if matched:
            img.filepath = matched
            try:
                img.reload()
                relinked_count += 1
                print(
                    f"  [贴图重连] {img.name} -> {os.path.relpath(matched, search_root)}"
                )
            except Exception as e:
                print(f"  [贴图警告] 加载失败 {matched}: {e}")

    print(
        f"  [贴图扫描] 扫描到 {sum(len(v) for v in by_filename.values())} 张本地贴图，重连 {relinked_count} 张"
    )


def ensure_image_loaded(img: bpy.types.Image) -> bool:
    if img is None:
        return False
    if img.size[0] == 0 or img.size[1] == 0:
        try:
            img.reload()
        except Exception:
            return False
    return img.size[0] > 0 and img.size[1] > 0


def get_image_pixels_rgba(img: bpy.types.Image):
    if not ensure_image_loaded(img):
        return None
    w, h = img.size[0], img.size[1]
    arr = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    return arr.reshape((h, w, 4))


def classify_alpha_mode(img: bpy.types.Image, material_alpha: float = 1.0):
    if material_alpha < 0.98:
        return "BLEND", 0.5

    if img is None or not ensure_image_loaded(img):
        return "OPAQUE", 0.5

    rgba = get_image_pixels_rgba(img)
    if rgba is None:
        return "OPAQUE", 0.5

    alpha = rgba[:, :, 3]
    if float(alpha.min()) >= 0.99:
        return "OPAQUE", 0.5

    semi_ratio = float(np.mean((alpha > 0.08) & (alpha < 0.92)))
    if semi_ratio < 0.05:
        return "CLIP", 0.5
    return "BLEND", 0.5


def bake_tint_and_convert_png(
    img: bpy.types.Image,
    tint_rgba,
    material_alpha: float,
    cache_dir: str,
    max_size: int = 2048,
    suffix: str = "",
):
    os.makedirs(cache_dir, exist_ok=True)
    tint = np.array(
        [tint_rgba[0], tint_rgba[1], tint_rgba[2], material_alpha],
        dtype=np.float32,
    )
    need_tint = not np.allclose(tint, [1.0, 1.0, 1.0, 1.0], atol=0.01)

    if img is None:
        return None

    rgba = get_image_pixels_rgba(img)
    if rgba is None:
        return img

    h, w, _ = rgba.shape
    src_format = getattr(img, "file_format", "PNG")
    need_resize = max(w, h) > max_size

    if src_format == "PNG" and not need_tint and not need_resize and img.packed_file:
        return img

    key_str = f"{img.name}|{w}x{h}|{tint.tolist()}|{max_size}|{suffix}"
    digest = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:10]
    safe_base = "".join(
        c if c.isalnum() or c in "_-" else "_" for c in os.path.splitext(img.name)[0]
    )[:32]
    out_name = f"{safe_base}_{digest}{suffix}.png"
    out_path = os.path.join(cache_dir, out_name)

    if out_name in bpy.data.images and bpy.data.images[out_name].size[0] > 0:
        return bpy.data.images[out_name]

    if not os.path.exists(out_path):
        # 新建独立图像对象，避免 packed_file 只读导致 save() 报错
        work_img = bpy.data.images.new(
            f"__tmp_{out_name}", width=w, height=h, alpha=True
        )
        try:
            pixels = np.clip(rgba * tint, 0.0, 1.0) if need_tint else rgba
            work_img.pixels.foreach_set(pixels.ravel())
            work_img.update()

            if need_resize:
                scale = max_size / float(max(w, h))
                new_w = max(16, int(round(w * scale)))
                new_h = max(16, int(round(h * scale)))
                work_img.scale(new_w, new_h)

            work_img.filepath_raw = out_path
            work_img.file_format = "PNG"
            work_img.save()
        finally:
            bpy.data.images.remove(work_img)

    loaded = bpy.data.images.load(out_path, check_existing=True)
    loaded.name = out_name
    try:
        loaded.pack()
    except Exception:
        pass
    return loaded
