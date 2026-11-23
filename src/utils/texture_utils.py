import hashlib
import os
import re

import bpy
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds", ".tif", ".tiff", ".webp"}

FORMAT_TO_EXT = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "TARGA": ".tga",
    "TARGA_RAW": ".tga",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "WEBP": ".webp",
}


def scan_texture_directory(search_root: str):
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


def resolve_packed_or_external_images(search_root: str, cache_dir: str):
    """
    同时处理：
    1) 直接打包在 .blend 内部的贴图 (packed_file) —— 若未解码则安全解出字节流加载
    2) 外部路径丢失的贴图 —— 扫描本地目录自动重连
    """
    by_filename, by_stem = scan_texture_directory(search_root)
    unpack_dir = os.path.join(cache_dir, "_unpacked")

    packed_count = 0
    external_ok = 0
    relinked_count = 0

    for img in list(bpy.data.images):
        if img.source == "GENERATED":
            external_ok += 1
            continue
        if img.source not in {"FILE", "SEQUENCE"}:
            continue

        # 情况 A：贴图直接打包在 .blend 里面
        if img.packed_file is not None:
            packed_count += 1
            if img.size[0] == 0 or img.size[1] == 0:
                os.makedirs(unpack_dir, exist_ok=True)
                ext = FORMAT_TO_EXT.get(getattr(img, "file_format", "PNG"), ".png")
                if img.filepath:
                    fp_ext = os.path.splitext(img.filepath.replace("\\", "/"))[
                        1
                    ].lower()
                    if fp_ext in IMAGE_EXTS:
                        ext = fp_ext
                safe = "".join(
                    c if c.isalnum() or c in "_-" else "_" for c in img.name
                )[:40]
                tmp_path = os.path.join(unpack_dir, f"{safe}{ext}")
                with open(tmp_path, "wb") as f:
                    f.write(img.packed_file.data)
                img.filepath = tmp_path
                try:
                    img.unpack(method="REMOVE")
                except Exception:
                    pass
                img.reload()
            continue

        # 情况 B：外部贴图路径有效
        abs_fp = bpy.path.abspath(img.filepath) if img.filepath else ""
        if abs_fp and os.path.isfile(abs_fp):
            if img.size[0] == 0:
                try:
                    img.reload()
                except Exception:
                    pass
            if img.size[0] > 0:
                external_ok += 1
                continue

        # 情况 C：外部路径失效 -> 扫描本地目录自动重连
        if not by_filename:
            continue

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
        f"  [贴图统计] .blend内嵌打包: {packed_count} 张 | "
        f"本地有效: {external_ok} 张 | 扫描重连: {relinked_count} 张"
    )


def ensure_image_loaded(img: bpy.types.Image) -> bool:
    if img is None:
        return False
    if img.source == "GENERATED":
        return img.size[0] > 0 and img.size[1] > 0
    if img.size[0] == 0 or img.size[1] == 0:
        if img.packed_file is None:
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
    alpha_mask_img: bpy.types.Image = None,
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
    need_alpha_merge = alpha_mask_img is not None and alpha_mask_img != img

    # 判断最终图像是否需要 Alpha 通道；法线贴图(_nrm)强制用 PNG 避免 JPEG 压缩噪点
    has_alpha = float(rgba[:, :, 3].min()) < 0.99 or material_alpha < 0.98
    if need_alpha_merge:
        mask_rgba = get_image_pixels_rgba(alpha_mask_img)
        if mask_rgba is not None and mask_rgba.shape[:2] == rgba.shape[:2]:
            has_alpha = True
    else:
        mask_rgba = None

    use_png = has_alpha or suffix == "_nrm"
    out_ext = ".png" if use_png else ".jpg"
    out_fmt = "PNG" if use_png else "JPEG"

    if (
        img.source != "GENERATED"
        and src_format == out_fmt
        and not need_tint
        and not need_resize
        and not need_alpha_merge
        and img.packed_file is not None
        and suffix == ""
    ):
        return img

    mask_key = alpha_mask_img.name if need_alpha_merge else ""
    key_str = (
        f"{img.name}|{w}x{h}|{tint.tolist()}|{max_size}|{suffix}|{mask_key}|{out_fmt}"
    )
    digest = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:10]
    safe_base = "".join(
        c if c.isalnum() or c in "_-" else "_" for c in os.path.splitext(img.name)[0]
    )[:32]
    out_name = f"{safe_base}_{digest}{suffix}{out_ext}"
    out_path = os.path.join(cache_dir, out_name)

    if out_name in bpy.data.images and bpy.data.images[out_name].size[0] > 0:
        return bpy.data.images[out_name]

    if not os.path.exists(out_path):
        pixels = rgba.copy()
        if need_alpha_merge and mask_rgba is not None:
            pixels[:, :, 3] = (
                mask_rgba[:, :, 3]
                if mask_rgba[:, :, 3].min() < 0.99
                else mask_rgba[:, :, 0]
            )
        if need_tint:
            pixels = np.clip(pixels * tint, 0.0, 1.0)

        work_img = bpy.data.images.new(
            f"__tmp_{out_name}", width=w, height=h, alpha=use_png
        )
        try:
            work_img.pixels.foreach_set(pixels.ravel())
            work_img.update()

            if need_resize:
                scale = max_size / float(max(w, h))
                new_w = max(16, int(round(w * scale)))
                new_h = max(16, int(round(h * scale)))
                work_img.scale(new_w, new_h)

            work_img.filepath_raw = out_path
            work_img.file_format = out_fmt
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
