"""Depth visualization from EXR without using Blender compositor.
Reads a float Z(m) EXR → clamp [0,zmax] → normalize [0,1] → save 16-bit PNG.
Prefers OpenEXR+Imath (system Python), falls back to imageio if available.
"""

from __future__ import annotations

import math
import os

import numpy as np

# Prefer pure-Python imageio; fall back to bpy Image I/O when unavailable.
_HAVE_IMAGEIO = True
try:
    import imageio.v3 as iio  # type: ignore
except Exception:
    _HAVE_IMAGEIO = False
    iio = None

# OpenEXR (preferred on system Python)
_HAVE_OPENEXR = True
try:
    import Imath
    import OpenEXR
except Exception:
    _HAVE_OPENEXR = False


def _read_exr_numpy_via_bpy(path: str) -> np.ndarray:
    """Fallback EXR reader using Blender's image loader, returns HxW float32."""
    import bpy  # local import to avoid hard dep when used outside Blender

    img = bpy.data.images.load(path)
    try:
        w, h = img.size
        # Blender stores pixels as flat RGBA floats [0..1 or meters]; take R
        buf = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        depth = buf[..., 0]
        return depth.copy()
    finally:
        # avoid leaking .blend datablocks
        bpy.data.images.remove(img)


def _write_png16_via_bpy(path: str, gray_u16: np.ndarray) -> None:
    """Fallback PNG writer via Blender (sRGB off; single channel expanded to RGBA)."""
    import bpy  # local import

    h, w = gray_u16.shape
    # Normalize back to [0,1] for Blender, but keep 16-bit when saving
    gray01 = gray_u16.astype(np.float32) / 65535.0
    # Blender Image is RGBA; replicate into RGB, keep A=1
    rgba = np.stack([gray01, gray01, gray01, np.ones_like(gray01)], axis=-1)
    img = bpy.data.images.new(
        name='DepthVizTmp', width=w, height=h, alpha=True, float_buffer=False
    )
    try:
        img.pixels = rgba.astype(np.float32).ravel().tolist()
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
    finally:
        bpy.data.images.remove(img)


def visualize_exr_to_png(
    depth_exr_path: str, depth_viz_png_path: str, zmax_m: float
) -> None:
    """Create a 16-bit grayscale PNG visualization from a depth EXR.

    Mapping: value_png = clip(depth_m / zmax_m, 0, 1) * 65535 (uint16)
    """
    if zmax_m <= 0 or not math.isfinite(zmax_m):
        raise ValueError(f'zmax_m must be positive finite, got {zmax_m}')

    # --- Read EXR to float32 ---
    if _HAVE_OPENEXR:
        # single-channel Z EXR (float32)
        exr = OpenEXR.InputFile(depth_exr_path)
        try:
            dw = exr.header()['dataWindow']
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            # Try common depth channel names first
            chan_name = 'Z'
            hdr_chs = exr.header().get('channels', {})
            if isinstance(hdr_chs, dict):
                if 'Z' not in hdr_chs and 'R' in hdr_chs:
                    chan_name = 'R'
                elif 'Z' not in hdr_chs and len(hdr_chs):
                    chan_name = next(iter(hdr_chs.keys()))
            # frombuffer -> read-only; make a writable copy
            z = np.frombuffer(exr.channel(chan_name, pt), dtype=np.float32)
            depth = z.copy().reshape(h, w)
        finally:
            exr.close()
    elif _HAVE_IMAGEIO:
        print('[INFO] visualizing exr to png, using python imageio')
        arr = iio.imread(depth_exr_path)  # HxW or HxWxC (float32/float64)
        if arr.ndim == 3:
            arr = arr[..., 0]
        depth = np.asarray(arr, dtype=np.float32)
    else:
        # Fallback using Blender's loader
        print('[INFO] visualizing exr to png, using blender')
        depth = _read_exr_numpy_via_bpy(depth_exr_path)

    # --- Sanitize values ---
    depth = np.nan_to_num(depth, copy=False, nan=np.inf, posinf=np.inf, neginf=0.0)
    # Clamp to [0, zmax] and normalize
    depth = np.clip(depth, 0.0, float(zmax_m)) / float(zmax_m)

    # --- To 16-bit single channel ---
    png_u16 = np.round(depth * 65535.0).astype(np.uint16)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(depth_viz_png_path), exist_ok=True)

    if _HAVE_IMAGEIO:
        # Save grayscale 16-bit
        iio.imwrite(depth_viz_png_path, png_u16)
    else:
        _write_png16_via_bpy(depth_viz_png_path, png_u16)


__all__ = ['visualize_exr_to_png']
