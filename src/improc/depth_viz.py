"""Depth visualization from EXR without using Blender compositor.
Reads a float Z(m) EXR → clamp [0,zmax] → normalize [0,1] → save 16-bit PNG.
"""

from __future__ import annotations

import os

import numpy as np


def visualize_exr_to_png(
    depth_m: np.ndarray,
    png_path: str,
    zmax_m: float,
    invalid_color: tuple[int, int, int] | None = None,
) -> None:
    import cv2

    d = depth_m.astype(np.float32)
    is_finite_pos = np.isfinite(d) & (d > 0)
    if zmax_m and zmax_m > 0:
        is_valid = is_finite_pos & (d <= zmax_m)
    else:
        is_valid = is_finite_pos

    if not (zmax_m and zmax_m > 0):
        # choose z_max from valid pixels
        vmax = float(np.max(d[is_valid])) if np.any(is_valid) else 0.0
        zmax_m = vmax if vmax > 0 else 1.0

    d_clip = np.zeros_like(d, dtype=np.float32)
    d_clip[is_valid] = np.minimum(d[is_valid], zmax_m)

    if invalid_color is None:
        # 16-bit single channel, invalid pixel has value 0
        png16 = (d_clip / zmax_m * 65535.0).astype(np.uint16)
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        cv2.imwrite(png_path, png16)
    else:
        # valid -> 8-bit grayscale, invalid -> "invalid color" e.g. green
        g = np.zeros_like(d, dtype=np.uint8)
        g[is_valid] = np.clip((d_clip[is_valid] / zmax_m) * 255.0, 0, 255).astype(
            np.uint8
        )
        rgb = np.stack([g, g, g], axis=-1)  # grayscale to 3-channel

        # paint invalid pixels with invalid_color
        ic = tuple(int(c) for c in invalid_color)
        mask = ~is_valid
        if np.any(mask):
            rgb[mask] = np.array(ic, dtype=np.uint8)

        # transform RGB->BGR since OpenCV expects BGR
        bgr = rgb[..., ::-1]
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        cv2.imwrite(png_path, bgr)


__all__ = ['visualize_exr_to_png']
