"""Depth visualization from EXR without using Blender compositor.
Reads a float Z(m) EXR → clamp [0,zmax] → normalize [0,1] → save 16-bit PNG.
"""

from __future__ import annotations

import os

import numpy as np


def visualize_exr_to_png(depth_m: np.ndarray, png_path: str, zmax_m: float) -> None:
    import cv2

    d = depth_m.astype(np.float32)
    if not (zmax_m and zmax_m > 0):
        zmax_m = float(np.max(d)) if np.max(d) > 0 else 1.0
    d = np.where(np.isfinite(d) & (d > 0), np.minimum(d, zmax_m), 0.0)
    png16 = (d / zmax_m * 65535.0).astype(np.uint16)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    cv2.imwrite(png_path, png16)


__all__ = ['visualize_exr_to_png']
