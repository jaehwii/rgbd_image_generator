# src/improc/depth_noise.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# -----------------------------
# EXR IO helpers
# -----------------------------


def read_exr_depth(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # returns float32 for EXR
    if img is None:
        raise IOError(f'Failed to read EXR: {path}')
    if img.ndim == 3:
        img = img[..., 0]  # take Z or first channel
    return img.astype(np.float32)


def write_exr_depth(path: str, depth: np.ndarray) -> None:
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, depth.astype(np.float32))
    if not ok:
        raise IOError(f'Failed to write EXR: {path}')


# -----------------------------
# Noise models
# -----------------------------
class DepthNoiseModel:
    def __call__(self, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class GaussianDepthNoise(DepthNoiseModel):
    sigma_m: float  # absolute meters

    def __call__(self, d: np.ndarray) -> np.ndarray:
        if self.sigma_m <= 0:
            return d
        return d + np.random.normal(0.0, self.sigma_m, size=d.shape).astype(np.float32)


@dataclass
class MultiplicativeDepthNoise(DepthNoiseModel):
    sigma_rel: float  # relative std (e.g., 0.01 = 1%)

    def __call__(self, d: np.ndarray) -> np.ndarray:
        if self.sigma_rel <= 0:
            return d
        scale = 1.0 + np.random.normal(0.0, self.sigma_rel, size=d.shape).astype(
            np.float32
        )
        return d * scale


@dataclass
class QuantizationDepthNoise(DepthNoiseModel):
    step_m: float  # quantization step in meters

    def __call__(self, d: np.ndarray) -> np.ndarray:
        if self.step_m <= 0:
            return d
        return np.round(d / self.step_m).astype(np.float32) * self.step_m


@dataclass
class DropoutDepthNoise(DepthNoiseModel):
    p: float  # probability to drop measurement
    fill: float = 0.0  # fill value (usually 0)

    def __call__(self, d: np.ndarray) -> np.ndarray:
        if self.p <= 0:
            return d
        mask = np.random.rand(*d.shape) < self.p
        out = d.copy()
        out[mask] = self.fill
        return out


# -----------------------------
# Composition + post rules
# -----------------------------
def apply_noise_chain(
    depth_m: np.ndarray,
    noises: List[DepthNoiseModel],
    *,
    zmax_m: Optional[float] = None,
    nonpositive_to_zero: bool = True,
) -> np.ndarray:
    """Apply noises sequentially, then clamp invalids."""
    x = depth_m.astype(np.float32)
    for n in noises:
        x = n(x)

    # negative / nan / inf handling
    if nonpositive_to_zero:
        x = np.where(np.isfinite(x) & (x > 0.0), x, 0.0).astype(np.float32)
    else:
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)

    # zmax clipping to 0 (simulate no return beyond range)
    if zmax_m is not None and zmax_m > 0:
        x = np.where(x <= zmax_m, x, 0.0).astype(np.float32)
    return x


# -----------------------------
# Depth clamping utility
# -----------------------------
def clamp_depth_to_zmax(
    depth_m: np.ndarray,
    zmax_m: float,
    nonpositive_to_zero: bool = True,
) -> np.ndarray:
    """Clamp depth to [0, zmax]; set values beyond zmax to 0 (simulate no return).
    Non-finite or non-positive values become 0 if nonpositive_to_zero is True.
    """
    x = depth_m.astype(np.float32)
    if nonpositive_to_zero:
        x = np.where(np.isfinite(x) & (x > 0.0), x, 0.0).astype(np.float32)
    else:
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    if zmax_m is not None and zmax_m > 0:
        x = np.where(x <= zmax_m, x, 0.0).astype(np.float32)
    return x
