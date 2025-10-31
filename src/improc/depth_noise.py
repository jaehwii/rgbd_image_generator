# src/improc/depth_noise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# -----------------------------
# Base noise API (mask-aware)
# -----------------------------
class DepthNoiseModel:
    """
    Mask-aware noise operator.
    Each noise must implement `apply(d, valid_mask)` and return (d, valid_mask).
    - d: float32 depth (meters)
    - valid_mask: boolean mask where measurements are currently valid
    The operator may modify `d` only at valid_mask==True, and may also shrink valid_mask
      (e.g., dropout). It must NEVER create new valid pixels outside the mask.
    """

    def apply(
        self,
        d: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class GaussianDepthNoise(DepthNoiseModel):
    """Additive Gaussian noise: d += N(0, sigma_m) on valid pixels."""

    sigma_m: float

    def apply(self, d, valid_mask):
        if self.sigma_m <= 0:
            return d, valid_mask
        n = np.random.normal(0.0, self.sigma_m, size=d.shape).astype(np.float32)
        d = d.copy()
        d[valid_mask] = (d[valid_mask] + n[valid_mask]).astype(np.float32)
        return d, valid_mask


@dataclass
class MultiplicativeDepthNoise(DepthNoiseModel):
    """Multiplicative noise: d *= (1 + N(0, sigma_rel)) on valid pixels."""

    sigma_rel: float

    def apply(self, d, valid_mask):
        if self.sigma_rel <= 0:
            return d, valid_mask
        scale = (1.0 + np.random.normal(0.0, self.sigma_rel, size=d.shape)).astype(
            np.float32
        )
        d = d.copy()
        d[valid_mask] = (d[valid_mask] * scale[valid_mask]).astype(np.float32)
        return d, valid_mask


@dataclass
class QuantizationDepthNoise(DepthNoiseModel):
    """Uniform quantization with step size (meters) on valid pixels."""

    step_m: float

    def apply(self, d, valid_mask):
        if self.step_m <= 0:
            return d, valid_mask
        d = d.copy()
        v = d[valid_mask]
        d[valid_mask] = (
            np.round(v / self.step_m).astype(np.float32) * self.step_m
        ).astype(np.float32)
        return d, valid_mask


@dataclass
class DropoutDepthNoise(DepthNoiseModel):
    """Randomly drop currently-valid measurements with prob p; fill with `fill`."""

    p: float
    fill: float = 0.0

    def apply(self, d, valid_mask):
        if self.p <= 0:
            return d, valid_mask
        drop = (
            np.random.rand(*d.shape) < self.p
        ) & valid_mask  # only drop where currently valid
        if not np.any(drop):
            return d, valid_mask
        d = d.copy()
        d[drop] = self.fill
        valid_mask = valid_mask.copy()
        valid_mask[drop] = False
        return d, valid_mask


# -----------------------------
# Utilities
# -----------------------------
def _initial_valid_mask(d: np.ndarray, zmax_m: Optional[float]) -> np.ndarray:
    is_finite_pos = np.isfinite(d) & (d > 0.0)
    if zmax_m is not None and zmax_m > 0:
        return is_finite_pos & (d <= zmax_m)
    return is_finite_pos


# -----------------------------
# Apply noises
# -----------------------------
def apply_noise_chain(
    depth_m: np.ndarray,
    noises: List[DepthNoiseModel],
    *,
    zmax_m: Optional[float] = None,
    invalid_fill: float = 0.0,
    nonpositive_to_zero: bool = True,
) -> np.ndarray:
    """
    Noise pipeline with robust invalid handling:
      1) Make initial valid mask (isfinite, >0, <=zmax if provided).
      2) Apply each noise with mask-awareness (only valid pixels are modified).
      3) Sanitize: remove non-finite / non-positive; re-apply zmax validity.
      4) Set invalid pixels to `invalid_fill` (default: 0.0).
    """
    d = depth_m.astype(np.float32, copy=True)

    # Step 1: initial validity (zmax included)
    valid = _initial_valid_mask(d, zmax_m)

    # Step 2: run noises
    for n in noises:
        d, valid = n.apply(d, valid)

    # Step 3: sanitize values after noise
    if nonpositive_to_zero:
        is_finite_pos = np.isfinite(d) & (d > 0.0)
    else:
        is_finite_pos = np.isfinite(d)

    if zmax_m is not None and zmax_m > 0:
        valid = is_finite_pos & (d <= zmax_m)
    else:
        valid = is_finite_pos

    # Step 4: finalize invalids
    out = d.copy()
    out[~valid] = invalid_fill
    return out.astype(np.float32)


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
