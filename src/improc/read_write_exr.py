import os

import cv2
import numpy as np


def read_exr_depth(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # returns float32 for EXR
    if img is None:
        raise IOError(f'Failed to read EXR: {path}')
    if img.ndim == 3:
        img = img[..., 0]  # take first channel
    return img.astype(np.float32)


def write_exr_depth(path: str, depth: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, depth.astype(np.float32))
    if not ok:
        raise IOError(f'Failed to write EXR: {path}')
