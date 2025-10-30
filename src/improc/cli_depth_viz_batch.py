"""Batch depth viz: read manifest.csv and make PNG16 from depth EXR using system Python.
Usage:
  python -m src.improc.cli_depth_viz_batch --manifest /abs/scene_root/manifest.csv --scene-root /abs/scene_root
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from src.improc.depth_viz import visualize_exr_to_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='Path to manifest.csv')
    ap.add_argument(
        '--scene-root',
        required=True,
        help='Absolute path to scene root (joins relative paths in manifest)',
    )
    args = ap.parse_args()

    scene_root = Path(args.scene_root).resolve()
    man_path = Path(args.manifest).resolve()
    if not man_path.exists():
        raise FileNotFoundError(f'manifest not found: {man_path}')

    with man_path.open(newline='') as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        exr_rel = r['depth_exr']
        viz_rel = r['depth_viz']
        zmax = float(r['zmax'])
        exr_abs = (scene_root / exr_rel).resolve()
        viz_abs = (scene_root / viz_rel).resolve()
        os.makedirs(viz_abs.parent, exist_ok=True)
        print(f'[POST] EXR -> PNG16 | {exr_abs} -> {viz_abs} | zmax={zmax}')
        visualize_exr_to_png(str(exr_abs), str(viz_abs), zmax)


if __name__ == '__main__':
    main()
