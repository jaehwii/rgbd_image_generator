"""Batch depth viz: read manifest.csv and make PNG16 from depth EXR using system Python.
Usage:
  python -m src.improc.cli_depth_viz_batch --manifest /abs/scene_root/manifest.csv --scene-root /abs/scene_root
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.improc.depth_noise import read_exr_depth
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
        zmax = float(r['zmax'])

        # Groundtruth
        exr_gt_rel = r.get('depth_exr_gt')
        viz_gt_rel = r.get('depth_viz_gt')
        if exr_gt_rel is None or viz_gt_rel is None:
            raise KeyError('manifest must contain both depth_exr_gt and depth_viz_gt')
        exr_gt_abs = (scene_root / exr_gt_rel).resolve()
        viz_gt_abs = (scene_root / viz_gt_rel).resolve()
        print(f'[POST] GT EXR -> PNG16 | {exr_gt_abs} -> {viz_gt_abs} | zmax={zmax}')
        depth_gt = read_exr_depth(str(exr_gt_abs))
        visualize_exr_to_png(depth_gt, str(viz_gt_abs), zmax)

        # Noisy
        exr_noisy_rel = r.get('depth_exr_noisy')
        viz_noisy_rel = r.get('depth_viz_noisy')
        if exr_noisy_rel is None or viz_noisy_rel is None:
            raise KeyError(
                'manifest must contain both depth_exr_noisy and depth_viz_noisy'
            )
        exr_noisy_abs = (scene_root / exr_noisy_rel).resolve()
        viz_noisy_abs = (scene_root / viz_noisy_rel).resolve()
        print(
            f'[POST] NOISY EXR -> PNG16 | {exr_noisy_abs} -> {viz_noisy_abs} | zmax={zmax}'
        )
        depth_noisy = read_exr_depth(str(exr_noisy_abs))
        visualize_exr_to_png(depth_noisy, str(viz_noisy_abs), zmax)


if __name__ == '__main__':
    main()
