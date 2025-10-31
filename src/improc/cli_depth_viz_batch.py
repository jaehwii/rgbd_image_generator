"""Batch depth viz driven by main config only.
Usage:
  python -m src.improc.cli_depth_viz_batch --config /abs/path/scene.toml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config.config_parser import load_config
from src.improc.depth_noise import clamp_depth_to_zmax, read_exr_depth
from src.improc.depth_viz import visualize_exr_to_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to scene .toml')
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f'[VIZ] loaded config: {args.config}')

    zmax = float(getattr(cfg.render, 'zmax_m', 0.0) or 0.0)

    scene_root = (Path(cfg.render.out_dir) / cfg.render.scene_id).resolve()
    manifest = scene_root / 'manifest.csv'
    if not manifest.exists():
        raise FileNotFoundError(f'manifest not found: {manifest}')

    with manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        # Groundtruth
        exr_gt_rel = r.get('depth_exr_gt')
        viz_gt_rel = r.get('depth_viz_gt')
        if exr_gt_rel is None or viz_gt_rel is None:
            raise KeyError('manifest must contain both depth_exr_gt and depth_viz_gt')
        exr_gt_abs = (scene_root / exr_gt_rel).resolve()
        viz_gt_abs = (scene_root / viz_gt_rel).resolve()
        print(f'[POST] GT EXR -> PNG16 | {exr_gt_abs} -> {viz_gt_abs} | zmax={zmax}')

        depth_gt = read_exr_depth(str(exr_gt_abs))
        if zmax > 0.0:
            depth_gt = clamp_depth_to_zmax(depth_gt, zmax)
        visualize_exr_to_png(depth_gt, str(viz_gt_abs), zmax, invalid_color=(0, 180, 0))

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
        if zmax > 0.0:
            depth_noisy = clamp_depth_to_zmax(depth_noisy, zmax)
        visualize_exr_to_png(
            depth_noisy, str(viz_noisy_abs), zmax, invalid_color=(0, 180, 0)
        )


if __name__ == '__main__':
    main()
