# src/improc/cli_depth_noise_batch.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config.config_parser import load_scene_cfg
from src.improc.depth_noise import (
    DropoutDepthNoise,
    GaussianDepthNoise,
    MultiplicativeDepthNoise,
    QuantizationDepthNoise,
    apply_noise_chain,
    read_exr_depth,
    write_exr_depth,
)
from src.improc.depth_viz import visualize_exr_to_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to scene .toml')
    args = ap.parse_args()

    cfg = load_scene_cfg(args.config)
    if not cfg.noise.enabled:
        print('[NOISE] noise.enabled = false → skip')
        return

    # scene root & manifest
    scene_root = (Path(cfg.render.out_dir) / cfg.render.scene_id).resolve()
    manifest = scene_root / 'manifest.csv'
    if not manifest.exists():
        raise FileNotFoundError(f'manifest not found: {manifest}')

    # load rows
    with manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))

    # build noise chain from config
    noises = []
    ncfg = cfg.noise
    if ncfg.gaussian.enabled and ncfg.gaussian.sigma_m > 0:
        noises.append(GaussianDepthNoise(ncfg.gaussian.sigma_m))
    if ncfg.multiplicative.enabled and ncfg.multiplicative.sigma_rel > 0:
        noises.append(MultiplicativeDepthNoise(ncfg.multiplicative.sigma_rel))
    if ncfg.quantization.enabled and ncfg.quantization.step_m > 0:
        noises.append(QuantizationDepthNoise(ncfg.quantization.step_m))
    if ncfg.dropout.enabled and ncfg.dropout.p > 0:
        noises.append(DropoutDepthNoise(ncfg.dropout.p, ncfg.dropout.fill))

    for r in rows:
        exr_gt_rel = r.get('depth_exr_gt')
        exr_noisy_rel = r.get('depth_exr_noisy')
        viz_noisy_rel = r.get('depth_viz_noisy')
        viz_gt_rel = r.get('depth_viz_gt')
        if None in (exr_gt_rel, exr_noisy_rel, viz_noisy_rel, viz_gt_rel):
            raise KeyError(
                'manifest must contain depth_exr_gt, depth_exr_noisy, depth_viz_noisy, and depth_viz_gt'
            )
        exr_gt_abs = (scene_root / exr_gt_rel).resolve()
        exr_noisy_abs = (scene_root / exr_noisy_rel).resolve()
        viz_noisy_abs = (scene_root / viz_noisy_rel).resolve()
        viz_gt_abs = (scene_root / viz_gt_rel).resolve()
        zmax = float(r.get('zmax', cfg.render.zmax_m or 0.0))

        # read GT, apply noise, clamp
        d_gt = read_exr_depth(str(exr_gt_abs))
        d_noisy = apply_noise_chain(d_gt, noises, zmax_m=zmax, nonpositive_to_zero=True)

        # write noisy exr
        write_exr_depth(str(exr_noisy_abs), d_noisy)
        # viz from NOISY
        visualize_exr_to_png(d_noisy, str(viz_noisy_abs), zmax)
        # viz from GROUNDTRUTH
        visualize_exr_to_png(d_gt, str(viz_gt_abs), zmax)

        print(
            f'[NOISE] {Path(exr_gt_rel).name}: noisy_exr -> {exr_noisy_rel}, viz_noisy -> {viz_noisy_rel}, viz_gt -> {viz_gt_rel}'
        )


if __name__ == '__main__':
    main()
