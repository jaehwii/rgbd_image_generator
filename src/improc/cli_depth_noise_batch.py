# src/improc/cli_depth_noise_batch.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config.config_parser import load_config
from src.improc.depth_noise import (
    DropoutDepthNoise,
    GaussianDepthNoise,
    MultiplicativeDepthNoise,
    QuantizationDepthNoise,
    apply_noise_chain,
    clamp_depth_to_zmax,
    read_exr_depth,
    write_exr_depth,
)
from src.improc.depth_viz import visualize_exr_to_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to scene .toml')
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f'[NOISE] loaded config: {args.config}')
    if not cfg.noise.enabled:
        print('[NOISE] noise.enabled = false → skip')
        return

    zmax = float(getattr(cfg.render, 'zmax_m', 0.0) or 0.0)

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

    print(
        f'[NOISE] enabled={cfg.noise.enabled} | chain={[type(n).__name__ for n in noises]}'
    )

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

        d_gt = read_exr_depth(str(exr_gt_abs))
        if zmax > 0.0:
            d_gt = clamp_depth_to_zmax(d_gt, zmax)
            write_exr_depth(str(exr_gt_abs), d_gt)

        # apply noise on clamped GT
        d_noisy = apply_noise_chain(d_gt, noises, zmax_m=zmax, nonpositive_to_zero=True)

        import numpy as np

        # Before/after clamp stats (apply_noise_chain already clamps; adjust if you clamp separately)
        diff = np.abs(d_noisy - d_gt)
        print(
            f'[DEBUG] diff(mean)={float(diff.mean()):.6e} diff(max)={float(diff.max()):.6e}'
        )

        # Symmetric clamp after noise as well
        if zmax > 0.0:
            d_noisy = clamp_depth_to_zmax(d_noisy, zmax)

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
