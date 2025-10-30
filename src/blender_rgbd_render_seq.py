# blender_rgbd_render_seq.py
# Render RGB + Depth for a sequence of camera poses.
# - Orientation is computed automatically so the camera looks at the given target_in_world.
# - Room (floor, 4 walls, ceiling) is created with distinct colors for easy orientation debugging.
#
# Usage (headless):
#   blender --background --python blender_rgbd_render_seq.py -- --config /abs/path/scene_example.toml
#

# --- Standard library ---
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

# --- Third-party (Blender) ---
import bpy
from mathutils import Vector

# --- Local project modules ---
from src.blender.object_utils import create_cube
from src.blender.render_ops import (
    render_depth_exr,
    render_obj_mask,
    render_rgb,
    set_render_settings,
)
from src.blender.scene_utils import (
    clear_scene,
    create_camera_from_intrinsics,
    create_key_light,
    create_room,
    set_obj_pose,
    world_matrix_evaluated,
)
from src.config.config_parser import load_scene_cfg  # must return SceneCfg
from src.config.config_types import SceneCfg
from src.utils.io_utils import ensure_dirs, write_matrix_txt
from src.utils.math_utils import look_at_quaternion, make_se3_matrix
from src.utils.summary import RenderSummary


def _parse_args_from_blender(argv: list[str]) -> argparse.Namespace:
    # Only parse args after the first "--"
    if '--' in argv:
        argv = argv[argv.index('--') + 1 :]
    else:
        argv = []  # allow nice error if user forgot to pass args
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to scene .toml')
    return ap.parse_args(argv)


_ARGS = _parse_args_from_blender(sys.argv)
config_path = os.path.abspath(_ARGS.config)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# -----------------------------------------------------------------------------
# CLI args
# -----------------------------------------------------------------------------
def parse_args(project_root: Path) -> Path:
    """Parse --config after Blender's '--' separator; return absolute Path."""
    # Blender passes script args after '--'. Extract them safely.
    argv = sys.argv[sys.argv.index('--') + 1 :] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description='RGB-D sequence renderer')
    parser.add_argument(
        '--config',
        default=str(project_root / 'config' / 'scene_example.toml'),
        help='Path to scene config TOML file',
    )
    args = parser.parse_args(argv)
    return Path(args.config).resolve()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    """Entry point called by Blender."""
    # Resolve project root relative to this file (no sys.path mutation)
    script_dir = Path(__file__).resolve().parent
    project_root = (script_dir / '..').resolve()

    cfg_path = parse_args(project_root)
    cfg: SceneCfg = load_scene_cfg(str(cfg_path), use_toml=True)

    # Prepare output directories and manifest
    scene_root = ensure_dirs(cfg.render.out_dir, cfg.render.scene_id)
    manifest_path = scene_root / 'manifest.csv'
    man_rows = []

    # --- Summary container
    summary = RenderSummary(
        scene_id=cfg.render.scene_id,
        scene_root=scene_root,
    )

    # Build scene
    clear_scene()
    create_key_light(location=(2.5, -2.5, 2.5), power=400.0)
    create_key_light(location=(-2.5, 2.5, 2.5), power=400.0)
    create_room(size=(6.0, 6.0, 3.0))

    # Object
    cube = create_cube(size=cfg.obj.size, T_WO=cfg.obj.T_WO, object_index=1)

    # Cameras
    cam_color = create_camera_from_intrinsics(
        name='ColorCamera',
        intrinsics=cfg.rig.color_intrinsics,
    )
    cam_depth = create_camera_from_intrinsics(
        name='ColorCamera',
        intrinsics=cfg.rig.depth_intrinsics,
    )

    # Depth rig relation (depth -> color)
    T_DC = make_se3_matrix(cfg.rig.T_DC.p, cfg.rig.T_DC.q_wxyz)

    # Common render settings
    set_render_settings(
        cfg.render.width,
        cfg.render.height,
        engine='CYCLES',
        quality='draft',
        device='GPU',
    )

    summary.engine_name = bpy.context.scene.render.engine
    summary.device_name = getattr(bpy.context.scene.cycles, 'device', 'CPU')

    # Iterate camera extrinsics (position + target), orientation computed automatically
    for k, cam_ext in enumerate(cfg.seq.camera_extrinsics):
        summary.start_frame_timer()

        eye = Vector(cam_ext.p_WC)
        target = Vector(cam_ext.p_W_target)

        # Orientation: look-at so that local -Z faces the target
        q = look_at_quaternion(eye, target)

        # Set color camera pose
        T_WC = make_se3_matrix(eye, (q.w, q.x, q.y, q.z))
        set_obj_pose(cam_color, T_WC)
        bpy.context.scene.camera = cam_color
        bpy.context.view_layer.update()

        T_WD = T_WC @ T_DC.inverted()
        set_obj_pose(cam_depth, T_WD)
        bpy.context.view_layer.update()

        # Filenames
        stem = f'frame_{k:04d}'
        rgb_path = os.path.join(scene_root, 'rgb', f'{stem}.png')
        d_exr_gt_path = os.path.join(scene_root, 'depth_exr_gt', f'{stem}.exr')
        d_exr_noisy_path = os.path.join(scene_root, 'depth_exr_noisy', f'{stem}.exr')
        d_viz_path = os.path.join(scene_root, 'depth_viz', f'{stem}.png')
        mask_path = os.path.join(scene_root, 'mask', f'{stem}.png')

        # Save poses
        T_WC_eval = world_matrix_evaluated(cam_color)
        T_WD_eval = world_matrix_evaluated(cam_depth)
        write_matrix_txt(
            os.path.join(scene_root, 'poses', f'T_WC_{stem}.txt'), T_WC_eval
        )
        write_matrix_txt(
            os.path.join(scene_root, 'poses', f'T_WD_{stem}.txt'), T_WD_eval
        )
        write_matrix_txt(
            os.path.join(scene_root, 'poses', f'T_WO_{stem}.txt'),
            world_matrix_evaluated(cube),
        )

        # Render
        print(f'[INFO] Rendering RGB: {rgb_path}')
        render_rgb(rgb_path, cam_color)
        print(f'[INFO] Rendering GT Depth EXR only: {d_exr_gt_path}')
        render_depth_exr(d_exr_gt_path, cam_depth)
        print(f'[INFO] Rendering Mask: {mask_path}')
        render_obj_mask(mask_path, cam_depth, object_index=1)

        # Manifest row
        man_rows.append(
            {
                'frame': k,
                'rgb': os.path.relpath(rgb_path, scene_root),
                'depth_exr_gt': os.path.relpath(d_exr_gt_path, scene_root),
                'depth_exr_noisy': os.path.relpath(d_exr_noisy_path, scene_root),
                'depth_viz': os.path.relpath(d_viz_path, scene_root),
                'mask': os.path.relpath(mask_path, scene_root),
                'T_WC_txt': f'poses/T_WC_{stem}.txt',
                'T_WD_txt': f'poses/T_WD_{stem}.txt',
                'T_WO_txt': f'poses/T_WO_{stem}.txt',
                'p_WC': f'{cam_ext.p_WC}',
                'p_W_target': f'{cam_ext.p_W_target}',
                'zmax': f'{cfg.render.zmax_m}',
            }
        )
        summary.add_frame_num(1)
        summary.stop_frame_timer()

    # Write manifest CSV
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(man_rows[0].keys()))
        writer.writeheader()
        writer.writerows(man_rows)

    # --- System Python postprocess ---
    # 1) Add noise & write noisy EXR + viz (noisy-based)
    # 2) (Optional) Re-run viz batch which prefers depth_exr_noisy if present.
    sys_py = os.environ.get('SYS_PY', 'python3')

    # Normalize environment and working directory so that 'src' is always importable
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        PROJECT_ROOT
        if 'PYTHONPATH' not in env or not env['PYTHONPATH']
        else f'{PROJECT_ROOT}:{env["PYTHONPATH"]}'
    )
    cwd = PROJECT_ROOT  # Execute from the project root

    noise_cmd = [
        sys_py,
        '-m',
        'src.improc.cli_depth_noise_batch',
        '--config',
        str(config_path),
    ]
    print(
        f'[INFO] Depth noise (system Python):\n  {" ".join(shlex.quote(c) for c in noise_cmd)}'
    )
    res = subprocess.run(noise_cmd, env=env, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print('[ERR] noise stderr:\n' + (res.stderr or '(empty)'))
        print('[ERR] noise stdout:\n' + (res.stdout or '(empty)'))
        print('[WARN] noise postprocess failed; you can run it manually later.')
    else:
        print('[OK ] noise done.')

    viz_cmd = [
        sys_py,
        '-m',
        'src.improc.cli_depth_viz_batch',
        '--manifest',
        str(manifest_path),
        '--scene-root',
        str(scene_root),
    ]
    print(
        f'[INFO] Depth viz (prefer noisy):\n  {" ".join(shlex.quote(c) for c in viz_cmd)}'
    )
    res = subprocess.run(viz_cmd, env=env, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print('[ERR] viz stderr:\n' + (res.stderr or '(empty)'))
        print('[ERR] viz stdout:\n' + (res.stdout or '(empty)'))
        print('[WARN] depth_viz postprocess failed; you can run it manually later.')
    else:
        print('[OK ] viz done.')


if __name__ == '__main__':
    main()
