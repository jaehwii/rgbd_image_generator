# blender_rgbd_render_seq.py
# Render RGB + Depth for a sequence of camera poses.
# - Orientation is computed automatically so the camera looks at the given target_in_world.
# - Room (floor, 4 walls, ceiling) is created with distinct colors for easy orientation debugging.
#
# Usage (headless):
#   blender --background --python blender_rgbd_render_seq.py -- --config /abs/path/scene_example.toml
#

# --- Standard library ---
import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Tuple

# --- Third-party (Blender) ---
import bpy
from mathutils import Matrix, Vector

# --- Local project modules (absolute imports; no sys.path hack) ---
from src.config.config_parser import load_scene_cfg  # must return SceneCfg
from src.config.config_types import CameraIntrinsics, SceneCfg
from src.renderer.blender_object_utils import create_cube
from src.utils.math_utils import look_at_quaternion, make_se3_matrix
from src.utils.summary import RenderSummary

Vec3 = Tuple[float, float, float]
QuatWXYZ = Tuple[float, float, float, float]


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
# Scene utils
# -----------------------------------------------------------------------------


def clear_scene():
    """Reset Blender to an empty scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def world_matrix_evaluated(obj) -> Matrix:
    """Return evaluated world matrix after depsgraph applies modifiers."""
    dg = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(dg).matrix_world.copy()


def set_obj_pose(obj, T_world: Matrix):
    """Assign a 4x4 world transform to an object."""
    obj.matrix_world = T_world


def create_camera_from_intrinsics(name: str, intrinsics: CameraIntrinsics):
    """Create a perspective camera with given intrinsics."""
    cam_data = bpy.data.cameras.new(name=name)
    cam_data.lens = intrinsics.focal_length_mm
    cam_data.clip_start = intrinsics.clip_start
    cam_data.clip_end = intrinsics.clip_end
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.rotation_mode = 'QUATERNION'
    return cam_obj


def create_depth_camera_from_color_camera(color_cam, T_DC: Matrix):
    """Clone color camera and apply T_WD = T_WC @ inv(T_DC)."""
    depth_cam = bpy.data.cameras.new(name='DepthCamera')
    depth_obj = bpy.data.objects.new('DepthCamera', depth_cam)
    bpy.context.collection.objects.link(depth_obj)

    # Copy intrinsics from color camera
    depth_cam.lens = color_cam.data.lens
    depth_cam.sensor_width = color_cam.data.sensor_width
    depth_cam.sensor_height = color_cam.data.sensor_height
    depth_cam.shift_x = color_cam.data.shift_x
    depth_cam.shift_y = color_cam.data.shift_y
    depth_cam.clip_start = color_cam.data.clip_start
    depth_cam.clip_end = color_cam.data.clip_end

    T_WC = world_matrix_evaluated(color_cam)
    T_WD = T_WC @ T_DC.inverted()
    depth_obj.matrix_world = T_WD
    depth_obj.rotation_mode = 'QUATERNION'
    bpy.context.view_layer.update()
    return depth_obj


def create_room(size=(6.0, 6.0, 3.0)):
    """Create a box-like room: floor, ceiling, and 4 colored walls. All normals inward."""
    xlen, ylen, zlen = size

    def make_material(name, rgba):
        m = bpy.data.materials.new(name=name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = rgba
        bsdf.inputs['Roughness'].default_value = 0.8
        return m

    m_floor = make_material('MatFloor', (0.7, 0.7, 0.7, 1.0))  # light gray
    m_ceiling = make_material('MatCeil', (0.92, 0.92, 0.96, 1.0))  # lighter gray
    m_xp = make_material('MatXpos', (0.9, 0.1, 0.1, 1.0))  # reddish
    m_xn = make_material('MatXneg', (0.9, 0.1, 0.9, 1.0))  # purple
    m_yp = make_material('MatYpos', (0.1, 0.90, 0.1, 1.0))  # green
    m_yn = make_material('MatYneg', (0.95, 0.95, 0.10, 1.0))  # yellow

    # Floor (Z=0)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.scale = (xlen, ylen, 0.0)  # set final dimensions in meters
    floor.name = 'RoomFloor'
    floor.data.materials.append(m_floor)

    # Ceiling
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, zlen))
    ceil = bpy.context.active_object
    ceil.scale = (xlen, ylen, 1.0)
    ceil.rotation_euler = (math.pi, 0.0, 0.0)
    ceil.name = 'RoomCeiling'
    ceil.data.materials.append(m_ceiling)

    # +X wall
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(xlen / 2.0, 0.0, zlen / 2.0))
    xp = bpy.context.active_object
    xp.scale = (zlen, ylen, 1.0)
    xp.rotation_euler = (0.0, -math.pi / 2.0, 0.0)
    xp.name = 'RoomWallXpos'
    xp.data.materials.append(m_xp)

    # -X wall
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-xlen / 2.0, 0.0, zlen / 2.0))
    xn = bpy.context.active_object
    xn.scale = (zlen, ylen, 1.0)
    xn.rotation_euler = (0.0, math.pi / 2.0, 0.0)
    xn.name = 'RoomWallXneg'
    xn.data.materials.append(m_xn)

    # +Y wall
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, ylen / 2.0, zlen / 2.0))
    yp = bpy.context.active_object
    yp.scale = (xlen, zlen, 1.0)
    yp.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    yp.name = 'RoomWallYpos'
    yp.data.materials.append(m_yp)

    # -Y wall
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, -ylen / 2.0, zlen / 2.0))
    yn = bpy.context.active_object
    yn.scale = (xlen, zlen, 1.0)
    yn.rotation_euler = (-math.pi / 2.0, 0.0, 0.0)
    yn.name = 'RoomWallYneg'
    yn.data.materials.append(m_yn)


def create_key_light(location=(3.0, -3.0, 5.0), power=1000.0):
    """Add a simple area light."""
    light_data = bpy.data.lights.new(name='KeyLight', type='AREA')
    light_data.energy = power
    light_data.size = 2.0
    light_obj = bpy.data.objects.new('KeyLight', light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = Vector(location)
    return light_obj


def set_render_settings(
    width: int,
    height: int,
    engine: str = 'CYCLES',
    quality: str = 'balanced',
    device: str = 'GPU',
):
    """
    Configure render engine and quality-speed tradeoffs.
    engine: "CYCLES" | "BLENDER_EEVEE"
    quality: "draft" | "balanced" | "high"
    device: "CPU" | "GPU"
    """
    scene = bpy.context.scene

    # --- Common resolution & color management ---
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.display_settings.display_device = 'sRGB'
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'None'

    req_engine = engine.upper().strip()
    req_device = device.upper().strip()

    if req_engine == 'CYCLES':
        scene.render.engine = 'CYCLES'

        # ---- GPU path (robust) ----
        if req_device == 'GPU':
            try:
                prefs = bpy.context.preferences
                cprefs = prefs.addons['cycles'].preferences

                # 1) Pick first available backend
                for backend in ['OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL']:
                    try:
                        cprefs.compute_device_type = backend
                        break
                    except Exception:
                        continue

                # 2) Refresh device list (version-safe)
                try:
                    cprefs.get_devices()
                except Exception:
                    try:
                        cprefs.refresh_devices()
                    except Exception:
                        pass

                # 3) Enable all devices discovered
                enabled_devices = []
                try:
                    for d in cprefs.devices:
                        d.use = True
                        enabled_devices.append(
                            f'{d.type}:{getattr(d, "name", "Unknown")}'
                        )
                except Exception:
                    enabled_devices = []

                # 4) If any GPU-like device is enabled, use GPU; else fallback to CPU
                gpu_enabled = any(
                    x.startswith(('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'))
                    for x in enabled_devices
                )
                if gpu_enabled:
                    scene.cycles.device = 'GPU'
                else:
                    print('[WARN] No usable Cycles GPU devices; falling back to CPU.')
                    scene.cycles.device = 'CPU'
                    # Honest backend state if none usable
                    try:
                        cprefs.compute_device_type = 'NONE'
                    except Exception:
                        pass

            except Exception as e:
                print(f'[WARN] GPU setup failed, falling back to CPU: {e}')
                scene.cycles.device = 'CPU'
                # try to mark backend none for clarity
                try:
                    bpy.context.preferences.addons[
                        'cycles'
                    ].preferences.compute_device_type = 'NONE'
                except Exception:
                    pass

        else:
            # CPU requested
            scene.cycles.device = 'CPU'
            try:
                bpy.context.preferences.addons[
                    'cycles'
                ].preferences.compute_device_type = 'NONE'
            except Exception:
                pass

        # ---- Quality presets ----
        presets = {
            'draft': dict(
                samples=32,
                use_adaptive_sampling=True,
                adaptive_threshold=0.05,
                denoise=True,
                max_bounces=3,
                diffuse_bounces=1,
                glossy_bounces=1,
                transmission_bounces=2,
                transparent_max_bounces=4,
                caustics_reflective=False,
                caustics_refractive=False,
                use_simplify=True,
                simplify_subdivision=0,
                texture_limit='1024',
            ),
            'balanced': dict(
                samples=128,
                use_adaptive_sampling=True,
                adaptive_threshold=0.02,
                denoise=True,
                max_bounces=6,
                diffuse_bounces=2,
                glossy_bounces=2,
                transmission_bounces=4,
                transparent_max_bounces=8,
                caustics_reflective=False,
                caustics_refractive=False,
                use_simplify=False,
                simplify_subdivision=0,
                texture_limit='OFF',
            ),
            'high': dict(
                samples=512,
                use_adaptive_sampling=True,
                adaptive_threshold=0.01,
                denoise=True,
                max_bounces=12,
                diffuse_bounces=4,
                glossy_bounces=4,
                transmission_bounces=6,
                transparent_max_bounces=16,
                caustics_reflective=False,
                caustics_refractive=False,
                use_simplify=False,
                simplify_subdivision=0,
                texture_limit='OFF',
            ),
        }
        p = presets.get(quality, presets['balanced'])

        scene.cycles.samples = p['samples']
        scene.cycles.use_adaptive_sampling = p['use_adaptive_sampling']
        scene.cycles.adaptive_threshold = p['adaptive_threshold']
        scene.cycles.max_bounces = p['max_bounces']
        scene.cycles.diffuse_bounces = p['diffuse_bounces']
        scene.cycles.glossy_bounces = p['glossy_bounces']
        scene.cycles.transmission_bounces = p['transmission_bounces']
        scene.cycles.transparent_max_bounces = p['transparent_max_bounces']
        scene.cycles.caustics_reflective = p['caustics_reflective']
        scene.cycles.caustics_refractive = p['caustics_refractive']

        try:
            scene.view_layers[0].cycles.use_denoising = p['denoise']
        except Exception:
            pass

        scene.render.use_simplify = p['use_simplify']
        scene.render.simplify_subdivision = p['simplify_subdivision']
        try:
            scene.render.simplify_texture_limit = p['texture_limit']
        except Exception:
            pass

    else:
        # EEVEE
        scene.render.engine = 'BLENDER_EEVEE'
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'

    # ---- Print final state ----
    print('[INFO] Render engine   :', scene.render.engine)

    if scene.render.engine == 'CYCLES':
        print('[INFO] Cycles device   :', getattr(scene.cycles, 'device', 'UNKNOWN'))
        backend = 'UNKNOWN'
        enabled = []
        try:
            cprefs = bpy.context.preferences.addons['cycles'].preferences
            backend = getattr(cprefs, 'compute_device_type', 'UNKNOWN')
            # refresh list to report accurately
            try:
                cprefs.get_devices()
            except Exception:
                try:
                    cprefs.refresh_devices()
                except Exception:
                    pass
            for d in getattr(cprefs, 'devices', []):
                if getattr(d, 'use', False):
                    enabled.append(f'{d.type}:{getattr(d, "name", "Unknown")}')
        except Exception:
            pass
        print('[INFO] Compute backend :', backend)
        print('[INFO] Enabled devices :', ', '.join(enabled) if enabled else '(none)')
    else:
        print('[INFO] Cycles device   : N/A')
        print('[INFO] Compute backend : N/A')
        print('[INFO] Enabled devices : (none)')


# -----------------------------------------------------------------------------
# Depth compositor
# -----------------------------------------------------------------------------


def setup_depth_compositor(depth_exr_path: str, depth_viz_png_path: str, zmax_m: float):
    """Configure compositor to write Z to EXR (raw) and PNG16 (viz)."""
    scene = bpy.context.scene
    scene.use_nodes = True

    # 1) activate z pass first (blender 4.x)
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    bpy.context.view_layer.update()

    # 2) initialize node tree
    ntree = scene.node_tree
    ntree.links.clear()
    ntree.nodes.clear()

    # 3) Create RLayers node
    n_rl = ntree.nodes.new('CompositorNodeRLayers')
    n_rl.location = (-300, 0)

    # If depth socket does not exist, recreate node to force
    if 'Depth' not in [s.name for s in n_rl.outputs]:
        ntree.nodes.remove(n_rl)
        bpy.context.view_layer.update()
        n_rl = ntree.nodes.new('CompositorNodeRLayers')
        n_rl.location = (-300, 0)

    # 4) Export EXR (raw meters)
    n_exr = ntree.nodes.new('CompositorNodeOutputFile')
    n_exr.label = 'DepthEXR'
    n_exr.format.file_format = 'OPEN_EXR'
    n_exr.format.color_depth = '32'
    n_exr.base_path = os.path.dirname(depth_exr_path)
    base_exr = os.path.splitext(os.path.basename(depth_exr_path))[0]
    n_exr.file_slots[0].path = base_exr + '_'

    # 5) Visualize as PNG16
    n_map = ntree.nodes.new('CompositorNodeMapRange')
    n_map.inputs[1].default_value = 0.0
    n_map.inputs[2].default_value = float(zmax_m)
    n_map.inputs[3].default_value = 0.0
    n_map.inputs[4].default_value = 1.0
    n_map.use_clamp = True

    n_png = ntree.nodes.new('CompositorNodeOutputFile')
    n_png.label = 'DepthVizPNG'
    n_png.format.file_format = 'PNG'
    n_png.format.color_mode = 'BW'
    n_png.format.color_depth = '16'
    n_png.base_path = os.path.dirname(depth_viz_png_path)
    base_png = os.path.splitext(os.path.basename(depth_viz_png_path))[0]
    n_png.file_slots[0].path = base_png + '_'

    # 6) Link (Ensure existence of depth socket now)
    ntree.links.new(n_rl.outputs['Depth'], n_exr.inputs[0])
    ntree.links.new(n_rl.outputs['Depth'], n_map.inputs['Value'])
    ntree.links.new(n_map.outputs['Value'], n_png.inputs[0])

    return n_exr, n_png


def setup_mask_compositor(mask_png_path: str, object_index: int = 1):
    """Build compositor graph to output an 8-bit single-channel ID mask PNG."""
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nt.links.clear()
    nt.nodes.clear()

    n_rl = nt.nodes.new('CompositorNodeRLayers')
    n_rl.location = (-400, 0)

    # Enable Object Index pass
    view_layer = scene.view_layers[0]
    view_layer.use_pass_object_index = True

    n_id = nt.nodes.new('CompositorNodeIDMask')
    n_id.index = object_index
    n_id.location = (-150, 0)

    n_out = nt.nodes.new('CompositorNodeOutputFile')
    n_out.label = 'ObjMaskPNG'
    n_out.format.file_format = 'PNG'
    n_out.format.color_mode = 'BW'  # single-channel
    n_out.format.color_depth = '8'  # 0 or 255
    n_out.base_path = os.path.dirname(mask_png_path)
    base_png = os.path.splitext(os.path.basename(mask_png_path))[0]
    n_out.file_slots[0].path = base_png + '_'  # will produce <name>_0001.png
    n_out.location = (150, 0)

    # Links: RLayers "IndexOB" -> IDMask -> FileOutput
    nt.links.new(n_rl.outputs['IndexOB'], n_id.inputs['ID value'])
    nt.links.new(n_id.outputs['Alpha'], n_out.inputs[0])

    return n_out


def render_obj_mask(mask_png_path: str, cam_obj, object_index: int = 1):
    """Render an object mask (from depth camera view) to a single-channel PNG."""
    scene = bpy.context.scene
    prev_camera = scene.camera
    prev_use_nodes = scene.use_nodes
    prev_filepath = scene.render.filepath
    tmp_path = os.path.join(os.path.dirname(mask_png_path), '__tmp_mask_main.png')

    try:
        scene.camera = cam_obj
        scene.render.filepath = tmp_path
        setup_mask_compositor(mask_png_path, object_index=object_index)
        bpy.ops.render.render(write_still=True)
        finalize_file_output(mask_png_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            print(f'[WARN] failed to remove tmp main render: {e}')
        scene.camera = prev_camera
        scene.use_nodes = prev_use_nodes
        scene.render.filepath = prev_filepath


def finalize_file_output(target_path: str) -> bool:
    """Rename compositor's <name>_0001.ext to target_path (overwrites if exists)."""
    import shutil

    base_dir = os.path.dirname(target_path)
    base_name = os.path.splitext(os.path.basename(target_path))[0]
    ext = os.path.splitext(target_path)[1]
    src = os.path.join(base_dir, base_name + '_0001' + ext)
    if os.path.exists(src):
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.move(src, target_path)
        return True
    return False


def render_rgb(out_path: str, cam_obj):
    """Render an RGB image to file using the given camera."""
    scene = bpy.context.scene
    scene.camera = cam_obj
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


def render_depth(depth_exr_path: str, depth_viz_png_path: str, cam_obj, zmax_m: float):
    """Render depth EXR + 16-bit PNG visualization via compositor."""
    scene = bpy.context.scene
    prev_camera = scene.camera
    prev_use_nodes = scene.use_nodes
    prev_filepath = scene.render.filepath
    tmp_path = os.path.join(os.path.dirname(depth_exr_path), '__tmp_depth_main.png')
    try:
        scene.camera = cam_obj
        scene.render.filepath = tmp_path
        setup_depth_compositor(depth_exr_path, depth_viz_png_path, zmax_m)
        bpy.ops.render.render(write_still=True)
        finalize_file_output(depth_exr_path)
        finalize_file_output(depth_viz_png_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            print(f'[WARN] failed to remove tmp main render: {e}')
        scene.camera = prev_camera
        scene.use_nodes = prev_use_nodes
        scene.render.filepath = prev_filepath


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def ensure_dirs(base_out: str | Path, scene_id: str) -> Path:
    """Create output subfolders and return the scene root path as Path."""
    base = Path(base_out).expanduser().resolve()
    scene_root = base / scene_id

    # Create required subdirectories
    for name in ('rgb', 'depth_exr', 'depth_viz', 'poses', 'mask'):
        (scene_root / name).mkdir(parents=True, exist_ok=True)

    return scene_root


def write_matrix_txt(path: str, M: Matrix):
    """Write a 4x4 matrix (row-major) as 4 lines of space-separated floats."""
    with open(path, 'w') as f:
        for r in range(4):
            f.write(' '.join(f'{M[r][c]:.9f}' for c in range(4)) + '\n')


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

    # --- Summary container (struct처럼 사용)
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
        d_exr_path = os.path.join(scene_root, 'depth_exr', f'{stem}.exr')
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
        print(f'[INFO] Rendering Depth (EXR/PNG): {d_exr_path}, {d_viz_path}')
        render_depth(d_exr_path, d_viz_path, cam_depth, cfg.render.zmax_m)
        print(f'[INFO] Rendering Mask: {mask_path}')
        render_obj_mask(mask_path, cam_depth, object_index=1)

        # Manifest row
        man_rows.append(
            {
                'frame': k,
                'rgb': os.path.relpath(rgb_path, scene_root),
                'depth_exr': os.path.relpath(d_exr_path, scene_root),
                'depth_viz': os.path.relpath(d_viz_path, scene_root),
                'mask': os.path.relpath(mask_path, scene_root),
                'T_WC_txt': f'poses/T_WC_{stem}.txt',
                'T_WD_txt': f'poses/T_WD_{stem}.txt',
                'T_WO_txt': f'poses/T_WO_{stem}.txt',
                'p_WC': f'{cam_ext.p_WC}',
                'p_W_target': f'{cam_ext.p_W_target}',
            }
        )
        summary.add_frame_num(1)
        summary.stop_frame_timer()

    # Write manifest CSV
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(man_rows[0].keys()))
        writer.writeheader()
        writer.writerows(man_rows)

    print('[INFO] Done. Summary:')
    summary.print()


if __name__ == '__main__':
    main()
