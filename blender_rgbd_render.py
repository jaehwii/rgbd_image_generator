# blender_rgbd_render.py
# Run:
#   blender --background --python blender_rgbd_render.py -- \
#     --out /path/to/cube_rgb.png --w 640 --h 480
#
# To see the GUI, remove "--background".

import math
import os
import shutil
import sys

import bpy
from mathutils import Euler, Matrix, Vector


# -----------------------
# CLI args (after "--")
# -----------------------
def parse_args():
    out_path = os.path.abspath('./cube_rgb.png')
    width = 640
    height = 480
    zmax_m = 10.0  # map [0, zmax] to [0, 1]

    if '--' in sys.argv:
        idx = sys.argv.index('--') + 1
        argv = sys.argv[idx:]
        i = 0
        while i < len(argv):
            if argv[i] == '--out' and i + 1 < len(argv):
                out_path = os.path.abspath(argv[i + 1])
                i += 2
            elif argv[i] == '--w' and i + 1 < len(argv):
                width = int(argv[i + 1])
                i += 2
            elif argv[i] == '--h' and i + 1 < len(argv):
                height = int(argv[i + 1])
                i += 2
            elif argv[i] == '--zmax' and i + 1 < len(argv):
                zmax_m = float(argv[i + 1])
                i += 2
            else:
                i += 1
    return out_path, width, height, zmax_m


# -----------------------
# Utils
# -----------------------
def clear_scene():
    """Reset to factory settings with an empty scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def world_matrix_evaluated(obj):
    """Return the evaluated world matrix (after modifiers/depsgraph)."""
    dg = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(dg).matrix_world.copy()


def create_camera(
    name='Camera', location=(2.5, -2.5, 2.0), target=(0, 0, 0.5), focal_length_mm=50.0
):
    """Create a camera, place it, and point it to the target."""
    cam_data = bpy.data.cameras.new(name=name)
    cam_data.lens = focal_length_mm
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = Vector(location)
    # Blender camera points along local -Z with +Y as up.
    direction = Vector(target) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = rot_quat.to_euler('XYZ')

    # Set as the active camera.
    bpy.context.scene.camera = cam_obj

    # Conservative clipping range.
    cam_obj.data.clip_start = 0.01
    cam_obj.data.clip_end = 100.0
    return cam_obj


def create_light(name='KeyLight', location=(3.0, -3.0, 5.0), power=1000.0):
    """Create a simple area light."""
    light_data = bpy.data.lights.new(name=name, type='AREA')
    light_data.energy = power
    light_data.size = 2.0
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = Vector(location)
    return light_obj


def create_cube(name='Cube', size=1.0, location=(0.0, 0.0, 0.5)):
    """Create a cube with a simple Principled BSDF material."""
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    cube = bpy.context.active_object
    cube.name = name
    mat = bpy.data.materials.new(name='CubeMat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.2, 0.5, 0.9, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.4
    cube.data.materials.append(mat)
    return cube


def create_floor(name='Floor', size=10.0, z=0.0):
    """Create a large plane to serve as the floor."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z))
    plane = bpy.context.active_object
    plane.name = name
    mat = bpy.data.materials.new(name='FloorMat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.8
    plane.data.materials.append(mat)
    return plane


def set_render_settings(out_path, width, height):
    """Configure render engine, resolution, color management, and output path."""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'  # Cycles also works if preferred.
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    # Recommended defaults.
    scene.display_settings.display_device = 'sRGB'
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'None'

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.filepath = out_path


def render_rgb(out_path, cam):
    """Render an RGB image from the given camera to out_path."""
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


# --- SE(3) helpers (mathutils.Matrix) ---
def se3_from_tr(txyz=(0, 0, 0), rxyz_deg=(0, 0, 0)):
    """Create a 4x4 transform from translation and XYZ Euler angles in degrees."""
    rx, ry, rz = [math.radians(a) for a in rxyz_deg]
    T = (
        Matrix.Translation(Vector(txyz))
        @ Euler((rx, ry, rz), 'XYZ').to_matrix().to_4x4()
    )
    return T


def set_obj_from_matrix(obj, T_world):
    """Assign a 4x4 world transform to an object."""
    obj.matrix_world = T_world


def create_depth_camera_from_color(color_cam, T_DC):
    """Create a depth camera by cloning intrinsics from the color camera and applying T_DC (depth->color)."""
    depth_cam = bpy.data.cameras.new(name='DepthCamera')
    depth_cam_obj = bpy.data.objects.new('DepthCamera', depth_cam)
    bpy.context.collection.objects.link(depth_cam_obj)

    # Copy camera data settings to keep intrinsics consistent.
    depth_cam.type = color_cam.data.type
    depth_cam.lens = color_cam.data.lens
    depth_cam.sensor_width = color_cam.data.sensor_width
    depth_cam.sensor_height = color_cam.data.sensor_height
    depth_cam.shift_x = color_cam.data.shift_x
    depth_cam.shift_y = color_cam.data.shift_y
    depth_cam.clip_start = color_cam.data.clip_start
    depth_cam.clip_end = color_cam.data.clip_end

    # World transform: T_WD = T_WC @ inv(T_DC).
    T_WC = world_matrix_evaluated(color_cam)
    T_WD = T_WC @ T_DC.inverted()

    depth_cam_obj.matrix_world = T_WD
    depth_cam_obj.rotation_mode = 'XYZ'
    bpy.context.view_layer.update()

    depth_cam_obj.hide_render = False
    depth_cam_obj.hide_set(False)

    return depth_cam_obj


# --- Compositor for depth ---
def setup_depth_compositor(depth_exr_path, depth_viz_png_path, zmax_m):
    """
    Configure a compositor to write the Z pass to files.
    - EXR (32f): raw depth in meters
    - PNG16 (optional visualization): map [0, zmax] to [0, 1] and save as single-channel 16-bit PNG
    Note: File Output nodes append frame numbers; we rename to the requested filename after rendering.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    ntree = scene.node_tree
    ntree.links.clear()
    ntree.nodes.clear()

    # Nodes
    n_rl = ntree.nodes.new('CompositorNodeRLayers')
    n_rl.location = (-300, 0)

    # EXR output (raw depth)
    n_out_exr = ntree.nodes.new('CompositorNodeOutputFile')
    n_out_exr.label = 'DepthEXR'
    n_out_exr.format.file_format = 'OPEN_EXR'
    n_out_exr.format.color_depth = '32'  # 32f
    n_out_exr.base_path = os.path.dirname(depth_exr_path)
    base_exr = os.path.splitext(os.path.basename(depth_exr_path))[0]
    n_out_exr.file_slots[0].path = base_exr + '_'  # => .../<name>_0001.exr
    ntree.links.new(n_rl.outputs['Depth'], n_out_exr.inputs[0])

    # PNG16 visualization (optional)
    n_map = ntree.nodes.new('CompositorNodeMapRange')
    n_map.inputs[1].default_value = 0.0  # From Min
    n_map.inputs[2].default_value = float(zmax_m)  # From Max
    n_map.inputs[3].default_value = 0.0  # To Min
    n_map.inputs[4].default_value = 1.0  # To Max
    n_map.use_clamp = True
    n_map.location = (-50, -150)

    n_out_png = ntree.nodes.new('CompositorNodeOutputFile')
    n_out_png.label = 'DepthVizPNG'
    n_out_png.format.file_format = 'PNG'
    n_out_png.format.color_mode = 'BW'  # single channel
    n_out_png.format.color_depth = '16'  # 16-bit
    n_out_png.base_path = os.path.dirname(depth_viz_png_path)
    base_png = os.path.splitext(os.path.basename(depth_viz_png_path))[0]
    n_out_png.file_slots[0].path = base_png + '_'
    n_out_png.location = (200, -150)

    # Links
    ntree.links.new(n_rl.outputs['Depth'], n_map.inputs['Value'])
    ntree.links.new(n_map.outputs['Value'], n_out_png.inputs[0])

    # Enable Z pass on the view layer.
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True

    return (n_out_exr, n_out_png)


def finalize_file_output(target_path):
    """
    Rename the compositor's '<name>_0001.ext' output to the requested target_path.
    Overwrites target_path if it exists.
    """
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


def render_depth(depth_exr_path, depth_viz_png_path, cam, zmax_m):
    """Render depth using the compositor, producing EXR (raw) and PNG16 (visualization)."""
    scene = bpy.context.scene

    prev_camera = scene.camera
    prev_use_nodes = scene.use_nodes
    prev_filepath = scene.render.filepath

    # Temporary path for the main render (we discard this file).
    tmp_path = os.path.join(os.path.dirname(depth_exr_path), '__tmp_depth_main.png')

    try:
        scene.render.filepath = tmp_path
        scene.camera = cam
        setup_depth_compositor(depth_exr_path, depth_viz_png_path, zmax_m)

        bpy.ops.render.render(write_still=True)

        # Rename compositor outputs to the target names.
        finalize_file_output(depth_exr_path)
        finalize_file_output(depth_viz_png_path)

    finally:
        # Clean up the temporary main render file.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            print(f'[WARN] failed to remove tmp main render: {e}')

        scene.camera = prev_camera
        scene.use_nodes = prev_use_nodes
        scene.render.filepath = prev_filepath


# -----------------------
# Main
# -----------------------
def main():
    out_rgb, width, height, zmax_m = parse_args()
    out_dir = os.path.dirname(out_rgb)
    base = os.path.splitext(os.path.basename(out_rgb))[0]
    out_depth_exr = os.path.join(out_dir, base + '_depth.exr')
    out_depth_viz = os.path.join(out_dir, base + '_depth_viz.png')

    clear_scene()

    # Build a simple scene.
    create_floor()
    create_cube(size=1.0, location=(0.0, 0.0, 0.5))
    create_light(power=1000.0)

    # RGB camera (color camera)
    cam_rgb = create_camera(
        name='ColorCamera',
        location=(0.0, -3.0, 1.5),  # from behind, facing forward
        target=(0.0, 0.0, 0.5),
        focal_length_mm=35.0,  # slightly wider FOV
    )

    # T_DC (depth -> color)
    T_DC = se3_from_tr(txyz=(-0.1, 0.0, 0.0), rxyz_deg=(0.0, 0.0, 0.0))
    cam_depth = create_depth_camera_from_color(cam_rgb, T_DC)

    print('[DBG] T_WC:\n', cam_rgb.matrix_world)
    print('[DBG] T_WD:\n', cam_depth.matrix_world)

    # Common render settings.
    set_render_settings(out_rgb, width, height)

    # RGB render.
    print(f'[INFO] Rendering RGB to: {out_rgb} ({width}x{height})')
    render_rgb(out_rgb, cam_rgb)

    debug_rgb_from_depth = os.path.join(out_dir, base + '_from_depth_rgb.png')
    render_rgb(debug_rgb_from_depth, cam_depth)
    print('[DBG] Saved depth-cam RGB:', debug_rgb_from_depth)

    # Depth render.
    print(
        f'[INFO] Rendering Depth to: {out_depth_exr} (viz: {out_depth_viz}), Zmax={zmax_m} m'
    )
    render_depth(out_depth_exr, out_depth_viz, cam_depth, zmax_m)

    print('[INFO] Done.')


if __name__ == '__main__':
    main()
