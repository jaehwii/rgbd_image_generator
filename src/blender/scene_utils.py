import math

# --- Third-party (Blender) ---
import bpy
from mathutils import Matrix, Vector

# --- Local project modules ---
from src.config.config_types import CameraIntrinsics


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
