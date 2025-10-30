from typing import Tuple

# --- Third-party (Blender) ---
import bpy

# --- Local project modules ---
from src.config.config_types import SE3
from src.utils.math_utils import make_scaled_se3_matrix

Vec3 = Tuple[float, float, float]


def create_cube(
    name: str = 'Cube',
    size: Vec3 = (1.0, 1.0, 1.0),  # Vec3 only
    T_WO: SE3 = SE3(
        p=(0.0, 0.0, 0.5),
        q_wxyz=(1.0, 0.0, 0.0, 0.0),
    ),
    base_color=(0.2, 0.5, 0.9, 1.0),
    roughness: float = 0.4,
    object_index: int = 1,
):
    """Create cube with non-uniform size (Vec3), material, and world transform."""
    # 1) spawn unit cube
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)
    )
    cube = bpy.context.active_object
    cube.name = name
    cube.pass_index = object_index

    # 2) material
    mat = bpy.data.materials.new(name=f'{name}Mat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Roughness'].default_value = roughness
    if len(cube.data.materials) == 0:
        cube.data.materials.append(mat)
    else:
        cube.data.materials[0] = mat

    # 3) apply world transform (including size)
    T = make_scaled_se3_matrix(T_WO.p, T_WO.q_wxyz, size)
    cube.matrix_world = T

    return cube
