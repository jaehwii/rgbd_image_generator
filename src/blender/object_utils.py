# src/blender/object_utils.py
from pathlib import Path
from typing import Tuple

# --- Third-party (Blender) ---
import bpy

# --- Local project modules ---
from src.config.config_types import SE3, ObjectCAD, ObjectPrimitive, ObjectSpec
from src.utils.math_utils import make_scaled_se3_matrix

Vec3 = Tuple[float, float, float]


def _make_material(name: str, base_color=(0.8, 0.8, 0.8, 1.0), roughness: float = 0.4):
    m = bpy.data.materials.new(name=name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Roughness'].default_value = roughness
    return m


# ----------------------------
# Primitive/CAD utilities
# ----------------------------


def _create_primitive(
    shape: str,
    size: Vec3,
    T_WO: SE3,
    base_color=(0.8, 0.8, 0.8, 1.0),
    roughness: float = 0.4,
    object_index: int = 1,
):
    """Create a primitive by shape and apply SE(3)+scale."""
    shape = shape.lower()
    # spawn a unit primitive at origin; final size by matrix_world scaling
    if shape == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
        obj = bpy.context.active_object
    elif shape == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=(0, 0, 0))
        obj = bpy.context.active_object
    elif shape == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 0))
        obj = bpy.context.active_object
    elif shape == 'cone':
        bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1.0, location=(0, 0, 0))
        obj = bpy.context.active_object
    elif shape == 'plane':
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        obj = bpy.context.active_object
    else:
        raise ValueError(f'Unsupported primitive shape: {shape!r}')

    obj.name = f'{shape.capitalize()}'
    obj.pass_index = object_index

    # material
    mat = _make_material(f'{obj.name}Mat', base_color, roughness)
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    # world transform = SE3 * diag(size)
    T = make_scaled_se3_matrix(T_WO.p, T_WO.q_wxyz, size)
    obj.matrix_world = T
    return obj


# src/blender/object_utils.py 중, 기존 _import_cad_mesh() 교체


def _import_cad_mesh(
    path: str,
    scale: Vec3,
    T_WO: SE3,
    base_color=(0.8, 0.8, 0.8, 1.0),
    roughness: float = 0.4,
    object_index: int = 1,
):
    """Import CAD mesh by extension and apply SE(3)*diag(scale).
    Robust to glTF importing a non-mesh active object (e.g., collection root).
    Supports: .glb/.gltf, .fbx, .obj (Blender 4.x: wm.obj_import)
    """
    p = Path(path)
    if not p.is_absolute():
        base = Path(bpy.path.abspath('//')).resolve()
        p = (base / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"[CAD] File not found: '{path}' → resolved '{str(p)}'")

    ext = p.suffix.lower()
    if ext in ('.glb', '.gltf'):
        op = bpy.ops.import_scene.gltf
        kwargs = dict(filepath=str(p))
    elif ext == '.fbx':
        op = bpy.ops.import_scene.fbx
        kwargs = dict(filepath=str(p))
    elif ext == '.obj':
        if not hasattr(bpy.ops.wm, 'obj_import'):
            raise RuntimeError('[CAD] OBJ importer not available in this build.')
        op = bpy.ops.wm.obj_import
        kwargs = dict(filepath=str(p))
    elif ext == '.stl':
        raise RuntimeError(
            '[CAD] STL importer not available in this build. '
            "Use GLB/GLTF/FBX/OBJ instead, or install/enable 'io_mesh_stl'."
        )
    else:
        raise ValueError(f'[CAD] Unsupported file extension: {ext}')

    # --- Import and collect only newly created objects
    before = set(bpy.data.objects)
    try:
        result = op(**kwargs)
    except Exception as e:
        raise RuntimeError(f"[CAD] Import failed for '{str(p)}': {e}")
    if 'FINISHED' not in set(result):
        raise RuntimeError(
            f"[CAD] Import did not finish. path='{str(p)}', result={result}"
        )

    new_objs = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new_objs if o.type == 'MESH']
    if not meshes:
        raise RuntimeError(
            '[CAD] Import succeeded but no MESH objects were created. '
            f'New object types: {[o.type for o in new_objs]}'
        )

    # --- Create a lightweight parent (EMPTY) and parent meshes under it
    parent = bpy.data.objects.new('CAD', None)
    bpy.context.scene.collection.objects.link(parent)

    # Material to override (optional)
    mat = _make_material(f'{parent.name}Mat', base_color, roughness)

    for m in meshes:
        # ensure linked to scene (some importers already link them)
        if m.name not in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.link(m)

        # parent under the EMPTY for unified transform
        m.parent = parent

        # mask index on each mesh (ID mask expects object-level index)
        m.pass_index = object_index

        # color override only if the mesh has NO materials
        if hasattr(m, 'data') and m.data is not None and len(m.data.materials) == 0:
            m.data.materials.append(mat)

    # --- Apply world transform on the parent (SE3 * diag(scale))
    T = make_scaled_se3_matrix(T_WO.p, T_WO.q_wxyz, scale)
    parent.matrix_world = T

    return parent


def create_object_from_spec(
    spec: ObjectSpec,
    object_index: int = 1,
):
    """
    Single entry point used by the renderer.

    - Primitive: uses `spec.shape`, `spec.size`, `spec.color_rgba`, `spec.T_WO`
    - CAD (.stl): uses `spec.path`, `spec.scale`, `spec.color_rgba`, `spec.T_WO`
    """
    if isinstance(spec, ObjectPrimitive):
        return _create_primitive(
            shape=spec.shape,
            size=spec.size,
            T_WO=spec.T_WO,
            base_color=spec.color_rgba,
            object_index=object_index,
        )

    if isinstance(spec, ObjectCAD):
        return _import_cad_mesh(
            path=spec.path,
            scale=spec.scale,
            T_WO=spec.T_WO,
            base_color=spec.color_rgba,
            object_index=object_index,
        )

    raise TypeError(f'Unknown object spec type: {type(spec)}')
