# TODO: rename file to avoid confusion with mathutils in Blender?

from typing import Tuple

# --- Third-party (Blender) ---
from mathutils import Matrix, Quaternion, Vector

Vec3 = Tuple[float, float, float]
QuatWXYZ = Tuple[float, float, float, float]


def make_se3_matrix(p_xyz: Vec3, q_wxyz: QuatWXYZ) -> Matrix:
    """Build 4x4 SE(3) from translation p, quaternion (w,x,y,z).
    Applies transform as:  T * R.
    """
    # Rotation
    Rot = Quaternion(q_wxyz).to_matrix().to_4x4()
    # Translation
    Trans = Matrix.Translation(Vector(p_xyz))

    return Trans @ Rot


def make_scaled_se3_matrix(
    p_xyz: Vec3, q_wxyz: QuatWXYZ, s_xyz: Vec3 = (1.0, 1.0, 1.0)
) -> Matrix:
    """Build 4x4 matrix from translation p, quaternion (w,x,y,z), and scale s.
    Applies transform as:  T * R * S.
    Note: this way, the scale is applied in local frame
    """
    T = make_se3_matrix(p_xyz, q_wxyz)
    # Scale (non-uniform)
    sx, sy, sz = map(float, s_xyz)
    S = Matrix.Identity(4)
    S[0][0] = sx
    S[1][1] = sy
    S[2][2] = sz

    return T @ S


def look_at_quaternion(eye: Vector, target: Vector) -> Quaternion:
    """Return a quaternion so that local -Z looks at (target - eye), +Y is up-ish."""
    d = target - eye
    if d.length < 1e-9:
        d = Vector((0, 0, -1))
    # Blender camera convention: forward = local -Z, up = local +Y
    return d.to_track_quat('-Z', 'Y')  # (w, x, y, z)
