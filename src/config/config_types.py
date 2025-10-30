from dataclasses import dataclass
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
QuatWXYZ = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SE3:
    p: Vec3
    q_wxyz: QuatWXYZ


@dataclass(frozen=True)
class RenderCfg:
    """Top-level render parameters."""

    out_dir: str
    scene_id: str
    width: int
    height: int
    zmax_m: float


@dataclass(frozen=True)
class CameraIntrinsics:
    """Minimal camera intrinsics used by Blender setup."""

    focal_length_mm: float
    clip_start: float
    clip_end: float


@dataclass(frozen=True)
class CameraRig:
    """Color and (optional) depth camera intrinsics."""

    color_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics
    T_DC: SE3  # depth camera -> color camera transform


@dataclass(frozen=True)
class ObjectCfg:
    """A simple parametric object placed in the scene."""

    type: str  # e.g., "cube"
    size: Vec3  # axis-aligned size in meters
    color_rgba: Tuple[float, float, float, float]
    T_WO: SE3  # world_T_object


@dataclass(frozen=True)
class CameraExtrinsics:
    """Per-frame camera pose inputs"""

    p_WC: Vec3  # camera position in world frame
    p_W_target: Vec3  # target point in world frame


@dataclass(frozen=True)
class SequenceCfg:
    camera_extrinsics: List[CameraExtrinsics]


@dataclass(frozen=True)
class SceneCfg:
    render: RenderCfg
    rig: CameraRig
    obj: ObjectCfg
    seq: SequenceCfg
