from dataclasses import dataclass
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
QuatWXYZ = Tuple[float, float, float, float]


@dataclass
class SE3:
    p: Vec3
    q_wxyz: QuatWXYZ


@dataclass
class RenderCfg:
    out_dir: str
    scene_id: str
    width: int
    height: int
    zmax_m: float


@dataclass
class CameraIntrinsics:
    focal_length_mm: float
    clip_start: float = 0.01
    clip_end: float = 100.0


@dataclass
class CameraRig:
    color_intrinsics: CameraIntrinsics
    T_DC: SE3  # depth camera -> color camera transform


@dataclass
class ObjectCfg:
    kind: str  # e.g., "cube"
    size: float
    T_WO: SE3  # world -> object


@dataclass
class CameraExtrinsics:
    p_WC: Vec3  # camera position in world frame
    p_W_target: Vec3  # target point in world frame


@dataclass
class SequenceCfg:
    camera_extrinsics: List[CameraExtrinsics]


@dataclass
class SceneCfg:
    render: RenderCfg
    rig: CameraRig
    obj: ObjectCfg
    seq: SequenceCfg
