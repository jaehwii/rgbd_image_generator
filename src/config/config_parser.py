from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, Tuple

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

from src.config.config_types import (
    SE3,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraRig,
    DropoutNoiseConfig,
    GaussianNoiseConfig,
    MultiplicativeNoiseConfig,
    NoiseConfig,
    ObjectCfg,
    QuantizationNoiseConfig,
    RenderCfg,
    SceneCfg,
    SequenceCfg,
)

# -----------------------------
# Helpers
# -----------------------------


def _as_vec3(v: Iterable[float]) -> Tuple[float, float, float]:
    x = tuple(float(x) for x in v)
    if len(x) != 3:
        raise ValueError(f'Vec3 must have length 3, got {len(x)}: {v}')
    return x  # type: ignore[return-value]


def _as_quat_wxyz(v: Iterable[float]) -> Tuple[float, float, float, float]:
    x = tuple(float(x) for x in v)
    if len(x) != 4:
        raise ValueError(f'QuatWXYZ must have length 4, got {len(x)}: {v}')
    return x  # type: ignore[return-value]


def _get(d, key, default):
    return d.get(key, default)


# -----------------------------
# Default (hardcoded) config
# -----------------------------

DEFAULT_CFG = SceneCfg(
    render=RenderCfg(
        out_dir='./outputs',
        scene_id='cube_demo',
        width=640,
        height=480,
        zmax_m=10.0,
    ),
    rig=CameraRig(
        color_intrinsics=CameraIntrinsics(
            focal_length_mm=35.0, clip_start=0.01, clip_end=100.0
        ),
        depth_intrinsics=CameraIntrinsics(
            focal_length_mm=35.0, clip_start=0.01, clip_end=100.0
        ),
        T_DC=SE3(p=(-0.1, 0.0, 0.0), q_wxyz=(1.0, 0.0, 0.0, 0.0)),
    ),
    obj=ObjectCfg(
        type='cube',
        size=(1.0, 1.0, 1.0),
        color_rgba=(0.8, 0.2, 0.2, 1.0),
        T_WO=SE3(p=(0.0, 0.0, 0.5), q_wxyz=(1.0, 0.0, 0.0, 0.0)),
    ),
    seq=SequenceCfg(
        camera_extrinsics=[
            CameraExtrinsics(p_WC=(1.5, -2.5, 1.4), p_W_target=(0.0, 0.0, 0.5)),
            CameraExtrinsics(p_WC=(1.5, 2.5, 1.4), p_W_target=(0.0, 0.0, 0.5)),
            CameraExtrinsics(p_WC=(-1.5, 2.5, 1.4), p_W_target=(0.0, 0.0, 0.5)),
            CameraExtrinsics(p_WC=(-1.5, -2.5, 1.4), p_W_target=(0.0, 0.0, 0.5)),
        ]
    ),
)


# -----------------------------
# TOML -> dataclasses
# -----------------------------


def _parse_rig(d: Dict[str, Any]) -> CameraRig:
    cin = d.get('color_intr', {})
    din = d.get('depth_intr', cin)  # default to color intr if depth missing

    color = CameraIntrinsics(
        focal_length_mm=float(cin['focal_length_mm']),
        clip_start=float(cin['clip_start']),
        clip_end=float(cin['clip_end']),
    )
    depth = CameraIntrinsics(
        focal_length_mm=float(din['focal_length_mm']),
        clip_start=float(din['clip_start']),
        clip_end=float(din['clip_end']),
    )

    tdc = d.get('T_DC')
    if not isinstance(tdc, dict):
        raise KeyError('[rig.T_DC] must be provided with fields `p` and `q_wxyz`')
    T_DC = SE3(
        p=_as_vec3(tdc['p']),
        q_wxyz=_as_quat_wxyz(tdc['q_wxyz']),
    )

    return CameraRig(color_intrinsics=color, depth_intrinsics=depth, T_DC=T_DC)


def _parse_obj(d: Dict[str, Any]) -> ObjectCfg:
    # REQUIRE Vec3 for size
    if 'size' not in d:
        raise KeyError(
            '[obj] size must be provided as a Vec3, e.g., size = [1.0, 1.0, 1.0]'
        )
    size_vec3 = _as_vec3(d['size'])

    # Allow either flat keys or nested tables for transform
    T = d.get('T_WO')
    if isinstance(T, dict):
        p = _as_vec3(T['p'])
        q = _as_quat_wxyz(T['q_wxyz'])
    else:
        # Backward-compatible keys
        p = _as_vec3(d['p_WO']) if 'p_WO' in d else (0.0, 0.0, 0.0)
        q = _as_quat_wxyz(d['q_WO_wxyz']) if 'q_WO_wxyz' in d else (1.0, 0.0, 0.0, 0.0)

    return ObjectCfg(
        type=str(d.get('type', 'cube')),
        size=size_vec3,
        color_rgba=tuple(float(x) for x in d.get('color_rgba', (0.8, 0.2, 0.2, 1.0))),  # type: ignore[return-value]
        T_WO=SE3(p=p, q_wxyz=q),
    )


def _parse_seq(d: Dict[str, Any]) -> SequenceCfg:
    items = []
    for it in d.get('camera_extrinsics', []):
        # support both {p_WC=[...], p_W_target=[...]} and {t=[...], target_in_world=[...]}
        p = _as_vec3(it.get('p_WC', it.get('t', (0.0, 0.0, 0.0))))
        tgt = _as_vec3(it.get('p_W_target', it.get('target_in_world', (0.0, 0.0, 0.0))))
        items.append(CameraExtrinsics(p_WC=p, p_W_target=tgt))
    return SequenceCfg(camera_extrinsics=items)


def _parse_render(d: Dict[str, Any]) -> RenderCfg:
    return RenderCfg(
        out_dir=str(d['out_dir']),
        scene_id=str(d['scene_id']),
        width=int(d['width']),
        height=int(d['height']),
        zmax_m=float(d['zmax_m']),
    )


def _parse_noise(d: Dict[str, Any]) -> NoiseConfig:
    nraw = d or {}
    graw = nraw.get('gaussian', {}) or {}
    mraw = nraw.get('multiplicative', {}) or {}
    qraw = nraw.get('quantization', {}) or {}
    draw = nraw.get('dropout', {}) or {}

    return NoiseConfig(
        enabled=bool(_get(nraw, 'enabled', True)),
        gaussian=GaussianNoiseConfig(
            enabled=bool(_get(graw, 'enabled', False)),
            sigma_m=float(_get(graw, 'sigma_m', 0.0)),
        ),
        multiplicative=MultiplicativeNoiseConfig(
            enabled=bool(_get(mraw, 'enabled', False)),
            sigma_rel=float(_get(mraw, 'sigma_rel', 0.0)),
        ),
        quantization=QuantizationNoiseConfig(
            enabled=bool(_get(qraw, 'enabled', False)),
            step_m=float(_get(qraw, 'step_m', 0.0)),
        ),
        dropout=DropoutNoiseConfig(
            enabled=bool(_get(draw, 'enabled', False)),
            p=float(_get(draw, 'p', 0.0)),
            fill=float(_get(draw, 'fill', 0.0)),
        ),
    )


def load_scene_cfg(toml_path: str | None = None, *, use_toml: bool = False) -> SceneCfg:
    """Return a complete :class:`SceneCfg`.

    - If ``use_toml`` is False (default), returns the hardcoded defaults (matches current behavior).
    - If ``use_toml`` is True, parses the TOML file at ``toml_path``.
    """
    if not use_toml:
        return DEFAULT_CFG

    if toml_path is None:
        raise ValueError('toml_path is required when use_toml=True')
    if tomllib is None:
        raise RuntimeError(
            'tomllib not available; use Python 3.11+ or set use_toml=False.'
        )

    with open(toml_path, 'rb') as f:
        raw = tomllib.load(f)

    render = _parse_render(raw['render'])
    rig = _parse_rig(raw['rig'])
    obj = _parse_obj(raw['obj'])
    seq = _parse_seq(raw['seq'])
    noise = _parse_noise(raw.get('noise', {}) or {})

    return SceneCfg(render=render, rig=rig, obj=obj, seq=seq, noise=noise)


# Small utility for debugging
def scene_cfg_asdict(cfg: SceneCfg) -> Dict[str, Any]:
    return asdict(cfg)
