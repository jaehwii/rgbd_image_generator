from .config_types import (
    SE3,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraRig,
    ObjectCfg,
    RenderCfg,
    SceneCfg,
    SequenceCfg,
)


def load_scene_cfg(toml_path: str) -> SceneCfg:
    # TODO: Replace with real TOML parsing.

    render = RenderCfg(
        out_dir='./outputs',
        scene_id='cube_demo',
        width=640,
        height=480,
        zmax_m=10.0,
    )

    rig = CameraRig(
        color_intrinsics=CameraIntrinsics(
            focal_length_mm=35.0, clip_start=0.01, clip_end=100.0
        ),
        T_DC=SE3(p=(-0.1, 0.0, 0.0), q_wxyz=(1.0, 0.0, 0.0, 0.0)),
    )

    obj = ObjectCfg(
        kind='cube',
        size=1.0,
        T_WO=SE3(p=(0.0, 0.0, 0.5), q_wxyz=(1.0, 0.0, 0.0, 0.0)),
    )

    seq = SequenceCfg(
        camera_extrinsics=[
            CameraExtrinsics(p_WC=(-8.0, -8.0, 8.0), p_W_target=(0.0, 0.0, 0.5)),
            CameraExtrinsics(p_WC=(1.5, -2.5, 1.4), p_W_target=(0.0, 0.0, 0.5)),
            CameraExtrinsics(p_WC=(1.5, 2.5, 1.4), p_W_target=(0.0, 0.0, 0.5)),
            CameraExtrinsics(p_WC=(-1.5, 2.5, 1.4), p_W_target=(0.0, 0.0, 0.5)),
            CameraExtrinsics(p_WC=(-1.5, -2.5, 1.4), p_W_target=(0.0, 0.0, 0.5)),
        ]
    )

    return SceneCfg(render=render, rig=rig, obj=obj, seq=seq)
