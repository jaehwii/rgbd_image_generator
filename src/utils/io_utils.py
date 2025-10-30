from pathlib import Path

# --- Third-party (Blender) ---
from mathutils import Matrix


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
