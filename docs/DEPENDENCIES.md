# Dependency Tiers

This project uses optional dependency gates instead of silent fallback in real
training, inference, and evaluation.

Install the minimal runtime:

```bash
pip install -r requirements/base.txt
```

Geometry and asset processing:

```bash
pip install -r requirements/geometry.txt
```

3DGS rendering and evaluation:

```bash
pip install -r requirements/gsplat.txt
```

SfM and differentiable camera utilities:

```bash
pip install -r requirements/sfm.txt
```

Visualization and debug extras:

```bash
pip install -r requirements/visualization.txt
```

Environment-sensitive render libraries are intentionally not installed by
default. Check your runtime first:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m geoss.utils.env_check
```

Then install matching builds of `pytorch3d`, `nvdiffrast`, and `kaolin` only
when configs require them. Real mode fails fast when a required optional library
is missing; dry-run mode can skip unavailable optional components.

## Library Roles

- `gsplat`: 3DGS render-back, novel-view render metrics, render-level loss, and Gaussian statistics.
- `pycolmap`: SfM/COLMAP reconstruction loading, sparse camera initialization, pose sanity checks.
- `kornia`: gated differentiable camera/projection path and warp/reprojection losses.
- `open3d`, `trimesh`, `plyfile`: point cloud, mesh, GLB/PLY IO and cleanup.
- `point-cloud-utils`: Chamfer, F-score, nearest-neighbor and point/surface metrics.
- `pytorch3d`, `nvdiffrast`, `kaolin`: advanced optional differentiable 3D ops.
- `viser`, `rerun-sdk`, `pyvista`: debug visualization only, never core training dependencies.
