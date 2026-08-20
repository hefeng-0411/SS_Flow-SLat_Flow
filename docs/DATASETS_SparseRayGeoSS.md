# Datasets

Supported sources are ShapeNet-SRN Cars, Objaverse Cars, and the local
MeshFleet_TRELLIS dataset. MeshFleet_TRELLIS is treated as an Objaverse Cars
vehicle subset that has already been preprocessed by the TRELLIS training
pipeline.

## ShapeNet-SRN Cars

Expected structure:

```text
SRN_ROOT/
  cars_train/<object_id>/rgb/*.png
  cars_train/<object_id>/pose/*.txt
  cars_train/<object_id>/intrinsics.txt
  list_train.txt
```

`SRNCarsDataset` returns images, masks, `K`, `c2w`, `w2c`, ids, and optional `gt_occ` if a ShapeNet mesh root is provided and a mesh is found.

SRN view sampling supports `fixed`, `random`, and `all`. RGBA alpha is used as
mask when available; otherwise a white-background threshold is used.

## Objaverse Cars

Raw GLB metadata is loaded by `ObjaverseCarsRawDataset`. Rendered RGB views are loaded by `ObjaverseCarsRenderedDataset`, supporting per-view JSON and NeRF-style `transforms.json`.

RGBA alpha becomes the mask. If alpha is missing or empty, a background threshold fallback is used. Depth and normal files are optional.

`ObjaverseCarsRawDataset` filters `annotations.json` for vehicle-like labels by
default when annotations are present.

## MeshFleet_TRELLIS

The local dataset path used for this project is:

```text
D:/VsCode/MVG/Base/MeshFleet_TRELLIS
```

The Hugging Face dataset may be downloaded as WebDataset shards. This code
expects the reconstructed TRELLIS folder layout. If the split contains only
shards, first run the dataset-card reconstruction command:

```bash
python reconstruct_data.py --shard_dir ./data/meshfleet_trellis/train --output_dir ./data/meshfleet_trellis/train_reconstructed
```

The reconstructed layout is:

```text
MeshFleet_TRELLIS/
  train/<category>/
    renders/<uid>/{000.png,...,transforms.json}
    renders_cond/<uid>/{*.png,transforms.json}
    voxels/<uid>.ply
    ss_latents/ss_enc_conv3d_16l8_fp16/<uid>.npz
    features/dinov2_vitl14_reg/<uid>.npz
    mesh_normalized/<uid>/mesh.glb
  test/<category>/
    ...
```

`MeshFleetTrellisDataset` consumes:

- RGBA renders as RGB images plus alpha masks.
- per-frame `camera_angle_x` and OpenGL `transform_matrix`, converted to OpenCV `c2w`.
- `voxels/<uid>.ply` as GT sparse occupancy supervision.
- `ss_latents/.../<uid>.npz` field `mean` as TRELLIS SS latent grid `[8,16,16,16]`.
- optional DINO/TRELLIS `features/<model>/<uid>.npz`.

MeshFleet voxel PLY files in the inspected sample store centers in a TRELLIS
centered half cube, approximately `[-0.5,0.5]^3`. The loader maps them to
GeoSS canonical `[-1,1]^3` using `xyz_canonical = xyz * 2` and records the
transform in `metadata["voxel_coordinate"]`.

Current local audit:

- `test/sdvas` contains one reconstructed object sample and is valid for smoke tests.
- `train` is currently empty on disk. Use `--meshfleet_split test --meshfleet_category sdvas` for local verification until train shards are reconstructed.

## Unified Batch

`VehicleMultiViewDataset.collate_fn` produces:

```text
images [B,N,3,H,W]
masks [B,N,1,H,W]
K [B,N,3,3]
c2w [B,N,4,4]
w2c [B,N,4,4]
gt_occ optional [B,R,R,R]
ss_latent_grid optional [B,8,16,16,16]
ss_latent_tokens optional [B,4096,8]
gt_sparse_xyz optional [B,P,3] or list when variable length
```
