# GeoVis-SLAT Adapter

GeoVis-SLAT adds a SLAT-only adapter after Sparse-Ray GeoSS SS generation. It keeps the architecture:

```text
VGGT multi-view evidence
+ SS active voxels and geometry confidence
-> active voxel projection
-> visibility-aware multi-view SLAT evidence
-> geometry-aligned SLAT condition tokens
-> confidence-gated TRELLIS SLAT Flow velocity residual
```

It does not modify VGGT, TRELLIS source, decoder, Mesh, or 3DGS stages.

## MeshFleet_TRELLIS

The local MeshFleet_TRELLIS sample is supported through `geoss/datasets/meshfleet_trellis_dataset.py`.

Both layouts are supported.

Full server flat split layout:

```text
/mnt/sda2/hf/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97/
  train/
    features/
    latents/
    mesh_normalized/
    renders/
    renders_cond/
    renders_eval_70/
    renders_eval_90/
    ss_latents/
    voxels/
  test/
    features/
    latents/
    mesh_normalized/
    renders/
    renders_cond/
    renders_eval_70/
    renders_eval_90/
    ss_latents/
    voxels/
```

Older local category/sample layout:

```text
MeshFleet_TRELLIS/<split>/<category>/
  renders/<uid>/transforms.json
  renders/<uid>/*.png
  voxels/<uid>.ply
  ss_latents/ss_enc_conv3d_16l8_fp16/<uid>.npz
  latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/<uid>.npz
  features/dinov2_vitl14_reg/<uid>.npz
```

`latents/.../*.npz` is treated as TRELLIS native SLAT target: `feats=[L,C]`, `coords=[L,3]`.

## Dry Run

```bash
python scripts/train_geovis_slat.py --config configs/geovis_slat.yaml --dry_run true --device cpu --output_dir outputs/geovis_slat_dry
python scripts/train_geovis_slat_joint.py --config configs/geovis_slat_joint.yaml --dry_run true --device cpu --output_dir outputs/geovis_slat_joint_dry
python scripts/infer_geovis_slat.py --config configs/geovis_slat.yaml --input examples/car_views --dry_run true --device cpu --output_dir outputs/geovis_slat_infer_dry
```

## MeshFleet 2-Step Sanity Training

```bash
python scripts/train_geovis_slat.py ^
  --config configs/geovis_slat.yaml ^
  --device cpu ^
  --output_dir outputs/geovis_slat_meshfleet_2step ^
  --steps 2 ^
  --batch_size 1 ^
  --num_views 3 ^
  --image_size 64 ^
  --active_tokens 128 ^
  --meshfleet_root D:\VsCode\MVG\Base\MeshFleet_TRELLIS ^
  --meshfleet_split test ^
  --meshfleet_category sdvas
```

## Inference Outputs

`scripts/infer_geovis_slat.py --dry_run true` writes:

- `original_slat.npz`
- `geovis_controlled_slat.npz`
- `ss_active_voxels.ply`
- `slat_confidence.ply`
- `slat_visibility_debug.npz`
- `view_weights.npz`
- `slat_velocity_debug.npz`
- `metrics.json`

## Evaluation

```bash
python scripts/eval_geovis_slat.py --input_dir outputs/geovis_slat_infer_dry --output_dir outputs/geovis_slat_eval_dry --ablation full_geovis_slat
python scripts/visualize_geovis_slat.py --input_dir outputs/geovis_slat_infer_dry --output_dir outputs/geovis_slat_vis_dry
```

Decoder-dependent RGB/LPIPS metrics remain optional and are not invoked by default.
