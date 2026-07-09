# Gap Audit: Sparse-Ray GeoSS Adapter

Status labels: `real implemented`, `mock only`, `scaffold only`, `risky`, `missing`, `fixed`.

| Area | Item | Status | Notes / fix status |
|---|---|---:|---|
| VGGTGeometryWrapper | Local checkpoint load | risky | Supports local state dict/checkpoint and `from_pretrained`; real VGGT still depends on the exact cloned VGGT API/checkpoint. |
| VGGTGeometryWrapper | Calls real VGGT forward / aggregator | risky | Wrapper has a single normalization point for output aliases. Without checkpoint it uses mock and logs shape-safe fallback. |
| VGGTGeometryWrapper | Real depth / pointmap / features | risky | Common field aliases are handled; needs one real checkpoint run to confirm latest VGGT field names. |
| VGGTGeometryWrapper | `[B,N,3,H,W]` / `[N,3,H,W]` input | real implemented | 5D training and 4D single-object calls are normalized. |
| VGGTGeometryWrapper | Freeze / eval / no_grad / cache | fixed | Wrapper and inner VGGT model default to eval; all VGGT params are frozen; tests cover no-grad mock outputs and real-checkpoint freeze when available. |
| RayEvidenceSampler | K/w2c projection and anchor depth | real implemented | Uses OpenCV projection with explicit `c2w` / `w2c`. |
| RayEvidenceSampler | mask/depth/VGGT depth sampling | real implemented | Samples mask and depth per projected anchor; features are sampled only when dense feature maps are available. |
| RayEvidenceSampler | occupied/free-space rules | fixed | Occupied = in bounds, mask inside, near depth surface. Free = mask outside or camera-to-surface free segment; `ray_free_space_loss` can now train directly from `free_geometry`. |
| RayEvidenceSampler | invalid/out-of-bound mask | real implemented | Invalid views are excluded by bounds and positive depth. |
| RayEvidenceSampler | multi-view conflict | real implemented | Outputs `conflict_score` and aggregator downweights confidence by conflict. |
| CrossViewEvidenceAggregator | Per-anchor view-only attention | real implemented | Attention operates over `[N]` view tokens for each anchor; no full-image brute-force attention. |
| CrossViewEvidenceAggregator | variable views / dropout | real implemented | Any `N` is accepted; training-time view dropout is supported. |
| CrossViewEvidenceAggregator | evidential confidence | real implemented | Returns `p_occ`, `uncertainty`, `geo_confidence`, and confidence stats. |
| SSVelocityAdapter | SS tokens cross-attend to geo tokens | real implemented | Uses SS latent tokens as queries and geo tokens as keys/values. |
| SSVelocityAdapter | token confidence aggregation | real implemented | Attention weights aggregate anchor confidence to SS-token confidence. |
| SSVelocityAdapter | alpha(t) and trust-region clip | real implemented | `fixed/cosine/learned` alpha and timestep-dependent clamp are active. |
| SSVelocityAdapter | disabled identity | real implemented | Disabled path returns `v_base`; tests assert `<1e-6`. |
| TRELLIS hook | Hook final SS velocity output | real implemented | External wrapper calls base SS flow, then applies velocity adapter. No TRELLIS source file is modified. |
| TRELLIS hook | sampling / CFG | fixed | CFG sampler wrapper applies GeoSS to conditional branch by default, leaves unconditional branch unmodified unless `geoss_apply_to_uncond=true`, and records sampler debug. Real TRELLIS sampler object still needs checkpoint-level integration run. |
| TRELLIS hook | training path | real implemented | Stage 2 can train velocity adapter using real MeshFleet `ss_latent_grid` and GeoSS context; real TRELLIS checkpoint is optional. |
| Dataset / coordinates | SRN / Objaverse | fixed | Parsers normalize to OpenCV canonical camera convention. SRN supports fixed/random/all views and alpha/white-background masks; raw Objaverse filters vehicle annotations when available. |
| Dataset / coordinates | MeshFleet_TRELLIS reconstructed layout | real implemented | Loader consumes `renders`, `transforms.json`, `voxels`, `ss_latents`, `features`, and `mesh_normalized`. |
| Dataset / coordinates | MeshFleet voxel coordinate | real implemented | PLY points in centered half cube are converted to GeoSS `[-1,1]^3` by `xyz * 2`; metadata records the transform. |
| Dataset / coordinates | WebDataset shard layout | missing | Current implementation expects reconstructed folders. If only shards exist, run dataset-card `reconstruct_data.py` first. |
| Training / eval | Stage 1 real batch | fixed | `train_sparse_ray_geoss.py` accepts MeshFleet root/split/category, supports resume, validation smoke, detailed loss/evidence JSONL logs, and required visualization/npz outputs. |
| Training / eval | Stage 2 real SS latent batch | fixed | `train_sparse_ray_ss_velocity.py` reads MeshFleet `ss_latent_grid` as `x0`, supports resume, logs identity error/velocity norms, and saves adapter+optimizer checkpoint. |
| Training / eval | Real TRELLIS checkpoint | risky | Hook is ready, but no local checkpoint path was provided/verified in this pass. |
| Training / eval | Eval metrics | real implemented | Synthetic/npz occupancy metrics, free-space/projection/confidence metrics are computed; some baselines remain fallback. |
| Ablations | Full list | scaffold only | Config labels exist. Baselines that require separate implementations are explicitly fallback and not reported as complete. |

Current local data audit:

- `D:/VsCode/MVG/Base/MeshFleet_TRELLIS/test/sdvas` is a reconstructed MeshFleet_TRELLIS category with one object sample.
- `D:/VsCode/MVG/Base/MeshFleet_TRELLIS/train` is currently empty on disk.
- The sample contains 150 RGBA renders, per-frame `camera_angle_x` and OpenGL `transform_matrix`, `voxels/<uid>.ply`, `ss_latents/ss_enc_conv3d_16l8_fp16/<uid>.npz`, DINO/TRELLIS features, and normalized mesh.
- The PLY voxel coordinates are centered around zero with max abs <= 0.5, so they are treated as TRELLIS centered half-cube coordinates and mapped to GeoSS `[-1,1]^3`.
