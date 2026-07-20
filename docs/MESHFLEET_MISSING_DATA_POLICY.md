# MeshFleet missing-data and UID identity policy

The repository does not require incomplete objects to be physically deleted.
Eligibility is decided per training stage because the stages consume different
supervision:

| Manifest | Required data |
| --- | --- |
| `stage1_train_uids.json` | usable `renders` and `voxels` |
| `stage2_train_uids.json` | usable `renders`, `voxels`, and `ss_latents` |
| `stage3_train_uids.json` | usable `renders`, `voxels`, and `latents` |
| `stage4_train_uids.json` | usable `renders`, `voxels`, and `latents` |
| `train_uids.json`, `validation_uids.json`, `test_uids.json` | every modality named by `--required_modalities`, complete primary/held-out render frames, at least one `renders_cond` image, and disjoint conditioning/evaluation cameras |
| `*_evaluation_uids.json` | usable conditioning renders, both held-out render sets, GT mesh, and disjoint cameras; unused training caches are not required |

Consequently, a UID missing only `features` can still be used by all current
training stages because those stages extract image evidence online. A UID
missing `latents` can still be used by Stages 1 and 2, but not by Stages 3 or 4.
A UID missing voxels is excluded from all geometry-supervised stages. This is
filtering of stage eligibility, not deletion of the underlying object.

Training loaders also enforce their required modalities. If a frozen manifest
names a UID that is no longer loadable, construction fails instead of silently
substituting another object. Missing or malformed individual render frames are
skipped only when at least one usable image/camera pair remains; the auditor's
minimum-view thresholds keep under-observed objects out of official manifests.
Use `--validate_payloads` on the remote audit so corrupt images, missing NPZ
arrays, malformed voxel PLY headers, and empty meshes are rejected before a
distributed job starts.

Official evaluation must receive a frozen `--uid_manifest`. Each subprocess is
addressed with `--meshfleet_uid`; numeric indices remain only as scheduling and
directory labels. This prevents a missing modality or render set from shifting
the sample order between inference, conditioning refinement, held-out rendering,
and GT-mesh lookup. Cached output directories contain `sample_identity.json` and
cannot be reused for a different UID unless the run is explicitly overwritten.

For the strictest comparison, use `test_uids.json`. If the benchmark definition
does not require cached training-only features/latents at test time, it is more
scientifically complete to freeze `test_evaluation_uids.json` before model
selection. Do not exclude a UID because the model fails on it: any failure for a
manifested UID remains in completeness accounting and makes
`official_complete=false`.
