# 1
**完成情况**
已围绕 `prompt.md` 在 `SS_Flow` 里落地一批真实代码改造，重点是禁止 real 模式 silent mock、补 VGGT→TRELLIS canonical alignment、动态 anchor、局部几何注意力、residual 训练目标、SS→SLAT learned/soft context、真实 TRELLIS pipeline wrapper、eval no-fake 和配置/测试。

注意：`git status` 里已有 `D da`，这是我开始前就存在的删除状态，我没有恢复或改动它。

**关键文件**
- [geoss/utils/run_mode.py](D:/VsCode/MVG/Base/SS_Flow/geoss/utils/run_mode.py): real/dry-run 统一校验，缺 VGGT/TRELLIS/dataset/decoder/prediction/render 直接 fail-fast。
- [geoss/geometry/alignment.py](D:/VsCode/MVG/Base/SS_Flow/geoss/geometry/alignment.py): 新增 Sim(3)/Umeyama、depth scale/shift、pose residual、alignment confidence。
- [geoss/models/sparse_anchor_queries.py](D:/VsCode/MVG/Base/SS_Flow/geoss/models/sparse_anchor_queries.py): learnable + VGGT surface + boundary/uncertainty 动态 anchors，带 metadata。
- [geoss/models/ss_velocity_adapter.py](D:/VsCode/MVG/Base/SS_Flow/geoss/models/ss_velocity_adapter.py): SS kNN local anchor attention，distance/confidence/metadata bias，保留 trust-region/alpha/confidence gate。
- [geoss/models/sparse_ray_geoss_adapter.py](D:/VsCode/MVG/Base/SS_Flow/geoss/models/sparse_ray_geoss_adapter.py): 使用 aligned geometry，支持 context-only real inference。
- [geoss/losses/ray_level_geometry_loss.py](D:/VsCode/MVG/Base/SS_Flow/geoss/losses/ray_level_geometry_loss.py): ray termination/mask/depth/free-space/surface 近似 loss。
- [geoss/losses/confidence_calibration.py](D:/VsCode/MVG/Base/SS_Flow/geoss/losses/confidence_calibration.py): Brier/BCE confidence calibration、ECE、evidential occupancy stats。
- [geoss/integration/real_trellis_pipeline.py](D:/VsCode/MVG/Base/SS_Flow/geoss/integration/real_trellis_pipeline.py): 真实 TRELLIS image→SS→SLAT→decoder wrapper，支持 adapter context 注入并保存 asset。
- [geoss/slat/integration/ss_slat_context.py](D:/VsCode/MVG/Base/SS_Flow/geoss/slat/integration/ss_slat_context.py), [geoss/slat/utils/slat_token_mapping.py](D:/VsCode/MVG/Base/SS_Flow/geoss/slat/utils/slat_token_mapping.py): 禁止 256→8 硬截断，改 kNN soft assignment。
- [scripts/train_sparse_ray_ss_velocity.py](D:/VsCode/MVG/Base/SS_Flow/scripts/train_sparse_ray_ss_velocity.py): SS 改为 `target_residual = target_v - v_base`，real 模式禁止 mock/zero base/random cond。
- [scripts/train_geovis_slat.py](D:/VsCode/MVG/Base/SS_Flow/scripts/train_geovis_slat.py): SLAT 改 residual loss，real 模式要求真实 `trellis_slat_base_velocity`。
- [scripts/infer_sparse_ray_geoss_ss.py](D:/VsCode/MVG/Base/SS_Flow/scripts/infer_sparse_ray_geoss_ss.py), [scripts/infer_geovis_slat.py](D:/VsCode/MVG/Base/SS_Flow/scripts/infer_geovis_slat.py): real infer 不再直接 not implemented，改走 real TRELLIS wrapper。
- [scripts/eval_sparse_ray_geoss.py](D:/VsCode/MVG/Base/SS_Flow/scripts/eval_sparse_ray_geoss.py), [scripts/eval_geovis_slat.py](D:/VsCode/MVG/Base/SS_Flow/scripts/eval_geovis_slat.py): real eval 禁止 fake pred/GT/render fallback。
- 新增配置：`configs/real_train_full_joint.yaml`, `real_train_ss_only.yaml`, `real_train_slat_only.yaml`, `real_eval.yaml`, `real_infer.yaml`, `dry_run_debug.yaml`, 四个 ablation config。
- 新增测试：`tests/test_real_mode_fail_fast.py`, `test_geometry_alignment.py`, `test_dynamic_anchor_local_attention.py`, `test_ss_slat_context_projector.py`, `test_ray_confidence_losses.py`。

**命令示例**
训练 SS residual：
```powershell
python scripts/train_sparse_ray_ss_velocity.py --config configs/real_train_ss_only.yaml --real_train --trellis_root D:\VsCode\MVG\Base\TRELLIS --trellis_model_path <TRELLIS-image-large> --vggt_root D:\VsCode\MVG\Base\vggt --vggt_checkpoint <vggt.pt> --meshfleet_root <dataset>
```

训练 SLAT residual：
```powershell
python scripts/train_geovis_slat.py --config configs/real_train_slat_only.yaml --real_train --trellis_model_path <TRELLIS-image-large> --meshfleet_root <dataset>
```

真实推理：
```powershell
python scripts/infer_sparse_ray_geoss_ss.py --config configs/real_infer.yaml --real_infer --trellis_root D:\VsCode\MVG\Base\TRELLIS --trellis_model_path <TRELLIS-image-large> --vggt_root D:\VsCode\MVG\Base\vggt --vggt_checkpoint <vggt.pt> --input <rendered_object_dir>
```

真实评估：
```powershell
python scripts/eval_sparse_ray_geoss.py --config configs/real_eval.yaml --real_eval --prediction <pred_occ.npz> --gt_occ <gt_occ.pt>
python scripts/eval_geovis_slat.py --real_eval --prediction <pred_render.npz> --gt_render <gt_render.npz>
```

消融：
```powershell
python scripts/infer_sparse_ray_geoss_ss.py --config configs/ablation_no_alignment.yaml --real_infer ...
python scripts/infer_sparse_ray_geoss_ss.py --config configs/ablation_global_attention.yaml --real_infer ...
```

**验证**
- `python -m compileall geoss scripts tests` 通过。
- 没能跑 `pytest`：当前 Python 环境没有 `pytest`。
- 没能做 PyTorch runtime smoke：默认 Python 和 Codex bundled Python 都没有 `torch`。

**仍需真实资源验证**
真实闭环仍需要实际 VGGT checkpoint/TRELLIS pipeline checkpoint、包含 TRELLIS cond/base velocity/latents 的 dataset，以及可用 CUDA/PyTorch/TRELLIS 渲染依赖。现在 real 模式缺这些会明确报错，不会保存 mock 指标冒充正式结果。


# 2
已继续落地“底层库接入”章节，重点不是堆 requirements，而是把 optional dependency、real-mode fail-fast、renderer/geometry/io/metrics/loss/baseline/debug 封装接进现有闭环。

**新增核心模块**
- [geoss/utils/optional_deps.py](D:/VsCode/MVG/Base/SS_Flow/geoss/utils/optional_deps.py): `is_gsplat_available()`、`is_pycolmap_available()`、`require_dependency(...)` 等统一检测。
- [geoss/utils/env_check.py](D:/VsCode/MVG/Base/SS_Flow/geoss/utils/env_check.py): 输出 Python/PyTorch/CUDA/GPU/optional deps 状态。
- [geoss/renderers/gsplat_renderer.py](D:/VsCode/MVG/Base/SS_Flow/geoss/renderers/gsplat_renderer.py): `render_gaussians(...)`，返回 RGB/alpha/depth/visibility。
- [geoss/renderers/pytorch3d_renderer.py](D:/VsCode/MVG/Base/SS_Flow/geoss/renderers/pytorch3d_renderer.py), [geoss/renderers/nvdiffrast_renderer.py](D:/VsCode/MVG/Base/SS_Flow/geoss/renderers/nvdiffrast_renderer.py): mesh 可微渲染接口，缺包 real-mode fail-fast。
- [geoss/geometry/colmap_utils.py](D:/VsCode/MVG/Base/SS_Flow/geoss/geometry/colmap_utils.py), [geoss/geometry/differentiable_camera.py](D:/VsCode/MVG/Base/SS_Flow/geoss/geometry/differentiable_camera.py), [geoss/geometry/pointcloud_ops.py](D:/VsCode/MVG/Base/SS_Flow/geoss/geometry/pointcloud_ops.py), [geoss/geometry/mesh_ops.py](D:/VsCode/MVG/Base/SS_Flow/geoss/geometry/mesh_ops.py): pycolmap/kornia/open3d/trimesh 接入点。
- [geoss/io/asset_io.py](D:/VsCode/MVG/Base/SS_Flow/geoss/io/asset_io.py): PLY/GLB/OBJ/Gaussian PLY 读写。
- [geoss/metrics](D:/VsCode/MVG/Base/SS_Flow/geoss/metrics): render、Gaussian、geometry metrics，含 Chamfer/F-score/floating splat/opacity/scale。
- [geoss/losses/render_losses.py](D:/VsCode/MVG/Base/SS_Flow/geoss/losses/render_losses.py): render-level RGB/mask/depth/SSIM/LPIPS/DINO/multiview loss 接口。
- [geoss/sfm/hloc_pipeline.py](D:/VsCode/MVG/Base/SS_Flow/geoss/sfm/hloc_pipeline.py): hloc/poselib/pycolmap SfM pipeline 接口。
- [geoss/baselines](D:/VsCode/MVG/Base/SS_Flow/geoss/baselines): Nerfstudio export 和 Splatfacto baseline runner。
- [geoss/visualization/vis_debug.py](D:/VsCode/MVG/Base/SS_Flow/geoss/visualization/vis_debug.py): 输出 aligned pointmap、anchors、voxels、camera JSON、metrics summary。

**接入到现有路径**
- [geoss/utils/run_mode.py](D:/VsCode/MVG/Base/SS_Flow/geoss/utils/run_mode.py): 增加配置依赖校验，例如 real_eval + gsplat metrics 但没装 `gsplat` 会直接报错。
- [geoss/models/ray_evidence_sampler.py](D:/VsCode/MVG/Base/SS_Flow/geoss/models/ray_evidence_sampler.py) 和 [geoss/slat/models/active_voxel_projector.py](D:/VsCode/MVG/Base/SS_Flow/geoss/slat/models/active_voxel_projector.py): 投影统一走 differentiable camera wrapper。
- [geoss/integration/real_trellis_pipeline.py](D:/VsCode/MVG/Base/SS_Flow/geoss/integration/real_trellis_pipeline.py): 保存真实 TRELLIS 输出时附带 Gaussian statistics。
- [scripts/eval_geovis_slat.py](D:/VsCode/MVG/Base/SS_Flow/scripts/eval_geovis_slat.py), [scripts/eval_sparse_ray_geoss.py](D:/VsCode/MVG/Base/SS_Flow/scripts/eval_sparse_ray_geoss.py): 输出 JSON + CSV，支持 render/geometry/Gaussian 指标。
- [scripts/eval_ablation_table.py](D:/VsCode/MVG/Base/SS_Flow/scripts/eval_ablation_table.py): 汇总 ablation JSON/CSV 表。

**依赖与配置**
- 新增拆分 requirements: [requirements/base.txt](D:/VsCode/MVG/Base/SS_Flow/requirements/base.txt), `geometry.txt`, `gsplat.txt`, `sfm.txt`, `render.txt`, `visualization.txt`, `dev.txt`, `all.txt`。
- 新增说明: [docs/DEPENDENCIES.md](D:/VsCode/MVG/Base/SS_Flow/docs/DEPENDENCIES.md)。
- 更新 real/dry-run/ablation 配置，加入 `dependencies/rendering/geometry/anchors/evaluation` 开关。
- 新增 ablation 配置：`with_gsplat_render_loss`, `no_pose_check`, `manual_projection`, `render_metrics_only`, `pytorch3d_mesh_loss`, `nvdiffrast_renderer`, `nerfstudio_splatfacto_baseline`。

**验证**
- `python -m compileall geoss scripts tests` 已通过。
- `python -m geoss.utils.env_check` 已通过，并显示当前环境缺 `torch`、`gsplat`、`pycolmap`、`kornia` 等。
- 未跑 pytest：当前环境没有 `pytest`，且没有 `torch`，张量单测无法执行。

**仍需真实验证**
真实 `real_train/real_eval/real_infer` 还需要安装匹配 PyTorch/CUDA 的 `gsplat`、`kornia`、`pycolmap`、`open3d`、`point-cloud-utils` 等，并提供 VGGT/TRELLIS checkpoints 与真实数据。现在缺关键库会 fail-fast，不会 silent fallback。`git status` 中的 `D da` 是既有删除状态，我没有处理它。|

# 3
已继续完成这轮“反向验收 + 修伪实现 + 主干防跑偏”审查。没有继续新增大模块，只做了边界收紧、fail-fast、hook 明确化和测试补充。

**结论**
Main backbone 仍是 `VGGT + TRELLIS`。当前没有发现主流程被改成 `COLMAP+3DGS`、`Nerfstudio/Splatfacto` 或 `gsplat-native optimizer` 的证据。`gsplat / pycolmap / kornia / nerfstudio` 现在定位为辅助渲染、几何校验、loss、baseline/exporter，不替代 VGGT/TRELLIS 主模型。

**主干架构检查**
```text
Main backbone: VGGT + TRELLIS
VGGT used as geometry predictor: yes
VGGT output actually enters alignment: yes
alignment output actually enters RayEvidenceSampler: yes
alignment output actually enters ActiveVoxelProjector: yes, with aligned_camera/depth/pointmap path
TRELLIS SS Flow used as base velocity: yes
GeoSS Adapter only predicts residual: yes
TRELLIS SLAT Flow used as base velocity: yes, but real path currently requires dataset/checkpoint-provided trellis_slat_base_velocity
GeoVis-SLAT Adapter only predicts residual: yes
TRELLIS decoder used for final asset output: yes
gsplat only used for render/eval/loss: yes
nerfstudio only used as baseline/exporter: yes
pycolmap only used for pose check/SfM baseline/alignment aid: yes
No replacement of VGGT/TRELLIS by native 3DGS: yes
```

**已修正的关键问题**
- [D:\VsCode\MVG\Base\SS_Flow\geoss\integration\real_trellis_pipeline.py](D:/VsCode/MVG/Base/SS_Flow/geoss/integration/real_trellis_pipeline.py): 增加明确的 `ss_velocity_hook` / `slat_velocity_hook` API，主 sampler 仍以 TRELLIS base velocity 为主，只注入 adapter residual。
- [D:\VsCode\MVG\Base\SS_Flow\geoss\models\sparse_ray_geoss_adapter.py](D:/VsCode/MVG/Base/SS_Flow/geoss/models/sparse_ray_geoss_adapter.py): real 路径缺 `v_base` 现在直接报错，不再允许 zero fallback。
- [D:\VsCode\MVG\Base\SS_Flow\scripts\train_sparse_ray_ss_velocity.py](D:/VsCode/MVG/Base/SS_Flow/scripts/train_sparse_ray_ss_velocity.py): real train 删除 random cond 分支；保持 `target_residual = target_v - v_base.detach()`。
- [D:\VsCode\MVG\Base\SS_Flow\scripts\train_geovis_slat.py](D:/VsCode/MVG/Base/SS_Flow/scripts/train_geovis_slat.py): real train 缺 dataloader、SLAT token、base velocity 时 fail-fast，不再生成 synthetic latent。
- [D:\VsCode\MVG\Base\SS_Flow\scripts\train_sparse_ray_geoss.py](D:/VsCode/MVG/Base/SS_Flow/scripts/train_sparse_ray_geoss.py): real train 禁止 mock VGGT、synthetic batch、random SS tokens、zero base velocity。
- [D:\VsCode\MVG\Base\SS_Flow\geoss\slat\utils\slat_token_mapping.py](D:/VsCode/MVG/Base/SS_Flow/geoss/slat/utils/slat_token_mapping.py) 及 SLAT adapter/aggregator: 移除 hard truncate，维度不匹配时 fail-fast 或要求 learned projector。
- [D:\VsCode\MVG\Base\SS_Flow\scripts\render_objaverse_cars.py](D:/VsCode/MVG/Base/SS_Flow/scripts/render_objaverse_cars.py): 去掉 `NotImplementedError`，改成外部 Blender 预处理的显式 fail-fast。
- dry-run 输出已统一标记 `not_for_paper_metrics=true`，相关 mock/synthetic 只允许 dry-run/tests。

**伪实现 / fallback 搜索处理**
重点搜索了：
```text
TODO / FIXME / pass / NotImplementedError / not implemented / dummy / mock / synthetic /
torch.randn / zeros_like / random cond / fake / fallback / placeholder
```

处理结果：
- `torch.randn` 仍存在于 dry-run、tests、TRELLIS diffusion sampling noise 中，未进入 real_train fake condition/base velocity。
- `mock/synthetic` 保留在 dry-run、测试、文档说明里，real 路径已 fail-fast。
- `baseline/splatfacto/ns-train` 只在 `geoss/baselines/` 和静态测试断言中出现，未被 main pipeline import 为主模型。
- `NotImplementedError` / `RuntimeError("not implemented")` 主路径已清掉。
- metrics 不返回随机数或 dummy 0；缺依赖时返回 unavailable 或 fail-fast，取决于 real-mode 配置。

**real_trellis_pipeline 审查**
该文件现在真实职责清晰：
```text
load real TRELLIS pipeline/checkpoint: yes
load VGGT/aligned geometry directly: receives geoss_context/geovis_slat_context from caller
encode image condition: yes
call TRELLIS SS sampler: yes
inject GeoSS residual: yes
call TRELLIS SLAT sampler: yes
inject GeoVis-SLAT residual: yes
call TRELLIS decoder: yes
save real Gaussian/mesh/latent outputs: yes
return structured outputs for eval/render: yes
```

它不是 VGGT/alignment 生成器，而是接收前序 aligned context 后接入 TRELLIS，这个边界是合理的。

**仍不能声称完成的地方**
- 当前本机未安装 `torch`，所以不能实际跑 tensor 单测、真实训练或真实推理。
- 当前没有真实 VGGT/TRELLIS checkpoint 和真实数据集，因此 real path 只能验证 fail-fast，不能验证论文级指标。
- SLAT real training 目前依赖数据集中提供 `trellis_slat_base_velocity` 或上游 frozen TRELLIS flow 输出；没有该字段会 fail-fast，这是正确边界。
- LPIPS/DINO 属于可用接口，不应在未安装模型/依赖前声称指标有效。
- pycolmap/kornia/gsplat/open3d 等都未安装，本机只能验证 optional dependency gate，不能验证真实几何/渲染数值。

**验证结果**
已通过：
```bash
python -m compileall geoss scripts tests
python -m geoss.utils.env_check
```

`env_check` 结果摘要：
```text
Python: 3.12.0
torch: not installed
CUDA: unavailable
gsplat: false
pycolmap: false
kornia: false
pytorch3d: false
nvdiffrast: false
kaolin: false
open3d: false
```

未能运行：
```bash
python -m pytest -q
```
原因：当前环境缺 `pytest`。

**新增/修改测试覆盖**
已补强：
- dry_run full path smoke 标记测试
- real-mode fail-fast 静态/配置测试
- no-random-cond / no-zero-base static test
- architecture ownership static test
- main pipeline 禁止导入 baselines/splatfacto 为主模型
- main pipeline 必须保留 VGGT/TRELLIS/adapter/decoder ownership flags

**下一步真实服务器验证命令**
在装好真实环境后跑：
```bash
python -m geoss.utils.env_check
python -m pytest -q
python scripts/train_sparse_ray_ss_velocity.py --config configs/real_train_ss.yaml
python scripts/train_geovis_slat.py --config configs/real_train_slat.yaml
python scripts/infer_sparse_ray_geoss_ss.py --config configs/real_infer.yaml
python scripts/eval_sparse_ray_geoss.py --config configs/real_eval.yaml
```

当前可以负责任地说：代码主干没有跑偏，real/dry_run 边界已明显收紧，伪实现入口已清理或隔离。但在缺真实 checkpoint、torch、gsplat/kornia/pycolmap 等依赖的本机环境下，不能声称真实训练指标或最终 3D 资产质量已经有效。