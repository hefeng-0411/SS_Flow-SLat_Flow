# 项目完整命令指南（详细中文版）

---

## 服务器目录结构概览

在开始之前，请先确认以下目录结构（基于当前配置文件假设）：

```bash
# ============================
# 数据集根目录
# ============================
/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97

# ============================
# 代码仓库目录
# ============================
/mnt/sda2/hef/Base/SS_Flow-SLat_Flow    # 主项目仓库（SS Flow + SLAT Flow）
/mnt/sda2/hef/Base/TRELLIS              # TRELLIS 3D生成模型仓库
/mnt/sda2/hef/Base/vggt                 # VGGT 视觉几何基础模型仓库
```

> **说明**：
> - **SS_Flow-SLat_Flow**：本项目核心仓库，包含所有训练、推理、评估和可视化脚本。
> - **TRELLIS**：微软的 3D 资产生成模型（`TRELLIS-image-large`），用作稀疏射线解码器后端。
> - **vggt**：Meta 的 VGGT-1B 视觉几何基础模型，提供特征提取能力。
> - **数据集**：MeshFleet 数据集，以 git LFS hash 形式存储。

---

## 第 1 节：进入项目目录

所有后续命令默认在项目根目录下执行：

```bash
cd /mnt/sda2/hef/Base/SS_Flow-SLat_Flow
```

> **提示**：建议在执行任何命令前先确认当前工作目录正确，且 Python 环境已激活（如 `conda activate <env_name>`）。

---

## 第 2 节：推荐完整多 GPU 训练流程（一键式两阶段）

这是**最推荐**的训练方式，自动完成从 Stage 1 到 Stage 2 的完整训练流程。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml
```

### 执行流程详解：

| 阶段 | 说明 | 对应脚本/配置 |
|------|------|---------------|
| **Stage 1** | SS 速度残差训练（Sparse Sampling Velocity Residual） | `configs/real_train_ss.yaml` |
| **自动触发 Stage 2** | 当 Stage 1 收敛（指标 plateau）后，自动启动 SLAT 训练 | `configs/real_train_slat_only.yaml` |

### 关键特性：
- **GPU 分配**：使用 GPU 4、5、6、7，共 4 个进程并行
- **自适应批大小**：由配置文件控制，自动探测可用显存并设置最优 batch size
- **早停机制（Early Stopping）**：在配置文件中启用，避免过拟合
- **自动阶段切换**：Stage 1 收敛后自动启动 Stage 2 的 SLAT 训练脚本：
  ```bash
  scripts/train_geovis_slat.py --config configs/real_train_slat_only.yaml
  ```

> **⚠️ 注意**：此命令会持续运行直到两个阶段全部完成。请确保服务器稳定且有足够的磁盘空间存储 checkpoint。

---

## 第 3 节：仅执行 Stage 1 —— SS 速度训练

如果你只想单独训练 Stage 1（例如调试、超参搜索、或者手动控制阶段切换），可以使用以下命令。

### 3.1 多 GPU 训练

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml
```

**输出文件**：
| 文件 | 路径 | 说明 |
|------|------|------|
| 最新 checkpoint | `outputs/real_train_ss/ss_velocity_adapter_last.pt` | 每个保存间隔自动覆盖 |
| 最优 checkpoint | `outputs/real_train_ss/ss_velocity_adapter_best.pt` | 验证指标最优时保存 |

### 3.2 单 GPU 训练

适用于调试或资源有限的场景：

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml
```

> **说明**：仅使用 GPU 4（单张 A800），训练速度会显著降低，但方便调试和快速验证。

### 3.3 恢复 Stage 1 训练（断点续训）

如果训练中断，可以从上次保存的 checkpoint 恢复（包括优化器状态和 adapter 权重）：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml \
  --resume outputs/real_train_ss/ss_velocity_adapter_last.pt
```

> **关键参数**：`--resume` 会加载 checkpoint 中的模型权重、优化器状态、学习率调度器状态和当前训练步数，实现无缝续训。

---

## 第 4 节：仅执行 Stage 2 —— SLAT 训练

当你禁用了自动 Stage 2 触发，或者想单独重训/微调 SLAT 模块时使用。

### 4.1 多 GPU 训练

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_geovis_slat.py \
  --config configs/real_train_slat_only.yaml
```

**输出文件**：
| 文件 | 路径 | 说明 |
|------|------|------|
| 最新 checkpoint | `outputs/real_train_slat/geovis_slat_adapter_last.pt` | 定期保存 |
| 最优 checkpoint | `outputs/real_train_slat/geovis_slat_adapter_best.pt` | 验证集最优 |

### 4.2 单 GPU 训练

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/train_geovis_slat.py \
  --config configs/real_train_slat_only.yaml
```

### 4.3 恢复 Stage 2 训练

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_geovis_slat.py \
  --config configs/real_train_slat_only.yaml \
  --resume outputs/real_train_slat/geovis_slat_adapter_last.pt
```

> **提示**：Stage 2 依赖 Stage 1 的输出（SS 速度 adapter），请确保 Stage 1 已完成且 checkpoint 路径在配置文件中正确指向。

---

## 第 5 节：高级多阶段顺序启动器

这是一个更高层次的启动脚本，封装了完整的分阶段训练逻辑，包含 **batch size 探测**、**阶段间自动衔接**和**重启恢复**功能。

### 5.1 完整四阶段训练

```bash
python scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --gpus 4,5,6,7 \
  --nproc_per_node 4 \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --vggt_pretrained facebook/VGGT-1B \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --output_root outputs/meshfleet_full_4gpu_sequence
```

**阶段划分**：

| 阶段 | 名称 | 说明 |
|------|------|------|
| `stage1` | GeoSS | 几何稀疏采样基础训练 |
| `stage2` | SS Velocity | 稀疏射线速度残差训练 |
| `slat` | GeoVis SLAT | 地理视觉 SLAT 适配训练 |
| `slat_joint` | SLAT Joint | SLAT 联合微调 |

> **特性**：每个阶段启动前会自动探测可用显存，确定最优 batch size（batch probing）。

### 5.2 从 Stage 2 开始训练

如果你已经完成了 Stage 1，想从 Stage 2 继续：

```bash
python scripts/launch_meshfleet_multigpu_sequence.py \
  --data_root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --gpus 4,5,6,7 \
  --nproc_per_node 4 \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large \
  --output_root outputs/meshfleet_full_4gpu_sequence \
  --start_at stage2
```

> **`--start_at` 可选值**：`stage1`、`stage2`、`slat`、`slat_joint`

---

## 第 6 节：完整联合训练（Joint Training）

> ⚠️ **警告**：虽然 `configs/real_train_full_joint.yaml` 配置文件存在，但当前 `scripts/train_sparse_ray_joint.py` 脚本并未像 SS/SLAT 专用脚本那样完整消费该配置。**建议优先使用第 5 节的高级启动器**来实现联合式分阶段训练。

如果你仍然想尝试现有的联合训练入口：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_joint.py \
  --config configs/sparse_ray_joint.yaml \
  --device cuda \
  --meshfleet_root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --meshfleet_split train \
  --vggt_root /mnt/sda2/hef/Base/vggt \
  --vggt_pretrained facebook/VGGT-1B \
  --trellis_root /mnt/sda2/hef/Base/TRELLIS \
  --trellis_model_path microsoft/TRELLIS-image-large
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `--config` | 联合训练配置文件 |
| `--device` | 计算设备 |
| `--meshfleet_root` | MeshFleet 数据集根目录 |
| `--meshfleet_split` | 数据划分（train/val/test） |
| `--vggt_root` | VGGT 仓库路径 |
| `--vggt_pretrained` | VGGT 预训练模型名称 |
| `--trellis_root` | TRELLIS 仓库路径 |
| `--trellis_model_path` | TRELLIS 模型标识 |

> ⚠️ **请在将其作为主要训练路径之前，仔细验证输出结果的正确性。**

---

## 第 7 节：推理 / 生成

### 7.1 SS + TRELLIS 解码推理（使用 Stage 1 checkpoint）

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_sparse_ray_geoss_ss.py \
  --config configs/real_infer.yaml \
  --ss_adapter_checkpoint outputs/real_train_ss/ss_velocity_adapter_last.pt \
  --output_dir outputs/real_infer_ss
```

**说明**：
- 加载 Stage 1 训练好的 SS 速度 adapter checkpoint
- 使用 TRELLIS 解码器生成 3D 输出
- 推理数据来自 `configs/real_infer.yaml` 中配置的 test split

### 7.2 推理指定 MeshFleet 索引的对象

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_sparse_ray_geoss_ss.py \
  --config configs/real_infer.yaml \
  --ss_adapter_checkpoint outputs/real_train_ss/ss_velocity_adapter_last.pt \
  --meshfleet_index 0 \
  --output_dir outputs/real_infer_ss_idx0
```

> **`--meshfleet_index`**：从 test split 中选择指定索引的对象进行推理，方便单个样本调试和可视化。

### 7.3 自定义数据集/输入路径

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_sparse_ray_geoss_ss.py \
  --config configs/real_infer.yaml \
  --ss_adapter_checkpoint <PATH_TO_STAGE1_CKPT> \
  --meshfleet_root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --meshfleet_split test \
  --output_dir outputs/real_infer_custom
```

### 7.4 SLAT 推理

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_geovis_slat.py \
  --config configs/real_infer.yaml \
  --slat_adapter_checkpoint outputs/real_train_slat/geovis_slat_adapter_last.pt \
  --geovis_context <PATH_TO_GEOVIS_CONTEXT_NPZ> \
  --input <PATH_TO_INPUT_IMAGE_OR_IMAGE_DIR> \
  --output_dir outputs/real_infer_slat
```

**重要参数**：
| 参数 | 说明 |
|------|------|
| `--slat_adapter_checkpoint` | Stage 2 训练产出的 SLAT adapter checkpoint |
| `--geovis_context` | GeoVis 上下文文件（`.npz`），**必须由 SS/GeoVis 上下文生成阶段产出** |
| `--input` | 输入图像路径或图像目录 |

> ⚠️ **前置依赖**：SLAT 推理需要 `--geovis_context` 参数提供的上下文文件，该文件通常由 SS 推理阶段生成。请确保先完成 SS 推理。

---

## 第 8 节：评估

### 8.1 评估 SS 占用预测（Occupancy Prediction）

```bash
python scripts/eval_sparse_ray_geoss.py \
  --real_eval \
  --prediction <PATH_TO_PREDICTION_NPZ> \
  --gt_occ <PATH_TO_GT_OCC_PT> \
  --output_dir outputs/eval_sparse_ray_geoss
```

**输入格式要求**：
| 参数 | 格式 | 说明 |
|------|------|------|
| `--prediction` | `.npz` 文件，包含 key `"occ"` | 模型预测的占用结果 |
| `--gt_occ` | `.pt` 文件（tensor 或 dict 含 `"gt_occ"` key） | Ground truth 占用标签 |

### 8.2 评估 SLAT 渲染 / 高斯输出

```bash
python scripts/eval_geovis_slat.py \
  --real_eval \
  --input_dir outputs/real_infer_slat \
  --prediction <PATH_TO_PRED_RENDER_NPZ> \
  --gt_render <PATH_TO_GT_RENDER_NPZ> \
  --gaussian_ply outputs/real_infer_ss/asset_gaussian.ply \
  --output_dir outputs/eval_geovis_slat
```

**输入格式要求**：
| 参数 | 格式 | 说明 |
|------|------|------|
| `--prediction` | `.npz`，含 `"rgb"` 和可选 `"mask"` | 预测渲染结果 |
| `--gt_render` | `.npz`，含 `"rgb"` 和可选 `"mask"` | Ground truth 渲染 |
| `--gaussian_ply` | `.ply` 文件 | 高斯点云文件，用于计算高斯统计指标 |

### 8.3 评估几何点云指标

如果你有预测和 GT 的点云 `.npz` 文件：

```bash
python scripts/eval_geovis_slat.py \
  --real_eval \
  --input_dir outputs/real_infer_slat \
  --prediction <PATH_TO_PRED_RENDER_NPZ> \
  --gt_render <PATH_TO_GT_RENDER_NPZ> \
  --pred_points <PATH_TO_PRED_POINTS_NPZ> \
  --gt_points <PATH_TO_GT_POINTS_NPZ> \
  --output_dir outputs/eval_geovis_slat_geometry
```

> **点云文件要求**：`.npz` 文件中应包含 key `"points"`。

### 8.4 消融实验汇总表

对多个实验运行目录进行批量扫描，生成统一的结果表格：

```bash
python scripts/eval_ablation_table.py \
  --runs_dir outputs/ablation_runs \
  --output_json outputs/ablation_summary.json \
  --output_csv outputs/ablation_summary.csv
```

> **说明**：自动扫描 `--runs_dir` 下的所有子目录，汇总各实验指标，输出 JSON 和 CSV 格式的结果表。适用于论文中的消融实验对比。

---

## 第 9 节：定性分析与可视化

### 9.1 可视化 SS/GeoSS Demo 产物

```bash
python scripts/visualize_sparse_ray_geoss.py \
  --output_dir outputs/vis_sparse_ray_geoss
```

> **输出**：生成 `visualize_sparse_ray_geoss_demo.ply` 文件，可用 MeshLab、CloudCompare 等工具查看。

### 9.2 可视化 SLAT 推理结果

```bash
python scripts/visualize_geovis_slat.py \
  --input_dir outputs/real_infer_slat \
  --output_dir outputs/vis_geovis_slat
```

> **输入依赖**：会读取 `slat_visibility_debug.npz` 和 `original_slat.npz`（如果存在），用于生成可视化结果。

### 9.3 绘制训练曲线

**单阶段训练**：
```bash
python scripts/plot_training_metrics.py \
  --output_root outputs/real_train_ss \
  --report_dir outputs/reports/ss_training \
  --smooth 0.9 \
  --formats png,pdf
```

**启动器多阶段训练**：
```bash
python scripts/plot_training_metrics.py \
  --output_root outputs/meshfleet_full_4gpu_sequence \
  --report_dir outputs/reports/full_sequence \
  --smooth 0.9 \
  --formats png,pdf
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `--output_root` | 训练输出根目录（包含 JSONL 训练日志） |
| `--report_dir` | 图表输出目录 |
| `--smooth` | 指数移动平均平滑系数（0.9 = 较强平滑），用于平滑训练曲线 |
| `--formats` | 输出图片格式，逗号分隔（支持 `png`、`pdf`） |

---

## 第 10 节：Checkpoint 路径速查表

以下是所有常用 checkpoint 路径的汇总，方便复制粘贴：

```bash
# =============================================
# Stage 1: SS 速度 Adapter
# =============================================
outputs/real_train_ss/ss_velocity_adapter_last.pt    # 最新（用于推理/续训）
outputs/real_train_ss/ss_velocity_adapter_best.pt    # 最优（用于最终评估）

# =============================================
# Stage 2: SLAT Adapter
# =============================================
outputs/real_train_slat/geovis_slat_adapter_last.pt  # 最新
outputs/real_train_slat/geovis_slat_adapter_best.pt  # 最优

# =============================================
# 启动器顺序训练输出（四阶段）
# =============================================
outputs/meshfleet_full_4gpu_sequence/stage1_geoss/geoss_adapter_last.pt
outputs/meshfleet_full_4gpu_sequence/stage2_ss_velocity/ss_velocity_adapter_last.pt
outputs/meshfleet_full_4gpu_sequence/stage3_geovis_slat/geovis_slat_adapter_last.pt
outputs/meshfleet_full_4gpu_sequence/stage4_geovis_slat_joint/geovis_slat_adapter_last.pt
```

> **建议**：在推理和评估时，优先使用 `_best.pt` checkpoint 以获得最优指标；续训时使用 `_last.pt` 以保留完整的训练状态。

---

## 第 11 节：快速迭代 / 调试命令

### 11.1 SS 快速冒烟测试（4 GPU，20 步）

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml \
  --steps 20 \
  --save_every 10 \
  --output_dir outputs/debug_ss_20step
```

> **用途**：快速验证训练循环是否正常工作（数据加载、前向传播、反向传播、checkpoint 保存），使用真实数据和模型但仅训练 20 步。

### 11.2 SLAT 快速冒烟测试（4 GPU，20 步）

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_geovis_slat.py \
  --config configs/real_train_slat_only.yaml \
  --steps 20 \
  --save_every 10 \
  --output_dir outputs/debug_slat_20step
```

### 11.3 干跑模式（Dry Run）—— 无真实指标

```bash
python scripts/train_sparse_ray_ss_velocity.py \
  --config configs/dry_run_debug.yaml \
  --dry_run true \
  --output_dir outputs/dry_run_ss
```

> **说明**：使用模拟/合成数据运行，不产生有效的论文指标。纯粹用于验证代码逻辑和流程是否正确，适合代码开发阶段的快速调试。

> ⚠️ **注意**：干跑模式的结果**不能**用于论文或正式评估。

---

## 附录：常用操作流程总结

### 流程 A：完整训练 → 推理 → 评估（推荐）

```bash
# 1. 进入项目
cd /mnt/sda2/hef/Base/SS_Flow-SLat_Flow

# 2. 完整两阶段训练
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml

# 3. SS 推理
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_sparse_ray_geoss_ss.py \
  --config configs/real_infer.yaml \
  --ss_adapter_checkpoint outputs/real_train_ss/ss_velocity_adapter_best.pt \
  --output_dir outputs/real_infer_ss

# 4. SLAT 推理
CUDA_VISIBLE_DEVICES=4 python \
  scripts/infer_geovis_slat.py \
  --config configs/real_infer.yaml \
  --slat_adapter_checkpoint outputs/real_train_slat/geovis_slat_adapter_best.pt \
  --geovis_context <SS推理产出的上下文npz> \
  --input <输入图像> \
  --output_dir outputs/real_infer_slat

# 5. 评估
python scripts/eval_sparse_ray_geoss.py \
  --real_eval \
  --prediction <推理产出npz> \
  --gt_occ <GT占用标签> \
  --output_dir outputs/eval_final
```

### 流程 B：快速调试验证

```bash
# 1. 冒烟测试 SS 训练
CUDA_VISIBLE_DEVICES=4 python \
  scripts/train_sparse_ray_ss_velocity.py \
  --config configs/real_train_ss.yaml \
  --steps 20 --save_every 10 \
  --output_dir outputs/debug_quick

# 2. 干跑验证代码逻辑
python scripts/train_sparse_ray_ss_velocity.py \
  --config configs/dry_run_debug.yaml \
  --dry_run true \
  --output_dir outputs/dry_run_quick
```

---

> **最后提醒**：
> - 所有 `CUDA_VISIBLE_DEVICES` 中的 GPU 编号请根据实际服务器可用 GPU 调整。
> - `torchrun --nproc_per_node` 的值应与可见 GPU 数量一致。
> - 训练前请确保配置文件中的数据集路径、模型路径、输出路径均正确。
> - 定期检查磁盘空间，checkpoint 文件可能会占用大量存储。