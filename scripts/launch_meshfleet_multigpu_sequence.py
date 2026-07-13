from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional


OOM_PATTERNS = (
    "out of memory",
    "cuda oom",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cuda error: out of memory",
    "torch.cuda.outofmemoryerror",
    "nccl error",
    "processgroupnccl",
)

TRANSIENT_SETUP_PATTERNS = (
    "remotedisconnected",
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection timed out",
    "read timed out",
    "temporary failure in name resolution",
    "urlopen error",
    "http error 5",
    "tlsv1_alert",
    "torch.hub",
)


DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
)


@dataclass
class Stage:
    name: str
    script: str
    config: str
    output_dir: Path
    steps: int
    max_batch_size: int
    resume_path: Optional[Path]
    best_path: Optional[Path]
    extra_args: List[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential multi-GPU MeshFleet training launcher with OOM batch probing.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs/meshfleet_multigpu_sequence")
    parser.add_argument("--gpus", type=str, default=None, help="CUDA_VISIBLE_DEVICES value, for example 0,1,2,3.")
    parser.add_argument("--nproc_per_node", type=int, default=0)
    parser.add_argument("--probe_steps", type=int, default=3)
    parser.add_argument("--probe_max_vram_util", type=float, default=0.96)
    parser.add_argument("--min_batch_size", type=int, default=1)
    parser.add_argument("--no_auto_batch", action="store_true")
    parser.add_argument("--batch_probe_strategy", choices=["binary", "halve"], default="binary")
    parser.add_argument("--oom_retry_limit", type=int, default=8)
    parser.add_argument("--setup_retry_limit", type=int, default=3)
    parser.add_argument("--restart_sleep_seconds", type=float, default=20.0)
    parser.add_argument("--force_rerun_completed", action="store_true")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--pin_memory", type=str, default="true")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Diagnostic compatibility flag; Stage 2 is architected to run without relying on this.")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--latent_tokens", type=int, default=4096)
    parser.add_argument("--active_tokens", type=int, default=4096)
    parser.add_argument("--meshfleet_split", type=str, default="train")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--vggt_root", type=str, default=None)
    parser.add_argument("--vggt_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_pretrained", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument("--trellis_model_path", type=str, default=None)
    parser.add_argument("--torch_hub_dir", type=str, default=None)
    parser.add_argument("--dinov2_repo", type=str, default=None)
    parser.add_argument("--cuda_alloc_conf", type=str, default="max_split_size_mb:512,garbage_collection_threshold:0.9")
    parser.add_argument("--nccl_socket_ifname", type=str, default=None)
    parser.add_argument("--stage1_steps", type=int, default=100000)
    parser.add_argument("--stage2_steps", type=int, default=100000)
    parser.add_argument("--slat_steps", type=int, default=100000)
    parser.add_argument("--slat_joint_steps", type=int, default=30000)
    parser.add_argument("--stage1_max_batch_size", type=int, default=6)
    parser.add_argument("--stage2_max_batch_size", type=int, default=8)
    parser.add_argument("--slat_max_batch_size", type=int, default=8)
    parser.add_argument("--slat_joint_max_batch_size", type=int, default=6)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_raw_residual_weight", type=float, default=1.0)
    parser.add_argument("--slat_lr", type=float, default=1e-4)
    parser.add_argument("--slat_joint_lr", type=float, default=5e-5)
    parser.add_argument("--slat_raw_residual_weight", type=float, default=1.0)
    parser.add_argument("--slat_effective_residual_weight", type=float, default=1.0)
    parser.add_argument("--slat_grad_accum_steps", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--fault_tolerant_save_every", type=int, default=25)
    parser.add_argument("--visualize_every", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--disable_early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_min_steps", type=int, default=300)
    parser.add_argument("--early_stop_warmup_steps", type=int, default=200)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.002)
    parser.add_argument("--early_stop_ema", type=float, default=0.6)
    parser.add_argument("--early_stop_window", type=int, default=8)
    parser.add_argument("--adaptive_min_batch_size", type=int, default=1)
    parser.add_argument("--adaptive_target_utilization", type=float, default=0.92)
    parser.add_argument("--adaptive_low_utilization", type=float, default=0.82)
    parser.add_argument("--adaptive_grow_patience_steps", type=int, default=8)
    parser.add_argument("--adaptive_cooldown_steps", type=int, default=3)
    parser.add_argument("--adaptive_oom_retries", type=int, default=8)
    parser.add_argument("--max_train_hours_per_stage", type=float, default=0.0)
    parser.add_argument("--start_at", choices=["stage1", "stage2", "slat", "slat_joint"], default="stage1")
    parser.add_argument("--stop_after", choices=["stage1", "stage2", "slat", "slat_joint"], default="slat_joint")
    parser.add_argument("--skip_slat_joint", action="store_true")
    parser.add_argument("--speculative_handoff_gpus", type=str, default=None, help="Optional spare CUDA_VISIBLE_DEVICES for concurrent next-stage warmup/probing.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env = _sanitize_torchrun_parent_env(env)
    env = _configure_training_env(env, args)
    visible_count = _visible_gpu_count(env)
    requested_nproc = args.nproc_per_node if args.nproc_per_node > 0 else visible_count
    nproc = min(requested_nproc, visible_count)
    if nproc < 1:
        raise RuntimeError("No visible CUDA GPU was found. Set --gpus or CUDA_VISIBLE_DEVICES correctly.")
    if requested_nproc != nproc:
        print(
            f"Requested nproc_per_node={requested_nproc} but CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')} "
            f"exposes only {visible_count} GPU(s); clamping nproc_per_node to {nproc}.",
            flush=True,
        )
    print(f"Launcher device topology: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}, nproc_per_node={nproc}", flush=True)

    stages = _make_stages(args, root, output_root)
    stages = _slice_stages(stages, args.start_at, args.stop_after)
    if args.skip_slat_joint:
        stages = [stage for stage in stages if stage.name != "slat_joint"]

    selected = {}
    handoff = HandoffCoordinator(args, env, nproc)
    for index, stage in enumerate(stages):
        next_stage = stages[index + 1] if index + 1 < len(stages) else None
        if _stage_is_complete(stage) and not args.force_rerun_completed:
            step = _checkpoint_step(_resolve_resume_path(stage))
            selected[stage.name] = "already_complete"
            print(f"\n==== {stage.name}: already complete at checkpoint step={step}; skipping ====", flush=True)
            continue
        if next_stage is not None:
            handoff.start_when_ready(next_stage, current_stage=stage)
        print(f"\n==== {stage.name}: probing per-GPU batch size up to {stage.max_batch_size} on {nproc} GPUs ====", flush=True)
        batch_size = handoff.consume(stage.name) or (stage.max_batch_size if args.no_auto_batch else _probe_batch(stage, args, env, nproc))
        selected[stage.name] = batch_size
        print(f"==== {stage.name}: selected per-GPU batch_size={batch_size}, global_batch_size={batch_size * nproc} ====", flush=True)
        _run_full_stage(stage, args, env, nproc, batch_size)
    handoff.close()

    summary = {"nproc_per_node": nproc, "selected_per_gpu_batch_size": selected}
    (output_root / "multigpu_sequence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


class HandoffCoordinator:
    def __init__(self, args: argparse.Namespace, env: dict, nproc: int) -> None:
        self.args = args
        self.base_env = dict(env)
        self.nproc = nproc
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="handoff")
        self.futures: dict[str, Future] = {}

    def start_when_ready(self, next_stage: Stage, *, current_stage: Stage) -> None:
        if not self.args.speculative_handoff_gpus or next_stage.name in self.futures:
            return
        env = dict(self.base_env)
        env["CUDA_VISIBLE_DEVICES"] = self.args.speculative_handoff_gpus
        nproc = len([item for item in self.args.speculative_handoff_gpus.split(",") if item.strip()])
        if nproc < 1:
            return
        self.futures[next_stage.name] = self.pool.submit(self._wait_and_probe, next_stage, current_stage, env, nproc)

    def consume(self, stage_name: str) -> Optional[int]:
        future = self.futures.pop(stage_name, None)
        if future is None:
            return None
        return future.result()

    def close(self) -> None:
        for future in self.futures.values():
            future.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _wait_and_probe(self, next_stage: Stage, current_stage: Stage, env: dict, nproc: int) -> int:
        while not _resolve_resume_path(current_stage):
            time.sleep(2.0)
        print(f"{next_stage.name}: speculative handoff probing on spare GPUs={env['CUDA_VISIBLE_DEVICES']}", flush=True)
        return next_stage.max_batch_size if self.args.no_auto_batch else _probe_batch(next_stage, self.args, env, nproc)


def _make_stages(args: argparse.Namespace, root: Path, output_root: Path) -> list[Stage]:
    stage1_out = output_root / "stage1_geoss"
    stage2_out = output_root / "stage2_ss_velocity"
    slat_out = output_root / "stage3_geovis_slat"
    slat_joint_out = output_root / "stage4_geovis_slat_joint"

    common_data = [
        "--device", "cuda",
        "--meshfleet_root", args.data_root,
        "--meshfleet_split", args.meshfleet_split,
        "--num_views", str(args.num_views),
        "--image_size", str(args.image_size),
        "--num_workers", str(args.num_workers),
        "--pin_memory", args.pin_memory,
        "--ddp_find_unused_parameters", "true" if args.ddp_find_unused_parameters else "false",
    ]
    if args.meshfleet_category:
        common_data += ["--meshfleet_category", args.meshfleet_category]
    early_stop_args = []
    if not args.disable_early_stop:
        early_stop_args = [
            "--early_stop", "true",
            "--early_stop_metric", "loss",
            "--early_stop_mode", "min",
            "--early_stop_patience", str(args.early_stop_patience),
            "--early_stop_min_steps", str(args.early_stop_min_steps),
            "--early_stop_warmup_steps", str(args.early_stop_warmup_steps),
            "--early_stop_min_delta", str(args.early_stop_min_delta),
            "--early_stop_relative_delta", "true",
            "--early_stop_ema", str(args.early_stop_ema),
            "--early_stop_window", str(args.early_stop_window),
            "--max_train_hours", str(args.max_train_hours_per_stage),
            "--save_best", "true",
        ]
    adaptive_args = [
        "--adaptive_batch", "true",
        "--adaptive_min_batch_size", str(args.adaptive_min_batch_size),
        "--adaptive_target_utilization", str(args.adaptive_target_utilization),
        "--adaptive_low_utilization", str(args.adaptive_low_utilization),
        "--adaptive_grow_patience_steps", str(args.adaptive_grow_patience_steps),
        "--adaptive_cooldown_steps", str(args.adaptive_cooldown_steps),
        "--adaptive_oom_retries", str(args.adaptive_oom_retries),
    ]
    vggt_args = []
    if args.vggt_root:
        vggt_args += ["--vggt_root", args.vggt_root]
    if args.vggt_checkpoint:
        vggt_args += ["--vggt_checkpoint", args.vggt_checkpoint]
    elif args.vggt_pretrained:
        vggt_args += ["--vggt_pretrained", args.vggt_pretrained]
    trellis_args = []
    if args.trellis_root:
        trellis_args += ["--trellis_root", args.trellis_root]
    if args.trellis_model_path:
        trellis_args += ["--trellis_model_path", args.trellis_model_path]
    slat_runtime_args = [
        "--raw_residual_weight", str(args.slat_raw_residual_weight),
        "--effective_residual_weight", str(args.slat_effective_residual_weight),
        "--grad_accum_steps", str(args.slat_grad_accum_steps),
    ]
    if args.torch_hub_dir:
        slat_runtime_args += ["--torch_hub_dir", args.torch_hub_dir]
    if args.dinov2_repo:
        slat_runtime_args += ["--dinov2_repo", args.dinov2_repo]
    stage1_geoss_checkpoint = stage1_out / "geoss_adapter_last.pt"
    if not stage1_geoss_checkpoint.exists() and (stage1_out / "geoss_adapter_best.pt").exists():
        stage1_geoss_checkpoint = stage1_out / "geoss_adapter_best.pt"

    return [
        Stage(
            name="stage1",
            script=str(root / "scripts" / "train_sparse_ray_geoss.py"),
            config=str(root / "configs" / "sparse_ray_geoss.yaml"),
            output_dir=stage1_out,
            steps=args.stage1_steps,
            max_batch_size=args.stage1_max_batch_size,
            resume_path=stage1_out / "geoss_adapter_last.pt",
            best_path=stage1_out / "geoss_adapter_best.pt",
            extra_args=common_data
            + early_stop_args
            + adaptive_args
            + vggt_args
            + [
                "--latent_tokens", str(args.latent_tokens),
                "--adaptive_max_batch_size", str(args.stage1_max_batch_size),
                "--lr", str(args.stage1_lr),
                "--save_every", str(args.save_every),
                "--fault_tolerant_save_every", str(args.fault_tolerant_save_every),
                "--visualize_every", str(args.visualize_every),
                "--val_every", str(args.val_every),
            ],
        ),
        Stage(
            name="stage2",
            script=str(root / "scripts" / "train_sparse_ray_ss_velocity.py"),
            config=str(root / "configs" / "sparse_ray_ss_velocity.yaml"),
            output_dir=stage2_out,
            steps=args.stage2_steps,
            max_batch_size=args.stage2_max_batch_size,
            resume_path=stage2_out / "ss_velocity_adapter_last.pt",
            best_path=stage2_out / "ss_velocity_adapter_best.pt",
            extra_args=common_data
            + early_stop_args
            + adaptive_args
            + vggt_args
            + trellis_args
            + [
                "--geoss_checkpoint", str(stage1_geoss_checkpoint),
                "--adaptive_max_batch_size", str(args.stage2_max_batch_size),
                "--lr", str(args.stage2_lr),
                "--raw_residual_weight", str(args.stage2_raw_residual_weight),
                "--save_every", str(args.save_every),
                "--fault_tolerant_save_every", str(args.fault_tolerant_save_every),
            ],
        ),
        Stage(
            name="slat",
            script=str(root / "scripts" / "train_geovis_slat.py"),
            config=str(root / "configs" / "geovis_slat.yaml"),
            output_dir=slat_out,
            steps=args.slat_steps,
            max_batch_size=args.slat_max_batch_size,
            resume_path=slat_out / "geovis_slat_adapter_last.pt",
            best_path=slat_out / "geovis_slat_adapter_best.pt",
            extra_args=common_data
            + early_stop_args
            + adaptive_args
            + trellis_args
            + slat_runtime_args
            + [
                "--active_tokens", str(args.active_tokens),
                "--adaptive_max_batch_size", str(args.slat_max_batch_size),
                "--lr", str(args.slat_lr),
                "--save_every", str(args.save_every),
                "--fault_tolerant_save_every", str(args.fault_tolerant_save_every),
                "--visualize_every", str(args.visualize_every),
            ],
        ),
        Stage(
            name="slat_joint",
            script=str(root / "scripts" / "train_geovis_slat_joint.py"),
            config=str(root / "configs" / "geovis_slat_joint.yaml"),
            output_dir=slat_joint_out,
            steps=args.slat_joint_steps,
            max_batch_size=args.slat_joint_max_batch_size,
            resume_path=slat_joint_out / "geovis_slat_adapter_last.pt",
            best_path=slat_joint_out / "geovis_slat_adapter_best.pt",
            extra_args=common_data
            + early_stop_args
            + adaptive_args
            + trellis_args
            + slat_runtime_args
            + [
                "--active_tokens", str(args.active_tokens),
                "--adaptive_max_batch_size", str(args.slat_joint_max_batch_size),
                "--lr", str(args.slat_joint_lr),
                "--save_every", str(args.save_every),
                "--fault_tolerant_save_every", str(args.fault_tolerant_save_every),
                "--visualize_every", str(args.visualize_every),
            ],
        ),
    ]


def _slice_stages(stages: list[Stage], start_at: str, stop_after: str) -> list[Stage]:
    names = [stage.name for stage in stages]
    start = names.index(start_at)
    stop = names.index(stop_after)
    if stop < start:
        raise ValueError("--stop_after must be the same as or later than --start_at.")
    return stages[start : stop + 1]


def _probe_batch(stage: Stage, args: argparse.Namespace, env: dict, nproc: int) -> int:
    if args.batch_probe_strategy == "binary":
        return _probe_batch_binary(stage, args, env, nproc)
    return _probe_batch_halve(stage, args, env, nproc)


def _probe_batch_halve(stage: Stage, args: argparse.Namespace, env: dict, nproc: int) -> int:
    batch_size = max(stage.max_batch_size, args.min_batch_size)
    last_error = ""
    setup_retries: dict[int, int] = {}
    while batch_size >= args.min_batch_size:
        probe_dir = stage.output_dir / f"_probe_bs{batch_size}"
        command = _torchrun_command(stage, nproc, batch_size, args.probe_steps, probe_dir, resume=False)
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        text = result.stdout or ""
        (probe_dir / "probe.log").parent.mkdir(parents=True, exist_ok=True)
        (probe_dir / "probe.log").write_text(text, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            if _probe_too_hot(text, args):
                print(f"{stage.name}: per-GPU batch_size={batch_size} exceeded probe VRAM headroom, retrying with {batch_size // 2}", flush=True)
                batch_size //= 2
                continue
            return batch_size
        last_error = text[-8000:]
        if not _looks_like_oom(text):
            if _looks_like_transient_setup(text) and setup_retries.get(batch_size, 0) < args.setup_retry_limit:
                setup_retries[batch_size] = setup_retries.get(batch_size, 0) + 1
                print(
                    f"{stage.name}: transient setup failure at per-GPU batch_size={batch_size}; "
                    f"retry {setup_retries[batch_size]}/{args.setup_retry_limit}",
                    flush=True,
                )
                time.sleep(max(0.0, args.restart_sleep_seconds))
                continue
            raise RuntimeError(f"{stage.name} batch probe failed for a non-OOM reason.\n{last_error}")
        print(f"{stage.name}: per-GPU batch_size={batch_size} OOM, retrying with {batch_size // 2}", flush=True)
        time.sleep(max(0.0, args.restart_sleep_seconds))
        batch_size //= 2
    raise RuntimeError(f"{stage.name} could not run even with per-GPU batch_size={args.min_batch_size}.\n{last_error}")


def _probe_batch_binary(stage: Stage, args: argparse.Namespace, env: dict, nproc: int) -> int:
    low = max(1, args.min_batch_size)
    high = max(stage.max_batch_size, low)
    best = 0
    last_error = ""
    tried = set()
    setup_retries: dict[int, int] = {}

    while low <= high:
        if high == stage.max_batch_size and high not in tried:
            batch_size = high
        else:
            batch_size = (low + high) // 2
        tried.add(batch_size)
        probe_dir = stage.output_dir / f"_probe_bs{batch_size}"
        command = _torchrun_command(stage, nproc, batch_size, args.probe_steps, probe_dir, resume=False)
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        text = result.stdout or ""
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "probe.log").write_text(text, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            if _probe_too_hot(text, args):
                last_error = text[-12000:]
                high = batch_size - 1
                print(f"{stage.name}: probe too close to VRAM limit at per-GPU batch_size={batch_size}; next search range [{low}, {high}]", flush=True)
                continue
            best = batch_size
            low = batch_size + 1
            print(f"{stage.name}: probe OK at per-GPU batch_size={batch_size}", flush=True)
        else:
            last_error = text[-12000:]
            if not _looks_like_oom(text):
                if _looks_like_transient_setup(text) and setup_retries.get(batch_size, 0) < args.setup_retry_limit:
                    setup_retries[batch_size] = setup_retries.get(batch_size, 0) + 1
                    print(
                        f"{stage.name}: transient setup failure at per-GPU batch_size={batch_size}; "
                        f"retry {setup_retries[batch_size]}/{args.setup_retry_limit}",
                        flush=True,
                    )
                    time.sleep(max(0.0, args.restart_sleep_seconds))
                    tried.discard(batch_size)
                    continue
                raise RuntimeError(f"{stage.name} batch probe failed for a non-OOM reason.\n{last_error}")
            high = batch_size - 1
            print(f"{stage.name}: probe OOM at per-GPU batch_size={batch_size}; next search range [{low}, {high}]", flush=True)
            time.sleep(max(0.0, args.restart_sleep_seconds))
    if best >= args.min_batch_size:
        return best
    raise RuntimeError(f"{stage.name} could not run even with per-GPU batch_size={args.min_batch_size}.\n{last_error}")


def _run_full_stage(stage: Stage, args: argparse.Namespace, env: dict, nproc: int, batch_size: int) -> None:
    current = batch_size
    oom_retries = 0
    setup_retries = 0
    launcher_log = stage.output_dir / "launcher_stage.log"
    while current >= args.min_batch_size:
        if _stage_is_complete(stage):
            print(f"{stage.name}: completed at checkpoint step={_checkpoint_step(_resolve_resume_path(stage))}", flush=True)
            return
        resume_path = _resolve_resume_path(stage)
        resume = resume_path is not None
        command = _torchrun_command(stage, nproc, current, stage.steps, stage.output_dir, resume=resume)
        print(
            f"{stage.name}: launching per-GPU batch_size={current}, global_batch_size={current * nproc}, "
            f"resume={resume}, checkpoint={resume_path}, checkpoint_step={_checkpoint_step(resume_path)}",
            flush=True,
        )
        returncode, tail = _run_command_logged(command, env, launcher_log)
        if returncode == 0:
            return
        if _stage_is_complete(stage):
            print(f"{stage.name}: subprocess failed after writing a completed checkpoint; treating stage as complete.", flush=True)
            return
        if _looks_like_oom(tail) and current > args.min_batch_size and oom_retries < args.oom_retry_limit:
            next_batch = max(args.min_batch_size, current // 2)
            print(
                f"{stage.name}: OOM/fatal CUDA detected, lowering per-GPU batch_size from {current} to {next_batch}; "
                f"will resume from checkpoint step={_checkpoint_step(_resolve_resume_path(stage))}",
                flush=True,
            )
            current = next_batch
            oom_retries += 1
            time.sleep(max(0.0, args.restart_sleep_seconds))
            continue
        if _looks_like_transient_setup(tail) and setup_retries < args.setup_retry_limit:
            setup_retries += 1
            print(
                f"{stage.name}: transient setup failure, relaunching same batch_size={current}; "
                f"retry {setup_retries}/{args.setup_retry_limit}",
                flush=True,
            )
            time.sleep(max(0.0, args.restart_sleep_seconds))
            continue
        raise RuntimeError(
            f"{stage.name} failed with per-GPU batch_size={current}, returncode={returncode}. "
            f"Check {launcher_log}. Tail:\n{tail[-4000:]}"
        )


def _torchrun_command(stage: Stage, nproc: int, batch_size: int, steps: int, output_dir: Path, *, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        stage.script,
        "--config",
        stage.config,
        "--output_dir",
        str(output_dir),
        "--steps",
        str(steps),
        "--batch_size",
        str(batch_size),
        "--steps_are_total",
        "true",
    ] + list(stage.extra_args)
    resume_path = _resolve_resume_path(stage)
    if resume and resume_path is not None:
        command += ["--resume", str(resume_path)]
    return command


def _configure_training_env(env: dict, args: argparse.Namespace) -> dict:
    tuned = dict(env)
    tuned.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    tuned.setdefault("NCCL_P2P_DISABLE", "0")
    tuned.setdefault("NCCL_SHM_DISABLE", "0")
    tuned.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    tuned.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    tuned.setdefault("TORCH_SHOW_CPP_STACKTRACES", "0")
    if args.nccl_socket_ifname:
        tuned["NCCL_SOCKET_IFNAME"] = args.nccl_socket_ifname
    if args.cuda_alloc_conf is not None:
        alloc_conf = str(args.cuda_alloc_conf).strip()
        if alloc_conf:
            tuned["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
        else:
            tuned.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if args.torch_hub_dir:
        hub = Path(args.torch_hub_dir)
        tuned["TORCH_HUB_DIR"] = str(hub)
        tuned.setdefault("TORCH_HOME", str(hub.parent if hub.name == "hub" else hub))
    return tuned


def _visible_gpu_count(env: dict) -> int:
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([item for item in visible.split(",") if item.strip()])
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _sanitize_torchrun_parent_env(env: dict) -> dict:
    clean = dict(env)
    for key in DISTRIBUTED_ENV_KEYS:
        clean.pop(key, None)
    return clean


def _looks_like_oom(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in OOM_PATTERNS)


def _looks_like_transient_setup(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in TRANSIENT_SETUP_PATTERNS)


def _probe_too_hot(text: str, args: argparse.Namespace) -> bool:
    ceiling = float(getattr(args, "probe_max_vram_util", 0.0) or 0.0)
    if ceiling <= 0.0:
        return False
    values = [float(match) for match in re.findall(r'"vram_utilization"\s*:\s*([0-9.]+)', text)]
    return bool(values and max(values) >= ceiling)


def _run_command_logged(command: list[str], env: dict, log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail_lines: list[str] = []
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n\n==== COMMAND ====\n")
        log.write(" ".join(command) + "\n")
        log.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            tail_lines.append(line)
            if len(tail_lines) > 300:
                tail_lines.pop(0)
        returncode = process.wait()
        log.write(f"\n==== RETURN CODE: {returncode} ====\n")
    return returncode, "".join(tail_lines)


def _stage_is_complete(stage: Stage) -> bool:
    resume_path = _resolve_resume_path(stage)
    if resume_path is None:
        return False
    step = _checkpoint_step(resume_path)
    if step is not None and step >= stage.steps:
        return True
    early_stop = _checkpoint_early_stop(resume_path)
    return bool(early_stop and early_stop.get("should_stop"))


def _resolve_resume_path(stage: Stage) -> Optional[Path]:
    candidates = [path for path in [stage.resume_path, stage.best_path] if path is not None and path.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: _checkpoint_step(path) or -1)


def _checkpoint_step(path: Optional[Path]) -> Optional[int]:
    state = _load_checkpoint_meta(path)
    if not state:
        return None
    try:
        return int(state.get("step", 0))
    except Exception:
        return None


def _checkpoint_early_stop(path: Optional[Path]) -> Optional[dict]:
    state = _load_checkpoint_meta(path)
    early_stop = state.get("early_stop") if state else None
    return early_stop if isinstance(early_stop, dict) else None


def _load_checkpoint_meta(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        import torch

        state = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    return state if isinstance(state, dict) else {}


if __name__ == "__main__":
    main()
