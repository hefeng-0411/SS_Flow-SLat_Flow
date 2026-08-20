from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from geoss.utils.elastic_engine import train_step_with_oom_retry


class DummyBatchController:
    def __init__(self):
        self.batch_size = 2
        self.ooms = 0

    def update_after_success(self, device):
        return DummyAdjustment(False, self.batch_size, self.batch_size, "success")

    def update_after_oom(self, device):
        self.ooms += 1
        return DummyAdjustment(False, self.batch_size, self.batch_size, "oom_replay")


class DummyAdjustment:
    def __init__(self, changed, old_batch_size, new_batch_size, reason):
        self.changed = changed
        self.old_batch_size = old_batch_size
        self.new_batch_size = new_batch_size
        self.reason = reason

    def as_dict(self):
        return {
            "changed": self.changed,
            "old_batch_size": self.old_batch_size,
            "new_batch_size": self.new_batch_size,
            "reason": self.reason,
        }


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)
    model = torch.nn.Linear(8, 4).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    controller = DummyBatchController()
    attempts = {"count": 0}

    def step_fn():
        attempts["count"] += 1
        x = torch.randn(2, 8, device=device)
        y = model(x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        y.backward()
        opt.step()
        if rank == 0 and attempts["count"] == 1:
            raise torch.cuda.OutOfMemoryError("synthetic post-mutation OOM")
        return float(y.detach().cpu())

    result = train_step_with_oom_retry(
        step_fn,
        model=model,
        optimizer=opt,
        device=device,
        batch_controller=controller,
        max_retries=2,
    )

    flat = torch.cat([p.detach().flatten().float() for p in model.parameters()]).to(device)
    gathered = [torch.empty_like(flat) for _ in range(world_size)]
    dist.all_gather(gathered, flat)
    max_diff = max(float((gathered[0] - item).abs().max().detach().cpu()) for item in gathered)
    if rank == 0:
        print({"status": "ok", "retries": result.retries, "max_param_diff": max_diff})
    assert result.retries == 1
    assert max_diff == 0.0
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
