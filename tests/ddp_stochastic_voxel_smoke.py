from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.models.ss_flow_adapter import SSFlowAdapter


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = DDP(SSFlowAdapter(latent_dim=8, condition_dim=9, hidden_dim=32, num_heads=4, num_blocks=1).to(device), device_ids=[local_rank])
    torch.nn.init.normal_(model.module.residual_head.weight, std=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(3):
        output = model(
            torch.randn(1, 16, 8, device=device),
            torch.randn(1, 16, 9, device=device),
            torch.tensor([500.0], device=device),
            torch.ones(1, 16, 1, device=device, dtype=torch.bool),
            torch.ones(1, 16, 1, device=device),
            v_base=torch.randn(1, 16, 8, device=device),
        )
        loss = output.v_final.square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
