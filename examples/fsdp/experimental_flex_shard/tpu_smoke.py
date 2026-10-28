"""Unvalidated TPU smoke test; use launch_tpu.py on a configured TPU host.

Requires mutually compatible TorchTPU, PyTorch, and public FlexShard builds.
FlexShard requires PyTorch >= 2.16.0.dev0; this test has no compatibility shim.
Passing this smoke test alone does not validate distributed numerical accuracy,
XLA collective lowering, checkpoint peak memory, or performance.
"""

from functools import partial

import torch
import torch.distributed as dist
from flex_shard import MixedPrecisionPolicy
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from graph_flex_shard import data_parallel


def main():
  dist.init_process_group("tpu_dist")
  world_size = dist.get_world_size()
  mesh = init_device_mesh("tpu", (world_size,), mesh_dim_names=("fsdp",))
  torch.manual_seed(0)
  with torch.device("tpu"):
    model = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 64))
  # SUM, like SimpleFSDP: keep the loss/world_size scaling exactly once.
  data_parallel(
    model,
    mesh,
    mode="fully_shard",
    mp_policy=MixedPrecisionPolicy(
      param_dtype=torch.bfloat16,
      reduce_dtype=torch.float32,
    ),
  )
  # Parameter-access gathers are inside forward, matching SimpleFSDP's pattern.
  for layer in (model[0], model[2]):
    layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
  x = torch.randn(8, 128, device="tpu", dtype=torch.bfloat16)
  y = torch.randn(8, 64, device="tpu", dtype=torch.bfloat16)

  def forward_backward(x, y):
    with torch.fx.traceback.annotate({"spmd_safe": True}):
      loss = nn.functional.mse_loss(model(x).float(), y.float())
      (loss / world_size).backward()
      return loss.detach()

  step = torch.compile(forward_backward, backend="tpu", fullgraph=True)
  for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    step(x, y)
    optimizer.step()
  torch.accelerator.synchronize()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
