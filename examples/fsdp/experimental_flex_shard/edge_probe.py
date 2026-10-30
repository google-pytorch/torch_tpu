"""Edge checks for explicit layouts, optimizer skip semantics, and validation."""

import copy
import os
import tempfile
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flex_shard import (
  BucketSpec,
  GlobalLayout,
  MixedPrecisionPolicy,
  set_global_layout,
  set_partial_grad_group,
)
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from graph_flex_shard import GraphShard, data_parallel, graph_flex_shard
from probe import storage_compat


class Weight(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.arange(5.0).reshape(1, 5))

  def forward(self, x):
    return x @ self.weight.T


def placement_fn(named, mesh, placement):
  return {name: (placement,) for name, _ in named}


def spec(mesh, patterns, placement, op=dist.ReduceOp.SUM, factor=None):
  return BucketSpec(
    patterns,
    partial(placement_fn, placement=placement),
    mesh,
    gradient_reduce_op=op,
    gradient_divide_factor=factor,
  )


class Tied(nn.Module):
  def __init__(self):
    super().__init__()
    self.a = nn.Linear(5, 5)
    self.b = nn.Linear(5, 5)
    self.b.weight = self.a.weight
    self.unused = nn.Parameter(torch.ones(5))

  def forward(self, x):
    return self.b(torch.sin(self.a(x)))


def run(rank, path):
  os.environ["GLOO_SOCKET_IFNAME"] = "lo"
  dist.init_process_group(
    "gloo", rank=rank, world_size=2, init_method=f"file://{path}"
  )
  mesh = init_device_mesh("cpu", (2,))
  for dim, op, factor in [
    (0, dist.ReduceOp.SUM, None),
    (-1, dist.ReduceOp.AVG, 5),
  ]:
    model = Weight()
    reference = copy.deepcopy(model)
    storage_compat(model)
    graph_flex_shard(
      model, [spec(mesh, ["weight"], GraphShard(dim), op, factor)]
    )
    x = torch.arange(20.0).reshape(4, 5) / 20 + rank
    all_x = torch.cat([
      torch.arange(20.0).reshape(4, 5) / 20 + r for r in range(2)
    ])
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    compiled(x).square().mean().backward()
    reference(all_x).square().mean().backward()
    p = model._parameters["weight"]
    multiplier = 2 if op == dist.ReduceOp.SUM else 2 / factor
    expected = (
      p._placements[0].extract_local_shard(reference.weight.grad, rank, 2)
      * multiplier
    )
    torch.testing.assert_close(p.grad, expected)
    assert p._placements[0].dim == (dim % 2)

  torch.manual_seed(33)
  reference = Tied()
  model = copy.deepcopy(reference)
  storage_compat(model)
  data_parallel(model, mesh, mode="fully_shard")
  assert model.a._parameters["weight"] is model.b._parameters["weight"]
  for layer in (model.a, model.b):
    layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.01, weight_decay=0.1, foreach=True
  )
  ref_optimizer = torch.optim.AdamW(
    reference.parameters(), lr=0.01, weight_decay=0.1, foreach=True
  )
  compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
  unused_initial = model._parameters["unused"].detach().clone()
  x = torch.arange(20.0).reshape(4, 5) / 20 + rank
  all_x = torch.cat([
    torch.arange(20.0).reshape(4, 5) / 20 + r for r in range(2)
  ])
  for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    ref_optimizer.zero_grad(set_to_none=True)
    (compiled(x).square().mean() / 2).backward()
    reference(all_x).square().mean().backward()
    assert model._parameters["unused"].grad is None
    for name, p in model.named_parameters():
      ref = dict(reference.named_parameters())[name]
      if ref.grad is not None:
        torch.testing.assert_close(
          p.grad, p._placements[0].extract_local_shard(ref.grad, rank, 2)
        )
    optimizer.step()
    ref_optimizer.step()
  torch.testing.assert_close(model._parameters["unused"], unused_initial)
  # state_dict reads the sharded slots, not the gathering properties.
  state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
  assert state["a.weight"].shape == model.a._parameters["weight"].shape
  model.load_state_dict(state)
  assert model.a._parameters["weight"] is model.b._parameters["weight"]

  validations = 0
  for invalid in (
    "op",
    "multi",
    "scalar",
    "partial",
    "outer",
    "nested",
    "override",
    "sparse",
    "max_norm",
    "reserved",
  ):
    fresh = nn.Linear(5, 5)
    storage_compat(fresh)
    original = tuple(fresh.parameters())
    if invalid == "partial":
      set_partial_grad_group(fresh.weight, mesh.get_group())
    if invalid == "outer":
      set_global_layout(
        fresh.weight, GlobalLayout((5, 5), ((0, 0),), ((0, 0),), ((5, 5),))
      )
    if invalid == "nested":
      data_parallel(fresh, mesh, mode="fully_shard")
      fresh = nn.Sequential(fresh)
    chosen = [
      spec(mesh, ["weight"], GraphShard(0)),
      spec(mesh, ["bias"], GraphShard(0)),
    ]
    if invalid == "op":
      chosen[0] = spec(mesh, ["weight"], GraphShard(0), dist.ReduceOp.MAX)
    if invalid == "multi":
      chosen = [spec(mesh, ["*"], GraphShard(0))]
    if invalid == "scalar":
      fresh = nn.ParameterDict({"scalar": nn.Parameter(torch.ones(()))})
      storage_compat(fresh)
      original = tuple(fresh.parameters())
      chosen = [spec(mesh, ["scalar"], GraphShard(0))]
    if invalid in ("sparse", "max_norm"):
      fresh = nn.Embedding(
        5,
        5,
        sparse=invalid == "sparse",
        max_norm=1.0 if invalid == "max_norm" else None,
      )
      storage_compat(fresh)
      original = tuple(fresh.parameters())
      chosen = [spec(mesh, ["weight"], GraphShard(0))]
    if invalid == "reserved":
      fresh = nn.ParameterDict({
        "_graph_param_getters": nn.Parameter(torch.ones(5))
      })
      storage_compat(fresh)
      original = tuple(fresh.parameters())
      chosen = [spec(mesh, ["_graph_param_getters"], GraphShard(0))]
    if invalid == "override":
      policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        param_dtype_overrides={"weigth": torch.float32},
      )
      chosen = [
        BucketSpec(
          [name],
          partial(placement_fn, placement=GraphShard(0)),
          mesh,
          mp_policy=policy,
        )
        for name in ("weight", "bias")
      ]
    try:
      graph_flex_shard(fresh, chosen)
    except (ValueError, NotImplementedError):
      if invalid != "nested":
        assert all(a is b for a, b in zip(original, fresh.parameters()))
      validations += 1
    else:
      raise AssertionError(f"Accepted unsupported {invalid}")
  if rank == 0:
    print(
      f"PASS: empty rank, negative dim, SUM/AVG custom factor, tied checkpointed weights, "
      f"AdamW unused skip, state_dict roundtrip, {validations} rejection/atomicity checks",
      flush=True,
    )
  dist.destroy_process_group()


if __name__ == "__main__":
  with tempfile.TemporaryDirectory() as directory:
    mp.start_processes(
      run, args=(f"{directory}/store",), nprocs=2, start_method="fork"
    )
