"""CPU/Gloo numerical and graph-capture probe; no TPU runtime is mocked."""

from __future__ import annotations

import copy
import json
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from flex_shard import MixedPrecisionPolicy
from functorch.compile import make_boxed_func
from torch import nn
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import aot_export_module
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
  checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh

from graph_flex_shard import data_parallel


_used_grad_dtype_compat = False


def storage_compat(model):
  # Test-only bridge for older wheels. No graph operation, collective,
  # autograd behavior, or runtime is mocked. The TPU probe has no bridge.
  global _used_grad_dtype_compat
  for param in model.parameters():
    if not hasattr(param, "_has_grad_dtype_override"):
      param._has_grad_dtype_override = False
      _used_grad_dtype_compat = True


class Net(nn.Module):
  def __init__(self, dtype=torch.float32):
    super().__init__()
    self.a = nn.Linear(5, 7, dtype=dtype)
    self.b = nn.Linear(7, 3, dtype=dtype)
    self.scalar = nn.Parameter(torch.ones((), dtype=dtype))
    self.unused = nn.Parameter(torch.randn(9, dtype=dtype))
    self.frozen = nn.Parameter(torch.randn(9, dtype=dtype), requires_grad=False)

  def forward(self, x):
    return self.b(torch.relu(self.a(x))) * self.scalar + self.frozen[:3]


class Loss(nn.Module):
  def __init__(self, model, scale):
    super().__init__()
    self.model = model
    self.scale = scale

  def forward(self, x, y):
    return (
      nn.functional.mse_loss(self.model(x).float(), y.float()) * self.scale,
    )


def run_case(
  mesh, rank, mode="fully_shard", dim=0, checkpoint=False, dtype=torch.float32
):
  torch._dynamo.reset()
  torch.manual_seed(12)
  reference = Net(dtype)
  model = copy.deepcopy(reference)
  storage_compat(model)
  mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
  data_parallel(model, mesh, mode=mode, shard_dim=dim, mp_policy=mp)
  if checkpoint == "forward":
    for layer in (model.a, model.b):
      layer.forward = partial(
        torch.utils.checkpoint.checkpoint, layer.forward, use_reentrant=False
      )
  elif checkpoint:
    model.a = checkpoint_wrapper(model.a)
    model.b = checkpoint_wrapper(model.b)
  world_size = mesh.size()
  inputs = torch.randn(world_size, 4, 5, dtype=dtype)
  targets = torch.randn(world_size, 4, 3, dtype=dtype)
  x, y = inputs[rank].contiguous(), targets[rank].contiguous()
  fw_graphs, bw_graphs = [], []

  def compiler(out, gm, args):
    out.append(gm)
    return make_boxed_func(gm.forward)

  backend = aot_autograd(
    fw_compiler=lambda gm, args: compiler(fw_graphs, gm, args),
    bw_compiler=lambda gm, args: compiler(bw_graphs, gm, args),
  )
  loss_model = Loss(model, 1 / world_size)
  compiled = torch.compile(loss_model, backend=backend, fullgraph=True)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=True)
  ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.02, foreach=True)
  max_error = 0.0
  for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    ref_optimizer.zero_grad(set_to_none=True)
    loss = compiled(x, y)[0]
    loss.backward()
    ref_loss = nn.functional.mse_loss(
      reference(inputs.reshape(-1, 5)).float(), targets.reshape(-1, 3).float()
    )
    ref_loss.backward()
    for name, param in model.named_parameters():
      clean = name.replace("_checkpoint_wrapped_module.", "")
      ref = dict(reference.named_parameters())[clean]
      if ref.grad is None:
        assert param.grad is None
        continue
      placement = param._placements[0]
      expected = placement.extract_local_shard(ref.grad, rank, world_size)
      torch.testing.assert_close(
        param.grad,
        expected,
        rtol=0.04 if dtype == torch.bfloat16 else 1e-5,
        atol=0.004 if dtype == torch.bfloat16 else 1e-6,
      )
      if param.numel():
        max_error = max(max_error, float((param.grad - expected).abs().max()))
    optimizer.step()
    ref_optimizer.step()
    for name, param in model.named_parameters():
      clean = name.replace("_checkpoint_wrapped_module.", "")
      ref = dict(reference.named_parameters())[clean]
      expected = param._placements[0].extract_local_shard(ref, rank, world_size)
      torch.testing.assert_close(
        param,
        expected,
        rtol=0.04 if dtype == torch.bfloat16 else 1e-5,
        atol=0.004 if dtype == torch.bfloat16 else 1e-6,
      )
  assert len(fw_graphs) == 1, len(fw_graphs)
  assert len(bw_graphs) == 1, len(bw_graphs)
  fw_ops = [
    str(node.target)
    for node in fw_graphs[0].graph.nodes
    if node.op == "call_function"
  ]
  bw_ops = [
    str(node.target)
    for node in bw_graphs[0].graph.nodes
    if node.op == "call_function"
  ]
  assert (
    any("all_gather_into_tensor" in op for op in fw_ops) or mode == "replicate"
  )
  assert (
    any("reduce_scatter_tensor" in op for op in bw_ops) or mode == "replicate"
  )
  assert not any("fsdp." in op for op in fw_ops + bw_ops)
  dim_label = sorted(set(dim.values())) if isinstance(dim, dict) else dim
  case = f"{mode}_dim{dim_label}_checkpoint{checkpoint}_{dtype}"
  output = Path("results")
  output.mkdir(exist_ok=True)
  if rank == 0:
    (output / f"{case}.forward.txt").write_text(fw_graphs[0].code)
    (output / f"{case}.backward.txt").write_text(bw_graphs[0].code)
  # Capture a true JOINT forward+backward graph in addition to separate AOT graphs.
  # The stock exporter rejects unused trainable parameters (PyTorch #101192).
  # Numerical/AOT tests above still cover that parameter; freeze it only here.
  model._parameters["unused"].requires_grad_(False)
  try:
    exported, _signature = aot_export_module(
      loss_model, (x, y), trace_joint=True, output_loss_index=0
    )
  finally:
    model._parameters["unused"].requires_grad_(True)
  joint_ops = [
    str(node.target)
    for node in exported.graph.nodes
    if node.op == "call_function"
  ]
  assert (
    any("all_gather_into_tensor" in op for op in joint_ops)
    or mode == "replicate"
  )
  assert (
    any("reduce_scatter_tensor" in op for op in joint_ops)
    or mode == "replicate"
  )
  # Execute the exported joint graph, supplying every parameter explicitly.
  assert not list(exported.named_buffers()), (
    "Unexpected captured tensor constants"
  )
  params = dict(loss_model.named_parameters(remove_duplicate=False))
  buffers = dict(loss_model.named_buffers())
  arguments = [params[name] for name in _signature.parameters]
  arguments += [buffers[name] for name in _signature.buffers]
  with torch.no_grad():
    actual_outputs = exported(*arguments, x, y)
  reference.zero_grad(set_to_none=True)
  nn.functional.mse_loss(
    reference(inputs.reshape(-1, 5)).float(), targets.reshape(-1, 3).float()
  ).backward()
  out_nodes = next(
    node for node in exported.graph.nodes if node.op == "output"
  ).args[0]
  gradient_names = _signature.backward_signature.gradients_to_parameters
  for node, value in zip(out_nodes, actual_outputs):
    if node.name not in gradient_names:
      continue
    parameter_name = gradient_names[node.name]
    clean = parameter_name.removeprefix("model.").replace(
      "_checkpoint_wrapped_module.", ""
    )
    expected = (
      params[parameter_name]
      ._placements[0]
      .extract_local_shard(
        dict(reference.named_parameters())[clean].grad,
        rank,
        world_size,
      )
    )
    torch.testing.assert_close(
      value,
      expected,
      rtol=0.04 if dtype == torch.bfloat16 else 1e-5,
      atol=0.004 if dtype == torch.bfloat16 else 1e-6,
    )
  if checkpoint:
    assert any("all_gather_into_tensor" in op for op in bw_ops), (
      "Gather not rematerialized"
    )
  if rank == 0:
    (output / f"{case}.joint.txt").write_text(exported.code)
  result = {
    "case": case,
    "steps": 3,
    "fw_graphs": len(fw_graphs),
    "bw_graphs": len(bw_graphs),
    "max_grad_error": max_error,
    "joint_nodes": len(joint_ops),
    "joint_executed_and_gradients_checked": True,
    "backward_all_gather": sum("all_gather_into_tensor" in op for op in bw_ops),
    "joint_all_gather": sum("all_gather_into_tensor" in op for op in joint_ops),
    "joint_reduce_scatter": sum(
      "reduce_scatter_tensor" in op for op in joint_ops
    ),
    "joint_all_reduce": sum("all_reduce" in op for op in joint_ops),
  }
  if rank == 0:
    print(json.dumps(result), flush=True)
  return result


def main():
  if not dist.is_initialized():
    dist.init_process_group("gloo")
  rank = dist.get_rank()
  mesh = init_device_mesh("cpu", (dist.get_world_size(),))
  results = []
  for mode, dim, checkpoint, dtype in [
    ("fully_shard", 0, False, torch.float32),
    ("fully_shard", 1, False, torch.float32),
    ("fully_shard", 0, True, torch.float32),
    ("fully_shard", 0, "forward", torch.float32),
    ("replicate", 0, False, torch.float32),
    ("fully_shard", 0, False, torch.bfloat16),
  ]:
    # Vectors always shard along dim 0, matrices may shard along dim 1.
    dimensions = {
      name: dim if param.ndim > 1 else 0
      for name, param in Net(dtype).named_parameters()
    }
    results.append(run_case(mesh, rank, mode, dimensions, checkpoint, dtype))
  if rank == 0:
    Path("results/summary.json").write_text(
      json.dumps(
        {
          "torch": torch.__version__,
          "world_size": mesh.size(),
          "test_only_grad_dtype_compat": _used_grad_dtype_compat,
          "results": results,
        },
        indent=2,
      )
    )
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
