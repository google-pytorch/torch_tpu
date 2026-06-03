# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SPMD graph rewriter.

It transforms high-level Dynamo graph module to StableHLO/Shardy sentinels.
"""

# pylint: disable=protected-access


from typing import Any, Dict, List, Sequence, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.tensor as dist_tensor
import torch.distributed.tensor._dtensor_spec as dtensor_spec
from torch.distributed.tensor._ops._pointwise_ops import pointwise_ops
import torch.fx
from torch_tpu._internal.distributed import tpu_distributed

_DTENSOR_POINTWISE_OPS = set(pointwise_ops)

_RANK_TO_DEVICE_MAP = None


def get_rank_to_device_map() -> List[int]:
  global _RANK_TO_DEVICE_MAP
  if _RANK_TO_DEVICE_MAP is not None:
    return _RANK_TO_DEVICE_MAP

  if not dist.is_initialized():
    return []

  world_size = dist.get_world_size()
  my_device_id = tpu_distributed.global_device_id()

  device_ids = [0] * world_size
  dist.all_gather_object(device_ids, my_device_id)
  _RANK_TO_DEVICE_MAP = device_ids
  return _RANK_TO_DEVICE_MAP


def to_generic_mesh_spec(device_mesh: dist_tensor.DeviceMesh) -> str:
  names = device_mesh.mesh_dim_names or [
      f"mesh_dim_{i}" for i in range(device_mesh.ndim)
  ]
  sizes = device_mesh.shape
  axes_str = ",".join(f"{name}:{size}" for name, size in zip(names, sizes))

  flat_ranks = device_mesh.mesh.flatten().tolist()
  rank_to_device = get_rank_to_device_map()
  if rank_to_device:
    flat_ranks = [rank_to_device[r] for r in flat_ranks]
  device_ids_str = ",".join(str(r) for r in flat_ranks)
  return f"{axes_str}[{device_ids_str}]"


def to_generic_sharding_spec(
    device_mesh: dist_tensor.DeviceMesh,
    placements: Sequence[Any],
    tensor_shape: List[int],
) -> str:
  rank = len(tensor_shape)
  dim_shardings = [[] for _ in range(rank)]
  partial_axes = []
  names = device_mesh.mesh_dim_names or [
      f"mesh_dim_{i}" for i in range(device_mesh.ndim)
  ]

  for i, p in enumerate(placements):
    if p.is_shard():
      dim_shardings[p.dim].append(names[i])
    elif p.is_partial():
      partial_axes.append(names[i])

  dim_str = ";".join(",".join(axes) for axes in dim_shardings)
  if partial_axes:
    return f"{dim_str}|{','.join(partial_axes)}"
  return dim_str


def resolve_eager_value(arg: Any) -> Any:
  """Resolves the eager value of an FX node or returns the arg if not a node."""
  if isinstance(arg, torch.fx.Node):
    return arg.meta.get("example_value", arg.meta.get("val", None))
  return arg


def get_node_sharding_spec(node: torch.fx.Node, active_mesh: Any) -> str | None:
  val = node.meta.get("example_value", node.meta.get("val", None))
  if val is None or not hasattr(val, "_spec") or val._spec is None:
    return None
  # Exclude 0-dimensional scalar tensors (which cannot be sharded)
  if len(val.shape) == 0:
    return None
  placements = val._spec.placements
  global_shape = list(val.shape)
  return to_generic_sharding_spec(active_mesh, placements, global_shape)


def get_active_fake_mode() -> Any | None:
  from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
  from torch._subclasses.fake_tensor import FakeTensorMode

  for mode in reversed(_get_current_dispatch_mode_stack()):
    if isinstance(mode, FakeTensorMode):
      return mode
  return None


def _find_fake_mode(graph_module: torch.fx.GraphModule) -> Any | None:
  for node in graph_module.graph.nodes:
    val = node.meta.get("example_value", node.meta.get("val", None))
    if val is not None:
      if hasattr(val, "_local_tensor") and val._local_tensor is not None:
        mode = getattr(val._local_tensor, "fake_mode", None)
        if mode is not None:
          return mode
      mode = getattr(val, "fake_mode", None)
      if mode is not None:
        return mode
  return None


def to_plain_global_fake(val: Any, fake_mode: Any = None) -> Any:
  if val is None:
    return None
  if not (hasattr(val, "_spec") and val._spec is not None):
    return val

  local_tensor = getattr(val, "_local_tensor", None)
  local_fake_mode = (
      getattr(local_tensor, "fake_mode", None)
      if local_tensor is not None
      else None
  )
  inp_fake_mode = getattr(val, "fake_mode", None)

  if fake_mode is None:
    fake_mode = get_active_fake_mode()
  if fake_mode is None:
    fake_mode = local_fake_mode
  if fake_mode is None:
    fake_mode = inp_fake_mode

  if fake_mode is None:
    return val

  global_shape = list(val.shape)
  meta_tensor = torch.empty(
      global_shape,
      dtype=val.dtype,
      device="meta",
      requires_grad=val.requires_grad,
  )
  plain_fake = fake_mode.fake_tensor_converter.from_meta_and_device(
      fake_mode, meta_tensor, val.device
  )
  return plain_fake


def rewrite_dynamo_graph(
    graph_module: torch.fx.GraphModule,
    inps: Sequence[Any],
    resolved_param_specs: Dict[str, Any] | None = None,
    resolved_output_specs: List[Any] | None = None,
) -> List[Any]:
  """Rewrites the FX graph to use custom Shardy sentinels and constraints."""
  g = graph_module.graph
  fake_mode = _find_fake_mode(graph_module)
  if fake_mode is None:
    raise AssertionError("Could not find active FakeTensorMode in the graph.")

  if resolved_param_specs is None:
    resolved_param_specs = {}
  else:
    resolved_param_specs.clear()

  # 1. Validation pass: Ensure no calls to to_local or from_local.

  for node in g.nodes:
    if node.op == "call_method" and node.target in ("to_local", "from_local"):
      raise AssertionError(
          f"Validation failed: FX graph contains '{node.target}' call: {node}."
          " Pure DTensor compiler does not support mixing or existing"
          " to_local/from_local."
      )
    if node.op == "call_function" and (
        getattr(node.target, "__name__", None) in ("from_local", "to_local")
        or "from_local" in str(node.target)
        or "to_local" in str(node.target)
    ):
      raise AssertionError(
          "Validation failed: FX graph contains from_local/to_local function"
          f" call: {node}."
      )

  # 2. Insert Input Sentinels after placeholders
  placeholders = [n for n in g.nodes if n.op == "placeholder"]
  new_inps = []
  global_node_map = {}
  active_mesh = None

  for i, node in enumerate(placeholders):
    inp = inps[i]
    if isinstance(inp, dist_tensor.DTensor):
      spec = inp._spec
      val = inp
      resolved_param_specs[node.name] = spec
      if active_mesh is None:
        active_mesh = spec.mesh

      global_shape = (
          list(val.shape)
          if val is not None and hasattr(val, "shape")
          else list(spec.tensor_meta.shape)
      )

      with g.inserting_after(node):
        sentinel_node = g.create_node(
            "call_function",
            torch.ops.tpu.spmd_metadata.default,
            args=(
                node,
                to_generic_mesh_spec(spec.mesh),
                "",
                to_generic_sharding_spec(
                    spec.mesh, spec.placements, global_shape
                ),
                global_shape,
                "input",
            ),
        )
        node.replace_all_uses_with(sentinel_node)
        sentinel_node.replace_input_with(sentinel_node, node)
        sentinel_node.meta = node.meta.copy()
        val = node.meta.get("example_value", node.meta.get("val", None))
        if val is not None:
          plain_val = to_plain_global_fake(val, fake_mode)
          sentinel_node.meta["val"] = plain_val
          sentinel_node.meta["example_value"] = plain_val
      global_node_map[node] = sentinel_node
      new_inps.append(to_plain_global_fake(inp, fake_mode))
    else:
      new_inps.append(inp)

  if active_mesh is None:
    return new_inps

  mesh_spec = to_generic_mesh_spec(active_mesh)

  # 3. Replace high-level redistribute calls with Sharding Constraints
  redistribute_nodes_to_remove = []
  for node in list(g.nodes):
    if node.op == "call_method" and node.target == "redistribute":
      input_node = node.args[0]
      global_input = global_node_map.get(input_node, input_node)
      placements = node.args[2]

      global_val = global_input.meta.get(
          "example_value", global_input.meta.get("val", None)
      )
      if global_val is None:
        raise AssertionError(
            "Could not resolve shape for global input node "
            f"'{global_input.name}'"
        )
      global_shape = list(global_val.shape)

      sharding_spec_out = to_generic_sharding_spec(
          active_mesh, placements, global_shape
      )

      sharding_spec_in = ""
      if (
          global_input.op == "call_function"
          and global_input.target == torch.ops.tpu.spmd_metadata.default
      ):
        sharding_spec_in = global_input.args[3]
      elif hasattr(global_val, "_spec") and global_val._spec is not None:
        sharding_spec_in = to_generic_sharding_spec(
            active_mesh, global_val._spec.placements, global_shape
        )

      with g.inserting_before(node):
        constraint_node = g.create_node(
            "call_function",
            torch.ops.tpu.spmd_metadata.default,
            args=(
                global_input,
                mesh_spec,
                sharding_spec_in,
                sharding_spec_out,
                global_shape,
                "intermediate",
            ),
        )
        cloned_val = global_val.clone()
        global_meta = (
            global_val._spec.tensor_meta
            if hasattr(global_val, "_spec") and global_val._spec is not None
            else None
        )
        cloned_val._spec = dtensor_spec.DTensorSpec(
            active_mesh, placements, tensor_meta=global_meta
        )
        constraint_node.meta["val"] = cloned_val
        constraint_node.meta["example_value"] = cloned_val

      node.replace_all_uses_with(constraint_node)
      redistribute_nodes_to_remove.append(node)
      global_node_map[node] = constraint_node

  for node in redistribute_nodes_to_remove:
    g.erase_node(node)

  # 4. Insert Output Sentinels at Graph exits
  output_node = [n for n in g.nodes if n.op == "output"][0]

  new_return_args = []
  for arg in output_node.args[0]:
    if arg is None:
      new_return_args.append(None)
      if resolved_output_specs is not None:
        resolved_output_specs.append(None)
      continue

    val = arg.meta.get("example_value", arg.meta.get("val", None))
    if val is not None and hasattr(val, "_spec") and val._spec is not None:
      spec = val._spec
      if resolved_output_specs is not None:
        resolved_output_specs.append((spec.mesh, spec.placements))

      global_shape = (
          list(val.shape)
          if hasattr(val, "shape")
          else list(spec.tensor_meta.shape)
      )

      for p in spec.placements:
        if p.is_partial():
          raise ValueError(
              f"Output node '{arg.name}' has Partial placement. "
              "Returning Partial (unreduced) tensors from compiled graphs is not supported yet. "
              "Please explicitly reduce the output using redistribute() before returning."
          )

      mesh_spec = to_generic_mesh_spec(spec.mesh)
      sharding_spec_in = to_generic_sharding_spec(
          spec.mesh, spec.placements, global_shape
      )

      placement = spec.placements[0]

      target_local_shape = list(global_shape)
      if isinstance(placement, torch.distributed.tensor.Shard):
        target_local_shape[placement.dim] //= spec.mesh.size()

      local_shape = target_local_shape

      with g.inserting_before(output_node):
        global_arg = global_node_map.get(arg, arg)
        slice_node = g.create_node(
            "call_function",
            torch.ops.tpu.spmd_metadata.default,
            args=(
                global_arg,
                mesh_spec,
                sharding_spec_in,
                "",
                global_shape,
                "output",
            ),
        )
        meta_tensor = torch.empty(
            local_shape,
            dtype=val.dtype,
            device="meta",
            requires_grad=val.requires_grad,
        )
        local_fake = fake_mode.fake_tensor_converter.from_meta_and_device(
            fake_mode, meta_tensor, val.device
        )
        slice_node.meta["val"] = local_fake
        slice_node.meta["example_value"] = local_fake

      new_return_args.append(slice_node)
    else:
      new_return_args.append(arg)
      if resolved_output_specs is not None:
        resolved_output_specs.append(None)

  output_node.args = (new_return_args,)

  # (Section 5 constraint insertion disabled to align with JAX propagation)

  for node in g.nodes:
    if node.op == "output":
      continue
    if (
        node.op == "call_function"
        and node.target == torch.ops.tpu.spmd_metadata.default
        and len(node.args) > 5
        and node.args[5] == "output"
    ):
      continue

    for key in ("val", "example_value"):
      if key in node.meta:
        node.meta[key] = to_plain_global_fake(node.meta[key], fake_mode)

  graph_module.recompile()
  return new_inps
