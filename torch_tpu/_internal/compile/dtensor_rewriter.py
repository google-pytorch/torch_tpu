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


from typing import Any, List, Sequence, cast

import torch
import torch.distributed.tensor as dist_tensor
import torch.distributed.tensor._dtensor_spec as dtensor_spec
import torch.fx

# Map parameter names to their resolved DTensorSpec to bridge FWD and BWD
# compile passes
RESOLVED_PARAM_SPECS = {}


def to_sdy_mesh_format(device_mesh: dist_tensor.DeviceMesh) -> str:
  """Converts a PyTorch DeviceMesh to the Shardy sdy.mesh MLIR format."""
  names = device_mesh.mesh_dim_names or [
      f"mesh_dim_{i}" for i in range(len(device_mesh.shape))
  ]
  sizes = device_mesh.shape
  axes_str = ", ".join(f'"{name}"={size}' for name, size in zip(names, sizes))
  return f"#sdy.mesh<[{axes_str}]>"


def to_sdy_sharding_format(
    device_mesh: dist_tensor.DeviceMesh,
    placements: Sequence[Any],
    tensor_shape: List[int],
) -> str:
  """Converts PyTorch DTensor placements to Shardy sdy.sharding MLIR format."""
  rank = len(tensor_shape)
  dim_shardings = [[] for _ in range(rank)]
  names = device_mesh.mesh_dim_names or [
      f"mesh_dim_{i}" for i in range(len(device_mesh.shape))
  ]

  for i, p in enumerate(placements):
    if p.is_shard():
      dim_shardings[p.dim].append(names[i])

  dim_shardings_str = []
  for axes in dim_shardings:
    if not axes:
      dim_shardings_str.append("{}")
    else:
      axes_str = ", ".join(f'"{axis}"' for axis in axes)
      dim_shardings_str.append(f"{{{axes_str}}}")

  dim_shardings_joined = ", ".join(dim_shardings_str)
  return f"#sdy.sharding<@mesh, [{dim_shardings_joined}]>"


def resolve_eager_value(arg: Any) -> Any:
  """Resolves the eager value of an FX node or returns the arg if not a node."""
  if isinstance(arg, torch.fx.Node):
    return arg.meta.get("example_value", arg.meta.get("val", None))
  return arg


def rewrite_fx_graph(
    graph_module: torch.fx.GraphModule, inps: Sequence[Any]
) -> List[Any]:
  """Rewrites the FX graph to use custom Shardy sentinels and constraints.

  This function runs pre-AOT and replaces high-level DTensor operations
  (like from_local, redistribute, and to_local) with custom StableHLO
  sharding sentinels, enabling deferred layout planning in JAX/XLA Shardy.

  Args:
    graph_module: The FX graph module to rewrite in-place.
    inps: The example eager inputs to the graph, used to resolve shapes and
      sharding specifications.

  Returns:
    The list of unwrapped physical local tensors to pass to downstream
    AOTAutograd tracing.
  """
  g = graph_module.graph
  RESOLVED_PARAM_SPECS.clear()

  # 1. Validate input guardrails and insert Input Sentinels dynamically
  placeholders = [n for n in g.nodes if n.op == "placeholder"]
  new_inps = []
  global_node_map = {}
  active_specs = {}
  active_mesh = None
  nodes_to_remove = []

  for i, node in enumerate(placeholders):
    inp = inps[i]
    val = node.meta.get("example_value", node.meta.get("val", None))

    # A. Upfront Guardrail check: is it a DTensor eagerly? (Model 2)
    if isinstance(inp, dist_tensor.DTensor):
      spec = inp._spec

      # Stash resolved spec globally under the parameter JIT name to bridge to
      # BWD pass!
      RESOLVED_PARAM_SPECS[node.name] = spec

      local_fake = val._local_tensor if hasattr(val, "_local_tensor") else val
      global_fake = val

      # Update placeholders in-place
      node.meta["val"] = local_fake
      node.meta["example_value"] = local_fake

      if active_mesh is None:
        active_mesh = spec.mesh

      mesh_sdy_format = to_sdy_mesh_format(spec.mesh)
      placements_sdy_format = to_sdy_sharding_format(
          spec.mesh, spec.placements, spec.tensor_meta.shape
      )
      global_shape = list(spec.tensor_meta.shape)

      # Insert Input Sentinel after placeholder
      with g.inserting_after(node):
        sentinel_node = g.create_node(
            "call_function",
            torch.ops.torch_tpu.sdy_sharding.default,
            args=(
                node,
                mesh_sdy_format,
                placements_sdy_format,
                global_shape,
                True,  # is_input = True
            ),
        )
        sentinel_node.meta["val"] = global_fake
        sentinel_node.meta["example_value"] = global_fake

        # Map original placeholder uses to the sentinel node
        users = list(node.users.keys())
        for user in users:
          if user != sentinel_node:
            user.replace_input_with(node, sentinel_node)

        global_node_map[node] = sentinel_node
        active_specs[sentinel_node] = spec

      new_inp = inp._local_tensor
      if inp.requires_grad and not new_inp.requires_grad:
        try:
          new_inp.requires_grad_(True)
        except RuntimeError:
          new_inp = new_inp.clone().requires_grad_(True)
      new_inps.append(new_inp)

    # B. Upfront Guardrail check: is it wrapped inside the model via
    # from_local? (Model 1)
    else:
      if not isinstance(inp, torch.Tensor):
        new_inps.append(inp)
        continue

      from_local_node = None
      global_val = None
      for user in list(node.users.keys()):
        if user.op == "call_function" and (
            "from_local" in getattr(user.target, "__name__", "")
            or "from_local" in getattr(user.target, "__qualname__", "")
        ):
          from_local_node = user
          break

      # Resolved specs fallback (FWD stashed bridge!) to natively support BWD
      # compilation pass
      stashed_spec = None
      if RESOLVED_PARAM_SPECS and node.name in RESOLVED_PARAM_SPECS:
        stashed_spec = RESOLVED_PARAM_SPECS[node.name]
      elif RESOLVED_PARAM_SPECS and node.name.endswith("_grad"):
        base_name = node.name.removesuffix("_grad")
        # Try parameter format first (with trailing '_')
        if f"{base_name}_" in RESOLVED_PARAM_SPECS:
          stashed_spec = RESOLVED_PARAM_SPECS[f"{base_name}_"]
        # Fallback to exact match (for inputs like arg0_1)
        elif base_name in RESOLVED_PARAM_SPECS:
          stashed_spec = RESOLVED_PARAM_SPECS[base_name]

      if from_local_node is None and stashed_spec is None:
        # Strict Upfront Guardrail Violation: BS mixed-and-matched local
        # tensor detected!
        raise ValueError(
            f"SPMD JIT Compilation Error: Input placeholder '{node.name}' "
            "is a plain torch.Tensor eagerly but is never wrapped in a "
            "DTensor inside the model. Mixing plain Tensors and sharded "
            "DTensors is not supported."
        )

      # Case B.1: Resolve layout natively from the JIT from_local node (FWD
      # compile pass!)
      if from_local_node is not None:
        mesh_node = from_local_node.args[1]
        placements_node = from_local_node.args[2]

        mesh_val = resolve_eager_value(mesh_node)
        placements_val = resolve_eager_value(placements_node)

        global_val = from_local_node.meta.get(
            "example_value", from_local_node.meta.get("val", None)
        )
        if global_val is None:
          raise AssertionError(
              "Could not resolve shape for from_local node "
              f"'{from_local_node.name}'"
          )

        global_shape = list(global_val.shape)
        global_spec = global_val._spec if hasattr(global_val, "_spec") else None
        global_meta = (
            global_spec.tensor_meta if global_spec is not None else None
        )
        spec = dtensor_spec.DTensorSpec(
            mesh_val, placements_val, tensor_meta=global_meta
        )

        # Stash resolved spec globally under the parameter JIT name to bridge
        # to BWD compile pass!
        RESOLVED_PARAM_SPECS[node.name] = spec

      # Case B.2: Resolve layout from the stashed FWD specs bridge (BWD
      # compile pass!)
      else:
        spec = cast(dtensor_spec.DTensorSpec, stashed_spec)
        mesh_val = spec.mesh
        placements_val = spec.placements
        global_shape = (
            list(spec.tensor_meta.shape)
            if spec.tensor_meta is not None
            else list(val.shape)
        )

      if active_mesh is None:
        active_mesh = mesh_val

      mesh_sdy_format = to_sdy_mesh_format(mesh_val)
      placements_sdy_format = to_sdy_sharding_format(
          mesh_val, placements_val, global_shape
      )

      local_fake = val
      node.meta["val"] = local_fake
      node.meta["example_value"] = local_fake

      # Insert Input Sentinel after placeholder
      with g.inserting_after(node):
        sentinel_node = g.create_node(
            "call_function",
            torch.ops.torch_tpu.sdy_sharding.default,
            args=(
                node,
                mesh_sdy_format,
                placements_sdy_format,
                global_shape,
                True,  # is_input = True
            ),
        )

        if from_local_node is not None:
          sentinel_node.meta["val"] = global_val
          sentinel_node.meta["example_value"] = global_val

          # Rewire all from_local users directly to the sentinel node!
          from_local_node.replace_all_uses_with(sentinel_node)
          nodes_to_remove.append(from_local_node)
          global_node_map[from_local_node] = sentinel_node
        else:
          # For BWD gradient inputs, shape is scaled to global shape
          fake_mode = local_fake.fake_mode
          global_fake = fake_mode.from_tensor(
              torch.empty(
                  global_shape,
                  dtype=local_fake.dtype,
                  device="cpu",
                  requires_grad=local_fake.requires_grad,
              )
          )
          sentinel_node.meta["val"] = global_fake
          sentinel_node.meta["example_value"] = global_fake

        # Rewire placeholder uses to sentinel node
        if from_local_node is None:
          users = list(node.users.keys())
          for user in users:
            if user != sentinel_node:
              user.replace_input_with(node, sentinel_node)

        global_node_map[node] = sentinel_node
        active_specs[sentinel_node] = spec

      new_inps.append(inp)

  # Erase the unwrapped from_local nodes natively
  for node in nodes_to_remove:
    g.erase_node(node)

  if active_mesh is None:

    return list(inps)

  mesh_sdy_format = to_sdy_mesh_format(active_mesh)

  # 2. Replace high-level redistribute calls with Sharding Constraints
  redistribute_nodes_to_remove = []
  for node in list(g.nodes):
    if node.op == "call_method" and node.target == "redistribute":
      input_node = node.args[0]
      global_input = global_node_map.get(input_node, input_node)

      # redistribute args: (self, device_mesh, placements)
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

      placements_sdy_format = to_sdy_sharding_format(
          active_mesh, placements, global_shape
      )

      with g.inserting_before(node):
        constraint_node = g.create_node(
            "call_function",
            torch.ops.torch_tpu.sdy_sharding_constraint.default,
            args=(
                global_input,
                mesh_sdy_format,
                placements_sdy_format,
            ),
        )
        constraint_node.meta["val"] = global_val.clone()
        constraint_node.meta["example_value"] = global_val.clone()

      node.replace_all_uses_with(constraint_node)
      redistribute_nodes_to_remove.append(node)
      global_node_map[node] = constraint_node

  for node in redistribute_nodes_to_remove:
    g.erase_node(node)

  # 3. Erase redundant JIT to_local calls inside the graph
  to_local_nodes_to_remove = []
  for node in list(g.nodes):
    if node.op == "call_method" and node.target == "to_local":
      input_node = node.args[0]
      global_input = global_node_map.get(input_node, input_node)

      # Rewire downstream users of to_local directly to its global input node!
      node.replace_all_uses_with(global_input)
      to_local_nodes_to_remove.append(node)
      global_node_map[node] = global_input

  for node in to_local_nodes_to_remove:
    g.erase_node(node)

  # 3. Insert Output Sentinels at Graph exits to slice back to local shapes
  output_node = [n for n in g.nodes if n.op == "output"][0]

  new_return_args = []
  for arg in output_node.args[0]:
    if arg is None:
      new_return_args.append(None)
      continue

    val = arg.meta.get("example_value", arg.meta.get("val", None))
    if val is not None and hasattr(val, "_spec"):
      spec = val._spec
      global_shape = list(spec.tensor_meta.shape)
      placement = spec.placements[0]

      # Compute target local shape based on sharding spec
      target_local_shape = list(global_shape)
      if isinstance(placement, torch.distributed.tensor.Shard):
        target_local_shape[placement.dim] //= spec.mesh.size()

      # If the JIT node's shape is already local-sized, bypass sentinel
      # slicing!
      if list(val.shape) == target_local_shape:
        new_return_args.append(global_node_map.get(arg, arg))
        continue

      local_shape = target_local_shape

      placements_sdy_format = to_sdy_sharding_format(
          spec.mesh, spec.placements, spec.tensor_meta.shape
      )

      with g.inserting_before(output_node):
        global_arg = global_node_map.get(arg, arg)
        slice_node = g.create_node(
            "call_function",
            torch.ops.torch_tpu.sdy_sharding.default,
            args=(
                global_arg,
                mesh_sdy_format,
                placements_sdy_format,
                local_shape,
                False,  # is_input = False
            ),
        )
        global_val = global_arg.meta.get(
            "example_value", global_arg.meta.get("val", None)
        )
        fake_mode = global_val._local_tensor.fake_mode
        local_fake = fake_mode.from_tensor(
            torch.empty(
                local_shape,
                dtype=global_val.dtype,
                device="cpu",
                requires_grad=global_val.requires_grad,
            )
        )
        slice_node.meta["val"] = local_fake
        slice_node.meta["example_value"] = local_fake

      new_return_args.append(slice_node)

    else:
      new_return_args.append(arg)

  output_node.args = (new_return_args,)

  # Eliminate dead nodes natively
  g.eliminate_dead_code()

  return new_inps
