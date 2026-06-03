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
"""DTensor sharding strategy and autograd registrations for custom sentinels."""

from typing import cast

import torch
import torch.distributed.tensor as dist_tensor
import torch.distributed.tensor._dtensor_spec as dtensor_spec
import torch.distributed.tensor._op_schema as dtensor_op_schema
import torch.distributed.tensor._ops.utils as dtensor_op_utils

# --- Python Fake Implementation ---


def compute_local_shape(
    global_shape: list[int], mesh_spec: str, sharding_spec: str
) -> list[int]:
  if not sharding_spec or not mesh_spec:
    return list(global_shape)

  mesh_dims = {}
  mesh_spec_axes = mesh_spec.split("[")[0]
  for pair in mesh_spec_axes.split(","):
    if ":" in pair:
      name, size = pair.split(":")
      mesh_dims[name] = int(size)

  dims = sharding_spec.split("|")[0].split(";")
  local_shape = list(global_shape)
  for i, dim in enumerate(dims):
    if i >= len(local_shape):
      break
    if not dim:
      continue
    for axis in dim.split(","):
      if axis in mesh_dims:
        local_shape[i] //= mesh_dims[axis]
  return local_shape


@torch.library.register_fake("tpu::spmd_metadata")
def spmd_metadata_fake(
    x, mesh_spec, sharding_spec_in, sharding_spec_out, global_shape, kind
):
  _ = sharding_spec_out
  if kind == "output":
    shape = compute_local_shape(global_shape, mesh_spec, sharding_spec_in)
  else:
    shape = global_shape

  return torch.empty(
      shape,
      dtype=x.dtype,
      layout=x.layout,
      device=x.device,
      requires_grad=x.requires_grad,
  )


# --- Autograd Registrations for Custom Sentinels ---


def spmd_metadata_setup_context(ctx, inputs, output):
  _ = output
  ctx.mesh_spec = inputs[1]
  ctx.sharding_spec_in = inputs[2]
  ctx.sharding_spec_out = inputs[3]
  ctx.source_shape = list(inputs[0].shape)
  ctx.kind = inputs[5]


def spmd_metadata_backward(ctx, grad_output):
  return (
      torch.ops.tpu.spmd_metadata(
          grad_output,
          ctx.mesh_spec,
          ctx.sharding_spec_out,  # Inverted: out becomes in
          ctx.sharding_spec_in,  # Inverted: in becomes out
          ctx.source_shape,
          ctx.kind,
      ),
      None,
      None,
      None,
      None,
      None,
  )


torch.library.register_autograd(
    "tpu::spmd_metadata",
    spmd_metadata_backward,
    setup_context=spmd_metadata_setup_context,
)


def _make_contiguous_strides(shape):
  strides = []
  current_stride = 1
  for dim in reversed(shape):
    strides.append(current_stride)
    current_stride *= dim
  return tuple(reversed(strides))


# --- Sharding Strategy Registrations for the DTensor Sharding Propagator ---


def parse_generic_sharding_str(
    sharding_str: str, device_mesh: dist_tensor.DeviceMesh
) -> tuple[dist_tensor.Placement, ...]:
  """Parses a generic Tensor-to-Mesh sharding string back into PyTorch DTensor placements."""
  if not sharding_str:
    return tuple(dist_tensor.Replicate() for _ in range(device_mesh.ndim))

  names = device_mesh.mesh_dim_names or [
      f"mesh_dim_{i}" for i in range(device_mesh.ndim)
  ]

  parts = sharding_str.split("|")
  dims_spec = parts[0]
  partial_spec = parts[1] if len(parts) > 1 else ""

  dims = dims_spec.split(";")

  placements: list[dist_tensor.Placement] = [
      dist_tensor.Replicate() for _ in range(device_mesh.ndim)
  ]

  for dim_idx, dim in enumerate(dims):
    if not dim:
      continue
    axis_names = dim.split(",")
    for name in axis_names:
      if name in names:
        mesh_dim = names.index(name)
        placements[mesh_dim] = dist_tensor.Shard(dim_idx)

  if partial_spec:
    partial_axes = partial_spec.split(",")
    for name in partial_axes:
      if name in names:
        mesh_dim = names.index(name)
        placements[mesh_dim] = dist_tensor.Partial("sum")

  return tuple(placements)


@dtensor_op_utils.register_op_strategy(torch.ops.tpu.spmd_metadata.default)
def spmd_metadata_strategy(
    schema: dtensor_op_schema.OpSchema,
) -> dtensor_op_schema.OpStrategy:
  """Defines the sharding strategy for spmd_metadata sentinels."""
  # args: (x, mesh_spec, sharding_spec_in, sharding_spec_out, global_shape,
  #        kind)
  input_strategy = cast(dtensor_op_schema.OpStrategy, schema.args_schema[0])
  sharding_spec_in = schema.args_schema[2]
  sharding_spec_out = schema.args_schema[3]
  global_shape = schema.args_schema[4]
  kind = schema.args_schema[5]

  parent_spec = input_strategy.strategies[0].output_specs
  if isinstance(parent_spec, (list, tuple)):
    parent_spec = parent_spec[0]
  parent_spec = cast(dtensor_spec.DTensorSpec, parent_spec)

  if kind == "output":
    placements = parse_generic_sharding_str(sharding_spec_in, parent_spec.mesh)
  else:
    placements = parse_generic_sharding_str(sharding_spec_out, parent_spec.mesh)

  new_tensor_meta = None
  if parent_spec.tensor_meta is not None:
    new_shape = torch.Size(global_shape)
    new_stride = _make_contiguous_strides(new_shape)
    new_tensor_meta = parent_spec.tensor_meta._replace(
        shape=new_shape, stride=new_stride
    )

  output_spec = dtensor_spec.DTensorSpec(
      mesh=parent_spec.mesh,
      placements=placements,
      tensor_meta=new_tensor_meta,
  )

  return dtensor_op_schema.OpStrategy([
      dtensor_op_schema.OpSpec(
          output_specs=output_spec, input_specs=(parent_spec,)
      )
  ])


def ensure_registered():
  """Ensures side-effects of this module (registrations) are executed."""
  pass
