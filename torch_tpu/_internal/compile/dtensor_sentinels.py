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

import re
from typing import cast

import torch
import torch.distributed.tensor as dist_tensor
import torch.distributed.tensor._dtensor_spec as dtensor_spec
import torch.distributed.tensor._op_schema as dtensor_op_schema
import torch.distributed.tensor._ops.utils as dtensor_op_utils

# --- Autograd Registrations for Custom Sentinels ---


def sdy_sharding_setup_context(ctx, inputs, output):
  # pylint: disable=unused-argument
  # inputs is (x, mesh_sdy_format, placements_sdy_format, target_shape,
  # is_input)
  ctx.mesh_sdy_format = inputs[1]
  ctx.placements_sdy_format = inputs[2]
  ctx.source_shape = list(inputs[0].shape)
  ctx.is_input = inputs[4]


def sdy_sharding_backward(ctx, grad_output):
  # Backward of input sentinel is output sentinel, and vice versa.
  # We toggle is_input.
  grad_is_input = not ctx.is_input

  # The target shape for the backward sentinel is the source shape of the
  # forward sentinel.
  # (i.e. if forward was local -> global, backward is global -> local, so
  # target shape is local).
  grad_target_shape = ctx.source_shape

  return (
      torch.ops.torch_tpu.sdy_sharding(
          grad_output,
          ctx.mesh_sdy_format,
          ctx.placements_sdy_format,
          grad_target_shape,
          grad_is_input,
      ),
      None,
      None,
      None,
      None,
  )


torch.library.register_autograd(
    "torch_tpu::sdy_sharding",
    sdy_sharding_backward,
    setup_context=sdy_sharding_setup_context,
)


def sdy_sharding_constraint_setup_context(ctx, inputs, output):
  # pylint: disable=unused-argument
  ctx.mesh_sdy_format = inputs[1]
  ctx.placements_sdy_format = inputs[2]


def sdy_sharding_constraint_backward(ctx, grad_output):
  return (
      torch.ops.torch_tpu.sdy_sharding_constraint(
          grad_output, ctx.mesh_sdy_format, ctx.placements_sdy_format
      ),
      None,
      None,
  )


torch.library.register_autograd(
    "torch_tpu::sdy_sharding_constraint",
    sdy_sharding_constraint_backward,
    setup_context=sdy_sharding_constraint_setup_context,
)


# --- Sharding Strategy Registrations for the DTensor Sharding Propagator ---


def parse_placements_str(placements_str: str):
  """Parses a serialized placements string back into DTensor placements."""
  placements = []
  for p_str in placements_str.split(","):
    p_str = p_str.strip()
    if "Shard" in p_str or p_str.startswith("S("):
      dim_match = re.search(r"Shard\((\d+)\)|S\((\d+)\)", p_str)
      if dim_match is not None:
        dim_str = dim_match.group(1) or dim_match.group(2)
        if dim_str is not None:
          placements.append(dist_tensor.Shard(int(dim_str)))
    elif "Replicate" in p_str or p_str == "R":
      placements.append(dist_tensor.Replicate())
  return tuple(placements)


@dtensor_op_utils.register_op_strategy(torch.ops.torch_tpu.sdy_sharding.default)
def sdy_sharding_strategy(
    schema: dtensor_op_schema.OpSchema,
) -> dtensor_op_schema.OpStrategy:
  """Defines the sharding strategy for sdy_sharding sentinels."""
  # args: (x, mesh_sdy_format, placements_sdy_format, target_shape, is_input)
  input_strategy = cast(dtensor_op_schema.OpStrategy, schema.args_schema[0])
  is_input = schema.args_schema[4]

  parent_spec = input_strategy.strategies[0].output_specs
  if isinstance(parent_spec, (list, tuple)):
    parent_spec = parent_spec[0]
  parent_spec = cast(dtensor_spec.DTensorSpec, parent_spec)

  if is_input:
    placements_str = schema.args_schema[2]
    placements = parse_placements_str(placements_str)
    target_shape = schema.args_schema[3]

    # Create global output DTensorSpec
    output_spec = dtensor_spec.DTensorSpec(
        mesh=parent_spec.mesh,
        placements=placements,
        tensor_meta=parent_spec.tensor_meta,
    )
    if output_spec.tensor_meta is not None:
      output_spec.tensor_meta.shape = torch.Size(target_shape)
  else:
    local_shape = schema.args_schema[3]

    # Create replicated / local physical output DTensorSpec
    output_spec = dtensor_spec.DTensorSpec(
        mesh=parent_spec.mesh,
        placements=tuple(
            dist_tensor.Replicate() for _ in range(parent_spec.mesh.ndim)
        ),
        tensor_meta=parent_spec.tensor_meta,
    )
    if output_spec.tensor_meta is not None:
      output_spec.tensor_meta.shape = torch.Size(local_shape)

  return dtensor_op_schema.OpStrategy([
      dtensor_op_schema.OpSpec(
          output_specs=output_spec, input_specs=(parent_spec,)
      )
  ])


@dtensor_op_utils.register_op_strategy(
    torch.ops.torch_tpu.sdy_sharding_constraint.default
)
def sdy_sharding_constraint_strategy(
    schema: dtensor_op_schema.OpSchema,
) -> dtensor_op_schema.OpStrategy:
  """Defines the strategy for sdy_sharding_constraint sentinels."""
  # args: (x, mesh_sdy_format, placements_sdy_format)
  input_strategy = cast(dtensor_op_schema.OpStrategy, schema.args_schema[0])
  placements_str = schema.args_schema[2]
  placements = parse_placements_str(placements_str)

  parent_spec = input_strategy.strategies[0].output_specs
  if isinstance(parent_spec, (list, tuple)):
    parent_spec = parent_spec[0]
  parent_spec = cast(dtensor_spec.DTensorSpec, parent_spec)

  output_spec = dtensor_spec.DTensorSpec(
      mesh=parent_spec.mesh,
      placements=placements,
      tensor_meta=parent_spec.tensor_meta,
  )
  return dtensor_op_schema.OpStrategy([
      dtensor_op_schema.OpSpec(
          output_specs=output_spec, input_specs=(parent_spec,)
      )
  ])
