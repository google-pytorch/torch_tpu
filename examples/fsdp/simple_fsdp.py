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
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SimpleFSDP: Compile-first Fully Sharded Data Parallel via DTensor.

This file is branched from TorchTitan's SimpleFSDP implementation
(https://github.com/pytorch/torchtitan/blob/d771abb062600127b9dd1630a15e5a9dd2259bc5/torchtitan/experiments/graph_trainer/simple_fsdp.py),
with TorchTitan-specific (`torchtitan.*`) and `spmd_types` dependencies
removed so that `simple_fsdp` is self-contained and only depends on PyTorch.
"""

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
import sys

import torch
from torch import nn
from torch.distributed._tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import Placement, _StridedShard

_active_parametrization = True


@contextmanager
def disable_active_parametrization() -> Generator[None, None, None]:
    global _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = True


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


def _distribute_dtensor(
    tensor: DTensor,
    device_mesh: DeviceMesh,
    dp_placements: Sequence[Placement],
) -> DTensor:
  """Below are experimental enhancements to distribute a DTensor.

  This helps enable Simple FSDP + TP/EP, in which

      inner spec/mesh is TP/EP spec/mesh
      outer spec/mesh is FSDP/DDP/HSDP spec/mesh
  """
  inner_spec = tensor._spec
  outer_mesh, inner_mesh = device_mesh, inner_spec.mesh
  spanned_mesh = DeviceMesh._concatenate([outer_mesh, inner_mesh])

  if len(dp_placements) == 1:
    assert dp_placements[0].is_replicate() or dp_placements[0].is_shard()
    if isinstance(dp_placements[0], Shard):
      # For FSDP + EP/TP/EP+TP
      assert len(inner_spec.placements) == 2 or len(inner_spec.placements) == 1
      shard_dim = dp_placements[0].dim
      split_factor = inner_spec.num_shards_map[shard_dim]
      tensor_placement = (
          (
              _StridedShard(shard_dim, split_factor=split_factor)
              if split_factor > 1
              else dp_placements[0]
          ),
      ) + inner_spec.placements
    else:
      # For DDP + TP/EP
      assert len(inner_spec.placements) == 1
      tensor_placement = (dp_placements[0], inner_spec.placements[0])
  elif len(dp_placements) == 2:
    assert dp_placements[0].is_replicate() and isinstance(
        dp_placements[1], Shard
    )
    # For HSDP + EP/TP/EP+TP
    assert len(inner_spec.placements) == 2 or len(inner_spec.placements) == 1
    shard_dim = dp_placements[1].dim
    split_factor = inner_spec.num_shards_map[shard_dim]
    tensor_placement = (
        dp_placements[0],
        (
            _StridedShard(shard_dim, split_factor=split_factor)
            if split_factor > 1
            else dp_placements[1]
        ),
    ) + inner_spec.placements
  else:
    raise ValueError(
        f"Unsupported placement {dp_placements} for distributing DTensor"
        f" {tensor}"
    )

  # HSDP case needs 2 placements for 2D outer_mesh
  current_placements = (Replicate(),) * len(dp_placements)
  target_placements = tuple(dp_placements)

  current_spec = DTensorSpec(
      mesh=outer_mesh,
      placements=current_placements,
      tensor_meta=inner_spec.tensor_meta,
  )
  target_spec = DTensorSpec(
      mesh=outer_mesh,
      placements=target_placements,
      tensor_meta=inner_spec.tensor_meta,
  )
  result_tensor = redistribute_local_tensor(
      tensor._local_tensor,
      current_spec=current_spec,
      target_spec=target_spec,
  )
  return DTensor(
      result_tensor.requires_grad_(tensor.requires_grad),
      DTensorSpec(
          mesh=spanned_mesh,
          placements=tensor_placement,
          tensor_meta=inner_spec.tensor_meta,
      ),
      requires_grad=tensor.requires_grad,
  )


_wrap_class_id = count()
_wrap_class_cache: dict[tuple[type, tuple[str, ...]], type] = {}


def _register_parametrization(
    module: nn.Module,
    param_names: list[str],
    parametrization_init: Callable[[str], nn.Module],
) -> None:
  """It works with state_dict without incurring parametrization calls because

  state_dict accesses parameters directly from self._parameters, not from
  getters
  https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L2141
  TODO: In checkpoint saving/loading, avoid parametrization calls when calling
  get_model_state_dict func in
  third_party/py/torchtitan/components/checkpointer/dcp.py.
  """
  parametrizations = [parametrization_init(pn) for pn in param_names]
  # Same-type modules share one wrapper class so per-layer torch.compile
  # guards on type(self) hit instead of recompiling every layer. Properties
  # read the per-instance parametrization from the module.
  cache_key = (module.__class__, tuple(param_names))
  module_cls = _wrap_class_cache.get(cache_key)
  if module_cls is None:
    param_name_to_property = {
        pn: property(
            lambda self, i=i, pn=pn: self._simple_fsdp_parametrization_list[i](
                self._parameters[pn]
            )
        )
        for i, pn in enumerate(param_names)
    }
    module_cls = type(
        f"SimpleFSDP{module.__class__.__name__}_{next(_wrap_class_id)}",
        (module.__class__,),
        param_name_to_property,
    )
    # Expose the dynamically created class as a real, importable symbol
    # so that pickle/GraphPickler can resolve it during serialization.
    sys.modules[module_cls.__module__].__dict__[
        module_cls.__name__
    ] = module_cls
    _wrap_class_cache[cache_key] = module_cls
  module.__class__ = module_cls
  module._simple_fsdp_parametrization_list = parametrizations


class ReplicateComputation(nn.Module):

  def __init__(
      self,
      device_mesh: DeviceMesh,
      param_sharding: tuple[Placement, ...],
      mode: str,
      full_dtensor: bool = False,
      mp_policy: MixedPrecisionPolicy | None = None,
  ) -> None:
    super().__init__()
    self.device_mesh = device_mesh
    self.param_sharding = param_sharding
    self.mode = mode
    self.compute_placements: list[Placement] = [
        Replicate()
    ] * self.device_mesh.ndim
    self.grad_placements: list[Placement] = [
        Partial(reduce_op="sum")
    ] * self.device_mesh.ndim
    self.full_dtensor = full_dtensor
    mp_policy = mp_policy or MixedPrecisionPolicy()
    self.param_dtype: torch.dtype | None = mp_policy.param_dtype
    self.reduce_dtype: torch.dtype | None = mp_policy.reduce_dtype

  def replicate_compute(self, x: DTensor) -> torch.Tensor:
    # data parallel runtime replicate parameters and do local compute
    # the gradients are partial tensors that needs to perform reduction
    # (i.e. DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)
    # support FSDP/DDP/HSDP + EP + TP (assuming TP shards the inner-most dim)
    non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
    assert non_dp_mesh_dims <= 2, "Only DP + EP/TP/EP+TP is supported"
    if non_dp_mesh_dims > 0:
      if self.full_dtensor:
        raise NotImplementedError(
            "full_dtensor is not supported for DP + EP/TP/EP+TP"
        )

      dp_mesh = self.device_mesh
      # re-wrap 2D DTensor to 1D DTensor on dp_mesh for efficient FSDP all-gather
      sharded_local_tensor = x.to_local()
      sharded_dtensor = DTensor.from_local(
          sharded_local_tensor, dp_mesh, self.param_sharding
      )

      # the actual FSDP's fwd all-gather & bwd reduce-scatter
      # DDP's bwd all-reduce on dp_mesh
      replicated_dtensor = sharded_dtensor.redistribute(
          placements=self.compute_placements,
          forward_dtype=self.param_dtype,
          backward_dtype=self.reduce_dtype,
      )

      # re-wrap all-gathered DTensor on dp_mesh to be on non_dp_mesh
      # TODO: DTensor should support this mesh collapsing operation
      replicated_local_tensor = replicated_dtensor.to_local(
          grad_placements=self.grad_placements
      )

      non_dp_placements = tuple(x._spec.placements[-non_dp_mesh_dims:])
      assert x._spec.mesh.mesh_dim_names is not None
      non_dp_mesh_dim_names = tuple(
          x._spec.mesh.mesh_dim_names[-non_dp_mesh_dims:]
      )
      non_dp_mesh = x._spec.mesh[non_dp_mesh_dim_names]
      output = DTensor.from_local(
          replicated_local_tensor, non_dp_mesh, non_dp_placements
      )
    elif non_dp_mesh_dims == 0:
      output = x.redistribute(
          placements=self.compute_placements,
          forward_dtype=self.param_dtype,
          backward_dtype=self.reduce_dtype,
      )
      if not self.full_dtensor:
        output = output.to_local(grad_placements=self.grad_placements)
    else:
      raise AssertionError(
          f"Unsupported replicate compute on placement {x._spec.placements} for"
          f" DTensor {x}"
      )

    return output

  def forward(self, x: DTensor) -> torch.Tensor:
    global _active_parametrization
    # This should never be set to true during forward, only outside for model
    # inspection / debugging / initialization
    # model initialization can be done now through
    # with disable_active_parametrization():
    #     model.init_states()
    if not _active_parametrization:
      return x

    output = self.replicate_compute(x)
    return output


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int | dict[str, int] = 0,
    full_dtensor: bool = False,
    ignored_params: set[nn.Parameter] | None = None,
) -> nn.Module:
  """Shard ``model`` and install the data-parallel parametrization."""
  # A dict `shard_dim` names an axis per parameter, for a module whose
  # parameters do not all divide on the same one. It has no single axis to
  # build `param_sharding` from, so the tuples below carry a placeholder:
  # every parameter overrides it, and the one reader that would see it
  # through -- `ReplicateComputation`'s EP/TP path -- is refused outright.
  template_dim = 0 if isinstance(shard_dim, dict) else shard_dim

  param_sharding: tuple[Placement, ...]
  if mode == "replicate":
    param_sharding = (Replicate(),)
  elif mode == "fully_shard":
    param_sharding = (Shard(template_dim),)
  elif mode == "hybrid_shard":
    # replicate inter-host, fully shard intra-host
    param_sharding = (Replicate(), Shard(template_dim))
    assert (
        device_mesh.ndim == 2
    ), "hybrid sharded data parallel requires 2D DeviceMesh"
  else:
    raise ValueError(f"Unsupported mode {mode}")

  modules = list(model.modules())

  for mod in modules:
    params_dict = dict(mod.named_parameters(recurse=False))
    # we shouldn't apply data parallel to the modules that are already
    # sharded by data parallel
    if "SimpleFSDP" in mod.__class__.__name__:
      continue

    # Skip parameters that are explicitly ignored (e.g., shared or frozen weights).
    if ignored_params is not None:
      params_dict = {
          p_name: p
          for p_name, p in params_dict.items()
          if p not in ignored_params
      }
      if not params_dict:
        continue

    for p_name, p in params_dict.items():
      if p is not None and p.numel() > 0:
        dim = shard_dim[p_name] if isinstance(shard_dim, dict) else shard_dim
        if isinstance(p, DTensor) and dim != template_dim:
          # Under EP/TP, `ReplicateComputation` re-wraps the parameter
          # with `param_sharding`, which carries `template_dim` and so
          # would drop this axis.
          raise NotImplementedError(
              f"{p_name} is already a DTensor, so sharding it on dim "
              f"{dim} would be dropped on the EP/TP path"
          )
        # Check if the parameter is smaller than the device mesh dimension.
        # If so, replicate it instead of sharding it.
        actual_sharding = [
            Shard(dim) if isinstance(pl, Shard) else pl for pl in param_sharding
        ]
        for i, placement in enumerate(actual_sharding):
          if isinstance(placement, Shard) and p.shape[
              placement.dim
          ] < device_mesh.size(i):
            actual_sharding[i] = Replicate()
        actual_sharding = tuple(actual_sharding)

        distribute_tensor_func = (
            _distribute_dtensor if isinstance(p, DTensor) else distribute_tensor
        )
        mod.register_parameter(
            p_name,
            nn.Parameter(
                distribute_tensor_func(p, device_mesh, actual_sharding),
                requires_grad=p.requires_grad,
            ),
        )

        # to be compatible with DCP, we use a customized _register_parametrization
        # instead of nn.utils.parametrize.register_parametrization here
        # nn.utils.parametrize.register_parametrization(
        #     mod,
        #     p_name,
        #     ReplicateComputation(
        #         device_mesh,
        #         param_sharding,
        #         mode,
        #         mp_policy=mp_policy,
        #     ),
        #     unsafe=True,
        # )

    valid_param_names = [
        p_name
        for p_name, p in params_dict.items()
        if p is not None and p.numel() > 0
    ]
    if valid_param_names:
      _register_parametrization(
          mod,
          valid_param_names,
          lambda param_name: ReplicateComputation(
              device_mesh=device_mesh,
              param_sharding=param_sharding,
              mode=mode,
              full_dtensor=full_dtensor,
              mp_policy=mp_policy,
          ),
      )
  return model
