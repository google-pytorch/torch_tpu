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
"""Utilities for SPMD execution in torch_tpu."""

from collections.abc import Callable
import functools
from typing import Any, TypeVar

import torch
from torch import distributed as dist
from torch.utils import _pytree
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.sync import sync
from torch_tpu._internal.utils import annotations

_F = TypeVar("_F", bound=Callable[..., Any])

SPMD_SAFE_METADATA_KEY = "spmd_safe"


def _synchronize_tensors(args, kwargs):
  flat_args, _ = _pytree.tree_flatten((args, kwargs))
  tensors = [
      t
      for t in flat_args
      if isinstance(t, torch.Tensor) and t.device.type != "cpu"
  ]
  if tensors:
    sync.synchronize(tensors, wait=False)


@annotations.experimental(
    "torch.tpu.distributed.spmd_safe is experimental and subject to change."
)
def spmd_safe(func: _F) -> _F:
  """Marks a function as SPMD-safe, i.e. run identically on every rank.

  By default, TorchTPU runs each eager collective, such as
  `torch.distributed.all_reduce`, in an execution graph of its own, separate
  from the surrounding operations, so that ranks that fuse different operations
  around a collective cannot deadlock. In a function decorated with
  `spmd_safe`, collectives are instead fused with the surrounding operations
  into the same graphs, which is more efficient. Only decorate functions that
  issue the same TPU operations, including collectives, in the same order and
  with the same shapes on all ranks; code with rank-dependent control flow must
  not be decorated.

  The decorated function also synchronizes its input and output TPU tensors
  without waiting, so that no operations from outside the function are fused
  into the graphs it produces. Under `torch.compile`, the decorator annotates
  the traced graph nodes instead.

  Args:
    func: The function to decorate.

  Returns:
    A wrapper of `func` with the same signature and return value.

  Example:

    @torch.tpu.distributed.spmd_safe
    def scaled_all_reduce(x: torch.Tensor) -> torch.Tensor:
      y = x * 5
      torch.distributed.all_reduce(y)
      return y
  """

  @functools.wraps(func)
  def wrapper(*args: Any, **kwargs: Any) -> Any:
    is_compiling = torch.compiler.is_compiling()

    if not is_compiling:
      _synchronize_tensors(args, kwargs)

      tpu_distributed.enter_spmd_safe_region()
      try:
        result = func(*args, **kwargs)
      finally:
        tpu_distributed.exit_spmd_safe_region()

      _synchronize_tensors(result, {})
      return result

    # In compile mode, annotate the FX tracing context with spmd_safe metadata.
    with torch.fx.traceback.annotate({SPMD_SAFE_METADATA_KEY: True}):
      return func(*args, **kwargs)

  return wrapper  # pyrefly: ignore[bad-return]


def make_standalone(group: dist.ProcessGroup) -> None:
  """Makes a process subgroup launch its programs without the other subgroups.

  By default, the collectives of a subgroup are part of a program for every
  device of the slice, which all subgroups launch together and in the same
  order. The collectives of a standalone subgroup, and the Pallas kernels over
  its mesh, are compiled for the devices of the subgroup alone, so its members
  launch them on their own, e.g. the stages of a pipeline. A program holds the
  collectives and kernels of one standalone subgroup only, and no
  sparse-dense-matmul lookup, whose table is sharded across the SparseCores of
  all devices.

  Every member of `group` must call this before the first collective of the
  group; the call waits for all of them. No-op for a group of the world size.

  Args:
    group: The process subgroup to make standalone.

  Raises:
    RuntimeError: If libtpu is older than 0.0.47; libtpu 0.0.44.1 to 0.0.46.1
      halt programs of subgroups that span several chips.

  Example:
    make_standalone(stage_group)
    dist.all_reduce(x, group=stage_group)  # Launched by the stage's ranks.
  """
  backend = group._get_backend(torch.device("tpu"))
  tpu_distributed.make_standalone(backend)
