# Copyright 2025 Google LLC
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
"""Handles Python type overloading for synchronize() function."""

import inspect
import os

import torch
import torch.distributed.tensor as dt
from torch.utils import _pytree
from torch_tpu._internal.device import _device_ops_backend
from torch_tpu._internal.sync import _tpu_torch_sync


def _maybe_unwrap(tensor: torch.Tensor) -> torch.Tensor:
  if isinstance(tensor, dt.DTensor):
    return tensor._local_tensor  # pylint: disable=protected-access
  if hasattr(tensor, "elem"):
    # This is a FakeTensor (or another wrapper). Extract the inner tensor.
    return tensor.elem
  return tensor


# TODO: deprecate this in favor of standard torch.tpu.synchronize()
# or torch.tpu.Stream.synchronize()
def synchronize(
    tensors: torch.Tensor | list[torch.Tensor] | None = None, wait: bool = False
) -> None:
  """Forces a materialization of one or more TPU tensors.

  Args:
    tensors: what to synchronize. If None, synchronizes the entire graph of
      deferred operations. Otherwise, synchronizes only the specified tensor or
      list of tensors.
    wait: Whether to wait for the tensor(s) to be ready. If wait is False, the
      function will compiled a graph to compute the tensors and enqueue it for
      execution, but will not wait for the results. If wait is True, the
      function will also wait for the results to be ready.
  """
  if tensors is None:
    if not wait:
      raise NotImplementedError(
          "torch_tpu._internal.sync.synchronize(None, wait=False) is"
          " deprecated. Please migrate your code to use torch.tpu.synchronize()"
      )
    current_device_id = _device_ops_backend._get_current_device_id()  # pylint: disable=protected-access
    _device_ops_backend._synchronize_device(current_device_id)  # pylint: disable=protected-access
  elif isinstance(tensors, list):
    _tpu_torch_sync._synchronize_list(  # pylint: disable=protected-access
        [_maybe_unwrap(t) for t in tensors], wait
    )
  else:
    _tpu_torch_sync._synchronize_tensor(  # pylint: disable=protected-access
        _maybe_unwrap(tensors), wait
    )


def is_materializing(tensor: torch.Tensor) -> bool:
  return _tpu_torch_sync._is_materializing(  # pylint: disable=protected-access
      _maybe_unwrap(tensor)
  )


def is_materialized(tensor: torch.Tensor) -> bool:
  return _tpu_torch_sync._is_materialized(  # pylint: disable=protected-access
      _maybe_unwrap(tensor)
  )


def _dump_computation(
    str_to_dump: str,
    file_name: str,
):
  """Dumps a string to a file."""

  directory = os.path.dirname(file_name)
  if directory:
    os.makedirs(directory, exist_ok=True)
  with open(file_name, "w") as f:
    f.write(str_to_dump)


def dump_computation_graphviz(tensors: list[torch.Tensor], file_name: str):
  """Dumps a graphviz compatible representation of the aten traversal of the given tensors, to a file_name."""
  _dump_computation(computation_graphviz(*tensors), file_name)


def dump_computation_mlir(tensors: list[torch.Tensor], file_name: str):
  """Dumps a graphviz compatible representation of the aten traversal of the given tensors, to a file_name."""
  _dump_computation(computation_mlir(*tensors), file_name)


def computation_graphviz(
    *tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]
) -> str:
  """Returns a graphviz compatible representation of the aten traversal of the given tensors."""
  caller_frame = inspect.stack()[1]  # get the caller's frame
  caller_locals = dict(caller_frame[0].f_locals)
  flat_tensors, _ = _pytree.tree_flatten(tensors)
  return _tpu_torch_sync._get_computation_graphviz(flat_tensors, caller_locals)  # pylint: disable=protected-access


def computation_mlir(
    *tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]
) -> str:
  """Returns a MLIR representation of the aten traversal of the given tensors."""
  flat_tensors, _ = _pytree.tree_flatten(tensors)
  return _tpu_torch_sync._get_computation_mlir(flat_tensors)  # pylint: disable=protected-access
