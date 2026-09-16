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
from torch.utils import _pytree
from torch_tpu._internal.sync import _tpu_torch_sync


# TODO: add ways to support all these methods through public API features
# (streams, events, etc)
synchronize = _tpu_torch_sync.synchronize
is_materializing = _tpu_torch_sync.is_materializing
is_materialized = _tpu_torch_sync.is_materialized


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
