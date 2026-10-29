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

"""Eager registration for torch.ops.higher_order.invoke_subgraph."""

from typing import Any
import torch
import torch._C


def _lazy_handle_invoke_subgraph(*args: Any, **kwargs: Any) -> Any:
  """Lazily imports nested_compile_region and dispatches invoke_subgraph."""
  # pylint: disable=g-import-not-at-top
  from torch_tpu._internal.compile import nested_compile_region
  # pylint: enable=g-import-not-at-top
  return nested_compile_region.handle_invoke_subgraph(*args, **kwargs)


def register() -> None:
  """Eagerly registers invoke_subgraph HOP for TPU and XLA dispatch keys."""
  try:
    for key in (
        torch._C.DispatchKey.XLA,
        torch._C.DispatchKey.PrivateUse1,
    ):
      torch.ops.higher_order.invoke_subgraph.py_impl(key)(
          _lazy_handle_invoke_subgraph
      )
  except (AttributeError, RuntimeError):
    # AttributeError: Safe to ignore if environment lacks XLA/TPU support.
    # RuntimeError: Safe to ignore if the implementation is already registered.
    pass
