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

"""Context managers for setting various execution modes."""

import contextlib
import sys
import types
from typing import TypeAlias

import torch
from torch_tpu._internal.execution_mode import execution_mode_impl
from torch_tpu._internal.execution_mode.execution_mode_impl import get_eager_mode
from torch_tpu._internal.execution_mode.execution_mode_impl import set_eager_mode

EagerMode: TypeAlias = execution_mode_impl.EagerMode


class _ExecutionModeModule(types.ModuleType):
  """Custom module for managing TorchTPU execution settings."""

  @property
  def enable_cpu_fallback(self) -> bool:
    # pylint: disable-next=protected-access
    return execution_mode_impl._is_cpu_fallback_enabled()

  @enable_cpu_fallback.setter
  def enable_cpu_fallback(self, value: bool):
    # pylint: disable-next=protected-access
    execution_mode_impl._enable_cpu_fallback(value)


sys.modules[__name__].__class__ = _ExecutionModeModule


@contextlib.contextmanager
def eager_mode(mode: EagerMode):
  """Context manager for setting the eager mode."""
  # Push the new state.
  execution_mode_impl._push_eager_mode(mode)  # pylint: disable=protected-access
  try:
    yield
  finally:
    # Guarantee the state is restored even if Python code throws an exception.
    execution_mode_impl._pop_eager_mode()  # pylint: disable=protected-access


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "EagerMode",
    "eager_mode",
    "get_eager_mode",
    "set_eager_mode",
    # go/keep-sorted end
]
