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

"""Properties and context managers for setting various execution modes."""

import contextlib
import enum
import sys
import types
from typing import TYPE_CHECKING, TypeAlias

from torch_tpu._internal.execution_mode import execution_mode_impl
from torch_tpu._internal.utils import annotations

_EagerMode: TypeAlias = execution_mode_impl._EagerMode


@annotations.experimental("TPU eager execution mode configuration.")
class EagerMode(enum.Enum):
  """Eager execution mode for TPU operations."""

  DEFER_AND_FUSE = _EagerMode.DEFER_AND_FUSE
  DEFER_NEVER = _EagerMode.DEFER_NEVER
  DEFER_NEVER_AND_LAUNCH_BLOCKING = _EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
  _INTERNAL_COMPILE_FX_GRAPH = _EagerMode.INTERNAL_COMPILE_FX_GRAPH

  def __eq__(self, other: object) -> bool:
    if isinstance(other, EagerMode):
      return self.value == other.value
    if isinstance(other, _EagerMode):
      return self.value == other
    return False

  def __hash__(self) -> int:
    return hash(self.value)


def _unwrap_eager_mode(
    mode: EagerMode | _EagerMode,
) -> _EagerMode:
  """Unwraps an EagerMode wrapper to extract the underlying C++ pybind _EagerMode.

  When users pass a `torch.tpu.EagerMode` instance, we unwrap its `.value` so
  the C++ backend functions receive the expected
  `execution_mode_impl._EagerMode` pybind object. If an un-wrapped pybind enum
  is passed directly (e.g. from internal modules), it is returned as-is.
  """
  if isinstance(mode, EagerMode):
    return mode.value
  return mode


def _wrap_eager_mode(
    mode: _EagerMode | EagerMode,
) -> EagerMode | _EagerMode:
  """Wraps a raw C++ pybind _EagerMode into the public EagerMode enum.

  If the raw mode corresponds to one of the 3 public modes or
  `_INTERNAL_COMPILE_FX_GRAPH`, returns the matching `EagerMode` enum member. If
  it is an internal mode (e.g., `INTERNAL_DEFER_ALL`), `EagerMode(mode)` will
  raise `ValueError` and we safely return the raw mode fallback.
  """
  if isinstance(mode, EagerMode):
    return mode
  try:
    return EagerMode(mode)
  except ValueError:
    return mode


if TYPE_CHECKING:
  disable_inplace_buffer_donation: bool
  eager_mode: _EagerMode
  enable_cpu_fallback: bool


class _ExecutionModeModule(types.ModuleType):
  """Custom module for managing TorchTPU execution settings."""

  @property
  def eager_mode(self) -> EagerMode | _EagerMode:
    # pylint: disable-next=protected-access
    return _wrap_eager_mode(execution_mode_impl._get_eager_mode())

  @eager_mode.setter
  def eager_mode(self, value: EagerMode | _EagerMode):
    # pylint: disable-next=protected-access
    execution_mode_impl._set_eager_mode(_unwrap_eager_mode(value))

  @property
  def enable_cpu_fallback(self) -> bool:
    # pylint: disable-next=protected-access
    return execution_mode_impl._is_cpu_fallback_enabled()

  @enable_cpu_fallback.setter
  def enable_cpu_fallback(self, value: bool):
    # pylint: disable-next=protected-access
    execution_mode_impl._enable_cpu_fallback(value)

  @property
  def disable_inplace_buffer_donation(self) -> bool:
    # pylint: disable-next=protected-access
    return not execution_mode_impl._is_inplace_buffer_donation_enabled()

  @disable_inplace_buffer_donation.setter
  def disable_inplace_buffer_donation(self, value: bool):
    # pylint: disable-next=protected-access
    execution_mode_impl._enable_inplace_buffer_donation(not value)


sys.modules[__name__].__class__ = _ExecutionModeModule


@annotations.experimental("Context manager for setting the eager mode.")
@contextlib.contextmanager
def set_eager_mode(mode: EagerMode | _EagerMode):
  """Context manager for setting the eager mode."""
  # Push the new state.
  execution_mode_impl._push_eager_mode(_unwrap_eager_mode(mode))  # pylint: disable=protected-access
  try:
    yield
  finally:
    # Guarantee the state is restored even if Python code throws an exception.
    execution_mode_impl._pop_eager_mode()  # pylint: disable=protected-access


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "EagerMode",
    "disable_inplace_buffer_donation",
    "set_eager_mode",
    # go/keep-sorted end
]
