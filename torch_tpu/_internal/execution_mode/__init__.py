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

from contextlib import contextmanager
from typing import Set, TypeAlias

from torch_tpu._internal import execution_mode_impl
from torch_tpu._internal.execution_mode_impl import get_eager_mode
from torch_tpu._internal.execution_mode_impl import set_eager_mode

EagerMode: TypeAlias = execution_mode_impl.EagerMode


@contextmanager
def eager_mode(mode: EagerMode):
  """Context manager for setting the execution mode."""
  old_eager_mode = execution_mode_impl.get_eager_mode()
  try:
    execution_mode_impl.set_eager_mode(mode)
    yield
  finally:
    execution_mode_impl.set_eager_mode(old_eager_mode)


@contextmanager
def cpu_fallback_mode(enabled: bool):
  """Context manager for setting the fallback mode."""
  old_enabled = execution_mode_impl.is_cpu_fallback_enabled()
  try:
    execution_mode_impl.enable_cpu_fallback(enabled)
    yield
  finally:
    execution_mode_impl.enable_cpu_fallback(old_enabled)


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "EagerMode",
    "cpu_fallback_mode",
    "eager_mode",
    "get_eager_mode",
    "set_eager_mode",
    # go/keep-sorted end
]
