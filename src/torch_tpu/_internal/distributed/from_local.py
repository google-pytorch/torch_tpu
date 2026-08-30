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

"""DTensor.from_local implementation for TorchTPU."""

from collections.abc import Callable
import os
from typing import Any, Final
import warnings

import torch
import torch.distributed.tensor as dt
from torch_tpu._internal import execution_mode

_UNEVEN_SHARDING_WARNING: Final[str] = (
    "`DTensor.from_local` called without `shape` and `stride`. Unevenly"
    " sharded tensors across ranks may cause collectives to hang or error out."
    " Provide global `shape` and `stride` when shapes differ across ranks."
)

_original_from_local: Callable[..., Any] = dt.DTensor.from_local


def _is_debug_eager_mode() -> bool:
  """Returns True if running in debug eager mode."""
  if torch.compiler.is_compiling():
    return False

  debug_check_env = os.getenv(
      "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS", ""
  ).lower()
  if debug_check_env in ("0", "false"):
    return False
  if debug_check_env in ("1", "true"):
    return True

  return (
      execution_mode.eager_mode
      == execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
  )


def from_local(
    *args: Any,
    run_check: bool | None = None,
    shape: torch.Size | None = None,
    stride: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> dt.DTensor:
  """Wraps `DTensor.from_local` with shape checks.

  Warns when global `shape` and `stride` are omitted, and enables
  `run_check=True` in debug eager mode unless explicitly set.
  """
  if shape is None or stride is None:
    warnings.warn(_UNEVEN_SHARDING_WARNING, UserWarning, stacklevel=2)

  if run_check is None:
    run_check = _is_debug_eager_mode()

  return _original_from_local(
      *args,
      run_check=run_check,
      shape=shape,
      stride=stride,
      **kwargs,
  )


def patch_from_local() -> None:
  """Overrides DTensor.from_local with TorchTPU's implementation."""
  if dt.DTensor.from_local is not from_local:
    dt.DTensor.from_local = staticmethod(from_local)
