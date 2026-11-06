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

"""Provides utilities for managing accelerator devices across backends in tests."""

from typing import Any, Final

import torch
import torch_tpu  # pylint: disable=unused-import  # ruff: ignore[unused-import]
from torch_tpu._internal.utils import hardware
from tests import oss_utils

# Default device cap applied to multi-chip TPU tests in OSS presubmit
# (`requires-tpu-v5lite:4`). Multi-chip tests cap at 4 chips (the smallest
# valid 2x2x1 multi-chip sub-slice topology on TPU v4/v5e/v5p/v6e) so that tests
# use 4 chips in OSS CI even when scheduled on 8-chip hosts (such as
# `ct5lp-224-8tpu` in OSS).
# This only affects OSS and is the default cap, not a global maximum (some
# tests can choose not to cap the device count).
_OSS_DEFAULT_DEVICE_CAP: Final[int] = 4


def get_tpu_device_count(default: int = 0) -> int:
  """Returns the TPU device count for multi-device tests, capped at 4 in OSS.

  In OSS environments, multi-chip TPU tests are capped at 4 chips (the smallest
  valid 2x2x1 multi-chip topology on TPU v4/v5e/v5p/v6e) so that tests use 4
  chips in OSS presubmit (`requires-tpu-v5lite:4`) even when scheduled on 8-chip
  hosts. In internal Google3 environments, the cap is not applied and the full
  device count is returned.

  Queries `hardware.get_tpu_device_count()` rather than
  `torch.tpu.device_count()` so that the host's physical chip count is returned
  even when PJRT has already been initialized in single-device mode (e.g. via
  `SeededTest.setUp()`).

  Args:
    default: Fallback device count to return when no TPU devices are detected
      (e.g. on CPU-only runners where emulated devices are used).
  """
  count = hardware.get_tpu_device_count()
  if count == 0:
    return default
  if oss_utils.is_oss():
    return min(count, _OSS_DEFAULT_DEVICE_CAP)
  return count


def get_backend_module(backend_name: str) -> Any:
  """Returns the torch backend module corresponding to backend_name."""
  if backend_name in ("gpu", "cuda"):
    return torch.cuda
  elif backend_name == "tpu":
    return torch.tpu
  raise ValueError(
      f"Unsupported backend '{backend_name}'. Supported backends are: 'gpu',"
      " 'cuda', 'tpu'"
  )


def get_device(
    backend_name: str, device_idx: int | None = None
) -> torch.device:
  """Returns a torch.device for backend name and optional device index.

  Maps the CLI flag backend names ('gpu', 'cuda', 'tpu') to their corresponding
  PyTorch device types ('cuda', 'tpu').

  Args:
    backend_name: Name of the backend ('gpu', 'cuda', or 'tpu').
    device_idx: Optional integer index of the device (e.g. 0 for 'cuda:0'). If
      None, returns a device without an index (e.g. 'cuda').
  """
  if backend_name in ("gpu", "cuda"):
    device_type = "cuda"
  elif backend_name == "tpu":
    device_type = "tpu"
  else:
    raise ValueError(
        f"Unsupported backend '{backend_name}'. Supported backends are: 'gpu',"
        " 'cuda', 'tpu'"
    )
  if device_idx is None:
    return torch.device(device_type)
  return torch.device(f"{device_type}:{device_idx}")


def set_active_device(backend_name: str, device_idx: int) -> torch.device:
  """Sets active device to device_idx and returns torch.device."""
  torch.accelerator.set_device_index(device_idx)
  return get_device(backend_name, device_idx)
