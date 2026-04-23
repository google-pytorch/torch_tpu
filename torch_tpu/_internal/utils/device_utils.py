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

"""Utilities for device operations in TorchBench."""

import json
import os
from typing import Any, Callable, Sequence

from absl import logging
import psutil
import torch
from torch.utils._pytree import tree_flatten
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal import sync as tpu_sync

from torch_tpu._internal.shims.xprof import xprof_analysis_client


_BYTES_IN_MB = 1024 * 1024


def _get_peak_hbm_memory_mb(
    session_id: str | None = None,
    client: xprof_analysis_client.XprofAnalysisClient | None = None,
) -> float:
  """Get peak HBM usage in MB from Xprof memory profile.

  If session_id or client is not provided, returns -1.0. This happens when
  Xprof is not enabled.

  Args:
    session_id: The session ID to get the memory profile for.
    client: The xprof client to use for profiling.

  Returns:
    The peak HBM usage in MB.
  """

  if not session_id:
    return -1.0

  if not client:
    return -1.0
  # The get_profile_data method returns a tuple of (content_type, data).
  _, data = client.get_profile_data('memory_profile.json', session_id)

  if not data:
    logging.warning(
        'No memory_profile.json data found for session_id: %s', session_id
    )
    return -1.0

  try:
    profile = json.loads(data.decode('utf-8'))
    # See
    # https://github.com/openxla/xprof/blob/master/plugin/xprof/protobuf/memory_profile.proto
    # for the output schema.
    # The key '0' corresponds to the allocator for device 0.
    allocator_stats = profile['memoryProfilePerAllocator']['0']
    peak_bytes = allocator_stats['profileSummary']['peakStats'][
        'peakBytesInUse'
    ]
    return float(peak_bytes) / _BYTES_IN_MB
  except (KeyError, json.JSONDecodeError, ValueError):
    logging.warning(
        'Failed to parse memory profile for session_id %s',
        session_id,
        exc_info=True,
    )
    return -1.0


def get_peak_memory_hbm(
    device,
    session_id: str | None = None,
    xprof_client: xprof_analysis_client.XprofAnalysisClient | None = None,
) -> float:
  """Get peak memory usage for the specified device.

  Args:
    device: The device type ('cuda', 'cpu', 'tpu', 'xla_cuda').
    session_id: Optional Xprof session ID to retrieve TPU/XLA_CUDA peak memory
      from. This is a no-op for CPU and CUDA.
    xprof_client: Optional Xprof analysis client to use for retrieving memory
      profile data.

  Returns:
    The peak memory usage in MB.
  """
  if device == 'cuda':
    peak_memory_usage_mb = (
        torch.cuda.memory.max_memory_allocated() / _BYTES_IN_MB
    )
  elif device == 'cpu':
    total = psutil.virtual_memory().total
    percentage = psutil.Process(os.getpid()).memory_percent()
    peak_memory_usage_mb = percentage / 100.0 * total / _BYTES_IN_MB
  elif device in ('tpu', 'xla_cuda'):
    peak_memory_usage_mb = _get_peak_hbm_memory_mb(session_id, xprof_client)
  else:
    raise ValueError(f'Unsupported device: {device}')
  return peak_memory_usage_mb


def reset_peak_memory_stats(device):
  if device == 'cuda':
    torch.cuda.memory.reset_max_memory_allocated()
  elif device == 'tpu' or device == 'cpu' or device == 'xla_cuda':
    pass
  else:
    raise ValueError(f'Unsupported device: {device}')


def _collect_tensors(tensors: Any) -> Sequence[torch.Tensor]:
  leaves, _ = tree_flatten(tensors)
  return [l for l in leaves if isinstance(l, torch.Tensor)]


def synchronize(
    device,
    tensor_to_sync: (
        torch.Tensor | dict[str, torch.Tensor] | Sequence[torch.Tensor]
    ),
):
  """Synchronizes the device with the given tensor(s).

  Args:
    device: The device type ('cuda', 'tpu', 'xla_cuda').
    tensor_to_sync: The tensor(s) to synchronize.
  """
  if device == 'cuda':
    torch.cuda.synchronize()
  elif device == 'tpu' or device == 'xla_cuda':
    # TODO(b/507181043): Investigate why sync(None, wait=True) doesn't work.
    tensors = _collect_tensors(tensor_to_sync)
    tpu_sync.synchronize(tensors, wait=True)
  else:
    raise ValueError(f'Unsupported device: {device}')


def cache_miss_count(device: str) -> int:
  """Get cache miss count for the specified device. Always returns 0 for CUDA.

  Args:
    device: The device type ('cuda', 'tpu', 'xla_cuda').

  Returns:
    The cache miss count.
  """
  if device == 'cuda':
    return 0
  elif device in ('tpu', 'xla_cuda'):
    # pylint: disable=protected-access
    return getattr(torch, device)._get_cache_misses()
  else:
    raise ValueError(f'Unsupported device: {device}')


def torch_compile(func: Callable[..., Any], device: str) -> Callable[..., Any]:
  """Wraps a callable with `torch.compile` based on the specified device.

  Args:
    func: The callable to be compiled.
    device: The device type ('cuda', 'tpu', 'xla_cuda').

  Returns:
    The compiled callable.

  Raises:
    ValueError: If the device is not supported.
  """
  if device == 'cuda':
    func = torch.compile(func, backend='inductor')
  elif device in ('tpu', 'xla_cuda'):
    func = torch.compile(
        func, dynamic=False, backend=torch_tpu_compile.TpuBackend()
    )
  else:
    raise ValueError(f'Unsupported device: {device}')
  return func


def clear_cache(device: str) -> None:
  """Clears the compilation cache for the specified device. No-op for CUDA.

  Args:
    device: The device type ('cuda', 'tpu', 'xla_cuda').

  Raises:
    ValueError: If the device is not supported.
  """
  if device == 'cuda':
    pass
  elif device in ('tpu', 'xla_cuda'):
    # pylint: disable=protected-access
    getattr(torch, device)._clear_cache()
  else:
    raise ValueError(f'Unsupported device: {device}')
