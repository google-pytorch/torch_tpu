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
from torch_tpu._internal import sync as tpu_sync
from torch_tpu._internal.benchmarks import xprof_adapter

_BYTES_IN_MB = 1024 * 1024


def _get_peak_hbm_memory_mb(
    session_id: str | None = None,
    client: xprof_adapter.XprofAnalysisClient | None = None,
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
    allocator_mem_profiles = profile.get('memoryProfilePerAllocator', {})
    if '0' in allocator_mem_profiles:
      allocator_stats = allocator_mem_profiles['0']
    elif allocator_mem_profiles:
      # If '0' is not present, try the first available allocator
      first_key = list(allocator_mem_profiles.keys())[0]
      logging.warning(
          "Allocator '0' not found, using '%s'. Available allocators: %s",
          first_key,
          list(allocator_mem_profiles.keys()),
      )
      allocator_stats = allocator_mem_profiles[first_key]
    else:
      logging.warning('No allocators found in memory profile')
      return -1.0

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


def get_max_total_device_time(
    session_id: str | None = None,
    client: xprof_adapter.XprofAnalysisClient | None = None,
) -> float:
  """Gets the max total device time across all devices running the model.

  Args:
    session_id: The session ID to get the profile for.
    client: The xprof client to use.

  Returns:
    The total device active time in seconds, or None if failed or not available.
  """
  logging.info(
      'get_total_device_time called with session_id=%s, client=%s',
      session_id,
      client,
  )
  if not session_id or not client:
    logging.warning('get_total_device_time: session_id or client is None')
    return -1.0

  try:
    logging.info('Fetching hosts for session_id=%s', session_id)
    hosts = client.get_hosts(session_id)
    if not hosts:
      logging.warning('No hosts found for session_id: %s', session_id)
      return -1.0

    max_device_time_s = 0.0
    has_any_device = False

    for host in hosts:
      logging.info(
          'Fetching XSpace for session_id=%s, host=%s', session_id, host
      )
      xspace = client.get_xspace(session_id, host=host)
      if xspace is None:
        logging.warning('get_xspace returned None for host: %s', host)
        continue

      for plane in xspace.planes:
        if plane.name.startswith('/device:'):
          logging.info('Analyzing plane: %s on host: %s', plane.name, host)
          device_time_ps = 0
          event_count = 0
          for line in plane.lines:
            if line.name.startswith('XLA Modules'):
              for e in line.events:
                device_time_ps += e.duration_ps
                event_count += 1

          if event_count > 0:
            device_time_s = device_time_ps / 1e12
            logging.info(
                'Device %s on host %s time: %.6f seconds (events=%d)',
                plane.name,
                host,
                device_time_s,
                event_count,
            )
            if device_time_s > max_device_time_s:
              max_device_time_s = device_time_s
              has_any_device = True

    if not has_any_device:
      logging.warning('No XLA Modules execution events found in any XSpace')
      return -1.0

    logging.info(
        'Calculated max_device_time_s across all hosts: %s',
        max_device_time_s,
    )
    return max_device_time_s

  except Exception:  # pylint: disable=broad-except
    logging.exception(
        'Failed to parse device time from XSpace for session_id %s',
        session_id,
    )
    return -1.0


def get_peak_memory_hbm(
    device,
    session_id: str | None = None,
    xprof_client: xprof_adapter.XprofAnalysisClient | None = None,
) -> float:
  """Get peak memory usage for the specified device.

  Args:
    device: The device type ('cuda', 'cpu', 'tpu', 'xla_cuda', 'xla_cpu',
      'jax').
    session_id: Optional Xprof session ID to retrieve TPU/XLA_CUDA/JAX peak
      memory from. This is a no-op for CPU and CUDA.
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
    if hasattr(torch, 'accelerator') and hasattr(
        torch.accelerator, 'max_memory_allocated'
    ):
      try:
        peak_memory_bytes = torch.accelerator.max_memory_allocated()
        if peak_memory_bytes >= 0:
          return peak_memory_bytes / _BYTES_IN_MB
      except Exception:  # Fallback to XProf if accelerator query fails
        pass
    peak_memory_usage_mb = _get_peak_hbm_memory_mb(session_id, xprof_client)
  elif device in ('xla_cpu', 'jax'):
    peak_memory_usage_mb = _get_peak_hbm_memory_mb(session_id, xprof_client)
  else:
    raise ValueError(f'Unsupported device: {device}')
  return peak_memory_usage_mb


def reset_peak_memory_stats(device):
  if device == 'cuda':
    torch.cuda.memory.reset_max_memory_allocated()
  elif device in ('tpu', 'xla_cuda'):
    if hasattr(torch, 'accelerator'):
      if hasattr(torch.accelerator, 'synchronize'):
        torch.accelerator.synchronize()
      if hasattr(torch.accelerator, 'reset_peak_memory_stats'):
        torch.accelerator.reset_peak_memory_stats()
  elif device in ('cpu', 'xla_cpu', 'jax'):
    pass
  else:
    raise ValueError(f'Unsupported device: {device}')


def _collect_tensors(tensors: Any) -> Sequence[torch.Tensor]:
  leaves, _ = tree_flatten(tensors)
  return [l for l in leaves if isinstance(l, torch.Tensor)]


def synchronize(
    device,
    tensor_to_sync: (
        torch.Tensor | dict[str, torch.Tensor] | Sequence[torch.Tensor] | None
    ) = None,
):
  """Synchronizes the device with the given tensor(s).

  Args:
    device: The device type ("cuda", "tpu", "xla_cuda").
    tensor_to_sync: The tensor(s) to synchronize.
  """
  if device == 'cuda':
    torch.cuda.synchronize()
  elif device == 'cpu':
    pass
  elif device in ('tpu', 'xla_cuda', 'xla_cpu'):
    if tensor_to_sync is None:
      torch.accelerator.synchronize()
    else:
      # TODO(b/507181043): Investigate why sync(None, wait=True) doesn't work.
      tensors = _collect_tensors(tensor_to_sync)
      # TODO(lukeboyer): Flip to global sync after cl/964180694.
      tpu_sync.synchronize(tensors, wait=True)
  else:
    raise ValueError(f'Unsupported device: {device}')


def cache_miss_count(device: str) -> int:
  """Get cache miss count for the specified device. Always returns 0 for CUDA.

  Args:
    device: The device type ('cuda', 'tpu', 'xla_cuda', 'xla_cpu').

  Returns:
    The cache miss count.
  """
  if device in ('cuda', 'cpu'):
    return 0
  elif device in ('tpu', 'xla_cuda', 'xla_cpu'):
    # pylint: disable=protected-access
    return getattr(torch, device)._get_cache_misses()
  else:
    raise ValueError(f'Unsupported device: {device}')


def get_peak_host_compilation_memory_mb(device: str) -> float | None:
  """Get peak host compilation memory for the specified device.

  This value represents the entire process. Technical limitations of the
  underlying infrastructure prevents resetting of this value during eviction.

  Args:
    device: The device type ('cuda', 'tpu', 'xla_cuda').

  Returns:
    The peak host compilation memory in MB. None if not supported by the
    backend, or not available.
  """
  if device != 'tpu':
    return None
  # pylint: disable=protected-access
  perf_stats = getattr(torch, 'tpu')._get_cache_stats()
  if perf_stats.peak_compilation_host_memory_bytes is None:
    return None
  return perf_stats.peak_compilation_host_memory_bytes / _BYTES_IN_MB


def get_peak_host_compilation_memory_mb(device: str) -> float | None:
  """Get peak host compilation memory for the specified device.

  This value represents the entire process. Technical limitations of the
  underlying infrastructure prevents resetting of this value during eviction.

  Args:
    device: The device type ('cuda', 'tpu', 'xla_cuda').

  Returns:
    The peak host compilation memory in MB. None if not supported by the
    backend, or not available.
  """
  if device != 'tpu':
    return None
  # pylint: disable=protected-access
  perf_stats = getattr(torch, 'tpu')._get_cache_stats()
  if perf_stats.peak_compilation_host_memory_bytes is None:
    return None
  return perf_stats.peak_compilation_host_memory_bytes / _BYTES_IN_MB


def torch_compile(
    func: Callable[..., Any],
    device: str,
    dynamic: bool = False,
    fullgraph: bool = False,
) -> Callable[..., Any]:
  """Wraps a callable with `torch.compile` based on the specified device.

  Args:
    func: The callable to be compiled.
    device: The device type ('cuda', 'tpu', 'xla_cuda', 'xla_cpu').
    dynamic: Whether to use dynamic shapes in compilation.
    fullgraph: Whether to require compiling the full graph without breaks.

  Returns:
    The compiled callable.

  Raises:
    ValueError: If the device is not supported.
  """
  if device in ('cuda', 'cpu'):
    func = torch.compile(func, backend='inductor', fullgraph=fullgraph)
  elif device in ('tpu', 'xla_cuda', 'xla_cpu'):
    if dynamic:
      func = torch.compile(
          func,
          # We want to start with static shapes and auto-detect dynamism, so
          # we use dynamic=None instead of dynamic=True, which starts with as
          # much dynamism as possible.
          dynamic=None,
          backend='tpu',
          options={'bounded_dynamism': True},
          fullgraph=fullgraph,
      )
    else:
      func = torch.compile(
          func,
          dynamic=False,
          backend='tpu',
          fullgraph=fullgraph,
      )
  else:
    raise ValueError(f'Unsupported device: {device}')
  return func


def clear_cache(device: str) -> None:
  """Clears the compilation cache for the specified device. No-op for CUDA.

  Args:
    device: The device type ('cuda', 'tpu', 'xla_cuda', 'xla_cpu').

  Raises:
    ValueError: If the device is not supported.
  """
  if device in ('cuda', 'cpu'):
    pass
  elif device in ('tpu', 'xla_cuda', 'xla_cpu'):
    # pylint: disable=protected-access
    getattr(torch, device)._clear_cache()
  else:
    raise ValueError(f'Unsupported device: {device}')
