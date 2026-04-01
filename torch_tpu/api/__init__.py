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

"""Core API for TorchTPU."""

import functools
import inspect
import os
import sys
import threading
from typing import Any, Callable

import torch
from torch._dynamo.backends import inductor
import torch._dynamo.backends.registry as backend_registry
import torch.distributed
# For monkeypatching torch.autograd.Variable._execution_engine.
import torch_tpu._internal.compile  # pylint: disable=unused-import
import torch_tpu._internal.sync  # pylint: disable=unused-import
from torch_tpu.api import _device_module

try:
  from torch_tpu._internal.distributed import tpu_distributed  # pylint: disable=g-import-not-at-top

  _TPU_DISTRIBUTED_AVAILABLE = True
except ImportError:
  _TPU_DISTRIBUTED_AVAILABLE = False

_INIT_LOCK = threading.Lock()


def _init_device_impl(device: str) -> torch.device:
  """Initializes the runtime for a specific device.

  If the underlying InitializePjRt call fails, _init_tpu_runtime() will
  throw a runtime error exception.

  This function is idempotent, the underlying _init_runtime checkes
  whether InitializePjRt has been called already.

  Args:
    device: Name of the device. Currently "tpu" and "xla_cuda" are supported.

  Raises:
    RuntimeError:  if initialization of PjRt runtime failed.

  Returns:
    torch.device   if initialization of PjRt runtime succeeded.
  """

  # Only "tpu / xla_cuda / xla_cpu" are supported.
  assert device == "tpu" or device == "xla_cuda" or device == "xla_cpu"

  torch.utils.rename_privateuse1_backend(device)
  device_d = torch.device(device)
  if device_d is None:
    raise RuntimeError("Failed to set privateuse1_backend in torch")
  print(
      f"Successfully renamed PrivateUse1 backend to '{device}'. "
      f"Device: {device_d}",
      file=sys.stderr,
  )

  # pylint: disable=protected-access
  torch._register_device_module(device, _device_module._DeviceModule)
  print(f"Registered Python module for '{device}'.", file=sys.stderr)

  # Call InitializePjRt runtime or raise a runtime error.
  getattr(torch, device)._init_runtime(device_type=device)
  print(
      f"Device type: {device_d.type}, Device index:"
      f" {device_d.index if device_d.index is not None else 'default'}"
  )

  if (
      device == "tpu"
      and not torch.distributed.is_initialized()
      and _TPU_DISTRIBUTED_AVAILABLE
      and len(tpu_distributed.all_global_device_ids()) > 1
  ):
    # Looks like we are running in a distributed setup.
    # Register the TPU distributed runtime; users will also need to
    # init_process_group() in their code.
    print("Initializing TPU distributed runtime")
    torch.distributed.Backend.register_backend(
        "tpu_dist", tpu_distributed.create_process_group, devices="tpu"
    )
  return device_d


def _init_device(device: str) -> torch.device:
  # Acquire a lock to ensure that initialization happens exactly once.
  with _INIT_LOCK:
    if hasattr(torch, device):
      return torch.device(device)
    else:
      return _init_device_impl(device)


# TODO: Remove these functions. Make `torch.device("tpu:)` just work.
@functools.lru_cache(maxsize=1)
def tpu_device() -> torch.device:
  """Common wrapper function to ensure execution on device."""
  return _init_device("tpu")


@functools.lru_cache(maxsize=1)
def _xla_cuda_device() -> torch.device:
  """Common wrapper function to ensure execution on device."""
  return _init_device("xla_cuda")


@functools.lru_cache(maxsize=1)
def _xla_cpu_device() -> torch.device:
  """Common wrapper function to ensure execution on device."""
  return _init_device("xla_cpu")


_torch_compile = torch.compile


def _get_default_backend_impl() -> backend_registry.CompilerFn:
  """Checks for TPU and returns the appropriate backend function.

  Note: The reason for this auxiliary method rather than just passing the "tpu"
  string in the monkeypatched torch.compile is that we need InitGoogle to be
  called before we call "tpu_device", therefore we defer till call time.

  Returns:
    The appropriate backend function for `torch.compile`.
  """
  try:
    # tpu_device() will raise a RuntimeError if a TPU is not available.
    tpu_device()
    return backend_registry.lookup_backend("tpu")
  except RuntimeError:
    original_signature = inspect.signature(_torch_compile)
    default_backend = original_signature.parameters["backend"].default
    return backend_registry.lookup_backend(default_backend)


def _default_backend_selector(graph_module, example_inputs, **kwargs):
  """The backend callable to pass to torch.compile."""
  backend = _get_default_backend_impl()
  return backend(graph_module, example_inputs, **kwargs)


# Monkeypatch torch.compile to default to TPU backend if available.
@functools.wraps(_torch_compile)
def _default_tpu_compile(*args, **kwargs):

  # If an explicit backend is not specified, use DefaultBackendSelector.
  if "backend" not in kwargs:
    kwargs["backend"] = _default_backend_selector
  return _torch_compile(*args, **kwargs)


# TODO(b/492505722): Update internal usage of torch.compile.
torch.compile = _default_tpu_compile

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "_xla_cpu_device",
    "_xla_cuda_device",
    "tpu_device",
    # go/keep-sorted end
]
