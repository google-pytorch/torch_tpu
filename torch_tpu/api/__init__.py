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

from collections.abc import Callable
import functools
import inspect
import os
import sys
import typing
import torch
from torch._dynamo.backends import inductor
import torch._dynamo.backends.registry as backend_registry

# TODO(pganssle): Migrate to PEP 695 syntax when PyTorch is 3.12+ only
_RV = typing.TypeVar("_RV", bound=torch.device | None)


def _ensure_loaded(
    f: Callable[[], _RV],
) -> Callable[[], _RV]:
  """Ensure that the torch_tpu backend is loaded before returning a device.

  Args:
    f: The decorated function

  Returns:
      Returns a wrapped version of the decorated function that makes sure that
      the TPU backend has been loaded before the function is called.
  """

  @functools.wraps(f)
  def _device_func() -> _RV:
    # pylint: disable=g-import-not-at-top
    from torch_tpu import _loader as loader

    # pylint: disable=protected-access
    if not loader._LOADED:
      # Set allow_xla_backend because unlike the automatic loading case, the
      # *_device() calls are explicitly requesting one of the backends. It is
      # possible that someone may call `torch.device("xla_cuda")` on a CPU
      # device, causing the `xla_cpu` backend to be loaded, but this will be
      # followed immediately by an error, and this case will be rare, so we
      # do not have to overly concern ourselves with this.
      loader.load(allow_xla_backend=True)
    return f()

  return _device_func


# TODO(b/432530222): Remove these functions now that `torch.device("tpu")` works
@functools.lru_cache(maxsize=1)
@_ensure_loaded
def tpu_device() -> torch.device:
  """Common wrapper function to ensure execution on device."""
  return torch.device("tpu")


@functools.lru_cache(maxsize=1)
@_ensure_loaded
def _xla_cuda_device() -> torch.device:
  """Common wrapper function to ensure execution on device."""
  return torch.device("xla_cuda")


@functools.lru_cache(maxsize=1)
@_ensure_loaded
def _xla_cpu_device() -> torch.device:
  """Common wrapper function to ensure execution on device."""
  return torch.device("xla_cpu")


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
