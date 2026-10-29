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

"""Public API for torch_tpu._internal.compile."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import torch
from torch_tpu._internal.compile import _nested_compile_region_registration
from torch_tpu._internal.compile import _scan_registration
from torch_tpu._internal.compile import flex_attention_redirect
from torch_tpu._internal.compile import grouped_mm_meta
from torch_tpu._internal.compile import meta_registration_overrides

if TYPE_CHECKING:
  from torch._dynamo.device_interface import DeviceInterface
  from torch_tpu._internal.compile._backend import TpuBackend

_backend_lock = threading.RLock()
_backend_instance: TpuBackend | None = None
_compile_backend_initialized: bool = False

_BACKEND_REEXPORTS = frozenset({
    "AsyncCompilationSubmitted",
    "TpuBackend",
    "async_compile",
    "clear_cache",
    "resolve_compilations",
})


def _initialize_graphsafe_rng() -> None:
  """Initializes graphsafe RNG operations for TPU."""
  # pylint: disable=g-import-not-at-top
  from torch._functorch._aot_autograd import utils as aot_utils
  # pylint: enable=g-import-not-at-top

  if hasattr(aot_utils, "register_graphsafe_rng_device_type"):
    aot_utils.register_graphsafe_rng_device_type("privateuseone")
    # Add "tpu" to the supported device types for graphsafe RNG, as torch_tpu
    # renames "privateuseone" to "tpu".
    # pylint: disable=protected-access
    aot_utils._GRAPHSAFE_RNG_DEVICE_TYPES.add("tpu")

    torch._functorch.config.graphsafe_rng_functionalization = True
    torch._functorch.config.functionalize_rng_ops = False
    # pylint: enable=protected-access


def _configure_dynamo_flags() -> None:
  """Enables TPU-specific TorchDynamo tracing flags."""
  # New Dynamo flag that enables tracing through the `backward` function call.
  # This enables the possibility of generating of a single fx graph
  # containing both forward and backward subgraphs, rather than a discrete fx
  # graph for each. See b/480979694 for details.
  # pylint: disable=protected-access
  if hasattr(torch._dynamo.config, "trace_autograd_ops"):
    torch._dynamo.config.trace_autograd_ops = True
  # Disables PyTorch Dynamo's default graph break on RNN, LSTM, and GRU modules,
  # allowing Dynamo to trace recurrent layers into aten::lstm and aten::gru
  # nodes for lowering to TorchTPU's StableHLO kernels.
  if hasattr(torch._dynamo.config, "allow_rnn"):
    torch._dynamo.config.allow_rnn = True
  # pylint: enable=protected-access


def _register_dynamo_device_interface() -> None:
  """Registers the TPU DeviceInterface in TorchDynamo."""
  # pylint: disable=g-import-not-at-top
  from torch._dynamo.device_interface import get_interface_for_device
  from torch._dynamo.device_interface import register_interface_for_device
  from torch_tpu._internal.device import _device_interface
  # pylint: enable=g-import-not-at-top

  # We call `get_interface_for_device` here to force the device interfaces
  # defined upstream (which as of torch 2.13 includes one for TPU) to be
  # initialized and mapped (which happens once), that way if any code somehow
  # manages to call `get_interface_for_device` before we're able to register
  # our own DeviceInterface it won't clobber the one we register below.
  # This will not be needed once we become an in-tree backend in which case we
  # will move our DeviceModule implementation upstream similar to existing
  # ones for CUDA and other backends.
  try:
    get_interface_for_device("tpu")
  except NotImplementedError:
    pass
  device_interface: type[DeviceInterface] = (
      _device_interface.get_device_interface("tpu")
  )
  register_interface_for_device("tpu", device_interface)


def ensure_initialized() -> None:
  """Idempotently initializes compiler backend flags and runtime hooks."""
  global _backend_instance, _compile_backend_initialized
  with _backend_lock:
    if _compile_backend_initialized:
      return
    _initialize_graphsafe_rng()
    _configure_dynamo_flags()
    _register_dynamo_device_interface()
    # pylint: disable=g-import-not-at-top
    from torch_tpu._internal.compile import _backend
    from torch_tpu._internal.compile import nested_compile_region  # noqa: F401
    from torch_tpu._internal.compile import scan  # noqa: F401
    # pylint: enable=g-import-not-at-top
    _backend_instance = _backend.TpuBackend()
    _compile_backend_initialized = True


def register_with_dynamo() -> None:
  """Registers the TPU compiler backend with TorchDynamo for fallback envs."""
  ensure_initialized()
  # pylint: disable=g-import-not-at-top
  from torch._dynamo.backends import registry
  # pylint: enable=g-import-not-at-top
  # TorchDynamo 2.11 has no public "is_registered" API to test if a backend
  # name exists without attempting discovery, so check the private
  # _COMPILER_FNS dict directly to avoid duplicate registration assertions.
  if "tpu" not in registry._COMPILER_FNS:
    registry.register_backend(
        compiler_fn=_backend_instance,
        name="tpu",
    )  # pyrefly: ignore[bad-argument-type]


def _get_backend() -> TpuBackend:
  """Entry-point target and attribute loader for the TPU backend.

  Initializes compiler hooks without registering into TorchDynamo directly,
  allowing `torch._dynamo.backends.registry.lookup_backend('tpu')` to register
  the returned backend callable itself.
  """
  ensure_initialized()
  assert _backend_instance is not None
  return _backend_instance


def __getattr__(name: str) -> Any:
  if name == "backend":
    b = _get_backend()
    globals()["backend"] = b
    return b

  if name in _BACKEND_REEXPORTS:
    # pylint: disable=g-import-not-at-top
    from torch_tpu._internal.compile import _backend
    # pylint: enable=g-import-not-at-top
    globals()["AsyncCompilationSubmitted"] = _backend.AsyncCompilationSubmitted
    globals()["TpuBackend"] = _backend.TpuBackend
    globals()["async_compile"] = _backend.async_compile
    globals()["clear_cache"] = _backend.clear_cache
    globals()["resolve_compilations"] = _backend.resolve_compilations
    return getattr(_backend, name)

  if name == "TpuCompileDebug":
    # pylint: disable=g-import-not-at-top
    from torch_tpu._internal.compile.debug import TpuCompileDebug
    # pylint: enable=g-import-not-at-top
    globals()["TpuCompileDebug"] = TpuCompileDebug
    return TpuCompileDebug

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Eager-relevant patches and higher-order op registrations required on TPU
# host initialization.
flex_attention_redirect.patch()
grouped_mm_meta.register()
meta_registration_overrides.register()
_scan_registration.register()
_nested_compile_region_registration.register()

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "AsyncCompilationSubmitted",
    "TpuBackend",
    "TpuCompileDebug",
    "async_compile",
    "backend",
    "clear_cache",
    "ensure_initialized",
    "register_with_dynamo",
    "resolve_compilations",
    # go/keep-sorted end
]
