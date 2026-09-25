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

"""Module to handle autoloading torch_tpu as torch.tpu."""

import functools
import importlib.metadata
import os
import sys
import threading
from typing import Final

from torch_tpu._internal import env_utils

_AUTOLOAD_ENV_VAR: Final[str] = "TORCH_DEVICE_BACKEND_AUTOLOAD"

# Disable autoload while the module loads to prevent circular imports, see:
# https://docs.pytorch.org/tutorials/unstable/python_extension_autoload.html
#
# These two lines run before `import torch` below, so they cannot go through
# `torch_tpu._internal.env`: that pybind11 extension links libtorch, and
# importing it here would trigger the very cycle they exist to break.
_OLD_AUTOLOAD_ENV: Final[str] = os.getenv(  # PY_ENVIRON_OK=pre-torch.
    _AUTOLOAD_ENV_VAR, "1"
)
os.environ[_AUTOLOAD_ENV_VAR] = "0"  # PY_ENVIRON_OK=pre-torch.

# pylint: disable=g-import-not-at-top
import torch
import torch._dynamo.backends.registry as backend_registry
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.device_interface import register_interface_for_device
from torch.utils import _pytree as pytree
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.device import _device_module
from torch_tpu._internal.device import _tpu_backend_config
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.utils import hardware

# pylint: enable=g-import-not-at-top

# Ensure that device is initialized exactly once
_INIT_LOCK: threading.Lock = threading.Lock()

# Make it easier to check whether load() has been called
_LOADED: bool = False

_torch_compile = torch.compile

# Register the TPU backend.
# pylint: disable=protected-access
if not hasattr(torch.backends, "tpu"):
  torch.backends.tpu = _tpu_backend_config._TpuBackendConfig()  # pyrefly: ignore[missing-attribute]

# Enable tracing through recurrent layers (RNN, LSTM, GRU) in Dynamo to avoid
# graph breaks and leverage TorchTPU's fused StableHLO kernels.
if hasattr(torch._dynamo.config, "allow_rnn"):
  torch._dynamo.config.allow_rnn = True


def _get_default_backend_impl(example_inputs) -> backend_registry.CompilerFn:
  """Checks for TPU and returns the appropriate backend function.

  Note: The reason for this auxiliary method rather than just passing the "tpu"
  string in the monkeypatched torch.compile is that we need InitGoogle to be
  called before we call "tpu_device", therefore we defer till call time.

  Args:
    example_inputs: The example inputs provided for the model to execute.

  Returns:
    The appropriate backend function for `torch.compile`.
  """
  flat_inputs, _ = pytree.tree_flatten(example_inputs)

  if hasattr(torch, "tpu"):
    for x in flat_inputs:
      if isinstance(x, torch.Tensor) and x.device.type == "tpu":
        return backend_registry.lookup_backend("tpu")

  # At this point all inputs are in CPU so call the default backend in such a
  # case.
  #
  # Hard-coding the default backend is fine for now as monkey-patching
  # `torch.compile` isn't expected to be a long-term solution. The previous
  # way of using the inspect module wasn't working since torch.compile is an
  # overloaded method.
  # We must specify aot_eager for cpu graphs internally since Inductor+CPU is
  # not supported.
  return backend_registry.lookup_backend(_backend.get_default_cpu_backend())


def _default_backend_selector(graph_module, example_inputs, **kwargs):
  """The backend callable to pass to torch.compile."""
  backend = _get_default_backend_impl(example_inputs)
  return backend(graph_module, example_inputs, **kwargs)


# Monkeypatch torch.compile to default to TPU backend if available.
@functools.wraps(_torch_compile)
def _default_tpu_compile(*args, **kwargs):

  # If an explicit backend is not specified, use DefaultBackendSelector.
  if "backend" not in kwargs:
    kwargs["backend"] = _default_backend_selector
  return _torch_compile(*args, **kwargs)


def _set_libtpu_version() -> None:
  """Passes the installed libtpu version to the C++ runtime."""
  version = ""
  try:
    version = importlib.metadata.version("libtpu")
  except importlib.metadata.PackageNotFoundError:
    pass

  from torch_tpu._internal import env  # pylint: disable=g-import-not-at-top

  env.set_libtpu_version(version)


def _warn_if_unoptimized() -> None:
  """Warns if TorchTPU is running in OSS with an unoptimized build."""
  from torch_tpu._internal import env  # pylint: disable=g-import-not-at-top

  if not env.IS_INTERNAL_TORCH_TPU and not env.TORCH_TPU_IS_OPTIMIZED_BUILD:
    import warnings  # pylint: disable=g-import-not-at-top

    warnings.warn(
        "[UNOPTIMIZED BUILD] TorchTPU was compiled in an unoptimized mode."
        " Performance will be degraded. For production workloads, compile with"
        " '-c opt' (default).",
        RuntimeWarning,
        stacklevel=7,
    )


def _patch_dtensor() -> None:
  """Overrides `from_local` to warn when shape/stride are omitted."""
  from torch_tpu._internal.distributed import from_local  # pylint: disable=g-import-not-at-top

  from_local.patch_from_local()


def _patch_max_pool() -> None:
  """Patches PyTorch max pooling operations with TPU-optimized variants."""
  from torch_tpu._internal.pooling import patch as pooling_patch  # pylint: disable=g-import-not-at-top

  pooling_patch.patch_max_pool()


def _init_device_impl(device: str) -> torch.device:
  """Initializes a lazy pytorch device.

  This will create the privateuse1 backend for the relevant device. The device
  will be a lazy-initialized device, so InitializePjRt will not be called
  until an attribute is accessed on the device.

  The InitializePjRt call can fail, so if you must handle that scenario,
  eagerly access an attrribute on the device after loading.

  Args:
    device: Name of the device. Currently only "tpu" is supported.

  Raises:
    RuntimeError:  if renaming the backend failed

  Returns:
    torch.device   if creating the backend succeeded
  """

  # These imports register things as import-time side effects, so we want to
  # make sure they are imported at load time.
  # pylint: disable=unused-import,g-import-not-at-top
  # For torch.compile() "tpu" backend registration.
  import torch_tpu._internal.compile  # pylint: disable=unused-import  # noqa: F401

  # For monkeypatching torch.autograd.Variable._execution_engine.
  import torch_tpu._internal.sync  # pylint: disable=unused-import  # noqa: F401
  # pylint: enable=unused-import,g-import-not-at-top

  # Only "tpu" is supported.
  assert device == "tpu"

  device_module = _device_module.get_device_module(device)
  device_module._init_runtime_options()  # pylint: disable=protected-access
  # We call `get_interface_for_device` here to force the device interfaces
  # defined upstream (which as of torch 2.13 includes one for TPU) to be
  # initialized and mapped (which happens once), that way if any code somehow
  # manages to call `get_interface_for_device` before we're able to register
  # our own DeviceInterface it won't clobber the one we register below.
  # This will not be needed once we become an in-tree backend in which case we
  # will move our DeviceModule implementation upstream similar to existing ones
  # for CUDA and other backends.
  try:
    get_interface_for_device(device)
  except NotImplementedError:
    pass
  device_interface = _device_module.get_device_interface(device)
  register_interface_for_device(device, device_interface)

  torch.utils.rename_privateuse1_backend(device)
  # Generate `Tensor.is_{device}`, `Tensor.{device}()`, `Module.{device}()`,
  # `PackedSequence.{device}()` attrs so user code can use the same idioms it
  # uses with cuda/xpu/npu (e.g. `if t.is_tpu`, `model.tpu()`). Without this,
  # `device.type == "{device}"` works but the convenience attrs are absent.
  # Storage methods are not generated (default `for_storage=False`).
  torch.utils.generate_methods_for_privateuse1_backend()
  device_d = torch.device(device)
  if device_d is None:
    raise RuntimeError("Failed to set privateuse1_backend in torch")
  print(
      f"Successfully renamed PrivateUse1 backend to '{device}'. "
      f"Device: {device_d!r}",
      file=sys.stderr,
  )

  # pylint: disable=protected-access
  torch._register_device_module(device, device_module)  # pyrefly: ignore[bad-argument-type]
  print(f"Registered Python module for '{device}'.", file=sys.stderr)

  print(
      f"Device type: {device_d.type}, Device index:"
      f" {device_d.index if device_d.index is not None else 'default'}"
  )

  if device == "tpu" and not torch.distributed.is_initialized():
    # Register the TPU distributed runtime; users will also need to
    # init_process_group() in their code. Registered unconditionally
    # (not gated on multi-host) so single-host runs that still go
    # through init_process_group(backend="tpu_dist") work.
    #
    # Pass `devices` as a list (not the bare string "tpu") to work around a
    # register_backend string-handling bug fixed upstream in
    # pytorch/pytorch#187960; the list form is correct regardless.
    print("Initializing TPU distributed runtime")
    from torch_tpu._internal.distributed import torchcomm_tpu  # pylint: disable=g-import-not-at-top

    torch.distributed.Backend.register_backend(
        "tpu_dist", tpu_distributed.create_process_group, devices=["tpu"]
    )
    torch.distributed.Backend.register_backend(
        "tpu", tpu_distributed.create_process_group, devices=["tpu"]
    )
    torchcomm_tpu.register_torchcomms_tpu()

  # Register the Kineto backend using the internal C++ lifecycle module.
  from torch_tpu._internal import profiler  # pylint: disable=g-import-not-at-top

  profiler.register_kineto_backend()

  # Expose the public profiler module under 'torch.tpu.profiler' in sys.modules
  # and attach it to the 'torch.tpu' device module for attribute access.
  from torch_tpu._internal.profiler import api as profiler_api  # pylint: disable=g-import-not-at-top

  sys.modules["torch.tpu.profiler"] = profiler_api
  setattr(device_module, "profiler", profiler_api)

  # Expose the public pallas module under 'torch.tpu.pallas' in sys.modules
  # and attach it to the 'torch.tpu' device module for attribute access.
  from torch_tpu._internal.pallas import api as pallas_api  # pylint: disable=g-import-not-at-top

  sys.modules["torch.tpu.pallas"] = pallas_api
  setattr(device_module, "pallas", pallas_api)

  # Pass installed libtpu version to C++ runtime if available.
  _set_libtpu_version()

  # Warn if running with an unoptimized build.
  _warn_if_unoptimized()

  # Override DTensor.from_local to warn when shape/stride are omitted.
  _patch_dtensor()

  # Patch max pooling operations with TPU-optimized variants.
  _patch_max_pool()

  # Monkey patch torch.set_float32_matmul_precision and
  # torch.get_float32_matmul_precision to maintain global precision state.
  from torch_tpu._internal.precision import precision_impl  # pylint: disable=g-import-not-at-top

  @functools.wraps(torch.set_float32_matmul_precision)
  def _tpu_set_precision(precision: str) -> None:
    match precision:
      case "medium":
        precision_impl._set_global_precision(precision_impl.Precision.DEFAULT)
      case "high":
        precision_impl._set_global_precision(precision_impl.Precision.HIGH)
      case "highest":
        precision_impl._set_global_precision(precision_impl.Precision.HIGHEST)
      case _:
        raise ValueError(
            "expected to be one of 'highest', 'high', or 'medium', got"
            f" '{precision}'"
        )

  @functools.wraps(torch.get_float32_matmul_precision)
  def _tpu_get_precision() -> str:
    global_precision = precision_impl._get_global_precision()
    match global_precision:
      case precision_impl.Precision.DEFAULT:
        return "medium"
      case precision_impl.Precision.HIGH:
        return "high"
      case precision_impl.Precision.HIGHEST:
        return "highest"
      case _:
        raise ValueError(
            "expected to be one of 'highest', 'high', or 'medium', got"
            f" {global_precision}"
        )

  torch.set_float32_matmul_precision = _tpu_set_precision
  torch.get_float32_matmul_precision = _tpu_get_precision

  # Monkey patch torch.compile until there is an official way to override the
  # backend: https://github.com/pytorch/pytorch/issues/178930
  # TODO(b/492505722): Update internal usage of torch.compile.
  torch.compile = _default_tpu_compile

  return device_d


@functools.cache
def _init_device(device: str) -> torch.device:
  with _INIT_LOCK:
    if hasattr(torch, device):
      return torch.device(device)
    return _init_device_impl(device)


def load() -> None:
  """Loads the torch_tpu backend.

  If the backend has been loaded, the `_LOADED` global variable will be set to
  `True`.
  """
  # On autoload restore the original autoload value
  env_utils.set_str_env(_AUTOLOAD_ENV_VAR, _OLD_AUTOLOAD_ENV)

  global _LOADED
  if hardware.get_tpu_device_count():
    _init_device("tpu")  # INIT_DEVICE_OK=root usage
    _LOADED = True

  # Start the eager-mode trace daemon if TORCH_TRACE was set at process
  # startup. No-op when TORCH_TRACE is unset.
  from torch_tpu._internal import tracing  # pylint: disable=g-import-not-at-top

  tracing.enable_if_requested()
