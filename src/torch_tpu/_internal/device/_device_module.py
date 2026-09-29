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

"""Device Python Module.

This module provides a Python interface to initialize and shutdown the device
and runtime.
"""

import abc
import atexit
from collections.abc import Mapping
import dataclasses
import sys
from typing import Any, Final, List
import torch
from torch._dynamo.device_interface import DeviceInterface
from torch_tpu._internal import execution_mode
from torch_tpu._internal.device import _device_ops_backend
import torch_tpu._internal.precision as _precision_module
from torch_tpu._internal.stream import streams
from torch_tpu._internal.utils import annotations
from torch_tpu._internal.utils import hardware

experimental = annotations.experimental

# TODO: add more dtypes here. Initial list was chosen arbitrarily.
_AMP_SUPPORTED_DTYPES = (
    # go/keep-sorted start
    torch.bfloat16,
    torch.bool,
    torch.complex128,
    torch.complex64,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.uint8,
    # go/keep-sorted end
)


class _DefaultGeneratorsProperty:
  """Descriptor that acts exactly like a tuple of default generators.

  Mirrors the type and lazy initialization behavior of PyTorch's native
  `torch.cuda.default_generators`, while dynamically reflecting export
  overrides.

  Attributes:
    _cached_default_generators: A lazily-initialized tuple holding the default
      `torch.Generator` instances for each TPU device. It contains exactly
      `owner.device_count()` elements, ordered by device ordinal (i.e. element
      at index `i` corresponds to device index `i`). Crossing the pybind11
      boundary creates new Python wrapper objects on each call to
      `_device_ops_backend.get_default_generator()`, so caching this tuple
      ensures identity persistence (`torch.tpu.default_generators is
      torch.tpu.default_generators`) matching upstream PyTorch semantics.
  """

  def __init__(self):
    self._cached_default_generators: tuple[torch.Generator, ...] | None = None

  def __get__(self, instance, owner):
    if owner is None:
      return ()
    _device_ops_backend._get_current_device_id()  # pylint: disable=protected-access
    if (
        execution_mode.eager_mode
        == execution_mode.EagerMode.INTERNAL_COMPILE_FX_GRAPH
    ):
      generators = []
      for i in range(owner.device_count()):
        try:
          generators.append(_device_ops_backend.get_default_generator(i))
        except RuntimeError:
          generators.append(_device_ops_backend.get_eager_default_generator(i))
      return tuple(generators)

    if self._cached_default_generators is None:
      self._cached_default_generators = tuple(
          _device_ops_backend.get_default_generator(i)
          for i in range(owner.device_count())
      )
    return self._cached_default_generators


# __tt_api_stages__ maps attribute names to ApiStageInfo stage metadata.
#
# Tracks lifecycle stage annotations for module attributes on `torch.tpu`. Each
# entry maps an attribute name to an `annotations.ApiStageInfo` containing
# `stage`, `reason`, optional `value`, and optional `version` for deprecated
# attributes.
#
# - Attributes present in __tt_api_stages__ emit stage warnings on access.
# - Attributes physically defined on the class but absent from this map
#   are considered STABLE (unannotated).
# - Attributes NOT defined on the class raise AttributeError in __getattr__ on
#   access.
__tt_api_stages__: dict[str, annotations.ApiStageInfo] = {
    "default_generators": annotations.ApiStageInfo(
        stage=annotations.Stage.EXPERIMENTAL,
        reason="Default generators for TPU devices.",
        value=_DefaultGeneratorsProperty(),
    ),
    "Precision": annotations.ApiStageInfo(
        stage=annotations.Stage.EXPERIMENTAL,
        reason="StableHLO precision configuration.",
        value=_precision_module.Precision,
    ),
}


def _rng_validate_device_index(
    device: int | str | torch.device, device_idx: int
) -> None:
  """Validates that the provided device is this process's TPU.

  Args:
    device: The device to validate. Can be an int (device index), "tpu", or a
      `torch.device("tpu")` object.
    device_idx: The index of the current process's TPU device.

  Raises:
    ValueError: If the device is not a TPU device or if the index does not
      match the current device index.
  """
  provided_str, provided_idx = None, None
  if isinstance(device, str):
    provided_str = device
  elif isinstance(device, torch.device):
    provided_str = device.type
    provided_idx = device.index
  elif isinstance(device, int):
    provided_idx = device
  else:
    raise TypeError(f"got unrecognized device type: {device}")

  if provided_str is not None and provided_str != "tpu":
    raise ValueError(
        f"expected device type 'tpu' for RNG state access, got '{provided_str}'"
    )
  if provided_idx is not None and provided_idx != device_idx:
    raise ValueError(
        f"expected local device index {device_idx}, got {provided_idx}:"
        " accessing RNG state of a non-current TPU device is not supported"
    )


# Copied from torch/cuda/_utils.py
def _get_device_index(
    device_arg: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
  """Get the device index from device, which can be a torch.device, int, str, or None."""
  if isinstance(device_arg, int):
    return device_arg
  if isinstance(device_arg, str):
    device_arg = torch.device(device_arg)
  if isinstance(device_arg, torch.device):
    if allow_cpu:
      if device_arg.type not in ["tpu", "cpu"]:
        raise ValueError(f"Expected a tpu or cpu device, but got: {device_arg}")
    elif device_arg.type != "tpu":
      raise ValueError(f"Expected a tpu device, but got: {device_arg}")
  if not torch.jit.is_scripting():
    if isinstance(device_arg, _DeviceContext):
      return device_arg.idx
  return torch._utils._get_device_index(  # pylint: disable=protected-access
      device_arg, optional=optional, allow_cpu=allow_cpu
  )


@experimental(
    "device() context manager is experimental and subject to change without"
    " notice."
)
class _DeviceContext:
  r"""Context-manager that changes the selected device.

  .. warning::
      This API is experimental and subject to change in future releases.

  Args:
    device: device index to select. It's a no-op if this argument is a negative
      integer or ``None``.
  """

  def __init__(self, device: Any):  # pylint: disable=redefined-outer-name
    self.idx = _get_device_index(device, optional=True)
    self.prev_idx = -1

  def __enter__(self):
    self.prev_idx = _DeviceModule._exchange_device(self.idx)  # pylint: disable=protected-access

  def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any):
    self.idx = _DeviceModule._maybe_exchange_device(self.prev_idx)  # pylint: disable=protected-access
    return False


@experimental(
    "device_of() context manager is experimental and subject to change without"
    " notice."
)
class _DeviceOfContext(_DeviceContext):
  r"""Context-manager that changes the current device to that of given object.

  .. warning::
      This API is experimental and subject to change in future releases.

  You can use both tensors and storages as arguments. If a given object is
  not allocated on a TPU, this is a no-op.

  Args:
    obj: object allocated on the selected device.
  """

  def __init__(self, obj: Any):
    idx = -1
    if getattr(obj, "is_tpu", False):
      idx = obj.get_device()
    elif getattr(getattr(obj, "device", None), "type", None) == "tpu":
      idx = obj.get_device()
    super().__init__(idx)


class _DeviceModuleMeta(abc.ABCMeta):
  """Metaclass for _DeviceModule to intercept class-level attribute access.

  Handles dynamic attribute resolution for future deprecated or fallback
  attributes on `torch.tpu` registered in `__tt_api_stages__`.
  """

  @property
  @annotations.experimental(
      "allow_excess_precision is experimental and may change or be removed "
      "without notice."
  )
  def allow_excess_precision(cls) -> bool:
    """Whether XLA is allowed to use excess precision.

    This property reflects the effective setting used by the XLA compiler even
    when not explicitly set. First time calling this property triggers parsing
    and memoization of the XLA_FLAGS environment variable.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    getter = getattr(_device_ops_backend, "_get_allow_excess_precision", None)
    if getter is None:
      raise RuntimeError("TorchTPU device ops backend is not initialized.")
    return getter()

  @allow_excess_precision.setter
  @annotations.experimental(
      "allow_excess_precision is experimental and may change or be removed "
      "without notice."
  )
  def allow_excess_precision(cls, value: bool):
    """Sets whether XLA is allowed to use excess precision.

    It sets the XLA flag of --allow_excess_precision globally for all future
    compilations. Changing its value frequently is not recommended as it will
    trigger recompilation for the new setting.
    File a feature request for a context manager for this flag if needed.

    The flag defaults to False if not set in the XLA_FLAGS environment variable
    nor explicitly set by this property.

    .. warning::
        This API is experimental and subject to change in future releases.

    Args:
      value: Whether to allow excess precision. Can be True or False.
    """
    setter = getattr(_device_ops_backend, "_set_allow_excess_precision", None)
    if setter is None:
      raise RuntimeError("TorchTPU device ops backend is not initialized.")
    setter(value)

  @property
  @annotations.experimental(
      "window_num_stripes is experimental and may change or be removed "
      "without notice."
  )
  def window_num_stripes(cls) -> int:
    """Maximum concurrent DMA stripes for WindowTpu peer transfers."""
    getter = getattr(_device_ops_backend, "_get_window_num_stripes", None)
    if getter is None:
      raise RuntimeError("TorchTPU device ops backend is not initialized.")
    return getter()

  @window_num_stripes.setter
  @annotations.experimental(
      "window_num_stripes is experimental and may change or be removed "
      "without notice."
  )
  def window_num_stripes(cls, value: int):
    """Sets maximum concurrent DMA stripes for WindowTpu peer transfers."""
    setter = getattr(_device_ops_backend, "_set_window_num_stripes", None)
    if setter is None:
      raise RuntimeError("TorchTPU device ops backend is not initialized.")
    setter(value)

  @property
  @annotations.experimental(
      "window_stripe_chunk_mb is experimental and may change or be removed "
      "without notice."
  )
  def window_stripe_chunk_mb(cls) -> int:
    """Minimum payload size in MB per parallel DMA stripe for WindowTpu."""
    getter = getattr(_device_ops_backend, "_get_window_stripe_chunk_mb", None)
    if getter is None:
      raise RuntimeError("TorchTPU device ops backend is not initialized.")
    return getter()

  @window_stripe_chunk_mb.setter
  @annotations.experimental(
      "window_stripe_chunk_mb is experimental and may change or be removed "
      "without notice."
  )
  def window_stripe_chunk_mb(cls, value: int):
    """Sets minimum payload size in MB per parallel DMA stripe for WindowTpu."""
    setter = getattr(_device_ops_backend, "_set_window_stripe_chunk_mb", None)
    if setter is None:
      raise RuntimeError("TorchTPU device ops backend is not initialized.")
    setter(value)

  def __getattr__(cls, name: str) -> Any:
    stages_map = getattr(cls, "__tt_api_stages__", {})
    # tt_api_globals is None because resolved attributes are cached directly on
    # `cls` via `type.__setattr__` below, bypassing __getattr__ on 2nd access.
    val = annotations._resolve_module_attribute(
        stages_map, name, "torch.tpu", tt_api_globals=None
    )
    # Install the resolved attribute or descriptor directly on `cls`, and return
    # via getattr so Python's native descriptor protocol handles binding.
    type.__setattr__(cls, name, val)
    return getattr(cls, name)


class _DeviceModule(abc.ABC, metaclass=_DeviceModuleMeta):
  """torch_tpu device equivalent to functions in "torch/cuda/__init__.py".

  https://github.com/pytorch/pytorch/blob/v2.7.0/torch/cuda/__init__.py

  For Automatic Mixed Precision (AMP), see:
  https://docs.pytorch.org/docs/stable/amp.html

  Since methods of this class are accessible via getattr(torch, "tpu"), we need
  to make sure they are not accidentally exposed to the public API by adding a
  leading underscore to the names of internal methods.
  """

  # Expose module-level __tt_api_stages__ map as a class attribute so
  # _DeviceModuleMeta.__getattr__ can access stage metadata on the class.
  __tt_api_stages__ = __tt_api_stages__

  # device_count and current_device are None until device_count() and
  # current_device() are called.
  _device_count: int | None = None
  _current_device: int | None = None
  _autocast_enabled: bool = False
  _autocast_dtype: torch.dtype | None = torch.bfloat16

  _device_type: str

  device = _DeviceContext  # pylint: disable=invalid-name
  device_of = _DeviceOfContext  # pylint: disable=invalid-name
  precision = _precision_module.precision

  # This method is called when a subclass of this abstract base class is
  # created. As of Python 3.15, there is no obviously supported way of declaring
  # a required abstract class attribute, but we can do the validation here.
  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)

    if not hasattr(cls, "_device_type"):
      raise TypeError(f"Class '{cls.__name__}' must define _device_type")

  @classmethod
  @experimental(
      "current_device() is experimental and subject to change without notice."
  )
  def current_device(cls) -> int:  # This is in torch/cuda/__init__.py.
    """Returns the index of the currently selected device.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    if cls._current_device is None:
      cls._current_device = _device_ops_backend._get_current_device_id()  # pylint: disable=protected-access

    assert (
        cls._current_device is not None
    ), "_DeviceModule is initialized but has no current device."
    return cls._current_device

  @classmethod
  def _get_generator(cls, device: torch.device) -> torch.Generator:
    """Returns the default generator for the given device.

    This is required for graphsafe RNG operations in AOTAutograd.

    Args:
      device: The torch.device object.

    Returns:
      The default torch.Generator for the specified device.
    """
    idx = device.index if device.index is not None else cls.current_device()
    return cls.default_generators[idx]

  @classmethod
  @experimental(
      "set_device() is experimental and subject to change without notice."
  )
  def set_device(
      cls, device: int | str | torch.device
  ) -> None:  # This is in torch/cuda/__init__.py.
    """Sets the current device.

    .. warning::
        This API is experimental and subject to change in future releases.

    For TPUs, this method primarily validates that the given device index
    matches the device index assigned to the current process. Unlike CUDA,
    switching the active device within a single process is not supported.

    Args:
      device: The device to set. Can be an int (device index), "tpu", or a
        `torch.device("tpu")` object.

    Raises:
      ValueError: If the provided device index does not match the current
        device index.
      TypeError: If the device type is not recognized.
    """
    current_idx = cls.current_device()
    provided_idx = None
    if isinstance(device, str):
      if device != "tpu":
        raise ValueError(f"Invalid device string {device}")
      provided_idx = current_idx  # Assume current device if only "tpu" is given
    elif isinstance(device, torch.device):
      if device.type != "tpu":
        raise ValueError(f"Invalid device type {device.type}")
      provided_idx = device.index if device.index is not None else current_idx
    elif isinstance(device, int):
      provided_idx = device
    else:
      raise TypeError(f"Got unrecognized device type, {device}")

    if provided_idx != current_idx:
      raise ValueError(
          f"Cannot set TPU device to index {provided_idx}, current process is "
          f"bound to device index {current_idx}. Changing the active TPU "
          "device within a process is not supported."
      )
    # No actual device switching occurs, so we just return.
    return

  @classmethod
  def _exchange_device(cls, device: int) -> int:
    if device < 0:
      return -1
    prev_device = cls.current_device()
    cls.set_device(device)
    return prev_device

  @classmethod
  def _maybe_exchange_device(cls, device: int) -> int:
    if device < 0:
      return -1
    prev_device = cls.current_device()
    cls.set_device(device)
    return prev_device

  @classmethod
  @experimental(
      "is_available() is experimental and subject to change without notice."
  )
  def is_available(cls) -> bool:  # This is in torch/cuda/__init__.py.
    """Returns if the TPU backend is currently available.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    if cls._device_type == "tpu":
      return cls.device_count() > 0
    return False

  @classmethod
  def _is_triton_capable(cls, device: torch.types.Device = None) -> bool:
    """Returns if the device has Triton support.

    Args:
      device: The device to check. Unused for custom backends unless explicitly
        implemented.

    Returns:
      False, as custom backends do not support Triton by default.
    """
    del device  # Unused
    return False

  @classmethod
  def _is_gpu(cls) -> bool:
    """Returns True if Inductor should treat this device as a GPU-class accelerator.

    Returns:
      False, as custom backends are conservatively treated as non-GPU until
      they explicitly opt in.
    """
    return False

  @classmethod
  def _get_multi_processor_count(cls, device: torch.types.Device = None) -> int:
    """Returns the number of compute units for the device."""
    get_props = getattr(cls, "get_device_properties", None)
    if get_props is not None:
      try:
        props = get_props(device)
        return getattr(props, "multi_processor_count", 1)
      except Exception:  # pylint: disable=broad-except
        return 1
    return 1

  @classmethod
  @experimental(
      "is_initialized() is experimental and subject to change without notice."
  )
  def is_initialized(cls) -> bool:  # This is in torch/cuda/__init__.py.
    """Returns whether PyTorch's TPU state has been initialized.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    return not cls._is_in_bad_fork() and _device_ops_backend._is_initialized()  # pylint: disable=protected-access

  @classmethod
  def _lazy_init(cls) -> None:
    """Initialization method to satisfy PyTorch's PrivateUse1 requirements."""
    if cls._is_in_bad_fork():
      raise RuntimeError(
          "cannot access or re-initialize TPU in a forked subprocess. "
          "To use TPU with multiprocessing, you must use the 'spawn' start"
          " method"
      )
    _device_ops_backend._register_fork_handler()  # pylint: disable=protected-access

  @classmethod
  def _is_in_bad_fork(cls) -> bool:
    """Returns whether the current process is in an unsafe forked child state."""
    return _device_ops_backend._is_in_bad_fork()  # pylint: disable=protected-access

  @classmethod
  @experimental(
      "manual_seed() is experimental and may change or be removed without"
      " notice."
  )
  def manual_seed(cls, seed: int) -> None:
    """Sets the seed for generating random numbers on the current device.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    _device_ops_backend.manual_seed(seed)

  @classmethod
  @experimental(
      "manual_seed_all() is experimental and may change or be removed without"
      " notice."
  )
  def manual_seed_all(cls, seed: int) -> None:
    """Sets the seed for generating random numbers on all devices.

    .. warning::
        This API is experimental and subject to change in future releases.

    Args:
      seed: The desired seed.
    """
    _device_ops_backend.manual_seed_all(seed)

  @classmethod
  @experimental(
      "initial_seed() is experimental and may change or be removed without"
      " notice."
  )
  def initial_seed(cls) -> int:
    """Returns the current random seed of the current TPU device.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    idx = cls.current_device()
    return cls.default_generators[idx].initial_seed()

  @classmethod
  def _get_local_device_attributes(cls) -> Mapping[str, Any]:
    """Returns attributes of the local PJRT device, initializing options if needed."""
    cls._init_runtime_options()
    return _device_ops_backend._get_local_device_attributes()  # pylint: disable=protected-access

  @classmethod
  @experimental(
      "get_rng_state() is experimental and subject to change without notice."
  )
  def get_rng_state(
      cls, device: int | str | torch.device = "tpu"
  ) -> torch.Tensor:
    """Returns the random number generator state of the specified TPU device.

    .. warning::
        This API is experimental and subject to change in future releases.

    Args:
      device: The device to return the RNG state of. Can be an int (device
        index), "tpu", or a `torch.device("tpu")` object.

    Returns:
      A `torch.Tensor` representing the RNG state of the default generator.
    """
    _rng_validate_device_index(device, cls.current_device())
    return _device_ops_backend.get_rng_state(-1)

  @classmethod
  @experimental(
      "set_rng_state() is experimental and subject to change without notice."
  )
  def set_rng_state(
      cls, new_state: torch.Tensor, device: int | str | torch.device = "tpu"
  ) -> None:
    """Sets the random number generator state of the specified TPU device.

    .. warning::
        This API is experimental and subject to change in future releases.

    Args:
      new_state: A `torch.Tensor` containing the desired RNG state.
      device: The device to set the RNG state on. Can be an int (device index),
        "tpu", or a `torch.device("tpu")` object.
    """
    _rng_validate_device_index(device, cls.current_device())
    _device_ops_backend.set_rng_state(new_state, -1)

  @classmethod
  def _init_runtime_options(
      cls,
  ):
    """Initializes the TPU runtime options."""
    if cls._is_in_bad_fork():
      raise RuntimeError(
          "cannot access or re-initialize TPU in a forked subprocess. "
          "To use TPU with multiprocessing, you must use the 'spawn' start"
          " method"
      )
    if not cls.is_initialized():
      _device_ops_backend._init_runtime_options(cls._device_type)  # pylint: disable=protected-access
      if not hasattr(cls, "_atexit_registered"):
        atexit.register(cls._shutdown_runtime)
        cls._atexit_registered = True

  @classmethod
  @experimental(
      "synchronize() is experimental and subject to change without notice."
  )
  def synchronize(cls, device: int | None = None) -> None:
    """Waits for all kernels in all streams on a TPU device to complete.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    streams.synchronize(device)

  @classmethod
  @experimental(
      "device_count() is experimental and subject to change without notice."
  )
  def device_count(cls) -> int:  # This is in torch/cuda/__init__.py.
    """Returns the count of devices. This is implemented for *local* devices.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    if cls._device_count is None:
      if cls.is_initialized():
        cls._device_count = _device_ops_backend._get_device_count()  # pylint: disable=protected-access
      elif cls._device_type == "tpu":
        return hardware.get_tpu_device_count()
      else:
        # Fallback to init for other devices if needed.
        cls._device_count = _device_ops_backend._get_device_count()  # pylint: disable=protected-access

    assert cls._device_count is not None
    return cls._device_count

  @classmethod
  @experimental(
      "stream() is experimental and subject to change without notice."
  )
  def stream(cls, stream: streams.TpuStream | None) -> "TpuStreamContext":
    """Wrapper for a context manager that selects a given TPUStream.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    return TpuStreamContext(stream)

  @classmethod
  @experimental(
      "Stream() is experimental and may change or be removed without notice."
  )
  def Stream(cls, device=None, priority=0, **kwargs):  # pylint: disable=invalid-name
    """Device-level wrapper for TpuStream object.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    return streams.TpuStream(device=device, priority=priority, **kwargs)

  @classmethod
  @experimental(
      "Event() is experimental and may change or be removed without notice."
  )
  def Event(  # pylint: disable=invalid-name
      cls,
      enable_timing: bool = False,
      blocking: bool = False,
      interprocess: bool = False,
      external: bool = False,
  ) -> streams.TpuEvent:
    """Device-level wrapper for TpuEvent object.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    return streams.TpuEvent(
        enable_timing=enable_timing,
        blocking=blocking,
        interprocess=interprocess,
        external=external,
    )

  @classmethod
  @experimental(
      "current_stream() is experimental and subject to change without notice."
  )
  def current_stream(
      cls, device: torch.device | None = None
  ) -> streams.TpuStream:
    """Returns the active stream for a given device.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    if device is None:
      device = torch.device("tpu")
    # pylint: disable=protected-access
    stream_id = _device_ops_backend._get_current_stream_id(device.index)
    return streams.TpuStream(device=device, stream_id=stream_id)

  @classmethod
  @experimental(
      "default_stream() is experimental and subject to change without notice."
  )
  def default_stream(
      cls, device: torch.device | None = None
  ) -> streams.TpuStream:
    """Returns the default stream for a given device.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    if device is None:
      device = torch.device("tpu")
    return streams.TpuStream(device=device, stream_id=0)

  @classmethod
  @experimental(
      "get_amp_supported_dtype() is experimental and may change or be removed "
      "without notice."
  )
  def get_amp_supported_dtype(cls) -> List[torch.dtype]:  # Needed for AMP.
    """Returns the list of supported dtypes for Automatic Mixed Precision (AMP).

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    return list(_AMP_SUPPORTED_DTYPES)

  @classmethod
  def _is_autocast_enabled(cls) -> bool:
    return cls._autocast_enabled

  @classmethod
  def _get_autocast_dtype(cls) -> torch.dtype | None:
    return cls._autocast_dtype

  @classmethod
  def _set_autocast_enabled(cls, enabled: bool) -> None:
    cls._autocast_enabled = enabled

  @classmethod
  def _set_autocast_dtype(cls, dtype: torch.dtype) -> None:
    if dtype not in cls.get_amp_supported_dtype():
      raise ValueError(
          f"Unsupported dtype {dtype} for AMP. Supported dtypes are:"
          f" {cls.get_amp_supported_dtype()}"
      )
    cls._autocast_dtype = dtype

  @classmethod
  def _set_allow_cache(cls, allow: bool = True):
    _device_ops_backend._set_allow_cache(allow)  # pylint: disable=protected-access

  @classmethod
  def _set_cache_only(cls, cache_only: bool = True):
    _device_ops_backend._set_cache_only(cache_only)  # pylint: disable=protected-access

  @classmethod
  def _set_dump_on_cache_miss(cls, enable: bool = True):
    _device_ops_backend._set_dump_on_cache_miss(enable)  # pylint: disable=protected-access

  @classmethod
  def _get_dump_on_cache_miss(cls):
    return _device_ops_backend._get_dump_on_cache_miss()  # pylint: disable=protected-access

  @classmethod
  def _get_cache_requests(cls):
    return _device_ops_backend._get_cache_requests()  # pylint: disable=protected-access

  @classmethod
  def _get_cache_hits(cls):
    return _device_ops_backend._get_cache_hits()  # pylint: disable=protected-access

  @classmethod
  def _get_cache_misses(cls):
    return _device_ops_backend._get_cache_misses()  # pylint: disable=protected-access

  @classmethod
  def _clear_cache(cls):
    backend_mod = sys.modules.get("torch_tpu._internal.compile._backend")
    if backend_mod is not None and hasattr(backend_mod, "clear_cache"):
      backend_mod.clear_cache()
    _device_ops_backend._clear_cache()  # pylint: disable=protected-access

  @classmethod
  def _hbm_usage_summary(cls):
    return _device_ops_backend._hbm_usage_summary()  # pylint: disable=protected-access

  @classmethod
  def _get_cache_stats(cls):
    return _device_ops_backend._get_cache_stats()  # pylint: disable=protected-access

  @classmethod
  def _shutdown_runtime(cls):
    if cls.is_initialized():
      if torch.distributed.is_available():
        if torch.distributed.is_initialized():
          torch.distributed.barrier()
      _device_ops_backend._shutdown_runtime()  # pylint: disable=protected-access


@experimental(
    "_TpuDeviceProperties is experimental and subject to change without notice."
)
@dataclasses.dataclass(frozen=True)
class _TpuDeviceProperties:
  """Static TPU device properties, analog of _CudaDeviceProperties.

  .. warning::
      This API is experimental and subject to change in future releases.

  ``total_memory`` is HBM visible to a single device, in bytes. Peak FLOP/s
  is intentionally absent: it is a dtype-dependent published constant, not a
  queryable hardware property, and no torch device module reports it.
  """

  name: str
  total_memory: int
  multi_processor_count: int = 1

  __tt_api_stages__ = {
      "multi_processor_count": annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason="multi_processor_count is experimental and subject to change.",
      ),
  }


class TpuDeviceModule(_DeviceModule):
  """Device module implementation for TPU devices."""

  _device_type: Final[str] = "tpu"  # pyrefly: ignore[bad-override]

  @classmethod
  @experimental(
      "get_device_name() is experimental and subject to change without notice."
  )
  def get_device_name(
      cls, device: int | str | torch.device | None = None
  ) -> str:
    """Human-readable name of the attached TPU, e.g. "TPU v7".

    .. warning::
        This API is experimental and subject to change in future releases.

    Args:
      device: Optional device index, string, or torch.device to query properties
        for.
    """
    del device  # Unused
    return hardware.get_tpu_device_name()

  @classmethod
  @experimental(
      "get_device_properties() is experimental and subject to change without"
      " notice."
  )
  def get_device_properties(
      cls, device: int | str | torch.device | None = None
  ) -> _TpuDeviceProperties:
    """Static device properties, mirroring torch.cuda.get_device_properties.

    .. warning::
        This API is experimental and subject to change in future releases.

    Args:
      device: Optional device index, string, or torch.device to query properties
        for.

    Returns:
      A _TpuDeviceProperties instance for the specified device.

    Raises:
      RuntimeError: If the TPU generation is unrecognized or memory properties
        cannot be determined.
    """
    total_memory = hardware.get_hbm_bytes_per_device()
    if total_memory is None:
      raise RuntimeError(
          f"unrecognized TPU {cls.get_device_name(device)!r}; cannot "
          "determine device memory"
      )
    return _TpuDeviceProperties(
        name=cls.get_device_name(device),
        total_memory=total_memory,
    )

  @classmethod
  def _get_compute_capability(
      cls,
      device: int | str | torch.device | None = None,
  ) -> str:
    """Returns the compute capability for the TPU device.

    TPUs do not have a CUDA-style compute capability version, so this returns
    an empty string to satisfy TorchDynamo's DeviceInterface contract.

    Args:
      device: Optional device identifier (unused, present for API
        compatibility).

    Returns:
      An empty string.
    """
    del device  # Unused.
    return ""

  @classmethod
  @experimental(
      "topology_aware_mesh() is experimental and may change in the future."
  )
  def topology_aware_mesh(
      cls,
      mesh_shape: tuple[int, ...],
      *,
      dcn_mesh_shape: tuple[int, ...] | None = None,
      allow_split_physical_axes: bool = False,
      contiguous_submeshes: bool = False,
  ) -> torch.Tensor:
    """Computes a topology-aware rank layout tensor.

    Returns an int64 tensor of global torch ranks with shape `mesh_shape`,
    mapping logically adjacent coordinate positions to physically adjacent TPU
    devices on the ICI fabric. Whether the job spans one slice or several is
    read off the devices, so the same call covers both.

    Args:
      mesh_shape: Full logical mesh shape, ordered by increasing network
        intensity. Its product must equal the world size, on multi-slice jobs
        too.
      dcn_mesh_shape: Which logical axes are split across slices. Same rank as
        `mesh_shape`, product equal to the number of slices, and must divide
        `mesh_shape` elementwise. Inferred when None.
      allow_split_physical_axes: Permits splitting a physical axis across
        logical axes when required.
      contiguous_submeshes: Forwards to create_device_mesh.

    Returns:
      An int64 tensor of global torch ranks mapping the requested layout.
    """
    from torch_tpu._internal.distributed import device_mesh  # pylint: disable=g-import-not-at-top

    return device_mesh.topology_aware_mesh(
        cls._device_type,
        mesh_shape,
        dcn_mesh_shape=dcn_mesh_shape,
        allow_split_physical_axes=allow_split_physical_axes,
        contiguous_submeshes=contiguous_submeshes,
    )


def get_device_module(device_type: str) -> type[_DeviceModule]:
  """Retrieves the _DeviceModule module for the specified device.

  Args:
    device_type: A valid device type, currently only "tpu".

  Returns:
    A module that can be passed to torch._register_device_module.
  """
  if device_type != "tpu":
    raise ValueError(
        f"Unknown device type {device_type!r}, only 'tpu' is supported."
    )

  return TpuDeviceModule


class TpuDeviceInterface(TpuDeviceModule, DeviceInterface):
  """Device interface implementation for TPU devices in TorchDynamo."""

  exchange_device = TpuDeviceModule._exchange_device
  get_compute_capability = TpuDeviceModule._get_compute_capability
  get_multi_processor_count = TpuDeviceModule._get_multi_processor_count
  is_gpu = TpuDeviceModule._is_gpu
  is_triton_capable = TpuDeviceModule._is_triton_capable
  maybe_exchange_device = TpuDeviceModule._maybe_exchange_device


def get_device_interface(device_type: str) -> type[DeviceInterface]:
  """Retrieves the DeviceInterface implementation for the specified device.

  Args:
    device_type: A valid device type, currently only "tpu".

  Returns:
    A class that can be passed to
    torch._dynamo.device_interface.register_interface_for_device.
  """
  if device_type != "tpu":
    raise ValueError(
        f"Unknown device type {device_type!r}, only 'tpu' is supported."
    )

  return TpuDeviceInterface


class TpuStreamContext:
  """Operations within this context manager are enqueued on the given stream.

  Currently non-functional. See b/452051142.
  """

  def __init__(self, stream: streams.TpuStream | None):
    self.stream = stream
    self.prev_stream_id = None

  def __enter__(self):
    if self.stream is None:
      return
    # pylint: disable=protected-access
    self.prev_stream_id = _device_ops_backend._get_current_stream_id(
        self.stream.device_index
    )
    _device_ops_backend._set_current_stream_id(
        self.stream.stream_id, self.stream.device_index
    )

  def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any):
    if self.stream is None or self.prev_stream_id is None:
      return
    # pylint: disable=protected-access
    _device_ops_backend._set_current_stream_id(
        self.prev_stream_id, self.stream.device_index
    )
