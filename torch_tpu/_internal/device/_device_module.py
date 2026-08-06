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
from typing import Any, Final, List
import torch
from torch_tpu._internal.device import _device_ops_backend
import torch_tpu._internal.precision as _precision_module
from torch_tpu._internal.stream import streams
from torch_tpu._internal.utils import hardware
from torch_tpu._internal.utils.annotations import experimental

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
  `torch.cuda.default_generators`.
  """

  def __init__(self):
    self._cached_tuple = None

  def __get__(self, instance, owner):
    if owner is None:
      return ()
    if self._cached_tuple is None:
      # Ensure initialized
      owner.current_device()
      self._cached_tuple = tuple(
          _device_ops_backend.get_default_generator(i)
          for i in range(owner.device_count())
      )
    return self._cached_tuple


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


class _DeviceModule(abc.ABC):
  """torch_tpu device equivalent to functions in "torch/cuda/__init__.py".

  https://github.com/pytorch/pytorch/blob/v2.7.0/torch/cuda/__init__.py

  For Automatic Mixed Precision (AMP), see:
  https://docs.pytorch.org/docs/stable/amp.html

  Since methods of this class are accessible via getattr(torch, "tpu"), we need
  to make sure they are not accidentally exposed to the public API by adding a
  leading underscore to the names of internal methods.
  """

  # device_count and current_device are None until device_count() and
  # current_device() are called.
  _device_count: int | None = None
  _current_device: int | None = None
  _autocast_enabled: bool = False
  _autocast_dtype: torch.dtype | None = torch.bfloat16

  _device_type: str

  Precision = _precision_module.Precision  # pylint: disable=invalid-name
  precision = _precision_module.precision
  # A sequence of default generators, one per device, matching the exact
  # type and behavior of PyTorch's native `torch.cuda.default_generators`.
  default_generators = _DefaultGeneratorsProperty()

  # This method is called when a subclass of this abstract base class is
  # created. As of Python 3.15, there is no obviously supported way of declaring
  # a required abstract class attribute, but we can do the validation here.
  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)

    if not hasattr(cls, "_device_type"):
      raise TypeError(f"Class '{cls.__name__}' must define _device_type")

  @classmethod
  def current_device(cls) -> int:  # This is in torch/cuda/__init__.py.
    """Returns the index of the currently selected device."""
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
  def set_device(
      cls, device: int | str | torch.device
  ) -> None:  # This is in torch/cuda/__init__.py.
    """Sets the current device.

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
  def is_available(cls) -> bool:  # This is in torch/cuda/__init__.py.
    """Returns if the TPU backend is currently available."""
    if cls._device_type == "tpu":
      return cls.device_count() > 0
    if cls._device_type == "xla_cpu":
      return True
    if cls._device_type == "xla_cuda":
      return hardware.has_nvidia_gpu()
    return False

  @classmethod
  def is_initialized(cls) -> bool:  # This is in torch/cuda/__init__.py.
    """Returns whether PyTorch's TPU state has been initialized."""
    return _device_ops_backend._is_initialized()  # pylint: disable=protected-access

  @classmethod
  def _lazy_init(cls) -> None:
    """No-op initialization method to satisfy PyTorch's PrivateUse1 requirements."""
    pass

  @classmethod
  def _is_in_bad_fork(cls) -> bool:
    """Returns whether we are in a bad fork.

    This is a CUDA specific concept, related to how the CUDA context and runtime
    interacts with multiprocesses. An implementation is required to support
    torch.manual_seed(), as it is called in _seed_custom_device() in
    py/torch/random.py.
    """
    # TODO(b/456545545): unda - revisit this.
    return False

  @classmethod
  def manual_seed_all(cls, seed: int) -> None:
    """Sets the seed for generating random numbers on all devices."""
    _device_ops_backend.manual_seed_all(seed)

  @classmethod
  def get_local_device_attributes(cls) -> Mapping[str, Any]:
    """Returns attributes of the local PJRT device, initializing options if needed."""
    cls._init_runtime_options()
    return _device_ops_backend._get_local_device_attributes()  # pylint: disable=protected-access

  @classmethod
  def get_rng_state(
      cls, device: int | str | torch.device = "tpu"
  ) -> torch.Tensor:
    _rng_validate_device_index(device, cls.current_device())
    return _device_ops_backend.get_rng_state(-1)

  @classmethod
  def set_rng_state(
      cls, new_state: torch.Tensor, device: int | str | torch.device = "tpu"
  ) -> None:
    _rng_validate_device_index(device, cls.current_device())
    _device_ops_backend.set_rng_state(new_state, -1)

  @classmethod
  def _init_runtime_options(
      cls,
  ):
    """Initializes the TPU runtime options."""
    if not cls.is_initialized():
      _device_ops_backend._init_runtime_options(cls._device_type)  # pylint: disable=protected-access
      if not hasattr(cls, "_atexit_registered"):
        atexit.register(cls._shutdown_runtime)
        cls._atexit_registered = True

  @classmethod
  def synchronize(cls, device: int | None = None) -> None:
    """Waits for all kernels in all streams on a TPU device to complete."""
    streams.synchronize(device)

  @classmethod
  def device_count(cls) -> int:  # This is in torch/cuda/__init__.py.
    """Returns the count of devices. This is implemented for *local* devices."""
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
  def stream(cls, stream: streams.TpuStream | None) -> "TpuStreamContext":
    """Wrapper for a context manager that selects a given TPUStream."""
    return TpuStreamContext(stream)

  @classmethod
  def Stream(cls, device=None, priority=0, **kwargs):  # pylint: disable=invalid-name
    """Device-level wrapper for TpuStream object."""
    return streams.TpuStream(device=device, priority=priority, **kwargs)

  @classmethod
  def Event(  # pylint: disable=invalid-name
      cls,
      enable_timing: bool = False,
      blocking: bool = False,
      interprocess: bool = False,
      external: bool = False,
  ) -> streams.TpuEvent:
    """Device-level wrapper for TpuEvent object."""
    return streams.TpuEvent(
        enable_timing=enable_timing,
        blocking=blocking,
        interprocess=interprocess,
        external=external,
    )

  @classmethod
  def set_stream(cls, stream: streams.TpuStream) -> None:
    """Sets the current stream."""
    # pylint: disable=protected-access
    _device_ops_backend._set_current_stream_id(stream.stream_id)

  @classmethod
  def current_stream(
      cls, device: torch.device | None = None
  ) -> streams.TpuStream:
    """Returns the active stream for a given device."""
    if device is None:
      device = torch.device("tpu")
    # pylint: disable=protected-access
    stream_id = _device_ops_backend._get_current_stream_id(device.index)
    return streams.TpuStream(device=device, stream_id=stream_id)

  @classmethod
  def default_stream(
      cls, device: torch.device | None = None
  ) -> streams.TpuStream:
    """Returns the default stream for a given device."""
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
    return _device_ops_backend._get_cache_requests()  # pylint: disable=protected-accessœ

  @classmethod
  def _get_cache_hits(cls):
    return _device_ops_backend._get_cache_hits()  # pylint: disable=protected-access

  @classmethod
  def _get_cache_misses(cls):
    return _device_ops_backend._get_cache_misses()  # pylint: disable=protected-access

  @classmethod
  def _clear_cache(cls):
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


@dataclasses.dataclass(frozen=True)
class _TpuDeviceProperties:
  """Static TPU device properties, analog of _CudaDeviceProperties.

  ``total_memory`` is HBM visible to a single device, in bytes. Peak FLOP/s
  is intentionally absent: it is a dtype-dependent published constant, not a
  queryable hardware property, and no torch device module reports it.
  """

  name: str
  total_memory: int
  multi_processor_count: int = 1


class TpuDeviceModule(_DeviceModule):
  """Device module implementation for TPU devices."""
  _device_type: Final[str] = "tpu"  # pyrefly: ignore[bad-override]

  @classmethod
  def get_device_name(
      cls, device: int | str | torch.device | None = None
  ) -> str:
    """Human-readable name of the attached TPU, e.g. "TPU v7"."""
    del device  # Unused
    return hardware.get_tpu_device_name()

  @classmethod
  def get_device_properties(
      cls, device: int | str | torch.device | None = None
  ) -> _TpuDeviceProperties:
    """Static device properties, mirroring torch.cuda.get_device_properties.

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
  def get_compute_capability(
      cls,
      device: int | str | torch.device | None = None,
  ) -> str:
    del device  # Unused.
    return ""


class XlaCudaDeviceModule(_DeviceModule):
  _device_type: Final[str] = "xla_cuda"  # pyrefly: ignore[bad-override]


class XlaCpuDeviceModule(_DeviceModule):
  _device_type: Final[str] = "xla_cpu"  # pyrefly: ignore[bad-override]


_DEVICE_MODULE_MAPPING: Final[Mapping[str, type[_DeviceModule]]] = {
    "tpu": TpuDeviceModule,
    "xla_cuda": XlaCudaDeviceModule,
    "xla_cpu": XlaCpuDeviceModule,
}


def get_device_module(device_type: str) -> type[_DeviceModule]:
  """Retrieves the _DeviceModule module for the specified device.

  Args:
    device_type: A valid device type, one of: {tpu, xla_cuda, xla_cpu}

  Returns:
    A module that can be passed to torch._register_device_module.
  """
  if device_type not in _DEVICE_MODULE_MAPPING:
    raise ValueError(
        f"Unknown device type {device_type!r}, please choose one of: "
        + ", ".join(_DEVICE_MODULE_MAPPING.keys())
    )

  return _DEVICE_MODULE_MAPPING[device_type]


class TpuStreamContext:
  """Operations within this context manager are enqueued on the given stream.

  Currently non-functional. See b/452051142.
  """

  def __init__(self, stream: streams.TpuStream | None):
    self.stream = stream
    self.prev_stream = None

  def __enter__(self):
    if self.stream is None:
      return
    self.prev_stream = torch.tpu.current_stream()
    torch.tpu.set_stream(self.stream)

  def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any):
    if self.stream is None:
      return
    torch.tpu.set_stream(self.prev_stream)
