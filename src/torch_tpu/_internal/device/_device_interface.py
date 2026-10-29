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

"""TorchDynamo DeviceInterface implementation for TPU devices."""

from __future__ import annotations

from torch._dynamo.device_interface import DeviceInterface
from torch_tpu._internal.device import _device_module


class TpuDeviceInterface(_device_module.TpuDeviceModule, DeviceInterface):
  """Device interface implementation for TPU devices in TorchDynamo."""

  exchange_device = _device_module.TpuDeviceModule._exchange_device
  get_compute_capability = (
      _device_module.TpuDeviceModule._get_compute_capability
  )
  get_multi_processor_count = (
      _device_module.TpuDeviceModule._get_multi_processor_count
  )
  is_gpu = _device_module.TpuDeviceModule._is_gpu
  is_triton_capable = _device_module.TpuDeviceModule._is_triton_capable
  maybe_exchange_device = (
      _device_module.TpuDeviceModule._maybe_exchange_device
  )


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
