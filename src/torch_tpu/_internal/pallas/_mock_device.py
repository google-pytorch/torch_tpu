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

"""Abstract device mocks for compile-only Pallas mesh generation."""

from torch import distributed as dist
from torch_tpu._internal import env


class _PallasMockClient:
  """Mock PJRT client for abstract compilation devices."""

  platform: str = "tpu"
  platform_version: str = "TPU"
  runtime_type: str = "tpu"

  def process_index(self) -> int:
    return 0

  def device_count(self) -> int:
    if dist.is_initialized():
      return dist.get_world_size()
    return env.get_int_env_once("WORLD_SIZE", default_value=1)

  def devices(self) -> list:
    return []

  def local_devices(self) -> list:
    return []


_MOCK_CLIENT = _PallasMockClient()


class _PallasDevice:
  """Compile-only abstract device compatible with JAX mesh/sharding inspection without jax.devices()."""

  def __init__(
      self,
      device_kind: str = "TPU",
      num_cores: int = 1,
      platform: str = "tpu",
      id: int = 0,
      process_index: int = 0,
  ):
    self.device_kind = device_kind
    self.num_cores = num_cores
    self.platform = platform
    self.id = id
    self.process_index = process_index
    self.client = _MOCK_CLIENT

  def __repr__(self) -> str:
    return (
        f"AbstractDevice(id={self.id}, device_kind='{self.device_kind}',"
        f" platform='{self.platform}')"
    )

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _PallasDevice):
      return False
    return (
        self.id == other.id
        and self.device_kind == other.device_kind
        and self.platform == other.platform
    )

  def __hash__(self) -> int:
    return hash((self.id, self.device_kind, self.platform))
