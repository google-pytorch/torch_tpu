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

"""Target: framework-agnostic hardware descriptor, selected by an environment variable.

Instead of detecting the hardware at runtime, the platform is declared using the
environment variable "BENCHMARK_PLATFORM". This variable must be set to one of
a fixed set of Platform enum values, and that name fully determines the Target.
This is to avoid loading PyTorch/JAX here which can conflict to hold device
lockfiles.
"""

import dataclasses
import enum
import os
from typing import Mapping

dataclass = dataclasses.dataclass

# The environment variable used to declare the hardware platform.
PLATFORM_ENV_VAR = "BENCHMARK_PLATFORM"

_DEFAULT_PLATFORM_STR = "cpu"
_DEFAULT_DTYPE_STR = "bf16"


class Platform(enum.Enum):
  """Set of platforms supported by the harness.

  Hardware platforms must be explicitly added to this enum to be supported.
  Names encode the physical shape only (chip + device count) and never imply
  a logical mesh configuration for dp/tp/pp etc.
  """

  CPU = "cpu"
  B200_1 = "b200_1"
  B200_2 = "b200_2"
  B200_4 = "b200_4"
  B200_8 = "b200_8"
  V5E_1X1 = "v5e_1x1"
  V6E_1X1 = "v6e_1x1"
  V5P_1X1X1 = "v5p_1x1x1"
  V7_1X1X1 = "v7_1x1x1"
  V7_1X1X1_2TC = "v7_1x1x1_2tc"
  V7_1X2X1 = "v7_1x2x1"
  V7_2X2X1 = "v7_2x2x1"
  V7_2X2X2 = "v7_2x2x2"
  V7_2X2X4 = "v7_2x2x4"
  V7_2X4X4 = "v7_2x4x4"


class DeviceKind(enum.Enum):
  """The backend family or hardware category.

  This categorization is coarse on purpose (e.g., CUDA, TPU, CPU).
  Specific chip generations are treated as labels within the topology,
  rather than distinct device kinds.
  """

  CUDA = "cuda"
  TPU = "tpu"
  CPU = "cpu"


@dataclass(frozen=True)
class Topology:
  """Physical device topology representing factual hardware configurations.

  Attributes:
    nnodes: The number of nodes in the cluster.
    nprocs_per_node: The number of processes (devices) per node.
    chip: A label for the chip generation (e.g., "cpu", "b200", "v5e").
  """

  nnodes: int = 1
  nprocs_per_node: int = 1
  chip: str = "cpu"


@dataclass(frozen=True)
class PlatformSpec:
  """The fixed hardware properties implied by a Platform name.

  Attributes:
    kind: The backend family (e.g., DeviceKind.TPU).
    topology: The device topology representing nodes and processes.
  """

  kind: DeviceKind
  topology: Topology


def _tpu(chip, *, nnodes: int = 1, nprocs_per_node: int = 1) -> PlatformSpec:
  return PlatformSpec(
      kind=DeviceKind.TPU,
      topology=Topology(
          nnodes=nnodes,
          nprocs_per_node=nprocs_per_node,
          chip=chip,
      ),
  )


def _gpu(chip, *, nnodes: int = 1, nprocs_per_node: int = 1) -> PlatformSpec:
  return PlatformSpec(
      kind=DeviceKind.CUDA,
      topology=Topology(
          nnodes=nnodes,
          nprocs_per_node=nprocs_per_node,
          chip=chip,
      ),
  )


_PLATFORMS: dict[Platform, PlatformSpec] = {
    Platform.CPU: PlatformSpec(
        kind=DeviceKind.CPU, topology=Topology(chip="cpu")
    ),
    Platform.B200_1: _gpu("b200"),
    Platform.B200_2: _gpu("b200", nnodes=1, nprocs_per_node=2),
    Platform.B200_4: _gpu("b200", nnodes=1, nprocs_per_node=4),
    Platform.B200_8: _gpu("b200", nnodes=1, nprocs_per_node=8),
    Platform.V5E_1X1: _tpu("v5e"),
    Platform.V6E_1X1: _tpu("v6e"),
    Platform.V5P_1X1X1: _tpu("v5p"),
    Platform.V7_1X1X1: _tpu("v7"),
    # A single v7 chip with one process per TensorCore (2 processes), for
    # world_size=2 distributed benchmarks. V7_1X1X1 above uses one process.
    Platform.V7_1X1X1_2TC: _tpu("v7", nnodes=1, nprocs_per_node=2),
    # Two v7 chips (a 1x2x1 sub-slice of a 4-chip host) with one process per
    # TensorCore (4 processes), for world_size=4 distributed benchmarks.
    Platform.V7_1X2X1: _tpu("v7", nnodes=1, nprocs_per_node=4),
    # Each v7 has two cores, we use one process per core. Hence, nprocs is 8
    # for the 2x2x1 case.
    Platform.V7_2X2X1: _tpu("v7", nnodes=1, nprocs_per_node=8),
    Platform.V7_2X2X2: _tpu("v7", nnodes=2, nprocs_per_node=8),
    Platform.V7_2X2X4: _tpu("v7", nnodes=4, nprocs_per_node=8),
    Platform.V7_2X4X4: _tpu("v7", nnodes=8, nprocs_per_node=8),
}


class DType(enum.Enum):
  """Framework-agnostic data type definitions.

  Each backend framework is responsible for mapping these abstract types
  to its own concrete types (e.g., BF16 maps to torch.bfloat16 or jnp.bfloat16).

  The dtype is passed to benchmark factories as context. It is the factory
  function's responsibility to use it appropriate when loading models and
  inputs.
  """

  BF16 = "bf16"
  FP32 = "fp32"


class UnsupportedBenchmark(Exception):
  """Raised from a benchmark factory when a target cannot cannot run the benchmark."""


@dataclass(frozen=True)
class Target:
  """A framework-agnostic descriptor of the target hardware environment.

  This class contains only pure data and avoids importing or using any
  framework-specific types (like torch or jax types). Framework-specific
  DeviceOps implementations will translate this descriptor into concrete
  device handles.

  Attributes:
    platform: The declared platform enum (e.g., Platform.B200_8).
    platform_spec: The hardware topology and device kind for the platform.
    dtype: The framework-agnostic data type.
  """

  platform: Platform
  platform_spec: PlatformSpec
  dtype: DType

  @property
  def device_kind(self) -> DeviceKind:
    return self.platform_spec.kind


def platform_from_env(
    env: Mapping[str, str] = os.environ,  # PY_ENVIRON_OK=benchmark
) -> Platform:
  """Reads the benchmark platform configuration from the environment.

  The platform is declared explicitly rather than detected.

  Args:
    env: A mapping representing the environment variables (defaults to
      `os.environ`).

  Returns:
    The resolved Platform enum corresponding to the environment variable.

  Raises:
    ValueError: If the environment variable contains an unsupported platform
    string.
  """
  raw_str = env.get(PLATFORM_ENV_VAR, _DEFAULT_PLATFORM_STR)
  return Platform(raw_str.lower())


def make_target(
    platform: Platform = Platform(_DEFAULT_PLATFORM_STR),
    dtype: DType = DType(_DEFAULT_DTYPE_STR),
) -> Target:
  """Builds a framework-agnostic Target descriptor for a given platform.

  No device handles are acquired during this step. The resulting Target
  is purely data; handles are materialized later by framework-specific
  DeviceOps implementations.

  Args:
    platform: The declared hardware platform to target.
    dtype: The data type policy to apply, which overrides any platform defaults.

  Returns:
    A populated Target dataclass representing the requested configuration.
  """
  spec = _PLATFORMS[platform]
  return Target(
      platform=platform,
      platform_spec=spec,
      dtype=dtype,
  )
