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

"""Common definitions for benchmarks."""

import enum
import random
from absl import flags
import torch


class Platform(enum.Enum):
  """The platform to run the benchmark on."""

  # The platform names should match the ones in the MLCompass config file. See
  # go/torchtpu-mlcompass#configuration-structure for more details.
  V7_1X1X1 = "v7_1x1x1"
  V7_2X2X1 = "v7_2x2x1"
  V7_2X2X2 = "v7_2x2x2"
  V7_2X2X4 = "v7_2x2x4"
  V7_2X4X4 = "v7_2x4x4"
  GFC_1X1X1 = "gfc_1x1x1"
  GFC_2X2X1 = "gfc_2x2x1"
  GFC_2X2X2 = "gfc_2x2x2"
  GFC_2X2X4 = "gfc_2x2x4"
  GFC_2X4X4 = "gfc_2x4x4"
  B200_1 = "b200_1"
  B200_4 = "b200_4"
  B200_8 = "b200_8"
  TORCH_CPU = "torch_cpu"
  V5E_1X1 = "v5e_1x1"


class Backend(enum.Enum):
  """The backend to use for the benchmark."""

  TORCH_TPU = "torch_tpu"
  TORCHAX = "torchax"


PLATFORM = flags.DEFINE_enum_class(
    "platform",
    Platform.GFC_1X1X1,
    Platform,
    "The platform to run the tests on.",
)

BACKEND = flags.DEFINE_enum_class(
    "backend",
    Backend.TORCH_TPU,
    Backend,
    "The backend to use for the benchmark.",
)

PLATFORM_DEVICE_MAP = {
    Platform.V7_1X1X1: "tpu",
    Platform.V7_2X2X1: "tpu",
    Platform.V7_2X2X2: "tpu",
    Platform.V7_2X2X4: "tpu",
    Platform.V7_2X4X4: "tpu",
    Platform.GFC_1X1X1: "tpu",
    Platform.GFC_2X2X1: "tpu",
    Platform.GFC_2X2X2: "tpu",
    Platform.GFC_2X2X4: "tpu",
    Platform.GFC_2X4X4: "tpu",
    Platform.B200_1: "cuda",
    Platform.B200_4: "cuda",
    Platform.B200_8: "cuda",
    Platform.TORCH_CPU: "cpu",
    Platform.V5E_1X1: "tpu",
}

PLATFORM_TO_NODE_CONFIG = {
    Platform.V7_2X2X2: {"num_nodes": 2, "nproc_per_node": 8},
    Platform.V7_2X2X4: {"num_nodes": 4, "nproc_per_node": 8},
    Platform.V7_2X4X4: {"num_nodes": 8, "nproc_per_node": 8},
    Platform.GFC_2X2X2: {"num_nodes": 2, "nproc_per_node": 8},
    Platform.GFC_2X2X4: {"num_nodes": 4, "nproc_per_node": 8},
    Platform.GFC_2X4X4: {"num_nodes": 8, "nproc_per_node": 8},
}

_RANDOM_SEED = 0


def seed_rngs() -> None:
  """Seeds the Python and PyTorch RNGs with the given seed."""
  random.seed(_RANDOM_SEED)
  torch.manual_seed(_RANDOM_SEED)


class RunMode(enum.Enum):
  """The mode to run the benchmark in.

  Make sure that no entry is a prefix of the other. Run mode name is appended
  to test names, and MLCompass runs tests based on a prefix match, it can lead
  to duplicate entries. For e.g., test_model_eager will match both
  test_model_eager and test_model_eager_optimized. Hence, we use
  eager_default and eager_optimized.
  """

  EAGER_DEFAULT = (  # Run the model in eager mode with DeferNever.
      "eager_default"
  )
  EAGER_OPTIMIZED = (  # Run the model in eager mode with DeferAndFuse.
      "eager_optimized"
  )
  # Run the model in eager mode with DeferNeverAndLaunchBlocking.
  EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING = (
      "eager_defer_never_and_launch_blocking"
  )
  COMPILED = "compiled"  # Run the model with torch.compile.


def get_torch_device(platform: Platform) -> torch.device:
  """Returns the torch device for the given platform."""
  return torch.device(PLATFORM_DEVICE_MAP[platform])


def is_torch_compile(run_mode: RunMode) -> bool:
  """Returns whether the given run mode uses torch.compile."""
  return run_mode == RunMode.COMPILED
