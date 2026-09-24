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

"""Execution modes for each framework and target kind."""

import contextlib
import enum
from typing import Iterator

import torch
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e.harness import target as target_lib


class Framework(enum.Enum):
  """Execution framework (torch vs. torchax).

  This is not a runtime command-line flag. It is decided by which test binary
  runs and passed in as that binary's module-level constant. It appears only in
  the test execution matrix.
  """

  TORCH = "torch"
  TORCHAX = "torchax"


# TODO: b/534438865 - Move RunMode defn to harness/mode.py
def modes_for(
    framework: Framework | str,
    device_kind: target_lib.DeviceKind | str,
) -> list[common.RunMode]:
  """Execution modes for the given framework and device kind.

  |          | torch                                            | torchax  |
  |----------|--------------------------------------------------|----------|
  | tpu      | eager_default, eager_optimized,                  | compiled |
  |          | eager_defer_never_and_launch_blocking, compiled  |          |
  | cuda/cpu | eager_default, compiled                          |    N/A   |

  Args:
    framework: The execution framework (Framework enum or string representation
      such as "torch" or "torchax").
    device_kind: The hardware device kind (DeviceKind enum or string
      representation such as "tpu", "cuda", or "cpu").

  Returns:
    A list of supported RunMode instances for this framework and device kind.

  Raises:
    ValueError: If an unknown framework or device kind string is provided.
  """
  # The passed values can be strings. Explicitly coerce them into enums to
  # catch invalid values.
  framework = Framework(framework)
  device_kind = target_lib.DeviceKind(device_kind)

  if framework is Framework.TORCHAX:
    # torchax runs compiled only on TPU.
    if device_kind is not target_lib.DeviceKind.TPU:
      raise ValueError("torchax only runs on TPU")
    return [common.RunMode.COMPILED]
  if device_kind is target_lib.DeviceKind.TPU:
    return [
        common.RunMode.EAGER_DEFAULT,
        common.RunMode.EAGER_OPTIMIZED,
        common.RunMode.COMPILED,
    ]
  if device_kind is target_lib.DeviceKind.CPU:
    return [common.RunMode.EAGER_DEFAULT]
  # torch on cuda.
  return [
      common.RunMode.EAGER_DEFAULT,
      common.RunMode.COMPILED,
  ]


@contextlib.contextmanager
def run_mode_context(
    run_mode: common.RunMode | str,
    target: target_lib.Target,
) -> Iterator[None]:
  """Configures the execution environment for different run modes.

  Args:
    run_mode: The execution mode to configure the environment for.
    target: The hardware target descriptor.

  Yields:
    None.

  Raises:
    ValueError: If an unexpected or unsupported run mode is provided.
  """
  # The passed values can be strings. Explicitly coerce them into enums to
  # catch invalid values.
  run_mode = common.RunMode(run_mode)
  if target.device_kind is not target_lib.DeviceKind.TPU:
    if target.device_kind is target_lib.DeviceKind.CUDA:
      torch.set_float32_matmul_precision("high")
    yield
    return

  from torch_tpu._internal import execution_mode  # pylint: disable=g-import-not-at-top

  original_eager_mode = execution_mode.eager_mode
  new_eager_mode = None

  match run_mode:
    case common.RunMode.EAGER_DEFAULT:
      new_eager_mode = execution_mode.EagerMode.DEFER_NEVER
    case common.RunMode.EAGER_OPTIMIZED:
      new_eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    case common.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING:
      new_eager_mode = execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
    case common.RunMode.COMPILED:
      pass  # Explicitly do nothing.

  if new_eager_mode is not None:
    execution_mode.eager_mode = new_eager_mode

  try:
    yield
  finally:
    # Revert back to the original execution mode.
    execution_mode.eager_mode = original_eager_mode
