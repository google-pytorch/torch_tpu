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

"""Registers layer benchmarks: one spec per layer config and train/eval step.

`add_layer_sweep` registers a benchmark for every config of a layer sweep and
every requested step (train or eval). Each spec is named
`layer_<benchmark>_<train|eval>_<microbenchmark name>`, where the
microbenchmark name (`configs.microbenchmark_name`) identifies the layer shape
and is also exported with the results.
"""

from collections.abc import Callable, Sequence
import enum
import re
from typing import Any, TypeVar

import torch
from torch.nn import attention
from perflab import common
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops
from perflab.layer_utils import builders
from perflab.layer_utils import configs as configs_lib

_ConfigT = TypeVar("_ConfigT")


class Step(enum.Enum):
  """A step that a layer benchmark takes."""

  # Forward and backward, without an optimizer.
  TRAIN = "train"
  # Forward only.
  EVAL = "eval"


ALL_STEPS: frozenset[Step] = frozenset(Step)

# The run modes a layer benchmark runs in unless it opts out of some. The debug
# mode `eager_defer_never_and_launch_blocking` is never skipped, so it can
# still be requested explicitly.
ALL_RUN_MODES: frozenset[common.RunMode] = frozenset({
    common.RunMode.EAGER_DEFAULT,
    common.RunMode.EAGER_OPTIMIZED,
    common.RunMode.COMPILED,
})

# The accelerator kinds a layer benchmark can run on, and runs on by default.
ACCELERATOR_KINDS: frozenset[target_lib.DeviceKind] = frozenset({
    target_lib.DeviceKind.TPU,
    target_lib.DeviceKind.CUDA,
})

_PRESUBMIT_REASON = (
    "Layer shape sweeps run in presubmit scope only in the FakeTensorMode CPU"
    " smoke test."
)

_NON_NAME_CHARS = re.compile(r"[^a-z0-9_]+")


def spec_name(benchmark: str, step: Step, microbenchmark_name: str) -> str:
  """Returns the benchmark name of a layer config.

  The microbenchmark name is lowercased, each run of characters other than
  `[a-z0-9_]` (e.g. the dot in `SDPBackend.MATH`) becomes `_`, and leading and
  trailing `_` are dropped, so that the name is a valid test method name and
  `--test_filter` pattern.

  Args:
    benchmark: The layer benchmark, e.g. `"linear"`.
    step: The step that the benchmark takes.
    microbenchmark_name: The microbenchmark name of the layer config.
  """
  micro = _NON_NAME_CHARS.sub("_", microbenchmark_name.lower()).strip("_")
  return f"layer_{benchmark}_{step.value}_{micro}"


class _SdpaKernelWrapper(torch.nn.Module):
  """Runs the wrapped layer with the overrideable SDPA backend enabled.

  The TorchTPU SDPA kernel is registered as the overrideable backend. An
  `sdpa_kernel` context inside the layer still selects its own backends.
  """

  def __init__(self, module: torch.nn.Module):
    super().__init__()
    self.module = module

  def forward(self, *args: Any, **kwargs: Any) -> Any:
    with attention.sdpa_kernel([attention.SDPBackend.OVERRIDEABLE]):
      return self.module(*args, **kwargs)


def _make_factory(
    layer_args: builders.LayerArgs,
    step: Step,
    device_kinds: frozenset[target_lib.DeviceKind],
    data_dependent: bool,
) -> registry_lib.Factory:
  """Returns a benchmark factory that builds the layer for `layer_args`."""
  is_training = step is Step.TRAIN

  def factory(ctx: context_lib.Context) -> tuple[Any, ...]:
    # The CPU always builds the layer, for the FakeTensorMode CPU smoke test.
    if (
        ctx.device_kind != target_lib.DeviceKind.CPU
        and ctx.device_kind not in device_kinds
    ):
      raise target_lib.UnsupportedBenchmark(
          f"the layer runs only on {sorted(k.value for k in device_kinds)},"
          f" not {ctx.device_kind.value}"
      )
    if data_dependent:
      mode_lib.allow_data_dependent_fake_outputs()
    model_and_input = builders.ml_layer_model_builder(
        layer_args,
        # A device string, as in the other perflab factories:
        # `torch.device("tpu")` only parses once the TPU backend is registered.
        ctx.device_kind.value,
        torch_device_ops.get_torch_dtype(ctx.dtype),
        is_training,
    )
    model = model_and_input.model
    if ctx.device_kind == target_lib.DeviceKind.TPU:
      model = _SdpaKernelWrapper(model)
    model.train(is_training)
    inputs = model_and_input.example_inputs
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    if is_training:
      # No optimizer: the training step only computes the gradients.
      return model, args, {}, None
    return model, args, {}

  return factory


def add_layer_sweep(
    benchmark: str,
    configs: Sequence[_ConfigT],
    to_layer_args: Callable[[_ConfigT], builders.LayerArgs],
    *,
    steps: frozenset[Step] = ALL_STEPS,
    run_modes: frozenset[common.RunMode] = ALL_RUN_MODES,
    device_kinds: frozenset[target_lib.DeviceKind] = ACCELERATOR_KINDS,
    data_dependent: bool = False,
    presubmit_skip_reason: str | None = None,
) -> None:
  """Registers a benchmark for each config and step of a layer.

  Every spec has `PresubmitPolicy.CPU_SMOKE_ONLY`, unless
  `presubmit_skip_reason` is set. Under `--fake_tensor_mode`,
  a layer that runs in an accelerator-only mode also runs `eager_default`, so
  that the FakeTensorMode CPU smoke test builds and steps it.

  Args:
    benchmark: The layer benchmark, e.g. `"linear"`. Benchmark names start with
      `layer_<benchmark>_`.
    configs: The layer config dataclasses to register, one spec each per step.
    to_layer_args: Returns the builder args for a config.
    steps: The steps to register, a non-empty subset of `ALL_STEPS`.
    run_modes: The run modes the layer runs in, a non-empty subset of
      `ALL_RUN_MODES`.
    device_kinds: The accelerator kinds the layer runs on, a non-empty subset of
      `ACCELERATOR_KINDS`. The factory raises `UnsupportedBenchmark` on other
      accelerators. The CPU always runs the layer, for the FakeTensorMode CPU
      smoke test.
    data_dependent: Whether the layer reads data-dependent values, e.g. the
      output size of `torch.unique`. The FakeTensorMode CPU smoke test then
      returns placeholders for them.
    presubmit_skip_reason: Why the layer does not run in presubmit scope, e.g.
      because the CPU cannot build it for the FakeTensorMode CPU smoke test. If
      set, every spec has `PresubmitPolicy.SKIP` with this reason.

  Raises:
    ValueError: If `steps`, `run_modes` or `device_kinds` is empty or has
      values outside `ALL_STEPS`, `ALL_RUN_MODES` or `ACCELERATOR_KINDS`, or if
      a benchmark name is already registered.
  """
  for arg_name, values, allowed in (
      ("steps", steps, ALL_STEPS),
      ("run_modes", run_modes, ALL_RUN_MODES),
      ("device_kinds", device_kinds, ACCELERATOR_KINDS),
  ):
    if not values or not values <= allowed:
      raise ValueError(
          f"{arg_name} must be a non-empty subset of"
          f" {sorted(v.value for v in allowed)}, got"
          f" {sorted(str(getattr(v, 'value', v)) for v in values)}"
      )
  _, fake_tensor_mode = flags_lib.collection_flag_values()
  skipped_run_modes = mode_lib.smoke_skipped_run_modes(
      frozenset(m.value for m in ALL_RUN_MODES - run_modes),
      fake_tensor_mode=fake_tensor_mode,
  )
  if presubmit_skip_reason is None:
    presubmit = registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY
    presubmit_reason = _PRESUBMIT_REASON
  else:
    presubmit = registry_lib.PresubmitPolicy.SKIP
    presubmit_reason = presubmit_skip_reason
  for config in configs:
    microbenchmark_name = configs_lib.microbenchmark_name(config)
    layer_args = to_layer_args(config)
    # In `Step` order, so that registration order does not depend on set order.
    for step in (s for s in Step if s in steps):
      registry_lib.add_benchmark(
          factory=_make_factory(layer_args, step, device_kinds, data_dependent),
          stepper=(
              step_lib.StepperType.TRAINING
              if step is Step.TRAIN
              else step_lib.StepperType.FORWARD
          ),
          # Only the `compiled` run mode reads the compile config. A training
          # step takes the default (`Scope.STEP`), which compiles the whole
          # step, forward and backward. A forward step compiles the model.
          compile_config=(
              None
              if step is Step.TRAIN
              else compile_lib.CompileConfig(scope=compile_lib.Scope.MODEL)
          ),
          name=spec_name(benchmark, step, microbenchmark_name),
          skipped_run_modes=skipped_run_modes,
          presubmit=presubmit,
          presubmit_reason=presubmit_reason,
          # Layer inputs come from the config shapes, not from a batch
          # workload, so layer benchmarks export no throughput.
          workload=None,
          microbenchmark_name=microbenchmark_name,
      )
