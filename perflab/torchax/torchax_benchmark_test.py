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

"""TorchAx benchmark binary.

Runs every registered single-host benchmark, once per applicable execution
mode using TorchAx.

Environment & Flags:
  BENCHMARK_PLATFORM   v5e_1x1 | v6e_1x1 | ... (default: cpu)
  --run_scope          full | presubmit         (default: full)
"""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from torch_tpu._internal.distributed import multiprocessing
from perflab import base_test
from perflab import cases
from perflab import common
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import device_ops as device_ops_lib
from perflab import discovery as discovery_lib
from perflab import mode as mode_lib
from perflab import models
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import workload as workload_lib
from perflab.torchax import torchax_device_ops
from perflab.torchax import torchax_step

# Pre-import module_registry so torchvision initializes before XLADispatchMode is enabled.
from tests import module_registry  # pylint: disable=unused-import # noqa: F401
import torch_xla2 as torchax  # pylint: disable=unused-import # noqa: F401
from torchax import interop  # pylint: disable=unused-import # noqa: F401

# Framework definition for this binary.
_FRAMEWORK = mode_lib.Framework.TORCHAX

# Resolved once at collection time.
_PLATFORM = target_lib.platform_from_env()

# Import all models and trigger registration of benchmarks and steppers.
failures = discovery_lib.import_submodules(models)

torchax.enable_globally()
torchax.enable_performance_mode()


_CASES = list(cases.get_cases(_PLATFORM, _FRAMEWORK))


def _make_run_step(
    spec: registry_lib.BenchmarkSpec,
    ctx: context_lib.Context,
    mode: common.RunMode,
) -> tuple[step_lib.Stepper, workload_lib.StepWork | None]:
  """Builds the stepper of a benchmark and the work of one of its steps."""
  common.seed_rngs()
  runner = torchax_step.resolve_torchax_stepper(
      spec.stepper, **spec.stepper_kwargs
  )
  runner.init_with_benchmark_args(spec, ctx)
  step_work = (
      None
      if ctx.workload is None
      else workload_lib.step_work(
          ctx.workload, spec.stepper, spec.stepper_kwargs
      )
  )

  if common.is_torch_compile(mode):
    compile_config = spec.compile_config or compile_lib.CompileConfig()
    runner.compile(compile_config, ctx.target)

  return runner, step_work


def reset_state() -> None:
  """Resets the JAX compilation cache."""
  jax.clear_caches()


class TorchaxBenchmarkTest(
    base_test.BaseBenchmarkTest, parameterized.TestCase  # ABSLTEST_OK=benchmark
):
  """One test method, parameterized over the registry x mode matrix for TorchAx."""

  framework = _FRAMEWORK
  platform = _PLATFORM
  case_params = base_test.case_params(_CASES)
  import_failures = failures

  def make_device_ops(
      self, target: target_lib.Target
  ) -> torchax_device_ops.TorchaxDeviceOps:
    return torchax_device_ops.TorchaxDeviceOps(target)

  def make_run_step(
      self,
      spec: registry_lib.BenchmarkSpec,
      ctx: context_lib.Context,
      mode: common.RunMode,
  ) -> tuple[step_lib.Stepper, workload_lib.StepWork | None]:
    return _make_run_step(spec, ctx, mode)

  def reset_state(
      self,
      target: target_lib.Target,
      device_ops: device_ops_lib.DeviceOps,
      mode: common.RunMode,
  ) -> None:
    del target, device_ops, mode  # Unused.
    reset_state()

  @parameterized.named_parameters(*_CASES)
  def test_benchmark(
      self, spec: registry_lib.BenchmarkSpec, mode: common.RunMode
  ):
    # TODO - b/534438865: Add support for decoder only decode stepper.
    if spec.stepper == step_lib.StepperType.DECODER_ONLY_DECODE:
      self.skipTest(
          f"TorchAx benchmark does not support stepper {spec.stepper}"
      )

    self.run_case(spec, mode)


def _test_main(*args: Any, **kwargs: Any) -> None:
  absltest.main(*args, testLoader=base_test.BenchmarkTestLoader(), **kwargs)


if __name__ == "__main__":
  multiprocessing.handle_test_main(_test_main)
