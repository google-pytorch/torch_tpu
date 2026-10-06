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

"""Tests for TorchTPU context manager support under `torch.compile`."""

from absl.testing import absltest
import torch
from torch_tpu._internal import compiler_options
from torch_tpu._internal import execution_mode
from torch_tpu._internal import profiler
from torch_tpu._internal.device_utils import annotations
from torch_tpu._internal.export import export
from tests import seed_test_utils

# See the assertion text in torch/_dynamo/symbolic_convert.py.
_ASSERTION_ERROR_REGEX = r"Attempt to trace forbidden callable"


class ContextManagerErrorsTest(seed_test_utils.RepeatableTest):
  """Tests that banned TorchTPU context managers fail fast under Dynamo."""

  def setUp(self):
    super().setUp()
    # Drop cached compilations so each test observes a fresh trace.
    torch._dynamo.reset()

  def test_set_eager_mode_raises_in_torch_compile_region(self):

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_layout_context_raises_in_torch_compile_region(self):

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with annotations.LayoutContext(
          annotations.TpuLayout(minor_to_major=[1, 0])
      ):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_custom_compiler_options_raises_in_torch_compile_region(self):

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with compiler_options.custom_compiler_options(
          {"xla_optimization_level": "O2"}
      ):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_enable_tracebacks_raises_in_torch_compile_region(self):

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with export.enable_tracebacks(False):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_profile_raises_in_torch_compile_region(self):
    handler = profiler.xprof_trace_handler(self.create_tempdir().full_path)

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with profiler.profile(
          activities=[profiler.ProfilerActivity.CPU],
          on_trace_ready=handler,
      ):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))


if __name__ == "__main__":
  absltest.main()
