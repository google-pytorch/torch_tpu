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

"""Tests for compiled mode error handling without using Dynamo."""

from typing import TypeAlias

from absl.testing import absltest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal import execution_mode
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.compiler import StaticCompiler
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

EagerMode: TypeAlias = execution_mode.EagerMode


def eager_mode_compile_fx_graph():
  """Enable EagerMode._INTERNAL_COMPILE_FX_GRAPH."""
  return execution_mode.set_eager_mode(EagerMode._INTERNAL_COMPILE_FX_GRAPH)


class CompileApiTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_traverse_and_compile_with_forced_layout(self):
    input_tensor = torch.ones(2, 3, device=torch.device("tpu"))
    input_tensor.cpu()  # Force materialization
    # Case 2: Force non-default layout [0, 1]. Execution should FAIL with
    # default input.
    with eager_mode_compile_fx_graph():
      x2 = torch.ones(2, 3, device="cpu").to(device=torch.device("tpu"))
      z2 = x2 + x2

    compile_result_fail = tpu_torch_compile.traverse_and_compile(
        [z2], [x2], argument_layouts=[[0, 1]]
    )
    self.assertIsNotNone(compile_result_fail.executable)

    results_fail = tpu_torch_compile.execute(
        compile_result_fail.executable, [input_tensor]
    )
    self.assertLen(results_fail, 1)

    with self.assertRaises(RuntimeError) as err:
      results_fail[0].cpu()

    self.assertIn("layout", str(err.exception).lower())
    tt_testing.reset_eager_state()

  def test_static_compiler_buffer_donation(self):
    def simple_fn(x, y):
      return x + y, x * 2

    x = torch.randn(2, 3, device=self.device)
    y = torch.randn(2, 3, device=self.device)

    gm = make_fx(simple_fn)(x, y)

    debug: TpuCompileDebug = TpuCompileDebug()
    comp = StaticCompiler(debug=debug)

    donated_inputs = [0]
    compiled_executable = comp(gm, [x, y], donated_inputs=donated_inputs)

    self.assertIsNotNone(compiled_executable)
    self.assertLen(debug.stablehlo_forward_text, 1)
    self.assertIn("jax.buffer_donor = true", debug.stablehlo_forward_text[0])

    x_cpu = x.cpu().clone()
    y_cpu = y.cpu().clone()
    expected_add = x_cpu + y_cpu
    expected_mul = x_cpu * 2

    res = compiled_executable(x, y)

    utils.assert_close(res[0].cpu(), expected_add)
    utils.assert_close(res[1].cpu(), expected_mul)

    with self.assertRaisesRegex(
        RuntimeError,
        r"to_copy\(\): cannot start a device to host copy from a donated or"
        r" deleted buffer",
    ):
      x.cpu()

    # y should still be accessible
    y.cpu()


if __name__ == "__main__":
  absltest.main()
