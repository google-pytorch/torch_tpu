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

"""Ensure we error out at torch.compile() when "dynamic" is applied."""

import os

from absl.testing import absltest
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile.compiler import has_dynamic_symints
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


def simple(x):
  a = 0.3 * x
  return a


class EnsureNoDynamicTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    torch.compiler.reset()

  def test_dynamic_none_and_not_size_change(self):
    # As long as it runs without raising error, we are good.
    input_1 = torch.tensor([0.1, 0.2], device=torch.device("tpu"))
    compiled = torch.compile(simple, backend="tpu")
    compiled(input_1).to("cpu")

  def test_dynamic_none_and_size_change_recompile_supported(self):
    input_1 = torch.tensor([0.1, 0.2])
    input_1_tpu = input_1.to(torch.device("tpu"))

    input_2 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    input_2_tpu = input_2.to(torch.device("tpu"))

    recompile_count = 0

    def count_recompile():
      nonlocal recompile_count
      recompile_count += 1
      return f"recompile: {recompile_count}"

    torch._dynamo.reset_recompile_user_contexts()
    self.addCleanup(torch._dynamo.reset_recompile_user_contexts)
    torch._dynamo.register_hook_for_recompile_user_context(count_recompile)

    compiled = torch.compile(simple, dynamic=None, backend="tpu")
    utils.assert_close(compiled(input_1_tpu).to("cpu"), simple(input_1))
    self.assertEqual(recompile_count, 1)

    # Shape change recompiles with bounded dynamism
    utils.assert_close(compiled(input_2_tpu).to("cpu"), simple(input_2))
    self.assertEqual(recompile_count, 2)

  def test_dynamic_false_and_size_change_recompile_supported(self):
    input_1 = torch.tensor([0.1, 0.2])
    input_1_tpu = input_1.to(torch.device("tpu"))

    input_2 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    input_2_tpu = input_2.to(torch.device("tpu"))

    compiled = torch.compile(simple, dynamic=False, backend="tpu")
    utils.assert_close(compiled(input_1_tpu).to("cpu"), simple(input_1))

    ## Shape change and recompile
    utils.assert_close(compiled(input_2_tpu).to("cpu"), simple(input_2))

  def test_default_dynamic_and_size_change_recompile_supported(self):
    input_1 = torch.tensor([0.1, 0.2])
    input_1_tpu = input_1.to(torch.device("tpu"))

    input_2 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    input_2_tpu = input_2.to(torch.device("tpu"))

    compiled = torch.compile(simple, backend="tpu")
    utils.assert_close(compiled(input_1_tpu).to("cpu"), simple(input_1))

    ## Shape change and recompile statically by default
    utils.assert_close(compiled(input_2_tpu).to("cpu"), simple(input_2))

  def test_concrete_symint_treated_as_static(self):
    """SymInt wrapping a concrete integer should not be flagged as dynamic."""
    shape_env = ShapeEnv()
    symint = shape_env.create_symintnode(sympy.Integer(32), hint=32)
    self.assertFalse(
        has_dynamic_symints([symint]),
        "Concrete SymInt (no free symbols) should be treated as static",
    )


if __name__ == "__main__":
  absltest.main()
