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

"""Unit tests for ops."""

import concurrent

from absl.testing import absltest
import torch
from torch_tpu import api
from torch_tpu._internal import compiler_options as compiler
from tests import op_testing


class CompilerOptionTest(op_testing.TorchTpuTestBase):
  """Unit tests for setting compiler options."""

  def setUp(self):
    super().setUp()
    self.tpu = api.tpu_device()

  def test_push_compiler_options_does_not_affect_result(self):
    """Tests setting compiler options.

    This is just a smoke test to make sure that setting the compiler options
    doesn't crash or affect the result.
    """

    with compiler.custom_compiler_options({
        "xla_tpu_enable_deduplicated_calls": "DISABLED",
    }):
      result = torch.matmul(
          torch.ones((2, 2), device=self.tpu),
          torch.ones((2, 2), device=self.tpu),
      ).to(
          "cpu"
      )  # to() triggers compilation.

    self.assert_close(
        golden_result=torch.tensor([[2, 2], [2, 2]], dtype=torch.float32),
        torch_tpu_result=result,
    )

  def test_push_compiler_options_with_opt_level(self):
    """Tests setting compiler options with optimization level.

    This is just a smoke test to make sure that setting the compiler options
    doesn't crash or affect the result.
    """

    with compiler.custom_compiler_options({
        "xla_optimization_level": "O1",
        "xla_memory_fitting_level": "O0",
    }):
      result = torch.matmul(
          torch.ones((2, 2), device=self.tpu),
          torch.ones((2, 2), device=self.tpu),
      ).to(
          "cpu"
      )  # to() triggers compilation.

    self.assert_close(
        golden_result=torch.tensor([[2, 2], [2, 2]], dtype=torch.float32),
        torch_tpu_result=result,
    )

  def test_push_compiler_options_in_threads(self):
    """Tests setting compiler options concurrently doesn't affect the result."""

    compiler_option_dicts = [
        {"xla_tpu_enable_deduplicated_calls": "AUTO"},
        {"xla_tpu_enable_deduplicated_calls": "ENABLED"},
        {"xla_tpu_enable_deduplicated_calls": "DISABLED"},
    ]

    def run_op(op, index):
      with compiler.custom_compiler_options(
          compiler_option_dicts[index % len(compiler_option_dicts)]
      ):
        arg = torch.tensor(2, dtype=torch.float32, device=api.tpu_device())
        res = op(arg)
        return res.to("cpu").item()  # to() triggers compilation.

    # Start 100 threads to run log2 and exp concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
      log2_futures = [
          executor.submit(run_op, torch.log2, i) for i in range(100)
      ]
      exp_futures = [executor.submit(run_op, torch.exp, i) for i in range(100)]
      for future in log2_futures:
        self.assertEqual(future.result(), 1)
      for future in exp_futures:
        self.assertEqual(future.result(), 7.389056)

  def test_nested_compiler_options_in_threads(self):
    """Tests setting compiler options in nested contexts concurrently."""

    compiler_option_dicts1 = [
        {"xla_tpu_enable_deduplicated_calls": "AUTO"},
        {"xla_tpu_enable_deduplicated_calls": "ENABLED"},
        {"xla_tpu_enable_deduplicated_calls": "DISABLED"},
    ]
    compiler_option_dicts2 = [
        {"xla_tpu_enable_deduplicated_calls": "ENABLED"},
        {"xla_optimization_level": "O1"},
        {"xla_memory_fitting_level": "O0"},
    ]

    def run_op(op, index):
      with compiler.custom_compiler_options(
          compiler_option_dicts1[index % len(compiler_option_dicts1)]
      ):
        with compiler.custom_compiler_options(
            compiler_option_dicts2[index % len(compiler_option_dicts2)]
        ):
          arg = torch.tensor(2, dtype=torch.float32, device=api.tpu_device())
          res = op(arg)
          return res.to("cpu").item()  # to() triggers compilation.

    # Start 100 threads to run log2 and exp concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
      log2_futures = [
          executor.submit(run_op, torch.log2, i) for i in range(100)
      ]
      exp_futures = [executor.submit(run_op, torch.exp, i) for i in range(100)]
      for future in log2_futures:
        self.assertEqual(future.result(), 1)
      for future in exp_futures:
        self.assertEqual(future.result(), 7.389056)

  def test_excess_precision_cache_conflict(self):
    """Tests excess precision flag."""
    device = api.tpu_device()
    input_tensor = torch.tensor(
        [-1.3594, 5.6875],
        dtype=torch.bfloat16,
        device=device,
    )

    def compute():
      return input_tensor.pow(2).sum(-1)

    with compiler.custom_compiler_options(
        {"xla_allow_excess_precision": "false"}
    ):
      with compiler.custom_compiler_options(
          {"xla_allow_excess_precision": "true"}
      ):
        self.assertEqual(compute().cpu().item(), 34.25)
      # Ensure it does not reuse the compiled cache when exiting the context
      # as the compiler setting changed.
      self.assertEqual(compute().cpu().item(), 34.00)
    self.assertEqual(compute().cpu().item(), 34.25)


if __name__ == "__main__":
  absltest.main()
