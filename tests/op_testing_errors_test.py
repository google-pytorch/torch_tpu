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

"""Error tests for the op_testing framework and assertions."""

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
import torch
from torch.testing._internal.common_methods_invocations import op_db
from tests import op_testing


class OpTestingErrorsTest(op_testing.TorchTpuVsCpuTestBase):
  """Tests error handling for tolerance checks and accuracy overrides."""

  def test_assert_close_tolerances_must_be_positive(self):
    """Tests that specifying rtol or atol <= 0 raises ValueError."""
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([1.0, 2.0])

    with self.assertRaisesRegex(ValueError, "rtol must be > 0"):
      self.assert_close(golden_result=t1, torch_tpu_result=t2, rtol=0)
    with self.assertRaisesRegex(ValueError, "rtol must be > 0"):
      self.assert_close(golden_result=t1, torch_tpu_result=t2, rtol=-1e-5)
    with self.assertRaisesRegex(ValueError, "atol must be > 0"):
      self.assert_close(golden_result=t1, torch_tpu_result=t2, atol=0)
    with self.assertRaisesRegex(ValueError, "atol must be > 0"):
      self.assert_close(golden_result=t1, torch_tpu_result=t2, atol=-1e-5)

    with self.assertRaisesRegex(ValueError, "rtol must be > 0"):
      self.assert_close_tpu_vs_cpu(
          lambda device: torch.tensor([1.0], device=device), rtol=0
      )
    with self.assertRaisesRegex(ValueError, "atol must be > 0"):
      self.assert_close_tpu_vs_cpu(
          lambda device: torch.tensor([1.0], device=device), atol=0
      )

    with self.assertRaisesRegex(ValueError, "rtol must be > 0"):
      self.set_accuracy_overrides(
          tpu_gpu_overrides={"op": {torch.float32: {"rtol": 0}}},
          grad_gpu_overrides={},
      )
    with self.assertRaisesRegex(ValueError, "atol must be > 0"):
      self.set_accuracy_overrides(
          tpu_gpu_overrides={"op": {torch.float32: {"atol": 0}}},
          grad_gpu_overrides={},
      )


class OpTestingFrameworkErrorsTest(op_testing.OpInfoTestBase):
  """Tests for the op_testing framework itself."""

  def test_torch_tpu_vs_gpu_missing_golden_fails(self):
    op = next(op for op in op_db if op.name == "add")
    with (
        flagsaver.flagsaver(test_mode=op_testing.TestMode.TORCH_TPU_VS_GPU),
        mock.patch.object(
            op_testing, "_GOLDEN_GPU_DATA", op_testing.GoldenGpuData()
        ),
    ):
      with self.assertRaisesRegex(
          AssertionError,
          "No GPU golden samples found for add.*Please re-generate the GPU"
          " golden files",
      ):
        self._get_golden_input_output_pairs(
            op=op,
            dtype=torch.float32,
            variant=op_testing.OpVariant.BASE,
            max_samples=1,
        )

      # Ensure that if samples are present, no error is raised and samples are
      # returned.
      fake_input = op_testing.OpInput(
          op_testing.SampleInput(torch.zeros(1), args=(), kwargs={}, name="s")
      )
      fake_output = op_testing.OpOutput(torch.zeros(1))
      fake_sample = (fake_input, fake_output)
      op_testing._GOLDEN_GPU_DATA.add(
          self._testMethodName,
          op_testing.OpVariant.BASE,
          torch.float32,
          fake_input,
          fake_output,
      )
      res = self._get_golden_input_output_pairs(
          op=op,
          dtype=torch.float32,
          variant=op_testing.OpVariant.BASE,
          max_samples=1,
      )
      self.assertEqual(res, [fake_sample])


if __name__ == "__main__":
  absltest.main()
