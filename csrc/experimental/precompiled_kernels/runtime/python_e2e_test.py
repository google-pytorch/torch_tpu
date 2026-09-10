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

"""Check first-call results and actual precompiled execution from Python."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import testing


class PythonE2ETest(parameterized.TestCase):  # ABSLTEST_OK=no RNG

  @parameterized.named_parameters(
      ("relu", torch.relu, 1), ("add", torch.add, 2)
  )
  def test_first_execution(self, operation, arity):
    enabled = int(os.environ["TORCH_TPU_EXPERIMENTAL_PRECOMPILED"])
    x = (torch.arange(9 * 129).reshape(9, 129) % 17 - 8).float()
    cpu_inputs = (x, x + 0.25)[:arity]
    inputs = tuple(t.to("tpu") for t in cpu_inputs)
    torch.tpu.synchronize()

    before = testing.precompiled_execution_count()
    actual = operation(*inputs).cpu()

    ref = operation(*cpu_inputs)
    torch.testing.assert_close(actual, ref)  # TORCH_ASSERT_CLOSE_OK=exact
    self.assertEqual(testing.precompiled_execution_count(), before + enabled)


if __name__ == "__main__":
  absltest.main()
