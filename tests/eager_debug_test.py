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

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode


class EagerDebugTest(absltest.TestCase):
  """Test for eager debug feature."""

  def test_failing_op_raises_in_correct_location(self):
    """Test that failing ops are not deferred."""
    # Act
    execution_mode.set_eager_mode(
        execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
    )

    # Arrange
    a = torch.rand((1_000_000_000, 1), device="tpu")
    b = torch.rand((1, 1_000), device="tpu")

    # Assert
    # The outerproduct of two tensors is 4 TB,
    # enough to trigger an OOM but not so large as to hit
    # int64 max value.

    # PyTorch on CPU gives this error:
    # ---------------------------------------------------------------------------
    # RuntimeError
    # Traceback (most recent call last)
    # /tmp/ipykernel_223/1573597190.py in <cell line: 0>()
    # ----> 1 a @ b
    #
    # RuntimeError: [enforce fail a t alloc_cpu.cpp:127] err == 0.
    # DefaultCPUAllocator:
    # can't allocate memory: you tried to allocate 4000000000000000000 bytes.
    # Error code 12 (Cannot allocate memory)

    with self.assertRaises(RuntimeError):
      # Act
      c = a @ b
      del c

    # If the outerproduct was deferred,
    # an OOM could be triggered by `c.sum().cpu()`.


if __name__ == "__main__":
  absltest.main()
