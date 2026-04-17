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


class H2DSyncTest(absltest.TestCase):

  def test_h2d_sync_race(self):
    """Verifies that torch.tpu.synchronize() properly waits for H2D transfers.

    If synchronize() fails to wait, modifying the host tensor immediately
    after the call may corrupt the data transferred to the TPU.
    """
    size = 4096

    # 1. Use pinned memory for async H2D transfer
    cpu_tensor = torch.ones(
        size, size, dtype=torch.float32, device="cpu"
    ).pin_memory()

    # Warmup
    _ = cpu_tensor.to("tpu", non_blocking=True)
    torch.tpu.synchronize()

    # Reset host tensor to 1s
    cpu_tensor.fill_(1.0)

    # 2. Enqueue async H2D transfer
    tpu_tensor = cpu_tensor.to("tpu", non_blocking=True)

    # 3. Synchronize
    # (Should wait for the H2D future registered via MarkStreamActive)
    torch.tpu.synchronize()

    # 4. Modify the host tensor immediately after synchronize
    cpu_tensor.fill_(2.0)

    # 5. Read back the TPU tensor to verify integrity
    result_tensor = tpu_tensor.cpu()

    # The TPU tensor must contain the original 1.0s.
    mean_val = result_tensor.mean().item()
    print(f"\n[RESULT] H2D transferred tensor mean: {mean_val:.4f}")

    self.assertEqual(
        mean_val,
        1.0,
        msg=(
            "H2D race condition detected! synchronize() failed to wait for"
            " transfer."
        ),
    )


if __name__ == "__main__":
  absltest.main()
