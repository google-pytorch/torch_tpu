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

from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import tracer_utils
from torch_tpu._internal.utils import utils


class TpuTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError(f"absltest.FLAGS.test_random_seed not an int: {seed!r}")

    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)
    self.device = torch.device("tpu")
    torch.set_default_device(self.device)

  def test_replay_on_linear_no_change(self):
    """Tests replay on linear on TPU with no change."""
    # Arrange
    model = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
    with torch.no_grad():
      model.weight.fill_(3.0)
      model.bias.fill_(0.0)

    # Need device.type, not just device, because sometimes there's an
    # extra device.index.
    self.assertEqual(model.weight.device.type, self.device.type)

    # Act
    with utils.ActivationTracer(model) as tracer:
      model(torch.tensor([[1.0]], dtype=torch.bfloat16))
    log, _ = tracer.forward_log, tracer.forward_pre_log
    replayed_log = tracer_utils.replay_log(log, "cpu")

    # Assert
    original = log[0]["output"]
    replayed = replayed_log[0]["output"]  # pyrefly: ignore[bad-index]
    utils.assert_close(replayed, original.cpu())
    utils.assert_close(replayed.to(self.device), original)


if __name__ == "__main__":
  absltest.main()
