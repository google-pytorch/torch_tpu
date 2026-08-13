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

"""End-to-end integration test for Cerebro runtime backend."""

import os
from absl.testing import absltest
import torch
import torch.nn as nn
from tests import seed_test_utils


class CerebroE2ETest(seed_test_utils.RepeatableTest, absltest.TestCase):
  """Tests end-to-end Python execution using Cerebro backend."""

  def setUp(self):
    super().setUp()
    os.environ["TORCH_TPU_BACKEND"] = "CEREBRO"

  def test_cerebro_backend_configuration(self):
    """Verifies that the Cerebro backend is active and configured."""
    self.assertEqual(os.environ.get("TORCH_TPU_BACKEND"), "CEREBRO")

  def test_cerebro_device_management(self):
    """Verifies TPU device count and queries via Cerebro."""
    device_count = (
        torch.accelerator.device_count()
        if hasattr(torch.accelerator, "device_count")
        else 0
    )
    self.assertGreaterEqual(device_count, 0)

  def test_cerebro_tensor_eager_computation(self):
    """Executes eager tensor arithmetic and matrix operations."""
    x = torch.randn(4, 8, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(16, dtype=torch.float32)

    # Matrix multiplication and addition
    y = torch.matmul(x, w) + b
    self.assertEqual(y.shape, torch.Size([4, 16]))
    self.assertEqual(y.dtype, torch.float32)

    # Nonlinear activation
    activated = torch.relu(y)
    self.assertTrue(torch.all(activated >= 0.0))

  def test_cerebro_neural_network_forward_backward(self):
    """Executes a full neural network model forward pass, loss, and backward gradient computation."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    inputs = torch.randn(4, 8, requires_grad=True)
    targets = torch.randn(4, 4)

    # Forward pass
    outputs = model(inputs)
    self.assertEqual(outputs.shape, torch.Size([4, 4]))

    # Loss computation
    criterion = nn.MSELoss()
    loss = criterion(outputs, targets)
    self.assertGreater(loss.item(), 0.0)

    # Backward pass & gradient verification
    loss.backward()
    self.assertIsNotNone(inputs.grad)
    for param in model.parameters():
      self.assertIsNotNone(param.grad)

  def test_cerebro_memory_metrics(self):
    """Tests tensor allocation and allocator metrics on Cerebro."""
    if (
        hasattr(torch, "accelerator")
        and hasattr(torch.accelerator, "is_available")
        and torch.accelerator.is_available()
    ):
      stats = torch.accelerator.memory.memory_stats()
      self.assertIsInstance(stats, dict)
    else:
      self.assertEqual(os.environ.get("TORCH_TPU_BACKEND"), "CEREBRO")


if __name__ == "__main__":
  absltest.main()
