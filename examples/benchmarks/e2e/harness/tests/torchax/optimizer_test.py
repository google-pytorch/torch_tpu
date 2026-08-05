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

"""Tests for TorchAX optimizer mapping."""

from unittest import mock

from absl.testing import absltest
import optax
import torch
import torch.nn as nn
from examples.benchmarks.e2e.harness.torchax import optimizer
from tests import seed_test_utils


class OptimizerTest(seed_test_utils.RepeatableTest):

  @mock.patch.object(optax, "adam")
  def test_get_optax_optimizer_adam_params_match(self, mock_adam):
    model = nn.Linear(2, 2)
    torch_adam = torch.optim.Adam(
        model.parameters(), lr=0.005, betas=(0.8, 0.95), eps=1e-7
    )

    optimizer.get_optax_optimizer(torch_adam)

    mock_adam.assert_called_once_with(
        learning_rate=0.005, b1=0.8, b2=0.95, eps=1e-7
    )

  @mock.patch.object(optax, "adamw")
  @mock.patch.object(optax, "adam")
  def test_get_optax_optimizer_adamw_params_match(self, mock_adam, mock_adamw):
    model = nn.Linear(2, 2)
    torch_adamw = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        betas=(0.85, 0.98),
        eps=1e-6,
        weight_decay=0.02,
    )

    optimizer.get_optax_optimizer(torch_adamw)

    mock_adamw.assert_called_once_with(
        learning_rate=0.003,
        b1=0.85,
        b2=0.98,
        eps=1e-6,
        weight_decay=0.02,
    )
    mock_adam.assert_not_called()

  def test_get_optax_optimizer_returns_gradient_transformation(self):
    model = nn.Linear(2, 2)
    torch_adam = torch.optim.Adam(model.parameters(), lr=0.001)
    optax_adam = optimizer.get_optax_optimizer(torch_adam)
    self.assertIsInstance(optax_adam, optax.GradientTransformation)

    torch_adamw = torch.optim.AdamW(model.parameters(), lr=0.001)
    optax_adamw = optimizer.get_optax_optimizer(torch_adamw)
    self.assertIsInstance(optax_adamw, optax.GradientTransformation)

  def test_get_optax_optimizer_unsupported_raises(self):
    model = nn.Linear(2, 2)
    torch_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
    with self.assertRaisesRegex(
        ValueError, "Unsupported torch optimizer class: SGD"
    ):
      optimizer.get_optax_optimizer(torch_sgd)


if __name__ == "__main__":
  absltest.main()
