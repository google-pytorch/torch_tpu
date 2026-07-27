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

"""Tests for loss."""

from absl.testing import absltest
import torch
import torch.nn as nn
from examples.benchmarks.e2e.harness import loss as loss_lib


class _LossModel(nn.Module):
  """Returns a scalar loss directly."""

  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(4, 3)

  def forward(self, x, labels=None):
    logits = self.lin(x)
    if labels is None:
      return logits
    return nn.functional.cross_entropy(logits, labels)


class _ModelOutput:
  """Stand-in for an HF ModelOutput with a .loss attribute."""

  def __init__(self, loss, logits):
    self.loss = loss
    self.logits = logits


class LossTest(absltest.TestCase):

  def test_real_loss_scalar_tensor_output(self):
    model = _LossModel()
    loss = loss_lib.real_loss(
        model, (torch.randn(2, 4),), {"labels": torch.tensor([0, 1])}
    )
    self.assertEqual(loss.ndim, 0)

  def test_real_loss_model_output_with_loss_attr(self):
    class M(nn.Module):

      def forward(self, x):
        return _ModelOutput(loss=x.sum(), logits=x)

    self.assertEqual(
        loss_lib.real_loss(M(), (torch.ones(2, 2),), {}).item(), 4.0
    )

  def test_real_loss_dict_with_loss_key(self):
    class M(nn.Module):

      def forward(self, x):
        return {"loss": x.sum(), "logits": x}

    self.assertEqual(
        loss_lib.real_loss(M(), (torch.ones(2, 2),), {}).item(), 4.0
    )

  def test_real_loss_tuple_loss_first(self):
    class M(nn.Module):

      def forward(self, x):
        return (x.sum(), x)

    self.assertEqual(
        loss_lib.real_loss(M(), (torch.ones(2, 2),), {}).item(), 4.0
    )

  def test_real_loss_per_token_loss_is_meaned(self):
    class M(nn.Module):

      def forward(self, x):
        return {"loss": torch.tensor([1.0, 3.0])}

    self.assertAlmostEqual(
        loss_lib.real_loss(M(), (torch.ones(1),), {}).item(), 2.0
    )

  def test_real_loss_raises_when_no_loss_found(self):
    class M(nn.Module):

      def forward(self, x):
        return {"logits": x}

    with self.assertRaises(TypeError):
      loss_lib.real_loss(M(), (torch.ones(2, 2),), {})


if __name__ == "__main__":
  absltest.main()
