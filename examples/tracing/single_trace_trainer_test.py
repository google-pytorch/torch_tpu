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

"""Sanity check tests for standalone SingleTraceTrainer using torch.compile."""

from absl.testing import absltest
import torch
import torch_tpu
from examples.tracing import optimizers
from examples.tracing import single_trace_trainer
from tests import seed_test_utils


class ToyLinearModule(torch.nn.Module):
  """Simple toy module with a single linear layer."""

  def __init__(self, in_features=128, out_features=64):
    super().__init__()
    self.linear = torch.nn.Linear(in_features, out_features)

  def forward(self, x):
    return self.linear(x)


class SingleTraceTrainerStandaloneSanityTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.device = torch.device("tpu")

  def test_torch_compile_adamw_step(self):
    model = ToyLinearModule().to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=optimizers.ReferenceAdamw(use_bfloat16_moments=False),
        compile_fn=torch.compile,
    )

    x = torch.randn((16, 128), device=self.device)
    target = torch.randn((16, 64), device=self.device)

    train_step = trainer.make_compiled_train_step(x, target)
    loss = train_step(x, target)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))

  def test_torch_compile_sgd_step(self):
    model = ToyLinearModule().to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=optimizers.ReferenceSgd(lr=1e-2, momentum=0.9),
    )

    x = torch.randn((16, 128), device=self.device)
    target = torch.randn((16, 64), device=self.device)

    # Pass compile_fn directly to make_compiled_train_step
    train_step = trainer.make_compiled_train_step(
        x, target, compile_fn=torch.compile
    )
    loss = train_step(x, target)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))

  def test_torch_compile_no_targets(self):
    model = ToyLinearModule().to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=optimizers.ReferenceAdamw(),
        compile_fn=torch.compile,
    )

    x = torch.randn((16, 128), device=self.device)
    train_step = trainer.make_compiled_train_step(x)
    loss = train_step(x)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
  absltest.main()
