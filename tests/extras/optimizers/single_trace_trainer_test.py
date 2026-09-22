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
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from torch_tpu.extras.optimizers import adamw
from torch_tpu.extras.optimizers import sgd
from torch_tpu.extras.optimizers import single_trace_trainer
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
        optimizer=adamw.ReferenceAdamw(use_bfloat16_moments=False),
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
        optimizer=sgd.ReferenceSgd(lr=1e-2, momentum=0.9),
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
        optimizer=adamw.ReferenceAdamw(),
        compile_fn=torch.compile,
    )

    x = torch.randn((16, 128), device=self.device)
    train_step = trainer.make_compiled_train_step(x)
    loss = train_step(x)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))

  def test_torch_compile_no_optimizer(self):
    model = ToyLinearModule().to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=None,
        compile_fn=torch.compile,
    )

    x = torch.randn((16, 128), device=self.device)
    train_step = trainer.make_compiled_train_step(x)
    loss = train_step(x)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))
    for param in model.parameters():
      self.assertIsNotNone(param.grad)
      self.assertFalse(torch.isnan(param.grad).any())

  def test_torch_compile_parameterless_module(self):
    model = torch.nn.Tanh().to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=None,
        compile_fn=torch.compile,
    )

    x = torch.randn((16, 128), device=self.device)
    train_step = trainer.make_compiled_train_step(x)
    loss = train_step(x)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))

  def test_torch_compile_sparse_embedding_no_optimizer(self):
    model = torch.nn.Embedding(128, 64, sparse=True).to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=None,
        compile_fn=torch.compile,
    )

    x = torch.randint(0, 128, (16, 8), device=self.device)
    target = torch.randn((16, 8, 64), device=self.device)

    train_step = trainer.make_compiled_train_step(x, target)
    loss = train_step(x, target)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))
    self.assertIsNotNone(model.weight.grad)
    self.assertTrue(model.weight.grad.is_sparse)
    self.assertFalse(torch.isnan(model.weight.grad._values()).any())  # pylint: disable=protected-access

  def test_torch_compile_sparse_embedding_bag_no_optimizer(self):
    model = torch.nn.EmbeddingBag(128, 64, sparse=True, mode="sum").to(
        device=self.device
    )
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=None,
        compile_fn=torch.compile,
    )

    x = torch.randint(0, 128, (16, 8), device=self.device)
    target = torch.randn((16, 64), device=self.device)

    train_step = trainer.make_compiled_train_step(x, target)
    loss = train_step(x, target)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))
    self.assertIsNotNone(model.weight.grad)
    self.assertTrue(model.weight.grad.is_sparse)
    self.assertFalse(torch.isnan(model.weight.grad._values()).any())  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
