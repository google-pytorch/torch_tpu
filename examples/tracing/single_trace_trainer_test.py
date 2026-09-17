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

  def test_torch_compile_eval_step(self):
    model = ToyLinearModule().to(device=self.device).eval()
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=None,
        compile_fn=torch.compile,
    )

    x = torch.randn((16, 128), device=self.device)
    eval_step = trainer.make_compiled_eval_step(x)
    out = eval_step(x)

    with torch.no_grad():
      expected = model(x)
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)

  def test_torch_compile_no_optimizer_with_targets(self):
    model = ToyLinearModule().to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=None,
        compile_fn=torch.compile,
    )

    x = torch.randn((16, 128), device=self.device)
    target = torch.randn((16, 64), device=self.device)

    train_step = trainer.make_compiled_train_step(x, target)
    loss = train_step(x, target)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))
    for param in model.parameters():
      self.assertIsNotNone(param.grad)
      self.assertFalse(torch.isnan(param.grad).any())

  def test_torch_compile_buffers_update(self):
    class ToyBufferedModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
        self.register_buffer("running_mean", torch.zeros(64))

      def forward(self, x):
        if self.training:
          self.running_mean = self.running_mean * 0.9 + x.mean(dim=0) * 0.1
        return self.linear(x) + self.running_mean

    model = ToyBufferedModule().to(device=self.device).train()
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=None,
        compile_fn=torch.compile,
    )

    x1 = torch.randn((16, 64), device=self.device) * 2.0 + 3.0
    x2 = torch.randn((16, 64), device=self.device) * 4.0 - 1.0
    train_step = trainer.make_compiled_train_step(x1)

    initial_mean = model.running_mean.clone().cpu()
    loss1 = train_step(x1)
    mean_after_step1 = model.running_mean.clone().cpu()
    self.assertFalse(torch.allclose(initial_mean, mean_after_step1))

    loss2 = train_step(x2)
    mean_after_step2 = model.running_mean.clone().cpu()
    self.assertFalse(torch.allclose(mean_after_step1, mean_after_step2))
    self.assertFalse(torch.isnan(loss1))
    self.assertFalse(torch.isnan(loss2))

  def test_torch_compile_alternating_train_and_eval_with_buffers(self):
    """Verifies make_compiled_eval_step picks up buffer updates after train steps."""

    class ToyBufferedModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
        self.register_buffer("running_mean", torch.zeros(64))

      def forward(self, x):
        if self.training:
          self.running_mean = self.running_mean * 0.5 + x.mean(dim=0) * 0.5
        return self.linear(x) + self.running_mean

    model = ToyBufferedModule().to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=None,
        compile_fn=torch.compile,
    )

    x = torch.randn((16, 64), device=self.device) * 3.0 + 5.0
    model.train()
    train_step = trainer.make_compiled_train_step(x)
    model.eval()
    eval_step = trainer.make_compiled_eval_step(x)

    out_before = eval_step(x).clone().cpu()
    model.train()
    train_step(x)
    model.eval()
    out_after = eval_step(x).clone().cpu()
    self.assertFalse(torch.allclose(out_before, out_after))
    with torch.no_grad():
      expected = model(x).cpu()
    torch.testing.assert_close(out_after, expected, rtol=1e-4, atol=1e-4)

  def test_torch_compile_eval_step_noop_and_view(self):
    """Verifies zero-compute and view-only FX graphs execute accurately via host metadata."""
    dropout_model = torch.nn.Dropout(0.5).to(device=self.device).eval()
    trainer_noop = single_trace_trainer.SingleTraceTrainer(
        model=dropout_model,
        optimizer=None,
        compile_fn=torch.compile,
    )
    x = torch.randn((4, 8, 16), device=self.device)
    eval_noop = trainer_noop.make_compiled_eval_step(x)
    out_noop = eval_noop(x)
    torch.testing.assert_close(out_noop.cpu(), x.cpu())

    flatten_model = torch.nn.Flatten(start_dim=1).to(device=self.device).eval()
    trainer_view = single_trace_trainer.SingleTraceTrainer(
        model=flatten_model,
        optimizer=None,
        compile_fn=torch.compile,
    )
    eval_view = trainer_view.make_compiled_eval_step(x)
    out_view = eval_view(x)
    self.assertEqual(out_view.shape, (4, 128))
    torch.testing.assert_close(out_view.cpu(), x.flatten(1).cpu())

  def test_torch_compile_eval_step_with_optimizer(self):
    """Verifies make_compiled_eval_step works when an optimizer is configured."""
    model = ToyLinearModule().to(device=self.device)
    trainer = single_trace_trainer.SingleTraceTrainer(
        model=model,
        optimizer=optimizers.ReferenceAdamw(use_bfloat16_moments=False),
        compile_fn=torch.compile,
    )
    x = torch.randn((16, 128), device=self.device)
    target = torch.randn((16, 64), device=self.device)

    model.train()
    train_step = trainer.make_compiled_train_step(x, target)
    train_step(x, target)

    model.eval()
    eval_step = trainer.make_compiled_eval_step(x)
    out_comp = eval_step(x).cpu()
    with torch.no_grad():
      out_eager = torch.func.functional_call(model, trainer.params, (x,)).cpu()
    torch.testing.assert_close(out_comp, out_eager, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
