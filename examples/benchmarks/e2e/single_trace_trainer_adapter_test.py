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

"""Tests for SingleTraceTrainerAdapter using StaticCompiler on TPU."""

import copy
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from examples.benchmarks.e2e import single_trace_trainer_adapter
from tests import seed_test_utils


class ToyLinearModule(torch.nn.Module):
  """Simple toy module with a single linear layer."""

  def __init__(self, in_features=128, out_features=64):
    super().__init__()
    self.linear = torch.nn.Linear(in_features, out_features)

  def forward(self, x):
    return self.linear(x)


class SingleTraceTrainerAdapterEagerCompareTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.device = torch.device("tpu")

  def _get_test_models(
      self, in_features=128, out_features=64, dtype=torch.float32
  ):
    model = ToyLinearModule(in_features, out_features)
    model.train()
    model_eager = copy.deepcopy(model)
    model_eager.train()
    return model.to(device=self.device, dtype=dtype), model_eager.to(
        device=self.device, dtype=dtype
    )

  def _assert_params_and_state_close(
      self,
      trainer,
      model_eager,
      optimizer_eager,
  ):
    for safe_name, p_comp in trainer.param_group.params.items():
      name = trainer._key_map[safe_name]
      p_eager = dict(model_eager.named_parameters())[name]

      test_utils.assert_close(
          p_comp,
          p_eager,
          rtol=1e-4,
          atol=1e-4,
      )

      if isinstance(optimizer_eager, torch.optim.AdamW):
        if safe_name in trainer.param_group.opt_state_m:
          m_comp = trainer.param_group.opt_state_m[safe_name]
          m_eager = optimizer_eager.state[p_eager]["exp_avg"]
          test_utils.assert_close(
              m_comp,
              m_eager,
              rtol=1e-4,
              atol=1e-4,
          )

        if safe_name in trainer.param_group.opt_state_v:
          v_comp = trainer.param_group.opt_state_v[safe_name]
          v_eager = optimizer_eager.state[p_eager]["exp_avg_sq"]
          test_utils.assert_close(
              v_comp,
              v_eager,
              rtol=1e-4,
              atol=1e-4,
          )

      elif isinstance(optimizer_eager, torch.optim.SGD):
        if safe_name in trainer.param_group.opt_state_m:
          m_comp = trainer.param_group.opt_state_m[safe_name]
          if "momentum_buffer" in optimizer_eager.state[p_eager]:
            m_eager = optimizer_eager.state[p_eager]["momentum_buffer"]
            test_utils.assert_close(
                m_comp,
                m_eager,
                rtol=1e-4,
                atol=1e-4,
            )

  def _verify_train_step_vs_eager(
      self, trainer, model_eager, x, target, ref_opt_cls, **ref_opt_kwargs
  ):
    optimizer_eager = ref_opt_cls(model_eager.parameters(), **ref_opt_kwargs)
    train_step = trainer.make_compiled_train_step(x, target)

    # ===== Step 1 =====
    loss = train_step(x, target)
    optimizer_eager.zero_grad()
    out_eager = model_eager(x)
    loss_eager = torch.nn.functional.mse_loss(out_eager, target)
    loss_eager.backward()
    optimizer_eager.step()

    torch.accelerator.synchronize()
    test_utils.assert_close(loss, loss_eager, rtol=1e-4, atol=1e-4)

    self._assert_params_and_state_close(trainer, model_eager, optimizer_eager)

    # ===== Step 2 =====
    loss = train_step(x, target)
    optimizer_eager.zero_grad()
    out_eager = model_eager(x)
    loss_eager = torch.nn.functional.mse_loss(out_eager, target)
    loss_eager.backward()
    optimizer_eager.step()

    torch.accelerator.synchronize()
    test_utils.assert_close(loss, loss_eager, rtol=1e-4, atol=1e-4)

    self._assert_params_and_state_close(
        trainer,
        model_eager,
        optimizer_eager,
    )

  def test_linear_layer_training_compiled_reference_adamw(self):
    model, model_eager = self._get_test_models()

    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model,
        single_trace_trainer_adapter.ReferenceAdamw(use_bfloat16_moments=False),
    )

    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    self._verify_train_step_vs_eager(
        trainer,
        model_eager,
        x,
        target,
        torch.optim.AdamW,
    )

  def test_linear_layer_training_compiled_torch_adamw(self):
    model, model_eager = self._get_test_models()

    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model,
        single_trace_trainer_adapter.TorchAdamw(use_bfloat16_moments=False),
    )

    self._verify_train_step_vs_eager(
        trainer,
        model_eager,
        x,
        target,
        torch.optim.AdamW,
    )

  def test_linear_layer_training_compiled_reference_sgd(self):
    model, model_eager = self._get_test_models()
    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model,
        single_trace_trainer_adapter.ReferenceSgd(lr=1e-2, momentum=0.9),
    )
    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    self._verify_train_step_vs_eager(
        trainer,
        model_eager,
        x,
        target,
        torch.optim.SGD,
        lr=1e-2,
        momentum=0.9,
    )

  def test_linear_layer_training_compiled_torch_sgd(self):
    model, model_eager = self._get_test_models()
    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, single_trace_trainer_adapter.TorchSgd(lr=1e-2, momentum=0.9)
    )
    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    self._verify_train_step_vs_eager(
        trainer,
        model_eager,
        x,
        target,
        torch.optim.SGD,
        lr=1e-2,
        momentum=0.9,
    )


class SingleTraceTrainerAdapterSignatureTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def test_with_targets(self):
    model = ToyLinearModule().to(device=self.device)

    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, single_trace_trainer_adapter.ReferenceAdamw()
    )

    train_step = trainer.make_compiled_train_step(x, target)
    self.assertIsNotNone(train_step(x, target))

  def test_no_targets(self):
    model = ToyLinearModule().to(device=self.device)

    x = torch.randn((32, 128), device=self.device)

    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, single_trace_trainer_adapter.ReferenceAdamw()
    )

    train_step = trainer.make_compiled_train_step(x)
    self.assertIsNotNone(train_step(x))

  def test_traced_with_targets_none_for_step(self):
    model = ToyLinearModule().to(device=self.device)

    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, single_trace_trainer_adapter.ReferenceAdamw()
    )

    train_step = trainer.make_compiled_train_step(x, target)

    with self.assertRaises(  # ASSERT_RAISES_OK=Testing mismatched signature error in example test
        AssertionError
    ):
      train_step(x)

  def test_traced_without_targets_provided_for_step(self):
    model = ToyLinearModule().to(device=self.device)

    x = torch.randn((32, 128), device=self.device)
    target = torch.randn((32, 64), device=self.device)

    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, single_trace_trainer_adapter.ReferenceAdamw()
    )

    train_step = trainer.make_compiled_train_step(x)

    with self.assertRaises(  # ASSERT_RAISES_OK=Testing mismatched signature error in example test
        AssertionError
    ):
      train_step(x, target)


class SingleTraceTrainerAdapterBufferMutationTest(
    seed_test_utils.RepeatableTest
):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def test_model_inplace_buf_update(self):
    class Model(torch.nn.Module):

      def __init__(self):
        super().__init__()

        self.register_buffer("add_count", torch.tensor(0))
        self.linear = torch.nn.Linear(128, 64)

      def forward(self, x):
        self.add_count.add_(1)
        return self.linear(x) + 1

    model = Model().to(self.device)
    self.assertEqual(model.add_count, 0)
    x = torch.randn((32, 128), device=self.device)

    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, single_trace_trainer_adapter.ReferenceAdamw()
    )

    train_step = trainer.make_compiled_train_step(x)
    _ = train_step(x)
    self.assertEqual(model.add_count, 1)


class SingleTraceTrainerAdapterTiedWeightsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def test_tied_weights_deduplication(self):
    class TiedModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(128, 64)
        self.lm_head = torch.nn.Linear(64, 128, bias=False)
        self.tie_weights()

      def tie_weights(self):
        self.lm_head.weight = self.embed.weight

      def forward(self, x):
        h = self.embed(x)
        return self.lm_head(h)

    model = TiedModel().to(self.device)
    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, single_trace_trainer_adapter.ReferenceAdamw()
    )
    # Verify that tied weights resulted in only one unique parameter entry
    self.assertEqual(len(trainer.param_group.params), 1)
    self.assertIn("embed_weight", trainer.param_group.params)
    self.assertEqual(
        list(trainer.params.keys()),
        ["embed.weight"],
    )

    x = torch.randint(0, 128, (4, 16), device=self.device)
    train_step = trainer.make_compiled_train_step(x)
    for _ in range(3):
      loss = train_step(x)
      self.assertFalse(torch.isnan(loss))
      # Ensure model parameters in both submodules stay synchronized and tied
      self.assertIs(model.embed.weight, model.lm_head.weight)


class SingleTraceTrainerAdapterFwdBwdOnlyTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.device = torch.device("tpu")

  def test_linear_layer_fwd_bwd_only_compiled_vs_eager(self):
    model = ToyLinearModule().to(device=self.device)
    model.train()
    model_eager = copy.deepcopy(model)
    model_eager.train()

    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, optimizer=None
    )

    x = torch.randn((32, 128), device=self.device)

    train_step = trainer.make_compiled_train_step(x)

    # Compiled step
    loss_comp = train_step(x)

    # Eager step
    out_eager = model_eager(x)
    loss_eager = torch.mean(out_eager)
    loss_eager.backward()

    torch.accelerator.synchronize()
    test_utils.assert_close(loss_comp, loss_eager, rtol=1e-4, atol=1e-4)

    # Check gradients match
    for name, param in model.named_parameters():
      param_eager = dict(model_eager.named_parameters())[name]
      self.assertIsNotNone(param.grad)
      self.assertIsNotNone(param_eager.grad)
      test_utils.assert_close(
          param.grad, param_eager.grad, rtol=1e-4, atol=1e-4
      )

  def test_parameterless_module_compiled(self):
    model = torch.nn.Tanh().to(device=self.device)
    trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model, optimizer=None
    )

    x = torch.randn((32, 128), device=self.device)
    train_step = trainer.make_compiled_train_step(x)
    loss = train_step(x)

    self.assertIsNotNone(loss)
    self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
  absltest.main()
