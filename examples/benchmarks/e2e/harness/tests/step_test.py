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

"""Tests for step."""

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
import torch
import torch.nn as nn
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import discovery as discovery_lib
from examples.benchmarks.e2e.harness import measure as measure_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import steps
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness import torch_device_ops
from examples.benchmarks.e2e.harness.steps import decode
from tests import seed_test_utils
import transformers


def mock_grad_probe(model, args, kwargs):
  return model(*args, **kwargs).sum()


def _mlp(in_dim=4, hidden=8, out_dim=3, seed=0):
  torch.manual_seed(seed)
  return nn.Sequential(
      nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
  )


discovery_lib.import_submodules(steps)


class StepperTypeTest(seed_test_utils.RepeatableTest):

  def test_all_steps_registered(self):
    self.assertEqual(
        set(step_lib.STEPPERS),
        {
            step_lib.StepperType.FORWARD,
            step_lib.StepperType.TRAINING,
            step_lib.StepperType.DECODER_ONLY_DECODE,
        },
    )

  def test_uniform_factory_shape(self):
    """Every entry is a factory that returns a callable."""
    for name in step_lib.StepperType:
      self.assertTrue(callable(step_lib.STEPPERS[name]))

  def test_training_bad_accum_steps_rejected(self):
    with self.assertRaises(ValueError):
      step_lib.resolve_stepper(step_lib.StepperType.TRAINING, accum_steps=0)

  def test_forward_runs_under_no_grad(self):
    model = _mlp()
    stepper = step_lib.resolve_stepper(step_lib.StepperType.FORWARD)
    stepper.init_with_benchmark_args(model, (torch.randn(2, 4),), {})
    out = stepper.get_step_fn()()
    self.assertFalse(
        out.requires_grad, "Expected forward to run under inference_mode"
    )

  def test_forward_splats_args_and_kwargs(self):
    model = mock.MagicMock()
    model.return_value = torch.tensor(1.0)
    stepper = step_lib.resolve_stepper(step_lib.StepperType.FORWARD)
    x, y = torch.randn(2, 4), torch.tensor([0, 1])
    stepper.init_with_benchmark_args(model, (x,), {"labels": y})
    out = stepper.get_step_fn()()
    self.assertEqual(out, torch.tensor(1.0))
    model.assert_called_once_with(x, labels=y)

  def test_forward_leaves_params_unchanged(self):
    model = _mlp()
    stepper = step_lib.resolve_stepper(step_lib.StepperType.FORWARD)
    before = [p.clone() for p in model.parameters()]
    stepper.init_with_benchmark_args(model, (torch.randn(2, 4),), {})
    stepper.get_step_fn()()
    for b, p in zip(before, model.parameters()):
      self.assertTrue(torch.equal(b, p))

  def _grads_after_training_step(self, accum_steps, batches, lr=0.0):
    """Run one training step and return the grads that were computed"""
    model = _mlp(seed=1234)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    def compute_loss(m, args, kwargs):  # pylint: disable=unused-argument
      # Deterministic per-call: pull the next micro-batch off the closure list.
      inputs, labels = kwargs["batches"].pop(0)
      return nn.functional.cross_entropy(m(inputs), labels)

    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING,
        accum_steps=accum_steps,
        compute_loss=compute_loss,
    )
    stepper.init_with_benchmark_args(model, (), {"batches": list(batches)}, opt)
    stepper.get_step_fn()()
    return [p.grad.clone() for p in model.parameters()]

  def test_accum_b_matches_accum_1_on_full_batch(self):
    """accum_steps=B over B micro-batches of size 1 must equal accum_steps=1 over the full batch of size B."""
    torch.manual_seed(7)
    B = 4
    inputs = torch.randn(B, 4)
    labels = torch.randint(0, 3, (B,))

    # accum=1 over the whole batch: mean CE over B samples.
    full = self._grads_after_training_step(1, [(inputs, labels)])

    # accum=B over B micro-batches of 1: each is a mean over 1 sample, divided by B.
    micro = self._grads_after_training_step(
        B, [(inputs[i : i + 1], labels[i : i + 1]) for i in range(B)]
    )

    for grad_full, grad_micro in zip(full, micro):
      torch.testing.assert_close(grad_full, grad_micro, rtol=1e-5, atol=1e-6)
      self.assertGreater(torch.norm(grad_full).item(), 0.0)
      self.assertGreater(torch.norm(grad_micro).item(), 0.0)

  def test_training_updates_params(self):
    """Catches when the optimizer is bound to stale tensors."""
    model = _mlp(seed=99)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING, compute_loss=mock_grad_probe
    )
    before = [p.clone() for p in model.parameters()]
    stepper.init_with_benchmark_args(model, (torch.randn(4, 4),), {}, opt)
    stepper.get_step_fn()()
    changed = [
        not torch.equal(b, p) for b, p in zip(before, model.parameters())
    ]
    self.assertTrue(
        all(changed), "Expected training params to change after training."
    )

  def test_training_one_step_is_one_optimizer_update(self):
    model = _mlp()
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING,
        accum_steps=3,
        compute_loss=mock_grad_probe,
    )

    class CountingSGD(torch.optim.SGD):
      steps = 0

      def step(self, *a, **kw):
        CountingSGD.steps += 1
        return super().step(*a, **kw)

    opt = CountingSGD(model.parameters(), lr=0.01)
    stepper.init_with_benchmark_args(model, (torch.randn(2, 4),), {}, opt)
    stepper.get_step_fn()()
    self.assertEqual(
        CountingSGD.steps,
        1,
        (
            "Expected one optimizer step per training step, got "
            f"{CountingSGD.steps}"
        ),
    )

  def test_training_grads_zeroed_each_step(self):
    """Without zero_grad, gradients accumulate across timed steps."""
    model = _mlp()
    # Learning rate is set to 0.0 to prevent the model parameters from updating.
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING, compute_loss=mock_grad_probe
    )
    x = torch.randn(4, 4)
    stepper.init_with_benchmark_args(model, (x,), {}, opt)
    stepper.get_step_fn()()
    first = [p.grad.clone() for p in model.parameters()]
    stepper.get_step_fn()()
    second = [p.grad.clone() for p in model.parameters()]
    for a, b in zip(first, second):
      torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

  def test_training_requires_optimizer(self):
    stepper = step_lib.resolve_stepper(step_lib.StepperType.TRAINING)
    with self.assertRaises(TypeError):
      stepper.init_with_benchmark_args(_mlp(), (torch.randn(2, 4),), {})

  def test_compute_loss_override_is_used(self):
    my_loss = mock.MagicMock(return_value=torch.tensor(1.0, requires_grad=True))

    model = _mlp()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING, compute_loss=my_loss
    )
    args = (torch.randn(2, 4),)
    kwargs = {}
    stepper.init_with_benchmark_args(model, args, kwargs, opt)
    stepper.get_step_fn()()
    my_loss.assert_called_once_with(model, args, kwargs)

  def _ops(self):
    return torch_device_ops.TorchDeviceOps(
        target_lib.make_target(target_lib.Platform.V5E_1X1)
    )

  def test_inference_through_measure(self):
    ops = self._ops()
    model, x = _mlp().to(ops.device), torch.randn(4, 4, device=ops.device)
    stepper = step_lib.resolve_stepper(step_lib.StepperType.FORWARD)
    stepper.init_with_benchmark_args(model, (x,), {})
    with flagsaver.flagsaver(
        min_warmup_steps=1, max_warmup_steps=5, post_warmup_steps=3
    ):
      m = measure_lib.measure(
          stepper,
          ops,
          name="mlp_inference",
      )
    self.assertGreaterEqual(m.post_warmup_step_time_seconds, 0.0)

  def test_training_through_measure(self):
    ops = self._ops()
    model = _mlp().to(ops.device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(4, 4, device=ops.device)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING,
        accum_steps=2,
        compute_loss=mock_grad_probe,
    )
    stepper.init_with_benchmark_args(model, (x,), {}, opt)
    with flagsaver.flagsaver(
        min_warmup_steps=1, max_warmup_steps=5, post_warmup_steps=3
    ):
      m = measure_lib.measure(
          stepper,
          ops,
          name="mlp_training",
      )
    self.assertGreaterEqual(m.post_warmup_step_time_seconds, 0.0)
    self.assertGreater(m.e2e_wall_time_seconds, 0.0)

  def test_decode_stepper_eager(self):
    ops = self._ops()

    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    if not hasattr(config, "head_dim"):
      config.head_dim = config.hidden_size // config.num_attention_heads
    model = transformers.LlamaForCausalLM(config).to(ops.device)
    model.eval()

    prompt_len = 4
    x = torch.randint(0, 16, (1, prompt_len), device=ops.device)

    stepper = decode.decode()
    stepper.init_with_benchmark_args(model, (), {"input_ids": x})

    cache = stepper.get_cache()

    self.assertTrue(cache.layers[0].is_initialized)

    self.assertEqual(cache.get_seq_length(), 0)
    cache_state_pre = cache.layers[0].keys.clone()

    stepper.pre_warmup_init()
    torch.accelerator.synchronize()
    self.assertEqual(cache.get_seq_length(), prompt_len)
    self.assertIsNotNone(stepper.next_token)

    cache_state_post_prefill = cache.layers[0].keys.clone()
    self.assertFalse(torch.equal(cache_state_pre, cache_state_post_prefill))

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(cache.get_seq_length(), prompt_len + stepper.output_tokens)

    cache_state_post_step1 = cache.layers[0].keys.clone()
    self.assertFalse(
        torch.equal(cache_state_post_prefill, cache_state_post_step1)
    )

    stepper.post_warmup_hook()
    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(cache.get_seq_length(), prompt_len + stepper.output_tokens)

  def test_decode_stepper_compiled(self):
    with torch.no_grad():
      ops = self._ops()

      config = transformers.LlamaConfig(
          vocab_size=16,
          hidden_size=8,
          intermediate_size=16,
          num_hidden_layers=2,
          num_attention_heads=2,
          num_key_value_heads=2,
          max_position_embeddings=128,
      )
      if not hasattr(config, "head_dim"):
        config.head_dim = config.hidden_size // config.num_attention_heads
      model = transformers.LlamaForCausalLM(config).to(ops.device)
      model.eval()

      prompt_len = 4
      x = torch.randint(0, 16, (1, prompt_len), device=ops.device)

      stepper = decode.decode(output_tokens=8)
      stepper.init_with_benchmark_args(model, (), {"input_ids": x})
      target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
      compile_config = compile_lib.CompileConfig(
          scope=compile_lib.Scope.MODEL, dynamic=False
      )
      stepper.compile(compile_config, target)

      cache = stepper.get_cache()
      cache_state_pre = cache.layers[0].keys.clone()
      prompt_len = stepper.prompt_len

      stepper.pre_warmup_init()
      torch.accelerator.synchronize()

      self.assertEqual(cache.get_seq_length(), prompt_len)
      self.assertIsNotNone(stepper.next_token)

      cache_state_post_prefill = cache.layers[0].keys.clone()
      self.assertFalse(torch.equal(cache_state_pre, cache_state_post_prefill))

      step_fn = stepper.get_step_fn()
      torch.accelerator.synchronize()

      for _ in range(3):
        _ = step_fn()
        torch.accelerator.synchronize()
        self.assertEqual(cache.get_seq_length(), prompt_len + 8)

        stepper.post_warmup_hook()
        torch.accelerator.synchronize()
        self.assertEqual(cache.get_seq_length(), prompt_len)

      post_warmup_comp_cache_miss = getattr(torch, "tpu")._get_cache_misses()

      _ = step_fn()

      torch.accelerator.synchronize()
      self.assertEqual(
          getattr(torch, "tpu")._get_cache_misses(),
          post_warmup_comp_cache_miss,
          "missed cache in iter",
      )

  def test_decode_through_measure(self):
    ops = self._ops()

    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=128,
    )
    if not hasattr(config, "head_dim"):
      config.head_dim = config.hidden_size // config.num_attention_heads
    model = transformers.LlamaForCausalLM(config).to(ops.device)
    model.eval()
    x = torch.randint(0, 16, (1, 4), device=ops.device)

    stepper = decode.decode(output_tokens=8)
    stepper.init_with_benchmark_args(model, (), {"input_ids": x})
    with flagsaver.flagsaver(
        min_warmup_steps=1, max_warmup_steps=5, post_warmup_steps=3
    ):
      m = measure_lib.measure(
          stepper,
          ops,
          name="llama_decode",
      )
    self.assertLessEqual(m.num_warmup_steps, 5)
    self.assertGreaterEqual(m.post_warmup_step_time_seconds, 0.0)
    self.assertGreater(m.e2e_wall_time_seconds, 0.0)


if __name__ == "__main__":
  absltest.main()
