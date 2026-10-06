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
from torch_tpu._internal.utils import test_utils
from torch_tpu.extras.optimizers import single_trace_trainer
from perflab import compile as compile_lib
from perflab import discovery as discovery_lib
from perflab import measure as measure_lib
from perflab import step_lib
from perflab import steps
from perflab import target as target_lib
from perflab import torch_device_ops
from perflab.steps import decode
from tests import seed_test_utils
import transformers


def mock_grad_probe(model, args, kwargs):
  return model(*args, **kwargs).sum()


def _mlp(in_dim=4, hidden=8, out_dim=3, seed=0):
  torch.manual_seed(seed)
  return nn.Sequential(
      nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
  )


def _state_tensors(states):
  """Normalizes a linear-attention state container to a dict of tensors.

  transformers exposes `conv_states`/`recurrent_states` as a single tensor in
  older releases and as a mapping of state index to tensor in newer ones.

  Args:
    states: The container read off a cache layer or a snapshot entry.

  Returns:
    A dict of state index to tensor.
  """
  if isinstance(states, torch.Tensor):
    return {0: states}
  return {k: v for k, v in states.items() if isinstance(v, torch.Tensor)}


discovery_lib.import_submodules(steps)


class StepperTypeTest(seed_test_utils.RepeatableTest):

  def test_all_steps_registered(self):
    self.assertEqual(
        set(step_lib.STEPPERS),
        {
            step_lib.StepperType.FORWARD,
            step_lib.StepperType.TRAINING,
            step_lib.StepperType.SINGLE_TRACE_TRAINING,
            step_lib.StepperType.DECODER_ONLY_DECODE,
        },
    )

  def test_uniform_factory_shape(self):
    """Every entry is a factory that returns a callable."""
    for name in step_lib.StepperType:
      self.assertTrue(callable(step_lib.STEPPERS[name]))

  def test_training_bad_accum_steps_rejected(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
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
      test_utils.assert_close(grad_full, grad_micro, rtol=1e-5, atol=1e-6)
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
      test_utils.assert_close(a, b, rtol=1e-5, atol=1e-6)

  def test_training_requires_optimizer(self):
    stepper = step_lib.resolve_stepper(step_lib.StepperType.TRAINING)
    with self.assertRaises(TypeError):  # ASSERT_RAISES_OK=benchmark
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

  def test_decode_stepper_dynamic_eager(self):
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

    stepper = decode.decode(output_tokens=8, dynamism=True)
    stepper.init_with_benchmark_args(model, (), {"input_ids": x})

    cache = stepper.get_cache()
    self.assertIsInstance(cache, transformers.cache_utils.DynamicCache)
    self.assertEqual(cache.get_seq_length(), 0)

    stepper.pre_warmup_init()
    torch.accelerator.synchronize()
    cache = stepper.get_cache()
    self.assertEqual(cache.get_seq_length(), prompt_len)
    self.assertIsNotNone(stepper.next_token)

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()
    cache = stepper.get_cache()
    self.assertEqual(cache.get_seq_length(), prompt_len + stepper.output_tokens)

    stepper.post_warmup_hook()
    cache = stepper.get_cache()
    self.assertEqual(cache.get_seq_length(), prompt_len)

    _ = step_fn()
    torch.accelerator.synchronize()
    cache = stepper.get_cache()
    self.assertEqual(cache.get_seq_length(), prompt_len + stepper.output_tokens)

  def test_dynamic_cache_crop_hybrid_layer_recording(self):
    if not hasattr(
        transformers.cache_utils.DynamicCache, "activate_past_recording"
    ) or not hasattr(
        transformers.cache_utils.DynamicSlidingWindowLayer,
        "activate_past_recording",
    ):
      self.skipTest(
          "DynamicCache does not support activate_past_recording in this"
          " transformers version"
      )

    sliding_layer = transformers.cache_utils.DynamicSlidingWindowLayer(
        sliding_window=4
    )
    cache = transformers.cache_utils.DynamicCache()
    cache.layers = [sliding_layer]

    # Without activate_past_recording, crop after exceeding sliding window raises RuntimeError.
    sliding_layer.update(
        torch.randn(1, 1, 6, 8),
        torch.randn(1, 1, 6, 8),
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing rollback error when past recording is disabled.
        RuntimeError, "does not track past states"
    ):
      cache.crop(-2)

    # With activate_past_recording, crop succeeds and rolls back state.
    sliding_layer_rec = transformers.cache_utils.DynamicSlidingWindowLayer(
        sliding_window=4
    )
    cache_rec = transformers.cache_utils.DynamicCache()
    cache_rec.layers = [sliding_layer_rec]
    cache_rec.activate_past_recording()
    sliding_layer_rec.update(
        torch.randn(1, 1, 6, 8),
        torch.randn(1, 1, 6, 8),
    )
    cache_rec.crop(-2)
    self.assertEqual(cache_rec.get_seq_length(), 4)

  def test_decode_stepper_dynamic_hybrid_model(self):
    if not hasattr(
        transformers.cache_utils.DynamicCache, "activate_past_recording"
    ):
      self.skipTest(
          "DynamicCache does not support activate_past_recording in this"
          " transformers version"
      )

    ops = self._ops()

    config = transformers.JambaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=1,
        num_experts_per_tok=1,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        attn_layer_offset=1,
        attn_layer_period=2,
        use_mamba_kernels=False,
    )
    model = transformers.JambaForCausalLM(config).to(ops.device)
    model.eval()

    prompt_len = 4
    x = torch.randint(0, 16, (1, prompt_len), device=ops.device)

    stepper = decode.decode(output_tokens=4, dynamism=True)
    stepper.init_with_benchmark_args(model, (), {"input_ids": x})

    cache = stepper.get_cache()
    self.assertIsInstance(cache, transformers.cache_utils.DynamicCache)

    stepper.pre_warmup_init()
    torch.accelerator.synchronize()
    self.assertIsNotNone(stepper.next_token)

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len + stepper.output_tokens)

    # post_warmup_hook on hybrid dynamic cache should succeed w/o crop error
    stepper.post_warmup_hook()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len)
    self.assertTrue(torch.equal(stepper.next_token, stepper.prefill_next_token))

    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len + stepper.output_tokens)

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

  def test_single_trace_training_updates_params(self):
    ops = self._ops()
    model = _mlp(seed=99).to(ops.device)
    opt = single_trace_trainer.ReferenceSgd(lr=0.1)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.SINGLE_TRACE_TRAINING
    )
    before = {
        name: p.detach().cpu().clone() for name, p in model.named_parameters()
    }
    x = torch.randn(4, 4, device=ops.device)
    stepper.init_with_benchmark_args(model, (x,), {}, opt)
    stepper.get_step_fn()()
    after = stepper.get_params()
    changed = [
        not torch.equal(before[name], after[name].cpu()) for name in before
    ]
    self.assertTrue(
        all(changed),
        "Expected training params to change after single trace training step.",
    )

  def test_training_stepper_composite_dispatch_eager(self):
    ops = self._ops()
    model = _mlp(seed=42).to(ops.device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING, compute_loss=mock_grad_probe
    )
    x = torch.randn(4, 4, device=ops.device)
    stepper.init_with_benchmark_args(model, (x,), {}, opt)
    self.assertIsNone(stepper._trainer)
    before = [p.clone() for p in model.parameters()]
    stepper.get_step_fn()()
    self.assertIsNone(stepper._trainer)
    changed = [
        not torch.equal(b, p) for b, p in zip(before, model.parameters())
    ]
    self.assertTrue(all(changed))

  def test_training_stepper_composite_dispatch_compiled(self):
    ops = self._ops()
    model = _mlp(seed=42).to(ops.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING, compute_loss=mock_grad_probe
    )
    x = torch.randn(4, 4, device=ops.device)
    with torch.no_grad():
      expected_loss = mock_grad_probe(model, (x,), {})
    stepper.init_with_benchmark_args(model, (x,), {}, opt)
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    compile_config = compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=False
    )
    stepper.compile(compile_config, target)
    self.assertIsNotNone(stepper._trainer)
    before = {
        name: p.detach().cpu().clone() for name, p in model.named_parameters()
    }
    loss = stepper.get_step_fn()()
    # Only the single-trace step returns the loss, and it must come from
    # `compute_loss` (a sum) rather than the built-in loss (a mean).
    test_utils.assert_close(loss, expected_loss)
    after = stepper.get_params()
    changed = [
        not torch.equal(before[name], after[name].cpu()) for name in before
    ]
    self.assertTrue(all(changed))

  def test_training_stepper_composite_dispatch_compiled_disabled(self):
    ops = self._ops()
    model = _mlp(seed=42).to(ops.device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING,
        compute_loss=mock_grad_probe,
        use_single_trace=False,
    )
    x = torch.randn(4, 4, device=ops.device)
    stepper.init_with_benchmark_args(model, (x,), {}, opt)
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    compile_config = compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=False
    )
    stepper.compile(compile_config, target)
    self.assertIsNone(stepper._trainer)

  def test_training_stepper_uses_standard_stepper_on_gpu(self):
    ops = self._ops()
    model = _mlp(seed=42).to(ops.device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING, compute_loss=mock_grad_probe
    )
    x = torch.randn(4, 4, device=ops.device)
    stepper.init_with_benchmark_args(model, (x,), {}, opt)
    target = target_lib.make_target(platform=target_lib.Platform.B200_1)
    compile_config = compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=False
    )
    stepper.compile(compile_config, target)
    self.assertIsNone(stepper._trainer)

  def test_training_stepper_composite_dispatch_accum_steps(self):
    ops = self._ops()
    model = _mlp(seed=42).to(ops.device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stepper = step_lib.resolve_stepper(
        step_lib.StepperType.TRAINING,
        accum_steps=2,
        compute_loss=mock_grad_probe,
    )
    x = torch.randn(4, 4, device=ops.device)
    stepper.init_with_benchmark_args(model, (x,), {}, opt)
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    compile_config = compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=False
    )
    stepper.compile(compile_config, target)
    self.assertIsNone(stepper._trainer)

  def test_decode_safe_get_helper(self):
    # Object with first attribute present
    class ConfigA:
      num_attention_heads = 8
      n_head = 4

    self.assertEqual(
        decode._safe_get(ConfigA(), "num_attention_heads", "n_head"), 8
    )

    # Object with only second attribute present
    class ConfigB:
      n_head = 4

    self.assertEqual(
        decode._safe_get(ConfigB(), "num_attention_heads", "n_head"), 4
    )

    # Dict object
    dict_cfg = {"num_heads": 6}
    self.assertEqual(
        decode._safe_get(dict_cfg, "num_attention_heads", "num_heads"), 6
    )

    # Missing attributes or None object
    self.assertIsNone(decode._safe_get(ConfigB(), "nonexistent"))
    self.assertIsNone(decode._safe_get(None, "foo", "bar"))

    # Config with per_layer_config as list
    class LayerConfig:
      head_dim = 64
      num_heads = 4

    class RootWithList:
      per_layer_config = [LayerConfig()]

    self.assertEqual(decode._safe_get(RootWithList(), "head_dim"), 64)
    self.assertEqual(decode._safe_get(RootWithList(), "num_heads"), 4)

    # Config with per_layer_config as dict
    class RootWithDict:
      per_layer_config = {0: {"head_dim": 128, "num_heads": 8}}

    self.assertEqual(decode._safe_get(RootWithDict(), "head_dim"), 128)
    self.assertEqual(decode._safe_get(RootWithDict(), "num_heads"), 8)

    # Config where accessing global attribute raises an exception
    class RaisingConfig:

      @property
      def head_dim(self):
        raise RuntimeError(
            "'head_dim' is a per-layer attribute and may vary across layers."
        )

      per_layer_config = [{"head_dim": 256}]

    self.assertEqual(decode._safe_get(RaisingConfig(), "head_dim"), 256)

  def test_decode_stepper_config_aliases(self):
    ops = self._ops()

    # 1. Config with n_head / n_embd (GPT-2 style)
    cfg_gpt2 = transformers.GPT2Config(
        vocab_size=16,
        n_head=4,
        n_embd=64,
        n_layer=2,
    )
    model_gpt2 = mock.MagicMock(config=cfg_gpt2, dtype=torch.bfloat16)
    x = torch.randint(0, 16, (2, 8), device=ops.device)

    stepper_gpt2 = decode.DecodeStepper(output_tokens=4, dynamism=True)
    stepper_gpt2.init_with_benchmark_args(model_gpt2, (), {"input_ids": x})
    self.assertEqual(stepper_gpt2.num_heads, 4)
    self.assertEqual(stepper_gpt2.head_dim, 16)  # 64 // 4

    # 2. Config with num_heads / d_model / d_kv (T5 / other style)
    cfg_t5 = transformers.T5Config(
        vocab_size=16,
        num_heads=8,
        d_model=128,
        d_kv=32,
        num_layers=2,
    )
    model_t5 = mock.MagicMock(config=cfg_t5, dtype=torch.float32)
    stepper_t5 = decode.DecodeStepper(output_tokens=4, dynamism=True)
    stepper_t5.init_with_benchmark_args(model_t5, (), {"input_ids": x})
    self.assertEqual(stepper_t5.num_heads, 8)
    self.assertEqual(stepper_t5.head_dim, 32)

    # 3. Multimodal / nested text_config (object and dict)
    nested_text_cfg = transformers.LlamaConfig(
        vocab_size=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_size=1024,
        num_hidden_layers=2,
    )
    nested_text_cfg.head_dim = 64
    root_cfg = mock.MagicMock(text_config=nested_text_cfg)
    root_cfg.get_text_config = mock.MagicMock(return_value=nested_text_cfg)
    model_multimodal = mock.MagicMock(config=root_cfg, dtype=torch.bfloat16)
    stepper_mm = decode.DecodeStepper(output_tokens=4, dynamism=True)
    stepper_mm.init_with_benchmark_args(model_multimodal, (), {"input_ids": x})
    self.assertEqual(stepper_mm.num_heads, 4)
    self.assertEqual(stepper_mm.head_dim, 64)

    # 4. Multimodal with dictionary text_config
    dict_text_cfg = {
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hidden_size": 512,
    }
    root_dict_cfg = transformers.PretrainedConfig(num_hidden_layers=2)
    root_dict_cfg.text_config = dict_text_cfg
    root_dict_cfg.get_text_config = mock.MagicMock(
        return_value=transformers.PretrainedConfig(num_hidden_layers=2)
    )
    model_dict = mock.MagicMock(config=root_dict_cfg, dtype=torch.bfloat16)
    stepper_dict = decode.DecodeStepper(output_tokens=4, dynamism=True)
    stepper_dict.init_with_benchmark_args(model_dict, (), {"input_ids": x})
    self.assertEqual(stepper_dict.num_heads, 2)
    self.assertEqual(stepper_dict.head_dim, 32)

    # 5. Fallback defaults when all config attributes are absent
    empty_cfg = transformers.PretrainedConfig(num_hidden_layers=1)
    model_empty = mock.MagicMock(config=empty_cfg, dtype=torch.bfloat16)
    stepper_empty = decode.DecodeStepper(output_tokens=4, dynamism=True)
    stepper_empty.init_with_benchmark_args(model_empty, (), {"input_ids": x})
    self.assertEqual(stepper_empty.num_heads, 1)
    self.assertEqual(stepper_empty.head_dim, 1)

  def test_decode_stepper_base_model_output_with_past_eager(self):
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
    # LlamaModel outputs BaseModelOutputWithPast (no logits field)
    model = transformers.LlamaModel(config).to(ops.device)
    model.eval()

    prompt_len = 4
    x = torch.randint(0, 16, (1, prompt_len), device=ops.device)

    stepper = decode.decode(output_tokens=4, dynamism=True)
    stepper.init_with_benchmark_args(model, (), {"input_ids": x})

    cache = stepper.get_cache()
    self.assertIsInstance(cache, transformers.cache_utils.DynamicCache)
    self.assertEqual(cache.get_seq_length(), 0)

    stepper.pre_warmup_init()
    torch.accelerator.synchronize()
    cache = stepper.get_cache()
    self.assertEqual(cache.get_seq_length(), prompt_len)
    self.assertIsNotNone(stepper.next_token)

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()
    cache = stepper.get_cache()
    self.assertEqual(cache.get_seq_length(), prompt_len + stepper.output_tokens)

  def test_decode_unpack_output_base_model_output_variations(self):
    batch_size, seq_len, hidden_dim, vocab_size = 2, 4, 8, 16

    # 1. Output with get_output_embeddings
    stepper = decode.DecodeStepper()
    linear_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.get_output_embeddings = mock.MagicMock(return_value=linear_head)
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(return_value=None)
    stepper._model = mock_model
    stepper._decode_model = mock_model

    hidden = torch.randn(batch_size, seq_len, hidden_dim)
    base_output = transformers.modeling_outputs.BaseModelOutputWithPast(
        last_hidden_state=hidden
    )
    logits = stepper._unpack_output_and_update_cache(base_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

    # 2. Output with lm_head
    stepper = decode.DecodeStepper()
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.get_output_embeddings = mock.MagicMock(return_value=None)
    mock_model.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=True)
    mock_model.get_input_embeddings = mock.MagicMock(return_value=None)
    stepper._model = mock_model
    stepper._decode_model = mock_model

    logits = stepper._unpack_output_and_update_cache(base_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

    # 3. Output with get_input_embeddings
    stepper = decode.DecodeStepper()
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.get_output_embeddings = mock.MagicMock(return_value=None)
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(
        return_value=torch.nn.Embedding(vocab_size, hidden_dim)
    )
    stepper._model = mock_model
    stepper._decode_model = mock_model

    logits = stepper._unpack_output_and_update_cache(base_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

    # 4. Fallback to hidden_state
    stepper = decode.DecodeStepper()
    mock_model = mock.MagicMock(spec=[])
    stepper._model = mock_model
    stepper._decode_model = mock_model

    logits = stepper._unpack_output_and_update_cache(base_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, hidden_dim))

    # 5. Output with tied nn.Embedding from get_output_embeddings
    stepper = decode.DecodeStepper()
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.get_output_embeddings = mock.MagicMock(
        return_value=torch.nn.Embedding(vocab_size, hidden_dim)
    )
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(return_value=None)
    stepper._model = mock_model
    stepper._decode_model = mock_model

    logits = stepper._unpack_output_and_update_cache(base_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

    # 6. Dictionary output with past_key_values and last_hidden_state
    stepper = decode.DecodeStepper()
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.get_output_embeddings = mock.MagicMock(return_value=None)
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(
        return_value=torch.nn.Embedding(vocab_size, hidden_dim)
    )
    stepper._model = mock_model
    stepper._decode_model = mock_model
    dummy_cache = transformers.cache_utils.DynamicCache()
    dict_output = {
        "last_hidden_state": hidden,
        "past_key_values": dummy_cache,
    }
    logits = stepper._unpack_output_and_update_cache(dict_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
    self.assertIs(stepper.cache, dummy_cache)

    # 7. Tuple output (return_dict=False) with base model hidden state
    stepper = decode.DecodeStepper()
    mock_config = mock.MagicMock(vocab_size=vocab_size)
    mock_model = mock.MagicMock(spec=torch.nn.Module, config=mock_config)
    mock_model.get_output_embeddings = mock.MagicMock(return_value=None)
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(
        return_value=torch.nn.Embedding(vocab_size, hidden_dim)
    )
    stepper._model = mock_model
    stepper._decode_model = mock_model
    tuple_output = (hidden, dummy_cache)
    logits = stepper._unpack_output_and_update_cache(tuple_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
    self.assertIs(stepper.cache, dummy_cache)

    # 8. Output with lm_head as nn.Embedding (tied weights)
    stepper = decode.DecodeStepper()
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.get_output_embeddings = mock.MagicMock(return_value=None)
    mock_model.lm_head = torch.nn.Embedding(vocab_size, hidden_dim)
    mock_model.get_input_embeddings = mock.MagicMock(return_value=None)
    stepper._model = mock_model
    stepper._decode_model = mock_model

    logits = stepper._unpack_output_and_update_cache(base_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

    # 9. Raw torch.Tensor output already representing logits
    stepper = decode.DecodeStepper()
    mock_config = mock.MagicMock(vocab_size=vocab_size)
    mock_model = mock.MagicMock(spec=torch.nn.Module, config=mock_config)
    mock_model.get_output_embeddings = mock.MagicMock(return_value=None)
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(
        return_value=torch.nn.Embedding(vocab_size, hidden_dim)
    )
    stepper._model = mock_model
    stepper._decode_model = mock_model
    stepper.vocab_size = vocab_size
    raw_logits_tensor = torch.randn(batch_size, seq_len, vocab_size)
    logits = stepper._unpack_output_and_update_cache(raw_logits_tensor)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
    self.assertIs(logits, raw_logits_tensor)

    # 10. Nested text_config with vocab_size
    stepper = decode.DecodeStepper()
    text_cfg = mock.MagicMock(vocab_size=vocab_size)
    root_cfg = mock.MagicMock(text_config=text_cfg)
    mock_model = mock.MagicMock(spec=torch.nn.Module, config=root_cfg)
    mock_model.get_output_embeddings = mock.MagicMock(return_value=None)
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(
        return_value=torch.nn.Embedding(vocab_size, hidden_dim)
    )
    stepper._model = mock_model
    stepper._decode_model = mock_model
    logits = stepper._unpack_output_and_update_cache(tuple_output)
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

  def test_decode_stepper_unpack_output_with_cross_attentions(self):
    last_hidden_state = torch.randn(2, 4, 16)
    dummy_cache = mock.MagicMock(spec=transformers.cache_utils.Cache)

    # 1. BaseModelOutputWithPastAndCrossAttentions without model projection
    stepper = decode.DecodeStepper(output_tokens=4, dynamism=True)
    output_cross_attn = (
        transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            past_key_values=dummy_cache,
        )
    )
    logits = stepper._unpack_output_and_update_cache(output_cross_attn)
    self.assertTrue(torch.equal(logits, last_hidden_state))
    self.assertEqual(stepper.cache, dummy_cache)

    # 2. BaseModelOutputWithPastAndCrossAttentions with model projection
    stepper = decode.DecodeStepper(output_tokens=4, dynamism=True)
    vocab_size = 32
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.get_output_embeddings = mock.MagicMock(
        return_value=torch.nn.Linear(16, vocab_size, bias=False)
    )
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(return_value=None)
    stepper._model = mock_model
    stepper._decode_model = mock_model
    logits = stepper._unpack_output_and_update_cache(output_cross_attn)
    self.assertEqual(logits.shape, (2, 4, vocab_size))

    # 3. BaseModelOutputWithPast
    stepper = decode.DecodeStepper(output_tokens=4, dynamism=True)
    output_past = transformers.modeling_outputs.BaseModelOutputWithPast(
        last_hidden_state=last_hidden_state,
        past_key_values=dummy_cache,
    )
    logits = stepper._unpack_output_and_update_cache(output_past)
    self.assertTrue(torch.equal(logits, last_hidden_state))

    # 4. CausalLMOutputWithCrossAttentions
    stepper = decode.DecodeStepper(output_tokens=4, dynamism=True)
    expected_logits = torch.randn(2, 4, 32)
    output_causal = (
        transformers.modeling_outputs.CausalLMOutputWithCrossAttentions(
            logits=expected_logits,
            past_key_values=dummy_cache,
        )
    )
    logits = stepper._unpack_output_and_update_cache(output_causal)
    self.assertTrue(torch.equal(logits, expected_logits))

    # 5. Indexable custom sequence object fallback (unprojected)
    stepper = decode.DecodeStepper(output_tokens=4, dynamism=True)

    class CustomIndexableOutput:

      def __init__(self, tensor):
        self._tensor = tensor

      def __getitem__(self, idx):
        if idx == 0:
          return self._tensor
        raise IndexError

    custom_output = CustomIndexableOutput(last_hidden_state)
    logits = stepper._unpack_output_and_update_cache(custom_output)
    self.assertTrue(torch.equal(logits, last_hidden_state))

    # 6. Indexable custom sequence object with tensor already sized to vocab_size
    stepper = decode.DecodeStepper(output_tokens=4, dynamism=True)
    stepper.vocab_size = vocab_size
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.get_output_embeddings = mock.MagicMock(
        return_value=torch.nn.Linear(16, vocab_size, bias=False)
    )
    mock_model.lm_head = None
    mock_model.get_input_embeddings = mock.MagicMock(return_value=None)
    stepper._model = mock_model
    stepper._decode_model = mock_model
    already_sized_tensor = torch.randn(2, 4, vocab_size)
    custom_output_sized = CustomIndexableOutput(already_sized_tensor)
    logits = stepper._unpack_output_and_update_cache(custom_output_sized)
    self.assertTrue(torch.equal(logits, already_sized_tensor))

  def test_prepare_forward_kwargs(self):
    cache = mock.MagicMock()
    base_kwargs = {
        "use_cache": True,
        "cache_position": torch.tensor([4]),
    }

    # Model taking (input_ids, past_key_values, cache_position)
    class ModelWithPastKV(torch.nn.Module):

      def forward(self, input_ids, past_key_values=None, cache_position=None):
        pass

    m1 = ModelWithPastKV()
    res1 = decode.DecodeStepper._prepare_forward_kwargs(m1, cache, base_kwargs)
    self.assertIn("past_key_values", res1)
    self.assertEqual(res1["past_key_values"], cache)
    self.assertIn("cache_position", res1)

    # Model taking (input_ids, cache_params) without cache_position
    class ModelWithCacheParams(torch.nn.Module):

      def forward(self, input_ids, cache_params=None):
        pass

    m2 = ModelWithCacheParams()
    res2 = decode.DecodeStepper._prepare_forward_kwargs(m2, cache, base_kwargs)
    self.assertIn("cache_params", res2)
    self.assertEqual(res2["cache_params"], cache)
    self.assertNotIn("cache_position", res2)

    # Model with **kwargs
    class ModelWithVarKwargs(torch.nn.Module):

      def forward(self, input_ids, **kwargs):
        pass

    m3 = ModelWithVarKwargs()
    res3 = decode.DecodeStepper._prepare_forward_kwargs(m3, cache, base_kwargs)
    self.assertIn("past_key_values", res3)
    self.assertIn("cache_position", res3)

  def test_decode_stepper_cache_selection_static_vs_dynamic(self):
    ops = self._ops()

    # Model supporting static cache with dynamism=False -> StaticCache
    config_static = transformers.GPT2Config(
        vocab_size=16,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    model_static = transformers.GPT2LMHeadModel(config_static).to(ops.device)
    stepper_static = decode.decode(output_tokens=4, dynamism=False)
    stepper_static.init_with_benchmark_args(
        model_static,
        (),
        {"input_ids": torch.randint(0, 16, (1, 4), device=ops.device)},
    )
    self.assertIsInstance(
        stepper_static.get_cache(), transformers.cache_utils.StaticCache
    )

    # Model with dynamism=True -> DynamicCache
    stepper_dynamic = decode.decode(output_tokens=4, dynamism=True)
    stepper_dynamic.init_with_benchmark_args(
        model_static,
        (),
        {"input_ids": torch.randint(0, 16, (1, 4), device=ops.device)},
    )
    self.assertIsInstance(
        stepper_dynamic.get_cache(), transformers.cache_utils.DynamicCache
    )

    # Recurrent/SSM model -> DynamicCache even with dynamism=False.
    config_mamba = transformers.MambaConfig(
        vocab_size=16,
        hidden_size=16,
        state_size=8,
        num_hidden_layers=2,
    )
    model_mamba = transformers.MambaForCausalLM(config_mamba).to(ops.device)
    stepper_mamba = decode.decode(output_tokens=4, dynamism=False)
    stepper_mamba.init_with_benchmark_args(
        model_mamba,
        (),
        {"input_ids": torch.randint(0, 16, (1, 4), device=ops.device)},
    )
    self.assertIsInstance(
        stepper_mamba.get_cache(), transformers.cache_utils.DynamicCache
    )

  def _mistral_static_stepper(self, ops, sliding_window, output_tokens):
    config = transformers.MistralConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        sliding_window=sliding_window,
    )
    torch.manual_seed(0)
    model = transformers.MistralForCausalLM(config).to(ops.device).eval()
    stepper = decode.decode(output_tokens=output_tokens, dynamism=False)
    torch.manual_seed(1)
    input_ids = torch.randint(0, 16, (1, 4)).to(ops.device)
    stepper.init_with_benchmark_args(model, (), {"input_ids": input_ids})
    return stepper

  def test_decode_stepper_static_cache_drops_unreachable_sliding_window(self):
    cache_utils = transformers.cache_utils
    ops = self._ops()
    # prompt 4 + 4 new tokens = 8 <= window 4096: the window never rolls.
    stepper = self._mistral_static_stepper(
        ops, sliding_window=4096, output_tokens=4
    )
    layers = stepper.get_cache().layers
    self.assertNotEmpty(layers)
    for layer in layers:
      self.assertIs(type(layer), cache_utils.StaticLayer)
      self.assertEqual(layer.max_cache_len, 8)
      self.assertTrue(layer.is_sliding)
    # Callers such as masking_utils look up `is_sliding.index(True)`.
    self.assertEqual(stepper.get_cache().is_sliding.index(True), 0)

    # A window shorter than the sequence rolls, so the layer must be kept.
    stepper = self._mistral_static_stepper(
        ops, sliding_window=4, output_tokens=4
    )
    for layer in stepper.get_cache().layers:
      self.assertIs(type(layer), cache_utils.StaticSlidingWindowLayer)

  def _run_static_decode(self, num_steps, compile_model):
    """Runs a static-cache Mistral decode and returns per-step outputs."""
    ops = self._ops()
    stepper = self._mistral_static_stepper(
        ops, sliding_window=4096, output_tokens=num_steps
    )
    stepper.pre_warmup_init()
    stepper.output_tokens = 1
    if compile_model:
      stepper.compile(
          compile_lib.CompileConfig(
              scope=compile_lib.Scope.MODEL, dynamic=False
          ),
          ops.target,
      )
    step_fn = stepper.get_step_fn()
    graphs_after_step = []
    logits = []
    with torch.inference_mode():
      for _ in range(num_steps):
        logits.append(step_fn().logits.detach().clone())
        graphs_after_step.append(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )
    return logits, graphs_after_step

  def test_decode_stepper_static_sliding_window_model_does_not_recompile(self):
    # Step 0 consumes the prefill token, which was created outside
    # inference_mode, so step 1 recompiles once on its dispatch keys. That is
    # absorbed by warmup in practice; the steady state starts at step 1.
    torch._dynamo.reset()
    _, graphs = self._run_static_decode(num_steps=5, compile_model=True)
    self.assertGreater(graphs[1], 0)
    self.assertEqual(graphs[-1], graphs[1])

    # Without the fix the sliding layer's Python int changes every step and
    # dynamo recompiles, which is what this test guards against.
    torch._dynamo.reset()
    with mock.patch.object(decode, "_drop_unreachable_sliding_windows"):
      _, graphs = self._run_static_decode(num_steps=5, compile_model=True)
    self.assertGreater(graphs[-1], graphs[1])
    torch._dynamo.reset()

  def test_decode_stepper_dropping_sliding_window_keeps_logits(self):
    fixed, _ = self._run_static_decode(num_steps=4, compile_model=False)
    with mock.patch.object(decode, "_drop_unreachable_sliding_windows"):
      original, _ = self._run_static_decode(num_steps=4, compile_model=False)
    for got, want in zip(fixed, original):
      test_utils.assert_close(got, want, rtol=0, atol=0)

  def test_decode_stepper_falcon_multi_query_heads(self):
    ops = self._ops()
    config_falcon_mq = transformers.FalconConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        multi_query=True,
        new_decoder_architecture=False,
    )
    model_falcon_mq = transformers.FalconForCausalLM(config_falcon_mq).to(
        ops.device
    )
    stepper_mq = decode.decode(output_tokens=4, dynamism=False)
    stepper_mq.init_with_benchmark_args(
        model_falcon_mq,
        (),
        {"input_ids": torch.randint(0, 16, (1, 4), device=ops.device)},
    )
    self.assertEqual(stepper_mq.num_heads, 1)

  def test_decode_stepper_custom_cache_snapshot_and_rollback(self):
    class CustomSSMCache:

      def __init__(self, state_val: int = 1):
        self.state_val = state_val

    class DummySSMModel(torch.nn.Module):

      def __init__(self, vocab_size: int = 16, hidden_size: int = 16):
        super().__init__()
        self.config = mock.MagicMock(
            model_type="custom_ssm",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=2,
            _supports_static_cache=False,
        )
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

      def forward(self, input_ids, past_key_values=None, **kwargs):
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        if past_key_values is not None and isinstance(
            past_key_values, CustomSSMCache
        ):
          past_key_values.state_val += 1
          return logits, past_key_values
        return logits

    ops = self._ops()
    model = DummySSMModel().to(ops.device)
    stepper = decode.decode(output_tokens=4, dynamism=False)
    stepper.init_with_benchmark_args(
        model,
        (),
        {"input_ids": torch.randint(0, 16, (1, 4), device=ops.device)},
    )
    custom_cache = CustomSSMCache(state_val=10)
    stepper.cache = custom_cache

    stepper.pre_warmup_init()
    self.assertEqual(stepper.cache.state_val, 11)
    self.assertIsNotNone(stepper._prefill_cache_snapshot)
    self.assertEqual(stepper._prefill_cache_snapshot.state_val, 11)

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    self.assertEqual(stepper.cache.state_val, 15)

    stepper.post_warmup_hook()
    self.assertEqual(stepper.cache.state_val, 11)

  def test_decode_stepper_mamba_warmup_lifecycle(self):
    ops = self._ops()
    config = transformers.MambaConfig(
        vocab_size=16,
        hidden_size=16,
        state_size=8,
        num_hidden_layers=2,
    )
    # The harness relies on transformers deriving linear-attention cache layers
    # from the config; older releases have no `layer_types` on `MambaConfig`.
    probe = transformers.cache_utils.DynamicCache(config=config)
    if not isinstance(
        probe.layers[0], transformers.cache_utils.LinearAttentionCacheLayerMixin
    ):
      self.skipTest(
          "this transformers version does not build linear-attention cache"
          " layers for Mamba"
      )

    model = transformers.MambaForCausalLM(config).to(ops.device)
    model.eval()

    prompt_len = 4
    x = torch.randint(0, 16, (1, prompt_len), device=ops.device)

    stepper = decode.decode(output_tokens=4, dynamism=False)
    stepper.init_with_benchmark_args(model, (), {"input_ids": x})
    # Mamba has no attention KV cache, so it must never get a StaticCache.
    self.assertIsInstance(
        stepper.get_cache(), transformers.cache_utils.DynamicCache
    )

    stepper.pre_warmup_init()
    torch.accelerator.synchronize()
    self.assertIsNotNone(stepper.next_token)
    # `crop()` cannot roll back recurrent state, so it must be snapshotted.
    self.assertIsNotNone(stepper._prefill_linear_states)
    prefill_states = {
        layer_idx: {
            state_idx: tensor.clone()
            for state_idx, tensor in _state_tensors(
                entry["recurrent_states"]
            ).items()
        }
        for layer_idx, entry in stepper._prefill_linear_states.items()
        if "recurrent_states" in entry
    }

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len + stepper.output_tokens)
    # Guard against a vacuous restore assertion below: warmup decode steps must
    # actually have advanced the recurrent state.
    layers = stepper.get_cache().layers
    self.assertTrue(
        any(
            not torch.equal(
                _state_tensors(layers[layer_idx].recurrent_states)[state_idx],
                want,
            )
            for layer_idx, states in prefill_states.items()
            for state_idx, want in states.items()
        ),
        "warmup decode did not mutate any recurrent state",
    )

    stepper.post_warmup_hook()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len)
    self.assertTrue(torch.equal(stepper.next_token, stepper.prefill_next_token))
    for layer_idx, states in prefill_states.items():
      actual = _state_tensors(layers[layer_idx].recurrent_states)
      for state_idx, want in states.items():
        self.assertTrue(
            torch.equal(actual[state_idx], want),
            f"recurrent state (layer {layer_idx}, state {state_idx}) was not"
            " restored to its post-prefill value",
        )

    # Decoding must still work after the rollback.
    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len + stepper.output_tokens)

  def test_linear_attention_states_legacy_tensor_layout(self):
    # transformers releases before the multi-state rework expose
    # `conv_states`/`recurrent_states` as a single tensor rather than a dict.
    class LegacyLinearLayer:

      def __init__(self):
        self.conv_states = torch.zeros(1, 4, 3)
        self.recurrent_states = torch.ones(1, 4, 2)

    class FakeCache:

      def __init__(self, layers):
        self.layers = layers

    layer = LegacyLinearLayer()
    cache = FakeCache([layer])

    snapshot = decode._snapshot_linear_attention_states(
        cache, include_conv=True
    )
    self.assertIsNotNone(snapshot)
    self.assertCountEqual(
        snapshot[0].keys(), ["conv_states", "recurrent_states"]
    )

    recurrent_buffer = layer.recurrent_states
    layer.conv_states.fill_(7.0)
    layer.recurrent_states.fill_(7.0)
    decode._restore_linear_attention_states(cache, snapshot)

    self.assertTrue(torch.equal(layer.conv_states, torch.zeros(1, 4, 3)))
    self.assertTrue(torch.equal(layer.recurrent_states, torch.ones(1, 4, 2)))
    # Restoring must write through the existing buffer so that the static
    # address the compiled decode step captured stays valid.
    self.assertIs(layer.recurrent_states, recurrent_buffer)

  def test_supports_early_initialization(self):
    class ModernLayer:
      supports_early_init = True

    class LegacyAttentionLayer:
      pass

    # A linear-attention layer on a release that predates `supports_early_init`
    # would be handed an attention-shaped tensor by `early_initialization`.
    class LegacyLinearLayer:
      pass

    transformers.cache_utils.LinearAttentionCacheLayerMixin.register(
        LegacyLinearLayer
    )

    class FakeCache:

      def __init__(self, layers):
        self.layers = layers

    self.assertTrue(
        decode._supports_early_initialization(FakeCache([ModernLayer()]))
    )
    self.assertTrue(
        decode._supports_early_initialization(
            FakeCache([LegacyAttentionLayer()])
        )
    )
    self.assertFalse(
        decode._supports_early_initialization(
            FakeCache([LegacyAttentionLayer(), LegacyLinearLayer()])
        )
    )

  def test_forward_param_spec_is_memoized(self):
    class ModelWithPastKV(torch.nn.Module):

      def forward(self, input_ids, past_key_values=None, cache_position=None):
        pass

    # The decode loop calls this once per generated token, so the underlying
    # `inspect.signature` call must only happen once per forward signature.
    decode._inspect_params.cache_clear()
    first = decode._forward_param_spec(ModelWithPastKV().forward)
    misses_after_first = decode._inspect_params.cache_info().misses
    for _ in range(5):
      self.assertEqual(
          decode._forward_param_spec(ModelWithPastKV().forward), first
      )
    self.assertEqual(
        decode._inspect_params.cache_info().misses, misses_after_first
    )
    self.assertIn("past_key_values", first[0])

  def test_decode_stepper_hybrid_static_cache_warmup_lifecycle(self):
    if "linear_attention" not in getattr(
        transformers.cache_utils, "STATIC_LAYER_TYPE_MAPPING", {}
    ):
      self.skipTest(
          "this transformers version predates STATIC_LAYER_TYPE_MAPPING, so"
          " the static cache layout for hybrid models is not guaranteed"
      )

    ops = self._ops()
    config = transformers.JambaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=1,
        num_experts_per_tok=1,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        attn_layer_offset=1,
        attn_layer_period=2,
        use_mamba_kernels=False,
    )
    # Jamba reports layer_types = ["linear_attention", "full_attention", ...].
    # transformers maps both through STATIC_LAYER_TYPE_MAPPING, so a hybrid
    # model must keep its statically shaped cache when dynamism is off.
    model = transformers.JambaForCausalLM(config).to(ops.device)
    model.eval()

    prompt_len = 4
    x = torch.randint(0, 16, (1, prompt_len), device=ops.device)

    stepper = decode.decode(output_tokens=4, dynamism=False)
    stepper.init_with_benchmark_args(model, (), {"input_ids": x})
    self.assertIsInstance(
        stepper.get_cache(), transformers.cache_utils.StaticCache
    )

    stepper.pre_warmup_init()
    torch.accelerator.synchronize()
    # The linear-attention layers mutate conv/recurrent buffers in place, so
    # they must be snapshotted for a static cache.
    self.assertIsNotNone(stepper._prefill_linear_states)
    prefill_states = {
        layer_idx: {
            name: {
                state_idx: tensor.clone()
                for state_idx, tensor in _state_tensors(states).items()
            }
            for name, states in entry.items()
        }
        for layer_idx, entry in stepper._prefill_linear_states.items()
    }

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len + stepper.output_tokens)

    stepper.post_warmup_hook()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len)
    layers = stepper.get_cache().layers
    for layer_idx, entry in prefill_states.items():
      for name, states in entry.items():
        actual = _state_tensors(getattr(layers[layer_idx], name))
        for state_idx, want in states.items():
          self.assertTrue(
              torch.equal(actual[state_idx], want),
              f"{name}[{state_idx}] on layer {layer_idx} was not restored",
          )

    # Attention layers must also be rewound, including the python-int counter
    # that StaticSlidingWindowLayer keeps alongside the tensor one.
    for layer in layers:
      if hasattr(layer, "cumulative_length_int"):
        self.assertEqual(layer.cumulative_length_int, prompt_len)

    _ = step_fn()
    torch.accelerator.synchronize()
    self.assertEqual(stepper.current_pos, prompt_len + stepper.output_tokens)

  def test_unpack_output_ignores_scalar_second_element(self):
    stepper = decode.DecodeStepper(output_tokens=4, dynamism=True)
    stepper.vocab_size = 32
    logits = torch.randn(2, 4, 32)

    # A scalar (e.g. a loss) in slot 1 must never be mistaken for a cache; if it
    # were, the next step would pass it as `past_key_values`.
    result = stepper._unpack_output_and_update_cache((logits, 0.5))
    self.assertTrue(torch.equal(result, logits))
    self.assertIsNone(stepper.cache)

    # A genuine custom (non-Cache, non-scalar) object is still picked up.
    class CustomCache:
      pass

    custom = CustomCache()
    result = stepper._unpack_output_and_update_cache((logits, custom))
    self.assertTrue(torch.equal(result, logits))
    self.assertIs(stepper.cache, custom)

  def test_state_param_and_module_recurrent_state_rollback(self):
    # Exercises both RWKV-style `state` list kwargs/outputs and
    # RecurrentGemma-style submodule tensor attributes (`conv1d_state`,
    # `recurrent_states`).
    class StateListOutput:

      def __init__(self, logits, state):
        self.logits = logits
        self.state = state

    class ToyRecurrentModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.config = type(
            "Cfg",
            (),
            {
                "vocab_size": 16,
                "num_attention_heads": 2,
                "hidden_size": 8,
                "model_type": "rwkv",
            },
        )()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)
        self.conv1d_state = None
        self.recurrent_states = None

      def forward(self, input_ids, state=None, use_cache=True, **kwargs):
        del use_cache, kwargs
        # A `Cache` instance would fail subscripting; prefill must pass `None`.
        assert not isinstance(state, transformers.cache_utils.Cache)
        if state is None:
          state = [torch.zeros(1, 8, device=input_ids.device)]
        state = [state[0] + float(input_ids.shape[1])]
        if self.conv1d_state is None:
          self.conv1d_state = torch.zeros(1, 8, 3, device=input_ids.device)
          self.recurrent_states = torch.zeros(1, 8, device=input_ids.device)
        self.conv1d_state = self.conv1d_state + 1.0
        self.recurrent_states = self.recurrent_states + 2.0
        logits = self.head(self.embed(input_ids))
        return StateListOutput(logits=logits, state=state)

    ops = self._ops()
    model = ToyRecurrentModel().to(ops.device).eval()
    x = torch.randint(0, 16, (1, 4), device=ops.device)

    stepper = decode.decode(output_tokens=3, dynamism=False)
    stepper.init_with_benchmark_args(model, (), {"input_ids": x})
    stepper.pre_warmup_init()
    torch.accelerator.synchronize()

    self.assertIsInstance(stepper.get_cache(), list)
    self.assertTrue(
        torch.equal(
            stepper.get_cache()[0], torch.full((1, 8), 4.0, device=ops.device)
        )
    )
    prefill_conv = model.conv1d_state.clone()
    prefill_rec = model.recurrent_states.clone()

    step_fn = stepper.get_step_fn()
    _ = step_fn()
    torch.accelerator.synchronize()
    conv_buf = model.conv1d_state
    rec_buf = model.recurrent_states
    self.assertTrue(
        torch.equal(
            stepper.get_cache()[0], torch.full((1, 8), 7.0, device=ops.device)
        )
    )
    self.assertFalse(torch.equal(model.conv1d_state, prefill_conv))
    self.assertFalse(torch.equal(model.recurrent_states, prefill_rec))

    stepper.post_warmup_hook()
    torch.accelerator.synchronize()
    self.assertTrue(
        torch.equal(
            stepper.get_cache()[0], torch.full((1, 8), 4.0, device=ops.device)
        )
    )
    self.assertIs(model.conv1d_state, conv_buf)
    self.assertIs(model.recurrent_states, rec_buf)
    self.assertTrue(torch.equal(model.conv1d_state, prefill_conv))
    self.assertTrue(torch.equal(model.recurrent_states, prefill_rec))

  def test_decode_stepper_heterogeneous_config(self):
    ops = self._ops()

    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        head_dim=16,
        per_layer_config={1: {"head_dim": 32, "num_key_value_heads": 2}},
    )
    if hasattr(config, "allow_global_per_layer_attribute_access"):
      self.assertFalse(config.allow_global_per_layer_attribute_access)
      with self.assertRaises(  # ASSERT_RAISES_OK=Verifying config raises before stepper init.
          Exception
      ):
        _ = config.head_dim

    model = mock.MagicMock(config=config, dtype=torch.bfloat16)
    x = torch.randint(0, 16, (1, 4), device=ops.device)

    # Static decode mode
    stepper_static = decode.DecodeStepper(output_tokens=4, dynamism=False)
    stepper_static.init_with_benchmark_args(model, (), {"input_ids": x})
    self.assertEqual(stepper_static.head_dim, 16)
    self.assertEqual(stepper_static.num_heads, 4)
    if hasattr(config, "allow_global_per_layer_attribute_access"):
      self.assertTrue(config.allow_global_per_layer_attribute_access)

    # Dynamic decode mode
    if hasattr(config, "allow_global_per_layer_attribute_access"):
      config.allow_global_per_layer_attribute_access = False
    stepper_dynamic = decode.DecodeStepper(output_tokens=4, dynamism=True)
    stepper_dynamic.init_with_benchmark_args(model, (), {"input_ids": x})
    self.assertEqual(stepper_dynamic.head_dim, 16)
    self.assertEqual(stepper_dynamic.num_heads, 4)
    if hasattr(config, "allow_global_per_layer_attribute_access"):
      self.assertTrue(config.allow_global_per_layer_attribute_access)


if __name__ == "__main__":
  absltest.main()
