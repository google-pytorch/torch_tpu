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


class _DummyDecodeModel(nn.Module):
  """Minimal stand-in for a decoder: echoes back whatever cache it was given."""

  def forward(self, input_ids=None, past_key_values=None, **kwargs):
    del input_ids, kwargs
    return past_key_values


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

  def test_decode_stepper_max_decode_tokens(self):
    stepper = decode.DecodeStepper(output_tokens=8)
    self.assertEqual(stepper.output_tokens, 8)
    self.assertEqual(stepper.max_decode_tokens, 8)

    stepper_override = decode.DecodeStepper(
        output_tokens=4, max_decode_tokens=32
    )
    self.assertEqual(stepper_override.output_tokens, 4)
    self.assertEqual(stepper_override.max_decode_tokens, 32)

    factory_stepper = decode.decode(output_tokens=6, max_decode_tokens=18)
    self.assertEqual(factory_stepper.output_tokens, 6)
    self.assertEqual(factory_stepper.max_decode_tokens, 18)

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
    ops = self._ops()
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

  def test_inplace_dynamic_cache_preserves_tensor_buffers_and_in_place_update(
      self,
  ):
    from torch_tpu._internal import cache as cache_lib

    ops = self._ops()
    cache = cache_lib.InPlaceDynamicCache(max_cache_len=16)
    k1 = torch.randn(1, 2, 4, 8, device=ops.device)
    v1 = torch.randn(1, 2, 4, 8, device=ops.device)

    # First prefill
    k_out, v_out = cache.update(k1, v1, layer_idx=0)
    self.assertEqual(cache.get_seq_length(), 4)
    self.assertEqual(k_out.shape, (1, 2, 4, 8))
    self.assertEqual(v_out.shape, (1, 2, 4, 8))
    layer = cache.layers[0]
    self.assertIsInstance(layer, cache_lib.InPlaceDynamicLayer)
    buf_k_ptr = layer._keys_buf.data_ptr()
    buf_v_ptr = layer._values_buf.data_ptr()

    # Step 1 token decode
    k2 = torch.randn(1, 2, 1, 8, device=ops.device)
    v2 = torch.randn(1, 2, 1, 8, device=ops.device)
    k_out2, v_out2 = cache.update(k2, v2, layer_idx=0)
    self.assertEqual(cache.get_seq_length(), 5)
    self.assertEqual(k_out2.shape, (1, 2, 5, 8))
    # Underlying buffer must remain unchanged in memory (no reallocations)
    self.assertEqual(layer._keys_buf.data_ptr(), buf_k_ptr)
    self.assertEqual(layer._values_buf.data_ptr(), buf_v_ptr)

    # Test crop resets length and retains backing buffer
    cache.crop(4)
    self.assertEqual(cache.get_seq_length(), 4)
    self.assertEqual(layer.keys.shape, (1, 2, 4, 8))
    self.assertEqual(layer._keys_buf.data_ptr(), buf_k_ptr)

  def test_patch_transformers_dynamic_cache(self):
    from torch_tpu._internal import cache as cache_lib

    orig_dynamic_cache = transformers.cache_utils.DynamicCache
    orig_dynamic_layer = transformers.cache_utils.DynamicLayer
    try:
      cache_lib.patch_transformers_dynamic_cache()
      self.assertIs(
          transformers.cache_utils.DynamicCache, cache_lib.InPlaceDynamicCache
      )
      self.assertIs(
          transformers.cache_utils.DynamicLayer, cache_lib.InPlaceDynamicLayer
      )
      # Instantiating standard DynamicCache now creates InPlaceDynamicCache
      c = transformers.cache_utils.DynamicCache(max_cache_len=32)
      self.assertIsInstance(c, cache_lib.InPlaceDynamicCache)
    finally:
      transformers.cache_utils.DynamicCache = orig_dynamic_cache
      transformers.cache_utils.DynamicLayer = orig_dynamic_layer

  def test_from_dynamic_cache_and_tpu_forward_hook(self):
    from torch_tpu._internal import cache as cache_lib

    ops = self._ops()
    std_cache = transformers.cache_utils.DynamicCache()
    k1 = torch.randn(1, 2, 4, 8, device=ops.device)
    v1 = torch.randn(1, 2, 4, 8, device=ops.device)
    std_cache.update(k1, v1, layer_idx=0)

    # Test conversion
    inplace_cache = cache_lib.InPlaceDynamicCache.from_dynamic_cache(
        std_cache, max_cache_len=16
    )
    self.assertIsInstance(inplace_cache, cache_lib.InPlaceDynamicCache)
    self.assertEqual(inplace_cache.get_seq_length(), 4)
    self.assertEqual(inplace_cache.layers[0].keys.shape, (1, 2, 4, 8))
    self.assertTrue(
        torch.equal(inplace_cache.layers[0].keys, std_cache.layers[0].keys)
    )

    # Test tpu_forward_hook
    args = (torch.tensor([1]),)
    kwargs = {"past_key_values": std_cache}
    new_args, new_kwargs = cache_lib.tpu_forward_hook(None, args, kwargs)
    self.assertIs(new_args, args)
    self.assertIsInstance(
        new_kwargs["past_key_values"], cache_lib.InPlaceDynamicCache
    )
    self.assertEqual(new_kwargs["past_key_values"].get_seq_length(), 4)

    # Test duck-typed DynamicCache (e.g. from v4 or external module)
    class ExternalDynamicLayer:

      def __init__(self, k, v):
        self.keys = k
        self.values = v
        self.is_initialized = True

    class ExternalDynamicCache:

      def __init__(self, layer):
        self.layers = [layer]

    ext_layer = ExternalDynamicLayer(k1, v1)
    # Set class name explicitly to DynamicLayer/DynamicCache
    ext_layer.__class__.__name__ = "DynamicLayer"
    ext_cache = ExternalDynamicCache(ext_layer)
    ext_cache.__class__.__name__ = "DynamicCache"

    _, ext_kwargs = cache_lib.tpu_forward_hook(
        None, (), {"past_key_values": ext_cache}
    )
    self.assertIsInstance(
        ext_kwargs["past_key_values"], cache_lib.InPlaceDynamicCache
    )
    self.assertEqual(ext_kwargs["past_key_values"].get_seq_length(), 4)

    # Test register_tpu_cache_hook on an nn.Module
    class DummyModule(nn.Module):

      def forward(self, x, past_key_values=None):
        return past_key_values

    mod = DummyModule()
    cache_lib.register_tpu_cache_hook(mod)
    out_cache = mod(torch.tensor([1]), past_key_values=std_cache)
    self.assertIsInstance(out_cache, cache_lib.InPlaceDynamicCache)

  def test_tpu_compile_installs_cache_hook(self):
    from torch_tpu._internal import cache as cache_lib

    std_cache = transformers.cache_utils.DynamicCache()

    model = _DummyDecodeModel()
    torch_tpu_compile.tpu_compile(model)

    # The upgrade is a forward pre-hook, so it is observable on the original
    # module; no need to pay for an actual TPU compilation here.
    out_cache = model(
        input_ids=torch.tensor([1]),
        past_key_values=transformers.cache_utils.DynamicCache(),
    )
    self.assertIsInstance(out_cache, cache_lib.InPlaceDynamicCache)

  def test_tpu_compile_cache_hook_is_registered_once(self):
    from torch_tpu._internal import compile as torch_tpu_compile

    model = _DummyDecodeModel()
    torch_tpu_compile.tpu_compile(model)
    torch_tpu_compile.tpu_compile(model)

    self.assertLen(model._forward_pre_hooks, 1)  # pylint: disable=protected-access

  def test_tpu_compile_passes_through_non_module(self):
    from torch_tpu._internal import compile as torch_tpu_compile

    def step(x):
      return x + 1

    self.assertIsNotNone(torch_tpu_compile.tpu_compile(step))

  def test_inplace_cache_reserves_real_trailing_slot(self):
    """The exposed view must never span its whole backing buffer.

    `repeat_kv()`'s `reshape()` guards on `Ne(head_dim * end, head_dim *
    buffer_len)`. Reserving a real trailing slot keeps `end` strictly below
    `buffer_len` so that guard can never be hit.
    """
    from torch_tpu._internal import cache as cache_lib

    ops = self._ops()
    max_cache_len = 6
    cache = cache_lib.InPlaceDynamicCache(max_cache_len=max_cache_len)
    keys = torch.randn(1, 2, 4, 8, device=ops.device)
    values = torch.randn(1, 2, 4, 8, device=ops.device)
    cache.update(keys, values, layer_idx=0)
    layer = cache.layers[0]

    # The headroom is a real tensor dimension, not merely padded storage: a
    # view of a padded allocation does not survive functionalization.
    self.assertEqual(layer._keys_buf.shape[-2], max_cache_len + 1)
    self.assertEqual(layer._values_buf.shape[-2], max_cache_len + 1)
    self.assertTrue(layer._keys_buf.is_contiguous())

    # Fill the cache right up to its usable capacity.
    for _ in range(max_cache_len - 4):
      cache.update(
          torch.randn(1, 2, 1, 8, device=ops.device),
          torch.randn(1, 2, 1, 8, device=ops.device),
          layer_idx=0,
      )
    self.assertEqual(cache.get_seq_length(), max_cache_len)
    self.assertLess(layer.keys.shape[-2], layer._keys_buf.shape[-2])
    self.assertFalse(layer.keys.is_contiguous())
    self.assertFalse(layer.values.is_contiguous())

  def test_exposed_view_keeps_full_buffer_stride(self):
    """The exposed view must carry the stride of the longer backing buffer.

    That is what decouples the traced upper bound of the sequence length (at
    most `capacity`) from the head-dimension stride (`capacity + 1` slots), so
    `repeat_kv()`'s reshape guard `Ne(head_dim * length, stride)` is
    unsatisfiable without any caller-side `mark_dynamic`.
    """
    from torch_tpu._internal import cache as cache_lib

    ops = self._ops()
    max_cache_len = 6
    head_dim = 8
    cache = cache_lib.InPlaceDynamicCache(max_cache_len=max_cache_len)
    cache.update(
        torch.randn(1, 2, 4, head_dim, device=ops.device),
        torch.randn(1, 2, 4, head_dim, device=ops.device),
        layer_idx=0,
    )
    layer = cache.layers[0]
    expected_stride = (max_cache_len + 1) * head_dim

    for _ in range(max_cache_len - 4 + 1):
      for tensor in (layer.keys, layer.values):
        self.assertEqual(tensor.stride(1), expected_stride)
        self.assertNotEqual(tensor.shape[-2] * head_dim, expected_stride)
        self.assertFalse(tensor.is_contiguous())
      if cache.get_seq_length() >= max_cache_len:
        break
      cache.update(
          torch.randn(1, 2, 1, head_dim, device=ops.device),
          torch.randn(1, 2, 1, head_dim, device=ops.device),
          layer_idx=0,
      )
    self.assertEqual(cache.get_seq_length(), max_cache_len)

  def test_inplace_cache_growth_preserves_headroom(self):
    from torch_tpu._internal import cache as cache_lib

    ops = self._ops()
    cache = cache_lib.InPlaceDynamicCache(max_cache_len=2)
    keys = torch.randn(1, 2, 2, 8, device=ops.device)
    values = torch.randn(1, 2, 2, 8, device=ops.device)
    cache.update(keys, values, layer_idx=0)
    layer = cache.layers[0]
    self.assertEqual(layer._keys_buf.shape[-2], 3)

    # Exceeding the usable capacity reallocates, and the replacement buffer
    # must keep the reserved trailing slot.
    for _ in range(3):
      cache.update(
          torch.randn(1, 2, 1, 8, device=ops.device),
          torch.randn(1, 2, 1, 8, device=ops.device),
          layer_idx=0,
      )
    self.assertEqual(cache.get_seq_length(), 5)
    self.assertGreater(layer._keys_buf.shape[-2], layer.keys.shape[-2])
    self.assertFalse(layer.keys.is_contiguous())
    self.assertTrue(torch.equal(layer.keys[:, :, :2, :], keys))


if __name__ == "__main__":
  absltest.main()
