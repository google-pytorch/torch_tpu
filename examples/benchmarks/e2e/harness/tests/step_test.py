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


if __name__ == "__main__":
  absltest.main()
