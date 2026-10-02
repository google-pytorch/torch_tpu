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

"""Tests for Gemma4 model implementation."""

from absl.testing import absltest
import torch
import torch_tpu
from examples.gemma4 import model


class Gemma4ModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(1234)

  def tearDown(self):
    if torch.accelerator.is_available():
      torch.accelerator.synchronize()
    super().tearDown()

  def test_dense_model_forward(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    m = model.Gemma4Model(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = m(input_ids)

    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_moe_model_forward(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        enable_moe=True,
        num_experts=4,
        top_k_experts=2,
        expert_dim=64,
        moe_dense_hidden_dim=64,
    )

    m = model.Gemma4Model(config)
    for module in m.modules():
      if isinstance(module, model.Gemma4MoERagged):
        module.ragged_dot_impl = lambda x, w, sizes: torch.matmul(x, w[0])

    input_ids = torch.randint(0, config.vocab_size, (2, 10))

    output = m(input_ids)

    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_multimodal_model_forward(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vision_proj_dim=64,
        audio_proj_dim=64,
    )
    m = model.Gemma4Model(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    pixel_values = torch.randn(2, 5, 64)
    audio_features = torch.randn(2, 5, 64)

    output = m(
        input_ids, pixel_values=pixel_values, audio_features=audio_features
    )

    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_sliding_attention_cpu_fallback(self):
    config = model.Gemma4Config(
        num_hidden_layers=1,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        sliding_window=16,
        layer_types=['sliding_attention'],
    )
    attn = model.Gemma4Attention(config, layer_idx=0)
    hidden_states = torch.randn(2, 32, config.hidden_size)
    position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)

    # On CPU, query_states.device.type is 'cpu'. use_splash must be False.
    output, _ = attn(hidden_states, position_ids)
    self.assertEqual(output.shape, (2, 32, config.hidden_size))

  def test_sliding_attention_with_custom_attention_mask(self):
    config = model.Gemma4Config(
        num_hidden_layers=1,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        sliding_window=16,
        layer_types=['sliding_attention'],
    )
    attn = model.Gemma4Attention(config, layer_idx=0)
    hidden_states = torch.randn(2, 32, config.hidden_size)
    position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)
    attention_mask = torch.zeros(2, 1, 32, 32)

    # When attention_mask is passed, use_splash must be False.
    output, _ = attn(hidden_states, position_ids, attention_mask=attention_mask)
    self.assertEqual(output.shape, (2, 32, config.hidden_size))

  def test_gradient_checkpointing_disabled_by_default(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    m = model.Gemma4Model(config)
    self.assertFalse(m.gradient_checkpointing)

  def test_gradient_checkpointing_training_without_checkpointing(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    m = model.Gemma4Model(config)
    m.train()
    self.assertFalse(m.gradient_checkpointing)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = m(input_ids)
    loss = output.sum()
    loss.backward()
    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_gradient_checkpointing_enabled_training(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    m = model.Gemma4Model(config)
    m.gradient_checkpointing_enable()
    self.assertTrue(m.gradient_checkpointing)
    m.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = m(input_ids)
    loss = output.sum()
    loss.backward()
    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def _assert_layer_calls(self, m, input_ids, *, forward, backward):
    """Runs forward + backward and checks how often decoder layers ran.

    Gradient checkpointing discards layer activations in the forward pass and
    re-runs every checkpointed layer during backward, so the backward count is
    the number of recomputed layers. Forward pre-hooks are used (rather than
    forward hooks) because non-reentrant checkpointing may stop a recompute
    early, before the layer's forward returns.

    Args:
      m: The model under test.
      input_ids: Input token ids.
      forward: Expected decoder-layer calls during the forward pass.
      backward: Expected decoder-layer calls (recomputes) during backward.
    """
    calls = []
    handles = [
        layer.register_forward_pre_hook(lambda *_: calls.append(1))
        for layer in m.layers
    ]
    try:
      output = m(input_ids)
      forward_calls = len(calls)
      output.sum().backward()
      backward_calls = len(calls) - forward_calls
    finally:
      for handle in handles:
        handle.remove()
    self.assertEqual(
        (forward_calls, backward_calls),
        (forward, backward),
        msg='(forward, backward) decoder-layer call counts',
    )

  def _small_config(self):
    return model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )

  def test_no_recompute_when_checkpointing_disabled_in_train_mode(self):
    config = self._small_config()
    m = model.Gemma4Model(config)
    m.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    self._assert_layer_calls(
        m, input_ids, forward=config.num_hidden_layers, backward=0
    )

  def test_no_recompute_when_checkpointing_enabled_in_eval_mode(self):
    config = self._small_config()
    m = model.Gemma4Model(config)
    m.gradient_checkpointing_enable()
    m.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    self._assert_layer_calls(
        m, input_ids, forward=config.num_hidden_layers, backward=0
    )

  def test_layers_recomputed_when_checkpointing_enabled_in_train_mode(self):
    config = self._small_config()
    m = model.Gemma4Model(config)
    m.gradient_checkpointing_enable()
    m.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    self._assert_layer_calls(
        m,
        input_ids,
        forward=config.num_hidden_layers,
        backward=config.num_hidden_layers,
    )

  def test_checkpoint_manual_attribute_default_kwargs_fallback(self):
    # Setting the attribute directly skips gradient_checkpointing_enable(), so
    # the forward falls back to default checkpoint kwargs. Those must select
    # non-reentrant checkpointing: reentrant checkpointing rejects the
    # skip_sliding_mask keyword argument and would raise here.
    config = self._small_config()
    m = model.Gemma4Model(config)
    m.gradient_checkpointing = True
    m.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    self._assert_layer_calls(
        m,
        input_ids,
        forward=config.num_hidden_layers,
        backward=config.num_hidden_layers,
    )


if __name__ == '__main__':
  absltest.main()
