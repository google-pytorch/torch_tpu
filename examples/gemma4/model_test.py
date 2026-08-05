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

from unittest import mock
from absl.testing import absltest
import torch
import torch_tpu
from examples.gemma4 import model
from tests import seed_test_utils


class Gemma4ModelTest(seed_test_utils.RepeatableTest):

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

  def test_checkpoint_not_called_when_disabled_in_train_mode(self):
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
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    with mock.patch('torch.utils.checkpoint.checkpoint') as mock_ckpt:
      m(input_ids)
      mock_ckpt.assert_not_called()

  def test_checkpoint_not_called_when_enabled_in_eval_mode(self):
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
    m.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    with mock.patch('torch.utils.checkpoint.checkpoint') as mock_ckpt:
      m(input_ids)
      mock_ckpt.assert_not_called()

  def test_checkpoint_called_when_enabled_in_train_mode(self):
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
    m.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    with mock.patch(
        'torch.utils.checkpoint.checkpoint',
        wraps=torch.utils.checkpoint.checkpoint,
    ) as mock_ckpt:
      m(input_ids)
      self.assertEqual(mock_ckpt.call_count, config.num_hidden_layers)

  def test_checkpoint_manual_attribute_default_kwargs_fallback(self):
    config = model.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    m = model.Gemma4Model(config)
    m.gradient_checkpointing = True
    m.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    with mock.patch(
        'torch.utils.checkpoint.checkpoint',
        wraps=torch.utils.checkpoint.checkpoint,
    ) as mock_ckpt:
      m(input_ids)
      self.assertEqual(mock_ckpt.call_count, config.num_hidden_layers)
      for call in mock_ckpt.call_args_list:
        self.assertFalse(call.kwargs.get('use_reentrant', True))


if __name__ == '__main__':
  absltest.main()
