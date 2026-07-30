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

"""Numerical correctness tests comparing Hugging Face Gemma4 to custom TPU implementation in model.py."""

from absl.testing import absltest
import torch
import torch_tpu
from examples.gemma4 import model
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model


try:
  from torch_tpu.ops import splash_attention
except (ImportError, ModuleNotFoundError):
  splash_attention = None


class Gemma4CorrectnessTest(absltest.TestCase):

  def test_equivalence_dense_model(self):
    config_dict = {
        'vocab_size': 100,
        'vocab_size_per_layer_input': 100,
        'hidden_size_per_layer_input': 0,
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_hidden_layers': 1,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'head_dim': 32,
        'global_head_dim': 32,
        'rms_norm_eps': 1e-6,
        'attention_bias': False,
        'layer_types': ['full_attention'],
        'rope_parameters': {
            'full_attention': {
                'rope_type': 'proportional',
                'partial_rotary_factor': 0.25,
                'rope_theta': 1000000.0,
            }
        },
    }

    try:
      hf_config = Gemma4Config(text_config=config_dict)

      hf_config.text_config.__class__ = Gemma4TextConfig
      hf_model = Gemma4Model(hf_config)
    except (AttributeError, TypeError, ValueError, KeyError, RuntimeError) as e:
      self.skipTest(
          f'transformers.Gemma4Model not available or failed to init: {e}'
      )

    custom_config = model.Gemma4Config(**config_dict)
    custom_model = model.Gemma4Model(custom_config)

    # Copy weights
    hf_state_dict = hf_model.state_dict()
    new_state_dict = {}
    for k, v in hf_state_dict.items():
      if k.startswith('language_model.'):
        new_state_dict[k[len('language_model.') :]] = v
      else:
        new_state_dict[k] = v
    custom_model.load_state_dict(new_state_dict, strict=True)

    device = 'cpu'
    hf_model.to(device)
    custom_model.to(device)
    hf_model.eval()
    custom_model.eval()

    input_ids = torch.randint(
        0, custom_config.vocab_size, (2, 10), device=device
    )

    with torch.no_grad():
      hf_output = hf_model(input_ids)
      custom_output = custom_model(input_ids)

    if hasattr(hf_output, 'last_hidden_state'):
      hf_hidden = hf_output.last_hidden_state
    else:
      hf_hidden = hf_output

    torch.testing.assert_close(hf_hidden, custom_output)

  def test_splash_mqa_output_layout_specification(self):
    """Verify splash_mqa / splash_sdpa output layout matches HEAD_DIM_MINOR (B, S, H, D)."""
    if splash_attention is None:
      self.skipTest('splash_attention module not available')
    try:
      torch.tensor([1.0]).to('tpu')
    except (RuntimeError, AttributeError):
      self.skipTest('TPU device not available in CPU test environment')

    bsz, seq_len, num_heads, head_dim = 2, 512, 4, 32
    # Input in HEAD_DIM_MINOR / (B, S, H, D) format
    q = torch.randn(bsz, seq_len, num_heads, head_dim)
    k = torch.randn(bsz, seq_len, 1, head_dim)
    v = torch.randn(bsz, seq_len, 1, head_dim)

    # splash_sdpa with QKVLayout.HEAD_DIM_MINOR (B, S, H, D)
    out = splash_attention.splash_sdpa(
        q,
        k,
        v,
        scale=1.0,
        is_causal=True,
        local_window_size=16,
        enable_gqa=True,
        q_layout='HEAD_DIM_MINOR',
        k_layout='HEAD_DIM_MINOR',
        v_layout='HEAD_DIM_MINOR',
    )
    # Expected output layout is (B, S, H, D) directly matching downstream o_proj input format
    self.assertEqual(
        out.shape,
        (bsz, seq_len, num_heads, head_dim),
        f'Expected output shape (B, S, H, D) = ({bsz}, {seq_len}, {num_heads},'
        f' {head_dim}), got {out.shape}',
    )


if __name__ == '__main__':
  absltest.main()
