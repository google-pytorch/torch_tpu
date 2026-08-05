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
from torch_tpu._internal.utils import test_utils as utils
from examples.gemma4 import model
from tests import seed_test_utils
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model


class Gemma4CorrectnessTest(seed_test_utils.RepeatableTest):

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

    utils.assert_close(hf_hidden, custom_output)


if __name__ == '__main__':
  absltest.main()
