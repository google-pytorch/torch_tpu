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

"""Tests for standalone native Gemma4 model_baseline implementation."""

from absl.testing import absltest
import torch
from examples.gemma4 import model_baseline
from tests import seed_test_utils


class Gemma4ModelBaselineTest(seed_test_utils.RepeatableTest):

  def test_dense_model_forward(self):
    config = model_baseline.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    m = model_baseline.Gemma4Model(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = m(input_ids)

    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_moe_model_forward(self):
    config = model_baseline.Gemma4Config(
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

    m = model_baseline.Gemma4Model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = m(input_ids)

    self.assertEqual(output.shape, (2, 10, config.hidden_size))

  def test_causal_lm_forward(self):
    config = model_baseline.Gemma4Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    causal_lm = model_baseline.Gemma4ForCausalLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = causal_lm(input_ids)

    self.assertEqual(logits.shape, (2, 10, config.vocab_size))


if __name__ == "__main__":
  absltest.main()
