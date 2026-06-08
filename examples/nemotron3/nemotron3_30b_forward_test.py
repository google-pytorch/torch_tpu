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

"""Tests for production-fidelity Nemotron3 model baseline."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from examples.nemotron3 import nemotron3
from transformers import NemotronHConfig, NemotronHForCausalLM


class NemotronHForwardTest(parameterized.TestCase):

  def _get_toy_config(self) -> nemotron3.NemotronHConfig:
    return nemotron3.NemotronHConfig(
        vocab_size=100,
        hidden_size=16,
        layers_block_type=["mamba", "attention", "moe"],
        # Scale down layer dimensions for fast tests
        ssm_state_size=8,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        conv_kernel=4,
        chunk_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        use_bias=False,
    )

  def test_forward(self):
    config = self._get_toy_config()
    model = nemotron3.NemotronHForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(input_ids)
    self.assertEqual(output.shape, (2, 8, config.vocab_size))

  def test_tpu_compile(self):
    device = torch.device("tpu")
    config = self._get_toy_config()
    model = nemotron3.NemotronHForCausalLM(config).to(device)
    compiled_model = torch.compile(model, backend="tpu", dynamic=False)
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
    output = compiled_model(input_ids)
    self.assertEqual(output.shape, (2, 8, config.vocab_size))

  def test_equivalence_with_huggingface(self):
    config = self._get_toy_config()
    hf_config = NemotronHConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        layer_types=config.layers_block_type,
        ssm_state_size=config.ssm_state_size,
        mamba_num_heads=config.mamba_num_heads,
        mamba_head_dim=config.mamba_head_dim,
        n_groups=config.n_groups,
        conv_kernel=config.conv_kernel,
        expand=config.expand,
        time_step_min=config.time_step_min,
        time_step_max=config.time_step_max,
        time_step_limit=config.time_step_limit,
        use_conv_bias=config.use_conv_bias,
        chunk_size=config.chunk_size,
        mamba_proj_bias=config.mamba_proj_bias,
        n_routed_experts=config.n_routed_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        routed_scaling_factor=config.routed_scaling_factor,
        n_group=config.n_group,
        topk_group=config.topk_group,
        norm_topk_prob=config.norm_topk_prob,
        moe_intermediate_size=config.moe_intermediate_size,
        moe_shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
        moe_latent_size=config.moe_latent_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        attention_dropout=config.attention_dropout,
        mlp_bias=config.mlp_bias,
        mlp_hidden_act=config.mlp_hidden_act,
        use_bias=config.use_bias,
        layer_norm_epsilon=config.layer_norm_epsilon,
    )
    hf_config._attn_implementation = "eager"

    hf_model = NemotronHForCausalLM(hf_config).eval()
    local_model = nemotron3.NemotronHForCausalLM(config).eval()

    local_model.load_state_dict(hf_model.state_dict())

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.int64,
    )
    with torch.no_grad():
      hf_out = hf_model(input_ids, attention_mask=attention_mask).logits
      local_out = local_model(input_ids, attention_mask=attention_mask)

    torch.testing.assert_close(local_out, hf_out, rtol=1e-5, atol=1e-5)

  def test_cpu_tpu_eager_parity(self):
    config = self._get_toy_config()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.int64,
    )

    model_cpu = nemotron3.NemotronHForCausalLM(config).eval()
    with torch.no_grad():
      output_cpu = model_cpu(input_ids, attention_mask=attention_mask)

    device = torch.device("tpu")
    model_tpu = nemotron3.NemotronHForCausalLM(config).to(device).eval()
    model_tpu.load_state_dict(model_cpu.state_dict())

    input_ids_tpu = input_ids.to(device)
    attention_mask_tpu = attention_mask.to(device)
    with torch.no_grad():
      output_tpu = model_tpu(
          input_ids_tpu, attention_mask=attention_mask_tpu
      ).cpu()

    torch.testing.assert_close(output_tpu, output_cpu, rtol=1e-2, atol=1e-2)

  def test_cpu_tpu_compiled_parity(self):
    config = self._get_toy_config()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.int64,
    )

    model_cpu = nemotron3.NemotronHForCausalLM(config).eval()
    with torch.no_grad():
      output_cpu = model_cpu(input_ids, attention_mask=attention_mask)

    device = torch.device("tpu")
    model_tpu = nemotron3.NemotronHForCausalLM(config).to(device).eval()
    model_tpu.load_state_dict(model_cpu.state_dict())
    compiled_model = torch.compile(model_tpu, backend="tpu", dynamic=False)

    input_ids_tpu = input_ids.to(device)
    attention_mask_tpu = attention_mask.to(device)
    with torch.no_grad():
      output_compiled = compiled_model(
          input_ids_tpu, attention_mask=attention_mask_tpu
      ).cpu()

    torch.testing.assert_close(
        output_compiled, output_cpu, rtol=1e-2, atol=1e-2
    )


if __name__ == "__main__":
  absltest.main()
