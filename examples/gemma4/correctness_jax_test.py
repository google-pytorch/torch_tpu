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

"""Numerical correctness tests for Gemma4 implementation against JAX."""

from absl.testing import absltest
from gemma.gm.nn.gemma4 import _config as jax_config
from gemma.gm.nn.gemma4 import _gemma4 as jax_gemma4
from gemma.gm.nn.gemma4 import _modules as jax_modules
from gemma.gm.nn.gemma4._gemma4 import Gemma4_E2B
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from torch_tpu._internal.utils import utils
from examples.gemma4 import model as pt_model

HIDDEN_SIZE = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
NUM_LAYERS = 1
VOCAB_SIZE = 262144


class MiniGemma4(jax_gemma4._Gemma4Base):
  config: jax_config.TransformerConfig = jax_config.TransformerConfig(
      num_embed=262144,
      embed_dim=128,
      hidden_dim=128 * 4,
      num_heads=4,
      head_dim=32,
      num_kv_heads=2,
      attention_types=tuple([jax_modules.AttentionType.GLOBAL] * NUM_LAYERS),
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      qk_norm_with_scale=True,
      final_logit_softcap=30.0,
      global_rope_proportion=1.0,
      local_rope_proportion=1.0,
  )


def attention_forward_no_scale(
    self, hidden_states, position_ids, attention_mask=None, **kwargs
):
  bsz, seq_len, _ = hidden_states.shape
  query_states = (
      self.q_proj(hidden_states)
      .view(bsz, seq_len, self.num_heads, self.head_dim)
      .transpose(1, 2)
  )
  key_states = (
      self.k_proj(hidden_states)
      .view(bsz, seq_len, self.current_kv_heads, self.head_dim)
      .transpose(1, 2)
  )
  value_states = (
      self.v_proj(hidden_states)
      .view(bsz, seq_len, self.current_kv_heads, self.head_dim)
      .transpose(1, 2)
  )
  print(f'DEBUG PT Q_proj mean: {query_states.float().mean().item()}')
  print(f'DEBUG PT K_proj mean: {key_states.float().mean().item()}')
  print(f'DEBUG PT V_proj mean: {value_states.float().mean().item()}')

  query_states = self.q_norm(query_states)
  key_states = self.k_norm(key_states)
  value_states = self.v_norm(value_states)
  print(f'DEBUG PT Q_norm mean: {query_states.float().mean().item()}')
  print(f'DEBUG PT K_norm mean: {key_states.float().mean().item()}')

  cos, sin = self.rotary_emb(query_states, position_ids)
  query_states, key_states = pt_model.apply_rotary_pos_emb(
      query_states, key_states, cos, sin
  )
  print(f'DEBUG PT Q_rope mean: {query_states.float().mean().item()}')
  print(f'DEBUG PT K_rope mean: {key_states.float().mean().item()}')

  num_key_value_groups = self.num_heads // self.current_kv_heads
  if num_key_value_groups > 1:
    key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
    value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

  # UN-SCALED dot product attention
  attn_weights = torch.matmul(
      query_states, key_states.transpose(-1, -2)
  )  # Removed division by sqrt(head_dim)

  if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

  print(f'DEBUG PT Pre-softmax mean: {attn_weights.float().mean().item()}')

  attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
      query_states.dtype
  )
  print(f'DEBUG PT Post-softmax mean: {attn_probs.float().mean().item()}')

  attn_output = torch.matmul(attn_probs, value_states)
  attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
  print(
      f'DEBUG PT attn_output_pre_proj mean: {attn_output.float().mean().item()}'
  )
  attn_output = self.o_proj(attn_output)
  print(f'DEBUG PT Final output mean: {attn_output.float().mean().item()}')
  return attn_output, None


class Gemma4JAXCorrectnessTest(absltest.TestCase):

  def test_equivalence_dense_model(self):
    import numpy as np

    np.random.seed(42)
    torch.manual_seed(42)

    # PyTorch Config
    p_config = pt_model.Gemma4Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 4,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        layer_types=['full_attention'] * NUM_LAYERS,
        use_post_ffw_norm=True,
        rope_theta=10000.0,
        global_rope_proportion=1.0,
    )

    # Instantiate JAX model
    j_model = MiniGemma4(text_only=True)

    # Initialize JAX model parameters
    rng = jax.random.PRNGKey(0)
    dummy_tokens = jnp.ones((1, 10), dtype=jnp.int32)
    variables = j_model.init(rng, dummy_tokens)
    params = variables['params']

    # Monkey-patch attention to remove scaling
    pt_model.Gemma4Attention.forward = attention_forward_no_scale

    # Instantiate PyTorch model
    p_model = pt_model.Gemma4Model(p_config)

    # Copy weights from JAX to PyTorch
    with torch.no_grad():
      # Embedding
      p_model.embed_tokens.weight.copy_(
          torch.from_numpy(
              np.array(params['embedder']['input_embedding']).astype(np.float32)
          ).to(torch.bfloat16)
      )

      # Final norm
      p_model.norm.weight.copy_(
          torch.from_numpy(
              np.array(params['final_norm']['scale']).astype(np.float32)
          ).to(torch.bfloat16)
      )

      for i in range(NUM_LAYERS):
        j_layer = params[f'layer_{i}']
        p_layer = p_model.layers[i]

        # Attention
        p_layer.input_layernorm.weight.copy_(
            torch.from_numpy(
                np.array(j_layer['pre_attention_norm']['scale']).astype(
                    np.float32
                )
            ).to(torch.bfloat16)
        )

        # Q, K, V, O
        print(
            f'DEBUG layer {i}: type of q_einsum:'
            f" {type(j_layer['attn']['q_einsum'])}"
        )
        if hasattr(j_layer['attn']['q_einsum'], 'keys'):
          print(
              f'DEBUG layer {i}: keys of q_einsum:'
              f" {list(j_layer['attn']['q_einsum'].keys())}"
          )
        print(
            f"DEBUG layer {i}: shape of q_einsum['w']:"
            f" {j_layer['attn']['q_einsum']['w'].shape}"
        )
        j_q = np.array(j_layer['attn']['q_einsum']['w']).astype(np.float32)
        p_layer.self_attn.q_proj.weight.copy_(
            torch.from_numpy(
                j_q.transpose(0, 2, 1).reshape(
                    NUM_HEADS * HEAD_DIM, HIDDEN_SIZE
                )
            ).to(torch.bfloat16)
        )

        print(
            f"DEBUG layer {i}: shape of kv_einsum['w']:"
            f" {j_layer['attn']['kv_einsum']['w'].shape}"
        )
        j_kv = np.array(j_layer['attn']['kv_einsum']['w']).astype(np.float32)
        p_layer.self_attn.k_proj.weight.copy_(
            torch.from_numpy(
                j_kv[0]
                .transpose(0, 2, 1)
                .reshape(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
            ).to(torch.bfloat16)
        )
        p_layer.self_attn.v_proj.weight.copy_(
            torch.from_numpy(
                j_kv[1]
                .transpose(0, 2, 1)
                .reshape(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
            ).to(torch.bfloat16)
        )

        print(
            f"DEBUG layer {i}: shape of attn_vec_einsum['w']:"
            f" {j_layer['attn']['attn_vec_einsum']['w'].shape}"
        )
        j_o = np.array(j_layer['attn']['attn_vec_einsum']['w']).astype(
            np.float32
        )
        p_layer.self_attn.o_proj.weight.copy_(
            torch.from_numpy(
                j_o.reshape(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE).T
            ).to(torch.bfloat16)
        )

        p_layer.self_attn.q_norm.weight.copy_(
            torch.from_numpy(
                np.array(j_layer['attn']['query_norm']['scale']).astype(
                    np.float32
                )
            ).to(torch.bfloat16)
        )
        p_layer.self_attn.k_norm.weight.copy_(
            torch.from_numpy(
                np.array(j_layer['attn']['key_norm']['scale']).astype(
                    np.float32
                )
            ).to(torch.bfloat16)
        )

        if p_layer.post_attention_layernorm is not None:
          p_layer.post_attention_layernorm.weight.copy_(
              torch.from_numpy(
                  np.array(j_layer['post_attention_norm']['scale']).astype(
                      np.float32
                  )
              ).to(torch.bfloat16)
          )

        # MLP
        p_layer.pre_feedforward_layernorm.weight.copy_(
            torch.from_numpy(
                np.array(j_layer['pre_ffw_norm']['scale']).astype(np.float32)
            ).to(torch.bfloat16)
        )

        j_gate_up = np.array(j_layer['mlp']['gating_einsum']).astype(np.float32)
        p_layer.mlp.gate_proj.weight.copy_(
            torch.from_numpy(j_gate_up[0]).to(torch.bfloat16)
        )
        p_layer.mlp.up_proj.weight.copy_(
            torch.from_numpy(j_gate_up[1]).to(torch.bfloat16)
        )

        j_down = np.array(j_layer['mlp']['linear']).astype(np.float32)
        p_layer.mlp.down_proj.weight.copy_(
            torch.from_numpy(j_down.T).to(torch.bfloat16)
        )

        if p_layer.post_feedforward_layernorm is not None:
          p_layer.post_feedforward_layernorm.weight.copy_(
              torch.from_numpy(
                  np.array(j_layer['post_ffw_norm']['scale']).astype(np.float32)
              ).to(torch.bfloat16)
          )

    # Run forward pass and compare
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 10))
    j_input = jnp.array(input_ids.numpy())

    # Create causal mask
    seq_len = 10  # matching input_ids shape (1, 10)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    # PyTorch mask (float with -inf)
    attention_mask = torch.zeros(seq_len, seq_len)
    attention_mask.masked_fill_(causal_mask, float('-inf'))
    attention_mask = attention_mask.view(1, 1, seq_len, seq_len)

    # JAX mask (boolean with True for allowed)
    jax_mask = ~causal_mask
    jax_mask = jax_mask.view(1, seq_len, seq_len)

    # Compare embeddings
    j_embed = variables['params']['embedder']['input_embedding'][
        j_input
    ] * np.sqrt(HIDDEN_SIZE)
    p_embed = p_model.embed_tokens(input_ids) * np.sqrt(HIDDEN_SIZE)
    try:
      np.testing.assert_allclose(
          np.array(j_embed).astype(np.float32),
          p_embed.detach().numpy().astype(np.float32),
          rtol=1e-3,
          atol=1e-3,
      )
      print('DEBUG: Embeddings match!')
    except AssertionError as e:
      print(f'DEBUG: Embeddings mismatch! {e}')

    j_output = j_model.apply(
        variables,
        j_input,
        attention_mask=jnp.array(jax_mask.numpy()),
        return_hidden_states=True,
    )
    p_output = p_model(input_ids, attention_mask=attention_mask)

    # Compare hidden states
    j_hidden = np.array(j_output.hidden_states)
    p_hidden = p_output.detach().numpy()

    np.testing.assert_allclose(
        j_hidden.astype(np.float32),
        p_hidden.astype(np.float32),
        rtol=1e-2,
        atol=0.2,
    )


if __name__ == '__main__':
  absltest.main()
