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

"""Unit tests for ragged_moe module."""

from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from perflab.model_utils import ragged_moe
from tests import seed_test_utils
from transformers.models.gemma4 import configuration_gemma4
from transformers.models.qwen3_5_moe import configuration_qwen3_5_moe


def manual_ragged_dot(
    x: torch.Tensor, w: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
  """CPU reference implementation of ragged dot for testing."""
  w = w.to(dtype=x.dtype, device=x.device)
  out = torch.zeros(x.shape[0], w.shape[-1], device=x.device, dtype=x.dtype)
  idx = 0
  for i, g in enumerate(groups):
    if g == 0:
      continue
    x_slice = x[idx : (idx + g), :]
    expert_w = w[i]
    out[idx : (idx + g), :] = x_slice @ expert_w
    idx += g
  return out


class RaggedMoeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.config = configuration_qwen3_5_moe.Qwen3_5MoeTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    )

  def test_forward_output_shape(self):
    layer = ragged_moe.RaggedMoeQwen35(
        config=self.config,
        is_tensor_parallel=False,
        ragged_dot_impl=manual_ragged_dot,
    )
    batch_size = 2
    seq_len = 8
    hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)

    output, router_logits = layer(hidden_states)

    self.assertEqual(
        output.shape, (batch_size, seq_len, self.config.hidden_size)
    )
    self.assertEqual(
        router_logits.shape,
        (batch_size * seq_len, self.config.num_experts),
    )

  def test_missing_ragged_dot_impl_raises(self):
    layer = ragged_moe.RaggedMoeQwen35(
        config=self.config,
        is_tensor_parallel=False,
        ragged_dot_impl=None,
    )
    hidden_states = torch.randn(2, 4, self.config.hidden_size)
    if layer.ragged_dot_impl is None:
      with self.assertRaises(  # ASSERT_RAISES_OK=Test runtime fallback.
          RuntimeError
      ):
        layer(hidden_states)

  def test_invalid_tensor_parallel_world_size_raises(self):
    self.config.moe_intermediate_size = 127
    with mock.patch("torch.distributed.is_initialized", return_value=True):
      with mock.patch("torch.distributed.get_world_size", return_value=8):
        with self.assertRaises(  # ASSERT_RAISES_OK=Test TP divisibility.
            ValueError
        ):
          ragged_moe.RaggedMoeQwen35(
              config=self.config,
              is_tensor_parallel=True,
          )

  def test_tensor_parallel_forward_all_reduce(self):
    layer = ragged_moe.RaggedMoeQwen35(
        config=self.config,
        is_tensor_parallel=True,
        ragged_dot_impl=manual_ragged_dot,
    )
    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)

    with mock.patch("torch.distributed.all_reduce") as mock_all_reduce:
      output, router_logits = layer(hidden_states)
      mock_all_reduce.assert_called_once()

    self.assertEqual(
        output.shape, (batch_size, seq_len, self.config.hidden_size)
    )
    self.assertEqual(
        router_logits.shape,
        (batch_size * seq_len, self.config.num_experts),
    )


class RaggedExpertsGemma4Test(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.config = configuration_gemma4.Gemma4TextConfig(
        hidden_size=32,
        moe_intermediate_size=48,
        num_experts=4,
        top_k_experts=2,
        hidden_activation="gelu_pytorch_tanh",
    )

  def test_forward_matches_per_token_reference(self):
    experts = ragged_moe.RaggedExpertsGemma4(
        config=self.config, ragged_dot_impl=manual_ragged_dot
    )
    num_tokens = 6
    hidden_states = torch.randn(num_tokens, self.config.hidden_size)
    top_k_index = torch.stack([
        torch.randperm(self.config.num_experts)[: self.config.top_k_experts]
        for _ in range(num_tokens)
    ])
    top_k_weights = torch.softmax(
        torch.randn(num_tokens, self.config.top_k_experts), dim=-1
    )

    output = experts(hidden_states, top_k_index, top_k_weights)

    expected = torch.zeros_like(hidden_states)
    for t in range(num_tokens):
      for k in range(self.config.top_k_experts):
        e = int(top_k_index[t, k])
        x = hidden_states[t]
        h = (x @ experts.up[e]) * experts.act_fn(x @ experts.gate[e])
        expected[t] += top_k_weights[t, k] * (h @ experts.down[e])
    test_utils.assert_close(output, expected, rtol=1e-4, atol=1e-5)

  def test_constructs_parameters_in_requested_dtype(self):
    experts = ragged_moe.RaggedExpertsGemma4(
        config=self.config, dtype=torch.bfloat16
    )
    self.assertEqual(experts.up.dtype, torch.bfloat16)
    self.assertEqual(experts.gate.dtype, torch.bfloat16)
    self.assertEqual(experts.down.dtype, torch.bfloat16)

  def test_missing_ragged_dot_impl_raises(self):
    experts = ragged_moe.RaggedExpertsGemma4(config=self.config)
    experts.ragged_dot_impl = None
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Test runtime fallback.
        RuntimeError, "torch.tpu.ops.ragged_dot is not available"
    ):
      experts(
          torch.randn(2, self.config.hidden_size),
          torch.zeros(2, self.config.top_k_experts, dtype=torch.long),
          torch.ones(2, self.config.top_k_experts),
      )


if __name__ == "__main__":
  absltest.main()
