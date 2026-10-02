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
import torch.nn.functional as F
from torch_tpu._internal.utils import test_utils
from perflab.model_utils import ragged_moe
from tests import seed_test_utils
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


def dense_moe_reference(
    layer: ragged_moe.RaggedMoeQwen35,
    hidden_states: torch.Tensor,
    sparse_scale: float = 1.0,
) -> torch.Tensor:
  """Computes the MoE output token by token, without sorting or grouping.

  Args:
    layer: The layer whose weights are used.
    hidden_states: Input of shape [batch, seq, hidden].
    sparse_scale: Factor applied to the routed-expert output, e.g. the number of
      simulated tensor-parallel ranks whose partial sums are all-reduced.

  Returns:
    The expected layer output of shape [batch, seq, hidden].
  """
  batch_size, seq_len, hidden_size = hidden_states.shape
  h = hidden_states.reshape(-1, hidden_size)
  with torch.no_grad():
    shared = layer.shared_expert(h) * torch.sigmoid(layer.shared_expert_gate(h))
    probs = F.softmax(layer.router(h), dim=-1, dtype=torch.float)
    sparse = torch.zeros_like(h)
    for t in range(h.shape[0]):
      weights, experts = torch.topk(probs[t], layer.top_k)
      weights = weights / weights.sum()
      for weight, e in zip(weights, experts):
        x = h[t]
        expert_hidden = (x @ layer.up[e]) * F.silu(x @ layer.gate[e])
        sparse[t] += weight * (expert_hidden @ layer.down[e])
  out = sparse * sparse_scale + shared
  return out.view(batch_size, seq_len, hidden_size)


class _FakeAllReduce:
  """Fake `torch.distributed.all_reduce` (SUM) over identical ranks.

  Simulates `world_size` ranks that each hold the same partial result, so the
  sum-reduction multiplies the tensor in place by `world_size`.
  """

  def __init__(self, world_size: int):
    self.world_size = world_size

  def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> None:
    del args, kwargs  # Unused.
    tensor.mul_(self.world_size)


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

  def test_forward_matches_dense_reference(self):
    layer = ragged_moe.RaggedMoeQwen35(
        config=self.config,
        is_tensor_parallel=False,
        ragged_dot_impl=manual_ragged_dot,
    )
    batch_size = 2
    seq_len = 8
    hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)

    with torch.no_grad():
      output, router_logits = layer(hidden_states)

    test_utils.assert_close(output, dense_moe_reference(layer, hidden_states))
    test_utils.assert_close(
        router_logits,
        layer.router(hidden_states.view(-1, self.config.hidden_size)),
    )

  def test_forward_matches_dense_reference_with_empty_experts(self):
    layer = ragged_moe.RaggedMoeQwen35(
        config=self.config,
        is_tensor_parallel=False,
        ragged_dot_impl=manual_ragged_dot,
    )
    # With positive inputs, experts 0 and 1 get positive logits and experts 2
    # and 3 negative ones, so experts 2 and 3 receive no tokens.
    with torch.no_grad():
      layer.router.weight[:2] = 1.0
      layer.router.weight[2:] = -1.0
    hidden_states = torch.rand(2, 4, self.config.hidden_size) + 0.1

    with torch.no_grad():
      output, _ = layer(hidden_states)

    test_utils.assert_close(output, dense_moe_reference(layer, hidden_states))

  def test_missing_ragged_dot_impl_raises(self):
    layer = ragged_moe.RaggedMoeQwen35(
        config=self.config,
        is_tensor_parallel=False,
        ragged_dot_impl=manual_ragged_dot,
    )
    # Importing torch_tpu registers torch.ops.tpu.ragged_dot, so the
    # constructor never leaves the impl unset here. Clear it to simulate an
    # environment where the op is unavailable.
    # TODO(wan): revisit DECISION (b/568593051): simulate the missing op by
    # clearing the attribute rather than adding a constructor seam.
    layer.ragged_dot_impl = None
    hidden_states = torch.randn(2, 4, self.config.hidden_size)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Test runtime fallback.
        RuntimeError, "ragged_dot is not available"
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

  def test_tensor_parallel_forward_all_reduces_sparse_output(self):
    layer = ragged_moe.RaggedMoeQwen35(
        config=self.config,
        is_tensor_parallel=True,
        ragged_dot_impl=manual_ragged_dot,
    )
    world_size = 2
    hidden_states = torch.randn(2, 4, self.config.hidden_size)

    # Only the routed-expert output is reduced across ranks; the shared expert
    # output must not be scaled.
    with mock.patch(
        "torch.distributed.all_reduce", _FakeAllReduce(world_size)
    ), torch.no_grad():
      output, _ = layer(hidden_states)

    test_utils.assert_close(
        output,
        dense_moe_reference(layer, hidden_states, sparse_scale=world_size),
    )


if __name__ == "__main__":
  absltest.main()
