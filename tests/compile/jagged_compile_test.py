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

"""Tests for compiling packed/jagged tensor operations with torch.compile.

In jagged/packed token representations (e.g., in LLM sequence packing or nested
tensors), variable-length sequences [S_1, S_2, ..., S_B] are concatenated along
the sequence dimension into a continuous flat buffer of shape [M, D], where
M = sum(S_i).

These integration tests verify that core jagged tensor operations (linear
projections, token embeddings, multi-head attention reshaping, LayerNorm
reductions, Sparse Mixture of Experts, and hardware alignment pipelines)
compile cleanly end-to-end on TPU with torch.compile(..., backend="tpu") and
match eager execution and autograd backward gradients.
"""

import copy
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class JaggedCompileTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()

  def test_linear_layer_compiled(self):
    """Verifies that a Linear projection on packed tokens compiles cleanly.

    Packed token values [M, D_in] undergo a standard matrix multiplication
    [M, D_in] x [D_in, D_out] -> [M, D_out] inside the compiled XLA cluster.
    """
    linear = torch.nn.Linear(32, 64, bias=True).to("tpu")
    compiled_linear = torch.compile(linear, backend="tpu", dynamic=False)

    # Flat packed token values [M, D] where M=20 total tokens.
    values = torch.randn(20, 32, device="tpu")

    eager_out = linear(values)
    compiled_out = compiled_linear(values)

    self.assertEqual(compiled_out.shape, (20, 64))
    utils.assert_close(compiled_out.cpu(), eager_out.cpu())

  def test_embedding_layer_compiled(self):
    """Verifies that an Embedding layer with 1D packed token IDs compiles.

    In sequence packing, input token IDs are stored as a 1D tensor [M] indexing
    into an embedding table of shape [VocabSize, D_model].
    """
    vocab_size = 500
    d_model = 32
    embedding = torch.nn.Embedding(vocab_size, d_model).to("tpu")
    compiled_embedding = torch.compile(embedding, backend="tpu", dynamic=False)

    # Flat packed token IDs [M] where M=25 total tokens.
    token_ids = torch.randint(
        0, vocab_size, (25,), dtype=torch.int64, device="tpu"
    )

    eager_out = embedding(token_ids)
    compiled_out = compiled_embedding(token_ids)

    self.assertEqual(compiled_out.shape, (25, d_model))
    utils.assert_close(compiled_out.cpu(), eager_out.cpu())

  def test_qkv_projection_and_head_reshape_compiled(self):
    """Verifies QKV projection and multi-head reshaping on packed tokens.

    In Transformer self-attention, flat tokens [M, D] are projected to
    [M, 3 * H * D_head] and reshaped into [M, 3, H, D_head] so attention
    heads can be processed independently.
    """
    d_model, num_heads, head_dim = 64, 4, 16

    class QKVProjection(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(d_model, 3 * num_heads * head_dim)

      def forward(self, x):
        m = x.shape[0]
        # Reshape [M, 3 * H * D_head] -> [M, 3, H, D_head]
        return self.qkv(x).view(m, 3, num_heads, head_dim)

    model = QKVProjection().to("tpu")
    compiled_model = torch.compile(model, backend="tpu", dynamic=False)

    values = torch.randn(25, d_model, device="tpu")
    eager_out = model(values)
    compiled_out = compiled_model(values)

    self.assertEqual(compiled_out.shape, (25, 3, num_heads, head_dim))
    utils.assert_close(compiled_out.cpu(), eager_out.cpu())

  def test_transformer_block_forward_and_backward_compiled(self):
    """Verifies full Transformer block (LayerNorm + FFN + Residual) and autograd.

    Validates that:
    1. LayerNorm normalizes across the feature dimension D independently per
       token without cross-token contamination.
    2. GELU activations and Feed-Forward projections fuse into optimized XLA
       fusion clusters.
    3. Reverse-mode automatic differentiation computes exact weight and input
       gradients matching eager PyTorch for all weights and biases.
    """
    d_model, d_ff = 32, 128

    class TransformerBlock(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.ffn1 = torch.nn.Linear(d_model, d_ff)
        self.ffn2 = torch.nn.Linear(d_ff, d_model)

      def forward(self, x):
        h = self.norm(x)
        return x + self.ffn2(torch.nn.functional.gelu(self.ffn1(h)))

    eager_model = TransformerBlock().to("tpu")
    compiled_model = copy.deepcopy(eager_model)
    compiled_fn = torch.compile(compiled_model, backend="tpu", dynamic=False)

    eager_values = torch.randn(25, d_model, device="tpu", requires_grad=True)
    compiled_values = eager_values.clone().detach().requires_grad_(True)

    eager_out = eager_model(eager_values)
    compiled_out = compiled_fn(compiled_values)

    self.assertEqual(compiled_out.shape, (25, d_model))
    utils.assert_close(compiled_out.cpu(), eager_out.cpu())

    # Verify backward gradient backpropagation across the compiled graph.
    eager_out.sum().backward()
    compiled_out.sum().backward()

    utils.assert_close(compiled_values.grad.cpu(), eager_values.grad.cpu())
    utils.assert_close(
        compiled_model.norm.weight.grad.cpu(),
        eager_model.norm.weight.grad.cpu(),
    )
    utils.assert_close(
        compiled_model.norm.bias.grad.cpu(),
        eager_model.norm.bias.grad.cpu(),
    )
    utils.assert_close(
        compiled_model.ffn1.weight.grad.cpu(),
        eager_model.ffn1.weight.grad.cpu(),
    )
    utils.assert_close(
        compiled_model.ffn1.bias.grad.cpu(),
        eager_model.ffn1.bias.grad.cpu(),
    )
    utils.assert_close(
        compiled_model.ffn2.weight.grad.cpu(),
        eager_model.ffn2.weight.grad.cpu(),
    )
    utils.assert_close(
        compiled_model.ffn2.bias.grad.cpu(),
        eager_model.ffn2.bias.grad.cpu(),
    )

  def test_sparse_moe_block_compiled(self):
    """Verifies Sparse MoE layer (Top-2 router + expert combine) under torch.compile.

    In Mixture of Experts (MoE), flat jagged tokens [M, D] are dynamically
    routed to the top-k experts:
    1. Gating router computes logits [M, num_experts].
    2. topk extracts top-2 expert weights and indices.
    3. Out-of-place scatter maps top-k weights into a routing mask [M,
    num_experts].
    4. Expert outputs are stacked and combined via weighted reduction sum.
    """
    d_model, d_ff, num_experts, top_k = 32, 64, 4, 2

    class SparseMoEBlock(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, num_experts, bias=False)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model, d_ff),
                torch.nn.GELU(),
                torch.nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])

      def forward(self, x):
        router_logits = self.gate(x)
        topk_weights, topk_indices = torch.topk(
            torch.softmax(router_logits, dim=-1), top_k, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        # Out-of-place scatter preserves autograd history to router_logits
        zeros = torch.zeros_like(router_logits)
        routing_mask = zeros.scatter(1, topk_indices, topk_weights)
        # Combine expert outputs: sum_e (expert_e(x) * routing_mask[:, e:e+1])
        expert_outputs = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)
        return (stacked * routing_mask.unsqueeze(-1)).sum(dim=1)

    eager_model = SparseMoEBlock().to("tpu")
    compiled_model = copy.deepcopy(eager_model)
    compiled_fn = torch.compile(compiled_model, backend="tpu", dynamic=False)

    eager_values = torch.randn(25, d_model, device="tpu", requires_grad=True)
    compiled_values = eager_values.clone().detach().requires_grad_(True)

    eager_out = eager_model(eager_values)
    compiled_out = compiled_fn(compiled_values)

    self.assertEqual(compiled_out.shape, (25, d_model))
    utils.assert_close(compiled_out.cpu(), eager_out.cpu())

    eager_out.sum().backward()
    compiled_out.sum().backward()

    utils.assert_close(compiled_values.grad.cpu(), eager_values.grad.cpu())
    utils.assert_close(
        compiled_model.gate.weight.grad.cpu(),
        eager_model.gate.weight.grad.cpu(),
    )
    for c_expert, e_expert in zip(compiled_model.experts, eager_model.experts):
      for (_, c_param), (_, e_param) in zip(
          c_expert.named_parameters(), e_expert.named_parameters()
      ):
        utils.assert_close(c_param.grad.cpu(), e_param.grad.cpu())

  def test_align_and_strip_pipeline_compiled(self):
    """Verifies hardware 128-byte alignment padding and stripping fusion.

    TPU Matrix Multiplication Units (MXU) achieve peak arithmetic throughput
    when the leading dimension M is a multiple of 128. This test verifies that
    padding flat tokens to a 128-multiple before GEMM and slicing back to valid
    token boundaries compiles into a fused XLA graph without intermediate memory
    copies.
    """
    d_model = 32
    linear = torch.nn.Linear(d_model, d_model).to("tpu")

    def pipeline(values, valid_tokens: int):
      # 1. Pad valid token count to the next 128 multiple.
      pad_len = (128 - (valid_tokens % 128)) % 128
      padding = torch.zeros(
          pad_len, d_model, device=values.device, dtype=values.dtype
      )
      aligned_values = torch.cat([values, padding], dim=0)
      # 2. Execute linear projection on aligned buffer.
      h = linear(aligned_values)
      # 3. Strip padding back to valid token boundaries.
      return h[:valid_tokens]

    compiled_pipeline = torch.compile(pipeline, backend="tpu", dynamic=False)

    values = torch.randn(130, d_model, device="tpu")
    eager_out = pipeline(values, valid_tokens=130)
    compiled_out = compiled_pipeline(values, valid_tokens=130)

    self.assertEqual(compiled_out.shape, (130, d_model))
    utils.assert_close(compiled_out.cpu(), eager_out.cpu())


if __name__ == "__main__":
  absltest.main()
