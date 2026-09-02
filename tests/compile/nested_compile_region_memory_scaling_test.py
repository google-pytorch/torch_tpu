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

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn as nn
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


@torch.compiler.nested_compile_region
def mlp_block_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
  """2-layer MLP block executed inside a nested compile region."""
  h = torch.relu(x @ w1 + b1)
  return h @ w2 + b2 + x


class MLPBlock(nn.Module):
  """Single MLP block with residual connection."""

  def __init__(self, dim: int):
    super().__init__()
    self.w1 = nn.Parameter(torch.randn(dim, dim * 2, device="tpu") * 0.1)
    self.b1 = nn.Parameter(torch.zeros(dim * 2, device="tpu"))
    self.w2 = nn.Parameter(torch.randn(dim * 2, dim, device="tpu") * 0.1)
    self.b2 = nn.Parameter(torch.zeros(dim, device="tpu"))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return mlp_block_forward(x, self.w1, self.b1, self.w2, self.b2)


class DeepStackedMLP(nn.Module):
  """Deep model stacking N identical nested MLP blocks."""

  def __init__(self, num_layers: int, dim: int = 16):
    super().__init__()
    self.num_layers = num_layers
    self.blocks = nn.ModuleList([MLPBlock(dim) for _ in range(num_layers)])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for block in self.blocks:
      x = block(x)
    return x


@torch.compiler.nested_compile_region
def transformer_block_forward(
    h: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    o_w: torch.Tensor,
) -> torch.Tensor:
  """Self-attention block executed inside a nested compile region."""
  q = h @ q_w
  k = h @ k_w
  v = h @ v_w
  attn = torch.softmax((q @ k.transpose(-1, -2)) * 0.5, dim=-1)
  attn_out = attn @ v @ o_w
  return h + attn_out


class DeepStackedTransformer(nn.Module):
  """Deep model stacking N transformer attention layers."""

  def __init__(self, num_layers: int, dim: int = 16):
    super().__init__()
    self.num_layers = num_layers
    self.q_weights = nn.ParameterList([
        nn.Parameter(torch.randn(dim, dim, device="tpu") * 0.1)
        for _ in range(num_layers)
    ])
    self.k_weights = nn.ParameterList([
        nn.Parameter(torch.randn(dim, dim, device="tpu") * 0.1)
        for _ in range(num_layers)
    ])
    self.v_weights = nn.ParameterList([
        nn.Parameter(torch.randn(dim, dim, device="tpu") * 0.1)
        for _ in range(num_layers)
    ])
    self.out_weights = nn.ParameterList([
        nn.Parameter(torch.randn(dim, dim, device="tpu") * 0.1)
        for _ in range(num_layers)
    ])

  def forward(self, h: torch.Tensor) -> torch.Tensor:
    for q_w, k_w, v_w, o_w in zip(
        self.q_weights, self.k_weights, self.v_weights, self.out_weights
    ):
      h = transformer_block_forward(h, q_w, k_w, v_w, o_w)
    return h


class NestedCompileRegionMemoryScalingTest(seed_test_utils.RepeatableTest):
  """Tests verifying O(1) MLIR instruction and host memory scaling for deep models."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)

  @parameterized.parameters(4, 8, 16, 32, 64)
  def test_mlp_depth_scaling_mlir_size_constant(self, num_layers: int):
    """Verifies that loop rolling bounds MLIR text size to O(1) regardless of depth N."""
    dim = 16
    model = DeepStackedMLP(num_layers=num_layers, dim=dim)
    x_tpu = torch.randn(4, dim, device="tpu")

    tpu_backend = _backend.TpuBackend(debug=True)
    compiled_model = torch.compile(model, backend=tpu_backend)

    with torch.no_grad(), torch._dynamo.config.patch(
        inline_single_use_invoke_subgraph=False
    ):
      out = compiled_model(x_tpu).to("cpu")

    # 1. Single compiled binary
    self.assertLen(tpu_backend._compiled_executables, 1)
    mlir_text = tpu_backend._compiled_executables[0].mlir_texts[0]

    # 2. Verify stablehlo.while loop exists
    self.assertIn("stablehlo.while", mlir_text)

    # 3. Verify MLIR text size is bounded O(1):
    # A rolled while-loop module is ~200-300 lines regardless of whether N=4 or N=64.
    mlir_line_count = len(mlir_text.splitlines())
    self.assertLess(
        mlir_line_count,
        350,
        f"Expected MLIR line count for N={num_layers} to be bounded (<350"
        f" lines), got {mlir_line_count}",
    )

    # 4. Check CPU eager reference correctness with tolerance appropriate for up to 64 compounding layers
    x_cpu = x_tpu.cpu()
    with torch.no_grad():
      ref = x_cpu
      for block in model.blocks:
        w1_c, b1_c = block.w1.cpu(), block.b1.cpu()
        w2_c, b2_c = block.w2.cpu(), block.b2.cpu()
        h_c = torch.relu(ref @ w1_c + b1_c)
        ref = h_c @ w2_c + b2_c + ref

    utils.assert_close(out, ref, rtol=2e-2, atol=3e-2)

  @parameterized.parameters(4, 8, 16, 32)
  def test_transformer_depth_scaling(self, num_layers: int):
    """Verifies loop rolling for deep transformer attention stacks."""
    dim = 16
    model = DeepStackedTransformer(num_layers=num_layers, dim=dim)
    h_tpu = torch.randn(4, dim, device="tpu") * 0.5

    tpu_backend = _backend.TpuBackend(debug=True)
    compiled_model = torch.compile(model, backend=tpu_backend)

    with torch.no_grad(), torch._dynamo.config.patch(
        inline_single_use_invoke_subgraph=False
    ):
      out = compiled_model(h_tpu).to("cpu")

    self.assertLen(tpu_backend._compiled_executables, 1)
    mlir_text = tpu_backend._compiled_executables[0].mlir_texts[0]
    self.assertIn("stablehlo.while", mlir_text)

    # Check CPU eager reference
    h_cpu = h_tpu.cpu()
    with torch.no_grad():
      for q_w, k_w, v_w, o_w in zip(
          model.q_weights, model.k_weights, model.v_weights, model.out_weights
      ):
        q = h_cpu @ q_w.cpu()
        k = h_cpu @ k_w.cpu()
        v = h_cpu @ v_w.cpu()
        attn = torch.softmax((q @ k.transpose(-1, -2)) * 0.5, dim=-1)
        attn_out = attn @ v @ o_w.cpu()
        h_cpu = h_cpu + attn_out

    utils.assert_close(out, h_cpu, rtol=5e-3, atol=5e-3)

  def test_unrolled_vs_loop_rolled_mlir_complexity(self):
    """Compares unrolled vs loop-rolled MLIR complexity for N=16 layers."""
    num_layers = 16
    dim = 16

    # 1. Plain unrolled model without nested compile region
    class PlainMLPBlock(nn.Module):

      def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim * 2, device="tpu") * 0.1)
        self.b1 = nn.Parameter(torch.zeros(dim * 2, device="tpu"))
        self.w2 = nn.Parameter(torch.randn(dim * 2, dim, device="tpu") * 0.1)
        self.b2 = nn.Parameter(torch.zeros(dim, device="tpu"))

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2 + x

    class PlainDeepMLP(nn.Module):

      def __init__(self, num_layers: int, dim: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PlainMLPBlock(dim) for _ in range(num_layers)]
        )

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
          x = block(x)
        return x

    model_unrolled = PlainDeepMLP(num_layers=num_layers, dim=dim)
    x_tpu = torch.randn(4, dim, device="tpu")
    backend_unrolled = _backend.TpuBackend(debug=True)
    compiled_unrolled = torch.compile(model_unrolled, backend=backend_unrolled)
    with torch.no_grad():
      out_unrolled = compiled_unrolled(x_tpu).to("cpu")
    self.assertIsNotNone(out_unrolled)
    mlir_unrolled = backend_unrolled._compiled_executables[0].mlir_texts[0]

    # 2. Loop-rolled model
    model_rolled = DeepStackedMLP(num_layers=num_layers, dim=dim)
    backend_rolled = _backend.TpuBackend(debug=True)
    compiled_rolled = torch.compile(model_rolled, backend=backend_rolled)
    with torch.no_grad(), torch._dynamo.config.patch(
        inline_single_use_invoke_subgraph=False
    ):
      out_rolled = compiled_rolled(x_tpu).to("cpu")
    self.assertIsNotNone(out_rolled)
    mlir_rolled = backend_rolled._compiled_executables[0].mlir_texts[0]

    self.assertNotIn("stablehlo.while", mlir_unrolled)
    self.assertIn("stablehlo.while", mlir_rolled)

    # In unrolled model, dot_general operations scale linearly with num_layers (2 * 16 = 32 dot ops).
    # In loop-rolled model, there are exactly 2 dot_general operations inside the while loop body!
    unrolled_dots = mlir_unrolled.count("stablehlo.dot_general")
    rolled_dots = mlir_rolled.count("stablehlo.dot_general")
    self.assertEqual(
        rolled_dots,
        2,
        f"Expected exactly 2 dot ops in while loop, got {rolled_dots}",
    )
    self.assertEqual(
        unrolled_dots,
        num_layers * 2,
        f"Expected {num_layers * 2} dot ops in unrolled graph, got"
        f" {unrolled_dots}",
    )


if __name__ == "__main__":
  absltest.main()
