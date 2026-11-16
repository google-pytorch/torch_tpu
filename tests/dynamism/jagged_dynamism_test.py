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

"""Tests for compiling packed/jagged tensor operations with bounded dynamism.

Bounded dynamism allows jagged tensor workloads with variable token counts
M <= M_max to execute on TPU using a single compiled XLA executable.

Key Architecture Mechanics:
1. Selective Dynamic Marking: Uses torch._dynamo.mark_dynamic(tensor, dim=0,
   min=1, max=M_max) on the flat token dimension.
2. Bounded StableHLO Emission: TorchTPU emits MLIR types with upper bounds
   (#stablehlo.type_extensions<bounds = [M_max, D]>), allowing XLA to allocate
   buffers for M_max while predicating computation up to the runtime size M.
3. Zero-Recompilation Execution: When batch sequence lengths change across
   iterations, the single compiled executable is reused without XLA
   recompilation latency or graph breaks.
"""

from absl.testing import absltest
import torch
from torch._dynamo import testing as dynamo_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class JaggedDynamismTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()

  def _run_and_count_compilations(
      self,
      model: torch.nn.Module,
      token_counts: list[int],
      feature_dim: int,
      max_tokens: int,
  ) -> list[int]:
    """Runs `model` compiled and eagerly on inputs with a dynamic token dim.

    Args:
      model: The model to compile.
      token_counts: The token count M of the input for each call.
      feature_dim: The size of the static feature dimension of the inputs.
      max_tokens: The upper bound of the dynamic token dimension.

    Returns:
      The number of graphs compiled by the TPU backend for each call of the
      compiled model.
    """
    compile_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled_model = torch.compile(model, backend=compile_counter)
    compilations_per_call = []
    for count in token_counts:
      values = torch.randn(count, feature_dim, device="tpu")
      torch._dynamo.mark_dynamic(values, 0, min=1, max=max_tokens)
      eager_out = model(values).cpu()

      compilations_before = compile_counter.frame_count
      compiled_out = compiled_model(values)
      compilations_per_call.append(
          compile_counter.frame_count - compilations_before
      )

      self.assertEqual(compiled_out.shape, eager_out.shape)
      self.assertEqual(compiled_out.shape[0], count)
      utils.assert_close(compiled_out.cpu(), eager_out)
    return compilations_per_call

  def test_linear_layer_bounded_dynamism(self):
    """Verifies Linear execution across variable token counts without recompilation.

    Executes multiple dynamic token counts M in [15, 27, 45] bounded by M_max=64
    on a single compiled executable, verifying only the first call compiles.
    """
    linear = torch.nn.Linear(32, 64, bias=True).to("tpu")

    compilations_per_call = self._run_and_count_compilations(
        linear, token_counts=[15, 27, 45], feature_dim=32, max_tokens=64
    )

    self.assertEqual(compilations_per_call, [1, 0, 0])

  def test_qkv_projection_and_head_reshape_bounded_dynamism(self):
    """Verifies QKV projection and head reshaping with dynamic token dimension M.

    Verifies that view(M, 3, num_heads, head_dim) tracks dynamic leading
    dimension M cleanly through Dynamo and StableHLO lowering with zero
    recompilations.
    """
    d_model, num_heads, head_dim = 64, 4, 16

    class QKVProjection(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(d_model, 3 * num_heads * head_dim)

      def forward(self, x):
        m = x.shape[0]
        # Reshape dynamic tokens [M, 3 * H * D_head] -> [M, 3, H, D_head]
        return self.qkv(x).view(m, 3, num_heads, head_dim)

    model = QKVProjection().to("tpu")

    compilations_per_call = self._run_and_count_compilations(
        model, token_counts=[18, 42, 70], feature_dim=d_model, max_tokens=96
    )

    self.assertEqual(compilations_per_call, [1, 0, 0])

  def test_feed_forward_network_bounded_dynamism(self):
    """Verifies FFN (Linear + ReLU + Linear + Residual) execution across dynamic token counts."""
    d_model, d_ff = 32, 128

    class FeedForwardBlock(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.ffn1 = torch.nn.Linear(d_model, d_ff)
        self.ffn2 = torch.nn.Linear(d_ff, d_model)

      def forward(self, x):
        return x + self.ffn2(torch.nn.functional.relu(self.ffn1(x)))

    model = FeedForwardBlock().to("tpu")

    compilations_per_call = self._run_and_count_compilations(
        model, token_counts=[20, 50, 80], feature_dim=d_model, max_tokens=128
    )

    self.assertEqual(compilations_per_call, [1, 0, 0])


if __name__ == "__main__":
  absltest.main()
