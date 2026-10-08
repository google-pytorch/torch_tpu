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

from unittest import mock
from absl.testing import absltest
import torch
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import test_utils as utils
from tests import oss_utils
from tests import seed_test_utils


@oss_utils.skip_if_libtpu_version(
    lambda v: v < (0, 0, 47),
    "Bounded dynamism requires libtpu >= 0.0.47",
)
class JaggedDynamismTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()

  def test_linear_layer_bounded_dynamism(self):
    """Verifies Linear execution across variable token counts without recompilation.

    Executes multiple dynamic token counts M in [15, 27, 45] bounded by M_max=64
    on a single compiled executable, verifying only 1 compilation event occurs.
    """
    linear = torch.nn.Linear(32, 64, bias=True).to("tpu")

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled_linear = torch.compile(linear, backend="tpu")

      # Varying batch token counts M <= 64 across iterations.
      for count in [15, 27, 45]:
        values = torch.randn(count, 32, device="tpu")
        torch._dynamo.mark_dynamic(values, 0, min=1, max=64)
        eager_out = linear(values)
        compiled_out = compiled_linear(values)
        self.assertEqual(compiled_out.shape, (count, 64))
        utils.assert_close(compiled_out.cpu(), eager_out.cpu())

      # Verify the model module compiled only once across all dynamic inputs.
      self.assertEqual(mock_dc.call_count, 1)

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

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled_model = torch.compile(model, backend="tpu")

      for count in [18, 42, 70]:
        values = torch.randn(count, d_model, device="tpu")
        torch._dynamo.mark_dynamic(values, 0, min=1, max=96)
        eager_out = model(values)
        compiled_out = compiled_model(values)
        self.assertEqual(compiled_out.shape, (count, 3, num_heads, head_dim))
        utils.assert_close(compiled_out.cpu(), eager_out.cpu())

      # Verify single model compilation event across dynamic token counts.
      self.assertEqual(mock_dc.call_count, 1)

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

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled_model = torch.compile(model, backend="tpu")

      for count in [20, 50, 80]:
        values = torch.randn(count, d_model, device="tpu")
        torch._dynamo.mark_dynamic(values, 0, min=1, max=128)

        eager_out = model(values)
        compiled_out = compiled_model(values)

        self.assertEqual(compiled_out.shape, (count, d_model))
        utils.assert_close(compiled_out.cpu(), eager_out.cpu())

      # Verify single model compilation event across dynamic token counts.
      self.assertEqual(mock_dc.call_count, 1)


if __name__ == "__main__":
  absltest.main()
