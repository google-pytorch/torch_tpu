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

"""Baseline for TorchTPU context manager graph breaks under `torch.compile`.

Unlike PyTorch-native context managers, TorchDynamo cannot symbolically trace
the opaque pybind11 calls used by TorchTPU context managers. Consequently, when
`torch.compile` encounters such a context, it falls back to eager execution.

The test target sets `TORCH_LOGS=graph_breaks`, so every invocation's test log
carries per-graph-break diagnostics from PyTorch. The diagnostics info is meant
for human inspection only.
"""

from absl.testing import absltest
import torch
from torch_tpu._internal import precision
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


def _torch_tpu_ctx_matmul(
    mat1: torch.Tensor, mat2: torch.Tensor
) -> torch.Tensor:
  """Matmul within a TorchTPU context manager."""
  with precision.precision(precision.Precision.HIGHEST):
    return mat1 @ mat2


def _torch_native_ctx_matmul(
    mat1: torch.Tensor, mat2: torch.Tensor
) -> torch.Tensor:
  """Matmul within a PyTorch-native context manager."""
  with torch.no_grad():
    return mat1 @ mat2


class ContextManagerGraphBreakTest(seed_test_utils.RepeatableTest):
  """Tests for TorchTPU context manager graph breaks under `torch.compile`."""

  def setUp(self):
    super().setUp()
    # Drop cached compilations so each test observes a fresh trace.
    torch._dynamo.reset()

  def test_torch_tpu_ctx_causes_graph_breaks(self):
    """Test that TorchDynamo breaks graphs on a TorchTPU context manager."""
    explanation = torch._dynamo.explain(_torch_tpu_ctx_matmul)(
        torch.randn(2, 2), torch.randn(2, 2)
    )
    # TorchDynamo cannot symbolically execute the opaque pybind11 calls inside
    # TorchTPU context managers. Consequently, it abandons the whole frame and
    # runs it eagerly: zero graphs captured.
    self.assertEqual(explanation.graph_count, 0)

  def test_torch_tpu_ctx_trace_fullgraph_raises(self):
    """Test that a TorchTPU context manager raises tracing exceptions."""
    # With `fullgraph=True`, a graph break becomes a hard compile error instead
    # of a silent eager fallback.
    compiled = torch.compile(
        _torch_tpu_ctx_matmul, backend="eager", fullgraph=True
    )
    with self.assertRaises(torch._dynamo.exc.Unsupported):
      compiled(torch.randn(2, 2), torch.randn(2, 2))

  def test_torch_native_ctx_compiles_without_graph_breaks(self):
    """Test that a PyTorch-native context manager compiles cleanly."""
    compiled = torch.compile(
        _torch_native_ctx_matmul, backend="eager", fullgraph=True
    )
    mat1, mat2 = torch.ones(2, 2), torch.ones(2, 2)
    utils.assert_close(compiled(mat1, mat2), mat1 @ mat2)

    explanation = torch._dynamo.explain(_torch_native_ctx_matmul)(
        torch.randn(2, 2), torch.randn(2, 2)
    )
    self.assertEqual(explanation.graph_count, 1)
    self.assertEqual(explanation.graph_break_count, 0)


if __name__ == "__main__":
  absltest.main()
