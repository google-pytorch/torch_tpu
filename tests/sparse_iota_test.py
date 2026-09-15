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
import numpy as np
import torch
from tests import op_testing

INT_MAX = 2**31 - 1
PAD_WIDTH = 8


def compute_expected_sparse_iota(
    row_sizes: list[int],
    pad_size: int,
    pad_value: int,
) -> list[int]:
  """Computes the expected sparse_iota output on CPU."""
  expected_output = []
  offset = 0
  for size in row_sizes:
    cols = list(range(offset, offset + size))
    padded_size = ((size + pad_size - 1) // pad_size) * pad_size
    num_pads = max(0, padded_size - size)
    pads = [pad_value] * num_pads
    expected_output.extend(cols)
    expected_output.extend(pads)
    offset += size

  return expected_output


class SparseIotaTest(op_testing.TorchTpuTestBase):
  """Tests for sparse_iota op on TPU SparseCore."""

  def _get_inputs(self, device: torch.device):
    row_sizes = [7, 14, 20, 0, 24, 0, 13, 0]
    pad_size = PAD_WIDTH
    pad_value = INT_MAX
    max_non_zeroes_per_row = 256
    max_non_zeroes = 2048

    padded_row_sizes = [
        ((size + pad_size - 1) // pad_size) * pad_size for size in row_sizes
    ]
    cumsum_padded = [0] + list(np.cumsum(padded_row_sizes)[:-1])
    row_pointers = [cumsum_padded[i] + size for i, size in enumerate(row_sizes)]
    row_pointers_tensor = torch.tensor(
        row_pointers, dtype=torch.int32, device=device
    )

    expected = torch.tensor(
        compute_expected_sparse_iota(row_sizes, pad_size, pad_value),
        dtype=torch.int32,
    )

    return (
        row_pointers_tensor,
        max_non_zeroes,
        max_non_zeroes_per_row,
        pad_value,
        pad_size,
        expected,
    )

  @parameterized.parameters(False, True)
  def test_sparse_iota_on_tpu(self, compile_op: bool):
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    (
        row_pointers,
        max_non_zeroes,
        max_non_zeroes_per_row,
        pad_value,
        pad_size,
        expected,
    ) = self._get_inputs(device)

    def iota_fn(rp):
      return torch.ops.tpu.sparse_iota(
          rp,
          max_non_zeroes=max_non_zeroes,
          max_non_zeroes_per_row=max_non_zeroes_per_row,
      )

    if compile_op:
      iota_fn = torch.compile(iota_fn, fullgraph=True)

    out = iota_fn(row_pointers)

    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu()[: expected.shape[0]],
    )


if __name__ == "__main__":
  absltest.main()
