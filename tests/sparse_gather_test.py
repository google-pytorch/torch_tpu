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

import math
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from tests import op_testing
from tests import seed_test_utils

PAD_WIDTH = 8
INT_MAX = 2**31 - 1

_SPARSECORE_SUPPORTED: bool | None = None


def is_sparsecore_supported() -> bool:
  """Returns True if the current device environment supports SparseCore."""
  global _SPARSECORE_SUPPORTED
  if _SPARSECORE_SUPPORTED is not None:
    return _SPARSECORE_SUPPORTED

  if not torch.accelerator.is_available():
    _SPARSECORE_SUPPORTED = False
    return False

  try:
    from torch_tpu._internal.device import _device_ops_backend

    _ = torch.zeros(1, device="tpu")
    attrs = _device_ops_backend._get_local_device_attributes()
    kind = attrs.get("device_kind", "")
    if not kind.startswith("TPU v7"):
      _SPARSECORE_SUPPORTED = False
      return False
    _SPARSECORE_SUPPORTED = True
    return True
  except Exception:
    pass

  # Fallback: probe sparse_gather op directly on a minimal input.
  try:
    rp = torch.tensor([0, 8], dtype=torch.int32, device="tpu")
    idx = torch.full((16,), INT_MAX, dtype=torch.int32, device="tpu")
    op = torch.zeros((1, 8), dtype=torch.float32, device="tpu")
    torch.ops.tpu.sparse_gather(rp, idx, op, max_non_zeroes_per_row=8)
    _SPARSECORE_SUPPORTED = True
    return True
  except RuntimeError as e:
    if "SparseCore support" in str(e):
      _SPARSECORE_SUPPORTED = False
      return False
    _SPARSECORE_SUPPORTED = False
    return False
  except Exception:
    _SPARSECORE_SUPPORTED = False
    return False


def pad_length(arr: list[int]) -> list[int]:
  """Pads the array to the nearest multiple of PAD_WIDTH."""
  pad_len = (PAD_WIDTH - len(arr)) % PAD_WIDTH
  assert 0 <= pad_len < PAD_WIDTH
  assert (len(arr) + pad_len) % PAD_WIDTH == 0
  return arr + [INT_MAX] * pad_len


def round_up(x: int, y: int) -> int:
  """Rounds up x to the nearest multiple of y."""
  return math.ceil(x / y) * y


def build_csr(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
  """Builds CSR row_pointers and indices from a list of lists (COO format)."""
  row_ptrs_list = []
  indices_list = []
  curr_row_end = 0
  for row in rows:
    curr_row_end += len(row)
    padded_row = pad_length(row)
    indices_list.extend(padded_row)
    row_ptrs_list.append(curr_row_end)
    curr_row_end = round_up(curr_row_end, PAD_WIDTH)

  return (
      torch.tensor(row_ptrs_list, dtype=torch.int32),
      torch.tensor(indices_list, dtype=torch.int32),
  )


class SparseGatherMetaTest(seed_test_utils.RepeatableTest):
  """Tests for sparse_gather and sparse_gather_backward shape and dtype inference on meta device.

  These tests verify meta tensor registration for abstract interpretation and
  torch.compile tracing, running on CPU without requiring an accelerator.
  """

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_sparse_gather_meta(self, dtype):
    """Verifies sparse_gather shape and dtype inference on meta device."""
    num_rows = 4
    max_non_zeroes = 8
    total_indices = num_rows * max_non_zeroes
    table_depth = 128
    table_width = 16

    row_pointers = torch.empty(num_rows, dtype=torch.int32, device="meta")
    indices = torch.empty(total_indices, dtype=torch.int32, device="meta")
    operand = torch.empty(table_depth, table_width, dtype=dtype, device="meta")

    out = torch.ops.tpu.sparse_gather(
        row_pointers, indices, operand, max_non_zeroes_per_row=max_non_zeroes
    )
    self.assertTrue(out.is_meta)
    self.assertEqual(out.shape, torch.Size([total_indices, table_width]))
    self.assertEqual(out.dtype, dtype)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_sparse_gather_backward_meta(self, dtype):
    """Verifies sparse_gather_backward shape and dtype inference on meta device."""
    num_indices = 64
    table_depth = 128
    table_width = 32

    grad_output = torch.empty(
        num_indices, table_width, dtype=dtype, device="meta"
    )
    indices = torch.empty(num_indices, dtype=torch.int32, device="meta")
    grad_operand = torch.empty(
        table_depth, table_width, dtype=dtype, device="meta"
    )

    out = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, grad_operand
    )
    self.assertTrue(out.is_meta)
    self.assertEqual(out.shape, torch.Size([table_depth, table_width]))
    self.assertEqual(out.dtype, dtype)


class SparseGatherTest(op_testing.TorchTpuTestBase):
  """Tests for sparse_gather op on TPU SparseCore and TensorCore."""

  def setUp(self) -> None:
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    torch._dynamo.reset()

  def _skip_if_sparsecore_unsupported(self) -> None:
    if not is_sparsecore_supported():
      self.skipTest("SparseCore is not supported on this TPU device.")

  def _get_inputs(
      self,
      device: torch.device,
      table_depth: int = 1024,
      table_width: int = 16,
      num_rows: int = 16,
      max_non_zeroes_per_row: int = 64,
      dtype: torch.dtype = torch.float32,
  ):
    torch.manual_seed(42)
    np.random.seed(42)

    operand = (
        torch.arange(0, table_depth, dtype=dtype)
        .unsqueeze(1)
        .repeat(1, table_width)
    )

    coos = []
    for _ in range(num_rows):
      row_size = np.random.randint(1, max_non_zeroes_per_row // 2 + 1)
      row_indices = np.random.randint(0, table_depth, size=row_size).tolist()
      coos.append(row_indices)

    row_pointers, indices = build_csr(coos)
    pad_needed = num_rows * max_non_zeroes_per_row - len(indices)
    if pad_needed > 0:
      indices = torch.cat(
          [indices, torch.full((pad_needed,), INT_MAX, dtype=torch.int32)]
      )

    return (
        row_pointers.to(device),
        indices.to(device),
        operand.to(device),
        coos,
        max_non_zeroes_per_row,
    )

  @parameterized.parameters(False, True)
  def test_sparse_gather_on_tpu(self, compile_op):
    """Verifies forward sparse_gather execution on TPU against CPU indexing.

    Checks:
      - Validates that torch.ops.tpu.sparse_gather correctly reads embedding
        rows from `operand` based on CSR `row_pointers` and `indices`.
      - Tests both eager and torch.compile(fullgraph=True) modes.

    Setup:
      - `operand`: 2D embedding table of shape [1024, 16] initialized with row
        values [0, 1, 2, ..., 1023].
      - `indices`: CSR indices containing valid embedding indices interspersed
        with INT_MAX padding tokens up to `num_rows * max_non_zeroes_per_row`.

    Expected Output:
      - For each logical non-zero element in `coos`, the output slice must match
        `operand.cpu()[torch.tensor(all_indices)]`.
    """
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    row_pointers, indices, operand, coos, max_non_zeroes_per_row = (
        self._get_inputs(device)
    )

    def gather_fn(rp, idx, op):
      return torch.ops.tpu.sparse_gather(
          rp, idx, op, max_non_zeroes_per_row=max_non_zeroes_per_row
      )

    if compile_op:
      gather_fn = torch.compile(gather_fn, fullgraph=True)

    out = gather_fn(row_pointers, indices, operand)

    all_indices = []
    for row in coos:
      all_indices.extend(row)
    expected = operand.cpu()[torch.tensor(all_indices, dtype=torch.long)]

    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu()[: expected.shape[0]],
    )

  @parameterized.parameters(False, True)
  def test_sparse_gather_minimal_row_size_forward(self, compile_op):
    """Verifies forward sparse_gather with minimal non-zero row size (1 element per row).

    Checks:
      - Multi-row CSR table where every row has exactly 1 non-zero element.
      - Tests that sparse_gather properly reads single-element rows followed by
        padding tokens up to max_non_zeroes_per_row.
      - Verifies both eager and torch.compile modes.
    """
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    num_rows = 16
    max_non_zeroes = 8
    table_depth = 64
    table_width = 8

    operand = (
        torch.arange(0, table_depth, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, table_width)
        .to(device)
    )
    # 1 element per row, padded with 7 INT_MAX tokens to meet 8-alignment
    row_ptrs = [i * max_non_zeroes + 1 for i in range(num_rows)]
    indices_list = []
    for i in range(num_rows):
      indices_list.extend([i] + [INT_MAX] * (max_non_zeroes - 1))

    row_pointers = torch.tensor(row_ptrs, dtype=torch.int32, device=device)
    indices = torch.tensor(indices_list, dtype=torch.int32, device=device)

    def gather_fn(rp, idx, op):
      return torch.ops.tpu.sparse_gather(
          rp, idx, op, max_non_zeroes_per_row=max_non_zeroes
      )

    if compile_op:
      gather_fn = torch.compile(gather_fn, fullgraph=True)

    out = gather_fn(row_pointers, indices, operand)
    expected = operand.cpu()[torch.tensor(range(num_rows), dtype=torch.long)]

    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu()[:num_rows],
    )

  @parameterized.parameters(False, True)
  def test_sparse_gather_full_rows_no_padding(self, compile_op):
    """Verifies forward sparse_gather when rows are 100% dense with zero padding tokens.

    Checks:
      - 16 rows where every slot in max_non_zeroes_per_row contains a valid
      index.
      - Indices tensor contains no INT_MAX elements.
      - Verifies both eager and torch.compile modes.
    """
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    num_rows = 16
    max_non_zeroes = 8
    table_depth = 256
    table_width = 8
    total_elements = num_rows * max_non_zeroes

    operand = (
        torch.arange(0, table_depth, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, table_width)
        .to(device)
    )
    row_ptrs = [(i + 1) * max_non_zeroes for i in range(num_rows)]
    indices_list = list(range(total_elements))

    row_pointers = torch.tensor(row_ptrs, dtype=torch.int32, device=device)
    indices = torch.tensor(indices_list, dtype=torch.int32, device=device)

    def gather_fn(rp, idx, op):
      return torch.ops.tpu.sparse_gather(
          rp, idx, op, max_non_zeroes_per_row=max_non_zeroes
      )

    if compile_op:
      gather_fn = torch.compile(gather_fn, fullgraph=True)

    out = gather_fn(row_pointers, indices, operand)
    expected = operand.cpu()[:total_elements]

    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu(),
    )

  @parameterized.parameters(False, True)
  def test_sparse_gather_repeated_indices_forward(self, compile_op):
    """Verifies forward sparse_gather gathering the same embedding index multiple times."""
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    num_rows = 16
    max_non_zeroes = 8
    table_depth = 32
    table_width = 8

    operand = (
        torch.arange(0, table_depth, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, table_width)
        .to(device)
    )
    row_ptrs = [i * max_non_zeroes + 4 for i in range(num_rows)]
    indices_list = []
    for i in range(num_rows):
      idx_val = i % table_depth
      indices_list.extend([idx_val] * 4 + [INT_MAX] * 4)

    row_pointers = torch.tensor(row_ptrs, dtype=torch.int32, device=device)
    indices = torch.tensor(indices_list, dtype=torch.int32, device=device)

    def gather_fn(rp, idx, op):
      return torch.ops.tpu.sparse_gather(
          rp, idx, op, max_non_zeroes_per_row=max_non_zeroes
      )

    if compile_op:
      gather_fn = torch.compile(gather_fn, fullgraph=True)

    out = gather_fn(row_pointers, indices, operand)

    all_indices = []
    for i in range(num_rows):
      idx_val = i % table_depth
      all_indices.extend([idx_val] * 4)
    expected = operand.cpu()[torch.tensor(all_indices, dtype=torch.long)]

    self.assert_close(
        golden_result=expected,
        torch_tpu_result=out.cpu()[: expected.shape[0]],
    )

  @parameterized.product(
      compile_op=(False, True),
      dtype=(torch.float32, torch.bfloat16),
  )
  def test_sparse_gather_autograd_on_tpu(self, compile_op, dtype):
    """Verifies end-to-end autograd backward pass for sparse_gather on TPU.

    Checks:
      - Full VJP backward pass (`out.backward(grad_output)`) produces correct
        gradients for the embedding table (`operand.grad`).
      - Tests both eager and torch.compile(fullgraph=True) modes.
      - Tests both torch.float32 and torch.bfloat16 precision.
    """
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    row_pointers, indices, operand, _, max_non_zeroes_per_row = (
        self._get_inputs(device, dtype=dtype)
    )
    operand = operand.clone().detach().requires_grad_(True)

    def gather_fn(rp, idx, op):
      return torch.ops.tpu.sparse_gather(
          rp, idx, op, max_non_zeroes_per_row=max_non_zeroes_per_row
      )

    if compile_op:
      gather_fn = torch.compile(gather_fn, fullgraph=True)

    out = gather_fn(row_pointers, indices, operand)
    grad_output = torch.randn_like(out) * 0.1

    out.backward(grad_output)
    self.assertIsNotNone(operand.grad)

    grad_output_cpu = grad_output.cpu()
    indices_cpu = indices.cpu()
    golden_grad = torch.zeros(
        operand.shape, dtype=dtype, device=torch.device("cpu")
    )
    valid_mask = (indices_cpu >= 0) & (indices_cpu < operand.shape[0])
    valid_indices = indices_cpu[valid_mask].long()
    valid_grad_output = grad_output_cpu[valid_mask]
    golden_grad.index_add_(0, valid_indices, valid_grad_output)

    kwargs = {}
    if dtype == torch.bfloat16:
      kwargs = {"rtol": 5e-2, "atol": 5e-2}

    self.assert_close(
        golden_result=golden_grad,
        torch_tpu_result=operand.grad.cpu(),
        **kwargs,
    )

  def test_sparse_gather_autograd_multiple_backward(self):
    """Verifies gradient accumulation over multiple backward passes with retain_graph=True."""
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    row_pointers, indices, operand, _, max_non_zeroes_per_row = (
        self._get_inputs(device, dtype=torch.float32)
    )
    operand = operand.clone().detach().requires_grad_(True)

    out = torch.ops.tpu.sparse_gather(
        row_pointers,
        indices,
        operand,
        max_non_zeroes_per_row=max_non_zeroes_per_row,
    )

    grad1 = torch.randn_like(out) * 0.1
    grad2 = torch.randn_like(out) * 0.1

    out.backward(grad1, retain_graph=True)
    out.backward(grad2)

    indices_cpu = indices.cpu()
    valid_mask = (indices_cpu >= 0) & (indices_cpu < operand.shape[0])
    valid_indices = indices_cpu[valid_mask].long()

    golden_grad = torch.zeros_like(operand.cpu())
    golden_grad.index_add_(0, valid_indices, grad1.cpu()[valid_mask])
    golden_grad.index_add_(0, valid_indices, grad2.cpu()[valid_mask])

    self.assert_close(
        golden_result=golden_grad,
        torch_tpu_result=operand.grad.cpu(),
    )

  def test_sparse_gather_autograd_requires_grad_false(self):
    """Verifies that autograd is bypassed when operand does not require grad."""
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    row_pointers, indices, operand, _, max_non_zeroes_per_row = (
        self._get_inputs(device)
    )
    operand = operand.clone().detach().requires_grad_(False)
    out = torch.ops.tpu.sparse_gather(
        row_pointers,
        indices,
        operand,
        max_non_zeroes_per_row=max_non_zeroes_per_row,
    )
    self.assertFalse(out.requires_grad)

  @parameterized.product(
      compile_op=(False, True),
      dtype=(torch.float32, torch.bfloat16),
  )
  def test_sparse_gather_backward_op_direct(self, compile_op, dtype):
    """Verifies direct invocation of torch.ops.tpu.sparse_gather_backward op."""
    device = torch.device("tpu")
    table_depth = 512
    table_width = 32
    num_indices = 256

    torch.manual_seed(42)
    indices_list = torch.randint(0, table_depth, (num_indices - 32,)).tolist()
    indices_list.extend([INT_MAX] * 32)
    indices = torch.tensor(indices_list, dtype=torch.int32, device=device)
    grad_output = (
        torch.randn(num_indices, table_width, dtype=dtype, device=device) * 0.1
    )
    grad_operand = torch.zeros(
        table_depth, table_width, dtype=dtype, device=device
    )

    def bwd_fn(go, idx, g_op):
      return torch.ops.tpu.sparse_gather_backward(go, idx, g_op)

    if compile_op:
      bwd_fn = torch.compile(bwd_fn, fullgraph=True)

    out_grad = bwd_fn(grad_output, indices, grad_operand)

    golden_grad = torch.zeros(table_depth, table_width, dtype=dtype)
    indices_cpu = indices.cpu()
    valid_mask = (indices_cpu >= 0) & (indices_cpu < table_depth)
    golden_grad.index_add_(
        0, indices_cpu[valid_mask].long(), grad_output.cpu()[valid_mask]
    )

    kwargs = {}
    if dtype == torch.bfloat16:
      kwargs = {"rtol": 5e-2, "atol": 5e-2}

    self.assert_close(
        golden_result=golden_grad,
        torch_tpu_result=out_grad.cpu(),
        **kwargs,
    )

  @parameterized.parameters(1, 8, 64)
  def test_sparse_gather_backward_various_widths(self, table_width):
    """Verifies backward scatter accumulation across varying table widths (e.g.

    1, 8, 64).
    """
    device = torch.device("tpu")
    table_depth = 64
    num_indices = 32

    torch.manual_seed(42)
    indices_list = torch.randint(0, table_depth, (num_indices - 8,)).tolist()
    indices_list.extend([INT_MAX] * 8)
    indices = torch.tensor(indices_list, dtype=torch.int32, device=device)
    grad_output = torch.randn(
        num_indices, table_width, dtype=torch.float32, device=device
    )
    grad_operand = torch.zeros(
        table_depth, table_width, dtype=torch.float32, device=device
    )

    out_grad = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, grad_operand
    )

    golden_grad = torch.zeros(table_depth, table_width, dtype=torch.float32)
    indices_cpu = indices.cpu()
    valid_mask = (indices_cpu >= 0) & (indices_cpu < table_depth)
    golden_grad.index_add_(
        0, indices_cpu[valid_mask].long(), grad_output.cpu()[valid_mask]
    )

    self.assert_close(
        golden_result=golden_grad,
        torch_tpu_result=out_grad.cpu(),
    )

  def test_sparse_gather_backward_accumulate_on_existing_grad_operand(self):
    """Verifies that backward accumulates on top of a pre-existing non-zero grad_operand buffer."""
    device = torch.device("tpu")
    table_depth = 16
    table_width = 8

    indices = torch.tensor([2, 5, 2, INT_MAX], dtype=torch.int32, device=device)
    grad_output = torch.ones(
        indices.shape[0], table_width, dtype=torch.float32, device=device
    )
    initial_grad_operand = torch.full(
        (table_depth, table_width), 3.0, dtype=torch.float32, device=device
    )

    out_grad = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, initial_grad_operand
    )
    out_cpu = out_grad.cpu()

    # Index 2 was gathered twice with grad 1.0 -> 3.0 + 2.0 = 5.0
    self.assert_close(
        golden_result=torch.full((table_width,), 5.0),
        torch_tpu_result=out_cpu[2],
    )
    # Index 5 was gathered once with grad 1.0 -> 3.0 + 1.0 = 4.0
    self.assert_close(
        golden_result=torch.full((table_width,), 4.0),
        torch_tpu_result=out_cpu[5],
    )
    # Ungathered rows retain initial value 3.0
    self.assert_close(
        golden_result=torch.full((table_width,), 3.0),
        torch_tpu_result=out_cpu[0],
    )

  def test_sparse_gather_backward_boundary_indices(self):
    """Verifies boundary indices (0 and table_depth - 1) accumulate correctly without dropping."""
    device = torch.device("tpu")
    table_depth = 16
    table_width = 8

    indices = torch.tensor(
        [0, table_depth - 1, 0, table_depth - 1, INT_MAX],
        dtype=torch.int32,
        device=device,
    )
    grad_output = torch.ones(
        indices.shape[0], table_width, dtype=torch.float32, device=device
    )
    grad_operand = torch.zeros(
        table_depth, table_width, dtype=torch.float32, device=device
    )

    out_grad = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, grad_operand
    )
    out_cpu = out_grad.cpu()

    # Boundary 0 gathered twice
    self.assert_close(
        golden_result=torch.full((table_width,), 2.0),
        torch_tpu_result=out_cpu[0],
    )
    # Boundary table_depth - 1 gathered twice
    self.assert_close(
        golden_result=torch.full((table_width,), 2.0),
        torch_tpu_result=out_cpu[table_depth - 1],
    )
    # Middle rows remain 0.0
    self.assert_close(
        golden_result=torch.zeros(table_width),
        torch_tpu_result=out_cpu[1],
    )

  def test_sparse_gather_backward_zero_grad_output(self):
    """Verifies that an all-zero grad_output produces an all-zero grad_operand."""
    device = torch.device("tpu")
    table_depth = 16
    table_width = 8
    num_indices = 32

    indices = torch.randint(
        0, table_depth, (num_indices,), dtype=torch.int32, device=device
    )
    grad_output = torch.zeros(
        num_indices, table_width, dtype=torch.float32, device=device
    )
    grad_operand = torch.zeros(
        table_depth, table_width, dtype=torch.float32, device=device
    )

    out_grad = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, grad_operand
    )

    self.assert_close(
        golden_result=torch.zeros(table_depth, table_width),
        torch_tpu_result=out_grad.cpu(),
    )

  def test_sparse_gather_repeated_indices_accumulation(self):
    """Verifies that multiple occurrences of the same index accumulate correctly."""
    device = torch.device("tpu")
    table_depth = 16
    table_width = 8

    indices = torch.tensor(
        [3, 7, 3, INT_MAX, 3, 7, INT_MAX, 3, 3, INT_MAX],
        dtype=torch.int32,
        device=device,
    )
    grad_output = torch.ones(
        indices.shape[0], table_width, dtype=torch.float32, device=device
    )
    grad_operand = torch.zeros(
        table_depth, table_width, dtype=torch.float32, device=device
    )

    out_grad = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, grad_operand
    )
    out_cpu = out_grad.cpu()

    self.assert_close(
        golden_result=torch.full((table_width,), 5.0),
        torch_tpu_result=out_cpu[3],
    )
    self.assert_close(
        golden_result=torch.full((table_width,), 2.0),
        torch_tpu_result=out_cpu[7],
    )
    self.assert_close(
        golden_result=torch.zeros(table_width),
        torch_tpu_result=out_cpu[0],
    )
    self.assert_close(
        golden_result=torch.zeros(table_width),
        torch_tpu_result=out_cpu[1],
    )

  def test_sparse_gather_backward_all_padding(self):
    """Verifies edge case where the entire indices tensor consists of padding."""
    device = torch.device("tpu")
    table_depth = 32
    table_width = 16
    num_indices = 64

    indices = torch.full(
        (num_indices,), INT_MAX, dtype=torch.int32, device=device
    )
    grad_output = torch.ones(
        num_indices, table_width, dtype=torch.float32, device=device
    )
    grad_operand = torch.zeros(
        table_depth, table_width, dtype=torch.float32, device=device
    )

    out_grad = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, grad_operand
    )

    self.assert_close(
        golden_result=torch.zeros(table_depth, table_width),
        torch_tpu_result=out_grad.cpu(),
    )

  def test_sparse_gather_backward_negative_and_out_of_bounds_indices(self):
    """Verifies that invalid indices (negative and >= table_depth) are safely masked."""
    device = torch.device("tpu")
    table_depth = 16
    table_width = 8

    indices = torch.tensor(
        [2, -1, 100, INT_MAX, 2, -5, 50, INT_MAX],
        dtype=torch.int32,
        device=device,
    )
    grad_output = torch.ones(
        indices.shape[0], table_width, dtype=torch.float32, device=device
    )
    grad_operand = torch.zeros(
        table_depth, table_width, dtype=torch.float32, device=device
    )

    out_grad = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, grad_operand
    )
    out_cpu = out_grad.cpu()

    self.assert_close(
        golden_result=torch.full((table_width,), 2.0),
        torch_tpu_result=out_cpu[2],
    )
    self.assert_close(
        golden_result=torch.zeros(table_width),
        torch_tpu_result=out_cpu[0],
    )
    self.assert_close(
        golden_result=torch.tensor(2.0 * table_width),
        torch_tpu_result=out_cpu.sum(),
    )

  def test_sparse_gather_backward_single_element(self):
    """Verifies edge case of single element gather backward with table_depth=1."""
    device = torch.device("tpu")
    table_depth = 1
    table_width = 4

    indices = torch.tensor([0], dtype=torch.int32, device=device)
    grad_output = torch.tensor(
        [[3.5, 4.5, -1.0, 2.0]], dtype=torch.float32, device=device
    )
    grad_operand = torch.zeros(
        table_depth, table_width, dtype=torch.float32, device=device
    )

    out_grad = torch.ops.tpu.sparse_gather_backward(
        grad_output, indices, grad_operand
    )

    self.assert_close(
        golden_result=grad_output.cpu(),
        torch_tpu_result=out_grad.cpu(),
    )


if __name__ == "__main__":
  absltest.main()
