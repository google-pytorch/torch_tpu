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

from collections.abc import Callable, Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._dynamo import testing as dynamo_testing
from torch._subclasses import fake_tensor
import torch.export
import torch.fx
from torch.fx.experimental import symbolic_shapes
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export
from tests import op_testing


# TODO(ziyinh): Wrap these utility functions into Python helpers for users to
# use.
def _compile_body_and_make_dummy_outputs(
    fn: Callable[..., Any],
    max_non_zeroes_per_row: int,
    csrs: Sequence[Sequence[torch.Tensor]],
    dense_args: Sequence[torch.Tensor],
    result_row_pointer_op_index: int = 0,
) -> tuple[str, list[torch.Tensor]]:
  """Traces the row function using fx_to_mlir and generates dummy meta outputs."""
  sample_body_args: list[torch.Tensor] = [
      tpu_torch_compile.placeholder([], torch.int32, False)
  ]
  for c in csrs:
    sample_body_args.append(
        tpu_torch_compile.placeholder([], torch.int32, False)
    )
    for t in c[1:]:
      shape = [max_non_zeroes_per_row, *t.shape[1:]]
      sample_body_args.append(
          tpu_torch_compile.placeholder(shape, t.dtype, t.requires_grad)
      )
  sample_body_args.extend(
      tpu_torch_compile.placeholder_like(a) for a in dense_args
  )

  actual_fn = fn.module if isinstance(fn, torch.export.ExportedProgram) else fn
  if not isinstance(actual_fn, torch.fx.GraphModule):
    actual_fn = torch.fx.symbolic_trace(actual_fn)

  body_mlir = export.fx_to_mlir(actual_fn, args=sample_body_args)
  serialized_mlir = body_mlir.serialize_text()

  ref_row_ptrs = csrs[result_row_pointer_op_index][0]
  ref_cols = csrs[result_row_pointer_op_index][1]
  out_device = ref_cols.device
  dummy_outputs = [torch.empty_like(ref_row_ptrs, device=out_device)]
  for r in body_mlir.mlir_result_tensors:
    if ref_cols.ndim == 1:
      full_shape = (ref_cols.shape[0], *r.shape[1:])
    else:
      full_shape = (*ref_cols.shape[:-1], *r.shape)
    dummy_outputs.append(
        torch.empty(full_shape, dtype=r.dtype, device=out_device)
    )

  return serialized_mlir, dummy_outputs


def _dispatch_sparse_map_row(
    fn: Callable[..., Any],
    csrs: Sequence[Sequence[torch.Tensor]],
    *dense_args: torch.Tensor,
    max_non_zeroes_per_row: int = 16,
    pad_value: int = 2147483647,
    max_pad_per_row: int = 8,
    wait_threshold: float = 0.5,
    result_row_pointer_op_index: int = 0,
) -> tuple[torch.Tensor, ...]:
  serialized_mlir, dummy_outputs = _compile_body_and_make_dummy_outputs(
      fn, max_non_zeroes_per_row, csrs, dense_args, result_row_pointer_op_index
  )
  csr_tensors = []
  for c in csrs:
    csr_tensors.extend(c)
  csr_lengths = [len(c) for c in csrs]
  results = torch.tpu.ops.sparse_map_row(
      csr_tensors,
      csr_lengths,
      list(dense_args),
      serialized_mlir,
      dummy_outputs,
      max_non_zeroes_per_row,
      pad_value,
      max_pad_per_row,
      wait_threshold,
      result_row_pointer_op_index,
  )
  return tuple(results)


class SparseMapRowTest(op_testing.TorchTpuTestBase):
  """Tests for sparse_map_row operator on TPU SparseCore."""

  def _get_dummy_padded_csr(self, device: torch.device, dtype=torch.float32):
    """Returns hardware-padded dummy CSR tensors for testing."""
    # Row pointers aligned to 8 elements.
    row_ptrs = torch.tensor(
        [0, 8, 16, 16, 16, 16, 16, 16], dtype=torch.int32, device=device
    )
    indices = torch.tensor(
        [0, 1, 1, 2, 2, 3, 4, -1, 0, 1, 1, 2, 2, 3, 4, -1],
        dtype=torch.int32,
        device=device,
    )
    values = torch.tensor(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            -1,
            90,
            100,
            110,
            120,
            130,
            140,
            150,
            -1,
        ],
        dtype=dtype,
        device=device,
    )
    return row_ptrs, indices, values

  @parameterized.named_parameters(
      dict(testcase_name="eager", compile_fn=False),
      dict(testcase_name="compile", compile_fn=True),
  )
  def test_sparse_map_row_add(self, compile_fn: bool):
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    row_ptrs, indices, values = self._get_dummy_padded_csr(
        device, dtype=torch.int32
    )
    scalar = torch.tensor(42, dtype=torch.int32, device=device)

    def simple_add(unused_row_idx, unused_nnz, cols, vals, s):
      return cols, vals + s

    if compile_fn:
      serialized_mlir, dummy_outputs = _compile_body_and_make_dummy_outputs(
          simple_add, 16, [(row_ptrs, indices, values)], [scalar]
      )

      @torch.compile(backend="tpu")
      def run_op(r, i, v, s):
        return torch.tpu.ops.sparse_map_row(
            [r, i, v],
            [3],
            [s],
            serialized_mlir,
            dummy_outputs,
            16,
            2147483647,
            8,
            0.5,
            0,
        )

    else:

      def run_op(r, i, v, s):
        return _dispatch_sparse_map_row(simple_add, [(r, i, v)], s)

    res_row_ptrs, res_indices, res_values = run_op(
        row_ptrs, indices, values, scalar
    )

    expected_values = values + scalar
    self.assert_close(
        golden_result=row_ptrs.cpu(), torch_tpu_result=res_row_ptrs.cpu()
    )
    self.assert_close(
        golden_result=indices.cpu(), torch_tpu_result=res_indices.cpu()
    )
    self.assert_close(
        golden_result=expected_values.cpu(), torch_tpu_result=res_values.cpu()
    )

  @parameterized.named_parameters(
      dict(testcase_name="eager", compile_fn=False),
      dict(testcase_name="compile", compile_fn=True),
  )
  def test_sparse_map_row_multiply(self, compile_fn: bool):
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    row_ptrs, indices, values = self._get_dummy_padded_csr(
        device, dtype=torch.int32
    )
    scale = torch.tensor(2, dtype=torch.int32, device=device)

    def complex_op(unused_row_idx, unused_nnz, cols, vals, s):
      res = (vals * s) + cols
      return cols, torch.clamp(res, min=30)

    if compile_fn:
      serialized_mlir, dummy_outputs = _compile_body_and_make_dummy_outputs(
          complex_op, 16, [(row_ptrs, indices, values)], [scale]
      )

      @torch.compile(backend="tpu")
      def run_op(r, i, v, s):
        return torch.tpu.ops.sparse_map_row(
            [r, i, v],
            [3],
            [s],
            serialized_mlir,
            dummy_outputs,
            16,
            2147483647,
            8,
            0.5,
            0,
        )

    else:

      def run_op(r, i, v, s):
        return _dispatch_sparse_map_row(complex_op, [(r, i, v)], s)

    res_row_ptrs, res_indices, res_values = run_op(
        row_ptrs, indices, values, scale
    )

    res_expected = (values * scale) + indices
    expected_values = torch.clamp(res_expected, min=30)

    self.assert_close(
        golden_result=row_ptrs.cpu(), torch_tpu_result=res_row_ptrs.cpu()
    )
    self.assert_close(
        golden_result=indices.cpu(), torch_tpu_result=res_indices.cpu()
    )
    self.assert_close(
        golden_result=expected_values.cpu(), torch_tpu_result=res_values.cpu()
    )

  @parameterized.named_parameters(
      dict(testcase_name="eager", compile_fn=False),
      dict(testcase_name="compile", compile_fn=True),
  )
  def test_sparse_map_row_mixed_args(self, compile_fn: bool):
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    row_ptrs, indices, values = self._get_dummy_padded_csr(
        device, dtype=torch.float32
    )
    scalar_int = torch.tensor(5, dtype=torch.int32, device=device)
    scalar_float = torch.tensor(2.5, dtype=torch.float32, device=device)

    def mixed_op(unused_row_idx, unused_nnz, cols, vals, s_int, s_float):
      return cols + s_int, vals + s_float

    if compile_fn:
      serialized_mlir, dummy_outputs = _compile_body_and_make_dummy_outputs(
          mixed_op,
          16,
          [(row_ptrs, indices, values)],
          [scalar_int, scalar_float],
      )

      @torch.compile(backend="tpu")
      def run_op(r, i, v, s_i, s_f):
        return torch.tpu.ops.sparse_map_row(
            [r, i, v],
            [3],
            [s_i, s_f],
            serialized_mlir,
            dummy_outputs,
            16,
            2147483647,
            8,
            0.5,
            0,
        )

    else:

      def run_op(r, i, v, s_i, s_f):
        return _dispatch_sparse_map_row(mixed_op, [(r, i, v)], s_i, s_f)

    res_row_ptrs, res_indices, res_values = run_op(
        row_ptrs, indices, values, scalar_int, scalar_float
    )

    expected_indices = indices + scalar_int
    expected_values = values + scalar_float
    self.assert_close(
        golden_result=row_ptrs.cpu(), torch_tpu_result=res_row_ptrs.cpu()
    )
    self.assert_close(
        golden_result=expected_indices.cpu(), torch_tpu_result=res_indices.cpu()
    )
    self.assert_close(
        golden_result=expected_values.cpu(), torch_tpu_result=res_values.cpu()
    )

  @parameterized.named_parameters(
      dict(testcase_name="eager", compile_fn=False),
      dict(testcase_name="compile", compile_fn=True),
  )
  def test_sparse_map_row_add_two_csrs(self, compile_fn: bool):
    self._skip_if_sparsecore_unsupported()
    device = torch.device("tpu")
    row_ptrs1, indices1, values1 = self._get_dummy_padded_csr(
        device, dtype=torch.float32
    )
    row_ptrs2, indices2, values2 = self._get_dummy_padded_csr(
        device, dtype=torch.float32
    )

    def map_fn(
        unused_row_idx,
        unused_nnz1,
        cols1,
        vals1,
        unused_nnz2,
        unused_cols2,
        vals2,
    ):
      return cols1, vals1 + vals2

    if compile_fn:
      serialized_mlir, dummy_outputs = _compile_body_and_make_dummy_outputs(
          map_fn,
          16,
          [(row_ptrs1, indices1, values1), (row_ptrs2, indices2, values2)],
          [],
      )

      @torch.compile(backend="tpu")
      def run_op(r1, i1, v1, r2, i2, v2):
        return torch.tpu.ops.sparse_map_row(
            [r1, i1, v1, r2, i2, v2],
            [3, 3],
            [],
            serialized_mlir,
            dummy_outputs,
            16,
            2147483647,
            8,
            0.5,
            0,
        )

    else:

      def run_op(r1, i1, v1, r2, i2, v2):
        return _dispatch_sparse_map_row(
            map_fn,
            [
                (r1, i1, v1),
                (r2, i2, v2),
            ],
        )

    res_row_ptrs, res_indices, res_values = run_op(
        row_ptrs1, indices1, values1, row_ptrs2, indices2, values2
    )

    expected_values = values1 + values2
    self.assert_close(
        golden_result=row_ptrs1.cpu(), torch_tpu_result=res_row_ptrs.cpu()
    )
    self.assert_close(
        golden_result=indices1.cpu(), torch_tpu_result=res_indices.cpu()
    )
    self.assert_close(
        golden_result=expected_values.cpu(), torch_tpu_result=res_values.cpu()
    )

  def test_sparse_map_row_meta(self):
    row_ptrs = torch.empty(8, dtype=torch.int32, device="meta")
    indices = torch.empty(16, dtype=torch.int32, device="meta")
    values = torch.empty(16, dtype=torch.float32, device="meta")
    dummy_outputs = [
        torch.empty(8, dtype=torch.int32, device="meta"),
        torch.empty(16, dtype=torch.int32, device="meta"),
        torch.empty(16, dtype=torch.float32, device="meta"),
    ]
    out = torch.tpu.ops.sparse_map_row(
        [row_ptrs, indices, values],
        [3],
        [],
        "dummy_mlir",
        dummy_outputs,
        16,
        2147483647,
        8,
        0.5,
        0,
    )
    self.assertLen(out, 3)
    self.assertEqual(out[0].shape, torch.Size([8]))
    self.assertEqual(out[0].dtype, torch.int32)
    self.assertEqual(out[0].device, torch.device("meta"))
    self.assertEqual(out[1].shape, torch.Size([16]))
    self.assertEqual(out[1].dtype, torch.int32)
    self.assertEqual(out[2].shape, torch.Size([16]))
    self.assertEqual(out[2].dtype, torch.float32)

  def test_sparse_map_row_meta_validation(self):
    dummy = torch.empty(8, dtype=torch.int32, device="meta")
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates CSR list.
        RuntimeError, "expected at least 1 CSR tensor"
    ):
      torch.tpu.ops.sparse_map_row(
          [], [], [], "mlir", [dummy], 16, 0, 8, 0.5, 0
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates lengths.
        RuntimeError, "expected at least 1 entry in csr_lengths"
    ):
      torch.tpu.ops.sparse_map_row(
          [dummy], [], [], "mlir", [dummy], 16, 0, 8, 0.5, 0
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates CSR count.
        RuntimeError, "sum of csr_lengths"
    ):
      torch.tpu.ops.sparse_map_row(
          [dummy], [2], [], "mlir", [dummy], 16, 0, 8, 0.5, 0
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates op index.
        RuntimeError, "expected result_row_pointer_op_index"
    ):
      torch.tpu.ops.sparse_map_row(
          [dummy], [1], [], "mlir", [dummy], 16, 0, 8, 0.5, 1
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates max nnz.
        RuntimeError, "expected max_non_zeroes_per_row"
    ):
      torch.tpu.ops.sparse_map_row(
          [dummy], [1], [], "mlir", [dummy], 0, 0, 8, 0.5, 0
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates MLIR.
        RuntimeError, "expected a non-empty body MLIR"
    ):
      torch.tpu.ops.sparse_map_row(
          [dummy], [1], [], "", [dummy], 16, 0, 8, 0.5, 0
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates output.
        RuntimeError, "expected at least 1 dummy output tensor"
    ):
      torch.tpu.ops.sparse_map_row(
          [dummy], [1], [], "mlir", [], 16, 0, 8, 0.5, 0
      )

  def test_sparse_map_row_eager_meta(self):
    row_ptrs = torch.empty(8, dtype=torch.int32, device="meta")
    indices = torch.empty(16, dtype=torch.int32, device="meta")
    values = torch.empty(16, dtype=torch.float32, device="meta")
    scalar = torch.empty((), dtype=torch.float32, device="meta")

    def map_fn(unused_row_idx, unused_nnz, cols, vals, s):
      return cols, vals + s

    res_row_ptrs, res_indices, res_values = _dispatch_sparse_map_row(
        map_fn, [(row_ptrs, indices, values)], scalar
    )
    self.assertEqual(res_row_ptrs.shape, torch.Size([8]))
    self.assertEqual(res_row_ptrs.dtype, torch.int32)
    self.assertEqual(res_row_ptrs.device, torch.device("meta"))
    self.assertEqual(res_indices.shape, torch.Size([16]))
    self.assertEqual(res_indices.dtype, torch.int32)
    self.assertEqual(res_indices.device, torch.device("meta"))
    self.assertEqual(res_values.shape, torch.Size([16]))
    self.assertEqual(res_values.dtype, torch.float32)
    self.assertEqual(res_values.device, torch.device("meta"))

  @parameterized.parameters("meta", "tpu")
  def test_sparse_map_row_fake_symbolic_shapes(self, device):
    """Tests sparse_map_row doesn't specialize symbolic FakeTensor dims."""
    shape_env = symbolic_shapes.ShapeEnv()
    mode = fake_tensor.FakeTensorMode(shape_env=shape_env)

    def make_fake(*shape, dtype):
      return mode.from_tensor(
          torch.empty(shape, dtype=dtype, device=device),
          symbolic_context=symbolic_shapes.StatelessSymbolicContext(
              dynamic_sizes=[symbolic_shapes.DimDynamic.DYNAMIC] * len(shape)
          ),
      )

    row_ptrs = make_fake(8, dtype=torch.int32)
    indices = make_fake(16, dtype=torch.int32)
    values = make_fake(16, 4, dtype=torch.float32)
    dummy_outputs = [
        make_fake(8, dtype=torch.int32),
        make_fake(16, dtype=torch.int32),
        make_fake(16, 4, dtype=torch.float32),
    ]
    with mode:
      out = torch.tpu.ops.sparse_map_row(
          [row_ptrs, indices, values],
          [3],
          [],
          "dummy_mlir",
          dummy_outputs,
          16,
          2147483647,
          8,
          0.5,
          0,
      )
    self.assertLen(out, len(dummy_outputs))
    sizes = [size for t in [*dummy_outputs, *out] for size in t.shape]
    for size in sizes:
      self.assertIsInstance(size, torch.SymInt)
    specialized = [
        size for size in sizes if shape_env.replace(size.node.expr).is_number
    ]
    self.assertEmpty(specialized)
    for dummy, result in zip(dummy_outputs, out):
      self.assertEqual(result.shape, dummy.shape)
      self.assertEqual(result.dtype, dummy.dtype)

  def test_sparse_map_row_compile_dynamic_shapes(self):
    """Tests compiled sparse_map_row recompiles once for dynamic shapes."""
    torch._dynamo.reset()
    self.addCleanup(torch._dynamo.reset)
    counter = dynamo_testing.CompileCounter()

    @torch.compile(backend=counter, fullgraph=True)
    def fn(row_ptrs, indices, values, dummy_outputs):
      return torch.tpu.ops.sparse_map_row(
          [row_ptrs, indices, values],
          [3],
          [],
          "dummy_mlir",
          dummy_outputs,
          16,
          2147483647,
          8,
          0.5,
          0,
      )

    for n in (2, 3, 4):
      # All dims change between calls.
      row_ptrs = torch.empty(n * 4, dtype=torch.int32, device="meta")
      indices = torch.empty(n * 8, dtype=torch.int32, device="meta")
      values = torch.empty(n * 8, n + 4, device="meta")
      dummy_outputs = [
          torch.empty(n * 4, dtype=torch.int32, device="meta"),
          torch.empty(n * 8, dtype=torch.int32, device="meta"),
          torch.empty(n * 8, n + 4, device="meta"),
      ]
      out = fn(row_ptrs, indices, values, dummy_outputs)
      self.assertEqual([t.shape for t in out], [t.shape for t in dummy_outputs])
    # One static frame, then one dynamic frame that serves all later sizes.
    self.assertEqual(counter.frame_count, 2)


if __name__ == "__main__":
  absltest.main()
