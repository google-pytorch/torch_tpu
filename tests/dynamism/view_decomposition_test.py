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

"""Unit tests for view decomposition."""

from __future__ import annotations

from typing import Any

from absl.testing import absltest
import sympy
import torch
from torch._dynamo.backends.common import aot_autograd
from torch_tpu._internal.compile.dynamic import view_decomposition
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class ViewDecompositionTest(seed_test_utils.RepeatableTest):

  def _is_dynamic_symint(self, val: Any) -> bool:
    """Checks if a value or node represents a dynamic (non-constant) SymInt."""
    if isinstance(val, torch.fx.Node) and "val" in val.meta:
      val = val.meta["val"]
    if isinstance(val, torch.SymInt):
      return True
    return False

  def _assert_base_shapes_match(
      self,
      node_base_shape: list[Any],
      tensor_base_shape: list[Any],
  ):
    """Verifies that base shapes match, ignoring dynamic symint dimensions."""
    self.assertLen(
        node_base_shape,
        len(tensor_base_shape),
        f"Rank mismatch: {node_base_shape} vs {tensor_base_shape}",
    )
    for i, (node_dim, tensor_dim) in enumerate(
        zip(node_base_shape, tensor_base_shape)
    ):
      if self._is_dynamic_symint(node_dim):
        # Ignore dynamic symint dimension: torch.fx.Node has SymInt while torch.Tensor has static int
        continue
      if isinstance(node_dim, torch.SymInt):
        if hasattr(node_dim, "node") and hasattr(node_dim.node, "expr"):
          expr = node_dim.node.expr
          if isinstance(expr, (int, sympy.Integer)):
            self.assertEqual(
                int(expr),
                int(tensor_dim),
                f"Dimension {i} mismatch: {node_dim} vs {tensor_dim}",
            )
            continue
        # Fallback if it's dynamic SymInt
        continue
      self.assertEqual(
          int(node_dim),
          int(tensor_dim),
          f"Dimension {i} mismatch: {node_dim} vs {tensor_dim}",
      )

  def _capture_view_node(
      self,
      base_tensor: torch.Tensor,
      view_fn: Any,
  ) -> tuple[
      torch.fx.Node, tuple[list[Any], list[tuple[str, tuple[Any, ...]]]] | None
  ]:
    """Compiles view_fn(base_tensor) and calls view_decomposition mid-compilation inside fw_compiler."""
    torch._dynamo.reset()
    captured_node = None
    captured_decomp = None
    self.guards_before = []
    self.guards_after = []

    def fw_compiler(graph_module, example_inputs):
      del example_inputs
      nonlocal captured_node, captured_decomp
      # The argument to add is the result of view_fn
      for node in reversed(list(graph_module.graph.nodes)):
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.add.Tensor
        ):
          captured_node = node.args[0]
          break
      if captured_node is None:
        for node in reversed(list(graph_module.graph.nodes)):
          if node.op != "output":
            captured_node = node
            break
      if captured_node is not None:
        shape_env = None
        val = captured_node.meta.get("val")
        if val is not None:
          for dim in list(val.shape) + list(val.stride()):
            if isinstance(dim, torch.SymInt) and hasattr(dim.node, "shape_env"):
              shape_env = dim.node.shape_env
              break

        if shape_env is not None:
          self.guards_before = list(shape_env.guards)

        captured_decomp = view_decomposition.decompose_into_view_sequence(
            captured_node
        )

        if shape_env is not None:
          self.guards_after = list(shape_env.guards)
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler))
    def dummy_fn(x):
      y = view_fn(x)
      return y + y

    dummy_fn(base_tensor)
    self.assertIsNotNone(captured_node, "Failed to capture view node")
    return captured_node, captured_decomp

  def _print_decomposition(
      self,
      node_or_val: torch.fx.Node | torch.Tensor,
      decomp: tuple[list[Any], list[tuple[str, tuple[Any, ...]]]] | None,
  ) -> None:
    """Prints the decomposition of a tensor or FX node along with its shape and stride."""
    if isinstance(node_or_val, torch.fx.Node):
      val = node_or_val.meta.get("val")
      print(
          f"Input FX Node '{node_or_val.name}' (op={node_or_val.op},"
          f" target={getattr(node_or_val, 'target', None)}):"
      )
      if val is not None:
        print(f"  Node val shape: {val.shape}")
        print(f"  Node val stride: {val.stride()}")
    else:
      print(
          f"Input Tensor shape: {node_or_val.shape}, stride:"
          f" {node_or_val.stride()}"
      )

    if decomp is None:
      print("  Decomposition is None")
      return
    base_shape, view_ops = decomp
    print(f"  Decomp Base shape: {base_shape}")
    print(f"  Decomp View ops: {view_ops}")

  def _print_guards(self) -> None:
    """Prints the before, after, and newly added guards recorded on self."""
    guards_before_exprs = [getattr(g, "expr", g) for g in self.guards_before]
    guards_after_exprs = [getattr(g, "expr", g) for g in self.guards_after]
    print(f"  Guards before ({len(self.guards_before)}): {guards_before_exprs}")
    print(f"  Guards after ({len(self.guards_after)}): {guards_after_exprs}")
    new_guards = guards_after_exprs[len(self.guards_before) :]
    print(f"  New guards added ({len(new_guards)}): {new_guards}")

  def test_decompose_dynamic_shape_and_dynamic_stride_transpose(self):
    """Tests view decomposition when torch.fx.Node has SymInt in both shape and stride."""
    # Dynamic tensor: dim 1 is dynamic
    base_dyn = torch.randn(16, 20, 8, 64)
    torch._dynamo.mark_dynamic(base_dyn, 1, min=10, max=100)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x.transpose(1, 2)
    )

    # Static tensor view
    base_static = torch.randn(16, 20, 8, 64)
    view_static = base_static.transpose(1, 2)

    # Decompose static tensor
    tensor_decomp = view_decomposition.decompose_into_view_sequence(view_static)

    self._print_decomposition(view_node, node_decomp)
    self._print_guards()
    self._print_decomposition(view_static, tensor_decomp)

    self.assertIsNotNone(node_decomp)
    self.assertIsNotNone(tensor_decomp)

    node_base_shape, _ = node_decomp
    tensor_base_shape, tensor_view_ops = tensor_decomp

    # Verify base shape matches ignoring symint dimension
    self._assert_base_shapes_match(node_base_shape, tensor_base_shape)
    self.assertEqual(tensor_base_shape, [16, 20, 8, 64])
    self.assertEqual(tensor_view_ops, [("permute", ([0, 2, 1, 3],))])

  def test_decompose_static_shape_dynamic_stride_slice(self):
    """Tests view decomposition when torch.fx.Node has static shape but dynamic stride."""
    # Dynamic base buffer: dim 2 is dynamic
    base_dyn = torch.randn(16, 8, 256, 8)
    torch._dynamo.mark_dynamic(base_dyn, 2, min=128, max=512)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x[:, :, -127:, :]
    )

    # Static tensor view
    base_static = torch.randn(16, 8, 256, 8)
    view_static = base_static[:, :, -127:, :]

    # Decompose static tensor
    tensor_decomp = view_decomposition.decompose_into_view_sequence(view_static)

    self._print_decomposition(view_node, node_decomp)
    self._print_guards()
    self._print_decomposition(view_static, tensor_decomp)

    self.assertIsNotNone(node_decomp)
    self.assertIsNotNone(tensor_decomp)

    node_base_shape, _ = node_decomp
    tensor_base_shape, _ = tensor_decomp

    self._assert_base_shapes_match(node_base_shape, tensor_base_shape)
    self.assertEqual(tensor_base_shape, [16, 8, 256, 8])

  def test_decompose_prefix_slice(self):
    """Tests view decomposition on prefix slice of dynamic tensor."""
    base_dyn = torch.randn(32, 200, 64)
    torch._dynamo.mark_dynamic(base_dyn, 1, min=100, max=500)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x[:, :50, :]
    )

    base_static = torch.randn(32, 200, 64)
    view_static = base_static[:, :50, :]

    tensor_decomp = view_decomposition.decompose_into_view_sequence(view_static)

    self._print_decomposition(view_node, node_decomp)
    self._print_guards()
    self._print_decomposition(view_static, tensor_decomp)

    self.assertIsNotNone(node_decomp)
    self.assertIsNotNone(tensor_decomp)

    node_base_shape, _ = node_decomp
    tensor_base_shape, _ = tensor_decomp

    self._assert_base_shapes_match(node_base_shape, tensor_base_shape)
    self.assertEqual(tensor_base_shape, [32, 200, 64])

  def test_decompose_dynamic_slice_dynamic_shape_and_stride(self):
    """Tests view decomposition on dynamic offset slice where shape and stride have SymInt."""
    base_dyn = torch.randn(10, 40, 20)
    torch._dynamo.mark_dynamic(base_dyn, 1)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x[:, 10:, :]
    )

    base_static = torch.randn(10, 40, 20)
    view_static = base_static[:, 10:, :]

    tensor_decomp = view_decomposition.decompose_into_view_sequence(view_static)

    self._print_decomposition(view_node, node_decomp)
    self._print_guards()
    self._print_decomposition(view_static, tensor_decomp)

    self.assertIsNotNone(node_decomp)
    self.assertIsNotNone(tensor_decomp)

    node_base_shape, node_view_ops = node_decomp
    tensor_base_shape, _ = tensor_decomp

    self._assert_base_shapes_match(node_base_shape, tensor_base_shape)
    self.assertEqual(tensor_base_shape, [10, 40, 20])

    # Verify that the view node shape and the slice end in view_ops are dynamic SymInts
    self.assertIsInstance(view_node.meta["val"].shape[1], torch.SymInt)
    self.assertEqual(node_view_ops[0][0], "slice")
    self.assertIsInstance(node_view_ops[0][1][2], torch.SymInt)

  def test_decompose_dynamic_dim0_slice_with_storage_offset(self):
    """Tests view decomposition on a tensor with dynamic dim 0 and offset 10.

    Matches the example in _delinearize_storage_offset docstring:
      base shape = (m, 3, 3) where dim 0 (m) is dynamic (SymInt).
      base_strides = (9, 3, 1).
      Slicing with [1:, :, 1:] produces offset 10 and shape (m - 1, 3, 2).
    """
    base_dyn = torch.randn(10, 3, 3)
    torch._dynamo.mark_dynamic(base_dyn, 0, min=3, max=100)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x[1:, :, 1:]
    )

    base_static = torch.randn(10, 3, 3)
    view_static = base_static[1:, :, 1:]

    tensor_decomp = view_decomposition.decompose_into_view_sequence(view_static)

    self._print_decomposition(view_node, node_decomp)
    self._print_guards()
    self._print_decomposition(view_static, tensor_decomp)

    self.assertIsNotNone(node_decomp)
    self.assertIsNotNone(tensor_decomp)

    node_base_shape, node_view_ops = node_decomp
    tensor_base_shape, tensor_view_ops = tensor_decomp

    self._assert_base_shapes_match(node_base_shape, tensor_base_shape)
    self.assertEqual(tensor_base_shape, [10, 3, 3])

    # Verify dim 0 is a dynamic SymInt in the view node, base shape, and slice op
    self.assertIsInstance(view_node.meta["val"].shape[0], torch.SymInt)
    self.assertIsInstance(node_base_shape[0], torch.SymInt)
    self.assertEqual(node_base_shape[1:], [3, 3])

    # Slices on dim 0 and dim 2
    self.assertEqual(len(node_view_ops), 2)
    self.assertEqual(node_view_ops[0][0], "slice")
    self.assertEqual(node_view_ops[0][1][0], 0)  # dim_idx 0
    self.assertEqual(node_view_ops[0][1][1], 1)  # start 1
    self.assertIsInstance(
        node_view_ops[0][1][2], torch.SymInt
    )  # end is dynamic m

    self.assertEqual(node_view_ops[1][0], "slice")
    self.assertEqual(node_view_ops[1][1][0], 2)  # dim_idx 2
    self.assertEqual(node_view_ops[1][1][1], 1)  # start 1
    self.assertEqual(node_view_ops[1][1][2], 3)  # end 3

    # Verify static view ops match
    self.assertEqual(
        tensor_view_ops, [("slice", (0, 1, 10, 1)), ("slice", (2, 1, 3, 1))]
    )

  def test_decompose_combined_permute_and_slice(self):
    """Tests view decomposition with both permute and slice operations."""
    base_dyn = torch.randn(16, 128, 8, 64)
    torch._dynamo.mark_dynamic(base_dyn, 1, min=64, max=256)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x.transpose(1, 2)[:, :, :50, :]
    )

    base_static = torch.randn(16, 128, 8, 64)
    view_static = base_static.transpose(1, 2)[:, :, :50, :]

    tensor_decomp = view_decomposition.decompose_into_view_sequence(view_static)

    self._print_decomposition(view_node, node_decomp)
    self._print_guards()
    self._print_decomposition(view_static, tensor_decomp)

    self.assertIsNotNone(node_decomp)
    self.assertIsNotNone(tensor_decomp)

    node_base_shape, _ = node_decomp
    tensor_base_shape, _ = tensor_decomp

    self._assert_base_shapes_match(node_base_shape, tensor_base_shape)
    self.assertEqual(tensor_base_shape, [16, 128, 8, 64])

  def test_decompose_multi_dimensional_slice(self):
    """Tests view decomposition with slicing along multiple dimensions."""
    base_dyn = torch.randn(20, 100, 50)
    torch._dynamo.mark_dynamic(base_dyn, 1, min=50, max=200)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x[5:15, :40, 10:30]
    )

    base_static = torch.randn(20, 100, 50)
    view_static = base_static[5:15, :40, 10:30]

    tensor_decomp = view_decomposition.decompose_into_view_sequence(view_static)

    self._print_decomposition(view_node, node_decomp)
    self._print_guards()
    self._print_decomposition(view_static, tensor_decomp)

    self.assertIsNotNone(node_decomp)
    self.assertIsNotNone(tensor_decomp)

    node_base_shape, _ = node_decomp
    tensor_base_shape, tensor_view_ops = tensor_decomp

    self._assert_base_shapes_match(node_base_shape, tensor_base_shape)
    self.assertEqual(tensor_base_shape, [15, 100, 50])
    self.assertEqual(
        tensor_view_ops,
        [
            ("slice", (0, 5, 15, 1)),
            ("slice", (1, 0, 40, 1)),
            ("slice", (2, 10, 30, 1)),
        ],
    )

  def test_delinearize_storage_offset(self):
    """Tests delinearizing storage offsets into multidimensional coordinate offsets."""
    base_strides = [5000, 50, 1]

    # Zero offset
    self.assertEqual(
        view_decomposition._delinearize_storage_offset(0, base_strides),
        [0, 0, 0],
    )

    # Offset entirely on innermost dimension (stride = 1)
    self.assertEqual(
        view_decomposition._delinearize_storage_offset(10, base_strides),
        [0, 0, 10],
    )

    # Offset across multiple dimensions (5 * 5000 + 0 * 50 + 10 = 25010)
    self.assertEqual(
        view_decomposition._delinearize_storage_offset(25010, base_strides),
        [5, 0, 10],
    )

    # Offset across all dimensions (3 * 5000 + 4 * 50 + 7 = 15207)
    self.assertEqual(
        view_decomposition._delinearize_storage_offset(15207, base_strides),
        [3, 4, 7],
    )

    # 1D tensor offset
    self.assertEqual(
        view_decomposition._delinearize_storage_offset(15, [1]),
        [15],
    )

    # Docstring example: base_strides = [9, 3, 1], offset = 10 -> [1, 0, 1]
    self.assertEqual(
        view_decomposition._delinearize_storage_offset(10, [9, 3, 1]),
        [1, 0, 1],
    )

  def test_decompose_contiguous_tensor_returns_none(self):
    """Tests that contiguous tensor returns None for both node and tensor."""
    base_dyn = torch.randn(16, 20, 8, 64)
    torch._dynamo.mark_dynamic(base_dyn, 1, min=10, max=100)

    view_node, node_decomp = self._capture_view_node(base_dyn, lambda x: x)

    base_static = torch.randn(16, 20, 8, 64)

    tensor_decomp = view_decomposition.decompose_into_view_sequence(base_static)

    self._print_decomposition(view_node, node_decomp)
    self._print_guards()
    self._print_decomposition(base_static, tensor_decomp)

    self.assertIsNone(node_decomp)
    self.assertIsNone(tensor_decomp)

  def test_decompose_adds_symint_constraint(self):
    """Tests that view decomposition works with dynamic SymInt shapes."""
    base_dyn = torch.randn(128, 1024)
    torch._dynamo.mark_dynamic(base_dyn, 1)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x.view(16, 8, -1, 8)[:, :, :127, :]
    )
    self._print_decomposition(view_node, node_decomp)
    self._print_guards()

    self.assertIsNotNone(node_decomp)
    base_shape, _ = node_decomp
    self.assertTrue(any(isinstance(dim, torch.SymInt) for dim in base_shape))
    self.assertGreater(len(self.guards_after), len(self.guards_before))

  def test_needs_view_decomposition(self):
    """Tests needs_view_decomposition and _is_contiguous_tensor."""
    # Contiguous tensor with storage_offset == 0
    contiguous_tensor = torch.randn(10, 20)
    self.assertEqual(contiguous_tensor.storage_offset(), 0)
    self.assertTrue(view_decomposition._is_contiguous_tensor(contiguous_tensor))

    # Sliced tensor with non-zero storage offset
    sliced_tensor = contiguous_tensor[5:, :]
    self.assertNotEqual(sliced_tensor.storage_offset(), 0)
    self.assertFalse(view_decomposition._is_contiguous_tensor(sliced_tensor))

    # Transposed non-contiguous tensor
    transposed_tensor = contiguous_tensor.transpose(0, 1)
    self.assertFalse(
        view_decomposition._is_contiguous_tensor(transposed_tensor)
    )

    # Empty / None checks
    with self.assertRaises(AssertionError):  # ASSERT_RAISES_OK=Input check.
      view_decomposition._is_contiguous_tensor(None)
    self.assertTrue(view_decomposition._is_contiguous_tensor(torch.randn(0)))

    # FX placeholder nodes
    graph = torch.fx.Graph()
    contig_node = graph.placeholder("x")
    contig_node.meta["val"] = contiguous_tensor
    self.assertFalse(view_decomposition.needs_view_decomposition(contig_node))

    offset_node = graph.placeholder("y")
    offset_node.meta["val"] = sliced_tensor
    self.assertTrue(view_decomposition.needs_view_decomposition(offset_node))

    call_node = graph.call_function(
        torch.ops.aten.add.Tensor, args=(contig_node, contig_node)
    )
    call_node.meta["val"] = sliced_tensor
    self.assertTrue(view_decomposition.needs_view_decomposition(call_node))

    scalar_node = graph.placeholder("s")
    scalar_node.meta["val"] = 42
    self.assertFalse(view_decomposition.needs_view_decomposition(scalar_node))

    none_node = graph.placeholder("none_val")
    self.assertFalse(view_decomposition.needs_view_decomposition(none_node))

    # Direct runtime torch.Tensor tests
    self.assertTrue(view_decomposition.needs_view_decomposition(sliced_tensor))
    self.assertFalse(
        view_decomposition.needs_view_decomposition(contiguous_tensor)
    )
    self.assertFalse(view_decomposition.needs_view_decomposition(42))
    self.assertFalse(view_decomposition.needs_view_decomposition(None))

  def test_validate_dimension_slice(self):
    """Tests _validate_dimension_slice for boundary checks and divisibility validation."""
    # Valid slice within bounds
    self.assertTrue(
        view_decomposition._validate_dimension_slice(
            dim_idx=0,
            size=10,
            stride=1,
            prev_stride=10,
            dim_offset=0,
            base_size=10,
            is_reshaped_dim=False,
        )
    )
    self.assertTrue(
        view_decomposition._validate_dimension_slice(
            dim_idx=1,
            size=5,
            stride=1,
            prev_stride=20,
            dim_offset=10,
            base_size=20,
            is_reshaped_dim=False,
        )
    )

    # Invalid slice: required_end (dim_offset + size) exceeds base_size
    self.assertFalse(
        view_decomposition._validate_dimension_slice(
            dim_idx=1,
            size=20,
            stride=1,
            prev_stride=25,
            dim_offset=10,
            base_size=25,
            is_reshaped_dim=False,
        )
    )
    self.assertFalse(
        view_decomposition._validate_dimension_slice(
            dim_idx=0,
            size=30,
            stride=1,
            prev_stride=25,
            dim_offset=0,
            base_size=25,
            is_reshaped_dim=False,
        )
    )
    # Invalid slice: negative dim_offset
    self.assertFalse(
        view_decomposition._validate_dimension_slice(
            dim_idx=1,
            size=5,
            stride=1,
            prev_stride=20,
            dim_offset=-2,
            base_size=20,
            is_reshaped_dim=False,
        )
    )

    # Reshaped dimension with valid divisibility (512 % 64 == 0)
    self.assertTrue(
        view_decomposition._validate_dimension_slice(
            dim_idx=1,
            size=8,
            stride=64,
            prev_stride=512,
            dim_offset=0,
            base_size=8,
            is_reshaped_dim=True,
        )
    )

    # Reshaped dimension with invalid divisibility (512 % 60 != 0)
    self.assertFalse(
        view_decomposition._validate_dimension_slice(
            dim_idx=1,
            size=8,
            stride=60,
            prev_stride=512,
            dim_offset=0,
            base_size=8,
            is_reshaped_dim=True,
        )
    )

  def test_decompose_unsupported_stride_zero(self):
    """Tests that decompose_into_view_sequence raises NotImplementedError for stride == 0."""
    t = torch.randn(1, 10).expand(4, 10)
    with self.assertRaises(NotImplementedError):  # ASSERT_RAISES_OK=Unit test.
      view_decomposition.decompose_into_view_sequence(t)

  def test_decompose_unsupported_non_unit_step_slice(self):
    """Tests that decompose_into_view_sequence raises NotImplementedError for non-unit step slice."""
    t = torch.randn(20, 20)[:, ::2]
    with self.assertRaises(NotImplementedError):  # ASSERT_RAISES_OK=Unit test.
      view_decomposition.decompose_into_view_sequence(t)

  def test_get_base_tensor_sliced(self):
    """Tests get_base_tensor on sliced view tensor."""
    orig = torch.randn(16, 32, 64)
    view = orig[2:10, 5:25, :]
    base = view_decomposition.get_base_tensor(view)
    self.assertTrue(base.is_contiguous())
    self.assertEqual(list(base.shape), [10, 32, 64])
    utils.assert_close(base[2:10, 5:25, :], view)

  def test_get_base_tensor_transposed(self):
    """Tests get_base_tensor on transposed view tensor."""
    orig = torch.randn(16, 32)
    view = orig.t()
    base = view_decomposition.get_base_tensor(view)
    self.assertTrue(base.is_contiguous())
    self.assertEqual(list(base.shape), [16, 32])
    utils.assert_close(base.t(), view)

  def test_get_base_tensor_contiguous(self):
    """Tests get_base_tensor returns input directly when already contiguous."""
    orig = torch.randn(16, 32)
    base = view_decomposition.get_base_tensor(orig)
    self.assertIs(base, orig)

  def _assert_view_replay(
      self,
      view: torch.Tensor,
      expected_base_shape: list[int],
      expected_view_ops: list[tuple[str, tuple[Any, ...]]],
  ) -> None:
    """Checks decomposition and that replaying it on the runtime base matches."""
    base_shape, view_ops = view_decomposition.decompose_into_view_sequence(view)
    self.assertEqual(base_shape, expected_base_shape)
    self.assertEqual(view_ops, expected_view_ops)

    base = view_decomposition.get_base_tensor(view)
    self.assertEqual(list(base.shape), expected_base_shape)
    self.assertTrue(base.is_contiguous())
    self.assertLessEqual(base.numel(), _storage_numel(view))
    replayed = base
    for op_type, op_args in view_ops:
      if op_type == "slice":
        dim_idx, start, end, step = op_args
        replayed = replayed.narrow(dim_idx, start, end - start)
        self.assertEqual(step, 1)
      else:
        replayed = replayed.permute(*op_args[0])
    self.assertEqual(replayed.shape, view.shape)
    self.assertEqual(replayed.data_ptr(), view.data_ptr())
    # Strides of size-1 dims are layout-irrelevant and may differ.
    for size, replayed_st, view_st in zip(
        view.shape, replayed.stride(), view.stride()
    ):
      if size != 1:
        self.assertEqual(replayed_st, view_st)
    utils.assert_close(replayed, view)

  def test_unit_dim_arbitrary_stride_is_contiguous(self):
    """Tests that strides of size-1 dims are ignored (c10 contiguity)."""
    # torch.zeros_like() copies the stride of a size-1 batch dim verbatim, e.g.
    # HF linear-attention recurrent states: storage holds 8 * 4 * 64 elements.
    shape, stride = (1, 8, 4, 64), (4096, 256, 64, 1)
    state = torch.zeros_like(torch.empty_strided(shape, stride))
    self.assertEqual(state.stride(), stride)
    self.assertTrue(state.is_contiguous())

    self.assertTrue(view_decomposition._is_contiguous_tensor(state))
    self.assertFalse(view_decomposition.needs_view_decomposition(state))
    self.assertIsNone(view_decomposition.decompose_into_view_sequence(state))
    self.assertIs(view_decomposition.get_base_tensor(state), state)

    # Stride-0 size-1 dims (expand to size 1) are not broadcasts either.
    self.assertTrue(
        view_decomposition._is_contiguous_tensor(torch.randn(8).expand(1, 8))
    )
    # Empty tensors are contiguous regardless of strides.
    self.assertTrue(
        view_decomposition._is_contiguous_tensor(
            torch.empty_strided((0, 4), (1, 0))
        )
    )

  def test_unit_dim_arbitrary_stride_transposed(self):
    """Tests a transposed view of a state whose size-1 dim has a bogus stride."""
    shape, stride = (1, 8, 4, 64), (4096, 256, 64, 1)
    state = torch.empty_strided(shape, stride).copy_(torch.randn(shape))
    view = state.transpose(1, 2)  # [1, 4, 8, 64], strides [4096, 64, 256, 1]

    # Sorting by stride would imply a [1, 16, 4, 64] base, twice the storage, so
    # the size-1 dim is ordered innermost; base covers exactly the 2048 elements.
    self._assert_view_replay(
        view,
        expected_base_shape=[8, 4, 64, 1],
        expected_view_ops=[("permute", ([3, 1, 0, 2],))],
    )

  def test_unit_dim_sliced_views(self):
    """Tests genuine views whose size-1 dims come from slicing or select."""
    orig = torch.randn(4, 8)

    # Column slice: size-1 inner dim.
    self._assert_view_replay(
        orig[:, 3:4],
        expected_base_shape=[4, 8],
        expected_view_ops=[("slice", (1, 3, 4, 1))],
    )
    # Row slice: size-1 outer dim keeps its slot.
    self._assert_view_replay(
        orig[2:3, 1:5],
        expected_base_shape=[3, 8],
        expected_view_ops=[("slice", (0, 2, 3, 1)), ("slice", (1, 1, 5, 1))],
    )

  def test_decompose_dynamic_unit_dim_transpose(self):
    """Tests a dynamic transposed view with a size-1 dim adds no guards."""
    base_dyn = torch.randn(1, 20, 8, 64)
    torch._dynamo.mark_dynamic(base_dyn, 1, min=10, max=100)

    view_node, node_decomp = self._capture_view_node(
        base_dyn, lambda x: x.transpose(1, 2)
    )
    self._print_decomposition(view_node, node_decomp)
    self._print_guards()

    self.assertIsNotNone(node_decomp)
    node_base_shape, node_view_ops = node_decomp
    # The stride-sorted base fits the storage, so the size-1 dim keeps its slot.
    self.assertIsInstance(node_base_shape[1], torch.SymInt)
    self._assert_base_shapes_match(node_base_shape, [1, 20, 8, 64])
    self.assertEqual(node_view_ops, [("permute", ([0, 2, 1, 3],))])
    self.assertEqual(self.guards_after, self.guards_before)

  def test_decompose_dynamic_unit_dim_slices_keep_layout(self):
    """Tests size-1 slices of dynamic tensors keep the base dims, one slice."""
    # (base shape, dynamic dim, view fn, slice op on the static base).
    cases = [
        ((1, 20, 64), 1, lambda x: x[:, -1:, :], (1, 19, 20, 1)),
        ((2, 20, 64), 1, lambda x: x[:, -1:, :], (1, 19, 20, 1)),
        ((1, 20, 64), 1, lambda x: x[:, 3:4, :], (1, 3, 4, 1)),
        ((1, 20), 1, lambda x: x[:, -1:], (1, 19, 20, 1)),
        ((2, 4, 20, 8), 2, lambda x: x[:, :, -1:, :], (2, 19, 20, 1)),
        ((1, 4, 20, 8), 2, lambda x: x[:, :, :1, :], (2, 0, 1, 1)),
    ]
    for base_shape, dyn_dim, view_fn, slice_op in cases:
      with self.subTest(base_shape=base_shape, slice_op=slice_op):
        self._assert_view_replay(
            view_fn(torch.randn(base_shape)),
            expected_base_shape=list(base_shape),
            expected_view_ops=[("slice", slice_op)],
        )

        base_dyn = torch.randn(base_shape)
        torch._dynamo.mark_dynamic(base_dyn, dyn_dim, min=10, max=100)
        view_node, node_decomp = self._capture_view_node(base_dyn, view_fn)
        self._print_decomposition(view_node, node_decomp)
        self._print_guards()

        self.assertIsNotNone(node_decomp)
        node_base_shape, node_view_ops = node_decomp
        self._assert_base_shapes_match(node_base_shape, list(base_shape))
        self.assertIsInstance(node_base_shape[dyn_dim], torch.SymInt)
        self.assertLen(node_view_ops, 1)
        self.assertEqual(node_view_ops[0][0], "slice")
        self.assertEqual(node_view_ops[0][1][0], slice_op[0])
        self.assertEqual(self.guards_after, self.guards_before)

  def test_zero_dim_view_with_storage_offset_is_not_decomposed(self):
    """Tests that 0-d views with a storage offset are passed through as is."""
    cases = [
        torch.tensor([0, 1]).select(0, 1),
        torch.arange(6.0).view(2, 3)[1, 2],
    ]
    for view in cases:
      with self.subTest(view=view):
        self.assertEqual(view.shape, ())
        self.assertNotEqual(view.storage_offset(), 0)
        self.assertIsNone(view_decomposition.decompose_into_view_sequence(view))
        self.assertIs(view_decomposition.get_base_tensor(view), view)

  def test_delinearize_storage_offset_zero_dims(self):
    """Tests that a non-zero offset over zero dims raises, and 0 returns []."""
    with self.assertRaises(  # ASSERT_RAISES_OK=internal signal.
        NotImplementedError
    ):
      view_decomposition._delinearize_storage_offset(1, [])
    self.assertEqual(view_decomposition._delinearize_storage_offset(0, []), [])

  def test_zero_dim_tensor_without_offset_is_contiguous(self):
    """Tests that a 0-d tensor at offset 0 goes through the contiguous path."""
    scalar = torch.arange(3.0)[0]
    self.assertIsNone(view_decomposition.decompose_into_view_sequence(scalar))
    self.assertIs(view_decomposition.get_base_tensor(scalar), scalar)

  def test_decompose_base_exceeding_storage_raises(self):
    """Tests that views whose inferred base exceeds their storage raise."""
    cases = [
        # [::2, None] on a 9-element buffer infers a [5, 2] base.
        torch.arange(9.0)[::2, None],
        # The reporter's original counterexample: diagonal of a reshaped
        # tensor, whose stride-sorted and unit-free bases both exceed
        # the 4-element storage.
        torch.arange(4).reshape(2, 2, 1).diagonal(0, 0, 1),
    ]
    for view in cases:
      with self.subTest(view=view):
        with self.assertRaises(  # ASSERT_RAISES_OK=internal signal.
            NotImplementedError
        ):
          view_decomposition.decompose_into_view_sequence(view)
        with self.assertRaises(  # ASSERT_RAISES_OK=internal signal.
            NotImplementedError
        ):
          view_decomposition.get_base_tensor(view)

  def test_decompose_strided_slice_trailing_unit_dim_fits_storage(self):
    """Tests that the same view decomposes when the base fits the storage."""
    # Same shape and strides as the 9-element case, one more element of
    # storage: the [5, 2] base fits and the decomposition still works.
    self._assert_view_replay(
        torch.arange(10.0)[::2, None],
        expected_base_shape=[5, 2],
        expected_view_ops=[("slice", (1, 0, 1, 1))],
    )


def _storage_numel(tensor: torch.Tensor) -> int:
  return tensor.untyped_storage().nbytes() // tensor.element_size()


if __name__ == "__main__":
  absltest.main()
