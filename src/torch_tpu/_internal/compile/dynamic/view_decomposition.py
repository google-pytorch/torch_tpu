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

"""View decomposition for dynamic placeholders in PyTorch FX."""

from __future__ import annotations

from collections.abc import Sequence
import functools
from typing import Any

from absl import logging
import sympy
import torch


def _fmt_sym(val: Any) -> str:
  """Formats a numeric value for debug logging."""
  if isinstance(val, torch.SymInt):
    expr = getattr(val.node, "expr", str(val)) if hasattr(val, "node") else val
    return f"SymInt({expr})"
  return f"{val}"


def _to_expr(val: Any) -> Any:
  """Extracts or converts a PyTorch value to a SymPy expression.

  Bridges PyTorch into the SymPy algebra engine:
  - For torch.SymInt, extracts the underlying SymPy AST node (val.node.expr).
  - For concrete integers, converts to sympy.Integer.
  - Passes through existing SymPy objects (sympy.Basic).

  Args:
    val: A torch.SymInt, Python int, or sympy.Basic expression.

  Returns:
    A SymPy expression suitable for algebraic manipulation (e.g. sympy.div).
  """
  if isinstance(val, torch.SymInt):
    return val.node.expr
  if isinstance(val, sympy.Basic):
    return val
  return sympy.Integer(int(val))


def _to_sym(expr: Any, shape_env: Any) -> Any:
  """Converts a SymPy expression back to a PyTorch representation (int or SymInt).

  Bridges the SymPy algebra engine back to PyTorch:
  - Constant integer expressions (sympy.Integer) are converted to native Python
    ints to avoid unnecessary symbolic node overhead in the FX graph.
  - Dynamic symbolic expressions are wrapped into a torch.SymInt via the
    provided ShapeEnv so downstream ATen operations (e.g. aten.slice) can accept
    them as valid tensor dimension/offset arguments.

  Args:
    expr: SymPy expression (e.g. sympy.Integer, sympy.Symbol, or composite AST).
    shape_env: Optional ShapeEnv used to construct torch.SymInt nodes.

  Returns:
    A Python int, torch.SymInt, or raw expression if shape_env is None.
  """
  if hasattr(expr, "is_Integer") and expr.is_Integer:
    return int(expr)
  if shape_env is not None:
    return shape_env.create_symintnode(expr, hint=None)  # pytype: disable=bad-argument-type
  return expr


def _div_expr(curr_expr: Any, st_expr: Any) -> tuple[Any, Any]:
  """Performs algebraic polynomial division on SymPy expressions.

  Raises:
    NotImplementedError: If curr_expr cannot be algebraically divided by
    st_expr.
  """
  gens = list(curr_expr.free_symbols | st_expr.free_symbols)
  if not gens:
    return curr_expr // st_expr, curr_expr % st_expr

  try:
    res = sympy.div(curr_expr, st_expr, *gens)
    return res
  except Exception as e:
    raise NotImplementedError(
        "Unsupported view decomposition: cannot algebraically divide storage"
        f" offset {_fmt_sym(curr_expr)} by stride {_fmt_sym(st_expr)}: {e}"
    ) from e


def _is_contiguous_tensor(val: Any) -> bool:
  """Checks if a tensor or FakeTensor metadata represents a standard contiguous layout."""
  assert (
      val is not None and hasattr(val, "shape") and hasattr(val, "stride")
  ), f"Expected tensor or FakeTensor with shape and stride, got {type(val)}"

  shape = list(val.shape)
  strides = list(val.stride())
  storage_offset = val.storage_offset()

  # Non-zero storage offset is not standard contiguous
  if storage_offset != 0:
    return False

  num_dims = len(shape)
  if num_dims == 0:
    return True

  # Check if standard contiguous row-major strides match current strides
  expected_st = 1
  for d, st in zip(reversed(shape), reversed(strides)):
    if st != expected_st:
      return False
    expected_st = expected_st * d

  return True


def needs_view_decomposition(
    node_or_tensor: torch.fx.Node | torch.Tensor,
) -> bool:
  """Checks if an FX node or runtime tensor requires view decomposition."""
  if isinstance(node_or_tensor, torch.fx.Node):
    val = node_or_tensor.meta.get("val")
  else:
    val = node_or_tensor

  if not isinstance(val, torch.Tensor) or not hasattr(val, "stride"):
    return False
  return not _is_contiguous_tensor(val)


def _build_tail(
    shape: Sequence[Any], strides: Sequence[Any]
) -> tuple[list[Any], list[Any], list[int] | None]:
  """Sorts dimensions into descending physical stride order and computes permutation.

  Tensors may have dimensions permuted (e.g. via transpose/permute). This
  function sorts dimensions by descending stride to recover the physical memory
  layout (row-major order). If permuted, it returns the permutation list mapping
  physical dimension order back to logical order.

  Args:
    shape: Sequence of dimension sizes in logical order.
    strides: Sequence of dimension strides in logical order.

  Returns:
    A tuple of:
      - phys_shape: Dimension sizes sorted in descending physical stride order.
      - phys_strides: Strides sorted in descending physical stride order.
      - perm_list: Permutation index list to restore logical order, or None if
        already in descending physical order.
  """
  num_dims = len(shape)

  def _cmp_strides(a_idx: int, b_idx: int) -> int:
    st_a = strides[a_idx]
    st_b = strides[b_idx]
    if st_a > st_b:
      return -1
    elif st_a < st_b:
      return 1
    # Relies on sort stability to preserve original dimension order on ties.
    return 0

  sorted_dims = sorted(range(num_dims), key=functools.cmp_to_key(_cmp_strides))
  is_permuted = sorted_dims != list(range(num_dims))

  if is_permuted:
    phys_shape = [shape[i] for i in sorted_dims]
    phys_strides = [strides[i] for i in sorted_dims]
    perm_list = [sorted_dims.index(i) for i in range(num_dims)]
  else:
    phys_shape = list(shape)
    phys_strides = list(strides)
    perm_list = None

  logging.debug(
      "_build_tail: shape=%s, strides=%s -> phys_shape=%s, phys_strides=%s,"
      " perm_list=%s",
      shape,
      strides,
      phys_shape,
      phys_strides,
      perm_list,
  )
  return phys_shape, phys_strides, perm_list


def _delinearize_storage_offset(
    storage_offset: Any, base_strides: Sequence[Any]
) -> list[Any]:
  """Converts a 1D linear storage offset into per-dimension coordinate offsets.

  Given a linear storage offset into a contiguous base buffer and its row-major
  strides (where storage_offset = sum(dim_offsets[i] * base_strides[i])), this
  decomposes the scalar offset into multi-dimensional index offsets [d_0, d_1,
  ..., d_{n-1}] corresponding to the slice start positions along each dimension.

  Example:
    base_strides = (9, 3, 1)
    storage_offset = 10
    dim_offsets = [1, 0, 1]  # 1 * 9 + 0 * 3 + 1 * 1 == 10

    base_strides imply a contiguous base shape of (m, 3, 3) for m > 1.
    Slicing the contiguous base buffer with [1:, 0:, 1:] yields a tensor with
    shape (m - 1, 3, 2) and the desired storage offset of 10.

  Args:
    storage_offset: 1D linear integer or SymInt offset into the storage buffer.
    base_strides: Sequence of contiguous base strides in descending physical
      order.

  Returns:
    A list of coordinate offsets for each dimension in physical order.
  """
  num_dims = len(base_strides)
  if storage_offset == 0:
    return [0] * num_dims

  # Fast path when all values are concrete integers
  if isinstance(storage_offset, int) and all(
      isinstance(s, int) for s in base_strides
  ):
    curr = storage_offset
    dim_offsets = [0] * num_dims
    for i in range(num_dims - 1):
      dim_offsets[i] = curr // base_strides[i]
      curr %= base_strides[i]
    dim_offsets[-1] = curr
    return dim_offsets

  shape_env = None
  for item in (storage_offset, *base_strides):
    if isinstance(item, torch.SymInt):
      shape_env = item.node.shape_env
      break

  curr_sym = storage_offset
  curr_expr = _to_expr(storage_offset)
  dim_offsets: list[Any] = [0] * num_dims

  for i in range(num_dims - 1):
    st = base_strides[i]
    if curr_sym == 0:
      dim_offsets[i] = 0
      continue

    # If the remaining offset is strictly less than this stride, offset along this dim is 0
    if st > 0 and curr_sym < st:
      dim_offsets[i] = 0
      continue

    q_expr, r_expr = _div_expr(curr_expr, _to_expr(st))
    curr_expr = r_expr
    dim_offsets[i] = _to_sym(q_expr, shape_env)
    curr_sym = _to_sym(r_expr, shape_env)

  dim_offsets[-1] = _to_sym(curr_expr, shape_env)
  return dim_offsets


def _compute_base_strides_and_shapes(
    phys_shape: Sequence[Any],
    phys_strides: Sequence[Any],
) -> tuple[list[Any], list[Any]]:
  """Computes contiguous base strides and shapes assuming slice step=1.

  Traverses dimensions backwards (innermost to outermost) to compute expected
  contiguous row-major strides (base_strides) and infer un-sliced base buffer
  shapes. If any stride does not match the expected contiguous stride, raises
  NotImplementedError (indicating non-unit slice step or unsupported layout).

  Args:
    phys_shape: Sequence of dimension sizes in physical stride order.
    phys_strides: Sequence of dimension strides in physical stride order.

  Returns:
    A tuple of (base_shape, base_strides).

  Raises:
    NotImplementedError: If any stride does not match expected contiguous
      stride (non-unit slice step or unsupported layout).
  """
  num_dims = len(phys_shape)
  base_shape = list(phys_shape)
  base_strides = [1] * num_dims

  # Traverse from innermost (stride 1) to outermost dimension.
  for dim_idx in reversed(range(num_dims)):
    size = phys_shape[dim_idx]
    stride = phys_strides[dim_idx]
    if dim_idx < num_dims - 1:
      base_strides[dim_idx] = (
          base_shape[dim_idx + 1] * base_strides[dim_idx + 1]
      )

    # Verify that the stride matches contiguous base stride (step == 1).
    if base_strides[dim_idx] > 0 and stride != base_strides[dim_idx]:
      raise NotImplementedError(
          f"Unsupported view decomposition: stride {_fmt_sym(stride)} at"
          f" dim_idx={dim_idx} does not match expected contiguous base stride"
          f" {_fmt_sym(base_strides[dim_idx])} (non-unit slice step or"
          " unsupported layout)."
      )

    # Check if this dimension was sliced or reshaped from an outer stride jump.
    if dim_idx > 0:
      prev_stride = phys_strides[dim_idx - 1]
      base_dim_size = (
          prev_stride // base_strides[dim_idx]
          if base_strides[dim_idx] > 0
          else size
      )
      is_reshaped_dim = bool((prev_stride % base_strides[dim_idx]) != 0)
      is_sliced_dim = not is_reshaped_dim and bool(size < base_dim_size)
      if is_reshaped_dim or is_sliced_dim:
        base_shape[dim_idx] = base_dim_size
  return base_shape, base_strides


def _validate_dimension_slice(
    dim_idx: int,
    size: Any,
    stride: Any,
    prev_stride: Any,
    dim_offset: Any,
    base_size: Any,
    is_reshaped_dim: bool,
) -> bool:
  """Validates that slice fits within base bounds and enforces divisibility guards.

  Performs three validation checks:
  1. Lower bound check: Ensures dim_offset >= 0 (slice start does not precede
  0).
  2. Upper bound check: Ensures the sliced window (dim_offset + size) does not
     exceed the reconstructed base dimension size.
  3. Divisibility check: For reshaped dimensions, ensures prev_stride is evenly
     divisible by stride.

  Args:
    dim_idx: Index of the dimension being validated.
    size: Sliced size of the dimension.
    stride: Physical stride of the dimension.
    prev_stride: Stride of the preceding outer dimension (dim_idx - 1).
    dim_offset: Starting coordinate offset along this dimension.
    base_size: Reconstructed base buffer size for this dimension.
    is_reshaped_dim: Whether this dimension was created by a reshape.

  Returns:
    True if the dimension slice is valid, False otherwise.
  """
  # Check that the slice start offset does not precede the start of the buffer.
  if dim_offset < 0:
    logging.debug(
        "_build_body [dim_idx=%d]: return False (negative slice offset: %s <"
        " 0)",
        dim_idx,
        _fmt_sym(dim_offset),
    )
    return False

  required_end = dim_offset + size
  # Check that the sliced window does not exceed the base buffer dimension.
  if base_size < required_end:
    logging.debug(
        "_build_body [dim_idx=%d]: return False (overlapping view: base_size"
        " %s < required end %s)",
        dim_idx,
        _fmt_sym(base_size),
        _fmt_sym(required_end),
    )
    return False

  # Enforce divisibility guard for reshaped dimensions.
  if dim_idx > 0 and is_reshaped_dim:
    if (prev_stride % stride) != 0:
      logging.debug(
          "_build_body [dim_idx=%d]: return False (divisibility failed %s %%"
          " %s != 0)",
          dim_idx,
          _fmt_sym(prev_stride),
          _fmt_sym(stride),
      )
      return False

  return True


def _assemble_slice_ops(
    phys_shape: Sequence[Any],
    phys_strides: Sequence[Any],
    dim_offsets: Sequence[Any],
    base_shape: list[Any],
    base_strides: Sequence[Any],
    compute_slice_ops: bool = True,
) -> list[tuple[int, Any, Any, Any]]:
  """Generates canonical slice operations after verifying dimension bounds and divisibility.

  Iterates backwards through dimensions to identify sliced or offset dimensions,
  updates the contiguous base shape accordingly, validates the slice bounds and
  divisibility constraints, and collects slice tuples (dim_idx, start, end,
  step).

  Args:
    phys_shape: Sequence of dimension sizes in descending physical stride order.
    phys_strides: Sequence of strides in descending physical stride order.
    dim_offsets: List of per-dimension start offsets from storage_offset.
    base_shape: Pre-initialized base shape list (modified in place).
    base_strides: Sequence of expected contiguous base strides.
    compute_slice_ops: Whether to construct and return the slice_ops list.

  Returns:
    List of slice operation tuples (dim_idx, start, end, step) in forward
    dimension order.

  Raises:
    NotImplementedError: If slice bounds exceed base buffer or divisibility
    check
      fails.
  """
  num_dims = len(phys_shape)
  slice_ops: list[tuple[int, Any, Any, Any]] = []

  # Iterate backwards (innermost to outermost) to determine slice bounds.
  for dim_idx in reversed(range(num_dims)):
    size = phys_shape[dim_idx]
    stride = phys_strides[dim_idx]
    dim_offset = dim_offsets[dim_idx]
    has_offset = bool(dim_offset != 0)
    is_reshaped_dim = False
    is_sliced_dim = False

    logging.debug(
        "_build_body [dim_idx=%d]: size=%s, stride=%s, dim_offset=%s,"
        " base_strides[%d]=%s",
        dim_idx,
        _fmt_sym(size),
        _fmt_sym(stride),
        _fmt_sym(dim_offset),
        dim_idx,
        _fmt_sym(base_strides[dim_idx]),
    )

    if dim_idx > 0:
      prev_stride = phys_strides[dim_idx - 1]
      base_dim_size = (
          prev_stride // base_strides[dim_idx]
          if base_strides[dim_idx] > 0
          else size
      )
      is_reshaped_dim = bool((prev_stride % base_strides[dim_idx]) != 0)
      is_sliced_dim = not is_reshaped_dim and bool(size < base_dim_size)
      logging.debug(
          "_build_body [dim_idx=%d]: prev_stride=%s, base_dim_size=%s ->"
          " is_reshaped_dim=%s, is_sliced_dim=%s",
          dim_idx,
          _fmt_sym(prev_stride),
          _fmt_sym(base_dim_size),
          is_reshaped_dim,
          is_sliced_dim,
      )
      if has_offset or is_reshaped_dim or is_sliced_dim:
        base_shape[dim_idx] = base_dim_size
        logging.debug(
            "_build_body [dim_idx=%d]: updated base_shape[%d]=%s",
            dim_idx,
            dim_idx,
            _fmt_sym(base_shape[dim_idx]),
        )
        # Validate slice window bounds and divisibility guards.
        if not _validate_dimension_slice(
            dim_idx,
            size,
            stride,
            prev_stride,
            dim_offset,
            base_shape[dim_idx],
            is_reshaped_dim,
        ):
          raise NotImplementedError(
              f"Unsupported view decomposition at dim_idx={dim_idx}: slice"
              f" bounds [offset={_fmt_sym(dim_offset)},"
              f" end={_fmt_sym(dim_offset + size)}] exceed base dimension size"
              f" {_fmt_sym(base_shape[dim_idx])} or indivisible stride jump"
              f" ({_fmt_sym(prev_stride)} % {_fmt_sym(stride)} != 0)."
          )
    elif has_offset:
      if dim_offset < 0:
        raise NotImplementedError(
            "Unsupported view decomposition at dim_idx=0: negative offset"
            f" {_fmt_sym(dim_offset)}."
        )
      base_shape[0] = dim_offset + size
      logging.debug(
          "_build_body [dim_idx=0]: updated base_shape[0]=%s",
          _fmt_sym(base_shape[0]),
      )

    # Append slice op if dimension has non-zero offset or is sliced/reshaped.
    if compute_slice_ops and (
        has_offset or (dim_idx > 0 and (is_reshaped_dim or is_sliced_dim))
    ):
      logging.debug(
          "_build_body [dim_idx=%d]: appended slice_op=(%d, start=%s, end=%s,"
          " step=1)",
          dim_idx,
          dim_idx,
          _fmt_sym(dim_offset),
          _fmt_sym(dim_offset + size),
      )
      slice_ops.append((dim_idx, dim_offset, dim_offset + size, 1))

  if compute_slice_ops:
    # Reverse so that slice operations apply in forward dimension order (0..n-1).
    slice_ops.reverse()
  return slice_ops


def _build_body(
    phys_shape: Sequence[Any],
    phys_strides: Sequence[Any],
    storage_offset: Any,
    compute_slice_ops: bool = True,
) -> tuple[list[Any], list[tuple[int, Any, Any, Any]]]:
  """Computes per-dimension contiguous base shape and slice operations."""
  num_dims = len(phys_shape)
  for dim_idx in range(1, num_dims):
    if phys_strides[dim_idx] == 0:
      raise NotImplementedError(
          f"Unsupported view decomposition: stride == 0 at dim_idx={dim_idx}"
          " (broadcasting is not supported)."
      )

  base_shape, base_strides = _compute_base_strides_and_shapes(
      phys_shape, phys_strides
  )
  dim_offsets = _delinearize_storage_offset(storage_offset, base_strides)
  slice_ops = _assemble_slice_ops(
      phys_shape,
      phys_strides,
      dim_offsets,
      base_shape,
      base_strides,
      compute_slice_ops=compute_slice_ops,
  )
  return base_shape, slice_ops


def _assemble_canonical_view_ops(
    slice_ops: Sequence[tuple[int, Any, Any, Any]],
    perm_list: Sequence[int] | None,
) -> list[tuple[str, tuple[Any, ...]]]:
  """Assembles canonical FX view operations (slice, permute)."""
  view_ops: list[tuple[str, tuple[Any, ...]]] = []
  for dim_idx, start, end, step in slice_ops:
    view_ops.append(("slice", (dim_idx, start, end, step)))

  if perm_list is not None:
    view_ops.append(("permute", (perm_list,)))
  return view_ops


def decompose_into_view_sequence(
    node_or_val: torch.fx.Node | torch.Tensor,
) -> tuple[list[Any], list[tuple[str, tuple[Any, ...]]]] | None:
  """Decomposes a non-contiguous view tensor or placeholder into:

  1. `base_shape`: The contiguous dynamic base buffer shape.
  2. `view_ops`: Sequence of canonical FX view operation specs:
     - ("permute", (perm_list,))
     - ("slice", (dim_idx, start, end, step))

  Returns None if layout is already contiguous or ambiguous.
  """
  if isinstance(node_or_val, torch.fx.Node):
    val = node_or_val.meta.get("val")
  else:
    val = node_or_val

  assert (
      val is not None and hasattr(val, "shape") and hasattr(val, "stride")
  ), f"Expected tensor or tensor-typed fx.Node, got {type(node_or_val)}"

  if _is_contiguous_tensor(val):
    return None

  shape = list(val.shape)
  strides = list(val.stride())
  storage_offset = val.storage_offset()

  logging.debug(
      "decompose_into_view_sequence: shape=[%s], strides=[%s],"
      " storage_offset=%s",
      ", ".join(_fmt_sym(d) for d in shape),
      ", ".join(_fmt_sym(s) for s in strides),
      _fmt_sym(storage_offset),
  )

  phys_shape, phys_strides, perm_list = _build_tail(shape, strides)
  base_shape, slice_ops = _build_body(phys_shape, phys_strides, storage_offset)
  view_ops = _assemble_canonical_view_ops(slice_ops, perm_list)

  logging.debug(
      "decompose_into_view_sequence: SUCCESS base_shape=[%s], view_ops=%s",
      ", ".join(_fmt_sym(d) for d in base_shape),
      view_ops,
  )
  return base_shape, view_ops


def get_base_tensor(tensor: torch.Tensor) -> torch.Tensor:
  """Reconstructs the contiguous base tensor view for a runtime view tensor."""
  assert isinstance(
      tensor, torch.Tensor
  ), f"Expected torch.Tensor, got {type(tensor)}"
  if _is_contiguous_tensor(tensor):
    return tensor

  shape = list(tensor.shape)
  strides = list(tensor.stride())
  storage_offset = tensor.storage_offset()

  phys_shape, phys_strides, _ = _build_tail(shape, strides)
  base_shape, _ = _build_body(
      phys_shape, phys_strides, storage_offset, compute_slice_ops=False
  )
  base_strides = [1] * len(base_shape)
  for i in reversed(range(len(base_shape) - 1)):
    base_strides[i] = base_strides[i + 1] * base_shape[i + 1]
  return torch.as_strided(
      tensor,
      size=tuple(base_shape),
      stride=tuple(base_strides),
      storage_offset=0,
  )
