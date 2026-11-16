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

"""TPU Meta kernels for `aten::_grouped_mm` and `aten::_scaled_grouped_mm`.

Upstream's meta requires 16-byte aligned strides, column-major `mat2`, and int32
`offs`, which TPU doesn't. Only TPU tensors are affected; other devices keep
upstream's meta. Checks mirror `ValidateGroupedMmInputs`, `ScaledGroupedMm`, and
`GetGroupedMmOutputSize` in csrc/ops/grouped_mm/grouped_mm_aten_kernels.cc so
eager and compile agree.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import _meta_registrations
from torch_tpu._internal import warnings
from torch_tpu._internal.utils import tt_warnings

_meta_lib: torch.library.Library | None = None


def _check(
    cond: bool | torch.SymBool, msg: str, op_name: str = "grouped_mm"
) -> None:
  # `torch._check` defers unbacked (data-dependent) conditions to runtime.
  torch._check(cond, lambda: f"{op_name}(): {msg}")  # pylint: disable=protected-access


def _check_not_implemented(
    cond: bool | torch.SymBool, msg: str, op_name: str = "grouped_mm"
) -> None:
  torch._check_not_implemented(cond, lambda: f"{op_name}(): {msg}")  # pylint: disable=protected-access


def _name(dtype: torch.dtype) -> str:
  return str(dtype).removeprefix("torch.")


def _get_grouped_mm_output_size(
    self: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor | None,
    op_name: str,
) -> Sequence[int | torch.SymInt]:
  """Validates ranks, offs presence, batch sizes, and contraction dim."""
  _check(
      self.dim() in (2, 3),
      f"expected self to be 2D or 3D, got {self.dim()}D",
      op_name,
  )
  _check(
      mat2.dim() in (2, 3),
      f"expected mat2 to be 2D or 3D, got {mat2.dim()}D",
      op_name,
  )

  a_is_2d, b_is_2d = self.dim() == 2, mat2.dim() == 2
  if a_is_2d or b_is_2d:
    _check(
        offs is not None,
        "offsets must be provided when at least one of self or mat2 is a 2D"
        " tensor",
        op_name,
    )
  else:
    _check(
        offs is None,
        "offsets must not be provided when both self and mat2 are 3D tensors",
        op_name,
    )
  num_offs = offs.size(0) if offs is not None else 0  # Presence checked above.

  if a_is_2d and b_is_2d:
    out_size = [num_offs, self.size(0), mat2.size(1)]
  elif a_is_2d:
    _check(
        num_offs == mat2.size(0),
        "expected offs batch size to match mat2 batch size, got"
        f" {num_offs} and {mat2.size(0)}",
        op_name,
    )
    out_size = [self.size(0), mat2.size(-1)]
  elif b_is_2d:
    _check(
        num_offs == self.size(0),
        "expected offs batch size to match self batch size, got"
        f" {num_offs} and {self.size(0)}",
        op_name,
    )
    out_size = [self.size(1), mat2.size(1)]
  else:
    _check(
        self.size(0) == mat2.size(0),
        "expected self batch size to match mat2 batch size, got"
        f" {self.size(0)} and {mat2.size(0)}",
        op_name,
    )
    out_size = [self.size(0), self.size(1), mat2.size(-1)]
  _check(
      self.size(-1) == mat2.size(-2),
      "expected the contraction dimension of self and mat2 to match, got"
      f" {self.size(-1)} and {mat2.size(-2)}",
      op_name,
  )
  return out_size


def _tpu_grouped_mm_meta(
    self: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Validates inputs like the TPU eager kernel; returns a contiguous output."""
  dtype = self.dtype
  # Eager raises NotImplementedError for bias; match it.
  _check_not_implemented(
      bias is None,
      "expected bias to be undefined, got defined tensor",
  )
  _check(
      dtype == mat2.dtype,
      "expected self and mat2 to have the same dtype, got"
      f" {_name(dtype)} and {_name(mat2.dtype)}",
  )
  _check(
      dtype in (torch.float32, torch.float16, torch.bfloat16),
      f"expected self to be float32, float16, or bfloat16, got {_name(dtype)}",
  )
  out_dtype = out_dtype or dtype
  _check(
      out_dtype == dtype,
      f"expected out_dtype to match self dtype {_name(dtype)}, got"
      f" {_name(out_dtype)}",
  )
  if offs is not None:
    _check(offs.dim() == 1, f"expected offs to be 1D, got {offs.dim()}D")
    _check(
        offs.dtype in (torch.int32, torch.int64),
        f"expected offs to be int32 or int64, got {_name(offs.dtype)}",
    )
    _check(
        offs.stride(0) == 1,
        f"expected offs to have stride 1 along dim 0, got {offs.stride(0)}",
    )
  out_size = _get_grouped_mm_output_size(self, mat2, offs, "grouped_mm")
  return self.new_empty(out_size, dtype=out_dtype)


def _is_scalar(t: torch.Tensor) -> bool | torch.SymBool:
  if t.dim() == 0:
    return True
  if t.dim() == 1:
    return t.size(0) == 1
  return False


def _has_shape_1d(
    t: torch.Tensor, d0: int | torch.SymInt
) -> bool | torch.SymBool:
  return (t.size(0) == d0) if t.dim() == 1 else False


def _has_shape_2d(
    t: torch.Tensor, d0: int | torch.SymInt, d1: int | torch.SymInt
) -> bool | torch.SymBool:
  return ((t.size(0) == d0) & (t.size(1) == d1)) if t.dim() == 2 else False


def _is_rowwise_scales(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor | None,
) -> bool | torch.SymBool:
  """Mirrors `IsRowwiseScales` in grouped_mm_aten_kernels.cc without guarding."""
  if self.dim() not in (2, 3) or mat2.dim() not in (2, 3):
    return False
  a_is_2d, b_is_2d = self.dim() == 2, mat2.dim() == 2
  if a_is_2d and not b_is_2d:
    return _has_shape_1d(scale_a, self.size(0)) & _has_shape_2d(
        scale_b, mat2.size(0), mat2.size(2)
    )
  elif not a_is_2d and b_is_2d:
    return _has_shape_2d(scale_a, self.size(0), self.size(1)) & _has_shape_1d(
        scale_b, mat2.size(1)
    )
  elif a_is_2d and b_is_2d:
    if offs is None or offs.dim() != 1:
      return False
    g = offs.size(0)
    return _has_shape_1d(scale_a, g * self.size(0)) & _has_shape_1d(
        scale_b, g * mat2.size(1)
    )
  else:
    return _has_shape_2d(scale_a, self.size(0), self.size(1)) & _has_shape_2d(
        scale_b, mat2.size(0), mat2.size(2)
    )


def _tpu_scaled_grouped_mm_meta(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale_result: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
  """Validates `_scaled_grouped_mm` inputs like the TPU eager kernel."""
  del use_fast_accum
  op_name = "scaled_grouped_mm"
  out_size = _get_grouped_mm_output_size(self, mat2, offs, op_name)
  tensorwise = _is_scalar(scale_a) & _is_scalar(scale_b)
  rowwise = _is_rowwise_scales(self, mat2, scale_a, scale_b, offs)
  _check_not_implemented(
      tensorwise | rowwise,
      "expected scale_a and scale_b to both be scalar/size-1 or match rowwise"
      f" shapes for self {list(self.shape)} and mat2 {list(mat2.shape)}, got"
      f" scale_a {list(scale_a.shape)} and scale_b {list(scale_b.shape)}",
      op_name,
  )
  if bias is not None:
    _check_not_implemented(
        _is_scalar(bias),
        f"expected bias to be 1D or scalar, got {bias.dim()}D",
        op_name,
    )
  if scale_result is not None:
    _check_not_implemented(
        _is_scalar(scale_result),
        f"expected scale_result to be 1D or scalar, got {scale_result.dim()}D",
        op_name,
    )
  target_dtype = out_dtype or self.dtype
  return self.new_empty(out_size, dtype=target_dtype)


def _install_meta_override(
    op: torch._ops.OpOverload,  # pylint: disable=protected-access
    tpu_fn: Callable[..., torch.Tensor],
) -> None:
  """Installs `tpu_fn` as the Meta override for `op` on TPU tensors."""
  assert _meta_lib is not None
  meta_key = torch._C.DispatchKey.Meta  # pylint: disable=protected-access
  upstream_meta = op.py_kernels.pop(meta_key, None)
  if upstream_meta is None:
    # Don't break `import torch` on torch versions without this meta; the
    # grouped_mm meta tests catch the drift.
    tt_warnings.warn(
        f"torch {torch.__version__} has no Python meta for {op}; TPU"
        " grouped_mm meta override not installed.",
        warnings.TORCH_TPU,
    )
    return

  def _meta_wrapper(
      self: torch.Tensor, *args: Any, **kwargs: Any
  ) -> torch.Tensor:
    # FakeTensors report `meta` as their device here; `device_hint` returns
    # the fake device.
    if _meta_registrations.device_hint(self) != "tpu":
      return upstream_meta(self, *args, **kwargs)
    return tpu_fn(self, *args, **kwargs)

  # Upstream registers its meta both as a dispatcher Meta kernel and as a
  # Python-dispatcher kernel (see `activate_meta()`); replace both. Older torch
  # defaults `allow_override` to False.
  _meta_lib.impl(op, _meta_wrapper, allow_override=True)
  op.py_impl(meta_key)(_meta_wrapper)


def register() -> None:
  """Replaces `_grouped_mm` and `_scaled_grouped_mm` Meta kernels for TPU."""
  global _meta_lib
  if _meta_lib is not None:
    return
  _meta_lib = torch.library.Library("aten", "IMPL", "Meta")

  _install_meta_override(
      torch.ops.aten._grouped_mm.default,  # pylint: disable=protected-access
      _tpu_grouped_mm_meta,
  )
  _install_meta_override(
      torch.ops.aten._scaled_grouped_mm.default,  # pylint: disable=protected-access
      _tpu_scaled_grouped_mm_meta,
  )
