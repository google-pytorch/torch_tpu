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

"""TPU Meta kernel for `aten::_grouped_mm`.

Upstream's meta requires 16-byte aligned strides and int32 `offs`, which TPU
doesn't. Only TPU tensors are affected; other devices keep upstream's meta.
Checks mirror `ValidateGroupedMmInputs` / `GetGroupedMmOutputSize` in
csrc/ops/grouped_mm/grouped_mm_aten_kernels.cc so eager and compile agree.
"""

import torch
from torch import _meta_registrations
from torch_tpu._internal.utils import tt_warnings

_meta_lib: torch.library.Library | None = None


def _check(cond: bool | torch.SymBool, msg: str) -> None:
  # `torch._check` defers unbacked (data-dependent) conditions to runtime.
  torch._check(cond, lambda: f"grouped_mm(): {msg}")  # pylint: disable=protected-access


def _name(dtype: torch.dtype) -> str:
  return str(dtype).removeprefix("torch.")


def _tpu_grouped_mm_meta(
    self: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
  """Validates inputs like the TPU eager kernel; returns a contiguous output."""
  dtype = self.dtype
  # Eager raises NotImplementedError for bias; match it.
  torch._check_not_implemented(  # pylint: disable=protected-access
      bias is None,
      lambda: "grouped_mm(): expected bias to be undefined, got defined tensor",
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
  _check(
      self.dim() in (2, 3), f"expected self to be 2D or 3D, got {self.dim()}D"
  )
  _check(
      mat2.dim() in (2, 3), f"expected mat2 to be 2D or 3D, got {mat2.dim()}D"
  )

  a_is_2d, b_is_2d = self.dim() == 2, mat2.dim() == 2
  if a_is_2d or b_is_2d:
    _check(
        offs is not None,
        "offsets must be provided when at least one of self or mat2 is a 2D"
        " tensor",
    )
  else:
    _check(
        offs is None,
        "offsets must not be provided when both self and mat2 are 3D tensors",
    )
  num_offs = offs.size(0) if offs is not None else 0  # Presence checked above.

  if a_is_2d and b_is_2d:
    out_size = [num_offs, self.size(0), mat2.size(1)]
  elif a_is_2d:
    _check(
        num_offs == mat2.size(0),
        "expected offs batch size to match mat2 batch size, got"
        f" {num_offs} and {mat2.size(0)}",
    )
    out_size = [self.size(0), mat2.size(-1)]
  elif b_is_2d:
    _check(
        num_offs == self.size(0),
        "expected offs batch size to match self batch size, got"
        f" {num_offs} and {self.size(0)}",
    )
    out_size = [self.size(1), mat2.size(1)]
  else:
    _check(
        self.size(0) == mat2.size(0),
        "expected self batch size to match mat2 batch size, got"
        f" {self.size(0)} and {mat2.size(0)}",
    )
    out_size = [self.size(0), self.size(1), mat2.size(-1)]
  _check(
      self.size(-1) == mat2.size(-2),
      "expected the contraction dimension of self and mat2 to match, got"
      f" {self.size(-1)} and {mat2.size(-2)}",
  )
  return self.new_empty(out_size, dtype=out_dtype)


def register() -> None:
  """Replaces the `aten._grouped_mm.default` Meta kernel for TPU tensors."""
  global _meta_lib
  if _meta_lib is not None:
    return
  op = torch.ops.aten._grouped_mm.default  # pylint: disable=protected-access
  meta_key = torch._C.DispatchKey.Meta  # pylint: disable=protected-access
  upstream_meta = op.py_kernels.pop(meta_key, None)
  if upstream_meta is None:
    # Don't break `import torch` on torch versions without this meta; the
    # grouped_mm meta tests catch the drift.
    tt_warnings.warn(
        f"torch {torch.__version__} has no Python meta for {op}; TPU"
        " grouped_mm meta override not installed."
    )
    return

  def _grouped_mm_meta(
      self: torch.Tensor,
      mat2: torch.Tensor,
      offs: torch.Tensor | None = None,
      bias: torch.Tensor | None = None,
      out_dtype: torch.dtype | None = None,
  ) -> torch.Tensor:
    # FakeTensors report `meta` as their device here; `device_hint` returns
    # the fake device.
    if _meta_registrations.device_hint(self) != "tpu":
      return upstream_meta(self, mat2, offs, bias, out_dtype)
    return _tpu_grouped_mm_meta(self, mat2, offs, bias, out_dtype)

  # Upstream registers its meta both as a dispatcher Meta kernel and as a
  # Python-dispatcher kernel (see `activate_meta()`); replace both. Older torch
  # defaults `allow_override` to False.
  _meta_lib = torch.library.Library("aten", "IMPL", "Meta")
  _meta_lib.impl(op, _grouped_mm_meta, allow_override=True)
  op.py_impl(meta_key)(_grouped_mm_meta)
