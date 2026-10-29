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

"""Eager registration for higher_order.scan and associative_scan."""

import types
from typing import Any
import torch
import torch._C
from torch._higher_order_ops import associative_scan as _orig_associative_scan

# Forward declaration for linting; resolved in caller globals at runtime.
_tpu_associative_scan: Any = None


def _delegating_entrypoint(
    combine_fn: Any,
    xs: Any,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> Any:
  # This code runs with upstream's module globals, where
  # `_install_associative_scan_patch` defines `_tpu_associative_scan`.
  return _tpu_associative_scan(
      combine_fn, xs, dim, reverse=reverse, combine_mode=combine_mode
  )


def _lazy_tpu_associative_scan(
    combine_fn: Any,
    xs: Any,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> Any:
  """Lazily imports compile.scan and dispatches associative_scan."""
  # pylint: disable=g-import-not-at-top
  from torch_tpu._internal.compile import scan
  # pylint: enable=g-import-not-at-top
  return scan._tpu_associative_scan(
      combine_fn, xs, dim, reverse=reverse, combine_mode=combine_mode
  )


# An unpatched copy of upstream `associative_scan`, captured before
# `_install_associative_scan_patch` replaces the original's `__code__`.
_SAVED_ORIG_ASSOCIATIVE_SCAN = types.FunctionType(
    getattr(
        _orig_associative_scan,
        "_orig_code",
        _orig_associative_scan.__code__,
    ),
    _orig_associative_scan.__globals__,
    _orig_associative_scan.__name__,
    _orig_associative_scan.__defaults__,
    _orig_associative_scan.__closure__,
)


def _install_associative_scan_patch() -> None:
  """Routes `torch._higher_order_ops.associative_scan` to the TPU fast path.

  This swaps the function's `__code__` in place rather than rebinding module
  attributes. That way references imported before `torch_tpu` also take the
  TPU path, and the function object's identity is unchanged.
  """
  orig_fn = _orig_associative_scan
  if getattr(orig_fn, "_tpu_patched", False):
    return
  setattr(orig_fn, "_orig_code", orig_fn.__code__)
  orig_fn.__globals__["_tpu_associative_scan"] = _lazy_tpu_associative_scan
  orig_fn.__code__ = _delegating_entrypoint.__code__
  orig_fn.__defaults__ = _delegating_entrypoint.__defaults__
  setattr(orig_fn, "_tpu_patched", True)


def _lazy_handle_scan_impl(*args: Any, **kwargs: Any) -> Any:
  """Lazily imports compile.scan and dispatches higher_order.scan."""
  # pylint: disable=g-import-not-at-top
  from torch_tpu._internal.compile import scan
  # pylint: enable=g-import-not-at-top
  return scan._handle_scan_impl(*args, **kwargs)


def _lazy_handle_associative_scan_impl(*args: Any, **kwargs: Any) -> Any:
  """Lazily imports compile.scan and dispatches associative_scan."""
  # pylint: disable=g-import-not-at-top
  from torch_tpu._internal.compile import scan
  # pylint: enable=g-import-not-at-top
  return scan._handle_associative_scan_impl(*args, **kwargs)


def register() -> None:
  """Eagerly registers scan HOPs and patches associative_scan."""
  try:
    for key in (
        torch._C.DispatchKey.XLA,
        torch._C.DispatchKey.PrivateUse1,
        torch._C.DispatchKey.Meta,
    ):
      torch.ops.higher_order.scan.py_impl(key)(_lazy_handle_scan_impl)

    for key in (
        torch._C.DispatchKey.XLA,
        torch._C.DispatchKey.PrivateUse1,
    ):
      torch.ops.higher_order.associative_scan.py_impl(key)(
          _lazy_handle_associative_scan_impl
      )

    _install_associative_scan_patch()
  except (AttributeError, RuntimeError):
    # AttributeError: Safe to ignore if environment lacks XLA/TPU support.
    # RuntimeError: Safe to ignore if the implementation is already registered.
    pass
