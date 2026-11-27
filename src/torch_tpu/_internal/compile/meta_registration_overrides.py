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

"""Overrides for upstream PyTorch Meta kernel registrations on TPU."""

from collections.abc import Callable
import inspect
from typing import Any

import torch
from torch import _decomp
from torch import _meta_registrations
from torch._decomp import decompositions
from torch._subclasses import fake_tensor

_meta_lib: torch.library.Library | None = None

_LOW_PRECISION_FLOATS: frozenset[torch.dtype] = frozenset(
    d
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float4_e2m1fn_x2",
    )
    if (d := getattr(torch, name, None)) is not None
)


def _format_dtype(dtype: torch.dtype) -> str:
  return str(dtype).removeprefix("torch.")


def _effective_device(tensor: torch.Tensor) -> torch.device:
  maybe_get_fake_device = getattr(fake_tensor, "maybe_get_fake_device", None)
  if maybe_get_fake_device is not None:
    fake_device = maybe_get_fake_device(tensor)
    if fake_device is not None:
      return fake_device
  elif isinstance(tensor, fake_tensor.FakeTensor):
    return tensor.fake_device
  return tensor.device


@decompositions.out_wrapper(exact_dtype=True)
def _addmm_dtype_meta(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: torch.dtype,
    beta: Any = 1,
    alpha: Any = 1,
):
  if not self.is_floating_point() and not self.is_complex():
    beta = int(beta)
    alpha = int(alpha)
  out = alpha * torch.mm(mat1, mat2, out_dtype=out_dtype)
  if beta == 0:
    return out
  return out + beta * self.to(out_dtype).expand_as(out)


# Expose the replacement decomposition on `torch._decomp.decompositions` so
# `FakeTensorMode` (`torch_decomp_decompositions`) recognizes it on both
# static and symbolic shape paths.
_addmm_dtype_meta.__module__ = decompositions.__name__
setattr(decompositions, _addmm_dtype_meta.__name__, _addmm_dtype_meta)


# TODO: Remove once minimum supported PyTorch includes
# https://github.com/pytorch/pytorch/pull/179634 and
# https://github.com/pytorch/pytorch/pull/195594 (PyTorch 2.12+).
def _patch_addmm_dtype_meta(meta_lib: torch.library.Library) -> None:
  """Patches `aten.addmm.dtype` and `aten.addmm.dtype_out` Meta/decompositions."""
  # pylint: disable=protected-access
  meta_key = torch._C.DispatchKey.Meta
  op_to_fn: tuple[tuple[Any, Callable[..., Any]], ...] = (
      (torch.ops.aten.addmm.dtype, _addmm_dtype_meta),
      (torch.ops.aten.addmm.dtype_out, _addmm_dtype_meta),
  )
  patched = False
  for op, fn in op_to_fn:
    existing = _decomp.decomposition_table.get(op)
    if (
        existing is not None
        and "out_dtype" in inspect.signature(existing).parameters
    ):
      continue
    _decomp.decomposition_table[op] = fn
    if op in _decomp.pre_autograd_decomposition_table:
      _decomp.pre_autograd_decomposition_table[op] = fn
    op.py_kernels.pop(meta_key, None)
    meta_lib.impl(op, fn, allow_override=True)
    op.py_impl(meta_key)(fn)
    patched = True
  if patched:
    fake_tensor.torch_decomp_decompositions.cache_clear()
  # pylint: enable=protected-access


@decompositions.out_wrapper(exact_dtype=True)
def _bmm_dtype_meta(
    self: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: torch.dtype,
):
  return _meta_registrations.common_meta_baddbmm_bmm(
      self, mat2, True, out_dtype=out_dtype
  )


# TODO: b/576283069 - Remove once minimum supported PyTorch includes
# https://github.com/pytorch/pytorch/pull/175619 (PyTorch 2.12+).
def _patch_bmm_dtype_meta(meta_lib: torch.library.Library) -> None:
  """Patches `aten.bmm.dtype` and `aten.bmm.dtype_out` Meta kernels."""
  # pylint: disable=protected-access
  meta_key = torch._C.DispatchKey.Meta
  for op in (torch.ops.aten.bmm.dtype, torch.ops.aten.bmm.dtype_out):
    existing = op.py_kernels.get(meta_key)
    if (
        existing is not None and "out" in inspect.signature(existing).parameters
    ) or (
        existing is None
        and torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), meta_key)
    ):
      continue
    if op not in _decomp.meta_table and hasattr(
        torch._C, "_fake_dispatch_register_meta"
    ):
      torch._C._fake_dispatch_register_meta(
          op._schema.name, op._schema.overload_name
      )
    _decomp.meta_table[op] = _bmm_dtype_meta
    op.py_kernels.pop(meta_key, None)
    meta_lib.impl(op, _bmm_dtype_meta, allow_override=True)
    op.py_impl(meta_key)(_bmm_dtype_meta)
  # pylint: enable=protected-access


# TODO: Remove once PyTorch includes https://github.com/pytorch/pytorch/pull/194271
# and https://github.com/pytorch/pytorch/pull/198303 (PyTorch 2.15+).
def _patch_sdpa_meta(meta_lib: torch.library.Library) -> None:
  """Patches `aten._scaled_dot_product_fused_attention_overrideable` Meta kernel."""
  # pylint: disable=protected-access
  op = torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
  meta_key = torch._C.DispatchKey.Meta
  upstream_meta = op.py_kernels.pop(meta_key, None)
  if upstream_meta is None:
    return

  def _sdpa_overrideable_meta(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      attn_bias: torch.Tensor | None = None,
      dropout_p: float = 0.0,
      is_causal: bool = False,
      return_debug_mask: bool = False,
      scale: float | None = None,
  ):
    if _meta_registrations.device_hint(query) == "tpu" and query.dim() in (
        3,
        4,
    ):
      s_kv = key.size(-2)
      torch._check(
          s_kv == value.size(-2),
          lambda: (
              f"key sequence length ({s_kv}) must match "
              f"value sequence length ({value.size(-2)})"
          ),
      )
      if attn_bias is not None:
        bias_s_kv = attn_bias.size(-1)
        if bias_s_kv != 1:
          torch._check(
              bias_s_kv == s_kv,
              lambda: (
                  f"attn_bias last dimension must match S_KV ({s_kv}) "
                  f"or be 1 for broadcasting, but got {bias_s_kv}"
              ),
          )
    return upstream_meta(
        query,
        key,
        value,
        attn_bias,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale,
    )

  meta_lib.impl(op, _sdpa_overrideable_meta, allow_override=True)
  op.py_impl(meta_key)(_sdpa_overrideable_meta)
  # pylint: enable=protected-access


# Upstream PyTorch's `_fused_dropout` decomposition does not accept a
# `generator` argument, rejects `complex64`, and on older PyTorch versions
# returns a non-`uint8` mask. Override Meta on TPU to match `AtenFusedDropout`.
def _patch_fused_dropout_meta(meta_lib: torch.library.Library) -> None:
  """Patches `aten._fused_dropout` default and out Meta kernels."""
  # pylint: disable=protected-access
  default_op = torch.ops.aten._fused_dropout.default
  out_op = torch.ops.aten._fused_dropout.out
  meta_key = torch._C.DispatchKey.Meta
  upstream_decomp = _decomp.decomposition_table.get(default_op)
  upstream_meta = default_op.py_kernels.get(meta_key)
  unsupported_float_or_complex = frozenset(
      d
      for name in (
          "float8_e4m3fn",
          "float8_e4m3fnuz",
          "float8_e5m2",
          "float8_e5m2fnuz",
          "float8_e8m0fnu",
          "float4_e2m1fn_x2",
          "complex32",
      )
      if (d := getattr(torch, name, None)) is not None
  )

  @decompositions.out_wrapper("out0", "out1")
  def _fused_dropout_meta(
      self: torch.Tensor,
      p: float,
      generator: torch.Generator | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if _meta_registrations.device_hint(self) == "tpu":
      torch._check(
          p >= 0.0,
          lambda: (
              f"fused_dropout(): expected p to be in the range [0, 1], got {p}"
          ),
      )
      torch._check(
          p <= 1.0,
          lambda: (
              f"fused_dropout(): expected p to be in the range [0, 1], got {p}"
          ),
      )
      dtype_str = str(self.dtype).removeprefix("torch.")
      torch._check(
          (self.is_floating_point() or self.is_complex())
          and self.dtype not in unsupported_float_or_complex,
          lambda: (
              "fused_dropout(): expected input to be floating point or"
              f" complex, got {dtype_str}"
          ),
      )
      return (
          torch.empty_like(self, memory_format=torch.contiguous_format),
          torch.empty_like(
              self, dtype=torch.uint8, memory_format=torch.contiguous_format
          ),
      )
    if upstream_meta is not None:
      return upstream_meta(self, p, generator=generator)
    if upstream_decomp is not None:
      return upstream_decomp(self, p, generator=generator)
    raise NotImplementedError(
        f"No upstream implementation for _fused_dropout on device {self.device}"
    )

  wrapped_fn = _decomp._convert_out_params(_fused_dropout_meta)
  for op in (default_op, out_op):
    _decomp.meta_table[op] = wrapped_fn
    if hasattr(torch._C, "_fake_dispatch_register_meta"):
      schema = op._schema
      torch._C._fake_dispatch_register_meta(schema.name, schema.overload_name)
    op.py_kernels.pop(meta_key, None)
    meta_lib.impl(op, wrapped_fn, allow_override=True)
    op.py_impl(meta_key)(wrapped_fn)
  # pylint: enable=protected-access


# Upstream PyTorch's `torch._refs.native_group_norm` Meta kernel casts `mean`
# and `rstd` to `input.dtype`, crashes on `N == 0` integral inputs and 1D
# inputs, and returns non-contiguous `channels_last` strides, whereas
# `AtenNativeGroupNorm` returns contiguous outputs with `mean`/`rstd` in the
# resolved stat dtype. Override Meta on TPU to match `AtenNativeGroupNorm`.
def _native_group_norm_tpu_meta(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """TPU Meta implementation matching `AtenNativeGroupNorm`."""
  # pylint: disable=protected-access
  for tensor in (input, weight, bias):
    if tensor is not None:
      dtype_str = _format_dtype(tensor.dtype)
      torch._check(
          tensor.dtype not in _LOW_PRECISION_FLOATS,
          lambda dtype_str=dtype_str: (
              f"native_group_norm(): not implemented for {dtype_str}"
          ),
      )
  torch._check(
      N >= 0,
      lambda: f"native_group_norm(): expected batch size to be >= 0, got {N}",
  )
  if not input.is_floating_point():
    input_dtype_str = _format_dtype(input.dtype)
    torch._check(
        N == 0,
        lambda: (
            "native_group_norm(): expected input dtype to be floating point,"
            f" got {input_dtype_str}"
        ),
    )
  torch._check(
      C > 0,
      lambda: (
          "native_group_norm(): expected number of channels to be greater than"
          f" 0, got {C}"
      ),
  )
  torch._check(
      HxW > 0,
      lambda: (
          f"native_group_norm(): expected HxW to be greater than 0, got {HxW}"
      ),
  )
  torch._check(
      group > 0,
      lambda: (
          "native_group_norm(): expected num groups to be greater than 0,"
          f" got {group}"
      ),
  )
  torch._check(
      C % group == 0,
      lambda: (
          "native_group_norm(): expected number of channels to be divisible by"
          f" num_groups ({group}), got {C}"
      ),
  )
  torch._check(
      input.numel() == N * C * HxW,
      lambda: (
          f"native_group_norm(): expected input to have {N * C * HxW} elements,"
          f" got {input.numel()}"
      ),
  )
  if weight is not None:
    torch._check(
        weight.ndim == 1 and weight.numel() == C,
        lambda: (
            "native_group_norm(): expected weight to be a 1D tensor of size"
            f" {C}, got {list(weight.shape)}"
        ),
    )
  if bias is not None:
    torch._check(
        bias.ndim == 1 and bias.numel() == C,
        lambda: (
            f"native_group_norm(): expected bias to be a 1D tensor of size {C},"
            f" got {list(bias.shape)}"
        ),
    )
  has_mixed = (weight is not None and weight.dtype != input.dtype) or (
      bias is not None and bias.dtype != input.dtype
  )
  if has_mixed:
    input_dtype_str = _format_dtype(input.dtype)
    torch._check(
        input.dtype in (torch.float16, torch.bfloat16),
        lambda: (
            "native_group_norm(): expected float16 or bfloat16 input for"
            f" mixed-dtype group_norm, got {input_dtype_str}"
        ),
    )
    if weight is not None:
      weight_dtype_str = _format_dtype(weight.dtype)
      torch._check(
          weight.dtype == torch.float32,
          lambda: (
              "native_group_norm(): expected weight dtype to be float32 for"
              f" mixed-dtype group_norm, got {weight_dtype_str}"
          ),
      )
    if bias is not None:
      bias_dtype_str = _format_dtype(bias.dtype)
      torch._check(
          bias.dtype == torch.float32,
          lambda: (
              "native_group_norm(): expected bias dtype to be float32 for"
              f" mixed-dtype group_norm, got {bias_dtype_str}"
          ),
      )
  stat_dtype = input.dtype
  if weight is not None and weight.dtype != input.dtype:
    stat_dtype = weight.dtype
  elif bias is not None and bias.dtype != input.dtype:
    stat_dtype = bias.dtype
  # pylint: enable=protected-access
  return (
      torch.empty_like(input, memory_format=torch.contiguous_format),
      input.new_empty((N, group), dtype=stat_dtype),
      input.new_empty((N, group), dtype=stat_dtype),
  )


def _patch_native_group_norm_meta(meta_lib: torch.library.Library) -> None:
  """Patches `aten.native_group_norm` default and out Meta kernels."""
  # pylint: disable=protected-access
  default_op = torch.ops.aten.native_group_norm.default
  out_op = torch.ops.aten.native_group_norm.out
  meta_key = torch._C.DispatchKey.Meta
  upstream_decomp = _decomp.decomposition_table.get(default_op)
  upstream_meta = default_op.py_kernels.get(meta_key)
  upstream_out_decomp = _decomp.decomposition_table.get(out_op)
  upstream_out_meta = out_op.py_kernels.get(meta_key)

  def _native_group_norm_meta(
      input: torch.Tensor,  # pylint: disable=redefined-builtin
      weight: torch.Tensor | None,
      bias: torch.Tensor | None,
      N: int,
      C: int,
      HxW: int,
      group: int,
      eps: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _meta_registrations.device_hint(input) == "tpu":
      return _native_group_norm_tpu_meta(input, weight, bias, N, C, HxW, group)
    if upstream_meta is not None:
      return upstream_meta(input, weight, bias, N, C, HxW, group, eps)
    if upstream_decomp is not None:
      return upstream_decomp(input, weight, bias, N, C, HxW, group, eps)
    raise NotImplementedError(
        "No upstream implementation for native_group_norm on device"
        f" {input.device}"
    )

  def _native_group_norm_out_meta(
      input: torch.Tensor,  # pylint: disable=redefined-builtin
      weight: torch.Tensor | None,
      bias: torch.Tensor | None,
      N: int,
      C: int,
      HxW: int,
      group: int,
      eps: float,
      *,
      out0: torch.Tensor,
      out1: torch.Tensor,
      out2: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _meta_registrations.device_hint(input) == "tpu":
      out0_device = _effective_device(out0)
      out1_device = _effective_device(out1)
      out2_device = _effective_device(out2)
      torch._check(
          out0_device.type == "tpu"
          and out1_device.type == "tpu"
          and out2_device.type == "tpu",
          lambda: (
              "native_group_norm(): expected all output tensors to be on tpu,"
              f" got {out0_device}, {out1_device}, and {out2_device}"
          ),
      )
      y, mean, rstd = _native_group_norm_tpu_meta(
          input, weight, bias, N, C, HxW, group
      )
      for out_tensor, expected in ((out0, y), (out1, mean), (out2, rstd)):
        exp_str = _format_dtype(expected.dtype)
        got_str = _format_dtype(out_tensor.dtype)
        torch._check(
            out_tensor.dtype == expected.dtype,
            lambda exp_str=exp_str, got_str=got_str: (
                "native_group_norm(): expected the output dtype to be"
                f" {exp_str}, got {got_str}"
            ),
        )
        out_tensor.resize_(expected.shape)
      return (out0, out1, out2)
    if upstream_out_meta is not None:
      return upstream_out_meta(
          input,
          weight,
          bias,
          N,
          C,
          HxW,
          group,
          eps,
          out0=out0,
          out1=out1,
          out2=out2,
      )
    if upstream_out_decomp is not None:
      return upstream_out_decomp(
          input,
          weight,
          bias,
          N,
          C,
          HxW,
          group,
          eps,
          out0=out0,
          out1=out1,
          out2=out2,
      )
    y, mean, rstd = _native_group_norm_meta(
        input, weight, bias, N, C, HxW, group, eps
    )
    for out_tensor, result in ((out0, y), (out1, mean), (out2, rstd)):
      out_tensor.resize_(result.shape)
      out_tensor.copy_(result)
    return (out0, out1, out2)

  op_to_fn: tuple[tuple[Any, Callable[..., Any]], ...] = (
      (default_op, _native_group_norm_meta),
      (out_op, _native_group_norm_out_meta),
  )
  for op, fn in op_to_fn:
    _decomp.meta_table[op] = fn
    if hasattr(torch._C, "_fake_dispatch_register_meta"):
      schema = op._schema
      torch._C._fake_dispatch_register_meta(schema.name, schema.overload_name)
    op.py_kernels.pop(meta_key, None)
    meta_lib.impl(op, fn, allow_override=True)
    op.py_impl(meta_key)(fn)
  # pylint: enable=protected-access


# Upstream PyTorch's `native_group_norm_backward` decomposition is decorated
# with `@pw_cast_for_opmath`, which promotes all returned outputs (including
# `d_input`) to `float32` whenever `mean`/`rstd` or `weight` are `float32`,
# whereas `AtenNativeGroupNormBackward` returns `d_input` in `input.dtype` and
# `d_weight`/`d_bias` in `weight.dtype` if `weight` is provided else
# `mean.dtype`. Override Meta on TPU so `aot_autograd` FakeTensors match the
# compiled TPU backward output dtypes.
def _native_group_norm_backward_tpu_meta(
    grad_out: torch.Tensor,
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
  """TPU Meta implementation matching `AtenNativeGroupNormBackward`."""
  # pylint: disable=protected-access
  for tensor in (grad_out, input, mean, rstd, weight):
    if tensor is not None:
      dtype_str = _format_dtype(tensor.dtype)
      torch._check(
          tensor.dtype not in _LOW_PRECISION_FLOATS,
          lambda dtype_str=dtype_str: (
              f"native_group_norm_backward(): not implemented for {dtype_str}"
          ),
      )
  torch._check(
      grad_out.numel() == N * C * HxW,
      lambda: (
          "native_group_norm_backward(): expected grad_out to have"
          f" {N * C * HxW} elements, got {grad_out.numel()}"
      ),
  )
  torch._check(
      input.numel() == N * C * HxW,
      lambda: (
          "native_group_norm_backward(): expected input to have"
          f" {N * C * HxW} elements, got {input.numel()}"
      ),
  )
  torch._check(
      grad_out.shape == input.shape,
      lambda: (
          "native_group_norm_backward(): expected grad_out and input to have"
          f" the same dimensions, got grad_out size {list(grad_out.shape)},"
          f" input size {list(input.shape)}"
      ),
  )
  torch._check(
      mean.shape == (N, group),
      lambda: (
          "native_group_norm_backward(): expected mean to have shape"
          f" [{N}, {group}], got {list(mean.shape)}"
      ),
  )
  torch._check(
      mean.shape == rstd.shape,
      lambda: (
          "native_group_norm_backward(): expected mean and rstd to have the"
          f" same shape, got mean size {list(mean.shape)} and rstd size"
          f" {list(rstd.shape)}"
      ),
  )
  torch._check(
      group > 0,
      lambda: (
          "native_group_norm_backward(): expected num groups to be greater than"
          f" 0, got {group}"
      ),
  )
  torch._check(
      C % group == 0,
      lambda: (
          "native_group_norm_backward(): expected number of channels to be"
          f" divisible by num_groups ({group}), got {C}"
      ),
  )
  if weight is not None:
    torch._check(
        weight.numel() == C,
        lambda: (
            f"native_group_norm_backward(): expected weight to have {C}"
            f" elements, got {weight.numel()}"
        ),
    )
  # pylint: enable=protected-access
  dparam_dtype = weight.dtype if weight is not None else mean.dtype
  d_input = (
      torch.empty_like(input, memory_format=torch.contiguous_format)
      if output_mask[0]
      else None
  )
  d_weight = (
      input.new_empty((C,), dtype=dparam_dtype) if output_mask[1] else None
  )
  d_bias = input.new_empty((C,), dtype=dparam_dtype) if output_mask[2] else None
  return (d_input, d_weight, d_bias)


def _patch_native_group_norm_backward_meta(
    meta_lib: torch.library.Library,
) -> None:
  """Patches `aten.native_group_norm_backward` default Meta kernel."""
  # pylint: disable=protected-access
  default_op = torch.ops.aten.native_group_norm_backward.default
  meta_key = torch._C.DispatchKey.Meta
  upstream_decomp = _decomp.decomposition_table.get(default_op)
  upstream_meta = default_op.py_kernels.get(meta_key)

  def _native_group_norm_backward_meta(
      grad_out: torch.Tensor,
      input: torch.Tensor,  # pylint: disable=redefined-builtin
      mean: torch.Tensor,
      rstd: torch.Tensor,
      weight: torch.Tensor | None,
      N: int,
      C: int,
      HxW: int,
      group: int,
      output_mask: list[bool],
  ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if _meta_registrations.device_hint(input) == "tpu":
      return _native_group_norm_backward_tpu_meta(
          grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
      )
    if upstream_meta is not None:
      return upstream_meta(
          grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
      )
    if upstream_decomp is not None:
      return upstream_decomp(
          grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
      )
    raise NotImplementedError(
        "No upstream implementation for native_group_norm_backward on device"
        f" {input.device}"
    )

  _decomp.meta_table[default_op] = _native_group_norm_backward_meta
  if hasattr(torch._C, "_fake_dispatch_register_meta"):
    schema = default_op._schema
    torch._C._fake_dispatch_register_meta(schema.name, schema.overload_name)
  default_op.py_kernels.pop(meta_key, None)
  meta_lib.impl(
      default_op, _native_group_norm_backward_meta, allow_override=True
  )
  default_op.py_impl(meta_key)(_native_group_norm_backward_meta)
  # pylint: enable=protected-access


def register() -> None:
  """Registers TPU Meta kernel overrides."""
  global _meta_lib
  if _meta_lib is not None:
    return
  _meta_lib = torch.library.Library("aten", "IMPL", "Meta")
  _patch_sdpa_meta(_meta_lib)
  _patch_addmm_dtype_meta(_meta_lib)
  _patch_bmm_dtype_meta(_meta_lib)
  _patch_fused_dropout_meta(_meta_lib)
  _patch_native_group_norm_meta(_meta_lib)
  _patch_native_group_norm_backward_meta(_meta_lib)
