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

"""TorchInductor backend for TPU that lowers fused kernels to Pallas.

`torch.compile(backend="tpu_inductor_pallas")` runs TorchInductor on graphs
of TPU tensors. Pointwise/reduction fusions are code-generated as JAX Pallas
kernels by reusing the upstream experimental Pallas codegen
(`torch._inductor.codegen.pallas`); everything Inductor does not lower
(matmuls, SDPA, embeddings, ...) falls back to aten ops executed by torch_tpu
eager mode.

Unlike the upstream `PALLAS_TARGET_TPU` debug path — which round-trips every
tensor through host memory into a separate JAX TPU client — the generated
kernels here are executed natively by torch_tpu: the `jax.jit`-wrapped
`pallas_call` is traced with `jax.export` (trace-only, platform "tpu"),
serialized to StableHLO, and run on torch_tpu's own PJRT runtime via the same
machinery that powers `torch_tpu.pallas.jax_op`. Tensors never leave the
torch_tpu runtime and no second TPU client is created.

Deviations from the upstream Pallas codegen (see `TpuPallasKernel`):
  * Output aliasing / input donation is disabled. The generated
    `<kernel>_jit_wrapper` therefore contains a plain (alias-free)
    `pallas_call`, which keeps the `jax.export` path simple; kernel results
    are copied back into the Inductor-allocated output buffers.
  * The generated `<kernel>_main` entry point (the function Inductor's
    wrapper calls) is replaced with one that dispatches through
    `torch_tpu._internal.inductor.pallas_runtime.run_kernel`.
"""

import re
from typing import Any, Optional, Sequence

from absl import logging
import torch
from torch._dynamo.backends import registry

_BACKEND_NAME = "tpu_inductor_pallas"

# Inductor config values this backend requires:
#   inplace_buffers=False   -- prevents in_out_ptr kernel params. In-place
#       params would require donating the corresponding jax buffers, which the
#       torch_tpu execution path does not support yet (results are copied back
#       into the Inductor-allocated buffers instead).
#   fallback_random=True    -- RNG ops run as aten fallbacks on the tpu device
#       instead of being lowered (Philox-style lowering is CUDA-specific).
#   implicit_fallbacks=True -- any op without an Inductor lowering falls back
#       to the aten op, executed by torch_tpu eager mode.
_INDUCTOR_CONFIG_PATCHES = {
    "inplace_buffers": False,
    "fallback_random": True,
    "implicit_fallbacks": True,
    "triton.cudagraphs": False,
}

_registered = False

# Prepended to every generated kernel module; used by the rewritten
# full-array stores. Runs under jax tracing, so jnp-only.
_TPU_STORE_HELPER = '''
import math as _tpu_math
import jax.numpy as _tpu_jnp

def _tpu_prepare_store(value, shape):
    value = _tpu_jnp.asarray(value)
    if value.size == _tpu_math.prod(shape):
        return value.reshape(shape)
    return _tpu_jnp.broadcast_to(value, shape)
'''


def _make_tpu_pallas_classes():
  """Builds the TPU scheduling/codegen classes.

  Deferred into a function so that importing this module does not import
  torch._inductor (which is heavy and, more importantly, must not be imported
  before torch_tpu finishes loading).
  """
  from torch._inductor.codegen import pallas as pallas_codegen
  from torch._inductor.codegen.common import DeviceOpOverrides
  from torch._inductor.virtualized import V
  from torch.utils._ordered_set import OrderedSet

  class TpuPallasKernelOverrides(pallas_codegen.PallasKernelOverrides):

    @staticmethod
    def fma(a: str, b: str, c: str) -> str:
      # jnp.fma has no Mosaic lowering; mul+add is equivalent numerics-wise
      # for these kernels.
      return f"(({a}) * ({b}) + ({c}))"

    @staticmethod
    def index_expr(expr, dtype: torch.dtype) -> str:
      """Iteration-position value with canonical broadcasting.

      Upstream renders index expressions against the flat 1-D iteration-var
      arrays, so an expression mixing pointwise and reduction vars (e.g. the
      causal mask `r0_2 - x0`) collapses elementwise instead of forming the
      (pw, r) outer product. Render each var as its mixed-radix digit of the
      flat position, shaped (pw, 1) / (1, r) inside reduction kernels.
      """
      from torch._inductor.utils import get_bounds_index_expr

      kernel = V.kernel
      prepared = kernel.prepare_indexing(expr)
      idx_str = kernel._tpu_index_value_str(prepared)  # pylint: disable=protected-access
      var = kernel.cse.generate(
          kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
      )
      return TpuPallasKernelOverrides.to_dtype(var, dtype)

  class TpuPallasKernel(pallas_codegen.PallasKernel):
    """Upstream Pallas kernel codegen, retargeted at torch_tpu execution."""

    overrides = TpuPallasKernelOverrides  # type: ignore[assignment]

    def _tpu_index_value_str(self, prepared) -> str:
      pw_numel, r_numel = self._iteration_numels()
      expr_str = self.kexpr(prepared)
      if pw_numel is None:
        return expr_str
      for var, entry in self.range_tree_nodes.items():
        name = str(var)
        if not re.search(rf"\b{re.escape(name)}\b", expr_str):
          continue
        if entry.prefix.startswith("r"):
          arr = (
              f"(((jnp.arange({r_numel})) // {int(entry.divisor)}) %"
              f" {int(entry.length)})"
          )
          if self.inside_reduction:
            arr += "[None, :]"
        else:
          arr = (
              f"(((jnp.arange({pw_numel})) // {int(entry.divisor)}) %"
              f" {int(entry.length)})"
          )
          if self.inside_reduction:
            arr += "[:, None]"
        expr_str = re.sub(rf"\b{re.escape(name)}\b", arr, expr_str)
      return expr_str

    def dtype_to_str(self, dtype: torch.dtype) -> str:  # type: ignore[override]
      # Needed by dtype propagation for kernels with indirect indexing
      # (e.g. embedding gathers); the base SIMDKernel leaves it abstract.
      from torch._inductor.runtime.runtime_utils import torch_dtype_to_jax

      return torch_dtype_to_jax(dtype)

    def _buffer_is_contiguous(self, buffer_name: str) -> bool:  # type: ignore[override]
      # Upstream marks contiguous outputs as "aliasable" and donates them to
      # the jax kernel. Returning False forces the no-alias path: every
      # output goes through out_shapes/out_dtypes and is copied back.
      del buffer_name
      return False

    # --- Canonical shapes over the iteration space ------------------------
    #
    # The upstream Pallas codegen mixes values whose shapes come from the
    # underlying buffers (e.g. a (4, 64, 128) load) with values shaped by
    # the flattened iteration space (e.g. a (256,) reduction result), and
    # relies on size-matching heuristics to reconcile them. Those heuristics
    # break for >2-D buffers, differently-viewed same-numel buffers, and
    # post-reduction pointwise math (all hit by RMSNorm / cross-entropy
    # style fusions).
    #
    # Instead, each kernel parameter is assigned a canonical shape over the
    # flattened iteration space:
    #     pointwise kernel, buffer spans the space   -> (numel,)
    #     reduction kernel, pointwise+reduction vars -> (pw_numel, r_numel)
    #     reduction kernel, pointwise vars only      -> (pw_numel, 1)
    #     reduction kernel, reduction vars only      -> (r_numel,)
    # Crucially the reshapes happen OUTSIDE the Pallas kernel, at the torch
    # boundary in pallas_runtime.run_kernel (free views on contiguous
    # tensors) — Mosaic, the Pallas TPU compiler, supports only a narrow set
    # of in-kernel reshapes, so the kernel must see already-canonical refs.
    # Partial reductions emit keepdims results, (pw_numel, 1), so downstream
    # pointwise math broadcasts against (pw_numel, r_numel) values.
    #
    # This assumes loads produce elements in iteration order (pointwise-
    # major, reduction-minor), which holds for the contiguous/broadcast
    # patterns Inductor generates here — the upstream `reshape(-1, r_numel)`
    # reduction path makes the same assumption.

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # Kernel param name -> canonical shape tuple, or None to pass the
      # buffer through in its original shape.
      self._tpu_canonical_shapes: dict[str, Optional[tuple]] = {}
      # (buffer name, spec) -> alias kernel param, for buffers accessed
      # through more than one pattern in the same kernel (e.g. the
      # interleaved real/imag reads of complex RoPE). The buffer is simply
      # passed twice with different boundary views.
      self._tpu_alias_params: dict[tuple, str] = {}
      # Set when a load/store emitted flat-offset index arrays; such
      # kernels must keep 1-D canonical shapes (see the fold-to-2-D step).
      self._tpu_requires_flat = False

    def _iteration_numels(self):
      pw_numel = 1
      r_numel = 1
      for prefix, numel in self.numels.items():
        try:
          n = int(numel)
        except (TypeError, ValueError):
          return None, None  # dynamic shapes: leave upstream behavior
        if prefix.startswith("r"):
          r_numel *= n
        else:
          pw_numel *= n
      return pw_numel, r_numel

    def _classify_index_vars(self, index) -> tuple[bool, bool]:
      """Returns (has_pointwise, has_reduction) for an index expression."""
      has_pw = False
      has_r = False
      for sym in index.free_symbols:
        entry = self.range_tree_nodes.get(sym)
        if entry is not None:
          if entry.prefix.startswith("r"):
            has_r = True
          else:
            has_pw = True
        elif str(sym).startswith("tmp"):
          # Indirect index (e.g. a gather): derived from a loaded value,
          # which in these kernels is a function of pointwise vars.
          has_pw = True
      return has_pw, has_r

    def _canonical_shape_for(self, buf_name: str, index) -> Optional[tuple]:
      pw_numel, r_numel = self._iteration_numels()
      if pw_numel is None:
        return None
      try:
        buf_numel = int(V.graph.get_buffer(buf_name).get_numel())
      except (TypeError, ValueError):
        return None
      if buf_numel == 1:
        # Pallas TPU only supports refs of rank >= 1; 0-d buffers must be
        # presented as (1,) and broadcast from there.
        return (1,)
      if not self.inside_reduction:
        if buf_numel == pw_numel * r_numel:
          return (pw_numel * r_numel,)
        return None
      has_pw, has_r = self._classify_index_vars(index)
      if has_pw and has_r and buf_numel == pw_numel * r_numel:
        return (pw_numel, r_numel)
      if has_pw and not has_r and buf_numel == pw_numel:
        return (pw_numel, 1)
      if has_r and not has_pw and buf_numel == r_numel:
        return (r_numel,)
      return None

    def _record_canonical_shape(self, param: str, shape: Optional[tuple]):
      if param in self._tpu_canonical_shapes:
        if self._tpu_canonical_shapes[param] != shape:
          # Conflicting access patterns (e.g. full load + partial slice):
          # pass the buffer through unreshaped.
          self._tpu_canonical_shapes[param] = None
      else:
        self._tpu_canonical_shapes[param] = shape

    def _tpu_param_for(self, name: str, spec) -> str:
      """Kernel param carrying buffer `name` with boundary view `spec`.

      If the buffer is already bound to a different view, an alias param is
      created (the wrapper passes the same buffer again; see call_kernel).
      """
      unset = object()
      param = str(self.args.input(name))
      existing = self._tpu_canonical_shapes.get(param, unset)
      if existing is unset or existing == spec:
        self._tpu_canonical_shapes[param] = spec
        return param
      key = (name, spec)
      alias = self._tpu_alias_params.get(key)
      if alias is None:
        alias_outer = f"{name}__tpu_alias{len(self._tpu_alias_params)}"
        alias = str(self.args.input(alias_outer))
        self._tpu_alias_params[key] = alias
        self._tpu_canonical_shapes[alias] = spec
        # Make dtype/size lookups (python_argdefs etc.) resolve the alias.
        if alias_outer not in V.graph.name_to_buffer:
          src = V.graph.name_to_buffer.get(name) or V.graph.graph_inputs.get(
              name
          )
          if src is not None:
            V.graph.name_to_buffer[alias_outer] = src
      return alias

    # The upstream _convert_to_jax_slice silently returns "..." for
    # multi-variable strided access ("TODO: Implement proper
    # multi-dimensional strided indexing"), which drops permutations and
    # corrupts values. Guard every non-indirect load/store: if the index is
    # not exactly contiguous over the iteration space, express the affine
    # access pattern as a torch-side as_strided view (recorded in the
    # canonical-shape table and applied by pallas_runtime at the kernel
    # boundary — Mosaic supports neither general reshapes nor general
    # gathers inside the kernel), and fail loudly on anything non-affine.

    def _tpu_index_is_contiguous(self, prepared) -> bool:
      """True iff the index reads/writes in flat iteration order.

      Flat positions are built from the vars the index actually uses — the
      range tree can hold several (redundant) decompositions of the same
      range, and a given index only ever uses one of them.
      """
      import sympy as sympy_mod

      free = prepared.free_symbols & set(self.range_tree_nodes.keys())
      if not free:
        return True  # constant index; parent handles
      pw_numel, r_numel = self._iteration_numels()
      if pw_numel is None:
        return True  # dynamic shapes: keep upstream behavior
      pw_flat = sympy_mod.S.Zero
      r_flat = sympy_mod.S.Zero
      try:
        for var in free:
          entry = self.range_tree_nodes[var]
          term = sympy_mod.Integer(int(entry.divisor)) * var
          if entry.prefix.startswith("r"):
            r_flat = r_flat + term
          else:
            pw_flat = pw_flat + term
      except (TypeError, ValueError):
        return True
      candidates = [
          pw_flat * sympy_mod.Integer(r_numel) + r_flat,
          pw_flat,
          r_flat,
      ]
      return any(
          sympy_mod.expand(prepared - cand) == 0 for cand in candidates
      )

    def _tpu_affine_terms(self, prepared):
      """Decomposes an index into mixed-radix digits, or None.

      Returns ``(digits, offset)`` where each digit is
      ``(is_reduction, position_divisor, length, stride)``: the digit's
      value at flat iteration position p is ``(p // position_divisor) %
      length`` and it contributes ``stride * value`` to the index. Tree
      vars are digits themselves (var = (p // divisor) % length), and
      ``var // q`` / ``ModularIndexing(var, q, m)`` are finer sub-digits.
      """
      from torch.utils._sympy.functions import FloorDiv, ModularIndexing

      def digit_of(base):
        # -> (is_reduction, position_divisor, length) or None
        if base in self.range_tree_nodes:
          entry = self.range_tree_nodes[base]
          return (
              entry.prefix.startswith("r"),
              int(entry.divisor),
              int(entry.length),
          )
        if isinstance(base, FloorDiv):
          inner, q = base.args
          d = digit_of(inner)
          if d is None or not q.is_Integer:
            return None
          is_r, div, length = d
          q = int(q)
          if q <= 0 or length % q != 0:
            return None
          return (is_r, div * q, length // q)
        if isinstance(base, ModularIndexing):
          inner, q, m = base.args
          d = digit_of(inner)
          if d is None or not q.is_Integer or not m.is_Integer:
            return None
          is_r, div, length = d
          q, m = int(q), int(m)
          if q <= 0 or m <= 0 or length % (q * m) != 0 and length > q * m:
            return None
          return (is_r, div * q, m)
        return None

      digits = []
      offset = 0
      for term, coeff in prepared.as_coefficients_dict().items():
        try:
          coeff = int(coeff)
        except (TypeError, ValueError):
          return None
        if term == 1:
          offset = coeff
          continue
        digit = digit_of(term)
        if digit is None:
          return None
        digits.append(digit + (coeff,))
      if not digits:
        return None
      return digits, offset

    def _tpu_view_dims(self, digits, numel, want_r, allow_broadcast):
      """Builds (sizes, strides) covering one iteration space exactly.

      Digits are mixed-radix positions of the flat iteration index (value =
      (p // position_divisor) % length). Present digits become view dims
      (outer-to-inner by position divisor); any gaps in the divisor chain —
      ranges the index does not depend on — become stride-0 broadcast dims.
      Overlapping decompositions of the same range raise.
      """
      entries = sorted(
          (
              (div, length, stride)
              for is_r, div, length, stride in digits
              if is_r == want_r
          ),
          key=lambda dls: -dls[0],
      )
      dims = []
      remaining = numel
      for div, length, stride in entries:
        span = div * length
        if span > remaining or remaining % span != 0:
          raise pallas_codegen.Unsupported(
              "tpu pallas: inconsistent index decomposition"
              f" (span {span}, remaining {remaining})"
          )
        if span < remaining:
          if not allow_broadcast:
            raise pallas_codegen.Unsupported(
                "tpu pallas store index does not cover the iteration space"
            )
          dims.append((remaining // span, 0))
        dims.append((length, stride))
        remaining = div
      if remaining > 1:
        if not allow_broadcast:
          raise pallas_codegen.Unsupported(
              "tpu pallas store index does not cover the iteration space"
          )
        dims.append((remaining, 0))
      return dims

    def _tpu_strided_load(self, name: str, prepared) -> Any:
      """Non-contiguous affine load: plain [...] read of a torch-side view.

      The access pattern (including broadcasts: absent vars get stride 0)
      becomes an as_strided view materialized by pallas_runtime, so the
      kernel sees a canonically-shaped ref covering the full iteration
      space in iteration order.
      """
      terms = self._tpu_affine_terms(prepared)
      pw_numel, r_numel = self._iteration_numels()
      if terms is None or pw_numel is None:
        raise pallas_codegen.Unsupported(
            f"tpu pallas load with non-affine index not supported: {prepared}"
        )
      digits, offset = terms
      dims = self._tpu_view_dims(
          digits, pw_numel, want_r=False, allow_broadcast=True
      ) + self._tpu_view_dims(
          digits, r_numel, want_r=True, allow_broadcast=True
      )
      sizes = tuple(s for s, _ in dims)
      view_strides = tuple(st for _, st in dims)
      canonical = (
          (pw_numel, r_numel) if self.inside_reduction else (pw_numel * r_numel,)
      )
      spec = ("as_strided", sizes, view_strides, offset, canonical)
      param = self._tpu_param_for(name, spec)
      self._tpu_requires_flat = True
      return self.cse.generate(
          self.compute,
          f"{param}[...]",
          dtype=V.graph.get_dtype(name),
      )

    def load(self, name: str, index) -> Any:  # type: ignore[override]
      if self._has_indirect_vars(index):
        result = super().load(name, index)
        self._record_canonical_shape(
            str(self.args.input(name)),
            self._canonical_shape_for(name, index),
        )
        return result
      prepared = self.prepare_indexing(index)
      free = prepared.free_symbols & set(self.range_tree_nodes.keys())
      if not free and prepared.is_Integer:
        # Constant index (e.g. a 0-d grad seed): `[0]` on a 0-d ref is
        # invalid, and integer indexing of larger refs is not lowerable.
        # A whole-ref load broadcasts fine for single-element buffers.
        try:
          buf_numel = int(V.graph.get_buffer(name).get_numel())
        except (TypeError, ValueError):
          buf_numel = None
        if buf_numel == 1:
          param = self._tpu_param_for(name, (1,))
          return self.cse.generate(
              self.compute,
              f"{param}[...]",
              dtype=V.graph.get_dtype(name),
          )
      if not self._tpu_index_is_contiguous(prepared):
        return self._tpu_strided_load(name, prepared)
      # Contiguous full-array load; emit it against the (possibly aliased)
      # param so a buffer can carry different views for different loads.
      param = self._tpu_param_for(
          name, self._canonical_shape_for(name, index)
      )
      self.load_index_exprs[name] = index
      if self.use_masked_ops is None:
        self.use_masked_ops = self._determine_masked_ops_for_kernel()
      return self.cse.generate(
          self.compute,
          f"{param}[...]",
          dtype=V.graph.get_dtype(name),
      )

    def _tpu_strided_store(self, name: str, prepared, value) -> None:
      """Non-contiguous affine store: kernel writes iteration order; the
      runtime scatters into the real buffer through an as_strided view.

      Requires a bijection (every index var present, sizes covering the
      buffer) — otherwise multiple iteration points would alias one output
      element.
      """
      terms = self._tpu_affine_terms(prepared)
      pw_numel, r_numel = self._iteration_numels()
      try:
        buf_numel = int(V.graph.get_buffer(name).get_numel())
      except (TypeError, ValueError):
        buf_numel = None
      if terms is None or pw_numel is None or buf_numel is None:
        raise pallas_codegen.Unsupported(
            f"tpu pallas store with non-affine index not supported: {prepared}"
        )
      digits, offset = terms
      dims = self._tpu_view_dims(
          digits, pw_numel, want_r=False, allow_broadcast=False
      )
      if self.inside_reduction and any(is_r for is_r, _, _, _ in digits):
        dims += self._tpu_view_dims(
            digits, r_numel, want_r=True, allow_broadcast=False
        )
      sizes = tuple(s for s, _ in dims)
      total = 1
      for s in sizes:
        total *= s
      if total != buf_numel:
        raise pallas_codegen.Unsupported(
            "tpu pallas store index does not cover the output buffer"
            f" bijectively: {prepared}"
        )
      view_strides = tuple(st for _, st in dims)
      spec = ("as_strided", sizes, view_strides, offset, (buf_numel,))
      out = self.args.output(name)
      self._record_canonical_shape(str(out), spec)
      if self._tpu_canonical_shapes.get(str(out)) != spec:
        raise pallas_codegen.Unsupported(
            f"tpu pallas: conflicting access patterns for buffer {name}"
        )
      self._tpu_requires_flat = True
      self.store_buffer_names.add(name)
      store_expr = f"{out}[...] = _tpu_prepare_store({value}, {out}.shape)"
      self.stores.writeline(store_expr)
      self.store_with_output.append((out, store_expr))

    def store(self, name: str, index, value, mode: Any = None) -> None:  # type: ignore[override]
      if not self._has_indirect_vars(index):
        prepared = self.prepare_indexing(index)
        if not self._tpu_index_is_contiguous(prepared):
          if mode is not None:
            raise pallas_codegen.Unsupported(
                f"tpu pallas store mode {mode} not supported"
            )
          self._tpu_strided_store(name, prepared, value)
          return
      self._record_canonical_shape(
          str(self.args.output(name)), self._canonical_shape_for(name, index)
      )
      super().store(name, index, value, mode=mode)

    def reduction(self, dtype, src_dtype, reduction_type, value):  # type: ignore[override]
      assert self.inside_reduction
      if isinstance(value, tuple):
        raise pallas_codegen.Unsupported(
            "Tuple reductions (e.g., welford_combine) not supported in"
            " Pallas backend"
        )
      reduction_ops = {
          "sum": "jnp.sum",
          "prod": "jnp.prod",
          "max": "jnp.max",
          "min": "jnp.min",
          "any": "jnp.any",
          "argmax": "jnp.argmax",
          "argmin": "jnp.argmin",
      }
      pw_numel, r_numel = self._iteration_numels()
      if (
          reduction_type not in reduction_ops
          or pw_numel is None
          or pw_numel == 1
      ):
        # Full reductions to scalar (and exotic types) are handled fine
        # upstream.
        return super().reduction(dtype, src_dtype, reduction_type, value)

      cache_key = (src_dtype, reduction_type, value)
      if cache_key in self.cse.reduction_cache:
        return self.cse.reduction_cache[cache_key]

      # broadcast_to handles every canonical input shape — (pw, r), (pw, 1),
      # (r,), scalars — and keepdims keeps the result 2-D, both natively
      # supported by Mosaic (unlike reshapes).
      op = reduction_ops[reduction_type]
      reduction_expr = (
          f"{op}(jnp.broadcast_to({value}, ({pw_numel}, {r_numel})),"
          " axis=-1, keepdims=True)"
      )
      result = self.cse.generate(self.compute, reduction_expr, dtype=dtype)
      self.cse.reduction_cache[cache_key] = result
      return result

    def call_kernel(self, name: str, node=None) -> None:  # type: ignore[override]
      """Same as upstream, but alias params resolve to their real buffer."""
      wrapper = V.graph.wrapper_code
      _, call_args, _, _ = self.args.python_argdefs()
      call_arg_strs = [
          re.sub(r"__tpu_alias\d+$", "", str(arg)) for arg in call_args
      ]
      wrapper.writeline(f"{name}.run({', '.join(call_arg_strs)})")

    def codegen_kernel(self, name: Optional[str] = None) -> str:  # type: ignore[override]
      src = super().codegen_kernel(name)
      # Keep iteration variables flat 1-D. Upstream reshapes them to the
      # first output's shape for N-D broadcasting; under the canonical flat
      # shapes used here that reshape is wrong, and flat index arrays also
      # keep gathers low-rank for Mosaic (which only supports 2-D gathers).
      src = re.sub(
          r"^(\s*(?:[xyz]\d+|r\d+_\d*) = jnp\.arange\([^)]*\))\.reshape\([^)]*\)\s*$",
          r"\1",
          src,
          flags=re.MULTILINE,
      )
      # Upstream full-array stores emit `out[...] = value.reshape(out.shape)`,
      # which breaks when the value only broadcasts over the output (e.g. a
      # scalar zero-fill kernel). Route them through a broadcast-aware helper.
      src = re.sub(
          r"(\w+)\[\.\.\.\] = (\w+)\.reshape\(\1\.shape\)",
          r"\1[...] = _tpu_prepare_store(\2, \1.shape)",
          src,
      )
      # Same for scalar-output stores (`out[...] = tmpN` with no reshape):
      # Pallas refs are at least rank-1, so a 0-d value needs broadcasting.
      src = re.sub(
          r"^(\s*)(\w+)\[\.\.\.\] = (\w+)\s*$",
          r"\1\2[...] = _tpu_prepare_store(\3, \2.shape)",
          src,
          flags=re.MULTILINE,
      )
      src = _TPU_STORE_HELPER + src
      # Shadow the upstream `<kernel>_main` (which converts tensors via
      # DLPack / host round-trips) with a torch_tpu-native entry point.
      # Later `def` of the same name wins when the module is executed.
      src += "\n\n" + self._codegen_tpu_main(name or "<KERNEL_NAME>")
      return src

    def _codegen_tpu_main(self, kernel_name: str) -> str:
      arg_defs, _, _, _ = self.args.python_argdefs()
      params = [a.name for a in arg_defs]
      in_out_params = [p for p in params if p.startswith("in_out_ptr")]
      if in_out_params:
        # inplace_buffers=False should make this unreachable; fail loudly
        # rather than corrupting donated buffers.
        raise pallas_codegen.Unsupported(
            "tpu_inductor_pallas does not support in_out_ptr kernel"
            f" parameters (got {in_out_params})"
        )
      out_params = [p for p in params if p.startswith("out_ptr")]
      in_params = [p for p in params if p.startswith("in_ptr")]

      def as_tuple(items) -> str:
        return "(" + ", ".join(str(i) for i in items) + (
            "," if items else ""
        ) + ")"

      out_canonical = [self._tpu_canonical_shapes.get(p) for p in out_params]
      in_canonical = [self._tpu_canonical_shapes.get(p) for p in in_params]

      # Flat 1-D buffers tile poorly in VMEM (Mosaic pads them massively).
      # For pointwise kernels where every value is canonical and the kernel
      # body never references iteration-var arrays (index_expr / gathers),
      # fold the flat shape into 2-D with a 128-wide lane dimension.
      pw_numel, r_numel = self._iteration_numels()
      total = None if pw_numel is None else pw_numel * r_numel
      if (
          not self.inside_reduction
          and not self._tpu_requires_flat
          and total
          and total % 128 == 0
          and all(c is not None for c in out_canonical + in_canonical)
      ):
        body_text = "\n".join(
            str(line) for line in list(self.compute._lines)  # pylint: disable=protected-access
        )
        uses_iter_vars = any(
            re.search(rf"\b{re.escape(str(var))}\b", body_text)
            for var in self.range_tree_nodes
        )
        if not uses_iter_vars:
          folded = (total // 128, 128)
          out_canonical = [
              folded if c == (total,) else c for c in out_canonical
          ]
          in_canonical = [
              folded if c == (total,) else c for c in in_canonical
          ]
      lines = [
          f"def {kernel_name}_main({', '.join(params)}, stream=None):",
          "    from torch_tpu._internal.inductor import pallas_runtime",
          "    pallas_runtime.run_kernel(",
          f"        {kernel_name}_jit_wrapper,",
          f"        '{kernel_name}',",
          f"        outs={as_tuple(out_params)},",
          f"        ins={as_tuple(in_params)},",
          f"        out_canonical={as_tuple(out_canonical)},",
          f"        in_canonical={as_tuple(in_canonical)},",
          "    )",
      ]
      return "\n".join(lines) + "\n"

  class TpuPallasScheduling(pallas_codegen.PallasScheduling):
    kernel_type = TpuPallasKernel  # type: ignore[assignment]

    @classmethod
    def get_tiling_and_scores(
        cls,
        node_schedule,
        numel,
        reduction_numel=None,
        coalesce_analysis=None,
    ):
      # Never tile. The upstream Pallas codegen silently drops the
      # permutation in tiled (multi-var) kernels — a full `[...]` load plus
      # a reshape store loses the y/x axis reordering. With a single flat
      # pointwise var, permuted access lowers to explicit index arrays,
      # which is correct (and whole-array kernels have no launch-grid reason
      # to tile anyway). select_tiling delegates here, so this covers every
      # tiling query.
      import sympy as sympy_mod

      del node_schedule, coalesce_analysis
      if reduction_numel is None:
        reduction_numel = sympy_mod.S.One
      return cls.create_tiling([numel], [reduction_numel]), None

  class TpuDeviceOpOverrides(DeviceOpOverrides):
    """Minimal device ops for the generated Python wrapper.

    Pallas kernels are invoked as plain Python calls (no streams, no device
    guards), so these only need to keep the wrapper codegen happy.
    """

    def import_get_raw_stream_as(self, name: str) -> str:
      return f"{name} = lambda device_idx: None"

    def set_device(self, device_idx: int) -> str:
      return f"pass  # tpu set_device({device_idx})"

    def synchronize(self) -> str:
      return "torch.tpu.synchronize()"

    def device_guard(self, device_idx: int) -> str:
      return f"contextlib.nullcontext()  # tpu device_guard({device_idx})"

    def cpp_device_guard(self) -> str:
      raise NotImplementedError(
          "tpu_inductor_pallas does not support cpp_wrapper"
      )

  del OrderedSet  # unused; kept import above minimal
  return TpuPallasScheduling, TpuDeviceOpOverrides


def _make_tpu_device_interface():
  """Builds a minimal dynamo DeviceInterface for the tpu device.

  Inductor queries it for device properties (e.g. in the reduction-split
  heuristic via `DeviceProperties.create`). Pallas kernels are whole-array
  `grid=(1,)` calls with no streams, so only the property/availability
  surface is implemented.
  """
  import types

  from torch._dynamo import device_interface as dynamo_device_interface

  # One TensorCore per chip from the kernel-launch perspective; the value
  # only feeds split-reduction heuristics, which Pallas handles either way.
  _tpu_props = types.SimpleNamespace(multi_processor_count=1)

  class TpuDeviceInterface(dynamo_device_interface.DeviceInterface):

    class Worker:

      @staticmethod
      def set_device(device: int) -> None:
        del device

      @staticmethod
      def current_device() -> int:
        return 0

      @staticmethod
      def get_device_properties(device=None):
        del device
        return _tpu_props

    @staticmethod
    def is_available() -> bool:
      return True

    @staticmethod
    def current_device() -> int:
      return 0

    @staticmethod
    def set_device(device) -> None:
      del device

    @staticmethod
    def device_count() -> int:
      tpu_module = getattr(torch, "tpu", None)
      if tpu_module is not None and hasattr(tpu_module, "device_count"):
        return tpu_module.device_count()
      return 1

    @staticmethod
    def get_device_properties(device=None):
      del device
      return _tpu_props

    @staticmethod
    def get_compute_capability(device=None) -> int:
      del device
      return 0

    @staticmethod
    def synchronize(device=None) -> None:
      tpu_module = getattr(torch, "tpu", None)
      if tpu_module is not None and hasattr(tpu_module, "synchronize"):
        tpu_module.synchronize()

  return TpuDeviceInterface


def _register_gather_scatter_fallbacks() -> None:
  """Routes gather/scatter-family ops to aten fallbacks (torch_tpu eager).

  The Pallas TPU compiler (Mosaic) only supports limited (2-D) gathers, so
  arbitrary gathers/scatters cannot be code-generated into whole-array Pallas
  kernels. XLA handles them well through torch_tpu eager dispatch instead.
  Ops that Inductor decomposes anyway (e.g. embedding_dense_backward) end up
  as the index/index_put leaves below, which are covered too.
  """
  from torch._inductor import lowering as inductor_lowering

  aten = torch.ops.aten
  fallback_op_names = [
      "embedding",
      "embedding_dense_backward",
      "gather",
      "index",
      "_unsafe_index",
      "index_select",
      "index_put",
      "index_put_",
      "_unsafe_index_put",
      "_unsafe_index_put_",
      "index_add",
      "scatter",
      "scatter_add",
      "scatter_reduce",
      "masked_fill",
      "take",
  ]
  for op_name in fallback_op_names:
    op = getattr(aten, op_name, None)
    if op is None:
      continue
    try:
      inductor_lowering.make_fallback(op, warn=False, override_decomp=True)
    except Exception:  # pylint: disable=broad-except
      logging.warning(
          "tpu_inductor_pallas: could not register fallback for %s", op
      )


def _ensure_registered() -> None:
  """Registers the tpu device with Inductor's codegen registries (once)."""
  global _registered
  if _registered:
    return

  from torch._dynamo import device_interface as dynamo_device_interface
  from torch._inductor.codegen import common as codegen_common
  from torch._inductor.codegen.wrapper import PythonWrapperCodegen

  tpu_scheduling, tpu_device_op_overrides = _make_tpu_pallas_classes()

  if codegen_common.get_scheduling_for_device("tpu") is None:
    codegen_common.register_backend_for_device(
        "tpu",
        tpu_scheduling,
        PythonWrapperCodegen,
    )
  codegen_common.register_device_op_overrides(
      "tpu", tpu_device_op_overrides()
  )
  registered_interfaces = dict(
      dynamo_device_interface.get_registered_device_interfaces()
  )
  if "tpu" not in registered_interfaces:
    dynamo_device_interface.register_interface_for_device(
        "tpu", _make_tpu_device_interface()
    )
  _register_gather_scatter_fallbacks()

  # Keep fused SDPA ops out of traced graphs: the dispatcher would pick
  # torch_tpu's _scaled_dot_product_fused_attention_overrideable (or a
  # CPU-flavored flash op), whose eager implementation / output-layout
  # contract does not survive Inductor's extern-kernel handling. With the
  # fused backends off, SDPA lowers to the math path — softmax through
  # Pallas codegen, matmuls through extern mm. Process-global, but only set
  # once this backend is actually used.
  torch._C._set_sdp_use_overrideable(False)  # pylint: disable=protected-access
  torch._C._set_sdp_use_flash(False)  # pylint: disable=protected-access
  torch._C._set_sdp_use_mem_efficient(False)  # pylint: disable=protected-access
  _registered = True
  logging.info(
      "Registered Inductor Pallas codegen for the 'tpu' device"
      " (backend name: %s).",
      _BACKEND_NAME,
  )


def _build_decompositions():
  """Inductor's decomp table + math decomps for fused SDPA ops.

  Fused SDPA ops (e.g. _scaled_dot_product_flash_attention_for_cpu, which
  the SDPA dispatcher picks for the tpu device) have CPU-specific output
  layout contracts that torch_tpu's generic eager execution does not honor,
  so keeping them as extern fallbacks risks silent stride corruption in the
  backward. Decomposing to math (matmul + softmax) instead routes the
  softmax through Pallas codegen and the matmuls through extern mm.
  """
  from torch._decomp import decomposition_table
  from torch._inductor.decomposition import select_decomp_table

  decompositions = dict(select_decomp_table())
  for op, decomp in decomposition_table.items():
    if "scaled_dot_product" in str(op):
      decompositions[op] = decomp

  aten = torch.ops.aten

  def _sdpa_overrideable_decomp(
      query,
      key,
      value,
      attn_bias=None,
      dropout_p=0.0,
      is_causal=False,
      return_debug_mask=False,
      *,
      scale=None,
  ):
    """Math decomposition for the privateuse1 fused SDPA op.

    torch_tpu's _fused_sdp_choice picks
    _scaled_dot_product_fused_attention_overrideable regardless of the
    global SDP backend flags, and there is no upstream decomposition for it.
    Autograd traces through the math ops, so only the primary output needs
    real data; the flash-style bookkeeping outputs are placeholders.
    """
    del return_debug_mask
    output, _ = aten._scaled_dot_product_attention_math.default(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        dropout_mask=None,
        scale=scale,
        enable_gqa=query.size(1) != key.size(1),
    )
    batch, num_heads, seq_q, _ = query.shape
    seq_k = key.size(2)
    logsumexp = torch.empty(
        (batch, num_heads, seq_q), dtype=torch.float32, device=query.device
    )
    empty_scalar = torch.empty((), dtype=torch.long, device=query.device)
    debug_mask = torch.empty(
        (0,), dtype=query.dtype, device=query.device
    )
    return (
        output,
        logsumexp,
        empty_scalar,  # cum_seq_q
        empty_scalar,  # cum_seq_k
        seq_q,  # max_q
        seq_k,  # max_k
        empty_scalar,  # philox_seed
        empty_scalar,  # philox_offset
        debug_mask,
    )

  def _sdpa_overrideable_backward_decomp(
      grad_out,
      query,
      key,
      value,
      attn_bias,
      grad_input_mask,
      out,
      logsumexp,
      cum_seq_q,
      cum_seq_k,
      max_q,
      max_k,
      dropout_p,
      is_causal,
      philox_seed,
      philox_offset,
      *,
      scale=None,
  ):
    """Math backward for the privateuse1 fused SDPA op.

    The joint trace records the fused backward before the forward decomp
    applies, so this must be decomposed too — otherwise it would consume
    the placeholder logsumexp from the forward decomp. The softmax is
    recomputed from query/key (logsumexp is ignored).
    """
    del out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k
    del philox_seed, philox_offset
    if dropout_p != 0.0:
      raise NotImplementedError(
          "tpu_inductor_pallas SDPA backward requires dropout_p == 0"
      )
    import math as math_mod

    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    key_e = key
    value_e = value
    if num_kv_heads != num_q_heads:
      repeat = num_q_heads // num_kv_heads
      key_e = key.repeat_interleave(repeat, dim=1)
      value_e = value.repeat_interleave(repeat, dim=1)

    softmax_scale = (
        scale if scale is not None else 1.0 / math_mod.sqrt(query.size(-1))
    )
    scores = torch.matmul(query, key_e.transpose(-2, -1)) * softmax_scale
    if attn_bias is not None:
      scores = scores + attn_bias
    if is_causal:
      seq_q = query.size(-2)
      seq_k = key_e.size(-2)
      causal_mask = torch.ones(
          seq_q, seq_k, dtype=torch.bool, device=query.device
      ).triu(1)
      scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(query.dtype)

    grad_value_e = torch.matmul(probs.transpose(-2, -1), grad_out)
    grad_probs = torch.matmul(grad_out, value_e.transpose(-2, -1))
    grad_scores = probs * (
        grad_probs - (grad_probs * probs).sum(dim=-1, keepdim=True)
    )
    grad_scores = grad_scores * softmax_scale
    grad_query = torch.matmul(grad_scores, key_e)
    grad_key_e = torch.matmul(grad_scores.transpose(-2, -1), query)

    if num_kv_heads != num_q_heads:
      repeat = num_q_heads // num_kv_heads
      batch, _, seq_k, head_dim = key.shape
      grad_key = grad_key_e.view(
          batch, num_kv_heads, repeat, seq_k, head_dim
      ).sum(dim=2)
      grad_value = grad_value_e.view(
          batch, num_kv_heads, repeat, seq_k, head_dim
      ).sum(dim=2)
    else:
      grad_key = grad_key_e
      grad_value = grad_value_e

    grad_bias = None
    if len(grad_input_mask) > 3 and grad_input_mask[3] and attn_bias is not None:
      grad_bias = grad_scores / softmax_scale
    return (grad_query, grad_key, grad_value, grad_bias)

  overrideable_op = getattr(
      aten, "_scaled_dot_product_fused_attention_overrideable", None
  )
  if overrideable_op is not None:
    decompositions[overrideable_op.default] = _sdpa_overrideable_decomp
  overrideable_bwd_op = getattr(
      aten, "_scaled_dot_product_fused_attention_overrideable_backward", None
  )
  if overrideable_bwd_op is not None:
    decompositions[overrideable_bwd_op.default] = (
        _sdpa_overrideable_backward_decomp
    )
  return decompositions


def tpu_inductor_pallas(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    **kwargs,
):
  """`torch.compile` backend: Inductor with Pallas codegen on TPU."""
  del kwargs  # Inductor mode/options are controlled via config patches.
  _ensure_registered()

  from torch._inductor import compile_fx

  return compile_fx.compile_fx(
      graph_module,
      example_inputs,
      config_patches=dict(_INDUCTOR_CONFIG_PATCHES),
      decompositions=_build_decompositions(),
  )


registry.register_backend(compiler_fn=tpu_inductor_pallas, name=_BACKEND_NAME)
