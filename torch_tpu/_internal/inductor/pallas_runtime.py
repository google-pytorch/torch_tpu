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

"""Runtime for Inductor-generated Pallas kernels on torch_tpu.

The Inductor Pallas backend (`pallas_backend.py`) generates modules of the
form::

    def <kernel>_kernel(out_ref0, in_ref0, ...): ...          # Pallas kernel
    @functools.partial(jax.jit, static_argnums=(0, 1), donate_argnums=())
    def <kernel>_jit_wrapper(out_shapes, out_dtypes, in0, ...): ...
    def <kernel>_main(out_ptr0, in_ptr0, ..., stream=None):
        ... pallas_runtime.run_kernel(<kernel>_jit_wrapper, ...) ...

`run_kernel` executes the jitted Pallas wrapper on torch_tpu TPU tensors by
wrapping it in a `JaxCallable` (jax.export trace -> serialized StableHLO ->
torch_tpu PJRT execution). The JaxCallable is cached per generated wrapper
function; JaxCallable itself additionally caches per input-shape
specialization, so steady-state calls are a dict lookup plus one executable
dispatch.
"""

from typing import Any, Callable, Sequence

import torch
from torch_tpu._internal.pallas import pallas as tpu_pallas

# Mosaic (the Pallas TPU compiler) has no 64-bit support, and JAX itself
# canonicalizes 64-bit types away unless jax_enable_x64 is set. Follow the
# same convention at the kernel boundary: 64-bit tensors are downcast on the
# way in and results are upcast by the final copy_ into the Inductor-allocated
# output buffer. Index-like values in these kernels (embedding rows, sequence
# positions, class counts) comfortably fit 32 bits.
_DTYPE_CANONICALIZATION = {
    torch.int64: torch.int32,
    torch.uint64: torch.uint32,
    torch.float64: torch.float32,
    torch.complex128: torch.complex64,
}


def _canonical_dtype(dtype: torch.dtype) -> torch.dtype:
  return _DTYPE_CANONICALIZATION.get(dtype, dtype)

# Keyed by id() of the generated jit wrapper function. The generated module
# (and thus the wrapper function object) is cached for the lifetime of the
# compiled model, so id() reuse is not a concern here.
_JAX_CALLABLES: dict[int, tpu_pallas.JaxCallable] = {}


def _get_callable(
    jit_fn: Callable[..., Any], kernel_name: str
) -> tpu_pallas.JaxCallable:
  key = id(jit_fn)
  jax_callable = _JAX_CALLABLES.get(key)
  if jax_callable is None:
    jax_callable = tpu_pallas.JaxCallable(
        name=f"inductor_{kernel_name}",
        jit_fn=jit_fn,
        trace_key=f"inductor_{kernel_name}_{key}",
        static_argnums=(0, 1),
    )
    _JAX_CALLABLES[key] = jax_callable
  return jax_callable


def run_kernel(
    jit_fn: Callable[..., Any],
    kernel_name: str,
    outs: Sequence[torch.Tensor],
    ins: Sequence[torch.Tensor],
    out_canonical: Sequence[Any] = (),
    in_canonical: Sequence[Any] = (),
) -> None:
  """Executes a generated Pallas kernel and fills the output buffers.

  Args:
    jit_fn: The generated `<kernel>_jit_wrapper` (a `jax.jit`-wrapped
      `pallas_call` taking `(out_shapes, out_dtypes, *inputs)` with
      `static_argnums=(0, 1)`).
    kernel_name: The generated kernel's name, for debugging/cache keys.
    outs: Inductor-allocated output buffers (written in place).
    ins: Input tensors, in the kernel's `in_ptr*` order.
    out_canonical: Per-output canonical shape over the kernel's iteration
      space, or None to use the buffer's own shape. Computed at codegen time
      by TpuPallasKernel; the Pallas kernel's refs carry these shapes, and
      reshaping here (free torch views) avoids in-kernel reshapes that the
      Mosaic compiler cannot lower.
    in_canonical: Same, for inputs.
  """
  jax_callable = _get_callable(jit_fn, kernel_name)

  if not out_canonical:
    out_canonical = (None,) * len(outs)
  if not in_canonical:
    in_canonical = (None,) * len(ins)

  def kernel_shape(t, canon):
    if canon is None:
      return tuple(t.shape)
    if canon[0] == "as_strided":
      return tuple(canon[4])
    return tuple(canon)

  def prepare_input(t, canon):
    # 64-bit tensors are downcast at the kernel boundary (see
    # _canonical_dtype); outputs are upcast again by the final copy_.
    dtype = _canonical_dtype(t.dtype)
    if canon is not None and canon[0] == "as_strided":
      # Strided/broadcast access pattern recorded at codegen time: read the
      # elements in kernel iteration order (free view + one XLA copy).
      _, sizes, strides, offset, shape = canon
      view = torch.as_strided(
          t, sizes, strides, t.storage_offset() + offset
      )
      return view.contiguous().to(dtype).reshape(shape)
    t = t.contiguous().to(dtype)
    return t.reshape(canon) if canon is not None else t

  out_shapes = tuple(
      kernel_shape(t, canon)
      for t, canon in zip(outs, out_canonical, strict=True)
  )
  out_dtypes = tuple(
      tpu_pallas.TORCH_TO_JAX_DTYPE_MAP[_canonical_dtype(t.dtype)]
      for t in outs
  )
  ins = tuple(
      prepare_input(t, canon)
      for t, canon in zip(ins, in_canonical, strict=True)
  )

  results = jax_callable(out_shapes, out_dtypes, *ins)
  if isinstance(results, torch.Tensor):
    results = (results,)

  for out, canon, result in zip(outs, out_canonical, results, strict=True):
    if canon is not None and canon[0] == "as_strided":
      # The kernel produced values in iteration order; scatter them into
      # the real buffer through the recorded strided view.
      _, sizes, strides, offset, _ = canon
      view = torch.as_strided(
          out, sizes, strides, out.storage_offset() + offset
      )
      view.copy_(result.reshape(sizes))
    elif out.is_contiguous():
      out.copy_(result.reshape(out.shape))
    else:
      # Full-array kernel stores write in the output buffer's memory order.
      # Inductor allocates dense (possibly permuted) buffers, so a flat
      # storage-ordered view receives the elements correctly.
      flat = torch.as_strided(
          out, (out.numel(),), (1,), storage_offset=out.storage_offset()
      )
      flat.copy_(result.reshape(-1))
