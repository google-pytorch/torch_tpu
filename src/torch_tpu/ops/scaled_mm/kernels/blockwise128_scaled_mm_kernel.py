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

"""Native block-scaled FP8 matmul Pallas kernel for 128-granularity recipes.

The scales are blocked at 128 along the contraction dimension K, with a
configurable 1-or-128 granularity on M (for `a`) and N (for `b`). This covers
the `BlockWise1x128` / `BlockWise128x128` recipe family, but deliberately not
`BlockWise1x32` (mxfp8), which uses e8m0 scales and a separate lowering.

Current supported recipes:

1. (`scaled_mm_1x128_128x128`):
  * Activation `a = [M, K]` with 1x128 scaling: `scale_a = [M, K // 128]`
    (one scale per row, per 128-wide K block).
  * Weight `b = [K, N]` with 128x128 scaling: `scale_b = [K // 128, N // 128]`
    (one scale per 128x128 tile), passed in its native 2-D layout.
2. (TBD) 128x128_1x128: backward dgrad for one weight with one
activation/gradient.
3. (TBD) 1x128_1x128: backward wgrad pairs two activation-like tensors (dY, X)
and involves no weight at all.
"""

import contextlib
import functools
import inspect
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_DEFAULT_TILE_M = 512
_DEFAULT_TILE_N = 512
_DEFAULT_TILE_K = 1024

_BLOCK_SIZE = 128


def make_abstract_device(
    device_kind: str = "TPU v6e", num_cores: int = 1, platform: str = "tpu"
) -> Any:
  """Compatible across Google3 HEAD (with platform) and OSS jax==0.9.2 (without platform)."""
  if "platform" in inspect.signature(jax.sharding.AbstractDevice).parameters:
    return jax.sharding.AbstractDevice(device_kind, num_cores, platform)  # type: ignore[call-arg]
  return jax.sharding.AbstractDevice(device_kind, num_cores)  # pyrefly: ignore[missing-argument]


def _next_multiple(x: int, multiple: int) -> int:
  return pl.cdiv(x, multiple) * multiple


_ceil_div = pl.cdiv


def _expand_scale_axis(
    scale: jax.Array, block: int, *, axis: int, size: int
) -> jax.Array:
  """Expands a coarse block-level scale into a per-element scale along `axis` by repeating each entry `block` times.

  It then trims any trailing overshoot if `size` is not an exact multiple of
  `block` (or acts as a no-op when `block == 1`).

  **Example:**
  For `size = 5` and `block = 3`, input `scale = [10.0, 20.0]` is repeated to
  `[10.0, 10.0, 10.0, 20.0, 20.0, 20.0]` and trimmed to `[10.0, 10.0, 10.0,
  20.0, 20.0]`.
  """
  if block != 1:
    scale = jnp.repeat(scale, block, axis=axis)
  return jax.lax.slice_in_dim(scale, 0, size, axis=axis)


def _pad_to_tiles(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int, int, int, int]:
  # Zero-pads the operands (a, b) and their corresponding block scales (scale_a, scale_b)
  # up to exact tile multiples required by the static kernel grid.
  m, k = a.shape
  n = b.shape[1]
  padded_m = _next_multiple(m, tile_m)
  padded_n = _next_multiple(n, tile_n)
  padded_k = _next_multiple(k, tile_k)
  blocks_k = _ceil_div(k, block_size)
  n_blocks_k = padded_k // block_size

  if padded_m != m or padded_k != k:
    a = jnp.pad(a, ((0, padded_m - m), (0, padded_k - k)))
  if padded_k != k or padded_n != n:
    b = jnp.pad(b, ((0, padded_k - k), (0, padded_n - n)))
  if padded_m != m or n_blocks_k != blocks_k:
    scale_a = jnp.pad(scale_a, ((0, padded_m - m), (0, n_blocks_k - blocks_k)))
  if n_blocks_k != blocks_k or padded_n != n:
    scale_b = jnp.pad(scale_b, ((0, n_blocks_k - blocks_k), (0, padded_n - n)))

  return a, b, scale_a, scale_b, padded_m, padded_n, padded_k, n_blocks_k


def _pad_bias(bias: jax.Array, n: int, padded_n: int) -> jax.Array:
  """Zero-pads the bias up to the exact column tile multiple."""
  if padded_n != n:
    bias = jnp.pad(bias, (0, padded_n - n))
  return bias.astype(jnp.float32)


def _blockwise_scaled_mm_kernel(
    a_ref,  # [tile_m, tile_k] fp8
    b_ref,  # [tile_k, tile_n] fp8
    sa_ref,  # [tile_k // block_size, tile_m] f32 (block-major)
    sb_ref,  # [tile_k // block_size, tile_n] f32
    *args,  # optionally bias_ref [tile_n] f32, then out_ref [tile_m, tile_n], acc_ref [tile_m, tile_n]
    block_size: int,
    steps_k: int,
    has_bias: bool = False,
):
  pid_k = pl.program_id(2)
  is_first_step = pid_k == 0
  is_last_step = pid_k == pl.num_programs(2) - 1

  if has_bias:
    bias_ref, out_ref, acc_ref = args
  else:
    bias_ref = None
    out_ref, acc_ref = args

  # Accumulator for one K-tile; [tile_m, tile_n] from the output block.
  acc = jnp.zeros(out_ref.shape, jnp.float32)
  for i in range(steps_k):
    k_start = i * block_size
    k_end = k_start + block_size
    a_blk = a_ref[:, k_start:k_end]  # [tile_m, block_size] fp8
    b_blk = b_ref[k_start:k_end, :]  # [block_size, tile_n] fp8
    dot = jax.lax.dot_general(
        a_blk,
        b_blk,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # [tile_m, tile_n]
    sa = sa_ref[i][:, None]  # [tile_m, 1] f32
    sb = sb_ref[i : i + 1, :]  # [1, tile_n] f32
    acc = acc + dot * sa * sb

  @pl.when(~is_first_step)
  def _():
    acc_ref[...] = acc_ref[...] + acc

  @pl.when(is_first_step)
  def _():
    acc_ref[...] = acc

  @pl.when(is_last_step)
  def _():
    res = acc_ref[...]
    if bias_ref is not None:
      res = res + bias_ref[...][None, :].astype(jnp.float32)
    out_ref[...] = res.astype(out_ref.dtype)


def _blockwise128_scaled_mm(
    a: jax.Array,  # [M, K] fp8_e4m3fn
    b: jax.Array,  # [K, N] fp8_e4m3fn
    scale_a: jax.Array,  # [ceil(M / scale_block_m), K // 128]
    scale_b: jax.Array,  # [K // 128, ceil(N / scale_block_n)]
    *,
    scale_block_m: int,  # Row block covered by each `scale_a` entry (1 or 128)
    scale_block_n: int,  # Column block covered by each `scale_b` entry (1 or 128)
    out_dtype: Any,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    bias: jax.Array | None = None,
) -> jax.Array:

  m, k = a.shape
  _, n = b.shape

  scale_a = _expand_scale_axis(scale_a, scale_block_m, axis=0, size=m)
  scale_b = _expand_scale_axis(scale_b, scale_block_n, axis=1, size=n)

  # Zero-pad operands and scales up to exact tile multiples for the static grid.
  a, b, scale_a, scale_b, padded_m, padded_n, padded_k, _ = _pad_to_tiles(
      a,
      b,
      scale_a,
      scale_b,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      block_size=_BLOCK_SIZE,
  )

  scale_a = scale_a.astype(jnp.float32)
  scale_b = scale_b.astype(jnp.float32)
  if bias is not None:
    bias = _pad_bias(bias, n, padded_n)

  # Store scale_a block-major and rank-2 ([M, K//bs] -> [K//bs, M]) so applying
  # the per-row scale needs no per-K-block cross-lane broadcast. The row-major (before swap)
  # layout paid an XLU `vperm` (plus `vrot.slane`/`vsel`) every block to shuffle
  # each row scale from a lane onto its sublane, which is expensive.
  scale_a = jnp.swapaxes(scale_a, 0, 1)

  steps_k = tile_k // _BLOCK_SIZE
  grid = (padded_m // tile_m, padded_n // tile_n, padded_k // tile_k)

  # Cost estimate enables latency-hiding scheduling and XProf/roofline FLOPs
  # reporting. Only applied for concrete-shape tracing (benchmarks/direct
  # callers); dropped under symbolic export (the precompiled op).
  out_itemsize = jnp.dtype(out_dtype).itemsize
  blocks_k = _ceil_div(k, _BLOCK_SIZE)
  scale_a_elems = _ceil_div(m, scale_block_m) * blocks_k
  scale_b_elems = blocks_k * _ceil_div(n, scale_block_n)
  try:
    cost_estimate = pl.CostEstimate(
        # GEMM FLOPs (2 per MAC); scale muls folded in are ~1/128 smaller, so omitted.
        flops=2 * m * n * k,
        transcendentals=0,  # no exp/log/rsqrt in this kernel
        # Ideal HBM traffic: fp8 a/b (1 byte), f32 scales (4 bytes), and output.
        bytes_accessed=(
            m * k
            + k * n
            + 4 * (scale_a_elems + scale_b_elems)
            + out_itemsize * m * n
            + (4 * n if bias is not None else 0)
        ),
    )
  except (
      # Dynamic or symbolic shapes (e.g. during export) cannot be converted to
      # concrete integers for arithmetic; gracefully fall back to None.
      jax.errors.ConcretizationTypeError,
      getattr(
          jax.core,
          "InconclusiveDimensionOperation",
          type("Dummy", (Exception,), {}),
      ),
      ValueError,
  ):
    cost_estimate = None

  in_specs = [
      pl.BlockSpec(
          (tile_m, tile_k),
          lambda mi, ni, ki: (mi, ki),
          memory_space=pltpu.VMEM,
      ),  # a
      pl.BlockSpec(
          (tile_k, tile_n),
          lambda mi, ni, ki: (ki, ni),
          memory_space=pltpu.VMEM,
      ),  # b
      pl.BlockSpec(
          (steps_k, tile_m),
          lambda mi, ni, ki: (ki, mi),
          memory_space=pltpu.VMEM,
      ),  # scale_a
      pl.BlockSpec(
          (steps_k, tile_n),
          lambda mi, ni, ki: (ki, ni),
          memory_space=pltpu.VMEM,
      ),  # scale_b
  ]
  if bias is not None:
    in_specs.append(
        pl.BlockSpec(
            (tile_n,),
            lambda mi, ni, ki: (ni,),
            memory_space=pltpu.VMEM,
        )
    )

  kernel = pl.pallas_call(
      functools.partial(
          _blockwise_scaled_mm_kernel,
          block_size=_BLOCK_SIZE,
          steps_k=steps_k,
          has_bias=bias is not None,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=pl.BlockSpec((tile_m, tile_n), lambda mi, ni, ki: (mi, ni)),
          scratch_shapes=[pltpu.VMEM((tile_m, tile_n), jnp.float32)],
          grid=grid,
      ),
      out_shape=jax.ShapeDtypeStruct((padded_m, padded_n), out_dtype),
      cost_estimate=cost_estimate,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
  )

  if bias is not None:
    out = kernel(a, b, scale_a, scale_b, bias)
  else:
    out = kernel(a, b, scale_a, scale_b)
  return out[:m, :n]


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_dtype",
        "tile_m",
        "tile_n",
        "tile_k",
    ),
)
def scaled_mm_1x128_128x128(
    a: jax.Array,  # [M, K] fp8_e4m3fn
    b: jax.Array,  # [K, N] fp8_e4m3fn
    scale_a: jax.Array,  # [M, ceil(K / 128)]
    scale_b: jax.Array,  # [ceil(K / 128), ceil(N / 128)]
    *,
    out_dtype: jnp.dtype = jnp.bfloat16,
    tile_m: int = _DEFAULT_TILE_M,
    tile_n: int = _DEFAULT_TILE_N,
    tile_k: int = _DEFAULT_TILE_K,
    bias: jax.Array | None = None,
) -> jax.Array:
  return _blockwise128_scaled_mm(
      a,
      b,
      scale_a,
      scale_b,
      scale_block_m=1,
      scale_block_n=_BLOCK_SIZE,
      out_dtype=out_dtype,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      bias=bias,
  )


@contextlib.contextmanager
def _maybe_use_abstract_tpu_mesh():
  """Uses an abstract TPU mesh when exporting without a TPU (build time)."""
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    abstract_mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=make_abstract_device(),
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      yield
  else:
    yield


def get_shapes(
    *,
    in_dtype: Any = jnp.float8_e4m3fn,
    has_bias: bool = False,
) -> tuple[tuple[jax.ShapeDtypeStruct, ...], jax.ShapeDtypeStruct | None]:
  """Symbolic input shapes for AOT export, as `(positional_args, bias)`.

  Only K is constrained, to a multiple of `_BLOCK_SIZE`, because that is the
  sole shape precondition the C++ op enforces. M and N are left free: this
  kernel zero-pads them up to tile multiples itself and slices the result back
  down, so the exported module must stay valid for partial trailing tiles.
  """
  m, n, k = jax.export.symbolic_shape(
      "m, n, k",
      constraints=(f"k >= {_BLOCK_SIZE}", f"mod(k, {_BLOCK_SIZE}) == 0"),
  )
  blocks_k = k // _BLOCK_SIZE
  blocks_n = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
  args = (
      jax.ShapeDtypeStruct((m, k), in_dtype),
      jax.ShapeDtypeStruct((k, n), in_dtype),
      jax.ShapeDtypeStruct((m, blocks_k), jnp.float32),
      jax.ShapeDtypeStruct((blocks_k, blocks_n), jnp.float32),
  )
  bias = jax.ShapeDtypeStruct((n,), jnp.float32) if has_bias else None
  return args, bias


def export_mlir_module(
    *,
    block_size: int = _BLOCK_SIZE,
    in_dtype: Any = jnp.float8_e4m3fn,
    out_dtype: Any = jnp.bfloat16,
    tile_m: int = _DEFAULT_TILE_M,
    tile_n: int = _DEFAULT_TILE_N,
    tile_k: int = _DEFAULT_TILE_K,
    has_bias: bool = False,
) -> str:
  """Exports the kernel and returns the serialized MLIR module string."""
  if block_size != _BLOCK_SIZE:
    raise ValueError(
        "The 1x128 / 128x128 recipe family only defines scale layouts at a"
        f" block size of {_BLOCK_SIZE}, got {block_size}."
    )
  args, bias = get_shapes(in_dtype=in_dtype, has_bias=has_bias)
  kwargs: dict[str, Any] = dict(
      out_dtype=out_dtype,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
  )
  if bias is not None:
    kwargs["bias"] = bias

  with _maybe_use_abstract_tpu_mesh():
    exported = jax.export.export(scaled_mm_1x128_128x128, platforms=["tpu"])(
        *args, **kwargs
    )
  return exported.mlir_module()
