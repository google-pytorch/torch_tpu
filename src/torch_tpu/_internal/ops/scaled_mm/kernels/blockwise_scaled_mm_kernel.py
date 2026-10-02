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

"""Native block-scaled FP8 matmul Pallas kernel for `torch._scaled_mm_v2`.

Keeps operands in FP8 and performs block-by-block matmul on the MXU (FP8 MAC,
FP32 accumulate), applying per-block scales inside the K-loop in VMEM.

Contract (matches GPU `_scaled_mm_v2` block recipes):
  * Operands `a = [M, K]` and `b = [K, N]` are quantized FP8 (`float8_e4m3fn`
    or `float8_e5m2`).
  * `scale_a = [M, K // block_size]`, `scale_b = [K // block_size, N]` in
    logical (unswizzled) layout, one scale per K-block.
  * `block_size` parametrizes the K-block (e.g., 32 for MXFP8).

Math per output element:
  out[i, j] = sum_b Sa[i, b] * Sb[b, j] * sum_{k in block b} A[i, k] * B[k, j]
"""

import contextlib
import functools
import inspect
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

pallas_export_experimental = pl.pallas_export_experimental

# Default static tile dimensions.
_DEFAULT_TILE_M = 256
_DEFAULT_TILE_N = 256
_DEFAULT_TILE_K = 512

# TPU f32 VREG tiling granularities (sublane=8, lane=128).
_VREG_SUBLANE_TILE = 8
_VREG_LANE_TILE = 128


def _next_multiple(x: int, multiple: int) -> int:
  rem = x % multiple
  return x if rem == 0 else x + multiple - rem


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
    transpose_rhs: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int, int, int, int]:
  """Zero-pads the operands and block scales up to exact tile multiples.

  The static Pallas grid needs every dimension to be an exact multiple of its
  tile. Zero padding is safe: padded rows/cols contribute 0 to the matmul and
  are
  sliced off after the kernel, and each padded K-block is all-zero so it adds
  nothing to the K-sum. K padding grows the scales by the matching number of
  blocks (block count = padded_k // block_size).

  Args:
    a: LHS `[M, K]`.
    b: RHS `[K, N]`.
    scale_a: LHS block scales `[M, K // block_size]`.
    scale_b: RHS block scales `[K // block_size, N]`.
    tile_m: Output row tile.
    tile_n: Output column tile.
    tile_k: K tile (a multiple of block_size).
    block_size: K-block size.

  Returns:
    The padded `(a, b, scale_a, scale_b)` followed by `padded_m, padded_n,
    padded_k, n_blocks_k` (the padded K-block count).
  """
  m, k = a.shape
  n = b.shape[0] if transpose_rhs else b.shape[1]
  blocks_k = scale_a.shape[1]
  steps_k = tile_k // block_size
  padded_m = _next_multiple(m, tile_m)
  padded_n = _next_multiple(n, tile_n)
  n_blocks_k = _next_multiple(blocks_k, steps_k)
  padded_k = n_blocks_k * block_size

  if padded_m != m or padded_k != k:
    a = jnp.pad(a, ((0, padded_m - m), (0, padded_k - k)))
  if transpose_rhs:
    if padded_n != n or padded_k != k:
      b = jnp.pad(b, ((0, padded_n - n), (0, padded_k - k)))
  elif padded_k != k or padded_n != n:
    b = jnp.pad(b, ((0, padded_k - k), (0, padded_n - n)))
  if padded_m != m or n_blocks_k != blocks_k:
    scale_a = jnp.pad(scale_a, ((0, padded_m - m), (0, n_blocks_k - blocks_k)))
  if n_blocks_k != blocks_k or padded_n != n:
    scale_b = jnp.pad(scale_b, ((0, n_blocks_k - blocks_k), (0, padded_n - n)))

  return a, b, scale_a, scale_b, padded_m, padded_n, padded_k, n_blocks_k


def _blockwise_scaled_mm_kernel(
    a_ref,  # [tile_m, tile_k] fp8
    b_ref,  # [tile_k, tile_n] fp8 (K-major) or [tile_n, tile_k] fp8 (K-minor)
    sa_ref,  # [tile_k // block_size, tile_m, 1] f32 (block-major)
    sb_ref,  # [tile_k // block_size, tile_n] f32
    out_ref,  # [tile_m, tile_n] out_dtype
    acc_ref,  # [tile_m, tile_n] f32 scratch
    *,
    block_size: int,
    steps_k: int,
    transpose_rhs: bool = False,
):
  """Accumulates one [tile_m, tile_n] output tile across K blocks."""
  pid_k = pl.program_id(2)
  is_first_step = pid_k == 0
  is_last_step = pid_k == pl.num_programs(2) - 1

  acc = jnp.zeros(out_ref.shape, jnp.float32)
  for i in range(steps_k):
    k_start = i * block_size
    k_end = k_start + block_size
    a_blk = a_ref[:, k_start:k_end]  # [tile_m, block_size] fp8
    if transpose_rhs:
      b_blk = b_ref[:, k_start:k_end]  # [tile_n, block_size] fp8
      dot = jax.lax.dot_general(
          a_blk,
          b_blk,
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.float32,
      )  # [tile_m, tile_n]
    else:
      b_blk = b_ref[k_start:k_end, :]  # [block_size, tile_n] fp8
      dot = jax.lax.dot_general(
          a_blk,
          b_blk,
          (((1,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )  # [tile_m, tile_n]
    sa = sa_ref[i].astype(jnp.float32)  # [tile_m, 1]
    sb = sb_ref[i : i + 1, :].astype(jnp.float32)  # [1, tile_n]
    acc = acc + dot * sa * sb

  @pl.when(~is_first_step)
  def _():
    acc_ref[...] = acc_ref[...] + acc

  @pl.when(is_first_step)
  def _():
    acc_ref[...] = acc

  @pl.when(is_last_step)
  def _():
    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "out_dtype",
        "tile_m",
        "tile_n",
        "tile_k",
        "transpose_rhs",
    ),
)
def blockwise_scaled_mm(
    a: jax.Array,  # [M, K] fp8
    b: jax.Array,  # [K, N] fp8
    scale_a: jax.Array,  # [M, K // block_size]
    scale_b: jax.Array,  # [K // block_size, N]
    *,
    block_size: int = 32,
    out_dtype: Any = jnp.bfloat16,
    tile_m: int = _DEFAULT_TILE_M,
    tile_n: int = _DEFAULT_TILE_N,
    tile_k: int = _DEFAULT_TILE_K,
    transpose_rhs: bool = False,
) -> jax.Array:
  """Block-scaled FP8 matmul: returns `a @ b` with per-block scales applied.

  Supports both JIT execution (with in-graph zero-padding for unaligned shapes)
  and AOT export via `jax.export` when M/N/K are constrained to tile multiples.

  Args:
    a: Quantized LHS, `[M, K]`, `float8_e4m3fn` or `float8_e5m2`.
    b: Quantized RHS, `[K, N]`, `float8_e4m3fn` or `float8_e5m2`.
    scale_a: LHS block scales, `[M, K // block_size]` (logical layout).
    scale_b: RHS block scales, `[K // block_size, N]` (logical layout).
    block_size: K-block size (32 for MXFP8).
    out_dtype: Output dtype (`bfloat16` or `float32`).
    tile_m: Output row tile (multiple of `_VREG_SUBLANE_TILE`).
    tile_n: Output column tile (multiple of `_VREG_LANE_TILE`).
    tile_k: K tile; must be a multiple of `block_size`.

  Returns:
    `[M, N]` result in `out_dtype`.
  """
  m, k = a.shape
  if transpose_rhs:
    n, k_b = b.shape
  else:
    k_b, n = b.shape
  if k_b != k:
    raise ValueError(f"contraction mismatch: a is {a.shape}, b is {b.shape}")
  if k % block_size != 0:
    raise ValueError(f"K={k} must be a multiple of block_size={block_size}")
  if tile_k % block_size != 0:
    raise ValueError(
        f"tile_k={tile_k} must be a multiple of block_size={block_size}"
    )
  if scale_a.shape != (m, k // block_size):
    raise ValueError(
        f"scale_a shape mismatch: expected ({m}, {k // block_size}), got"
        f" {scale_a.shape}"
    )
  if scale_b.shape != (k // block_size, n):
    raise ValueError(
        f"scale_b shape mismatch: expected ({k // block_size}, {n}), got"
        f" {scale_b.shape}"
    )

  # Zero-pad operands and scales up to exact tile multiples for the static grid.
  a, b, scale_a, scale_b, padded_m, padded_n, padded_k, _ = _pad_to_tiles(
      a,
      b,
      scale_a,
      scale_b,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      block_size=block_size,
      transpose_rhs=transpose_rhs,
  )

  # Store scale_a block-major [K//bs, M, 1] so sa_ref[i] is a contiguous
  # register-group slice with tile_m on sublanes.
  scale_a = jnp.swapaxes(scale_a.astype(jnp.float32), 0, 1)[:, :, None]
  scale_b = scale_b.astype(jnp.float32)

  steps_k = tile_k // block_size
  grid = (
      pl.cdiv(padded_m, tile_m),
      pl.cdiv(padded_n, tile_n),
      pl.cdiv(padded_k, tile_k),
  )

  kernel = pl.pallas_call(
      functools.partial(
          _blockwise_scaled_mm_kernel,
          block_size=block_size,
          steps_k=steps_k,
          transpose_rhs=transpose_rhs,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(
                  (tile_m, tile_k),
                  lambda mi, ni, ki: (mi, ki),
                  memory_space=pltpu.VMEM,
              ),  # a
              pl.BlockSpec(
                  (tile_n, tile_k) if transpose_rhs else (tile_k, tile_n),
                  (lambda mi, ni, ki: (ni, ki))
                  if transpose_rhs
                  else (lambda mi, ni, ki: (ki, ni)),
                  memory_space=pltpu.VMEM,
              ),  # b
              pl.BlockSpec(
                  (steps_k, tile_m, 1),
                  lambda mi, ni, ki: (ki, mi, 0),
                  memory_space=pltpu.VMEM,
              ),  # scale_a
              pl.BlockSpec(
                  (steps_k, tile_n),
                  lambda mi, ni, ki: (ki, ni),
                  memory_space=pltpu.VMEM,
              ),  # scale_b
          ],
          out_specs=pl.BlockSpec((tile_m, tile_n), lambda mi, ni, ki: (mi, ni)),
          scratch_shapes=[pltpu.VMEM((tile_m, tile_n), jnp.float32)],
          grid=grid,
      ),
      out_shape=jax.ShapeDtypeStruct((padded_m, padded_n), out_dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
  )

  out = kernel(a, b, scale_a, scale_b)
  if padded_m == m and padded_n == n:
    return out
  return out[:m, :n]


def _make_abstract_device(device_kind: str, num_cores: int, platform: str):
  if "platform" in inspect.signature(jax.sharding.AbstractDevice).parameters:
    return jax.sharding.AbstractDevice(device_kind, num_cores, platform)  # type: ignore[call-arg]
  return jax.sharding.AbstractDevice(device_kind, num_cores)  # pyrefly: ignore[missing-argument]


@contextlib.contextmanager
def _maybe_use_abstract_tpu_mesh():
  """Uses an abstract TPU mesh when exporting without a TPU (build time)."""
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    abstract_mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=_make_abstract_device("TPU v5p", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      yield
  else:
    yield


def get_shapes(
    *,
    block_size: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: Any = jnp.float8_e4m3fn,
) -> tuple[jax.ShapeDtypeStruct, ...]:
  """Symbolic input shapes `(a, b, scale_a, scale_b)` for AOT export.

  Constrains M/N/K to exact tile multiples so the exported kernel never sees a
  partial tile (the C++ op pads to satisfy this). `blocks_k` is tied to K via a
  linear constraint rather than a floordiv so it stays an integer symbol.
  """
  steps_k = tile_k // block_size
  m, n, k, blocks_k = jax.export.symbolic_shape(
      "m, n, k, blocks_k",
      constraints=(
          f"mod(m, {tile_m}) == 0",
          f"mod(n, {tile_n}) == 0",
          f"mod(k, {tile_k}) == 0",
          f"mod(blocks_k, {steps_k}) == 0",
          f"k == blocks_k * {block_size}",
      ),
  )
  return (
      jax.ShapeDtypeStruct((m, k), in_dtype),
      jax.ShapeDtypeStruct((k, n), in_dtype),
      jax.ShapeDtypeStruct((m, blocks_k), jnp.float32),
      jax.ShapeDtypeStruct((blocks_k, n), jnp.float32),
  )


def export_mlir_module(
    *,
    block_size: int = 32,
    out_dtype: Any = jnp.float32,
    in_dtype: Any = jnp.float8_e4m3fn,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 256,
) -> str:
  """Exports the kernel and returns the serialized MLIR module string."""
  a, b, scale_a, scale_b = get_shapes(
      block_size=block_size,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      in_dtype=in_dtype,
  )
  f = functools.partial(
      blockwise_scaled_mm,
      block_size=block_size,
      out_dtype=out_dtype,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
  )
  with pallas_export_experimental(dynamic_shapes=True):
    with _maybe_use_abstract_tpu_mesh():
      exp = jax.export.export(jax.jit(f), platforms=["tpu"])(
          a, b, scale_a, scale_b
      )
  return exp.mlir_module()
