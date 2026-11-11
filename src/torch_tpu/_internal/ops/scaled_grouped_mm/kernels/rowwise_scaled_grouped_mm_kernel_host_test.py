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

"""Host-side tests for the rowwise-scaled grouped FP8 matmul kernel.

These run on CPU: the visit schedule against a brute-force model, argument
checks, the AOT export and its generator, and kernel numerics in the Pallas
interpreter. On-device numerics are in rowwise_scaled_grouped_mm_kernel_test.
"""

from collections.abc import Sequence
import functools
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torch_tpu._internal.ops.scaled_grouped_mm.kernels import rowwise_scaled_grouped_mm_kernel as kernel
from torch_tpu._internal.ops.scaled_grouped_mm.kernels import scaled_grouped_mm_generate as generate
from tests import seed_test_utils

# Group layouts as (rows of `a`, rows per group). Rows past the last group must
# come out zero.
_LAYOUTS = {
    "even": (1024, (256, 256, 256, 256)),
    "uneven_unaligned": (1000, (100, 300, 1, 512)),
    "empty_first_middle_last": (1000, (0, 500, 0, 300, 0)),
    "single_group": (640, (640,)),
    "all_empty": (384, (0, 0, 0)),
    "tiny_m": (50, (20, 30)),
}


def _offs(sizes) -> np.ndarray:
  return np.cumsum(sizes, dtype=np.int32)


def _expected_visits(
    offs: np.ndarray, num_rows: int, tile_m: int
) -> Sequence[tuple[int, int]]:
  """Brute-force (group, m-tile) visits in execution order."""
  bounds = [0, *np.maximum.accumulate(offs).tolist(), num_rows]
  visits = []
  for group in range(len(bounds) - 1):
    start, stop = bounds[group], bounds[group + 1]
    if stop > start:
      first, last = start // tile_m, (stop - 1) // tile_m
      visits.extend((group, tile) for tile in range(first, last + 1))
  return visits


def _row_owners(offs: np.ndarray, num_rows: int) -> np.ndarray:
  """Group owning each row (torch semantics); `len(offs)` past the last group."""
  return np.searchsorted(offs, np.arange(num_rows), side="right")


class GroupMetadataTest(seed_test_utils.RepeatableTest):
  """The visit schedule computed from `offs`."""

  def _check_schedule(self, offs: np.ndarray, num_rows: int, tile_m: int):
    group_offsets, group_ids, m_tile_ids, num_visits = (
        kernel.make_group_metadata(
            jnp.asarray(offs), num_rows=num_rows, tile_m=tile_m
        )
    )
    group_offsets, group_ids, m_tile_ids = map(
        np.asarray, (group_offsets, group_ids, m_tile_ids)
    )
    num_visits = int(num_visits)
    num_groups = len(offs)
    tiles_m = -(-num_rows // tile_m)

    visits = list(zip(group_ids[:num_visits], m_tile_ids[:num_visits]))
    self.assertEqual(visits, _expected_visits(offs, num_rows, tile_m))

    # Every row is written by exactly one visit: that of the group owning it.
    written = np.full(num_rows, -1)
    for group, tile in visits:
      rows = np.arange(tile * tile_m, min((tile + 1) * tile_m, num_rows))
      rows = rows[
          (rows >= group_offsets[group]) & (rows < group_offsets[group + 1])
      ]
      self.assertTrue(np.all(written[rows] == -1), f"rows rewritten: {rows}")
      written[rows] = group
    np.testing.assert_array_equal(written, _row_owners(offs, num_rows))

    # A tile's visits are adjacent, so its output block stays resident.
    tiles = m_tile_ids[:num_visits]
    num_runs = 1 + np.count_nonzero(tiles[1:] != tiles[:-1])
    self.assertEqual(num_runs, len(set(tiles.tolist())))

    # Trailing (never executed) visits still name valid blocks.
    self.assertEqual(group_ids.shape, (tiles_m + num_groups,))
    self.assertTrue(np.all(group_ids[num_visits:] == num_groups))
    self.assertTrue(np.all((m_tile_ids >= 0) & (m_tile_ids < tiles_m)))

  @parameterized.product(layout=tuple(_LAYOUTS), tile_m=(128, 256, 512))
  def test_named_layouts(self, layout, tile_m):
    num_rows, sizes = _LAYOUTS[layout]
    self._check_schedule(_offs(sizes), num_rows, tile_m)

  def test_random_layouts(self):
    rng = np.random.default_rng(self.test_random_seed)
    for _ in range(50):
      num_groups = int(rng.integers(1, 10))
      num_rows = int(rng.integers(1, 2000))
      tile_m = int(rng.choice([128, 256, 512]))
      covered = int(rng.integers(0, num_rows + 1))
      cuts = np.sort(rng.integers(0, covered + 1, size=num_groups - 1))
      offs = np.append(cuts, covered).astype(np.int32)
      with self.subTest(offs=offs.tolist(), num_rows=num_rows, tile_m=tile_m):
        self._check_schedule(offs, num_rows, tile_m)

  def test_clamps_malformed_offs(self):
    group_offsets, *_ = kernel.make_group_metadata(
        jnp.asarray([-5, 300, 200, 5000], jnp.int32), num_rows=1000, tile_m=128
    )
    np.testing.assert_array_equal(group_offsets, [0, 0, 300, 300, 1000, 1000])


class DefaultTileTest(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      (1, 128),
      (200, 256),
      (300, 384),
      (768, 768),
      (2048, 2048),
      (4096, 2048),
      (5000, 1280),
  )
  def test_default_tile(self, dim, expected):
    self.assertEqual(kernel.default_tile(dim), expected)


class CheckArgsTest(seed_test_utils.RepeatableTest):
  """Argument validation, which runs while tracing."""

  _M, _K, _N, _G = 256, 128, 384, 2

  def _trace(self, *, tile_m=128, out_dtype=jnp.bfloat16, **overrides):
    args = dict(
        a=((self._M, self._K), jnp.float8_e4m3fn),
        b=((self._G, self._N, self._K), jnp.float8_e4m3fn),
        scale_a=((self._M,), jnp.float32),
        scale_b=((self._G, self._N), jnp.float32),
        offs=((self._G,), jnp.int32),
    )
    args.update(overrides)
    fn = functools.partial(
        kernel.rowwise_scaled_grouped_mm,
        transpose_rhs=True,
        tile_m=tile_m,
        out_dtype=out_dtype,
    )
    return jax.eval_shape(
        fn, **{k: jax.ShapeDtypeStruct(*v) for k, v in args.items()}
    )

  def test_accepts_valid_args(self):
    out = self._trace()
    self.assertEqual(out.shape, (self._M, self._N))
    self.assertEqual(out.dtype, jnp.bfloat16)

  @parameterized.named_parameters(
      ("lhs_rank", dict(a=((1, _M, _K), jnp.float8_e4m3fn)), "2-D"),
      ("lhs_dtype", dict(a=((_M, _K), jnp.bfloat16)), "float8"),
      ("rhs_dtype", dict(b=((_G, _N, _K), jnp.float32)), "float8"),
      ("k_mismatch", dict(b=((_G, _N, 64), jnp.float8_e4m3fn)), "Contraction"),
      ("scale_a_shape", dict(scale_a=((_M, 1), jnp.float32)), "scale_a"),
      ("scale_b_shape", dict(scale_b=((_G, 1), jnp.float32)), "scale_b"),
      ("offs_dtype", dict(offs=((_G,), jnp.float32)), "offs"),
      ("offs_shape", dict(offs=((_G + 1,), jnp.int32)), "offs"),
      ("tile_not_lane_aligned", dict(tile_m=100), "multiple of 128"),
      ("out_dtype", dict(out_dtype=jnp.float16), "out_dtype"),
  )
  def test_rejects(self, overrides, regex):
    with self.assertRaisesRegex(ValueError, regex):
      self._trace(**overrides)


class ExportTest(seed_test_utils.RepeatableTest):
  """The AOT export the precompiled variants are generated from."""

  @parameterized.product(
      transpose_rhs=(False, True), out_dtype=(jnp.bfloat16, jnp.float32)
  )
  def test_export_keeps_all_dims_symbolic(self, transpose_rhs, out_dtype):
    exported = kernel.export(
        transpose_rhs=transpose_rhs,
        out_dtype=out_dtype,
        tile_m=256,
        tile_k=256,
        tile_n=256,
    )
    self.assertEqual(exported.platforms, ("tpu",))
    a, b, scale_a, scale_b, offs = exported.in_avals
    m, k = a.shape
    num_groups = b.shape[0]
    n = b.shape[1] if transpose_rhs else b.shape[2]
    for dim in (m, k, n, num_groups):
      self.assertNotIsInstance(dim, int)
    self.assertEqual(scale_a.shape, (m,))
    self.assertEqual(scale_b.shape, (num_groups, n))
    self.assertEqual(offs.shape, (num_groups,))
    (out,) = exported.out_avals
    self.assertEqual(out.shape, (m, n))
    self.assertEqual(out.dtype, out_dtype)
    self.assertIn("tpu_custom_call", exported.mlir_module())

  def test_export_mlir_module(self):
    mlir = kernel.export_mlir_module(
        lhs_dtype=jnp.float8_e4m3fn,
        rhs_dtype=jnp.float8_e4m3fn,
        out_dtype=jnp.bfloat16,
        transpose_rhs=True,
        tile_m=256,
        tile_k=256,
        tile_n=256,
    )
    self.assertIsInstance(mlir, str)
    self.assertIn("module @", mlir)
    self.assertIn("tpu_custom_call", mlir)

  @parameterized.parameters(False, True)
  def test_lowers_multi_tile_k(self, transpose_rhs):
    m, k, n, g = 512, 512, 256, 4
    b_shape = (g, n, k) if transpose_rhs else (g, k, n)
    args = (
        jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct(b_shape, jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((m,), jnp.float32),
        jax.ShapeDtypeStruct((g, n), jnp.float32),
        jax.ShapeDtypeStruct((g,), jnp.int32),
    )
    with kernel.aot_target.use_abstract_tpu_mesh():
      exported = jax.export.export(
          kernel.rowwise_scaled_grouped_mm, platforms=["tpu"]
      )(
          *args,
          out_dtype=jnp.bfloat16,
          transpose_rhs=transpose_rhs,
          tile_m=256,
          tile_k=256,
          tile_n=256,
      )
    self.assertIn("tpu_custom_call", exported.mlir_module())


class GenerateTest(seed_test_utils.RepeatableTest):
  """The build-time generator behind the embedded kernel genrules."""

  _VARIANT = dict(
      lhs_dtype_name="float8_e4m3fn",
      rhs_dtype_name="float8_e4m3fn",
      out_dtype_name="bfloat16",
      transpose_rhs=True,
      tile_m=256,
      tile_k=256,
      tile_n=256,
  )

  def test_lowers_for_v7x(self):
    with mock.patch.object(
        jax.sharding, "use_abstract_mesh", wraps=jax.sharding.use_abstract_mesh
    ) as use_mesh:
      generate.generate_kernel_module(**self._VARIANT)
    use_mesh.assert_called_once()
    (mesh,), _ = use_mesh.call_args
    self.assertEqual(mesh.abstract_device.device_kind, "TPU7x")

  def test_rejects_unknown_dtype(self):
    with self.assertRaisesRegex(ValueError, "float16"):
      generate.generate_kernel_module(
          **(self._VARIANT | dict(lhs_dtype_name="float16"))
      )

  def test_main_embeds_module(self):
    out_dir = self.create_tempdir().full_path
    header = os.path.join(out_dir, "kernel_embed.h")
    implementation = os.path.join(out_dir, "kernel_embed.cc")
    with flagsaver.flagsaver(
        header=header,
        implementation=implementation,
        kernel_name="kTestKernel",
        lhs_dtype="float8_e4m3fn",
        rhs_dtype="float8_e4m3fn",
        out_dtype="float32",
        transpose_rhs=False,
        tile_m=128,
        tile_k=128,
        tile_n=128,
    ):
      generate.main([])
    with open(header) as f:
      self.assertIn("extern const unsigned char kTestKernel_data[];", f.read())
    with open(implementation) as f:
      body = f.read()
    hex_bytes = body.split("{", 1)[1].split("}", 1)[0].split(",")
    module = bytes(int(x, 16) for x in hex_bytes).decode("utf-8")
    self.assertIn("tpu_custom_call", module)
    self.assertIn("f32", module)


def _ternary(rng: np.random.Generator, shape, dtype) -> jax.Array:
  return jnp.asarray(rng.integers(-1, 2, size=shape), jnp.float32).astype(dtype)


def _pow2(rng: np.random.Generator, shape) -> jax.Array:
  return jnp.asarray(np.exp2(rng.integers(-3, 4, size=shape)), jnp.float32)


def reference(a, b, scale_a, scale_b, offs, *, transpose_rhs) -> np.ndarray:
  """f32 numpy `_scaled_grouped_mm` with rowwise scales; zeros past offs[-1]."""
  a = np.asarray(a.astype(jnp.float32))
  b = np.asarray(b.astype(jnp.float32))
  if transpose_rhs:
    b = b.transpose(0, 2, 1)
  scale_a, scale_b = np.asarray(scale_a), np.asarray(scale_b)
  out = np.zeros((a.shape[0], b.shape[2]), np.float32)
  start = 0
  for group, end in enumerate(np.asarray(offs)):
    out[start:end] = (a[start:end] @ b[group]) * (
        scale_a[start:end, None] * scale_b[group]
    )
    start = end
  return out


class InterpretNumericsTest(seed_test_utils.RepeatableTest):
  """Kernel numerics in the Pallas interpreter.

  Ternary payloads with K <= 256 and power-of-two scales make every product and
  partial sum exact in any accumulator, so results must match bit-for-bit.
  """

  @parameterized.product(
      layout=("uneven_unaligned", "empty_first_middle_last", "tiny_m"),
      transpose_rhs=(False, True),
      # (tile_m, tile_k, tile_n): several K steps and N tiles, then one each.
      tiles=((128, 128, 128), (256, 256, 384)),
  )
  def test_matches_reference(self, layout, transpose_rhs, tiles):
    num_rows, sizes = _LAYOUTS[layout]
    k, n, num_groups = 256, 384, len(sizes)
    tile_m, tile_k, tile_n = tiles
    rng = np.random.default_rng(self.test_random_seed)
    a = _ternary(rng, (num_rows, k), jnp.float8_e4m3fn)
    b_shape = (num_groups, n, k) if transpose_rhs else (num_groups, k, n)
    b = _ternary(rng, b_shape, jnp.float8_e4m3fn)
    scale_a = _pow2(rng, (num_rows,))
    scale_b = _pow2(rng, (num_groups, n))
    offs = jnp.asarray(_offs(sizes))

    out = kernel.rowwise_scaled_grouped_mm(
        a,
        b,
        scale_a,
        scale_b,
        offs,
        out_dtype=jnp.float32,
        transpose_rhs=transpose_rhs,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        interpret=True,
    )
    np.testing.assert_array_equal(
        np.asarray(out),
        reference(a, b, scale_a, scale_b, offs, transpose_rhs=transpose_rhs),
    )

  @parameterized.parameters(False, True)
  def test_unaligned_k_and_n_matches_reference(self, transpose_rhs):
    sizes = (150, 0, 120)
    num_rows, k, n, num_groups = 300, 180, 200, len(sizes)
    tile_m, tile_k, tile_n = 128, 128, 128
    rng = np.random.default_rng(self.test_random_seed)
    a = _ternary(rng, (num_rows, k), jnp.float8_e4m3fn)
    b_shape = (num_groups, n, k) if transpose_rhs else (num_groups, k, n)
    b = _ternary(rng, b_shape, jnp.float8_e4m3fn)
    scale_a = _pow2(rng, (num_rows,))
    scale_b = _pow2(rng, (num_groups, n))
    offs = jnp.asarray(_offs(sizes))

    out = kernel.rowwise_scaled_grouped_mm(
        a,
        b,
        scale_a,
        scale_b,
        offs,
        out_dtype=jnp.float32,
        transpose_rhs=transpose_rhs,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        interpret=True,
    )
    self.assertEqual(out.shape, (num_rows, n))
    np.testing.assert_array_equal(
        np.asarray(out),
        reference(a, b, scale_a, scale_b, offs, transpose_rhs=transpose_rhs),
    )


if __name__ == "__main__":
  absltest.main()
