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

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torch_tpu._internal.ops.scaled_mm.kernels import blockwise128_scaled_mm_kernel as kernels
from torch_tpu._internal.ops.scaled_mm.kernels import scaled_mm_generate

_BLOCK_SIZE = 128

# CheckAbslTestCase suppression: these are pure host-logic tests with fixed
# inputs and no RNG, so plain parameterized test classes are safe here.
_TestCase = parameterized.TestCase  # ABSLTEST_OK=pure host logic; no RNG


class ExpandScaleAxisTest(_TestCase):
  """Unit tests for `_expand_scale_axis` (pure host logic; no TPU required)."""

  def test_block_one_is_noop(self):
    scale = jnp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    out = kernels._expand_scale_axis(scale, 1, axis=1, size=4)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(scale))

  def test_expand_axis1_full_bands(self):
    # [2, 2] -> repeat 128 along N -> [2, 256].
    scale = jnp.asarray(np.array([[1.0, 2.0], [3.0, 4.0]], np.float32))
    out = np.asarray(kernels._expand_scale_axis(scale, 128, axis=1, size=256))
    self.assertEqual(out.shape, (2, 256))
    np.testing.assert_array_equal(out[:, :128], np.tile([[1.0], [3.0]], 128))
    np.testing.assert_array_equal(out[:, 128:], np.tile([[2.0], [4.0]], 128))

  def test_expand_axis1_partial_trailing_band(self):
    # N=200 -> ceil(200/128)=2 blocks -> expand to 256 -> trim to 200.
    scale = jnp.asarray(np.array([[1.0, 2.0], [3.0, 4.0]], np.float32))
    out = np.asarray(kernels._expand_scale_axis(scale, 128, axis=1, size=200))
    self.assertEqual(out.shape, (2, 200))
    np.testing.assert_array_equal(out[:, :128], np.tile([[1.0], [3.0]], 128))
    # Only 200-128=72 elements of the second (partial) band survive the trim.
    np.testing.assert_array_equal(out[:, 128:], np.tile([[2.0], [4.0]], 72))

  def test_expand_axis0_partial_trailing_band(self):
    # M=130 -> ceil(130/128)=2 blocks -> expand to 256 -> trim to 130.
    scale = jnp.asarray(np.array([[5.0], [6.0]], np.float32))
    out = np.asarray(kernels._expand_scale_axis(scale, 128, axis=0, size=130))
    self.assertEqual(out.shape, (130, 1))
    np.testing.assert_array_equal(out[:128], np.full((128, 1), 5.0))
    np.testing.assert_array_equal(out[128:], np.full((2, 1), 6.0))


class CeilDivTest(_TestCase):

  @parameterized.named_parameters(
      ("exact", 256, 128, 2),
      ("partial", 200, 128, 2),
      ("one", 1, 128, 1),
      ("block_one", 130, 1, 130),
  )
  def test_ceil_div(self, x, divisor, expected):
    self.assertEqual(kernels._ceil_div(x, divisor), expected)


class PadToTilesTest(_TestCase):
  """Unit tests for `_pad_to_tiles` at block=128 (pure host logic; no TPU)."""

  @parameterized.named_parameters(
      # (name, m, n, k, tile_m, tile_n, tile_k)
      ("already_aligned", 128, 128, 256, 128, 128, 256),
      ("pad_m_only", 100, 128, 256, 128, 128, 256),
      ("pad_n_only", 128, 100, 256, 128, 128, 256),
      ("pad_k_grows_scales", 128, 128, 128, 128, 128, 256),
      ("pad_unaligned_k", 128, 128, 200, 128, 128, 256),
      ("pad_all", 100, 100, 128, 128, 128, 256),
  )
  def test_pad_shapes_and_fill(self, m, n, k, tile_m, tile_n, tile_k):
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((m, k)).astype(np.float32))
    b = jnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
    blocks_k = kernels._ceil_div(k, _BLOCK_SIZE)
    # Per-row / per-col (post-expansion) scale layout.
    sa = jnp.asarray(rng.standard_normal((m, blocks_k)).astype(np.float32))
    sb = jnp.asarray(rng.standard_normal((blocks_k, n)).astype(np.float32))

    pa, pb, psa, psb, pm, pn, pk, nbk = kernels._pad_to_tiles(
        a,
        b,
        sa,
        sb,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_size=_BLOCK_SIZE,
    )

    self.assertEqual(pm % tile_m, 0)
    self.assertEqual(pn % tile_n, 0)
    self.assertEqual(pk % tile_k, 0)
    self.assertGreaterEqual(pm, m)
    self.assertGreaterEqual(pn, n)
    self.assertGreaterEqual(pk, k)
    self.assertEqual(nbk, pk // _BLOCK_SIZE)

    self.assertEqual(pa.shape, (pm, pk))
    self.assertEqual(pb.shape, (pk, pn))
    self.assertEqual(psa.shape, (pm, nbk))
    self.assertEqual(psb.shape, (nbk, pn))

    pa, pb, psa, psb = map(np.asarray, (pa, pb, psa, psb))
    np.testing.assert_array_equal(pa[:m, :k], np.asarray(a))
    np.testing.assert_array_equal(pb[:k, :n], np.asarray(b))
    np.testing.assert_array_equal(psa[:m, :blocks_k], np.asarray(sa))
    np.testing.assert_array_equal(psb[:blocks_k, :n], np.asarray(sb))
    self.assertEqual(float(pa[m:, :].sum() + pa[:, k:].sum()), 0.0)
    self.assertEqual(float(pb[k:, :].sum() + pb[:, n:].sum()), 0.0)
    self.assertEqual(float(psa[m:, :].sum() + psa[:, blocks_k:].sum()), 0.0)
    self.assertEqual(float(psb[blocks_k:, :].sum() + psb[:, n:].sum()), 0.0)


class PadBiasTest(_TestCase):
  """Unit tests for bias padding logic (pure host logic; no TPU)."""

  def test_bias_padding(self):
    bias = jnp.ones((100,), dtype=jnp.float32)
    padded_bias = kernels._pad_bias(bias, n=100, padded_n=128)
    self.assertEqual(padded_bias.shape, (128,))
    np.testing.assert_array_equal(np.asarray(padded_bias)[:100], np.ones(100))
    np.testing.assert_array_equal(np.asarray(padded_bias)[100:], np.zeros(28))


class ExportSymbolicShapesTest(_TestCase):
  """Tests the AOT export path the genrules use to precompile the kernel."""

  # `float16` is deliberately absent: the ATen op rejects it because a
  # `float16` result lowers to `tpu.pack_subelements`, which Mosaic cannot
  # legalize on some TPU generations.
  @parameterized.product(
      out_dtype=(jnp.bfloat16, jnp.float32),
      has_bias=(False, True),
  )
  def test_export_mlir_module(self, out_dtype, has_bias):
    mlir_text = kernels.export_mlir_module(
        out_dtype=out_dtype, has_bias=has_bias
    )
    self.assertIn("@main", mlir_text)
    # M, N and K stay symbolic, so the entry point keeps dynamic dimensions.
    self.assertIn("tensor<?x", mlir_text)

  def test_export_rejects_non_128_block_size(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Host validation test.
        ValueError, "block size"
    ):
      kernels.export_mlir_module(block_size=32)

  def test_get_shapes_bias_is_n_vector(self):
    args, bias = kernels.get_shapes(has_bias=True)
    a, b, scale_a, scale_b = args
    m, k = a.shape
    n = b.shape[1]
    self.assertEqual(scale_a.shape, (m, k // _BLOCK_SIZE))
    self.assertEqual(scale_b.shape[0], k // _BLOCK_SIZE)
    self.assertEqual(bias.shape, (n,))
    self.assertEqual(bias.dtype, jnp.float32)

  def test_get_shapes_without_bias(self):
    _, bias = kernels.get_shapes(has_bias=False)
    self.assertIsNone(bias)


class RecipeRegistryTest(_TestCase):
  """Tests that this kernel is reachable through the shared recipe registry."""

  def test_recipe_is_registered(self):
    self.assertIn("blockwise_128", scaled_mm_generate.get_supported_recipes())

  @parameterized.product(
      out_dtype_name=("bfloat16", "float32"),
      has_bias=(False, True),
  )
  def test_generate_kernel_module(self, out_dtype_name, has_bias):
    mlir_text = scaled_mm_generate.generate_kernel_module(
        recipe="blockwise_128",
        in_dtype_name="float8_e4m3fn",
        out_dtype_name=out_dtype_name,
        block_size=_BLOCK_SIZE,
        tile_m=512,
        tile_n=512,
        tile_k=1024,
        has_bias=has_bias,
    )
    self.assertIn("@main", mlir_text)


if __name__ == "__main__":
  absltest.main()
