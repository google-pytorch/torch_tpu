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

"""Host-side tests for the block-scaled FP8 matmul kernel.

These cover pure host logic -- input precondition validation, tile padding,
condition unfolding, and AOT export -- that runs without a TPU.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torch_tpu._internal.ops.scaled_mm.kernels import blockwise_scaled_mm_kernel as kernels
from torch_tpu._internal.ops.scaled_mm.kernels import scaled_mm_generate
from tests import seed_test_utils


class PreconditionValidationTest(seed_test_utils.RepeatableTest):
  """Tests that blockwise_scaled_mm raises ValueError on invalid inputs."""

  def setUp(self):
    super().setUp()
    self.m, self.n, self.k = 128, 128, 256
    self.block_size = 32
    self.a_q = jnp.zeros((self.m, self.k), dtype=jnp.float8_e4m3fn)
    self.b_q = jnp.zeros((self.k, self.n), dtype=jnp.float8_e4m3fn)
    self.sa = jnp.ones((self.m, self.k // self.block_size), dtype=jnp.float32)
    self.sb = jnp.ones((self.k // self.block_size, self.n), dtype=jnp.float32)

  def test_contraction_mismatch_raises(self):
    bad_b = jnp.zeros((self.k + 32, self.n), dtype=jnp.float8_e4m3fn)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Host validation test.
        ValueError, "contraction mismatch"
    ):
      kernels.blockwise_scaled_mm(
          self.a_q, bad_b, self.sa, self.sb, block_size=self.block_size
      )

  def test_k_not_divisible_by_block_size_raises(self):
    bad_a = jnp.zeros((self.m, 250), dtype=jnp.float8_e4m3fn)
    bad_b = jnp.zeros((250, self.n), dtype=jnp.float8_e4m3fn)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Host validation test.
        ValueError, "must be a multiple of block_size"
    ):
      kernels.blockwise_scaled_mm(
          bad_a, bad_b, self.sa, self.sb, block_size=self.block_size
      )

  def test_tile_k_not_divisible_by_block_size_raises(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Host validation test.
        ValueError, "must be a multiple of block_size"
    ):
      kernels.blockwise_scaled_mm(
          self.a_q,
          self.b_q,
          self.sa,
          self.sb,
          block_size=self.block_size,
          tile_k=250,
      )

  def test_scale_a_shape_mismatch_raises(self):
    bad_sa = jnp.ones(
        (self.m, (self.k // self.block_size) + 1), dtype=jnp.float32
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Host validation test.
        ValueError, "scale_a shape mismatch"
    ):
      kernels.blockwise_scaled_mm(
          self.a_q, self.b_q, bad_sa, self.sb, block_size=self.block_size
      )

  def test_scale_b_shape_mismatch_raises(self):
    bad_sb = jnp.ones(
        ((self.k // self.block_size) + 1, self.n), dtype=jnp.float32
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Host validation test.
        ValueError, "scale_b shape mismatch"
    ):
      kernels.blockwise_scaled_mm(
          self.a_q, self.b_q, self.sa, bad_sb, block_size=self.block_size
      )


class PadToTilesTest(seed_test_utils.RepeatableTest):
  """Unit tests for `_pad_to_tiles` (pure host logic; no TPU required)."""

  @parameterized.named_parameters(
      # (name, m, n, k, tile_m, tile_n, tile_k, block_size)
      ("already_aligned", 16, 128, 256, 16, 128, 256, 32),
      ("pad_m_only", 10, 128, 256, 16, 128, 256, 32),
      ("pad_n_only", 16, 100, 256, 16, 128, 256, 32),
      ("pad_k_grows_scales", 16, 128, 192, 16, 128, 256, 32),
      ("pad_all", 10, 100, 192, 16, 128, 256, 32),
  )
  def test_pad_shapes_and_fill(
      self, m, n, k, tile_m, tile_n, tile_k, block_size
  ):
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((m, k)).astype(np.float32))
    b = jnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
    blocks_k = k // block_size
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
        block_size=block_size,
    )

    # Padded dims are exact tile multiples, never smaller than the originals.
    self.assertEqual(pm % tile_m, 0)
    self.assertEqual(pn % tile_n, 0)
    self.assertEqual(pk % tile_k, 0)
    self.assertGreaterEqual(pm, m)
    self.assertGreaterEqual(pn, n)
    self.assertGreaterEqual(pk, k)
    self.assertEqual(nbk, pk // block_size)

    # Padded shapes line up with the reported padded dims.
    self.assertEqual(pa.shape, (pm, pk))
    self.assertEqual(pb.shape, (pk, pn))
    self.assertEqual(psa.shape, (pm, nbk))
    self.assertEqual(psb.shape, (nbk, pn))

    pa, pb, psa, psb = map(np.asarray, (pa, pb, psa, psb))
    # The original region is preserved verbatim.
    np.testing.assert_array_equal(pa[:m, :k], np.asarray(a))
    np.testing.assert_array_equal(pb[:k, :n], np.asarray(b))
    np.testing.assert_array_equal(psa[:m, :blocks_k], np.asarray(sa))
    np.testing.assert_array_equal(psb[:blocks_k, :n], np.asarray(sb))
    # Everything outside the original region is zero.
    self.assertEqual(float(pa[m:, :].sum() + pa[:, k:].sum()), 0.0)
    self.assertEqual(float(pb[k:, :].sum() + pb[:, n:].sum()), 0.0)
    self.assertEqual(float(psa[m:, :].sum() + psa[:, blocks_k:].sum()), 0.0)
    self.assertEqual(float(psb[blocks_k:, :].sum() + psb[:, n:].sum()), 0.0)

  def test_already_aligned_is_noop(self):
    a = jnp.ones((16, 256), dtype=jnp.float32)
    b = jnp.ones((256, 128), dtype=jnp.float32)
    sa = jnp.ones((16, 8), dtype=jnp.float32)
    sb = jnp.ones((8, 128), dtype=jnp.float32)
    pa, pb, psa, psb, pm, pn, pk, nbk = kernels._pad_to_tiles(
        a, b, sa, sb, tile_m=16, tile_n=128, tile_k=256, block_size=32
    )
    self.assertEqual((pm, pn, pk, nbk), (16, 128, 256, 8))
    # No padding needed, so the arrays pass through unchanged.
    self.assertIs(pa, a)
    self.assertIs(pb, b)
    self.assertIs(psa, sa)
    self.assertIs(psb, sb)


class ExportTest(seed_test_utils.RepeatableTest):
  """Tests AOT export of the kernel to StableHLO."""

  def test_export_host_e4m3(self):
    mlir_str = kernels.export_mlir_module(
        block_size=32,
        out_dtype=jnp.float32,
        in_dtype=jnp.float8_e4m3fn,
        tile_m=128,
        tile_n=128,
        tile_k=256,
    )
    self.assertIn("func.func", mlir_str)
    self.assertIn("f8E4M3FN", mlir_str)

  def test_export_host_e5m2(self):
    mlir_str = kernels.export_mlir_module(
        block_size=32,
        out_dtype=jnp.float32,
        in_dtype=jnp.float8_e5m2,
        tile_m=128,
        tile_n=128,
        tile_k=256,
    )
    self.assertIn("func.func", mlir_str)
    self.assertIn("f8E5M2", mlir_str)


class KernelGeneratorRegistryTest(seed_test_utils.RepeatableTest):
  """Unit tests for recipe-based kernel generator registry and CLI validation."""

  def test_supported_recipes_contains_mxfp8(self):
    recipes = scaled_mm_generate.get_supported_recipes()
    self.assertIn("mxfp8", recipes)

  def test_generate_mxfp8_success(self):
    mlir_str = scaled_mm_generate.generate_kernel_module(
        recipe="mxfp8",
        in_dtype_name="float8_e4m3fn",
        out_dtype_name="float32",
        block_size=32,
        tile_m=128,
        tile_n=128,
        tile_k=256,
    )
    self.assertIn("func.func", mlir_str)
    self.assertIn("f8E4M3FN", mlir_str)

  def test_unsupported_recipe_raises(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validating CLI generator argument rejection.
        ValueError, "Unsupported recipe 'invalid_recipe'. Supported recipes:"
    ):
      scaled_mm_generate.generate_kernel_module(
          recipe="invalid_recipe",
          in_dtype_name="float8_e4m3fn",
          out_dtype_name="float32",
      )

  def test_unsupported_in_dtype_raises(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validating CLI generator argument rejection.
        ValueError, "Unsupported in_dtype: bad_dtype"
    ):
      scaled_mm_generate.generate_kernel_module(
          recipe="mxfp8",
          in_dtype_name="bad_dtype",
          out_dtype_name="float32",
      )

  def test_unsupported_out_dtype_raises(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validating CLI generator argument rejection.
        ValueError, "Unsupported out_dtype: bad_dtype"
    ):
      scaled_mm_generate.generate_kernel_module(
          recipe="mxfp8",
          in_dtype_name="float8_e4m3fn",
          out_dtype_name="bad_dtype",
      )

  def test_register_custom_recipe(self):
    def dummy_generator(**kwargs):
      del kwargs
      return "module { func.func @custom_kernel() { return } }"

    scaled_mm_generate.register_recipe("test_recipe", dummy_generator)
    try:
      self.assertIn("test_recipe", scaled_mm_generate.get_supported_recipes())
      result = scaled_mm_generate.generate_kernel_module(
          recipe="test_recipe",
          in_dtype_name="float8_e4m3fn",
          out_dtype_name="float32",
      )
      self.assertIn("@custom_kernel", result)
    finally:
      del scaled_mm_generate._RECIPE_GENERATORS["test_recipe"]


if __name__ == "__main__":
  absltest.main()
