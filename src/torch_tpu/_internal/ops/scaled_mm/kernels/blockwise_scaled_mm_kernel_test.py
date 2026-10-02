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

"""Numerics suite for the MXFP8 1x32 block-scaled FP8 matmul Pallas kernel.

Format: E4M3 payload with one FP32 (unswizzled) scale per 32-value K block,
applied by multiplication. The E8M0 scale dtype and SWIZZLE_32_4_4 layout are
dequantized/unswizzled by the C++ op layer before the kernel runs, so they are
out of scope here; analytical cases use power-of-two scales (E8M0 encodes only
powers of two).

Grading:
  * Structural cases assert bit-exact equality on analytically exact outputs,
    each isolating one bug class (addressing, layout, reduction, accumulation,
    overflow) to pinpoint a root cause.
  * Randomized cases align with PyTorch CUDA (`test_scaled_matmul_cuda.py`)
    validation: elementwise `rtol/atol` against the quantized reference and
    whole-tensor cosine similarity (`>= 0.999`) against both quantized and FP32
    references.
"""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torch_tpu._internal.ops.scaled_mm.kernels import blockwise_scaled_mm_kernel as kernels
from tests import seed_test_utils

_FP8_E4M3 = jnp.float8_e4m3fn.dtype
_E4M3_MAX = 448.0
# MX scale-recipe constants: E8M0 exponent bias and E4M3 target pow2 (2**8).
_F8E4M3_LARGEST_POW2 = 8
_F8E8M0_EXP_BIAS = 127

# PyTorch CUDA numerical validation thresholds for blockwise scaled matmul
# (see test_scaled_matmul_cuda.py: test_scaled_mm_block_wise_numerics).
_COSINE_SIM_FLOOR = 0.999
_BF16_RTOL = 7e-2
_BF16_ATOL = 6e-1

# (name, M, K, N). K is a multiple of the 32-wide block (partial K blocks are
# handled by the C++ op layer, not the kernel).
_SHAPES = (
    # normal / aligned to MXU + block tiles
    ("128x128x128", 128, 128, 128),
    ("256x256x256", 256, 256, 256),
    ("128x256x512", 128, 256, 512),
    # irregular M/N (K stays block-valid): exercises operand + scale pad/unpad
    ("65x96x112", 65, 96, 112),
    ("197x224x272", 197, 224, 272),
    # very unbalanced: heavy M or N padding, thin reduction or thin output
    ("1023x64x48", 1023, 64, 48),
    ("31x1024x64", 31, 1024, 64),
    ("45x96x1024", 45, 96, 1024),
    # mixed large/small + M around the MXU sublane tile boundary (127/128/129)
    ("2x1024x128", 2, 1024, 128),
    ("127x96x1024", 127, 96, 1024),
    ("129x128x96", 129, 128, 96),
)

# Square subset for identity x identity, which is only defined for M == K == N.
_SQUARE_SHAPES = tuple(s for s in _SHAPES if s[1] == s[2] == s[3])

# Representative subset for the marker tests (scale/payload addressing, layout):
# one aligned, one padded, one unbalanced, one on the MXU sublane boundary
# (M=129). Reduction/accumulation coverage is left to the other tests.
_MARKER_SHAPES = tuple(
    s
    for s in _SHAPES
    if s[0] in ("128x128x128", "197x224x272", "1023x64x48", "129x128x96")
)

_BLOCK_SIZE = 32


def _cosine_similarity(expected: np.ndarray, actual: np.ndarray) -> float:
  """Cosine similarity over the whole tensor in float32."""
  e = expected.astype(np.float32).reshape(-1)
  a = actual.astype(np.float32).reshape(-1)
  norm_e = float(np.linalg.norm(e))
  norm_a = float(np.linalg.norm(a))
  if norm_e == 0.0 and norm_a == 0.0:
    return 1.0
  return float(np.dot(e, a) / (norm_e * norm_a + 1e-30))


def _max_abs_error(expected: np.ndarray, actual: np.ndarray) -> float:
  return float(
      np.max(np.abs(expected.astype(np.float32) - actual.astype(np.float32)))
  )


def _run_kernel(a_fp8, b_fp8, sa, sb, out_dtype) -> np.ndarray:
  """Runs the kernel and returns the result as an FP32 numpy array."""
  out = kernels.blockwise_scaled_mm(
      jnp.asarray(a_fp8),
      jnp.asarray(b_fp8),
      jnp.asarray(sa),
      jnp.asarray(sb),
      block_size=_BLOCK_SIZE,
      out_dtype=out_dtype,
  )
  return np.asarray(out.astype(jnp.float32))


def _decode(payload_fp8, scale, axis: int) -> np.ndarray:
  """Reconstructs logical real values: payload * per-block scale, in float32.

  This is the MXFP8 dequant definition (`value = E4M3_payload * block_scale`).
  Used to build references that are structurally independent of the optimized
  kernel, so a shared bug cannot hide.
  """
  p = np.asarray(payload_fp8).astype(np.float32)
  s = np.repeat(np.asarray(scale).astype(np.float32), _BLOCK_SIZE, axis=axis)
  return p * s


def _reference_matmul(a_real: np.ndarray, b_real: np.ndarray) -> np.ndarray:
  """Reference matmul in FP32, independent of the kernel."""
  return a_real.astype(np.float32) @ b_real.astype(np.float32)


def _random_e4m3(
    rng: np.random.Generator, shape
) -> tuple[np.ndarray, np.ndarray]:
  """Random standard-normal values quantized to exact E4M3 code points.

  Returns `(payload_fp8, decoded_f32)`. Because `decoded_f32` is the exact
  float32 value of `payload_fp8`, there is zero FP32->FP8 input quantization
  error between the kernel inputs and the reference inputs, isolating the
  kernel's GEMM accumulation and output-cast numerics.
  """
  x = rng.standard_normal(size=shape).astype(np.float32)
  payload = np.clip(x, -_E4M3_MAX, _E4M3_MAX).astype(_FP8_E4M3)
  decoded = payload.astype(np.float32)
  return payload, decoded


def _mx_quantize(x_f32: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
  """Quantizes to MXFP8: power-of-two (E8M0) block scale + E4M3 payload.

  Mirrors the OCP MX recipe (`data_to_mx_scale`): the per-block scale is
  `2**(floor(log2(amax)) - 8)`, i.e. a power of two, which is exactly what E8M0
  stores. And we represented it here as FP32 (the kernel's contract).
  The dequant is `payload * scale`.
  """
  n = x_f32.shape[axis]
  assert n % _BLOCK_SIZE == 0
  new_shape = list(x_f32.shape)
  new_shape[axis] = n // _BLOCK_SIZE
  new_shape.insert(axis + 1, _BLOCK_SIZE)
  xb = x_f32.reshape(new_shape)
  amax = np.abs(xb).max(axis=axis + 1, keepdims=True)
  amax = np.maximum(amax, 1e-30)  # guard log2 for all-zero blocks
  exp = np.floor(np.log2(amax)) - _F8E4M3_LARGEST_POW2
  exp = np.clip(exp, -_F8E8M0_EXP_BIAS, _F8E8M0_EXP_BIAS)
  scale = np.exp2(exp).astype(
      np.float32
  )  # power of two: exact in FP32 and E8M0
  payload = np.clip(xb / scale, -_E4M3_MAX, _E4M3_MAX).astype(_FP8_E4M3)
  payload = payload.reshape(x_f32.shape)
  scale = scale.squeeze(axis + 1).astype(np.float32)
  return payload, scale


class _BlockwiseScaledMmTestBase(seed_test_utils.RepeatableTest):
  """Base test case that skips execution if no TPU backend is present."""

  def setUp(self):
    super().setUp()
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest(
          "Blockwise scaled matmul Pallas kernel requires TPU backend."
      )


class BlockwiseScaledMmStructuralTest(_BlockwiseScaledMmTestBase):

  @parameterized.named_parameters(*_SQUARE_SHAPES)
  def test_a_eye_b_eye(self, m, k, n):
    # Capture accidental transpose, row/col swap, layout error, or bad payload decode.
    a = np.eye(m, dtype=np.float32).astype(_FP8_E4M3)
    b = np.eye(k, dtype=np.float32).astype(_FP8_E4M3)
    sa = np.ones((m, k // _BLOCK_SIZE), np.float32)
    sb = np.ones((k // _BLOCK_SIZE, n), np.float32)
    out = _run_kernel(a, b, sa, sb, jnp.float32)
    expected = np.eye(m, dtype=np.float32)
    np.testing.assert_array_equal(out, expected)

  @parameterized.named_parameters(*_SHAPES)
  def test_a_ones_b_ones(self, m, k, n):
    # Capture a dropped K block, early-stopped reduction, or padded-tail leak.
    a = np.ones((m, k), np.float32).astype(_FP8_E4M3)
    b = np.ones((k, n), np.float32).astype(_FP8_E4M3)
    sa = np.ones((m, k // _BLOCK_SIZE), np.float32)
    sb = np.ones((k // _BLOCK_SIZE, n), np.float32)
    out = _run_kernel(a, b, sa, sb, jnp.float32)
    expected = np.full((m, n), float(k), np.float32)
    np.testing.assert_array_equal(out, expected)

  @parameterized.named_parameters(*_MARKER_SHAPES)
  def test_a_payload_marker(self, m, k, n):
    # Capture A payload/block-addressing errors, not scale handling.
    a = np.ones((m, k), np.float32)
    a[1, 0:_BLOCK_SIZE] = 2.0
    b = np.ones((k, n), np.float32)
    sa = np.ones((m, k // _BLOCK_SIZE), np.float32)
    sb = np.ones((k // _BLOCK_SIZE, n), np.float32)
    out = _run_kernel(
        a.astype(_FP8_E4M3), b.astype(_FP8_E4M3), sa, sb, jnp.float32
    )
    expected = _reference_matmul(a, b)
    np.testing.assert_array_equal(out, expected)

  @parameterized.named_parameters(*_MARKER_SHAPES)
  def test_b_payload_marker(self, m, k, n):
    # Capture B row/col mapping, B block addressing, and B-only transpose/layout mistakes.
    a = np.ones((m, k), np.float32)
    b = np.ones((k, n), np.float32)
    b[0:_BLOCK_SIZE, 1] = 2.0
    sa = np.ones((m, k // _BLOCK_SIZE), np.float32)
    sb = np.ones((k // _BLOCK_SIZE, n), np.float32)
    out = _run_kernel(
        a.astype(_FP8_E4M3), b.astype(_FP8_E4M3), sa, sb, jnp.float32
    )
    expected = _reference_matmul(a, b)
    np.testing.assert_array_equal(out, expected)

  @parameterized.named_parameters(*_MARKER_SHAPES)
  def test_a_scale_marker(self, m, k, n):
    # Capture if a_scale is ignored or addressed to the wrong block
    a_payload = np.ones((m, k), np.float32)
    a_payload[1, 0:_BLOCK_SIZE] = 2.0
    b_payload = np.ones((k, n), np.float32)
    sa = np.ones((m, k // _BLOCK_SIZE), np.float32)
    sa[1, 0] = 2.0
    sb = np.ones((k // _BLOCK_SIZE, n), np.float32)
    out = _run_kernel(
        a_payload.astype(_FP8_E4M3),
        b_payload.astype(_FP8_E4M3),
        sa,
        sb,
        jnp.float32,
    )
    a_real = _decode(a_payload.astype(_FP8_E4M3), sa, axis=1)
    expected = _reference_matmul(a_real, b_payload)
    np.testing.assert_array_equal(out, expected)

  @parameterized.named_parameters(*_MARKER_SHAPES)
  def test_b_scale_marker(self, m, k, n):
    # Capture if b_scale is ignored or addressed to the wrong block
    a_payload = np.ones((m, k), np.float32)
    b_payload = np.ones((k, n), np.float32)
    b_payload[0:_BLOCK_SIZE, 1] = 2.0
    sa = np.ones((m, k // _BLOCK_SIZE), np.float32)
    sb = np.ones((k // _BLOCK_SIZE, n), np.float32)
    sb[0, 1] = 2.0
    out = _run_kernel(
        a_payload.astype(_FP8_E4M3),
        b_payload.astype(_FP8_E4M3),
        sa,
        sb,
        jnp.float32,
    )
    b_real = _decode(b_payload.astype(_FP8_E4M3), sb, axis=0)
    expected = _reference_matmul(a_payload, b_real)
    np.testing.assert_array_equal(out, expected)


class BlockwiseScaledMmRandomTest(_BlockwiseScaledMmTestBase):
  """Randomized MXFP8 cases graded on rtol/atol and cosine similarity."""

  @parameterized.named_parameters(*_SHAPES)
  def test_random_exact_e4m3_scales_one(self, m, k, n):
    # Exact E4M3 inputs (scales=1): isolates kernel GEMM numerics.
    rng = np.random.default_rng(0)
    a_fp8, a_real = _random_e4m3(rng, (m, k))
    b_fp8, b_real = _random_e4m3(rng, (k, n))
    sa = np.ones((m, k // _BLOCK_SIZE), np.float32)
    sb = np.ones((k // _BLOCK_SIZE, n), np.float32)

    out = _run_kernel(a_fp8, b_fp8, sa, sb, jnp.bfloat16)
    expected = _reference_matmul(a_real, b_real)

    cos_sim = _cosine_similarity(expected, out)
    logging.info(
        "random_exact_e4m3 shape=(%d,%d,%d) cos_sim=%.6f max_abs_err=%.3g",
        m,
        k,
        n,
        cos_sim,
        _max_abs_error(expected, out),
    )
    self.assertEqual(out.shape, (m, n))
    self.assertGreaterEqual(cos_sim, _COSINE_SIM_FLOOR)
    np.testing.assert_allclose(out, expected, rtol=_BF16_RTOL, atol=_BF16_ATOL)

  @parameterized.named_parameters(*_SHAPES)
  def test_random_data_derived_scales(self, m, k, n):
    # Capture scale-application errors under realistic data-derived E8M0 scales.
    rng = np.random.default_rng(1)
    a_real = rng.standard_normal((m, k)).astype(np.float32)
    b_real = rng.standard_normal((k, n)).astype(np.float32)
    a_fp8, sa = _mx_quantize(a_real, axis=1)
    b_fp8, sb = _mx_quantize(b_real, axis=0)

    out = _run_kernel(a_fp8, b_fp8, sa, sb, jnp.bfloat16)

    original_ref = _reference_matmul(a_real, b_real)
    cos_sim_original = _cosine_similarity(original_ref, out)
    quantized_ref = _reference_matmul(
        _decode(a_fp8, sa, axis=1), _decode(b_fp8, sb, axis=0)
    )
    cos_sim_quantized = _cosine_similarity(quantized_ref, out)
    logging.info(
        "data_derived shape=(%d,%d,%d) cos_sim_original=%.6f "
        "cos_sim_quantized=%.6f max_abs_err=%.3g",
        m,
        k,
        n,
        cos_sim_original,
        cos_sim_quantized,
        _max_abs_error(quantized_ref, out),
    )
    self.assertEqual(out.shape, (m, n))
    self.assertGreaterEqual(cos_sim_original, _COSINE_SIM_FLOOR)
    self.assertGreaterEqual(cos_sim_quantized, _COSINE_SIM_FLOOR)
    np.testing.assert_allclose(
        out, quantized_ref, rtol=_BF16_RTOL, atol=_BF16_ATOL
    )


class BlockwiseScaledMmEdgeCaseTest(_BlockwiseScaledMmTestBase):
  """Backend-specific edge tripwires at the top of the FP8 range."""

  def test_fp8_saturation(self):
    # Capture accumulation overflow at the top of the E4M3 range (operands=448).
    m, n, k = 128, 128, 256
    block_size = 32
    a = np.full((m, k), _E4M3_MAX, np.float32).astype(_FP8_E4M3)
    b = np.full((k, n), _E4M3_MAX, np.float32).astype(_FP8_E4M3)
    sa = np.full((m, k // block_size), 2.0**-6, np.float32)
    sb = np.full((k // block_size, n), 2.0**-6, np.float32)
    out = np.asarray(
        kernels.blockwise_scaled_mm(
            jnp.asarray(a),
            jnp.asarray(b),
            jnp.asarray(sa),
            jnp.asarray(sb),
            block_size=block_size,
            out_dtype=jnp.float32,
        )
    )
    expected = _reference_matmul(_decode(a, sa, axis=1), _decode(b, sb, axis=0))
    np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
  absltest.main()
