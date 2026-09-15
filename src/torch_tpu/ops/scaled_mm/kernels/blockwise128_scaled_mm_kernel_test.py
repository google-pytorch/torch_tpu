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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torch_tpu.ops.scaled_mm.kernels import blockwise128_scaled_mm_kernel as kernels

_FP8_E4M3 = jnp.float8_e4m3fn.dtype
_E4M3_MAX = 448.0

_BLOCK_SIZE = 128

# From PyTorch's fp8 block-wise test. f32 shares bf16's bound: the MXU
# accumulates FP8 products in a ~bf16 (E8M8) container, so output dtype does not
# change accumulation precision.
_ATOL_RTOL = {jnp.bfloat16: (6e-1, 7e-2), jnp.float32: (6e-1, 7e-2)}

# Cosine floor: cos >= 0.999 permits ~4.5% whole-tensor L2 error;
# a dropped block / transpose / mis-addressed scale drops it far below.
_COSINE_FLOOR = 0.999

_SHAPES = (
    ("128x128x128", 128, 128, 128),
    ("256x256x256", 256, 256, 256),
    ("128x256x512", 128, 256, 512),
    ("128x128x200", 128, 128, 200),
    ("64x256x300", 64, 256, 300),
    ("1x128x256", 1, 128, 256),
    ("512x128x128", 512, 128, 128),
    ("129x256x256", 129, 256, 256),
)

_LARGE_SHAPES = (
    ("256x1024x256", 256, 1024, 256),
    ("128x1280x256", 128, 1280, 256),
    ("128x7168x512", 128, 7168, 512),
    # DeepSeek-V3/R1 MoE-expert production GEMM (hidden=7168).
    ("128x7168x2048", 128, 7168, 2048),
    # DeepSeek-V3/R1 o_proj prefill (TP8-row shard): per-device K=2048, N=7168.
    ("2048x2048x7168", 2048, 2048, 7168),
)

_STRUCTURAL_TILE = (128, 128, 1024)

# Data-derived quantization needs whole 128x128 B tiles, so restrict it to
# shapes with K and N multiples of 128.
_ALIGNED_SHAPES = tuple(
    s for s in _SHAPES if s[2] % _BLOCK_SIZE == 0 and s[3] % _BLOCK_SIZE == 0
)

_BIAS_ISOLATION_SHAPES = tuple(
    s
    for s in _SHAPES + _LARGE_SHAPES
    if s[0]
    in (
        "1x128x256",  # M=1 (single-row) bias broadcast.
        "256x256x256",
        "128x1280x256",
        "128x7168x2048",
    )
)

_RANDOM_DTYPES = (("bfloat16", jnp.bfloat16), ("float32", jnp.float32))

_EXACT_DTYPE_SHAPES = tuple(
    (f"{name}_{dname}", m, k, n, dtype)
    for (name, m, k, n) in _SHAPES
    for (dname, dtype) in _RANDOM_DTYPES
)

_DERIVED_DTYPE_SHAPES = tuple(
    (f"{name}_{dname}", m, k, n, dtype)
    for (name, m, k, n) in _ALIGNED_SHAPES + _LARGE_SHAPES
    for (dname, dtype) in _RANDOM_DTYPES
)


def _cosine(expected: np.ndarray, actual: np.ndarray) -> float:
  e = expected.astype(np.float32).reshape(-1)
  a = actual.astype(np.float32).reshape(-1)
  return float(np.dot(e, a) / (np.linalg.norm(e) * np.linalg.norm(a) + 1e-30))


def _cast_ref(ref: np.ndarray, out_dtype) -> np.ndarray:
  # PyTorch compares the kernel against the reference *after* rounding the
  # reference to the output dtype, so both sides carry the same output rounding.
  return np.asarray(jnp.asarray(ref).astype(out_dtype).astype(jnp.float32))


def _run_kernel(
    a_fp8, b_fp8, sa, sb, out_dtype, bias=None, tile=None
) -> np.ndarray:
  tile_kwargs = {}
  if tile is not None:
    tile_kwargs = dict(tile_m=tile[0], tile_n=tile[1], tile_k=tile[2])
  out = kernels.scaled_mm_1x128_128x128(
      jnp.asarray(a_fp8),
      jnp.asarray(b_fp8),
      jnp.asarray(sa),
      jnp.asarray(sb),
      bias=jnp.asarray(bias) if bias is not None else None,
      out_dtype=out_dtype,
      **tile_kwargs,
  )
  return np.asarray(out.astype(jnp.float32))


def _dequant_a(payload_fp8, scale) -> np.ndarray:
  p = np.asarray(payload_fp8).astype(np.float32)
  s = np.repeat(np.asarray(scale).astype(np.float32), _BLOCK_SIZE, axis=1)
  return p * s[:, : p.shape[1]]


def _dequant_b(payload_fp8, scale) -> np.ndarray:
  p = np.asarray(payload_fp8).astype(np.float32)
  s = np.asarray(scale).astype(np.float32)
  s = np.repeat(np.repeat(s, _BLOCK_SIZE, axis=0), _BLOCK_SIZE, axis=1)
  return p * s[: p.shape[0], : p.shape[1]]


def _reference_matmul(a_real: np.ndarray, b_real: np.ndarray) -> np.ndarray:
  return a_real.astype(np.float32) @ b_real.astype(np.float32)


def _quantize_blocked(
    x_f32: np.ndarray, block_r: int, block_c: int
) -> tuple[np.ndarray, np.ndarray]:
  r, c = x_f32.shape
  assert r % block_r == 0 and c % block_c == 0
  xb = x_f32.reshape(r // block_r, block_r, c // block_c, block_c)
  amax = np.abs(xb).max(axis=(1, 3), keepdims=True)
  amax = np.maximum(amax, 1e-30)
  scale = (amax / _E4M3_MAX).astype(np.float32)
  payload = np.clip(xb / scale, -_E4M3_MAX, _E4M3_MAX).astype(_FP8_E4M3)
  payload = payload.reshape(r, c)
  scale = scale.reshape(r // block_r, c // block_c).astype(np.float32)
  return payload, scale


def _ones_sa(m: int, k: int) -> np.ndarray:
  return np.ones((m, k // _BLOCK_SIZE), np.float32)


def _ones_sb(k: int, n: int) -> np.ndarray:
  cols = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
  return np.ones((k // _BLOCK_SIZE, cols), np.float32)


# CheckAbslTestCase suppression: these tests are deterministic (fixed inputs or
# fixed RNG seeds), so plain parameterized test classes are safe here.
_TestCase = parameterized.TestCase  # ABSLTEST_OK=deterministic; fixed seeds


class BlockWise128ScaledMmStructuralTest(_TestCase):
  # Structural tests catch logic bugs (wrong scale/operand addressing,
  # layout/transpose, K-reduction, overflow/masking) with bit-exact equality.
  # Inputs are FP8-representable (identity, all-ones, integer markers), so any
  # deviation is a bug and each case isolates one bug class.

  def test_a_eye_b_eye(self):
    # Capture wrong out-tile (mi, ni) placement, row/column swaps, and layout bugs.
    m, k, n = 256, 256, 256
    a = np.eye(m, dtype=np.float32).astype(_FP8_E4M3)
    b = np.eye(k, dtype=np.float32).astype(_FP8_E4M3)
    out = _run_kernel(
        a, b, _ones_sa(m, k), _ones_sb(k, n), jnp.float32, tile=_STRUCTURAL_TILE
    )
    np.testing.assert_array_equal(out, np.eye(m, dtype=np.float32))

  def test_a_ones_b_ones(self):
    # Capture a dropped/duplicated/leaked k-tile and a broken middle-tile
    # accumulator carry (the middle carry only exists with 3+ k-tiles).
    m, k, n = 128, 2176, 128
    a = np.ones((m, k), np.float32).astype(_FP8_E4M3)
    b = np.ones((k, n), np.float32).astype(_FP8_E4M3)
    out = _run_kernel(
        a, b, _ones_sa(m, k), _ones_sb(k, n), jnp.float32, tile=_STRUCTURAL_TILE
    )
    np.testing.assert_array_equal(out, np.full((m, n), float(k), np.float32))

  @parameterized.named_parameters(
      ("float32", jnp.float32), ("bfloat16", jnp.bfloat16)
  )
  def test_bias_added(self, bias_dtype):
    m, k, n = 128, 1152, 256
    a = np.ones((m, k), np.float32).astype(_FP8_E4M3)
    b = np.ones((k, n), np.float32).astype(_FP8_E4M3)
    bias = np.arange(n, dtype=np.float32).astype(bias_dtype)
    bias_f32 = np.asarray(bias).astype(np.float32)
    out = _run_kernel(
        a,
        b,
        _ones_sa(m, k),
        _ones_sb(k, n),
        jnp.float32,
        bias=bias,
        tile=_STRUCTURAL_TILE,
    )
    expected = np.full((m, n), float(k), np.float32) + bias_f32
    np.testing.assert_array_equal(out, expected)

  def test_a_payload_marker(self):
    # Capture A-payload m-tile (128 + 1 partial) and K-block addressing.
    m, k, n = 129, 1152, 128
    a = np.ones((m, k), np.float32)
    a[1, 0:_BLOCK_SIZE] = 2.0
    b = np.ones((k, n), np.float32)
    out = _run_kernel(
        a.astype(_FP8_E4M3),
        b.astype(_FP8_E4M3),
        _ones_sa(m, k),
        _ones_sb(k, n),
        jnp.float32,
        tile=_STRUCTURAL_TILE,
    )
    np.testing.assert_array_equal(out, _reference_matmul(a, b))

  def test_b_payload_marker(self):
    # Capture B-payload n-tile and K-block addressing (a transposed read
    # surfaces the column as a row).
    m, k, n = 128, 1152, 256
    a = np.ones((m, k), np.float32)
    b = np.ones((k, n), np.float32)
    b[0:_BLOCK_SIZE, 1] = 2.0
    out = _run_kernel(
        a.astype(_FP8_E4M3),
        b.astype(_FP8_E4M3),
        _ones_sa(m, k),
        _ones_sb(k, n),
        jnp.float32,
        tile=_STRUCTURAL_TILE,
    )
    np.testing.assert_array_equal(out, _reference_matmul(a, b))

  def test_a_scale_marker(self):
    # Capture scale_a[row, K-block] addressing.
    m, k, n = 129, 1152, 128
    a_payload = np.ones((m, k), np.float32)
    a_payload[1, 0:_BLOCK_SIZE] = 2.0
    b_payload = np.ones((k, n), np.float32)
    sa = _ones_sa(m, k)
    sa[1, 0] = 2.0
    out = _run_kernel(
        a_payload.astype(_FP8_E4M3),
        b_payload.astype(_FP8_E4M3),
        sa,
        _ones_sb(k, n),
        jnp.float32,
        tile=_STRUCTURAL_TILE,
    )
    a_real = _dequant_a(a_payload.astype(_FP8_E4M3), sa)
    np.testing.assert_array_equal(out, _reference_matmul(a_real, b_payload))

  def test_b_scale_marker(self):
    # Capture scale_b[K-band, N-band] addressing of the 2-D weight scale.
    m, k, n = 128, 1152, 256
    a_payload = np.ones((m, k), np.float32)
    b_payload = np.ones((k, n), np.float32)
    sb = _ones_sb(k, n)
    sb[0, 1] = 2.0
    out = _run_kernel(
        a_payload.astype(_FP8_E4M3),
        b_payload.astype(_FP8_E4M3),
        _ones_sa(m, k),
        sb,
        jnp.float32,
        tile=_STRUCTURAL_TILE,
    )
    b_real = _dequant_b(b_payload.astype(_FP8_E4M3), sb)
    np.testing.assert_array_equal(out, _reference_matmul(a_payload, b_real))

  def test_b_scale_tail_corner_marker(self):
    # Capture scale_b's tail corner (last K-block x partial trailing N-band).
    m, k, n = 128, 1152, 300
    a_payload = np.ones((m, k), np.float32)
    b_payload = np.ones((k, n), np.float32)
    sb = _ones_sb(k, n)
    sb[-1, -1] = 2.0
    out = _run_kernel(
        a_payload.astype(_FP8_E4M3),
        b_payload.astype(_FP8_E4M3),
        _ones_sa(m, k),
        sb,
        jnp.float32,
        tile=_STRUCTURAL_TILE,
    )
    b_real = _dequant_b(b_payload.astype(_FP8_E4M3), sb)
    np.testing.assert_array_equal(out, _reference_matmul(a_payload, b_real))


class BlockWise128ScaledMmRandomTest(_TestCase):
  # Randomized tests catch precision bugs (quantization, accumulation drift, a
  # dropped scale/block, a bad output cast) via two gates: element-wise atol/rtol
  # against the fp8 reference (dequantized-FP8 matmul), plus cosine >= 0.999 vs
  # the fp8 reference and the hp truth (original-data matmul). Cosine catches
  # near-zero cancelling outputs whose unbounded relative error atol/rtol miss.

  def _assert_pytorch_gates(
      self, tag, m, k, n, out, fp8_ref, hp_ref, out_dtype
  ):
    atol, rtol = _ATOL_RTOL[out_dtype]
    fp8_ref_cast = _cast_ref(fp8_ref, out_dtype)
    abs_err = np.abs(out - fp8_ref_cast)
    max_abs_err = float(np.max(abs_err))
    # max_rel_err is undefined where the reference is exactly zero; skip those.
    denom = np.abs(fp8_ref_cast)
    nz = denom > 0
    max_rel_err = float(np.max(abs_err[nz] / denom[nz])) if np.any(nz) else 0.0
    cos_fp8 = _cosine(fp8_ref, out)
    cos_hp = _cosine(hp_ref, out)
    cos_fp8_hp = _cosine(hp_ref, fp8_ref)
    # Per-gate report: min threshold that would pass ("need"), configured
    # threshold ("set"), and their ratio (headroom). "infx" headroom means the
    # measured error is exactly zero.
    min_cos = min(cos_fp8, cos_hp, cos_fp8_hp)
    atol_headroom = atol / max_abs_err if max_abs_err > 0 else float("inf")
    rtol_headroom = rtol / max_rel_err if max_rel_err > 0 else float("inf")
    logging.info(
        "%s shape=(%d,%d,%d) out=%s | atol need>=%.3g set=%.3g (%.0fx) |"
        " rtol need>=%.3g set=%.3g (%.0fx) | cos min=%.6f set=%.3f (+%.4f)"
        " [fp8=%.6f hp=%.6f fp8_hp=%.6f]",
        tag,
        m,
        k,
        n,
        out_dtype.__name__,
        max_abs_err,
        atol,
        atol_headroom,
        max_rel_err,
        rtol,
        rtol_headroom,
        min_cos,
        _COSINE_FLOOR,
        min_cos - _COSINE_FLOOR,
        cos_fp8,
        cos_hp,
        cos_fp8_hp,
    )
    self.assertEqual(out.shape, (m, n))
    # Validate the emulation reference doing the right thing
    self.assertGreaterEqual(cos_fp8_hp, _COSINE_FLOOR)
    # Validate the kernel matches the emulation reference element-wise
    np.testing.assert_allclose(out, fp8_ref_cast, atol=atol, rtol=rtol)
    # Validate the kernel agrees in direction with the emulation reference
    self.assertGreaterEqual(cos_fp8, _COSINE_FLOOR)
    # Validate the kernel agrees in direction with the full-precision truth
    self.assertGreaterEqual(cos_hp, _COSINE_FLOOR)

  @parameterized.named_parameters(*_EXACT_DTYPE_SHAPES)
  def test_random_exact_e4m3_scales_one(self, m, k, n, out_dtype):
    # Samples large values spanning the E4M3 range (integers in [0, 255)) with
    # all-ones scales. No signed cancellation, so outputs are large/positive and
    # rtol grades the bulk (PyTorch's data_random_scales_one).
    rng = np.random.default_rng(0)
    a_hp = rng.integers(0, 255, size=(m, k)).astype(np.float32)
    b_hp = rng.integers(0, 255, size=(k, n)).astype(np.float32)
    a_fp8 = a_hp.astype(_FP8_E4M3)
    b_fp8 = b_hp.astype(_FP8_E4M3)
    a_real = np.asarray(a_fp8).astype(np.float32)
    b_real = np.asarray(b_fp8).astype(np.float32)

    out = _run_kernel(a_fp8, b_fp8, _ones_sa(m, k), _ones_sb(k, n), out_dtype)

    fp8_ref = _reference_matmul(a_real, b_real)
    hp_ref = _reference_matmul(a_hp, b_hp)
    self._assert_pytorch_gates(
        "exact_e4m3", m, k, n, out, fp8_ref, hp_ref, out_dtype
    )

  @parameterized.named_parameters(*_DERIVED_DTYPE_SHAPES)
  def test_random_data_derived_scales(self, m, k, n, out_dtype):
    # Samples small real values (activation ~O(1), weight ~O(1e-3)) mapped into
    # E4M3 by data-derived per-block scales, stressing the wide scale_a*scale_b
    # range. Includes the large-K production shapes (PyTorch's
    # data_random_calc_scales).
    rng = np.random.default_rng(1)
    a_real = rng.standard_normal((m, k)).astype(np.float32)
    b_real = (rng.standard_normal((k, n)) * 1e-3).astype(np.float32)
    a_fp8, sa = _quantize_blocked(a_real, 1, _BLOCK_SIZE)
    b_fp8, sb = _quantize_blocked(b_real, _BLOCK_SIZE, _BLOCK_SIZE)

    out = _run_kernel(a_fp8, b_fp8, sa, sb, out_dtype)

    fp8_ref = _reference_matmul(_dequant_a(a_fp8, sa), _dequant_b(b_fp8, sb))
    hp_ref = _reference_matmul(a_real, b_real)
    self._assert_pytorch_gates(
        "data_derived", m, k, n, out, fp8_ref, hp_ref, out_dtype
    )

  @parameterized.named_parameters(*_BIAS_ISOLATION_SHAPES)
  def test_random_bias_isolation(self, m, k, n):
    rng = np.random.default_rng(4)
    a_real = rng.standard_normal((m, k)).astype(np.float32)
    b_real = rng.standard_normal((k, n)).astype(np.float32)
    a_fp8, sa = _quantize_blocked(a_real, 1, _BLOCK_SIZE)
    b_fp8, sb = _quantize_blocked(b_real, _BLOCK_SIZE, _BLOCK_SIZE)
    bias = (rng.standard_normal(n) * 100.0).astype(np.float32)

    out_nobias = _run_kernel(a_fp8, b_fp8, sa, sb, jnp.float32)
    out_bias = _run_kernel(a_fp8, b_fp8, sa, sb, jnp.float32, bias=bias)

    delta = out_bias - out_nobias
    expected_delta = np.broadcast_to(bias, (m, n))

    np.testing.assert_allclose(delta, expected_delta, atol=1e-3, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
