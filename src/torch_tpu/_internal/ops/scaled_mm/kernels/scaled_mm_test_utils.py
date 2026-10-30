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

"""Shared numerics helpers for the FP8 scaled_mm / scaled_grouped_mm kernel tests.

Randomized FP8 GEMM tests follow PyTorch's CUDA validation
(`test_scaled_matmul_cuda.py`): elementwise rtol/atol against an f32 reference,
plus whole-tensor cosine similarity, which catches near-zero cancelling outputs
whose unbounded relative error rtol/atol cannot grade.
"""

from collections.abc import Mapping
from typing import Any

from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

# Cosine floor: cos >= 0.999 permits ~4.5% whole-tensor L2 error; a dropped
# block or K tile, transposed operand or misaddressed scale drops it far below.
COSINE_FLOOR = 0.999

# Elementwise bound from PyTorch's FP8 block-wise numerics test. TPU v7x and
# newer feed FP8 to the MXU natively and accumulate products in a ~bf16 (E8M8)
# container, so f32 output shares bf16's bound: the output dtype does not change
# accumulation precision.
FP8_MXU_RTOL = 7e-2
FP8_MXU_ATOL = 6e-1

# Tighter bound for chips whose FP8 GEMM matches the f32 reference up to output
# rounding.
F32_ACCUM_RTOL = 1e-2
F32_ACCUM_ATOL = 2e-2
_F32_ACCUM_CHIPS = (pltpu.ChipVersion.TPU_V5E, pltpu.ChipVersion.TPU_V6E)


def fp8_tolerance() -> Mapping[str, float]:
  """Returns `rtol`/`atol` kwargs for FP8 GEMM outputs on the current TPU."""
  if pltpu.get_tpu_info().chip_version in _F32_ACCUM_CHIPS:
    return dict(rtol=F32_ACCUM_RTOL, atol=F32_ACCUM_ATOL)
  return dict(rtol=FP8_MXU_RTOL, atol=FP8_MXU_ATOL)


def reference_matmul(a: Any, b: Any) -> np.ndarray:
  """Reference matmul in f32, independent of the kernel."""
  return np.asarray(a).astype(np.float32) @ np.asarray(b).astype(np.float32)


def round_to_dtype(x: Any, dtype: Any) -> np.ndarray:
  """Rounds `x` to `dtype` and returns it as an f32 numpy array.

  PyTorch compares the kernel against the reference *after* rounding the
  reference to the output dtype, so both sides carry the same output rounding.

  Args:
    x: values to round.
    dtype: dtype to round through, e.g. the kernel's output dtype.

  Returns:
    `x` rounded to `dtype`, as f32.
  """
  return np.asarray(jnp.asarray(x).astype(dtype).astype(jnp.float32))


def cosine_similarity(expected: Any, actual: Any) -> float:
  """Cosine similarity over the whole tensor in f32 (1.0 if both are zero)."""
  e = np.asarray(expected).astype(np.float32).reshape(-1)
  a = np.asarray(actual).astype(np.float32).reshape(-1)
  norm_e = float(np.linalg.norm(e))
  norm_a = float(np.linalg.norm(a))
  if norm_e == 0.0 and norm_a == 0.0:
    return 1.0
  return float(np.dot(e, a) / (norm_e * norm_a + 1e-30))


def max_abs_error(expected: Any, actual: Any) -> float:
  return float(
      np.max(
          np.abs(
              np.asarray(expected).astype(np.float32)
              - np.asarray(actual).astype(np.float32)
          )
      )
  )


def assert_fp8_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    cosine_floor: float = COSINE_FLOOR,
) -> None:
  """Asserts elementwise closeness and whole-tensor cosine >= `cosine_floor`."""
  np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
  cosine = cosine_similarity(expected, actual)
  if cosine < cosine_floor:
    raise AssertionError(
        f"Cosine similarity {cosine:.6f} is below the floor {cosine_floor}."
    )
