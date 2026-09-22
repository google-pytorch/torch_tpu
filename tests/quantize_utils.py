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

"""Utilities to work with sub-byte dtypes and quantization."""

from collections.abc import Callable, Mapping, Sequence
from typing import Final

import torch

# Dtypes that use fewer than 8 bits (less than one full byte) to represent a
# single value.
SUB_BYTE_DTYPES: Final[Sequence[torch.dtype]] = (
    torch.int4,
    torch.float4_e2m1fn_x2,
)

# Low-precision dtypes for quantization.
QUANTIZED_NUMERIC_DTYPES: Final[Sequence[torch.dtype]] = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    *SUB_BYTE_DTYPES,
)

# torch.float8_e8m0fnu was added in PyTorch 2.12, and oss/pyproject.toml
# declares `torch>=2.11.0`, so the attribute can legitimately be absent when
# torch_tpu is built against the oldest supported release.
_E8M0_DTYPES: Final[Sequence[torch.dtype]] = (
    (torch.float8_e8m0fnu,) if hasattr(torch, "float8_e8m0fnu") else ()
)

# Dtypes that safetensors cannot serialize directly. Callers store the raw
# bytes as uint8 and record the original dtype alongside them. Sub-byte dtypes
# have no safetensors encoding at all, and e8m0 is missing from its dtype map.
RAW_BYTE_DTYPES: Final[Sequence[torch.dtype]] = (
    *SUB_BYTE_DTYPES,
    *_E8M0_DTYPES,
)

# FP8 encodings that only ROCm produces. Neither TPU nor CUDA implements them,
# so ops exclude them wholesale rather than per-kernel.
ROCM_ONLY_FLOAT8_DTYPES: Final[Sequence[torch.dtype]] = (
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)

# The FP4 (E2M1) value denoted by each 4-bit encoding (bit 3 is the sign).
# Stored as float32 so that decoded values can be numerically compared.
_FP4_E2M1_VALUES: torch.Tensor = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)  # pyformat: disable


def quantize_to_float4_e2m1fn_x2(t: torch.Tensor) -> torch.Tensor:
  """Quantizes a float tensor to packed float4_e2m1fn_x2.

  Neither CPU nor CUDA implements casts to float4_e2m1fn_x2, so the cast
  is emulated with bit manipulation. The result matches the TPU cast
  semantics (see fp4_test.py): the shape is preserved, values are quantized
  with round-to-nearest-even and saturation to +/-6, packed two per byte (low
  nibble first) at the front of the buffer, and the rest is zero-padded.

  Args:
    t: The tensor to quantize. Must have a dtype convertible to float32.

  Returns:
    A float4_e2m1fn_x2 tensor with the same shape as `t`.
  """
  x = t.detach().to(torch.float32)
  # E2M1 has no NaN/inf encoding; saturate them like out-of-range values.
  x = torch.nan_to_num(x, nan=0.0, posinf=6.0, neginf=-6.0)
  magnitude = x.abs().clamp(max=6.0).flatten()
  # Midpoints between consecutive E2M1 magnitudes [0, .5, 1, 1.5, 2, 3, 4, 6].
  midpoints = torch.tensor(
      [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device
  )
  codes = torch.searchsorted(midpoints, magnitude, right=True)
  # right=True rounds all ties up; ties whose lower code is even must round
  # down instead (round-to-nearest-even).
  tie_down = (
      (magnitude == 0.25)
      | (magnitude == 1.25)
      | (magnitude == 2.5)
      | (magnitude == 5.0)
  )
  codes = torch.where(tie_down, codes - 1, codes).to(torch.uint8)
  codes |= torch.signbit(x).flatten().to(torch.uint8) << 3
  # Pack two values per byte, low nibble first, into a buffer with one byte
  # per container element (i.e. twice the needed capacity), zero-padded.
  nibbles = torch.zeros(2 * codes.numel(), dtype=torch.uint8, device=x.device)
  nibbles[: codes.numel()] = codes
  nibbles = nibbles.view(-1, 2)
  packed = nibbles[:, 0] | (nibbles[:, 1] << 4)
  return packed.view(t.shape).view(torch.float4_e2m1fn_x2)


def quantize_to_int4(t: torch.Tensor) -> torch.Tensor:
  """Quantizes an integer tensor to int4.

  Neither CPU nor CUDA implements casts to int4, so saturate it to the int4
  value range and reinterpret the bytes. An int4 element occupies one byte,
  sign-extended, thus the clamped int8 bit patterns are already correct.

  Args:
    t: The tensor to quantize. Must have an integer dtype.

  Returns:
    An int4 tensor with the same shape as `t`.
  """
  return t.detach().clamp(-8, 7).to(torch.int8).contiguous().view(torch.int4)


def decode_sub_byte_tensor(t: torch.Tensor) -> torch.Tensor:
  """Decodes a sub-byte dtype tensor for value comparison.

  CPU kernels for the sub-byte dtypes are mostly unimplemented (even casts),
  so decode them manually:

  - int4: one element per byte; the low nibble is sign/zero-extended to
    int8 (tolerant of both sign-extended and nibble-only encodings).
  - float4_e2m1fn_x2: each byte packs two FP4 (E2M1) values, low nibble first;
    decoded to a float32 tensor whose trailing dimension is doubled.

  Args:
    t: The tensor to decode.

  Returns:
    The decoded tensor, or `t` unchanged if its dtype needs no decoding.
  """
  if t.dtype == torch.int4:
    return (t.view(torch.int8) << 4) >> 4
  if t.dtype == torch.float4_e2m1fn_x2:
    data = t.view(torch.uint8)
    codes = torch.stack((data & 0xF, data >> 4), dim=-1)
    if t.dim() > 0:
      # Not reshape(..., -1): -1 is ambiguous when a leading dim is 0.
      codes = codes.reshape(*t.shape[:-1], 2 * t.shape[-1])
    return _FP4_E2M1_VALUES.to(codes.device)[codes.long()]
  return t


# Maps a sub-byte dtype to a conversion function from the fallback dtype to
# the sub-byte dtype.  Used by the test framework to produce sample inputs
# for dtypes that torch.testing cannot generate directly.
# See also DESIRED_DTYPE_TO_SAMPLE_DTYPE below.
SAMPLE_DTYPE_CONVERTERS: Final[
    Mapping[torch.dtype, Callable[[torch.Tensor], torch.Tensor]]
] = {
    torch.float4_e2m1fn_x2: quantize_to_float4_e2m1fn_x2,
    torch.int4: quantize_to_int4,
}

# Maps a sub-byte sample dtype to the supported dtype that should be sampled
# first, before converting with the matching converter in
# SAMPLE_DTYPE_CONVERTERS above.
DESIRED_DTYPE_TO_SAMPLE_DTYPE: Final[Mapping[torch.dtype, torch.dtype]] = {
    # Why torch.float8_e4m3fn:
    # - it matches the finite floating-point encoding semantics (fn) of
    #   torch.float4_e2m1fn_x2, and
    # - it provides 3 mantissa bits for fine-grained quantization resolution
    #   before casting down to FP4.
    torch.float4_e2m1fn_x2: torch.float8_e4m3fn,
    torch.int4: torch.int8,
}
