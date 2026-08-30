# Copyright 2025 Google LLC
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

"""Test utilities for torch_tpu."""

import collections
import enum
import math
import re
from typing import Any, Callable, TypeAlias

import numpy as np
import torch
from torch_tpu._internal.utils import utils

# A tolerance value can be either a float or a callable that takes the expected
# value as the only argument and returns the tolerance to use for comparing
# the expected and the actual values. This allows the tolerance to depend on
# the value range, and thus be tighter than the same global tolerance.
Tolerance: TypeAlias = float | Callable[[float], float]

_SAFETY_FACTOR: float = 1.2


def _format_sci_ceil(val: float) -> str:
  """Formats a float in scientific notation, rounded up to 1 decimal place."""
  if val == 0:
    return "0.0e+00"
  if not np.isfinite(val):
    return str(val)
  exponent = int(np.floor(np.log10(abs(val))))
  mantissa = val / (10**exponent)
  mantissa_ceil = np.ceil(mantissa * 10) / 10
  if mantissa_ceil >= 10:
    mantissa_ceil = 1.0
    exponent += 1
  return f"{mantissa_ceil:.1f}e{exponent:+03d}"


class CheckValueMode(enum.Enum):
  """How to check the values when comparing tensors.

  Options:
  - SKIP:   Don't check the values.
  - LOOSE:  Check the values loosely, i.e. using PyTorch's default algorithm,
            where two values are considered close if the error is less than
            atol + rtol * abs(expected).
  - STRICT: Check the values strictly, where two values are considered close
            if the error is less than min(atol, rtol * abs(expected)).
            Note: STRICT mode enforces relative tolerance even for small values
            (down to 1e-7), so it may fail if the expected value is exactly
            zero but the actual value is small non-zero (even if < atol).
            In such cases, use LOOSE mode.
  """

  SKIP = "skip"
  LOOSE = "loose"
  STRICT = "strict"


_RE_INDEX_CAPTURE = re.compile(r"at index \((.*?)\) \(up to .* allowed\)")

_RE_STRICT_REL_FAILURE = re.compile(r"\(up to 1.*e-07 allowed\)")

# Regex for identifying scalar outputs.
_RE_SCALAR_COMP_FAILURE = re.compile(r"Scalars are not (equal|close)!")


def assert_close(
    actual: Any,
    expected: Any,
    rtol: float | None = None,
    atol: Tolerance | None = None,
    preamble: str | None = None,
    check_value: CheckValueMode | None = None,
    check_dtype: bool = True,
) -> None:
  """Assert that `actual` and `expectec` are sufficiently close.

  This function is typically used for tensors or collections of tensors but
  is flexible. Internally it uses`torch.testing.assert_close`, and detailed
  information about how tensors are compared can be found in that function's
  docstring. This implementation provides a more detailed error message.

  Args:
    actual: The computed result.
    expected: The expected result.
    rtol: The relative tolerance for checking numerical values. If omitted, a
      default value is determined base on the dtype.
    atol: The absolute tolerance for checking numerical values. If a callable,
      rtol must be None and the result of calling it on the expected value will
      be used as the absolute tolerance. This allows the tolerance to depend on
      the value range (e.g. we might need higher precisions for certain ranges).
      If omitted, a default value is determined based on the dtype.
    preamble: Optional string to describe the objects being compared.
    check_value: The mode for checking the values. None (default) means LOOSE.
    check_dtype: Check if the dtypes are the same. Use sparsingly, only when
      dtypes are different due to PyTorch inconsistency between their CPU and
      GPU backend.
  """

  if check_value == CheckValueMode.LOOSE:
    raise ValueError(
        "LOOSE mode is the default. Please omit the check_value argument."
    )

  if actual is None or expected is None:
    assert actual is None and expected is None, (
        "expected and actual values for comparison must both be None or not"
        f" None, got {actual=} and {expected=}"
    )
    return

  if not check_dtype:
    promoted_dtype = torch.promote_types(actual.dtype, expected.dtype)
    actual = actual.to(promoted_dtype)
    expected = expected.to(promoted_dtype)

  actual_is_seq = isinstance(
      actual, collections.abc.Sequence
  ) and not isinstance(actual, str)
  expected_is_seq = isinstance(
      expected, collections.abc.Sequence
  ) and not isinstance(expected, str)
  assert actual_is_seq == expected_is_seq, (
      "actual and expected values for comparison must be both objects or both"
      f" sequences, got one sequence and one object\n{actual=}\n{expected=}"
  )
  if actual_is_seq and expected_is_seq:
    assert len(actual) == len(expected), (
        "sequences of tensors for comparison must be the same length, got"
        f" {len(actual)} actual objects and {len(expected)} expected objects"
    )
    for actual_item, expected_item in zip(actual, expected):
      assert_close(
          actual=actual_item,
          expected=expected_item,
          rtol=rtol,
          atol=atol,
          preamble=preamble,
          check_value=check_value,
          check_dtype=check_dtype,
      )
    return

  assert isinstance(actual, torch.Tensor) and isinstance(
      expected, torch.Tensor
  ), (
      "expected and actual values for comparison must be both tensors, got"
      f" {type(actual)} and {type(expected)}"
  )
  _assert_tensor_close(
      actual=actual,
      expected=expected,
      rtol=rtol,
      atol=atol,
      preamble=preamble,
      check_value=check_value,
  )


def _lookup_tolerances(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float | None,
    atol: float | None,
) -> tuple[float, float]:
  """Looks up the tolerances for comparing the given tensors.

  Args:
    actual: The computed result.
    expected: The expected result.
    rtol: The relative tolerance for checking numerical values. If omitted, a
      default value is determined based on the dtype.
    atol: The absolute tolerance for checking numerical values. If omitted, a
      default value is determined based on the dtype.

  Returns:
    The relative and absolute tolerances to use for comparing the given tensors.
  """
  if rtol is not None and atol is not None:
    return rtol, atol

  default_rtol, default_atol = torch.testing._comparison.default_tolerances(  # pylint: disable=protected-access
      actual, expected
  )
  if rtol is None:
    rtol = default_rtol
  if atol is None:
    atol = default_atol
  return rtol, atol


def _raw_assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    msg: Callable[[str], str],
    rtol: float | None = None,
    atol: Tolerance | None = None,
) -> None:
  """Raw assert_close implementation."""

  if callable(atol):
    assert rtol is None, f"rtol must be None if atol is a callable, got {rtol=}"

    # Reuse torch.testing.assert_close to check that the dtypes and shapes
    # match.
    torch.testing.assert_close(  # TORCH_ASSERT_CLOSE_OK=root usage
        actual=actual,
        expected=expected,
        # Treat all numbers as equal as we are only comparing the dtypes and
        # shapes so far.
        rtol=0.0,
        atol=float("inf"),
        equal_nan=True,
        check_device=False,
        msg=msg,
    )

    def _close(actual_value: float, expected_value: float) -> bool:
      """Returns True if the values are close."""
      return abs(actual_value - expected_value) <= atol(expected_value)

    # Verify that the values are close elementwise.
    # Get the values of the tensors as N-dimensional numpy arrays:
    actual_np = actual.cpu().numpy()
    expected_np = expected.cpu().numpy()
    # Enumerate the indices where the values are not close:
    mismatches = []
    for i, actual_value in np.ndenumerate(actual_np):
      expected_value = expected_np[i]
      if not _close(actual_value, expected_value):
        mismatches.append(
            f"at index {i}, expected={expected_value}, "
            f"actual={actual_value}, "
            f"diff={abs(actual_value - expected_value)}, "
            f"tolerance={atol(expected_value)}"
        )
    if mismatches:
      raise AssertionError(
          f"Values are not close at {len(mismatches)} indices:\n"
          + "\n".join(mismatches)
      )
  else:
    torch.testing.assert_close(  # TORCH_ASSERT_CLOSE_OK=root usage
        actual=actual,
        expected=expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        check_device=False,
        msg=msg,
    )


def _round_up_float(val: float) -> float:
  """Rounds float upward towards +infinity to prevent truncation errors."""
  # Rounds float upward toward +infinity using nextafter to avoid truncation.
  if not math.isfinite(val) or val == 0.0:
    return val
  return math.nextafter(val, math.inf)


def compute_optimal_tolerances(
    actual: Any,
    expected: Any,
    safety_factor: float = _SAFETY_FACTOR,
) -> tuple[float, float]:
  """Computes optimal (atol, rtol) bounds for comparing actual and expected.

  Follows the optimal error bound framework in go/tt-optimal-error-bounds:
  - Objective: Minimize atol + rtol subject to |a - e| <= atol + rtol * |e|.
  - Case 1: |e| <= 1 -> atol = |a - e|, rtol = 0.
  - Case 2: |e| > 1  -> atol = 0, rtol = |a - e| / |e|.
  The final bounds are max(atol_i) and max(rtol_i) over all elements,
  multiplied by safety_factor (default 1.2x), rounded up to avoid truncation.

  Args:
    actual: Computed tensor or collection of tensors or scalar.
    expected: Expected tensor or collection of tensors or scalar.
    safety_factor: Safety margin multiplier (default 1.2).

  Returns:
    Tuple of (optimal_atol, optimal_rtol).
  """
  assert actual is not None and expected is not None, (
      "actual and expected values for tolerance computation must both be"
      f" non-None, got actual={actual} and expected={expected}"
  )

  # Enforce matching container types for sequences.
  assert isinstance(actual, list) == isinstance(expected, list), (
      f"Container type mismatch: actual is {type(actual).__name__}, expected is"
      f" {type(expected).__name__}"
  )
  assert isinstance(actual, tuple) == isinstance(expected, tuple), (
      f"Container type mismatch: actual is {type(actual).__name__}, expected is"
      f" {type(expected).__name__}"
  )

  # Recursively compute optimal error bounds across list/tuple elements.
  if isinstance(actual, (list, tuple)):
    assert len(actual) == len(expected), (
        f"Sequence length mismatch: len(actual)={len(actual)} vs"
        f" len(expected)={len(expected)}"
    )
    atols, rtols = [], []
    for a, e in zip(actual, expected, strict=True):
      a_tol, r_tol = compute_optimal_tolerances(a, e, safety_factor)
      atols.append(a_tol)
      rtols.append(r_tol)
    if any(math.isnan(x) for x in atols) or any(math.isnan(x) for x in rtols):
      return float("nan"), float("nan")
    return (max(atols, default=0.0), max(rtols, default=0.0))

  # Enforce matching input types before coercion.
  assert isinstance(actual, torch.Tensor) == isinstance(
      expected, torch.Tensor
  ), (
      f"Input type mismatch: actual is {type(actual).__name__}, expected is"
      f" {type(expected).__name__}"
  )

  # Coerce inputs to tensors and return zero tolerances if empty.
  if not isinstance(actual, torch.Tensor):
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected)

  # Assert structural and dtype compatibility.
  assert isinstance(actual, torch.Tensor) and isinstance(
      expected, torch.Tensor
  ), f"Expected torch.Tensors, got {type(actual)} and {type(expected)}"
  assert actual.dtype == expected.dtype, (
      f"Dtype mismatch: actual.dtype={actual.dtype} vs"
      f" expected.dtype={expected.dtype}"
  )
  assert actual.shape == expected.shape, (
      f"Shape mismatch: actual.shape={actual.shape} vs"
      f" expected.shape={expected.shape}"
  )

  if actual.numel() == 0 or expected.numel() == 0:
    return 0.0, 0.0

  # Synchronize TPU execution and cast CPU tensors to float64/complex128.
  if torch.accelerator.is_available():
    torch.accelerator.synchronize()

  actual = actual.detach().cpu()
  expected = expected.detach().cpu()

  if actual.dtype.is_complex:
    actual = actual.to(torch.complex128)
    expected = expected.to(torch.complex128)
  else:
    actual = actual.to(torch.float64)
    expected = expected.to(torch.float64)

  # Return NaN for unmatched NaNs, or zero tolerances for matching NaNs.
  actual_nan = torch.isnan(actual)
  expected_nan = torch.isnan(expected)
  if (actual_nan ^ expected_nan).any():
    return float("nan"), float("nan")

  # Both tensors are guaranteed to have matching NaN positions at this point.
  valid_mask = ~actual_nan
  # If valid_mask has no True elements, all elements in both actual and
  # expected are NaNs.
  if not valid_mask.any():
    return 0.0, 0.0

  # At this point, actual[valid_mask] and expected[valid_mask] contain no NaN
  # values. Compute absolute diffs and partition into Case 1 (|e|<=1) and
  # Case 2 (|e|>1).
  diff = torch.abs(actual[valid_mask] - expected[valid_mask])
  if torch.isinf(diff).any():
    return float("inf"), float("inf")

  abs_expected = torch.abs(expected[valid_mask])

  mask_case1 = abs_expected <= 1.0
  mask_case2 = ~mask_case1

  atol_opt = 0.0
  rtol_opt = 0.0

  # Case 1 (|e| <= 1): Minimize atol + rtol by setting rtol = 0 and atol =
  # max diff. Round up to avoid floating-point truncation during difference
  # calculation.
  if mask_case1.any():
    atol_opt = _round_up_float(float(diff[mask_case1].max().item()))

  # Case 2 (|e| > 1): Minimize atol + rtol by setting atol = 0, rtol =
  # max rel diff.
  if mask_case2.any():
    case2_expected = abs_expected[mask_case2]
    case2_diff = diff[mask_case2]
    finite_mask = torch.isfinite(case2_expected)
    if finite_mask.any():
      # Round up to avoid floating-point truncation introduced by division.
      rtol_opt = _round_up_float(
          float(
              (case2_diff[finite_mask] / case2_expected[finite_mask])
              .max()
              .item()
          )
      )
    elif torch.isinf(case2_diff).any():
      rtol_opt = float("inf")

  # Apply safety margin multiplier and round up to avoid truncation errors.
  final_atol = _round_up_float(safety_factor * atol_opt)
  final_rtol = _round_up_float(safety_factor * rtol_opt)

  return final_atol, final_rtol


def _add_tolerance_suggestions(
    msg: str,
    check_mode: CheckValueMode,
    actual: Any,
    expected: Any,
) -> str:
  """Augments the error message with tolerance suggestions.

  Tolerances are determined by computing tight bounds using
  `compute_optimal_tolerances(actual, expected)`,
  which minimizes atol + rtol with a safety factor margin
  (_SAFETY_FACTOR = 1.2).

  Args:
    msg: Original assertion error message string.
    check_mode: CheckValueMode (LOOSE, STRICT, or SKIP).
    actual: Computed result (torch.Tensor, scalar, or nested sequence thereof).
    expected: Expected result (torch.Tensor, scalar, or nested sequence
      thereof).

  Returns:
    Augmented error message string containing suggested tolerance values.
  """
  try:
    atol_val, rtol_val = compute_optimal_tolerances(actual, expected)
  except Exception:  # pylint: disable=broad-except
    return msg

  formatted_rtol = _format_sci_ceil(rtol_val)
  formatted_atol = _format_sci_ceil(atol_val)

  suggestion_lines = ["\n\nTolerance Suggestions:"]

  if check_mode == CheckValueMode.LOOSE:
    suggestion_lines.append("\n  Loose check failed.")
  elif check_mode == CheckValueMode.STRICT:
    suggestion_lines.append("\n  Strict check failed.")
    suggestion_lines.append(
        "\n  Note: Optimal single-bound tradeoffs (rtol=0 or atol=0) apply to"
        " LOOSE mode."
    )

  if np.isnan(atol_val) or np.isnan(rtol_val):
    suggestion_lines.append(
        "\n    - Automatic tolerance calculation omitted due to NaN values."
    )
    return msg + "".join(suggestion_lines)

  suggestion_lines.append(
      "\n  Optimal tight tolerances (with 1.2x safety margin):"
  )

  if np.isinf(rtol_val):
    suggestion_lines.append("\n    - rtol check is impossible or infinite")
  else:
    suggestion_lines.append(f"\n    - rtol >= {rtol_val} ({formatted_rtol})")

  if np.isinf(atol_val):
    suggestion_lines.append("\n    - atol check is impossible or infinite")
  else:
    suggestion_lines.append(f"\n    - atol >= {atol_val} ({formatted_atol})")

  return msg + "".join(suggestion_lines)


# The FP4 (E2M1) value denoted by each 4-bit encoding (bit 3 is the sign).
# Stored as float32 so that decoded values can be numerically compared.
_FP4_E2M1_VALUES: torch.Tensor = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)  # pyformat: disable


def _decode_sub_byte_tensor(t: torch.Tensor) -> torch.Tensor:
  """Decodes a sub-byte dtype tensor for value comparison.

  CPU kernels for the sub-byte dtypes are mostly unimplemented (even casts),
  so decode them manually:

  - int4/uint4: one element per byte; the low nibble is sign/zero-extended to
    int8/uint8 (tolerant of both sign-extended and nibble-only encodings).
  - float4_e2m1fn_x2: each byte packs two FP4 (E2M1) values, low nibble first;
    decoded to a float32 tensor whose trailing dimension is doubled.

  Args:
    t: The tensor to decode.

  Returns:
    The decoded tensor, or `t` unchanged if its dtype needs no decoding.
  """
  # A same-itemsize view() works on non-contiguous tensors, unlike contiguous(),
  # which needs a copy kernel the sub-byte dtypes don't have.
  if t.dtype == torch.int4:
    return (t.view(torch.int8) << 4) >> 4
  if t.dtype == torch.uint4:
    return t.view(torch.uint8) & 0xF
  if t.dtype == torch.float4_e2m1fn_x2:
    data = t.view(torch.uint8)
    codes = torch.stack((data & 0xF, data >> 4), dim=-1)
    if t.dim() > 0:
      # Not reshape(..., -1): -1 is ambiguous when a leading dim is 0.
      codes = codes.reshape(*t.shape[:-1], 2 * t.shape[-1])
    return _FP4_E2M1_VALUES.to(codes.device)[codes.long()]
  return t


def _assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float | None = None,
    atol: Tolerance | None = None,
    preamble: str | None = None,
    check_value: CheckValueMode | None = None,
) -> None:
  """Single-tensor assert_close implementation.

  Compared to _raw_assert_tensor_close, this function provides a more detailed
  error message, including the actual and expected tensor values at the
  locations where the tolerances are exceeded.

  Args:
    actual: The computed result.
    expected: The expected result.
    rtol: The relative tolerance for checking numerical values. If omitted, a
      default value is determined based on the dtype.
    atol: The absolute tolerance for checking numerical values. If a callable,
      rtol must be None and the result of calling it on the expected value will
      be used as the absolute tolerance. This allows the tolerance to depend on
      the value range (e.g. we might need higher precisions for certain ranges).
      If omitted, a default value is determined based on the dtype.
    preamble: Optional string to describe the objects being compared.
    check_value: The mode for checking the values. None (default) means LOOSE.
  """

  if check_value == CheckValueMode.LOOSE:
    raise ValueError(
        "LOOSE mode is the default. Please omit the check_value argument."
    )

  if not preamble:
    preamble = ""
  else:
    preamble = preamble.rstrip() + "\n"

  if expected.shape != actual.shape:
    msg = (
        f"{preamble}Tensor shape mismatch:\n"
        f"Expected tensor shape: {expected.shape}\n"
        f"  Actual tensor shape: {actual.shape}.\n"
        f"      Expected tensor:\n{expected}\n\n"
        f"      Actual tensor:\n{actual}"
    )
    raise AssertionError(msg)

  if expected.dtype != actual.dtype:
    msg = (
        f"{preamble}Tensor dtype mismatch:\n"
        f"Expected tensor dtype: {expected.dtype}\n"
        f"  Actual tensor dtype: {actual.dtype}.\n"
        f"      Expected tensor:\n{expected}\n\n"
        f"      Actual tensor:\n{actual}"
    )
    raise AssertionError(msg)

  # Sub-byte dtypes have no CPU kernels for comparison math; decode them into
  # standard dtypes first.
  actual = _decode_sub_byte_tensor(actual)
  expected = _decode_sub_byte_tensor(expected)

  if torch.accelerator.is_available():
    torch.accelerator.synchronize()

  def get_indices(msg: str) -> list[tuple[int, ...]]:
    """Returns the indices from the given message."""
    # The message looks like:
    #
    # ...
    # Greatest absolute difference: 1.8008867502212524 at index (2, 0, 0) \
    # (up to 0 allowed)
    # Greatest relative difference: 0.10574676096439362 at index (1, 2, 2) \
    # (up to 1e-05 allowed)
    indices = []
    while True:
      match = _RE_INDEX_CAPTURE.search(msg)
      if match:
        indices.append(
            tuple(
                int(s.strip())
                for s in match.group(1).split(",")
                # we discard empty strings, e.g. if the shape is "(3,)" we
                # don't want to get [int('3'), int('')]
                if s.strip()
            )
        )
        msg = msg[match.end() :]
      else:
        break
    return indices

  def refine_assert_close_message(msg: str) -> str:
    """Refines the message from torch.testing.assert_close with more details."""
    # Parses scalar vs tensor error messages from torch.testing.assert_close
    # to extract failure indices and display detailed actual/expected element
    # values.
    details = []
    if _RE_SCALAR_COMP_FAILURE.search(msg):
      atol_str = "None"
      rtol_str = "None"
      display_rtol = rtol
      display_atol = atol
      if not callable(display_atol):
        try:
          display_rtol, display_atol = _lookup_tolerances(
              actual, expected, display_rtol, display_atol
          )
        except Exception:  # pylint: disable=broad-except
          pass
      if isinstance(display_atol, (int, float)):
        atol_str = f"{display_atol:.3g}"
      elif display_atol is not None:
        atol_str = str(display_atol)
      if isinstance(display_rtol, (int, float)):
        rtol_str = f"{display_rtol:.3g}"
      elif display_rtol is not None:
        rtol_str = str(display_rtol)
      details.append(f"\nAbsolute difference allowed: up to {atol_str}\n")
      details.append(f"Relative difference allowed: up to {rtol_str}\n")
    else:
      for index in get_indices(msg):
        golden_elem = expected[index]
        torch_tpu_elem = actual[index]
        diff = abs(golden_elem - torch_tpu_elem)
        rel_diff = 1 if golden_elem == 0 else diff / abs(golden_elem)
        details.append(
            f"    at index {index}, expected={golden_elem},"
            f" actual={torch_tpu_elem},"
            f" relative diff={rel_diff},"
            f" diff={diff}\n"
        )
    return (
        f"{preamble}{msg}\n{''.join(details)}\n"
        f"Expected tensor:\n{expected}\n\n"
        f"Actual tensor:\n{actual}"
    )

  # If atol is a callable, we don't check the relative tolerance, so we always
  # use the loose mode.
  if (
      check_value is None
      or check_value == CheckValueMode.LOOSE
      or callable(atol)
  ):

    def msg_handler(msg: str) -> str:
      return _add_tolerance_suggestions(
          refine_assert_close_message(msg),
          check_mode=CheckValueMode.LOOSE,
          actual=actual,
          expected=expected,
      )

    # To support partial overrides (where only rtol or only atol is specified),
    # we pre-fill the missing tolerance with the PyTorch default. This prevents
    # assert_close from failing due to one of the tolerances being None.
    if not callable(atol):
      rtol, atol = _lookup_tolerances(actual, expected, rtol, atol)

    _raw_assert_tensor_close(
        actual=actual,
        expected=expected,
        rtol=rtol,
        atol=atol,
        msg=msg_handler,
    )
  elif check_value == CheckValueMode.STRICT:
    # In the strict mode, we compare the tensors twice: once with the absolute
    # tolerance only, and once with the relative tolerance only. This ensures
    # that both thresholds are satisfied.
    rtol, atol = _lookup_tolerances(actual, expected, rtol, atol)

    def strict_msg_handler(msg: str, is_relative: bool = False) -> str:
      msg = refine_assert_close_message(msg)
      if is_relative:
        # The relative check uses a small atol=1e-7 to handle zero expected
        # values. The error message from torch.testing.assert_close reports this
        # as the allowed absolute difference, which is confusing to users who
        # provided a different atol. We replace it with a clarification.
        # Note: We do this replacement AFTER refine_assert_close_message because
        # get_indices depends on the format.
        msg = _RE_STRICT_REL_FAILURE.sub("(strict relative check failure)", msg)
      return _add_tolerance_suggestions(
          msg,
          check_mode=CheckValueMode.STRICT,
          actual=actual,
          expected=expected,
      )

    _raw_assert_tensor_close(
        actual=actual,
        expected=expected,
        rtol=0,
        atol=atol,
        msg=lambda msg: strict_msg_handler(msg, is_relative=False),
    )

    # When expected is 0.0 and actual is -0.0, torch.testing.assert_close
    # thinks the relative error is 1.0, so we need to set atol to a small
    # value (as opposed to 0) to allow this case to pass.
    # For float8 types, torch.testing.assert_close requires atol=0.0 for
    # bitwise comparison.
    atol = 1e-7
    if actual.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
      atol = 0.0

    _raw_assert_tensor_close(
        actual=actual,
        expected=expected,
        rtol=rtol,
        atol=atol,
        msg=lambda msg: strict_msg_handler(msg, is_relative=True),
    )


libtpu_at_least = utils.libtpu_at_least
libtpu_version = utils.libtpu_version
parse_version = utils.parse_version
