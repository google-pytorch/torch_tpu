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

"""Tools and helper routines common to all examples and benchmarks."""

import collections
from collections.abc import MutableSequence
import contextlib
import enum
import functools
import os
import re
import sys
import time
from typing import Any, Callable, Sequence, TextIO, TypeAlias

import numpy as np
import torch
from torch.utils import _pytree
from torch_tpu._internal import execution_mode
from torch_tpu._internal import export as torch_tpu_export
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile

EagerMode: TypeAlias = execution_mode.EagerMode


# A tolerance value can be either a float or a callable that takes the expected
# value as the only argument and returns the tolerance to use for comparing
# the expected and the actual values. This allows the tolerance to depend on
# the value range, and thus be tighter than the same global tolerance.
Tolerance: TypeAlias = float | Callable[[float], float]


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


_RE_INDEX_CAPTURE = re.compile(r"at index \((.*)\) \(up to .* allowed\)")

# Regex for extracting the actual absolute and relative differences for both
# tensor and scalar outputs.
_RE_TOLERANCE = re.compile(
    r"\n("
    # Tensor tolerance error message.
    r"Greatest absolute difference: (?P<tensor_atol>\S*) at index.*\n"
    r"Greatest relative difference: (?P<tensor_rtol>\S*) at index.*"
    r"|"
    # Scalar tolerance error message.
    r"Absolute difference: (?P<scalar_atol>\S*) \(up to .*\).*\n"
    r"Relative difference: (?P<scalar_rtol>\S*) \(up to .*\)"
    r")"
)

_RE_STRICT_REL_FAILURE = re.compile(r"\(up to 1.*e-07 allowed\)")

# Regex for identifying scalar outputs.
_RE_SCALAR_COMP_FAILURE = re.compile(r"Scalars are not (equal|close)!")


def assert_close(
    actual: Any,
    expected: Any,
    rtol: float | None = None,
    atol: Tolerance | None = None,
    preamble: str | None = None,
    check_value: CheckValueMode = CheckValueMode.STRICT,
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
    check_value: The mode for checking the values.
    check_dtype: Check if the dtypes are the same. Use sparsingly, only when
      dtypes are different due to PyTorch inconsistency between their CPU and
      GPU backend.
  """

  if actual is None or expected is None:
    assert actual is None and expected is None, (
        "expected and actual values for comparison must both be None or not"
        f" None, got {actual=} and {expected=}"
    )
    return

  if not check_dtype:
    promoted_dtype = torch.promote_types(actual.dtype, expected.dtype)
    actual = actual.to(promoted_dtype)
    expected = actual.to(promoted_dtype)

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


def _assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float | None = None,
    atol: Tolerance | None = None,
    preamble: str | None = None,
    check_value: CheckValueMode = CheckValueMode.STRICT,
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
    check_value: The mode for checking the values.
  """

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

  def _add_tolerance_suggestions(msg: str, check_mode: CheckValueMode) -> str:
    """Augments the error message with tolerance suggestions."""
    # We use regex to find the max absolute and relative differences reported by
    # torch.testing.assert_close.
    #
    # Example lines (for tensor outputs):
    # Greatest absolute difference: 0.02114 at index (21, 36) (up to 1e-07 ...)
    # Greatest relative difference: 23.9766 at index (41, 104) (up to 0.1 ...)
    #
    # Example lines (for scalar outputs):
    # Expected 0.5 but got 0.5029680728912354.
    # Absolute difference: 0.0029680728912353516
    # Relative difference: 0.005936145782470703

    # Try to find the actual tolerances.
    tol_match = _RE_TOLERANCE.search(msg)

    assert tol_match, (
        "could not extract the tolerance from the error message given by"
        " PyTorch `torch.testing.assert_close` function, using the"
        " `_RE_TOLERANCE` regex. You need to update the regex to fix this"
        " issue.\n"
        " \n"
        " Error message:\n"
        + msg
    )

    def get_tol(tol: str) -> float:
      """Returns the tolerance value from the tensor or scalar match."""
      for ty in ("scalar", "tensor"):
        val = tol_match.group(f"{ty}_{tol}")
        if val is not None:
          return float(val)
      raise ValueError(
          f"could not find tolerance '{tol}' in: {tol_match.groupdict()}"
      )

    try:
      max_abs = get_tol("atol")
      max_rel = get_tol("rtol")
    except ValueError:
      return msg

    def _format_sci_ceil(val: float) -> str:
      """Formats a float in scientific notation, rounded up to 1 decimal."""
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

    suggestion_lines = ["\n\nTolerance Suggestions:"]

    if check_mode == CheckValueMode.LOOSE:
      suggestion_lines.append("\n  Loose check failed.")
      suggestion_lines.append("\n  To pass in LOOSE mode, you need either:")
      if np.isfinite(max_rel):
        suggestion_lines.append(
            f"\n    - rtol >= {max_rel} ({_format_sci_ceil(max_rel)}) (with"
            " atol=0)"
        )
      else:
        suggestion_lines.append(
            "\n    - (rtol check impossible due to zero expected values)"
        )
      suggestion_lines.append(
          f"\n    - OR atol >= {max_abs} ({_format_sci_ceil(max_abs)}) (with"
          " rtol=0)"
      )
      suggestion_lines.append(
          "\n    - OR a combination such that atol + rtol * |expected| covers"
          " the diffs."
      )

    elif check_mode == CheckValueMode.STRICT:
      suggestion_lines.append("\n  Strict check failed.")
      suggestion_lines.append("\n  To pass in STRICT mode, you need BOTH:")

      if np.isinf(max_rel):
        suggestion_lines.append(
            "\n    - rtol check is IMPOSSIBLE because `expected` is 0 but"
            " `actual` is not."
        )
        suggestion_lines.append("\n      (Use LOOSE mode or fix the values)")
      else:
        suggestion_lines.append(
            f"\n    - rtol >= {max_rel} ({_format_sci_ceil(max_rel)})"
        )

      suggestion_lines.append(
          f"\n    - atol >= {max_abs} ({_format_sci_ceil(max_abs)})"
      )

    return msg + "".join(suggestion_lines)

  def refine_assert_close_message(msg: str) -> str:
    """Refines the message from torch.testing.assert_close with more details."""

    # The message from torch.testing.assert_close looks like (for scalars):
    #
    # Scalars are not equal!
    #
    # Expected 0.5 but got 0.5029680728912354.
    # Absolute difference: 0.0029680728912353516
    # Relative difference: 0.005936145782470703
    #
    # or (for tensors):
    #
    # ...
    # Greatest absolute difference: 1.8008867502212524 at index (2, 0, 0) \
    # (up to 0 allowed)
    # Greatest relative difference: 0.10574676096439362 at index (1, 2, 2) \
    # (up to 1e-05 allowed)
    #
    # We want to show the actual and expected tensor elements at the given
    # indices, so we need to parse the message and extract the indices.
    details = ""
    if _RE_SCALAR_COMP_FAILURE.search(msg):
      details += f"\nAbsolute difference allowed: up to {atol:.3g}\n"
      details += f"Relative difference allowed: up to {rtol:.3g}\n"
    else:
      for index in get_indices(msg):
        golden_elem = expected[index]
        torch_tpu_elem = actual[index]
        diff = abs(golden_elem - torch_tpu_elem)
        rel_diff = 1 if golden_elem == 0 else diff / abs(golden_elem)
        details += (
            f"    at index {index}, expected={golden_elem},"
            f" actual={torch_tpu_elem},"
            f" relative diff={rel_diff},"
            f" diff={diff}\n"
        )
    return (
        f"{preamble}{msg}\n{details}\n"
        f"Expected tensor:\n{expected}\n\n"
        f"Actual tensor:\n{actual}"
    )

  # If atol is a callable, we don't check the relative tolerance, so we always
  # use the loose mode.
  if check_value == CheckValueMode.LOOSE or callable(atol):

    def msg_handler(msg: str) -> str:
      return _add_tolerance_suggestions(
          refine_assert_close_message(msg), CheckValueMode.LOOSE
      )

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
      return _add_tolerance_suggestions(msg, CheckValueMode.STRICT)

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


def model_memory(
    model: torch.nn.Module, dtype: torch.dtype = torch.float32
) -> float:
  """Estimate the physical memory requirements for a model.

  This function counts the total number of parameters plus
  the parameters that require gradients. It also collects
  all model buffers before finally multiplying this sum
  with the size of the model's data type.

  The function currently assumes that the whole model
  utilizes a single dtype.

  Args:
    model:        The PyTorch model to analyze.
    dtype:        The assumed dtype for the whole model.

  Returns:
    The estimated model memory requirements in GB.
  """
  total_params = 0
  for param in model.parameters():
    total_params += param.numel() * 2 if param.requires_grad else param.numel()
  total_buffers = sum(buf.numel() for buf in model.buffers())

  element_size = torch.tensor(0, dtype=dtype).element_size()
  total_memory_bytes = (total_params + total_buffers) * element_size

  return total_memory_bytes / (1024.0**3)


@contextlib.contextmanager
def open_output(
    filepath: str | None, mode: str = "w", default: TextIO = sys.stdout
):
  """Opens an output file if filepath is provided.

  Args:
    filepath: The path to the file to open. If None or empty, the default file
      is used.
    mode: The mode to open the file in.
    default: The default file to use if filepath is None.

  Only closes the file if it was opened (i.e., not the default).
  """
  if filepath:
    f = open(os.path.abspath(os.path.expanduser(filepath)), mode)
    try:
      yield f
    finally:
      f.close()
  else:
    yield default


def format_model(
    model: torch.nn.Module | Callable[..., Any],
    *input_tensors: torch.Tensor | tuple[torch.Tensor, ...],
    params: bool = False,
    pt: bool = False,
    aten: bool = False,
    shlo: bool = False,
) -> str:
  """Formats a model or tensor function in various ways.

  Use the various print method to provide details about a given model.

  Args:
    model:          The PyTorch model or tensor function to format.
    *input_tensors: The input tensor(s) to the model.
    params:         If True, include model parameter sizes.
    pt:             If True, include the output of PyTorch's print(model).
    aten:           If True, utilize OpTracer to collect and print the aten ops.
    shlo:           If True, dump generated StableHLO.

  Raises:
    NotImplementedError: For unrecognized dumper options.

  Returns:
    The formatted model as a string.
  """

  result = ""
  if not isinstance(model, torch.nn.Module):
    if isinstance(input_tensors, torch.Tensor):
      placeholders = (tpu_torch_compile.placeholder_like(input_tensors),)
    else:
      to_placeholder = lambda x: tpu_torch_compile.placeholder_like(x)
      placeholders = _pytree.tree_map_only(
          torch.Tensor, to_placeholder, input_tensors
      )

    # Trace the model in INTERNAL_DEFER_ALL mode so that we can capture the full
    # graph.
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
      model = torch.fx.experimental.proxy_tensor.make_fx(model)(*placeholders)

  if params:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result += f"Parameters: {total}, Trainable: {train}\n"
    result += (
        f"Estimated Memory: {model_memory(model):.3f} "
        "GB (based on # of parameters and dtype)\n"
    )

  if pt:
    # If printing GraphModule, use print_readable to include type information.
    module_str = (
        model.print_readable(print_output=False)
        if isinstance(model, torch.fx.GraphModule)
        else str(model)
    )
    result += module_str + "\n"

  if aten:
    with OpTracer() as tracer:
      model(*input_tensors)
    result += tracer._pformat() + "\n"  # pylint: disable=protected-access

  if shlo:
    if isinstance(model, torch.fx.GraphModule):
      exported = torch.export.export(model, args=(*input_tensors,))
      mlir_text = torch_tpu_export.exported_to_mlir(exported).serialize_text()
      result += mlir_text + "\n"
    else:
      with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
        results = model(*input_tensors)

      shlo = sync.computation_mlir(results)
      result += shlo + "\n"

  return result


def format_tensor(
    tensor: torch.Tensor, desc: str = "", data: bool = False
) -> str:
  """Formats the most common properties of a PyTorch tensor.

  Args:
      tensor: The tensor to inspect.
      desc:   An optional descriptive string.
      data:   If True, also include the tensor's contents. If the tensor is on a
        device, it is first copied to CPU.

  Returns:
      A string representation of the tensor's properties.
  """

  if not isinstance(tensor, torch.Tensor):
    return f"Error: Expected a torch.Tensor, but got {type(tensor)}"

  lines = []
  if desc:
    lines.append(desc)
  lines.append(f"Shape (Size):     {tensor.shape}")
  lines.append(f"Number of Dims:   {tensor.ndim}")
  lines.append(f"Data Type (dtype):{tensor.dtype}")
  lines.append(f"Device:           {tensor.device}")
  lines.append(f"Min:              {tensor.cpu().min().item():.2f}")
  lines.append(f"Max:              {tensor.cpu().max().item():.2f}")
  lines.append(f"Mean:             {tensor.cpu().mean().item():.2f}")
  lines.append(f"Var:              {tensor.cpu().var().item():.2f}")
  lines.append(f"Requires Grad:    {tensor.requires_grad}")
  lines.append(f"Memory Layout:    {tensor.layout}")
  lines.append(f"Is a Leaf:        {tensor.is_leaf}")
  lines.append(f"Memory Usage (B): {tensor.nbytes}")
  lines.append(f"Strides:          {tensor.stride()}")
  if tensor.grad_fn is not None:
    lines.append(f"Gradient Function: {tensor.grad_fn}")
  else:
    lines.append("Gradient Function: None (or is a leaf tensor)")

  if data:
    if str(tensor.device) != "cpu":
      lines.append(tensor.to("cpu"))
    else:
      lines.append(str(tensor))

  return "\n".join(lines)


def get_tensor_summary(tensor: torch.Tensor, data: bool = True) -> str:
  """Returns a one-line, compact summary of a PyTorch tensor.

  The summary includes shape, dtype, device, min, max, mean, variance, and
  requires grad.

  Example output:
  (2, 3) Min:-0.85 Max:1.69 Mean:0.31 Var:0.77 Grad:Y float32 cpu

  Note: variance is calculated with correction=0, since we want the variance of
  the actual values, not the estimated variance a sample of a population.

  Args:
    tensor (torch.Tensor): The tensor to summarize.
    data (bool): If True, also include details on tensors data (Min, Max, etc)
      else only returns shape and dtype info.

  Returns:
    A compact string representation of the tensor with summary statistics.

  Raises:
    TypeError: If the tensor is not a torch.Tensor.
  """
  if not isinstance(tensor, torch.Tensor):
    raise TypeError(f"Expected a torch.Tensor, but got {type(tensor)}")

  shape = tuple(tensor.shape)
  grad = "Y" if tensor.requires_grad else "N"
  dtype = str(tensor.dtype).split("torch.")[-1]
  device = tensor.device

  if not data:
    template = "torch.tensor(...{shape}, dtype={dtype}, device={device}{grad})"
    grad = ", requires_grad=True" if tensor.requires_grad else ""
    return template.format(
        shape=shape, dtype=tensor.dtype, device=f"'{device}'", grad=grad
    )

  template = (
      "{shape}  Min:{min_val: .2f}  Max:{max_val: .2f}  Mean:{mean_val: .2f}  "
      "Var:{var_val: .2f}  Grad:{grad}  {dtype}  {device}"
  )

  if tensor.numel() > 0:
    t_cpu = tensor.to("cpu")
    min_val = t_cpu.min().item()
    max_val = t_cpu.max().item()
    if tensor.dtype.is_floating_point:
      mean_val = t_cpu.mean().item()
      var_val = t_cpu.var(correction=0).item()
    else:
      mean_val = t_cpu.float().mean().item()
      var_val = t_cpu.float().var(correction=0).item()
    return template.format(
        shape=shape,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        var_val=var_val,
        grad=grad,
        dtype=dtype,
        device=device,
    )
  else:
    return (
        f"{shape}  Min:n/a  Max:n/a  Mean:n/a  Var:n/a  Grad:{grad}  {dtype}  "
        f"{device}"
    )


class _InputArgumentMetadata:
  """A class to log a single argument's metadata.

  Note the stringification is pushed into the `__str__` method which makes
  this class more beneficial for python logging.
  """

  def __init__(self, arg: Any):
    self._arg = arg

  def __str__(self):
    if isinstance(self._arg, torch.Tensor):
      return get_tensor_summary(self._arg, data=False)
    return str(self._arg)

  def __repr__(self):
    return self.__str__()


class InputMetadata:
  """A class to log call inputs without requiring access to their data.

  This avoids the need to send any part of a tensor back to the host when
  printing. This also defers the evaluation of the text until `__str__` eval
  time allowing for lazy evaluation in logging.
  """

  def __init__(self, inputs: Sequence[Any]):
    self._inputs = _pytree.tree_map_only(
        torch.Tensor, _InputArgumentMetadata, inputs
    )
    if not isinstance(self._inputs, Sequence):
      self._inputs = [self._inputs]

  def __str__(self):
    return str(self._inputs)


class OpTracer(
    torch.utils._python_dispatch.TorchDispatchMode,  # pylint: disable=protected-access
    contextlib.ContextDecorator,
):
  """This context manager intercepts and stores PyTorch ops.

  It can also be used a decorator.

  Example usage as context manager:
    x = torch.tensor(2.0)
    with OpTracer() as tracer:
      _ = x ** 0.5
    print(tracer_utils.pformat_op_tracer(tracer))

  Example usage as a wrapper:
    def f(x):
      return x ** 0.5

    tracer = OpTracer()
    wrapper = tracer(f)
    _ = wrapper(torch.tensor(2.0))
    print(tracer_utils.pformat_op_tracer(tracer))

  Example usage as a decorator:
    tracer = OpTracer()
    @tracer
    def f(x):
      return x ** 0.5
    _ = f(torch.tensor(2.0))
    print(tracer_utils.pformat_op_tracer(tracer))

  Attributes:
    ops_log (dict[str, set[str]]): A dict mapping stringified namespaces to a
      set of stringified ops.
  """

  def __enter__(self):
    super().__enter__()  # pytype: disable=attribute-error
    self.ops_log = collections.defaultdict(dict)
    return self

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    aten = str(func)
    mapping = self.ops_log[func.namespace]
    mapping[aten] = mapping.get(aten, 0) + 1
    return func(*args, **kwargs)

  def num_atens(self) -> int:
    return sum(self.ops_log["aten"].values())

  ### TODO(yho): This really belongs in tracer_utils.py, but format_model
  ### uses it, so we'll leave it here for now.
  def _pformat(self) -> str:
    """Pretty-formats the collected data from OpTracer.

    Returns:
      A string representation of the collected data.
    """
    lines = []

    for key in sorted(self.ops_log):
      num_ops = len(self.ops_log[key])
      lines.append(f"Ops for namespace: '{key}' ({num_ops} ops)")
      for op in self.ops_log[key]:
        num_calls = self.ops_log[key][op]
        lines.append(f"  {op}: {num_calls}")
    return "\n".join(lines)


Event: TypeAlias = dict[str, Any]


class ActivationTracer(contextlib.ContextDecorator):
  """Traces activations in a model's forward and backward pass.

  This uses PyTorch's hooks API to intercept nn.Modules and
  log the input and output tensors of each module.

  Example usage:
    model = MyModel()
    with ActivationTracer(model) as collector:
      # Forward pass within this block will trigger hooks
      output = model(input_tensor)
    forward_log = collector.forward_log
    forward_pre_log = collector.forward_pre_log
    print(collector.pformat())

  Attributes:
    forward_pre_log: Events logged before the forward pass of each module.
    forward_log: Events logged after the forward pass of each module.
    backward_pre_log: Events logged before the backward pass of each module.
    backward_log: Events logged after the backward pass of each module.
  """

  forward_pre_log: MutableSequence[Event]
  forward_log: MutableSequence[Event]
  backward_pre_log: MutableSequence[Event]
  backward_log: MutableSequence[Event]

  def __init__(self, model: torch.nn.Module):
    self._model = model
    self._hook_handles: MutableSequence[torch.utils.hooks.RemovableHandle] = []
    self.forward_pre_log: MutableSequence[Event] = []
    self.forward_log: MutableSequence[Event] = []
    self.backward_pre_log: MutableSequence[Event] = []
    self.backward_log: MutableSequence[Event] = []

  # Based on signature at
  # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
  def _forward_pre_hook(
      self,
      module: torch.nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      depth: int,
  ) -> None:
    """Logs the forward pass of a module. Called before a forward pass.

    Args:
      module: The module that was called.
      args: The positional arguments of the module.
      kwargs: The keyword arguments of the module.
      depth: The depth of the module in the model hierarchy.
    """

    self.forward_pre_log.append({
        "module": module,
        "args": args,
        "kwargs": kwargs,
        "depth": depth,
        "time": time.perf_counter(),
        "idx": self._idx,
    })
    self._idx += 1

  # Based on signature at
  # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
  def _forward_hook(
      self,
      module: torch.nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      output: Any,
      depth: int,
  ) -> None:
    """Logs the forward pass of a module. Called after a forward pass.

    Args:
      module: The module that was called.
      args: The positional arguments of the module.
      kwargs: The keyword arguments of the module.
      output: The output tensor(s) of the module.
      depth: The depth of the module in the model hierarchy.
    """

    self.forward_log.append({
        "module": module,
        "args": args,
        "kwargs": kwargs,
        "output": output,
        "depth": depth,
        "time": time.perf_counter(),
        "idx": self._idx,
    })
    self._idx += 1

  # Based on signature at
  # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook
  def _backward_pre_hook(
      self,
      module: torch.nn.Module,
      grad_output: tuple[torch.Tensor, ...],
      depth: int,
  ) -> None:
    """Logs the backward pass of a module. Called before a backward pass.

    Args:
      module: The module that was called.
      grad_output: The gradient of the output tensor(s) of the module.
      depth: The depth of the module in the model hierarchy.
    """

    self.backward_pre_log.append({
        "module": module,
        "grad_output": grad_output,
        "depth": depth,
        "time": time.perf_counter(),
        "idx": self._idx,
    })
    self._idx += 1

  # Based on signature at
  # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
  def _backward_hook(
      self,
      module: torch.nn.Module,
      grad_input: tuple[torch.Tensor, ...],
      grad_output: tuple[torch.Tensor, ...],
      depth: int,
  ) -> None:
    """Logs the backward pass of a module. Called after a backward pass.

    Args:
      module: The module that was called.
      grad_input: The gradient of the input tensor(s) to the module.
      grad_output: The gradient of the output tensor(s) of the module.
      depth: The depth of the module in the model hierarchy.
    """

    self.backward_log.append({
        "module": module,
        "grad_input": grad_input,
        "grad_output": grad_output,
        "depth": depth,
        "time": time.perf_counter(),
        "idx": self._idx,
    })
    self._idx += 1

  def _register_hooks_recursive(self, module: torch.nn.Module, depth: int):
    """Registers hooks on the current module and recursively on its children."""
    self._hook_handles.append(
        module.register_forward_pre_hook(
            functools.partial(self._forward_pre_hook, depth=depth),
            with_kwargs=True,
        )
    )
    self._hook_handles.append(
        module.register_forward_hook(
            functools.partial(self._forward_hook, depth=depth), with_kwargs=True
        )
    )
    self._hook_handles.append(
        module.register_full_backward_pre_hook(
            functools.partial(self._backward_pre_hook, depth=depth)
        )
    )
    self._hook_handles.append(
        module.register_full_backward_hook(
            functools.partial(self._backward_hook, depth=depth)
        )
    )

    # Only increase depth for non-container modules.
    child_depth = (
        depth
        if isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict))
        else depth + 1
    )
    for _, child in module.named_children():
      self._register_hooks_recursive(child, child_depth)

  def __enter__(self):
    self.forward_log.clear()
    self.forward_pre_log.clear()
    self.backward_log.clear()
    self.backward_pre_log.clear()
    self._idx = 0

    self._register_hooks_recursive(self._model, 0)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # Unregister and remove the hooks at the end of the context manager
    # in case the use wants to call the model outside the context manager.
    for handle in self._hook_handles:
      handle.remove()
    self._hook_handles.clear()
    return False
