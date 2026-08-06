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

"""Fuzz tests for ops.

For each op, we test that the result on TPU matches the result on the golden
device (either CPU or GPU, depending on the test mode).
"""

import collections.abc
import copy
import os
import sys
from typing import Any

from absl import flags
from absl.testing import absltest
import torch
from torch.nn import attention
from torch.testing._internal import common_methods_invocations  # pylint: disable=unused-import
from torch_tpu._internal.utils import test_utils
from tests import dynamism_test_utils
from tests import op_testing
from tests import oss_utils

_TEST_CATEGORIES = flags.DEFINE_list(
    "test_categories",
    [],
    (
        "List of test categories to include or exclude (prefixed with '-')."
        " By default, all test categories are included. The order of entries"
        " does not matter, and duplicates are ignored. If any category is"
        " excluded (e.g., '-foreach'), any test in that category is skipped."
        " If any categories are included, a test must belong to at least one"
        " included category to run. Exclusions take precedence over"
        " inclusions. E.g., --test_categories=foreach or"
        " --test_categories=-foreach."
        " This flag is ignored if --test_filter is specified on the bazel"
        " command-line, in order to respect the user's intent to run the"
        " specific tests."
    ),
)


def category(*names):
  """Decorator to associate categories with test methods."""

  def decorator(func):
    func.categories = set(names)
    return func

  return decorator


def _has_test_filter() -> bool:
  """Returns True if a test filter was specified on the command-line or env.

  How Bazel and Blaze handle --test_filter:
  Per the Bazel Test Encyclopedia
  (https://bazel.build/reference/test-encyclopedia#initial-environment):
  - '--test_filter' is a build-tool flag for Blaze and Bazel, not a binary flag.
  - When 'blaze test --test_filter=<filter>' or 'bazel test --test_filter=...'
    is executed, Bazel/Blaze passes the filter string to the test executable
    via the 'TESTBRIDGE_TEST_ONLY' environment variable.
  - Python's absltest framework inspects 'TESTBRIDGE_TEST_ONLY' during main()
    initialization and converts it into '-k=<filter>' arguments in sys.argv.
  - If the test binary is executed directly (outside Bazel/Blaze), filters may
    also be passed via sys.argv flags ('-k', '--test_filter',
    '--default_filter').
  """
  # 1. Check for TESTBRIDGE_TEST_ONLY env var set by Bazel or Blaze.
  if os.environ.get("TESTBRIDGE_TEST_ONLY"):
    return True

  # 2. Check for absl flags if registered.
  if hasattr(flags.FLAGS, "test_filter") and flags.FLAGS["test_filter"].present:
    return True

  # 3. Check for specific filter flags passed in sys.argv (-k, --test_filter,
  #    --default_filter).
  for arg in sys.argv[1:]:
    if arg in ("--test_filter", "--default_filter", "-k") or arg.startswith(
        ("--test_filter=", "--default_filter=", "-k=")
    ):
      return True

  return False


# In this file, we use the following naming convention for variables:
# - golden_*: a value for the device used for computing the golden results
#   (either CPU or GPU)
# - tpu_*: a value for the TPU device

COMPLEX_DTYPES = op_testing.COMPLEX_DTYPES
FLOAT_DTYPES = op_testing.FLOAT_DTYPES
INTEGRAL_DTYPES = op_testing.INTEGRAL_DTYPES
NUMERIC_DTYPES = op_testing.NUMERIC_DTYPES

CheckValueMode = test_utils.CheckValueMode

# OpInput and OpOutput are not used directly in this file, but needed for
# pickling the golden results.
OpInput = op_testing.OpInput
OpOutput = op_testing.OpOutput
OpVariant = op_testing.OpVariant
TorchTpuTestBase = op_testing.TorchTpuTestBase


def _expm1_atol(y: float) -> float:
  """Returns the absolute tolerance for expm1() given the expected value y."""

  # For expm1 (i.e. y = e^x - 1), we especially care about the precision when
  # y is close to 0. However, a tolerance tight enough for this case is
  # too tight for y >> 0. Without variable-tolerance, we'd be forced to
  # pick a large tolerance for all cases, which would cause unnecessary
  # slack in our tests.
  #
  # Using variable-tolerance, we:
  #   - use a small tolerance for y < 0,
  #   - use a large (relative) tolerance for y > 1,
  #   - linearly interpolate the tolerance for y in [0, 1].
  #
  # This allows the test to be tight for y near 0 or being negative, while
  # allowing more slack for y >> 0.
  atol0 = 3e-6  # small tolerance
  atol1 = 1e-5  # large tolerance
  if y < 0:
    return atol0
  if y > 1:
    return atol1 * y
  return atol0 + (atol1 - atol0) * y


# Overrides the default relative and absolute tolerance for the given op and
# dtype between TPU and CPU. Two tensors are considered to match if
#
#   abs(tpu - cpu) <= atol + rtol * abs(cpu)
#
# (A common misconception is that the criteria is
#   abs(tpu - cpu) <= min(atol, rtol * abs(cpu))
# but that's incorrect.)
#
# The overrides apply to both the op and its inplace variant.
#
# The default tolerances are defined here:
#   https://docs.pytorch.org/docs/stable/testing.html
ACCURACY_OVERRIDES_VS_CPU: dict[str, dict[torch.dtype, dict[str, float]]] = {
    # go/keep-sorted start
    "_foreach_acos": {
        torch.complex64: {"rtol": 3.6e-5, "atol": 8e-5},
    },
    "_foreach_add": {
        torch.bfloat16: {"rtol": 7.6e-1, "atol": 2.9e-2},
        torch.float16: {"rtol": 3.6e-2, "atol": 4e-3},
    },
    "_foreach_addcdiv": {
        torch.bfloat16: {"rtol": 1, "atol": 5.5e-2},
        torch.float16: {"rtol": 2.8e-1, "atol": 6.3e-2},
    },
    "_foreach_addcmul": {
        torch.bfloat16: {"rtol": 1, "atol": 1.9e-1},
        torch.float16: {"rtol": 1.9e-1, "atol": 2.5e-1},
    },
    "_foreach_asin": {
        torch.complex64: {"rtol": 1.3e-4, "atol": 1.1e-4},
    },
    "_foreach_atan": {
        torch.complex64: {"rtol": 1.8e-5, "atol": 2.6e-5},
    },
    "_foreach_cos": {
        torch.complex64: {"rtol": 4.2e-6, "atol": 1.3e-2},
    },
    "_foreach_cosh": {
        torch.complex64: {"rtol": 4.3e-6, "atol": 1.3e-2},
        torch.float32: {"rtol": 3.9e-6, "atol": 1.3e-2},
        torch.int16: {"rtol": 3.4e-6, "atol": 1.2e-3},
        torch.int32: {"rtol": 3.4e-6, "atol": 1.2e-3},
        torch.int64: {"rtol": 3.4e-6, "atol": 1.2e-3},
        torch.int8: {"rtol": 3.4e-6, "atol": 1.2e-3},
        torch.uint8: {"rtol": 3.4e-6, "atol": 1.2e-3},
    },
    "_foreach_exp": {
        torch.complex64: {"rtol": 3.9e-6, "atol": 2.1e-2},
        torch.float32: {"rtol": 3.9e-6, "atol": 2.2e-2},
        torch.int16: {"rtol": 3.7e-6, "atol": 2.4e-3},
        torch.int32: {"rtol": 3.7e-6, "atol": 2.4e-3},
        torch.int64: {"rtol": 3.7e-6, "atol": 2.4e-3},
        torch.int8: {"rtol": 3.7e-6, "atol": 2.4e-3},
        torch.uint8: {"rtol": 3.7e-6, "atol": 2.4e-3},
    },
    "_foreach_expm1": {
        torch.complex64: {"rtol": 1e-4, "atol": 2e-2},
        torch.float32: {"rtol": 1e-4, "atol": 3e-2},
        torch.int16: {"rtol": 3.7e-6, "atol": 1.1e-2},
        torch.int32: {"rtol": 3.7e-6, "atol": 1.1e-2},
        torch.int64: {"rtol": 3.7e-6, "atol": 1.1e-2},
        torch.int8: {"rtol": 3.7e-6, "atol": 1.1e-2},
        torch.uint8: {"rtol": 3.7e-6, "atol": 1.1e-2},
    },
    "_foreach_frac": {
        torch.bfloat16: {"rtol": 1e-3, "atol": 1},
        torch.float16: {"rtol": 1e-3, "atol": 1},
        torch.float32: {"rtol": 1e-5, "atol": 1},
        torch.float64: {"rtol": 1e-5, "atol": 1.1},
    },
    "_foreach_lerp": {
        torch.bfloat16: {"rtol": 1.5, "atol": 6.1e-2},
        torch.float16: {"rtol": 5.1e-1, "atol": 6.3e-2},
    },
    "_foreach_lgamma": {
        torch.bfloat16: {"rtol": 5e-2, "atol": 1e-2},
        torch.float32: {"rtol": 1e-1, "atol": 1e-3},
    },
    "_foreach_log": {
        torch.complex64: {"rtol": 5.8e-5, "atol": 1.1e-4},
        torch.float32: {"rtol": 2.5e-4, "atol": 9e-5},
        torch.int16: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int32: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int64: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int8: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.uint8: {"rtol": 5.7e-5, "atol": 6.3e-5},
    },
    "_foreach_log10": {
        torch.complex64: {"rtol": 9.4e-5, "atol": 4.4e-5},
        torch.float32: {"rtol": 2.7e-4, "atol": 4.5e-5},
        torch.int16: {"rtol": 5.7e-5, "atol": 2.8e-5},
        torch.int32: {"rtol": 5.7e-5, "atol": 2.8e-5},
        torch.int64: {"rtol": 5.7e-5, "atol": 2.8e-5},
        torch.int8: {"rtol": 5.7e-5, "atol": 2.8e-5},
        torch.uint8: {"rtol": 5.7e-5, "atol": 2.8e-5},
    },
    "_foreach_log1p": {
        torch.complex64: {"rtol": 3.1e-5, "atol": 4.4e-5},
        torch.float32: {"rtol": 2.5e-4, "atol": 6.9e-5},
        torch.int16: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int32: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int64: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int8: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.uint8: {"rtol": 5.7e-5, "atol": 6.3e-5},
    },
    "_foreach_log2": {
        torch.complex64: {"rtol": 1.5e-4, "atol": 1.6e-4},
        torch.float32: {"rtol": 2.6e-4, "atol": 9.9e-5},
        torch.int16: {"rtol": 5.7e-5, "atol": 9.1e-5},
        torch.int32: {"rtol": 5.7e-5, "atol": 9.1e-5},
        torch.int64: {"rtol": 5.7e-5, "atol": 9.1e-5},
        torch.int8: {"rtol": 5.7e-5, "atol": 9.1e-5},
        torch.uint8: {"rtol": 5.7e-5, "atol": 9.1e-5},
    },
    "_foreach_pow": {
        torch.int8: {"rtol": 1, "atol": 1},
        torch.int16: {"rtol": 1, "atol": 1},
        torch.int32: {"rtol": 1, "atol": 1},
        torch.int64: {"rtol": 1, "atol": 1},
        torch.complex64: {"rtol": 1e-3, "atol": 1e-3},
        torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    },
    "_foreach_reciprocal": {
        torch.float32: {"rtol": 1.2e-07, "atol": 1.6e-5},
    },
    "_foreach_sigmoid": {
        torch.complex64: {"rtol": 1.6e-5, "atol": 9.9e-5},
        torch.float16: {"rtol": 6.2e-2, "atol": 1.6e-5},
        torch.float32: {"rtol": 6.2e-2, "atol": 1.6e-5},
        torch.int16: {"rtol": 5.9e-3, "atol": 1.5e-5},
        torch.int32: {"rtol": 5.9e-3, "atol": 1.5e-5},
        torch.int64: {"rtol": 5.9e-3, "atol": 1.5e-5},
        torch.int8: {"rtol": 5.9e-3, "atol": 1.5e-5},
        torch.uint8: {"rtol": 1.5e-5, "atol": 1.5e-5},
    },
    "_foreach_sin": {
        torch.complex64: {"rtol": 4.1e-6, "atol": 9.6e-3},
    },
    "_foreach_sinh": {
        torch.complex64: {"rtol": 3.8e-6, "atol": 1.2e-2},
        torch.float32: {"rtol": 4e-6, "atol": 1.1e-2},
        torch.int16: {"rtol": 3.3e-6, "atol": 1.1e-3},
        torch.int32: {"rtol": 3.3e-6, "atol": 1.1e-3},
        torch.int64: {"rtol": 3.3e-6, "atol": 1.1e-3},
        torch.int8: {"rtol": 3.3e-6, "atol": 1.1e-3},
        torch.uint8: {"rtol": 3.3e-6, "atol": 1.1e-3},
    },
    "_foreach_sub": {
        torch.bfloat16: {"rtol": 7.6e-2, "atol": 3.2e-2},
        torch.float16: {"rtol": 1.1e-2, "atol": 4e-3},
    },
    "_foreach_tan": {
        torch.complex64: {"rtol": 2.1e-5, "atol": 3.1e-4},
    },
    "_foreach_tanh": {
        torch.complex64: {"rtol": 3.2e-5, "atol": 1.8e-3},
        torch.float32: {"rtol": 5.2e-5, "atol": 3.3e-5},
        torch.int16: {"rtol": 3e-5, "atol": 3e-5},
        torch.int32: {"rtol": 3e-5, "atol": 3e-5},
        torch.int64: {"rtol": 3e-5, "atol": 3e-5},
        torch.int8: {"rtol": 3e-5, "atol": 3e-5},
        torch.uint8: {"rtol": 3e-5, "atol": 3e-5},
    },
    "_log_softmax_backward_data": {
        torch.bfloat16: {"rtol": 1.5e-1, "atol": 6.3e-2},
        torch.float16: {"rtol": 1.4e-2, "atol": 3.2e-2},
        torch.float32: {"rtol": 6.9e-5, "atol": 2.3e-4},
    },
    "_native_batch_norm_legit": {
        torch.bfloat16: {"rtol": 7.4e-2, "atol": 3.1e-2},
        torch.float16: {"rtol": 4.4e-3, "atol": 4e-3},
        torch.float32: {"rtol": 1e-4, "atol": 5e-4},
    },
    "_softmax_backward_data": {
        torch.bfloat16: {"rtol": 2.9, "atol": 4.7e-2},
        torch.float16: {"rtol": 1.3, "atol": 4.3e-3},
    },
    "_thnn_fused_gru_cell": {
        torch.bfloat16: {"rtol": 2.9e-1, "atol": 2.9e-5},
        torch.complex64: {"rtol": 1.2e-4, "atol": 3.3e-2},
        torch.float16: {"rtol": 5.3e-2, "atol": 2.8e-5},
        torch.float32: {"rtol": 8.1e-3, "atol": 8.7e-5},
    },
    "_thnn_fused_lstm_cell": {
        torch.complex64: {"rtol": 2.9e-4, "atol": 7.1e-4},
        torch.float16: {"rtol": 4.8e-2, "atol": 3.1e-5},
        torch.float32: {"rtol": 4.8e-2, "atol": 4.7e-5},
    },
    "abs": {
        torch.float64: {"rtol": 4.7e-6, "atol": 1e-6},
    },
    "acos": {
        torch.complex64: {"rtol": 8.9e-5, "atol": 9.8e-5},
    },
    "acosh": {
        torch.complex64: {"rtol": 3.4e-5, "atol": 7e-5},
        torch.float32: {"rtol": 2.7e-5, "atol": 4.4e-5},
        torch.int16: {"rtol": 2.3e-5, "atol": 5.7e-5},
        torch.int32: {"rtol": 2.3e-5, "atol": 5.7e-5},
        torch.int64: {"rtol": 2.3e-5, "atol": 5.7e-5},
        torch.int8: {"rtol": 2.3e-5, "atol": 5.7e-5},
        torch.uint8: {"rtol": 2.3e-5, "atol": 5.7e-5},
    },
    "addcdiv": {
        torch.bfloat16: {"rtol": 3.4e-1, "atol": 1.6e-2},
        torch.float16: {"rtol": 8.2e-2, "atol": 4e-3},
    },
    "addcmul": {
        torch.bfloat16: {"rtol": 2.5e-1, "atol": 4.7e-2},
        torch.float16: {"rtol": 1.8e-2, "atol": 7.9e-3},
    },
    "addmm": {
        torch.bfloat16: {"rtol": 1.9e-2, "atol": 1.5e-2},
        torch.float16: {"rtol": 1.6e-1, "atol": 1.8e-1},
        torch.float32: {"rtol": 1.8e-1, "atol": 1.9e-1},
    },
    "addmv": {
        torch.bfloat16: {"rtol": 1.1e-1, "atol": 2.5e-1},
        torch.float16: {"rtol": 2.4e-2, "atol": 5.1e-2},
    },
    "asin": {
        torch.complex64: {"rtol": 2.6e-4, "atol": 9.6e-5},
    },
    "asinh": {
        torch.bool: {"rtol": 2.7e-5, "atol": 2.4e-5},
        torch.complex64: {"rtol": 3.9e-5, "atol": 8.1e-5},
        torch.float32: {"rtol": 2.5e-4, "atol": 9.5e-5},
        torch.int16: {"rtol": 2.7e-5, "atol": 3.2e-5},
        torch.int32: {"rtol": 2.7e-5, "atol": 3.2e-5},
        torch.int64: {"rtol": 2.7e-5, "atol": 3.2e-5},
        torch.int8: {"rtol": 2.7e-5, "atol": 3.2e-5},
        torch.uint8: {"rtol": 2.7e-5, "atol": 3.2e-5},
    },
    "atan": {
        torch.complex64: {"rtol": 2.1e-5, "atol": 1.8e-5},
    },
    "atanh": {
        torch.complex64: {"rtol": 9.3e-5, "atol": 2.5e-5},
        torch.float32: {"rtol": 2.2e-4, "atol": 6.3e-5},
    },
    "baddbmm": {
        torch.bfloat16: {"rtol": 3.5e-2, "atol": 3.8e-2},
        torch.complex64: {"rtol": 1.2, "atol": 2.5},
        torch.float16: {"rtol": 6.7e-1, "atol": 6.6e-1},
        torch.float32: {"rtol": 6.7e-1, "atol": 5.9e-1},
    },
    "bmm": {
        torch.complex64: {"rtol": 1.1e-1, "atol": 1.9},
        torch.float16: {"rtol": 1.7e-1, "atol": 6.1e-1},
        torch.float32: {"rtol": 1.1e-1, "atol": 5.8e-1},
    },
    "cdist": {
        torch.float16: {"rtol": 1.2e-3, "atol": 7.9e-3},
        torch.float32: {"rtol": 6.0e-3, "atol": 4.1e-2},
    },
    "cos": {
        torch.complex64: {"rtol": 2.8e-6, "atol": 4.8e-4},
    },
    "cosh": {
        torch.complex64: {"rtol": 3.5e-6, "atol": 1.2e-2},
        torch.float32: {"rtol": 3.5e-6, "atol": 1.2e-2},
        torch.int16: {"rtol": 3.4e-6, "atol": 1.2e-3},
        torch.int32: {"rtol": 3.4e-6, "atol": 1.2e-3},
        torch.int64: {"rtol": 3.4e-6, "atol": 1.2e-3},
        torch.int8: {"rtol": 3.4e-6, "atol": 1.2e-3},
        torch.uint8: {"rtol": 3.4e-6, "atol": 1.2e-3},
    },
    "cummin": {
        torch.float64: {"atol": 5},
    },
    "cumprod": {
        torch.bfloat16: {"rtol": 1.7e-1, "atol": 5.9e-3},
        torch.float16: {"rtol": 2.3e-2, "atol": 1.5e-3},
    },
    "cumsum": {
        torch.bfloat16: {"rtol": 1.1e-1, "atol": 1.6e-2},
        torch.float16: {"rtol": 1.5e-2, "atol": 4.9e-3},
    },
    "digamma": {
        torch.uint8: {"rtol": 3.9e-5, "atol": 4.9e-5},
        torch.int8: {"rtol": 3.9e-5, "atol": 4.9e-5},
        torch.int16: {"rtol": 3.9e-5, "atol": 4.9e-5},
        torch.int32: {"rtol": 3.9e-5, "atol": 4.9e-5},
        torch.int64: {"rtol": 3.9e-5, "atol": 4.9e-5},
        torch.float32: {"rtol": 4.7e-5, "atol": 6.1e-5},
    },
    "erfinv": {
        torch.float32: {"rtol": 1.5e-5, "atol": 2e-5},
    },
    "exp": {
        torch.complex64: {"rtol": 3.3e-6, "atol": 4.4e-3},
        torch.float32: {"rtol": 3.8e-6, "atol": 1.8e-2},
        torch.int16: {"rtol": 3.7e-6, "atol": 1.5e-3},
        torch.int32: {"rtol": 3.7e-6, "atol": 1.5e-3},
        torch.int64: {"rtol": 3.7e-6, "atol": 1.5e-3},
        torch.int8: {"rtol": 3.7e-6, "atol": 1.5e-3},
        torch.uint8: {"rtol": 3.7e-6, "atol": 2.4e-3},
    },
    "exp2": {
        torch.bfloat16: {"rtol": 0.032, "atol": 8.4},
        torch.complex64: {"rtol": 4.00e-06, "atol": 2.0e-03},
        torch.float16: {"rtol": 0.0045, "atol": 0.53},
        torch.float32: {"rtol": 3.50e-06, "atol": 2.0e-03},
    },
    "expm1": {
        torch.complex64: {"rtol": 3.9e-6, "atol": 2e-2},
        torch.float32: {"rtol": 5.7e-6, "atol": 2.1e-2},
        torch.int16: {"rtol": 3.7e-6, "atol": 2.4e-3},
        torch.int32: {"rtol": 3.7e-6, "atol": 2.4e-3},
        torch.int64: {"rtol": 3.7e-6, "atol": 2.4e-3},
        torch.int8: {"rtol": 3.7e-6, "atol": 2.4e-3},
        torch.uint8: {"rtol": 3.7e-6, "atol": 2.4e-3},
    },
    "ldexp": {
        torch.complex64: {"rtol": 4.8e-6, "atol": 1.7e-2},
        torch.float16: {"rtol": 1.3e-3, "atol": 1.6e-2},
        torch.float32: {"rtol": 4.9e-6, "atol": 1.6e-2},
    },
    "lgamma": {
        torch.float16: {"rtol": 1.2e-3, "atol": 2.5e-4},
        torch.float32: {"rtol": 8.2e-4, "atol": 1.8e-4},
        torch.int16: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int32: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int64: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int8: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.uint8: {"rtol": 1.3e-4, "atol": 2.3e-4},
    },
    "linalg.lu": {
        torch.float64: {"rtol": 3.6e-5, "atol": 2.1e-6},
    },
    "linalg.lu_factor_ex": {
        torch.float64: {"rtol": 8.8e-6, "atol": 6.8e-6},
    },
    "linalg.solve_ex": {
        torch.float64: {"rtol": 3.5e-6, "atol": 1.2e-6},
    },
    "linalg.vector_norm": {
        torch.bfloat16: {"rtol": 2e-2, "atol": 9.2e-5},
        torch.complex64: {"rtol": 2.5e-6, "atol": 3.5e-5},
        torch.float32: {"rtol": 4e-6, "atol": 3e-5},
    },
    "log": {
        torch.complex64: {"rtol": 4.6e-5, "atol": 9.2e-5},
        torch.float32: {"rtol": 5.9e-5, "atol": 6.2e-5},
        torch.int16: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int32: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int64: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int8: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.uint8: {"rtol": 5.7e-5, "atol": 6.3e-5},
    },
    "log10": {
        torch.complex64: {"rtol": 5.4e-5, "atol": 2.9e-5},
        torch.float32: {"rtol": 1.9e-4, "atol": 2.8e-5},
        torch.int16: {"rtol": 5.7e-5, "atol": 2.8e-5},
        torch.int32: {"rtol": 5.7e-5, "atol": 2.8e-5},
        torch.int64: {"rtol": 5.7e-5, "atol": 2.8e-5},
        torch.int8: {"rtol": 5.7e-5, "atol": 2.8e-5},
        torch.uint8: {"rtol": 5.7e-5, "atol": 2.8e-5},
    },
    "log1p": {
        torch.complex64: {"rtol": 8.9e-5, "atol": 4.7e-5},
        torch.float32: {"rtol": 1.8e-4, "atol": 9e-5},
        torch.int16: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int32: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int64: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.int8: {"rtol": 5.7e-5, "atol": 6.3e-5},
        torch.uint8: {"rtol": 5.7e-5, "atol": 6.3e-5},
    },
    "log2": {
        torch.complex64: {"rtol": 5.8e-5, "atol": 8.8e-5},
        torch.float32: {"rtol": 7e-5, "atol": 9.6e-5},
        torch.int16: {"rtol": 5.7e-5, "atol": 9.1e-5},
        torch.int32: {"rtol": 5.7e-5, "atol": 9.1e-5},
        torch.int64: {"rtol": 5.7e-5, "atol": 9.1e-5},
        torch.int8: {"rtol": 5.7e-5, "atol": 9.1e-5},
        torch.uint8: {"rtol": 5.7e-5, "atol": 9.1e-5},
    },
    "log_softmax": {
        torch.bfloat16: {"rtol": 1, "atol": 7.8e-3},
        torch.float16: {"rtol": 5e-1, "atol": 9.8e-4},
        torch.float32: {"rtol": 2.7e-4, "atol": 6.9e-5},
    },
    # exp/log accumulation plus the associative scan reordering diverge from the
    # sequential CPU reference more than a plain cumulative reduction does.
    "logcumsumexp": {
        torch.bfloat16: {"rtol": 2.7e-2, "atol": 5.9e-3},
        torch.float16: {"rtol": 8e-3, "atol": 1.5e-3},
        torch.float32: {"rtol": 3e-4, "atol": 1e-4},
    },
    "logit": {
        torch.bfloat16: {"atol": 9.4e-3},
        torch.bool: {"atol": 7.5e-5},
        torch.float16: {"rtol": 1.2e-3, "atol": 1.2e-3},
        torch.float32: {"rtol": 1.1e-4, "atol": 1.3e-4},
        torch.int16: {"atol": 7.5e-5},
        torch.int32: {"atol": 7.5e-5},
        torch.int64: {"atol": 7.5e-5},
        torch.int8: {"atol": 7.5e-5},
        torch.uint8: {"atol": 7.5e-5},
    },
    "matmul": {
        torch.bfloat16: {"rtol": 3.9e-1, "atol": 2.9e-1},
        torch.complex64: {"rtol": 4e-1, "atol": 1.9},
        torch.float16: {"rtol": 1.3e-2, "atol": 8e-1},
        torch.float32: {"rtol": 4.5, "atol": 8.5e-1},
    },
    "mm": {
        torch.complex64: {"rtol": 1.4e-2, "atol": 1.5},
        torch.float16: {"rtol": 5e-2, "atol": 4.1e-1},
        torch.float32: {"rtol": 2.4e-2, "atol": 4.7e-1},
    },
    "native_batch_norm": {
        torch.bfloat16: {"rtol": 7.4e-2, "atol": 3.1e-2},
        torch.float16: {"rtol": 4.4e-3, "atol": 4e-3},
        torch.float32: {"rtol": 1e-4, "atol": 5e-4},
    },
    "native_layer_norm": {
        torch.bfloat16: {"rtol": 1.0, "atol": 2.5e-5},
        torch.float16: {"rtol": 6.9e-3, "atol": 3.0e-3},
    },
    "nn.functional.adaptive_avg_pool2d": {
        torch.bfloat16: {"rtol": 2, "atol": 3e-2},
        torch.float16: {"rtol": 1.9e-1, "atol": 4e-3},
    },
    "nn.functional.adaptive_avg_pool3d": {
        torch.bfloat16: {"rtol": 1, "atol": 3.2e-2},
        torch.float16: {"rtol": 2, "atol": 5.9e-3},
    },
    "nn.functional.avg_pool2d": {
        torch.bfloat16: {"rtol": 1.7, "atol": 7.9e-2},
        torch.float16: {"rtol": 3.4e-1, "atol": 2.4e-2},
    },
    "nn.functional.batch_norm": {
        torch.bfloat16: {"rtol": 7.4e-2, "atol": 2.6e-2},
        torch.float16: {"rtol": 4.4e-3, "atol": 2e-3},
    },
    "nn.functional.binary_cross_entropy": {
        torch.float32: {"rtol": 1.4e-4, "atol": 5.1e-5},
        torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
        torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    },
    "nn.functional.conv1d": {
        torch.bfloat16: {"rtol": 2.6e-2, "atol": 1.2e-2},
        torch.float16: {"rtol": 5e-2, "atol": 5.4e-1},
        torch.float32: {"rtol": 3.7e-2, "atol": 6.2e-1},
    },
    "nn.functional.conv2d": {
        torch.bfloat16: {"rtol": 1.8e-1, "atol": 3e-2},
        torch.float16: {"rtol": 1.1, "atol": 1},
        torch.float32: {"rtol": 1.1, "atol": 1.1},
    },
    "nn.functional.conv_transpose1d": {
        torch.bfloat16: {"rtol": 2.8e-2, "atol": 1.1e-1},
        torch.float16: {"rtol": 7.4e-2, "atol": 5e-1},
        torch.float32: {"rtol": 1.2e-1, "atol": 6.2e-1},
    },
    "nn.functional.conv_transpose2d": {
        torch.bfloat16: {"rtol": 1.8, "atol": 7.5e-1},
        torch.float16: {"rtol": 8, "atol": 7.5e-1},
        torch.float32: {"rtol": 6.3e-1, "atol": 6.9e-1},
    },
    "nn.functional.embedding_bag": {
        torch.bfloat16: {"rtol": 3.4e-1, "atol": 6.3e-2},
        torch.float16: {"rtol": 3.4e-1, "atol": 4e-3},
        torch.float32: {"rtol": 2.9e-6, "atol": 1.6e-5},
    },
    "nn.functional.gelu": {
        torch.float16: {"rtol": 6.8e-3, "atol": 4.2e-5},
        torch.float32: {"rtol": 6.7e-3, "atol": 4.1e-5},
        torch.float64: {"rtol": 8e-7, "atol": 1.3e-7},
    },
    "nn.functional.glu": {
        torch.float16: {"rtol": 2e-3, "atol": 1.5e-3},
        torch.float32: {"rtol": 6.5e-5, "atol": 1.7e-5},
    },
    "nn.functional.group_norm": {
        torch.bfloat16: {"rtol": 1.7e-1, "atol": 2.8e-2},
        torch.float16: {"rtol": 5.3e-2, "atol": 4e-3},
    },
    "nn.functional.hardsigmoid": {
        torch.bfloat16: {"rtol": 3.4e-2, "atol": 4e-3},
        torch.float16: {"rtol": 4.3e-3, "atol": 4.9e-4},
    },
    "nn.functional.hardswish": {
        torch.bfloat16: {"rtol": 3e-2, "atol": 3.2e-2},
    },
    "nn.functional.interpolate": {
        torch.bfloat16: {"rtol": 5e-2, "atol": 2e-2},
        torch.float16: {"rtol": 2e-2, "atol": 5e-3},
    },
    "nn.functional.logsigmoid": {
        torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
        torch.float16: {"rtol": 1e-3, "atol": 1e-3},
        torch.float32: {"rtol": 3e-4, "atol": 1e-4},
    },
    "nn.functional.mse_loss": {
        torch.bfloat16: {"rtol": 7e-3, "atol": 3e-1},
        torch.float16: {"rtol": 6.2e-4, "atol": 3.2e-2},
        torch.float32: {"rtol": 1.2e-6, "atol": 7.7e-5},
        torch.float64: {"rtol": 2.8e-9, "atol": 1.8e-7},
    },
    "nn.functional.nll_loss": {
        torch.float16: {"rtol": 1e-2, "atol": 1e-1},
    },
    "nn.functional.pdist": {
        torch.float32: {"rtol": 3.1e-6, "atol": 3.3e-5},
    },
    "nn.functional.prelu": {
        torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
        torch.float16: {"rtol": 2e-3, "atol": 1.5e-3},
        torch.float32: {"rtol": 6.5e-5, "atol": 1.7e-5},
    },
    "nn.functional.scaled_dot_product_attention": {
        # this op internally calls bmm, which runs on DEFAULT precision
        # as that is faster but requires higher tolerances.
        # changing it to HIGHEST precision allows for lower tolerances:
        # torch.float16: {"rtol": 1e-3, "atol": 1.2},
        # torch.float32: {"rtol": 1.3e-6, "atol": 1.2}
        # torch.bfloat16: {"rtol": 1.6e-2, "atol": 7.5e-1}
        torch.bfloat16: {"rtol": 5e-1, "atol": 8.4e-1},
        torch.float16: {"rtol": 5e-1, "atol": 9.7e-1},
        torch.float32: {"rtol": 5e-1, "atol": 9.5e-1},
        torch.float64: {"rtol": 1e-6, "atol": 1e-5},
    },
    "nn.functional.silu": {
        torch.bfloat16: {"rtol": 3.6e-2, "atol": 7.7e-5},
        torch.float16: {"rtol": 4.1e-2, "atol": 8.6e-5},
        torch.float32: {"rtol": 4.2e-2, "atol": 8.7e-5},
    },
    "nn.functional.softplus": {
        torch.bfloat16: {"rtol": 2e-2, "atol": 9.2e-5},
    },
    "norm": {
        torch.complex64: {"rtol": 6.8e-6},
        torch.float32: {"rtol": 6.9e-6},
    },
    "polygamma": {
        torch.bfloat16: {"rtol": 2.2e-1, "atol": 2.5e-4},
        torch.float16: {"rtol": 1.3e-1, "atol": 2.6e-4},
        torch.float32: {"rtol": 3.3e-5, "atol": 5.4e-4},
    },
    "pow": {
        torch.complex64: {"rtol": 6e-4, "atol": 1e-5},
        torch.float32: {"rtol": 5e-6, "atol": 1e-5},
    },
    "prod": {
        torch.float16: {"rtol": 1.8e-3, "atol": 1.3e-4},
    },
    "remainder": {
        torch.float16: {"rtol": 1, "atol": 0.4},
    },
    "sigmoid": {
        torch.float32: {"rtol": 1.7e-5, "atol": 1.6e-5},
        torch.int16: {"rtol": 5.9e-3, "atol": 1.5e-5},
        torch.int32: {"rtol": 5.9e-3, "atol": 1.5e-5},
        torch.int64: {"rtol": 5.9e-3, "atol": 1.5e-5},
        torch.int8: {"rtol": 5.9e-3, "atol": 1.5e-5},
        torch.uint8: {"rtol": 1.5e-5, "atol": 1.5e-5},
    },
    "sin": {
        torch.complex64: {"rtol": 3.9e-6, "atol": 1.2e-2},
    },
    "sinh": {
        torch.complex64: {"rtol": 4.2e-6, "atol": 1.1e-2},
        torch.float32: {"rtol": 3.8e-6, "atol": 1.1e-2},
        torch.int16: {"rtol": 3.3e-6, "atol": 1.1e-3},
        torch.int32: {"rtol": 3.3e-6, "atol": 1.1e-3},
        torch.int64: {"rtol": 3.3e-6, "atol": 1.1e-3},
        torch.int8: {"rtol": 3.3e-6, "atol": 1.1e-3},
        torch.uint8: {"rtol": 3.3e-6, "atol": 1.1e-3},
    },
    "softmax": {
        torch.bfloat16: {"rtol": 2e-2, "atol": 2.5e-4},
        torch.float16: {"rtol": 2.2e-3, "atol": 9.8e-4},
    },
    "sqrt": {
        torch.float64: {"rtol": 2e-6, "atol": 3e-6},
    },
    "sum": {
        torch.float16: {"rtol": 3.5e-3, "atol": 1.2e-2},
    },
    "tan": {
        torch.complex64: {"rtol": 6.1e-6, "atol": 3.1e-5},
    },
    "tanh": {
        torch.complex64: {"rtol": 8.3e-6, "atol": 1.7e-5},
        torch.float32: {"rtol": 5.8e-5, "atol": 3.9e-5},
        torch.int16: {"rtol": 3e-5, "atol": 3e-5},
        torch.int32: {"rtol": 3e-5, "atol": 3e-5},
        torch.int64: {"rtol": 3e-5, "atol": 3e-5},
        torch.int8: {"rtol": 3e-5, "atol": 3e-5},
        torch.uint8: {"rtol": 3e-5, "atol": 3e-5},
    },
    "var": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"rtol": 2.6e-3, "atol": 5.9e-4},
    },
    "xlogy": {
        torch.float32: {"rtol": 2.6e-4, "atol": 7.5e-4},
        torch.int16: {"rtol": 5.7e-5, "atol": 5.7e-4},
        torch.int32: {"rtol": 5.7e-5, "atol": 5.7e-4},
        torch.int64: {"rtol": 5.7e-5, "atol": 5.7e-4},
        torch.int8: {"rtol": 5.7e-5, "atol": 5.7e-4},
        torch.uint8: {"rtol": 5.7e-5, "atol": 5.7e-4},
    },
    # go/keep-sorted end
}  # end of ACCURACY_OVERRIDES_VS_CPU


def update_dict(d, u):
  """Recursively update a dictionary to override tolerance values."""
  for k, v in u.items():
    if isinstance(v, collections.abc.Mapping):
      d[k] = update_dict(d.get(k, {}), v)
    else:
      d[k] = v
  return d


# Like ACCURACY_OVERRIDES_VS_CPU, but for TPU vs GPU instead.
ACCURACY_OVERRIDES_VS_GPU = {
    # go/keep-sorted start
    "_foreach_acos": {
        torch.complex64: {"rtol": 4.2e-5, "atol": 7.1e-5},
    },
    "_foreach_add": {
        torch.bfloat16: {"rtol": 3.4e-2, "atol": 3.3e-2},
        torch.float16: {"rtol": 6.4e-3, "atol": 6.5e-3},
    },
    "_foreach_addcdiv": {
        torch.bfloat16: {"rtol": 3.7e-2, "atol": 5.2e-2},
        torch.float16: {"rtol": 4.8e-3, "atol": 9.1e-3},
    },
    "_foreach_addcmul": {
        torch.bfloat16: {"rtol": 5.1e-2, "atol": 7.6e-2},
        torch.float16: {"rtol": 5.5e-3, "atol": 7.7e-3},
    },
    "_foreach_asin": {
        torch.complex64: {"rtol": 6e-5, "atol": 6.1e-5},
    },
    "_foreach_atan": {
        torch.complex64: {"rtol": 1.9e-5, "atol": 1.5e-5},
    },
    "_foreach_cos": {
        torch.complex64: {"rtol": 3.6e-6, "atol": 1e-2},
    },
    "_foreach_cosh": {
        torch.complex64: {"rtol": 5.1e-6},
        torch.float32: {"rtol": 5e-6},
        torch.int16: {"rtol": 4e-6},
        torch.int32: {"rtol": 4e-6},
        torch.int64: {"rtol": 4e-6},
        torch.int8: {"rtol": 4e-6},
        torch.uint8: {"rtol": 4e-6},
    },
    "_foreach_exp": {
        torch.complex64: {"rtol": 4.8e-6},
        torch.float32: {"rtol": 4.9e-6},
        torch.int16: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.uint8: {"rtol": 4.4e-6},
    },
    "_foreach_expm1": {
        torch.complex64: {"rtol": 5.9e-6},
        torch.float32: {"rtol": 6.3e-6},
        torch.int16: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.uint8: {"rtol": 4.4e-6},
    },
    "_foreach_frac": {
        torch.bfloat16: {"atol": 1.3},
        torch.float16: {"atol": 1.3},
        torch.float32: {"atol": 1.3},
        torch.float64: {"atol": 1.3},
    },
    "_foreach_lerp": {
        torch.bfloat16: {"atol": 9.4e-2},
        torch.float16: {"atol": 1.3e-1},
    },
    "_foreach_log": {
        torch.complex64: {"rtol": 6.3e-5, "atol": 6.8e-5},
        torch.float32: {"rtol": 7.4e-5, "atol": 8.3e-5},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "_foreach_log10": {
        torch.complex64: {"rtol": 3.9e-5, "atol": 5.3e-5},
        torch.float32: {"rtol": 3.3e-5, "atol": 4.2e-5},
        torch.int16: {"atol": 3.3e-5},
        torch.int32: {"atol": 3.3e-5},
        torch.int64: {"atol": 3.3e-5},
        torch.int8: {"atol": 3.3e-5},
        torch.uint8: {"atol": 3.3e-5},
    },
    "_foreach_log1p": {
        torch.complex64: {"rtol": 3.6e-5, "atol": 3.7e-5},
        torch.float32: {"rtol": 7.6e-5, "atol": 8.3e-5},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "_foreach_log2": {
        torch.complex64: {"rtol": 1.1e-4, "atol": 3.4e-5},
        torch.float32: {"rtol": 1.1e-4, "atol": 8.1e-5},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "_foreach_pow": {
        torch.float32: {"rtol": 5.5e-6},
        torch.int16: {"atol": 1.3},
        torch.int32: {"atol": 1.3},
        torch.int8: {"atol": 1.3},
    },
    "_foreach_sigmoid": {
        torch.complex64: {"rtol": 1.2e-5, "atol": 6e-5},
        torch.float16: {"atol": 1.6e-5},
        torch.float32: {"rtol": 6.2e-2, "atol": 1.7e-5},
        torch.int16: {"atol": 1.5e-5},
        torch.int32: {"atol": 1.5e-5},
        torch.int64: {"atol": 1.5e-5},
        torch.int8: {"atol": 1.5e-5},
        torch.uint8: {"atol": 1.5e-5},
    },
    "_foreach_sin": {
        torch.complex64: {"rtol": 3.7e-6, "atol": 9.2e-3},
    },
    "_foreach_sinh": {
        torch.complex64: {"atol": 1.1e-2},
        torch.float32: {"atol": 9.3e-3},
        torch.int16: {"atol": 1.2e-3},
        torch.int32: {"atol": 1.2e-3},
        torch.int64: {"atol": 1.2e-3},
        torch.int8: {"atol": 1.2e-3},
        torch.uint8: {"atol": 1.2e-3},
    },
    "_foreach_sub": {
        torch.bfloat16: {"atol": 2.8e-2},
        torch.float16: {"atol": 1.6e-2},
    },
    "_foreach_tan": {
        torch.complex64: {"atol": 7e-4},
    },
    "_foreach_tanh": {
        torch.complex64: {"atol": 1.4e-4},
        torch.float32: {"rtol": 6.2e-5, "atol": 3.4e-5},
        torch.int16: {"atol": 3e-5},
        torch.int32: {"atol": 3e-5},
        torch.int64: {"atol": 3e-5},
        torch.int8: {"atol": 3e-5},
        torch.uint8: {"atol": 3e-5},
    },
    "_log_softmax_backward_data": {
        torch.bfloat16: {"rtol": 2.2e-2, "atol": 5.4e-2},
        torch.float16: {"rtol": 2.7e-3, "atol": 8.5e-3},
        torch.float32: {"rtol": 3.5e-5, "atol": 3.6e-5},
    },
    "_native_batch_norm_legit": {
        torch.float32: {"rtol": 1e-5, "atol": 2.1e-5},
    },
    "_softmax_backward_data": {
        torch.bfloat16: {"rtol": 1.9e-2, "atol": 4.6e-2},
        torch.float16: {"rtol": 1.5e-3, "atol": 4.7e-3},
    },
    "_thnn_fused_gru_cell": {
        torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    },
    "_thnn_fused_lstm_cell": {
        torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    },
    "acos": {
        torch.complex64: {"rtol": 5.5e-5, "atol": 2.2e-5},
    },
    "acosh": {
        torch.complex64: {"rtol": 2.9e-5},
        torch.float32: {"rtol": 3e-5, "atol": 1.3e-5},
        torch.int16: {"rtol": 2.8e-5},
        torch.int32: {"rtol": 2.8e-5},
        torch.int64: {"rtol": 2.8e-5},
        torch.int8: {"rtol": 2.8e-5},
        torch.uint8: {"rtol": 2.8e-5},
    },
    "add": {
        torch.bfloat16: {"rtol": 1.8e-2, "atol": 1.5e-2},
        torch.float16: {"rtol": 4.5e-3, "atol": 1.2e-3},
    },
    "addcdiv": {
        torch.bfloat16: {"rtol": 2.5e-2, "atol": 2.9e-2},
        torch.float16: {"rtol": 4.9e-3, "atol": 6.5e-3},
    },
    "addcmul": {
        torch.bfloat16: {"atol": 1.5e-2},
        torch.float16: {"rtol": 2.1e-3, "atol": 3.1e-3},
    },
    "addmm": {
        torch.float16: {"rtol": 1.5e-2},
        torch.float32: {"rtol": 7.7e-3},
    },
    "arange": {
        torch.bfloat16: {"atol": 9.4e-3},
        torch.float16: {"atol": 5.9e-4},
    },
    "asin": {
        torch.complex64: {"rtol": 7.9e-5, "atol": 1e-4},
    },
    "asinh": {
        torch.bool: {"atol": 2.9e-5},
        torch.complex64: {"rtol": 4.4e-5, "atol": 7.1e-5},
        torch.float32: {"rtol": 7.1e-5, "atol": 8.3e-5},
        torch.int16: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int32: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int64: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int8: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.uint8: {"rtol": 1.6e-5, "atol": 2.9e-5},
    },
    "atan": {
        torch.complex64: {"rtol": 2.1e-5, "atol": 1.8e-5},
    },
    "atanh": {
        torch.complex64: {"rtol": 1.8e-5, "atol": 3.2e-5},
        torch.float32: {"rtol": 5.8e-5, "atol": 7.7e-5},
    },
    "baddbmm": {
        torch.bfloat16: {"rtol": 1.7e-2, "atol": 1.5e-2},
        torch.complex64: {"rtol": 8.3e-2},
        torch.float16: {"rtol": 2.8e-1, "atol": 3.3e-1},
        torch.float32: {"rtol": 3.4e-1, "atol": 3.6e-1},
    },
    "bmm": {
        torch.complex64: {"rtol": 9.6e-2},
        torch.float16: {"rtol": 1.1e-1, "atol": 3.5e-1},
        torch.float32: {"rtol": 6.6e-2, "atol": 3.4e-1},
    },
    "cdist": {
        torch.bfloat16: {"atol": 2.1e-1},
        torch.float16: {"rtol": 2e-1, "atol": 3e-1},
        torch.float32: {"rtol": 5.1e-1, "atol": 3e-1},
    },
    "cos": {
        torch.complex64: {"rtol": 4.5e-6},
    },
    "cosh": {
        torch.complex64: {"rtol": 3.9e-6},
        torch.float32: {"rtol": 3.4e-6},
        torch.int16: {"rtol": 4e-6},
        torch.int32: {"rtol": 4e-6},
        torch.int64: {"rtol": 4e-6},
        torch.int8: {"rtol": 4e-6},
        torch.uint8: {"rtol": 4e-6},
    },
    # bf16/f16 cumsum accumulation rounds differently from the GPU
    # reference, which is itself not bit-exact (mirrors the vs-CPU tol).
    "cumsum": {
        torch.bfloat16: {"rtol": 2.3e-2, "atol": 3.8e-2},
        torch.float16: {"rtol": 3.7e-3, "atol": 4.7e-3},
    },
    "digamma": {
        torch.float32: {"rtol": 4e-5, "atol": 1.3e-5},
        torch.int16: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int32: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int64: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int8: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.uint8: {"rtol": 4.7e-5, "atol": 3.6e-5},
    },
    "erfinv": {
        torch.float32: {"rtol": 1.6e-5, "atol": 1.7e-5},
    },
    "exp": {
        torch.complex64: {"rtol": 2.8e-6},
        torch.float32: {"rtol": 4.6e-6},
        torch.int16: {"rtol": 3.1e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 3.1e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 3.1e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 3.1e-6, "atol": 7.9e-7},
        torch.uint8: {"rtol": 4.4e-6},
    },
    "exp2": {
        torch.bfloat16: {"rtol": 2.4e-2, "atol": 2.4e-3},
        torch.complex64: {"rtol": 3.3e-6},
        torch.float16: {"rtol": 2.3e-3, "atol": 5.9e-4},
        torch.float32: {"rtol": 2.4e-6},
    },
    "expm1": {
        torch.complex64: {"rtol": 4.7e-6},
        torch.float32: {"rtol": 6.9e-6},
        torch.int16: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.uint8: {"rtol": 4.4e-6},
    },
    "ldexp": {
        torch.complex64: {"rtol": 4.4e-6},
        torch.float32: {"rtol": 4.1e-6},
    },
    "lgamma": {
        torch.bfloat16: {"rtol": 2.6e-2, "atol": 1.6e-4},
        torch.float16: {"rtol": 1.3e-2, "atol": 1.5e-4},
        torch.float32: {"rtol": 1.3e-2, "atol": 2.6e-4},
        torch.int16: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int32: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int64: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int8: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.uint8: {"rtol": 1.3e-4, "atol": 2.3e-4},
    },
    "linalg.lu": {
        torch.complex64: {"rtol": 5, "atol": 10},
        torch.float64: {"atol": 2.5e-6},
    },
    "linalg.lu_factor_ex": {
        torch.complex64: {"rtol": 5.4, "atol": 19},
        torch.float32: {"rtol": 2.5e-6, "atol": 2.4e-5},
    },
    "linalg.vector_norm": {
        torch.complex64: {"rtol": 4.8e-6},
        torch.float32: {"rtol": 4.8e-6},
    },
    "linspace": {
        torch.int16: {"rtol": 4.1e-1, "atol": 1.3},
        torch.int32: {"rtol": 4.1e-1, "atol": 1.3},
        torch.int64: {"rtol": 4.1e-1, "atol": 1.3},
        torch.int8: {"rtol": 4.1e-1, "atol": 1.3},
        torch.uint8: {"rtol": 4.1e-1},
    },
    "log": {
        torch.complex64: {"rtol": 4.7e-5},
        torch.float32: {"rtol": 4.9e-5, "atol": 4.3e-5},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "log10": {
        torch.complex64: {"rtol": 1.4e-5, "atol": 2.2e-5},
        torch.float32: {"atol": 3.3e-5},
        torch.int16: {"atol": 3.3e-5},
        torch.int32: {"atol": 3.3e-5},
        torch.int64: {"atol": 3.3e-5},
        torch.int8: {"atol": 3.3e-5},
        torch.uint8: {"atol": 3.3e-5},
    },
    "log1p": {
        torch.complex64: {"rtol": 3.3e-5, "atol": 3.7e-5},
        torch.float32: {"rtol": 1.1e-4, "atol": 1.2e-4},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "log2": {
        torch.complex64: {"rtol": 4.5e-5},
        torch.float32: {"rtol": 5.6e-5, "atol": 2.8e-5},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "log_softmax": {
        torch.bfloat16: {"atol": 6.5e-3},
        torch.float16: {"rtol": 1.2e-3, "atol": 6.3e-4},
        torch.float32: {"rtol": 5.7e-5, "atol": 8.3e-5},
    },
    "logcumsumexp": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"rtol": 1.8e-3, "atol": 1.4e-3},
        torch.float32: {"rtol": 5e-5, "atol": 3.2e-5},
    },
    "logit": {
        torch.bool: {"atol": 7.5e-5},
        torch.float16: {"atol": 5.9e-4},
        torch.float32: {"rtol": 7.5e-5, "atol": 8.3e-5},
        torch.int16: {"atol": 7.5e-5},
        torch.int32: {"atol": 7.5e-5},
        torch.int64: {"atol": 7.5e-5},
        torch.int8: {"atol": 7.5e-5},
        torch.uint8: {"atol": 7.5e-5},
    },
    "matmul": {
        torch.bfloat16: {"rtol": 1.2e-1, "atol": 1.9e-1},
        torch.complex64: {"rtol": 1.4e-1},
        torch.float16: {"rtol": 3e-1, "atol": 4.7e-1},
        torch.float32: {"rtol": 2.7e-1, "atol": 3.9e-1},
    },
    "mm": {
        torch.complex64: {"rtol": 3.8e-2},
        torch.float16: {"rtol": 4.8e-2},
        torch.float32: {"rtol": 5.2e-2},
    },
    "nn.functional.adaptive_avg_pool2d": {
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 1.8e-3, "atol": 3.6e-3},
    },
    "nn.functional.adaptive_avg_pool3d": {
        torch.bfloat16: {"rtol": 1.9e-2, "atol": 1.9e-2},
        torch.float16: {"rtol": 3.1e-3, "atol": 3.1e-3},
    },
    "nn.functional.avg_pool2d": {
        torch.bfloat16: {"rtol": 3.8e-2, "atol": 4.5e-2},
        torch.float16: {"rtol": 3.5e-3, "atol": 7.1e-3},
    },
    "nn.functional.binary_cross_entropy": {
        torch.bfloat16: {"rtol": 9e-3, "atol": 5.9e-3},
        torch.float16: {"rtol": 8.5e-3, "atol": 4.4e-4},
        torch.float32: {"rtol": 1.8e-4, "atol": 6.9e-5},
    },
    "nn.functional.conv1d": {
        torch.float16: {"rtol": 8.2e-2},
        torch.float32: {"rtol": 9e-2},
    },
    "nn.functional.conv2d": {
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 3.9e-1, "atol": 4.2e-1},
        torch.float32: {"rtol": 4.2e-1, "atol": 4.6e-1},
    },
    "nn.functional.conv_transpose1d": {
        torch.float16: {"rtol": 6.7e-2, "atol": 1.6e-1},
        torch.float32: {"rtol": 7.1e-2, "atol": 2.1e-1},
    },
    "nn.functional.conv_transpose2d": {
        torch.bfloat16: {"rtol": 7.4e-2, "atol": 7.6e-2},
        torch.float16: {"rtol": 2.2e-1, "atol": 6.6e-2},
        torch.float32: {"rtol": 3e-1, "atol": 9.4e-2},
    },
    "nn.functional.embedding_bag": {
        torch.bfloat16: {"atol": 3.2e-2},
        torch.float16: {"atol": 4e-3},
    },
    "nn.functional.gelu": {
        torch.float16: {"atol": 7.4e-5},
        torch.float32: {"rtol": 2.1e-5, "atol": 4.3e-5},
    },
    "nn.functional.glu": {
        torch.float16: {"rtol": 1.2e-3, "atol": 1.2e-3},
    },
    "nn.functional.group_norm": {
        torch.bfloat16: {"rtol": 3.3e-2, "atol": 8e-2},
    },
    "nn.functional.hardsigmoid": {
        torch.bfloat16: {"atol": 4.7e-3},
    },
    "nn.functional.logsigmoid": {
        torch.float32: {"rtol": 3.2e-5},
    },
    "nn.functional.mse_loss": {
        torch.float32: {"rtol": 2.3e-6},
    },
    "nn.functional.nll_loss": {
        torch.float16: {"rtol": 3e-3, "atol": 2.4e-3},
    },
    "nn.functional.silu": {
        torch.bfloat16: {"atol": 7.4e-5},
        torch.float16: {"rtol": 1.2e-3, "atol": 1.5e-4},
        torch.float32: {"rtol": 1.3e-5, "atol": 7.2e-5},
    },
    "norm": {
        torch.complex64: {"rtol": 5.7e-6},
        torch.float32: {"rtol": 5.7e-6},
    },
    "polygamma": {
        torch.float32: {"rtol": 1.1e-5},
    },
    "pow": {
        torch.complex64: {"rtol": 7e-4, "atol": 3.8e-4},
        torch.float32: {"rtol": 5.6e-6},
    },
    "remainder": {
        torch.float16: {"rtol": 2.4e-3, "atol": 4.4e-3},
    },
    "sigmoid": {
        torch.float16: {"atol": 1.5e-5},
        torch.float32: {"atol": 1.5e-5},
        torch.int16: {"atol": 1.8e-5},
        torch.int32: {"atol": 1.8e-5},
        torch.int64: {"atol": 1.8e-5},
        torch.int8: {"atol": 1.8e-5},
        torch.uint8: {"atol": 1.8e-5},
    },
    "sin": {
        torch.complex64: {"rtol": 4.9e-6},
    },
    "sinh": {
        torch.complex64: {"rtol": 4.6e-6},
        torch.float32: {"rtol": 5.8e-6},
        torch.int16: {"rtol": 4e-6},
        torch.int32: {"rtol": 4e-6},
        torch.int64: {"rtol": 4e-6},
        torch.int8: {"rtol": 4e-6},
        torch.uint8: {"rtol": 4e-6},
    },
    "softmax": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"atol": 5.9e-4},
    },
    "tan": {
        torch.complex64: {"rtol": 7.1e-6},
    },
    "tanh": {
        torch.complex64: {"rtol": 1.1e-5, "atol": 1.1e-5},
        torch.float32: {"atol": 3.8e-5},
        torch.int16: {"atol": 3.6e-5},
        torch.int32: {"atol": 3.6e-5},
        torch.int64: {"atol": 3.6e-5},
        torch.int8: {"atol": 3.6e-5},
        torch.uint8: {"atol": 3.6e-5},
    },
    "var": {
        torch.float16: {"rtol": 1.5e-3},
    },
    "xlogy": {
        torch.float32: {"rtol": 3.2e-4, "atol": 3e-4},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    # go/keep-sorted end
}  # end of ACCURACY_OVERRIDES_VS_GPU

# Like ACCURACY_OVERRIDES_VS_CPU, but for TPU vs GPU with torch.compile().
ACCURACY_OVERRIDES_VS_GPU_COMPILED = {
    # go/keep-sorted start
    "_foreach_acos": {
        torch.complex64: {"rtol": 4.2e-5, "atol": 7e-5},
    },
    "_foreach_asin": {
        torch.complex64: {"rtol": 6e-5, "atol": 6.1e-5},
    },
    "_foreach_atan": {
        torch.complex64: {"rtol": 1.9e-5, "atol": 1.5e-5},
    },
    "_foreach_cos": {
        torch.complex64: {"atol": 9.6e-3},
    },
    "_foreach_cosh": {
        torch.complex64: {"rtol": 5.1e-6},
    },
    "_foreach_exp": {
        torch.complex64: {"rtol": 4.8e-6},
    },
    "_foreach_expm1": {
        torch.complex64: {"rtol": 5.9e-6},
    },
    "_foreach_log": {
        torch.complex64: {"rtol": 6.3e-5, "atol": 6.6e-5},
    },
    "_foreach_log10": {
        torch.complex64: {"rtol": 3.9e-5, "atol": 4.5e-5},
    },
    "_foreach_log1p": {
        torch.complex64: {"rtol": 3.6e-5, "atol": 3.7e-5},
    },
    "_foreach_log2": {
        torch.complex64: {"rtol": 1.1e-4},
    },
    "_foreach_sigmoid": {
        torch.complex64: {"atol": 6e-5},
    },
    "_foreach_sin": {
        torch.complex64: {"atol": 9.2e-3},
    },
    "_foreach_sinh": {
        torch.complex64: {"rtol": 3.8e-6, "atol": 8.3e-3},
    },
    "_foreach_tan": {
        torch.complex64: {"atol": 6.3e-5},
    },
    "_foreach_tanh": {
        torch.complex64: {"atol": 1.4e-4},
    },
    "_log_softmax_backward_data": {
        torch.float16: {"rtol": 1.8e-3, "atol": 2.4e-3},
        torch.float32: {"rtol": 3.5e-5, "atol": 3.6e-5},
    },
    "_native_batch_norm_legit": {
        torch.float32: {"rtol": 1e-5, "atol": 2.1e-5},
    },
    "_softmax_backward_data": {
        torch.bfloat16: {"atol": 3.5e-2},
        torch.float16: {"atol": 4.6e-3},
    },
    "_thnn_fused_gru_cell": {
        torch.float32: {"rtol": 1e-5, "atol": 3e-5},
    },
    "acos": {
        torch.complex64: {"rtol": 5.5e-5, "atol": 2.2e-5},
    },
    "acosh": {
        torch.complex64: {"rtol": 2.9e-5},
    },
    "addcmul": {
        torch.float16: {"rtol": 1.4e-3, "atol": 1.2e-3},
    },
    "asin": {
        torch.complex64: {"rtol": 7.9e-5, "atol": 1e-4},
    },
    "asinh": {
        torch.bool: {"atol": 2.9e-5},
        torch.complex64: {"rtol": 4.4e-5, "atol": 7.1e-5},
        torch.float32: {"rtol": 7.1e-5, "atol": 8.3e-5},
        torch.int16: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int32: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int64: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int8: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.uint8: {"rtol": 1.6e-5, "atol": 2.9e-5},
    },
    "atan": {
        torch.complex64: {"rtol": 2.1e-5, "atol": 1.8e-5},
    },
    "atanh": {
        torch.complex64: {"rtol": 1.8e-5, "atol": 3.2e-5},
        torch.float32: {"rtol": 5.8e-5, "atol": 7.7e-5},
    },
    "baddbmm": {
        torch.complex64: {"rtol": 6.9e-2},
    },
    "bmm": {
        torch.complex64: {"rtol": 9.6e-2},
        torch.float16: {"rtol": 1.1e-1, "atol": 3.5e-1},
        torch.float32: {"rtol": 6.6e-2, "atol": 3.4e-1},
    },
    "cos": {
        torch.complex64: {"rtol": 4.5e-6},
    },
    "cosh": {
        torch.complex64: {"rtol": 3.9e-6},
    },
    # bf16/f16 cumsum accumulation rounds differently from the GPU
    # reference, which is itself not bit-exact (mirrors the vs-CPU tol).
    "cumsum": {
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 3.5e-3, "atol": 2.4e-3},
    },
    "erfinv": {
        torch.float32: {"rtol": 1.6e-5, "atol": 1.7e-5},
    },
    "exp": {
        torch.complex64: {"rtol": 2.8e-6},
    },
    "exp2": {
        torch.complex64: {"rtol": 3.3e-6},
    },
    "expm1": {
        torch.complex64: {"rtol": 4.7e-6},
        torch.float32: {"rtol": 6.9e-6},
        torch.int16: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.uint8: {"rtol": 4.4e-6},
    },
    "ldexp": {
        torch.complex64: {"rtol": 4.4e-6},
    },
    "linalg.lu_factor_ex": {
        torch.complex64: {"rtol": 5.4, "atol": 19},
    },
    "linalg.vector_norm": {
        torch.complex64: {"rtol": 4.8e-6},
        torch.float32: {"rtol": 4.8e-6},
    },
    "linspace": {
        torch.bfloat16: {"atol": 9.4e-3},
        torch.float16: {"rtol": 1.2e-3, "atol": 1.2e-3},
        torch.int16: {"rtol": 4.1e-1, "atol": 1.3},
        torch.int32: {"rtol": 4.1e-1, "atol": 1.3},
        torch.int64: {"rtol": 4.1e-1, "atol": 1.3},
        torch.int8: {"rtol": 4.1e-1, "atol": 1.3},
        torch.uint8: {"rtol": 4.1e-1},
    },
    "log": {
        torch.complex64: {"rtol": 4.7e-5},
    },
    "log10": {
        torch.complex64: {"rtol": 1.4e-5, "atol": 2.2e-5},
    },
    "log1p": {
        torch.complex64: {"rtol": 3.3e-5, "atol": 3.7e-5},
        torch.float32: {"rtol": 1.1e-4, "atol": 1.2e-4},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "log2": {
        torch.complex64: {"rtol": 4.5e-5},
    },
    "logcumsumexp": {
        torch.bfloat16: {"atol": 3.7e-5},
        torch.float16: {"rtol": 1.2e-3, "atol": 5.9e-4},
        torch.float32: {"rtol": 5e-5, "atol": 3.2e-5},
    },
    "logit": {
        torch.bool: {"atol": 7.5e-5},
        torch.float16: {"atol": 5.9e-4},
        torch.float32: {"rtol": 7.5e-5, "atol": 8.3e-5},
        torch.int16: {"atol": 7.5e-5},
        torch.int32: {"atol": 7.5e-5},
        torch.int64: {"atol": 7.5e-5},
        torch.int8: {"atol": 7.5e-5},
        torch.uint8: {"atol": 7.5e-5},
    },
    "matmul": {
        torch.complex64: {"rtol": 5.2e-2},
    },
    "mm": {
        torch.complex64: {"rtol": 3.8e-2},
    },
    "nn.functional.adaptive_avg_pool2d": {
        torch.bfloat16: {"rtol": 1.8e-2, "atol": 1.9e-2},
        torch.float16: {"rtol": 1.8e-3, "atol": 1.8e-3},
    },
    "nn.functional.binary_cross_entropy": {
        torch.bfloat16: {"rtol": 9e-3, "atol": 5.9e-3},
        torch.float16: {"rtol": 8.5e-3, "atol": 4.4e-4},
        torch.float32: {"rtol": 1.8e-4, "atol": 6.9e-5},
    },
    "nn.functional.conv1d": {
        torch.float16: {"rtol": 8.2e-2},
        torch.float32: {"rtol": 9e-2},
    },
    "nn.functional.conv2d": {
        torch.bfloat16: {"atol": 1.5e-2},
        torch.float16: {"rtol": 3.9e-1, "atol": 4.2e-1},
        torch.float32: {"rtol": 4.2e-1, "atol": 4.6e-1},
    },
    "nn.functional.conv_transpose1d": {
        torch.bfloat16: {"atol": 8.3e-3},
        torch.float16: {"rtol": 6.7e-2, "atol": 1.6e-1},
        torch.float32: {"rtol": 7.1e-2, "atol": 2.1e-1},
    },
    "nn.functional.conv_transpose2d": {
        torch.bfloat16: {"rtol": 6.2e-2, "atol": 8e-2},
        torch.float16: {"rtol": 2.2e-1, "atol": 6.7e-2},
        torch.float32: {"rtol": 3e-1, "atol": 9.4e-2},
    },
    "nn.functional.embedding_bag": {
        torch.bfloat16: {"atol": 6.5e-2},
        torch.float16: {"atol": 2.5e-1},
    },
    "nn.functional.mse_loss": {
        torch.float32: {"rtol": 2.3e-6},
    },
    "nn.functional.nll_loss": {
        torch.bfloat16: {"rtol": 2.3e-2, "atol": 5.9e-3},
        torch.float16: {"rtol": 3e-3, "atol": 2.4e-3},
    },
    "norm": {
        torch.complex64: {"rtol": 5.7e-6},
        torch.float32: {"rtol": 5.7e-6},
    },
    "polygamma": {
        torch.float32: {"rtol": 1.1e-5},
    },
    "pow": {
        torch.complex64: {"rtol": 7e-4, "atol": 3.8e-4},
    },
    "sin": {
        torch.complex64: {"rtol": 4.9e-6},
    },
    "sinh": {
        torch.complex64: {"rtol": 4.6e-6},
        torch.float32: {"rtol": 5.8e-6},
        torch.int16: {"rtol": 4e-6},
        torch.int32: {"rtol": 4e-6},
        torch.int64: {"rtol": 4e-6},
        torch.int8: {"rtol": 4e-6},
        torch.uint8: {"rtol": 4e-6},
    },
    "tan": {
        torch.complex64: {"rtol": 7.1e-6},
    },
    "tanh": {
        torch.complex64: {"rtol": 1.1e-5, "atol": 1.1e-5},
        torch.float32: {"atol": 3.8e-5},
        torch.int16: {"atol": 3.6e-5},
        torch.int32: {"atol": 3.6e-5},
        torch.int64: {"atol": 3.6e-5},
        torch.int8: {"atol": 3.6e-5},
        torch.uint8: {"atol": 3.6e-5},
    },
    # go/keep-sorted end
}  # end of ACCURACY_OVERRIDES_VS_GPU_COMPILED

# The gradient tolerances are based on the forward pass tolerances.
ACCURACY_OVERRIDES_GRAD: dict[str, dict[torch.dtype, dict[str, float]]] = (
    update_dict(
        copy.deepcopy(ACCURACY_OVERRIDES_VS_CPU),
        {
            # go/keep-sorted start
            "_foreach_erfc": {
                torch.bfloat16: {"rtol": 3.5e-2, "atol": 4.3e-4},
            },
            "_foreach_log10": {
                torch.float16: {"rtol": 1.3e-3, "atol": 1.3e-4},
            },
            "_foreach_norm": {
                torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
                torch.float16: {"rtol": 2e-3, "atol": 2e-3},
            },
            "_foreach_reciprocal": {
                torch.float32: {"rtol": 1e-6},
            },
            "_foreach_rsqrt": {
                torch.bfloat16: {"rtol": 2.4e-2, "atol": 1e-3},
                torch.float32: {"rtol": 2.1e-6, "atol": 1.4e-4},
            },
            "_foreach_sigmoid": {
                torch.float16: {"atol": 1e-3},
            },
            "_foreach_tanh": {
                torch.bfloat16: {"rtol": 2.4e-2, "atol": 2e-3},
                torch.float16: {"rtol": 5e-1, "atol": 9.8e-4},
                torch.float32: {"rtol": 4.9e-1, "atol": 6.1e-5},
            },
            "_thnn_fused_gru_cell": {
                torch.bfloat16: {"rtol": 3e-1, "atol": 5e-3},
                torch.float16: {"rtol": 8e-2, "atol": 3e-4},
                torch.float32: {"rtol": 1.5e-2, "atol": 2e-4},
            },
            "_thnn_fused_lstm_cell": {
                torch.bfloat16: {"rtol": 6e-2, "atol": 8e-3},
                torch.float16: {"rtol": 6e-2, "atol": 1e-3},
                torch.float32: {"rtol": 5e-2, "atol": 2e-4},
            },
            "acos": {
                torch.float16: {"rtol": 3e-3, "atol": 2e-2},
            },
            "asin": {
                torch.float16: {"rtol": 3e-3, "atol": 2e-2},
            },
            "atan": {
                torch.float16: {"rtol": 2e-3, "atol": 1e-4},
            },
            "atan2": {
                torch.bfloat16: {"rtol": 5.2e-2, "atol": 4e-2},
                torch.float16: {"rtol": 1.3e-2, "atol": 2e-3},
            },
            "atanh": {
                torch.bfloat16: {"rtol": 2e-2, "atol": 3e-1},
                torch.float16: {"rtol": 1e-3, "atol": 3e-1},
                torch.float32: {"rtol": 4e-4, "atol": 8e-2},
            },
            "cdist": {
                torch.float16: {"rtol": 3.2e-2, "atol": 7.9e-3},
                torch.float32: {"rtol": 3.7e-2, "atol": 8.5e-3},
            },
            "erf": {
                torch.bfloat16: {"rtol": 2e-2, "atol": 3e-4},
                torch.float16: {"rtol": 2e-3, "atol": 1e-4},
            },
            "erfinv": {
                # TODO(b/488121035)
                torch.float32: {"rtol": 5.3e-5, "atol": 1.3e-2},
                torch.bfloat16: {"rtol": 2e-2, "atol": 5e-1},
                torch.float16: {"rtol": 5.9e-3, "atol": 4e-2},
            },
            "lerp": {
                torch.bfloat16: {"rtol": 4e-2, "atol": 8e-3},
                torch.float16: {"rtol": 1.6e-3, "atol": 3.2e-2},
            },
            "linalg.inv": {
                torch.float32: {"rtol": 1.2e-2, "atol": 7e-3},
                torch.float64: {"rtol": 4.9e-6, "atol": 2.5e-7},
            },
            "linalg.lu": {
                torch.float32: {"rtol": 5.6e-3, "atol": 4e-2},
            },
            "linalg.lu_factor_ex": {
                torch.float32: {"rtol": 2.9e-3, "atol": 5e-2},
            },
            "linalg.solve_ex": {
                torch.float32: {"rtol": 8.8e-3, "atol": 5.8e-2},
            },
            "linalg.solve_triangular": {
                torch.float32: {"rtol": 4.8e-3, "atol": 4.5e-3},
            },
            "linalg.vector_norm": {
                torch.bfloat16: {"rtol": 2.2e-2, "atol": 1.6e-2},
                torch.float16: {"rtol": 2.1e-3, "atol": 3.7e-4},
            },
            "log10": {
                torch.float16: {"rtol": 2e-3, "atol": 2e-4},
            },
            "log2": {
                torch.float16: {"rtol": 1.3e-3, "atol": 2.5e-4},
            },
            # The logcumsumexp backward is a reverse cumulative softmax (exp and
            # division), so bf16 gradients diverge more than the forward pass.
            "logcumsumexp": {
                torch.bfloat16: {"rtol": 8e-2, "atol": 2e-1},
            },
            "matmul": {
                torch.float16: {"rtol": 1e-1, "atol": 1e-1},
                torch.float32: {"rtol": 1e-1, "atol": 1e-1},
            },
            "mm": {
                torch.float16: {"rtol": 1e-1, "atol": 7e-2},
                torch.float32: {"rtol": 9e-1, "atol": 6e-2},
            },
            "mul": {
                torch.float16: {"rtol": 1.1e-3, "atol": 4e-3},
            },
            "nn.functional.batch_norm": {
                torch.float16: {"rtol": 1, "atol": 2.6e-3},
            },
            "nn.functional.conv2d": {
                torch.bfloat16: {"rtol": 2e-5, "atol": 3e-1},
            },
            "nn.functional.ctc_loss": {
                torch.float32: {"rtol": 6.9e-3, "atol": 2.3e-4},
            },
            "nn.functional.embedding_bag": {
                torch.float16: {"rtol": 2e-3, "atol": 8e-3},
            },
            "nn.functional.gelu": {
                torch.bfloat16: {"rtol": 2e-2, "atol": 4e-4},
                torch.float32: {"rtol": 7e-3, "atol": 2e-4},
            },
            "nn.functional.glu": {
                torch.float32: {"rtol": 6.7e-5, "atol": 1.1e-5},
            },
            "nn.functional.group_norm": {
                torch.bfloat16: {"rtol": 6e-1, "atol": 3e-2},
                torch.float32: {"rtol": 1.7, "atol": 3e-1},
                torch.float64: {"rtol": 1.7, "atol": 5e-2},
            },
            "nn.functional.hardswish": {
                torch.bfloat16: {"rtol": 2.8e-2, "atol": 1e-2},
                torch.float16: {"rtol": 6e-3, "atol": 1e-3},
            },
            "nn.functional.softplus": {
                torch.float16: {"rtol": 2.3e-3, "atol": 3.1e-5},
                torch.float32: {"rtol": 2.1e-4, "atol": 1.4e-5},
            },
            "polygamma": {
                torch.float32: {"rtol": 3.1e-5, "atol": 1e-3},
            },
            "rsqrt": {
                torch.bfloat16: {"rtol": 2e-2, "atol": 1e-3},
                torch.float16: {"rtol": 3.7e-3, "atol": 2e-3},
            },
            "softmax": {
                torch.bfloat16: {"rtol": 1e-3, "atol": 5.8e-3},
                torch.float16: {"rtol": 1e-3, "atol": 1.1e-3},
            },
            "tan": {
                torch.float32: {"rtol": 1e-5, "atol": 2e-2},
            },
            "tanh": {
                torch.bfloat16: {"rtol": 2.4e-2, "atol": 2e-3},
                torch.float16: {"rtol": 1.2e-1, "atol": 1.1e-3},
                torch.float32: {"rtol": 3.8e-1, "atol": 6.6e-5},
            },
            "var": {
                torch.bfloat16: {"rtol": 3e-1, "atol": 2e-2},
            },
            # go/keep-sorted end
        },
    )
)  # end of ACCURACY_OVERRIDES_GRAD


# Returns true for test cases that trigger GPU bug.
#
# This function detects test cases that run `clamp_()` on an input tensor of
# dtype `uint8`, and on tensor arguments that hold negative elements. This
# pattern triggers a known PyTorch GPU bug.
#
# Note that this also affects both `clamp_min_()` and `clamp_max_()`, since
# their implementation is technically the same as `clamp_()`.
#
# More details: the bug is actually in the GPU implementation of `maximum()` and
# `minimum()`. More specifically, PyTorch wasn't promoting the inputs for those
# ops.
#
# Ref: https://github.com/pytorch/pytorch/issues/173110
#
# TODO: b/478321000 remove when PyTorch#173110 is fixed.
def _inplace_clamp_input_has_negative_values_uint8_gpu(
    golden_device_type: str,
    variant: OpVariant,
    op_input: OpInput,
) -> bool:

  # Returns True if `arg` is a tensor, and it holds a negative value.
  def is_tensor_and_has_negative_values(arg: Any) -> bool:
    # Short-circuit non-tensor arguments.
    if not isinstance(arg, torch.Tensor):
      return False

    # Check that `torch.any()` is actually run on CPU so as not to cause any
    # undesireable noise when measuring performance.
    assert (
        arg.device.type == "cpu"
    ), "pre-processing golden results should be run on CPU device."

    return torch.any(arg < 0)

  return (
      # PyTorch GPU implementation bug.
      golden_device_type == "gpu"
      # Main input should be a `uint8` tensor.
      and isinstance(op_input.input_value, torch.Tensor)
      and op_input.input_value.dtype == torch.uint8
      # Only on in-place variant.
      # It also happens on out-of-place variant, but only when the output tensor
      # is of the same dtype as the main input tensor.
      and variant == OpVariant.INPLACE
      # There should be at least one tensor argument (min or max) that holds a
      # negative value. This will cause overflow on dtype conversion.
      and any(is_tensor_and_has_negative_values(arg) for arg in op_input.args)
  )


# Returns true for test cases that set `pivot=False`.
# TODO: support `linalg.lu_factor_ex(pivot=False)` on TorchTPU.
# TODO: support `linalg.lu(pivot=False)` on TorchTPU.
def _linalg_lu_without_pivot_gpu(
    golden_device_type: str, unused_variant: OpVariant, op_input: OpInput
) -> bool:
  return (
      # Only GPU supports `pivot=False`.
      golden_device_type == "gpu"
      # The `pivot` keyword argument is set to False.
      and not op_input.kwargs.get("pivot", True)
  )


# Returns true for test cases with `training=False` when comparing against GPU.
# TODO(b/541256155): fix outputs to match GPU behavior when `training=False`.
def _native_batch_norm_legit_notrain_gpu(
    golden_device_type: str, unused_variant: OpVariant, op_input: OpInput
) -> bool:
  return (
      golden_device_type == "gpu"
      # _native_batch_norm_legit() has 1 (input) + 7 parameters.
      and len(op_input.args) == 7
      # Excluding the input, `training` is the 5-th parameter.
      and not op_input.args[4]
  )


# Returns true for test cases in eager mode, with 0-sized tensor input (trivial
# case) whose dtype is not floating point.
# TODO: support 0-sized tensor inputs whose dtype is not floating point in TPU
# eager mode.
def _native_layer_norm_0sized_nonfloat_eager_gpu(
    golden_device_type: str, unused_variant: OpVariant, op_input: OpInput
) -> bool:
  return (
      golden_device_type == "gpu"
      and not op_testing.is_compiled_mode()
      and not torch.is_floating_point(op_input.input_value)
      # Trivial case: 0-sized tensors.
      and op_input.input_value.numel() == 0
  )


# Returns true for test cases with, at least one of `row` (first argument) or
# `col` (second argument) is 0, and `dtype=bool`.
# TODO: support bool inputs.
def _tril_indices_bool_zero_arg_eager_gpu(
    golden_device_type: str, unused_variant: OpVariant, op_input: OpInput
) -> bool:
  return (
      golden_device_type == "gpu"
      and not op_testing.is_compiled_mode()
      # The input is an integer.
      and isinstance(op_input.input_value, int)
      # The second argument (i.e. first is input) is an integer.
      and len(op_input.args) > 0
      and isinstance(op_input.args[0], int)
      # One of them is 0.
      and (op_input.input_value == 0 or op_input.args[0] == 0)
      # The dtype kwarg is torch.bool.
      and op_input.kwargs.get("dtype") == torch.bool
  )


def _if_tpu_vs_gpu_compiled(true_value: Any, false_value: Any) -> Any:
  """Returns true_value if the test is TPU vs GPU compiled, false_value otherwise."""
  return true_value if op_testing.is_tpu_vs_gpu_compiled() else false_value


class TestOps(TorchTpuTestBase):
  """Tests for ops using randomly generated inputs."""

  # TODO: add tests for the following ops:
  # go/keep-sorted start
  # _local_scalar_dense
  # index_select
  # native_dropout
  # relu_
  # go/keep-sorted end

  def setUp(self):
    super().setUp()

    # Filter the tests by --test_categories only if --test_filter is not on
    # the bazel command line. If --test_filter is specified, we should respect
    # the user's intent and not apply any additional filtering based on
    # --test_categories.
    if not _has_test_filter() and _TEST_CATEGORIES.value:
      exclusions = {c[1:] for c in _TEST_CATEGORIES.value if c.startswith("-")}
      inclusions = {c for c in _TEST_CATEGORIES.value if not c.startswith("-")}

      method = getattr(self, self._testMethodName, None)
      test_categories = getattr(method, "categories", set())

      if any(cat in exclusions for cat in test_categories):
        self.skipTest(
            f"Skipping test because category is excluded: {test_categories}"
        )

      if inclusions and not any(cat in inclusions for cat in test_categories):
        self.skipTest(
            f"Skipping test because category is not included: {test_categories}"
        )

    self.set_accuracy_overrides(
        tpu_cpu_overrides=ACCURACY_OVERRIDES_VS_CPU,
        tpu_gpu_overrides=ACCURACY_OVERRIDES_VS_GPU_COMPILED
        if op_testing.is_compiled_mode()
        else ACCURACY_OVERRIDES_VS_GPU,
        grad_overrides=ACCURACY_OVERRIDES_GRAD,
    )
    self.set_dynamism_handlers(
        dynamism_test_utils.verify_op_supports_dynamism,
        dynamism_test_utils.mark_input_dynamic,
    )

  def test_abs(self):
    self.do_test_op(
        "abs",
        # TODO(b/495929595): can be incorrect close to 0.
        # TODO: stablehlo gradients for Real/Imag/Abs on complex are not
        # working.
        skip_if=lambda _1, _2, op_input: (
            op_testing._COMPUTE_GRAD.value
            and op_input.input_value.dtype == torch.complex64
        ),
        exclude_dtypes={"cpu": (torch.bool,)},
        exclude_inplace_dtypes={"cpu": (torch.bool,)},
    )

  def test_acos(self):
    self.do_test_op("acos")

  def test_acosh(self):
    self.do_test_op("acosh")

  def test_adaptive_avg_pool2d(self):
    self.do_test_op("nn.functional.adaptive_avg_pool2d")

  def test_adaptive_avg_pool3d(self):
    self.do_test_op(
        "nn.functional.adaptive_avg_pool3d",
        exclude_dtypes={
            "gpu": INTEGRAL_DTYPES + COMPLEX_DTYPES,
        },
    )

  def test_add(self):
    self.do_test_op("add")

  def test_addcdiv(self):
    self.do_test_op("addcdiv")

  def test_addcmul(self):
    self.do_test_op("addcmul")

  def test_addmm(self):
    self.do_test_op(
        "addmm",
        # TODO: XLA doesn't support complex dtypes currently. Fails with:
        # algebraic_simplifier.cc:584] Check failed: computation->Accept(this)
        # is OK (UNIMPLEMENTED: Converting from type C128 to type F32 is not
        # implemented.
        exclude_dtypes={
            "cpu": COMPLEX_DTYPES,
            # TODO: make addmm fail for integral dtypes to match GPU.
            "gpu": (
                COMPLEX_DTYPES
                + (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
            ),
        },
        # TODO: XLA doesn't support complex dtypes currently. Fails with:
        # algebraic_simplifier.cc:584] Check failed: computation->Accept(this)
        # is OK (UNIMPLEMENTED: Converting from type C128 to type F32 is not
        # implemented.
        exclude_inplace_dtypes={
            "cpu": COMPLEX_DTYPES,
            "gpu": (
                COMPLEX_DTYPES
                + (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
            ),
        },
    )

  def test_addmv(self):
    # TODO: make addmv fail for integral dtypes to match GPU.
    self.do_test_op(
        "addmv",
        # GPU (CUDA) does not support integral dtypes for addmv.
        exclude_dtypes={
            "gpu": (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
        },
        exclude_inplace_dtypes={
            "gpu": (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
        },
    )

  def test_all(self):
    self.do_test_op("all")

  def test_any(self):
    self.do_test_op("any")

  def test_amax(self):
    self.do_test_op("amax")

  def test_amin(self):
    self.do_test_op("amin")

  def test_aminmax(self):
    self.do_test_op(
        "aminmax",
        # TODO: make this work with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
    )

  def test_angle(self):
    self.do_test_op(
        "angle",
        # TODO(b/540345849): succeeded on TPU but failed on GPU for bfloat16 and
        # float16. CUDA kernel is not implemented for bfloat16 and float16.
        exclude_dtypes={"gpu": (torch.bfloat16, torch.float16)},
    )

  def test_arange(self):
    """Tests arange, arange.start, arange.out, arange.start_step, arange.start_out."""

    self.do_test_op(
        "arange",
        # TODO: fix arange(out=...) failing.
        check_out_variant=False,
        # TODO: fix arange() succeeding for bool and complex types (it should
        # fail).
        # TODO: fix arange() returning wrong results for uint8 and int64.
        exclude_dtypes=(torch.bool, torch.uint8, torch.int64) + COMPLEX_DTYPES,
        # NOTE: the test sample contains torch.arange(5), which *should*
        # return a tensor on CPU. Therefore we don't check that the result
        # is on TPU.
        check_device=False,
    )

  def test_argmax(self):
    self.do_test_op("argmax")

  def test_argmin(self):
    self.do_test_op("argmin")

  def test_as_strided(self):
    self.do_test_op("as_strided")

  def test_asin(self):
    self.do_test_op("asin")

  def test_asinh(self):
    self.do_test_op("asinh")

  def test_atan(self):
    self.do_test_op("atan")

  def test_atan2(self):
    self.do_test_op("atan2")

  def test_atanh(self):
    self.do_test_op("atanh")

  def test_avg_pool2d(self):
    self.do_test_op(
        "nn.functional.avg_pool2d",
        # TODO: fix avg_pool2d() succeeding for int64 on TPU.
        # GPU (CUDA) does not support int64 for avg_pool2d.
        exclude_dtypes={"gpu": (torch.int64,)},
    )

  def test_avg_pool3d(self):
    self.do_test_op(
        "nn.functional.avg_pool3d",
        # TODO: GPU does not support integral, complex,
        # `bfloat16` and `float16` dtypes.
        exclude_dtypes={
            "gpu": (
                (torch.uint8, torch.int8, torch.int16, torch.int64)
                + COMPLEX_DTYPES
                + (torch.bfloat16, torch.float16)
            )
        },
    )

  def test_baddbmm(self):
    self.do_test_op(
        "baddbmm",
        # TODO(b/495524286): Failed to generate integral golden results on GPU
        # GPU (CUDA) does not support integral dtypes for baddbmm.
        exclude_dtypes={
            "cpu": (torch.bool,),
            "gpu": INTEGRAL_DTYPES,
        },
        exclude_inplace_dtypes={
            "cpu": (torch.bool,),
            "gpu": INTEGRAL_DTYPES,
        },
    )

  def test_bernoulli(self):
    self.do_test_op(
        "bernoulli",
        # By definition, bernoulli() returns a tensor with random values, so
        # there's no point in checking the values.
        check_value=CheckValueMode.SKIP,
        # GPU (CUDA) does not support complex dtypes for bernoulli.
        exclude_dtypes=COMPLEX_DTYPES,
    )

  def test_binary_cross_entropy(self):
    self.do_test_op(
        "nn.functional.binary_cross_entropy",
        # Exclude non-floating dtypes as they are not supported by the op
        # and because GPU golden sample generation fails for the integer
        # dtypes.
        exclude_dtypes=INTEGRAL_DTYPES  # EXCLUDE_DTYPES_OK=unsupported by op
        + COMPLEX_DTYPES,
        # TODO(b/540303890): remove when backwards ops are implemented.
        check_grad=False,
    )

  def test_bincount(self):
    self.do_test_op(
        "bincount",
        # Excluded because they are not supported as input to bincount and
        # because the op_testing code fails to generate random inputs when
        # these types are enabled.
        exclude_dtypes={
            "cpu": COMPLEX_DTYPES + FLOAT_DTYPES + (torch.bool,),
            "gpu": COMPLEX_DTYPES + (torch.float64,) + (torch.bool,),
        },
    )

  def test_bitwise_and(self):
    self.do_test_op("bitwise_and")

  def test_bitwise_left_shift(self):
    self.do_test_op("bitwise_left_shift")

  def test_bitwise_not(self):
    self.do_test_op(
        "bitwise_not",
        # TODO: fix bitwise_not() succeeding on TPU with float inputs.
        check_op_failures=False,
        # TODO: fix bitwise_not_() succeeding on TPU with float inputs.
        check_inplace_op_failures=False,
    )

  def test_bitwise_or(self):
    self.do_test_op("bitwise_or")

  def test_bitwise_right_shift(self):
    self.do_test_op("bitwise_right_shift")

  def test_bitwise_xor(self):
    self.do_test_op("bitwise_xor")

  def test_bmm(self):
    self.do_test_op(
        "bmm",
        # GPU (CUDA) does not support integral dtypes for bmm.
        exclude_dtypes={
            "gpu": (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ),
        },
    )

  def test_cat(self):
    self.do_test_op("cat")

  def test_cdist(self):
    # TODO: b/470453016 look into the high relative error for bfloat16.
    # torch.bfloat16: {"rtol": 3.1, "atol": 3.9e-1}
    self.do_test_op(
        "cdist",
        # TODO: look into sometimes tests will fall into certain
        # CPU implementation.
    )

  def test_ceil(self):
    self.do_test_op(
        "ceil",
        exclude_dtypes=[torch.bool],
        exclude_inplace_dtypes=[torch.bool],
    )

  def test_clamp(self):
    self.do_test_op(
        "clamp",
        # b/446131726 - clamp() fails on TPU with bool dtypes.
        exclude_dtypes=(torch.bool,),
        exclude_inplace_dtypes=(torch.bool,),
        # TODO: b/478321000 remove when PyTorch#173110 is fixed.
        skip_if=_inplace_clamp_input_has_negative_values_uint8_gpu,
        # TODO: fix clamp() returning enormous errors or nans when dynamism is
        # enabled.
        check_dynamism=False,
    )

  def test_clamp_min(self):
    self.do_test_op(
        "clamp_min",
        # TODO: b/478321000 remove when PyTorch#173110 is fixed.
        skip_if=_inplace_clamp_input_has_negative_values_uint8_gpu,
        exclude_dtypes=(
            torch.complex64,
            torch.bool,
        ),
        exclude_inplace_dtypes=(
            torch.complex64,
            torch.bool,
        ),
    )

  def test_clamp_max(self):
    self.do_test_op(
        "clamp_max",
        # TODO: b/478321000 remove when PyTorch#173110 is fixed.
        skip_if=_inplace_clamp_input_has_negative_values_uint8_gpu,
        exclude_dtypes=(
            torch.complex64,
            torch.bool,
        ),
        exclude_inplace_dtypes=(
            torch.complex64,
            torch.bool,
        ),
    )

  def test_clone(self):
    self.do_test_op("clone")

  def test_complex(self):
    self.do_test_op(
        "complex",
        # TODO: XLA does not support complex<f16> dtype and complex<f64>.
        exclude_dtypes=(torch.float16, torch.float64),
    )

  def test_conj(self):
    self.do_test_op("conj")

  def test_conj_physical(self):
    self.do_test_op(
        "conj_physical",
        # TODO: b/448907643 - there is a problem with the plumbing of the
        # inplace variant.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
    )

  def test_constant_pad_nd(self):
    self.do_test_op("constant_pad_nd")

  def test_cos(self):
    self.do_test_op("cos")

  def test_cosh(self):
    self.do_test_op("cosh")

  def test_count_nonzero(self):
    self.do_test_op("count_nonzero")

  def test_ctc_loss(self):
    self.do_test_op(
        "nn.functional.ctc_loss",
        # Excluded because PyTorch's sample generation (via log_softmax on CPU)
        # does not support integral, bfloat16, float16, and complex dtypes.
        # Additionally, CPU and GPU do not support bfloat16 and float16.
        exclude_dtypes={
            "cpu": (
                INTEGRAL_DTYPES
                + COMPLEX_DTYPES
                + (torch.bfloat16, torch.float16)
            ),
            "gpu": (
                INTEGRAL_DTYPES
                + COMPLEX_DTYPES
                + (torch.bfloat16, torch.float16)
            ),
        },
    )

  def test_cumprod(self):
    self.do_test_op("cumprod")

  # TODO(b/529376045): Scan HLO lowering failing on GitHub
  # def test_cummax(self):
  #   self.do_test_op("cummax")

  def test_cumsum(self):
    self.do_test_op("cumsum")

  # TODO(bawilson): Scan HLO lowering failing on GitHub
  # def test_cummin(self):
  #   self.do_test_op("cummin")

  def test_diagonal(self):
    self.do_test_op("diagonal")

  def test_digamma(self):
    self.do_test_op("digamma")

  def test_div(self):
    self.do_test_op("div")

  def test_dot(self):
    self.do_test_op(
        "dot",
        # Relaxes the comparison mode from STRICT to LOOSE for floating point
        # precision differences. The absolute difference between the expected
        # and actual results can be slightly above the strict tolerance of
        # 1e-05.
        # GPU (CUDA) does not support integral dtypes for dot.
        exclude_dtypes={
            "gpu": (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ),
        },
    )

  def test_native_dropout_backward(self):
    self.do_test_op(
        "native_dropout_backward",
        # GPU (CUDA) does not support complex and integral dtypes for
        # native_dropout_backward.
        exclude_dtypes={
            "gpu": INTEGRAL_DTYPES + COMPLEX_DTYPES,
        },
    )

  def test_embedding(self):
    self.do_test_op(
        "nn.functional.embedding",
        # TODO: fix embedding() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: add support for sparse embeddings.
        skip_if=lambda device, variant, op_input: op_input.kwargs.get(
            "sparse", False
        ),
    )

  # TODO(b/529376045): Scan HLO lowering failing on GitHub
  # def test_embedding_bag(self):
  #   self.do_test_op(
  #       "nn.functional.embedding_bag",
  #       # TODO: add support for sparse embeddings.
  #       skip_if=lambda device, variant, op_input: (
  #           op_input.kwargs.get("sparse", False)
  #           or op_input.kwargs.get("scale_grad_by_freq", False)
  #       ),
  #   )

  def test_empty(self):
    self.do_test_op(
        "empty",
        # empty returns a tensor filled with uninitialized data. Tensor content
        # has no meaning.
        check_value=CheckValueMode.SKIP,
    )

  def test_empty_strided(self):
    self.do_test_op(
        "empty_strided",
        # empty returns a tensor filled with uninitialized data. Tensor content
        # has no meaning.
        check_value=CheckValueMode.SKIP,
    )

  def test_eq(self):
    self.do_test_op("eq")

  def test_equal(self):
    self.do_test_op("equal")

  def test_erf(self):
    self.do_test_op("erf")

  def test_erfinv(self):
    self.do_test_op("erfinv")

  def test_exp(self):
    self.do_test_op("exp")

  def test_exp2(self):
    self.do_test_op("exp2")

  def test_expm1(self):
    self.do_test_op("expm1")

  def test_expand(self):
    self.do_test_op("expand")

  def test_exponential(self):
    self.do_test_op(
        "exponential",
        # By definition, exponential() returns a tensor with random values, so
        # there's no point in checking the values.
        check_value=CheckValueMode.SKIP,
    )

  def test_eye(self):
    self.do_test_op("eye")

  def test_fft_rfft(self):
    self.do_test_op(
        "fft.rfft",
        exclude_dtypes={
            "cpu": (torch.float64,),
            # TODO: b/518595804 - PyTorch currently does not support half
            # precision FFTs for PrivateUse1 backends. See promote_type_fft
            # in SpectralOps.cpp.
            "gpu": (torch.float64, torch.float16, torch.bfloat16),
        },
    )

  def test_fft_fft(self):
    self.do_test_op(
        "fft.fft",
        exclude_dtypes={
            "cpu": (torch.float64,),
            # TODO: b/518595804 - PyTorch currently does not support half
            # precision FFTs for PrivateUse1 backends. See promote_type_fft
            # in SpectralOps.cpp.
            "gpu": (torch.float64, torch.float16, torch.bfloat16),
        },
    )

  def test_fft_ifft(self):
    self.do_test_op(
        "fft.ifft",
        exclude_dtypes={
            "cpu": (torch.float64,),
            # TODO: b/518595804 - PyTorch currently does not support half
            # precision FFTs for PrivateUse1 backends. See promote_type_fft
            # in SpectralOps.cpp.
            "gpu": (torch.float64, torch.float16, torch.bfloat16),
        },
    )

  def test_fft_irfft(self):
    self.do_test_op(
        "fft.irfft",
        exclude_dtypes={
            "cpu": (torch.float64,),
            # TODO: b/518595804 - PyTorch currently does not support half
            # precision FFTs for PrivateUse1 backends. See promote_type_fft
            # in SpectralOps.cpp.
            "gpu": (torch.float64, torch.float16, torch.bfloat16),
        },
    )

  def test_fill(self):
    self.do_test_op("fill")

  def test_flatten(self):
    # TODO: check flatten is being properly exercised.
    self.do_test_op("flatten")

  def test_floor(self):
    self.do_test_op(
        "floor",
        exclude_dtypes=[torch.bool],
        exclude_inplace_dtypes=[torch.bool],
    )

  def test_floor_divide(self):
    self.do_test_op(
        "floor_divide",
        # TODO: cpu does incorrect rounding for bfloat16 and float16.
        exclude_dtypes=(torch.bfloat16, torch.float16),
        exclude_inplace_dtypes=(torch.bfloat16, torch.float16),
    )

  def test_flip(self):
    self.do_test_op("flip")

  def test_fmax(self):
    self.do_test_op("fmax")

  def test_fmin(self):
    self.do_test_op("fmin")

  def test_fmod(self):
    self.do_test_op("fmod")

  @category("foreach")
  def test_foreach_abs(self):
    self.do_test_op(
        "_foreach_abs",
        exclude_dtypes={"cpu": (torch.bool,)},
        exclude_inplace_dtypes={"cpu": (torch.bool,)},
    )

  @category("foreach")
  def test_foreach_acos(self):
    self.do_test_op("_foreach_acos")

  @category("foreach")
  def test_foreach_add(self):
    self.do_test_op(
        "_foreach_add",
        # TODO(b/485291373): fix _foreach_add() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO(b/485291373): fix _foreach_add_() failing with complex dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_addcdiv(self):
    self.do_test_op(
        "_foreach_addcdiv",
        # TODO(b/485291373): fix _foreach_addcdiv() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO(b/485291373): fix _foreach_addcdiv_() failing with complex
        # dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
        check_dynamism=False,  # TODO(b/488338235): dynamism is flaky
    )

  @category("foreach")
  def test_foreach_addcmul(self):
    # TODO(b/494218929): Fix the high tolerance of 1e-2.
    self.do_test_op(
        "_foreach_addcmul",
        # TODO(b/485291373): fix _foreach_addcmul_() failing with complex
        # dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_asin(self):
    self.do_test_op("_foreach_asin")

  @category("foreach")
  def test_foreach_atan(self):
    self.do_test_op("_foreach_atan")

  @category("foreach")
  def test_foreach_ceil(self):
    self.do_test_op(
        "_foreach_ceil",
        exclude_dtypes=[torch.bool],
        exclude_inplace_dtypes=[torch.bool],
    )

  @category("foreach")
  def test_foreach_clamp_max(self):
    self.do_test_op(
        "_foreach_clamp_max",
        # TODO: "clamp_max_scalar_cpu" not implemented for 'Bool'.
        # TODO: fix _foreach_clamp_max() failing with complex dtypes.
        exclude_dtypes=(torch.bool,) + COMPLEX_DTYPES,
        # TODO: fix _foreach_clamp_max_() failing with complex dtypes.
        exclude_inplace_dtypes=(torch.bool,) + COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_clamp_min(self):
    self.do_test_op(
        "_foreach_clamp_min",
        # TODO: "clamp_min_scalar_cpu" not implemented for 'Bool'.
        # TODO: fix _foreach_clamp_min() failing with complex dtypes.
        exclude_dtypes=(torch.bool,) + COMPLEX_DTYPES,
        # TODO: fix _foreach_clamp_min_() failing with complex dtypes.
        exclude_inplace_dtypes=(torch.bool,) + COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_copy(self):
    self.do_test_op("_foreach_copy")

  @category("foreach")
  def test_foreach_cos(self):
    self.do_test_op("_foreach_cos")

  @category("foreach")
  def test_foreach_cosh(self):
    self.do_test_op("_foreach_cosh")

  @category("foreach")
  def test_foreach_div(self):
    self.do_test_op(
        "_foreach_div",
        # TODO: CPU returns nans but TPU returns 0.
        # TODO(b/485291373): fix _foreach_div() failing with complex dtypes.
        exclude_dtypes=(torch.bool,) + COMPLEX_DTYPES,
        # TODO: CPU returns nans but TPU returns 0.
        # TODO(b/485291373): fix _foreach_div_() failing with integral and
        # complex dtypes.
        exclude_inplace_dtypes=INTEGRAL_DTYPES + COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_erf(self):
    self.do_test_op("_foreach_erf")

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  @category("foreach")
  @oss_utils.skip_in_oss()
  def test_foreach_erfc(self):
    self.do_test_op("_foreach_erfc")

  @category("foreach")
  def test_foreach_exp(self):
    self.do_test_op("_foreach_exp")

  @category("foreach")
  def test_foreach_expm1(self):
    self.do_test_op("_foreach_expm1")

  @category("foreach")
  def test_foreach_floor(self):
    self.do_test_op(
        "_foreach_floor",
        exclude_dtypes=[torch.bool],
        exclude_inplace_dtypes=[torch.bool],
    )

  @category("foreach")
  def test_foreach_frac(self):
    self.do_test_op("_foreach_frac")

  @category("foreach")
  def test_foreach_lerp(self):
    self.do_test_op(
        "_foreach_lerp",
        # TODO(b/485291373): fix _foreach_lerp_() failing with complex dtypes.
        exclude_inplace_dtypes=(torch.complex64,),
    )

  @category("foreach")
  def test_foreach_lgamma(self):
    self.do_test_op(
        "_foreach_lgamma",
        # Too slow for float64.
        exclude_dtypes=(torch.float64,),
        exclude_inplace_dtypes=(torch.float64,),
    )

  @category("foreach")
  def test_foreach_log(self):
    self.do_test_op("_foreach_log")

  @category("foreach")
  def test_foreach_log10(self):
    self.do_test_op("_foreach_log10")

  @category("foreach")
  def test_foreach_log1p(self):
    self.do_test_op("_foreach_log1p")

  @category("foreach")
  def test_foreach_log2(self):
    self.do_test_op("_foreach_log2")

  @category("foreach")
  def test_foreach_max(self):
    # TODO(b/485291373): fix _foreach_max() failing with complex dtypes.
    self.do_test_op("_foreach_max", exclude_dtypes=COMPLEX_DTYPES)

  @category("foreach")
  def test_foreach_maximum(self):
    self.do_test_op(
        "_foreach_maximum",
        # TODO: _foreach_maximum() with dtype torch.bool failed on CPU, so it
        # should fail on TorchTPU too.
        # TODO: fix _foreach_maximum() failing with complex dtypes.
        exclude_dtypes=(torch.bool,) + COMPLEX_DTYPES,
        # TODO: _foreach_maximum_() with dtype torch.bool failed on CPU, so it
        # should fail on TorchTPU too.
        # TODO: fix _foreach_maximum_() failing with complex dtypes.
        exclude_inplace_dtypes=(torch.bool,) + COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_minimum(self):
    self.do_test_op(
        "_foreach_minimum",
        # TODO: _foreach_minimum() with dtype torch.bool failed on CPU, so it
        # should fail on TorchTPU too.
        # TODO: fix _foreach_minimum() failing with complex dtypes.
        exclude_dtypes=(torch.bool,) + COMPLEX_DTYPES,
        # TODO: _foreach_minimum_() with dtype torch.bool failed on CPU, so it
        # should fail on TorchTPU too.
        # TODO: fix _foreach_minimum_() failing with complex dtypes.
        exclude_inplace_dtypes=(torch.bool,) + COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_mul(self):
    self.do_test_op(
        "_foreach_mul",
        # TODO(b/485291373): fix _foreach_mul() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO(b/485291373): fix _foreach_mul_() failing with complex dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_neg(self):
    self.do_test_op("_foreach_neg")

  @category("foreach")
  def test_foreach_norm(self):

    def skip_if(device_type, variant, op_input):
      _ = device_type  # Unused, suppress linter error
      _ = variant  # Unused, suppress linter error
      # TODO(b/488385491): Enable grad check for negative ord when the issues
      # are fixed.
      ord_val = op_input.kwargs.get("ord")
      # For _foreach_norm, the first argument in the signature is `tensors`,
      # which is stored in op_input.input_value. The `ord` parameter is the
      # second argument. Thus, if `ord` is passed positionally, it will be
      # at index 0 of `op_input.args`.
      if ord_val is None and len(op_input.args) > 0:
        ord_val = op_input.args[0]
      if op_testing._COMPUTE_GRAD.value and ord_val is not None:
        try:
          if float(ord_val) < 0:
            return f"Skip ord={ord_val} as it has issues on TPU."
        except (ValueError, TypeError):
          pass
      return None

    self.do_test_op(
        "_foreach_norm",
        # TODO(b/485291373): fix _foreach_norm() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        skip_if=skip_if,
    )

  @category("foreach")
  def test_foreach_pow(self):
    self.do_test_op(
        "_foreach_pow",
        # TODO: fix TPU failure for these dtypes.
        exclude_dtypes=(torch.bool, torch.int64, torch.complex64),
        exclude_inplace_dtypes=(torch.bool, torch.int64, torch.complex64),
    )

  @category("foreach")
  def test_foreach_reciprocal(self):
    self.do_test_op("_foreach_reciprocal")

  @category("foreach")
  def test_foreach_round(self):
    self.do_test_op("_foreach_round")

  @category("foreach")
  def test_foreach_rsqrt(self):
    self.do_test_op("_foreach_rsqrt")

  @category("foreach")
  def test_foreach_sigmoid(self):
    self.do_test_op("_foreach_sigmoid")

  @category("foreach")
  def test_foreach_sign(self):
    self.do_test_op("_foreach_sign")

  @category("foreach")
  def test_foreach_sin(self):
    self.do_test_op("_foreach_sin")

  @category("foreach")
  def test_foreach_sinh(self):
    self.do_test_op("_foreach_sinh")

  @category("foreach")
  def test_foreach_sqrt(self):
    self.do_test_op("_foreach_sqrt")

  @category("foreach")
  def test_foreach_sub(self):
    self.do_test_op(
        "_foreach_sub",
        # TODO: fix _foreach_sub() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: fix _foreach_sub_() failing with complex dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
    )

  @category("foreach")
  def test_foreach_tan(self):
    self.do_test_op("_foreach_tan")

  @category("foreach")
  def test_foreach_tanh(self):
    self.do_test_op("_foreach_tanh")

  @category("foreach")
  def test_foreach_trunc(self):
    self.do_test_op("_foreach_trunc")

  @category("foreach")
  def test_foreach_zero(self):
    self.do_test_op("_foreach_zero")

  def test_full(self):
    self.do_test_op("full")

  def test_full_like(self):
    self.do_test_op("full_like")

  def test_gather(self):
    self.do_test_op("gather")

  def test_ge(self):
    self.do_test_op("ge")

  def test_gt(self):
    self.do_test_op("gt")

  def test_grid_sample(self):
    self.do_test_op(
        "nn.functional.grid_sample",
        exclude_dtypes={
            # CPU implementation has precision issues leading to incorrect
            # addressing for float16 and bfloat16
            "cpu": (torch.bfloat16, torch.float16),
        },
    )

  def test_histc(self):
    # TODO: b/487653210 - skipping `int8` and `uint8` dtypes due to incorrect
    # results. Test those dtypes once the bug is fixed.

    self.do_test_op(
        "histc",
        # NOTE: Exclude integral dtypes on CPU, because histc() is not
        # implemented in the PyTorch CPU backend.
        # Additionally, exclude float16 and bfloat16 because (expected)
        # precision variations can lead to binning errors.
        # TODO: GPU does not support `bool`, `bfloat16`, and `float16` dtypes.
        exclude_dtypes={
            "cpu": INTEGRAL_DTYPES + (torch.float16,),
            "gpu": (
                torch.bool,
                torch.int8,
                torch.uint8,
                torch.bfloat16,
                torch.float16,
            ),
        },
        # TODO(b/542534758): `histc()` does not support dynamism.
        check_dynamism=False,
    )

  def test_histc_skip_value_check_due_to_precision_variations(self):
    # NOTE: Verify output shapes and dtypes of previously excluded dtypes that
    # lead to binning errors due to precision variations.
    self.do_test_op(
        "histc",
        # TODO: GPU does not support `bool`, `bfloat16`, and `float16` dtypes.
        exclude_dtypes={
            "cpu": INTEGRAL_DTYPES,
            "gpu": (
                torch.bool,
                torch.int8,
                torch.uint8,
                torch.bfloat16,
                torch.float16,
            ),
        },
        check_value=CheckValueMode.SKIP,
        # TODO(b/542534758): `histc()` does not support dynamism.
        check_dynamism=False,
    )

  def test_index_add(self):
    self.do_test_op("index_add")

  def test_index_copy(self):
    self.do_test_op("index_copy")

  def test_index_fill(self):
    self.do_test_op("index_fill")

  def test_index_put(self):
    self.do_test_op("index_put")

  def test_index_select(self):
    self.do_test_op(
        "index_select",
        # TODO: fix index_select(out=...) failing.
        check_out_variant=False,
    )

  def test_isfinite(self):
    self.do_test_op("isfinite")

  def test_isin(self):
    self.do_test_op("isin")

  def test_isinf(self):
    self.do_test_op("isinf")

  def test_isnan(self):
    self.do_test_op("isnan")

  def test_isneginf(self):
    self.do_test_op("isneginf")

  def test_isposinf(self):
    self.do_test_op("isposinf")

  def test_kron(self):
    self.do_test_op(
        "kron",
        # TODO: fix kron(out=...) having huge numeric errors.
        check_out_variant=False,
    )

  def test_ldexp(self):
    self.do_test_op("ldexp")

  def test_le(self):
    self.do_test_op("le")

  def test_leaky_relu(self):
    self.do_test_op("nn.functional.leaky_relu")

  def test_lerp(self):
    self.do_test_op(
        "lerp",
        exclude_dtypes=(torch.complex64,),
        exclude_inplace_dtypes=(torch.complex64,),
    )

  def test_lgamma(self):
    self.do_test_op(
        "lgamma",
        # TODO: fix lgamma() failing for complex.
        exclude_dtypes=(torch.complex64,),
        exclude_inplace_dtypes=(torch.complex64,),
    )

  def test_linalg_lu_factor_ex(self):
    self.do_test_op(
        "linalg.lu_factor_ex",
        # TODO(b/528412744): Unskip float64 and complex128 when libtpu has been upreved.
        exclude_dtypes=(  # EXCLUDE_DTYPES_OK=unsupported 64-bit dtypes in older libtpu
            torch.float64,
            torch.complex128,
        ),
        skip_if=_linalg_lu_without_pivot_gpu,
    )

  def test_linalg_triangular_solve(self):
    self.do_test_op(
        "linalg.solve_triangular",
        # bool triggers an error in the sample generation code
        exclude_dtypes=(torch.bool,),
    )

  def test_logit(self):
    self.do_test_op("logit")

  def test_lu_unpack(self):
    self.do_test_op(
        "lu_unpack",
        exclude_dtypes={
            "cpu": (
                INTEGRAL_DTYPES
                + (torch.half, torch.bfloat16, torch.float64, torch.complex128)
            ),
            "gpu": (
                torch.uint8,
                torch.int64,  # Cannot generate GPU sample for this dtype.
                torch.int8,  # Cannot generate GPU sample for this dtype.
                torch.int16,
                torch.int32,  # Cannot generate GPU sample for this dtype.
                torch.bool,  # Cannot generate GPU sample for this dtype.
                torch.half,
                torch.bfloat16,  # Cannot generate GPU sample for this dtype.
                torch.float64,
                torch.complex128,
            ),
        },
        # ApplyPivotsInPlace in linalg_lu_kernels.cc calls .item() in C++ loop,
        # which fails placeholder tensor materialization in compiled mode.
        skip_if=lambda device, variant, op_input: op_testing.is_compiled_mode(),
    )

  def test_linalg_lu_solve(self):
    self.do_test_op(
        "linalg.lu_solve",
        exclude_dtypes={
            "cpu": (
                INTEGRAL_DTYPES
                + (torch.half, torch.bfloat16, torch.float64, torch.complex128)
            ),
            "gpu": (
                torch.uint8,
                torch.int16,
                torch.int64,  # Cannot generate GPU sample for this dtype.
                torch.int32,  # Cannot generate GPU sample for this dtype.
                torch.int8,  # Cannot generate GPU sample for this dtype.
                torch.bool,  # Cannot generate GPU sample for this dtype.
                torch.half,
                torch.bfloat16,  # Cannot generate GPU sample for this dtype.
                torch.float64,
                torch.complex128,
            ),
        },
        # ApplyPivotsInPlace in linalg_lu_kernels.cc calls .item() in C++ loop,
        # which fails placeholder tensor materialization in compiled mode.
        skip_if=lambda device, variant, op_input: op_testing.is_compiled_mode(),
    )

  def test_linalg_solve_ex(self):
    self.do_test_op(
        "linalg.solve_ex",
        exclude_dtypes={
            "cpu": (
                INTEGRAL_DTYPES
                + (torch.half, torch.bfloat16, torch.float64, torch.complex128)
            ),
            "gpu": (
                torch.uint8,
                torch.int64,  # Cannot generate GPU sample for this dtype.
                torch.int32,  # Cannot generate GPU sample for this dtype.
                torch.int16,
                torch.int8,  # Cannot generate GPU sample for this dtype.
                torch.bool,  # Cannot generate GPU sample for this dtype.
                torch.half,
                torch.bfloat16,  # Cannot generate GPU sample for this dtype.
                torch.float64,
                torch.complex128,
            ),
        },
        # ApplyPivotsInPlace in linalg_lu_kernels.cc calls .item() in C++ loop,
        # which fails placeholder tensor materialization in compiled mode.
        skip_if=lambda device, variant, op_input: op_testing.is_compiled_mode(),
    )

  def test_linalg_lu_out(self):
    self.do_test_op(
        "linalg.lu",
        exclude_dtypes=(
            torch.float64,
            torch.complex128,
        ),
        # ApplyPivotsInPlace in linalg_lu_kernels.cc calls .item() in C++ loop,
        # which fails placeholder tensor materialization in compiled mode.
        skip_if=lambda device, variant, op_input: (
            _linalg_lu_without_pivot_gpu(device, variant, op_input)
            or op_testing.is_compiled_mode()
        ),
    )

  def test_linalg_inv_ex_out(self):
    self.do_test_op(
        "linalg.inv",
        exclude_dtypes={
            "cpu": INTEGRAL_DTYPES + (torch.half, torch.bfloat16),
            "gpu": (
                torch.uint8,
                torch.int8,  # Cannot generate GPU sample for this dtype.
                torch.int16,
                torch.int32,  # Cannot generate GPU sample for this dtype.
                torch.int64,  # Cannot generate GPU sample for this dtype.
                torch.bool,  # Cannot generate GPU sample for this dtype.
                torch.bfloat16,  # Cannot generate GPU sample for this dtype.
                torch.half,
            ),
        },
        # TODO(b/495521055): linalg.inv fails with complex64 with compile.
        skip_if=lambda device, variant, op_input: (
            op_testing.is_compiled_mode()
            and op_input.input_value.dtype in COMPLEX_DTYPES
        ),
    )

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  @oss_utils.skip_in_oss()
  def test_norm(self):
    self.do_test_op("norm")

  def test_linalg_vector_norm_other_dtypes(self):
    self.do_test_op("linalg.vector_norm")

  def test_linspace(self):
    self.do_test_op(
        "linspace",
        # PyTorch's upstream sample generator for linspace includes a hardcoded
        # sample without a device kwarg: `yield SampleInput(1, args=(3, 1))`
        # (see common_methods_invocations.py).
        # Same issue as test_arange() above.
        skip_if=lambda _1, _2, op_input: "device" not in op_input.kwargs,
    )

  def test_lt(self):
    self.do_test_op("lt")

  def test_log(self):
    self.do_test_op("log")

  def test_log1p(self):
    self.do_test_op("log1p")

  def test_log10(self):
    self.do_test_op("log10")

  def test_log2(self):
    self.do_test_op("log2")

  def test_log_sigmoid(self):
    self.do_test_op("nn.functional.logsigmoid")

  def test_log_softmax(self):
    self.do_test_op("log_softmax")

  def test_log_softmax_backward_data(self):
    self.do_test_op(
        "_log_softmax_backward_data",
        # TODO(unda): this fails for complex and integral dtypes because the
        # sample generation process calls log_softmax() which is not supported
        # for these dtypes.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )

  def test_logcumsumexp(self):
    self.do_test_op(
        "logcumsumexp",
        # logcumsumexp is a floating-point op: integer/bool inputs are
        # unsupported (the reference sample generator itself overflows for
        # them), and the max-based logaddexp combiner has no defined extension
        # to complex.
        exclude_dtypes=(COMPLEX_DTYPES + INTEGRAL_DTYPES),
    )

  def test_logical_and(self):
    self.do_test_op("logical_and")

  def test_logical_or(self):
    self.do_test_op("logical_or")

  def test_logical_xor(self):
    self.do_test_op("logical_xor")

  def test_logical_not(self):
    self.do_test_op("logical_not")

  # TODO(b/529376045): Scan HLO lowering failing on GitHub
  # def test_masked_scatter(self):
  #   self.do_test_op("masked_scatter")

  def test_masked_select(self):
    self.do_test_op("masked_select")

  def test_masked_fill(self):
    self.do_test_op("masked_fill")

  def test_matmul(self):
    self.do_test_op(
        "matmul",
        # TODO: TPU supports bool dtypes but not all CPU and GPU lowerings do.
        # Due to how PyTorch decomposes this op, some cases bool dtype is
        # supported and other cases it's not. This op is supposed to be
        # delegated, but due to the same reason above, there are some funky
        # lowering causing delegation to fail for us. This requires additional
        # work to fix. For now, we just exclude bool dtypes.
        # TODO: fix matmul(out=...) failing with int64 dtypes. Error:
        # While rewriting computation to not contain X64 element types, XLA
        # encountered an HLO for which this rewriting is not implemented:
        # %_run_op.1 = s64[] dot(%Arg_1.1, %Arg_0.1), lhs_contracting_dims={0},
        # rhs_contracting_dims={0}, operand_precision={highest,highest}
        exclude_dtypes={
            "cpu": (
                torch.bool,
                torch.int64,
            ),
            # TODO: make matmul fail for integral dtypes to match GPU.
            "gpu": INTEGRAL_DTYPES,
        },
    )

  def test_multinomial(self):
    self.do_test_op(
        "multinomial",
        # TODO: multinomial() returns a tensor with random values, so we can't
        # check the values until we have a way to seed the random number
        # generator.
        check_value=CheckValueMode.SKIP,
        # TODO: float64 is not supported for rng on TPU.
        exclude_dtypes=(torch.float64,),
    )

  def test_max(self):
    self.do_test_op("max")

  # TODO: b/478321000 remove this comment when PyTorch#173110 is fixed.
  # If this test starts failing due to large absolute errors, look at the
  # comments of `_inplace_clamp_input_has_negative_values_uint8()` function.
  # That might be the cause.
  def test_maximum(self):
    self.do_test_op("maximum")

  def test_max_pool2d(self):
    self.do_test_op(
        "nn.functional.max_pool2d",
        # TODO: complex64, float64, and int64 dtypes are not supported on TPU.
        exclude_dtypes={
            "cpu": (
                torch.complex64,
                torch.float64,
                torch.int64,
            ),
            "gpu": (
                (
                    torch.complex64,
                    torch.float64,
                )
                # TODO: b/476417319 reject integer dtypes in TPU implementation,
                # so that it matches the GPU implementation.
                + (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
            ),
        },
    )

  def test_max_pool3d(self):
    self.do_test_op(
        "nn.functional.max_pool3d",
        # TODO: b/467347286 - complex64, float64, and int64 dtypes
        # are not supported on TPU.
        exclude_dtypes={
            "cpu": (
                torch.complex64,
                torch.float64,
                torch.int64,
            ),
            "gpu": (
                (
                    torch.complex64,
                    torch.float64,
                )
                # TODO: b/476417319 reject integer dtypes in TPU implementation,
                # so that it matches the GPU implementation.
                + (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
            ),
        },
    )

  def test_mean(self):
    self.do_test_op("mean")

  def test_min(self):
    self.do_test_op("min")

  # TODO: b/478321000 remove this comment when PyTorch#173110 is fixed.
  # If this test starts failing due to large absolute errors, look at the
  # comments of `_inplace_clamp_input_has_negative_values_uint8()` function.
  # That might be the cause.
  def test_minimum(self):
    self.do_test_op("minimum")

  def test_mm(self):
    self.do_test_op(
        "mm",
        # TODO: fix mm() failing with integral dtypes.
        exclude_dtypes=INTEGRAL_DTYPES,
    )

  def test_mul(self):
    self.do_test_op("mul")

  def test_nan_to_num(self):
    self.do_test_op("nan_to_num")

  def test_native_batch_norm(self):
    self.do_test_op(
        "native_batch_norm",
        # Due to a PyTorch output dtype inconsistency bw CPU and GPU, we skip
        # checking the output dtype against CPU.
        check_dtype=self.golden_device_type == "gpu",
        # TODO: fix native_batch_norm(out=...) failing.
        check_out_variant=False,
        exclude_dtypes={
            "gpu": (
                torch.bfloat16,
                torch.float16,
                torch.float32,
                torch.float64,
            )
        },
    )

  def test_native_batch_norm_legit(self):
    self.do_test_op(
        "_native_batch_norm_legit",
        # Due to a PyTorch output dtype inconsistency bw CPU and GPU, we skip
        # checking the output dtype against CPU.
        check_dtype=self.golden_device_type == "gpu",
        # TODO: fix _native_batch_norm_legit(out=...) failing.
        check_out_variant=False,
        exclude_dtypes={
            "gpu": (torch.float64,),
        },
        # TODO(b/541256155): match GPU behavior when `training=False`.
        skip_if=_native_batch_norm_legit_notrain_gpu,
    )

  def test_native_group_norm(self):
    # TODO: b/470451730 look into the high errors for bfloat16.
    # torch.bfloat16: {"rtol": 4.1, "atol": 4.0},
    self.do_test_op(
        "nn.functional.group_norm",
        # TODO: fix native_group_norm() succeeding with integral and
        # complex dtypes (it should fail).
        # TODO: b/470458807 look into why native_group_norm() returns NaN values
        # when using float16 dtype, while GPU succeeds.
        exclude_dtypes=(torch.uint8, torch.int8, torch.int16)
        + (torch.complex64,)
        + (torch.float16,)
        + (torch.float64,),
    )

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  @oss_utils.skip_in_oss()
  def test_native_layer_norm(self):
    self.do_test_op(
        "native_layer_norm",
        # TODO: before cl/833944280 introduced the backward ops, this test runs
        # with check_grad=True. As there is no default implementation of
        # backward for native_layer_norm, it should fail. Investigate why it's
        # not failing.
        # NOTE: native_layer_norm() is not implemented for complex64 on CPU,
        # so we have to exclude complex64 here.
        exclude_dtypes={
            "cpu": COMPLEX_DTYPES,
            "gpu": (
                (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                )
                + COMPLEX_DTYPES
            ),
        },
        # TODO: look into why this test produces float32 (default dtype) tensors
        # on GPU for the last 2 tensors being compared, while TPU produces
        # tensors of the same dtype as the input.
        skip_output_indices=[1, 2],
        # TODO(b/542277260): support 0-sized tensor inputs whose dtype is not
        # floating point in TPU eager mode.
        skip_if=_native_layer_norm_0sized_nonfloat_eager_gpu,
    )

  def test_ne(self):
    self.do_test_op("ne")

  def test_new_ones(self):
    self.do_test_op("new_ones")

  def test_new_zeros(self):
    self.do_test_op("new_zeros")

  def test_neg(self):
    self.do_test_op("neg")

  def test_nll_loss(self):
    self.do_test_op("nn.functional.nll_loss")

  @category("nonzero")
  def test_nonzero(self):
    self.do_test_op("nonzero")

  def test_normal_(self):
    self.do_test_op(
        "normal",
        # By definition, normal() returns a tensor with random values, so
        # there's no point in checking the values.
        check_value=CheckValueMode.SKIP,
        exclude_inplace_dtypes=(torch.float64,),
    )

  def test_ones(self):
    self.do_test_op("ones")

  def test_ones_like(self):
    self.do_test_op("ones_like")

  def test_permute(self):
    self.do_test_op("permute")

  def test_pow(self):
    self.do_test_op("pow")

  def test_nn_functional_conv1d(self):
    self.do_test_op(
        "nn.functional.conv1d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes.
        # Known issues:
        # 1. CPU sometimes fails with low-bitwidth integers, even though XLA
        #    succeeds; possible bug in CPU kernel?
        # 2. TPU lowering for int64 crashes due to "While rewriting computation
        #    to not contain X64 element types, XLA encountered an HLO for which
        #    this rewriting is not implemented: %convolution [...]"
        exclude_dtypes=COMPLEX_DTYPES
        + (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
    )

  def test_nn_functional_conv2d(self):
    self.do_test_op(
        "nn.functional.conv2d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes. See comments in test_nn_functional_conv1d.
        exclude_dtypes=COMPLEX_DTYPES
        + (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
    )

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  @oss_utils.skip_in_oss()
  def test_nn_functional_conv_transpose1d(self):
    self.do_test_op(
        "nn.functional.conv_transpose1d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes.
        # Known issues:
        # 1. CPU sometimes fails with low-bitwidth integers, even though XLA
        #    succeeds; possible bug in CPU kernel?
        # 2. TPU lowering for int64 crashes due to "While rewriting computation
        #    to not contain X64 element types, XLA encountered an HLO for which
        #    this rewriting is not implemented: %convolution [...]"
        exclude_dtypes=COMPLEX_DTYPES
        + (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
    )

  def test_nn_functional_conv_transpose2d(self):
    self.do_test_op(
        "nn.functional.conv_transpose2d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes.
        # Known issues:
        # 1. CPU sometimes fails with low-bitwidth integers, even though XLA
        #    succeeds; possible bug in CPU kernel?
        # 2. TPU lowering for int64 crashes due to "While rewriting computation
        #    to not contain X64 element types, XLA encountered an HLO for which
        #    this rewriting is not implemented: %convolution [...]"
        exclude_dtypes=COMPLEX_DTYPES
        + (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
    )

  # TODO(gleasonk): why does compilation time out on this input?
  #    input shape: (1, 1, 11, 12, 13),
  #                 created from (1, 1, 10, 11, 12) + pad (0, 0, 1, 1, 1)
  #    kernel shape: (1, 1, 4, 4, 4)
  #    strides: (1, 1, 1)
  #    padding: (4, 4, 4), symmetric
  #    dilation: (3, 3, 3)
  #    -> output shape: (1, 1, 10, 11, 12)
  # Timeout occurs with both XlaBuilder and StableHLO implementations.
  # def test_nn_functional_conv3d(self):
  #   self.do_test_op(
  #       "nn.functional.conv3d",
  #       # TODO: fix nn.functional.conv*d() failing with integral and complex
  #       # dtypes. See comments in test_nn_functional_conv1d.
  #       exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
  #   )

  def test_nn_functional_relu(self):
    self.do_test_op(
        "nn.functional.relu",
        # TODO: fix relu() succeeding with integral dtypes (it
        # should fail).  xla_cuda has no support for complex64::min()
        # xla_cuda: https://github.com/openxla/stablehlo/issues/560
        exclude_dtypes=(torch.uint8, torch.int8, torch.int16, torch.bool)
        + COMPLEX_DTYPES,
    )

  def test_nn_functional_rms_norm(self):
    self.do_test_op(
        "nn.functional.rms_norm",
        exclude_dtypes=INTEGRAL_DTYPES + COMPLEX_DTYPES,
    )

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  # TODO: b/476147793 association of (inputs, outputs) pairs with the op name
  # and dtype only causes comparison of outputs of different tests.
  @op_testing.skip_if_torch_tpu_vs_gpu_mode
  @oss_utils.skip_in_oss()
  def test_nn_functional_scaled_dot_product_attention_math(self):
    # Force the MATH backend for both CPU and TPU.
    with attention.sdpa_kernel(attention.SDPBackend.MATH):
      self.do_test_op(
          "nn.functional.scaled_dot_product_attention",
          exclude_dtypes=(torch.uint8, torch.int8, torch.int16),
      )

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  # TODO: b/476147793 association of (inputs, outputs) pairs with the op name
  # and dtype only causes comparison of outputs of different tests.
  @op_testing.skip_if_torch_tpu_vs_gpu_mode
  @oss_utils.skip_in_oss()
  def test_nn_functional_scaled_dot_product_attention_overrideable(self):
    # TODO: b/476166586 change the SDP backend for GPU device. It doesn't
    # support `OVERRIDEABLE`, so it fails unconditionally.
    #
    # Use OVERRIDEABLE backend for TPU, MATH for CPU.
    # Only do the forward pass (backward is not implemented yet)
    with attention.sdpa_kernel(
        [attention.SDPBackend.OVERRIDEABLE, attention.SDPBackend.MATH],
        set_priority=True,
    ):
      self.do_test_op(
          "nn.functional.scaled_dot_product_attention",
      )

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  # TODO: b/476147793 association of (inputs, outputs) pairs with the op name
  # and dtype only causes comparison of outputs of different tests.
  @op_testing.skip_if_torch_tpu_vs_gpu_mode
  @oss_utils.skip_in_oss()
  def test_nn_functional_scaled_dot_product_attention_efficient(self):
    # Use EFFICIENT_ATTENTION backend for TPU, and MATH for CPU.
    with attention.sdpa_kernel(
        [attention.SDPBackend.EFFICIENT_ATTENTION, attention.SDPBackend.MATH],
        set_priority=True,
    ):
      self.do_test_op(
          "nn.functional.scaled_dot_product_attention",
      )

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  # TODO: b/476147793 association of (inputs, outputs) pairs with the op name
  # and dtype only causes comparison of outputs of different tests.
  @op_testing.skip_if_torch_tpu_vs_gpu_mode
  @oss_utils.skip_in_oss()
  def test_nn_functional_scaled_dot_product_attention_flash(self):
    # Use FLASH_ATTENTION backend for TPU, and MATH for CPU.
    with attention.sdpa_kernel(
        [attention.SDPBackend.FLASH_ATTENTION, attention.SDPBackend.MATH],
        set_priority=True,
    ):
      self.do_test_op(
          "nn.functional.scaled_dot_product_attention",
      )

  def test_nn_functional_batch_norm(self):
    self.do_test_op("nn.functional.batch_norm")

  def test_nn_functional_elu(self):
    self.do_test_op("nn.functional.elu")

  def test_nn_functional_gelu(self):
    self.do_test_op("nn.functional.gelu")

  def test_nn_functional_glu(self):
    self.do_test_op("nn.functional.glu")

  def test_nn_functional_prelu(self):
    self.do_test_op(
        "nn.functional.prelu",
        exclude_dtypes=INTEGRAL_DTYPES,
        # supports floating point types
    )

  def test_nn_functional_hardsigmoid(self):
    self.do_test_op("nn.functional.hardsigmoid")

  def test_nn_functional_hardswish(self):
    self.do_test_op("nn.functional.hardswish")

  def test_nn_functional_hardtanh(self):
    self.do_test_op("nn.functional.hardtanh")

  def test_nn_functional_silu(self):
    self.do_test_op(
        "nn.functional.silu",
        # TODO: fix nn.functional.silu() succeeding with integral dtypes (it
        # should fail).
        # TODO: fix nn.functional.silu() failing with complex dtypes.
        exclude_dtypes=INTEGRAL_DTYPES + COMPLEX_DTYPES,
        # TODO: fix nn.functional.silu_() succeeding with integral dtypes (it
        # should fail).
        # TODO: fix nn.functional.silu_() failing with complex dtypes.
        exclude_inplace_dtypes=INTEGRAL_DTYPES + COMPLEX_DTYPES,
    )

  def test_nn_functional_softplus(self):
    self.do_test_op("nn.functional.softplus")

  def test_nn_functional_mse_loss(self):
    self.do_test_op("nn.functional.mse_loss")

  def test_pdist_forward(self):
    self.do_test_op("nn.functional.pdist")

  def test_polar(self):
    self.do_test_op(
        "polar",
        # TODO: fix polar() succeeding with these dtypes (it
        # should fail).
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        + COMPLEX_DTYPES
        + (torch.float64,),
    )

  def test_polygamma(self):
    self.do_test_op("polygamma")

  def test_prod(self):
    self.do_test_op("prod")

  def test_put(self):
    self.do_test_op("put")

  def test_randn(self):
    self.do_test_op(
        "randn",
        # TODO: fix randn(out=...) failing.
        check_out_variant=False,
        check_value=CheckValueMode.SKIP,  # randn() returns random values.
    )

  def test_randint(self):
    self.do_test_op("randint", check_value=CheckValueMode.SKIP)

  def test_reciprocal(self):
    self.do_test_op("reciprocal")

  def test_reflection_pad(self):
    self.do_test_op(
        "nn.functional.pad",
        variant_test_name="reflect",
        # TODO: bool dtype is not yet supported on TPU.
        exclude_dtypes={
            "gpu": (torch.bool,),
        },
    )

  def test_remainder(self):
    self.do_test_op(
        "remainder",
        # bfloat16 remainder is unstable at division boundaries: quotient
        # rounding flips shift the result by a full divisor vs CPU and GPU.
        exclude_dtypes=(torch.bfloat16,),
        exclude_inplace_dtypes=(torch.bfloat16,),
    )

  def test_repeat(self):
    self.do_test_op("repeat")

  def test_replication_pad(self):
    # TODO: Check why this is failing with torch.bool.
    self.do_test_op(
        "nn.functional.pad",
        variant_test_name="replicate",
        # TODO: bool dtype is not yet supported on TPU.
        exclude_dtypes={
            "gpu": (torch.bool,),
        },
    )

  def test_reshape(self):
    self.do_test_op("reshape")

  def test_resize_(self):
    self.do_test_op("resize_")

  def test_resolve_conj(self):
    self.do_test_op("resolve_conj")

  def test_resolve_neg(self):
    self.do_test_op("resolve_neg")

  def test_roll(self):
    self.do_test_op("roll")

  def test_round(self):
    self.do_test_op("round")

  def test_rsqrt(self):
    self.do_test_op("rsqrt")

  def test_rsub(self):
    self.do_test_op("rsub")

  def test_scaled_mm_v2(self):
    self.do_test_op(
        "torch._scaled_mm_v2",
        extra_dtypes=common_methods_invocations.float8_types(),
        exclude_dtypes=NUMERIC_DTYPES,
    )

  def test_scatter(self):
    self.do_test_op("scatter")

  def test_scatter_add(self):
    self.do_test_op("scatter_add")

  def test_scatter_reduce(self):
    self.do_test_op("scatter_reduce")

  def test_select(self):
    self.do_test_op("select")

  def test_select_scatter(self):
    self.do_test_op("select_scatter")

  def test_safe_softmax(self):
    self.do_test_op("torch.ops.aten._safe_softmax.default")

  def test_scalar_tensor(self):
    self.do_test_op("scalar_tensor")

  @category("searchsorted")
  def test_searchsorted(self):
    self.do_test_op(
        "searchsorted",
        exclude_dtypes=COMPLEX_DTYPES
        # and bool dtypes not supported.
        + (torch.bool,),
        # Upstream generates 288 samples per dtype; cap to prevent test
        # shard timeouts (10 samples per op dtype results in ~15m test time).
        max_samples_per_op_dtype=6,
    )

  def test_sigmoid(self):
    self.do_test_op("sigmoid")

  def test_sgn(self):
    self.do_test_op("sgn")

  def test_sign(self):
    self.do_test_op("sign")

  def test_signbit(self):
    self.do_test_op("signbit")

  def test_sin(self):
    self.do_test_op("sin")

  def test_sinh(self):
    self.do_test_op("sinh")

  def test_slice(self):
    self.do_test_op("slice")

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  @oss_utils.skip_in_oss()
  def test_softmax(self):
    self.do_test_op("softmax")

  def test_softmax_backward_data(self):
    self.do_test_op(
        "_softmax_backward_data",
        # TODO: fix the op for these dtypes.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )

  def test_split(self):
    self.do_test_op("split")

  def test_split_with_sizes(self):
    self.do_test_op("split_with_sizes")

  def test_split_with_sizes_copy(self):
    self.do_test_op(
        "split_with_sizes_copy",
        check_dynamism=False,  # b/483055194
    )

  def test_sqrt(self):
    self.do_test_op("sqrt")

  def test_squeeze(self):
    self.do_test_op("squeeze")

  def test_squeeze_copy(self):
    self.do_test_op("squeeze_copy")

  def test_sort(self):
    self.do_test_op(
        "sort",
        # sort() returns a (values, indices) tuple, where indices is
        # non-deterministic (as there might be duplicates in values).
        # Therefore we only check the values of the first output.
        check_value=[CheckValueMode.STRICT, CheckValueMode.SKIP],
        # TODO: fix sort() result not on TPU.
        check_device=False,
    )

  def test_sub(self):
    self.do_test_op("sub")

  def test_sum(self):
    self.do_test_op("sum")

  def test_stack(self):
    self.do_test_op("stack")

  def test_t(self):
    self.do_test_op("t")

  def test_take(self):
    self.do_test_op("take")

  def test_tan(self):
    self.do_test_op("tan")

  def test_tanh(self):
    self.do_test_op("tanh")

  def test_threshold(self):
    self.do_test_op("nn.functional.threshold")

  def test_to(self):
    self.do_test_op("to")

  def test_topk(self):
    # Skip the indices output in topk as torch doesn't specify the order of the
    # sorting when multiple indices have the same value.
    self.do_test_op(
        "topk",
        skip_output_indices=[1],
        # TODO: fix topk() result not on TPU.
        check_device=False,
    )

  def test_transpose(self):
    self.do_test_op("transpose")

  def test_tril(self):
    self.do_test_op("tril")

  def test_tril_indices(self):
    self.do_test_op(
        "tril_indices",
        # TODO: b/476115671 add support for other dtypes that GPU supports:
        # integers other than int32 and int64, floats and complex.
        exclude_dtypes={
            "gpu": (
                (torch.uint8, torch.int8, torch.int16)
                + FLOAT_DTYPES
                + COMPLEX_DTYPES
            ),
        },
        # TODO(b/542277009): add support for bool dtype when either `row` or
        # `col` is 0 (trivial case).
        skip_if=_tril_indices_bool_zero_arg_eager_gpu,
    )

  def test_triu(self):
    self.do_test_op("triu")

  def test_trunc(self):
    self.do_test_op("trunc")

  def test_unbind(self):
    self.do_test_op("unbind")

  def test_unfold(self):
    self.do_test_op("unfold")

  def test_unsqueeze(self):
    self.do_test_op("unsqueeze")

  def test_unsqueeze_copy(self):
    self.do_test_op("unsqueeze_copy")

  def test_uniform(self):
    self.do_test_op(
        "uniform",
        # By definition, uniform() returns a tensor with random values, so
        # there's no point in checking the values.
        check_value=CheckValueMode.SKIP,
    )

  def test_unsafe_view(self):
    self.do_test_op("torch.ops.aten._unsafe_view")

  def test_upsample_nearest(self):
    # TODO: The CPU side fails for complex dtypes and integers.
    self.do_test_op(
        "nn.functional.upsample_nearest",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )

  def test_upsample_bicubic2d(self):
    self.do_test_op(
        "nn.functional.interpolate",
        variant_test_name="bicubic",
        exclude_dtypes=(COMPLEX_DTYPES + INTEGRAL_DTYPES),
    )

  def test_upsample_bilinear(self):
    # TODO: The CPU side fails for complex dtypes and integers.
    self.do_test_op(
        "nn.functional.upsample_bilinear",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # TODO: STRICT fails for some types. Look into narrowing this down.
    )

  def test_upsample_nearest_exact(self):
    # TODO: The CPU side fails for complex dtypes and integers.
    self.do_test_op(
        "nn.functional.interpolate",
        variant_test_name="nearest-exact",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )

  def test_var(self):
    self.do_test_op("var")

  def test_var_mean(self):
    self.do_test_op("var_mean")

  def test_vdot(self):
    self.do_test_op(
        "vdot",
        # GPU (CUDA) does not support integral dtypes for vdot.
        exclude_dtypes={
            "gpu": (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ),
        },
    )

  def test_view(self):
    self.do_test_op("view")

  def test_view_as_real(self):
    self.do_test_op("view_as_real")

  def test_view_as_complex(self):
    self.do_test_op(
        "view_as_complex",
        # XLA only allows creating complex dtypes from f32 and f64.
        # TODO: However, the lowering for converting f64 to complex<f64> isn't
        # implemented yet, which needs further investigation.
        exclude_dtypes=(
            torch.float16,
            torch.float64,
        ),
    )

  def test_where(self):
    self.do_test_op("where")

  # TODO(b/535650392): Re-enable this testin OS once the bug is fixed.
  @oss_utils.skip_in_oss()
  def test_xlogy(self):
    self.do_test_op("xlogy")

  def test_zeros(self):
    self.do_test_op("zeros")

  def test_zero_(self):
    self.do_test_op("zero_")

  def test_zeros_like(self):
    self.do_test_op("zeros_like")

  def test_bucketize(self):
    self.do_test_op(
        "bucketize",
        exclude_dtypes={
            # - No total order for complex numbers.
            # - Inconsistent behaviour on CPU with bool.
            "cpu": COMPLEX_DTYPES + (torch.bool,),
            "gpu": COMPLEX_DTYPES + (torch.bool,),
        },
    )

  # geqrf testing isn't currently supported by PyTorch in other modes due to
  # certain operations not being implemented on CPU/GPU.
  @op_testing.skip_unless_torch_tpu_vs_gpu_mode
  def test_geqrf(self):
    self.do_test_op(
        "geqrf",
        exclude_dtypes=(
            torch.complex64,  # Cannot generate GPU sample for this dtype.
            torch.float64,  # Cannot generate GPU sample for this dtype.
            torch.float32,  # Cannot generate GPU sample for this dtype.
            torch.float16,  # Cannot generate GPU sample for this dtype.
            torch.bfloat16,  # Cannot generate GPU sample for this dtype.
            torch.uint8,  # Cannot generate GPU sample for this dtype.
            torch.int8,  # Cannot generate GPU sample for this dtype.
            torch.int16,  # Cannot generate GPU sample for this dtype.
            torch.int32,  # Cannot generate GPU sample for this dtype.
            torch.int64,  # Cannot generate GPU sample for this dtype.
            torch.bool,  # Cannot generate GPU sample for this dtype.
        ),
    )

  # qr testing isn't currently supported by PyTorch in other modes due to
  # certain operations not being implemented on CPU/GPU.
  @op_testing.skip_unless_torch_tpu_vs_gpu_mode
  def test_linalg_qr(self):
    self.do_test_op(
        "linalg.qr",
        exclude_dtypes=(
            torch.complex64,  # Cannot generate GPU sample for this dtype.
            torch.float64,  # Cannot generate GPU sample for this dtype.
            torch.float32,  # Cannot generate GPU sample for this dtype.
            torch.float16,  # Cannot generate GPU sample for this dtype.
            torch.bfloat16,  # Cannot generate GPU sample for this dtype.
            torch.uint8,  # Cannot generate GPU sample for this dtype.
            torch.int8,  # Cannot generate GPU sample for this dtype.
            torch.int16,  # Cannot generate GPU sample for this dtype.
            torch.int32,  # Cannot generate GPU sample for this dtype.
            torch.int64,  # Cannot generate GPU sample for this dtype.
            torch.bool,  # Cannot generate GPU sample for this dtype.
        ),
    )

  def test_thnn_fused_gru_cell(self):
    self.do_test_op(
        "_thnn_fused_gru_cell",
        # TODO(b/540346402): succeeded on TPU but failed on GPU for complex64.
        # CUDA kernel is not implemented for this dtype.
        exclude_dtypes={
            "gpu": (torch.complex64,),
        },
        check_dynamism=False,
    )

  def test_thnn_fused_lstm_cell(self):
    self.do_test_op(
        "_thnn_fused_lstm_cell",
        # TODO(b/540345869): succeeded on TPU but failed on GPU for complex64.
        # CUDA kernel is not implemented for this dtype.
        exclude_dtypes=(
            _if_tpu_vs_gpu_compiled(
                (torch.complex64,) + FLOAT_DTYPES,
                (torch.complex64,),
            )
        ),
        check_dynamism=False,
    )


def setUpModule() -> None:
  """Called by absltest.main() after flags are parsed but before tests are run."""

  op_testing.set_up_test_module()

  if (
      op_testing._torch_tpu_vs_gpu_mode() or op_testing._gen_gpu_golden_mode()
  ) and op_testing.is_compiled_mode():
    assert torch.backends.tpu.allow_excess_precision  # pytype: disable=module-attr
  else:
    assert not torch.backends.tpu.allow_excess_precision  # pytype: disable=module-attr


def tearDownModule() -> None:
  """Called by absltest.main() after running all tests."""

  op_testing.tear_down_test_module()


if __name__ == "__main__":
  absltest.main()
