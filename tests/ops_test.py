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

For each op, we test that the result on TPU matches the golden result recorded
on GPU.
"""

import collections.abc
import copy
import functools
import os
import sys
from typing import Any

from absl import flags
from absl.testing import absltest
import torch
from torch.nn import attention
from torch_tpu._internal.utils import test_utils
from tests import op_testing
from tests import oss_utils
from tests import quantize_utils

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


# The attribute that `also_run_static` sets on the tests it marks.
_ALSO_RUN_STATIC_ATTR = "also_run_static"

_UNBOUNDED_DYNAMIC_KERNELS_ENV_VAR = (
    "TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS"
)


def _unbounded_dynamic_kernels_disabled() -> bool:
  """Returns whether the env var disables unbounded dynamic kernels.

  Mirrors the runtime, which disables them only if the variable is "none".
  """
  return os.environ.get(_UNBOUNDED_DYNAMIC_KERNELS_ENV_VAR, "") == "none"


def _unbounded_dynamic_hits() -> int:
  """Returns the number of lookups served by unbounded dynamic kernels."""
  return torch.tpu._get_cache_stats().num_unbounded_dynamic_hits  # pylint: disable=protected-access


def also_run_static(func):
  """Also runs the test in ops_test_static, asserting no dynamic kernels ran.

  In the other targets, which load the dynamic kernels enabled by default, the
  test fails if no dynamic kernel served any of its samples, so that the
  kernels can't silently lose their coverage.

  Args:
    func: The test method of an op with dynamic kernels enabled by default.

  Returns:
    The wrapped test method.
  """

  @functools.wraps(func)
  def wrapper(self, *args, **kwargs):
    # Samples only run on the TPU if there is one, e.g. not when recording GPU
    # goldens or dtype exclusions.
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or accelerator.type != "tpu":
      return func(self, *args, **kwargs)
    hits_before = _unbounded_dynamic_hits()
    result = func(self, *args, **kwargs)
    torch.accelerator.synchronize()
    hits = _unbounded_dynamic_hits() - hits_before
    if _unbounded_dynamic_kernels_disabled():
      self.assertEqual(
          hits,
          0,
          "Lookups served by unbounded dynamic kernels, although"
          f" {_UNBOUNDED_DYNAMIC_KERNELS_ENV_VAR} disables them.",
      )
    # The served samples may be filtered out by the dtypes or the sample limit.
    elif self._subtests_run and op_testing.runs_all_samples():
      self.assertGreater(
          hits,
          0,
          f"No unbounded dynamic kernel served {self._testMethodName}, which"
          " is marked with @also_run_static: its dynamic kernels have no"
          " coverage.",
      )
    return result

  setattr(wrapper, _ALSO_RUN_STATIC_ATTR, True)
  return wrapper


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
#   (GPU)
# - tpu_*: a value for the TPU device

COMPLEX_DTYPES = op_testing.COMPLEX_DTYPES
FLOAT_DTYPES = op_testing.FLOAT_DTYPES
INTEGRAL_DTYPES = op_testing.INTEGRAL_DTYPES
_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
UNQUANTIZED_NUMERIC_DTYPES = op_testing.UNQUANTIZED_NUMERIC_DTYPES
ALL_NUMERIC_DTYPES = op_testing.ALL_NUMERIC_DTYPES
_SPARSE_COO_EXCLUDE_DTYPES = tuple(
    set(ALL_NUMERIC_DTYPES) - set(FLOAT_DTYPES) - set(INTEGRAL_DTYPES)
)

CheckValueMode = test_utils.CheckValueMode

# OpInput and OpOutput are not used directly in this file, but needed for
# pickling the golden results.
OpInput = op_testing.OpInput
OpOutput = op_testing.OpOutput
OpVariant = op_testing.OpVariant


def update_dict(d, u):
  """Recursively update a dictionary to override tolerance values."""
  for k, v in u.items():
    if isinstance(v, collections.abc.Mapping):
      d[k] = update_dict(d.get(k, {}), v)
    else:
      d[k] = v
  return d


# Overrides the default relative and absolute tolerance for the given op and
# dtype between TPU and GPU. Two tensors are considered to match if
#
#   abs(tpu - gpu) <= atol + rtol * abs(gpu)
#
# (A common misconception is that the criteria is
#   abs(tpu - gpu) <= min(atol, rtol * abs(gpu))
# but that's incorrect.)
#
# The overrides apply to both the op and its inplace variant.
#
# The default tolerances are defined here:
#   https://docs.pytorch.org/docs/stable/testing.html
ACCURACY_OVERRIDES_VS_GPU = {
    # go/keep-sorted start
    "_addmm_activation": {
        torch.bfloat16: {"rtol": 1.7e-2, "atol": 1.5e-2},
        torch.float16: {"rtol": 1.6e-1, "atol": 5.4e-1},
        torch.float32: {"rtol": 5.5e-2, "atol": 4.3e-2},
        torch.float64: {"rtol": 1.5e-4, "atol": 5.7e-4},
    },
    "_foreach_acos": {
        torch.complex64: {"rtol": 5.2e-5, "atol": 6.7e-5},
    },
    "_foreach_cos": {
        torch.complex64: {"rtol": 5.5e-6},
    },
    "_foreach_cosh": {
        torch.complex64: {"rtol": 5.1e-6},
        torch.float32: {"rtol": 5e-6},
    },
    "_foreach_exp": {
        torch.complex64: {"rtol": 4.8e-6, "atol": 2.8e-6},
        torch.int16: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 4.4e-6, "atol": 7.9e-7},
    },
    "_foreach_expm1": {
        torch.complex64: {"rtol": 6.1e-6},
    },
    "_foreach_log": {
        torch.complex64: {"rtol": 6.3e-5, "atol": 6.8e-5},
        torch.float32: {"rtol": 7.4e-5, "atol": 8.3e-5},
    },
    "_foreach_log10": {
        torch.complex64: {"rtol": 3.9e-5, "atol": 5.3e-5},
        torch.float32: {"rtol": 3.3e-5, "atol": 4.2e-5},
    },
    "_foreach_log2": {
        torch.complex64: {"rtol": 1.1e-4, "atol": 3.4e-5},
        torch.float32: {"rtol": 1.1e-4, "atol": 8.1e-5},
    },
    "_foreach_pow": {
        torch.complex64: {"rtol": 8.8e-4, "atol": 3.4e-4},
        torch.float32: {"rtol": 5.5e-6},
    },
    "_foreach_sigmoid": {
        torch.complex64: {"rtol": 1.4e-5},
        torch.float32: {"atol": 2.7e-5},
    },
    "_foreach_tan": {
        torch.complex64: {"rtol": 3.7e-5},
    },
    "_foreach_tanh": {
        torch.complex64: {"rtol": 1.4e-4, "atol": 1.1e-5},
        torch.float32: {"atol": 5.2e-5},
    },
    "_grouped_mm": {
        torch.float16: {"rtol": 1.1e-2, "atol": 1.2e-2},
        torch.float32: {"rtol": 8.6e-3, "atol": 1.3e-2},
    },
    "_log_softmax_backward_data": {
        torch.float32: {"rtol": 3.5e-5, "atol": 3.6e-5},
    },
    "_masked_softmax_backward": {
        torch.bfloat16: {"atol": 3e-3},
        torch.float16: {"atol": 5.9e-4},
    },
    "_native_batch_norm_legit": {
        torch.float32: {"rtol": 1e-5, "atol": 2.1e-5},
    },
    "_native_multi_head_attention": {
        torch.float16: {"rtol": 1.4e-1, "atol": 1.3e-1},
        torch.float32: {"rtol": 2.0e-1, "atol": 3.2e-1},
    },
    "_softmax_backward_data": {
        torch.bfloat16: {"atol": 3.6e-3},
        torch.float16: {"atol": 5.9e-4},
    },
    "_transformer_encoder_layer_fwd": {
        torch.bfloat16: {"rtol": 4.5e-2, "atol": 1.1e-1},
        torch.float16: {"rtol": 7.6e-2, "atol": 1.8e-1},
        torch.float32: {"rtol": 7.0e-2, "atol": 2.1e-1},
    },
    "_upsample_bicubic2d_aa": {
        torch.bfloat16: {"rtol": 3.5e-2, "atol": 3.3e-2},
        torch.float16: {"rtol": 3.8e-3, "atol": 4e-3},
        torch.float32: {"rtol": 2.0e-5, "atol": 2.0e-5},
    },
    "_upsample_bilinear2d_aa": {
        torch.bfloat16: {"rtol": 4.7e-2, "atol": 3.7e-2},
        torch.float16: {"rtol": 6.9e-3, "atol": 6.5e-3},
        torch.float32: {"rtol": 1.7e-5, "atol": 2.5e-5},
    },
    "abs": {
        torch.complex64: {"rtol": 2.0e-6, "atol": 2.0e-6},
        torch.float64: {"rtol": 4.7e-6, "atol": 1e-6},
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
    "addmm": {
        torch.float16: {"rtol": 1.5e-2},
        torch.float32: {"rtol": 7.7e-3},
    },
    "addmm.dtype": {
        torch.bfloat16: {"atol": 6.1e-2},
        torch.float16: {"rtol": 2.9e-2, "atol": 9.3e-2},
        torch.float32: {"rtol": 2.8e-2, "atol": 8e-2},
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
    "baddbmm.dtype": {
        torch.complex64: {"rtol": 1.5e-1},
        torch.float16: {"rtol": 6e-2, "atol": 1.2e-2},
        torch.float32: {"rtol": 7.7e-2, "atol": 8.4e-3},
    },
    "bmm": {
        torch.complex64: {"rtol": 9.6e-2},
        torch.float16: {"rtol": 1.1e-1, "atol": 3.5e-1},
        torch.float32: {"rtol": 6.6e-2, "atol": 3.4e-1},
    },
    "bmm.dtype": {
        torch.complex64: {"rtol": 3.5e-2, "atol": 2e-2},
        torch.float16: {"rtol": 1e-1, "atol": 8.2e-2},
        torch.float32: {"rtol": 5.4e-2, "atol": 7.4e-2},
    },
    "cdist": {
        torch.bfloat16: {"atol": 2.1e-1},
        torch.float16: {"rtol": 2e-1, "atol": 3e-1},
        torch.float32: {"rtol": 5.1e-1, "atol": 3e-1},
    },
    "col2im": {
        torch.complex64: {"rtol": 2.3e-2},
        torch.float16: {"rtol": 2.1e-2, "atol": 4.2e-2},
        torch.float32: {"rtol": 1.8e-2, "atol": 3.4e-2},
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
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 3.5e-3, "atol": 2.4e-3},
    },
    "digamma": {
        torch.float32: {"rtol": 4e-5, "atol": 1.3e-5},
        torch.int16: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int32: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int64: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int8: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.uint8: {"rtol": 4.7e-5, "atol": 3.6e-5},
    },
    "einsum": {
        torch.bfloat16: {"rtol": 3.9e-2},
        torch.complex64: {"rtol": 2.7e-2},
        torch.float16: {"rtol": 1.4e-2, "atol": 1.5e-1},
        torch.float32: {"rtol": 1.1e-2, "atol": 1.6e-1},
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
        torch.complex64: {"rtol": 3.3e-6},
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
    "floor_divide": {
        torch.bfloat16: {"rtol": 2.1e-1},
    },
    "gru.input": {
        torch.bfloat16: {"atol": 1.2e-3},
        torch.float16: {"atol": 1.1e-3},
        torch.float32: {"atol": 9.3e-4},
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
        torch.float64: {"atol": 2.5e-6},
    },
    "linalg.lu_factor_ex": {
        torch.float32: {"rtol": 2.5e-6, "atol": 2.4e-5},
    },
    "linalg.qr": {
        torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    },
    "linalg.svdvals": {
        torch.complex64: {"rtol": 5.6e-3},
        torch.float32: {"rtol": 4.9e-3},
    },
    "linalg.vector_norm": {
        torch.complex64: {"rtol": 4.8e-6},
        torch.float32: {"rtol": 4.8e-6},
    },
    "linspace": {
        torch.bfloat16: {"rtol": 9.2e-3, "atol": 9.4e-3},
        torch.float16: {"rtol": 1.2e-3, "atol": 1.8e-3},
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
        torch.float32: {"rtol": 5.7e-5, "atol": 8.3e-5},
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
    "lstm.input": {
        torch.bfloat16: {"atol": 1.2e-3},
        torch.float16: {"atol": 3.7e-4},
        torch.float32: {"atol": 4e-4},
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
    "mm.dtype": {
        torch.complex64: {"rtol": 3.6e-2},
        torch.float16: {"rtol": 2.9e-2, "atol": 4.2e-3},
        torch.float32: {"rtol": 4.9e-2, "atol": 2.9e-3},
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
    "nn.functional.conv3d": {
        torch.float16: {"rtol": 6.5e-1, "atol": 6.6e-1},
        torch.float32: {"rtol": 5e-1, "atol": 7e-1},
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
    "nn.functional.conv_transpose3d": {
        torch.bfloat16: {"rtol": 2.2e-1, "atol": 2.3e-1},
        torch.float16: {"rtol": 1.1, "atol": 8.2e-1},
        torch.float32: {"rtol": 6.2e-1, "atol": 1.1},
    },
    "nn.functional.embedding": {
        torch.float16: {
            "rtol": 3.9e-3,
            "atol": 2.4e-3,
        },
    },
    "nn.functional.embedding_bag": {
        torch.bfloat16: {"atol": 3.2e-2},
        torch.float16: {"atol": 4e-3},
    },
    "nn.functional.group_norm": {
        torch.bfloat16: {"rtol": 3.3e-2, "atol": 8e-2},
    },
    "nn.functional.linear": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.complex64: {"rtol": 2.2e-2, "atol": 3.5e-2},
        torch.float16: {"rtol": 1.9e-2, "atol": 1.5e-2},
        torch.float32: {"rtol": 2.0e-2, "atol": 1.7e-2},
    },
    "nn.functional.scaled_dot_product_attention": {
        torch.bfloat16: {"rtol": 3.3e-1, "atol": 4e-1},
        torch.float16: {"rtol": 5e-1, "atol": 5.4e-1},
        torch.float32: {"rtol": 5.6e-1, "atol": 5.6e-1},
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
        torch.complex64: {"rtol": 3.0e-5, "atol": 5.0e-6},
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
    "torch._scaled_grouped_mm": {
        torch.float8_e4m3fn: {"rtol": 9.4e-3, "atol": 4.7e-3},
    },
    "torch._scaled_mm": {
        torch.float8_e4m3fn: {"rtol": 9.4e-3, "atol": 4.7e-3},
    },
    "torch.ops.aten._safe_softmax.default": {
        torch.float16: {"atol": 5.9e-4},
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

# Like ACCURACY_OVERRIDES_VS_GPU, but with torch.compile().
ACCURACY_OVERRIDES_VS_GPU_COMPILED = {
    # go/keep-sorted start
    "_addmm_activation": {
        torch.float16: {"rtol": 4.2e-3, "atol": 4.7e-3},
        torch.float32: {"rtol": 3.6e-3, "atol": 4.1e-3},
        torch.float64: {"atol": 1.8e-4},
    },
    "_foreach_acos": {
        torch.complex64: {"rtol": 5e-5, "atol": 3e-5},
    },
    "_foreach_asin": {
        torch.complex64: {"rtol": 6e-5, "atol": 6.1e-5},
    },
    "_foreach_atan": {
        torch.complex64: {"rtol": 1.9e-5, "atol": 1.5e-5},
    },
    "_foreach_cos": {
        torch.complex64: {"rtol": 4.7e-6},
    },
    "_foreach_cosh": {
        torch.complex64: {"rtol": 5.1e-6},
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
    "_foreach_lerp": {
        torch.bfloat16: {"rtol": 3.5e-2, "atol": 7.3e-2},
        torch.float16: {"rtol": 4.2e-3, "atol": 8.7e-3},
    },
    "_foreach_lgamma": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"rtol": 1.2e-3, "atol": 5.9e-4},
        torch.float32: {"rtol": 2.8e-4, "atol": 3.3e-4},
        torch.int16: {"rtol": 1.5e-4, "atol": 6.3e-5},
        torch.int32: {"rtol": 1.5e-4, "atol": 6.3e-5},
        torch.int64: {"rtol": 1.5e-4, "atol": 6.3e-5},
        torch.int8: {"rtol": 1.5e-4, "atol": 6.3e-5},
        torch.uint8: {"rtol": 1.5e-4, "atol": 6.3e-5},
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
        torch.complex64: {"rtol": 8.8e-4, "atol": 4.1e-4},
        torch.float32: {"rtol": 5.1e-6},
    },
    "_foreach_sigmoid": {
        torch.complex64: {"rtol": 1.4e-5},
    },
    "_foreach_sin": {
        torch.complex64: {"rtol": 4.5e-6},
    },
    "_foreach_sinh": {
        torch.complex64: {"rtol": 5.6e-6},
    },
    "_foreach_tan": {
        torch.complex64: {"rtol": 3.1e-5},
    },
    "_foreach_tanh": {
        torch.complex64: {"rtol": 6.2e-6},
    },
    "_grouped_mm": {
        torch.float16: {"rtol": 4e-3, "atol": 9.4e-3},
        torch.float32: {"rtol": 3.5e-3, "atol": 9e-3},
    },
    "_log_softmax_backward_data": {
        torch.float32: {"rtol": 3.5e-5, "atol": 3.7e-5},
    },
    "_masked_softmax_backward": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"rtol": 1.1e-3, "atol": 5.9e-4},
    },
    "_native_batch_norm_legit": {
        torch.float32: {"rtol": 2.1e-6, "atol": 2.2e-5},
    },
    "_native_multi_head_attention": {
        torch.bfloat16: {"rtol": 3.7e-2, "atol": 4.7e-2},
        torch.float16: {"rtol": 1.6e-1, "atol": 1.7e-1},
        torch.float32: {"rtol": 2e-1, "atol": 3.2e-1},
    },
    "_segment_reduce": {
        torch.bfloat16: {"rtol": 1.7e-2, "atol": 3.8e-2},
        torch.float16: {"rtol": 1.6e-3, "atol": 2.1e-3},
    },
    "_transformer_encoder_layer_fwd": {
        torch.bfloat16: {"rtol": 4.5e-2, "atol": 1.1e-1},
        torch.float16: {"rtol": 7.6e-2, "atol": 1.8e-1},
        torch.float32: {"rtol": 7e-2, "atol": 2.1e-1},
    },
    "_upsample_bicubic2d_aa": {
        torch.bfloat16: {"rtol": 3.5e-2, "atol": 3.3e-2},
        torch.float16: {"rtol": 3.8e-3, "atol": 4e-3},
        torch.float32: {"rtol": 2.1e-5, "atol": 2.4e-5},
    },
    "_upsample_bilinear2d_aa": {
        torch.bfloat16: {"rtol": 4.7e-2, "atol": 3.7e-2},
        torch.float16: {"rtol": 6.9e-3, "atol": 6.5e-3},
        torch.float32: {"rtol": 1.7e-5, "atol": 2.5e-5},
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
    "addcmul": {
        torch.float16: {"rtol": 1.4e-3, "atol": 1.2e-3},
    },
    "addmm": {
        torch.float16: {"rtol": 1.4e-2},
        torch.float32: {"rtol": 7.7e-3},
    },
    "addmm.dtype": {
        torch.float16: {"rtol": 5.1e-3, "atol": 2.5e-3},
        torch.float32: {"rtol": 4.6e-3, "atol": 3.8e-3},
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
        torch.bfloat16: {"atol": 4.7e-3},
        torch.complex64: {"rtol": 7.7e-2},
        torch.float16: {"rtol": 2.8e-1, "atol": 3.3e-1},
        torch.float32: {"rtol": 3.4e-1, "atol": 3.6e-1},
    },
    "baddbmm.dtype": {
        torch.complex64: {"rtol": 1.1e-2, "atol": 6e-3},
        torch.float16: {"rtol": 4.7e-3, "atol": 7.7e-3},
        torch.float32: {"rtol": 4.4e-3, "atol": 6.8e-3},
    },
    "bmm": {
        torch.complex64: {"rtol": 9.6e-2},
        torch.float16: {"rtol": 1.1e-1, "atol": 3.5e-1},
        torch.float32: {"rtol": 6.6e-2, "atol": 3.4e-1},
    },
    "bmm.dtype": {
        torch.complex64: {"rtol": 2.3e-2, "atol": 2e-2},
        torch.float16: {"rtol": 6.1e-3, "atol": 8e-3},
        torch.float32: {"rtol": 7.1e-3, "atol": 1.2e-2},
    },
    "cdist": {
        torch.float32: {"rtol": 5.1e-6},
    },
    "col2im": {
        torch.complex64: {"rtol": 5.8e-3, "atol": 6.1e-3},
        torch.float16: {"rtol": 1.9e-3, "atol": 2.4e-3},
        torch.float32: {"rtol": 1.4e-3, "atol": 2.4e-3},
    },
    "cos": {
        torch.complex64: {"rtol": 4.6e-6},
    },
    "cosh": {
        torch.complex64: {"rtol": 3.9e-6},
    },
    "cumsum": {
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 3.5e-3, "atol": 2.4e-3},
    },
    "digamma": {
        torch.float32: {"rtol": 4e-5, "atol": 1.3e-5},
        torch.int16: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int32: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int64: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int8: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.uint8: {"rtol": 4.7e-5, "atol": 3.6e-5},
    },
    "einsum": {
        torch.complex64: {"rtol": 2.7e-2},
        torch.float16: {"rtol": 1.4e-2, "atol": 1.5e-1},
        torch.float32: {"rtol": 1.1e-2, "atol": 1.6e-1},
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
        torch.complex64: {"rtol": 3.3e-6},
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
    "fft.fft": {
        torch.int16: {"rtol": 1.8e-5, "atol": 1.9e-5},
    },
    "fft.ifft": {
        torch.int16: {"rtol": 6.9e-6, "atol": 6.9e-6},
    },
    "fft.rfft": {
        torch.int16: {"rtol": 2e-5, "atol": 9.1e-6},
    },
    "floor_divide": {
        torch.bfloat16: {"rtol": 6.1e-2, "atol": 1.3},
    },
    "gru.input": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"atol": 5.9e-4},
        torch.float32: {"atol": 9.1e-4},
    },
    "ldexp": {
        torch.complex64: {"rtol": 4.4e-6},
        torch.float32: {"rtol": 4.1e-6},
    },
    "lgamma": {
        torch.float16: {"atol": 1.8e-4},
        torch.float32: {"rtol": 1.7e-4, "atol": 1.8e-4},
        torch.int16: {"rtol": 1.5e-4, "atol": 6.3e-5},
        torch.int32: {"rtol": 1.5e-4, "atol": 6.3e-5},
        torch.int64: {"rtol": 1.5e-4, "atol": 6.3e-5},
        torch.int8: {"rtol": 1.5e-4, "atol": 6.3e-5},
        torch.uint8: {"rtol": 1.5e-4, "atol": 6.3e-5},
    },
    "linalg.lu_factor_ex": {
        torch.float32: {"rtol": 2.5e-6, "atol": 2.4e-5},
    },
    "linalg.svd": {
        torch.complex64: {"rtol": 5.1e-3, "atol": 3.9e-3},
        torch.float32: {"rtol": 5.6e-3, "atol": 2.3e-3},
    },
    "linalg.svdvals": {
        torch.complex64: {"rtol": 6.4e-3},
        torch.float32: {"rtol": 4.9e-3},
    },
    "linalg.vector_norm": {
        torch.complex64: {"rtol": 2.5e-6},
        torch.float32: {"rtol": 3.9e-6},
    },
    "linspace": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"atol": 5.9e-4},
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
        torch.float32: {"rtol": 5.7e-5, "atol": 8.3e-5},
    },
    "logcumsumexp": {
        torch.bfloat16: {"atol": 3.7e-5},
        torch.float16: {"rtol": 1.2e-3, "atol": 5.9e-4},
        torch.float32: {"rtol": 5e-5, "atol": 3.2e-5},
    },
    "logit": {
        torch.float32: {"rtol": 7.5e-5, "atol": 8.3e-5},
        torch.int16: {"atol": 7.5e-5},
        torch.int32: {"atol": 7.5e-5},
        torch.int64: {"atol": 7.5e-5},
        torch.int8: {"atol": 7.5e-5},
        torch.uint8: {"atol": 7.5e-5},
    },
    "lstm.input": {
        torch.bfloat16: {"atol": 1.2e-3},
        torch.float16: {"atol": 3e-4},
        torch.float32: {"atol": 1.4e-4},
    },
    "matmul": {
        torch.complex64: {"rtol": 1.4e-1},
        torch.float16: {"rtol": 3e-1, "atol": 4.7e-1},
        torch.float32: {"rtol": 2.7e-1, "atol": 3.9e-1},
    },
    "mm": {
        torch.complex64: {"rtol": 3.8e-2},
        torch.float16: {"rtol": 4.8e-2},
        torch.float32: {"rtol": 5.2e-2},
    },
    "mm.dtype": {
        torch.complex64: {"rtol": 8.5e-3, "atol": 1.1e-2},
        torch.float16: {"rtol": 2.1e-3, "atol": 4.2e-3},
        torch.float32: {"rtol": 7.2e-4, "atol": 2.9e-3},
    },
    "nn.functional.binary_cross_entropy": {
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
    "nn.functional.conv3d": {
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 6.5e-1, "atol": 6.6e-1},
        torch.float32: {"rtol": 5e-1, "atol": 7e-1},
    },
    "nn.functional.conv_transpose1d": {
        torch.bfloat16: {"atol": 8.3e-3},
        torch.float16: {"rtol": 6.7e-2, "atol": 1.6e-1},
        torch.float32: {"rtol": 7.1e-2, "atol": 2.1e-1},
    },
    "nn.functional.conv_transpose2d": {
        torch.bfloat16: {"rtol": 6.2e-2, "atol": 8e-2},
        torch.float16: {"rtol": 2.2e-1, "atol": 6.7e-2},
        torch.float32: {"rtol": 3.2e-1, "atol": 2.1e-1},
    },
    "nn.functional.conv_transpose3d": {
        torch.bfloat16: {"rtol": 2.4e-1, "atol": 2.3e-1},
        torch.float16: {"rtol": 1.1, "atol": 8.2e-1},
        torch.float32: {"rtol": 6.2e-1, "atol": 1.1},
    },
    "nn.functional.embedding": {
        torch.float16: {"rtol": 3.9e-3, "atol": 3e-3},
    },
    "nn.functional.embedding_bag": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"atol": 1.2e-3},
    },
    "nn.functional.group_norm": {
        torch.bfloat16: {"rtol": 2e-2, "atol": 5.7e-2},
    },
    "nn.functional.linear": {
        torch.bfloat16: {"atol": 4.7e-3},
        torch.float16: {"rtol": 1.9e-2, "atol": 1.5e-2},
        torch.float32: {"rtol": 2e-2, "atol": 1.7e-2},
    },
    "norm": {
        torch.complex64: {"rtol": 5.7e-6},
        torch.float32: {"rtol": 4.9e-6},
    },
    "polygamma": {
        torch.float32: {"rtol": 1.1e-5},
    },
    "pow": {
        torch.complex64: {"rtol": 7e-4, "atol": 3.8e-4},
        torch.float32: {"rtol": 5.6e-6},
    },
    "sin": {
        torch.complex64: {"rtol": 5.3e-6},
    },
    "sinh": {
        torch.complex64: {"rtol": 4.6e-6},
    },
    "torch._scaled_mm": {
        torch.float8_e4m3fn: {"rtol": 9.4e-3, "atol": 4.7e-3},
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
}  # end of ACCURACY_OVERRIDES_VS_GPU_COMPILED
_LIBTPU_VERSION = test_utils.libtpu_version()
if (
    oss_utils.is_oss()
    and _LIBTPU_VERSION is not None
    and _LIBTPU_VERSION < (0, 0, 51)
):
  # libtpu < 0.0.51 predates XLA's float32 Cody-Waite sinh/cosh precision
  # improvements (and complex64 tan lowering depends on sinh/cosh). Keep the
  # tightened head tolerances for internal runs and libtpu >= 0.0.51.
  # _foreach_cosh integer dtypes and _foreach_sinh fall back per-dtype to
  # cosh/sinh in op_testing.py.
  # TODO(b/584809741): Remove once OSS pins libtpu >= 0.0.51.
  update_dict(
      ACCURACY_OVERRIDES_VS_GPU_COMPILED,
      {
          # go/keep-sorted start
          "_foreach_cosh": {
              torch.float32: {"rtol": 5e-6},
              torch.int16: {"rtol": 4e-6},
              torch.int32: {"rtol": 4e-6},
              torch.int64: {"rtol": 4e-6},
              torch.int8: {"rtol": 4e-6},
              torch.uint8: {"rtol": 4e-6},
          },
          "_foreach_sinh": {
              torch.float32: {"rtol": 5.8e-6},
              torch.int16: {"rtol": 4e-6},
              torch.int32: {"rtol": 4e-6},
              torch.int64: {"rtol": 4e-6},
              torch.int8: {"rtol": 4e-6},
              torch.uint8: {"rtol": 4e-6},
          },
          "cosh": {
              torch.float32: {"rtol": 3.4e-6},
              torch.int16: {"rtol": 4e-6},
              torch.int32: {"rtol": 4e-6},
              torch.int64: {"rtol": 4e-6},
              torch.int8: {"rtol": 4e-6},
              torch.uint8: {"rtol": 4e-6},
          },
          "sinh": {
              torch.float32: {"rtol": 5.8e-6},
              torch.int16: {"rtol": 4e-6},
              torch.int32: {"rtol": 4e-6},
              torch.int64: {"rtol": 4e-6},
              torch.int8: {"rtol": 4e-6},
              torch.uint8: {"rtol": 4e-6},
          },
          "tan": {
              torch.complex64: {"rtol": 3.0e-5},
          },
          # go/keep-sorted end
      },
  )

# The GPU gradient tolerances are based on the TPU vs GPU forward pass tolerances.
ACCURACY_OVERRIDES_GRAD_VS_GPU: dict[
    str, dict[torch.dtype, dict[str, float]]
] = update_dict(
    copy.deepcopy(ACCURACY_OVERRIDES_VS_GPU),
    {
        # go/keep-sorted start
        "_foreach_erf": {
            torch.float16: {"rtol": 2.5e-3, "atol": 1e-4},
        },
        "_foreach_erfc": {
            torch.bfloat16: {"atol": 4.7e-3},
        },
        "_foreach_log10": {
            torch.float16: {"rtol": 1.2e-3, "atol": 5.9e-4},
        },
        "_foreach_pow": {
            torch.float16: {"rtol": 1.3e-3},
        },
        "_foreach_rsqrt": {
            torch.float32: {"rtol": 4e-6},
        },
        "_foreach_sigmoid": {
            torch.float16: {"atol": 5.9e-4},
        },
        "_native_batch_norm_legit": {
            torch.bfloat16: {"atol": 1.1e-2},
            torch.float16: {"atol": 1.4e-3},
        },
        "_thnn_fused_gru_cell": {
            torch.bfloat16: {"atol": 3.6e-3},
            torch.float16: {"atol": 1.1e-3},
            torch.float32: {"atol": 4.7e-5},
        },
        "_thnn_fused_lstm_cell": {
            torch.bfloat16: {"atol": 4.7e-3},
            torch.float16: {"atol": 5.2e-4},
            torch.float32: {"rtol": 6.5e-6, "atol": 5e-5},
        },
        "clamp": {
            torch.bfloat16: {"atol": 6e-1},
            torch.float16: {"atol": 6e-1},
        },
        "clamp_max": {
            torch.bfloat16: {"atol": 6e-1},
        },
        "clamp_min": {
            torch.bfloat16: {"atol": 6e-1},
        },
        "cumprod": {
            torch.bfloat16: {"rtol": 1.7e-1, "atol": 1.2e-2},
            torch.float16: {"rtol": 1.8e-2, "atol": 2e-3},
        },
        "einsum": {
            torch.float16: {"rtol": 2.4e-2},
            torch.float32: {"rtol": 2.3e-2},
        },
        "erf": {
            torch.bfloat16: {"rtol": 3.5e-2, "atol": 4.3e-4},
        },
        "erfinv": {
            torch.float16: {"rtol": 1.8e-3, "atol": 5.9e-3},
            torch.float32: {"rtol": 5e-5, "atol": 3.1e-3},
        },
        "fmax": {
            torch.bfloat16: {"atol": 6e-1},
        },
        "fmin": {
            torch.bfloat16: {"atol": 6e-1},
        },
        "gru.input": {
            torch.bfloat16: {"atol": 9.4e-3},
            torch.float16: {"rtol": 4.3e-3, "atol": 5e-3},
            torch.float32: {"rtol": 5.3e-3, "atol": 4.9e-3},
        },
        "kron": {
            torch.float16: {"rtol": 2.8e-3, "atol": 6.3e-2},
        },
        "ldexp": {
            torch.float32: {"rtol": 3.3e-6, "atol": 1.2e-3},
        },
        "lerp": {
            torch.bfloat16: {"rtol": 1.3e-1, "atol": 1.9e-1},
            torch.float16: {"rtol": 6.6e-3, "atol": 5.9e-3},
        },
        "linalg.cholesky_ex": {
            torch.float32: {"atol": 1.3e-3},
        },
        "linalg.inv": {
            torch.float32: {"rtol": 7.8e-2, "atol": 1.3e-2},
            torch.float64: {"rtol": 9e-6, "atol": 2.5e-7},
        },
        "linalg.lu": {
            torch.float32: {"rtol": 7.4e-1, "atol": 3.6e-2},
            torch.float64: {"rtol": 2.3e-5, "atol": 7.6e-6},
        },
        "linalg.lu_factor_ex": {
            torch.float32: {"rtol": 2.3e-5, "atol": 4.8e-4},
            torch.float64: {"rtol": 2.1e-5, "atol": 2e-6},
        },
        "linalg.lu_solve": {
            torch.float32: {"rtol": 4.7e-2, "atol": 1.4e-1},
        },
        "linalg.qr": {
            torch.float32: {"rtol": 9e-3, "atol": 1.5e-2},
        },
        "linalg.solve_ex": {
            torch.float32: {"rtol": 1.2e-1, "atol": 6e-2},
            torch.float64: {"rtol": 2.6e-6, "atol": 5.2e-6},
        },
        "linalg.svdvals": {
            torch.complex64: {"atol": 9.9e-3},
            torch.float32: {"atol": 6.6e-3},
        },
        "linalg.vector_norm": {
            torch.complex64: {"rtol": 1.3e-5, "atol": 1.3e-5},
            torch.float16: {"rtol": 2.1e-3, "atol": 1.5e-3},
            torch.float32: {"rtol": 1.6e-5, "atol": 1.6e-5},
        },
        "log2": {
            torch.float16: {"rtol": 1.2e-3, "atol": 2.5e-4},
        },
        "log_softmax": {
            torch.bfloat16: {"rtol": 3.5e-1, "atol": 9.8e-3},
            torch.float16: {"rtol": 1.8e-2, "atol": 5.9e-3},
            torch.float32: {"rtol": 8e-4, "atol": 2.6e-4},
        },
        "logcumsumexp": {
            torch.bfloat16: {"rtol": 7.3e-2, "atol": 3.3e-2},
            torch.float16: {"rtol": 9.4e-3, "atol": 4.7e-3},
            torch.float32: {"rtol": 1.4e-4, "atol": 6.5e-5},
        },
        "lstm.input": {
            torch.bfloat16: {"atol": 4.7e-3},
            torch.float16: {"rtol": 3.4e-3, "atol": 4.2e-3},
            torch.float32: {"rtol": 2.9e-3, "atol": 4e-3},
        },
        "native_batch_norm": {
            torch.bfloat16: {"atol": 1.1e-2},
            torch.float16: {"atol": 1.4e-3},
        },
        "nn.functional.conv3d": {
            torch.float16: {"rtol": 3.9e-2, "atol": 6.6e-2},
            torch.float32: {"rtol": 4.5e-2, "atol": 8.8e-2},
        },
        "nn.functional.conv_transpose3d": {
            torch.float16: {"rtol": 5e-2, "atol": 5.6e-2},
            torch.float32: {"rtol": 4e-2, "atol": 6.5e-2},
        },
        "nn.functional.cross_entropy": {
            torch.bfloat16: {"atol": 1.2e-3},
            torch.float16: {"atol": 3e-4},
        },
        "nn.functional.ctc_loss": {
            torch.float32: {"rtol": 3.6e-3, "atol": 2.2e-4},
        },
        "nn.functional.grid_sample": {
            torch.bfloat16: {"rtol": 1.9e-2, "atol": 1.9e-2},
            torch.float16: {"rtol": 3.6e-3, "atol": 2.4e-3},
        },
        "nn.functional.hardswish": {
            torch.bfloat16: {"rtol": 5.2e-2, "atol": 3e-3},
            torch.float16: {"rtol": 1.3e-2, "atol": 3.7e-4},
        },
        "nn.functional.interpolate": {
            torch.bfloat16: {"rtol": 2.7e-2, "atol": 7.9e-2},
            torch.float16: {"rtol": 1.8e-3, "atol": 4e-3},
        },
        "nn.functional.linear": {
            torch.float16: {"rtol": 3.6e-3, "atol": 5.9e-3},
            torch.float32: {"rtol": 4.1e-3, "atol": 2.5e-3},
        },
        "nn.functional.pdist": {
            torch.float32: {"rtol": 1.5e-5, "atol": 2.6e-5},
        },
        "nn.functional.scaled_dot_product_attention": {
            torch.bfloat16: {"rtol": 2.3e-1, "atol": 2.1e-1},
            torch.float16: {"rtol": 2.7e-1, "atol": 2.5e-1},
            torch.float32: {"rtol": 4.3e-1, "atol": 9.3e-1},
            torch.float64: {"rtol": 6e-6, "atol": 7.7e-6},
        },
        "norm": {
            torch.complex64: {"rtol": 8.2e-6},
            torch.float16: {"atol": 1.2e-3},
            torch.float32: {"rtol": 8.8e-6},
        },
        "polygamma": {
            torch.bfloat16: {"rtol": 3.6e-2},
            torch.float32: {"rtol": 1.1e-5},
        },
        "prod": {
            torch.float16: {"rtol": 1.6e-3, "atol": 9.8e-4},
        },
        "std": {
            torch.float16: {"atol": 6e-4},
        },
        "tan": {
            torch.float32: {"rtol": 3.8e-6, "atol": 8.4e-3},
        },
        # go/keep-sorted end
    },
)
if test_utils.parse_version(torch.__version__) < (2, 14, 0):
  # PyTorch 2.14+ changed clamp_min/clamp_max derivative tie-breaking at
  # self == bound from 1.0 to 0.0/0.5 ( matching the PyTorch 2.15 GPU goldens).
  # In OSS (which pins PyTorch 2.11), autograd decomposition of
  # _foreach_clamp_min/_foreach_clamp_max uses the pre-2.14 formula.
  update_dict(
      ACCURACY_OVERRIDES_GRAD_VS_GPU,
      {
          # go/keep-sorted start
          "_foreach_clamp_max": {
              torch.bfloat16: {"atol": 1.0},
              torch.float16: {"atol": 1.0},
          },
          "_foreach_clamp_min": {
              torch.bfloat16: {"atol": 1.0},
              torch.float16: {"atol": 1.0},
          },
          # go/keep-sorted end
      },
  )


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
      # Main input should be a `uint8` tensor.
      isinstance(op_input.input_value, torch.Tensor)
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
    unused_variant: OpVariant, op_input: OpInput
) -> bool:
  # The `pivot` keyword argument is set to False.
  return not op_input.kwargs.get("pivot", True)


# Returns true for test cases in eager mode, with 0-sized tensor input (trivial
# case) whose dtype is not floating point.
# TODO: support 0-sized tensor inputs whose dtype is not floating point in TPU
# eager mode.
def _native_layer_norm_0sized_nonfloat_eager_gpu(
    unused_variant: OpVariant, op_input: OpInput
) -> bool:
  return (
      not op_testing.is_compiled_mode()
      and not torch.is_floating_point(op_input.input_value)
      # Trivial case: 0-sized tensors.
      and op_input.input_value.numel() == 0
  )


# Returns true for test cases with, at least one of `row` (first argument) or
# `col` (second argument) is 0, and `dtype=bool`.
# TODO: support bool inputs.
def _tril_indices_bool_zero_arg_eager_gpu(
    unused_variant: OpVariant, op_input: OpInput
) -> bool:
  return (
      not op_testing.is_compiled_mode()
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


# Ops that TorchTPU specializes or throws on symbolic dims for, while CPU
# doesn't, under --check_symbolic_shapes (see the ops_test_symbolic_shapes
# target). Maps the op name (with `.{variant_test_name}` if the test uses one)
# to the reason. Such specializations make torch.compile recompile for every new
# input shape. They usually come from calling `numel()`, `sizes()` etc. on
# symbolic tensors in C++ kernels: use `torch_tpu::ConstTensor` to fix them.
# TODO(wan): Fix these ops and make this empty.
SYMBOLIC_SHAPES_KNOWN_FAILURES = {
    # go/keep-sorted start
    # The CPU control of the fused RNN cells runs their Python reference
    # (`OpInfo.ref`).
    "_thnn_fused_gru_cell": (
        "TorchTPU's autograd pins all input and hidden dims"
    ),
    "_thnn_fused_lstm_cell": (
        "TorchTPU's autograd pins all input and hidden dims"
    ),
    "fft.fft": "TorchTPU's FFT pins all input dims",
    "fft.ifft": "TorchTPU's FFT pins all input dims",
    "fft.rfft": "TorchTPU's FFT pins all input dims with requires_grad",
    "nn.functional.max_pool1d": "TorchTPU's max_pool pins the channel dim",
    "nn.functional.max_pool2d": (
        "TorchTPU's max_pool pins the channel and spatial dims"
    ),
    "nn.functional.max_pool3d": (
        "TorchTPU's max_pool pins the channel and spatial dims"
    ),
    # go/keep-sorted end
}
if oss_utils.is_oss():
  # The older PyTorch that OSS pins throws on symbolic dims on CPU for the
  # samples that TorchTPU's FFT fails on, so the check skips those samples and
  # these ops pass there.
  del SYMBOLIC_SHAPES_KNOWN_FAILURES["fft.fft"]
  del SYMBOLIC_SHAPES_KNOWN_FAILURES["fft.ifft"]
  del SYMBOLIC_SHAPES_KNOWN_FAILURES["fft.rfft"]


def _if_tpu_vs_gpu_compiled(true_value: Any, false_value: Any) -> Any:
  """Returns true_value for compiled GPU-golden exclusions.

  Evaluates to `true_value` during compiled TPU replay and golden recording
  so `_out_casting_tests` selects matching pairs. Call
  `op_testing.is_tpu_vs_gpu_compiled()` for replay-only options.
  """
  if op_testing.is_tpu_vs_gpu_compiled() or (
      op_testing._gen_gpu_golden_mode() and op_testing.is_compiled_mode()
  ):
    return true_value
  return false_value


class TestOps(op_testing.OpInfoTestBase):
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

    # With unbounded dynamic kernels disabled (the ops_test_static target), run
    # only the tests marked with @also_run_static: the others run in the
    # targets with them enabled.
    method = getattr(self, self._testMethodName, None)
    if _unbounded_dynamic_kernels_disabled() and not getattr(
        method, _ALSO_RUN_STATIC_ATTR, False
    ):
      self.skipTest(
          f"Skipping test because {_UNBOUNDED_DYNAMIC_KERNELS_ENV_VAR}"
          " disables unbounded dynamic kernels, and the test is not marked"
          " with @also_run_static."
      )

    # Filter the tests by --test_categories only if --test_filter is not on
    # the bazel command line. If --test_filter is specified, we should respect
    # the user's intent and not apply any additional filtering based on
    # --test_categories.
    if not _has_test_filter() and _TEST_CATEGORIES.value:
      exclusions = {c[1:] for c in _TEST_CATEGORIES.value if c.startswith("-")}
      inclusions = {c for c in _TEST_CATEGORIES.value if not c.startswith("-")}

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
        tpu_gpu_overrides=ACCURACY_OVERRIDES_VS_GPU_COMPILED
        if op_testing.is_compiled_mode()
        else ACCURACY_OVERRIDES_VS_GPU,
        grad_gpu_overrides=ACCURACY_OVERRIDES_GRAD_VS_GPU,
    )
    self.set_symbolic_shapes_known_failures(SYMBOLIC_SHAPES_KNOWN_FAILURES)

  def test_abs(self):
    self.do_test_op(
        "abs",
        # TODO(b/495929595): can be incorrect close to 0.
        # TODO: stablehlo gradients for Real/Imag/Abs on complex are not
        # working.
        skip_if=lambda _, op_input: (
            op_testing._COMPUTE_GRAD.value
            and op_input.input_value.dtype == torch.complex64
        ),
    )

  def test_acos(self):
    self.do_test_op("acos")

  def test_acosh(self):
    self.do_test_op("acosh")

  def test_adaptive_avg_pool2d(self):
    self.do_test_op(
        "nn.functional.adaptive_avg_pool2d",
        exclude_dtypes=(torch.bool,)
        # TODO(b/589910660): Eager CUDA rejects uint8/int8/int16/int32
        # (TorchTPU matches eager); compiled CUDA/Inductor widens them.
        + _if_tpu_vs_gpu_compiled(
            (torch.uint8, torch.int8, torch.int16, torch.int32), ()
        ),
    )

  def test_adaptive_avg_pool3d(self):
    self.do_test_op(
        "nn.functional.adaptive_avg_pool3d",
        exclude_dtypes=COMPLEX_DTYPES
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
    )

  def test_adaptive_max_pool2d(self):
    self.do_test_op(
        "nn.functional.adaptive_max_pool2d",
        # TODO: adaptive_max_pool2d_backward is not implemented on TPU yet.
        check_grad=False,
        # TODO(b/589910660): Eager CUDA rejects INTEGRAL_DTYPES (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
    )

  def test_add(self):
    self.do_test_op("add")

  def test_addcdiv(self):
    self.do_test_op("addcdiv")

  def test_addcmul(self):
    self.do_test_op(
        "addcmul",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        # TODO(b/568080282): Fix addcmul_ succeeding on bool in compiled mode.
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_addmm(self):
    self.do_test_op(
        "addmm",
        # TODO: XLA doesn't support complex dtypes currently. Fails with:
        # algebraic_simplifier.cc:584] Check failed: computation->Accept(this)
        # is OK (UNIMPLEMENTED: Converting from type C128 to type F32 is not
        # implemented.
        # TODO: make addmm fail for integral dtypes to match GPU.
        exclude_dtypes=COMPLEX_DTYPES
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
        # TODO: XLA doesn't support complex dtypes currently. Fails with:
        # algebraic_simplifier.cc:584] Check failed: computation->Accept(this)
        # is OK (UNIMPLEMENTED: Converting from type C128 to type F32 is not
        # implemented.
        exclude_inplace_dtypes=COMPLEX_DTYPES
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
    )

  def test_addmm_activation(self):
    self.do_test_op(
        "_addmm_activation",
        # _addmm_activation only supports floating-point dtypes on GPU and TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # _addmm_activation is an inference/lowering fusion without an autograd
        # backward formula in PyTorch.
        check_grad=False,
    )

  def test_addmm_dtype(self):
    self.do_test_op(
        "addmm.dtype",
        # TorchTPU's addmm.dtype kernel does not yet support complex or integral
        # dtypes.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # ATen's addmm.dtype overload does not register an autograd derivative.
        check_grad=False,
    )

  def test_addmv(self):
    self.do_test_op(
        "addmv",
        # GPU (CUDA) does not support integral dtypes for addmv.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.complex64,
        )
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
        exclude_inplace_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.float64,
            torch.complex64,
        )
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
    )

  def test_all(self):
    self.do_test_op(
        "all",
        exclude_dtypes=(torch.int4,),
    )

  def test_any(self):
    self.do_test_op(
        "any",
        exclude_dtypes=(torch.int4,)
        # TODO(b/589923495): Inductor returns bool instead of uint8 for
        # any(uint8), unlike eager CUDA and TorchTPU.
        + _if_tpu_vs_gpu_compiled((torch.uint8,), ()),
    )

  def test_amax(self):
    self.do_test_op("amax")

  def test_amin(self):
    self.do_test_op("amin")

  def test_aminmax(self):
    self.do_test_op(
        "aminmax",
        # TODO(b/562505887): PyTorch autograd backward fails with:
        # RuntimeError: derivative for aten::aminmax is not implemented.
        check_grad=False,
        # TODO: make this work with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
    )

  def test_angle(self):
    self.do_test_op("angle")

  def test_arange(self):
    """Tests arange, arange.start, arange.out, arange.start_step, arange.start_out."""

    self.do_test_op(
        "arange",
        # TODO: fix arange(out=...) failing.
        check_out_variant=False,
        # TODO: fix arange() succeeding for complex types (it should
        # fail).
        # TODO: fix arange() returning wrong results for uint8.
        exclude_dtypes=(torch.uint8,) + COMPLEX_DTYPES,
        # NOTE: the test sample contains torch.arange(5), which *should*
        # return a tensor on CPU. Therefore we don't check that the result
        # is on TPU.
        check_device=False,
    )

  def test_argmax(self):
    self.do_test_op(
        "argmax",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_argmin(self):
    self.do_test_op(
        "argmin",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

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

  def test_avg_pool1d(self):
    self.do_test_op(
        "nn.functional.avg_pool1d",
        # GPU short-circuits empty-batch inputs in avg_pool1d before checking
        # dtype, whereas TorchTPU's ValidateAvgPool1dDtype rejects bool, uint8,
        # int8, int16, int32, and complex64 upfront.
        exclude_dtypes=(
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
        )
        + COMPLEX_DTYPES
        # TODO(b/589910660): Eager CUDA rejects int64 (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.int64,), ()),
    )

  def test_avg_pool2d(self):
    self.do_test_op(
        "nn.functional.avg_pool2d",
        exclude_dtypes=(torch.bool,)
        # TODO(b/589910660): Eager CUDA rejects int8/int16/int32/int64
        # (TorchTPU matches eager); compiled CUDA/Inductor widens them.
        + _if_tpu_vs_gpu_compiled(
            (torch.int8, torch.int16, torch.int32, torch.int64), ()
        ),
    )

  def test_avg_pool3d(self):
    self.do_test_op(
        "nn.functional.avg_pool3d",
        # TODO: GPU does not support integral, complex, and `float16` dtypes.
        exclude_dtypes=(torch.uint8, torch.int8, torch.int16)
        + COMPLEX_DTYPES
        + (torch.float16,)
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
    )

  def test_baddbmm(self):
    self.do_test_op(
        "baddbmm",
        # TODO(b/495524286): Failed to generate integral golden results on GPU
        # GPU (CUDA) does not support integral dtypes for baddbmm.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
        exclude_inplace_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
    )

  def test_baddbmm_dtype(self):
    self.do_test_op(
        "baddbmm.dtype",
        # CUDA cublas does not support integral or boolean dtypes for
        # baddbmm.dtype, whereas TorchTPU lowers them via DotGeneral.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.bool,),
        # ATen's baddbmm.dtype overload does not register an autograd
        # derivative.
        check_grad=False,
        # In torch.compile mode, baddbmm.dtype_out with beta=1 aliases self and
        # out in AOTAutograd functionalization and raises a RuntimeError.
        skip_if=lambda variant, op_input: (
            op_testing.is_compiled_mode()
            and variant == OpVariant.OUT
            and op_input.kwargs.get("beta") == 1
        ),
    )

  def test_bernoulli(self):
    self.do_test_op(
        "bernoulli",
        # By definition, bernoulli() returns a tensor with random values, so
        # there's no point in checking the values.
        check_value=CheckValueMode.SKIP,
        # GPU (CUDA) does not support complex dtypes for bernoulli.
        # TPU does not support complex or integral dtypes for bernoulli.
        exclude_dtypes=(
            *COMPLEX_DTYPES,
            *INTEGRAL_DTYPES,
            torch.int4,
            torch.bool,
        ),
    )

  def test_binary_cross_entropy(self):
    self.do_test_op(
        "nn.functional.binary_cross_entropy",
        # TODO(b/557286629): support data sampling for non-floating-point dtypes.
        exclude_dtypes=INTEGRAL_DTYPES + COMPLEX_DTYPES,
    )

  def test_bincount(self):
    self.do_test_op(
        "bincount",
        # TODO(b/557286629): support data sampling for booleans.
        exclude_dtypes=COMPLEX_DTYPES + FLOAT_DTYPES + (torch.bool,),
        # TODO(b/544855480): Fix bincount consistency failure on GPU vs TPU for float dtypes.
        # GPU reference fails: "bincount_cuda" not implemented for 'Float'.
        # However, it succeeds on TorchTPU, causing a consistency check failure.
        skip_if=lambda variant, op_input: (
            op_testing._torch_tpu_vs_gpu_mode()
            and op_testing._COMPUTE_GRAD.value
            and op_input.input_value.dtype in FLOAT_DTYPES
        ),
    )

  def test_bitwise_and(self):
    self.do_test_op("bitwise_and")

  def test_bitwise_left_shift(self):
    self.do_test_op(
        "bitwise_left_shift",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_bitwise_not(self):
    self.do_test_op(
        "bitwise_not",
        exclude_dtypes=FLOAT_DTYPES + COMPLEX_DTYPES,
        exclude_inplace_dtypes=FLOAT_DTYPES + COMPLEX_DTYPES,
    )

  def test_bitwise_or(self):
    self.do_test_op("bitwise_or")

  def test_bitwise_right_shift(self):
    self.do_test_op(
        "bitwise_right_shift",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_bitwise_xor(self):
    self.do_test_op("bitwise_xor")

  def test_bmm(self):
    self.do_test_op(
        "bmm",
        # GPU (CUDA) does not support integral dtypes for bmm.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
    )

  def test_bmm_dtype(self):
    self.do_test_op(
        "bmm.dtype",
        # CUDA cublas does not support integral or boolean dtypes for bmm.dtype,
        # whereas TorchTPU lowers them via DotGeneral.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.bool,),
        # ATen's bmm.dtype overload does not register an autograd derivative.
        check_grad=False,
    )

  def test_cat(self):
    self.do_test_op(
        "cat",
        # In gradient mode, the test harness computes losses via `t.sum()`,
        # which fails on CUDA (`"sum_cuda" not implemented for 'Float8_e4m3fn'`)
        # but succeeds on TorchTPU, causing a failure-consistency mismatch.
        exclude_dtypes=(
            (torch.float8_e4m3fn,) if op_testing._COMPUTE_GRAD.value else None
        ),
    )

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
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_channel_shuffle(self):
    self.do_test_op("nn.functional.channel_shuffle")

  def test_clamp(self):
    self.do_test_op(
        "clamp",
        # TODO: b/478321000 remove when PyTorch#173110 is fixed.
        skip_if=_inplace_clamp_input_has_negative_values_uint8_gpu,
        exclude_dtypes=(torch.bool,),
        exclude_inplace_dtypes=(torch.bool,),
    )

  def test_clamp_min(self):
    self.do_test_op(
        "clamp_min",
        # TODO: b/478321000 remove when PyTorch#173110 is fixed.
        skip_if=_inplace_clamp_input_has_negative_values_uint8_gpu,
        exclude_dtypes=(torch.complex64,),
        exclude_inplace_dtypes=(torch.complex64,),
    )

  def test_clamp_max(self):
    self.do_test_op(
        "clamp_max",
        # TODO: b/478321000 remove when PyTorch#173110 is fixed.
        skip_if=_inplace_clamp_input_has_negative_values_uint8_gpu,
        exclude_dtypes=(torch.complex64,),
        exclude_inplace_dtypes=(torch.complex64,),
    )

  def test_clone(self):
    self.do_test_op("clone")

  def test_col2im(self):
    self.do_test_op(
        "col2im",
        # CUDA does not support integral or boolean dtypes for col2im, and
        # TorchTPU does not yet support complex128.
        exclude_dtypes=(torch.complex128, torch.bool) + INTEGRAL_DTYPES,
    )

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
    self.do_test_op(
        "constant_pad_nd",
        # In gradient mode, the test harness computes losses via `t.sum()`,
        # which fails on CUDA (`"sum_cuda" not implemented for 'Float8_e4m3fn'`)
        # but succeeds on TorchTPU, causing a failure-consistency mismatch.
        # TODO(b/590143907): MakeConstantAttr rejects float8_e4m3fn scalar
        # constant in compiled mode.
        exclude_dtypes=(
            (torch.float8_e4m3fn,)
            if op_testing._COMPUTE_GRAD.value
            else _if_tpu_vs_gpu_compiled((torch.float8_e4m3fn,), ())
        ),
    )

  def test_cos(self):
    self.do_test_op("cos")

  def test_cosh(self):
    self.do_test_op("cosh")

  def test_count_nonzero(self):
    self.do_test_op("count_nonzero")

  def test_cross_entropy(self):
    self.do_test_op(
        "nn.functional.cross_entropy",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_ctc_loss(self):
    self.do_test_op(
        "nn.functional.ctc_loss",
        # TODO(b/557286629): support data sampling for integral dtypes.
        exclude_dtypes=INTEGRAL_DTYPES + COMPLEX_DTYPES
        # PyTorch's native CUDA kernel (LossCTC.cu) dynamic shared
        # memory allocation per thread block exceeds hardware limits
        # for float64 on long target sequence lengths, leading to
        # cudaErrorLaunchOutOfResources.
        + (torch.float64,),
    )

  def test_cumprod(self):
    self.do_test_op(
        "cumprod",
        # TODO(b/589910660): Eager CUDA rejects dtype=bool output on cumprod
        # (TorchTPU matches eager); compiled CUDA/Inductor widens it.
        skip_if=lambda variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and variant in (OpVariant.BASE, OpVariant.INPLACE)
            and op_input.kwargs.get("dtype") == torch.bool
        ),
    )

  def test_cummax(self):
    self.do_test_op("cummax")

  def test_cumsum(self):
    self.do_test_op(
        "cumsum",
        # TODO(b/568080282): Fix cumsum_ succeeding on bool in compiled mode.
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        # TODO(b/589910660): Eager CUDA rejects dtype=bool output on cumsum
        # (TorchTPU matches eager); compiled CUDA/Inductor widens it.
        skip_if=lambda variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and variant == OpVariant.BASE
            and op_input.kwargs.get("dtype") == torch.bool
        ),
    )

  def test_cummin(self):
    self.do_test_op("cummin")

  def test_diagonal(self):
    self.do_test_op("diagonal")

  def test_digamma(self):
    self.do_test_op("digamma")

  def test_div(self):
    exclude_inplace_dtypes: tuple[torch.dtype, ...] = ()
    if (
        op_testing.is_tpu_vs_gpu_compiled()
        and test_utils.get_torch_version() < (2, 13, 0)
    ):
      # Before PyTorch 2.13, aten.div_ lacks safe-cast checks in
      # torch._meta_registrations (meta_div_inplace), so compiled div_() on
      # bool/integral tensors succeeds instead of raising like the 2.15 goldens.
      exclude_inplace_dtypes = INTEGRAL_DTYPES
    self.do_test_op("div", exclude_inplace_dtypes=exclude_inplace_dtypes)

  def test_dot(self):
    self.do_test_op(
        "dot",
        # Relaxes the comparison mode from STRICT to LOOSE for floating point
        # precision differences. The absolute difference between the expected
        # and actual results can be slightly above the strict tolerance of
        # 1e-05.
        # GPU (CUDA) does not support integral dtypes for dot.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
    )

  def test_native_dropout_backward(self):
    self.do_test_op(
        "native_dropout_backward",
        # GPU (CUDA) does not support complex and integral dtypes for
        # native_dropout_backward.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        + COMPLEX_DTYPES
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
    )

  def test_einsum(self):
    self.do_test_op(
        "einsum",
        # GPU (CUDA) does not support integral or boolean dtypes for einsum.
        exclude_dtypes=(torch.bool,) + INTEGRAL_DTYPES,
    )

  def test_embedding(self):
    self.do_test_op(
        "nn.functional.embedding",
        # TODO: fix embedding() failing with complex dtypes.
        exclude_dtypes=(
            *COMPLEX_DTYPES,
            # TODO(b/555391048): support float8 golden data sampling.
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float4_e2m1fn_x2,
        ),
    )

  def test_embedding_bag(self):
    self.do_test_op(
        "nn.functional.embedding_bag",
        # TODO: add support for sparse embeddings, scale_grad_by_freq, and
        # _embedding_bag_per_sample_weights_backward on TPU.
        skip_if=lambda variant, op_input: (
            op_input.kwargs.get("sparse", False)
            or op_input.kwargs.get("scale_grad_by_freq", False)
            or (
                op_testing._COMPUTE_GRAD.value
                and (
                    op_input.kwargs.get("per_sample_weights") is not None
                    or len(op_input.args) >= 8
                    and op_input.args[7] is not None
                )
            )
        ),
        # Override _OSS_SAMPLE_LIMIT (5) in grad mode since the first 7 golden
        # samples use per_sample_weights and are filtered out by skip_if.
        max_samples_per_op_dtype=(
            16 if op_testing._COMPUTE_GRAD.value else None
        ),
    )

  def test_embedding_renorm_(self):
    self.do_test_op(
        "embedding_renorm_",
        # embedding_renorm_ only supports floating-point weights on GPU and TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES + (torch.bool,),
        exclude_inplace_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES + (torch.bool,),
        # embedding_renorm_ is an in-place weight renormalization utility and is
        # non-differentiable.
        check_grad=False,
    )

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
    self.do_test_op(
        "eq",
        exclude_inplace_dtypes=(torch.int4,),
    )

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
        exclude_dtypes=(torch.int4,),
    )

  def test_eye(self):
    self.do_test_op("eye")

  def test_fake_quantize_per_tensor_affine_cachemask(self):
    self.do_test_op(
        "fake_quantize_per_tensor_affine_cachemask",
        # fake_quantize_per_tensor_affine_cachemask only supports floating-point
        # inputs on GPU and TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES + (torch.bool,),
        # Internal helper op returning (output, mask); autograd is defined on
        # the higher-level fake_quantize_per_tensor_affine op.
        check_grad=False,
    )

  def test_fft_rfft(self):
    self.do_test_op(
        "fft.rfft",
        # TODO: b/518595804 - PyTorch currently does not support half
        # precision FFTs for PrivateUse1 backends. See promote_type_fft
        # in SpectralOps.cpp.
        exclude_dtypes=(
            torch.float64,
            torch.float16,
            torch.int4,
        ),
    )

  def test_fft_fft(self):
    self.do_test_op(
        "fft.fft",
        # TODO: b/518595804 - PyTorch currently does not support half
        # precision FFTs for PrivateUse1 backends. See promote_type_fft
        # in SpectralOps.cpp.
        exclude_dtypes=(
            torch.float64,
            torch.float16,
            torch.int4,
        ),
    )

  def test_fft_ifft(self):
    self.do_test_op(
        "fft.ifft",
        # TODO: b/518595804 - PyTorch currently does not support half
        # precision FFTs for PrivateUse1 backends. See promote_type_fft
        # in SpectralOps.cpp.
        exclude_dtypes=(
            torch.float64,
            torch.float16,
            torch.int4,
        ),
    )

  def test_fft_irfft(self):
    self.do_test_op(
        "fft.irfft",
        # TODO: b/518595804 - PyTorch currently does not support half
        # precision FFTs for PrivateUse1 backends. See promote_type_fft
        # in SpectralOps.cpp.
        exclude_dtypes=(
            torch.float64,
            torch.float16,
            torch.int4,
        ),
    )

  def test_fill(self):
    self.do_test_op("fill")

  @test_utils.skip_if_asan("b/585576468: Slow / times out in ASAN mode.")
  def test_flatten(self):
    # TODO: check flatten is being properly exercised.
    self.do_test_op("flatten")

  def test_floor(self):
    self.do_test_op(
        "floor",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_floor_divide(self):
    self.do_test_op(
        "floor_divide",
        exclude_dtypes=(torch.float16,),
        exclude_inplace_dtypes=(torch.float16,),
    )

  def test_flip(self):
    self.do_test_op("flip")

  def test_fmax(self):
    self.do_test_op("fmax")

  def test_fmin(self):
    self.do_test_op("fmin")

  def test_fmod(self):
    self.do_test_op(
        "fmod",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  @category("foreach")
  def test_foreach_abs(self):
    self.do_test_op(
        "_foreach_abs",
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
        exclude_dtypes=COMPLEX_DTYPES
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool,), ()),
        # TODO(b/485291373): fix _foreach_addcdiv_() failing with complex
        # dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES
        + _if_tpu_vs_gpu_compiled((torch.bool,), ()),
        # Skip calls where 'value' is a list of scalars (PyTorch OpInfo passes
        # 'value' kwarg which the C++ binding rejects for list of scalars).
        skip_if=lambda _variant, op_input: (
            "value" in op_input.kwargs
            and isinstance(op_input.kwargs["value"], (list, tuple))
        ),
    )

  @category("foreach")
  def test_foreach_addcmul(self):
    self.do_test_op(
        "_foreach_addcmul",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        # TODO(b/485291373): fix _foreach_addcmul_() failing with complex
        # dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES
        + _if_tpu_vs_gpu_compiled((torch.bool,), ()),
        # Skip calls where 'value' is a list of scalars (PyTorch OpInfo passes
        # 'value' kwarg which the C++ binding rejects for list of scalars).
        skip_if=lambda _variant, op_input: (
            "value" in op_input.kwargs
            and isinstance(op_input.kwargs["value"], (list, tuple))
        ),
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
    )

  @category("foreach")
  def test_foreach_clamp_max(self):
    self.do_test_op(
        "_foreach_clamp_max",
        # TODO: fix _foreach_clamp_max() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: fix _foreach_clamp_max_() failing with complex dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
        # TODO(b/589910660): Eager CUDA rejects bool scalar overload on
        # _foreach_clamp_max (TorchTPU matches eager); Inductor widens it.
        skip_if=lambda _variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and bool(op_input.input_value)
            and op_input.input_value[0].dtype == torch.bool
            and bool(op_input.args)
            and isinstance(op_input.args[0], bool)
        ),
    )

  @category("foreach")
  def test_foreach_clamp_min(self):
    self.do_test_op(
        "_foreach_clamp_min",
        # TODO: fix _foreach_clamp_min() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: fix _foreach_clamp_min_() failing with complex dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
        # TODO(b/589910660): Eager CUDA rejects bool scalar overload on
        # _foreach_clamp_min (TorchTPU matches eager); Inductor widens it.
        skip_if=lambda _variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and bool(op_input.input_value)
            and op_input.input_value[0].dtype == torch.bool
            and bool(op_input.args)
            and isinstance(op_input.args[0], bool)
        ),
    )

  @category("foreach")
  def test_foreach_copy(self):
    self.do_test_op(
        "_foreach_copy",
        exclude_dtypes=(_if_tpu_vs_gpu_compiled((torch.complex64,), ())),
    )

  @category("foreach")
  def test_foreach_cos(self):
    self.do_test_op("_foreach_cos")

  @category("foreach")
  def test_foreach_cosh(self):
    self.do_test_op("_foreach_cosh")

  @category("foreach")
  def test_foreach_div(self):
    exclude_inplace_dtypes = (
        torch.uint8,
        torch.int8,
        torch.int16,
    ) + COMPLEX_DTYPES
    if (
        op_testing.is_tpu_vs_gpu_compiled()
        and test_utils.get_torch_version() < (2, 13, 0)
    ):
      # Before PyTorch 2.13, _foreach_div_ slow-path FakeTensor tracing calls
      # aten.div_, which lacks safe-cast checks in torch._meta_registrations
      # (meta_div_inplace), so compiled _foreach_div_() on bool/int32/int64
      # succeeds instead of raising like the 2.15 goldens.
      exclude_inplace_dtypes += (torch.bool, torch.int32, torch.int64)
    self.do_test_op(
        "_foreach_div",
        # TODO(b/485291373): fix _foreach_div() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO(b/485291373): fix _foreach_div_() failing with integral and
        # complex dtypes.
        exclude_inplace_dtypes=exclude_inplace_dtypes,
    )

  @category("foreach")
  def test_foreach_erf(self):
    self.do_test_op("_foreach_erf")

  @category("foreach")
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
    )

  @category("foreach")
  def test_foreach_frac(self):
    self.do_test_op("_foreach_frac")

  @category("foreach")
  def test_foreach_lerp(self):
    self.do_test_op(
        "_foreach_lerp",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
        # TODO(b/485291373): fix _foreach_lerp_() failing with complex dtypes.
        exclude_inplace_dtypes=(torch.complex64,)
        + _if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
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
        # TODO: fix _foreach_maximum() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: fix _foreach_maximum_() failing with complex dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
        # TODO(b/589910660): Eager CUDA rejects bool scalar overload on
        # _foreach_maximum (TorchTPU matches eager); Inductor widens it.
        skip_if=lambda _variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and bool(op_input.input_value)
            and op_input.input_value[0].dtype == torch.bool
            and bool(op_input.args)
            and isinstance(op_input.args[0], bool)
        ),
    )

  @category("foreach")
  def test_foreach_minimum(self):
    self.do_test_op(
        "_foreach_minimum",
        # TODO: fix _foreach_minimum() failing with complex dtypes.
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: fix _foreach_minimum_() failing with complex dtypes.
        exclude_inplace_dtypes=COMPLEX_DTYPES,
        # TODO(b/589910660): Eager CUDA rejects bool scalar overload on
        # _foreach_minimum (TorchTPU matches eager); Inductor widens it.
        skip_if=lambda _variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and bool(op_input.input_value)
            and op_input.input_value[0].dtype == torch.bool
            and bool(op_input.args)
            and isinstance(op_input.args[0], bool)
        ),
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
    self.do_test_op(
        "_foreach_neg",
        exclude_dtypes=(torch.bool,),
        exclude_inplace_dtypes=(torch.bool,),
    )

  @category("foreach")
  def test_foreach_norm(self):

    def skip_if(variant, op_input):
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

    exclude_dtypes = (
        # TODO(b/485291373): fix _foreach_norm() grad failing with complex
        # dtypes.
        (*COMPLEX_DTYPES, torch.int4)
        if op_testing._COMPUTE_GRAD.value
        else (torch.int4,)
    )
    self.do_test_op(
        "_foreach_norm",
        exclude_dtypes=exclude_dtypes,
        skip_if=skip_if,
    )

  @category("foreach")
  def test_foreach_pow(self):
    self.do_test_op(
        "_foreach_pow",
        # TODO(b/589910660): Eager CUDA rejects bool tensor list on
        # _foreach_pow (TorchTPU matches eager); Inductor widens it.
        skip_if=lambda variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and variant == OpVariant.BASE
            and bool(op_input.input_value)
            and op_input.input_value[0].dtype == torch.bool
            and bool(op_input.args)
            and isinstance(op_input.args[0], (list, tuple))
            and bool(op_input.args[0])
            and isinstance(op_input.args[0][0], torch.Tensor)
        ),
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
        # TODO(b/589910660): Eager CUDA rejects bool scalar arg and bool
        # tensor minus scalar (TorchTPU matches eager); Inductor widens them.
        skip_if=lambda _variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and bool(op_input.args)
            and (
                isinstance(op_input.args[0], bool)
                or (
                    bool(op_input.input_value)
                    and op_input.input_value[0].dtype == torch.bool
                    and isinstance(op_input.args[0], (int, float))
                )
            )
        ),
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

  def test_frac(self):
    self.do_test_op(
        "frac",
        # TODO(b/568080282): Clean up exclude_dtypes once Dynamo recompile_limit
        # eager fallback is fixed.
        exclude_dtypes=(_if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ())),
        # TODO(b/589910660): Eager CUDA rejects integral dtypes on frac_
        # (TorchTPU matches eager); compiled CUDA/Inductor widens them.
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_full(self):
    self.do_test_op("full")

  def test_full_like(self):
    self.do_test_op("full_like")

  def test_fused_adagrad(self):
    self.do_test_op(
        "_fused_adagrad_",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Intentional override (desired by design, not a bug): optimizer steps
        # are non-differentiable.
        check_grad=False,
    )

  def test_fused_adagrad_tensor_lr(self):
    self.do_test_op(
        "_fused_adagrad_.tensor_lr",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Intentional override (desired by design, not a bug): optimizer steps
        # are non-differentiable.
        check_grad=False,
    )

  def test_fused_adam(self):
    self.do_test_op(
        "_fused_adam_",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Intentional override (desired by design, not a bug): optimizer steps
        # are non-differentiable.
        check_grad=False,
    )

  def test_fused_adam_tensor_lr(self):
    self.do_test_op(
        "_fused_adam_.tensor_lr",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Intentional override (desired by design, not a bug): optimizer steps
        # are non-differentiable.
        check_grad=False,
    )

  def test_fused_adamw(self):
    self.do_test_op(
        "_fused_adamw_",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Intentional override (desired by design, not a bug): optimizer steps
        # are non-differentiable.
        check_grad=False,
    )

  def test_fused_adamw_tensor_lr(self):
    self.do_test_op(
        "_fused_adamw_.tensor_lr",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Intentional override (desired by design, not a bug): optimizer steps
        # are non-differentiable.
        check_grad=False,
    )

  def test_fused_moving_avg_obs_fq_helper(self):
    self.do_test_op(
        "_fused_moving_avg_obs_fq_helper",
        # CUDA only implements _fused_moving_avg_obs_fq_helper for float32;
        # TorchTPU supports float32/bfloat16/float16 and rejects float64,
        # complex, integral, and boolean dtypes.
        exclude_dtypes=(
            (torch.float16, torch.bfloat16, torch.float64)
            + COMPLEX_DTYPES
            + INTEGRAL_DTYPES
        ),
    )

  def test_fused_sgd(self):
    self.do_test_op(
        "_fused_sgd_",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Intentional override (desired by design, not a bug): optimizer steps
        # are non-differentiable.
        check_grad=False,
    )

  def test_fused_sgd_tensor_lr(self):
    self.do_test_op(
        "_fused_sgd_.tensor_lr",
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Intentional override (desired by design, not a bug): optimizer steps
        # are non-differentiable.
        check_grad=False,
    )

  def test_gather(self):
    self.do_test_op("gather")

  def test_ge(self):
    self.do_test_op(
        "ge",
        exclude_inplace_dtypes=(torch.int4,),
    )

  def test_gt(self):
    self.do_test_op(
        "gt",
        exclude_inplace_dtypes=(torch.int4,),
    )

  def test_grid_sample(self):
    self.do_test_op(
        "nn.functional.grid_sample",
        # TODO(b/589910660): Eager CUDA rejects INTEGRAL_DTYPES (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(
            (torch.complex64,) + INTEGRAL_DTYPES, ()
        ),
        # TODO(b/544855640): TPU implementation hangs on float64:
        # test_grid_sample (subtest_name='grid_sample_float64_subtest2') hangs
        # indefinitely with float64 on TPU, causing test timeout.
        # TODO(b/590144652): PyTorch FakeTensor meta check_grid_sampler_3d
        # rejects 5D bicubic on device "tpu" in compiled mode.
        skip_if=lambda variant, op_input: (
            (
                op_testing._torch_tpu_vs_gpu_mode()
                and op_testing._COMPUTE_GRAD.value
                and op_input.input_value.dtype == torch.float64
            )
            or (
                op_testing.is_tpu_vs_gpu_compiled()
                and op_input.input_value.ndim == 5
                and op_input.kwargs.get("mode") == "bicubic"
            )
        ),
        # Cap samples to avoid shard timeouts in grad_tpu_vs_gpu mode, as
        # grid_sample gradient computation is slow on TPU and generates many
        # samples across dtypes.
        max_samples_per_op_dtype=(
            8
            if op_testing._torch_tpu_vs_gpu_mode()
            and op_testing._COMPUTE_GRAD.value
            else None
        ),
    )

  def test_grouped_mm(self):
    self.do_test_op(
        "_grouped_mm",
        # _grouped_mm only supports floating-point dtypes and does not support
        # float64 on GPU or TPU.
        exclude_dtypes=(torch.float64, torch.bool)
        + COMPLEX_DTYPES
        + INTEGRAL_DTYPES,
        # _grouped_mm does not define an autograd backward formula in PyTorch.
        check_grad=False,
    )

  def test_gru_input(self):
    self.do_test_op(
        "gru.input",
        # gru.input only supports floating-point dtypes on GPU and TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES + (torch.bool,),
    )

  def test_histc(self):
    self.do_test_op(
        "histc",
        # TODO: GPU does not support `bool` and `float16` dtypes.
        exclude_dtypes=(torch.float16,),
    )

  def test_histc_skip_value_check_due_to_precision_variations(self):
    # NOTE: Verify output shapes and dtypes of previously excluded dtypes that
    # lead to binning errors due to precision variations.
    self.do_test_op(
        "histc",
        # TODO: GPU does not support `bool` and `float16` dtypes.
        exclude_dtypes=(
            torch.bool,
            torch.float16,
        ),
        check_value=CheckValueMode.SKIP,
    )

  def test_index_add(self):
    self.do_test_op("index_add")

  def test_index_copy(self):
    self.do_test_op("index_copy")

  def test_index_fill(self):
    self.do_test_op("index_fill")

  def test_index_put(self):
    self.do_test_op("index_put")

  def test_index_reduce_amax(self):
    self.do_test_op(
        "index_reduce",
        variant_test_name="amax",
        # TorchTPU only supports 0D/1D inputs in BuildScatterShlo.
        skip_if=lambda _variant, op_input: (op_input.input_value.ndim > 1),
    )

  def test_index_reduce_amin(self):
    self.do_test_op(
        "index_reduce",
        variant_test_name="amin",
        # TorchTPU only supports 0D/1D inputs in BuildScatterShlo.
        skip_if=lambda _variant, op_input: (op_input.input_value.ndim > 1),
    )

  def test_index_reduce_mean(self):
    self.do_test_op(
        "index_reduce",
        variant_test_name="mean",
        # TODO(b/589910660): Eager CUDA rejects complex64 (TorchTPU matches
        # eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(COMPLEX_DTYPES, ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled(COMPLEX_DTYPES, ()),
        # TorchTPU only supports 0D/1D inputs in BuildScatterShlo.
        skip_if=lambda _variant, op_input: (op_input.input_value.ndim > 1),
    )

  def test_index_reduce_prod(self):
    self.do_test_op(
        "index_reduce",
        variant_test_name="prod",
        # TorchTPU only supports 0D/1D inputs in BuildScatterShlo.
        skip_if=lambda _variant, op_input: (op_input.input_value.ndim > 1),
    )

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

  def test_jagged_to_padded_dense_forward(self):
    self.do_test_op(
        "_jagged_to_padded_dense_forward",
        # CUDA does not implement _jagged_to_padded_dense_forward for complex
        # dtypes, and upstream PyTorch lacks a Meta kernel for torch.compile.
        exclude_dtypes=COMPLEX_DTYPES
        + _if_tpu_vs_gpu_compiled(
            UNQUANTIZED_NUMERIC_DTYPES + (torch.bool,), ()
        ),
        # Forward-only primitive; autograd is handled by higher-level nested
        # tensor ops.
        check_grad=False,
    )

  def test_kron(self):
    self.do_test_op(
        "kron",
        # TODO: fix kron(out=...) having huge numeric errors.
        check_out_variant=False,
    )

  def test_lcm(self):
    self.do_test_op(
        "lcm",
        # Cap samples to avoid shard timeouts when lcm shares a shard with other
        # multi-dtype ops (lcm compiles a while loop per shape/dtype across
        # base, out, and inplace variants).
        max_samples_per_op_dtype=6,
    )

  def test_ldexp(self):
    self.do_test_op(
        "ldexp",
        # TODO(b/589923495): Eager CUDA and TorchTPU reject ldexp_(int8/uint8).
        # Under torch.compile, AOTAutograd functionalization bypasses the guard,
        # hitting undefined float-to-int overflow (Triton wraps vs XLA clamps).
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled(
            (torch.int8, torch.uint8), ()
        ),
    )

  def test_le(self):
    self.do_test_op(
        "le",
        exclude_inplace_dtypes=(torch.int4,),
    )

  def test_leaky_relu(self):
    self.do_test_op("nn.functional.leaky_relu")

  def test_lerp(self):
    self.do_test_op(
        "lerp",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_lgamma(self):
    self.do_test_op(
        "lgamma",
        # TODO: fix lgamma() failing for complex.
        exclude_dtypes=(torch.complex64,),
        exclude_inplace_dtypes=(torch.complex64,),
    )

  def test_linalg_cholesky_ex(self):
    self.do_test_op(
        "linalg.cholesky_ex",
        exclude_dtypes=INTEGRAL_DTYPES,
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
        exclude_dtypes=INTEGRAL_DTYPES,
    )

  def test_logit(self):
    self.do_test_op(
        "logit",
        exclude_dtypes=(torch.bool,),
        exclude_inplace_dtypes=INTEGRAL_DTYPES,
    )

  def test_lu_unpack(self):
    self.do_test_op(
        "lu_unpack",
        # TODO(b/553658275): support data sampling of integral and
        # half-precision float dtypes for SVD-based linear algebra ops.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.half, torch.bfloat16),
    )

  def test_linalg_lu_solve(self):
    self.do_test_op(
        "linalg.lu_solve",
        # TODO(b/553658275): support data sampling of integral and
        # half-precision float dtypes for SVD-based linear algebra ops.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.half, torch.bfloat16),
    )

  def test_linalg_solve_ex(self):
    self.do_test_op(
        "linalg.solve_ex",
        # TODO(b/553658275): support data sampling of integral and
        # half-precision float dtypes for SVD-based linear algebra ops.
        exclude_dtypes=INTEGRAL_DTYPES
        + (torch.half, torch.bfloat16, torch.float64),
    )

  def test_linalg_svd(self):
    self.do_test_op(
        "linalg.svd",
        # TODO(b/553658275): support data sampling of integral and
        # half-precision float dtypes for SVD-based linear algebra ops.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.half, torch.bfloat16),
        # Singular vectors U (index 0) and Vh (index 2) have sign/phase
        # ambiguity across backends; singular value gradients are tested in
        # test_linalg_svdvals.
        skip_output_indices=[0, 2],
        check_grad=False,
    )

  def test_linalg_svdvals(self):
    self.do_test_op(
        "linalg.svdvals",
        # TODO(b/553658275): support data sampling of integral and
        # half-precision float dtypes for SVD-based linear algebra ops.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.half, torch.bfloat16),
        # TODO(b/544855640): TPU implementation of _linalg_svd(compute_uv=True)
        # invoked by svdvals backward hangs on float64 (subtest18), causing
        # shard timeouts.
        skip_if=lambda _variant, op_input: (
            op_testing._COMPUTE_GRAD.value
            and op_input.input_value.dtype == torch.float64
        ),
    )

  def test_linalg_lu_out(self):
    self.do_test_op(
        "linalg.lu",
        exclude_dtypes=(
            torch.float64,
            torch.complex128,
        ),
        skip_if=_linalg_lu_without_pivot_gpu,
    )

  def test_linalg_inv_ex_out(self):
    self.do_test_op(
        "linalg.inv",
        # TODO(b/553658275): support data sampling of integral and
        # half-precision float dtypes for SVD-based linear algebra ops.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.half, torch.bfloat16),
        # TODO(b/495521055): linalg.inv fails with complex64 with compile.
        skip_if=lambda variant, op_input: (
            op_testing.is_compiled_mode()
            and op_input.input_value.dtype in COMPLEX_DTYPES
        ),
    )

  def test_norm(self):
    self.do_test_op(
        "norm",
        exclude_dtypes=(
            # TODO(b/555391048): support float4 golden data sampling.
            torch.float4_e2m1fn_x2,
            torch.int4,
        ),
    )

  def test_linalg_vector_norm_other_dtypes(self):
    # b/568044911: covers all dtypes, grad, and dynamism for vector norm.
    self.do_test_op(
        "linalg.vector_norm",
        exclude_dtypes=(torch.int4,),
        # TODO(b/544854527): Fix complex64 ord=inf/-inf gradient accuracy mismatch:
        # Greatest absolute difference: nan at index (2, 0).
        # Expected (GPU): (nan+nanj), Actual (TPU): 0j.
        skip_if=lambda variant, op_input: (
            op_testing._torch_tpu_vs_gpu_mode()
            and op_testing._COMPUTE_GRAD.value
            and op_input.input_value.dtype == torch.complex64
            and op_input.kwargs.get("ord") in (float("inf"), float("-inf"))
        ),
    )

  def test_linspace(self):
    self.do_test_op(
        "linspace",
        # PyTorch's upstream sample generator for linspace includes a hardcoded
        # sample without a device kwarg: `yield SampleInput(1, args=(3, 1))`
        # (see common_methods_invocations.py).
        # Same issue as test_arange() above.
        # In compiled mode, also skip:
        # - TODO(b/590144260): linspace.out(0, 7, 50, dtype=uint8) F32 rounding.
        # - TODO(b/589923495): Inductor keeps float start (e.g. 4.3) instead of
        #   truncating like eager CUDA/TorchTPU for integer dtypes.
        # - TODO(b/589910660): Eager CUDA rejects float-start bool linspace
        #   (TorchTPU matches eager); compiled CUDA/Inductor widens it.
        skip_if=lambda variant, op_input: (
            "device" not in op_input.kwargs
            or (
                op_testing.is_tpu_vs_gpu_compiled()
                and (
                    (
                        variant == OpVariant.OUT
                        and op_input.kwargs.get("dtype") == torch.uint8
                        and op_input.input_value == 0
                        and op_input.args == (7, 50)
                    )
                    or (
                        variant == OpVariant.BASE
                        and op_input.kwargs.get("dtype") in _INTEGER_DTYPES
                        and isinstance(op_input.input_value, float)
                        and not op_input.input_value.is_integer()
                        and len(op_input.args) >= 2
                        and op_input.args[1] > 1
                    )
                    or (
                        variant == OpVariant.BASE
                        and op_input.kwargs.get("dtype") == torch.bool
                        and isinstance(op_input.input_value, float)
                        and len(op_input.args) >= 2
                        and op_input.args[1] > 1
                    )
                )
            )
        ),
    )

  def test_lt(self):
    self.do_test_op(
        "lt",
        exclude_inplace_dtypes=(torch.int4,),
    )

  def test_log(self):
    self.do_test_op("log")

  def test_log1p(self):
    self.do_test_op("log1p")

  def test_log10(self):
    self.do_test_op("log10")

  def test_log2(self):
    self.do_test_op("log2")

  def test_log_sigmoid(self):
    self.do_test_op(
        "nn.functional.logsigmoid",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_log_softmax(self):
    self.do_test_op(
        "log_softmax",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_log_softmax_backward_data(self):
    self.do_test_op(
        "_log_softmax_backward_data",
        # TODO(b/557286629): support data sampling for non-floating-point dtypes.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )

  def test_logcumsumexp(self):
    self.do_test_op(
        "logcumsumexp",
        # logcumsumexp is a floating-point op: integer inputs are
        # unsupported (the reference sample generator itself overflows for
        # them), and the max-based logaddexp combiner has no defined extension
        # to complex.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )

  def test_logical_and(self):
    self.do_test_op("logical_and")

  def test_logical_or(self):
    self.do_test_op("logical_or")

  def test_logical_xor(self):
    self.do_test_op("logical_xor")

  def test_logical_not(self):
    self.do_test_op("logical_not")

  def test_lshift(self):
    self.do_test_op(
        "__lshift__",
        # Bitwise left shift only supports integral dtypes (excluding bool).
        exclude_dtypes=FLOAT_DTYPES + COMPLEX_DTYPES + (torch.bool,),
        exclude_inplace_dtypes=FLOAT_DTYPES + COMPLEX_DTYPES + (torch.bool,),
        # Integral bitshift is non-differentiable.
        check_grad=False,
    )

  def test_lstm_input(self):
    self.do_test_op(
        "lstm.input",
        # lstm.input only supports floating-point dtypes on GPU and TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES + (torch.bool,),
    )

  def test_masked_scatter(self):
    self.do_test_op("masked_scatter")

  def test_masked_select(self):
    self.do_test_op("masked_select")

  def test_masked_fill(self):
    self.do_test_op("masked_fill")

  def test_masked_softmax(self):
    self.do_test_op(
        "_masked_softmax",
        # _masked_softmax only supports floating-point dtypes on GPU and TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # TODO(b/589923495): Upstream PyTorch CUDA `SoftMax.cu` bugs on 0-D
        # scalar inputs:
        # - Forward (`dim=-1`): `masked_softmax_cuda` leaves `dim=-1` unwrapped
        #   after `.view(1)`, hitting the fallback and returning shape `(1,)`
        #   instead of `()`.
        # - Backward (`dim=None`): `masked_softmax_backward_cuda` sets
        #   `dim = output.dim() - 1 = -1` before `.view(1)`, tripping its own
        #   `TORCH_CHECK(dim >= 0)`.
        skip_if=lambda _variant, op_input: (
            op_input.input_value.ndim == 0
            and (
                op_input.kwargs.get("dim") == -1
                or (
                    op_testing._COMPUTE_GRAD.value
                    and op_input.kwargs.get("dim") is None
                )
            )
        ),
    )

  def test_masked_softmax_backward(self):
    self.do_test_op(
        "_masked_softmax_backward",
        # _masked_softmax_backward only supports floating-point dtypes on GPU
        # and TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Backward primitive without a second-order autograd formula.
        check_grad=False,
    )

  def test_matmul(self):
    self.do_test_op(
        "matmul",
        # TODO: TPU supports bool dtypes but not all CPU and GPU lowerings do.
        # Due to how PyTorch decomposes this op, some cases bool dtype is
        # supported and other cases it's not. This op is supposed to be
        # delegated, but due to the same reason above, there are some funky
        # lowering causing delegation to fail for us. This requires additional
        # work to fix. For now, we just exclude bool dtypes.
        # TODO: make matmul fail for integral dtypes to match GPU.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 dot/matmul
        # samples (TorchTPU matches eager); compiled CUDA/Inductor widens them.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
    )

  def test_multinomial(self):
    self.do_test_op(
        "multinomial",
        # TODO: multinomial() returns a tensor with random values, so we can't
        # check the values until we have a way to seed the random number
        # generator.
        check_value=CheckValueMode.SKIP,
        # TODO: float64 is not supported for rng on TPU.
        exclude_dtypes=(
            torch.float64,
            torch.int4,
        ),
    )

  def test_max(self):
    self.do_test_op("max")

  # TODO: b/478321000 remove this comment when PyTorch#173110 is fixed.
  # If this test starts failing due to large absolute errors, look at the
  # comments of `_inplace_clamp_input_has_negative_values_uint8()` function.
  # That might be the cause.
  def test_maximum(self):
    self.do_test_op("maximum")

  def test_max_pool1d(self):
    self.do_test_op(
        "nn.functional.max_pool1d",
        # TODO: complex64, float64, and int64 dtypes are not supported on TPU.
        exclude_dtypes=(
            torch.complex64,
            torch.float64,
        )
        # TODO: b/476417319 reject integer dtypes in TPU implementation,
        # so that it matches the GPU implementation.
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
    )

  def test_max_pool2d(self):
    self.do_test_op(
        "nn.functional.max_pool2d",
        # TODO: complex64, float64, and int64 dtypes are not supported on TPU.
        exclude_dtypes=(
            torch.complex64,
            torch.float64,
        )
        # TODO: b/476417319 reject integer dtypes in TPU implementation,
        # so that it matches the GPU implementation.
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
    )

  def test_max_pool3d(self):
    self.do_test_op(
        "nn.functional.max_pool3d",
        # TODO: b/467347286 - complex64, float64, and int64 dtypes
        # are not supported on TPU.
        exclude_dtypes=(
            torch.complex64,
            torch.float64,
        )
        # TODO: b/476417319 reject integer dtypes in TPU implementation,
        # so that it matches the GPU implementation.
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
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
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        ),
    )

  def test_mm_dtype(self):
    self.do_test_op(
        "mm.dtype",
        # CUDA cublas does not support integral or boolean dtypes for mm.dtype,
        # whereas TorchTPU lowers them via DotGeneral.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.bool,),
        # ATen's mm.dtype overload does not register an autograd derivative.
        check_grad=False,
    )

  def test_mul(self):
    self.do_test_op("mul")

  def test_nan_to_num(self):
    exclude_inplace_dtypes: tuple[torch.dtype, ...] = ()
    if (
        op_testing.is_tpu_vs_gpu_compiled()
        and test_utils.get_torch_version() < (2, 15, 0)
    ):
      # Before PyTorch 2.15, torch._refs.nan_to_num lacks a complex branch and
      # fails FakeTensor tracing on complex64 via torch.isneginf.
      exclude_inplace_dtypes = (torch.complex64,)
    self.do_test_op(
        "nan_to_num",
        exclude_dtypes=(_if_tpu_vs_gpu_compiled((torch.complex64,), ())),
        exclude_inplace_dtypes=exclude_inplace_dtypes,
    )

  def test_native_batch_norm(self):
    self.do_test_op(
        "native_batch_norm",
        # TODO: fix native_batch_norm(out=...) failing.
        check_out_variant=False,
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=(torch.float64,)
        + _if_tpu_vs_gpu_compiled((torch.complex64,) + _INTEGER_DTYPES, ()),
    )

  def test_native_batch_norm_legit(self):
    self.do_test_op(
        "_native_batch_norm_legit",
        # TODO: fix _native_batch_norm_legit(out=...) failing.
        check_out_variant=False,
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=(torch.float64,)
        + _if_tpu_vs_gpu_compiled((torch.complex64,) + _INTEGER_DTYPES, ()),
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

  def test_native_layer_norm(self):
    self.do_test_op(
        "native_layer_norm",
        # TODO: before cl/833944280 introduced the backward ops, this test runs
        # with check_grad=True. As there is no default implementation of
        # backward for native_layer_norm, it should fail. Investigate why it's
        # not failing.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        + COMPLEX_DTYPES,
        # TODO: look into why this test produces float32 (default dtype) tensors
        # on GPU for the last 2 tensors being compared, while TPU produces
        # tensors of the same dtype as the input.
        skip_output_indices=[1, 2],
        # TODO(b/542277260): support 0-sized tensor inputs whose dtype is not
        # floating point in TPU eager mode.
        skip_if=_native_layer_norm_0sized_nonfloat_eager_gpu,
    )

  def test_ne(self):
    self.do_test_op(
        "ne",
        exclude_inplace_dtypes=(torch.int4,),
    )

  def test_new_full(self):
    self.do_test_op("new_full")

  def test_new_ones(self):
    self.do_test_op("new_ones")

  def test_new_zeros(self):
    self.do_test_op("new_zeros")

  def test_neg(self):
    self.do_test_op("neg")

  def test_nll_loss(self):
    self.do_test_op(
        "nn.functional.nll_loss",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(
            (torch.complex64,) + _INTEGER_DTYPES, ()
        ),
    )

  @also_run_static
  @category("nonzero")
  def test_nonzero(self):
    self.do_test_op(
        "nonzero",
        exclude_dtypes=(torch.int4,),
    )

  def test_normal_(self):
    self.do_test_op(
        "normal",
        variant_test_name="in_place",
        # By definition, normal() returns a tensor with random values, so
        # there's no point in checking the values.
        check_value=CheckValueMode.SKIP,
        exclude_inplace_dtypes=(torch.float64,),
    )

  def test_ones(self):
    self.do_test_op("ones")

  def test_ones_like(self):
    self.do_test_op("ones_like")

  def test_padded_dense_to_jagged_forward(self):
    self.do_test_op(
        "_padded_dense_to_jagged_forward",
        # CUDA only implements float/bfloat16/half and int64 for
        # _padded_dense_to_jagged_forward, and upstream PyTorch lacks a Meta
        # kernel for torch.compile.
        exclude_dtypes=COMPLEX_DTYPES
        + (torch.uint8, torch.int8, torch.int16, torch.int32, torch.bool)
        + _if_tpu_vs_gpu_compiled(UNQUANTIZED_NUMERIC_DTYPES, ()),
        # Forward-only primitive; autograd is handled by higher-level nested
        # tensor ops.
        check_grad=False,
    )

  def test_permute(self):
    self.do_test_op("permute")

  def test_pow(self):
    self.do_test_op(
        "pow",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_nn_functional_conv1d(self):
    self.do_test_op(
        "nn.functional.conv1d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes.
        # Known issues:
        # 1. TPU lowering for int64 crashes due to "While rewriting computation
        #    to not contain X64 element types, XLA encountered an HLO for which
        #    this rewriting is not implemented: %convolution [...]"
        exclude_dtypes=COMPLEX_DTYPES + (torch.uint8, torch.int8, torch.int16),
    )

  def test_nn_functional_conv2d(self):
    self.do_test_op(
        "nn.functional.conv2d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes. See comments in test_nn_functional_conv1d.
        exclude_dtypes=COMPLEX_DTYPES + (torch.uint8, torch.int8, torch.int16),
    )

  def test_nn_functional_conv_transpose1d(self):
    self.do_test_op(
        "nn.functional.conv_transpose1d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes.
        exclude_dtypes=COMPLEX_DTYPES + (torch.uint8, torch.int8, torch.int16),
    )

  def test_nn_functional_conv_transpose2d(self):
    self.do_test_op(
        "nn.functional.conv_transpose2d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes.
        exclude_dtypes=COMPLEX_DTYPES + (torch.uint8, torch.int8, torch.int16),
    )

  def test_nn_functional_conv_transpose3d(self):
    self.do_test_op(
        "nn.functional.conv_transpose3d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes.
        exclude_dtypes=COMPLEX_DTYPES + (torch.uint8, torch.int8, torch.int16),
    )

  def test_nn_functional_conv3d(self):
    self.do_test_op(
        "nn.functional.conv3d",
        # TODO: fix nn.functional.conv*d() failing with integral and complex
        # dtypes. See comments in test_nn_functional_conv1d.
        exclude_dtypes=COMPLEX_DTYPES + (torch.uint8, torch.int8, torch.int16),
    )

  def test_nn_functional_relu(self):
    self.do_test_op(
        "nn.functional.relu",
        # TODO: fix relu() with uint8/int8/int16 dtypes.
        # xla_cuda has no support for complex64::min():
        # https://github.com/openxla/stablehlo/issues/560
        exclude_dtypes=(torch.uint8, torch.int8, torch.int16) + COMPLEX_DTYPES
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_nn_functional_rms_norm(self):
    self.do_test_op(
        "nn.functional.rms_norm",
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        + COMPLEX_DTYPES,
    )

  def test_nn_functional_scaled_dot_product_attention_math(self):
    # Force the MATH backend for both CPU and TPU.
    with attention.sdpa_kernel(attention.SDPBackend.MATH):
      self.do_test_op(
          "nn.functional.scaled_dot_product_attention",
          exclude_dtypes=(torch.uint8, torch.int8, torch.int16),
      )

  def test_nn_functional_scaled_dot_product_attention_overrideable(self):
    # b/476166586: GPU device doesn't support `OVERRIDEABLE`, so use `MATH` on
    # GPU when generating goldens. On TPU, use `OVERRIDEABLE`.
    # Only do the forward pass (backward is not implemented yet)
    backends = (
        [attention.SDPBackend.MATH]
        if op_testing._gen_gpu_golden_mode()
        else [attention.SDPBackend.OVERRIDEABLE, attention.SDPBackend.MATH]
    )
    with attention.sdpa_kernel(
        backends,
        set_priority=True,
    ):
      self.do_test_op(
          "nn.functional.scaled_dot_product_attention",
      )

  def test_nn_functional_scaled_dot_product_attention_flash(self):
    # Use CUDNN_ATTENTION on GPU when generating goldens, and FLASH_ATTENTION
    # for TPU (with MATH fallback).
    backends = (
        [attention.SDPBackend.CUDNN_ATTENTION, attention.SDPBackend.MATH]
        if op_testing._gen_gpu_golden_mode()
        else [attention.SDPBackend.FLASH_ATTENTION, attention.SDPBackend.MATH]
    )
    with attention.sdpa_kernel(
        backends,
        set_priority=True,
    ):
      self.do_test_op(
          "nn.functional.scaled_dot_product_attention",
      )

  def test_nn_functional_batch_norm(self):
    self.do_test_op(
        "nn.functional.batch_norm",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(
            (torch.complex64,) + _INTEGER_DTYPES, ()
        ),
    )

  def test_nn_functional_elu(self):
    self.do_test_op("nn.functional.elu")

  def test_nn_functional_gelu(self):
    self.do_test_op("nn.functional.gelu")

  def test_nn_functional_glu(self):
    self.do_test_op(
        "nn.functional.glu",
        # TODO(b/589910660): Eager CUDA rejects INTEGRAL_DTYPES (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
    )

  def test_nn_functional_prelu(self):
    self.do_test_op(
        "nn.functional.prelu",
        # TODO(b/557286629): support data sampling for integral dtypes.
        exclude_dtypes=INTEGRAL_DTYPES,
    )

  def test_nn_functional_hardsigmoid(self):
    self.do_test_op(
        "nn.functional.hardsigmoid",
        # TODO(b/589910660): Eager CUDA rejects INTEGRAL_DTYPES (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
    )

  def test_nn_functional_hardswish(self):
    self.do_test_op(
        "nn.functional.hardswish",
        exclude_dtypes=(_if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ())),
    )

  def test_nn_functional_hardtanh(self):
    self.do_test_op(
        "nn.functional.hardtanh",
        # TODO(b/589923495): Inductor mis-clamps uint8 hardtanh(0, -1, 1) -> 1
        # (signed min_val treated as unsigned). Eager CUDA and TorchTPU agree.
        skip_if=lambda variant, op_input: (
            op_testing.is_tpu_vs_gpu_compiled()
            and variant == OpVariant.BASE
            and op_input.input_value.dtype == torch.uint8
            and op_input.kwargs.get("min_val", -1.0) < 0
        ),
    )

  def test_nn_functional_linear(self):
    self.do_test_op(
        "nn.functional.linear",
        # GPU (CUDA) does not support integral or boolean dtypes for linear.
        # In compiled mode, linear decomposes to addmm which does not yet
        # support complex64.
        exclude_dtypes=INTEGRAL_DTYPES
        + _if_tpu_vs_gpu_compiled((torch.complex64,), ()),
    )

  def test_nn_functional_silu(self):
    self.do_test_op(
        "nn.functional.silu",
        # TODO: fix nn.functional.silu() failing with complex dtypes.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        + COMPLEX_DTYPES
        # TODO(b/589910660): Eager CUDA rejects int32/int64 (TorchTPU matches
        # eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.int32, torch.int64), ()),
        # TODO: fix nn.functional.silu_() failing with complex dtypes.
        exclude_inplace_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        + COMPLEX_DTYPES
        + _if_tpu_vs_gpu_compiled((torch.int32, torch.int64), ()),
    )

  def test_nn_functional_softplus(self):
    self.do_test_op(
        "nn.functional.softplus",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_nn_functional_unfold(self):
    self.do_test_op(
        "nn.functional.unfold",
        # TorchTPU supports integral dtypes for im2col/unfold, whereas CUDA
        # does not implement im2col_out_cuda for integral dtypes.
        exclude_dtypes=INTEGRAL_DTYPES,
    )

  def test_nn_functional_mse_loss(self):
    self.do_test_op(
        "nn.functional.mse_loss",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(
            (torch.complex64,) + _INTEGER_DTYPES, ()
        ),
    )

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
        + (torch.float64, torch.float16),
    )

  def test_polygamma(self):
    self.do_test_op(
        "polygamma",
        # Inplace polygamma_ has alias annotation issues in torch.compile
        # (b/568080282).
        skip_if=lambda variant, _: (
            op_testing.is_compiled_mode()
            and variant == op_testing.OpVariant.INPLACE
        ),
    )

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
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_remainder(self):
    self.do_test_op(
        "remainder",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_repeat(self):
    self.do_test_op("repeat")

  @test_utils.skip_if_asan("b/585577001: Slow / times out in ASAN mode.")
  def test_repeat_interleave(self):
    self.do_test_op(
        "repeat_interleave",
        skip_if=lambda _variant, op_input: (
            op_testing.is_compiled_mode()
            and (
                isinstance(op_input.kwargs.get("repeats"), torch.Tensor)
                or (
                    op_input.args and isinstance(op_input.args[0], torch.Tensor)
                )
            )
            and op_input.kwargs.get("output_size") is None
        ),
    )

  def test_replication_pad(self):
    self.do_test_op(
        "nn.functional.pad",
        variant_test_name="replicate",
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
    self.do_test_op(
        "round",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_rsqrt(self):
    self.do_test_op("rsqrt")

  def test_rshift(self):
    self.do_test_op(
        "__rshift__",
        # Bitwise right shift only supports integral dtypes (excluding bool).
        exclude_dtypes=FLOAT_DTYPES + COMPLEX_DTYPES + (torch.bool,),
        exclude_inplace_dtypes=FLOAT_DTYPES + COMPLEX_DTYPES + (torch.bool,),
        # Integral bitshift is non-differentiable.
        check_grad=False,
    )

  def test_rsub(self):
    self.do_test_op("rsub")

  # Recorded on Hopper as only there do the GPU kernels honor use_fast_accum
  # for rowwise scaling: cuBLASLt implements it, whereas Blackwell runs CUTLASS
  # kernels that ignore it.
  @op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)
  def test_scaled_mm_rowwise(self):
    self.do_test_op(
        "torch._scaled_mm",
        variant_test_name="rowwise",
        exclude_dtypes=UNQUANTIZED_NUMERIC_DTYPES,
        # aten._scaled_mm.out only supports tensorwise scaling in PyTorch.
        check_out_variant=False,
        check_grad=False,
        # TODO(amainz): Test the use_fast_accum samples once cl/990400735 lets
        # TorchTPU's torch._scaled_mm accept use_fast_accum=True.
        skip_if=lambda _, op_input: op_input.kwargs["use_fast_accum"],
        # Cover all 7 samples regardless of --max_samples_per_op_dtype.
        max_samples_per_op_dtype=7,
    )

  # Recorded on Hopper as PyTorch's CUDA rowwise _scaled_grouped_mm CUTLASS
  # kernel (f8f8bf16_rowwise_grouped_mm) is Hopper SM90-only (rejects SM100+).
  @op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)
  def test_scaled_grouped_mm_rowwise(self):
    self.do_test_op(
        "torch._scaled_grouped_mm",
        variant_test_name="rowwise",
        exclude_dtypes=(
            *UNQUANTIZED_NUMERIC_DTYPES,
            # TODO(amainz): Enable once cl/995189060 lands rowwise scale
            # lowering for torch._scaled_grouped_mm on TPU.
            torch.float8_e4m3fn,
        ),
        check_out_variant=False,
        check_grad=False,
        # TODO(amainz): Enable compiled mode once cl/995189061 registers a TPU
        # meta override that accepts non-col-major mat_b strides, and enable
        # use_fast_accum samples once cl/990400735 lands.
        skip_if=lambda _variant, op_input: (
            op_testing.is_compiled_mode() or op_input.kwargs["use_fast_accum"]
        ),
        # Cover all 5 samples regardless of --max_samples_per_op_dtype.
        max_samples_per_op_dtype=5,
    )

  # CUDA's _f8_f8_bf16_rowwise_grouped_mm CUTLASS kernel is Hopper SM90-only.
  @op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)
  def test_scaled_grouped_mm_v2(self):
    self.do_test_op(
        "torch._scaled_grouped_mm_v2",
        exclude_dtypes=(
            *UNQUANTIZED_NUMERIC_DTYPES,
            # TODO(amainz): Enable float8_e4m3fn once Hopper GPU goldens are
            # recorded and cl/995189060 lands rowwise scale lowering on TPU.
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            *quantize_utils.ROCM_ONLY_FLOAT8_DTYPES,
        ),
        check_out_variant=False,
        check_grad=False,
        max_samples_per_op_dtype=5,
    )

  def test_scaled_mm_v2(self):
    self.do_test_op(
        "torch._scaled_mm_v2",
        exclude_dtypes=(
            *UNQUANTIZED_NUMERIC_DTYPES,
            # cuBLASLt does not support e5m2 x e5m2 multiplication on CUDA
            torch.float8_e5m2,
            *quantize_utils.ROCM_ONLY_FLOAT8_DTYPES,
        ),
        check_out_variant=False,
        # Override _OSS_SAMPLE_LIMIT to cover all 8 scaled_mm_v2 test samples.
        max_samples_per_op_dtype=8,
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
    self.do_test_op(
        "torch.ops.aten._safe_softmax.default",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_scalar_tensor(self):
    self.do_test_op("scalar_tensor")

  @category("searchsorted")
  def test_searchsorted(self):
    self.do_test_op(
        "searchsorted",
        exclude_dtypes=(
            *COMPLEX_DTYPES,
            # TODO(b/555391048): support float8 golden data sampling.
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float4_e2m1fn_x2,
            torch.int4,
        )
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool,), ()),
        # Upstream generates 288 samples per dtype; cap to prevent test
        # shard timeouts (10 samples per op dtype results in ~15m test time).
        max_samples_per_op_dtype=4,
    )

  def test_segment_reduce_lengths(self):
    self.do_test_op(
        "_segment_reduce",
        variant_test_name="lengths",
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: _segment_reduce_backward is not implemented on TPU.
        check_grad=False,
    )

  def test_segment_reduce_offsets(self):
    self.do_test_op(
        "_segment_reduce",
        variant_test_name="offsets",
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: _segment_reduce_backward is not implemented on TPU.
        check_grad=False,
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

  def test_slice_scatter(self):
    self.do_test_op("slice_scatter")

  def test_softmax(self):
    self.do_test_op(
        "softmax",
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_softmax_backward_data(self):
    self.do_test_op(
        "_softmax_backward_data",
        # TODO(b/557286629): support data sampling for non-floating-point dtypes.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )

  def test_sparse_coo_add(self):
    self.do_test_op(
        "sparse_coo_add",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
        exclude_inplace_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_sparse_coo_add_dense_sparse(self):
    self.do_test_op(
        "sparse_coo_add_dense_sparse",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
        exclude_inplace_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_sparse_coo_clone(self):
    self.do_test_op(
        "sparse_coo_clone",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_sparse_coo_coalesce(self):
    self.do_test_op(
        "sparse_coo_coalesce",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_sparse_coo_copy_(self):
    self.do_test_op(
        "sparse_coo_copy_",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
        exclude_inplace_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_sparse_coo_empty_like(self):
    self.do_test_op(
        "sparse_coo_empty_like",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_sparse_coo_mul(self):
    self.do_test_op(
        "sparse_coo_mul",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
        exclude_inplace_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_sparse_coo_neg(self):
    self.do_test_op(
        "sparse_coo_neg",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES + (torch.uint8, torch.bool),
        exclude_inplace_dtypes=_SPARSE_COO_EXCLUDE_DTYPES
        + (torch.uint8, torch.bool),
    )

  def test_sparse_coo_sub(self):
    self.do_test_op(
        "sparse_coo_sub",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES + (torch.uint8, torch.bool),
        exclude_inplace_dtypes=_SPARSE_COO_EXCLUDE_DTYPES
        + (torch.uint8, torch.bool),
    )

  def test_sparse_coo_sub_dense_sparse(self):
    self.do_test_op(
        "sparse_coo_sub_dense_sparse",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES + (torch.uint8, torch.bool),
        exclude_inplace_dtypes=_SPARSE_COO_EXCLUDE_DTYPES
        + (torch.uint8, torch.bool),
    )

  def test_sparse_coo_to_dense(self):
    self.do_test_op(
        "sparse_coo_to_dense",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_sparse_coo_zero_(self):
    self.do_test_op(
        "sparse_coo_zero_",
        check_grad=False,
        check_op_failures=False,
        check_inplace_op_failures=False,
        exclude_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
        exclude_inplace_dtypes=_SPARSE_COO_EXCLUDE_DTYPES,
    )

  def test_split(self):
    self.do_test_op("split")

  def test_split_with_sizes(self):
    self.do_test_op("split_with_sizes")

  def test_split_with_sizes_copy(self):
    self.do_test_op(
        "split_with_sizes_copy",
    )

  def test_sqrt(self):
    self.do_test_op("sqrt")

  def test_squeeze(self):
    self.do_test_op(
        "squeeze",
        # Upstream PyTorch's squeeze_(dim) passes self's own symbolic sizes and
        # strides to as_strided_(), which frees them before copying them, so it
        # crashes on FakeTensors with symbolic dims (on CPU too).
        skip_if=lambda variant, _: (
            op_testing.is_check_symbolic_shapes_mode()
            and variant == OpVariant.INPLACE
        ),
    )

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
    self.do_test_op(
        "nn.functional.threshold",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_to(self):
    self.do_test_op(
        "to",
        exclude_dtypes=(
            # CPU and CUDA don't support casting to/from float4_e2m1fn_x2,
            # but TorchTPU intentionally does to support data sampling via
            # cross-dtype casts.
            torch.float4_e2m1fn_x2,
            torch.int4,
        ),
    )

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
    self.do_test_op(
        "transpose",
        exclude_dtypes=_if_tpu_vs_gpu_compiled(
            INTEGRAL_DTYPES + (torch.int4,), ()
        ),
    )

  def test_tril(self):
    self.do_test_op("tril")

  def test_tril_indices(self):
    self.do_test_op(
        "tril_indices",
        # TODO: b/476115671 add support for other dtypes that GPU supports:
        # integers other than int32 and int64, floats and complex.
        exclude_dtypes=(torch.uint8, torch.int8, torch.int16)
        + (torch.float64, torch.float16)
        + COMPLEX_DTYPES,
        # TODO(b/542277009): add support for bool dtype when either `row` or
        # `col` is 0 (trivial case).
        skip_if=_tril_indices_bool_zero_arg_eager_gpu,
    )

  def test_triu(self):
    self.do_test_op("triu")

  def test_trunc(self):
    self.do_test_op(
        "trunc",
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_unbind(self):
    self.do_test_op("unbind")

  def test_unfold(self):
    self.do_test_op("unfold")

  @also_run_static
  @category("unique")
  def test_unique(self):
    if (
        not op_testing._CHECK_DYNAMISM.value
        and op_testing.is_compiled_mode()
        and not op_testing._gen_gpu_golden_mode()
    ):
      self.skipTest(
          "TODO(b/584809741): unique is not supported in static compiled mode."
      )
    self.do_test_op(
        "unique",
        # PyTorch autograd backward is not implemented for _unique2.
        check_grad=False,
        exclude_dtypes=COMPLEX_DTYPES,
        # TODO: _unique2 on 0D inputs with return_inverse=True produces a 1D
        # PjRtBuffer for a 0D ATen inverse_indices output tensor.
        skip_if=lambda _variant, op_input: (
            not op_testing._CHECK_DYNAMISM.value
            and (
                (
                    not op_input.kwargs.get("sorted", True)
                    and op_input.kwargs.get("dim") is None
                )
                or (
                    op_input.input_value.ndim == 0
                    and op_input.kwargs.get("return_inverse", False)
                )
            )
        ),
        # Override _OSS_SAMPLE_LIMIT (5) since the first 13 golden samples have
        # sorted=False and are filtered out by skip_if.
        max_samples_per_op_dtype=20,
    )

  def test_unique_consecutive(self):
    self.do_test_op(
        "unique_consecutive",
        # TODO(b/561796146): PyTorch autograd backward fails with:
        # RuntimeError: the derivative for 'unique_consecutive' is not implemented.
        check_grad=False,
        exclude_dtypes=COMPLEX_DTYPES,
    )

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
        exclude_dtypes=(torch.int4,),
    )

  def test_unsafe_view(self):
    self.do_test_op("torch.ops.aten._unsafe_view")

  def test_upsample_nearest(self):
    self.do_test_op(
        "nn.functional.upsample_nearest",
        exclude_dtypes=COMPLEX_DTYPES
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/568080282): Fix upsample ops succeeding on integral
        # dtypes in compiled mode when GPU fails.
        + _if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
    )

  def test_upsample_bicubic2d(self):
    self.do_test_op(
        "nn.functional.interpolate",
        variant_test_name="bicubic",
        exclude_dtypes=COMPLEX_DTYPES
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/568080282): Fix upsample ops succeeding on integral
        # dtypes in compiled mode when GPU fails.
        + _if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
    )

  def test_upsample_bicubic2d_aa(self):
    self.do_test_op(
        "_upsample_bicubic2d_aa",
        check_grad=False,
    )

  def test_upsample_bilinear2d_aa(self):
    self.do_test_op(
        "_upsample_bilinear2d_aa",
        check_grad=False,
    )

  def test_upsample_linear1d(self):
    self.do_test_op(
        "nn.functional.interpolate",
        variant_test_name="linear",
        exclude_dtypes=_if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
    )

  def test_upsample_bilinear(self):
    self.do_test_op(
        "nn.functional.upsample_bilinear",
        exclude_dtypes=COMPLEX_DTYPES
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/568080282): Fix upsample ops succeeding on integral
        # dtypes in compiled mode when GPU fails.
        + _if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
        # TODO: STRICT fails for some types. Look into narrowing this down.
    )

  def test_upsample_nearest_exact(self):
    self.do_test_op(
        "nn.functional.interpolate",
        variant_test_name="nearest-exact",
        exclude_dtypes=COMPLEX_DTYPES
        + (
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/568080282): Fix upsample ops succeeding on integral
        # dtypes in compiled mode when GPU fails.
        + _if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
    )

  def test_std(self):
    self.do_test_op(
        "std",
        # TODO(b/589910660): Eager CUDA rejects INTEGRAL_DTYPES (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        exclude_dtypes=_if_tpu_vs_gpu_compiled(INTEGRAL_DTYPES, ()),
    )

  def test_var(self):
    self.do_test_op(
        "var",
        exclude_dtypes=COMPLEX_DTYPES + (torch.bool,)
        # TODO(b/589910660): Eager CUDA rejects integer dtypes (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled(_INTEGER_DTYPES, ()),
    )

  def test_var_mean(self):
    self.do_test_op("var_mean")

  def test_vdot(self):
    self.do_test_op(
        "vdot",
        # GPU (CUDA) does not support integral dtypes for vdot.
        exclude_dtypes=(
            torch.uint8,
            torch.int8,
            torch.int16,
        )
        # TODO(b/589910660): Eager CUDA rejects bool/int32/int64 (TorchTPU
        # matches eager); compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool, torch.int32, torch.int64), ()),
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

  def test_weight_norm_interface(self):
    self.do_test_op(
        "_weight_norm_interface",
        # _weight_norm_interface only supports floating-point dtypes on GPU and
        # TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
    )

  def test_weight_norm_interface_backward(self):
    self.do_test_op(
        "_weight_norm_interface_backward",
        # _weight_norm_interface_backward only supports floating-point dtypes on
        # GPU and TPU.
        exclude_dtypes=COMPLEX_DTYPES + INTEGRAL_DTYPES,
        # Backward primitive without a second-order autograd formula.
        check_grad=False,
    )

  def test_where(self):
    self.do_test_op("where")

  def test_xlogy(self):
    self.do_test_op(
        "xlogy",
        exclude_dtypes=(_if_tpu_vs_gpu_compiled((torch.complex64,), ())),
        # TODO(b/589910660): Eager CUDA rejects complex64 on xlogy_
        # (TorchTPU matches eager); compiled CUDA/Inductor widens it.
        exclude_inplace_dtypes=_if_tpu_vs_gpu_compiled((torch.complex64,), ()),
    )

  def test_zeros(self):
    self.do_test_op("zeros")

  def test_zero_(self):
    self.do_test_op("zero_")

  def test_zeros_like(self):
    self.do_test_op("zeros_like")

  def test_bucketize(self):
    self.do_test_op(
        "bucketize",
        # No total order for complex numbers.
        exclude_dtypes=COMPLEX_DTYPES
        # TODO(b/589910660): Eager CUDA rejects bool (TorchTPU matches eager);
        # compiled CUDA/Inductor widens via decomposition.
        + _if_tpu_vs_gpu_compiled((torch.bool,), ()),
    )

  def test_geqrf(self):
    if op_testing.is_check_symbolic_shapes_mode():
      # Triggers use-of-uninitialized errors on as_strided
      self.skipTest("bypassing SymInt MSAN errors in upstream PyTorch")
    self.do_test_op(
        "geqrf",
        # TODO(b/561796361): PyTorch autograd backward fails with:
        # RuntimeError: the derivative for 'geqrf' is not implemented.
        check_grad=False,
        # TODO(b/553658275): support data sampling of integral and
        # half-precision float dtypes for SVD-based linear algebra ops.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.half, torch.bfloat16),
    )

  def test_linalg_qr(self):
    self.do_test_op(
        "linalg.qr",
        # TODO(b/553658275): support data sampling of integral and
        # half-precision float dtypes for SVD-based linear algebra ops.
        exclude_dtypes=INTEGRAL_DTYPES + (torch.half, torch.bfloat16),
    )

  def test_thnn_fused_gru_cell(self):
    self.do_test_op(
        "_thnn_fused_gru_cell",
        # Exclude integral and complex dtypes: unsupported on TPU and GPU.
        # Error parity is verified in errors_test.py.
        exclude_dtypes=(INTEGRAL_DTYPES + COMPLEX_DTYPES),
    )

  def test_thnn_fused_lstm_cell(self):
    self.do_test_op(
        "_thnn_fused_lstm_cell",
        exclude_dtypes=(
            _if_tpu_vs_gpu_compiled(FLOAT_DTYPES, ())
            + INTEGRAL_DTYPES
            + COMPLEX_DTYPES
        ),
    )

  def test_native_multi_head_attention(self):
    self.do_test_op(
        "_native_multi_head_attention",
        # TODO(b/557286629): Attention only supports floating point dtypes.
        exclude_dtypes=(
            COMPLEX_DTYPES
            + INTEGRAL_DTYPES
            + _if_tpu_vs_gpu_compiled((torch.float64,), ())
        ),
        check_grad=False,
    )

  def test_transformer_encoder_layer_fwd(self):
    self.do_test_op(
        "_transformer_encoder_layer_fwd",
        # TODO(b/557286629): support data sampling for non-floating-point dtypes.
        exclude_dtypes=((torch.float64,) + COMPLEX_DTYPES + INTEGRAL_DTYPES),
        check_grad=False,
    )

  @op_testing.unimplemented_on_tpu
  def test_H(self):
    self.do_test_op("H")

  @op_testing.unimplemented_on_tpu
  def test_T(self):
    self.do_test_op("T")

  @op_testing.unimplemented_on_tpu
  def test_dunder_getitem(self):
    self.do_test_op("__getitem__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_radd(self):
    self.do_test_op("__radd__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_rand(self):
    self.do_test_op("__rand__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_rdiv(self):
    self.do_test_op("__rdiv__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_rmatmul(self):
    self.do_test_op("__rmatmul__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_rmod(self):
    self.do_test_op("__rmod__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_rmul(self):
    self.do_test_op("__rmul__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_ror(self):
    self.do_test_op("__ror__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_rpow(self):
    self.do_test_op("__rpow__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_rsub(self):
    self.do_test_op("__rsub__")

  @op_testing.unimplemented_on_tpu
  def test_dunder_rxor(self):
    self.do_test_op("__rxor__")

  @op_testing.unimplemented_on_tpu
  def test_batch_norm_with_update(self):
    self.do_test_op("_batch_norm_with_update")

  @op_testing.unimplemented_on_tpu
  def test_chunk_cat(self):
    self.do_test_op("_chunk_cat")

  @category("foreach")
  @op_testing.unimplemented_on_tpu
  def test_foreach_clone(self):
    self.do_test_op("_foreach_clone")

  @category("foreach")
  @op_testing.unimplemented_on_tpu
  def test_foreach_mm(self):
    self.do_test_op("_foreach_mm")

  @op_testing.unimplemented_on_tpu
  def test_unsafe_masked_index(self):
    self.do_test_op("_unsafe_masked_index")

  @op_testing.unimplemented_on_tpu
  def test_unsafe_masked_index_put_accumulate(self):
    self.do_test_op("_unsafe_masked_index_put_accumulate")

  @op_testing.unimplemented_on_tpu
  def test_addbmm(self):
    self.do_test_op("addbmm")

  @op_testing.unimplemented_on_tpu
  def test_addmm_decomposed(self):
    self.do_test_op("addmm", variant_test_name="decomposed")

  @op_testing.unimplemented_on_tpu
  def test_addr(self):
    self.do_test_op("addr")

  @op_testing.unimplemented_on_tpu
  def test_alias_copy(self):
    self.do_test_op("alias_copy")

  @op_testing.unimplemented_on_tpu
  def test_allclose(self):
    self.do_test_op("allclose")

  @op_testing.unimplemented_on_tpu
  def test_argsort(self):
    self.do_test_op("argsort")

  @op_testing.unimplemented_on_tpu
  def test_argwhere(self):
    self.do_test_op("argwhere")

  @op_testing.unimplemented_on_tpu
  def test_as_strided_partial_views(self):
    self.do_test_op("as_strided", variant_test_name="partial_views")

  @op_testing.unimplemented_on_tpu
  def test_as_strided_copy(self):
    self.do_test_op("as_strided_copy")

  @op_testing.unimplemented_on_tpu
  def test_as_strided_scatter(self):
    self.do_test_op("as_strided_scatter")

  @op_testing.unimplemented_on_tpu
  def test_atleast_1d(self):
    self.do_test_op("atleast_1d")

  @op_testing.unimplemented_on_tpu
  def test_atleast_2d(self):
    self.do_test_op("atleast_2d")

  @op_testing.unimplemented_on_tpu
  def test_atleast_3d(self):
    self.do_test_op("atleast_3d")

  @op_testing.unimplemented_on_tpu
  def test_bfloat16(self):
    self.do_test_op("bfloat16")

  @op_testing.unimplemented_on_tpu
  def test_block_diag(self):
    self.do_test_op("block_diag")

  @op_testing.unimplemented_on_tpu
  def test_bool(self):
    self.do_test_op("bool")

  @op_testing.unimplemented_on_tpu
  def test_broadcast_shapes(self):
    self.do_test_op("broadcast_shapes")

  @op_testing.unimplemented_on_tpu
  def test_broadcast_tensors(self):
    self.do_test_op("broadcast_tensors")

  @op_testing.unimplemented_on_tpu
  def test_broadcast_to(self):
    self.do_test_op("broadcast_to")

  @op_testing.unimplemented_on_tpu
  def test_byte(self):
    self.do_test_op("byte")

  @op_testing.unimplemented_on_tpu
  def test_cartesian_prod(self):
    self.do_test_op("cartesian_prod")

  @op_testing.unimplemented_on_tpu
  def test_cauchy(self):
    self.do_test_op("cauchy")

  @op_testing.unimplemented_on_tpu
  def test_cdouble(self):
    self.do_test_op("cdouble")

  @op_testing.unimplemented_on_tpu
  def test_cfloat(self):
    self.do_test_op("cfloat")

  @op_testing.unimplemented_on_tpu
  def test_chalf(self):
    self.do_test_op("chalf")

  @op_testing.unimplemented_on_tpu
  def test_char(self):
    self.do_test_op("char")

  @op_testing.unimplemented_on_tpu
  def test_cholesky_inverse(self):
    self.do_test_op("cholesky_inverse")

  @op_testing.unimplemented_on_tpu
  def test_cholesky_solve(self):
    self.do_test_op("cholesky_solve")

  @op_testing.unimplemented_on_tpu
  def test_chunk(self):
    self.do_test_op("chunk")

  @op_testing.unimplemented_on_tpu
  def test_column_stack(self):
    self.do_test_op("column_stack")

  @op_testing.unimplemented_on_tpu
  def test_combinations(self):
    self.do_test_op("combinations")

  @op_testing.unimplemented_on_tpu
  def test_contiguous(self):
    self.do_test_op("contiguous")

  @op_testing.unimplemented_on_tpu
  def test_copysign(self):
    self.do_test_op("copysign")

  @op_testing.unimplemented_on_tpu
  def test_corrcoef(self):
    self.do_test_op("corrcoef")

  @op_testing.unimplemented_on_tpu
  def test_cov(self):
    self.do_test_op("cov")

  @op_testing.unimplemented_on_tpu
  def test_cross(self):
    self.do_test_op("cross")

  @op_testing.unimplemented_on_tpu
  def test_cumulative_trapezoid(self):
    self.do_test_op("cumulative_trapezoid")

  @op_testing.unimplemented_on_tpu
  def test_deg2rad(self):
    self.do_test_op("deg2rad")

  @op_testing.unimplemented_on_tpu
  def test_diag(self):
    self.do_test_op("diag")

  @op_testing.unimplemented_on_tpu
  def test_diag_embed(self):
    self.do_test_op("diag_embed")

  @op_testing.unimplemented_on_tpu
  def test_diagflat(self):
    self.do_test_op("diagflat")

  @op_testing.unimplemented_on_tpu
  def test_diagonal_copy(self):
    self.do_test_op("diagonal_copy")

  @op_testing.unimplemented_on_tpu
  def test_diagonal_scatter(self):
    self.do_test_op("diagonal_scatter")

  @op_testing.unimplemented_on_tpu
  def test_diff(self):
    self.do_test_op("diff")

  @op_testing.unimplemented_on_tpu
  def test_dist(self):
    self.do_test_op("dist")

  @op_testing.unimplemented_on_tpu
  def test_div_floor_rounding(self):
    self.do_test_op("div", variant_test_name="floor_rounding")

  @op_testing.unimplemented_on_tpu
  def test_div_trunc_rounding(self):
    self.do_test_op("div", variant_test_name="trunc_rounding")

  @op_testing.unimplemented_on_tpu
  def test_double(self):
    self.do_test_op("double")

  @op_testing.unimplemented_on_tpu
  def test_dsplit(self):
    self.do_test_op("dsplit")

  @op_testing.unimplemented_on_tpu
  def test_dstack(self):
    self.do_test_op("dstack")

  @op_testing.unimplemented_on_tpu
  def test_empty_like(self):
    self.do_test_op("empty_like")

  @op_testing.unimplemented_on_tpu
  def test_empty_permuted(self):
    self.do_test_op("empty_permuted")

  @op_testing.unimplemented_on_tpu
  def test_erfc(self):
    self.do_test_op("erfc")

  @op_testing.unimplemented_on_tpu
  def test_expand_as(self):
    self.do_test_op("expand_as")

  @op_testing.unimplemented_on_tpu
  def test_expand_copy(self):
    self.do_test_op("expand_copy")

  @op_testing.unimplemented_on_tpu
  def test_fft_fft2(self):
    self.do_test_op("fft.fft2")

  @op_testing.unimplemented_on_tpu
  def test_fft_fftn(self):
    self.do_test_op("fft.fftn")

  @op_testing.unimplemented_on_tpu
  def test_fft_fftshift(self):
    self.do_test_op("fft.fftshift")

  @op_testing.unimplemented_on_tpu
  def test_fft_hfft(self):
    self.do_test_op("fft.hfft")

  @op_testing.unimplemented_on_tpu
  def test_fft_hfft2(self):
    self.do_test_op("fft.hfft2")

  @op_testing.unimplemented_on_tpu
  def test_fft_hfftn(self):
    self.do_test_op("fft.hfftn")

  @op_testing.unimplemented_on_tpu
  def test_fft_ifft2(self):
    self.do_test_op("fft.ifft2")

  @op_testing.unimplemented_on_tpu
  def test_fft_ifftn(self):
    self.do_test_op("fft.ifftn")

  @op_testing.unimplemented_on_tpu
  def test_fft_ifftshift(self):
    self.do_test_op("fft.ifftshift")

  @op_testing.unimplemented_on_tpu
  def test_fft_ihfft(self):
    self.do_test_op("fft.ihfft")

  @op_testing.unimplemented_on_tpu
  def test_fft_ihfft2(self):
    self.do_test_op("fft.ihfft2")

  @op_testing.unimplemented_on_tpu
  def test_fft_ihfftn(self):
    self.do_test_op("fft.ihfftn")

  @op_testing.unimplemented_on_tpu
  def test_fft_irfft2(self):
    self.do_test_op("fft.irfft2")

  @op_testing.unimplemented_on_tpu
  def test_fft_irfftn(self):
    self.do_test_op("fft.irfftn")

  @op_testing.unimplemented_on_tpu
  def test_fft_rfft2(self):
    self.do_test_op("fft.rfft2")

  @op_testing.unimplemented_on_tpu
  def test_fft_rfftn(self):
    self.do_test_op("fft.rfftn")

  @op_testing.unimplemented_on_tpu
  def test_fliplr(self):
    self.do_test_op("fliplr")

  @op_testing.unimplemented_on_tpu
  def test_flipud(self):
    self.do_test_op("flipud")

  @op_testing.unimplemented_on_tpu
  def test_float(self):
    self.do_test_op("float")

  @op_testing.unimplemented_on_tpu
  def test_float_power(self):
    self.do_test_op("float_power")

  @op_testing.unimplemented_on_tpu
  def test_frexp(self):
    self.do_test_op("frexp")

  @op_testing.unimplemented_on_tpu
  def test_gcd(self):
    self.do_test_op("gcd")

  @op_testing.unimplemented_on_tpu
  def test_geometric(self):
    self.do_test_op("geometric")

  @op_testing.unimplemented_on_tpu
  def test_gradient(self):
    self.do_test_op("gradient")

  @op_testing.unimplemented_on_tpu
  def test_grid_sampler_2d(self):
    self.do_test_op("grid_sampler_2d")

  @op_testing.unimplemented_on_tpu
  def test_grid_sampler_3d(self):
    self.do_test_op("grid_sampler_3d")

  @op_testing.unimplemented_on_tpu
  def test_half(self):
    self.do_test_op("half")

  @op_testing.unimplemented_on_tpu
  def test_hash_tensor(self):
    self.do_test_op("hash_tensor")

  @op_testing.unimplemented_on_tpu
  def test_heaviside(self):
    self.do_test_op("heaviside")

  @op_testing.unimplemented_on_tpu
  def test_histogram(self):
    self.do_test_op("histogram")

  @op_testing.unimplemented_on_tpu
  def test_histogramdd(self):
    self.do_test_op("histogramdd")

  @op_testing.unimplemented_on_tpu
  def test_hsplit(self):
    self.do_test_op("hsplit")

  @op_testing.unimplemented_on_tpu
  def test_hstack(self):
    self.do_test_op("hstack")

  @op_testing.unimplemented_on_tpu
  def test_hypot(self):
    self.do_test_op("hypot")

  @op_testing.unimplemented_on_tpu
  def test_i0(self):
    self.do_test_op("i0")

  @op_testing.unimplemented_on_tpu
  def test_igamma(self):
    self.do_test_op("igamma")

  @op_testing.unimplemented_on_tpu
  def test_igammac(self):
    self.do_test_op("igammac")

  @op_testing.unimplemented_on_tpu
  def test_imag(self):
    self.do_test_op("imag")

  @op_testing.unimplemented_on_tpu
  def test_inner(self):
    self.do_test_op("inner")

  @op_testing.unimplemented_on_tpu
  def test_int(self):
    self.do_test_op("int")

  @op_testing.unimplemented_on_tpu
  def test_isclose(self):
    self.do_test_op("isclose")

  @op_testing.unimplemented_on_tpu
  def test_isreal(self):
    self.do_test_op("isreal")

  @op_testing.unimplemented_on_tpu
  def test_istft(self):
    self.do_test_op("istft")

  @op_testing.unimplemented_on_tpu
  def test_item(self):
    self.do_test_op("item")

  @op_testing.unimplemented_on_tpu
  def test_kthvalue(self):
    self.do_test_op("kthvalue")

  @op_testing.unimplemented_on_tpu
  def test_linalg_cholesky(self):
    self.do_test_op("linalg.cholesky")

  @op_testing.unimplemented_on_tpu
  def test_linalg_cond(self):
    self.do_test_op("linalg.cond")

  @op_testing.unimplemented_on_tpu
  def test_linalg_cross(self):
    self.do_test_op("linalg.cross")

  @op_testing.unimplemented_on_tpu
  def test_linalg_det(self):
    self.do_test_op("linalg.det")

  @op_testing.unimplemented_on_tpu
  def test_linalg_diagonal(self):
    self.do_test_op("linalg.diagonal")

  @op_testing.unimplemented_on_tpu
  def test_linalg_eig(self):
    self.do_test_op("linalg.eig")

  @op_testing.unimplemented_on_tpu
  def test_linalg_eigh(self):
    self.do_test_op("linalg.eigh")

  @op_testing.unimplemented_on_tpu
  def test_linalg_eigvals(self):
    self.do_test_op("linalg.eigvals")

  @op_testing.unimplemented_on_tpu
  def test_linalg_eigvalsh(self):
    self.do_test_op("linalg.eigvalsh")

  @op_testing.unimplemented_on_tpu
  def test_linalg_householder_product(self):
    self.do_test_op("linalg.householder_product")

  @op_testing.unimplemented_on_tpu
  def test_linalg_inv_ex(self):
    self.do_test_op("linalg.inv_ex")

  @op_testing.unimplemented_on_tpu
  def test_linalg_ldl_factor(self):
    self.do_test_op("linalg.ldl_factor")

  @op_testing.unimplemented_on_tpu
  def test_linalg_ldl_factor_ex(self):
    self.do_test_op("linalg.ldl_factor_ex")

  @op_testing.unimplemented_on_tpu
  def test_linalg_ldl_solve(self):
    self.do_test_op("linalg.ldl_solve")

  @op_testing.unimplemented_on_tpu
  def test_linalg_lstsq(self):
    self.do_test_op("linalg.lstsq")

  @op_testing.unimplemented_on_tpu
  def test_linalg_lstsq_grad_oriented(self):
    self.do_test_op("linalg.lstsq", variant_test_name="grad_oriented")

  @op_testing.unimplemented_on_tpu
  def test_linalg_lu_factor(self):
    self.do_test_op("linalg.lu_factor")

  @op_testing.unimplemented_on_tpu
  def test_linalg_matrix_norm(self):
    self.do_test_op("linalg.matrix_norm")

  @op_testing.unimplemented_on_tpu
  def test_linalg_matrix_power(self):
    self.do_test_op("linalg.matrix_power")

  @op_testing.unimplemented_on_tpu
  def test_linalg_matrix_rank(self):
    self.do_test_op("linalg.matrix_rank")

  @op_testing.unimplemented_on_tpu
  def test_linalg_matrix_rank_hermitian(self):
    self.do_test_op("linalg.matrix_rank", variant_test_name="hermitian")

  @op_testing.unimplemented_on_tpu
  def test_linalg_matrix_sqrth(self):
    self.do_test_op("linalg.matrix_sqrth")

  @op_testing.unimplemented_on_tpu
  def test_linalg_multi_dot(self):
    self.do_test_op("linalg.multi_dot")

  @op_testing.unimplemented_on_tpu
  def test_linalg_norm(self):
    self.do_test_op("linalg.norm")

  @op_testing.unimplemented_on_tpu
  def test_linalg_norm_subgradients_at_zero(self):
    self.do_test_op("linalg.norm", variant_test_name="subgradients_at_zero")

  @op_testing.unimplemented_on_tpu
  def test_linalg_pinv(self):
    self.do_test_op("linalg.pinv")

  @op_testing.unimplemented_on_tpu
  def test_linalg_pinv_hermitian(self):
    self.do_test_op("linalg.pinv", variant_test_name="hermitian")

  @op_testing.unimplemented_on_tpu
  def test_linalg_pinv_singular(self):
    self.do_test_op("linalg.pinv", variant_test_name="singular")

  @op_testing.unimplemented_on_tpu
  def test_linalg_polar(self):
    self.do_test_op("linalg.polar")

  @op_testing.unimplemented_on_tpu
  def test_linalg_slogdet(self):
    self.do_test_op("linalg.slogdet")

  @op_testing.unimplemented_on_tpu
  def test_linalg_solve(self):
    self.do_test_op("linalg.solve")

  @op_testing.unimplemented_on_tpu
  def test_linalg_tensorinv(self):
    self.do_test_op("linalg.tensorinv")

  @op_testing.unimplemented_on_tpu
  def test_linalg_tensorsolve(self):
    self.do_test_op("linalg.tensorsolve")

  @op_testing.unimplemented_on_tpu
  def test_linalg_vander(self):
    self.do_test_op("linalg.vander")

  @op_testing.unimplemented_on_tpu
  def test_linalg_vecdot(self):
    self.do_test_op("linalg.vecdot")

  @op_testing.unimplemented_on_tpu
  def test_linspace_tensor_overload(self):
    self.do_test_op("linspace", variant_test_name="tensor_overload")

  @op_testing.unimplemented_on_tpu
  def test_log_normal(self):
    self.do_test_op("log_normal")

  @op_testing.unimplemented_on_tpu
  def test_log_softmax_with_dtype(self):
    self.do_test_op("log_softmax", variant_test_name="with_dtype")

  @op_testing.unimplemented_on_tpu
  def test_logaddexp(self):
    self.do_test_op("logaddexp")

  @op_testing.unimplemented_on_tpu
  def test_logaddexp2(self):
    self.do_test_op("logaddexp2")

  @op_testing.unimplemented_on_tpu
  def test_logdet(self):
    self.do_test_op("logdet")

  @op_testing.unimplemented_on_tpu
  def test_logspace(self):
    self.do_test_op("logspace")

  @op_testing.unimplemented_on_tpu
  def test_logspace_tensor_overload(self):
    self.do_test_op("logspace", variant_test_name="tensor_overload")

  @op_testing.unimplemented_on_tpu
  def test_logsumexp(self):
    self.do_test_op("logsumexp")

  @op_testing.unimplemented_on_tpu
  def test_long(self):
    self.do_test_op("long")

  @op_testing.unimplemented_on_tpu
  def test_lu(self):
    self.do_test_op("lu")

  @op_testing.unimplemented_on_tpu
  def test_lu_solve(self):
    self.do_test_op("lu_solve")

  @op_testing.unimplemented_on_tpu
  def test_mH(self):
    self.do_test_op("mH")

  @op_testing.unimplemented_on_tpu
  def test_mT(self):
    self.do_test_op("mT")

  @op_testing.unimplemented_on_tpu
  def test_masked_amax(self):
    self.do_test_op("masked.amax")

  @op_testing.unimplemented_on_tpu
  def test_masked_amin(self):
    self.do_test_op("masked.amin")

  @op_testing.unimplemented_on_tpu
  def test_masked_argmax(self):
    self.do_test_op("masked.argmax")

  @op_testing.unimplemented_on_tpu
  def test_masked_argmin(self):
    self.do_test_op("masked.argmin")

  @op_testing.unimplemented_on_tpu
  def test_masked_cumprod(self):
    self.do_test_op("masked.cumprod")

  @op_testing.unimplemented_on_tpu
  def test_masked_cumsum(self):
    self.do_test_op("masked.cumsum")

  @op_testing.unimplemented_on_tpu
  def test_masked_log_softmax(self):
    self.do_test_op("masked.log_softmax")

  @op_testing.unimplemented_on_tpu
  def test_masked_logaddexp(self):
    self.do_test_op("masked.logaddexp")

  @op_testing.unimplemented_on_tpu
  def test_masked_logsumexp(self):
    self.do_test_op("masked.logsumexp")

  @op_testing.unimplemented_on_tpu
  def test_masked_mean(self):
    self.do_test_op("masked.mean")

  @op_testing.unimplemented_on_tpu
  def test_masked_median(self):
    self.do_test_op("masked.median")

  @op_testing.unimplemented_on_tpu
  def test_masked_norm(self):
    self.do_test_op("masked.norm")

  @op_testing.unimplemented_on_tpu
  def test_masked_normalize(self):
    self.do_test_op("masked.normalize")

  @op_testing.unimplemented_on_tpu
  def test_masked_prod(self):
    self.do_test_op("masked.prod")

  @op_testing.unimplemented_on_tpu
  def test_masked_softmax_2(self):
    self.do_test_op("masked.softmax")

  @op_testing.unimplemented_on_tpu
  def test_masked_softmin(self):
    self.do_test_op("masked.softmin")

  @op_testing.unimplemented_on_tpu
  def test_masked_std(self):
    self.do_test_op("masked.std")

  @op_testing.unimplemented_on_tpu
  def test_masked_sum(self):
    self.do_test_op("masked.sum")

  @op_testing.unimplemented_on_tpu
  def test_masked_var(self):
    self.do_test_op("masked.var")

  @op_testing.unimplemented_on_tpu
  def test_matrix_exp(self):
    self.do_test_op("matrix_exp")

  @op_testing.unimplemented_on_tpu
  def test_max_binary(self):
    self.do_test_op("max", variant_test_name="binary")

  @op_testing.unimplemented_on_tpu
  def test_max_reduction_no_dim(self):
    self.do_test_op("max", variant_test_name="reduction_no_dim")

  @op_testing.unimplemented_on_tpu
  def test_max_pool2d_with_indices_backward(self):
    self.do_test_op("max_pool2d_with_indices_backward")

  @op_testing.unimplemented_on_tpu
  def test_median(self):
    self.do_test_op("median")

  @op_testing.unimplemented_on_tpu
  def test_meshgrid_list_of_tensors(self):
    self.do_test_op("meshgrid", variant_test_name="list_of_tensors")

  @op_testing.unimplemented_on_tpu
  def test_meshgrid_variadic_tensors(self):
    self.do_test_op("meshgrid", variant_test_name="variadic_tensors")

  @op_testing.unimplemented_on_tpu
  def test_min_binary(self):
    self.do_test_op("min", variant_test_name="binary")

  @op_testing.unimplemented_on_tpu
  def test_min_reduction_no_dim(self):
    self.do_test_op("min", variant_test_name="reduction_no_dim")

  @op_testing.unimplemented_on_tpu
  def test_mode(self):
    self.do_test_op("mode")

  @op_testing.unimplemented_on_tpu
  def test_movedim(self):
    self.do_test_op("movedim")

  @op_testing.unimplemented_on_tpu
  def test_msort(self):
    self.do_test_op("msort")

  @op_testing.unimplemented_on_tpu
  def test_mv(self):
    self.do_test_op("mv")

  @op_testing.unimplemented_on_tpu
  def test_mvlgamma_mvlgamma_p_1(self):
    self.do_test_op("mvlgamma", variant_test_name="mvlgamma_p_1")

  @op_testing.unimplemented_on_tpu
  def test_mvlgamma_mvlgamma_p_3(self):
    self.do_test_op("mvlgamma", variant_test_name="mvlgamma_p_3")

  @op_testing.unimplemented_on_tpu
  def test_mvlgamma_mvlgamma_p_5(self):
    self.do_test_op("mvlgamma", variant_test_name="mvlgamma_p_5")

  @op_testing.unimplemented_on_tpu
  def test_nanmean(self):
    self.do_test_op("nanmean")

  @op_testing.unimplemented_on_tpu
  def test_nanmedian(self):
    self.do_test_op("nanmedian")

  @op_testing.unimplemented_on_tpu
  def test_nanquantile(self):
    self.do_test_op("nanquantile")

  @op_testing.unimplemented_on_tpu
  def test_nansum(self):
    self.do_test_op("nansum")

  @op_testing.unimplemented_on_tpu
  def test_narrow(self):
    self.do_test_op("narrow")

  @op_testing.unimplemented_on_tpu
  def test_narrow_copy(self):
    self.do_test_op("narrow_copy")

  @op_testing.unimplemented_on_tpu
  def test_native_group_norm_2(self):
    self.do_test_op("native_group_norm")

  @op_testing.unimplemented_on_tpu
  def test_new_empty(self):
    self.do_test_op("new_empty")

  @op_testing.unimplemented_on_tpu
  def test_new_empty_strided(self):
    self.do_test_op("new_empty_strided")

  @op_testing.unimplemented_on_tpu
  def test_nextafter(self):
    self.do_test_op("nextafter")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_adaptive_avg_pool1d(self):
    self.do_test_op("nn.functional.adaptive_avg_pool1d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_adaptive_max_pool1d(self):
    self.do_test_op("nn.functional.adaptive_max_pool1d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_adaptive_max_pool3d(self):
    self.do_test_op("nn.functional.adaptive_max_pool3d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_alpha_dropout(self):
    self.do_test_op("nn.functional.alpha_dropout")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_batch_norm_without_cudnn(self):
    self.do_test_op(
        "nn.functional.batch_norm", variant_test_name="without_cudnn"
    )

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_bilinear(self):
    self.do_test_op("nn.functional.bilinear")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_binary_cross_entropy_with_logits(self):
    self.do_test_op("nn.functional.binary_cross_entropy_with_logits")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_celu(self):
    self.do_test_op("nn.functional.celu")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_cosine_embedding_loss(self):
    self.do_test_op("nn.functional.cosine_embedding_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_cosine_similarity(self):
    self.do_test_op("nn.functional.cosine_similarity")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_dropout(self):
    self.do_test_op("nn.functional.dropout")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_dropout2d(self):
    self.do_test_op("nn.functional.dropout2d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_dropout3d(self):
    self.do_test_op("nn.functional.dropout3d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_feature_alpha_dropout_with_train(self):
    self.do_test_op(
        "nn.functional.feature_alpha_dropout", variant_test_name="with_train"
    )

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_feature_alpha_dropout_without_train(self):
    self.do_test_op(
        "nn.functional.feature_alpha_dropout", variant_test_name="without_train"
    )

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_fractional_max_pool2d(self):
    self.do_test_op("nn.functional.fractional_max_pool2d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_fractional_max_pool3d(self):
    self.do_test_op("nn.functional.fractional_max_pool3d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_gaussian_nll_loss(self):
    self.do_test_op("nn.functional.gaussian_nll_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_hardshrink(self):
    self.do_test_op("nn.functional.hardshrink")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_hinge_embedding_loss(self):
    self.do_test_op("nn.functional.hinge_embedding_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_huber_loss(self):
    self.do_test_op("nn.functional.huber_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_instance_norm(self):
    self.do_test_op("nn.functional.instance_norm")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_interpolate_area(self):
    self.do_test_op("nn.functional.interpolate", variant_test_name="area")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_interpolate_bilinear(self):
    self.do_test_op("nn.functional.interpolate", variant_test_name="bilinear")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_interpolate_nearest(self):
    self.do_test_op("nn.functional.interpolate", variant_test_name="nearest")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_interpolate_trilinear(self):
    self.do_test_op("nn.functional.interpolate", variant_test_name="trilinear")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_kl_div(self):
    self.do_test_op("nn.functional.kl_div")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_l1_loss(self):
    self.do_test_op("nn.functional.l1_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_layer_norm(self):
    self.do_test_op("nn.functional.layer_norm")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_linear_cross_entropy(self):
    self.do_test_op("nn.functional.linear_cross_entropy")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_linear_cross_entropy_chunked(self):
    self.do_test_op(
        "nn.functional.linear_cross_entropy", variant_test_name="chunked"
    )

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_linear_cross_entropy_chunked_none(self):
    self.do_test_op(
        "nn.functional.linear_cross_entropy", variant_test_name="chunked_none"
    )

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_local_response_norm(self):
    self.do_test_op("nn.functional.local_response_norm")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_margin_ranking_loss(self):
    self.do_test_op("nn.functional.margin_ranking_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_max_unpool1d(self):
    self.do_test_op("nn.functional.max_unpool1d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_max_unpool1d_grad(self):
    self.do_test_op("nn.functional.max_unpool1d", variant_test_name="grad")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_max_unpool2d(self):
    self.do_test_op("nn.functional.max_unpool2d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_max_unpool2d_grad(self):
    self.do_test_op("nn.functional.max_unpool2d", variant_test_name="grad")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_max_unpool3d(self):
    self.do_test_op("nn.functional.max_unpool3d")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_max_unpool3d_grad(self):
    self.do_test_op("nn.functional.max_unpool3d", variant_test_name="grad")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_mish(self):
    self.do_test_op("nn.functional.mish")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_multi_head_attention_forward(self):
    self.do_test_op("nn.functional.multi_head_attention_forward")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_multi_margin_loss(self):
    self.do_test_op("nn.functional.multi_margin_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_multilabel_margin_loss(self):
    self.do_test_op("nn.functional.multilabel_margin_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_multilabel_soft_margin_loss(self):
    self.do_test_op("nn.functional.multilabel_soft_margin_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_normalize(self):
    self.do_test_op("nn.functional.normalize")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_one_hot(self):
    self.do_test_op("nn.functional.one_hot")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_pad_circular(self):
    self.do_test_op("nn.functional.pad", variant_test_name="circular")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_pad_constant(self):
    self.do_test_op("nn.functional.pad", variant_test_name="constant")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_pad_replicate_negative(self):
    self.do_test_op("nn.functional.pad", variant_test_name="replicate_negative")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_pairwise_distance(self):
    self.do_test_op("nn.functional.pairwise_distance")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_pixel_shuffle(self):
    self.do_test_op("nn.functional.pixel_shuffle")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_pixel_unshuffle(self):
    self.do_test_op("nn.functional.pixel_unshuffle")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_poisson_nll_loss(self):
    self.do_test_op("nn.functional.poisson_nll_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_relu6(self):
    self.do_test_op("nn.functional.relu6")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_rrelu(self):
    self.do_test_op("nn.functional.rrelu")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_selu(self):
    self.do_test_op("nn.functional.selu")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_silu_complex(self):
    self.do_test_op("nn.functional.silu", variant_test_name="complex")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_smooth_l1_loss(self):
    self.do_test_op("nn.functional.smooth_l1_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_soft_margin_loss(self):
    self.do_test_op("nn.functional.soft_margin_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_softmin(self):
    self.do_test_op("nn.functional.softmin")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_softmin_with_dtype(self):
    self.do_test_op("nn.functional.softmin", variant_test_name="with_dtype")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_softshrink(self):
    self.do_test_op("nn.functional.softshrink")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_softsign(self):
    self.do_test_op("nn.functional.softsign")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_tanhshrink(self):
    self.do_test_op("nn.functional.tanhshrink")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_triplet_margin_loss(self):
    self.do_test_op("nn.functional.triplet_margin_loss")

  @op_testing.unimplemented_on_tpu
  def test_nn_functional_triplet_margin_with_distance_loss(self):
    self.do_test_op("nn.functional.triplet_margin_with_distance_loss")

  @op_testing.unimplemented_on_tpu
  def test_nonzero_static(self):
    self.do_test_op("nonzero_static")

  @op_testing.unimplemented_on_tpu
  def test_norm_fro(self):
    self.do_test_op("norm", variant_test_name="fro")

  @op_testing.unimplemented_on_tpu
  def test_norm_inf(self):
    self.do_test_op("norm", variant_test_name="inf")

  @op_testing.unimplemented_on_tpu
  def test_norm_nuc(self):
    self.do_test_op("norm", variant_test_name="nuc")

  @op_testing.unimplemented_on_tpu
  def test_normal(self):
    self.do_test_op("normal", variant_test_name="")

  @op_testing.unimplemented_on_tpu
  def test_normal_number_mean(self):
    self.do_test_op("normal", variant_test_name="number_mean")

  @op_testing.unimplemented_on_tpu
  def test_ormqr(self):
    self.do_test_op("ormqr")

  @op_testing.unimplemented_on_tpu
  def test_outer(self):
    self.do_test_op("outer")

  @op_testing.unimplemented_on_tpu
  def test_pca_lowrank(self):
    self.do_test_op("pca_lowrank")

  @op_testing.unimplemented_on_tpu
  def test_permute_copy(self):
    self.do_test_op("permute_copy")

  @op_testing.unimplemented_on_tpu
  def test_pinverse(self):
    self.do_test_op("pinverse")

  @op_testing.unimplemented_on_tpu
  def test_polygamma_polygamma_n_1(self):
    self.do_test_op("polygamma", variant_test_name="polygamma_n_1")

  @op_testing.unimplemented_on_tpu
  def test_polygamma_polygamma_n_2(self):
    self.do_test_op("polygamma", variant_test_name="polygamma_n_2")

  @op_testing.unimplemented_on_tpu
  def test_polygamma_polygamma_n_3(self):
    self.do_test_op("polygamma", variant_test_name="polygamma_n_3")

  @op_testing.unimplemented_on_tpu
  def test_polygamma_polygamma_n_4(self):
    self.do_test_op("polygamma", variant_test_name="polygamma_n_4")

  @op_testing.unimplemented_on_tpu
  def test_positive(self):
    self.do_test_op("positive")

  @op_testing.unimplemented_on_tpu
  def test_quantile(self):
    self.do_test_op("quantile")

  @op_testing.unimplemented_on_tpu
  def test_rad2deg(self):
    self.do_test_op("rad2deg")

  @op_testing.unimplemented_on_tpu
  def test_rand_like(self):
    self.do_test_op("rand_like")

  @op_testing.unimplemented_on_tpu
  def test_randint_like(self):
    self.do_test_op("randint_like")

  @op_testing.unimplemented_on_tpu
  def test_randn_like(self):
    self.do_test_op("randn_like")

  @op_testing.unimplemented_on_tpu
  def test_ravel(self):
    self.do_test_op("ravel")

  @op_testing.unimplemented_on_tpu
  def test_real(self):
    self.do_test_op("real")

  @op_testing.unimplemented_on_tpu
  def test_renorm(self):
    self.do_test_op("renorm")

  @op_testing.unimplemented_on_tpu
  def test_reshape_as(self):
    self.do_test_op("reshape_as")

  @op_testing.unimplemented_on_tpu
  def test_resize_as(self):
    self.do_test_op("resize_as_")

  @op_testing.unimplemented_on_tpu
  def test_rot90(self):
    self.do_test_op("rot90")

  @op_testing.unimplemented_on_tpu
  def test_round_decimals_0(self):
    self.do_test_op("round", variant_test_name="decimals_0")

  @op_testing.unimplemented_on_tpu
  def test_round_decimals_3(self):
    self.do_test_op("round", variant_test_name="decimals_3")

  @op_testing.unimplemented_on_tpu
  def test_round_decimals_neg_3(self):
    self.do_test_op("round", variant_test_name="decimals_neg_3")

  @op_testing.unimplemented_on_tpu
  def test_scatter_reduce_amax(self):
    self.do_test_op("scatter_reduce", variant_test_name="amax")

  @op_testing.unimplemented_on_tpu
  def test_scatter_reduce_amin(self):
    self.do_test_op("scatter_reduce", variant_test_name="amin")

  @op_testing.unimplemented_on_tpu
  def test_scatter_reduce_mean(self):
    self.do_test_op("scatter_reduce", variant_test_name="mean")

  @op_testing.unimplemented_on_tpu
  def test_scatter_reduce_prod(self):
    self.do_test_op("scatter_reduce", variant_test_name="prod")

  @op_testing.unimplemented_on_tpu
  def test_short(self):
    self.do_test_op("short")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_bartlett(self):
    self.do_test_op("signal.windows.bartlett")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_blackman(self):
    self.do_test_op("signal.windows.blackman")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_cosine(self):
    self.do_test_op("signal.windows.cosine")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_exponential(self):
    self.do_test_op("signal.windows.exponential")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_gaussian(self):
    self.do_test_op("signal.windows.gaussian")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_general_cosine(self):
    self.do_test_op("signal.windows.general_cosine")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_general_hamming(self):
    self.do_test_op("signal.windows.general_hamming")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_hamming(self):
    self.do_test_op("signal.windows.hamming")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_hann(self):
    self.do_test_op("signal.windows.hann")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_kaiser(self):
    self.do_test_op("signal.windows.kaiser")

  @op_testing.unimplemented_on_tpu
  def test_signal_windows_nuttall(self):
    self.do_test_op("signal.windows.nuttall")

  @op_testing.unimplemented_on_tpu
  def test_sinc(self):
    self.do_test_op("sinc")

  @op_testing.unimplemented_on_tpu
  def test_softmax_with_dtype(self):
    self.do_test_op("softmax", variant_test_name="with_dtype")

  @op_testing.unimplemented_on_tpu
  def test_sparse_mm_reduce(self):
    self.do_test_op("sparse.mm", variant_test_name="reduce")

  @op_testing.unimplemented_on_tpu
  def test_sparse_sampled_addmm(self):
    self.do_test_op("sparse.sampled_addmm")

  @op_testing.unimplemented_on_tpu
  def test_special_airy_ai(self):
    self.do_test_op("special.airy_ai")

  @op_testing.unimplemented_on_tpu
  def test_special_bessel_j0(self):
    self.do_test_op("special.bessel_j0")

  @op_testing.unimplemented_on_tpu
  def test_special_bessel_j1(self):
    self.do_test_op("special.bessel_j1")

  @op_testing.unimplemented_on_tpu
  def test_special_bessel_y0(self):
    self.do_test_op("special.bessel_y0")

  @op_testing.unimplemented_on_tpu
  def test_special_bessel_y1(self):
    self.do_test_op("special.bessel_y1")

  @op_testing.unimplemented_on_tpu
  def test_special_chebyshev_polynomial_t(self):
    self.do_test_op("special.chebyshev_polynomial_t")

  @op_testing.unimplemented_on_tpu
  def test_special_chebyshev_polynomial_u(self):
    self.do_test_op("special.chebyshev_polynomial_u")

  @op_testing.unimplemented_on_tpu
  def test_special_chebyshev_polynomial_v(self):
    self.do_test_op("special.chebyshev_polynomial_v")

  @op_testing.unimplemented_on_tpu
  def test_special_chebyshev_polynomial_w(self):
    self.do_test_op("special.chebyshev_polynomial_w")

  @op_testing.unimplemented_on_tpu
  def test_special_entr(self):
    self.do_test_op("special.entr")

  @op_testing.unimplemented_on_tpu
  def test_special_erfcx(self):
    self.do_test_op("special.erfcx")

  @op_testing.unimplemented_on_tpu
  def test_special_hermite_polynomial_h(self):
    self.do_test_op("special.hermite_polynomial_h")

  @op_testing.unimplemented_on_tpu
  def test_special_hermite_polynomial_he(self):
    self.do_test_op("special.hermite_polynomial_he")

  @op_testing.unimplemented_on_tpu
  def test_special_i0e(self):
    self.do_test_op("special.i0e")

  @op_testing.unimplemented_on_tpu
  def test_special_i1(self):
    self.do_test_op("special.i1")

  @op_testing.unimplemented_on_tpu
  def test_special_i1e(self):
    self.do_test_op("special.i1e")

  @op_testing.unimplemented_on_tpu
  def test_special_laguerre_polynomial_l(self):
    self.do_test_op("special.laguerre_polynomial_l")

  @op_testing.unimplemented_on_tpu
  def test_special_legendre_polynomial_p(self):
    self.do_test_op("special.legendre_polynomial_p")

  @op_testing.unimplemented_on_tpu
  def test_special_log_ndtr(self):
    self.do_test_op("special.log_ndtr")

  @op_testing.unimplemented_on_tpu
  def test_special_modified_bessel_i0(self):
    self.do_test_op("special.modified_bessel_i0")

  @op_testing.unimplemented_on_tpu
  def test_special_modified_bessel_i1(self):
    self.do_test_op("special.modified_bessel_i1")

  @op_testing.unimplemented_on_tpu
  def test_special_modified_bessel_k0(self):
    self.do_test_op("special.modified_bessel_k0")

  @op_testing.unimplemented_on_tpu
  def test_special_modified_bessel_k1(self):
    self.do_test_op("special.modified_bessel_k1")

  @op_testing.unimplemented_on_tpu
  def test_special_ndtr(self):
    self.do_test_op("special.ndtr")

  @op_testing.unimplemented_on_tpu
  def test_special_ndtri(self):
    self.do_test_op("special.ndtri")

  @op_testing.unimplemented_on_tpu
  def test_special_polygamma_special_polygamma_n_0(self):
    self.do_test_op(
        "special.polygamma", variant_test_name="special_polygamma_n_0"
    )

  @op_testing.unimplemented_on_tpu
  def test_special_scaled_modified_bessel_k0(self):
    self.do_test_op("special.scaled_modified_bessel_k0")

  @op_testing.unimplemented_on_tpu
  def test_special_scaled_modified_bessel_k1(self):
    self.do_test_op("special.scaled_modified_bessel_k1")

  @op_testing.unimplemented_on_tpu
  def test_special_shifted_chebyshev_polynomial_t(self):
    self.do_test_op("special.shifted_chebyshev_polynomial_t")

  @op_testing.unimplemented_on_tpu
  def test_special_shifted_chebyshev_polynomial_u(self):
    self.do_test_op("special.shifted_chebyshev_polynomial_u")

  @op_testing.unimplemented_on_tpu
  def test_special_shifted_chebyshev_polynomial_v(self):
    self.do_test_op("special.shifted_chebyshev_polynomial_v")

  @op_testing.unimplemented_on_tpu
  def test_special_shifted_chebyshev_polynomial_w(self):
    self.do_test_op("special.shifted_chebyshev_polynomial_w")

  @op_testing.unimplemented_on_tpu
  def test_special_spherical_bessel_j0(self):
    self.do_test_op("special.spherical_bessel_j0")

  @op_testing.unimplemented_on_tpu
  def test_special_xlog1py(self):
    self.do_test_op("special.xlog1py")

  @op_testing.unimplemented_on_tpu
  def test_special_zeta(self):
    self.do_test_op("special.zeta")

  @op_testing.unimplemented_on_tpu
  def test_split_list_args(self):
    self.do_test_op("split", variant_test_name="list_args")

  @op_testing.unimplemented_on_tpu
  def test_square(self):
    self.do_test_op("square")

  @op_testing.unimplemented_on_tpu
  def test_squeeze_multiple(self):
    self.do_test_op("squeeze", variant_test_name="multiple")

  @op_testing.unimplemented_on_tpu
  def test_std_unbiased(self):
    self.do_test_op("std", variant_test_name="unbiased")

  @op_testing.unimplemented_on_tpu
  def test_std_mean(self):
    self.do_test_op("std_mean")

  @op_testing.unimplemented_on_tpu
  def test_std_mean_unbiased(self):
    self.do_test_op("std_mean", variant_test_name="unbiased")

  @op_testing.unimplemented_on_tpu
  def test_stft(self):
    self.do_test_op("stft")

  @op_testing.unimplemented_on_tpu
  def test_sum_to_size(self):
    self.do_test_op("sum_to_size")

  @op_testing.unimplemented_on_tpu
  def test_svd(self):
    self.do_test_op("svd")

  @op_testing.unimplemented_on_tpu
  def test_svd_lowrank(self):
    self.do_test_op("svd_lowrank")

  @op_testing.unimplemented_on_tpu
  def test_t_copy(self):
    self.do_test_op("t_copy")

  @op_testing.unimplemented_on_tpu
  def test_take_along_dim(self):
    self.do_test_op("take_along_dim")

  @op_testing.unimplemented_on_tpu
  def test_tensor_split(self):
    self.do_test_op("tensor_split")

  @op_testing.unimplemented_on_tpu
  def test_tensordot(self):
    self.do_test_op("tensordot")

  @op_testing.unimplemented_on_tpu
  def test_tile(self):
    self.do_test_op("tile")

  @op_testing.unimplemented_on_tpu
  def test_to_sparse(self):
    self.do_test_op("to_sparse")

  @op_testing.unimplemented_on_tpu
  def test_torch__scaled_mm(self):
    self.do_test_op("torch._scaled_mm", variant_test_name="")

  @op_testing.unimplemented_on_tpu
  def test_torch_ops_aten__efficient_attention_forward(self):
    self.do_test_op("torch.ops.aten._efficient_attention_forward")

  @op_testing.unimplemented_on_tpu
  def test_torch_ops_aten__flash_attention_forward(self):
    self.do_test_op("torch.ops.aten._flash_attention_forward")

  @op_testing.unimplemented_on_tpu
  def test_torch_ops_aten__scaled_dot_product_flash_attention_for_cpu(self):
    self.do_test_op(
        "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu"
    )

  @op_testing.unimplemented_on_tpu
  def test_trace(self):
    self.do_test_op("trace")

  @op_testing.unimplemented_on_tpu
  def test_transpose_copy(self):
    self.do_test_op("transpose_copy")

  @op_testing.unimplemented_on_tpu
  def test_trapezoid(self):
    self.do_test_op("trapezoid")

  @op_testing.unimplemented_on_tpu
  def test_trapz(self):
    self.do_test_op("trapz")

  @op_testing.unimplemented_on_tpu
  def test_triangular_solve(self):
    self.do_test_op("triangular_solve")

  @op_testing.unimplemented_on_tpu
  def test_triu_indices(self):
    self.do_test_op("triu_indices")

  @op_testing.unimplemented_on_tpu
  def test_true_divide(self):
    self.do_test_op("true_divide")

  @op_testing.unimplemented_on_tpu
  def test_unbind_copy(self):
    self.do_test_op("unbind_copy")

  @op_testing.unimplemented_on_tpu
  def test_unflatten(self):
    self.do_test_op("unflatten")

  @op_testing.unimplemented_on_tpu
  def test_unfold_copy(self):
    self.do_test_op("unfold_copy")

  @op_testing.unimplemented_on_tpu
  def test_unravel_index(self):
    self.do_test_op("unravel_index")

  @op_testing.unimplemented_on_tpu
  def test_unsafe_chunk(self):
    self.do_test_op("unsafe_chunk")

  @op_testing.unimplemented_on_tpu
  def test_unsafe_split(self):
    self.do_test_op("unsafe_split")

  @op_testing.unimplemented_on_tpu
  def test_var_unbiased(self):
    self.do_test_op("var", variant_test_name="unbiased")

  @op_testing.unimplemented_on_tpu
  def test_var_mean_unbiased(self):
    self.do_test_op("var_mean", variant_test_name="unbiased")

  @op_testing.unimplemented_on_tpu
  def test_view_as(self):
    self.do_test_op("view_as")

  @op_testing.unimplemented_on_tpu
  def test_view_copy(self):
    self.do_test_op("view_copy")

  @op_testing.unimplemented_on_tpu
  def test_vsplit(self):
    self.do_test_op("vsplit")

  @op_testing.unimplemented_on_tpu
  def test_vstack(self):
    self.do_test_op("vstack")


def setUpModule() -> None:
  """Called by absltest.main() after flags are parsed but before tests are run."""

  op_testing.set_up_test_module()

  if (
      op_testing._torch_tpu_vs_gpu_mode() or op_testing._gen_gpu_golden_mode()
  ) and op_testing.is_compiled_mode():
    assert torch.tpu.allow_excess_precision  # pytype: disable=module-attr
  else:
    assert not torch.tpu.allow_excess_precision  # pytype: disable=module-attr


def tearDownModule() -> None:
  """Called by absltest.main() after running all tests."""

  op_testing.tear_down_test_module()


if __name__ == "__main__":
  absltest.main()
