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

"""Testing framework for ops.

See go/torch-tpu-op-test for more details.
"""

import builtins
import collections
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
import copy
import dataclasses
import enum
import functools
import gzip
import importlib
import itertools
import logging  # PYTHON_LOGGING_OK=Setting module log level for compile testing
import os
import plistlib
import random
import re
import statistics
import sys
import time
import traceback
import typing
from typing import Any, Final
import unittest

from absl import flags
from etils import epath
import safetensors.torch as st
import sympy
import torch
from torch._subclasses import fake_tensor
from torch.fx.experimental import symbolic_shapes
from torch.testing._internal import autograd_function_db
from torch.testing._internal import common_dtype
from torch.testing._internal import common_methods_invocations
from torch.testing._internal import common_quantized
from torch.testing._internal import common_utils
from torch.testing._internal import custom_op_db
from torch.testing._internal import hop_db
from torch.testing._internal.opinfo import core
from torch.testing._internal.opinfo.definitions import _masked
from torch.testing._internal.opinfo.definitions import fft
from torch.testing._internal.opinfo.definitions import linalg
from torch.utils import _pytree
from torch_tpu._internal import compile as tt_compile
from torch_tpu._internal import compiler_options as compiler
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.utils import utils
from tests import dynamism_test_utils
from tests import quantize_utils
from tests import seed_test_utils
from tests import test_utils as tests_test_utils
import yaml

# In this file, we use the following naming convention for variables:
# - golden_*: a value for the device used for computing the golden results
#   (either CPU or GPU)
# - torch_tpu_*: a value for the TorchTPU device

TestCase = common_utils.TestCase
OpInfo = core.OpInfo  # pylint: disable=protected-access
SampleInput = core.SampleInput  # pylint: disable=protected-access
CheckValueMode = test_utils.CheckValueMode
Tolerance = test_utils.Tolerance
DynamismMode = dynamism_test_utils.DynamismMode

AccuracyOverrides = Mapping[str, Mapping[torch.dtype, Mapping[str, Tolerance]]]

FilterFn = Callable[..., str | None]
MarkDynamicFn = Callable[..., None]

GPU_SM90: Final[str] = "gpu_sm90"
GPU_SM100: Final[str] = "gpu_sm100"

# GPU architecture that records the goldens of tests not tagged with
# @record_gpu_goldens_on.
_DEFAULT_GPU_GOLDEN_ARCH: Final[str] = GPU_SM100
# Test method attribute set by @record_gpu_goldens_on.
_GPU_GOLDEN_ARCH_ATTR: Final[str] = "gpu_golden_arch"
# Test method attribute set by @test_unbounded_dynamic.
_UNBOUNDED_DYNAMIC_ATTR: Final[str] = "unbounded_dynamic_spec"

_GPU_GOLDEN_HOPPER_SUFFIX: Final[str] = "_hopper"

# Suffix of the directory artifact holding per-test-case golden files, see
# split_gpu_goldens() in BUILD.
_GPU_GOLDEN_PER_TEST_CASE_SUFFIX: Final[str] = "_per_test_case"

# Identifies a list of golden samples: (test case name, op variant, dtype key).
_GoldenKey = tuple[str, str, str]


def _validate_tolerances(
    rtol: Tolerance | None = None,
    atol: Tolerance | None = None,
) -> None:
  """Validates that rtol and atol, if specified, are > 0."""
  if rtol is not None and isinstance(rtol, (int, float)) and not (rtol > 0):
    raise ValueError(f"rtol must be > 0, got {rtol}")
  if atol is not None and isinstance(atol, (int, float)) and not (atol > 0):
    raise ValueError(f"atol must be > 0, got {atol}")


def _validate_accuracy_overrides(overrides: AccuracyOverrides) -> None:
  """Validates that all tolerances in the overrides are > 0."""
  for op_overrides in overrides.values():
    for tol_dict in op_overrides.values():
      _validate_tolerances(
          rtol=tol_dict.get("rtol"),
          atol=tol_dict.get("atol"),
      )


class TestMode(enum.Enum):
  """Mode to run the test in."""

  TORCH_TPU_VS_GPU = (  # Compare TorchTPU results to recorded GPU results.
      "torch_tpu_vs_gpu"
  )
  GEN_GPU_GOLDEN = "gen_gpu_golden"  # Generate GPU golden results.
  PERF = "perf"  # Measure op performance.
  DUMP_DTYPE_EXCLUSIONS = (  # Dump dtype exclusions in `do_test_op`.
      "dump_dtype_exclusions"
  )


_TEST_MODE: Final[flags.FlagHolder[TestMode]] = flags.DEFINE_enum_class(
    "test_mode", TestMode.TORCH_TPU_VS_GPU, TestMode, "Mode to run the test in."
)

# The --dtypes flag accepts a comma-separated list of dtypes. We define this
# as a list of strings, where each string is either the name of a dtype or
# "all".
_DTYPES: Final[flags.FlagHolder[Sequence[str]]] = flags.DEFINE_list(
    "dtypes",
    ["all"],
    "dtypes to test. Can be a comma-separated list of dtype names (e.g. "
    " 'float32,bfloat16'), or a single 'all'.",
)

_MAX_SAMPLES_PER_OP_DTYPE: Final[flags.FlagHolder[int]] = flags.DEFINE_integer(
    "max_samples_per_op_dtype",
    -1,
    "Maximum number of samples to test for each (op variant, dtype) pair. "
    "Negative values mean no limit.",
)

_COMPUTE_GRAD: Final[flags.FlagHolder[bool]] = flags.DEFINE_bool(
    "compute_grad",
    False,
    "Compute the gradient of the op instead of the normal op result.",
)

_USE_COMPILED: Final[flags.FlagHolder[bool]] = flags.DEFINE_bool(
    "use_compiled",
    False,
    "Use compiled torch_tpu backend.",
)

_UNBOUNDED_DYNAMIC_TESTS_ONLY: Final[flags.FlagHolder[bool]] = (
    flags.DEFINE_bool(
        "unbounded_dynamic_tests_only",
        False,
        "Run only the tests decorated with @test_unbounded_dynamic, and check"
        " that their matching subtests are served by an unbounded dynamic"
        " kernel. Requires TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS=1.",
    )
)

_IGNORE_ACCURACY_OVERRIDES: Final[flags.FlagHolder[bool]] = flags.DEFINE_bool(
    "ignore_accuracy_overrides",
    False,
    "If set, accuracy overrides will be ignored, forcing strict checks.",
)

_CHECK_DYNAMISM_USING_SEED: Final[flags.FlagHolder[int]] = flags.DEFINE_integer(
    "check_dynamism_using_seed",
    0,
    "If non-zero, marks a random dimension as bounded dynamic and runs the ops "
    "test, comparing the bounded TorchTPU result with the golden result. "
    "Only picks dimensions with size >=2, "
    "and bounds for size+10. Allows for explicitly setting the seed value for "
    "reproducibility, or -1 for random seed.",
)

_CHECK_SYMBOLIC_SHAPES: Final[flags.FlagHolder[bool]] = flags.DEFINE_bool(
    "check_symbolic_shapes",
    False,
    "If set, instead of comparing results with the golden results, runs each"
    " golden input on FakeTensors whose dims of size >= 2 are symbolic, and"
    " fails if TorchTPU specializes (pins to a constant) or throws on a"
    " symbolic dim that CPU keeps symbolic. Such specializations make"
    " torch.compile recompile for every new input shape.",
)

_OPT_LEVEL: Final[flags.FlagHolder[str]] = flags.DEFINE_string(
    "opt_level",
    "",
    "Optimization level to run the test with. Valid options are '' (default),"
    " 'O0', 'O1', 'O2', 'O3', and 'O4'. Only used in the perf mode.",
)

_PRINT_OP_INPUTS: Final[flags.FlagHolder[int]] = flags.DEFINE_integer(
    "print_op_inputs",
    0,
    "Print the inputs and reproducer function for the ops being tested. Useful"
    " for debugging. Set to 0 for no printing, 1 for to print op input"
    " summaries (no data), or 2 to include verbose / real tensor data for"
    " reproducers.",
)

_UPDATE_PERF_DATA: Final[flags.FlagHolder[bool]] = flags.DEFINE_bool(
    "update_perf_data",
    False,
    "Update the output perf data files. Currently only used in the perf mode.",
)

_ANALYZE: Final[flags.FlagHolder[bool]] = flags.DEFINE_bool(
    "analyze",
    False,
    "Analyze the performance results stored in `--perf_dir`.",
)

_PERF_DIR: Final[flags.FlagHolder[str]] = flags.DEFINE_string(
    "perf_dir",
    "",
    "Directory for storing the performance results. Must be a "
    "non-empty string in the perf mode.",
)

_GOLDEN_DATA_BASE_DIR: Final[flags.FlagHolder[str]] = flags.DEFINE_string(
    "golden_data_base_dir",
    "",
    "Directory containing golden data files. If not specified, standard test"
    " and runfiles locations are searched.",
)


# Example of how to use --base_perf_dir to study the impact of a change on op
# performance:
_COMPARISON_TO_BASE_EXAMPLE: Final[str] = r"""
# Gather baseline results (i.e., without change). The results will be stored in
# an output directory with the current timestamp. See op_perf.py for more
# details.
op_perf.py -- --num_runs_per_subtest=20

# Make changes to test.

# Gather current results (i.e., with change). The results will be stored in a
# similar directory with a more recent timestamp.
op_perf.py -- --num_runs_per_subtest=20

# Note the directories where the results were stored for each run. See the logic
# creating the perf_dir directory paths in op_perf.py.

# Run the comparison analysis.
blaze test -c opt torch_tpu/tests:ops_test_perf                       \
                  --test_arg=--base_perf_dir=/path/to/first/perf_dir  \
                  --test_arg=--perf_dir=/path/to/second/perf_dir      \
                  --test_arg=--analyze                                \
                  --test_arg=--opt_level=""
"""

_BASE_PERF_DIR: Final[flags.FlagHolder[str]] = flags.DEFINE_string(
    "base_perf_dir",
    "",
    "Directory where the baseline performance results are stored (i.e., the"
    " --perf_dir from a prior run). Only relevant when using `--analyze`. If"
    " specified, the analysis will include a comparison of the results in"
    " `--perf_dir` to the results in this directory.",
)

_NUM_RUNS_PER_SUBTEST: Final[flags.FlagHolder[int]] = flags.DEFINE_integer(
    "num_runs_per_subtest",
    1,
    "Number of times to run each sub-test with the same input. Useful for "
    "performance testing or checking flakiness on the same input. Does not "
    "generate new inputs.",
)


class OpVariant(enum.Enum):
  """Variant of an op to test."""

  BASE = "base"  # op(...)
  INPLACE = "inplace"  # op_(...)
  OUT = "out"  # op(..., out=...)


@dataclasses.dataclass(frozen=True)
class CastPairs:
  """Container for a pair of dtypes representing an illegal cast and an allowed cast.

  Attributes:
    illegal_pair: A tuple `(src_dtype, dst_dtype)` where casting `src_dtype` to
      `dst_dtype` is illegal according to `torch.can_cast()`, or `None` if no
      such pair exists in the provided dtypes.
    allowed_pair: A tuple `(src_dtype, dst_dtype)` where casting `src_dtype` to
      `dst_dtype` is allowed according to `torch.can_cast()`, or `None` if no
      such pair exists in the provided dtypes.
  """

  illegal_pair: tuple[torch.dtype, torch.dtype] | None = None
  allowed_pair: tuple[torch.dtype, torch.dtype] | None = None


def find_cast_pairs(
    dtypes: Iterable[torch.dtype],
) -> CastPairs:
  """Finds one allowed cast pair and one illegal cast pair from a set of dtypes.

  Args:
    dtypes: An iterable of `torch.dtype`s to be tested. The function considers
      all ordered pairs `(src, dst)` of distinct dtypes from this iterable.

  Returns:
    A `CastPairs` instance containing `illegal_pair` and `allowed_pair`.

    `illegal_pair` or `allowed_pair` (or both) will be `None` in the following
    cases:
    - `dtypes` contains fewer than 2 distinct dtypes (no pairs can be formed).
    - `dtypes` contains only dtypes where all pairs are allowed casts (e.g.,
      `[float32, float64]` has no illegal cast, so `illegal_pair` is `None`).
    - `dtypes` contains only dtypes where all pairs are illegal casts (e.g.,
      `[float32, int32]` has no allowed cast, so `allowed_pair` is `None`).
  """
  dtypes_list = list(dict.fromkeys(dtypes))

  illegal_pair = None
  allowed_pair = None

  # permutations considers ordered pairs, so (a, b) and (b, a)
  # are tested separately.
  for src, dst in itertools.permutations(dtypes_list, 2):
    if torch.can_cast(src, dst):
      allowed_pair = allowed_pair or (src, dst)
    else:
      illegal_pair = illegal_pair or (src, dst)

    if illegal_pair is not None and allowed_pair is not None:
      break

  return CastPairs(illegal_pair=illegal_pair, allowed_pair=allowed_pair)


# Operators known to fail out-variant dtype casting (either by rejecting valid
# promotional casts or permitting illegal downcasts without throwing an exception).
# We skip out-variant mismatched dtype testing for these ops until their C++
# out-kernels are updated with proper ATen type checking.
_OUT_DTYPE_CAST_KNOWN_FAILURES: Final[set[str]] = {
    # go/keep-sorted start
    "addcdiv",
    "addcmul",
    "addmv",
    "bitwise_not",
    # TODO: Remove "histc" once find_cast_pairs avoids real->complex pairs.
    "histc",
    "index_reduce",
    "lerp",
    "sparse_coo_mul",
    "sparse_coo_neg",
    # TODO: Remove "std" once find_cast_pairs avoids real->complex pairs.
    "std",
    "var",
    # go/keep-sorted end
}

# TODO(b/568080282): Dynamo FakeTensor rejects complex64 -> float64 out-variant
# dtype casts for these ops even though GPU golden succeeded.
_COMPILED_OUT_DTYPE_CAST_KNOWN_FAILURES: Final[set[str]] = {
    # go/keep-sorted start
    "abs",
    "mean",
    # go/keep-sorted end
}


def _sample_inputs_thnn_fused_rnn_cell(
    op_info: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    num_gates: int | None = None,
    **kwargs,
):
  """Sample inputs generator for THNN fused LSTM and GRU cells.

  Args:
    op_info: The OpInfo object for the tested operator.
    device: The target device for generated tensors.
    dtype: The data type for generated tensors.
    requires_grad: Whether generated tensors should require gradients.
    num_gates: The number of gates in the RNN cell (4 for LSTM, 3 for GRU). If
      None, inferred from op_info.name.
    **kwargs: Additional unused keyword arguments.

  Yields:
    SampleInput instances for testing fused RNN cells.
  """
  del kwargs
  if num_gates is None:
    num_gates = 4 if "lstm" in op_info.name else 3
  # Although fused LSTM/GRU cells only support floating-point dtypes,
  # we must generate valid integer/bool sample tensors when requested
  # by the test harness (e.g. when iterating across all op_db dtypes
  # without explicit exclude_dtypes). Otherwise, calling `torch.randn`
  # with integral dtypes throws NotImplementedError during sample
  # generation (`sample_inputs()`), crashing the test harness before
  # the op is invoked. Generating valid inputs here allows the op to
  # run, raise its expected unsupported-dtype exception on CPU/TPU,
  # and cleanly verify failure handling (`_assert_failure_consistency`).
  if dtype.is_floating_point or dtype.is_complex:
    make_arg = functools.partial(
        torch.randn, device=device, dtype=dtype, requires_grad=requires_grad
    )
  elif dtype == torch.bool:
    make_arg = lambda size: torch.randint(
        0, 2, size, device=device, dtype=dtype
    )
  elif dtype == torch.uint8:
    make_arg = lambda size: torch.randint(
        0, 10, size, device=device, dtype=dtype
    )
  else:
    make_arg = lambda size: torch.randint(
        -10, 10, size, device=device, dtype=dtype
    )
  batch, hidden = 4, 8
  ig = make_arg((batch, num_gates * hidden))
  hg = make_arg((batch, num_gates * hidden))
  hx = make_arg((batch, hidden))
  ib = make_arg((num_gates * hidden,))
  hb = make_arg((num_gates * hidden,))
  yield SampleInput(ig, args=(hg, hx, ib, hb))
  yield SampleInput(
      ig.detach().clone().requires_grad_(requires_grad),
      args=(
          hg.detach().clone().requires_grad_(requires_grad),
          hx.detach().clone().requires_grad_(requires_grad),
          None,
          None,
      ),
  )


def _sample_inputs_native_multi_head_attention(
    op_info: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs,
):
  """Sample inputs generator for native_multi_head_attention."""
  del op_info, kwargs
  if not dtype.is_floating_point:
    return
  make_arg = functools.partial(
      torch.randn, device=device, dtype=dtype, requires_grad=requires_grad
  )

  embed_dim = 8
  num_head = 2
  batch_size = 2
  seq_len = 4

  query = make_arg((batch_size, seq_len, embed_dim))
  key = make_arg((batch_size, seq_len, embed_dim))
  value = make_arg((batch_size, seq_len, embed_dim))
  qkv_weight = make_arg((3 * embed_dim, embed_dim))
  qkv_bias = make_arg((3 * embed_dim,))
  proj_weight = make_arg((embed_dim, embed_dim))
  proj_bias = make_arg((embed_dim,))

  # Case 1: need_weights=True, average_attn_weights=True, no mask
  yield SampleInput(
      query,
      args=(
          key,
          value,
          embed_dim,
          num_head,
          qkv_weight,
          qkv_bias,
          proj_weight,
          proj_bias,
          None,
          True,
          True,
          None,
      ),
  )

  # Case 2: need_weights=True, average_attn_weights=False, with bool mask
  mask = torch.randint(
      0, 2, (seq_len, seq_len), device=device, dtype=torch.bool
  )
  yield SampleInput(
      query.detach().clone().requires_grad_(requires_grad),
      args=(
          key.detach().clone().requires_grad_(requires_grad),
          value.detach().clone().requires_grad_(requires_grad),
          embed_dim,
          num_head,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          mask,
          True,
          False,
          0,
      ),
  )

  # Case 3: need_weights=False
  yield SampleInput(
      query.detach().clone().requires_grad_(requires_grad),
      args=(
          key.detach().clone().requires_grad_(requires_grad),
          value.detach().clone().requires_grad_(requires_grad),
          embed_dim,
          num_head,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          None,
          False,
          True,
          None,
      ),
  )

  # Case 4: mask_type=1 (Key Padding mask [B, L])
  key_padding_mask = torch.randint(
      0, 2, (batch_size, seq_len), device=device, dtype=torch.bool
  )
  yield SampleInput(
      query.detach().clone().requires_grad_(requires_grad),
      args=(
          key.detach().clone().requires_grad_(requires_grad),
          value.detach().clone().requires_grad_(requires_grad),
          embed_dim,
          num_head,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          key_padding_mask,
          True,
          True,
          1,
      ),
  )

  # Case 5: 4D bool mask [B, H, L, L]
  mask_4d = torch.randint(
      0,
      2,
      (batch_size, num_head, seq_len, seq_len),
      device=device,
      dtype=torch.bool,
  )
  yield SampleInput(
      query.detach().clone().requires_grad_(requires_grad),
      args=(
          key.detach().clone().requires_grad_(requires_grad),
          value.detach().clone().requires_grad_(requires_grad),
          embed_dim,
          num_head,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          mask_4d,
          False,
          False,
          0,
      ),
  )

  # Case 6: Float additive mask [L, L]
  mask_float = torch.randn(seq_len, seq_len, device=device, dtype=dtype)
  yield SampleInput(
      query.detach().clone().requires_grad_(requires_grad),
      args=(
          key.detach().clone().requires_grad_(requires_grad),
          value.detach().clone().requires_grad_(requires_grad),
          embed_dim,
          num_head,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          mask_float,
          True,
          True,
          0,
      ),
  )

  # Case 7: Empty tensor input
  query_empty = make_arg((0, seq_len, embed_dim))
  key_empty = make_arg((0, seq_len, embed_dim))
  value_empty = make_arg((0, seq_len, embed_dim))
  yield SampleInput(
      query_empty,
      args=(
          key_empty,
          value_empty,
          embed_dim,
          num_head,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          None,
          True,
          True,
          None,
      ),
  )


def _sample_inputs_transformer_encoder_layer_fwd(
    op_info: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs,
):
  """Sample inputs generator for _transformer_encoder_layer_fwd."""
  del op_info, kwargs
  if not dtype.is_floating_point:
    return
  make_arg = functools.partial(
      torch.randn, device=device, dtype=dtype, requires_grad=requires_grad
  )

  embed_dim = 8
  num_heads = 2
  d_ff = 16
  batch_size = 2
  seq_len = 4
  eps = 1e-5

  src = make_arg((batch_size, seq_len, embed_dim))
  qkv_weight = make_arg((3 * embed_dim, embed_dim))
  qkv_bias = make_arg((3 * embed_dim,))
  proj_weight = make_arg((embed_dim, embed_dim))
  proj_bias = make_arg((embed_dim,))
  norm_weight_1 = torch.ones(embed_dim, device=device, dtype=dtype)
  norm_bias_1 = torch.zeros(embed_dim, device=device, dtype=dtype)
  norm_weight_2 = torch.ones(embed_dim, device=device, dtype=dtype)
  norm_bias_2 = torch.zeros(embed_dim, device=device, dtype=dtype)
  ffn_weight_1 = make_arg((d_ff, embed_dim))
  ffn_bias_1 = make_arg((d_ff,))
  ffn_weight_2 = make_arg((embed_dim, d_ff))
  ffn_bias_2 = make_arg((embed_dim,))

  # Case 1: Post-LN ReLU without mask
  yield SampleInput(
      src,
      args=(
          embed_dim,
          num_heads,
          qkv_weight,
          qkv_bias,
          proj_weight,
          proj_bias,
          False,
          False,
          eps,
          norm_weight_1,
          norm_bias_1,
          norm_weight_2,
          norm_bias_2,
          ffn_weight_1,
          ffn_bias_1,
          ffn_weight_2,
          ffn_bias_2,
          None,
          None,
      ),
  )

  # Case 2: Pre-LN GELU without mask
  yield SampleInput(
      src.detach().clone().requires_grad_(requires_grad),
      args=(
          embed_dim,
          num_heads,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          True,
          True,
          eps,
          norm_weight_1,
          norm_bias_1,
          norm_weight_2,
          norm_bias_2,
          ffn_weight_1.detach().clone().requires_grad_(requires_grad),
          ffn_bias_1.detach().clone().requires_grad_(requires_grad),
          ffn_weight_2.detach().clone().requires_grad_(requires_grad),
          ffn_bias_2.detach().clone().requires_grad_(requires_grad),
          None,
          None,
      ),
  )

  # Case 3: 2D Bool Attention Mask (mask_type=0)
  mask_2d_bool = torch.triu(
      torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
  )
  yield SampleInput(
      src.detach().clone().requires_grad_(requires_grad),
      args=(
          embed_dim,
          num_heads,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          False,
          False,
          eps,
          norm_weight_1,
          norm_bias_1,
          norm_weight_2,
          norm_bias_2,
          ffn_weight_1.detach().clone().requires_grad_(requires_grad),
          ffn_bias_1.detach().clone().requires_grad_(requires_grad),
          ffn_weight_2.detach().clone().requires_grad_(requires_grad),
          ffn_bias_2.detach().clone().requires_grad_(requires_grad),
          mask_2d_bool,
          0,
      ),
  )

  # Case 4: 2D Bool Key Padding Mask (mask_type=1)
  mask_2d_padding = torch.zeros(
      batch_size, seq_len, device=device, dtype=torch.bool
  )
  mask_2d_padding[:, -1] = True
  yield SampleInput(
      src.detach().clone().requires_grad_(requires_grad),
      args=(
          embed_dim,
          num_heads,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          False,
          False,
          eps,
          norm_weight_1,
          norm_bias_1,
          norm_weight_2,
          norm_bias_2,
          ffn_weight_1.detach().clone().requires_grad_(requires_grad),
          ffn_bias_1.detach().clone().requires_grad_(requires_grad),
          ffn_weight_2.detach().clone().requires_grad_(requires_grad),
          ffn_bias_2.detach().clone().requires_grad_(requires_grad),
          mask_2d_padding,
          1,
      ),
  )

  # Case 5: Float Additive Mask [B, num_heads, seq_len, seq_len] (mask_type=2)
  mask_float = torch.zeros(
      batch_size, num_heads, seq_len, seq_len, device=device, dtype=dtype
  )
  mask_float.masked_fill_(
      torch.triu(
          torch.ones(
              batch_size,
              num_heads,
              seq_len,
              seq_len,
              device=device,
              dtype=torch.bool,
          ),
          diagonal=1,
      ),
      -1e4,
  )
  yield SampleInput(
      src.detach().clone().requires_grad_(requires_grad),
      args=(
          embed_dim,
          num_heads,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          False,
          False,
          eps,
          norm_weight_1,
          norm_bias_1,
          norm_weight_2,
          norm_bias_2,
          ffn_weight_1.detach().clone().requires_grad_(requires_grad),
          ffn_bias_1.detach().clone().requires_grad_(requires_grad),
          ffn_weight_2.detach().clone().requires_grad_(requires_grad),
          ffn_bias_2.detach().clone().requires_grad_(requires_grad),
          mask_float,
          2,
      ),
  )

  # Case 6: Empty tensor input
  src_empty = make_arg((0, seq_len, embed_dim))
  yield SampleInput(
      src_empty,
      args=(
          embed_dim,
          num_heads,
          qkv_weight.detach().clone().requires_grad_(requires_grad),
          qkv_bias.detach().clone().requires_grad_(requires_grad),
          proj_weight.detach().clone().requires_grad_(requires_grad),
          proj_bias.detach().clone().requires_grad_(requires_grad),
          False,
          False,
          eps,
          norm_weight_1,
          norm_bias_1,
          norm_weight_2,
          norm_bias_2,
          ffn_weight_1.detach().clone().requires_grad_(requires_grad),
          ffn_bias_1.detach().clone().requires_grad_(requires_grad),
          ffn_weight_2.detach().clone().requires_grad_(requires_grad),
          ffn_bias_2.detach().clone().requires_grad_(requires_grad),
          None,
          None,
      ),
  )


def _ref_thnn_fused_lstm_cell(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    cx: torch.Tensor,
    input_bias: torch.Tensor | None = None,
    hidden_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Analytical CPU reference implementation for `aten::_thnn_fused_lstm_cell`.

  Upstream PyTorch registers `aten::_thnn_fused_lstm_cell` strictly for CUDA
  (`aten/src/ATen/native/cuda/RNN.cu`) and does not provide a native CPU C++
  kernel (`AutogradCPU`). This function provides an analytical pure-PyTorch CPU
  reference implementation so that `do_test_op` can compute golden forward
  values (`ops_test`) and analytical autograd gradients (`ops_test_grad`)
  on CPU when validating TorchTPU accelerated kernels.

  Implements the standard LSTM cell recurrence:
  https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

    i = sigmoid(W_ii * x + b_ii + W_hi * h + b_hi)
    f = sigmoid(W_if * x + b_if + W_hf * h + b_hf)
    g = tanh   (W_ig * x + b_ig + W_hg * h + b_hg)
    o = sigmoid(W_io * x + b_io + W_ho * h + b_ho)
    c' = f * c + i * g
    h' = o * tanh(c')

  Args:
    input_gates: Pre-activation linear projection of input `x` [batch, 4*H].
    hidden_gates: Pre-activation linear projection of state `hx` [batch, 4*H].
    cx: Previous cell state [batch, H].
    input_bias: Optional input projection bias [4*H].
    hidden_bias: Optional hidden projection bias [4*H].

  Returns:
    A tuple `(hy, cy, workspace)` where:
      - `hy`: New hidden state `h'` [batch, H].
      - `cy`: New cell state `c'` [batch, H].
      - `workspace`: Concat of activated gates `[i, f, g, o]` [batch, 4*H],
        retained for backward gradient computation.

  Raises:
    RuntimeError: If `input_gates.dtype` is not floating-point or complex.
  """
  orig_dtype = input_gates.dtype
  if not (orig_dtype.is_floating_point or orig_dtype.is_complex):
    raise RuntimeError(
        f"_thnn_fused_lstm_cell not implemented for {orig_dtype}"
    )
  if orig_dtype in (torch.bfloat16, torch.float16):
    input_gates = input_gates.to(torch.float32)
    hidden_gates = hidden_gates.to(torch.float32)
    cx = cx.to(torch.float32)
    if input_bias is not None:
      input_bias = input_bias.to(torch.float32)
    if hidden_bias is not None:
      hidden_bias = hidden_bias.to(torch.float32)

  ig = input_gates if input_bias is None else input_gates + input_bias
  hg = hidden_gates if hidden_bias is None else hidden_gates + hidden_bias
  gates = ig + hg
  i_pre, f_pre, g_pre, o_pre = gates.chunk(4, dim=1)
  ingate = torch.sigmoid(i_pre)
  forgetgate = torch.sigmoid(f_pre)
  cellgate = torch.tanh(g_pre)
  outgate = torch.sigmoid(o_pre)
  cy = forgetgate * cx + ingate * cellgate
  hy = outgate * torch.tanh(cy)
  workspace = torch.cat([ingate, forgetgate, cellgate, outgate], dim=1)
  if orig_dtype in (torch.bfloat16, torch.float16):
    hy = hy.to(orig_dtype)
    cy = cy.to(orig_dtype)
    workspace = workspace.to(orig_dtype)
  return hy, cy, workspace.detach()


def _ref_thnn_fused_gru_cell(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    hx: torch.Tensor,
    input_bias: torch.Tensor | None = None,
    hidden_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Analytical CPU reference implementation for `aten::_thnn_fused_gru_cell`.

  Upstream PyTorch registers `aten::_thnn_fused_gru_cell` strictly for CUDA
  (`aten/src/ATen/native/cuda/RNN.cu`) and does not provide a native CPU C++
  kernel (`AutogradCPU`). This function provides an analytical pure-PyTorch CPU
  reference implementation so that `do_test_op` can compute golden forward
  values (`ops_test`) and analytical autograd gradients (`ops_test_grad`)
  on CPU when validating TorchTPU accelerated kernels.

  Implements the standard GRU cell recurrence:
  https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html

    r  = sigmoid(i_r + h_r)
    z  = sigmoid(i_z + h_z)
    n  = tanh(i_n + r * h_n)
    h' = (1 - z) * n + z * hx  ==  (hx - n) * z + n

  Args:
    input_gates: Pre-activation linear projection of input `x` [batch, 3*H].
    hidden_gates: Pre-activation linear projection of state `hx` [batch, 3*H].
    hx: Previous hidden state [batch, H].
    input_bias: Optional input projection bias [3*H].
    hidden_bias: Optional hidden projection bias [3*H].

  Returns:
    A tuple `(hy, workspace)` where:
      - `hy`: New hidden state `h'` [batch, H].
      - `workspace`: Concat of `[r, z, n, hx, h_n]` [batch, 5*H], retained for
        backward gradient computation.

  Raises:
    RuntimeError: If `input_gates.dtype` is not floating-point or complex.
  """
  orig_dtype = input_gates.dtype
  if not (orig_dtype.is_floating_point or orig_dtype.is_complex):
    raise RuntimeError(f"_thnn_fused_gru_cell not implemented for {orig_dtype}")
  if orig_dtype in (torch.bfloat16, torch.float16):
    input_gates = input_gates.to(torch.float32)
    hidden_gates = hidden_gates.to(torch.float32)
    hx = hx.to(torch.float32)
    if input_bias is not None:
      input_bias = input_bias.to(torch.float32)
    if hidden_bias is not None:
      hidden_bias = hidden_bias.to(torch.float32)

  ig = input_gates if input_bias is None else input_gates + input_bias
  hg = hidden_gates if hidden_bias is None else hidden_gates + hidden_bias
  ir, iz, in_ = ig.chunk(3, dim=1)
  hr, hz, hn = hg.chunk(3, dim=1)
  resetgate = torch.sigmoid(ir + hr)
  updategate = torch.sigmoid(iz + hz)
  newgate = torch.tanh(in_ + resetgate * hn)
  hy = (hx - newgate) * updategate + newgate
  workspace = torch.cat([resetgate, updategate, newgate, hx, hn], dim=1)
  if orig_dtype in (torch.bfloat16, torch.float16):
    hy = hy.to(orig_dtype)
    workspace = workspace.to(orig_dtype)
  return hy, workspace.detach()


def _swizzle_lhs_scale(logical_scale):
  """Converts an (M, K // 32) LHS scale tensor to the SWIZZLE_32_4_4 layout.

  SWIZZLE_32_4_4 is NVIDIA's blocked scale-factor layout. Divides the matrix
  into 128x4 tiles and rearranges internal 32-row chunks:
    1. Pad to ceil_div multiples of (128, 4) if needed.
    2. .view(M_blocks, 128, K_blocks, 4) # (M_tiles, 128, K_tiles, 4)
    3. .transpose(1, 2) # (M_tiles, K_tiles, 128, 4)
    4. .reshape(-1, 4, 32, 4) # (num_tiles, 4_chunks, 32_rows, 4_cols)
    5. .transpose(1, 2) # (num_tiles, 32_rows, 4_chunks, 4_cols)
    6. .reshape(-1) # Flat (padded_m * padded_k,)

  Args:
    logical_scale: Scale tensor of shape (M, K // 32).

  Returns:
    A flat tensor of padded elements.
  """
  m, k_logical = logical_scale.shape
  m_blocks = common_quantized.ceil_div(m, 128)
  k_blocks = common_quantized.ceil_div(k_logical, 4)
  padded_m = m_blocks * 128
  padded_k = k_blocks * 4
  if (m, k_logical) != (padded_m, padded_k):
    padded = torch.zeros(
        (padded_m, padded_k),
        dtype=logical_scale.dtype,
        device=logical_scale.device,
    )
    padded[:m, :k_logical] = logical_scale
  else:
    padded = logical_scale
  tiles = padded.view(m_blocks, 128, k_blocks, 4).transpose(1, 2)
  return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)


def _unswizzle_lhs_scale(swizzled_scale, m, k_logical):
  """Inverts `_swizzle_lhs_scale`.

  Undoes the tile rearrangement in reverse order:
    1. .reshape(M_blocks, K_blocks, 32, 4, 4) # tiles of (32_rows, 4_chunks, 4)
    2. .transpose(2, 3) # tiles of (4_chunks, 32_rows, 4_cols)
    3. .reshape(M_blocks, K_blocks, 128, 4) # (M_tiles, K_tiles, 128, 4)
    4. .transpose(1, 2) # (M_tiles, 128, K_tiles, 4)
    5. .reshape(padded_m, padded_k) # (padded_m, padded_k)
    6. Slice to [:m, :k_logical]

  Args:
    swizzled_scale: Flat blocked scale buffer.
    m: Logical row count, i.e. the M of the matmul.
    k_logical: Logical column count, i.e. `K // 32`.

  Returns:
    A tensor of shape (M, K // 32).
  """
  m_blocks = common_quantized.ceil_div(m, 128)
  k_blocks = common_quantized.ceil_div(k_logical, 4)
  tiles = swizzled_scale.reshape(m_blocks, k_blocks, 32, 4, 4).transpose(2, 3)
  tiles = tiles.reshape(m_blocks, k_blocks, 128, 4).transpose(1, 2)
  return tiles.reshape(m_blocks * 128, k_blocks * 4)[:m, :k_logical]


def _swizzle_rhs_scale(logical_scale):
  """Converts a (K // 32, N) RHS scale tensor to the SWIZZLE_32_4_4 layout.

  SWIZZLE_32_4_4 always tiles the non-contraction dimension by 128, which is N
  for the RHS, so the *logical* matrix is transposed before being blocked. The
  blocked buffer itself is then laid out exactly as it is for the LHS.
  """
  return _swizzle_lhs_scale(logical_scale.t().contiguous())


def _unswizzle_rhs_scale(swizzled_scale, k_logical, n):
  """Inverts `_swizzle_rhs_scale`, returning a (K // 32, N) tensor."""
  return _unswizzle_lhs_scale(swizzled_scale, n, k_logical).t()


def _get_scaled_mm_v2_default_inputs(device="tpu"):
  """Returns default input tensors for torch._scaled_mm_v2 tests."""
  m, n, k = 16, 16, 16
  dtype = torch.float8_e4m3fn
  self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype).to(device)
  mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype).to(device)
  scale_a = [torch.tensor([1.5], dtype=torch.float32, device=device)]
  scale_b = [torch.tensor([2.0], dtype=torch.float32, device=device)]
  recipe_a = [0]
  recipe_b = [0]
  swizzle_a = [0]
  swizzle_b = [0]
  return (
      self_fp8,
      mat2_fp8,
      scale_a,
      recipe_a,
      swizzle_a,
      scale_b,
      recipe_b,
      swizzle_b,
  )


def _ref_scaled_mm_v2(
    self_mat,
    mat2,
    scale_a,
    recipe_a,
    swizzle_a,
    scale_b,
    recipe_b,
    swizzle_b,
    bias=None,
    out_dtype=None,
    # Unused in CPU ref (defaults to 2D matmul contraction)
    contraction_dim=None,
    # Unused in CPU ref (hardware precision hint)
    use_fast_accum=False,
):
  """CPU reference implementation for torch._scaled_mm_v2."""
  del contraction_dim, use_fast_accum
  s_f32 = self_mat.float()
  m_f32 = mat2.float()
  sa_raw = scale_a[0]
  sb_raw = scale_b[0]
  rec_a = recipe_a[0]
  rec_b = recipe_b[0]
  swiz_a = swizzle_a[0]
  swiz_b = swizzle_b[0]

  if rec_a == 0 or rec_a == 1:
    s_scaled = s_f32 * sa_raw
  elif rec_a == 3:
    m, k = self_mat.shape
    if swiz_a == 1:
      sa_logical = _unswizzle_lhs_scale(sa_raw, m, k // 32)
    else:
      sa_logical = sa_raw
    sa_expanded = sa_logical.unsqueeze(-1).expand(m, k // 32, 32).reshape(m, k)
    s_scaled = s_f32 * sa_expanded
  else:
    s_scaled = s_f32 * sa_raw

  if rec_b == 0 or rec_b == 1:
    m_scaled = m_f32 * sb_raw
  elif rec_b == 3:
    k, n = mat2.shape
    if swiz_b == 1:
      sb_logical = _unswizzle_rhs_scale(sb_raw, k // 32, n)
    else:
      sb_logical = sb_raw
    sb_expanded = sb_logical.unsqueeze(1).expand(k // 32, 32, n).reshape(k, n)
    m_scaled = m_f32 * sb_expanded
  else:
    m_scaled = m_f32 * sb_raw

  out_f32 = torch.mm(s_scaled, m_scaled)
  if bias is not None:
    out_f32 = out_f32 + bias
  if out_dtype is not None:
    return out_f32.to(out_dtype)
  return out_f32


def _sample_inputs_scaled_mm_v2(
    op_info, device, dtype, requires_grad=False, **kwargs
):
  """Generates sample inputs for torch._scaled_mm_v2."""
  del op_info, requires_grad, kwargs
  make_mat = (
      lambda shape: torch.randn(*shape, dtype=torch.float32)
      .to(dtype)
      .to(device)
  )
  make_scale = lambda shape: (
      torch.randn(*shape, dtype=torch.float32).abs() + 0.1
  ).to(device)

  # 1. TensorWise scaling (recipe=0, swizzle=0)
  m, n, k = 16, 16, 16
  mat1 = make_mat((m, k))
  mat2 = make_mat((k, n)).t().contiguous().t()
  scale1 = [make_scale((1,))]
  scale2 = [make_scale((1,))]
  yield core.SampleInput(
      mat1,
      mat2,
      scale1,
      [0],
      [0],
      scale2,
      [0],
      [0],
      None,
      None,
  )

  # 2. RowWise scaling (recipe=1, swizzle=0)
  scale1_row = [make_scale((m, 1))]
  scale2_row = [make_scale((1, n))]
  yield core.SampleInput(
      mat1,
      mat2,
      scale1_row,
      [1],
      [0],
      scale2_row,
      [1],
      [0],
      None,
      torch.bfloat16,
  )

  # 3. BlockWise1x32 scaling with SWIZZLE_32_4_4 (recipe=3, swizzle=1)
  m2, n2, k2 = 128, 128, 128
  mat1_bw = make_mat((m2, k2))
  mat2_bw = make_mat((k2, n2)).t().contiguous().t()
  scale1_bw = [make_scale((m2, k2 // 32)).to(torch.float8_e8m0fnu)]
  scale2_bw = [make_scale((k2 // 32, n2)).to(torch.float8_e8m0fnu)]
  scale1_swiz = [_swizzle_lhs_scale(scale1_bw[0])]
  scale2_swiz = [_swizzle_rhs_scale(scale2_bw[0])]
  yield core.SampleInput(
      mat1_bw,
      mat2_bw,
      scale1_swiz,
      [3],
      [1],
      scale2_swiz,
      [3],
      [1],
      None,
      torch.bfloat16,
  )

  # 4. BlockWise1x32 scaling with SWIZZLE_32_4_4 and bias (recipe=3, swizzle=1)
  bias_bw = torch.randn(n2, dtype=torch.bfloat16, device=device)
  yield core.SampleInput(
      mat1_bw,
      mat2_bw,
      scale1_swiz,
      [3],
      [1],
      scale2_swiz,
      [3],
      [1],
      bias_bw,
      torch.bfloat16,
  )

  # 5. TensorWise with bias and out_dtype
  bias = torch.randn(16, dtype=torch.bfloat16, device=device)
  yield core.SampleInput(
      mat1,
      mat2,
      scale1,
      [0],
      [0],
      scale2,
      [0],
      [0],
      bias,
      torch.bfloat16,
  )

  # 6. BlockWise1x32 with SWIZZLE_32_4_4 where M needs padding (M < 128)
  m3, n3, k3 = 32, 128, 128
  mat1_m_pad = make_mat((m3, k3))
  mat2_m_pad = make_mat((k3, n3)).t().contiguous().t()
  scale1_m_pad = [make_scale((m3, k3 // 32)).to(torch.float8_e8m0fnu)]
  scale2_m_pad = [make_scale((k3 // 32, n3)).to(torch.float8_e8m0fnu)]
  yield core.SampleInput(
      mat1_m_pad,
      mat2_m_pad,
      [_swizzle_lhs_scale(scale1_m_pad[0])],
      [3],
      [1],
      [_swizzle_rhs_scale(scale2_m_pad[0])],
      [3],
      [1],
      None,
      torch.bfloat16,
  )

  # 7. BlockWise1x32 with SWIZZLE_32_4_4 where K needs padding (K // 32 < 4)
  m4, n4, k4 = 128, 128, 64
  mat1_k_pad = make_mat((m4, k4))
  mat2_k_pad = make_mat((k4, n4)).t().contiguous().t()
  scale1_k_pad = [make_scale((m4, k4 // 32)).to(torch.float8_e8m0fnu)]
  scale2_k_pad = [make_scale((k4 // 32, n4)).to(torch.float8_e8m0fnu)]
  yield core.SampleInput(
      mat1_k_pad,
      mat2_k_pad,
      [_swizzle_lhs_scale(scale1_k_pad[0])],
      [3],
      [1],
      [_swizzle_rhs_scale(scale2_k_pad[0])],
      [3],
      [1],
      None,
      torch.bfloat16,
  )

  # 8. BlockWise1x32 with SWIZZLE_32_4_4 where both M and K need padding
  m5, n5, k5 = 32, 64, 64
  mat1_mk_pad = make_mat((m5, k5))
  mat2_mk_pad = make_mat((k5, n5)).t().contiguous().t()
  scale1_mk_pad = [make_scale((m5, k5 // 32)).to(torch.float8_e8m0fnu)]
  scale2_mk_pad = [make_scale((k5 // 32, n5)).to(torch.float8_e8m0fnu)]
  yield core.SampleInput(
      mat1_mk_pad,
      mat2_mk_pad,
      [_swizzle_lhs_scale(scale1_mk_pad[0])],
      [3],
      [1],
      [_swizzle_rhs_scale(scale2_mk_pad[0])],
      [3],
      [1],
      None,
      torch.bfloat16,
  )


def _ref_scaled_grouped_mm_v2(
    self,
    mat2,
    scale_a,
    scale_recipe_a,
    swizzle_a,
    scale_b,
    scale_recipe_b,
    swizzle_b,
    offs=None,
    bias=None,
    out_dtype=None,
):
  """Float32 reference implementation for torch._scaled_grouped_mm_v2."""
  del scale_recipe_a, swizzle_a, scale_recipe_b, swizzle_b
  sa = scale_a[0].to(torch.float32)
  sb = scale_b[0].to(torch.float32)
  target_dtype = out_dtype if out_dtype is not None else torch.bfloat16

  if offs is None:
    g, m, _ = self.shape
    _, _, n = mat2.shape
    a_scaled = self.to(torch.float32) * sa.reshape(g, m, 1)
    b_scaled = mat2.to(torch.float32) * sb.reshape(g, 1, n)
    out_f32 = torch.bmm(a_scaled, b_scaled)
  elif self.ndim == 2 and mat2.ndim == 3:
    g, _, n = mat2.shape
    sa_col = sa.reshape(-1, 1)
    sb_rows = sb.reshape(g, 1, n)
    chunks = []
    start = 0
    for idx, end in enumerate(offs.tolist()):
      a_g = self[start:end].to(torch.float32) * sa_col[start:end]
      b_g = mat2[idx].to(torch.float32) * sb_rows[idx]
      chunks.append(torch.mm(a_g, b_g))
      start = end
    out_f32 = torch.cat(chunks, dim=0)
  elif self.ndim == 3 and mat2.ndim == 2:
    g, m, _ = self.shape
    sa_cols = sa.reshape(g, m, 1)
    sb_row = sb.reshape(1, -1)
    chunks = []
    start = 0
    for idx, end in enumerate(offs.tolist()):
      a_g = self[idx].to(torch.float32) * sa_cols[idx]
      b_g = mat2[:, start:end].to(torch.float32) * sb_row[:, start:end]
      chunks.append(torch.mm(a_g, b_g))
      start = end
    out_f32 = torch.cat(chunks, dim=1)
  else:
    g = offs.numel()
    m = self.shape[0]
    n = mat2.shape[1]
    sa_2d = sa.reshape(g, m)
    sb_2d = sb.reshape(g, n)
    chunks = []
    start = 0
    for idx, end in enumerate(offs.tolist()):
      a_g = self[:, start:end].to(torch.float32) * sa_2d[idx].unsqueeze(1)
      b_g = mat2[start:end, :].to(torch.float32) * sb_2d[idx].unsqueeze(0)
      chunks.append(torch.mm(a_g, b_g))
      start = end
    out_f32 = torch.stack(chunks, dim=0)

  if bias is not None:
    out_f32 = out_f32 + bias.to(torch.float32)
  return out_f32.to(target_dtype)


def _sample_inputs_scaled_grouped_mm_v2(
    op_info, device, dtype, requires_grad=False, **kwargs
):
  """Generates sample inputs for torch._scaled_grouped_mm_v2."""
  del op_info, requires_grad, kwargs
  make_mat_row_major = (
      lambda shape: torch.randn(*shape, dtype=torch.float32)
      .to(dtype)
      .to(device)
  )
  make_mat_col_major = (
      lambda shape: torch.randn(*shape, dtype=torch.float32)
      .to(dtype)
      .to(device)
      .transpose(-2, -1)
      .contiguous()
      .transpose(-2, -1)
  )
  make_scale = lambda shape: (
      torch.randn(*shape, dtype=torch.float32).abs() + 0.1
  ).to(device)

  # 1. 2D x 3D (ragged M) with RowWise scaling (recipe=1, swizzle=0)
  yield core.SampleInput(
      make_mat_row_major((48, 32)),
      make_mat_col_major((2, 32, 64)),
      args=(
          [make_scale((48,))],
          [1],
          [0],
          [make_scale((2, 64))],
          [1],
          [0],
      ),
      kwargs={
          "offs": torch.tensor([16, 48], dtype=torch.int32, device=device),
          "out_dtype": torch.bfloat16,
      },
  )

  # 2. 2D x 3D (ragged M) with zero-sized group and default out_dtype (None)
  yield core.SampleInput(
      make_mat_row_major((48, 32)),
      make_mat_col_major((3, 32, 64)),
      args=(
          [make_scale((48,))],
          [1],
          [0],
          [make_scale((3, 64))],
          [1],
          [0],
      ),
      kwargs={
          "offs": torch.tensor([16, 16, 48], dtype=torch.int32, device=device),
          "out_dtype": None,
      },
  )

  # 3. 3D x 3D (batched, offs=None) with RowWise scaling
  yield core.SampleInput(
      make_mat_row_major((2, 32, 32)),
      make_mat_col_major((2, 32, 64)),
      args=(
          [make_scale((2, 32))],
          [1],
          [0],
          [make_scale((2, 64))],
          [1],
          [0],
      ),
      kwargs={
          "offs": None,
          "out_dtype": torch.bfloat16,
      },
  )

  # 4. 2D x 2D (ragged K) with RowWise scaling
  yield core.SampleInput(
      make_mat_row_major((32, 48)),
      make_mat_col_major((48, 64)),
      args=(
          [make_scale((2, 32))],
          [1],
          [0],
          [make_scale((2, 64))],
          [1],
          [0],
      ),
      kwargs={
          "offs": torch.tensor([16, 48], dtype=torch.int32, device=device),
          "out_dtype": torch.bfloat16,
      },
  )

  # 5. 3D x 2D (ragged N) with RowWise scaling
  yield core.SampleInput(
      make_mat_row_major((2, 32, 32)),
      make_mat_col_major((32, 48)),
      args=(
          [make_scale((2, 32))],
          [1],
          [0],
          [make_scale((48,))],
          [1],
          [0],
      ),
      kwargs={
          "offs": torch.tensor([16, 48], dtype=torch.int32, device=device),
          "out_dtype": torch.bfloat16,
      },
  )


# The in_features (K) of the Llama3-8B linear layers, whose (in_features,
# out_features) are wq/wo (4096, 4096), wk/wv (4096, 1024), w1/w3 (4096, 14336)
# and w2 (14336, 4096).
_LLAMA3_8B_DIM: Final[int] = 4096
_LLAMA3_8B_FFN_DIM: Final[int] = 14336


def _quantize_rowwise_to_float8(
    hp: torch.Tensor,
    float8_dtype: torch.dtype,
    *,
    round_scales_to_power_of_2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantizes each row of `hp` to `float8_dtype` like torchao's Float8Linear.

  Mirrors torchao's rowwise scaling (`hp_tensor_to_float8_dynamic` with
  `axiswise_dim=-1`): the scale of each row maps its amax to the largest finite
  `float8_dtype` value, and the scaled values saturate before the cast.

  Args:
    hp: The high-precision tensor of rank >= 2 and shape (..., K) to quantize.
    float8_dtype: The float8 dtype to quantize to.
    round_scales_to_power_of_2: Whether to round the scales down to a power of
      2, as torchao's ROWWISE recipe does by default.

  Returns:
    The quantized tensor, and the float32 dequantization scales of shape
    (*hp.shape[:-1], 1), i.e. the reciprocals of the quantization scales, which
    is what `torch._scaled_mm` expects.
  """
  float8_max = torch.finfo(float8_dtype).max
  amax = hp.abs().amax(dim=-1, keepdim=True).to(torch.float32)
  scale = float8_max / amax.clamp(min=1e-12)
  if round_scales_to_power_of_2:
    scale = torch.exp2(torch.floor(torch.log2(scale)))
  data = (hp.to(torch.float32) * scale).clamp(-float8_max, float8_max)
  return data.to(float8_dtype), scale.reciprocal()


def _sample_inputs_scaled_mm_rowwise(
    op_info: OpInfo,
    device: torch.device | str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    **kwargs: Any,
) -> Iterator[SampleInput]:
  """Generates rowwise-scaled torch._scaled_mm inputs like torchao's.

  Each sample quantizes activations `x` of shape (M, K) and a weight `w` of
  shape (N, K) row by row and computes `x @ w.t()`, the forward of torchao's
  Float8Linear. Hence mat2 is column-major, as the GPU kernels require.

  The Llama3-8B samples keep the real accumulation depth K, which drives the
  accumulated rounding error, but slice M and N to keep the golden files small.
  """
  del op_info, requires_grad, kwargs
  if dtype != torch.float8_e4m3fn:
    return

  def make_sample(
      name: str,
      m: int,
      k: int,
      n: int,
      *,
      use_fast_accum: bool = False,
      out_dtype: torch.dtype = torch.bfloat16,
      with_bias: bool = False,
      round_scales_to_power_of_2: bool = True,
  ) -> SampleInput:
    x_fp8, x_scale = _quantize_rowwise_to_float8(
        torch.randn(m, k, device=device),
        dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    w_fp8, w_scale = _quantize_rowwise_to_float8(
        torch.randn(n, k, device=device) * k**-0.5,
        dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    sample_kwargs = {"out_dtype": out_dtype, "use_fast_accum": use_fast_accum}
    if with_bias:
      # The GPU kernels require the bias dtype to match a bfloat16 output.
      sample_kwargs["bias"] = torch.randn(n, dtype=out_dtype, device=device)
    return SampleInput(
        x_fp8,
        args=(w_fp8.t(), x_scale, w_scale.t()),
        kwargs=sample_kwargs,
        name=name,
    )

  # wq/wk/wv/wo/w1/w3 contract over the model dim and w2 over the FFN dim.
  llama3_8b_shapes = ((16, _LLAMA3_8B_DIM, 32), (16, _LLAMA3_8B_FFN_DIM, 16))
  for m, k, n in llama3_8b_shapes:
    yield make_sample(f"llama3_8b_k{k}", m, k, n)
  yield make_sample("bias", 32, 128, 48, with_bias=True)
  yield make_sample(
      "float32_out",
      32,
      256,
      96,
      out_dtype=torch.float32,
      round_scales_to_power_of_2=False,
  )
  yield make_sample("unaligned_m", 17, 128, 32)
  # torchao's Float8Linear enables use_fast_accum in the forward by default.
  for m, k, n in llama3_8b_shapes:
    yield make_sample(
        f"llama3_8b_k{k}_fast_accum", m, k, n, use_fast_accum=True
    )


# The hidden dim and the expert FFN dim of Qwen3-30B-A3B.
_QWEN3_30B_A3B_DIM: Final[int] = 2048
_QWEN3_30B_A3B_MOE_FFN_DIM: Final[int] = 768


def _sample_inputs_scaled_grouped_mm_rowwise(
    op_info: OpInfo,
    device: torch.device | str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    **kwargs: Any,
) -> Iterator[SampleInput]:
  """Generates rowwise-scaled torch._scaled_grouped_mm inputs like torchao's.

  Mirrors the grouped GEMMs of torchao's FP8 MoE training: the forward and the
  input gradient multiply tokens of shape (M, K), grouped by `offs` along M,
  with expert weights of shape (G, K, N), scaling the tokens per row and the
  weights per expert and column. The weight gradient multiplies two 2D tensors
  grouped by `offs` along the contraction dim, scaling each group's rows and
  columns separately.

  The Qwen3-30B-A3B samples keep the real accumulation depth K but slice N to
  keep the golden files small.
  """
  del op_info, requires_grad, kwargs
  if dtype != torch.float8_e4m3fn:
    return

  def quantize(hp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _quantize_rowwise_to_float8(
        hp, dtype, round_scales_to_power_of_2=True
    )

  def make_offs(group_sizes: Sequence[int]) -> torch.Tensor:
    return torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)

  def make_2d_3d_sample(
      name: str,
      group_sizes: Sequence[int],
      k: int,
      n: int,
      *,
      use_fast_accum: bool = False,
  ) -> SampleInput:
    a_fp8, a_scale = quantize(torch.randn(sum(group_sizes), k, device=device))
    # Quantizing the (G, N, K) weights along K and transposing them yields
    # column-major (G, K, N) weights with (G, N) scales, as the GPU expects.
    b_fp8, b_scale = quantize(
        torch.randn(len(group_sizes), n, k, device=device) * k**-0.5
    )
    return SampleInput(
        a_fp8,
        args=(
            b_fp8.transpose(-2, -1),
            a_scale.squeeze(-1),
            b_scale.squeeze(-1),
        ),
        kwargs={
            "offs": make_offs(group_sizes),
            "out_dtype": torch.bfloat16,
            "use_fast_accum": use_fast_accum,
        },
        name=name,
    )

  def make_2d_2d_sample(
      name: str,
      group_sizes: Sequence[int],
      m: int,
      n: int,
      *,
      use_fast_accum: bool = False,
  ) -> SampleInput:
    # grad_w[g] = grad_out.t()[:, group g] @ x[group g, :] for each expert g.
    k_total = sum(group_sizes)
    a_hp = torch.randn(m, k_total, device=device)
    b_hp = torch.randn(n, k_total, device=device) * k_total**-0.5
    a_groups, a_scales, b_groups, b_scales = [], [], [], []
    for a_group_hp, b_group_hp in zip(
        a_hp.split(group_sizes, dim=1), b_hp.split(group_sizes, dim=1)
    ):
      a_group, a_group_scale = quantize(a_group_hp)
      b_group, b_group_scale = quantize(b_group_hp)
      a_groups.append(a_group)
      a_scales.append(a_group_scale)
      b_groups.append(b_group)
      b_scales.append(b_group_scale)
    return SampleInput(
        torch.cat(a_groups, dim=1),
        args=(
            # Column-major (K, N), as the GPU expects.
            torch.cat(b_groups, dim=1).t(),
            # The scales of group g occupy [g * m, (g + 1) * m).
            torch.cat(a_scales).squeeze(-1),
            torch.cat(b_scales).squeeze(-1),
        ),
        kwargs={
            "offs": make_offs(group_sizes),
            "out_dtype": torch.bfloat16,
            "use_fast_accum": use_fast_accum,
        },
        name=name,
    )

  # w1/w3 contract over the model dim and w2 over the expert FFN dim. Expert 2
  # receives no tokens. Group sizes are multiples of 16 as the GPU kernels need.
  yield make_2d_3d_sample(
      f"qwen3_30b_a3b_k{_QWEN3_30B_A3B_DIM}",
      (16, 32, 0, 16),
      _QWEN3_30B_A3B_DIM,
      32,
  )
  yield make_2d_3d_sample(
      f"qwen3_30b_a3b_k{_QWEN3_30B_A3B_MOE_FFN_DIM}",
      (32, 16, 0, 16),
      _QWEN3_30B_A3B_MOE_FFN_DIM,
      32,
  )
  yield make_2d_2d_sample("weight_grad", (64, 16, 128, 48), 32, 48)
  # torchao's FP8 MoE training enables use_fast_accum in all grouped GEMMs.
  yield make_2d_3d_sample(
      f"qwen3_30b_a3b_k{_QWEN3_30B_A3B_DIM}_fast_accum",
      (16, 32, 0, 16),
      _QWEN3_30B_A3B_DIM,
      32,
      use_fast_accum=True,
  )
  yield make_2d_2d_sample(
      "weight_grad_fast_accum", (64, 16, 128, 48), 32, 48, use_fast_accum=True
  )


def _sample_inputs_safe_softmax(
    opinfo: Any,
    device: str | torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> collections.abc.Iterator[core.SampleInput]:
  """Custom sample generator for torch.ops.aten._safe_softmax.default supporting quantized dtypes."""
  # The upstream impl adds float32 masks to input tensors, which raises a type
  # promotion error if sampled directly on float8 tensors. To work it around,
  # generate samples in float32 and cast them to the requested dtype.
  for sample in common_methods_invocations.sample_inputs_safe_softmax(
      opinfo, device, torch.float32, requires_grad, **kwargs
  ):
    if isinstance(sample.input, torch.Tensor):
      sample.input = sample.input.to(dtype)
    yield sample


def _fused_adam_wrapper(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    **kwargs,
):
  return torch.ops.aten._fused_adam.default(
      self,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
      **kwargs,
  )


def _fused_adam_tensor_lr_wrapper(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    **kwargs,
):
  return torch.ops.aten._fused_adam.tensor_lr(
      self,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
      **kwargs,
  )


def _fused_adamw_wrapper(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    **kwargs,
):
  return torch.ops.aten._fused_adamw.default(
      self,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
      **kwargs,
  )


def _fused_adamw_tensor_lr_wrapper(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    **kwargs,
):
  return torch.ops.aten._fused_adamw.tensor_lr(
      self,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
      **kwargs,
  )


def _fused_sgd_wrapper(
    self,
    grads,
    momentum_buffer_list,
    **kwargs,
):
  return torch.ops.aten._fused_sgd.default(
      self,
      grads,
      momentum_buffer_list,
      **kwargs,
  )


def _fused_sgd_tensor_lr_wrapper(
    self,
    grads,
    momentum_buffer_list,
    **kwargs,
):
  return torch.ops.aten._fused_sgd.tensor_lr(
      self,
      grads,
      momentum_buffer_list,
      **kwargs,
  )


def _fused_adagrad_wrapper(
    self,
    grads,
    state_sums,
    state_steps,
    **kwargs,
):
  return torch.ops.aten._fused_adagrad.default(
      self,
      grads,
      state_sums,
      state_steps,
      **kwargs,
  )


def _fused_adagrad_tensor_lr_wrapper(
    self,
    grads,
    state_sums,
    state_steps,
    **kwargs,
):
  return torch.ops.aten._fused_adagrad.tensor_lr(
      self,
      grads,
      state_sums,
      state_steps,
      **kwargs,
  )


def _make_fused_opt_tensors(
    device: str | torch.device,
    param_dtype: torch.dtype,
    state_dtype: torch.dtype | None = None,
    shapes: Sequence[Sequence[int]] = ((4,), (2, 2)),
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
  """Constructs deterministic multi-tensor inputs for fused optimizers (no randn)."""
  if state_dtype is None:
    state_dtype = param_dtype
  compute_dtype = (
      torch.float64 if param_dtype == torch.float64 else torch.float32
  )
  ps, gs, ms, vs, max_vs, steps = [], [], [], [], [], []
  for i, shape in enumerate(shapes):
    numel = torch.Size(shape).numel()
    ps.append(
        torch.linspace(
            -1.5 + 0.25 * i,
            2.0 - 0.15 * i,
            steps=numel,
            dtype=compute_dtype,
            device=device,
        )
        .reshape(shape)
        .to(param_dtype)
    )
    gs.append(
        torch.linspace(
            -0.4 + 0.05 * i,
            0.35 - 0.05 * i,
            steps=numel,
            dtype=compute_dtype,
            device=device,
        )
        .reshape(shape)
        .to(param_dtype)
    )
    ms.append(
        torch.linspace(
            -0.04 + 0.01 * i,
            0.05 - 0.01 * i,
            steps=numel,
            dtype=compute_dtype,
            device=device,
        )
        .reshape(shape)
        .to(state_dtype)
    )
    # Strictly positive second-moment / sum accumulators (> 0) to prevent 1/sqrt(0).
    vs.append(
        torch.linspace(
            0.02 + 0.01 * i,
            0.25 + 0.05 * i,
            steps=numel,
            dtype=compute_dtype,
            device=device,
        )
        .reshape(shape)
        .to(state_dtype)
    )
    max_vs.append(
        torch.linspace(
            0.05 + 0.01 * i,
            0.35 + 0.05 * i,
            steps=numel,
            dtype=compute_dtype,
            device=device,
        )
        .reshape(shape)
        .to(state_dtype)
    )
    # Auxiliary step counters must always be float32 on device for CUDA kernels.
    steps.append(torch.tensor(float(i + 1), dtype=torch.float32, device=device))
  return ps, gs, ms, vs, max_vs, steps


def _make_dummy_unsupported_tensors(
    device: str | torch.device, dtype: torch.dtype
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
  if dtype in quantize_utils.SUB_BYTE_DTYPES:
    t = torch.zeros((4,), dtype=torch.uint8, device=device).view(dtype)
  elif dtype == torch.bool:
    t = torch.tensor([True, False, True, False], dtype=dtype, device=device)
  elif dtype.is_complex:
    t = torch.tensor(
        [1.0 + 1.0j, 2.0 + 1.0j, 1.0 + 1.0j, 2.0 + 1.0j],
        dtype=dtype,
        device=device,
    )
  else:
    t = torch.tensor([1, 2, 1, 2], dtype=dtype, device=device)
  step = torch.tensor(1.0, dtype=torch.float32, device=device)
  return [t], [t.clone()], [t.clone()], [t.clone()], [t.clone()], [step]


def _sample_inputs_fused_adam(
    opinfo: Any,
    device: str | torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> collections.abc.Iterator[core.SampleInput]:
  """Deterministic sample generator for _fused_adam_ and _fused_adamw_ (no randn)."""
  del requires_grad, kwargs
  use_tensor_lr = opinfo.name.endswith(".tensor_lr")
  lr_val = (
      torch.tensor(0.01, dtype=torch.float32, device=device)
      if use_tensor_lr
      else 0.01
  )

  if not dtype.is_floating_point:
    ps, gs, ms, vs, _, steps = _make_dummy_unsupported_tensors(device, dtype)
    yield core.SampleInput(
        ps,
        args=(gs, ms, vs, [], steps),
        kwargs={
            "lr": lr_val,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
        },
        name="unsupported_dtype",
    )
    return

  # 1. Basic step with default hyperparameters.
  ps, gs, ms, vs, _, steps = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms, vs, [], steps),
      kwargs={
          "lr": lr_val,
          "beta1": 0.9,
          "beta2": 0.999,
          "weight_decay": 0.0,
          "eps": 1e-8,
          "amsgrad": False,
          "maximize": False,
      },
      name="basic_step",
  )

  # 2. amsgrad=True, non-zero weight_decay=0.01, maximize=True.
  ps, gs, ms, vs, max_vs, steps = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms, vs, max_vs, steps),
      kwargs={
          "lr": lr_val,
          "beta1": 0.9,
          "beta2": 0.999,
          "weight_decay": 0.01,
          "eps": 1e-8,
          "amsgrad": True,
          "maximize": True,
      },
      name="amsgrad_wd_maximize",
  )

  # 3. grad_scale + found_inf=0.0 (scaled gradient step).
  ps, gs, ms, vs, _, steps = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms, vs, [], steps),
      kwargs={
          "lr": lr_val,
          "beta1": 0.9,
          "beta2": 0.999,
          "weight_decay": 0.01,
          "eps": 1e-8,
          "amsgrad": False,
          "maximize": False,
          "grad_scale": torch.tensor(2.0, dtype=torch.float32, device=device),
          "found_inf": torch.tensor(0.0, dtype=torch.float32, device=device),
      },
      name="grad_scale_step",
  )

  # 4. grad_scale + found_inf=1.0 (verifying no-op when inf is found).
  ps, gs, ms, vs, _, steps = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms, vs, [], steps),
      kwargs={
          "lr": lr_val,
          "beta1": 0.9,
          "beta2": 0.999,
          "weight_decay": 0.01,
          "eps": 1e-8,
          "amsgrad": False,
          "maximize": False,
          "grad_scale": torch.tensor(2.0, dtype=torch.float32, device=device),
          "found_inf": torch.tensor(1.0, dtype=torch.float32, device=device),
      },
      name="grad_scale_found_inf",
  )

  # 5. Multi-rank/multi-size tensor list (1D, 2D, 3D, 4D).
  ps, gs, ms, vs, _, steps = _make_fused_opt_tensors(
      device, dtype, shapes=[(7,), (3, 5), (2, 3, 4), (2, 2, 2, 2)]
  )
  yield core.SampleInput(
      ps,
      args=(gs, ms, vs, [], steps),
      kwargs={
          "lr": lr_val,
          "beta1": 0.9,
          "beta2": 0.999,
          "weight_decay": 0.01,
          "eps": 1e-8,
          "amsgrad": False,
          "maximize": False,
      },
      name="multi_rank_shapes",
  )

  # 6. Single-tensor parameter list (len(params) == 1).
  ps, gs, ms, vs, _, steps = _make_fused_opt_tensors(
      device, dtype, shapes=[(3, 5)]
  )
  yield core.SampleInput(
      ps,
      args=(gs, ms, vs, [], steps),
      kwargs={
          "lr": lr_val,
          "beta1": 0.9,
          "beta2": 0.999,
          "weight_decay": 0.0,
          "eps": 1e-8,
          "amsgrad": False,
          "maximize": False,
      },
      name="single_tensor",
  )

  # 7. Mixed-precision (float32 params/grads with bfloat16 state tensors).
  if dtype == torch.float32 and str(device) != "cpu":
    ps, gs, ms, vs, max_vs, steps = _make_fused_opt_tensors(
        device, param_dtype=torch.float32, state_dtype=torch.bfloat16
    )
    yield core.SampleInput(
        ps,
        args=(gs, ms, vs, max_vs, steps),
        kwargs={
            "lr": lr_val,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "amsgrad": True,
            "maximize": False,
        },
        name="mixed_precision_bf16_state",
    )


def _sample_inputs_fused_sgd(
    opinfo: Any,
    device: str | torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> collections.abc.Iterator[core.SampleInput]:
  """Deterministic sample generator for _fused_sgd_ (no randn)."""
  del requires_grad, kwargs
  use_tensor_lr = opinfo.name.endswith(".tensor_lr")
  lr_val = (
      torch.tensor(0.01, dtype=torch.float32, device=device)
      if use_tensor_lr
      else 0.01
  )

  if not dtype.is_floating_point:
    ps, gs, _, _, _, _ = _make_dummy_unsupported_tensors(device, dtype)
    yield core.SampleInput(
        ps,
        args=(gs, []),
        kwargs={
            "weight_decay": 0.0,
            "momentum": 0.0,
            "lr": lr_val,
            "dampening": 0.0,
            "nesterov": False,
            "maximize": False,
            "is_first_step": False,
        },
        name="unsupported_dtype",
    )
    return

  # 1. Basic momentum step (is_first_step=False).
  ps, gs, ms, _, _, _ = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms),
      kwargs={
          "weight_decay": 0.0,
          "momentum": 0.9,
          "lr": lr_val,
          "dampening": 0.0,
          "nesterov": False,
          "maximize": False,
          "is_first_step": False,
      },
      name="basic_momentum_step",
  )

  # 2. Vanilla SGD (momentum=0.0, momentum_buffer_list=[], weight_decay=0.01).
  ps, gs, _, _, _, _ = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, []),
      kwargs={
          "weight_decay": 0.01,
          "momentum": 0.0,
          "lr": lr_val,
          "dampening": 0.0,
          "nesterov": False,
          "maximize": False,
          "is_first_step": False,
      },
      name="vanilla_sgd_wd",
  )

  # 3a. Nesterov momentum (nesterov=True, dampening=0.0) with is_first_step=True.
  ps, gs, ms, _, _, _ = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms),
      kwargs={
          "weight_decay": 0.01,
          "momentum": 0.9,
          "lr": lr_val,
          "dampening": 0.0,
          "nesterov": True,
          "maximize": False,
          "is_first_step": True,
      },
      name="nesterov_first_step",
  )

  # 3b. Nesterov momentum (nesterov=True, dampening=0.0) with is_first_step=False.
  ps, gs, ms, _, _, _ = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms),
      kwargs={
          "weight_decay": 0.01,
          "momentum": 0.9,
          "lr": lr_val,
          "dampening": 0.0,
          "nesterov": True,
          "maximize": False,
          "is_first_step": False,
      },
      name="nesterov_subsequent_step",
  )

  # 4. grad_scale + found_inf=0.0 (is_first_step=True).
  ps, gs, ms, _, _, _ = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms),
      kwargs={
          "weight_decay": 0.01,
          "momentum": 0.9,
          "lr": lr_val,
          "dampening": 0.1,
          "nesterov": False,
          "maximize": True,
          "is_first_step": True,
          "grad_scale": torch.tensor(2.0, dtype=torch.float32, device=device),
          "found_inf": torch.tensor(0.0, dtype=torch.float32, device=device),
      },
      name="grad_scale_first_step",
  )

  # 5. grad_scale + found_inf=1.0 (verifying no-op when inf is found).
  ps, gs, ms, _, _, _ = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, ms),
      kwargs={
          "weight_decay": 0.01,
          "momentum": 0.9,
          "lr": lr_val,
          "dampening": 0.1,
          "nesterov": False,
          "maximize": False,
          "is_first_step": False,
          "grad_scale": torch.tensor(2.0, dtype=torch.float32, device=device),
          "found_inf": torch.tensor(1.0, dtype=torch.float32, device=device),
      },
      name="grad_scale_found_inf",
  )

  # 6. Multi-rank/multi-size tensor list (1D, 2D, 3D, 4D).
  ps, gs, ms, _, _, _ = _make_fused_opt_tensors(
      device, dtype, shapes=[(7,), (3, 5), (2, 3, 4), (2, 2, 2, 2)]
  )
  yield core.SampleInput(
      ps,
      args=(gs, ms),
      kwargs={
          "weight_decay": 0.01,
          "momentum": 0.9,
          "lr": lr_val,
          "dampening": 0.0,
          "nesterov": False,
          "maximize": False,
          "is_first_step": False,
      },
      name="multi_rank_shapes",
  )

  # 7. Single-tensor parameter list (len(params) == 1).
  ps, gs, ms, _, _, _ = _make_fused_opt_tensors(device, dtype, shapes=[(3, 5)])
  yield core.SampleInput(
      ps,
      args=(gs, ms),
      kwargs={
          "weight_decay": 0.0,
          "momentum": 0.9,
          "lr": lr_val,
          "dampening": 0.0,
          "nesterov": False,
          "maximize": False,
          "is_first_step": False,
      },
      name="single_tensor",
  )


def _sample_inputs_fused_adagrad(
    opinfo: Any,
    device: str | torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> collections.abc.Iterator[core.SampleInput]:
  """Deterministic sample generator for _fused_adagrad_ (no randn)."""
  del requires_grad, kwargs
  use_tensor_lr = opinfo.name.endswith(".tensor_lr")
  lr_val = (
      torch.tensor(0.01, dtype=torch.float32, device=device)
      if use_tensor_lr
      else 0.01
  )

  if not dtype.is_floating_point:
    ps, gs, _, vs, _, steps = _make_dummy_unsupported_tensors(device, dtype)
    yield core.SampleInput(
        ps,
        args=(gs, vs, steps),
        kwargs={
            "lr": lr_val,
            "lr_decay": 0.0,
            "weight_decay": 0.0,
            "eps": 1e-10,
            "maximize": False,
        },
        name="unsupported_dtype",
    )
    return

  # 1. Basic step.
  ps, gs, _, vs, _, steps = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, vs, steps),
      kwargs={
          "lr": lr_val,
          "lr_decay": 0.0,
          "weight_decay": 0.0,
          "eps": 1e-10,
          "maximize": False,
      },
      name="basic_step",
  )

  # 2. Non-zero lr_decay=0.01, weight_decay=0.01, maximize=True.
  ps, gs, _, vs, _, steps = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, vs, steps),
      kwargs={
          "lr": lr_val,
          "lr_decay": 0.01,
          "weight_decay": 0.01,
          "eps": 1e-10,
          "maximize": True,
      },
      name="lr_decay_wd_maximize",
  )

  # 3. grad_scale + found_inf=0.0.
  ps, gs, _, vs, _, steps = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, vs, steps),
      kwargs={
          "lr": lr_val,
          "lr_decay": 0.01,
          "weight_decay": 0.01,
          "eps": 1e-10,
          "maximize": False,
          "grad_scale": torch.tensor(2.0, dtype=torch.float32, device=device),
          "found_inf": torch.tensor(0.0, dtype=torch.float32, device=device),
      },
      name="grad_scale_step",
  )

  # 4. grad_scale + found_inf=1.0 (verifying no-op when inf is found).
  ps, gs, _, vs, _, steps = _make_fused_opt_tensors(device, dtype)
  yield core.SampleInput(
      ps,
      args=(gs, vs, steps),
      kwargs={
          "lr": lr_val,
          "lr_decay": 0.01,
          "weight_decay": 0.01,
          "eps": 1e-10,
          "maximize": False,
          "grad_scale": torch.tensor(2.0, dtype=torch.float32, device=device),
          "found_inf": torch.tensor(1.0, dtype=torch.float32, device=device),
      },
      name="grad_scale_found_inf",
  )

  # 5. Multi-rank/multi-size tensor list (1D, 2D, 3D, 4D).
  ps, gs, _, vs, _, steps = _make_fused_opt_tensors(
      device, dtype, shapes=[(7,), (3, 5), (2, 3, 4), (2, 2, 2, 2)]
  )
  yield core.SampleInput(
      ps,
      args=(gs, vs, steps),
      kwargs={
          "lr": lr_val,
          "lr_decay": 0.01,
          "weight_decay": 0.01,
          "eps": 1e-10,
          "maximize": False,
      },
      name="multi_rank_shapes",
  )

  # 6. Single-tensor parameter list (len(params) == 1).
  ps, gs, _, vs, _, steps = _make_fused_opt_tensors(
      device, dtype, shapes=[(3, 5)]
  )
  yield core.SampleInput(
      ps,
      args=(gs, vs, steps),
      kwargs={
          "lr": lr_val,
          "lr_decay": 0.0,
          "weight_decay": 0.0,
          "eps": 1e-10,
          "maximize": False,
      },
      name="single_tensor",
  )


def _make_sparse_coo_sample_tensor(
    shape: Sequence[int],
    sparse_dim: int,
    nnz: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    coalesced: bool = True,
) -> torch.Tensor:
  """Helper to create a sparse COO sample tensor for testing."""
  dense_dim = len(shape) - sparse_dim
  if nnz == 0:
    return torch.ops.aten._sparse_coo_tensor_with_dims(
        sparse_dim,
        dense_dim,
        shape,
        dtype=dtype,
        layout=torch.sparse_coo,
        device=device,
    )
  indices = torch.stack(
      [
          torch.randint(
              0, max(1, shape[d]), (nnz,), device=device, dtype=torch.int64
          )
          for d in range(sparse_dim)
      ],
      dim=0,
  )
  if not coalesced and nnz >= 2:
    indices[:, 1] = indices[:, 0]
  values_shape = (nnz, *shape[sparse_dim:])
  values = common_utils.make_tensor(
      values_shape, dtype=dtype, device=device, low=-5, high=5
  )
  sp = torch.sparse_coo_tensor(
      indices, values, size=shape, dtype=dtype, device=device
  )
  if coalesced:
    sp = sp.coalesce()
  return sp


def _sample_inputs_sparse_coo_binary(
    op_info: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> Iterable[SampleInput]:
  del op_info, requires_grad, kwargs
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (5, 6), 2, 4, device, dtype, coalesced=True
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (5, 6), 2, 3, device, dtype, coalesced=True
          ),
      ),
      kwargs={"alpha": 1},
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (5, 6), 2, 5, device, dtype, coalesced=False
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (5, 6), 2, 4, device, dtype, coalesced=True
          ),
      ),
      kwargs={"alpha": 2},
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (4, 5, 3), 2, 3, device, dtype, coalesced=True
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (4, 5, 3), 2, 3, device, dtype, coalesced=True
          ),
      ),
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (4, 4), 2, 0, device, dtype, coalesced=True
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (4, 4), 2, 3, device, dtype, coalesced=True
          ),
      ),
  )


def _sample_inputs_dense_sparse_binary(
    op_info: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> Iterable[SampleInput]:
  del op_info, requires_grad, kwargs
  yield SampleInput(
      common_utils.make_tensor(
          (5, 6), dtype=dtype, device=device, low=-5, high=5
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (5, 6), 2, 4, device, dtype, coalesced=True
          ),
      ),
      kwargs={"alpha": 1},
  )
  yield SampleInput(
      common_utils.make_tensor(
          (5, 6), dtype=dtype, device=device, low=-5, high=5
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (5, 6), 2, 5, device, dtype, coalesced=False
          ),
      ),
      kwargs={"alpha": 2},
  )
  yield SampleInput(
      common_utils.make_tensor(
          (4, 5, 3), dtype=dtype, device=device, low=-5, high=5
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (4, 5, 3), 2, 3, device, dtype, coalesced=True
          ),
      ),
  )
  yield SampleInput(
      common_utils.make_tensor(
          (4, 4), dtype=dtype, device=device, low=-5, high=5
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (4, 4), 2, 0, device, dtype, coalesced=True
          ),
      ),
  )


def _sample_inputs_sparse_coo_mul(
    op_info: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> Iterable[SampleInput]:
  del op_info, requires_grad, kwargs
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (5, 6), 2, 4, device, dtype, coalesced=True
      ),
      args=(torch.tensor(3, dtype=dtype, device=device),),
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (5, 6), 2, 5, device, dtype, coalesced=False
      ),
      args=(torch.tensor(2, dtype=dtype, device=device),),
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (4, 5, 3), 2, 3, device, dtype, coalesced=True
      ),
      args=(torch.tensor(4, dtype=dtype, device=device),),
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (4, 4), 2, 0, device, dtype, coalesced=True
      ),
      args=(torch.tensor(2, dtype=dtype, device=device),),
  )


def _sample_inputs_sparse_coo_unary(
    op_info: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> Iterable[SampleInput]:
  del op_info, requires_grad, kwargs
  yield SampleInput(
      _make_sparse_coo_sample_tensor((8,), 1, 4, device, dtype, coalesced=True)
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (5, 6), 2, 4, device, dtype, coalesced=True
      )
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (5, 6), 2, 5, device, dtype, coalesced=False
      )
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (4, 5, 3), 2, 4, device, dtype, coalesced=False
      )
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (4, 4), 2, 0, device, dtype, coalesced=True
      )
  )


def _sample_inputs_sparse_coo_copy_(
    op_info: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    **kwargs: Any,
) -> Iterable[SampleInput]:
  del op_info, requires_grad, kwargs
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (5, 6), 2, 2, device, dtype, coalesced=True
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (5, 6), 2, 4, device, dtype, coalesced=True
          ),
      ),
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (4, 5, 3), 2, 1, device, dtype, coalesced=True
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (4, 5, 3), 2, 3, device, dtype, coalesced=True
          ),
      ),
  )
  yield SampleInput(
      _make_sparse_coo_sample_tensor(
          (4, 4), 2, 3, device, dtype, coalesced=True
      ),
      args=(
          _make_sparse_coo_sample_tensor(
              (4, 4), 2, 0, device, dtype, coalesced=True
          ),
      ),
  )


# Ops not included in the list of tested ops for pytorch.
_ADDITIONAL_TORCH_TPU_OPS: Final[Sequence[OpInfo]] = [
    OpInfo(
        "_fused_adagrad_",
        op=_fused_adagrad_wrapper,
        aten_name="_fused_adagrad_",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_fused_adagrad,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "_fused_adagrad_.tensor_lr",
        op=_fused_adagrad_tensor_lr_wrapper,
        aten_name="_fused_adagrad_.tensor_lr",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_fused_adagrad,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "_fused_adam_",
        op=_fused_adam_wrapper,
        aten_name="_fused_adam_",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_fused_adam,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "_fused_adam_.tensor_lr",
        op=_fused_adam_tensor_lr_wrapper,
        aten_name="_fused_adam_.tensor_lr",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_fused_adam,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "_fused_adamw_",
        op=_fused_adamw_wrapper,
        aten_name="_fused_adamw_",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_fused_adam,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "_fused_adamw_.tensor_lr",
        op=_fused_adamw_tensor_lr_wrapper,
        aten_name="_fused_adamw_.tensor_lr",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_fused_adam,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "_fused_sgd_",
        op=_fused_sgd_wrapper,
        aten_name="_fused_sgd_",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_fused_sgd,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "_fused_sgd_.tensor_lr",
        op=_fused_sgd_tensor_lr_wrapper,
        aten_name="_fused_sgd_.tensor_lr",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_fused_sgd,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "torch._scaled_mm",
        # PyTorch's own torch._scaled_mm OpInfo only covers tensorwise scaling.
        # Note that _get_op("torch._scaled_mm") without a variant_test_name
        # returns this OpInfo, since it precedes PyTorch's in _KNOWN_OPS.
        variant_test_name="rowwise",
        op=torch._scaled_mm,  # pylint: disable=protected-access
        dtypes=common_dtype._dispatch_dtypes((torch.float8_e4m3fn,)),  # pylint: disable=protected-access
        sample_inputs_func=_sample_inputs_scaled_mm_rowwise,
        supports_out=True,
        # Quantized primitive without autograd formula.
        supports_autograd=False,
    ),
    OpInfo(
        "torch._scaled_grouped_mm",
        variant_test_name="rowwise",
        op=torch._scaled_grouped_mm,  # pylint: disable=protected-access
        dtypes=common_dtype._dispatch_dtypes((torch.float8_e4m3fn,)),  # pylint: disable=protected-access
        sample_inputs_func=_sample_inputs_scaled_grouped_mm_rowwise,
        supports_out=False,
        # Quantized primitive without autograd formula.
        supports_autograd=False,
    ),
    OpInfo(
        "torch._scaled_grouped_mm_v2",
        op=torch._scaled_grouped_mm_v2,  # pylint: disable=protected-access
        ref=_ref_scaled_grouped_mm_v2,
        # Input matrices for _scaled_grouped_mm_v2 are quantized FP8 tensors.
        dtypes=common_methods_invocations.float8_types(),
        sample_inputs_func=_sample_inputs_scaled_grouped_mm_v2,
        supports_out=True,
        # Quantized primitive without autograd formula.
        supports_autograd=False,
    ),
    OpInfo(
        "torch._scaled_mm_v2",
        op=torch._scaled_mm_v2,  # pylint: disable=protected-access
        ref=_ref_scaled_mm_v2,
        # Input matrices for _scaled_mm_v2 are quantized FP8 tensors.
        dtypes=common_methods_invocations.float8_types(),
        sample_inputs_func=_sample_inputs_scaled_mm_v2,
        supports_out=True,
        # Inference-only quantized primitive without autograd formula.
        supports_autograd=False,
    ),
    OpInfo(
        "torch.ops.aten._unsafe_view",
        op=lambda x, shape: x.view(shape),
        dtypes=common_dtype.all_types_and_complex_and(
            torch.complex32, torch.bool, torch.float16, torch.bfloat16
        ),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_jit_shape_analysis=True,
        sample_inputs_func=common_methods_invocations.sample_inputs_view_reshape,
        reference_inputs_func=common_methods_invocations.reference_inputs_view_reshape,
        error_inputs_func=common_methods_invocations.error_inputs_view_reshape,
        skips=(
            core.DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # RuntimeError: view size is not compatible with input tensor's size
            # and stride (at least one dimension spans across two contiguous
            # subspaces). Use .reshape(...) instead.
            core.DecorateInfo(
                unittest.expectedFailure,
                "TestMeta",
                "test_dispatch_symbolic_meta_outplace_all_strides",
            ),
        ),
    ),
    OpInfo(
        "_log_softmax_backward_data",
        op=torch.ops.aten._log_softmax_backward_data,  # pylint: disable=protected-access
        aten_name="_log_softmax_backward_data",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=(
            common_methods_invocations.sample_inputs_softmax_backward_data
        ),
        assert_autodiffed=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),
    OpInfo(
        "_thnn_fused_lstm_cell",
        op=torch.ops.aten._thnn_fused_lstm_cell,  # pylint: disable=protected-access
        ref=_ref_thnn_fused_lstm_cell,
        aten_name="_thnn_fused_lstm_cell",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_thnn_fused_rnn_cell,
        assert_autodiffed=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=True,
    ),
    OpInfo(
        "_thnn_fused_gru_cell",
        op=torch.ops.aten._thnn_fused_gru_cell,  # pylint: disable=protected-access
        ref=_ref_thnn_fused_gru_cell,
        aten_name="_thnn_fused_gru_cell",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_thnn_fused_rnn_cell,
        assert_autodiffed=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=True,
    ),
    OpInfo(
        "_native_multi_head_attention",
        op=torch.ops.aten._native_multi_head_attention,  # pylint: disable=protected-access
        aten_name="_native_multi_head_attention",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_native_multi_head_attention,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),
    OpInfo(
        "_transformer_encoder_layer_fwd",
        op=torch.ops.aten._transformer_encoder_layer_fwd,  # pylint: disable=protected-access
        aten_name="_transformer_encoder_layer_fwd",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_transformer_encoder_layer_fwd,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),
    OpInfo(
        "_upsample_bicubic2d_aa",
        op=torch.ops.aten._upsample_bicubic2d_aa,  # pylint: disable=protected-access
        aten_name="_upsample_bicubic2d_aa",
        dtypes=common_dtype.floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=functools.partial(
            common_methods_invocations.sample_inputs_upsample_aa, "bicubic"
        ),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),
    OpInfo(
        "torch.ops.aten._safe_softmax.default",
        dtypes=common_dtype.all_types_and(
            torch.half, torch.bfloat16, torch.bool
        ),
        sample_inputs_func=_sample_inputs_safe_softmax,
        assert_jit_shape_analysis=True,
        assert_autodiffed=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
        supports_cow_input_no_materialize_backward=False,
    ),
    OpInfo(
        "sparse_coo_add",
        op=torch.add,
        inplace_variant=lambda x, y, **kwargs: x.add_(y, **kwargs),
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_sparse_coo_binary,
        supports_out=True,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_add_dense_sparse",
        op=torch.add,
        inplace_variant=lambda x, y, **kwargs: x.add_(y, **kwargs),
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_dense_sparse_binary,
        supports_out=True,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_sub",
        op=torch.sub,
        inplace_variant=lambda x, y, **kwargs: x.sub_(y, **kwargs),
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_sparse_coo_binary,
        supports_out=True,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_sub_dense_sparse",
        op=torch.sub,
        inplace_variant=lambda x, y, **kwargs: x.sub_(y, **kwargs),
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_dense_sparse_binary,
        supports_out=True,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_mul",
        op=torch.mul,
        inplace_variant=lambda x, y: x.mul_(y),
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_sparse_coo_mul,
        supports_out=True,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_neg",
        op=torch.neg,
        inplace_variant=lambda x: x.neg_(),
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=_sample_inputs_sparse_coo_unary,
        supports_out=True,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_coalesce",
        op=lambda x: x.coalesce(),
        dtypes=common_dtype.all_types_and(
            torch.bfloat16, torch.float16, torch.bool
        ),
        sample_inputs_func=_sample_inputs_sparse_coo_unary,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_to_dense",
        op=lambda x: x.to_dense(),
        dtypes=common_dtype.all_types_and(
            torch.bfloat16, torch.float16, torch.bool
        ),
        sample_inputs_func=_sample_inputs_sparse_coo_unary,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_clone",
        op=torch.clone,
        dtypes=common_dtype.all_types_and(
            torch.bfloat16, torch.float16, torch.bool
        ),
        sample_inputs_func=_sample_inputs_sparse_coo_unary,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_zero_",
        op=lambda x: x.clone().zero_(),
        inplace_variant=lambda x: x.zero_(),
        dtypes=common_dtype.all_types_and(
            torch.bfloat16, torch.float16, torch.bool
        ),
        sample_inputs_func=_sample_inputs_sparse_coo_unary,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_empty_like",
        op=lambda x: torch.empty_like(x).zero_(),
        dtypes=common_dtype.all_types_and(
            torch.bfloat16, torch.float16, torch.bool
        ),
        sample_inputs_func=_sample_inputs_sparse_coo_unary,
        supports_out=False,
        supports_autograd=False,
    ),
    OpInfo(
        "sparse_coo_copy_",
        op=lambda x, y: x.clone().copy_(y),
        inplace_variant=lambda x, y: x.copy_(y),
        dtypes=common_dtype.all_types_and(
            torch.bfloat16, torch.float16, torch.bool
        ),
        sample_inputs_func=_sample_inputs_sparse_coo_copy_,
        supports_out=False,
        supports_autograd=False,
    ),
]
for _op_info in _ADDITIONAL_TORCH_TPU_OPS:
  if _op_info.name in (
      "_thnn_fused_lstm_cell",
      "_thnn_fused_gru_cell",
      "torch._scaled_grouped_mm_v2",
      "torch._scaled_mm_v2",
  ):
    _op_info.use_ref_for_cpu_golden = True


def _dtype_str(dtype: torch.dtype) -> str:
  """Returns a string representation of the dtype without the "torch." prefix."""
  return str(dtype).removeprefix("torch.")


def _golden_dtype_key(
    dtype: torch.dtype, out_dtype_override: torch.dtype | None = None
) -> str:
  """Returns the dictionary key for golden results in `_GOLDEN_GPU_DATA`.

  For standard tests, returns the string representation of `dtype` (e.g.,
  `"float32"`). For out-variant tests with an overridden out dtype, returns a
  composite key formatted as `"{input_dtype}_out_{out_dtype}"` (e.g.,
  `"float32_out_float64"`).

  In out-variant tests, "override" means the `out=` tensor is allocated with a
  different dtype (`out_dtype_override`) than the input tensor's dtype (`dtype`)
  to test out-variant dtype casting (e.g., `at::can_cast` upcasting or
  exception parity on illegal downcasting).

  Example:
    `_golden_dtype_key(torch.float32, torch.float64)` returns
    `"float32_out_float64"`, indicating an out-variant test where the input has
    `float32` dtype but the `out=` tensor is allocated with `float64` dtype.
  """
  if out_dtype_override is not None:
    return f"{_dtype_str(dtype)}_out_{_dtype_str(out_dtype_override)}"
  return _dtype_str(dtype)


_GoldenSample = (
    tuple["OpInput", "OpOutput"] | Callable[[], tuple["OpInput", "OpOutput"]]
)


class GoldenGpuData:
  """Golden GPU input/output samples collected per op test.

  The samples are organized in a three-level structure keyed by the test case
  name, the op variant (e.g. BASE, INPLACE, or OUT), and the dtype_key:

    test_case_name -> op variant -> dtype_key -> [sample, ...]

  Each sample is either a decoded `(OpInput, OpOutput)` pair or a zero-argument
  callable that lazily deserializes and returns the pair on demand.

  The `dtype_key` is a string:
    - For standard tests: the dtype name without "torch." prefix (e.g.,
    "float32").
    - For out-variant tests with overridden out dtypes: a composite key
      formatted as `"{input_dtype}_out_{out_dtype}"` (e.g.,
      "float32_out_float64").

  Because each test case can contain at most one `do_test_op()` (which is
  enforced), a `(test_case_name, variant, dtype_key)` triple uniquely identifies
  a list of golden input/output samples.

  Usage:
    * In the gen_gpu_golden mode, `add()` collects the golden results as they
      are computed, and `to_plistlib_pytree()` serializes them to a golden file.
    * In the torch_tpu_vs_gpu mode, `merge_plistlib_pytree()` populates this
      from the GPU golden files and `get_samples()` reads the results back to
      compare TorchTPU results against the GPU results.
  """

  def __init__(self) -> None:
    # Maps test_case_name -> OpVariant value -> dtype_key -> samples.
    self._data: dict[
        str,
        dict[str, dict[str, list[_GoldenSample]]],
    ] = {}

  def add(
      self,
      test_case_name: str,
      variant: "OpVariant",
      dtype: torch.dtype,
      op_input: "OpInput",
      op_output: "OpOutput",
      out_dtype_override: torch.dtype | None = None,
  ) -> None:
    """Records a single golden (input, output) sample."""
    key = _golden_dtype_key(dtype, out_dtype_override)
    self._samples_for(test_case_name, variant.value, key).append(
        (op_input, op_output)
    )

  def _ensure_test_case_loaded(self, test_case_name: str) -> None:
    """Lazily loads the golden samples for `test_case_name`, if present."""
    if test_case_name in self._data:
      return
    candidate_dirs = tests_test_utils.goldens_candidate_local_dirs(
        base_dir=_GOLDEN_DATA_BASE_DIR.value
    )
    per_test_case_dir = tests_test_utils.find_per_test_case_goldens_dir(
        _golden_file_prefix() + _GPU_GOLDEN_PER_TEST_CASE_SUFFIX, candidate_dirs
    )
    if per_test_case_dir is None:
      # TODO(b/576014545): Remove fallback once we add per-test-case loading
      # in OSS. If `per_test_case_dir` is `None`, goldens were already all
      # loaded by _load_golden_files().
      return
    op_tree = tests_test_utils.load_single_test_case_golden(
        test_case_name,
        per_test_case_dir,
    )
    if op_tree is not None:
      self.merge_plistlib_pytree({test_case_name: op_tree})

  def get_samples(
      self,
      test_case_name: str,
      variant: "OpVariant",
      dtype: torch.dtype,
      out_dtype_override: torch.dtype | None = None,
  ) -> Sequence[tuple["OpInput", "OpOutput"]]:
    """Returns the recorded samples for the key, or an empty list if absent."""
    self._ensure_test_case_loaded(test_case_name)
    key = _golden_dtype_key(dtype, out_dtype_override)
    samples = (
        self._data.get(test_case_name, {}).get(variant.value, {}).get(key, [])
    )
    # Lazily deserialize callable samples on first access, caching the result
    # in-place to avoid re-decoding on subsequent reads.
    for i, sample in enumerate(samples):
      if callable(sample):
        samples[i] = sample()
    return typing.cast(Sequence[tuple["OpInput", "OpOutput"]], samples)

  def clear(self) -> None:
    """Removes all recorded golden data."""
    self._data.clear()

  def _samples_for(
      self, test_case_name: str, variant_name: str, dtype_key: str
  ) -> list[_GoldenSample]:
    """Returns the mutable sample list for the key, creating it if needed."""
    return (
        self._data.setdefault(test_case_name, {})
        .setdefault(variant_name, {})
        .setdefault(dtype_key, [])
    )

  def to_plistlib_pytree(self) -> _pytree.PyTree:
    """Encodes the golden data into a plistlib-compatible pytree."""
    # Ensure any lazily-deferred samples are fully deserialized before saving.
    for variant_to_dt in self._data.values():
      for dt_to_samples in variant_to_dt.values():
        for samples in dt_to_samples.values():
          for i, sample in enumerate(samples):
            if callable(sample):
              samples[i] = sample()

    def leaf_func(x: "OpInput | OpOutput") -> _pytree.PyTree:
      return x.to_plistlib_pytree()

    def is_leaf(x: Any) -> bool:
      return isinstance(x, (OpInput, OpOutput))

    return _pytree.tree_map(leaf_func, self._data, is_leaf=is_leaf)

  def merge_plistlib_pytree(self, encoded_data: _pytree.PyTree) -> None:
    """Decodes a plistlib-compatible pytree and merges it into this data."""
    for test_case_name, variant_to_dt in encoded_data.items():
      for variant_name, dt_to_encoded_samples in variant_to_dt.items():
        for dtype_name, encoded_samples in dt_to_encoded_samples.items():
          samples = self._samples_for(test_case_name, variant_name, dtype_name)
          # Store lazy callables instead of eagerly deserializing all tensors.
          # Default args (inp=..., out=...) are needed so each loop iteration
          # captures its sample values at function definition time.
          for encoded_op_input, encoded_op_output in encoded_samples:
            samples.append(
                lambda inp=encoded_op_input, out=encoded_op_output: (
                    OpInput.from_plistlib_pytree(inp),
                    OpOutput.from_plistlib_pytree(out),
                )
            )


# Collects the golden GPU results for each op test. See GoldenGpuData for the
# structure and the modes that read from / write to it.
_GOLDEN_GPU_DATA: GoldenGpuData = GoldenGpuData()

# Used in the dump_dtype_exclusions mode to collect GPU dtype exclusions of each
# `do_test_op` call site. The key is op name, and the value is a mapping from
# exclusion kinds (`exclude_dtypes` or `exclude_inplace_dtypes`) to a set of
# dtype names.
#
# Ops with multiple `do_test_op` call sites are intersected. For instance, for
# a given op O with exclude_dtypes = {a, b} in one call site and
# exclude_dtypes = {b, c} in the other, the recorded exclusion will be {b}.
_DTYPE_EXCLUSIONS: MutableMapping[str, dict[str, set[str]]] = {}

# The full list of known ops. We will test a subset of these.
_KNOWN_OPS: Final[Sequence[OpInfo]] = (
    typing.cast(list[OpInfo], _ADDITIONAL_TORCH_TPU_OPS)
    + _masked.op_db
    + autograd_function_db.autograd_function_db
    + common_methods_invocations.foreach_binary_op_db
    + common_methods_invocations.foreach_other_op_db
    + common_methods_invocations.foreach_pointwise_op_db
    + common_methods_invocations.foreach_reduce_op_db
    + common_methods_invocations.foreach_unary_op_db
    + common_methods_invocations.op_db
    + custom_op_db.custom_op_db
    + fft.op_db
    + hop_db.hop_db
    + linalg.op_db
)


COMPLEX_DTYPES: Final[Sequence[torch.dtype]] = (
    # TODO: add complex128.
    torch.complex64,
)
FLOAT_DTYPES: Final[Sequence[torch.dtype]] = (
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
)
INTEGRAL_DTYPES: Final[Sequence[torch.dtype]] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)

UNQUANTIZED_NUMERIC_DTYPES: Final[Sequence[torch.dtype]] = (
    *COMPLEX_DTYPES,
    *FLOAT_DTYPES,
    *INTEGRAL_DTYPES,
)

ALL_NUMERIC_DTYPES: Final[Sequence[torch.dtype]] = (
    *UNQUANTIZED_NUMERIC_DTYPES,
    *quantize_utils.QUANTIZED_NUMERIC_DTYPES,
)

# Maps a dtype to its tier (smaller is more important).
TIER_OF_DTYPE: Final[Mapping[torch.dtype, int]] = {
    # go/keep-sorted start numeric=yes by_regex=:\s*(\d+)
    torch.bfloat16: 1,  # Primary FP type for modern models.
    torch.bool: 1,  # Masks and logic ops.
    torch.float32: 1,  # Default FP type for PyTorch.
    torch.float4_e2m1fn_x2: 1,  # FP4 supported by TPU v8.
    torch.float8_e4m3fn: 1,  # FP8 for weights/activations (higher precision).
    torch.float8_e5m2: 1,  # FP8 for gradients (wider range).
    torch.int32: 1,  # Tensor dimensions, lighter indexing.
    torch.int4: 1,  # Quantized inference for vLLM.
    torch.int64: 1,  # Standard type for indexing.
    torch.float16: 2,  # Half precision for modern models.
    torch.int8: 2,  # Quantized inference.
    torch.uint8: 2,  # Raw image.
    torch.complex64: 3,  # Signal processing.
    torch.float64: 3,  # Scientific computing.
    torch.int16: 3,
    torch.uint16: 3,
    torch.uint32: 3,
    torch.complex128: 4,  # High-precision complex, no native TPU support.
    torch.uint4: 4,
    torch.uint64: 4,  # No native TPU support.
    # go/keep-sorted end
}


@functools.lru_cache(maxsize=None)
def _dtypes_to_test() -> Sequence[torch.dtype]:
  """Returns the dtypes to test."""

  if "all" in _DTYPES.value:
    return ALL_NUMERIC_DTYPES
  else:
    return tuple(_parse_dtype(dtype_str) for dtype_str in _DTYPES.value)


@functools.lru_cache(maxsize=1)
def all_xla_supported_dtypes() -> Sequence[torch.dtype]:
  """Returns the dtypes supported by XLA."""
  dtypes = common_dtype.get_all_dtypes()
  dtypes = filter(lambda x: x != torch.complex128, dtypes)
  return list(dtypes)


# The number of slowest ops to print for each dtype.
_TOP_N_SLOWEST_OPS: Final[int] = 100

# The glob pattern matching data files in the performance result output
# directories.
_PERF_DATA_GLOB_PATTERN: Final[str] = "perf_data*.txt"

# Match a line in the perf result file. A perf result line looks like this:
#   test shard 3, sample 5: add float32 [add_float32_subtest0] 0.05ms
_PERF_RESULT_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^test shard (\d+), sample (\d+): (\S+) (\S+) \[(.*)\] (.*)ms"
)


class PerfResult:
  """Holds the performance result of running an op with a given input dtype."""

  # How many PerfResult instances have been created so far.
  num_instances: int = 0

  def __init__(
      self,
      *,
      op_name: str,
      dtype: torch.dtype,
      subtest_name: str,
      duration_sec: float,
  ) -> None:
    # Sponge uses 1-based shard indices.
    self.test_shard = _get_test_shard() + 1
    self.sequence_number = PerfResult.num_instances
    PerfResult.num_instances += 1
    self.op_name = op_name
    self.dtype = dtype
    self.subtest_name = subtest_name
    self.duration_sec = duration_sec

  def __str__(self) -> str:
    return (
        f"test shard {self.test_shard}, sample {self.sequence_number}:"
        f" {self.op_name} {_dtype_str(self.dtype)} [{self.subtest_name}]"
        f" {1000*self.duration_sec:.02f}ms"
    )

  def __repr__(self) -> str:
    return self.__str__()

  def __eq__(self, other: "PerfResult") -> bool:  # pyrefly: ignore[bad-override]
    return (
        self.op_name == other.op_name
        and self.dtype == other.dtype
        and self.subtest_name == other.subtest_name
        and self.duration_sec == other.duration_sec
    )

  def __hash__(self) -> int:
    return hash(
        (self.op_name, self.dtype, self.subtest_name, self.duration_sec)
    )

  def __lt__(self, other: "PerfResult") -> bool:
    return (
        self.duration_sec,
        self.op_name,
        _dtype_str(self.dtype),
        self.subtest_name,
    ) < (
        other.duration_sec,
        other.op_name,
        _dtype_str(other.dtype),
        other.subtest_name,
    )

  @classmethod
  def parse(cls, s: str) -> "PerfResult":
    """Parses the string representation of the perf result."""
    # A perf result line looks like this:
    #   test shard 3, sample 5: add float32 [add_float32_subtest0] 0.05ms
    m = _PERF_RESULT_LINE_RE.match(s)
    assert m, f"Invalid perf result line: {s}"
    test_shard, sample_number, op_name, dtype, subtest_name, duration_ms_str = (
        m.groups()
    )
    duration_ms = float(duration_ms_str)
    result = cls(
        op_name=op_name,
        dtype=_parse_dtype(dtype),
        subtest_name=subtest_name,
        duration_sec=duration_ms / 1000,
    )
    result.test_shard = int(test_shard)
    result.sequence_number = int(sample_number)
    return result


_PERF_RESULTS: MutableSequence[PerfResult] = []


_DTYPE_NAME_TO_DTYPE: Final[Mapping[str, torch.dtype]] = {
    _dtype_str(dtype): dtype for dtype in ALL_NUMERIC_DTYPES
}


def _parse_dtype(dtype_str: str) -> torch.dtype:
  """Parses the dtype from the string representation of the dtype."""
  # For example, "float32" becomes torch.float32.
  return _DTYPE_NAME_TO_DTYPE[dtype_str]


def _add_perf_result(result: PerfResult) -> None:
  """Adds the perf result to _PERF_RESULTS."""
  _PERF_RESULTS.append(result)
  print(f"Perf result: {result}", flush=True)


def _perf_dir() -> epath.Path:
  """Returns the perf directory."""
  assert (
      _PERF_DIR.value
  ), "--perf_dir must be set to a non-empty string when --test_mode=perf."
  return epath.Path(_PERF_DIR.value)


def _get_test_shard() -> int:
  """Returns the 0-based test shard index."""
  return int(os.environ.get("TEST_SHARD_INDEX", "0"))


def _save_perf_data() -> None:
  """Saves the perf data to the perf directory."""

  # Create the output directory if it doesn't exist.
  output_dir = _perf_dir()
  output_dir.mkdir(parents=True, exist_ok=True)

  # Save the perf data to a file in the output directory, in the order
  # the samples were tested.
  test_shard = _get_test_shard()
  output_file = output_dir / f"perf_data{test_shard:02d}.txt"
  output_file.write_text("\n".join(str(r) for r in _PERF_RESULTS))
  print(f"Saved perf data to {output_file}", flush=True)


def _print_perf_debug_guide() -> None:
  """Prints the performance debug guide."""

  print(
      """
Debug guide:

The performance report file lists the sequence numbers of the slowest samples.
Use the sequence numbers to find the corresponding logs in the test log.

For example,

  test shard 3, sample 21: take.out float32 1009.79ms

means that in test shard 3 (1-based, as shown in Sponge), test sample #21
caused the take.out op with a float32 input to take 1009.79ms.

To debug this,

1. open the test log on Sponge.
2. select "3" in the Shard dropdown menu.
3. click on "Target Log".
4. find "Perf result: test shard 3, sample 21:" in the log.
5. find the nearest "Compiling Module for key" log line *before* this line.
   The SHLO corresponding to this sample is logged there.
""",
      flush=True,
  )


def _analyze_perf_data() -> None:
  """Analyzes the perf data stored in perf directory."""

  output_dir = _perf_dir()
  test_shard = _get_test_shard()
  # Only analyze the perf data when we are test shard 0, as there's no point
  # repeating the same analysis N times.
  if test_shard != 0:
    return

  # Parse the perf data files.
  perf_data_files = output_dir.glob(_PERF_DATA_GLOB_PATTERN)
  perf_results = []
  for file in perf_data_files:
    print(f"Parsing perf data file {file}", flush=True)
    perf_results.extend(
        [PerfResult.parse(line) for line in file.read_text().splitlines()]
    )

  # Parse the perf data from the baseline directory, if provided.
  base_perf_results = []
  if _BASE_PERF_DIR.value:
    base_perf_dir = epath.Path(_BASE_PERF_DIR.value)
    base_perf_data_files = base_perf_dir.glob(_PERF_DATA_GLOB_PATTERN)
    for file in base_perf_data_files:
      print(f"Parsing base perf data file {file}", flush=True)
      base_perf_results.extend(
          [PerfResult.parse(line) for line in file.read_text().splitlines()]
      )

  def _analyze(dtype: torch.dtype) -> None:
    """Analyzes the perf data for a given dtype."""

    results = [r for r in perf_results if r.dtype == dtype]
    base_results = [r for r in base_perf_results if r.dtype == dtype]
    if not results:
      # The dtype is not tested.
      return

    def print_stats(results: list[PerfResult], label: str):
      """Prints perf stats for a list of results."""
      if not results:
        return
      num = len(results)
      print(80 * "-", flush=True)
      print(
          f"{label} perf stats from {num} samples for dtype {dtype}.",
          flush=True,
      )
      durations = [r.duration_sec for r in results]
      sorted_durations_ms = sorted(1000 * d for d in durations)
      stddev = statistics.stdev(sorted_durations_ms) if num > 1 else 0.0
      print(
          f"""\
Min: {min(results)}
Max: {max(results)}
Stddev: {stddev:.2f}ms
Mean: {sum(sorted_durations_ms)/num:.2f}ms
Median: {sorted_durations_ms[num//2]:.2f}ms
P90: {sorted_durations_ms[90*num//100]:.2f}ms
P95: {sorted_durations_ms[95*num//100]:.2f}ms
P99: {sorted_durations_ms[99*num//100]:.2f}ms""",
          flush=True,
      )

    if base_results:
      print_stats(base_results, "Baseline")
    print_stats(results, "Current")

    # For each op, find the slowest sample. Print the slowest ops.
    op_to_results = collections.defaultdict(list)
    for r in results:
      op_to_results[r.op_name].append(r)
    op_to_slowest_result = {
        op: max(results, key=lambda r: r.duration_sec)
        for op, results in op_to_results.items()
    }
    sorted_result_desc = sorted(op_to_slowest_result.values(), reverse=True)
    print(
        f"Top {_TOP_N_SLOWEST_OPS} slowest ops for the current run:", flush=True
    )
    for r in sorted_result_desc[:_TOP_N_SLOWEST_OPS]:
      print(r, flush=True)
    print(flush=True)

    durations_by_subtest = collections.defaultdict(list)
    for r in results:
      durations_by_subtest[r.subtest_name].append(r.duration_sec * 1000)

    base_durations_by_subtest = collections.defaultdict(list)
    if base_results:
      for r in base_results:
        base_durations_by_subtest[r.subtest_name].append(r.duration_sec * 1000)

    factors = []
    for subtest_name in sorted(durations_by_subtest.keys()):
      durations_ms = durations_by_subtest[subtest_name]
      n = len(durations_ms)
      mean = statistics.mean(durations_ms)
      median = statistics.median(durations_ms)
      if n > 1:
        stdev = statistics.stdev(durations_ms)
        print(
            f" Subtest: {subtest_name}, Count: {n}\n  Mean: {mean:.2f}ms, Min:"
            f" {min(durations_ms):.2f}ms, Max: {max(durations_ms):.2f}ms,"
            f" Median: {median:.2f}ms, Stddev: {stdev:.2f}ms",
            flush=True,
        )
      if (
          base_durations_by_subtest
          and subtest_name in base_durations_by_subtest
      ):
        base_mean = statistics.mean(base_durations_by_subtest[subtest_name])
        if base_mean > 0:
          factors.append(mean / base_mean)
    if factors:
      geomean_factor = statistics.geometric_mean(factors)
      print("")
      print(
          f"Current mean duration is {geomean_factor:.4f}x of the baseline"
          " (geomean over all tests).",
          flush=True,
      )

  for dtype in ALL_NUMERIC_DTYPES:
    _analyze(dtype)

  _print_perf_debug_guide()


def _get_op(op_name: str, *, variant_test_name: str | None = None) -> OpInfo:
  """Returns the op with the given name."""
  try:
    return next(
        op
        for op in _KNOWN_OPS
        if op.name == op_name
        and (not variant_test_name or op.variant_test_name == variant_test_name)
    )
  except StopIteration as e:
    raise ValueError(f"Unknown op: {op_name}") from e


def _supports_out(op: OpInfo) -> bool:
  """Returns whether the op supports out-variant testing.

  Foreach operations in PyTorch (`ForeachFuncInfo`) do not accept an `out`
  keyword argument in Python or in TorchTPU C++ kernels. In `ForeachFuncInfo`,
  PyTorch overloads `supports_out` to denote functional (out-of-place) execution
  rather than out-argument support (see
  `third_party/py/torch/testing/_internal/opinfo/core.py:ForeachFuncInfo.__post_init__`).
  """
  if isinstance(op, core.ForeachFuncInfo):
    return False
  return op.supports_out


def _op_name_for_logging(op: OpInfo, variant: OpVariant) -> str:
  """Returns the name of the op with the given variant.

  Args:
    op: The op to get the name of.
    variant: The variant of the op to get the name of.

  Used only for logging. The test logic shouldn't depend on this.
  """
  if variant == OpVariant.BASE:
    return op.name
  if variant == OpVariant.INPLACE:
    return f"{op.name}_"
  return f"{op.name}.out"


def _symbolic_shapes_key(op: OpInfo) -> str:
  """Returns the key of `op` in the --check_symbolic_shapes known failures."""
  if op.variant_test_name:
    return f"{op.name}.{op.variant_test_name}"
  return op.name


def _tensor_tree_map(
    leaf_func: Callable[[Any], Any], x: _pytree.PyTree
) -> _pytree.PyTree:
  """A version of tree_map that transforms torch.Tensors and devices.

  Args:
    leaf_func: The function to transform the leaves. It needs to handle the
      following input types: torch.Tensor, torch.device, str, OpInput, OpOutput.
    x: The PyTree to transform.

  Returns:
    A copy of the PyTree with the leaves transformed by leaf_func.
  """

  def is_leaf(obj: Any) -> bool:
    """Returns True if obj is a leaf node (i.e. one we want to transform)."""
    return (
        isinstance(obj, torch.Tensor)
        or isinstance(obj, torch.device)
        or isinstance(obj, str)
        or isinstance(obj, OpInput)
        or isinstance(obj, OpOutput)
    )

  return _pytree.tree_map(leaf_func, x, is_leaf=is_leaf)


def _convert_sample_dtype(
    sample: SampleInput, from_dtype: torch.dtype, to_dtype: torch.dtype
) -> SampleInput:
  """Converts all tensors in sample with dtype == from_dtype to to_dtype."""
  convert = quantize_utils.SAMPLE_DTYPE_CONVERTERS.get(
      to_dtype, lambda t: t.to(to_dtype)
  )

  def transform_leaf(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor) and obj.dtype == from_dtype:
      return convert(obj)
    if isinstance(obj, torch.dtype) and obj == from_dtype:
      # Also translate dtype-valued args, e.g. `dtype=` kwargs of factory ops,
      # so the sample really exercises the desired dtype.
      return to_dtype
    return obj

  sample.input = _tensor_tree_map(transform_leaf, sample.input)
  sample.args = _tensor_tree_map(transform_leaf, sample.args)
  sample.kwargs = _tensor_tree_map(transform_leaf, sample.kwargs)
  return sample


def _sample_inputs(
    op: OpInfo,
    device: torch.device,
    dtype: torch.dtype,
) -> list[SampleInput]:
  """Generates sample inputs for an op on the given device.

  Args:
    op: The op to generate sample inputs for.
    device: The device to generate sample inputs for.
    dtype: The desired dtype of the sample inputs.

  Returns:
    A list of sample inputs.
  """
  sample_dtype = quantize_utils.DESIRED_DTYPE_TO_SAMPLE_DTYPE.get(dtype, dtype)
  try:
    samples = list(op.sample_inputs(device, sample_dtype, requires_grad=False))
    if sample_dtype != dtype:
      samples = [_convert_sample_dtype(s, sample_dtype, dtype) for s in samples]
    return samples
  except Exception as e:  # pylint: disable=broad-except
    if device.type == "cpu":
      raise

    # CUDA kernels have gaps in supporting certain dtypes, especially the
    # low-precision quantized dtypes. To avoid (op, dtype) being silently
    # dropped, fall back to CPU and transfer back to the requested device.
    try:
      cpu_samples = _sample_inputs(op, torch.device("cpu"), dtype)
      print(
          f"WARNING: Generated samples for {op.name}() with dtype {dtype}"
          f" on CPU because generation failed on {device}: {e}",
          flush=True,
      )
      return [_sample_to_device(s, device) for s in cpu_samples]
    except Exception as ce:  # pylint: disable=broad-except
      print(
          f"WARNING: Skipping golden generation for {op.name}() with dtype"
          f" {dtype} because sample generation failed on both"
          f" {device} and CPU fallback with\n"
          f"  Exception on {device}: {e}\n"
          f"  Exception on CPU: {ce}",
          flush=True,
      )
      return []


def to(
    x: _pytree.PyTree,
    device: torch.device | str,
    *,
    convert_tensors: bool = True,
) -> _pytree.PyTree:
  """Converts all torch.Tensor and devices in x to the given device.

  Args:
    x: The PyTree to convert.
    device: The device to convert to. Can be a string (e.g. "cpu") or a
      torch.device object.
    convert_tensors: If True, convert torch.Tensor to the given device.
      Otherwise, only convert torch.device to the given device.

  Returns:
    A copy of the PyTree with all torch.Tensor and devices converted to
    the given device. Note that this does NOT guarantee a deep copy of
    the PyTree. For example, if a tensor in the PyTree is already on the
    given device, it will not be copied.
  """

  if isinstance(device, str):
    device_str = device
    device = torch.device(device)
  else:
    device_str = str(device)

  def transform_leaf(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor) and convert_tensors:
      if (
          obj.dtype in quantize_utils.SUB_BYTE_DTYPES
          and obj.device.type != "tpu"
          and device.type != "tpu"
      ):
        # Copies between CPU and GPU are not supported for sub-byte dtypes, thus
        # move the raw bytes instead. TPU transfers support them natively.
        return obj.view(torch.uint8).contiguous().to(device).view(obj.dtype)
      return obj.to(device)
    if isinstance(obj, torch.device):
      return device
    if isinstance(obj, str):
      # Translate device names to the desired device. For example, "cpu:0"
      # becomes "tpu:0" if the desired device is TPU.
      segs = obj.split(":", maxsplit=1)
      if segs[0] in ("cuda", "cpu", "tpu", "gpu"):
        segs[0] = device_str
      return ":".join(segs)
    # By default, _pytree.tree_map() does NOT traverse into classes,
    # so we need to handle these types explicitly.
    if isinstance(obj, OpInput):
      obj.input_value = to(
          obj.input_value, device, convert_tensors=convert_tensors
      )
      obj.args = to(obj.args, device, convert_tensors=convert_tensors)
      obj.kwargs = to(obj.kwargs, device, convert_tensors=convert_tensors)
      return obj
    if isinstance(obj, OpOutput):
      obj.output_value = to(
          obj.output_value, device, convert_tensors=convert_tensors
      )
      return obj
    return obj

  return _tensor_tree_map(transform_leaf, x)


def _sample_to_device(sample: SampleInput, device: torch.device) -> SampleInput:
  """Moves all tensors and devices in the sample to the given device."""
  sample.input = to(sample.input, device)
  sample.args = to(sample.args, device)
  sample.kwargs = to(sample.kwargs, device)
  return sample


def _prepare_zero_element_out_tensors(
    x: _pytree.PyTree,
    out_dtype_override: torch.dtype | None = None,
) -> _pytree.PyTree:
  """Prepares zero-element out tensors (shape (0,)) from the base computation result.

  We use zero-element tensors to force and verify out-variant shape resizing
  compliance. When testing out-variant dtype casting, out_dtype_override is
  provided to override the out tensor's dtype.
  """
  override_applied = False

  def transform_leaf(obj: Any) -> Any:
    nonlocal override_applied
    if isinstance(obj, torch.Tensor):
      dtype = obj.dtype
      if out_dtype_override is not None and not override_applied:
        # For multi-output operators (e.g., max() returning (values, indices)),
        # only override the primary result tensor to prevent corrupting auxiliary
        # index tensors (which must remain integer dtypes).
        # Note: This relies on `_tensor_tree_map` traversing the PyTree in a
        # deterministic, depth-first order so that the first `torch.Tensor`
        # encountered is always the primary result tensor.
        dtype = out_dtype_override
        override_applied = True
      if obj.layout == torch.sparse_coo:
        return torch.ops.aten._sparse_coo_tensor_with_dims(
            obj.sparse_dim(),
            obj.dense_dim(),
            (0,) * obj.ndim,
            dtype=dtype,
            layout=torch.sparse_coo,
            device=obj.device,
        )
      return torch.empty((0,), dtype=dtype, device=obj.device)
    return obj

  return _tensor_tree_map(transform_leaf, x)


def _gen_gpu_golden_mode() -> bool:
  """Returns true if the test is running in gen_gpu_golden mode."""
  return _TEST_MODE.value == TestMode.GEN_GPU_GOLDEN


def golden_target_gpu_arch() -> str:
  """Returns the architecture of the GPU that golden generation runs on.

  Used in gen_gpu_golden mode to decide which tests to run and which golden
  files to write (e.g. `ops_test_gpu_golden_hopper0.gz` on Hopper).

  Returns:
    `GPU_SM90` on Hopper or `GPU_SM100` on Blackwell.

  Raises:
    RuntimeError: If CUDA is unavailable or the GPU is neither Hopper nor
      Blackwell.
  """
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available for GPU golden generation.")
  major = torch.cuda.get_device_properties(0).major
  if major == 9:
    return GPU_SM90
  elif major == 10:
    return GPU_SM100
  raise RuntimeError(
      f"Unsupported GPU compute capability {major} for golden generation: "
      "requires Hopper (SM90, major=9) or Blackwell (SM100, major=10)."
  )


def record_gpu_goldens_on(arch: str):
  """Decorator that records a test method's GPU goldens on `arch`.

  Untagged test methods record their goldens on the default GPU (Blackwell).
  Only affects gen_gpu_golden mode: `OpInfoTestBase.setUp` runs the test on the
  generators for `arch` and skips it on every other architecture. Other modes
  run the test as usual.

  Example:

    class TestOps(op_testing.OpInfoTestBase):

      @op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)
      def test_asinh(self):
        ...

  Args:
    arch: The architecture to record goldens on, e.g. `GPU_SM90`.

  Returns:
    A decorator that tags the test method with `arch` and returns it unchanged.
  """

  def decorator(func):
    setattr(func, _GPU_GOLDEN_ARCH_ATTR, arch)
    return func

  return decorator


@dataclasses.dataclass(frozen=True)
class UnboundedDynamicSpec:
  """The subtests of an op that must run on a registered unbounded dynamic kernel.

  Attributes:
    dtypes: The dtypes the kernel serves.
    is_compatible: Returns whether the kernel serves the given op input tensor
      (e.g. based on its rank and extents). If None, any input of a listed dtype
      is served.
    variants: The op variants the kernel serves.
  """

  dtypes: frozenset[torch.dtype]
  is_compatible: Callable[[torch.Tensor], bool] | None
  variants: frozenset[OpVariant]

  def matches(
      self, variant: OpVariant, dtype: torch.dtype, op_input: "OpInput"
  ) -> bool:
    """Returns whether the subtest must run on the unbounded dynamic kernel."""
    if variant not in self.variants or dtype not in self.dtypes:
      return False
    if not isinstance(op_input.input_value, torch.Tensor):
      return False
    return self.is_compatible is None or bool(
        self.is_compatible(op_input.input_value)
    )


def test_unbounded_dynamic(
    *,
    dtypes: Iterable[torch.dtype],
    is_compatible: Callable[[torch.Tensor], bool] | None = None,
    variants: Iterable[OpVariant] = (OpVariant.BASE,),
):
  """Decorator that checks the op's matching subtests use an unbounded kernel.

  Only takes effect with --unbounded_dynamic_tests_only, which runs just the
  decorated tests (see the ops_test_unbounded_dynamic target, which also sets
  TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS=1). There, every subtest
  whose variant, dtype and input match must be served by a kernel from the
  `UnboundedDynamicKernelRegistry`, checked with the compilation cache's
  `num_unbounded_dynamic_registry_hits` stat.

  Example:

    class TestOps(op_testing.OpInfoTestBase):

      @op_testing.test_unbounded_dynamic(
          dtypes=[torch.float32], is_compatible=lambda x: x.dim() == 2
      )
      def test_nonzero(self):
        ...

  Args:
    dtypes: The dtypes the kernel serves.
    is_compatible: Returns whether the kernel serves the given op input tensor.
      If None, any input of a listed dtype is served.
    variants: The op variants the kernel serves.

  Returns:
    A decorator that tags the test method and returns it unchanged.
  """
  spec = UnboundedDynamicSpec(
      dtypes=frozenset(dtypes),
      is_compatible=is_compatible,
      variants=frozenset(variants),
  )

  def decorator(func):
    setattr(func, _UNBOUNDED_DYNAMIC_ATTR, spec)
    return func

  return decorator


def _dump_dtype_exclusions_mode() -> bool:
  """Returns true if the test is running in dump_dtype_exclusions mode."""
  return _TEST_MODE.value == TestMode.DUMP_DTYPE_EXCLUSIONS


def _torch_tpu_vs_gpu_mode() -> bool:
  """Returns true if the test is running in torch_tpu_vs_gpu mode."""
  return _TEST_MODE.value == TestMode.TORCH_TPU_VS_GPU


def _perf_mode() -> bool:
  """Returns true if the test is running in perf mode."""
  return _TEST_MODE.value == TestMode.PERF


def is_compiled_mode() -> bool:
  """Returns true if --use_compiled is set."""
  return _USE_COMPILED.value


@functools.lru_cache(maxsize=None)
def is_tpu_vs_gpu_compiled() -> bool:
  """Returns true if running TPU vs GPU test in compiled mode."""
  return _torch_tpu_vs_gpu_mode() and is_compiled_mode()


def _is_enum(obj: Any) -> bool:
  """Returns True if obj is a Python enum or a pybind11 enum object."""
  cls = obj if isinstance(obj, type) else type(obj)
  return issubclass(cls, enum.Enum) or hasattr(cls, "__entries")


def _to_plistlib_compatible(ptree: _pytree.PyTree) -> _pytree.PyTree:
  """Converts values in the PyTree to values that plistlib can handle.

  In particular,

    - A torch.Tensor is serialized to a byte string using st.save().
    - A tuple is converted to a {"$$tuple": ...} dict. This is needed
      for preserving the tuple/list distinction because plistlib always
      serializes tuples as lists.
    - A torch.dtype is converted to a {"$$dtype": dtype_name} dict.
    - A torch.device is converted to a {"$$device": device_name} dict.
    - A torch.memory_format is converted to a {"$$memory_format": format_name}
      dict.
    - An enum (Python or pybind11) is converted to a {"$$enum": ...} dict.
    - A complex number is converted to a {"$$real": ..., "$$imag": ...} dict.
    - An Exception is serialized to a {"$$exception": {"type": ...,
      "message": ...}} dict.

  The "$$" prefix is used to avoid name conflicts with the keys in the
  original PyTree.

  Args:
    ptree: The PyTree to convert.

  Returns:
    A copy of the PyTree with the values converted to values that plistlib can
    handle.
  """

  def leaf_func(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
      # Use st.save() to serialize the tensor to a byte string. This format is
      # lossless and compact. Since st.save() takes a dict from str to tensor,
      # we need to provide a dummy str key.
      #
      # st.save() doesn't handle sub-byte dtypes or float8_e8m0fnu, so serialize
      # the raw bytes as uint8 and record the original dtype in the key after a
      # "b:" prefix.
      if x.dtype in quantize_utils.RAW_BYTE_DTYPES:
        return st.save({
            f"b:{_dtype_str(x.dtype)}":
            # Densify the uint8 view rather than the tensor itself because
            # contiguous() needs a copy kernel the sub-byte dtypes don't have.
            x.view(torch.uint8).contiguous()
        })
      if x.layout == torch.sparse_coo:
        return {
            "$$sparse_coo": {
                "sparse_dim": x.sparse_dim(),
                "dense_dim": x.dense_dim(),
                "size": list(x.shape),
                "is_coalesced": x.is_coalesced(),
                "indices": _to_plistlib_compatible(x._indices().contiguous()),
                "values": _to_plistlib_compatible(x._values().contiguous()),
            }
        }
      # st.save() doesn't handle non-contiguous tensors or non-strided tensors.
      if x.layout != torch.strided:
        x = x.to_dense()
      x = x.contiguous()
      # st.save() doesn't handle complex tensors, so encode them as real + imag.
      # The "c" key indicates that the original tensor is complex-typed.
      if x.dtype.is_complex:
        return st.save({"c": torch.view_as_real(x)})
      # The "" key indicates that we are serializing the tensor as is.
      return st.save({"": x})
    if isinstance(x, tuple):
      return {"$$tuple": [_to_plistlib_compatible(e) for e in x]}
    if isinstance(x, torch.dtype):
      # The value is the name of the dtype, e.g. "float32".
      return {"$$dtype": str(x).split(".")[-1]}
    if isinstance(x, torch.device):
      # The value is the name of the device, e.g. "cpu" or "tpu".
      return {"$$device": str(x)}
    if isinstance(x, complex):  # plistlib doesn't handle complex numbers.
      # Encode complex numbers as real + imag.
      return {"$$real": x.real, "$$imag": x.imag}
    if isinstance(x, torch.memory_format):
      # The value is the name of the memory format, e.g. "contiguous_format".
      return {"$$memory_format": str(x).split(".")[-1]}
    if _is_enum(x):
      enum_class = type(x)
      return {
          "$$enum": {
              "module": enum_class.__module__,
              "class": enum_class.__qualname__,
              "name": getattr(x, "name", str(x)),
              "value": getattr(x, "value", None),
          }
      }
    if isinstance(x, Exception):
      # Preserve the exception type and message separately so the harness can
      # enforce exception parity between the golden device and TPU.
      return {"$$exception": {"type": type(x).__name__, "message": str(x)}}
    return x

  def is_leaf(obj: Any) -> bool:
    return isinstance(
        obj,
        (
            tuple,
            torch.Tensor,
            torch.dtype,
            torch.device,
            torch.memory_format,
            complex,
            Exception,
        ),
    ) or _is_enum(obj)

  return _pytree.tree_map(leaf_func, ptree, is_leaf=is_leaf)


def _resolve_enum_class(mod_name: str, class_name: str) -> Any:
  """Resolves an enum class by module name and class name."""
  if mod_name in sys.modules:
    mod = sys.modules[mod_name]
    try:
      cls = mod
      for part in class_name.split("."):
        cls = getattr(cls, part)
      return cls
    except AttributeError:
      pass

  if mod_name:
    try:
      mod = importlib.import_module(mod_name)
      cls = mod
      for part in class_name.split("."):
        cls = getattr(cls, part)
      return cls
    except (ImportError, AttributeError):
      pass

  # pylint: disable-next=protected-access
  for target_mod in (torch, torch._C, torch.nn.functional):
    cls = target_mod
    found = True
    for part in class_name.split("."):
      if hasattr(cls, part):
        cls = getattr(cls, part)
      else:
        found = False
        break
    if found and cls is not target_mod:
      return cls

  return None


def _deserialize_exception(payload: Any) -> Exception:
  """Deserializes an Exception object from a serialized "$$exception" payload."""
  # TODO(b/542331196): remove this check once exceptions are consistently in dict
  # format after golden updates.
  if not isinstance(payload, dict):
    return Exception(payload or "")

  type_name = str(payload.get("type", "Exception"))
  message = payload.get("message", "")

  builtin_exception = getattr(builtins, type_name, None)
  if isinstance(builtin_exception, type) and issubclass(
      builtin_exception, Exception
  ):
    try:
      return builtin_exception(message)
    except Exception:  # pylint: disable=broad-except
      # Some builtins cannot be built from a message alone, so fall back to
      # the synthetic exception.
      pass

  return type(type_name, (Exception,), {})(message)


def _from_plistlib_compatible(ptree: _pytree.PyTree) -> _pytree.PyTree:
  """Converts a plistlib-compatible PyTree to the original PyTree.

  In particular,

    - A byte string is deserialized to a torch.Tensor using st.load().
    - A {"$$tuple": ...} dict is deserialized to a tuple.
    - A {"$$dtype": ...} dict is deserialized to a torch.dtype.
    - A {"$$device": ...} dict is deserialized to a torch.device.
    - A {"$$memory_format": ...} dict is deserialized to a torch.memory_format.
    - A {"$$enum": ...} dict is deserialized to an enum instance.
    - A {"$$real": ..., "$$imag": ...} dict is deserialized to a complex number.
    - A {"$$exception": {"type": ..., "message": ...}} dict is deserialized to
      an Exception of the recorded type carrying the message.

  Args:
    ptree: The PyTree to convert.

  Returns:
    A copy of the PyTree with the values converted to the original types.
  """

  def leaf_func(x: Any) -> Any:
    if isinstance(x, bytes):
      d = st.load(x)
      if "c" in d:
        # The tensor is complex-typed.
        return torch.view_as_complex(d["c"])
      key, tensor = next(iter(d.items()))
      if key.startswith("b:"):
        # The tensor has a sub-byte dtype, serialized as raw uint8 bytes.
        return tensor.view(getattr(torch, key.removeprefix("b:")))
      # The tensor is not complex-typed.
      return d[""]
    if isinstance(x, dict):
      if "$$sparse_coo" in x:
        info = x["$$sparse_coo"]
        indices = _from_plistlib_compatible(info["indices"])
        values = _from_plistlib_compatible(info["values"])
        return torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
            info["sparse_dim"],
            info["dense_dim"],
            info["size"],
            indices,
            values,
            dtype=values.dtype,
            layout=torch.sparse_coo,
            device=values.device,
            is_coalesced=info["is_coalesced"],
        )
      if "$$tuple" in x:
        return tuple(_from_plistlib_compatible(e) for e in x["$$tuple"])
      if "$$dtype" in x:
        dtype_name = x["$$dtype"]
        return getattr(torch, dtype_name)
      if "$$device" in x:
        return torch.device(x["$$device"])
      if "$$memory_format" in x:
        format_name = x["$$memory_format"]
        return getattr(torch, format_name)
      if "$$enum" in x:
        enum_info = x["$$enum"]
        mod_name = enum_info.get("module", "")
        class_name = enum_info.get("class", "")
        enum_name = enum_info.get("name", "")
        enum_val = enum_info.get("value", None)

        cls = _resolve_enum_class(mod_name, class_name)
        if cls is not None:
          if hasattr(cls, enum_name):
            return getattr(cls, enum_name)
          if hasattr(cls, "__getitem__"):
            try:
              return cls[enum_name]
            except (KeyError, TypeError, IndexError):
              pass
          if enum_val is not None:
            try:
              return cls(enum_val)
            except (ValueError, TypeError):
              pass

        if enum_val is not None:
          return enum_val
        return enum_name
      if "$$real" in x and "$$imag" in x:
        return complex(x["$$real"], x["$$imag"])
      if "$$exception" in x:
        return _deserialize_exception(x["$$exception"])
      return {k: _from_plistlib_compatible(v) for k, v in x.items()}
    return x

  def is_leaf(obj: Any) -> bool:
    return isinstance(
        obj,
        (
            bytes,
            dict,
        ),
    )

  return _pytree.tree_map(leaf_func, ptree, is_leaf=is_leaf)


def _safe_tensor_repr(obj: Any) -> Any:
  """Returns a representation of obj safe against sub-byte tensor repr crashes."""

  def transform_leaf(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
      if x.dtype in quantize_utils.SUB_BYTE_DTYPES:
        return f"tensor({x.view(torch.uint8).tolist()}, dtype={x.dtype})"
      return repr(x)
    return x

  if isinstance(obj, torch.Tensor):
    return transform_leaf(obj)
  return _tensor_tree_map(transform_leaf, obj)


class OpInput:
  """Holds the input to an op."""

  def __init__(self, sample: SampleInput) -> None:
    cpu = torch.device("cpu")
    self.name = sample.name
    self.input_value = to(copy.deepcopy(sample.input), cpu)
    self.args = to(copy.deepcopy(sample.args), cpu)
    self.kwargs = to(copy.deepcopy(sample.kwargs), cpu)

  def __str__(self) -> str:
    return self.__repr__()

  def __repr__(self) -> str:
    return (
        f"Name: {self.name}\nInput:"
        f" {_safe_tensor_repr(self.input_value)}\nArgs:"
        f" {_safe_tensor_repr(self.args)}\nKwargs:"
        f" {_safe_tensor_repr(self.kwargs)}"
    )

  def summary(self) -> str:
    input_summary = utils.InputMetadata(self.input_value)
    args_summary = utils.InputMetadata(self.args)
    return (
        f"Name: {self.name}\nInput: {input_summary}\nArgs:"
        f" {args_summary}\nKwargs: {self.kwargs}"
    )

  def to_plistlib_pytree(self) -> _pytree.PyTree:
    """Converts this object to a PyTree that plistlib can handle."""
    d = {"$$name": self.name, "$$input": self.input_value, **self.kwargs}
    for i, a in enumerate(self.args):
      d[f"{i}"] = a

    return _to_plistlib_compatible(d)

  @classmethod
  def from_plistlib_pytree(cls, ptree: _pytree.PyTree) -> "OpInput":
    """Converts a PyTree to an OpInput."""

    decoded_ptree = _from_plistlib_compatible(ptree)
    name = decoded_ptree.pop("$$name")
    input_value = decoded_ptree.pop("$$input")
    kwargs = {}
    index_to_arg = {}
    for k, v in decoded_ptree.items():
      # If the key contains an integer, it's an arg. Otherwise, it's a kwarg.
      try:
        int_key = int(k)
        index_to_arg[int_key] = v
      except ValueError:
        kwargs[k] = v
    args = tuple(index_to_arg[k] for k in sorted(index_to_arg.keys()))
    sample = SampleInput(name=name, input=input_value, args=args, kwargs=kwargs)
    return cls(sample)


class OpOutput:
  """Holds the output of an op."""

  def __init__(self, output_value: Any) -> None:
    """Initializates the object with the output of an op.

    Args:
      output_value: The output of the op, or an Exception if the op failed.
    """

    # Keep exceptions and None results (e.g. of ops returning None) as is.
    if output_value is None or isinstance(output_value, Exception):
      self.output_value = output_value
      return

    cpu = torch.device("cpu")
    self.output_value = to(copy.deepcopy(output_value), cpu)

  def __str__(self) -> str:
    return self.__repr__()

  def __repr__(self) -> str:
    return f"Output: {_safe_tensor_repr(self.output_value)}"

  def to_plistlib_pytree(self) -> _pytree.PyTree:
    """Converts this object to a PyTree that plistlib can handle."""
    return _to_plistlib_compatible(self.output_value)

  @classmethod
  def from_plistlib_pytree(cls, ptree: _pytree.PyTree) -> "OpOutput":
    """Converts a PyTree to an OpOutput."""
    decoded_ptree = _from_plistlib_compatible(ptree)
    return cls(decoded_ptree)


def _to_tuple(x: Any) -> tuple[Any, ...]:
  """Converts x to a tuple if it's a list, or a 1-tuple if it's not a tuple or list."""
  if isinstance(x, list):
    return tuple(x)
  return x if isinstance(x, tuple) else (x,)


def _to_torch_tpu_printable_input(golden_input: OpInput) -> OpInput:
  """Converts the golden input to a printable input on TorchTPU.

  Args:
    golden_input: The golden input to convert.

  Returns:
    A copy of the golden input with the devices replaced with TorchTPU devices,
    but without transferring the tensors to TorchTPU.

  In failure messages, we need to print the op input on TorchTPU. However,
  tensors on TorchTPU are not printable, so we cannot print the actual op
  input on TorchTPU directly. If we print golden_input instead, we will
  be misleading the readers as the devices in golden_input are
  CPU instead of TorchTPU. Therefore we replace the devices in golden_input
  with TorchTPU devices but don't transfer the tensors there to TorchTPU.

  For example, suppose golden_input contains
    (..., torch.ones(1, device="cpu"), (1,), {"device": "cpu"})
  the _run_op(..., golden_input, torch.device("tpu"), ...) call above
  will translate it to
    (..., torch.ones(1, device="tpu"), (1,), {"device": "tpu"})
  before invoking the op. Trying to print this input will result in
  a RuntimeError, as we cannot print torch.ones(1, device="tpu")
  directly.

  If we just print golden_input as is, the readers will be misled
  as they will see {"device": "cpu"} and think that the op ran on
  CPU. Therefore, we create torch_tpu_printable_input as
    (..., torch.ones(1, device="cpu"), (1,), {"device": "tpu"})
  and use it in failure messages. Note how this object contains
  tensors on CPU and the device string "tpu". When we print
  torch_tpu_printable_input, the output looks like:
    Input: torch.tensor([1])
    Args: (1,)
    Kwargs: {"device": "tpu"}
  which is not misleading.
  """
  return to(golden_input, torch.device("tpu"), convert_tensors=False)


def print_op_input(op_input: OpInput, *, data: bool = True) -> None:
  """Prints the OpInput for the given test case.

  Args:
    op_input: The OpInput to print.
    data: Print the tensor data if true, else print summary only.
  """

  if data:
    print(op_input, flush=True)
    return
  print(op_input.summary(), flush=True)


def print_reproducer(
    subtest_name: str, op: OpInfo, op_input: OpInput, variant: OpVariant
) -> None:
  """Prints a copy-pastable python function reproducer for the given op.

  Args:
    subtest_name: The name of the subtest.
    op: The op to print the reproducer for.
    op_input: The op input to trace the function with.
    variant: The op variant to determine which function to print.
  """

  op_func = op.inplace_variant if variant == OpVariant.INPLACE else op

  def wrapped_op_func(*args):
    return op_func(*args, **op_input.kwargs)

  args = [op_input.input_value] + list(op_input.args)

  # Not all test cases are expected to pass, skip reproducers for these.
  try:
    fx_reproducer = utils.format_model(wrapped_op_func, *args, pt=True)
  # Broad exception catch as this is a best-effort reproducer generation and
  # invalid inputs may raise varied exceptions.
  except Exception as e:  # pylint: disable=broad-except
    fx_reproducer = e
  print("Reproducer for:", subtest_name, flush=True)
  print(fx_reproducer, flush=True)


def _run_and_print_exception(func: Callable[[], Any]) -> None:
  """Runs func() and prints the exception if one is raised."""
  try:
    func()
  except unittest.case.SkipTest:
    # Explicitly catching and re-raising unittest.case.SkipTest) is necessary to
    # prevent skipped tests from cluttering the logs with misleading tracebacks.
    raise
  except Exception:
    # Print the exception with the traceback to stderr.
    # absltest only prints the error message at the end of the test.
    # This allows us to see the error message next to the other outputs
    # of the test, making debugging easier.
    traceback.print_exc(file=sys.stderr)
    raise


def _should_skip_dtype(
    dtype: torch.dtype, *, exclude_dtypes: Sequence[torch.dtype]
) -> bool:
  """Returns true if the dtype should be skipped for the test."""
  return dtype in exclude_dtypes or (
      # Gradients only work for complex and floating-point dtypes.
      _COMPUTE_GRAD.value
      and not dtype.is_complex
      and not dtype.is_floating_point
  )


def _dummy_grad(device: torch.device) -> torch.Tensor:
  """Returns a dummy gradient for use in tests."""
  return torch.tensor([], dtype=torch.float32, device=device)


def _compiled_supports_op(
    op: OpInfo, device_op_input: OpInput, variant: OpVariant
) -> bool:
  """Returns true if torch.compile supports the given op.

  All ops should be supported in compiled mode, as a user can write all of these
  patterns. This includes data dependent ops since dynamo should graph break and
  handle them. In that regard, consider this is a list of bugs to be fixed.

  Note some of these bugs are only in grad mode.

  Args:
    op: The op to check.
    device_op_input: The op input on the device.
    variant: The op variant to check.
  """

  ####
  # TODO: Fail to handle empty tensor views in compile mode
  #  This is likely because of our special handling of size zero tensors.
  def has_empty(t: torch.Tensor | Sequence[Any] | Mapping[Any, Any]) -> bool:
    """Returns true if `t` is (or contains) a size zero tensor."""
    if isinstance(t, (list, tuple)):
      return any(has_empty(i) for i in t)
    return isinstance(t, torch.Tensor) and t.numel() == 0

  def has_sparse(t: torch.Tensor | Sequence[Any] | Mapping[Any, Any]) -> bool:
    """Returns true if `t` is (or contains) a sparse tensor."""
    if isinstance(t, (list, tuple)):
      return any(has_sparse(i) for i in t)
    if isinstance(t, dict):
      return any(has_sparse(v) for v in t.values())
    return isinstance(t, torch.Tensor) and t.layout != torch.strided

  if (
      has_empty(device_op_input.input_value)
      or has_empty(device_op_input.args)
      or has_empty(device_op_input.kwargs)
      or has_sparse(device_op_input.input_value)
      or has_sparse(device_op_input.args)
      or has_sparse(device_op_input.kwargs)
  ):
    return False

  compiled_deny_list = [
      ####
      # TODO: Materialize was called on a placeholder tensor.
      #  Dynamo handing our compile backend a graph with data dependent ops.
      "index_put",
      "masked_scatter",
      "masked_select",
      "nonzero",
      ####
      # TODO: failed to validate and reorder inputs.
      # This is mostly the same as the numel == 0 case, but for embedded
      # constants in the graph.
      "arange",
      "empty",
      "split_with_sizes",
      "tril_indices",
      ####
      # TODO: NaN result
      "prod",
      "topk",
      ####
      # TODO: Precision mismatch, need to update tolerance.
      "nn.functional.gelu",
      ####
      # TODO: Unsupported CVT X64 expansion from c64[] to c128[]
      #  Likely only shows in compiled mode due to optimization level difference
      "addcdiv",
      ####
      # TODO: Timeout, looks like this causes a hang somewhere.
      "_foreach_erf",
      "_foreach_erf.out",
      ####
      # TODO: dynamo assert _PyEval_EvalFrameDefault is EMPTY
      #  Unclear if these are dynamo bugs or due to the special dynamo flags we
      #  set in our backend.
      "_foreach_add",
      "_foreach_add.out",
      "bincount",
      "exponential",  # exponential_complex64_sample0
      "empty_strided",
      "empty_strided.out",
      "multinomial",
      "nn.functional.scaled_dot_product_attention",
      "normal",
      "normal.out",
      "randint",
      "randint.out",
      "randn",
      "resize_",
      "uniform",
      # TODO(b/568080282): Fix list index out of range and 0D shape mismatch in
      # compiled .out variant with dim.
      "max.out",
      "min.out",
  ]
  if utils.get_torch_version() < (2, 15, 0):
    # Before PyTorch 2.15, Dynamo does not treat a torch.return_types `out=`
    # (e.g. cummax's (values, indices)) as a tuple, so it misses the graph
    # break for resizing the zero-element out tensors. Tracing then fails with
    # "list index out of range" (or leaves the wrong out shape for 0-D inputs).
    compiled_deny_list += [
        "cummax.out",
        "cummin.out",
    ]

  key = op.name
  if variant == OpVariant.OUT:
    key += ".out"
  return key not in compiled_deny_list


def _target_device(dtype: torch.dtype, device: torch.device) -> torch.device:
  """Returns the device to actually run an op with `dtype` inputs on.

  Args:
    dtype: The dtype of the op's inputs.
    device: The device requested for running the op.

  Returns:
    `device`, unless `device` is a CUDA device and `dtype` is a sub-byte dtype
    (see `quantize_utils.SUB_BYTE_DTYPES`), in which case the CPU device is
    returned instead. This only happens when generating GPU goldens, which is
    the only mode that runs ops on CUDA.
  """
  if dtype in quantize_utils.SUB_BYTE_DTYPES and device.type == "cuda":
    # Sub-byte dtypes do not have CUDA copy or cast kernels. Invoking an
    # operator or .to("cuda") on a sub-byte tensor causes a CUDA assertion
    # failure, putting CUDA context into an unrecoverable error state.
    return torch.device("cpu")
  return device


def _op_func(
    op: OpInfo, variant: OpVariant, target_device: torch.device
) -> Callable[..., Any]:
  """Returns the function to call for running `variant` of `op`."""
  # For operators that lack a native PyTorch CPU kernel (e.g. internal CUDA
  # ops like _thnn_fused_lstm_cell / _thnn_fused_gru_cell), invoke the custom
  # analytical reference function `.ref` when computing CPU golden results,
  # i.e. for sub-byte dtypes in the gen_gpu_golden mode.
  ref = getattr(op, "ref", None)
  if (
      target_device.type == "cpu"
      and ref is not None
      and (
          op.name.startswith("_thnn_fused_")
          or getattr(op, "use_ref_for_cpu_golden", False)
      )
  ):
    return ref
  return op.inplace_variant if variant == OpVariant.INPLACE else op


def _requires_grad_input(input_value: _pytree.PyTree) -> _pytree.PyTree:
  """Sets requires_grad=True for the non-empty float/complex input tensors."""

  def set_requires_grad(t: Any) -> Any:
    if (
        isinstance(t, torch.Tensor)
        and (t.is_floating_point() or t.is_complex())
        and t.numel() > 0
    ):
      return t.requires_grad_(True)
    return t

  return _tensor_tree_map(set_requires_grad, input_value)


def _can_compute_grad(variant: OpVariant, input_value: _pytree.PyTree) -> bool:
  """Returns whether the gradient w.r.t. `input_value` can be computed."""

  def is_differentiable(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.requires_grad and t.numel() > 0

  # Autograd doesn't work with inplace ops and out variants, as they
  # erase the input tensor.
  if (
      variant == OpVariant.INPLACE
      or variant == OpVariant.OUT
      or not any(
          _pytree.tree_leaves(_tensor_tree_map(is_differentiable, input_value))
      )
  ):
    return False

  # TODO: fix the bug where computing the gradient of a 0-sized input tensor
  # fails with an exception "Function SumBackward0 returned an invalid
  # gradient at index 0 - expected device tpu:0 but got cpu".
  if isinstance(input_value, torch.Tensor) and input_value.numel() == 0:
    return False
  return True


def _compute_grad(
    result: Any, input_value: _pytree.PyTree, device: torch.device
) -> Any:
  """Returns the gradient of the sum of `result` w.r.t. `input_value`."""
  losses = [
      t.sum()
      for t in _pytree.tree_leaves(result)
      if isinstance(t, torch.Tensor)
      and (t.is_floating_point() or t.is_complex())
      and t.numel() > 0
  ]
  if not losses:
    return _dummy_grad(device)

  torch.sum(torch.stack(losses)).backward()

  def get_grad(t: Any) -> torch.Tensor | None:
    if isinstance(t, torch.Tensor) and t.grad is not None:
      return t.grad.clone()
    return None

  return _tensor_tree_map(get_grad, input_value)


@dataclasses.dataclass(frozen=True)
class _SymbolicRunResult:
  """The result of running an op on FakeTensors with symbolic dims.

  Attributes:
    specialized: The names (e.g. `args[0].size(1)`) of the symbolic input dims
      that running the op specialized, i.e. pinned to a constant.
    error: The exception raised by the op, or None if it succeeded.
  """

  specialized: frozenset[str]
  error: Exception | None = None


def _has_differentiable_input(op_input: OpInput) -> bool:
  """Returns whether `op_input.input_value` has a float or complex tensor."""
  return any(
      isinstance(t, torch.Tensor)
      and (t.is_floating_point() or t.is_complex())
      and t.numel() > 0
      for t in _pytree.tree_leaves(op_input.input_value)
  )


def _pinned_symbols(shape_env: symbolic_shapes.ShapeEnv) -> set[sympy.Basic]:
  """Returns the symbols that `shape_env` pinned to a constant.

  A symbol is pinned if it was replaced with a constant (e.g. by a guard
  `s0 == 5` from calling `size()` on a FakeTensor), or if it occurs in a guard
  that equates an expression with a constant (e.g. `s0 * s1 == 12` from calling
  `numel()`). Either way, torch.compile recompiles when the dim changes.

  Args:
    shape_env: The ShapeEnv that the op ran with.
  """
  pinned: set[sympy.Basic] = {
      symbol
      for symbol in shape_env.var_to_val
      if shape_env.replace(symbol).is_number
  }
  for guard in shape_env.guards:
    expr = guard.expr
    if isinstance(expr, sympy.Eq) and (
        expr.lhs.is_number or expr.rhs.is_number
    ):
      pinned |= expr.free_symbols
  return pinned


def _run_with_symbolic_dims(
    op_func: Callable[..., Any],
    op_input: OpInput,
    device: torch.device,
    *,
    requires_grad: bool,
) -> _SymbolicRunResult:
  """Runs `op_func` on FakeTensors made from `op_input` with symbolic dims.

  Every dim of size >= 2 of every input tensor (including tensors nested in args
  and kwargs) gets a backed symbolic size whose hint is the real size. Dims of
  size 0 and 1 stay static, as torch.compile always specializes them. The input
  tensors must be strided.

  Args:
    op_func: The function to run.
    op_input: The input to run `op_func` on. Not mutated.
    device: The device of the FakeTensors.
    requires_grad: If True, the non-empty float and complex tensors in
      `op_input.input_value` require grad (like with --compute_grad), and the
      gradients of the outputs w.r.t. them are computed too, so that autograd
      kernels and backward formulas are covered.

  Returns:
    The input dims that the op specialized, or the exception it raised.
  """
  device_input = to(copy.deepcopy(op_input), device)
  shape_env = symbolic_shapes.ShapeEnv()
  fake_mode = fake_tensor.FakeTensorMode(
      shape_env=shape_env, allow_non_fake_inputs=True
  )
  # Maps each input symbol to the input dims that have it. With duck sizing,
  # dims that have the same size share a symbol.
  dim_names: dict[sympy.Basic, list[str]] = collections.defaultdict(list)
  grad_inputs = []

  def make_fake(
      prefix: str, needs_grad: bool, path: _pytree.KeyPath, t: Any
  ) -> Any:
    if not isinstance(t, torch.Tensor):
      return t
    differentiable = (t.is_floating_point() or t.is_complex()) and t.numel() > 0
    fake = fake_mode.from_tensor(
        t,
        symbolic_context=symbolic_shapes.StatelessSymbolicContext(
            dynamic_sizes=[
                symbolic_shapes.DimDynamic.DYNAMIC
                if size >= 2
                else symbolic_shapes.DimDynamic.STATIC
                for size in t.shape
            ]
        ),
    )
    for dim, size in enumerate(fake.shape):
      if isinstance(size, torch.SymInt):
        dim_names[size.node.expr].append(
            f"{prefix}{_pytree.keystr(path)}.size({dim})"
        )
    if needs_grad and differentiable:
      fake.requires_grad_(True)
      grad_inputs.append(fake)
    return fake

  def make_fakes(
      prefix: str, tree: _pytree.PyTree, needs_grad: bool = False
  ) -> _pytree.PyTree:
    return _pytree.tree_map_with_path(
        functools.partial(make_fake, prefix, needs_grad), tree
    )

  try:
    input_value = make_fakes(
        "input", device_input.input_value, needs_grad=requires_grad
    )
    args = make_fakes("args", device_input.args)
    kwargs = make_fakes("kwargs", device_input.kwargs)
    with fake_mode:
      result = op_func(input_value, *args, **kwargs)
      outputs = [
          t
          for t in _pytree.tree_leaves(result)
          if isinstance(t, torch.Tensor) and t.requires_grad
      ]
      if grad_inputs and outputs:
        torch.autograd.grad(
            outputs,
            grad_inputs,
            grad_outputs=[torch.ones_like(t) for t in outputs],
            allow_unused=True,
        )
  except Exception as e:  # pylint: disable=broad-except
    return _SymbolicRunResult(specialized=frozenset(), error=e)
  return _SymbolicRunResult(
      specialized=frozenset(
          name
          for symbol in _pinned_symbols(shape_env)
          for name in dim_names.get(symbol, ())
      )
  )


def _dtypes_to_test_for_op(
    op: OpInfo, recording: bool
) -> Sequence[torch.dtype]:
  """Returns the candidate input dtypes to test `op` with."""
  dtypes_to_test = list(UNQUANTIZED_NUMERIC_DTYPES)
  if recording:
    # Include quantized numeric dtypes during GPU golden data generation.
    dtypes_to_test.extend(quantize_utils.QUANTIZED_NUMERIC_DTYPES)
  else:
    # When reading GPU goldens, include quantized numeric dtypes if the op
    # explicitly declares support for them in its OpInfo.
    dtypes_to_test.extend([
        d
        for d in getattr(op, "dtypes", ())
        if d in quantize_utils.QUANTIZED_NUMERIC_DTYPES
    ])
  return list(dict.fromkeys(dtypes_to_test))


def _out_casting_tests(
    op: OpInfo,
    dtypes_to_test: Iterable[torch.dtype],
    *,
    exclude_dtypes: Sequence[torch.dtype],
    check_op_failures: bool,
    compute_grad: bool,
    use_compiled: bool,
    recording: bool,
) -> Mapping[torch.dtype, Sequence[tuple[torch.dtype, bool]]]:
  """Returns the out-variant dtype casting tests to run for each input dtype.

  Args:
    op: The op to test.
    dtypes_to_test: The candidate input dtypes.
    exclude_dtypes: The input dtypes excluded by the test.
    check_op_failures: Whether to check op failures for the allowed cast.
    compute_grad: Whether the test computes gradients.
    use_compiled: Whether the test compiles the op.
    recording: Whether the test records GPU goldens.

  Returns:
    A mapping from an input dtype to `(out_dtype_override, check_op_failures)`
    pairs, each describing an additional out-variant test to run.
  """
  # Out-variant casting tests exception parity and type promotion in
  # forward evaluations and is not applicable when computing gradients.
  if compute_grad:
    return {}
  # In golden generation mode, always capture golden data regardless of
  # whether the op is known to fail on TPU.
  if not recording and (
      op.name in _OUT_DTYPE_CAST_KNOWN_FAILURES
      or (use_compiled and op.name in _COMPILED_OUT_DTYPE_CAST_KNOWN_FAILURES)
  ):
    return {}

  # Select one valid promotional upcast pair (allowed_pair, e.g., float32
  # -> float64) and one invalid downcast pair (illegal_pair, e.g.,
  # float32 -> int32) from the dtypes to be tested. These will be used
  # below to test out-variant dtype casting and exception parity between
  # TorchTPU and GPU.
  cast_pairs = find_cast_pairs([
      d
      for d in dtypes_to_test
      if not _should_skip_dtype(d, exclude_dtypes=exclude_dtypes)
  ])
  tests = collections.defaultdict(list)
  # For the input dtype of the allowed cast pair, run an additional
  # out-variant test with out_dtype_override set to the target upcast dtype to
  # verify that TorchTPU correctly supports out-variant promotional upcasting.
  if cast_pairs.allowed_pair is not None:
    src, dst = cast_pairs.allowed_pair
    tests[src].append((dst, check_op_failures))
  # For the input dtype of the illegal cast pair, run an additional out-variant
  # test with out_dtype_override set to the target downcast dtype to verify
  # that TorchTPU raises the same RuntimeError (exception parity) as
  # PyTorch/GPU.
  if cast_pairs.illegal_pair is not None:
    src, dst = cast_pairs.illegal_pair
    tests[src].append((dst, True))
  return tests


# How `do_test_op` logs the start of testing each op variant.
_VARIANT_CALL_FORMAT: Final[Mapping[OpVariant, str]] = {
    OpVariant.BASE: "{}()",
    OpVariant.OUT: "{}(out=...)",
    OpVariant.INPLACE: "{}_()",
}


class TorchTpuTestBase(seed_test_utils.RepeatableTest, common_utils.TestCase):
  """Base class for TorchTPU tests."""

  dynamism_filter_fn: FilterFn
  dynamism_mark_dynamic_fn: MarkDynamicFn

  tpu_gpu_accuracy_overrides: AccuracyOverrides
  grad_gpu_accuracy_overrides: AccuracyOverrides

  def _skip_if_sparsecore_unsupported(self) -> None:
    # `torch.tpu.get_device_properties()` raises without TPU devices. Checking
    # the accelerator type first keeps e.g. the GPUs of a GPU run from counting
    # as TPU devices.
    accelerator = torch.accelerator.current_accelerator()
    if (
        accelerator is None
        or accelerator.type != "tpu"
        or not torch.accelerator.is_available()
        or torch.tpu.get_device_properties().sparse_cores_per_device == 0
    ):
      self.skipTest("SparseCore is not supported on this TPU device.")

  def setUp(self) -> None:
    super().setUp()
    # Make sure we call `do_test_op()` method only once.
    self._do_test_op_called = False
    self._subtests_run = 0
    tt_testing.reset_eager_state()
    # Show long diffs in assertEqual.
    self.maxDiff = None  # pylint: disable=invalid-name

    # Device used for computing the golden results, or None if the golden
    # results are read from the golden file.
    self.golden_device = (
        torch.device("cuda") if _gen_gpu_golden_mode() else None
    )
    self.symbolic_shapes_known_failures: Mapping[str, str] = {}
    # Whether a known --check_symbolic_shapes failure was observed.
    self._known_symbolic_shapes_failure_seen = False

  def set_accuracy_overrides(
      self,
      *,
      tpu_gpu_overrides: Mapping[
          str, Mapping[torch.dtype, Mapping[str, Tolerance]]
      ],
      grad_gpu_overrides: Mapping[
          str, Mapping[torch.dtype, Mapping[str, Tolerance]]
      ],
  ) -> None:
    """Sets the accuracy overrides for the test.

    Args:
      tpu_gpu_overrides: Accuracy overrides for TorchTPU vs GPU.
      grad_gpu_overrides: Accuracy overrides for gradients vs GPU.

    To be called by a subclass's setUp() method.
    """
    _validate_accuracy_overrides(tpu_gpu_overrides)
    _validate_accuracy_overrides(grad_gpu_overrides)
    self.tpu_gpu_accuracy_overrides = tpu_gpu_overrides
    self.grad_gpu_accuracy_overrides = grad_gpu_overrides

  def set_dynamism_handlers(
      self,
      filter_fn: FilterFn,
      mark_dynamic_fn: MarkDynamicFn,
  ) -> None:
    """Sets the dynamism handler for bounded dynamism testing.

    Args:
      filter_fn: A function that returns None if the op should be tested for
        dynamism, or a string with the reason to skip if not.
      mark_dynamic_fn: A function that marks an input tensor as dynamic as well
        as any dependent arguments that should be marked dynamic as well.
    """

    self.dynamism_filter_fn = filter_fn
    self.dynamism_mark_dynamic_fn = mark_dynamic_fn

  def set_symbolic_shapes_known_failures(
      self, known_failures: Mapping[str, str]
  ) -> None:
    """Sets the ops that are known to fail --check_symbolic_shapes.

    Args:
      known_failures: Maps the name of an op (with `.{variant_test_name}` if the
        test uses one) to the reason why TorchTPU specializes or throws on its
        symbolic dims. Such an op must still fail the check on some sample, so
        that the entry is removed once the op is fixed.

    To be called by a subclass's setUp() method.
    """
    self.symbolic_shapes_known_failures = known_failures

  def skipTest(self, reason: typing.Any) -> typing.NoReturn:
    """Overrides TestCase.skipTest to bypass skips during dtype exclusion collection."""
    # Never skip tests when dumping dtype exclusions. Every test must run so
    # that its `do_test_op` calls can record its dtype exclusions, regardless of
    # any mode-based or environment-based skip conditions.
    if _dump_dtype_exclusions_mode():
      return  # pyrefly: ignore[bad-return]
    super().skipTest(reason)

  def golden_device_name(self) -> str:
    """Returns the name of the golden device for logging purposes."""
    return "GPU"

  def assert_devices_equivalent(
      self,
      device1: torch.device,
      device2: torch.device,
  ) -> None:
    """Asserts that two devices are equivalent, treating None index as 0."""
    self.assertEqual(device1.type, device2.type, f"{device1} vs {device2}")
    # "tpu:0" and "tpu" mean the same device.
    self.assertEqual(
        device1.index or 0, device2.index or 0, f"{device1} vs {device2}"
    )

  def assert_close(
      self,
      *,
      golden_result: Any,
      torch_tpu_result: Any,
      check_value: CheckValueMode | None = None,
      check_dtype: bool = True,
      rtol: float | None = None,
      atol: Tolerance | None = None,
      preamble: str | None = None,
  ) -> None:
    """Asserts that the TPU result is close to the golden result.

    This function does nothing in the perf mode, as we only want to
    measure the performance of the op there.

    Args:
      golden_result: The golden result.
      torch_tpu_result: The TPU result.
      check_value: The mode for checking the values. None (default) means LOOSE.
      check_dtype: Check if the dtypes are the same.
      rtol: The relative tolerance for checking the values.
      atol: The absolute tolerance for checking the values.
      preamble: Optional string to describe the objects being compared.
    """

    if check_value == CheckValueMode.LOOSE:
      raise ValueError(
          "LOOSE mode is the default. Please omit the check_value argument."
      )

    _validate_tolerances(rtol=rtol, atol=atol)

    if _perf_mode():
      return

    if isinstance(golden_result, tuple):
      self.assertIsInstance(torch_tpu_result, tuple)
      self.assertEqual(len(golden_result), len(torch_tpu_result))
      for golden_item, torch_tpu_item in zip(golden_result, torch_tpu_result):
        self.assert_close(
            golden_result=golden_item,
            torch_tpu_result=torch_tpu_item,
            check_value=check_value,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            preamble=preamble,
        )
      return

    if isinstance(golden_result, bool):
      self.assertEqual(golden_result, torch_tpu_result)
      return

    # In GPU golden mode with torch.compile, gradients for ops like
    # embedding(..., sparse=True) are densified to strided layout, whereas
    # TorchTPU eager backward produces sparse_coo. Convert to dense for value
    # comparison when layouts differ.
    if isinstance(golden_result, torch.Tensor) and isinstance(
        torch_tpu_result, torch.Tensor
    ):
      if torch_tpu_result.is_sparse and not golden_result.is_sparse:
        torch_tpu_result = torch_tpu_result.to_dense()
      elif golden_result.is_sparse and not torch_tpu_result.is_sparse:
        golden_result = golden_result.to_dense()

    test_utils.assert_close(
        actual=torch_tpu_result,
        expected=golden_result,
        rtol=rtol,
        atol=atol,
        check_value=check_value,
        check_dtype=check_dtype,
        preamble=preamble or "Comparing TorchTPU to golden result",
    )

  # TODO: Move the failure-handling logic to util.assert_close to avoid
  # duplication.
  def assert_close_tpu_vs_cpu(
      self,
      tensor_from_device: Callable[[torch.device], torch.Tensor],
      *,
      check_value: CheckValueMode | None = None,
      check_dtype: bool = True,
      check_exception_type: bool = True,
      allow_failure: bool = False,
      rtol: float | None = None,
      atol: Tolerance | None = None,
  ) -> None:
    """Checks that the TorchTPU result is close to the CPU result.

    If we get an exception on one device, checks that we get an exception on the
    other device.

    Args:
      tensor_from_device: A function that takes a device and returns a tensor
        from that device.
      check_value: The mode for checking the values. None (default) means LOOSE.
      check_dtype: Check if the dtypes are the same.
      check_exception_type: If True, check that the exception type is the same
        on both devices.
      allow_failure: If True, allow the CPU to throw an exception.
      rtol: The relative tolerance for checking the values.
      atol: The absolute tolerance for checking the values.
    """

    if check_value == CheckValueMode.LOOSE:
      raise ValueError(
          "LOOSE mode is the default. Please omit the check_value argument."
      )

    _validate_tolerances(rtol=rtol, atol=atol)

    cpu_result = None
    cpu_thrown = None
    torch_tpu_result = None
    torch_tpu_thrown = None
    try:
      cpu_result = tensor_from_device("cpu")
    except Exception as e:  # pylint: disable=broad-except
      cpu_thrown = e
    if cpu_thrown and not allow_failure:
      message = (
          "CPU threw an exception but allow_failure=False.\n"
          f"Exception type: {cpu_thrown.__class__.__name__}\n"
          f"Exception message: {cpu_thrown}"
      )
      self.fail(message)

    try:
      torch_tpu_result = to(tensor_from_device(torch.device("tpu")), "cpu")
    except Exception as e:  # pylint: disable=broad-except
      torch_tpu_thrown = e
    if cpu_thrown and not torch_tpu_thrown:
      self.fail(
          "TorchTPU did not throw but CPU threw an exception of type"
          f" {cpu_thrown.__class__.__name__}:\n{cpu_thrown}",
      )
    elif not cpu_thrown and torch_tpu_thrown:
      self.fail(
          "CPU did not throw an exception but TorchTPU threw an exception of"
          f" type {torch_tpu_thrown.__class__.__name__}:\n{torch_tpu_thrown}",
      )
    if check_exception_type and cpu_thrown and torch_tpu_thrown:
      self.assertIs(
          type(cpu_thrown),
          type(torch_tpu_thrown),
          "CPU and TorchTPU threw different types of exceptions.",
      )
    else:
      self.assert_close(
          golden_result=cpu_result,
          torch_tpu_result=torch_tpu_result,
          check_value=check_value,
          check_dtype=check_dtype,
          rtol=rtol,
          atol=atol,
      )


class OpInfoTestBase(TorchTpuTestBase):
  """Base class for PyTorch OpInfo database matrix tests (like ops_test.py).

  Extends TorchTpuTestBase with do_test_op() and OpInfo sample generation/golden
  logic.
  """

  def setUp(self) -> None:
    super().setUp()
    # Each test records its goldens on one GPU: the one set by
    # @record_gpu_goldens_on, or the default GPU if untagged.
    method = getattr(self, self._testMethodName, None)
    if _gen_gpu_golden_mode():
      arch = getattr(method, _GPU_GOLDEN_ARCH_ATTR, _DEFAULT_GPU_GOLDEN_ARCH)
      if golden_target_gpu_arch() != arch:
        self.skipTest(
            f"GPU goldens for {self._testMethodName} are recorded on {arch}."
        )
    # The subtests that must run on an unbounded dynamic kernel, set by
    # @test_unbounded_dynamic.
    self._unbounded_dynamic_spec: UnboundedDynamicSpec | None = getattr(
        method, _UNBOUNDED_DYNAMIC_ATTR, None
    )
    if _UNBOUNDED_DYNAMIC_TESTS_ONLY.value:
      if self._unbounded_dynamic_spec is None:
        self.skipTest(
            f"{self._testMethodName} is not decorated with"
            " @test_unbounded_dynamic."
        )
      if os.environ.get(
          "TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS", ""
      ).lower() not in ("1", "true"):
        self.fail(
            "--unbounded_dynamic_tests_only requires"
            " TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS=1."
        )

  def _get_golden_input_output_pairs(
      self,
      op: OpInfo,
      dtype: torch.dtype,
      variant: OpVariant,
      *,
      max_samples: int | None,
      compute_grad: bool = False,
      use_compiled: bool = False,
      verbose: bool = True,
      out_dtype_override: torch.dtype | None = None,
  ) -> Sequence[tuple[OpInput, OpOutput]]:
    """Returns a list of (input, output) pairs for the op.

    In the gen_gpu_golden mode, the inputs are generated randomly using
    op.sample_inputs(), and the outputs are computed on GPU.

    In the other modes, the input-output pairs are read from the GPU golden
    file.

    Args:
      op: The op to test.
      dtype: The dtype to test.
      variant: The variant of the op to test.
      max_samples: The maximum number of samples to generate for the given (op
        variant, dtype) combination. If None, the number is determined by the
        --max_samples_per_op flag.
      compute_grad: If True, compute the gradient of the op with respect to the
        input instead of the op outputs.
      use_compiled: If True, use torch.compile to compile the op before running.
      verbose: If True, print a log message when starting to fetch golden
        results.
      out_dtype_override: The data type of the out tensor to use when testing
        out-variant dtype casting.
    """

    op_name = _op_name_for_logging(op, variant)
    if verbose:
      print(
          f">>> Getting golden results for {op_name}() with dtype {dtype} ...",
          flush=True,
      )

    if max_samples is None:
      max_samples = _MAX_SAMPLES_PER_OP_DTYPE.value
    if _gen_gpu_golden_mode():
      return self._generate_golden_pairs(
          op,
          dtype,
          variant,
          max_samples,
          compute_grad=compute_grad,
          use_compiled=use_compiled,
          out_dtype_override=out_dtype_override,
      )
    return self._read_golden_pairs(
        op, dtype, variant, max_samples, out_dtype_override
    )

  def _read_golden_pairs(
      self,
      op: OpInfo,
      dtype: torch.dtype,
      variant: OpVariant,
      max_samples: int,
      out_dtype_override: torch.dtype | None,
  ) -> Sequence[tuple[OpInput, OpOutput]]:
    """Returns up to `max_samples` recorded GPU golden pairs for the op."""
    samples = _GOLDEN_GPU_DATA.get_samples(
        self._testMethodName,
        variant,
        dtype,
        out_dtype_override=out_dtype_override,
    )
    # TODO(b/540887166): Enable this check for compiled mode too when the bug
    # is fixed.
    if not samples and not is_compiled_mode():
      self.fail(
          "No GPU golden samples found for"
          f" {_op_name_for_logging(op, variant)}() with dtype {dtype}. Please"
          " re-generate the GPU golden files to cover the new (op, dtype)."
      )
    if max_samples >= 0 and len(samples) > max_samples:
      print(
          f">>> Taking {max_samples} samples from "
          f" {len(samples)} test samples ...",
          flush=True,
      )
      samples = samples[:max_samples]
    return samples

  def _generate_golden_pairs(
      self,
      op: OpInfo,
      dtype: torch.dtype,
      variant: OpVariant,
      max_samples: int,
      **run_op_kwargs: Any,
  ) -> Sequence[tuple[OpInput, OpOutput]]:
    """Generates up to `max_samples` golden pairs on the golden device."""
    golden_samples = _sample_inputs(op, self.golden_device, dtype)
    if max_samples >= 0 and len(golden_samples) > max_samples:
      print(
          f">>> Taking {max_samples} random samples from "
          f" {len(golden_samples)} test samples ...",
          flush=True,
      )
      # Take at most max_samples from the generated samples.
      golden_samples = random.sample(golden_samples, max_samples)

    pairs = []
    for golden_sample in golden_samples:
      # sample_inputs for scaled_dot_product_attention sometimes includes a
      # non-zero dropout parameter, but we don't guarantee randomness matches
      # and would need to compare distributions.
      if op.name == "nn.functional.scaled_dot_product_attention":
        golden_sample.kwargs.update({"dropout_p": 0.0})

      golden_input = OpInput(golden_sample)
      golden_result = self._run_op(
          op=op,
          variant=variant,
          dtype=dtype,
          op_input=golden_input,
          device=self.golden_device,
          subtest_name="golden",
          # When generating golden results, we should trust the op's output
          # to be correct and not check the device. E.g. even in the
          # gen_gpu_golden mode, torch.arange(5) should return a tensor on
          # CPU instead of GPU.
          check_device=False,
          **run_op_kwargs,
      )
      pairs.append((golden_input, OpOutput(golden_result)))
    return pairs

  def _run_op(
      self,
      op: OpInfo,
      variant: OpVariant,
      dtype: torch.dtype,
      op_input: OpInput,
      device: torch.device,
      *,
      subtest_name: str,
      check_device: bool,
      out: Any = None,
      compute_grad: bool = False,
      use_compiled: bool = False,
      measure_perf: bool = False,
      out_dtype_override: torch.dtype | None = None,
  ) -> Any:
    """Run the op on the given device.

    Args:
      op: The op to run.
      variant: The variant of the op to run.
      dtype: The dtype to run the op with.
      op_input: A sample input generated by the op's sample_inputs() method. The
        devices in this sample may not be the same as the given device. This
        function does not mutate op_input. Instead, it clones op_input, converts
        the clone to the given device, and runs the op on the clone.
      device: The device to run the op on.
      subtest_name: A name for the subtest being run. Used to group results that
        represent multiple runs of the same subtest.
      check_device: If True, check that the result tensors are on the given
        device.
      out: If variant is OUT, this can be either the out argument to be used
        when running the op or None if this function should create the out
        argument itself. Otherwise this must be None.
      compute_grad: If True, compute the gradient of the op with respect to the
        input instead of the op outputs.
      use_compiled: If True, use torch.compile to compile the op before running.
      measure_perf: If True, measure the performance of the op.
      out_dtype_override: The data type of the out tensor to use when testing
        out-variant dtype casting.

    Returns:
      When compute_grad is false: the result of the op, transferred to the CPU
      device; or the Exception thrown by the op if it failed.
      When compute_grad is true: the gradients of the op with respect to the
      input, transferred to the CPU device; or the Exception thrown by the op
      if it failed.
    """

    if variant != OpVariant.OUT:
      assert (
          out is None
      ), f"out must be None when testing the {variant} variant of an op."

    op_name = _op_name_for_logging(op, variant)
    target_device = _target_device(dtype, device)
    if variant == OpVariant.OUT and out is None:
      out = self._prepare_out(
          op,
          dtype,
          op_input,
          device,
          subtest_name=subtest_name,
          check_device=check_device,
          use_compiled=use_compiled,
          out_dtype_override=out_dtype_override,
      )
      if isinstance(out, Exception):
        return out

    op_func = _op_func(op, variant, target_device)

    # Clone the sample to prevent the op from mutating it.
    # We must deepcopy op_input *before* setting its device to the given device,
    # because deepcopying a TPU tensor is not implemented yet.
    device_op_input = to(copy.deepcopy(op_input), target_device)
    if variant == OpVariant.OUT:
      device_op_input.kwargs["out"] = out

    is_dynamic_test = bool(
        _CHECK_DYNAMISM_USING_SEED.value and device == torch.device("tpu")
    )
    if is_dynamic_test:
      self._mark_dynamic(
          op,
          variant,
          dtype,
          op_input,
          device_op_input,
          use_compiled=use_compiled,
      )
    if use_compiled:
      op_func = self._compile(
          op_func,
          op,
          variant,
          device,
          device_op_input,
          is_dynamic_test=is_dynamic_test,
      )
    if compute_grad:
      # We only support testing gradients when input_value is a tensor or a
      # (possibly nested) list/tuple of tensors.
      device_op_input.input_value = _requires_grad_input(
          device_op_input.input_value
      )
      if not _can_compute_grad(variant, device_op_input.input_value):
        return _dummy_grad(device)
    input_value = device_op_input.input_value

    try:
      start_time = time.perf_counter()
      result = op_func(  # pyrefly: ignore[not-callable]
          input_value,
          *device_op_input.args,
          **device_op_input.kwargs,
      )
      if variant == OpVariant.INPLACE:
        # For inplace ops, we want to check how the input was mutated.
        result = input_value
      elif variant == OpVariant.OUT:
        # For the out variant, we want to check how the out tensor was
        # mutated.
        result = out

      if compute_grad:
        result = _compute_grad(result, input_value, device)

      if check_device:
        self._assert_on_device(result, target_device)

      # In the eager mode of pytorch CPU/GPU, ops are run synchronously: if
      # an op encounters an error, the op() call itself will raise an
      # exception. In torch_tpu's eager mode, however, ops are usually deferred:
      # if an op encounters an error, an exception can be raised either during
      # the op() call or when the op is actually run (e.g. when we materialize
      # the result of the op via .to("cpu")). Most users don't care about
      # this subtle difference.
      #
      # Therefore, to compare the behavior of an op between TorchTPU and CPU/GPU
      # fairly, we should treat the .to("cpu") call as if it's part of the op
      # on TorchTPU. Hence the following call is is done inside _run_op(), and
      # any exception it raises is attributed to the op.
      result = to(result, "cpu")
      if measure_perf:
        # We must measure the time after the .to("cpu") call because that's the
        # time when the compilation and execution of the op is guaranteed to be
        # done.
        _add_perf_result(
            PerfResult(
                op_name=op_name,
                dtype=dtype,
                subtest_name=subtest_name,
                duration_sec=time.perf_counter() - start_time,
            )
        )
    except Exception as e:  # pylint: disable=broad-except
      # The op raised an exception.
      if device.type != "tpu":
        print(
            f"Exception caught during golden generation for {op_name}: {e}",
            flush=True,
        )
      result = e
    return result

  def _prepare_out(
      self,
      op: OpInfo,
      dtype: torch.dtype,
      op_input: OpInput,
      device: torch.device,
      *,
      subtest_name: str,
      check_device: bool,
      use_compiled: bool,
      out_dtype_override: torch.dtype | None,
  ) -> Any:
    """Returns the out argument for running the out variant of the op.

    Returns:
      Zero-element tensors with the structure of the base variant's result, on
      the device to run the op on; or the Exception thrown by the base variant.
    """
    # First, execute the base variant to dynamically determine the output
    # PyTree structure (e.g., single tensor vs. tuple of tensors) and natural
    # dtypes (e.g., bool for comparison ops or int64 for auxiliary index
    # outputs).
    base_result = self._run_op(
        op=op,
        variant=OpVariant.BASE,
        dtype=dtype,
        op_input=op_input,
        compute_grad=False,
        use_compiled=use_compiled,
        device=device,
        subtest_name=subtest_name,
        check_device=check_device,
        # This _run_op() call is for preparing test data, not part of the test
        # itself, so don't measure its performance to avoid skewing metrics.
        measure_perf=False,
    )
    if isinstance(base_result, Exception):
      return base_result
    # Next, generate zero-element out tensors (shape (0,)) matching the
    # discovered structure to verify out-variant shape resizing compliance.
    return to(
        _prepare_zero_element_out_tensors(
            base_result,
            out_dtype_override=out_dtype_override,
        ),
        _target_device(dtype, device),
    )

  def _mark_dynamic(
      self,
      op: OpInfo,
      variant: OpVariant,
      dtype: torch.dtype,
      op_input: OpInput,
      device_op_input: OpInput,
      *,
      use_compiled: bool,
  ) -> None:
    """Marks dimensions of `device_op_input` dynamic, or skips the test."""
    op_name = _op_name_for_logging(op, variant)
    dynamism_mode = DynamismMode.COMPILE if use_compiled else DynamismMode.EAGER
    print(
        f">>> Marking dynamism for {op_name} with dtype {dtype}"
        f" (mode={dynamism_mode.value}) ...",
        flush=True,
    )

    # Verify op is supported with dynamism.
    skip = self.dynamism_filter_fn(
        op, op_input.input_value, op_input.args, mode=dynamism_mode
    )
    if skip is not None:
      self.skipTest(skip)

    # Use a seed value to enable reproducing a failure deterministically.
    seed = _CHECK_DYNAMISM_USING_SEED.value
    seed = seed if seed > 0 else (time.time_ns() % 100000)
    print(f">>>> Reproduce using `--check_dynamism_using_seed={seed}`")
    self.dynamism_mark_dynamic_fn(
        seed,
        op,
        device_op_input.input_value,
        device_op_input.args,
        mode=dynamism_mode,
        kwargs=device_op_input.kwargs,
    )

  def _compile(
      self,
      op_func: Callable[..., Any],
      op: OpInfo,
      variant: OpVariant,
      device: torch.device,
      device_op_input: OpInput,
      *,
      is_dynamic_test: bool,
  ) -> Callable[..., Any]:
    """Returns `op_func` compiled for `device`, or skips the test."""
    op_name = _op_name_for_logging(op, variant)
    if not _compiled_supports_op(op, device_op_input, variant):
      self.skipTest(f"TpuBackend does not support {op_name} currently.")
    logging.getLogger(tt_compile.TpuBackend.__module__).setLevel(logging.DEBUG)
    backend = "tpu" if str(device) == "tpu" else "inductor"
    print(f"Compiling {op_name} for device {device} ...", flush=True)
    if is_dynamic_test:
      # Reset Dynamo state so dimensions marked dynamic in earlier subtests
      # do not carry over across compilations.
      torch._dynamo.reset()  # pylint: disable=protected-access
      return torch.compile(op_func, backend=backend)
    return torch.compile(op_func, dynamic=False, backend=backend)

  def _assert_on_device(self, result: Any, device: torch.device) -> None:
    """Asserts that the tensors in `result` are on `device`."""

    # To prevent bugs in ops and the test itself, we check that
    # tensors in the result are indeed on the expected device.
    def assert_on_device(obj: Any) -> None:
      if isinstance(obj, torch.Tensor):
        self.assert_devices_equivalent(obj.device, device)

    try:
      _tensor_tree_map(assert_on_device, result)
    except AssertionError as e:
      self.fail(f"Expected result to be on {device}, but got {result}: {e}")

  def _assert_failure_consistency(
      self,
      *,
      golden_result: Any,
      torch_tpu_result: Any,
      op_description: str,
      torch_tpu_printable_input: OpInput,
  ) -> None:
    """Asserts that golden_result and torch_tpu_result are both exceptions or both not exceptions."""

    golden_thrown = isinstance(golden_result, Exception)
    torch_tpu_thrown = isinstance(torch_tpu_result, Exception)

    if golden_thrown and not torch_tpu_thrown:
      self.fail(
          f"\n{op_description} failed on"
          f" {self.golden_device_name()}, so it should fail on TorchTPU too."
          " However, it succeeded on TorchTPU with\n"
          f"{torch_tpu_printable_input}\n"
          f"Error on {self.golden_device_name()}: {golden_result}",
      )
    elif not golden_thrown and torch_tpu_thrown:
      self.fail(
          f"\n{op_description} succeeded on"
          f" {self.golden_device_name()}, so it should succeed on TorchTPU"
          " too. However, it failed on TorchTPU with error:\n"
          "<<< ERROR START\n"
          f"{torch_tpu_result}\n"
          "<<< ERROR END\n"
          f"{torch_tpu_printable_input}\n"
          f"{self.golden_device_name()} result: {golden_result}",
      )
    elif golden_thrown and torch_tpu_thrown:
      golden_type = type(golden_result).__name__
      torch_tpu_type = type(torch_tpu_result).__name__

      if golden_type != torch_tpu_type:
        pass
        # TODO(b/542331196): enable it once exceptions are recorded in the
        # structured dict format.
        # self.fail(
        #     f"\n{op_description} failed on both {self.golden_device_name()} and"
        #     " TorchTPU, but raised different exception types.\n"
        #     f"{self.golden_device_name()}: {golden_type}: {golden_result}\n"
        #     f"TorchTPU: {torch_tpu_type}: {torch_tpu_result}\n"
        #     f"{torch_tpu_printable_input}",
        # )

  def _assert_structure_consistency(
      self, *, golden_result: Any, torch_tpu_result: Any
  ) -> None:
    """Asserts that golden_result and torch_tpu_result are both None, tensors, or tuples of the same size."""
    if golden_result is None:
      self.assertIsNone(
          torch_tpu_result,
          f"{self.golden_device_name()} result is None, but TorchTPU result is"
          f" a {type(torch_tpu_result)}",
      )
      return

    if isinstance(golden_result, torch.Tensor):
      self.assertIsInstance(
          torch_tpu_result,
          torch.Tensor,
          f"{self.golden_device_name()} result is a tensor, but TorchTPU"
          f" result is a {type(torch_tpu_result)}",
      )
      return

    if isinstance(golden_result, list):
      self.assertIsInstance(
          torch_tpu_result,
          list,
          f"{self.golden_device_name()} result is a list, but TorchTPU"
          f" result is a {type(torch_tpu_result)}",
      )
      self.assertEqual(len(golden_result), len(torch_tpu_result))
      return

    if isinstance(golden_result, bool):
      self.assertIsInstance(
          torch_tpu_result,
          bool,
          f"{self.golden_device_name()} result is a bool, but TorchTPU"
          f" result is a {type(torch_tpu_result)}",
      )
      return

    self.assertIsInstance(
        golden_result,
        tuple,
        f"{self.golden_device_name()} result is neither a tensor, a bool, nor"
        f" a list or tuple: {type(golden_result)}",
    )
    self.assertIsInstance(
        torch_tpu_result,
        tuple,
        "TorchTPU result is neither a tensor nor a list or tuple:"
        f" {type(torch_tpu_result)}",
    )
    self.assertEqual(len(golden_result), len(torch_tpu_result))

  def _assert_tuple_close(
      self,
      *,
      golden_result: Any,
      torch_tpu_result: Any,
      torch_tpu_printable_input: OpInput,
      check_value: CheckValueMode | Iterable[CheckValueMode] | None,
      check_dtype: bool,
      skip_output_indices: Sequence[int],
      accuracy_override: Mapping[str, Tolerance],
  ) -> None:
    """Asserts that golden_result and torch_tpu_result, as tuples, are close."""

    if check_value == CheckValueMode.LOOSE:
      raise ValueError(
          "LOOSE mode is the default. Please omit the check_value argument."
      )

    _validate_tolerances(
        rtol=accuracy_override.get("rtol"),
        atol=accuracy_override.get("atol"),
    )

    golden_result_tuple = _to_tuple(golden_result)
    torch_tpu_result_tuple = _to_tuple(torch_tpu_result)
    if check_value is None or isinstance(check_value, CheckValueMode):
      check_value = [check_value] * len(torch_tpu_result_tuple)  # pyrefly: ignore[bad-assignment]

    printable_input_str = str(torch_tpu_printable_input)
    for i, golden_result_i in enumerate(golden_result_tuple):
      if i in skip_output_indices:
        continue

      preamble_parts = []
      preamble_parts.append(printable_input_str)
      if isinstance(torch_tpu_result, tuple):
        preamble_parts.append(f"Mismatching result item: {i}")
      preamble_parts.append("Comparing TorchTPU to golden result")
      preamble = "\n".join(preamble_parts)

      self.assert_close(
          golden_result=golden_result_i,
          torch_tpu_result=torch_tpu_result_tuple[i],
          check_value=check_value[i],  # pyrefly: ignore[bad-index, unsupported-operation]
          check_dtype=check_dtype,
          preamble=preamble,
          **accuracy_override,  # pyrefly: ignore[bad-argument-type]
      )

  def _run_on_tpu(
      self,
      subtest_name: str,
      op: OpInfo,
      variant: OpVariant,
      dtype: torch.dtype,
      *,
      golden_input: OpInput,
      **run_op_kwargs: Any,
  ) -> tuple[Any, OpInput]:
    """Returns the TorchTPU result and printable input for the golden input."""
    print(f">>>> Testing {subtest_name} ...", flush=True)
    torch_tpu_printable_input = _to_torch_tpu_printable_input(golden_input)
    torch_tpu_printable_input.name = subtest_name
    if _PRINT_OP_INPUTS.value > 0:
      # TODO b/486861095
      if op.name != "normal":
        print_reproducer(subtest_name, op, golden_input, variant)
      print_op_input(torch_tpu_printable_input, data=_PRINT_OP_INPUTS.value > 1)

    compiler_options = (
        {"xla_optimization_level": _OPT_LEVEL.value} if _OPT_LEVEL.value else {}
    )
    spec = getattr(self, "_unbounded_dynamic_spec", None)
    check_unbounded_dynamic = (
        _UNBOUNDED_DYNAMIC_TESTS_ONLY.value
        and spec is not None
        and not run_op_kwargs.get("compute_grad", False)
        and not run_op_kwargs.get("use_compiled", False)
        and spec.matches(variant, dtype, golden_input)
    )
    hits_before = (
        torch.tpu._get_cache_stats().num_unbounded_dynamic_registry_hits  # pylint: disable=protected-access
        if check_unbounded_dynamic
        else 0
    )
    with compiler.custom_compiler_options(compiler_options):
      torch_tpu_result = self._run_op(
          op=op,
          variant=variant,
          dtype=dtype,
          op_input=golden_input,
          device=torch.device("tpu"),
          subtest_name=subtest_name,
          **run_op_kwargs,
      )
    if check_unbounded_dynamic and not isinstance(torch_tpu_result, Exception):
      self.assertGreater(
          torch.tpu._get_cache_stats().num_unbounded_dynamic_registry_hits,  # pylint: disable=protected-access
          hits_before,
          f"{subtest_name} was not served by an unbounded dynamic kernel, but"
          " it matches @test_unbounded_dynamic.",
      )
    return torch_tpu_result, torch_tpu_printable_input

  def _sub_test(
      self,
      subtest_name: str,
      op: OpInfo,
      variant: OpVariant,
      dtype: torch.dtype,
      *,
      golden_input: OpInput,
      golden_result: Any,
      check_value: CheckValueMode | Iterable[CheckValueMode] | None,
      check_dtype: bool,
      skip_output_indices: Sequence[int],
      compute_grad: bool,
      use_compiled: bool,
      **run_op_kwargs: Any,
  ) -> None:
    # do_test_op() already rejects LOOSE, so this never fires today. Keep it to
    # catch mistakes by future callers.
    if check_value == CheckValueMode.LOOSE:
      raise ValueError(
          "LOOSE mode is the default. Please omit the check_value argument."
      )

    op_name = _op_name_for_logging(op, variant)
    if _IGNORE_ACCURACY_OVERRIDES.value:
      accuracy_overrides = {}
    elif compute_grad:
      accuracy_overrides = self.grad_gpu_accuracy_overrides
    else:
      accuracy_overrides = self.tpu_gpu_accuracy_overrides

    # Default to the accuracy override for the base op name for _foreach_ ops.
    accuracy_override = accuracy_overrides.get(op.name, {}).get(dtype, {})
    if not accuracy_override and op.name.startswith("_foreach_"):
      base_op_name = op.name[9:]
      accuracy_override = accuracy_overrides.get(base_op_name, {}).get(
          dtype, {}
      )

    torch_tpu_result, torch_tpu_printable_input = self._run_on_tpu(
        subtest_name,
        op,
        variant,
        dtype,
        golden_input=golden_input,
        compute_grad=compute_grad,
        use_compiled=use_compiled,
        **run_op_kwargs,
    )
    golden_thrown = isinstance(golden_result, Exception)
    torch_tpu_thrown = isinstance(torch_tpu_result, Exception)

    self._assert_failure_consistency(
        golden_result=golden_result,
        torch_tpu_result=torch_tpu_result,
        op_description=(
            f"{op_name}() with dtype {dtype} (variant={variant.value})"
        ),
        torch_tpu_printable_input=torch_tpu_printable_input,
    )

    if not golden_thrown and not torch_tpu_thrown:
      self._assert_structure_consistency(
          golden_result=golden_result,
          torch_tpu_result=torch_tpu_result,
      )
      self._assert_tuple_close(
          golden_result=golden_result,
          torch_tpu_result=torch_tpu_result,
          torch_tpu_printable_input=torch_tpu_printable_input,
          check_value=check_value,
          check_dtype=check_dtype,
          skip_output_indices=skip_output_indices,
          accuracy_override=accuracy_override,
      )

  def _check_symbolic_shapes_sub_test(
      self,
      subtest_name: str,
      op: OpInfo,
      variant: OpVariant,
      dtype: torch.dtype,
      *,
      golden_input: OpInput,
      golden_result: Any,
      **unused_run_op_kwargs: Any,
  ) -> None:
    """Checks that TorchTPU keeps the symbolic dims that CPU keeps.

    Runs `golden_input` on CPU and TorchTPU FakeTensors with symbolic dims (see
    `_run_with_symbolic_dims`), once without and once with autograd, and fails
    if TorchTPU specializes a dim that CPU doesn't, or throws when CPU doesn't.
    CPU is the control: specializations that upstream PyTorch also makes (e.g.
    for data-dependent output shapes or shape arguments) are expected.

    Args:
      subtest_name: The name of the subtest.
      op: The op to check.
      variant: The variant of the op to check.
      dtype: The dtype of the op's inputs.
      golden_input: The input to check the op with.
      golden_result: The golden result for `golden_input`. If it's an exception,
        the check is skipped, as the op fails on real tensors too.
      **unused_run_op_kwargs: The `_run_op` kwargs of the other test modes.
    """
    del dtype, unused_run_op_kwargs  # Unused.
    op_name = _op_name_for_logging(op, variant)
    if isinstance(golden_result, Exception):
      print(
          f"Skipping {subtest_name} as it failed on"
          f" {self.golden_device_name()}: {golden_result}",
          flush=True,
      )
      return
    # torch.compile doesn't support sparse tensors (see _compiled_supports_op).
    if any(
        isinstance(t, torch.Tensor) and t.layout != torch.strided
        for t in _pytree.tree_leaves(
            (golden_input.input_value, golden_input.args, golden_input.kwargs)
        )
    ):
      print(f"Skipping {subtest_name} as it has sparse tensors.", flush=True)
      return

    print(f">>>> Checking symbolic shapes in {subtest_name} ...", flush=True)
    if _PRINT_OP_INPUTS.value > 0:
      print_op_input(golden_input, data=_PRINT_OP_INPUTS.value > 1)
    grad_modes = [False]
    # Autograd doesn't work with inplace ops, as they erase the input tensor.
    if (
        op.supports_autograd
        and variant != OpVariant.INPLACE
        and _has_differentiable_input(golden_input)
    ):
      grad_modes.append(True)
    failures = []
    for requires_grad in grad_modes:
      run_with_symbolic_dims = functools.partial(
          _run_with_symbolic_dims, requires_grad=requires_grad
      )
      cpu_device = torch.device("cpu")
      cpu = run_with_symbolic_dims(
          _op_func(op, variant, cpu_device), golden_input, cpu_device
      )
      if cpu.error is not None:
        # The control can't tell which specializations are expected.
        print(
            f"Skipping {subtest_name} (requires_grad={requires_grad}) as CPU"
            f" threw on symbolic dims: {type(cpu.error).__name__}:"
            f" {cpu.error}",
            flush=True,
        )
        continue
      tpu_device = torch.device("tpu")
      tpu = run_with_symbolic_dims(
          _op_func(op, variant, tpu_device), golden_input, tpu_device
      )
      if tpu.error is not None:
        failures.append(
            f"With requires_grad={requires_grad}, TorchTPU threw on symbolic"
            f" dims but CPU didn't: {type(tpu.error).__name__}: {tpu.error}"
        )
      elif extra := tpu.specialized - cpu.specialized:
        failures.append(
            f"With requires_grad={requires_grad}, TorchTPU specialized"
            f" {sorted(extra)}, which CPU keeps symbolic (CPU specialized"
            f" {sorted(cpu.specialized)})."
        )
    if not failures:
      return

    message = (
        f"{op_name}() specializes or throws on symbolic dims on TorchTPU but"
        " not on CPU, which makes torch.compile recompile for every new"
        f" shape.\n{golden_input.summary()}\n"
        + "\n".join(failures)
    )
    known_failure = self.symbolic_shapes_known_failures.get(
        _symbolic_shapes_key(op)
    )
    if known_failure is not None:
      self._known_symbolic_shapes_failure_seen = True
      print(f"Known failure ({known_failure}): {message}", flush=True)
      return
    self.fail(message)

  def _test_torch_tpu_vs_golden(
      self,
      op: OpInfo,
      dtype: torch.dtype,
      variant: OpVariant,
      *,
      compute_grad: bool,
      use_compiled: bool,
      check_value: CheckValueMode | Iterable[CheckValueMode] | None,
      check_dtype: bool,
      check_device: bool,
      check_op_failures: bool,
      skip_output_indices: Sequence[int],
      skip_if: Callable[[OpVariant, OpInput], bool] | None,
      max_samples_per_op_dtype: int | None,
      out_dtype_override: torch.dtype | None = None,
  ) -> None:
    """Tests that the op produces similar results on TorchTPU and the golden device.

    Args:
      op: The op to test.
      dtype: The dtype to test.
      variant: The variant of the op to test.
      compute_grad: Whether to compute the gradient of the op.
      use_compiled: Whether to use the compiled version of the op.
      check_value: The mode for checking the values. If SKIP, only the output's
        dtype and shape will be checked. If an iterable is provided, it must
        have the same length as the op's output list, and the i-th element will
        determine whether to check the i-th output's values (this is useful for
        ops that return multiple outputs).
      check_dtype: Check if the dtypes are the same.
      check_device: Whether to check that the result tensors are on the expected
        device.
      check_op_failures: Whether to check that the op fails on TorchTPU when it
        fails on the golden device.
      skip_output_indices: The indices of the output to skip checking.
      skip_if: A function that returns True if a test case should be skipped,
        given the op variant and the op input.
      max_samples_per_op_dtype: The maximum number of samples to test for each
        (op variant, dtype) combination. If None, the number is determined by
        the --max_samples_per_op_dtype flag.
    """

    # do_test_op() already rejects LOOSE, so this never fires today. Keep it to
    # catch mistakes by future callers.
    if check_value == CheckValueMode.LOOSE:
      raise ValueError(
          "LOOSE mode is the default. Please omit the check_value argument."
      )

    if dtype not in _dtypes_to_test():
      print(
          f"Skipping test for dtype {dtype} as it is not in --dtypes.",
          flush=True,
      )
      return

    op_name = _op_name_for_logging(op, variant)

    spec = getattr(self, "_unbounded_dynamic_spec", None)
    if _UNBOUNDED_DYNAMIC_TESTS_ONLY.value and (
        spec is None
        or compute_grad
        or use_compiled
        or dtype not in spec.dtypes
        or variant not in spec.variants
    ):
      print(
          f"Skipping test for {op_name}() (variant={variant.value}) with dtype"
          f" {dtype} as no unbounded dynamic kernel serves it.",
          flush=True,
      )
      return

    print(
        f">>> Testing {op_name}() (variant={variant.value}) with dtype"
        f" {dtype} ...",
        flush=True,
    )

    golden_pairs = self._get_golden_input_output_pairs(
        op=op,
        dtype=dtype,
        variant=variant,
        compute_grad=compute_grad,
        use_compiled=use_compiled,
        max_samples=max_samples_per_op_dtype,
        out_dtype_override=out_dtype_override,
    )
    if _gen_gpu_golden_mode():
      for golden_input, golden_output in golden_pairs:
        _GOLDEN_GPU_DATA.add(
            self._testMethodName,
            variant,
            dtype,
            golden_input,
            golden_output,
            out_dtype_override=out_dtype_override,
        )
      return

    for i, [golden_input, golden_output] in enumerate(golden_pairs):
      golden_result = golden_output.output_value
      golden_thrown = isinstance(golden_result, Exception)
      if golden_thrown and not check_op_failures:
        print(
            f"Skipping test sample for {op_name}() with dtype {dtype} on"
            " TorchTPU as it failed on"
            f" {self.golden_device_name()}.\n{golden_input}\n"
            f"Error: {golden_result}\n",
            flush=True,
        )
        continue

      if skip_if and skip_if(variant, golden_input):
        print(
            f"Skipping test sample for {op_name}() with dtype {dtype} on"
            " TorchTPU as filtered by the skip_if function.\n"
            f"{golden_input}",
            flush=True,
        )
        continue

      if (
          _UNBOUNDED_DYNAMIC_TESTS_ONLY.value
          and spec is not None
          and not spec.matches(variant, dtype, golden_input)
      ):
        print(
            f"Skipping test sample for {op_name}() with dtype {dtype} on"
            " TorchTPU as it is not compatible with the unbounded dynamic"
            f" kernel.\n{golden_input}",
            flush=True,
        )
        continue

      # Test each run of each sample in a subTest s.t. one sample's failure
      # doesn't prevent other runs.
      # Construct a meaningful and unique subtest name.
      subtest_name = f"{op_name}_{_dtype_str(dtype)}_"
      subtest_name += golden_input.name if golden_input.name else f"subtest{i}"

      if _perf_mode():
        test = functools.partial(self._run_on_tpu, measure_perf=True)
      elif _CHECK_SYMBOLIC_SHAPES.value:
        test = functools.partial(
            self._check_symbolic_shapes_sub_test, golden_result=golden_result
        )
      else:
        test = functools.partial(
            self._sub_test,
            golden_result=golden_result,
            check_value=check_value,
            check_dtype=check_dtype,
            skip_output_indices=skip_output_indices,
        )
      for perf_run_idx in range(_NUM_RUNS_PER_SUBTEST.value):
        self._subtests_run += 1
        with self.subTest(subtest_name=subtest_name, run_index=perf_run_idx):
          _run_and_print_exception(
              functools.partial(
                  test,
                  subtest_name,
                  op,
                  variant,
                  dtype,
                  golden_input=golden_input,
                  check_device=check_device,
                  compute_grad=compute_grad,
                  use_compiled=use_compiled,
                  out_dtype_override=out_dtype_override,
              )
          )

  def _assert_known_symbolic_shapes_failure_seen(
      self, op_name: str, variant_test_name: str | None
  ) -> None:
    """Fails if a known --check_symbolic_shapes failure no longer fails."""
    key = f"{op_name}.{variant_test_name}" if variant_test_name else op_name
    reason = self.symbolic_shapes_known_failures.get(key)
    if reason is None or self._known_symbolic_shapes_failure_seen:
      return
    # Runs restricted to some dtypes may miss the failing samples.
    if _DTYPES.value != ["all"] or self._subtests_run == 0:
      return
    self.fail(
        f"{key} is a known --check_symbolic_shapes failure ({reason}), but no"
        " sample specialized or threw on TorchTPU. If it's fixed, remove it"
        " from the known failures."
    )

  def do_test_op(
      self,
      op_name: str,
      *,
      exclude_dtypes: Iterable[torch.dtype] | None = None,
      exclude_inplace_dtypes: Iterable[torch.dtype] | None = None,
      check_out_variant: bool = True,
      check_grad: bool = True,
      check_device: bool = True,
      check_value: CheckValueMode | Iterable[CheckValueMode] | None = None,
      check_dtype: bool = True,
      check_op_failures: bool = True,
      check_inplace_op_failures: bool = True,
      skip_output_indices: Sequence[int] | None = None,
      skip_if: Callable[[OpVariant, OpInput], bool] | None = None,
      max_samples_per_op_dtype: int | None = None,
      variant_test_name: str | None = None,
  ) -> None:
    """Does the standard suite of testing for the given op.

    Args:
      op_name: The name of the op to test.
      exclude_dtypes: A list of input dtypes to exclude from testing. If None,
        no dtypes will be excluded.
      exclude_inplace_dtypes: Similar to exclude_dtypes, but for the inplace
        variant of the op.
      check_out_variant: Whether to check the out variant of the op.
      check_grad: Whether to check the gradient of the op.
      check_device: Whether to check that the op's result is on the correct
        device.
      check_value: Mode for checking the values. If SKIP, only the output's
        dtype and shape will be checked. If an iterable is provided, it must
        have the same length as the op's output list, and the i-th element will
        determine whether to check the i-th output's values (this is useful for
        ops that return multiple outputs). None (default) means LOOSE.
      check_dtype: Check if the dtypes are the same.
      check_op_failures: If True, check that the op fails on TorchTPU when it
        fails on the golden device. If False, the op will be skipped on TorchTPU
        if it fails on the golden device.
      check_inplace_op_failures: If True, check that the inplace variant of the
        op fails on TorchTPU when it fails on the golden device. If False, the
        inplace variant of the op will be skipped on TorchTPU if the input
        causes it to fail on the golden device.
      skip_output_indices: A list of output indices to skip when checking the
        values of the outputs when the op has multiple outputs.
      skip_if: A function that returns True if a test case should be skipped,
        given the op variant and the op input.
      max_samples_per_op_dtype: The maximum number of samples to test for a
        given (op_variant, dtype) combination. If None, the number is determined
        by the --max_samples_per_op_dtype flag.
      variant_test_name: The name of the variant to test, referring to the
        variant_test_name field of the OpInfo, when named variants are
        available. If None, the base variant set, filtering only on the op_name,
        will be tested.

    Raises:
      RuntimeError: If this is called twice in the same test method.
    """

    # Each test method must call do_test_op() at most once because the test
    # method name is used as the key to store and retrieve GPU golden results.
    # Calling it multiple times would cause different operations or variants
    # to share the same key, corrupting the golden data structure.
    if self._do_test_op_called:
      raise RuntimeError(
          f"do_test_op() was called multiple times in {self._testMethodName}. "
          "Each test method must call do_test_op() at most once."
      )
    self._do_test_op_called = True
    self._subtests_run = 0

    if _dump_dtype_exclusions_mode():
      _record_gpu_dtype_exclusions(
          op_name,
          exclude_dtypes=exclude_dtypes,
          exclude_inplace_dtypes=exclude_inplace_dtypes,
      )
      return

    if check_value == CheckValueMode.LOOSE:
      raise ValueError(
          "LOOSE mode is the default. Please omit the check_value argument."
      )

    recording = _gen_gpu_golden_mode()
    raw_exclude_dtypes = exclude_dtypes
    # During golden generation, we ignore all exclusions (both general and
    # device-specific) to proactively collect as much golden data as possible
    # for future compatibility. Robustness against GPU-unsupported cases is
    # handled by catching exceptions during sample generation.
    exclude_dtypes = () if recording else tuple(exclude_dtypes or ())
    exclude_inplace_dtypes = (
        () if recording else tuple(exclude_inplace_dtypes or ())
    )
    skip_output_indices = skip_output_indices or []
    compute_grad = _COMPUTE_GRAD.value
    use_compiled = is_compiled_mode()

    if compute_grad and not check_grad:
      self.skipTest(
          f"Skipping gradient tests for {op_name} as check_grad is False."
      )

    def _test_op_variants() -> None:
      """Tests all enabled variants of the op.

      Currently only the out variant can be disabled (via check_out_variant).
      """
      op = _get_op(op_name, variant_test_name=variant_test_name)
      dtypes_to_test = _dtypes_to_test_for_op(op, recording)
      # No applicable dtypes means exclude_dtypes intentionally excluded
      # everything, so skipping is expected.
      if all(
          _should_skip_dtype(d, exclude_dtypes=exclude_dtypes)
          for d in dtypes_to_test
      ):
        self.skipTest(
            f"All dtypes excluded for {op_name} on"
            f" {self.golden_device_name()}: exclude_dtypes={exclude_dtypes}"
        )

      # (variant, excluded dtypes, check_op_failures) for each variant to test.
      variants = [(OpVariant.BASE, exclude_dtypes, check_op_failures)]
      # --check_symbolic_shapes skips the out variant: it dispatches to the same
      # autograd and Meta kernels as the base variant on FakeTensors.
      if (
          check_out_variant
          and _supports_out(op)
          and not _CHECK_SYMBOLIC_SHAPES.value
      ):
        variants.append((OpVariant.OUT, exclude_dtypes, check_op_failures))
      if op.inplace_variant:
        variants.append((
            OpVariant.INPLACE,
            exclude_inplace_dtypes,
            check_inplace_op_failures,
        ))
      out_casting_tests = _out_casting_tests(
          op,
          dtypes_to_test,
          exclude_dtypes=tuple(raw_exclude_dtypes or ()),
          check_op_failures=check_op_failures,
          compute_grad=compute_grad,
          use_compiled=use_compiled,
          recording=recording,
      )

      for (
          variant,
          variant_exclude_dtypes,
          variant_check_op_failures,
      ) in variants:
        print(
            f"Testing {_VARIANT_CALL_FORMAT[variant].format(op_name)}.",
            flush=True,
        )
        for dtype in dtypes_to_test:
          if _should_skip_dtype(dtype, exclude_dtypes=variant_exclude_dtypes):
            continue
          test = functools.partial(
              self._test_torch_tpu_vs_golden,
              op,
              dtype,
              variant,
              compute_grad=compute_grad,
              use_compiled=use_compiled,
              check_value=check_value,
              check_dtype=check_dtype,
              check_device=check_device,
              skip_output_indices=skip_output_indices,
              skip_if=skip_if,
          )
          test(
              check_op_failures=variant_check_op_failures,
              max_samples_per_op_dtype=max_samples_per_op_dtype,
          )
          if variant != OpVariant.OUT:
            continue
          for out_dtype, cast_check_op_failures in out_casting_tests.get(
              dtype, ()
          ):
            test(
                check_op_failures=cast_check_op_failures,
                max_samples_per_op_dtype=1,
                out_dtype_override=out_dtype,
            )

    _run_and_print_exception(_test_op_variants)

    if _CHECK_SYMBOLIC_SHAPES.value:
      self._assert_known_symbolic_shapes_failure_seen(
          op_name, variant_test_name
      )

    # We have available dtypes to check and not intentionally excluded, but every sample disappeared, which is not expected.
    if _torch_tpu_vs_gpu_mode() and not is_compiled_mode():
      if self._subtests_run == 0:
        self.fail(
            f"do_test_op({op_name!r}) executed 0 subtests for "
            f"golden_device={self.golden_device_name()}. "
            f"Check exclude_dtypes ({exclude_dtypes}) vs candidate dtypes."
        )


class TorchTpuVsCpuTestBase(TorchTpuTestBase):
  """Base class for tests that compare TorchTPU results with CPU results.

  These tests always compute the golden results on CPU, regardless of
  --test_mode, since they have no recorded GPU goldens.
  """

  def setUp(self) -> None:
    super().setUp()
    self.golden_device = torch.device("cpu")

  def golden_device_name(self) -> str:
    return "CPU"


def _golden_file_prefix() -> str:
  """Returns the prefix for the golden file name."""
  if is_compiled_mode():
    return (
        "ops_test_gpu_golden_grad_compiled"
        if _COMPUTE_GRAD.value
        else "ops_test_gpu_golden_compiled"
    )
  return (
      "ops_test_gpu_golden_grad"
      if _COMPUTE_GRAD.value
      else "ops_test_gpu_golden"
  )


def _golden_file_basename(shard: int) -> str:
  """Returns the golden file name for `shard` on the current GPU."""
  if golden_target_gpu_arch() == GPU_SM90:
    return f"{_golden_file_prefix()}{_GPU_GOLDEN_HOPPER_SUFFIX}{shard}.gz"
  return f"{_golden_file_prefix()}{shard}.gz"


def _save_golden_file() -> None:
  """Saves the golden data to the undeclared outputs directory on Forge.

  The data is saved in a gzipped binary format. We don't use the .npz format
  because it relies on pickle, which is insecure and not portable. See
  go/nopickle.

  Instead, we:

  1. use safetensors to serialize the tensors in the golden data to a compact
     binary format,
  2. convert the transformed data into a plistlib-compatible pytree,
  3. use plistlib to serialize the final data to a compact binary file, which
     is then gzipped to further reduce its size.
  """

  golden_file = os.path.join(
      os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"),
      _golden_file_basename(_get_test_shard()),
  )
  print(
      f"Dumping the collected input/output pairs for each op to {golden_file}",
      flush=True,
  )

  encoded_data = _GOLDEN_GPU_DATA.to_plistlib_pytree()
  try:
    # pytype thinks `plistlib.FMT_BINARY` is `int` for whatever reason
    bin_data = plistlib.dumps(
        encoded_data, fmt=typing.cast(plistlib.PlistFormat, plistlib.FMT_BINARY)
    )
  except Exception as e:
    print(
        f"Failed to dump golden data: {e}, where:\n{encoded_data=}",
        flush=True,
    )
    raise
  with gzip.open(golden_file, "wb") as f:
    f.write(bin_data)


def _golden_keys(encoded_data: _pytree.PyTree) -> set[_GoldenKey]:
  """Returns the golden keys `encoded_data` holds samples for."""
  return {
      (test_case_name, variant_name, dtype_name)
      for test_case_name, variant_to_dt in encoded_data.items()
      for variant_name, dt_to_encoded_samples in variant_to_dt.items()
      for dtype_name in dt_to_encoded_samples
  }


def _check_no_cross_arch_overlap(
    filename: str,
    encoded_data: _pytree.PyTree,
    hopper_sources: Mapping[_GoldenKey, str],
) -> None:
  """Fails if `encoded_data` holds a golden key a Hopper file also holds."""
  overlap = sorted(_golden_keys(encoded_data) & set(hopper_sources))
  if not overlap:
    return
  test_case_name, variant_name, dtype_name = overlap[0]
  raise ValueError(
      f"{filename} and {hopper_sources[overlap[0]]} both hold goldens for"
      f" {len(overlap)} key(s), including {test_case_name} (variant"
      f" {variant_name}, dtype {dtype_name}), so it is ambiguous which one"
      " applies. Each test method must record its goldens on exactly one GPU"
      f" architecture; check whether {test_case_name} is tagged with"
      " @op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)."
  )


def _load_golden_files() -> None:
  """Loads the pinned Blackwell and Hopper goldens into _GOLDEN_GPU_DATA."""
  prefix = _golden_file_prefix()
  candidate_dirs = tests_test_utils.goldens_candidate_local_dirs(
      base_dir=_GOLDEN_DATA_BASE_DIR.value
  )
  if tests_test_utils.find_per_test_case_goldens_dir(
      prefix + _GPU_GOLDEN_PER_TEST_CASE_SUFFIX, candidate_dirs
  ):
    # Per-test-case golden artifacts are present, so skip eager monolithic
    # loading; GoldenGpuData.get_samples() lazily loads each test case's
    # goldens on demand.
    return
  # TODO(b/576014545): Remove monolithic fallback once we add per-test-case
  # loading in OSS.
  goldens = tests_test_utils.load_golden_files(
      prefix=prefix, candidate_dirs=candidate_dirs
  )
  hopper_prefix = f"{prefix}{_GPU_GOLDEN_HOPPER_SUFFIX}"
  hopper = [g for g in goldens if g.filename.startswith(hopper_prefix)]
  base = [g for g in goldens if not g.filename.startswith(hopper_prefix)]
  hopper_sources = {
      key: golden.filename
      for golden in hopper
      for key in _golden_keys(golden.tree)
  }
  _GOLDEN_GPU_DATA.clear()
  for golden in base:
    _check_no_cross_arch_overlap(golden.filename, golden.tree, hopper_sources)
  for golden in goldens:
    _GOLDEN_GPU_DATA.merge_plistlib_pytree(golden.tree)


def _record_gpu_dtype_exclusions(
    op_name: str,
    *,
    exclude_dtypes: Iterable[torch.dtype] | None,
    exclude_inplace_dtypes: Iterable[torch.dtype] | None,
) -> None:
  """Records a do_test_op call's GPU dtype exclusions in _DTYPE_EXCLUSIONS.

  Intersects new GPU exclusions with previously recorded exclusions for the
  given
  op across multiple call sites.

  Args:
    op_name: The name of the op being recorded.
    exclude_dtypes: Out-of-place dtype exclusions for the op call site.
    exclude_inplace_dtypes: In-place dtype exclusions for the op call site.
  """
  for kind, exclusions in (
      ("exclude_dtypes", exclude_dtypes),
      ("exclude_inplace_dtypes", exclude_inplace_dtypes),
  ):
    dtypes = {str(dtype) for dtype in exclusions or ()}
    op_exclusions = _DTYPE_EXCLUSIONS.setdefault(op_name, {})
    if kind not in op_exclusions:
      op_exclusions[kind] = dtypes
    else:
      op_exclusions[kind] &= dtypes


def _build_dtype_exclusions_yaml() -> str:
  """Builds a YAML document for collected GPU dtype exclusions."""
  tier_of_dtype_name = {
      str(dtype): tier for dtype, tier in TIER_OF_DTYPE.items()
  }

  # The dtypes of each exclusion kind are sorted by their tier in TIER_OF_DTYPE
  # first and then alphabetically. In this way, the most important dtypes show up at the top of YAML.
  def sort_key(dtype_name: str) -> tuple[int, str]:
    return (tier_of_dtype_name.get(dtype_name, sys.maxsize), dtype_name)

  data = {
      op_name: {
          kind: sorted(dtypes, key=sort_key)
          for kind, dtypes in kinds.items()
          if dtypes
      }
      for op_name, kinds in _DTYPE_EXCLUSIONS.items()
      if any(kinds.values())
  }
  return yaml.safe_dump(data, sort_keys=True, default_flow_style=False)


# YAML markers to facilitate parsing collected op dtype exclusions from test
# stdout output.
DTYPE_EXCLUSIONS_YAML_BEGIN: Final[str] = "BEGIN_DTYPE_EXCLUSIONS_YAML"
DTYPE_EXCLUSIONS_YAML_END: Final[str] = "END_DTYPE_EXCLUSIONS_YAML"


def _print_dtype_exclusions() -> None:
  """Prints collected GPU dtype exclusions to stdout in YAML format.

  The YAML is delimited by marker lines so that analysis tools can easily
  extract it from the test output.
  """
  print(DTYPE_EXCLUSIONS_YAML_BEGIN, flush=True)
  # `_build_dtype_exclusions_yaml` already ends with a newline, so suppress
  # print function's own newline to avoid emitting a trailing blank line.
  print(_build_dtype_exclusions_yaml(), end="", flush=True)
  print(DTYPE_EXCLUSIONS_YAML_END, flush=True)


def float_random_perm(num_elem: int, dtype: torch.dtype) -> torch.Tensor | None:
  """Generates a random permutation of unique numbers.

  Args:
    num_elem: The number of elements in the tensor.
    dtype: The dtype of the tensor. Must be a floating-point type.

  Returns:
    A 1-D tensor whose elements are a random permutation of unique numbers.
    Or None if num_elem is larger than the maximum number of unique values
    representable in the given dtype.
  """

  # A floating-point number of N bits (1 sign bit, E exponent bits, M mantissa
  # bits, where N = 1 + E + M) has this bit pattern:
  #   < sign> <exponent> <mantissa>
  #   <1 bit> < E bits > < M bits >
  #
  # - If all exponent bits are 1, the number is either infinity or NaN.
  # - If the exponent bits are 0111...1, the number is in the range [1.0, 2.0).
  #   An implicit 1. is assumed to be present in the mantissa, so the number is
  #   1.mmmm....
  #
  # To guarantee uniqueness, we map an integer in [0, 2^M) to the range
  # [1.0, 2.0) by using the integer as the mantissa and setting the exponent
  # bits to 0111...1 (2**(E - 1) - 1).

  # 1. Configuration mapping: maps a floating-point dtype to an integer dtype
  #    of the same size and the number of mantissa bits.
  config = {
      torch.float16: (torch.int16, 10),
      torch.bfloat16: (torch.int16, 7),
      torch.float32: (torch.int32, 23),
      torch.float64: (torch.int64, 52),
  }

  if dtype not in config:
    raise ValueError(f"Unsupported dtype: {dtype}")

  int_type, mantissa_bits = config[dtype]
  max_unique = 2**mantissa_bits

  # 2. Uniqueness Guarantee Check
  if num_elem > max_unique:
    return None

  exp_bits = dtype.itemsize * 8 - mantissa_bits - 1
  exp_pattern = (2 ** (exp_bits - 1) - 1) << mantissa_bits

  # 3. Generate unique mantissa patterns.
  # randperm(n) produces unique values from 0 to n-1.
  mantissas = torch.randperm(num_elem, dtype=int_type)

  # 4. Bitwise OR to inject the exponent and sign bits.
  # This places the unique values in the [1.0, 2.0) range.
  float_bits = torch.bitwise_or(
      mantissas, torch.tensor(exp_pattern, dtype=int_type)
  )

  # 5. Bit-cast to the target floating point type
  return float_bits.view(dtype)


def set_up_test_module() -> None:
  """Sets up the entire test module."""

  if _gen_gpu_golden_mode() and not torch.cuda.is_available():
    print(
        "WARNING: the gen_gpu_golden test mode requires compiling the test with"
        " --config=cuda on the blaze command line. Please rerun the test with"
        " that flag.",
        flush=True,
    )
    # We return success here because when someone runs a test case/suite
    # in the Cider UI by clicking on the green "triangles" button, the test
    # will run in all three modes without --config=cuda, and we don't want the
    # test to always fail in this case.
    sys.exit(0)

  if _CHECK_SYMBOLIC_SHAPES.value and (
      not _torch_tpu_vs_gpu_mode()
      or is_compiled_mode()
      or _COMPUTE_GRAD.value
      or _CHECK_DYNAMISM_USING_SEED.value
  ):
    raise ValueError(
        "--check_symbolic_shapes reads the GPU golden inputs of the"
        " torch_tpu_vs_gpu test mode and checks gradients itself, so it can't"
        " be combined with another --test_mode, --use_compiled,"
        " --compute_grad or --check_dynamism_using_seed."
    )
  if _CHECK_SYMBOLIC_SHAPES.value:
    # Don't log every symbol and guard that the checks create.
    logging.getLogger(symbolic_shapes.__name__).setLevel(logging.WARNING)
    # Don't log the traceback of every exception that an op raises on
    # FakeTensors: the checks report them, and the logs can exceed the size of
    # the test output that CI shows.
    logging.getLogger(fake_tensor.__name__).setLevel(logging.CRITICAL)

  # Note that global APIs must be configured before seeding PyTorch RNGs on TPU
  # as it triggers compilation of the seed modules.
  #
  # The very first invocation of `CompilationCache::GetOrCompile` will determine
  # the globally shared default compilation context.
  if _torch_tpu_vs_gpu_mode() or _gen_gpu_golden_mode():
    if is_compiled_mode():
      torch.tpu.allow_excess_precision = True

  # Assert that `torch.get_default_dtype()` returns `torch.float` when `setUp`
  # and `tearDown` are called.
  TestCase._default_dtype_check_enabled = True  # pylint: disable=protected-access

  if _torch_tpu_vs_gpu_mode():
    print("Running in TorchTPU vs GPU mode.", flush=True)
    _load_golden_files()
    return

  if _perf_mode():
    if _ANALYZE.value:
      _analyze_perf_data()
      sys.exit(0)  # Don't run the tests.
    else:
      # Disable compilation cache so that we can measure the time
      # for compiling the same op repeatedly.
      torch_tpu_module = getattr(torch, "tpu")
      torch_tpu_module._set_allow_cache(False)  # pylint: disable=protected-access
      # Dump the SHLO module for each compilation.
      torch_tpu_module._set_dump_on_cache_miss(True)  # pylint: disable=protected-access
      # Run an op to warm up the XLA compiler, so that we can measure
      # the compilation time of ops fairly in the inividual tests.
      torch.ones(1, dtype=torch.float32, device="tpu").to("cpu")
      # Measure the ops on the same inputs as the accuracy tests.
      _load_golden_files()

  # The other modes read the recorded GPU golden inputs, so only golden
  # generation needs the sample generators below.
  if not _gen_gpu_golden_mode():
    return

  # Operators that sort or select values and can have non-deterministic
  # tie-breaking, leading to flaky gradient tests when inputs have duplicates.
  # We intercept their sample generators to add a small uniqueness ramp.
  ops_needing_unique_values = {
      "topk",
      "sort",
      "max",
      "min",
      "argmax",
      "argmin",
      "cummax",
      "cummin",
  }

  for op in _KNOWN_OPS:
    if op.name not in ops_needing_unique_values:
      continue
    else:

      def create_unique_samples(
          orig_sample_inputs: collections.abc.Callable[
              ..., collections.abc.Iterable[Any]
          ],
      ) -> collections.abc.Callable[..., collections.abc.Iterator[Any]]:
        def unique_samples(
            op_info: Any,
            device: str,
            dtype: torch.dtype,
            requires_grad: bool,
            **kwargs: Any,
        ) -> collections.abc.Iterator[Any]:
          for sample in orig_sample_inputs(
              op_info, device, dtype, requires_grad, **kwargs
          ):
            if (
                _COMPUTE_GRAD.value
                and isinstance(sample.input, torch.Tensor)
                and sample.input.is_floating_point()
            ):
              # Ignore the original input.
              # Create a random tensor with the same shape, dtype, and
              # requires_grad state as the original input.
              rand_tensor = float_random_perm(sample.input.numel(), dtype)
              if rand_tensor is None:
                continue

              rand_tensor = rand_tensor.reshape(sample.input.shape).to(device)
              sample.input = rand_tensor.requires_grad_(requires_grad)

            yield sample

        return unique_samples

      # Override the default generator with our tie-breaking generator
      # This handles the base op and its variants (like 'out')
      op.sample_inputs_func = create_unique_samples(op.sample_inputs_func)


def tear_down_test_module() -> None:
  """Tears down the entire test module."""

  if _gen_gpu_golden_mode():
    _save_golden_file()
  elif _perf_mode() and _UPDATE_PERF_DATA.value:
    _save_perf_data()
  elif _dump_dtype_exclusions_mode():
    _print_dtype_exclusions()
