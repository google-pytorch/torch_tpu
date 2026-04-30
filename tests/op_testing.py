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

import collections
from collections.abc import Callable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence
import copy
import enum
import functools
import gzip
import logging  # PYTHON_LOGGING_OK=Setting module log level for compile testing
import os
import pathlib
import plistlib
import random
import re
import statistics
import sys
import time
import traceback
import typing
from typing import Any, Final, IO
import unittest

from absl import flags
from absl.testing import absltest
from etils import epath
import safetensors.torch as st
import torch
from torch.testing._internal import autograd_function_db
from torch.testing._internal import common_dtype
from torch.testing._internal import common_methods_invocations
from torch.testing._internal import common_utils
from torch.testing._internal import custom_op_db
from torch.testing._internal import hop_db
from torch.testing._internal.opinfo import core
from torch.testing._internal.opinfo.definitions import _masked
from torch.testing._internal.opinfo.definitions import fft
from torch.testing._internal.opinfo.definitions import linalg
from torch.utils import _pytree
from torch_tpu import api
from torch_tpu._internal import compile as tt_compile
from torch_tpu._internal import compiler_options as compiler
from torch_tpu._internal.utils import utils

# In this file, we use the following naming convention for variables:
# - golden_*: a value for the device used for computing the golden results
#   (either CPU or GPU)
# - torch_tpu_*: a value for the TorchTPU device

TestCase = common_utils.TestCase
OpInfo = core.OpInfo  # pylint: disable=protected-access
SampleInput = core.SampleInput  # pylint: disable=protected-access
CheckValueMode = utils.CheckValueMode
Tolerance = utils.Tolerance

AccuracyOverrides = Mapping[str, Mapping[torch.dtype, Mapping[str, Tolerance]]]

FilterFn = Callable[[OpInfo, torch.Tensor, Sequence[Any]], str | None]
MarkDynamicFn = Callable[[int, OpInfo, torch.Tensor, Sequence[Any]], None]

# The seed for both Python and PyTorch RNGs. Set before the test program starts
# and before each test method starts.
_RANDOM_SEED: Final[int] = 0


def _seed_rngs(seed: int) -> None:
  """Seeds the Python and PyTorch RNGs with the given seed."""
  random.seed(seed)
  torch.manual_seed(seed)


class TestMode(enum.Enum):
  """Mode to run the test in."""

  TORCH_TPU_VS_CPU = "torch_tpu_vs_cpu"  # Compare TorchTPU to CPU results.
  TORCH_TPU_VS_GPU = (  # Compare TorchTPU results to recorded GPU results.
      "torch_tpu_vs_gpu"
  )
  GEN_GPU_GOLDEN = "gen_gpu_golden"  # Generate GPU golden results.
  PERF = "perf"  # Measure op performance.


_TEST_MODE: Final[flags.FlagHolder[TestMode]] = flags.DEFINE_enum_class(
    "test_mode", TestMode.TORCH_TPU_VS_CPU, TestMode, "Mode to run the test in."
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

_CHECK_DYNAMISM_USING_SEED: Final[flags.FlagHolder[int]] = flags.DEFINE_integer(
    "check_dynamism_using_seed",
    0,
    "If non-zero, marks a random dimension as bounded dynamic and runs the ops"
    " test, comparing bounded XLA result with PyTorch CPU result. Only picks"
    " dimensions with size >=2, and bounds for size+10. Allows for explicitly"
    " setting the seed value for reproducibility, or -1 for random seed.",
)

_OPT_LEVEL: Final[flags.FlagHolder[str]] = flags.DEFINE_string(
    "opt_level",
    "",
    "Optimization level to run the test with. Valid options are '' (default)ß,"
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


class OpVariant(enum.Enum):
  """Variant of an op to test."""

  BASE = "base"  # op(...)
  INPLACE = "inplace"  # op_(...)
  OUT = "out"  # op(..., out=...)


# Ops not included in the list of tested ops for pytorch.
_ADDITIONAL_TORCH_TPU_OPS: Final[Sequence[OpInfo]] = [
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
]

# Used in the gen_gpu_golden mode to collect the golden results for each op.
# The key is the op name, and the value is a dictionary from dtype to a list
# of input-output pairs.
#
# In the torch_tpu_vs_gpu mode, we will populate this with results read from the
# GPU golden file and then use it when comparing the TorchTPU results against
# the GPU results.
#
# We cannot use a defaultdict here because it contains a function object, which
# cannot be pickled.
_GOLDEN_GPU_DATA: MutableMapping[
    str,
    MutableMapping[torch.dtype, MutableSequence[tuple["OpInput", "OpOutput"]]],
] = {}

# The full list of known ops. We will test a subset of these.
_KNOWN_OPS: Final[Sequence[OpInfo]] = (
    _masked.op_db
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
    + typing.cast(list[OpInfo], _ADDITIONAL_TORCH_TPU_OPS)
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

NUMERIC_DTYPES: Final[Sequence[torch.dtype]] = (
    *COMPLEX_DTYPES,
    *FLOAT_DTYPES,
    *INTEGRAL_DTYPES,
)


@functools.lru_cache(maxsize=None)
def _dtypes_to_test() -> Sequence[torch.dtype]:
  """Returns the dtypes to test."""

  if "all" in _DTYPES.value:
    return NUMERIC_DTYPES
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

# Match a line in the perf result file. A perf result line looks like this:
#   test shard 3, sample 5: add float32 0.05ms
_PERF_RESULT_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^test shard (\d+), sample (\d+): (\S+) (\S+) (.*)ms"
)


class PerfResult:
  """Holds the performance result of running an op with a given input dtype."""

  # How many PerfResult instances have been created so far.
  num_instances: int = 0

  def __init__(
      self, op_name: str, dtype: torch.dtype, duration_sec: float
  ) -> None:
    # Sponge uses 1-based shard indices.
    self.test_shard = _get_test_shard() + 1  # pytype: disable=name-error
    self.sequence_number = PerfResult.num_instances
    PerfResult.num_instances += 1
    self.op_name = op_name
    self.dtype = dtype
    self.duration_sec = duration_sec

  def __str__(self) -> str:
    return (
        f"test shard {self.test_shard}, sample {self.sequence_number}:"
        f" {self.op_name} {_format_dtype(self.dtype)}"
        f" {1000*self.duration_sec:.02f}ms"
    )

  def __repr__(self) -> str:
    return self.__str__()

  def __eq__(self, other: "PerfResult") -> bool:
    return (
        self.op_name == other.op_name
        and self.dtype == other.dtype
        and self.duration_sec == other.duration_sec
    )

  def __hash__(self) -> int:
    return hash((self.op_name, self.dtype, self.duration_sec))

  def __lt__(self, other: "PerfResult") -> bool:
    return (self.duration_sec, self.op_name, _format_dtype(self.dtype)) < (
        other.duration_sec,
        other.op_name,
        _format_dtype(other.dtype),
    )

  # Defines how to parse the string representation of the perf result.
  @classmethod
  def parse(cls, s: str) -> "PerfResult":
    # A perf result line looks like this:
    #   test shard 3, sample 5: add float32 0.05ms
    m = _PERF_RESULT_LINE_RE.match(s)
    assert m, f"Invalid perf result line: {s}"
    test_shard, sample_number, op_name, dtype, duration_ms_str = m.groups()
    duration_ms = float(duration_ms_str)
    result = cls(op_name, _parse_dtype(dtype), duration_ms / 1000)
    result.test_shard = int(test_shard)
    result.sequence_number = int(sample_number)
    return result


_PERF_RESULTS: MutableSequence[PerfResult] = []


def _format_dtype(dtype: torch.dtype) -> str:
  """Prints the dtype without the "torch." prefix."""
  return str(dtype).removeprefix("torch.")


_DTYPE_NAME_TO_DTYPE: Final[Mapping[str, torch.dtype]] = {
    _format_dtype(dtype): dtype for dtype in NUMERIC_DTYPES
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

  # Find all the perf data files in the output directory.
  perf_data_files = output_dir.glob("perf_data*.txt")

  # Parse the perf data files.
  perf_results = []
  for file in perf_data_files:
    print(f"Parsing perf data file {file}", flush=True)
    perf_results.extend(
        [PerfResult.parse(line) for line in file.read_text().splitlines()]
    )

  def _analyze(dtype: torch.dtype) -> None:
    """Analyzes the perf data for a given dtype."""

    results = [r for r in perf_results if r.dtype == dtype]
    if not results:
      # The dtype is not tested.
      return
    num = len(results)
    print(80 * "-", flush=True)
    print(f"Perf stats from {num} samples for dtype {dtype}.", flush=True)
    durations = [r.duration_sec for r in results]
    sorted_durations_ms = sorted(1000 * d for d in durations)
    print(
        f"""\
Min: {min(results)}
Max: {max(results)}
Stddev: {statistics.stdev(sorted_durations_ms):.2f}ms
Mean: {sum(sorted_durations_ms)/num:.2f}ms
Median: {sorted_durations_ms[num//2]:.2f}ms
P90: {sorted_durations_ms[90*num//100]:.2f}ms
P95: {sorted_durations_ms[95*num//100]:.2f}ms
P99: {sorted_durations_ms[99*num//100]:.2f}ms""",
        flush=True,
    )

    # For each op, find the slowest sample. Print the slowest ops.
    op_to_results = collections.defaultdict(list)
    for r in results:
      op_to_results[r.op_name].append(r)
    op_to_slowest_result = {
        op: max(results, key=lambda r: r.duration_sec)
        for op, results in op_to_results.items()
    }
    sorted_result_desc = sorted(op_to_slowest_result.values(), reverse=True)
    print(f"Top {_TOP_N_SLOWEST_OPS} slowest ops:", flush=True)
    for r in sorted_result_desc[:_TOP_N_SLOWEST_OPS]:
      print(r, flush=True)
    print(flush=True)

  for dtype in NUMERIC_DTYPES:
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


def _make_tensors_empty(x: _pytree.PyTree) -> _pytree.PyTree:
  """Makes all torch.Tensors in x empty."""

  def transform_leaf(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
      return torch.empty_like(obj)
    return obj

  return _tensor_tree_map(transform_leaf, x)


def _gen_gpu_golden_mode() -> bool:
  """Returns true if the test is running in gen_gpu_golden mode."""
  return _TEST_MODE.value == TestMode.GEN_GPU_GOLDEN


def _torch_tpu_vs_cpu_mode() -> bool:
  """Returns true if the test is running in torch_tpu_vs_cpu mode."""
  return _TEST_MODE.value == TestMode.TORCH_TPU_VS_CPU


def _torch_tpu_vs_gpu_mode() -> bool:
  """Returns true if the test is running in torch_tpu_vs_gpu mode."""
  return _TEST_MODE.value == TestMode.TORCH_TPU_VS_GPU


def _perf_mode() -> bool:
  """Returns true if the test is running in perf mode."""
  return _TEST_MODE.value == TestMode.PERF


def is_compiled_mode() -> bool:
  """Returns true if --use_compiled is set."""
  return _USE_COMPILED.value


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
    - A complex number is converted to a {"$$real": ..., "$$imag": ...} dict.
    - An Exception is serialized to a {"$$exception": ""} dict.

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
      # st.save() doesn't handle non-contiguous tensors.
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
    if isinstance(x, Exception):
      # We don't care about the actual exception, but just need to know it's
      # an exception. Therefore we use a dummy value.
      return {"$$exception": ""}
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
    )

  return _pytree.tree_map(leaf_func, ptree, is_leaf=is_leaf)


def _from_plistlib_compatible(ptree: _pytree.PyTree) -> _pytree.PyTree:
  """Converts a plistlib-compatible PyTree to the original PyTree.

  In particular,

    - A byte string is deserialized to a torch.Tensor using st.load().
    - A {"$$tuple": ...} dict is deserialized to a tuple.
    - A {"$$dtype": ...} dict is deserialized to a torch.dtype.
    - A {"$$device": ...} dict is deserialized to a torch.device.
    - A {"$$memory_format": ...} dict is deserialized to a torch.memory_format.
    - A {"$$real": ..., "$$imag": ...} dict is deserialized to a complex number.
    - A {"$$exception": ""} dict is deserialized to an Exception.

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
      # The tensor is not complex-typed.
      return d[""]
    if isinstance(x, dict):
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
      if "$$real" in x and "$$imag" in x:
        return complex(x["$$real"], x["$$imag"])
      if "$$exception" in x:
        return Exception()
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
        f"Name: {self.name}\nInput: {self.input_value}\nArgs:"
        f" {self.args}\nKwargs: {self.kwargs}"
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

    if isinstance(output_value, Exception):
      self.output_value = output_value
      return

    cpu = torch.device("cpu")
    self.output_value = to(copy.deepcopy(output_value), cpu)

  def __str__(self) -> str:
    return self.__repr__()

  def __repr__(self) -> str:
    return f"Output: {self.output_value}"

  def to_plistlib_pytree(self) -> _pytree.PyTree:
    """Converts this object to a PyTree that plistlib can handle."""
    return _to_plistlib_compatible(self.output_value)

  @classmethod
  def from_plistlib_pytree(cls, ptree: _pytree.PyTree) -> "OpOutput":
    """Converts a PyTree to an OpOutput."""
    decoded_ptree = _from_plistlib_compatible(ptree)
    return cls(decoded_ptree)


def _add_golden_result(
    op_name: str,
    dtype: torch.dtype,
    op_input: OpInput,
    op_output: OpOutput,
) -> None:
  """Adds the golden result to _GOLDEN_GPU_DATA."""
  if op_name not in _GOLDEN_GPU_DATA:
    _GOLDEN_GPU_DATA[op_name] = {}
  dtype_to_pairs = _GOLDEN_GPU_DATA[op_name]
  if dtype not in dtype_to_pairs:
    dtype_to_pairs[dtype] = []
  pairs = dtype_to_pairs[dtype]
  pairs.append((op_input, op_output))


def _dtype_str(dtype: torch.dtype) -> str:
  """Returns a string representation of the dtype without the "torch." prefix."""
  return str(dtype).removeprefix("torch.")


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
  the _run_op(..., golden_input, api.tpu_device(), ...) call above
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
  return to(golden_input, api.tpu_device(), convert_tensors=False)


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
  except RuntimeError as e:
    fx_reproducer = e
  print("Reproducer for:", subtest_name, flush=True)
  print(fx_reproducer, flush=True)


def _run_and_print_exception(func: Callable[[], None]) -> None:
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

  if (
      has_empty(device_op_input.input_value)
      or has_empty(device_op_input.args)
      or has_empty(device_op_input.kwargs)
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
  ]

  key = op.name
  if variant == OpVariant.OUT:
    key += ".out"
  return key not in compiled_deny_list


class TorchTpuTestBase(TestCase):
  """Base class for TorchTPU tests."""

  dynamism_filter_fn: FilterFn
  dynamism_mark_dynamic_fn: MarkDynamicFn

  tpu_cpu_accuracy_overrides: AccuracyOverrides
  tpu_gpu_accuracy_overrides: AccuracyOverrides
  grad_accuracy_overrides: AccuracyOverrides

  def setUp(self) -> None:
    super().setUp()
    # Show long diffs in assertEqual.
    self.maxDiff = None  # pylint: disable=invalid-name

    # Reseed the RNGs to prevent test methods from interfering with each other.
    _seed_rngs(_RANDOM_SEED)

    # Device used for computing the golden results, or None if the golden
    # results are read from the golden file.
    self.golden_device = {
        # go/keep-sorted start
        TestMode.GEN_GPU_GOLDEN: torch.device("cuda"),
        TestMode.PERF: torch.device("cpu"),
        TestMode.TORCH_TPU_VS_CPU: torch.device("cpu"),
        TestMode.TORCH_TPU_VS_GPU: None,
        # go/keep-sorted end
    }[_TEST_MODE.value]

    if _TEST_MODE.value in (TestMode.TORCH_TPU_VS_CPU, TestMode.PERF):
      self.golden_device_type = "cpu"
    elif _TEST_MODE.value in (
        TestMode.GEN_GPU_GOLDEN,
        TestMode.TORCH_TPU_VS_GPU,
    ):
      self.golden_device_type = "gpu"
    else:
      self.fail(f"Unknown test mode: {_TEST_MODE.value}")

    if not _gen_gpu_golden_mode():
      api.tpu_device()  # Initialize the TorchTPU device.

  def set_accuracy_overrides(
      self,
      *,
      tpu_cpu_overrides: Mapping[
          str, Mapping[torch.dtype, Mapping[str, Tolerance]]
      ],
      tpu_gpu_overrides: Mapping[
          str, Mapping[torch.dtype, Mapping[str, Tolerance]]
      ],
      grad_overrides: Mapping[
          str, Mapping[torch.dtype, Mapping[str, Tolerance]]
      ],
  ) -> None:
    """Sets the accuracy overrides for the test.

    Args:
      tpu_cpu_overrides: Accuracy overrides for TorchTPU vs CPU.
      tpu_gpu_overrides: Accuracy overrides for TorchTPU vs GPU.
      grad_overrides: Accuracy overrides for gradients.

    To be called by a subclass's setUp() method.
    """
    self.tpu_cpu_accuracy_overrides = tpu_cpu_overrides
    self.tpu_gpu_accuracy_overrides = tpu_gpu_overrides
    self.grad_accuracy_overrides = grad_overrides

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

  def skip_unless_torch_tpu_vs_cpu(self) -> None:
    """Skips the test unless it is running in the TorchTPU vs CPU mode."""
    if not _torch_tpu_vs_cpu_mode():
      self.skipTest("This test is only relevant for the TorchTPU vs CPU mode.")

  def golden_device_name(self) -> str:
    """Returns the name of the golden device for logging purposes."""
    return self.golden_device_type.upper()

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
      check_value: CheckValueMode = CheckValueMode.STRICT,
      check_dtype: bool = True,
      rtol: float | None = None,
      atol: Tolerance | None = None,
  ) -> None:
    """Asserts that the TPU result is close to the golden result.

    This function does nothing in the perf mode, as we only want to
    measure the performance of the op there.

    Args:
      golden_result: The golden result.
      torch_tpu_result: The TPU result.
      check_value: The mode for checking the values.
      check_dtype: Check if the dtypes are the same.
      rtol: The relative tolerance for checking the values.
      atol: The absolute tolerance for checking the values.
    """

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
        )
      return

    if isinstance(golden_result, bool):
      self.assertEqual(golden_result, torch_tpu_result)
      return

    utils.assert_close(
        actual=torch_tpu_result,
        expected=golden_result,
        rtol=rtol,
        atol=atol,
        check_value=check_value,
        check_dtype=check_dtype,
        preamble="Comparing TorchTPU to golden result",
    )

  # TODO: Move the failure-handling logic to util.assert_close to avoid
  # duplication.
  def assert_close_tpu_vs_cpu(
      self,
      tensor_from_device: Callable[[torch.device], torch.Tensor],
      *,
      check_value: CheckValueMode = CheckValueMode.STRICT,
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
      check_value: The mode for checking the values.
      check_dtype: Check if the dtypes are the same.
      check_exception_type: If True, check that the exception type is the same
        on both devices.
      allow_failure: If True, allow the CPU to throw an exception.
      rtol: The relative tolerance for checking the values.
      atol: The absolute tolerance for checking the values.
    """

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
      torch_tpu_result = to(tensor_from_device(api.tpu_device()), "cpu")
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

  def _get_golden_input_output_pairs(
      self,
      op: OpInfo,
      dtype: torch.dtype,
      variant: OpVariant,
      *,
      max_samples: int | None,
      compute_grad: bool = False,
      use_compiled: bool = False,
  ) -> Sequence[tuple[OpInput, OpOutput]]:
    """Returns a list of (input, output) pairs for the op.

    In the gen_gpu_golden and torch_tpu_vs_cpu modes, the inputs are generated
    randomly using op.sample_inputs(), and the outputs are computed on GPU
    and CPU respectively.

    In the torch_tpu_vs_gpu mode, the input-output pairs are read from the GPU
    golden file.

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
    """

    op_name = _op_name_for_logging(op, variant)
    print(
        f">>> Getting golden results for {op_name}() with dtype {dtype} ...",
        flush=True,
    )

    if _torch_tpu_vs_gpu_mode():
      return _GOLDEN_GPU_DATA.get(op_name, {}).get(dtype, [])

    # Generate sample inputs on the golden device.
    golden_samples = list(
        op.sample_inputs(self.golden_device, dtype, requires_grad=False)
    )
    if max_samples is None:
      max_samples = _MAX_SAMPLES_PER_OP_DTYPE.value
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
          compute_grad=compute_grad,
          use_compiled=use_compiled,
          device=self.golden_device,
          # When generating golden results, we should trust the op's output
          # to be correct and not check the device. E.g. even in the
          # gen_gpu_golden mode, torch.arange(5) should return a tensor on
          # CPU instead of GPU.
          check_device=False,
          # No need to mark inputs as dynamic when computing golden results.
          check_dynamism=False,
      )
      golden_output = OpOutput(golden_result)
      pairs.append((golden_input, golden_output))
      if _gen_gpu_golden_mode():
        _add_golden_result(op_name, dtype, golden_input, golden_output)
    return pairs

  def _run_op(
      self,
      op: OpInfo,
      variant: OpVariant,
      dtype: torch.dtype,
      op_input: OpInput,
      device: torch.device,
      *,
      check_device: bool,
      check_dynamism: bool,
      out: Any = None,
      compute_grad: bool = False,
      use_compiled: bool = False,
      measure_perf: bool = True,
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
      check_device: If True, check that the result tensors are on the given
        device.
      check_dynamism: If True and the test is running in dynamism checking mode
        (i.e. the --check_dynamism_using_seed flag is set), check that the op
        supports dynamism.
      out: If variant is OUT, this can be either the out argument to be used
        when running the op or None if this function should create the out
        argument itself. Otherwise this must be None.
      compute_grad: If True, compute the gradient of the op with respect to the
        input instead of the op outputs.
      use_compiled: If True, use torch.compile to compile the op before running.
      measure_perf: If True, measure the performance of the op in the perf mode.

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
    if variant == OpVariant.OUT and out is None:
      # We need to create the out tensor ourselves.
      #
      # First, run the base variant to get the output's dtype and size.
      base_result = self._run_op(
          op=op,
          variant=OpVariant.BASE,
          dtype=dtype,
          op_input=op_input,
          compute_grad=False,
          use_compiled=use_compiled,
          device=device,
          check_device=check_device,
          check_dynamism=check_dynamism,
          # This _run_op() call is for preparing the test data, not part of
          # the test itself, so we don't want to measure its performance
          # (otherwise it would skew the performance results).
          measure_perf=False,
      )
      if isinstance(base_result, Exception):
        # The base variant failed, so we cannot generate the out tensor.
        return base_result
      # Next, create the out argument with the correct dtype and size.
      # Some ops expect the out argument to be a tuple as opposed to a
      # single tensor (e.g. sort()), so we need _make_tensors_empty()
      # instead of torch.empty_like() here.
      out = to(_make_tensors_empty(base_result), device)

    op_func = op.inplace_variant if variant == OpVariant.INPLACE else op

    # Clone the sample to prevent the op from mutating it.
    # We must deepcopy op_input *before* setting its device to the given device,
    # because deepcopying a TPU tensor is not implemented yet.
    device_op_input = to(copy.deepcopy(op_input), device)

    if variant == OpVariant.OUT:
      device_op_input.kwargs["out"] = out

    input_value = device_op_input.input_value

    if use_compiled:
      if not _compiled_supports_op(op, device_op_input, variant):
        self.skipTest(f"TpuBackend does not support {op_name} currently.")
      logging.getLogger(tt_compile.TpuBackend.__module__).setLevel(
          logging.DEBUG
      )
      backend = "tpu" if str(device) == "tpu" else "inductor"
      print(f"Compiling {op_name} for device {device} ...", flush=True)
      op_func = torch.compile(op_func, dynamic=False, backend=backend)

    if _CHECK_DYNAMISM_USING_SEED.value and device == api.tpu_device():
      if not check_dynamism:
        self.skipTest(f"Dynamism check is explicitly disabled for {op_name}.")

      print(
          f">>> Marking dynamism for {op_name} with dtype {dtype} ...",
          flush=True,
      )

      # Verify op is supported with dynamism.
      skip = self.dynamism_filter_fn(op, op_input.input_value, op_input.args)
      if skip is not None:
        self.skipTest(skip)

      # Use a seed value to enable reproducing a failure deterministically.
      seed = _CHECK_DYNAMISM_USING_SEED.value
      seed = seed if seed > 0 else (time.time_ns() % 100000)
      print(f">>>> Reproduce using `--check_dynamism_using_seed={seed}`")
      self.dynamism_mark_dynamic_fn(
          seed, op, device_op_input.input_value, device_op_input.args
      )

    if compute_grad:

      def set_requires_grad(t: Any) -> Any:
        if (
            isinstance(t, torch.Tensor)
            and (t.is_floating_point() or t.is_complex())
            and t.numel() > 0
        ):
          return t.requires_grad_(True)
        return t

      # Set requires_grad=True for all floating-point and complex tensors in
      # the input. We only support testing gradients when input_value is a
      # tensor or a (possibly nested) list/tuple of tensors.
      input_value = _tensor_tree_map(set_requires_grad, input_value)
      device_op_input.input_value = input_value

      def is_differentiable(t: Any) -> bool:
        return isinstance(t, torch.Tensor) and t.requires_grad and t.numel() > 0

      # Autograd doesn't work with inplace ops and out variants, as they
      # erase the input tensor.
      if (
          variant == OpVariant.INPLACE
          or variant == OpVariant.OUT
          or not any(
              _pytree.tree_leaves(
                  _tensor_tree_map(is_differentiable, input_value)
              )
          )
      ):
        return _dummy_grad(device)

      # TODO: fix the bug where computing the gradient of a 0-sized input tensor
      # fails with an exception "Function SumBackward0 returned an invalid
      # gradient at index 0 - expected device tpu:0 but got cpu".
      if isinstance(input_value, torch.Tensor) and input_value.numel() == 0:
        return _dummy_grad(device)

    try:
      start_time = time.time()
      result = op_func(
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
        losses = [
            t.sum()
            for t in _pytree.tree_leaves(result)
            if isinstance(t, torch.Tensor)
            and (t.is_floating_point() or t.is_complex())
            and t.numel() > 0
        ]

        if losses:
          torch.sum(torch.stack(losses)).backward()

          def get_grad(t: Any) -> torch.Tensor | None:
            if isinstance(t, torch.Tensor) and t.grad is not None:
              return t.grad.clone()
            return None

          result = _tensor_tree_map(get_grad, input_value)
        else:
          result = _dummy_grad(device)

      if check_device:
        # To prevent bugs in ops and the test itself, we check that
        # tensors in the result are indeed on the expected device.
        def assert_on_device(obj: Any) -> None:
          if isinstance(obj, torch.Tensor):
            self.assert_devices_equivalent(obj.device, device)

        try:
          _tensor_tree_map(assert_on_device, result)
        except AssertionError as e:
          self.fail(f"Expected result to be on {device}, but got {result}: {e}")

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
      # Don't measure performance for the golden device. We are only interested
      # in the TPU performance.
      if _perf_mode() and measure_perf and device != self.golden_device:
        # We must measure the time after the .to("cpu") call because that's the
        # time when the compilation and execution of the op is guaranteed to be
        # done.
        _add_perf_result(PerfResult(op_name, dtype, time.time() - start_time))
    except Exception as e:  # pylint: disable=broad-except
      # The op raised an exception.
      result = e
    return result

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

  def _assert_structure_consistency(
      self, *, golden_result: Any, torch_tpu_result: Any
  ) -> None:
    """Asserts that golden_result and torch_tpu_result are both tensors or both tuples of the same size."""
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
      check_value: CheckValueMode | Iterable[CheckValueMode],
      check_dtype: bool,
      skip_output_indices: Sequence[int],
      accuracy_override: Mapping[str, Tolerance],
  ) -> None:
    """Asserts that golden_result and torch_tpu_result, as tuples, are close."""

    golden_result_tuple = _to_tuple(golden_result)
    torch_tpu_result_tuple = _to_tuple(torch_tpu_result)
    if isinstance(check_value, CheckValueMode):
      check_value = [check_value] * len(torch_tpu_result_tuple)

    for i, golden_result_i in enumerate(golden_result_tuple):
      if i in skip_output_indices:
        continue
      try:
        self.assert_close(
            golden_result=golden_result_i,
            torch_tpu_result=torch_tpu_result_tuple[i],
            check_value=check_value[i],
            check_dtype=check_dtype,
            **accuracy_override,
        )
      except AssertionError as e:
        e.add_note(
            f"{torch_tpu_printable_input}\n"
            + (
                f"Mismatching result item: {i}\n"
                if isinstance(torch_tpu_result, tuple)
                else ""
            )
            + f"{self.golden_device_name()} result: {golden_result}\n"
            f"TorchTPU result: {torch_tpu_result}",
        )
        raise

  def _sub_test(
      self,
      subtest_name: str,
      op: OpInfo,
      variant: OpVariant,
      dtype: torch.dtype,
      *,
      golden_input: OpInput,
      golden_result: Any,
      check_device: bool,
      check_dynamism: bool,
      check_value: CheckValueMode | Iterable[CheckValueMode],
      check_dtype: bool,
      skip_output_indices: Sequence[int],
      compute_grad: bool,
      use_compiled: bool,
  ) -> None:
    op_name = _op_name_for_logging(op, variant)
    print(f">>>> Testing {subtest_name} ...", flush=True)
    if compute_grad:
      accuracy_overrides = self.grad_accuracy_overrides
    elif _torch_tpu_vs_cpu_mode():
      accuracy_overrides = self.tpu_cpu_accuracy_overrides
    else:
      accuracy_overrides = self.tpu_gpu_accuracy_overrides

    # Default to the accuracy override for the base op name for _foreach_ ops.
    accuracy_override = accuracy_overrides.get(op.name, {}).get(dtype, {})
    if not accuracy_override and op.name.startswith("_foreach_"):
      base_op_name = op.name[9:]
      accuracy_override = accuracy_overrides.get(base_op_name, {}).get(
          dtype, {}
      )

    torch_tpu_printable_input = _to_torch_tpu_printable_input(golden_input)
    if _PRINT_OP_INPUTS.value > 0:
      # TODO b/486861095
      if op.name != "normal":
        print_reproducer(subtest_name, op, golden_input, variant)
      print_op_input(torch_tpu_printable_input, data=_PRINT_OP_INPUTS.value > 1)

    # TODO(wan): measure compilation time and execution time separately
    # in the perf mode.
    opt_level = _OPT_LEVEL.value
    if opt_level:
      compiler_options = {"xla_optimization_level": opt_level}
    else:
      compiler_options = {}
    with compiler.custom_compiler_options(compiler_options):
      torch_tpu_result = self._run_op(
          op=op,
          variant=variant,
          dtype=dtype,
          op_input=golden_input,
          compute_grad=compute_grad,
          use_compiled=use_compiled,
          device=api.tpu_device(),
          check_device=check_device,
          check_dynamism=check_dynamism,
      )
    golden_thrown = isinstance(golden_result, Exception)
    torch_tpu_thrown = isinstance(torch_tpu_result, Exception)

    self._assert_failure_consistency(
        golden_result=golden_result,
        torch_tpu_result=torch_tpu_result,
        op_description=f"{op_name}() with dtype {dtype}",
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

  def _test_torch_tpu_vs_golden(
      self,
      op: OpInfo,
      dtype: torch.dtype,
      variant: OpVariant,
      *,
      compute_grad: bool,
      use_compiled: bool,
      check_value: CheckValueMode | Iterable[CheckValueMode],
      check_dtype: bool,
      check_device: bool,
      check_dynamism: bool,
      check_op_failures: bool,
      skip_output_indices: Sequence[int],
      skip_if: Callable[[str, OpVariant, OpInput], bool] | None,
      max_samples_per_op_dtype: int | None,
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
      check_dynamism: If True and the test is running in dynamism checking mode,
        i.e. --check_dynamsim_using_seed is set, check that the op supports
        dynamism.
      check_op_failures: Whether to check that the op fails on TorchTPU when it
        fails on the golden device.
      skip_output_indices: The indices of the output to skip checking.
      skip_if: A function that returns True if a test case should be skipped,
        given the golden device, the op variant, and the op input.
      max_samples_per_op_dtype: The maximum number of samples to test for each
        (op variant, dtype) combination. If None, the number is determined by
        the --max_samples_per_op_dtype flag.
    """

    if dtype not in _dtypes_to_test():
      print(
          f"Skipping test for dtype {dtype} as it is not in --dtypes.",
          flush=True,
      )
      return

    if variant == OpVariant.INPLACE and not op.inplace_variant:
      print(f"Op {op.name} does not have an inplace variant.", flush=True)
      return

    op_name = _op_name_for_logging(op, variant)
    print(f">>> Testing {op_name}() with dtype {dtype} ...", flush=True)

    golden_pairs = self._get_golden_input_output_pairs(
        op=op,
        dtype=dtype,
        variant=variant,
        compute_grad=compute_grad,
        use_compiled=use_compiled,
        max_samples=max_samples_per_op_dtype,
    )
    for i, [golden_input, golden_output] in enumerate(golden_pairs):
      golden_result = golden_output.output_value
      golden_thrown = isinstance(golden_result, Exception)
      if golden_thrown and not check_op_failures:
        print(
            f"Skipping test for {op_name}() with dtype {dtype} on TorchTPU as"
            " it failed on"
            f" {self.golden_device_name()}.\n{golden_input}\n"
            f"Error: {golden_result}\n",
            flush=True,
        )
        continue

      if _gen_gpu_golden_mode():
        continue

      if skip_if and skip_if(self.golden_device_type, variant, golden_input):
        print(
            f"Skipping test sample for {op_name}() with dtype {dtype} on"
            " TorchTPU as filtered by the skip_if function.\n"
            f"{golden_input}",
            flush=True,
        )
        continue

      # Test each sample in a subtest s.t. one sample's failure doesn't prevent
      # other samples from being tested.
      # Construct a meaningful and unique subtest name.
      subtest_name = f"{op_name}_{_dtype_str(dtype)}_"
      subtest_name += golden_input.name if golden_input.name else f"sample{i}"

      with self.subTest(subtest_name):
        _run_and_print_exception(
            functools.partial(
                self._sub_test,
                subtest_name,
                op,
                variant,
                dtype,
                golden_input=golden_input,
                golden_result=golden_result,
                check_device=check_device,
                check_dynamism=check_dynamism,
                check_value=check_value,
                check_dtype=check_dtype,
                skip_output_indices=skip_output_indices,
                compute_grad=compute_grad,
                use_compiled=use_compiled,
            )
        )

  def _resolve_exclude_dtypes(
      self,
      exclude_dtypes: (
          Iterable[torch.dtype] | Mapping[str, Iterable[torch.dtype]] | None
      ),
  ) -> Sequence[torch.dtype]:
    """Resolves the exclude_dtypes argument."""
    exclude_dtypes = exclude_dtypes or ()
    if isinstance(exclude_dtypes, dict):
      # Ensure that only "cpu" and "gpu" are used as keys.
      excessive_keys = set(exclude_dtypes.keys()) - {"cpu", "gpu"}
      if excessive_keys:
        raise ValueError(
            "Expected only 'cpu' and 'gpu' keys in exclude_dtypes, got keys"
            f" {excessive_keys}"
        )

      exclude_dtypes = exclude_dtypes.get(self.golden_device_type, ())
    return tuple(exclude_dtypes)

  def do_test_op(
      self,
      op_name: str,
      *,
      exclude_dtypes: (
          Iterable[torch.dtype] | Mapping[str, Iterable[torch.dtype]] | None
      ) = None,
      exclude_inplace_dtypes: (
          Iterable[torch.dtype] | Mapping[str, Iterable[torch.dtype]] | None
      ) = None,
      check_out_variant: bool = True,
      check_grad: bool = True,
      check_device: bool = True,
      check_dynamism: bool = True,
      check_value: (
          CheckValueMode | Iterable[CheckValueMode]
      ) = CheckValueMode.STRICT,
      check_dtype: bool = True,
      check_op_failures: bool = True,
      check_inplace_op_failures: bool = True,
      skip_output_indices: Sequence[int] | None = None,
      skip_if: Callable[[str, OpVariant, OpInput], bool] | None = None,
      max_samples_per_op_dtype: int | None = None,
      variant_test_name: str | None = None,
  ) -> None:
    """Does the standard suite of testing for the given op.

    Args:
      op_name: The name of the op to test.
      exclude_dtypes: A list of input dtypes to exclude from testing, or a
        dictionary mapping device type ("cpu" or "gpu") to such a list (useful
        when different dtypes should be excluded for different golden devices).
        If None, no dtypes will be excluded.
      exclude_inplace_dtypes: Similar to exclude_dtypes, but for the inplace
        variant of the op.
      check_out_variant: Whether to check the out variant of the op.
      check_grad: Whether to check the gradient of the op.
      check_device: Whether to check that the op's result is on the correct
        device.
      check_dynamism: If True and the test is running in dynamism checking mode,
        i.e. --check_dynamsim_using_seed is set, check that the op supports
        dynamism.
      check_value: Mode for checking the values. If SKIP, only the output's
        dtype and shape will be checked. If an iterable is provided, it must
        have the same length as the op's output list, and the i-th element will
        determine whether to check the i-th output's values (this is useful for
        ops that return multiple outputs).
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
        given the golden device, the op variant, and the op input.
      max_samples_per_op_dtype: The maximum number of samples to test for a
        given (op_variant, dtype) combination. If None, the number is determined
        by the --max_samples_per_op_dtype flag.
      variant_test_name: The name of the variant to test, referring to the
        variant_test_name field of the OpInfo, when named variants are
        available. If None, the base variant set, filtering only on the op_name,
        will be tested.
    """

    exclude_dtypes = self._resolve_exclude_dtypes(exclude_dtypes)
    exclude_inplace_dtypes = self._resolve_exclude_dtypes(
        exclude_inplace_dtypes
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

      # Test the base variant.
      print(f"Testing {op_name}().", flush=True)
      for dtype in NUMERIC_DTYPES:
        if _should_skip_dtype(dtype, exclude_dtypes=exclude_dtypes):
          continue
        self._test_torch_tpu_vs_golden(
            op,
            dtype,
            OpVariant.BASE,
            compute_grad=compute_grad,
            use_compiled=use_compiled,
            check_value=check_value,
            check_dtype=check_dtype,
            check_device=check_device,
            check_dynamism=check_dynamism,
            check_op_failures=check_op_failures,
            skip_output_indices=skip_output_indices,
            skip_if=skip_if,
            max_samples_per_op_dtype=max_samples_per_op_dtype,
        )

      if check_out_variant:
        print(f"Testing {op_name}(out=...).", flush=True)
        for dtype in NUMERIC_DTYPES:
          if _should_skip_dtype(dtype, exclude_dtypes=exclude_dtypes):
            continue
          # TODO: cover more ways to specify the out arguments (e.g. correct
          # dtype but incorrect dimensions, correct dimensions but incorrect
          # dtype, both dtype and dimensions incorrect, incorrect structure
          # (e.g. tuple vs tensor for vice versa)).
          self._test_torch_tpu_vs_golden(
              op,
              dtype,
              OpVariant.OUT,
              compute_grad=compute_grad,
              use_compiled=use_compiled,
              check_value=check_value,
              check_dtype=check_dtype,
              check_device=check_device,
              check_dynamism=check_dynamism,
              check_op_failures=check_op_failures,
              skip_output_indices=skip_output_indices,
              skip_if=skip_if,
              max_samples_per_op_dtype=max_samples_per_op_dtype,
          )

      if not op.inplace_variant:
        return

      # Test the inplace variant.
      print(f"Testing {op_name}_().", flush=True)
      for dtype in NUMERIC_DTYPES:
        if _should_skip_dtype(dtype, exclude_dtypes=exclude_inplace_dtypes):
          continue
        self._test_torch_tpu_vs_golden(
            op,
            dtype,
            OpVariant.INPLACE,
            compute_grad=compute_grad,
            use_compiled=use_compiled,
            check_value=check_value,
            check_dtype=check_dtype,
            check_device=check_device,
            check_dynamism=check_dynamism,
            check_op_failures=check_inplace_op_failures,
            skip_output_indices=skip_output_indices,
            skip_if=skip_if,
            max_samples_per_op_dtype=max_samples_per_op_dtype,
        )

    _run_and_print_exception(_test_op_variants)


class TorchTpuVsCpuTestBase(TorchTpuTestBase):
  """Base class for tests that don't need to compare with GPU results."""

  def setUp(self) -> None:
    super().setUp()
    self.skip_unless_torch_tpu_vs_cpu()


def _golden_file_prefix() -> str:
  """Returns the prefix for the golden file name."""
  return (
      "ops_test_gpu_golden_compiled"
      if is_compiled_mode()
      else "ops_test_gpu_golden"
  )


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

  test_shard = _get_test_shard()
  golden_file = (
      os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
      + f"/{_golden_file_prefix()}{test_shard}.gz"
  )
  print(
      f"Dumping the collected input/output pairs for each op to {golden_file}",
      flush=True,
  )

  def to_plistlib_pytree(
      golden_data: Mapping[
          str, Mapping[torch.dtype, Sequence[tuple[OpInput, OpOutput]]]
      ],
  ) -> Mapping[
      str, Mapping[str, Sequence[tuple[_pytree.PyTree, _pytree.PyTree]]]
  ]:
    """Converts the golden data to a plistlib-compatible pytree."""

    def leaf_func(x: OpInput | OpOutput) -> _pytree.PyTree:
      return x.to_plistlib_pytree()

    def is_leaf(x: Any) -> bool:
      return isinstance(x, (OpInput, OpOutput))

    # tree_map() never translates dict keys (doing so may break the structure of
    # the dict), so we have to translate the dtypes in the dict keys separately.
    data_with_str_dtypes = {
        op: {
            str(dt).split(".")[-1]: samples
            for dt, samples in dt_to_samples.items()
        }
        for op, dt_to_samples in golden_data.items()
    }
    return _pytree.tree_map(
        leaf_func,
        data_with_str_dtypes,
        is_leaf=is_leaf,
    )

  encoded_data = to_plistlib_pytree(_GOLDEN_GPU_DATA)
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


def _load_golden_files() -> None:
  """Loads the checked in golden files into _GOLDEN_GPU_DATA."""
  golden_file_pattern = f"{_golden_file_prefix()}*.gz"
  golden_files = list(pathlib.Path(__file__).parent.glob(golden_file_pattern))
  if not golden_files:
    raise ValueError(
        f"No golden files found matching the pattern: {golden_file_pattern}"
    )

  _GOLDEN_GPU_DATA.clear()
  for golden_file in golden_files:
    print(
        "Loading the collected input/output pairs for each op from"
        f" {golden_file}",
        flush=True,
    )
    with gzip.open(golden_file, "rb") as f:
      # TODO(pganssle): Figure out why this is required
      f = typing.cast(IO[bytes], f)
      plist_ptree = plistlib.load(
          f, fmt=typing.cast(plistlib.PlistFormat, plistlib.FMT_BINARY)
      )
    for op, dt_to_encoded_samples in plist_ptree.items():
      if op not in _GOLDEN_GPU_DATA:
        _GOLDEN_GPU_DATA[op] = {}
      for dtype_name, encoded_samples in dt_to_encoded_samples.items():
        dtype = getattr(torch, dtype_name)
        samples = []
        for encoded_op_input, encoded_op_output in encoded_samples:
          op_input = OpInput.from_plistlib_pytree(encoded_op_input)
          op_output = OpOutput.from_plistlib_pytree(encoded_op_output)
          samples.append((op_input, op_output))
        _GOLDEN_GPU_DATA[op][dtype] = samples


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

  # Pick a random seed for the test.
  global _RANDOM_SEED
  if absltest.FLAGS["test_random_seed"].present:
    # The user explicitly passed --test_random_seed=N, so we use that value.
    _RANDOM_SEED = absltest.FLAGS.test_random_seed
  elif _torch_tpu_vs_cpu_mode():
    # The user did not pass --test_random_seed, so we pick a 5-digit seed
    # based on the current time.
    _RANDOM_SEED = time.time_ns() % 100000
  else:
    # We are either generating the golden GPU file or using it.
    # Pick a fixed seed to ensure that the golden file is stable and we
    # run the tests in the same condition where the golden file was generated.
    _RANDOM_SEED = 1234

  # Set the random seed for Python and Torch.
  _seed_rngs(_RANDOM_SEED)
  print(f"Repro with --test_random_seed={_RANDOM_SEED}", flush=True)
  print(f"Torch initial seed: {torch.initial_seed()}", flush=True)

  # Assert that `torch.get_default_dtype()` returns `torch.float` when `setUp`
  # and `tearDown` are called.
  TestCase._default_dtype_check_enabled = True  # pylint: disable=protected-access

  if _torch_tpu_vs_gpu_mode():
    print("Running in TorchTPU vs GPU mode.", flush=True)
    _load_golden_files()
    # TODO: b/502610173 - When the default value is changed to False for excess
    # precision, keep configuring the value as True for compile mode to compare
    # with the GPU inductor backend. For other tests, like gradient tests, tpu
    # vs cpu tests, keep the value as True before updating their accuracy
    # overrides.
    if not is_compiled_mode():
      torch.backends.tpu.allow_excess_precision = False  # pytype: disable=module-attr
    return

  if _perf_mode():
    if _ANALYZE.value:
      _analyze_perf_data()
      sys.exit(0)  # Don't run the tests.
    else:
      # Initialize torch_tpu.
      api.tpu_device()
      # Disable compilation cache so that we can measure the time
      # for compiling the same op repeatedly.
      torch_tpu_module = getattr(torch, "tpu")
      torch_tpu_module._set_allow_cache(False)  # pylint: disable=protected-access
      # Run an op to warm up the XLA compiler, so that we can measure
      # the compilation time of ops fairly in the inividual tests.
      torch.ones(1, dtype=torch.float32, device="tpu").to("cpu")

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
              # Create a small ramp (e.g., 0.0 to 0.01) to break any ties
              # For bfloat16, we need a larger scale.
              scale = 0.05 if dtype == torch.bfloat16 else 1e-2
              ramp = (
                  torch.linspace(
                      0, 1, sample.input.numel(), device=device, dtype=dtype
                  ).reshape(sample.input.shape)
                  * scale
              )

              # Apply the ramp and preserve the requires_grad state
              if requires_grad:
                sample.input = (sample.input.detach() + ramp).requires_grad_(
                    True
                )
              else:
                sample.input = sample.input + ramp

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


def skip_if_torch_tpu_vs_gpu_mode(
    test_item: Callable[..., None],
) -> Callable[..., None]:
  """Decorator that skips a test if on `TORCH_TPU_VS_GPU` mode.

  This is useful for tests where TPU doesn't output the same results as GPU only
  (i.e. CPU works as expected) on its samples. In other words, these are
  pontential bugs. If you use this decorator, you should open an issue.

  Args:
    test_item: The test item to be skipped.

  Returns:
    The test item, wrapped in a function that skips it if on
    `TORCH_TPU_VS_GPU` mode.
  """

  @functools.wraps(test_item)
  def skip_wrapper(self: unittest.TestCase, *args, **kwargs) -> None:
    if _torch_tpu_vs_gpu_mode():
      self.skipTest("Does not work on TORCH_TPU_VS_GPU mode.")
    return test_item(self, *args, **kwargs)

  return skip_wrapper


def skip_unless_torch_tpu_vs_gpu_mode(
    test_item: Callable[..., None],
) -> Callable[..., None]:
  """Decorator that runs a test only if on `TORCH_TPU_VS_GPU` mode."""

  @functools.wraps(test_item)
  def skip_wrapper(self: unittest.TestCase, *args, **kwargs) -> None:
    if not _torch_tpu_vs_gpu_mode():
      self.skipTest("Does not work on non-TORCH_TPU_VS_GPU modes.")
    return test_item(self, *args, **kwargs)

  return skip_wrapper
