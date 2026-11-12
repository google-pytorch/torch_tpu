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

"""Common command-line flags for benchmark tests."""

import copy
import json
import os
import sys
from typing import Any

from absl import flags
from perflab import context as context_lib
from perflab import target as target_lib

SKIP_BEHAVIOR = flags.DEFINE_enum(  # PYTHON_FLAG_OK=benchmark
    "skip_behavior",
    "skip",
    ["skip", "assert_raise", "ignore", "run_skipped"],
    "Behavior for run modes that are marked as skipped in the config."
    " Options:\n  skip: Skips the test if the benchmark is marked as skipped.\n"
    "  assert_raise: Runs the test and expects it to raise an exception if"
    " skipped.\n  ignore: Runs the test normally even if it is marked as"
    " skipped.\n  run_skipped: Only runs skipped tests (skips those not marked"
    " as skipped).",
)

ENABLE_XPROF = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "enable_xprof",
    False,
    "Whether to enable xprof profiling.",
    allow_override=True,
)

DRY_RUN = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "dry_run",
    False,
    "Whether to run in dry-run mode. If True, actual benchmark execution is"
    " skipped, and the list of targets that would have run are logged and"
    " written to the test outputs directory specified by environment variable"
    " TEST_UNDECLARED_OUTPUTS_DIR.",
)


RUN_IN_SUBPROCESS = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "run_in_subprocess",
    True,
    "Whether to run each benchmark case in its own child process. A case that"
    " crashes the interpreter (a CUDA device-side assert, a SIGSEGV in a"
    " kernel) or poisons the device context then fails alone instead of"
    " taking every later case in the shard with it. Disable to run cases"
    " in-process, e.g. to attach a debugger or use --pdb_post_mortem.",
)

FAKE_TENSOR_MODE = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "fake_tensor_mode",
    False,
    "Whether to construct models and execute benchmark steps inside PyTorch's"
    " FakeTensorMode(allow_non_fake_inputs=True). Only valid with"
    " BENCHMARK_PLATFORM=cpu, so it cannot skew real measurements; flag parsing"
    " fails otherwise. FakeTensorMode validates PyTorch op dispatch, tensor"
    " shapes, dtypes, and autograd graphs without allocating storage or"
    " executing CPU compute kernels, so it can also build models at their"
    " --run_scope=full sizes. Model configs are then read without the local"
    " weights cache directory. Building each case (not its steps) must also"
    " fit in the HBM of a v7 TensorCore; see perflab/fake_memory.py.",
)

COUNT_FLOPS = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "count_flops",
    False,
    "Whether to count the model FLOPs of one step of each benchmark after it is"
    " measured, and export them with the achieved TFLOP/s and model FLOPs"
    " utilization (MFU). The FLOPs are counted on a second copy of the"
    " benchmark built for CPU under FakeTensorMode, so counting neither touches"
    " the measured device nor changes the measured times; see"
    " perflab/flops.py.",
)


def _check_fake_tensor_mode(fake_tensor_mode: bool) -> bool:
  """Checks that `--fake_tensor_mode` is only set in CPU runs."""
  if not fake_tensor_mode:
    return True
  platform = target_lib.platform_from_env()
  if platform is not target_lib.Platform.CPU:
    raise flags.ValidationError(
        f"--{FAKE_TENSOR_MODE.name} is only valid with"
        f" {target_lib.PLATFORM_ENV_VAR}=cpu; got"
        f" {target_lib.PLATFORM_ENV_VAR}={platform.value}"
    )
  return True


# Registered with the flag so that every binary that accepts it fails at flag
# parsing, before any benchmark is built or measured.
flags.register_validator(FAKE_TENSOR_MODE, _check_fake_tensor_mode)

BATCH_SWEEP = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "batch_sweep",
    False,
    "Whether to also run the batch sweep: in addition to the default cases,"
    " which run each benchmark at its default workload, one case per"
    " benchmark that supports it (`BenchmarkSpec.batch_sweep`), run mode and"
    " other batch size of --batch_sweep_sizes, named"
    " `<benchmark>_bs<N>_<run mode>`, whose results are keyed by the batch"
    " size (`bs<N>`). The batch sweep is independent of the run mode. A batch"
    " size that runs out of device memory is exported as such rather than as"
    " a failure, and the larger batch sizes of the benchmark in that run mode"
    " are then not run (and exported as skipped), as are those after a batch"
    " size that fails and those past which its throughput stops improving"
    " (--batch_sweep_min_improvement, --batch_sweep_patience). Only valid with"
    " --run_scope=full. Only torch_benchmark_test runs batch sweeps.",
)

BATCH_SWEEP_SIZES = flags.DEFINE_list(  # PYTHON_FLAG_OK=benchmark
    "batch_sweep_sizes",
    ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"],
    "The batch sizes that --batch_sweep measures each benchmark at, unless the"
    " benchmark sets its own (`BatchSweep.batch_sizes`). The sweep of a"
    " benchmark stops early once a batch size runs out of memory or fails, or"
    " once its throughput stops improving (--batch_sweep_min_improvement), so"
    " the large batch sizes only run for benchmarks that still gain from"
    " them.",
)


BATCH_SWEEP_MIN_IMPROVEMENT = flags.DEFINE_float(  # PYTHON_FLAG_OK=benchmark
    "batch_sweep_min_improvement",
    0.03,
    "Stop the batch sweep of a benchmark in a run mode once its throughput"
    " (samples, or else tokens, per second) stops improving: once"
    " --batch_sweep_patience consecutive batch sizes each improve it by less"
    " than this fraction over the best smaller batch size, the larger batch"
    " sizes are not run (and are exported as skipped). 0 measures every batch"
    " size.",
    lower_bound=0.0,
)


BATCH_SWEEP_PATIENCE = flags.DEFINE_integer(  # PYTHON_FLAG_OK=benchmark
    "batch_sweep_patience",
    2,
    "The number of consecutive batch sizes without enough throughput"
    " improvement (--batch_sweep_min_improvement) after which the batch sweep"
    " of a benchmark in a run mode stops. Only batch sizes of at least"
    " --batch_sweep_patience_min_bs count.",
    lower_bound=1,
)


BATCH_SWEEP_PATIENCE_MIN_BS = flags.DEFINE_integer(  # PYTHON_FLAG_OK=benchmark
    "batch_sweep_patience_min_bs",
    8,
    "The smallest batch size that counts towards --batch_sweep_patience"
    " when it does not improve the throughput enough. Small batch sizes"
    " are often bound by the host's dispatch rather than by the device, so"
    " their throughput varies from run to run and can dip before larger"
    " batch sizes improve on it; they still set the best throughput that"
    " larger batch sizes must beat. 1 counts every batch size.",
    lower_bound=1,
)


def _parse_batch_sizes(values: list[str]) -> tuple[int, ...]:
  """Returns the sorted, distinct batch sizes of a --batch_sweep_sizes value."""
  batch_sizes = sorted({int(v) for v in values})
  if not batch_sizes or batch_sizes[0] < 1:
    raise ValueError(f"batch sizes must be positive integers, got {values}")
  return tuple(batch_sizes)


def _check_batch_sweep_sizes(values: list[str]) -> bool:
  """Checks that --batch_sweep_sizes lists positive integers."""
  try:
    _parse_batch_sizes(values)
  except ValueError as e:
    raise flags.ValidationError(f"--{BATCH_SWEEP_SIZES.name}: {e}") from e
  return True


def _check_batch_sweep(values: dict[str, object]) -> bool:
  """Checks that --batch_sweep runs in the full run scope."""
  if values[BATCH_SWEEP.name] and (
      values[context_lib.RUN_SCOPE.name] is not context_lib.RunScope.FULL
  ):
    raise flags.ValidationError(
        f"--{BATCH_SWEEP.name} is only valid with"
        f" --{context_lib.RUN_SCOPE.name}=full"
    )
  return True


flags.register_validator(BATCH_SWEEP_SIZES, _check_batch_sweep_sizes)
flags.register_multi_flags_validator(
    [BATCH_SWEEP, context_lib.RUN_SCOPE], _check_batch_sweep
)


# The environment variable in which a benchmark binary records its command line
# for the forkserver that runs its isolated test cases (see base_test.py). The
# forkserver imports the binary's `__main__`, and so collects its test cases,
# with a command line of its own that lacks the binary's flags.
COLLECTION_ARGV_ENV_VAR = "TORCH_TPU_PERFLAB_COLLECTION_ARGV"


def _collection_argv() -> list[str]:
  """Returns the command line that a benchmark binary collects its cases from.

  That is `sys.argv`, which this records for the child processes of the binary,
  except in a child process of a binary that recorded its command line (its
  forkserver), which collects its test cases from the binary's command line so
  that it has the same test cases, e.g. those of `--batch_sweep`.
  """
  recorded = os.environ.get(COLLECTION_ARGV_ENV_VAR)  # PY_ENVIRON_OK=benchmark
  if recorded:
    record = json.loads(recorded)
    if record.get("pid") == os.getppid():
      return list(record["argv"])
  os.environ[COLLECTION_ARGV_ENV_VAR] = json.dumps(  # PY_ENVIRON_OK=benchmark
      {"pid": os.getpid(), "argv": list(sys.argv)}
  )
  return list(sys.argv)


def collection_flag_values() -> tuple[context_lib.RunScope, bool]:
  """Returns the values of `--run_scope` and `--fake_tensor_mode`.

  Benchmark binaries collect their test cases at import time, usually before
  `flags.FLAGS` is parsed. These two flags are then parsed from `sys.argv` by
  absl's parser, so `--`, `--noflag`, `=false` and `--flagfile` work as in the
  full parse, but into a scratch `FlagValues` that holds copies of just them.
  Parsing `flags.FLAGS` itself here would mark it parsed with only the flags
  defined so far, and the full parse in `app.run` would then append the values
  of multi-value flags a second time. An invalid `--fake_tensor_mode` still
  fails the full parse.
  """
  run_scope, fake_tensor_mode = _collection_values(
      context_lib.RUN_SCOPE, FAKE_TENSOR_MODE
  )
  return run_scope, fake_tensor_mode


def batch_sweep_flag_values() -> tuple[bool, tuple[int, ...]]:
  """Returns the values of `--batch_sweep` and `--batch_sweep_sizes`.

  Like `collection_flag_values`, for benchmark binaries that collect their test
  cases at import, before `flags.FLAGS` is parsed.

  Returns:
    Whether to run the batch sweep, and its sorted, distinct batch sizes.
  """
  batch_sweep, batch_sweep_sizes = _collection_values(
      BATCH_SWEEP, BATCH_SWEEP_SIZES
  )
  return bool(batch_sweep), _parse_batch_sizes(batch_sweep_sizes)


def _collection_values(*holders: flags.FlagHolder) -> tuple[Any, ...]:
  """Returns the values of `holders`, parsed from the command line if need be.

  See `collection_flag_values` and `_collection_argv`.

  Args:
    *holders: The flags to return the values of.
  """
  if flags.FLAGS.is_parsed():
    return tuple(holder.value for holder in holders)
  flag_values = flags.FlagValues()
  for holder in holders:
    flag = copy.deepcopy(flags.FLAGS[holder.name])
    # Validators may read flags that are not copied here (e.g. that of
    # --batch_sweep reads --run_scope); the full parse still runs them.
    flag.validators = []
    flag_values[holder.name] = flag
  flag_values(_collection_argv(), known_only=True)
  return tuple(getattr(flag_values, holder.name) for holder in holders)
