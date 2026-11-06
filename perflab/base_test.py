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

"""Base test class for benchmark tests with subprocess isolation.

Runs individual test methods in isolated child processes.

A test method that crashes the interpreter or leaves the device in an unusable
state must not take every later test in the binary down with it.
BaseBenchmarkTest.run executes each test method in a fresh forkserver child and
relays the outcome to the parent's TestResult, so a hard crash, or a hang past
the per-test timeout, is recorded against that one test and the run moves on.

IPC protocol:
  The child transmits a single (outcome, detail, payload) tuple over a
  one-way pipe before exiting:
    (Outcome.SUCCESS, None, payload)      the method passed.
    (Outcome.SKIPPED, reason, payload)    the method called skipTest().
    (Outcome.FAILURE, traceback, payload) an assertion failed.
    (Outcome.ERROR, traceback, payload)   an unexpected exception was raised.
  payload is whatever the method stored in self.isolated_payload (None by
  default); it must be picklable. If the child dies without transmitting, or
  transmits nothing within the timeout, the parent synthesizes Outcome.CRASHED
  or Outcome.TIMED_OUT with a None payload.

Why forkserver rather than subprocess or the other start methods:
  1. fork is unsafe once multithreaded C++ runtimes (CUDA, gRPC, logging)
     are loaded: the child deadlocks or crashes.
  2. spawn re-execs the binary per case, which costs a full interpreter start
     and a re-import of torch for every benchmark. forkserver pays that once.
  3. subprocess.run cannot hand a bound (parameterized) test method to the
     child; re-running the same method by name in the child via pickling the
     TestCase class does that without serializing the method's arguments.
"""

from collections.abc import Mapping, Sequence
import contextlib
import enum
from multiprocessing import connection as mp_connection
import os
import sys
import traceback
from typing import Any
import unittest

from absl import flags
from absl import logging
from absl.testing import absltest
from torch_tpu._internal.distributed import multiprocessing
from perflab import cases
from perflab import common
from perflab import context as context_lib
from perflab import device_ops as device_ops_lib
from perflab import discovery as discovery_lib
from perflab import export as export_lib
from perflab import flags as flags_lib
from perflab import known_failures
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import workload as workload_lib

_DRY_RUN_OUTPUT_FILE = "test_targets.txt"

# Timeout for a child process that has already reported its result to exit
# on its own before killing it. A lingering non-daemon thread (a profiler
# uploader, a compile worker pool) must not block the next test.
_EXIT_GRACE_SECONDS = 60.0

# Timeout for a child process to report its result.
SUBPROCESS_TIMEOUT_SECONDS = 900.0


CaseParams = cases.CaseParams

# The smallest batch size at which each benchmark ran out of device memory in
# this process's batch sweep, by benchmark name and run mode (memory use differs
# between run modes). Batch sweep cases at larger batch sizes are skipped,
# assuming that memory use grows with the batch size.
_out_of_memory_batch_sizes: dict[tuple[str, common.RunMode], int] = {}


def note_out_of_memory(
    spec_name: str, mode: common.RunMode, batch_size: int
) -> None:
  """Records that a benchmark ran out of device memory at a batch size.

  Args:
    spec_name: The name of the benchmark.
    mode: The run mode of the case that ran out of memory.
    batch_size: The batch size that ran out of memory.
  """
  key = (spec_name, mode)
  current = _out_of_memory_batch_sizes.get(key)
  if current is None or batch_size < current:
    _out_of_memory_batch_sizes[key] = batch_size


def get_out_of_memory_skip_reason(
    spec_name: str, mode: common.RunMode, batch_size: int | None
) -> str | None:
  """Returns why a batch sweep case is skipped as too large, or None.

  Args:
    spec_name: The name of the benchmark.
    mode: The run mode of the case.
    batch_size: The batch size of the case, or None if it is not a batch sweep
      case.
  """
  if batch_size is None:
    return None
  out_of_memory_batch_size = _out_of_memory_batch_sizes.get((spec_name, mode))
  if out_of_memory_batch_size is None or batch_size < out_of_memory_batch_size:
    return None
  return (
      f"Benchmark {spec_name} ran out of device memory in run mode"
      f" {mode.value} at batch size {out_of_memory_batch_size}, so batch size"
      f" {batch_size} is not run"
  )


def _microbenchmark_name(
    spec: registry_lib.BenchmarkSpec, batch_size: int | None
) -> str | None:
  """Returns the key that a case exports its results under, if any.

  Args:
    spec: The benchmark of the case; its `microbenchmark_name` keys all of its
      results.
    batch_size: The batch size of a batch sweep case, which also keys its
      results (`bs<N>`), or None.
  """
  keys = [spec.microbenchmark_name]
  if batch_size is not None:
    keys.append(export_lib.batch_size_key(batch_size))
  return "_".join(k for k in keys if k) or None


class Outcome(enum.Enum):
  """How an isolated test method finished."""

  SUCCESS = "success"
  SKIPPED = "skipped"
  FAILURE = "failure"
  ERROR = "error"
  # Synthesized by the parent; never sent by the child.
  CRASHED = "crashed"
  TIMED_OUT = "timed_out"


def _run_in_child(
    cls: type[absltest.TestCase],  # ABSLTEST_OK=Reference to absltest.TestCase
    method_name: str,
    argv: Sequence[str],
    conn: mp_connection.Connection,
) -> None:
  """Executes a single test method and reports its outcome over conn.

  Must be called in a child process.

  Args:
    cls: The test case class.
    method_name: The name of the test method to run.
    argv: Flag arguments to parse before running.
    conn: The connection to send the (outcome, detail, payload) tuple to.
  """
  try:
    flags.FLAGS(list(argv), known_only=True)
    # Enable logging to stderr in the child process. This is the same as the
    # default behavior in the parent process.
    flags.FLAGS.set_default("alsologtostderr", True)
    logging.use_absl_handler()

    instance = cls(method_name)
    setattr(instance, "in_isolated_child", True)
    res = unittest.TestResult()  # UNITTEST_OK=TestResult object for subprocess
    absltest.TestCase.run(  # ABSLTEST_OK=Subprocess runner execution
        instance, res
    )
    payload = getattr(instance, "isolated_payload", None)
    if res.failures:
      conn.send((Outcome.FAILURE, res.failures[0][1], payload))
    elif res.errors:
      conn.send((Outcome.ERROR, res.errors[0][1], payload))
    elif res.skipped:
      conn.send((Outcome.SKIPPED, res.skipped[0][1], payload))
    else:
      conn.send((Outcome.SUCCESS, None, payload))
  except Exception:  # pylint: disable=broad-exception-caught
    conn.send((Outcome.ERROR, traceback.format_exc(), None))
  finally:
    conn.close()
    sys.stdout.flush()
    sys.stderr.flush()


def _kill(process) -> None:
  if process.is_alive():
    process.kill()
  process.join()


def run_isolated(
    cls: type[absltest.TestCase],  # ABSLTEST_OK=Reference to absltest.TestCase
    method_name: str,
    *,
    timeout_seconds: float | None = None,
) -> tuple[Outcome, str | None, Any]:
  """Runs cls(method_name) in a forkserver child and returns its outcome.

  Args:
    cls: The test case class; must be importable by the child (pickled by
      reference).
    method_name: The test method to run.
    timeout_seconds: Kill the child and report Outcome.TIMED_OUT if it has not
      reported within this many seconds. None waits indefinitely.

  Returns:
    The (outcome, detail, payload) tuple described in the module docstring.
  """
  ctx = multiprocessing.get_context("forkserver")
  parent_conn, child_conn = ctx.Pipe(duplex=False)
  process = ctx.Process(
      target=_run_in_child,
      args=(cls, method_name, list(sys.argv), child_conn),
  )
  process.start()
  child_conn.close()

  try:
    if not parent_conn.poll(timeout_seconds):
      _kill(process)
      return (
          Outcome.TIMED_OUT,
          (
              f"{method_name} reported no result within {timeout_seconds} s;"
              " the child process was killed."
          ),
          None,
      )
    try:
      outcome, detail, payload = parent_conn.recv()
    except EOFError:
      # The write end closed without a message: the child died before
      # reporting (device-side assert, SIGSEGV, os._exit).
      process.join(_EXIT_GRACE_SECONDS)
      _kill(process)
      return (
          Outcome.CRASHED,
          (
              f"{method_name} child process exited with code"
              f" {process.exitcode} before reporting a result."
          ),
          None,
      )
  finally:
    parent_conn.close()

  process.join(_EXIT_GRACE_SECONDS)
  _kill(process)
  return outcome, detail, payload


def _record(result, test, outcome: Outcome, detail: str | None) -> None:
  """Translates an Outcome into the matching TestResult call."""
  if outcome == Outcome.SUCCESS:
    result.addSuccess(test)
  elif outcome == Outcome.SKIPPED:
    result.addSkip(test, detail)
  elif outcome == Outcome.FAILURE:
    # Raise and catch so sys.exc_info() provides the (type, value, traceback)
    # tuple expected by TestResult.addFailure.
    try:
      raise AssertionError(f"Subprocess test failed:\n{detail}")
    except AssertionError:
      result.addFailure(test, sys.exc_info())
  else:
    # Raise and catch so sys.exc_info() provides the (type, value, traceback)
    # tuple expected by TestResult.addError.
    try:
      raise RuntimeError(f"Subprocess test {outcome.value}:\n{detail}")
    except RuntimeError:
      result.addError(test, sys.exc_info())


def _get_output_dir(key: str) -> str:
  try:
    return os.environ[key]
  except KeyError:
    raise RuntimeError(
        f"Output directory not set using environment variable {key}"
    )


case_params = cases.case_params
normalize_randomize_ordering_seed = cases.normalize_randomize_ordering_seed
BenchmarkTestLoader = cases.BenchmarkTestLoader


class BaseBenchmarkTest(
    absltest.TestCase  # ABSLTEST_OK=avoid shared lib import
):
  """Base test class for benchmark tests.

  Allows running every test method in its own child process. A test case that
  crashes the device is contained to that case.

  A binary subclasses this together with the `TestCase` of absl's
  `parameterized` module, sets the `framework`, `platform`, `case_params`, and
  `import_failures` attributes, implements the framework hooks
  (`make_device_ops`, `make_run_step`, `reset_state`, and optionally
  `run_context` and `after_measure`), and defines a test method parameterized
  over its cases that calls `run_case`.

  This is a `TestCase` base class with a test method of its own,
  `test_benchmark_imports`. Import this module rather than the class, so that
  test loaders do not collect the class itself.

  Attributes:
    framework: The framework this binary measures; set by the subclass.
    platform: The platform the binary targets; set by the subclass.
    case_params: (spec, mode), plus the batch size of a batch sweep case, behind
      each generated test method name, from case_params(); set by the subclass.
      Lets the parent export a failed record for a child that died before it
      could report.
    import_failures: The benchmark modules that the binary failed to import;
      `test_benchmark_imports` fails if there are any.
    in_isolated_child: True while the method runs inside the child process.
    isolated_payload: Set by the test method in the child; exported by the
      parent. Must be picklable.
  """

  framework: mode_lib.Framework
  platform: target_lib.Platform
  case_params: Mapping[str, CaseParams] = {}
  import_failures: Sequence[discovery_lib.ImportFailure] = ()
  in_isolated_child: bool = False
  isolated_payload: Any = None

  def setUp(self):
    super().setUp()
    log_counter = getattr(logging, "_log_counter_per_token", None)
    if log_counter is not None:
      log_counter.clear()

  def test_benchmark_imports(self):
    if self.import_failures:
      self.fail(
          "benchmark module(s) failed to import:\n  "
          + "\n  ".join(str(f) for f in self.import_failures)
      )

  def should_isolate(self) -> bool:
    # Dry runs only append to a file in TEST_UNDECLARED_OUTPUTS_DIR; keep
    # them in-process rather than paying a child start per line.
    return (
        not self.in_isolated_child
        and flags_lib.RUN_IN_SUBPROCESS.value
        and not flags_lib.DRY_RUN.value
    )

  def _get_parent_skip_reason(self) -> str | None:
    """Returns skip reason if the test method can be skipped without spawning a child.

    Returns:
      The human-readable skip reason string if the test should be skipped in the
      parent process, or None if the test should execute.
    """
    params = self.case_params.get(self._testMethodName)
    if params is None:
      return None
    spec, mode, _ = cases.unpack_case_params(params)
    if (
        self.framework == mode_lib.Framework.TORCHAX
        and spec.stepper == step_lib.StepperType.DECODER_ONLY_DECODE
    ):
      return f"TorchAx benchmark does not support stepper {spec.stepper}"
    skip_reason = cases.get_skip_reason(
        spec,
        mode,
        framework=self.framework,
        device_kind=target_lib.make_target(self.platform).device_kind,
    )
    if flags_lib.SKIP_BEHAVIOR.value == "skip":
      return skip_reason
    if flags_lib.SKIP_BEHAVIOR.value == "run_skipped" and skip_reason is None:
      return (
          f"Benchmark {spec.name} is not skipped for run mode {mode.value},"
          " skipping due to skip_behavior=run_skipped"
      )
    return None

  def run(self, result=None):
    """Runs the test method in a child process and records its outcome.

    Mirrors regular test's startTest/stopTest and the default-result handling
    so runner output, counts and XML reports see the isolated method as an
    ordinary one.

    Args:
      result: The TestResult to populate.

    Returns:
      The populated TestResult.
    """
    if not self.should_isolate():
      return super().run(result)

    orig_result = result
    if result is None:
      result = self.defaultTestResult()
      start_test_run = getattr(result, "startTestRun", None)
      if start_test_run is not None:
        start_test_run()

    result.startTest(self)
    try:
      params = self.case_params.get(self._testMethodName)
      if params is not None:
        spec, mode, batch_size = cases.unpack_case_params(params)
        skip_reason = self._get_parent_skip_reason()
        if skip_reason is not None:
          self.report_skip(spec, mode)
          _record(result, self, Outcome.SKIPPED, skip_reason)
          return result
        if batch_size is not None:
          skip_reason = get_out_of_memory_skip_reason(
              spec.name, mode, batch_size
          )
          if skip_reason is not None:
            self.report_out_of_memory_skip(spec, mode, batch_size)
            _record(result, self, Outcome.SKIPPED, skip_reason)
            return result

      outcome, detail, payload = run_isolated(
          self.__class__,
          self._testMethodName,
          timeout_seconds=SUBPROCESS_TIMEOUT_SECONDS,
      )
      if payload is not None:
        export_lib.export(payload)
        if getattr(payload, "out_of_memory", False):
          note_out_of_memory(
              payload.spec_name, payload.run_mode, payload.batch_size
          )
      elif outcome in (Outcome.CRASHED, Outcome.TIMED_OUT):
        # No payload: the case was skipped, failed before measuring, or the
        # child died. Only the last has no record of its own yet.
        if params is not None:
          spec, mode, batch_size = cases.unpack_case_params(params)
          self.report_result(spec, mode, succeeded=False, batch_size=batch_size)
      _record(result, self, outcome, detail)
    finally:
      result.stopTest(self)
      if orig_result is None:
        stop_test_run = getattr(result, "stopTestRun", None)
        if stop_test_run is not None:
          stop_test_run()
    return result

  def report_result(
      self,
      spec: registry_lib.BenchmarkSpec,
      mode: common.RunMode,
      *,
      succeeded: bool,
      metrics: metrics_lib.MetricsInterface | None = None,
      batch_size: int | None = None,
  ) -> None:
    """Exports the case's result, or hands it to the parent to export.

    Args:
      spec: The benchmark of the case.
      mode: The run mode of the case.
      succeeded: Whether the case succeeded.
      metrics: The metrics of the case, if it measured any.
      batch_size: The batch size of a batch sweep case, which keys its results.
    """
    self._report(
        export_lib.BenchmarkData(
            spec_name=spec.name,
            platform=self.platform,
            framework=self.framework,
            run_mode=mode,
            succeeded=succeeded,
            metrics=metrics or metrics_lib.PerformanceMetrics(),
            microbenchmark_name=_microbenchmark_name(spec, batch_size),
            batch_size=batch_size,
        )
    )

  def report_out_of_memory(
      self,
      spec: registry_lib.BenchmarkSpec,
      mode: common.RunMode,
      batch_size: int,
  ) -> None:
    """Exports that a batch sweep case ran out of device memory.

    The parent then skips the larger batch sizes of the benchmark.

    Args:
      spec: The benchmark of the case.
      mode: The run mode of the case.
      batch_size: The batch size that ran out of memory.
    """
    self._report(
        self._batch_sweep_record(spec, mode, batch_size, out_of_memory=True)
    )

  def report_out_of_memory_skip(
      self,
      spec: registry_lib.BenchmarkSpec,
      mode: common.RunMode,
      batch_size: int,
  ) -> None:
    """Exports that a batch sweep case was skipped after an out of memory.

    A smaller batch size of the benchmark ran out of device memory in the run
    mode, so the case is not run. It is exported as skipped rather than as out
    of memory, which it was not measured to be.

    Args:
      spec: The benchmark of the case.
      mode: The run mode of the case.
      batch_size: The batch size of the case.
    """
    self._report(self._batch_sweep_record(spec, mode, batch_size, skipped=True))

  def _batch_sweep_record(
      self,
      spec: registry_lib.BenchmarkSpec,
      mode: common.RunMode,
      batch_size: int,
      *,
      out_of_memory: bool = False,
      skipped: bool = False,
  ) -> export_lib.BenchmarkData:
    """Returns the record of a batch sweep case that measured no metrics."""
    return export_lib.BenchmarkData(
        spec_name=spec.name,
        platform=self.platform,
        framework=self.framework,
        run_mode=mode,
        succeeded=True,
        metrics=metrics_lib.PerformanceMetrics(),
        microbenchmark_name=_microbenchmark_name(spec, batch_size),
        batch_size=batch_size,
        out_of_memory=out_of_memory,
        skipped=skipped,
    )

  def _report(self, data: export_lib.BenchmarkData) -> None:
    """Exports `data`, or hands it to the parent process to export."""
    if self.in_isolated_child:
      self.isolated_payload = data
      return
    export_lib.export(data)
    if data.out_of_memory and data.batch_size is not None:
      note_out_of_memory(data.spec_name, data.run_mode, data.batch_size)

  def report_skip(
      self, spec: registry_lib.BenchmarkSpec, mode: common.RunMode
  ) -> None:
    """Exports a skip record if the skipped case is a known failure.

    The record lets dashboards drop the last failure of a benchmark that became
    a known failure. The run modes in `BenchmarkSpec.skipped_run_modes` are not
    recorded, as most of them never ran.

    Args:
      spec: The skipped benchmark.
      mode: The skipped run mode.
    """
    if flags_lib.DRY_RUN.value:
      return
    failure_reason = known_failures.get_known_failure_reason(
        spec.name,
        framework=self.framework,
        device_kind=target_lib.make_target(self.platform).device_kind,
        run_mode=mode,
    )
    if failure_reason is None:
      return
    export_lib.export_skip(
        spec_name=spec.name,
        platform=self.platform,
        framework=self.framework,
        run_mode=mode,
    )

  def _dry_run_test(self) -> None:
    """Logs and writes dry run test details to a text file."""
    logging.info(
        "[DRY RUN]: Would have run benchmark test method name %s",
        self._testMethodName,
    )
    output_dir = _get_output_dir("TEST_UNDECLARED_OUTPUTS_DIR")
    txt_path = os.path.join(output_dir, _DRY_RUN_OUTPUT_FILE)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, mode="a", encoding="utf-8") as f:
      f.write(self._testMethodName + "\n")

  def make_device_ops(
      self, target: target_lib.Target
  ) -> device_ops_lib.DeviceOps:
    """Returns the device operations that measure the cases on `target`."""
    raise NotImplementedError

  def make_run_step(
      self,
      spec: registry_lib.BenchmarkSpec,
      ctx: context_lib.Context,
      mode: common.RunMode,
  ) -> tuple[step_lib.Stepper, workload_lib.StepWork | None]:
    """Builds the stepper of a benchmark and the work of one of its steps.

    Called inside `run_context`.

    Args:
      spec: The benchmark to build.
      ctx: The context to build the benchmark for, with its resolved workload.
      mode: The run mode to build the benchmark for.

    Returns:
      The stepper of the benchmark and the work of one of its steps, or None if
      the benchmark declares no workload.
    """
    raise NotImplementedError

  def reset_state(
      self,
      target: target_lib.Target,
      device_ops: device_ops_lib.DeviceOps,
      mode: common.RunMode,
  ) -> None:
    """Resets framework and hardware compilation/runtime state before a case."""
    raise NotImplementedError

  def run_context(
      self, mode: common.RunMode, target: target_lib.Target
  ) -> contextlib.AbstractContextManager[Any]:
    """Returns the context to build and measure a case in."""
    del mode, target  # Unused.
    return contextlib.nullcontext()

  def after_measure(
      self,
      spec: registry_lib.BenchmarkSpec,
      ctx: context_lib.Context,
      target: target_lib.Target,
      case_name: str,
      metrics: metrics_lib.PerformanceMetrics,
  ) -> None:
    """Adds to the metrics of a measured case, outside `run_context`."""
    del spec, ctx, target, case_name, metrics  # Unused.

  def run_case(
      self,
      spec: registry_lib.BenchmarkSpec,
      mode: common.RunMode,
      batch_size: int | None = None,
  ) -> None:
    """Runs, measures, and reports the case of `spec` in run mode `mode`.

    Args:
      spec: The benchmark to run.
      mode: The run mode to run it in.
      batch_size: The batch size of a batch sweep case, or None to measure the
        benchmark at its own batch size.
    """
    skip_reason = cases.get_skip_reason(
        spec,
        mode,
        framework=self.framework,
        device_kind=target_lib.make_target(self.platform).device_kind,
    )

    if skip_reason is not None and flags_lib.SKIP_BEHAVIOR.value == "skip":
      self.report_skip(spec, mode)
      self.skipTest(skip_reason)
    if skip_reason is None and flags_lib.SKIP_BEHAVIOR.value == "run_skipped":
      self.skipTest(
          f"Benchmark {spec.name} is not skipped for run mode {mode.value},"
          " skipping due to skip_behavior=run_skipped"
      )

    if flags_lib.DRY_RUN.value:
      self._dry_run_test()
      return

    # Batch sizes above one that ran out of memory; the parent skips them
    # before starting a child, so this only applies to in-process runs.
    out_of_memory_reason = get_out_of_memory_skip_reason(
        spec.name, mode, batch_size
    )
    if batch_size is not None and out_of_memory_reason is not None:
      self.report_out_of_memory_skip(spec, mode, batch_size)
      self.skipTest(out_of_memory_reason)

    target = target_lib.make_target(self.platform, dtype=spec.dtype)
    device_ops = self.make_device_ops(target)
    ctx = registry_lib.resolve_context(
        spec,
        context_lib.Context(
            target=target, run_scope=context_lib.RUN_SCOPE.value
        ),
        batch_size,
    )

    self.reset_state(target, device_ops, mode)

    if (
        skip_reason is not None
        and flags_lib.SKIP_BEHAVIOR.value == "assert_raise"
    ):
      with self.assertRaises(Exception):  # ASSERT_RAISES_OK=benchmark
        self._run_and_measure(spec, mode, target, device_ops, ctx, batch_size)
      return

    try:
      metrics = self._run_and_measure(
          spec, mode, target, device_ops, ctx, batch_size
      )
    except unittest.SkipTest:
      raise
    except Exception as e:
      if batch_size is not None and common.is_out_of_memory(e):
        # The batch size bounds what fits, rather than failing the sweep.
        self.report_out_of_memory(spec, mode, batch_size)
        self.skipTest(
            f"{spec.name} ran out of device memory at batch size"
            f" {batch_size}: {e}"
        )
      self.report_result(spec, mode, succeeded=False, batch_size=batch_size)
      raise

    self.report_result(
        spec, mode, succeeded=True, metrics=metrics, batch_size=batch_size
    )

  def _run_and_measure(
      self, spec, mode, target, device_ops, ctx, batch_size
  ) -> metrics_lib.PerformanceMetrics:
    case_name = (
        f"{spec.name}_{mode.value}"
        if batch_size is None
        else cases.batch_sweep_case_name(spec.name, batch_size, mode)
    )
    try:
      with self.run_context(mode, target):
        run_step, step_work = self.make_run_step(spec, ctx, mode)
        metrics = measure_lib.measure(
            run_step,
            device_ops,
            name=case_name,
            enable_xprof=flags_lib.ENABLE_XPROF.value,
        )
    except target_lib.UnsupportedBenchmark as e:
      self.skipTest(f"{spec.name}: {e}")
    metrics.step_work = step_work

    # After measuring and outside the run context, so that it changes neither
    # the measured times nor the state of the measured device.
    self.after_measure(spec, ctx, target, case_name, metrics)

    logging.info("Metrics for %s:\n%s", case_name, metrics)
    self._assert_measured(metrics)
    return metrics

  def _assert_measured(self, metrics: metrics_lib.PerformanceMetrics) -> None:
    self.assertGreater(metrics.e2e_wall_time_seconds, 0.0)
    self.assertGreater(metrics.first_step_time_seconds, 0.0)
    if measure_lib.POST_WARMUP_STEPS.value > 0:
      self.assertGreater(metrics.post_warmup_step_time_seconds, 0.0)
