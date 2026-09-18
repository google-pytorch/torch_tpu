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
  2. spawn (and the resource_tracker thread a Queue starts) fails in
     self-contained Bazel .par binaries, where sys.executable is None.
  3. subprocess.run cannot hand a bound (parameterized) test method to the
     child; re-running the same method by name in the child via pickling the
     TestCase class does that without serializing the method's arguments.
"""

import enum
from multiprocessing import connection as mp_connection
import os
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence
import unittest

from absl import flags
from absl import logging
from absl.testing import absltest
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e.harness import export as export_lib
from examples.benchmarks.e2e.harness import flags as flags_lib
from examples.benchmarks.e2e.harness import metrics as metrics_lib
from examples.benchmarks.e2e.harness import mode as mode_lib
from examples.benchmarks.e2e.harness import registry as registry_lib
from examples.benchmarks.e2e.harness import target as target_lib
from torch_tpu._internal.distributed import multiprocessing

_DRY_RUN_OUTPUT_FILE = "test_targets.txt"

# Timeout for a child process that has already reported its result to exit
# on its own before killing it. A lingering non-daemon thread (a profiler
# uploader, a compile worker pool) must not block the next test.
_EXIT_GRACE_SECONDS = 60.0

# Timeout for a child process to report its result.
SUBPROCESS_TIMEOUT_SECONDS = 900.0


CaseParams = tuple[registry_lib.BenchmarkSpec, common.RunMode]


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


def case_params(
    cases: Iterable[tuple[str, registry_lib.BenchmarkSpec, common.RunMode]],
    method_name: str = "test_benchmark",
) -> Mapping[str, CaseParams]:
  """Maps the names parameterized.named_parameters generates to (spec, mode).

  Args:
    cases: The (name, spec, mode) triples passed to named_parameters.
    method_name: The decorated method; the generated name is <method_name>_<case
      name>.

  Returns:
    A mapping from generated test method name to its (spec, mode).
  """
  return {f"{method_name}_{name}": (spec, mode) for name, spec, mode in cases}


class BaseBenchmarkTest(
    absltest.TestCase  # ABSLTEST_OK=avoid shared lib import
):
  """Base test class for benchmark tests.

  Allows running every test method in its own child process. A test case that
  crashes the device is contained to that case.

  Attributes:
    framework: The framework this binary measures; set by the subclass.
    platform: The platform the binary targets; set by the subclass.
    case_params: (spec, mode) behind each generated test method name, from
      case_params(); set by the subclass. Lets the parent export a failed record
      for a child that died before it could report.
    in_isolated_child: True while the method runs inside the child process.
    isolated_payload: Set by the test method in the child; exported by the
      parent. Must be picklable.
  """

  framework: mode_lib.Framework
  platform: target_lib.Platform
  case_params: Mapping[str, CaseParams] = {}
  in_isolated_child: bool = False
  isolated_payload: Any = None

  def setUp(self):
    super().setUp()
    log_counter = getattr(logging, "_log_counter_per_token", None)
    if log_counter is not None:
      log_counter.clear()

  def should_isolate(self) -> bool:
    # Dry runs only append to a file in TEST_UNDECLARED_OUTPUTS_DIR; keep
    # them in-process rather than paying a child start per line.
    return (
        not self.in_isolated_child
        and flags_lib.RUN_IN_SUBPROCESS.value
        and not flags_lib.DRY_RUN.value
    )

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
      outcome, detail, payload = run_isolated(
          self.__class__,
          self._testMethodName,
          timeout_seconds=SUBPROCESS_TIMEOUT_SECONDS,
      )
      if payload is not None:
        export_lib.export(payload)
      elif outcome in (Outcome.CRASHED, Outcome.TIMED_OUT):
        # No payload: the case was skipped, failed before measuring, or the
        # child died. Only the last has no record of its own yet.
        params = self.case_params.get(self._testMethodName)
        if params is not None:
          spec, mode = params
          self.report_result(spec, mode, succeeded=False)
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
  ) -> None:
    """Exports the case's result, or hands it to the parent to export."""
    data = export_lib.BenchmarkData(
        spec_name=spec.name,
        platform=self.platform,
        framework=self.framework,
        run_mode=mode,
        succeeded=succeeded,
        metrics=metrics or metrics_lib.PerformanceMetrics(),
    )
    if self.in_isolated_child:
      self.isolated_payload = data
    else:
      export_lib.export(data)

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
