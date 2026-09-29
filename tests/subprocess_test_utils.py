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

"""Utilities for two-stage parent-child subprocess test execution.

Several TorchTPU test files use a two-stage execution model where a "parent"
test class spawns isolated child subprocesses to run individual test methods.
This is needed when the code under test relies on process-global state (e.g.,
C++ static caches, environment variables read once at startup) that cannot be
safely reset between test cases within a single process.

This module extracts the common scaffolding shared by those test files:

* `QueueWriter` — a file-like object that forwards writes to a
  `multiprocessing.Queue`, used to redirect child stdout/stderr back to the
  parent process for logging.

* `sub_test_worker_entry` — the generic child-process entry point that
  installs `QueueWriter`, rewrites `sys.argv` to target a single test method,
  and calls `absltest.main()`.

* `SubprocessTestMixin` — a mixin providing `run_sub_test()`, which spawns an
  isolated child process with the requested environment, drains its output
  queue, and asserts a zero exit code.

Execution Model:
----------------
1. Parent Stage (Default / Full Test Run):
   When running without `--test_mode`, `_TEST_MODE.value` is `None`. The worker
   test class skips its test methods because `--test_mode` is unset. The parent
   test class executes its test methods, each calling `run_sub_test()` to spawn
   a fresh child process running `sub_test_worker_entry` for a specific mode.

2. Worker Stage (Isolated Child Subprocess or Direct Debugging):
   In the worker subprocess (or when manually specifying `--test_mode`), the
   parent test class skips itself because `_TEST_MODE.value` is set, and only
   the target worker test method runs.

The parent process must not initialize the TPU runtime. A TPU device accepts one
process at a time, so a parent that holds it leaves the child unable to
initialize.

Examples:
---------
1. Running the entire test target (Parent Stage):
   Runs the full suite where the parent test spawns an isolated subprocess for
   each subtest:

   ```bash
   bazel test torch_tpu/tests/compile:cache_test
   # or
   bazel test torch_tpu/tests:env_vars_test
   ```

2. Debugging an individual test mode directly:
   To debug a specific test case in the current process (e.g., with breakpoints,
   `pdb`, or faster iteration without subprocess isolation), supply
   `--test_mode=<MODE>` and filter to the worker test method. Remember to set
   any environment variables expected by that test mode:

   * Using `bazel test`:
     ```bash
     bazel test torch_tpu/tests/compile:cache_test \\
         --test_sharding_strategy=disabled \\
         --test_filter="CacheTest.test_persistent_cache_tier2" \\
         --test_arg="--test_mode=TIER2" \\
         --test_env=TORCH_TPU_TIER2_COMPILATION_CACHE=compile_cache_test_tier2
     ```

   * Running the compiled test binary directly (e.g., under a debugger):
     ```bash
     TORCH_TPU_TIER2_COMPILATION_CACHE=compile_cache_test_tier2 \\
         bazel-bin/third_party/py/torch_tpu/tests/compile/cache_test \\
         --test_mode=TIER2 CacheTest.test_persistent_cache_tier2
     ```

   * Using `pytest` (OSS environment):
     ```bash
     TORCH_TPU_TIER2_COMPILATION_CACHE=compile_cache_test_tier2 \\
         pytest third_party/py/torch_tpu/tests/compile/cache_test.py \\
         -k "test_persistent_cache_tier2" -o addopts="--test_mode=TIER2"
     ```
"""

import enum
import io
import os
import queue
import sys
from typing import Mapping, Sequence
from unittest import mock

from absl import flags
from absl import logging
from absl.testing import absltest
from torch_tpu._internal.device import _device_ops_backend
from torch_tpu._internal.distributed import multiprocessing


class QueueWriter:
  """File-like object forwarding writes to a multiprocessing queue.

  Used to redirect stdout/stderr in child subprocesses so that output appears
  in the parent process's test logs.
  """

  def __init__(self, q: queue.Queue[str]):
    self.q = q

  def write(self, s: str | None) -> None:
    if s:
      self.q.put(s)

  def flush(self) -> None:
    pass

  def fileno(self) -> None:
    raise io.UnsupportedOperation("QueueWriter does not have a fileno")


def sub_test_worker_entry(
    q: queue.Queue[str],
    mode: enum.Enum,
    test_method_template: str,
) -> None:
  """Generic child-process entry point for two-stage subprocess tests.

  This function executes inside a newly spawned child process. It redirects
  stdout/stderr to the parent via `q`, and rewrites `sys.argv` so that
  `absltest.main()` runs only the single test method identified by
  `test_method_template`.

  Args:
    q: A multiprocessing queue for forwarding stdout/stderr to the parent.
    mode: The enum member selecting which test mode / method to run.
    test_method_template: A format string producing the fully qualified test
      method name. It is formatted with `mode=mode`, so it can contain
      `{mode.name}` or `{mode.value}` placeholders. For example:
      `"EnvVarsTest.test_{mode.value}"` or
      `"CacheTest.test_persistent_cache_{mode.value}"`.
  """
  qw = QueueWriter(q)
  sys.stdout = qw
  sys.stderr = qw

  sys.argv = [
      sys.argv[0],
      f"--test_mode={mode.name}",
      test_method_template.format(mode=mode),
  ]
  # Flags must be parsed from the rewritten argv exactly once, but whether that
  # happens depends on the context:
  #
  # In OSS testing, flags are still unparsed at this point. `absltest.main()`
  # below will invoke `app.run()`, which parses the rewritten argv. Avoid
  # parsing argv here to prevent duplicates in multi-value flags.
  #
  # In Google-internal testing, this function already runs inside `app.run()`,
  # which parsed the argv inherited from the parent. Inside `app.run()`,
  # `absltest.main()` only strips flags from argv and does not parse them.
  # Without the re-parse below, flags passed to the child will be silently
  # dropped, causing the worker tests to skip.
  if flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv)

  absltest.main()


class SubprocessTestMixin:
  """Mixin providing `run_sub_test` for parent test classes.

  The mixin is intentionally not a full `TestCase` subclass so it can be
  freely composed with any test base class (e.g., `RepeatableTest`).

  Subclasses must set the class attribute `WORKER_TEST_METHOD_TEMPLATE` to
  a format string that produces the target test method name when formatted
  with `mode=<TestMode enum member>`.

  Example::

      class ParentCacheTest(
          subprocess_test_utils.SubprocessTestMixin,
          seed_test_utils.MultiProcessRepeatableTest,
      ):
        WORKER_TEST_METHOD_TEMPLATE = (
            "CacheTest.test_persistent_cache_{mode.value}"
        )
  """

  # Subclasses must override with a format string, e.g.
  # "CacheTest.test_persistent_cache_{mode.value}"
  WORKER_TEST_METHOD_TEMPLATE: str

  def run_sub_test(
      self,
      mode: enum.Enum,
      env_updates: Mapping[str, str] | None = None,
      env_removals: Sequence[str] | None = None,
  ) -> None:
    """Spawns an isolated child subprocess to run a single test method.

    Args:
      mode: The test-mode enum member selecting the child test method.
      env_updates: Environment variables to set in the child process.
      env_removals: Environment variable keys to remove in the child process.
    """
    if _device_ops_backend._is_initialized():  # pylint: disable=protected-access
      raise AssertionError(
          "The parent process initialized the TPU runtime, so the child process"
          " cannot. To fix the test setup, make sure to\n"
          "  1. Derive the parent test class from seed_test_utils.MultiProcessRepeatableTest,\n"  # pylint: disable=line-too-long
          "  2. Skip worker tests before calling super().setUp()."
      )

    logging.info("Running subtest %s in subprocess for isolation.", mode.name)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=sub_test_worker_entry,
        args=(q, mode, self.WORKER_TEST_METHOD_TEMPLATE),
    )
    # Apply requested environment mutations before the child process, which
    # copies `os.environ`, starts. It ensures environment variables take effect
    # at import time.
    with mock.patch.dict(os.environ):
      for key in env_removals or ():
        os.environ.pop(key, None)
      os.environ.update(env_updates or {})
      p.start()

    while p.is_alive() or not q.empty():
      try:
        output = q.get(timeout=0.1)
        sys.stderr.write(output)
        sys.stderr.flush()
      except queue.Empty:
        continue

    p.join()
    # Prefer raising an exception over providing a fail() stub: if the concrete
    # test class does not override `fail()`, the default stub on this mixin
    # would shadow TestCase.fail() in MRO and silently pass the test even
    # when the child process aborts.
    if p.exitcode != 0:
      raise AssertionError(
          f"Subtest {mode.name} failed with exit code {p.exitcode}."
      )
