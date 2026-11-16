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

"""Tests for `continuous_test_utils`."""

import os
from typing import Final
import unittest
from unittest import mock

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal import common_distributed
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import continuous_test_utils

_FAILURE_MESSAGE: Final[str] = "DELIBERATE_FAILURE_ON_RANK_3"
_SKIP_MESSAGE: Final[str] = "DELIBERATE_SKIP_IN_WORKERS"


class _Harness(continuous_test_utils.TpuMultiProcContinuousTest):
  """Skips in the workers, runs a collective, then fails on a single rank.

  Its methods have no `test` prefix, so the test runner doesn't collect them.
  `TpuMultiProcContinuousTestTest` runs them explicitly, in this order, on one
  set of workers (spawning workers is most of this test's runtime). The failing
  method runs last because a failure skips the rest of the class.
  """

  world_size = 8

  def skip_in_workers(self) -> None:
    self.skipTest(_SKIP_MESSAGE)

  def fail_on_rank_3(self) -> None:
    dist.all_reduce(torch.ones(1, device="tpu"))
    if self.rank == 3:
      raise AssertionError(_FAILURE_MESSAGE)


class TpuMultiProcContinuousTestTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.result = unittest.TestResult()
    unittest.TestSuite([
        _Harness("skip_in_workers"),
        _Harness("fail_on_rank_3"),
    ]).run(cls.result)

  def test_worker_skip_is_reported_as_skip(self) -> None:
    """A skip raised in the workers is a skip, and later tests still run."""
    self.assertEqual(self.result.testsRun, 2)
    self.assertLen(self.result.skipped, 1)
    test, reason = self.result.skipped[0]
    self.assertEqual(test.id(), _Harness("skip_in_workers").id())
    self.assertIn(_SKIP_MESSAGE, reason)

  def test_single_rank_failure_is_reported_without_hanging(self) -> None:
    """A failure on one rank is reported, and the workers shut down.

    Guards the failure handling in `TpuMultiProcContinuousTest`:
    - The barrier between tests also runs when a test fails. Otherwise the other
      ranks block in it and the failure is never reported.
    - `tearDownClass()` kills the workers after a failure. Otherwise it waits
      forever for workers that can't exit cleanly.

    A regression shows up as this test timing out.
    """
    problems = self.result.errors + self.result.failures
    self.assertLen(problems, 1)
    test, traceback_str = problems[0]
    self.assertEqual(test.id(), _Harness("fail_on_rank_3").id())
    self.assertIn(_FAILURE_MESSAGE, traceback_str)

  def test_contract_invariants(self):
    """Guards against regressions in MultiProcessRepeatableTest invariants.

    If someone enables `seed_in_setup` or `synchronize_tensors_in_tear_down`,
    the main process will initialize the PJRT client / claim TPU devices,
    preventing child workers from acquiring them.
    """
    test_instance = _Harness("fail_on_rank_3")
    self.assertFalse(test_instance.seed_in_setup)
    self.assertFalse(test_instance.synchronize_tensors_in_tear_down)
    self.assertIsInstance(
        test_instance, seed_test_utils.MultiProcessRepeatableTest
    )


class _TimeoutHarness(continuous_test_utils.TpuMultiProcContinuousTest):
  """Harness whose class fixtures are called directly; spawns no workers."""

  world_size = 1

  def noop(self) -> None:
    pass


class TpuMultiProcContinuousTestTimeoutTest(
    seed_test_utils.MultiProcessRepeatableTest
):
  """Checks the harness's per-test-case worker timeout."""

  def _harness_wait_timeout(self) -> int:
    """Returns the timeout the harness would use to wait on workers."""
    return common_distributed.get_timeout(_TimeoutHarness("noop").id())

  def _set_up_class_with_env(self, **env: str) -> None:
    """Runs `_TimeoutHarness.setUpClass()` with only `env` timeout variables."""
    self.addCleanup(_TimeoutHarness.doClassCleanups)
    with mock.patch.dict(os.environ):
      os.environ.pop("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", None)
      os.environ.update(env)
      _TimeoutHarness.setUpClass()

  def test_harness_timeout_is_effectively_unlimited(self) -> None:
    upstream_timeout = self._harness_wait_timeout()
    self._set_up_class_with_env()
    self.assertEqual(self._harness_wait_timeout(), 100_000)
    # The override is scoped to the class.
    _TimeoutHarness.doClassCleanups()
    self.assertEqual(self._harness_wait_timeout(), upstream_timeout)

  def test_distributed_tests_default_timeout_raises(self) -> None:
    upstream_timeout = self._harness_wait_timeout()
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Test helper error.
        ValueError, "DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"
    ):
      self._set_up_class_with_env(DISTRIBUTED_TESTS_DEFAULT_TIMEOUT="900")
    self.assertEqual(self._harness_wait_timeout(), upstream_timeout)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
