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

"""Tests persistent Tier-2 and Tier-3 cache for torch.compile."""

import contextlib
import enum
import os
import queue
import sys
from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu import api
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

FLAGS = flags.FLAGS


class TestMode(enum.Enum):
  TIER2 = "tier2"
  TIER3 = "tier3"
  DISABLED = "disabled"


_TEST_MODE = flags.DEFINE_enum_class(
    "test_mode", None, TestMode, "The test mode to run."
)


def eager_mode_defer_all() -> contextlib._GeneratorContextManager[None]:
  """Enable eager mode with internal defer all."""
  return execution_mode.eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL)


class CachingTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    if _TEST_MODE.value is None:
      self.skipTest("test_mode flag not provided.")

  def _run_cache_test(self, expected_hit_increment: int) -> None:
    """Run the cache test and verifies hits."""
    with eager_mode_defer_all():
      x = torch.ones(10, device="cpu").to(device=api.tpu_device())
      y = torch.ones(10, device="cpu").to(device=api.tpu_device())
      z = x + y
    result_tensors = [z]
    argument_tensors = [x, y]

    traversal = tpu_torch_compile.build_traversal(
        result_tensors, argument_tensors
    )
    self.assertIsNotNone(traversal)

    # Clear in-memory cache to ensure clean slate
    torch.tpu._clear_cache()
    initial_hits = torch.tpu._get_cache_hits()

    # First compile (cache miss)
    exe1 = tpu_torch_compile.compile_traversal(traversal, fast_compile=False)
    self.assertEqual(torch.tpu._get_cache_hits(), initial_hits)

    # Clear in-memory cache to force persistent cache lookup
    torch.tpu._clear_cache()

    # Second compile
    exe2 = tpu_torch_compile.compile_traversal(traversal, fast_compile=False)
    self.assertEqual(
        torch.tpu._get_cache_hits(),
        initial_hits + expected_hit_increment,
    )

    self.assertIsNotNone(exe1)
    self.assertIsNotNone(exe2)

  def test_tier2_cache(self):
    if _TEST_MODE.value != TestMode.TIER2:
      self.skipTest("Skipping since test_mode is not tier2")

    self.assertEqual(
        os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"), "default"
    )
    self._run_cache_test(expected_hit_increment=1)

  def test_tier3_cache(self):
    if _TEST_MODE.value != TestMode.TIER3:
      self.skipTest("Skipping since test_mode is not tier3")

    self.assertIsNotNone(
        os.environ.get("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT")
    )
    self._run_cache_test(expected_hit_increment=1)

  def test_disabled_caching(self):
    if _TEST_MODE.value != TestMode.DISABLED:
      self.skipTest("Skipping since test_mode is not disabled")

    self.assertEqual(
        os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"), "disabled"
    )
    self._run_cache_test(expected_hit_increment=0)


class QueueWriter:
  """A helper to write stdout/stderr to a queue for multiprocessing."""

  def __init__(self, q: queue.Queue[str]):
    self.q = q

  def write(self, s: str | None):
    if s:
      self.q.put(s)

  def flush(self):
    del self


def _worker_entry(q, mode, env_updates) -> None:
  """Serves as the entry point for the worker process."""

  # Update env in child process
  os.environ.update(env_updates)

  qw = QueueWriter(q)
  sys.stdout = qw
  sys.stderr = qw

  test_method = (
      f"test_{mode}_cache" if mode != "disabled" else "test_disabled_caching"
  )
  sys.argv = [
      sys.argv[0],
      f"--test_mode={mode.upper()}",
      f"CachingTest.{test_method}",
  ]

  absltest.main()


class MasterCachingTest(parameterized.TestCase, absltest.TestCase):
  """A master test that spawns subprocesses for isolation."""

  def setUp(self):
    super().setUp()
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping master test in sub-test mode.")

  def _run_sub_test(self, mode, env_updates) -> None:
    """Runs a sub-test in a separate process."""

    logging.info("Running test in a subprocess for isolation.")
    ctx = g3_multiprocessing.get_context(g3_multiprocessing.ABSL_SPAWN)
    q = ctx.Queue()
    p = ctx.Process(
        target=_worker_entry,
        args=(q, mode, env_updates),
    )
    p.start()

    while p.is_alive() or not q.empty():
      try:
        output = q.get(timeout=0.1)
        sys.stderr.write(output)
        sys.stderr.flush()
      except queue.Empty:
        continue

    p.join()
    if p.exitcode != 0:
      self.fail(f"caching_{mode}_test failed with exit code {p.exitcode}")

  @parameterized.named_parameters(
      dict(
          testcase_name="tier2",
          mode="tier2",
          env_updates={
              "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "default",
          },
      ),
      dict(
          testcase_name="tier3",
          mode="tier3",
          env_updates={
              "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "default",
              "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "USE_TEMP_DIR",
          },
      ),
      dict(
          testcase_name="disabled",
          mode="disabled",
          env_updates={
              "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "disabled"
          },
      ),
  )
  def test_caching(self, mode, env_updates):
    if (
        env_updates.get("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT")
        == "USE_TEMP_DIR"
    ):
      env_updates["TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"] = (
          self.create_tempdir().full_path
      )
    self._run_sub_test(mode, env_updates)


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
