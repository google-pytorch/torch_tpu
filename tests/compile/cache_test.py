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

"""Tests persistent Tier-2 and Tier-3 cache for torch.compile.

This test file uses a two-stage execution model to test compilation caching in
isolated subprocesses:

1. Parent Stage (`ParentCacheTest`):
   - When executed by standard test runners (without `--test_mode`), the flag
     `_TEST_MODE.value` is None.
   - `CacheTest` skips all of its tests because `_TEST_MODE` is unset.
   - `ParentCacheTest` runs its test methods (`test_cache_tier2`,
     `test_cache_tier3`), each of which spawns a child worker subprocess using
     `multiprocessing` to execute `sub_test_worker_entry`.

2. Worker Stage (`CacheTest` in subprocess):
   - Each worker subprocess starts with the cache environment variables that
     the parent requested. `sub_test_worker_entry` sets the `--test_mode` flag
     (e.g., `tier2` or `tier3`) and modifies `sys.argv` so that
     `absltest.main()` runs only the target `CacheTest` method.
   - When `absltest` runs in the worker subprocess, `ParentCacheTest` skips
     itself because `_TEST_MODE` is now set, while `CacheTest` executes the
     requested test method.

Why subprocesses are needed:
   `torch.compile` and TorchTPU cache initialization rely on process-global
   state. Running each test mode in a clean, isolated subprocess ensures that
   environment variables and in-memory/on-disk cache states do not leak between
   different test cases.

See `torch_tpu.tests.subprocess_test_utils` for shared execution scaffolding,
as well as instructions and examples for running the full test suite or
debugging individual test modes directly.
"""

import enum
import os
import pathlib
import shutil
import uuid

from absl import flags
from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests import subprocess_test_utils


class TestMode(enum.Enum):
  """Enum for the different test modes."""

  TIER2 = "tier2"
  TIER3 = "tier3"


# The --test_mode flag controls whether the process runs as the parent
# test (when None) or as a worker subprocess running a specific CacheTest mode
# (when set to TIER2 or TIER3).
_TEST_MODE = flags.DEFINE_enum_class(
    "test_mode", None, TestMode, "The test mode to run."
)


def get_tier2_cache_dir(tier2_cache_name: str) -> pathlib.Path:
  """Returns the Tier-2 cache directory."""
  return pathlib.Path("/dev/shm/torch_tpu_cache") / tier2_cache_name


def clear_in_memory_caches() -> None:
  """Clears the in-memory caches."""
  torch.compiler.reset()
  torch.tpu._clear_cache()


def empty_dir(dir_path: pathlib.Path) -> None:
  """Empties the given directory."""
  try:
    shutil.rmtree(dir_path)
  except FileNotFoundError:
    pass
  dir_path.mkdir(parents=True, exist_ok=True)


def list_bin_files(dir_path: pathlib.Path) -> list[pathlib.Path]:
  """Lists all .bin files in the given directory and its subdirectories."""
  return list(dir_path.glob("**/*.bin"))


class CacheTest(seed_test_utils.RepeatableTest):
  """Tests persistent Tier-2 and Tier-3 cache for torch.compile.

  These test methods are intended to run only inside isolated child worker
  subprocesses where `--test_mode` has been explicitly set.
  """

  def setUp(self) -> None:
    # When running in the parent process, --test_mode is None. Skip these tests
    # so that only ParentCacheTest executes in the parent process.
    if _TEST_MODE.value is None:
      self.skipTest("test_mode flag not provided.")
    super().setUp()
    tt_testing.reset_eager_state()

  def _run_persistent_cache_logic(self, cache_dir: pathlib.Path | str):
    """Verifies that compilation artifacts persist to disk and hit on reload.

    This helper performs a two-step verification:
    1. First compile (Cache Miss): Compiles and runs `fn`, asserting that new
       cache binary files (.bin) are written to `cache_dir`.
    2. Second compile (Cache Hit): Clears in-memory caches and re-runs `fn`,
       asserting that persistent cache lookup succeeds and no new cache files
       are written to disk.
    """

    device = torch.accelerator.current_accelerator()

    def get_args() -> tuple[torch.Tensor, torch.Tensor]:
      return (
          torch.ones((2, 2), device=device),
          torch.ones((2, 2), device=device),
      )

    # Warmup/initialize by calling args once to populate any initial cache
    # caused by tensor creation, generator initialization, or other
    # torch_tpu initializations.
    for arg in get_args():
      _ = arg.cpu()
    _ = torch.tpu.get_rng_state().cpu()

    # Clear in-memory caches before starting so we only measure persistent
    # cache artifacts generated by the explicit compile below.
    clear_in_memory_caches()

    # Record existing .bin files before the first compile.
    initial_caches = set(list_bin_files(cache_dir))
    logging.info("Initial caches: %s", initial_caches)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      return torch.sin(x + y) * torch.cos(x - y)

    with self.subTest("TestCachePopulation"):
      # First compile: Should trigger a compilation cache miss and write new
      # persistent cache files (.bin) to cache_dir.
      with torch.inference_mode():
        fn(*get_args())

      caches_after_first_compile = set(list_bin_files(cache_dir))
      logging.info("Caches after first compile: %s", caches_after_first_compile)
      self.assertNotEmpty(
          caches_after_first_compile - initial_caches,
          "No new caches written after first compile",
      )

    # Clear in-memory cache to force a persistent cache lookup from disk.
    clear_in_memory_caches()

    with self.subTest("TestCacheHit"):
      # Second compile: Should trigger a persistent cache hit from disk.
      # No additional cache files should be created.
      with torch.inference_mode():
        fn(*get_args())

      caches_after_second_compile = set(list_bin_files(cache_dir))
      logging.info(
          "Caches after second compile: %s", caches_after_second_compile
      )
      self.assertEmpty(
          caches_after_second_compile - caches_after_first_compile,
          "New caches written after second compile",
      )

  def test_persistent_cache_tier2(self) -> None:
    """Tests Tier-2 persistent compilation caching in shared memory."""
    self.assertEqual(_TEST_MODE.value, TestMode.TIER2)
    cache_dir = get_tier2_cache_dir(
        os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE")
    )
    empty_dir(cache_dir)
    self._run_persistent_cache_logic(cache_dir)

  def test_persistent_cache_tier3(self) -> None:
    """Tests Tier-3 persistent compilation caching in a temporary root dir."""
    self.assertEqual(_TEST_MODE.value, TestMode.TIER3)
    cache_dir = pathlib.Path(
        os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")
    )
    empty_dir(cache_dir)
    self._run_persistent_cache_logic(cache_dir)


class ParentCacheTest(
    subprocess_test_utils.SubprocessTestMixin,
    seed_test_utils.MultiProcessRepeatableTest,
):
  """A parent test that spawns subprocesses for isolation.

  When invoked by the standard test runner without `--test_mode`, this class
  executes and spawns an isolated subprocess for each cache tier test.
  When running inside a worker subprocess (where `--test_mode` is set),
  `ParentCacheTest` skips itself to prevent recursion.
  """

  WORKER_TEST_METHOD_TEMPLATE = "CacheTest.test_persistent_cache_{mode.value}"

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    # In worker subprocesses (--test_mode is set), skip ParentCacheTest so only
    # the target CacheTest method is executed.
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping parent test in sub-test mode.")

  def _create_tier2_cache_name(self) -> str:
    """Returns a tier-2 cache name that no other process uses.

    The tier-2 cache lives in /dev/shm and outlives the test. A shared name
    such as "default" can already hold the compiled program, and concurrent
    runs of this test would empty each other's cache.
    """
    name = f"compile_cache_test_{uuid.uuid4().hex}"
    self.addCleanup(
        shutil.rmtree, get_tier2_cache_dir(name), ignore_errors=True
    )
    return name

  def test_cache_tier2(self) -> None:
    """Spawns an isolated worker subprocess to run the Tier-2 cache test."""
    self.run_sub_test(
        TestMode.TIER2,
        env_updates={
            # Disable PyTorch AOTAutograd cache because it persists TorchTPU executables
            # under /tmp across runs. A hit there bypasses TorchTPU compilation cache.
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TORCH_TPU_TIER2_COMPILATION_CACHE": (
                self._create_tier2_cache_name()
            ),
        },
    )

  def test_cache_tier3(self) -> None:
    """Spawns an isolated worker subprocess to run the Tier-3 cache test."""
    self.run_sub_test(
        TestMode.TIER3,
        env_updates={
            # Disable PyTorch AOTAutograd cache because it persists TorchTPU executables
            # under /tmp across runs. A hit there bypasses TorchTPU compilation cache.
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TORCH_TPU_TIER2_COMPILATION_CACHE": (
                self._create_tier2_cache_name()
            ),
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(
                self.create_tempdir().full_path
            ),
        },
    )


if __name__ == "__main__":
  # Use multiprocessing.handle_test_main to ensure compatibility with
  # absltest in multi-process Google3 test environments.
  multiprocessing.handle_test_main(absltest.main)
