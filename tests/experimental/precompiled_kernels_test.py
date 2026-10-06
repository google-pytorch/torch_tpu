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

"""Persistent-cache tests for `torch.tpu.experimental.precompiled_kernels`.

This test file uses a two-stage execution model (following `cache_test.py`) to
test persistent Tier-2 / Tier-3 caching of precompiled kernels in isolated
subprocesses. In-memory prefilling is covered in-process by
`precompiled_kernels_prefill_test.py`.

1. Parent Stage (`ParentLoadPrecompiledKernelsTest`):
   - When executed by standard test runners (without `--test_mode`), the flag
     `_TEST_MODE.value` is None.
   - `LoadPrecompiledKernelsTest` skips all of its tests before initializing
     the TPU runtime because `_TEST_MODE` is unset.
   - `ParentLoadPrecompiledKernelsTest` runs its test methods:
     * `test_tier2_cache`: spawns one subprocess to compile and populate the
       Tier-2 cache on the fly, then spawns a second subprocess to load and
       execute the kernel from the Tier-2 cache without recompiling.
     * `test_tier3_cache`: spawns one subprocess to compile and populate the
       Tier-3 cache on the fly, then spawns a second subprocess (with a fresh,
       empty Tier-2 cache) to load and execute the kernel from the Tier-3 cache
       without recompiling.

2. Worker Stage (`LoadPrecompiledKernelsTest` in subprocess):
   - Each worker subprocess starts with the cache environment variables that
     the parent requested and runs the single target test method specified by
     `--test_mode`.
"""

import enum
import pathlib
import shutil
import time
import uuid

from absl import flags
from absl import logging
from absl.testing import absltest
import torch
from torch.tpu.experimental import precompiled_kernels  # pyrefly: ignore[missing-import]
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests import subprocess_test_utils

# Must match kTorchTpuTier2CompilationCacheEnvVar in csrc/common/env_vars.h.
_TIER2_CACHE_ENV_VAR = "TORCH_TPU_TIER2_COMPILATION_CACHE"
# Must match kTorchTpuTier3CompilationCacheRootEnvVar in csrc/common/env_vars.h.
_TIER3_CACHE_ROOT_ENV_VAR = "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
# Must match kTorchTpuInternalTier3CompilationCacheLocalBackupTaskEnvVar in
# csrc/common/env_vars.h.
_TIER3_LOCAL_BACKUP_TASK_ENV_VAR = (
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK"
)
# LINT.IfChange(tier2_cache_root_dir)
# Must match kTier2CacheRootDir in csrc/common/tier2_compilation_cache.cc.
_TIER2_CACHE_ROOT_DIR = pathlib.Path("/dev/shm/torch_tpu_cache")
# LINT.ThenChange()


class TestMode(enum.Enum):
  """Enum for the different worker test modes."""

  POPULATE_CACHE = "populate_cache"
  USE_CACHE = "use_cache"


_TEST_MODE = flags.DEFINE_enum_class(
    "test_mode", None, TestMode, "The worker test mode to run."
)


def get_tier2_cache_dir(tier2_cache_name: str) -> pathlib.Path:
  """Returns the Tier-2 cache directory."""
  return _TIER2_CACHE_ROOT_DIR / tier2_cache_name


def list_bin_files(dir_path: pathlib.Path) -> list[pathlib.Path]:
  """Lists all .bin files in the given directory and its subdirectories."""
  return list(dir_path.glob("**/*.bin"))


class LoadPrecompiledKernelsTest(seed_test_utils.RepeatableTest):
  """Worker tests for precompiled kernels and persistent caching.

  These test methods are intended to run only inside isolated child worker
  subprocesses where `--test_mode` has been explicitly set.
  """

  def setUp(self) -> None:
    # When running in the parent process, --test_mode is None. Skip these tests
    # before calling super().setUp() so the parent process does not initialize
    # the TPU runtime.
    if _TEST_MODE.value is None:
      self.skipTest("test_mode flag not provided.")
    super().setUp()
    tt_testing.reset_eager_state()

  def test_populate_cache(self) -> None:
    """Compiles the precompiled kernels and populates the configured cache."""
    self.assertEqual(_TEST_MODE.value, TestMode.POPULATE_CACHE)
    precompiled_kernels.load_kernels()

  def test_use_cache(self) -> None:
    """Loads the precompiled kernels from persistent cache without compiling."""
    self.assertEqual(_TEST_MODE.value, TestMode.USE_CACHE)
    tensor = torch.zeros((128, 128), dtype=torch.float32)
    tensor[:64, :64] = 1
    expected = torch.nonzero(tensor)
    tensor = tensor.to("tpu")

    num_compilations_before = torch.tpu._get_cache_stats().num_compilations
    start = time.perf_counter()
    precompiled_kernels.load_kernels()
    self.assertEqual(
        torch.tpu._get_cache_stats().num_compilations, num_compilations_before
    )

    dynamic_res = torch.nonzero(tensor).cpu()
    logging.info("Nonzero kernel took %s s", time.perf_counter() - start)
    test_utils.assert_close(dynamic_res, expected)
    self.assertEqual(torch.tpu._get_cache_hits(), 1)


class ParentLoadPrecompiledKernelsTest(
    subprocess_test_utils.SubprocessTestMixin,
    seed_test_utils.MultiProcessRepeatableTest,
):
  """Parent test that spawns isolated subprocesses for each test case."""

  WORKER_TEST_METHOD_TEMPLATE = "LoadPrecompiledKernelsTest.test_{mode.value}"

  def setUp(self) -> None:
    super().setUp()
    tt_testing.reset_eager_state()
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping parent test in sub-test mode.")

  def _create_tier2_cache_name(self) -> str:
    """Returns a unique tier-2 cache name and registers cleanup for it."""
    name = f"precompiled_kernels_test_{uuid.uuid4().hex}"
    self.addCleanup(
        shutil.rmtree, get_tier2_cache_dir(name), ignore_errors=True
    )
    return name

  def test_tier2_cache(self) -> None:
    """Populates a Tier-2 cache in one process and uses it in a second process."""
    tier2_cache_name = self._create_tier2_cache_name()
    tier2_cache_dir = get_tier2_cache_dir(tier2_cache_name)
    env_updates = {_TIER2_CACHE_ENV_VAR: tier2_cache_name}
    env_removals = [_TIER3_CACHE_ROOT_ENV_VAR]

    with self.subTest("PopulateTier2Cache"):
      self.run_sub_test(
          TestMode.POPULATE_CACHE,
          env_updates=env_updates,
          env_removals=env_removals,
      )
      self.assertNotEmpty(
          list_bin_files(tier2_cache_dir),
          f"No .bin cache files written to {tier2_cache_dir}",
      )

    with self.subTest("UseTier2Cache"):
      self.run_sub_test(
          TestMode.USE_CACHE,
          env_updates=env_updates,
          env_removals=env_removals,
      )

  def test_tier3_cache(self) -> None:
    """Populates a Tier-3 cache in one process and uses it in a second process."""
    tier3_cache_dir = pathlib.Path(self.create_tempdir().full_path)

    with self.subTest("PopulateTier3Cache"):
      populate_tier2_name = self._create_tier2_cache_name()
      self.run_sub_test(
          TestMode.POPULATE_CACHE,
          env_updates={
              _TIER2_CACHE_ENV_VAR: populate_tier2_name,
              _TIER3_CACHE_ROOT_ENV_VAR: str(tier3_cache_dir),
          },
      )
      self.assertNotEmpty(
          list_bin_files(tier3_cache_dir),
          f"No .bin cache files written to {tier3_cache_dir}",
      )

    with self.subTest("UseTier3Cache"):
      # The tier-3 cache is only read on a tier-2 miss. Use a fresh, initially
      # empty tier-2 cache in the second process to guarantee a tier-2 miss.
      use_tier2_name = self._create_tier2_cache_name()
      use_tier2_dir = get_tier2_cache_dir(use_tier2_name)
      self.run_sub_test(
          TestMode.USE_CACHE,
          env_updates={
              _TIER2_CACHE_ENV_VAR: use_tier2_name,
              _TIER3_CACHE_ROOT_ENV_VAR: str(tier3_cache_dir),
              # A tier-3 read schedules a backup compilation by default, which
              # races with the read and may increment `num_compilations` even
              # on a tier-3 hit. Disable it so the worker can assert that no
              # compilation happened.
              _TIER3_LOCAL_BACKUP_TASK_ENV_VAR: "false",
          },
      )
      self.assertNotEmpty(
          list_bin_files(use_tier2_dir),
          f"Expected tier-2 cache {use_tier2_dir} to be backfilled from tier-3",
      )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
