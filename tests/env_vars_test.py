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

"""Unit tests for TorchTPU environment variables and experimental warnings.

This test file uses a two-stage execution model to test environment variable
warnings in isolated subprocesses:

1. Parent Stage (`ParentEnvVarsTest`):
   - When executed by standard test runners (without `--test_mode`), the flag
     `_TEST_MODE.value` is None.
   - `EnvVarsTest` skips all of its tests because `_TEST_MODE` is unset.
   - `ParentEnvVarsTest` runs its test methods, each of which spawns a child
     worker subprocess using `multiprocessing` to execute
     `sub_test_worker_entry`.

2. Worker Stage (`EnvVarsTest` in subprocess):
   - In each worker subprocess, `sub_test_worker_entry` sets the `--test_mode`
     flag, configures environment variables, and rewrites `sys.argv` so that
     `absltest.main()` runs only the target `EnvVarsTest` method.
   - When `absltest` runs in the worker subprocess, `ParentEnvVarsTest` skips
     itself because `_TEST_MODE` is now set, while `EnvVarsTest` executes the
     requested test method.

Why subprocesses are needed:
   `GetEnvOnce<name>()` in TorchTPU C++ caches environment variable values in
   process-global static storage on first read and emits `TORCH_WARN_ONCE`.
   Running each test scenario in a clean, isolated subprocess ensures that
   environment variables and memoized C++ static states do not leak between
   different test cases.

See `torch_tpu.tests.subprocess_test_utils` for shared execution scaffolding,
as well as instructions and examples for running the full test suite or
debugging individual test modes directly.
"""

import enum
import warnings

from absl import flags
from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests import subprocess_test_utils


class TestMode(enum.Enum):
  """Enum for the different test modes."""

  TIER2_WARN_ONCE = "tier2_warn_once"
  TIER2_DISABLED = "tier2_disabled"
  TIER2_UNSET = "tier2_unset"
  TIER3_WARN_ONCE = "tier3_warn_once"
  NON_EXPERIMENTAL = "non_experimental"


_TEST_MODE = flags.DEFINE_enum_class(
    "test_mode", None, TestMode, "The test mode to run."
)


class EnvVarsTest(seed_test_utils.RepeatableTest):
  """Tests that experimental environment variables trigger Python warnings.

  These test methods are intended to run only inside isolated child worker
  subprocesses where `--test_mode` has been explicitly set.
  """

  def setUp(self) -> None:
    if _TEST_MODE.value is None:
      self.skipTest("test_mode flag not provided.")
    super().setUp()
    tt_testing.reset_eager_state()

  def test_tier2_warn_once(self) -> None:
    """Verifies TORCH_TPU_TIER2_COMPILATION_CACHE warns once on read."""
    device = torch.accelerator.current_accelerator()
    with warnings.catch_warnings(record=True) as recorded_warnings:
      warnings.simplefilter("always")

      # First TPU operation triggers compilation, reading the env var.
      t = torch.ones((2, 2), device=device)
      res1 = t + t
      res1.cpu()

      matching_warnings = [
          w
          for w in recorded_warnings
          if "TORCH_TPU_TIER2_COMPILATION_CACHE" in str(w.message)
      ]
      self.assertLen(
          matching_warnings,
          1,
          f"Expected exactly 1 warning, got {len(matching_warnings)}:"
          f" {[str(w.message) for w in recorded_warnings]}",
      )
      warning_obj = matching_warnings[0]
      self.assertTrue(issubclass(warning_obj.category, UserWarning))
      self.assertIn(
          "the TORCH_TPU_TIER2_COMPILATION_CACHE environment variable is an"
          " experimental feature and may change or be removed without notice.",
          str(warning_obj.message),
      )

      # Second TPU operation must NOT emit another warning for the same env var.
      res2 = t * t
      res2.cpu()

      matching_warnings_after = [
          w
          for w in recorded_warnings
          if "TORCH_TPU_TIER2_COMPILATION_CACHE" in str(w.message)
      ]
      self.assertLen(
          matching_warnings_after,
          1,
          "Second read must not trigger an additional warning",
      )

  def test_tier2_disabled(self) -> None:
    """Verifies TORCH_TPU_TIER2_COMPILATION_CACHE='disabled' still warns as experimental."""
    device = torch.accelerator.current_accelerator()
    with warnings.catch_warnings(record=True) as recorded_warnings:
      warnings.simplefilter("always")

      t = torch.ones((2, 2), device=device)
      res = t + t
      res.cpu()

      matching_warnings = [
          w
          for w in recorded_warnings
          if "TORCH_TPU_TIER2_COMPILATION_CACHE" in str(w.message)
      ]
      self.assertLen(matching_warnings, 1)
      self.assertIn(
          "TORCH_TPU_TIER2_COMPILATION_CACHE environment variable is an"
          " experimental feature",
          str(matching_warnings[0].message),
      )

  def test_tier2_unset(self) -> None:
    """Verifies unset TORCH_TPU_TIER2_COMPILATION_CACHE does NOT trigger a warning."""
    device = torch.accelerator.current_accelerator()
    with warnings.catch_warnings(record=True) as recorded_warnings:
      warnings.simplefilter("always")

      t = torch.ones((2, 2), device=device)
      res = t + t
      res.cpu()

      matching_warnings = [
          w
          for w in recorded_warnings
          if "TORCH_TPU_TIER2_COMPILATION_CACHE" in str(w.message)
      ]
      self.assertEmpty(
          matching_warnings,
          "Expected no warning when TORCH_TPU_TIER2_COMPILATION_CACHE is unset,"
          f" got: {[str(w.message) for w in matching_warnings]}",
      )

  def test_tier3_warn_once(self) -> None:
    """Verifies TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT warns once on read."""
    device = torch.accelerator.current_accelerator()
    with warnings.catch_warnings(record=True) as recorded_warnings:
      warnings.simplefilter("always")

      t = torch.ones((2, 2), device=device)
      res = t + t
      res.cpu()

      matching_warnings = [
          w
          for w in recorded_warnings
          if "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" in str(w.message)
      ]
      self.assertLen(
          matching_warnings,
          1,
          "Expected exactly 1 warning for tier-3 cache root, got"
          f" {len(matching_warnings)}",
      )
      warning_obj = matching_warnings[0]
      self.assertTrue(issubclass(warning_obj.category, UserWarning))
      self.assertIn(
          "the TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT environment variable is"
          " an experimental feature and may change or be removed without"
          " notice.",
          str(warning_obj.message),
      )

  def test_non_experimental(self) -> None:
    """Verifies non-experimental env vars do not emit experimental warnings."""
    device = torch.accelerator.current_accelerator()
    with warnings.catch_warnings(record=True) as recorded_warnings:
      warnings.simplefilter("always")

      t = torch.ones((2, 2), device=device)
      res = t + t
      res.cpu()

      matching_warnings = [
          w
          for w in recorded_warnings
          if "TORCH_SHOW_CPP_STACKTRACES" in str(w.message)
      ]
      self.assertEmpty(
          matching_warnings,
          "Expected no warning for non-experimental env var"
          f" TORCH_SHOW_CPP_STACKTRACES, got: {matching_warnings}",
      )


class ParentEnvVarsTest(
    subprocess_test_utils.SubprocessTestMixin,
    seed_test_utils.MultiProcessRepeatableTest,
):
  """Parent test running isolated worker subprocesses for each test case."""

  WORKER_TEST_METHOD_TEMPLATE = "EnvVarsTest.test_{mode.value}"

  def setUp(self) -> None:
    super().setUp()
    tt_testing.reset_eager_state()
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping parent test in sub-test mode.")

  @absltest.skip("b/567323733 - The warning does not reach Python `warnings`.")
  def test_tier2_compilation_cache_warns_once_on_read(self) -> None:
    """Tests TORCH_TPU_TIER2_COMPILATION_CACHE warning in isolated subprocess."""
    self.run_sub_test(
        TestMode.TIER2_WARN_ONCE,
        env_updates={"TORCH_TPU_TIER2_COMPILATION_CACHE": "my_tier2_cache"},
    )

  @absltest.skip("b/567323733 - The warning does not reach Python `warnings`.")
  def test_tier2_compilation_cache_disabled_warns_on_read(self) -> None:
    """Tests TORCH_TPU_TIER2_COMPILATION_CACHE='disabled' warning in isolated subprocess."""
    self.run_sub_test(
        TestMode.TIER2_DISABLED,
        env_updates={"TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled"},
    )

  def test_tier2_compilation_cache_unset_no_warning(self) -> None:
    """Tests that unset TORCH_TPU_TIER2_COMPILATION_CACHE does not warn in isolated subprocess."""
    self.run_sub_test(
        TestMode.TIER2_UNSET,
        env_removals=["TORCH_TPU_TIER2_COMPILATION_CACHE"],
    )

  @absltest.skip("b/567323733 - The warning does not reach Python `warnings`.")
  def test_tier3_compilation_cache_root_warns_once_on_read(self) -> None:
    """Tests TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT warning in isolated subprocess."""
    temp_dir = self.create_tempdir().full_path
    self.run_sub_test(
        TestMode.TIER3_WARN_ONCE,
        env_updates={
            "TORCH_TPU_TIER2_COMPILATION_CACHE": "default",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(temp_dir),
        },
    )

  def test_non_experimental_env_var_no_warning(self) -> None:
    """Tests that non-experimental env vars do not warn in isolated subprocess."""
    self.run_sub_test(
        TestMode.NON_EXPERIMENTAL,
        env_updates={"TORCH_SHOW_CPP_STACKTRACES": "1"},
        env_removals=[
            "TORCH_TPU_TIER2_COMPILATION_CACHE",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        ],
    )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
