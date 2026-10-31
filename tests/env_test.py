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

"""Unit tests for TorchTPU environment and libtpu version handling."""

import enum
import os

from absl import flags
from absl.testing import absltest
import torch
from torch_tpu import _loader
from torch_tpu._internal import env
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests import subprocess_test_utils


class TestMode(enum.Enum):
  """Enum for the different test modes."""

  HAS_TORCH_TPU_ANCESTOR_PROCESS_CHILD = "has_torch_tpu_ancestor_process_child"
  HAS_TORCH_TPU_ANCESTOR_PROCESS_ROOT = "has_torch_tpu_ancestor_process_root"
  LIBTPU_VERSION = "libtpu_version"
  LIBTPU_VERSION_NOT_LOADED = "libtpu_version_not_loaded"
  GET_BOOL_ENV_ONCE_UNSET = "get_bool_env_once_unset"
  GET_BOOL_ENV_ONCE_TRUE = "get_bool_env_once_true"
  GET_BOOL_ENV_ONCE_FALSE = "get_bool_env_once_false"
  GET_BOOL_ENV_ONCE_INVALID = "get_bool_env_once_invalid"
  GET_BOOL_ENV_ONCE_MEMOIZED = "get_bool_env_once_memoized"
  GET_INT_ENV_ONCE_UNSET = "get_int_env_once_unset"
  GET_INT_ENV_ONCE_VALID = "get_int_env_once_valid"
  GET_INT_ENV_ONCE_INVALID = "get_int_env_once_invalid"
  GET_INT_ENV_ONCE_MEMOIZED = "get_int_env_once_memoized"
  GET_STR_ENV_ONCE_UNSET = "get_str_env_once_unset"
  GET_STR_ENV_ONCE_SET = "get_str_env_once_set"
  GET_STR_ENV_ONCE_EMPTY = "get_str_env_once_empty"
  GET_STR_ENV_ONCE_MEMOIZED = "get_str_env_once_memoized"


_TEST_MODE = flags.DEFINE_enum_class(
    "test_mode", None, TestMode, "The test mode to run."
)


class EnvTest(seed_test_utils.RepeatableTest):
  """Worker test class running individual test cases in isolated subprocesses."""

  def setUp(self) -> None:
    if _TEST_MODE.value is None:
      self.skipTest("test_mode flag not provided.")
    if _TEST_MODE.value in (
        TestMode.HAS_TORCH_TPU_ANCESTOR_PROCESS_CHILD,
        TestMode.HAS_TORCH_TPU_ANCESTOR_PROCESS_ROOT,
    ):
      self.seed_in_setup = False
      self.synchronize_tensors_in_tear_down = False
    super().setUp()

  def test_has_torch_tpu_ancestor_process_child(self) -> None:
    self.assertFalse(torch.tpu.is_initialized())
    self.assertTrue(env.has_torch_tpu_ancestor_process())
    self.assertFalse(torch.tpu.is_initialized())

  def test_has_torch_tpu_ancestor_process_root(self) -> None:
    self.assertFalse(torch.tpu.is_initialized())
    self.assertFalse(env.has_torch_tpu_ancestor_process())
    self.assertEqual(
        os.environ.get("TORCH_TPU_INTERNAL_TOPMOST_PID"), str(os.getpid())
    )
    self.assertFalse(torch.tpu.is_initialized())

  def test_libtpu_version(self) -> None:
    _loader.load()
    try:
      import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

      expected_version = libtpu.__version__
    except ImportError:
      expected_version = ""

    self.assertEqual(env.get_libtpu_version(), expected_version)

  def test_libtpu_version_not_loaded(self) -> None:
    self.assertIsNone(env.get_libtpu_version())

  def test_get_bool_env_once_unset(self) -> None:
    self.assertIsNone(env.get_bool_env_once("TORCH_TPU_ENABLE_DEBUG_CHECKS"))
    self.assertTrue(
        env.get_bool_env_once(
            "TORCH_TPU_ENABLE_DEBUG_CHECKS", default_value=True
        )
    )

  def test_get_bool_env_once_true(self) -> None:
    self.assertTrue(env.get_bool_env_once("TORCH_TPU_ENABLE_DEBUG_CHECKS"))
    # The default value is ignored when the variable is set.
    self.assertTrue(
        env.get_bool_env_once(
            "TORCH_TPU_ENABLE_DEBUG_CHECKS", default_value=False
        )
    )

  def test_get_bool_env_once_false(self) -> None:
    self.assertFalse(env.get_bool_env_once("TORCH_TPU_ENABLE_DEBUG_CHECKS"))

  def test_get_bool_env_once_invalid(self) -> None:
    self.assertIsNone(env.get_bool_env_once("TORCH_TPU_ENABLE_DEBUG_CHECKS"))
    self.assertTrue(
        env.get_bool_env_once(
            "TORCH_TPU_ENABLE_DEBUG_CHECKS", default_value=True
        )
    )

  def test_get_bool_env_once_memoized(self) -> None:
    self.assertTrue(env.get_bool_env_once("TORCH_TPU_ENABLE_DEBUG_CHECKS"))
    os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "0"
    self.assertTrue(env.get_bool_env_once("TORCH_TPU_ENABLE_DEBUG_CHECKS"))

  def test_get_int_env_once_unset(self) -> None:
    self.assertIsNone(env.get_int_env_once("TORCH_TPU_INTERNAL_HANDSHAKE_PORT"))
    self.assertEqual(
        env.get_int_env_once(
            "TORCH_TPU_INTERNAL_HANDSHAKE_PORT", default_value=42
        ),
        42,
    )

  def test_get_int_env_once_valid(self) -> None:
    self.assertEqual(
        env.get_int_env_once("TORCH_TPU_INTERNAL_HANDSHAKE_PORT"), 12345
    )
    # The default value is ignored when the variable is set.
    self.assertEqual(
        env.get_int_env_once(
            "TORCH_TPU_INTERNAL_HANDSHAKE_PORT", default_value=42
        ),
        12345,
    )

  def test_get_int_env_once_invalid(self) -> None:
    self.assertIsNone(env.get_int_env_once("TORCH_TPU_INTERNAL_HANDSHAKE_PORT"))
    self.assertEqual(
        env.get_int_env_once(
            "TORCH_TPU_INTERNAL_HANDSHAKE_PORT", default_value=42
        ),
        42,
    )

  def test_get_int_env_once_memoized(self) -> None:
    self.assertEqual(
        env.get_int_env_once("TORCH_TPU_INTERNAL_HANDSHAKE_PORT"), 12345
    )
    os.environ["TORCH_TPU_INTERNAL_HANDSHAKE_PORT"] = "54321"
    self.assertEqual(
        env.get_int_env_once("TORCH_TPU_INTERNAL_HANDSHAKE_PORT"), 12345
    )

  def test_get_str_env_once_unset(self) -> None:
    self.assertIsNone(env.get_str_env_once("MASTER_ADDR"))
    self.assertEqual(
        env.get_str_env_once("MASTER_ADDR", default_value="localhost"),
        "localhost",
    )

  def test_get_str_env_once_set(self) -> None:
    self.assertEqual(env.get_str_env_once("MASTER_ADDR"), "worker0")
    # The default value is ignored when the variable is set.
    self.assertEqual(
        env.get_str_env_once("MASTER_ADDR", default_value="localhost"),
        "worker0",
    )

  def test_get_str_env_once_empty(self) -> None:
    # An empty value is still a value, so the default value is ignored.
    self.assertEqual(env.get_str_env_once("MASTER_ADDR"), "")
    self.assertEqual(
        env.get_str_env_once("MASTER_ADDR", default_value="localhost"), ""
    )

  def test_get_str_env_once_memoized(self) -> None:
    self.assertEqual(env.get_str_env_once("MASTER_ADDR"), "worker0")
    os.environ["MASTER_ADDR"] = "worker1"
    self.assertEqual(env.get_str_env_once("MASTER_ADDR"), "worker0")


class ParentEnvTest(
    subprocess_test_utils.SubprocessTestMixin,
    seed_test_utils.MultiProcessRepeatableTest,
):
  """Parent test running isolated worker subprocesses for each test case."""

  WORKER_TEST_METHOD_TEMPLATE = "EnvTest.test_{mode.value}"

  def setUp(self) -> None:
    super().setUp()
    if _TEST_MODE.value is not None:
      self.skipTest("Skipping parent test in sub-test mode.")

  def test_has_torch_tpu_ancestor_process(self) -> None:
    self.assertFalse(torch.tpu.is_initialized())
    self.assertFalse(env.has_torch_tpu_ancestor_process())
    self.assertEqual(
        os.environ.get("TORCH_TPU_INTERNAL_TOPMOST_PID"), str(os.getpid())
    )
    self.assertFalse(torch.tpu.is_initialized())
    self.run_sub_test(TestMode.HAS_TORCH_TPU_ANCESTOR_PROCESS_CHILD)
    self.run_sub_test(
        TestMode.HAS_TORCH_TPU_ANCESTOR_PROCESS_ROOT,
        env_removals=["TORCH_TPU_INTERNAL_TOPMOST_PID"],
    )

  def test_libtpu_version(self) -> None:
    self.run_sub_test(TestMode.LIBTPU_VERSION)

  def test_libtpu_version_not_loaded(self) -> None:
    # `import torch_tpu` loads the backend, which sets the libtpu version,
    # unless autoload is off.
    self.run_sub_test(
        TestMode.LIBTPU_VERSION_NOT_LOADED,
        env_updates={"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )

  def test_get_bool_env_once_unset(self) -> None:
    self.run_sub_test(
        TestMode.GET_BOOL_ENV_ONCE_UNSET,
        env_removals=["TORCH_TPU_ENABLE_DEBUG_CHECKS"],
    )

  def test_get_bool_env_once_true(self) -> None:
    self.run_sub_test(
        TestMode.GET_BOOL_ENV_ONCE_TRUE,
        env_updates={"TORCH_TPU_ENABLE_DEBUG_CHECKS": "1"},
    )

  def test_get_bool_env_once_false(self) -> None:
    self.run_sub_test(
        TestMode.GET_BOOL_ENV_ONCE_FALSE,
        env_updates={"TORCH_TPU_ENABLE_DEBUG_CHECKS": "0"},
    )

  def test_get_bool_env_once_invalid(self) -> None:
    self.run_sub_test(
        TestMode.GET_BOOL_ENV_ONCE_INVALID,
        env_updates={"TORCH_TPU_ENABLE_DEBUG_CHECKS": "not_a_bool"},
    )

  def test_get_bool_env_once_memoized(self) -> None:
    self.run_sub_test(
        TestMode.GET_BOOL_ENV_ONCE_MEMOIZED,
        env_updates={"TORCH_TPU_ENABLE_DEBUG_CHECKS": "true"},
    )

  def test_get_int_env_once_unset(self) -> None:
    self.run_sub_test(
        TestMode.GET_INT_ENV_ONCE_UNSET,
        env_removals=["TORCH_TPU_INTERNAL_HANDSHAKE_PORT"],
    )

  def test_get_int_env_once_valid(self) -> None:
    self.run_sub_test(
        TestMode.GET_INT_ENV_ONCE_VALID,
        env_updates={"TORCH_TPU_INTERNAL_HANDSHAKE_PORT": "12345"},
    )

  def test_get_int_env_once_invalid(self) -> None:
    self.run_sub_test(
        TestMode.GET_INT_ENV_ONCE_INVALID,
        env_updates={"TORCH_TPU_INTERNAL_HANDSHAKE_PORT": "abc"},
    )

  def test_get_int_env_once_memoized(self) -> None:
    self.run_sub_test(
        TestMode.GET_INT_ENV_ONCE_MEMOIZED,
        env_updates={"TORCH_TPU_INTERNAL_HANDSHAKE_PORT": "12345"},
    )

  def test_get_str_env_once_unset(self) -> None:
    self.run_sub_test(
        TestMode.GET_STR_ENV_ONCE_UNSET,
        env_removals=["MASTER_ADDR"],
    )

  def test_get_str_env_once_set(self) -> None:
    self.run_sub_test(
        TestMode.GET_STR_ENV_ONCE_SET,
        env_updates={"MASTER_ADDR": "worker0"},
    )

  def test_get_str_env_once_empty(self) -> None:
    self.run_sub_test(
        TestMode.GET_STR_ENV_ONCE_EMPTY,
        env_updates={"MASTER_ADDR": ""},
    )

  def test_get_str_env_once_memoized(self) -> None:
    self.run_sub_test(
        TestMode.GET_STR_ENV_ONCE_MEMOIZED,
        env_updates={"MASTER_ADDR": "worker0"},
    )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
