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

"""Tests for the `env_utils` module."""

import os

from absl.testing import absltest
from torch_tpu._internal import env_utils
from tests import seed_test_utils


class GetAndSetStrEnvOnceTest(seed_test_utils.RepeatableTest):
  """Tests for `env_utils.get_and_set_str_env_once()`.

  Each test uses a distinct environment variable name so that the memoization
  performed by the function under test does not leak between tests.
  """

  def test_unset_env_var_is_set_to_the_generated_value(self):
    name = "TORCH_TPU_TEST_ENV_UTILS_UNSET"
    os.environ.pop(name, None)

    value = env_utils.get_and_set_str_env_once(name, lambda: "generated")

    self.assertEqual(value, "generated")
    # The generated value is persisted so that child processes inherit it.
    self.assertEqual(os.environ[name], "generated")

  def test_set_env_var_is_returned_without_calling_the_factory(self):
    name = "TORCH_TPU_TEST_ENV_UTILS_SET"
    os.environ[name] = "preset"

    def fail_if_called() -> str:
      self.fail("the value factory must not be called when the var is set")

    value = env_utils.get_and_set_str_env_once(name, fail_if_called)

    self.assertEqual(value, "preset")
    self.assertEqual(os.environ[name], "preset")

  def test_empty_env_var_counts_as_set(self):
    name = "TORCH_TPU_TEST_ENV_UTILS_EMPTY"
    os.environ[name] = ""

    def fail_if_called() -> str:
      self.fail("the value factory must not be called when the var is set")

    value = env_utils.get_and_set_str_env_once(name, fail_if_called)

    self.assertEqual(value, "")
    self.assertEqual(os.environ[name], "")

  def test_value_factory_is_called_at_most_once(self):
    name = "TORCH_TPU_TEST_ENV_UTILS_MEMOIZED"
    os.environ.pop(name, None)
    call_count = 0

    def count_calls() -> str:
      nonlocal call_count
      call_count += 1
      return "generated"

    first = env_utils.get_and_set_str_env_once(name, count_calls)
    second = env_utils.get_and_set_str_env_once(name, count_calls)

    self.assertEqual(first, "generated")
    self.assertEqual(second, "generated")
    self.assertEqual(call_count, 1)

  def test_current_env_var_value_wins_over_the_resolved_one(self):
    name = "TORCH_TPU_TEST_ENV_UTILS_STALE"
    os.environ.pop(name, None)

    first = env_utils.get_and_set_str_env_once(name, lambda: "generated")
    os.environ[name] = "changed out from under us"
    second = env_utils.get_and_set_str_env_once(name, lambda: "unused")

    self.assertEqual(first, "generated")
    self.assertEqual(second, "changed out from under us")

  def test_unset_env_var_is_restored_without_calling_the_factory_again(self):
    # Regression test: a caller that resolves a variable, then has it removed
    # from the environment -- as `mock.patch.dict(os.environ, ...)` does when it
    # restores the environment on exit -- must still end up with the variable
    # set, or child processes will not inherit it.
    name = "TORCH_TPU_TEST_ENV_UTILS_RESTORED"
    os.environ.pop(name, None)
    call_count = 0

    def count_calls() -> str:
      nonlocal call_count
      call_count += 1
      return "generated"

    env_utils.get_and_set_str_env_once(name, count_calls)
    del os.environ[name]
    value = env_utils.get_and_set_str_env_once(name, count_calls)

    self.assertEqual(value, "generated")
    self.assertEqual(os.environ[name], "generated")
    self.assertEqual(call_count, 1)


class SetStrEnvTest(seed_test_utils.RepeatableTest):
  """Tests for `env_utils.set_str_env()`."""

  def test_unset_env_var_is_set(self):
    name = "TORCH_TPU_TEST_ENV_UTILS_SET_UNSET"
    os.environ.pop(name, None)

    env_utils.set_str_env(name, "value")

    self.assertEqual(os.environ[name], "value")

  def test_set_env_var_is_overwritten(self):
    name = "TORCH_TPU_TEST_ENV_UTILS_SET_OVERWRITE"
    os.environ[name] = "old"

    env_utils.set_str_env(name, "new")

    self.assertEqual(os.environ[name], "new")

  def test_empty_value_is_set(self):
    name = "TORCH_TPU_TEST_ENV_UTILS_SET_EMPTY"
    os.environ[name] = "old"

    env_utils.set_str_env(name, "")

    self.assertEqual(os.environ[name], "")


if __name__ == "__main__":
  absltest.main()
