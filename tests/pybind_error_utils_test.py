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

from absl.testing import absltest
import torch  # pylint: disable=unused-import  # ruff: ignore[unused-import]

# Pre-loads `libtorch.so` for the `_internal.testing` C++ extension in OSS.
from torch_tpu._internal import testing
from tests import seed_test_utils


class PyBindErrorUtilsTest(seed_test_utils.RepeatableTest):

  def test_field_read(self):
    obj = testing.TestMembersClass()
    self.assertEqual(obj.read_only_field, 100)
    self.assertEqual(obj.read_write_field, 42)

  def test_field_write(self):
    obj = testing.TestMembersClass()
    self.assertEqual(obj.read_write_field, 42)
    obj.read_write_field = 99
    self.assertEqual(obj.read_write_field, 99)

  def test_write_readonly_field(self):
    obj = testing.TestMembersClass()

    with self.assertRaises(  # ASSERT_RAISES_OK=Testing the pybind11 wrapper.
        AttributeError
    ):
      obj.read_only_field = 99

  def test_throw_tterror_with_call_guard(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing this doesn't crash.
        RuntimeError, "throwing invalid argument"
    ):
      testing.throw_tterror_with_call_guard()


if __name__ == "__main__":
  absltest.main()
