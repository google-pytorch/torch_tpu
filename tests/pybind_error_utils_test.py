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
from torch_tpu.csrc.internal import testing
from tests import seed_test_utils


class PyBindErrorUtilsErrorsTest(seed_test_utils.RepeatableTest):

  def test_free_function_translation(self):
    # kInvalidArgument should translate to RuntimeError (c10::Error)
    # And the message should contain the calling API prefix:
    # "throw_tterror_in_free_function():"
    with self.assertRaisesRegex(
        RuntimeError,
        r"throw_tterror_in_free_function\(\):.*throwing invalid argument",
    ):
      testing.throw_tterror_in_free_function()

  def test_index_error_translation(self):
    # kPythonIndexError should translate to IndexError (c10::IndexError)
    with self.assertRaisesRegex(
        IndexError, r"throw_tterror_index_error\(\):.*throwing index error"
    ):
      testing.throw_tterror_index_error()

  def test_class_method_translation(self):
    obj = testing.TestErrorClass()
    # custom name "TestErrorClass.throw_tterror_in_member_function" should be
    # prepended
    with self.assertRaisesRegex(
        RuntimeError,
        r"TestErrorClass\.throw_tterror_in_member_function\(\):.*class throwing"
        r" invalid argument",
    ):
      obj.throw_tterror_in_member_function()


if __name__ == "__main__":
  absltest.main()
