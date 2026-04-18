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

"""Tests for `error_testing.py` functions."""

from absl.testing import absltest
from tests import error_testing as et


class ErrorTestingTest(absltest.TestCase):

  def test_append_error_test_failure_protocol(self):
    """Tests that the error test failure protocol is appended to the error."""

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        Exception,
        # Check that the appended note does not have leading spaces.
        r"\nNOTE: This test might fail depending on which PyTorch version is"
        " being used.",
    ):
      with et.assert_raises_message(
          RuntimeError,
          tpu="""TPU dummy error message for errors_test.py failure protocol""",
          cpu="""CPU dummy error message for errors_test.py failure protocol""",
          message_reviewed_by="wan",
      ):
        raise RuntimeError("Not the dummy error message")


if __name__ == "__main__":
  absltest.main()
