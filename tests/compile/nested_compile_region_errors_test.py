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

"""Error tests for TorchTPU's `nested_compile_region`."""

from absl.testing import absltest
from torch_tpu._internal.compile import nested_compile_region as tt_ncr
from tests import seed_test_utils


class NestedCompileRegionErrorsTest(seed_test_utils.RepeatableTest):

  def test_scan_unroll_must_be_positive(self):
    with self.assertRaisesRegex(ValueError, "scan_unroll must be at least 1"):
      tt_ncr.nested_compile_region(scan_unroll=0)


if __name__ == "__main__":
  absltest.main()
