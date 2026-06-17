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

"""Test that the split XLA and MLIR dependencies load properly."""

from absl.testing import absltest
import torch  # pylint: disable=unused-import
import torch_tpu  # pylint: disable=unused-import
from torch_tpu import _loader  # pylint: disable=unused-import


class PywrapSplitTest(absltest.TestCase):

  def test_extensions_load(self):
    # By this point, torch_tpu has been imported, which means the underlying
    # pybind extensions were successfully dlopen'd.
    # If any symbols were missing due to incorrect library splitting,
    # the import would have raised an ImportError with undefined symbol.
    from torch_tpu._internal.compile import tpu_torch_compile  # pylint: disable=g-import-not-at-top
    from torch_tpu._internal.sync import _tpu_torch_sync  # pylint: disable=g-import-not-at-top

    self.assertIsNotNone(tpu_torch_compile)
    self.assertIsNotNone(_tpu_torch_sync)


if __name__ == '__main__':
  absltest.main()
