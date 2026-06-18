# Copyright 2025 Google LLC
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
import torch  # pylint: disable=unused-import  # Required to load libtorch.so into memory.
from torch_tpu._internal import env


class EnvTest(absltest.TestCase):

  def test_pytorch_build_version(self):
    version = env.PYTORCH_BUILD_VERSION
    self.assertNotEqual(version, "unknown")
    self.assertRegex(version, r"^\d+(\.\d+)?(\.\d+)?")


if __name__ == "__main__":
  absltest.main()
