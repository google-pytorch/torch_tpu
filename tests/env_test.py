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

import os
from typing import Final

from absl.testing import absltest
from torch_tpu._internal import env

KNOWN_PYTORCH_BUILD_VERSION: Final[str] = os.environ.get(
    "TORCH_TPU_PYTORCH_BUILD_VERSION", None
)


class EnvTest(absltest.TestCase):

  def test_pytorch_build_version_not_unknown(self):
    self.assertNotEqual(env.PYTORCH_BUILD_VERSION, "unknown")

  @absltest.skipIf(
      KNOWN_PYTORCH_BUILD_VERSION is None,
      reason="Test only valid with known PyTorch version.",
  )
  def test_pytorch_build_version(self):
    """Test that the build version is exactly what is expected."""
    self.assertEqual(env.PYTORCH_BUILD_VERSION, KNOWN_PYTORCH_BUILD_VERSION)

  @absltest.skipIf(
      KNOWN_PYTORCH_BUILD_VERSION is not None,
      reason="Test unnecessary with known PyTorch version.",
  )
  def test_pytorch_build_version_format(self):
    """Check the format of PYTORCH_BUILD_VERSION when it is not known."""
    self.assertRegex(env.PYTORCH_BUILD_VERSION, r"^\d+(\.\d+)?(\.\d+)?")


if __name__ == "__main__":
  absltest.main()
