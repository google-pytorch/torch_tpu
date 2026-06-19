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

"""This tests that the XLA backends are NOT loaded unless explicitly enabled."""

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from absl.testing import parameterized  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top


class AutoloadDisabledBackendTest(parameterized.TestCase):

  @parameterized.parameters("xla_cuda", "xla_cpu")
  def test_backend_unavailable(self, module: str) -> None:
    with self.assertRaisesRegex(AttributeError, f".*{module}.*"):
      getattr(torch, module)

  @parameterized.parameters("xla_cuda", "xla_cpu")
  def test_device_unavailable(self, device_name: str) -> None:
    with self.assertRaisesRegex(RuntimeError, "Expected one of"):
      torch.device(device_name)


if __name__ == "__main__":
  absltest.main()
