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

"""Covers `torch_tpu.__version__`."""

import importlib.metadata

from absl.testing import absltest
import torch_tpu
from tests import seed_test_utils


class VersionTest(seed_test_utils.RepeatableTest):

  def test_version_matches_distribution_metadata(self) -> None:
    self.assertEqual(
        torch_tpu.__version__, importlib.metadata.version("torch_tpu")
    )

  def test_falls_back_to_valid_version_without_metadata(self) -> None:
    self.assertEqual(
        torch_tpu._get_version("torch_tpu_no_such_distribution"),  # pylint: disable=protected-access
        "0.0.0+unknown",
    )


if __name__ == "__main__":
  absltest.main()
