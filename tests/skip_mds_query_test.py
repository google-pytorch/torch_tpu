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

"""Tests that `import torch_tpu` skips libtpu's metadata server lookup."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch_tpu
from tests import seed_test_utils

_ENV_VAR = "TPU_SKIP_MDS_QUERY"


class SkipMetadataServerQueryTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.dict(os.environ))
    os.environ.pop(_ENV_VAR, None)

  def _run(self, num_chips: int) -> None:
    with mock.patch.object(
        torch_tpu._hardware,  # pylint: disable=protected-access
        "num_tpu_chips_on_pci",
        return_value=num_chips,
    ):
      torch_tpu._skip_metadata_server_query_without_tpus()  # pylint: disable=protected-access

  def test_sets_skip_without_tpus(self):
    self._run(num_chips=0)

    self.assertEqual(os.environ.get(_ENV_VAR), "1")

  @parameterized.parameters(1, 4)
  def test_leaves_unset_with_tpus(self, num_chips):
    self._run(num_chips=num_chips)

    self.assertNotIn(_ENV_VAR, os.environ)

  @parameterized.parameters("", "0")
  def test_overrides_user_value_without_tpus(self, user_value):
    os.environ[_ENV_VAR] = user_value

    self._run(num_chips=0)

    self.assertEqual(os.environ[_ENV_VAR], "1")

  def test_keeps_user_value_with_tpus(self):
    os.environ[_ENV_VAR] = "true"

    self._run(num_chips=4)

    self.assertEqual(os.environ[_ENV_VAR], "true")


if __name__ == "__main__":
  absltest.main()
