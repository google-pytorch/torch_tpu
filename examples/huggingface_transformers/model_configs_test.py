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

"""Unit tests for model_configs.py."""

from absl.testing import absltest
from examples.huggingface_transformers import model_configs
from tests import seed_test_utils


class ModelConfigsTest(seed_test_utils.RepeatableTest):

  def test_invalid_model_id(self):
    # Arrange
    model_id = "invalid_model_id"

    # Assert
    with self.assertRaises(ValueError):
      # Act
      model_configs.create_config_loader(model_id)()

  def test_valid_model_id(self):
    # Arrange
    # README.md has more details, but this model_id is from the HuggingFace
    # model hub, plus the -MINI suffix to indicate it has been downsized.
    model_id = "Qwen/Qwen3-235B-A22B-Instruct-2507-MINI"

    # Act
    _ = model_configs.create_config_loader(model_id)()

    # Assert
    # No assertion. Test passes if no exception is raised.


if __name__ == "__main__":
  absltest.main()
