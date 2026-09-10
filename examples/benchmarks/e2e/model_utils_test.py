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

"""Tests for model_utils."""

from unittest import mock

from absl.testing import absltest
import torch
from examples.benchmarks.e2e import model_utils
from tests import seed_test_utils


class DummyTiedModel(torch.nn.Module):

  def __init__(self, tie_weights_val=None):
    super().__init__()
    self.config = mock.MagicMock()
    self.tie_weights_called = False
    self.gradient_checkpointing_enabled = False
    if tie_weights_val is not None:
      self.tie_weights = tie_weights_val

  def tie_weights(self):
    self.tie_weights_called = True

  def gradient_checkpointing_enable(self, **kwargs):
    self.gradient_checkpointing_enabled = True


class ModelUtilsTest(seed_test_utils.RepeatableTest):

  def test_tie_model_weights_callable(self):
    model = DummyTiedModel()
    self.assertFalse(model.tie_weights_called)
    model_utils.tie_model_weights(model)
    self.assertTrue(model.tie_weights_called)

  def test_tie_model_weights_non_callable(self):
    model = DummyTiedModel(tie_weights_val=False)
    # Should not raise an exception when tie_weights is a non-callable attribute
    model_utils.tie_model_weights(model)

  def test_tie_model_weights_no_attribute(self):
    model = torch.nn.Linear(4, 4)
    self.assertFalse(hasattr(model, "tie_weights"))
    # Should not raise an exception when tie_weights is absent
    model_utils.tie_model_weights(model)

  def test_huggingface_llm_model_builder_calls_tie_weights(self):
    dummy_model = DummyTiedModel()
    mock_spec = mock.MagicMock()
    mock_spec.module_factory.return_value = dummy_model
    mock_spec.sample_inputs_factory.return_value = (None, {})
    mock_registry = mock.MagicMock()
    mock_registry.get_module_spec.return_value = mock_spec

    mock_args = mock.MagicMock()
    mock_args.model_name = "test_model"
    mock_args.sequence_length = 128
    mock_args.batch_size = 2
    mock_args.custom_kwargs = {"dist_strat": "none"}

    with mock.patch.object(
        model_utils, "get_module_registry", return_value=mock_registry
    ):
      result = model_utils.huggingface_llm_model_builder(
          mock_args,
          device=torch.device("cpu"),
          weights_dtype=torch.float32,
          is_training=False,
      )

    self.assertTrue(result.model.tie_weights_called)

  def test_huggingface_llm_model_builder_ddp_calls_tie_weights(self):
    dummy_model = DummyTiedModel()
    mock_spec = mock.MagicMock()
    mock_spec.module_factory.return_value = dummy_model
    mock_spec.sample_inputs_factory.return_value = (None, {})
    mock_registry = mock.MagicMock()
    mock_registry.get_module_spec.return_value = mock_spec

    mock_args = mock.MagicMock()
    mock_args.model_name = "test_model"
    mock_args.sequence_length = 128
    mock_args.batch_size = 2
    mock_args.custom_kwargs = {"dist_strat": "ddp"}

    with (
        mock.patch.object(
            model_utils, "get_module_registry", return_value=mock_registry
        ),
        mock.patch.object(
            model_utils.parallel,
            "DistributedDataParallel",
            side_effect=lambda m, **kwargs: m,
        ),
    ):
      result = model_utils.huggingface_llm_model_builder(
          mock_args,
          device=torch.device("cpu"),
          weights_dtype=torch.float32,
          is_training=False,
      )

    self.assertTrue(result.model.tie_weights_called)

  def test_huggingface_llm_model_builder_non_callable_tie_weights(self):
    dummy_model = DummyTiedModel(tie_weights_val=False)
    mock_spec = mock.MagicMock()
    mock_spec.module_factory.return_value = dummy_model
    mock_spec.sample_inputs_factory.return_value = (None, {})
    mock_registry = mock.MagicMock()
    mock_registry.get_module_spec.return_value = mock_spec

    mock_args = mock.MagicMock()
    mock_args.model_name = "test_model"
    mock_args.sequence_length = 128
    mock_args.batch_size = 2
    mock_args.custom_kwargs = {"dist_strat": "none"}

    with mock.patch.object(
        model_utils, "get_module_registry", return_value=mock_registry
    ):
      result = model_utils.huggingface_llm_model_builder(
          mock_args,
          device=torch.device("cpu"),
          weights_dtype=torch.float32,
          is_training=False,
      )

    self.assertIs(result.model, dummy_model)


if __name__ == "__main__":
  absltest.main()
