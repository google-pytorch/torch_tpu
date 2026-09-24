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
from absl.testing import flagsaver
import torch
from examples.benchmarks.e2e import benchmark_function_db
from examples.benchmarks.e2e import common
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

  def test_precompute_attention_mask_cuda_skips_4d_materialization(self):
    config = model_utils.transformers.PreTrainedConfig(
        hidden_size=64, _attn_implementation="sdpa"
    )
    example_inputs = {"input_ids": torch.zeros((2, 16), dtype=torch.long)}

    cuda_mask = model_utils._precompute_attention_mask(
        config=config,
        example_inputs=example_inputs,
        device=torch.device("cuda"),
        weights_dtype=torch.bfloat16,
        masking_device="cpu",
    )
    self.assertIsNone(cuda_mask)

    cpu_mask = model_utils._precompute_attention_mask(
        config=config,
        example_inputs=example_inputs,
        device=torch.device("cpu"),
        weights_dtype=torch.bfloat16,
        masking_device="cpu",
    )
    self.assertIsNotNone(cpu_mask)
    self.assertEqual(cpu_mask.shape, (2, 1, 16, 16))

  def test_get_train_factory_compiled_cuda_uses_eager_train_factory(self):
    with flagsaver.flagsaver((common.PLATFORM, common.Platform.B200_1)):
      out_cuda = benchmark_function_db.get_train_factory(
          common.RunMode.COMPILED, benchmark_function_db.simple_train_factory
      )
      self.assertIs(out_cuda, benchmark_function_db.simple_train_factory)

    with flagsaver.flagsaver((common.PLATFORM, common.Platform.GFC_1X1X1)):
      out_tpu = benchmark_function_db.get_train_factory(
          common.RunMode.COMPILED, benchmark_function_db.simple_train_factory
      )
      self.assertIs(
          out_tpu.func, benchmark_function_db.functional_train_factory
      )

    model_utils.device_utils.configure_inductor_for_benchmarks()
    self.assertFalse(
        model_utils.transformers.masking_utils.is_tracing(torch.zeros(1))
    )


if __name__ == "__main__":
  absltest.main()
