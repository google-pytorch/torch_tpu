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

"""Tests for load_model."""

import types
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn
from perflab import context as context_lib
from perflab import model_helpers
from perflab import target as target_lib
from perflab import workload as workload_lib
from perflab.gen_utils import load_model
from tests import seed_test_utils


class _FakeForSequenceClassification(nn.Module):
  """Stand-in for a HuggingFace `*ForSequenceClassification` model."""

  def __init__(self, config: types.SimpleNamespace):
    super().__init__()
    self.config = config
    self.linear = nn.Linear(4, 4)


def _deberta_config(
    num_labels: int, problem_type: str | None = None
) -> types.SimpleNamespace:
  """Returns a minimal DeBERTa-v2 sequence-classification config."""
  return types.SimpleNamespace(
      model_type="deberta-v2", num_labels=num_labels, problem_type=problem_type
  )


def _fake_registry(model: nn.Module, inputs: dict[str, Any]) -> Any:
  """Returns a module registry whose only module spec yields `model`."""
  module_spec = types.SimpleNamespace(
      module_factory=lambda dtype=None: (
          model.to(dtype=dtype) if dtype is not None else model
      ),
      sample_inputs_factory=lambda shape, device: (None, dict(inputs)),
  )
  return types.SimpleNamespace(
      get_module_spec=lambda *args, **kwargs: module_spec
  )


class _FakeTimeSeriesModel(nn.Module):
  """Stand-in for a HuggingFace time series model such as TimesFM."""

  def __init__(self, config: types.SimpleNamespace):
    super().__init__()
    self.config = config


def _fake_time_series_registry(config: types.SimpleNamespace) -> Any:
  """Returns a module registry that sizes inputs by `context_length`.

  Like the TimesFM input builder, its inputs ignore the requested sequence
  length, and it applies the config hook of the caller.

  Args:
    config: The model config, passed to the config hook.
  """

  def get_module_spec(*args, modify_config_hook, **kwargs):
    del args, kwargs  # Unused.
    hooked_config = modify_config_hook(config)
    return types.SimpleNamespace(
        module_factory=lambda dtype=None: _FakeTimeSeriesModel(hooked_config),
        sample_inputs_factory=lambda shape, device: (
            None,
            {
                "past_values": torch.randn(
                    shape[0], hooked_config.context_length
                )
            },
        ),
    )

  return types.SimpleNamespace(get_module_spec=get_module_spec)


class _FakeTextModel(nn.Module):
  """Stand-in for a HuggingFace model that takes token IDs."""

  def __init__(self, config: types.SimpleNamespace):
    super().__init__()
    self.config = config


def _fake_text_registry(config: types.SimpleNamespace) -> Any:
  """Returns a module registry that applies the config hook of the caller.

  Args:
    config: The model config, passed to the config hook.
  """

  def get_module_spec(*args, modify_config_hook, **kwargs):
    del args, kwargs  # Unused.
    hooked_config = modify_config_hook(config)
    return types.SimpleNamespace(
        module_factory=lambda dtype=None: _FakeTextModel(hooked_config),
        sample_inputs_factory=lambda shape, device: (
            None,
            {"input_ids": torch.zeros(shape, dtype=torch.long)},
        ),
    )

  return types.SimpleNamespace(get_module_spec=get_module_spec)


class LoadModelTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.ctx = context_lib.Context(
        target=target_lib.make_target(platform=target_lib.Platform.CPU),
        run_scope=context_lib.RunScope.PRESUBMIT,
        workload=workload_lib.Workload(1),
    )
    self.inputs = {
        "input_ids": torch.randint(0, 10, (1, 128)),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }

  def _load(
      self, model: nn.Module, is_training: bool
  ) -> tuple[nn.Module, dict[str, Any]]:
    with mock.patch.object(
        model_helpers,
        "get_module_registry",
        return_value=_fake_registry(model, self.inputs),
    ):
      return load_model.default_load_model(
          self.ctx, "fake/model", is_training=is_training
      )

  def test_sequence_classification_training_sets_single_label_problem_type(
      self,
  ):
    config = _deberta_config(num_labels=3)
    model, inputs = self._load(
        _FakeForSequenceClassification(config), is_training=True
    )

    self.assertEqual(model.config.problem_type, "single_label_classification")
    self.assertEqual(inputs["labels"].shape, (1,))
    self.assertEqual(inputs["labels"].dtype, torch.long)
    self.assertTrue(torch.all(inputs["labels"] < 3))

  def test_sequence_classification_regression_leaves_problem_type_unset(self):
    config = _deberta_config(num_labels=1)
    model, _ = self._load(
        _FakeForSequenceClassification(config), is_training=True
    )

    self.assertIsNone(model.config.problem_type)

  def test_sequence_classification_preserves_preset_problem_type(self):
    config = _deberta_config(
        num_labels=3, problem_type="multi_label_classification"
    )
    model, _ = self._load(
        _FakeForSequenceClassification(config), is_training=True
    )

    self.assertEqual(model.config.problem_type, "multi_label_classification")

  def test_sequence_classification_inference_leaves_problem_type_unset(self):
    config = _deberta_config(num_labels=3)
    model, inputs = self._load(
        _FakeForSequenceClassification(config), is_training=False
    )

    self.assertIsNone(model.config.problem_type)
    self.assertNotIn("labels", inputs)

  def _load_tapas(
      self, model: nn.Module, is_training: bool
  ) -> tuple[nn.Module, dict[str, Any]]:
    with mock.patch.object(
        model_helpers,
        "get_module_registry",
        return_value=_fake_registry(model, self.inputs),
    ):
      return load_model.tapas_load_model(
          self.ctx, "google/tapas-base-finetuned-sqa", is_training=is_training
      )

  def test_tapas_load_model_training_generates_binary_cell_selection_labels(
      self,
  ):
    _, inputs = self._load_tapas(
        _FakeTextModel(types.SimpleNamespace(vocab_size=30522)),
        is_training=True,
    )

    self.assertEqual(inputs["labels"].shape, (1, 128))
    self.assertEqual(inputs["labels"].dtype, torch.long)
    self.assertFalse(inputs["labels"].requires_grad)
    self.assertTrue(
        torch.all((inputs["labels"] == 0) | (inputs["labels"] == 1))
    )

  def test_tapas_load_model_inference_omits_labels(self):
    _, inputs = self._load_tapas(
        _FakeTextModel(types.SimpleNamespace(vocab_size=30522)),
        is_training=False,
    )

    self.assertNotIn("labels", inputs)

  def _load_time_series(
      self,
      context_length: int,
      run_scope: context_lib.RunScope = context_lib.RunScope.FULL,
      batch_size: int = 1,
      **config_kwargs: int,
  ) -> tuple[nn.Module, dict[str, Any]]:
    ctx = context_lib.Context(
        target=target_lib.make_target(platform=target_lib.Platform.CPU),
        run_scope=run_scope,
        workload=workload_lib.Workload(batch_size),
    )
    config = types.SimpleNamespace(
        context_length=context_length, **config_kwargs
    )
    with mock.patch.object(
        model_helpers,
        "get_module_registry",
        return_value=_fake_time_series_registry(config),
    ):
      return load_model.default_load_model(
          ctx,
          "fake/time_series_model",
          is_training=False,
      )

  def test_workload_batch_size_used(self):
    _, inputs = self._load_time_series(context_length=512, batch_size=8)

    self.assertEqual(inputs["past_values"].shape, (8, 512))

  def test_presubmit_uses_shorter_sequences(self):
    _, inputs = self._load_time_series(
        context_length=512,
        run_scope=context_lib.RunScope.PRESUBMIT,
        batch_size=2,
    )

    self.assertEqual(inputs["past_values"].shape, (2, 128))

  def test_context_length_capped_at_seq_len(self):
    model, inputs = self._load_time_series(context_length=16384)

    self.assertEqual(model.config.context_length, 2048)
    self.assertEqual(inputs["past_values"].shape, (1, 2048))

  def test_shorter_context_length_kept(self):
    model, inputs = self._load_time_series(context_length=512)

    self.assertEqual(model.config.context_length, 512)
    self.assertEqual(inputs["past_values"].shape, (1, 512))

  def test_capped_context_length_recomputes_num_patches(self):
    # The config of ibm-granite/granite-timeseries-patchtsmixer: 512 steps in
    # patches of 16, i.e. 32 patches, and 8 patches in the 128 presubmit steps.
    model, inputs = self._load_time_series(
        context_length=512,
        run_scope=context_lib.RunScope.PRESUBMIT,
        num_patches=32,
        patch_length=16,
        patch_stride=16,
    )

    self.assertEqual(model.config.context_length, 128)
    self.assertEqual(model.config.num_patches, 8)
    self.assertEqual(inputs["past_values"].shape, (1, 128))

  def test_uncapped_context_length_keeps_num_patches(self):
    model, _ = self._load_time_series(
        context_length=512, num_patches=32, patch_length=16, patch_stride=16
    )

    self.assertEqual(model.config.context_length, 512)
    self.assertEqual(model.config.num_patches, 32)

  def _load_text(
      self, config: types.SimpleNamespace, run_scope: context_lib.RunScope
  ) -> nn.Module:
    ctx = context_lib.Context(
        target=target_lib.make_target(platform=target_lib.Platform.CPU),
        run_scope=run_scope,
        workload=workload_lib.Workload(1),
    )
    with mock.patch.object(
        model_helpers,
        "get_module_registry",
        return_value=_fake_text_registry(config),
    ):
      model, _ = load_model.default_load_model(
          ctx, "fake/text_model", is_training=False
      )
    return model

  def test_presubmit_keeps_two_layers(self):
    model = self._load_text(
        types.SimpleNamespace(
            num_hidden_layers=12,
            text_config=types.SimpleNamespace(num_hidden_layers=12),
            vision_config=types.SimpleNamespace(num_hidden_layers=12, depth=12),
        ),
        context_lib.RunScope.PRESUBMIT,
    )

    self.assertEqual(model.config.dtype, torch.bfloat16)
    self.assertEqual(model.config.torch_dtype, torch.bfloat16)
    self.assertEqual(model.config.text_config.dtype, torch.bfloat16)
    self.assertEqual(model.config.text_config.torch_dtype, torch.bfloat16)
    self.assertEqual(model.config.vision_config.dtype, torch.bfloat16)
    self.assertEqual(model.config.vision_config.torch_dtype, torch.bfloat16)
    self.assertEqual(model.config.num_hidden_layers, 2)
    self.assertEqual(model.config.text_config.num_hidden_layers, 2)
    self.assertEqual(model.config.vision_config.num_hidden_layers, 2)
    self.assertEqual(model.config.vision_config.depth, 2)

  def test_set_config_dtype_updates_pretrained_config_property_and_dicts(self):
    class _FakePretrainedConfig:
      """Mirrors HF PretrainedConfig where torch_dtype is a dtype property."""

      def __init__(self, dtype=torch.float32, text_config=None):
        self.dtype = dtype
        self.text_config = text_config

      @property
      def torch_dtype(self):
        return self.dtype

      @torch_dtype.setter
      def torch_dtype(self, value):
        self.dtype = value

    # Simulate _ConfigWrapper.__setattr__ writing directly to __dict__.
    class _DictWritingWrapper:

      def __init__(self, data):
        object.__setattr__(self, "_data", data)

      def __getattr__(self, name):
        val = getattr(self._data, name)
        if isinstance(val, _FakePretrainedConfig):
          return _DictWritingWrapper(val)
        return val

      def __setattr__(self, name, value):
        self._data.__dict__[name] = value

    raw_cfg = _FakePretrainedConfig(
        dtype=torch.float32,
        text_config={"dtype": torch.float32, "torch_dtype": torch.float32},
    )
    wrapped = _DictWritingWrapper(raw_cfg)
    load_model._set_config_dtype(wrapped, torch.bfloat16)

    self.assertEqual(raw_cfg.dtype, torch.bfloat16)
    self.assertEqual(raw_cfg.torch_dtype, torch.bfloat16)
    self.assertEqual(raw_cfg.text_config["dtype"], torch.bfloat16)
    self.assertEqual(raw_cfg.text_config["torch_dtype"], torch.bfloat16)

  def test_presubmit_keeps_a_layer_per_dpt_neck_stage(self):
    # The config of Intel/dpt-hybrid-midas.
    model = self._load_text(
        types.SimpleNamespace(
            num_hidden_layers=12, backbone_out_indices=[2, 5, 8, 11]
        ),
        context_lib.RunScope.PRESUBMIT,
    )

    self.assertEqual(model.config.num_hidden_layers, 4)
    self.assertEqual(model.config.backbone_out_indices, [0, 1, 2, 3])

  def test_presubmit_reads_the_kept_layers_for_few_dpt_neck_stages(self):
    model = self._load_text(
        types.SimpleNamespace(
            num_hidden_layers=12, backbone_out_indices=[5, 11]
        ),
        context_lib.RunScope.PRESUBMIT,
    )

    self.assertEqual(model.config.num_hidden_layers, 2)
    self.assertEqual(model.config.backbone_out_indices, [0, 1])

  def test_full_scope_keeps_dpt_layers(self):
    model = self._load_text(
        types.SimpleNamespace(
            num_hidden_layers=12, backbone_out_indices=[2, 5, 8, 11]
        ),
        context_lib.RunScope.FULL,
    )

    self.assertEqual(model.config.num_hidden_layers, 12)
    self.assertEqual(model.config.backbone_out_indices, [2, 5, 8, 11])

  def test_blip_forward_compiled_graph_has_no_cpu_tensors_on_accelerator(self):
    class _FakeVisionOutput:

      def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = None
        self.attentions = None

    class _FakeBlipVisionModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

      def forward(
          self,
          pixel_values: torch.Tensor,
          interpolate_pos_encoding: bool = False,
          **kwargs: Any,
      ) -> _FakeVisionOutput:
        del interpolate_pos_encoding, kwargs  # Unused.
        return _FakeVisionOutput(self.proj(pixel_values))

    class _FakeBlipTextEncoder(nn.Module):

      def forward(
          self,
          input_ids: torch.Tensor,
          attention_mask: torch.Tensor | None = None,
          encoder_hidden_states: torch.Tensor | None = None,
          encoder_attention_mask: torch.Tensor | None = None,
          **kwargs: Any,
      ) -> tuple[torch.Tensor]:
        del input_ids, attention_mask, kwargs  # Unused.
        assert encoder_hidden_states is not None
        assert encoder_attention_mask is not None
        mask = encoder_attention_mask.to(
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )
        return (encoder_hidden_states * mask.unsqueeze(-1),)

    class _FakeDecoderOutput:

      def __init__(self, loss: torch.Tensor | None):
        self.loss = loss

    class _FakeBlipTextDecoder(nn.Module):

      def forward(
          self,
          input_ids: torch.Tensor,
          attention_mask: torch.Tensor | None = None,
          encoder_hidden_states: torch.Tensor | None = None,
          encoder_attention_mask: torch.Tensor | None = None,
          labels: torch.Tensor | None = None,
          reduction: str = "mean",
          **kwargs: Any,
      ) -> _FakeDecoderOutput:
        del input_ids, attention_mask, encoder_attention_mask, reduction, kwargs
        assert encoder_hidden_states is not None
        loss = encoder_hidden_states.sum() if labels is not None else None
        return _FakeDecoderOutput(loss)

    class BlipForQuestionAnswering(nn.Module):

      def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="blip")
        self.vision_model = _FakeBlipVisionModel()
        self.text_encoder = _FakeBlipTextEncoder()
        self.text_decoder = _FakeBlipTextDecoder()

      def forward(
          self,
          input_ids: torch.Tensor,
          pixel_values: torch.Tensor,
          decoder_input_ids: torch.Tensor | None = None,
          **kwargs: Any,
      ) -> Any:
        vision_outputs = self.vision_model(pixel_values=pixel_values, **kwargs)
        image_embeds = vision_outputs.last_hidden_state
        # Mirrors `modeling_blip.BlipForQuestionAnswering.forward`: omits
        # `device=image_embeds.device`.
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long
        )
        question_embeds = self.text_encoder(
            input_ids=input_ids,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )[0]
        return self.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=question_embeds,
            labels=decoder_input_ids,
        )

    with torch.device("meta"):
      model = BlipForQuestionAnswering()
      input_ids = torch.zeros((2, 6), dtype=torch.long)
      decoder_input_ids = torch.zeros((2, 6), dtype=torch.long)
      pixel_values = torch.randn((2, 6, 4))

    model, _ = self._load(model, is_training=False)
    eager_out = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        pixel_values=pixel_values,
    )
    self.assertEqual(eager_out.loss.device.type, "meta")

    captured_graphs = []

    def _capture_backend(gm: torch.fx.GraphModule, example_inputs):
      del example_inputs  # Unused.
      captured_graphs.append(gm)
      return gm.forward

    compiled = torch.compile(model, backend=_capture_backend)
    out = compiled(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        pixel_values=pixel_values,
    )

    self.assertEqual(out.loss.device.type, "meta")
    self.assertNotEmpty(captured_graphs)
    tensor_devices = {
        val.device.type
        for gm in captured_graphs
        for node in gm.graph.nodes
        if isinstance((val := node.meta.get("example_value")), torch.Tensor)
    }
    self.assertEqual(tensor_devices, {"meta"})


if __name__ == "__main__":
  absltest.main()
