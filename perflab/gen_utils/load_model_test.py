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
      module_factory=lambda: model,
      sample_inputs_factory=lambda shape, device: (None, dict(inputs)),
  )
  return types.SimpleNamespace(
      get_module_spec=lambda *args, **kwargs: module_spec
  )


class LoadModelTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.ctx = context_lib.Context(
        target=target_lib.make_target(platform=target_lib.Platform.CPU),
        run_scope=context_lib.RunScope.PRESUBMIT,
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


@torch.jit.script
def _scaled_size_sqrt(x: torch.Tensor, scale_factor: int):
  # Mirrors `transformers.models.deberta.modeling_deberta.scaled_size_sqrt`.
  return torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float) * scale_factor)


class _ImplicitCpuTensorsModel(nn.Module):
  """Creates tensors without `device=` in forward, like BLIP and DeBERTa."""

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirrors `BlipForQuestionAnswering.forward`'s `image_attention_mask`.
    mask = torch.ones(x.shape[:-1], dtype=torch.long)
    scale = _scaled_size_sqrt(x, 3)
    return x / scale.to(dtype=x.dtype), mask


class _GraphBreakModel(_ImplicitCpuTensorsModel):
  """Graph-breaks before creating device-less tensors, like DeBERTa."""

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    torch._dynamo.graph_break()
    return super().forward(x)


def _compile_and_record_graph_devices(
    model: nn.Module, fullgraph: bool = True
) -> tuple[tuple[torch.Tensor, ...], set[str]]:
  """Compiles `model` on a meta input; returns outputs and graph devices."""
  graph_devices = set()

  def recording_backend(gm, example_inputs):
    del example_inputs
    for node in gm.graph.nodes:
      value = node.meta.get("example_value")
      if isinstance(value, torch.Tensor):
        graph_devices.add(value.device.type)
    return gm.forward

  torch.compiler.reset()
  try:
    outputs = torch.compile(
        model, backend=recording_backend, fullgraph=fullgraph
    )(torch.zeros(2, 3, 4, device="meta"))
  finally:
    torch.compiler.reset()
  return outputs, graph_devices


class DefaultDeviceForwardTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # Keeps a failing test from leaking a global default device.
    self.addCleanup(torch.set_default_device, None)

  def test_eager_forward_creates_tensors_on_benchmark_device(self):
    # Graph breaks fall back to eager, so eager must use the device too.
    model = _ImplicitCpuTensorsModel()
    load_model.run_forward_on_default_device(model, "meta")

    _, mask = model(torch.zeros(2, 3, 4, device="meta"))

    self.assertEqual(mask.device.type, "meta")
    # The default device does not leak outside `forward`.
    self.assertEqual(torch.ones(1).device.type, "cpu")

  def test_compiled_graph_has_no_cpu_tensors(self):
    model = _ImplicitCpuTensorsModel()
    load_model.run_forward_on_default_device(model, "meta")

    (out, mask), graph_devices = _compile_and_record_graph_devices(model)

    self.assertEqual(out.device.type, "meta")
    self.assertEqual(mask.device.type, "meta")
    self.assertEqual(graph_devices, {"meta"})
    # The default device does not leak outside `forward`.
    self.assertEqual(torch.ones(1).device.type, "cpu")

  def test_compiled_graphs_after_graph_break_have_no_cpu_tensors(self):
    model = _GraphBreakModel()
    load_model.run_forward_on_default_device(model, "meta")

    (_, mask), graph_devices = _compile_and_record_graph_devices(
        model, fullgraph=False
    )

    self.assertEqual(mask.device.type, "meta")
    self.assertEqual(graph_devices, {"meta"})
    self.assertEqual(torch.ones(1).device.type, "cpu")

  def test_compiled_graph_without_wrapper_has_cpu_tensors(self):
    # Guards the tests above: without the wrapper the graph does contain CPU
    # tensors, which is what makes Inductor emit C++ kernels.
    _, graph_devices = _compile_and_record_graph_devices(
        _ImplicitCpuTensorsModel()
    )

    self.assertIn("cpu", graph_devices)


if __name__ == "__main__":
  absltest.main()
