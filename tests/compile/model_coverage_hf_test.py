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

"""Tests inference and training with torch.compile across HuggingFace models.

Compares perplexity (for inference) and loss (for training) between TPU and CPU
with popular HuggingFace model implementations using the Transformers API.
"""

import enum
import functools
import gc
from typing import Any

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.accelerator
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.utils import utils
from tests import module_registry


class ModelSize(enum.StrEnum):
  TINY = enum.auto()
  SMALL = enum.auto()
  MEDIUM = enum.auto()
  LARGE = enum.auto()


FLAGS = flags.FLAGS
_MODEL_SIZES_TO_RUN = flags.DEFINE_multi_enum_class(
    "model_sizes_to_run",
    default=[ModelSize.TINY],
    enum_class=ModelSize,
    help=(
        "Selects models to run based on their parameter count. 'TINY' <= 500M,"
        " 500M < 'SMALL' <= 1B, 1B < 'MEDIUM' <= 30B, and 'LARGE' > 30B."
    ),
)

# Threshold number for which perplexity scores should be less than. Anything
# higher indicates a significant bug in the model or test.
_PERPLEXITY_THRESHOLD = 100
_NUM_TRAIN_STEPS = 5


def _calculate_perplexity(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
  """Calculates perplexity from logits and input_ids for a Causal LM."""
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = input_ids[..., 1:].contiguous()

  loss = torch.nn.functional.cross_entropy(
      shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
  )
  return torch.exp(loss)


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: dict[str, Any],
) -> torch.Tensor:
  """Performs a single training step (Forward, Backward, Optimizer Step).

  Args:
    model: The PyTorch model to train.
    optimizer: The optimizer to use for the training step.
    inputs: A dictionary of input tensors.

  Returns:
    The calculated loss for the training step.
  """
  optimizer.zero_grad()

  outputs = model(**inputs, labels=inputs["input_ids"])
  loss = outputs.loss
  loss.backward()
  optimizer.step()

  return loss.detach()


class ModelCoverageHFTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.tpu_device = torch.accelerator.current_accelerator()
    cls.module_registry = module_registry.ModuleRegistry()

  def _check_model_size(
      self, provider: str, module_name: str, model_size: ModelSize
  ) -> None:
    if model_size not in _MODEL_SIZES_TO_RUN.value:
      self.skipTest(
          f"Skipping {provider}/{module_name} (size: {model_size}) because it's"
          " not in the provided model_sizes_to_run flag:"
          f" {_MODEL_SIZES_TO_RUN.value}."
      )

  def _create_inputs(self, provider: str, module_name: str):
    module_spec = self.module_registry.get_module_spec(
        provider, module_name, load_weights=True
    )
    tokenizer = module_spec.preprocessor_factory()
    prompt = (
        "Despite the heavy rain and strong winds forecasted for the afternoon,"
        " the dedicated marathon runners refused to cancel the"
    )
    return tokenizer(prompt, return_tensors="pt")

  def _create_model(
      self,
      provider: str,
      module_name: str,
      *,
      device=torch.device("cpu"),
      is_training=False,
  ):
    module_spec = self.module_registry.get_module_spec(
        provider, module_name, load_weights=True
    )
    with device:
      model = module_spec.module_factory()
    if is_training:
      model.train()
    else:
      model.eval()

    compiled_model = None

    if device == torch.accelerator.current_accelerator():
      fn_to_compile = (
          functools.partial(_train_step, model) if is_training else model
      )
      compiled_model = torch.compile(
          fn_to_compile,
          dynamic=False,
          backend=torch_tpu_compile.TpuBackend(),
      )

    return model, compiled_model

  # rtol and atol were determined manually by running the test case once,
  # checking the results and determining tolerances based on the delta between
  # CPU and TPU.
  @parameterized.named_parameters(
      dict(
          testcase_name="transformers/google/gemma-3-270m",
          provider="transformers",
          module_name="google/gemma-3-270m",
          model_size=ModelSize.TINY,
          rtol=2e-2,
          atol=5e-2,
      ),
      dict(
          testcase_name="transformers/Qwen/Qwen3-0.6B",
          provider="transformers",
          module_name="Qwen/Qwen3-0.6B",
          model_size=ModelSize.SMALL,
          rtol=1e-3,
          atol=2e-4,
      ),
      dict(
          testcase_name="transformers/Qwen/Qwen-4B",
          provider="transformers",
          module_name="Qwen/Qwen3-4B",
          model_size=ModelSize.MEDIUM,
          rtol=8e-3,
          atol=2e-2,
      ),
      dict(
          testcase_name="transformers/meta-llama/Llama-3.2-1B",
          provider="transformers",
          module_name="meta-llama/Llama-3.2-1B",
          model_size=ModelSize.SMALL,
          rtol=6e-3,
          atol=1e-3,
      ),
  )
  def test_hf_transformers_training(
      self,
      provider: str,
      module_name: str,
      model_size: ModelSize,
      rtol: float | None = None,
      atol: float | None = None,
  ) -> None:
    self._check_model_size(provider, module_name, model_size)

    # To reduce memory pressure, we first conduct a training step on CPU and
    # garbage collect the model and other objects we no longer need before doing
    # the train step on TPU.
    cpu_model, _ = self._create_model(provider, module_name, is_training=True)
    cpu_inputs = self._create_inputs(provider, module_name)
    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01)
    cpu_losses = []
    for _ in range(_NUM_TRAIN_STEPS):
      cpu_loss = _train_step(cpu_model, cpu_optimizer, cpu_inputs)
      cpu_losses.append(cpu_loss)
    del cpu_model, cpu_optimizer
    gc.collect()

    tpu_model, tpu_compiled_model = self._create_model(
        provider, module_name, device=self.tpu_device, is_training=True
    )
    tpu_inputs = {k: v.to(self.tpu_device) for k, v in cpu_inputs.items()}
    tpu_optimizer = torch.optim.SGD(tpu_model.parameters(), lr=0.01)
    tpu_losses = []
    for _ in range(_NUM_TRAIN_STEPS):
      tpu_loss = tpu_compiled_model(tpu_optimizer, tpu_inputs)
      tpu_losses.append(tpu_loss)

    for i in range(_NUM_TRAIN_STEPS):
      utils.assert_close(
          actual=tpu_losses[i],
          expected=cpu_losses[i],
          rtol=rtol,
          atol=atol,
          preamble="TPU vs CPU loss",
          check_value=utils.CheckValueMode.LOOSE,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="transformers/google/gemma-3-270m",
          provider="transformers",
          module_name="google/gemma-3-270m",
          model_size=ModelSize.TINY,
          rtol=6e-2,
          # This model's perplexity is actually lower than CPU by the tolerance
          # below which is good but the delta is somewhat large thus need to
          # specify a higher tolerance.
          atol=3e-1,
      ),
      dict(
          testcase_name="transformers/google/gemma-3-4b-pt",
          provider="transformers",
          module_name="google/gemma-3-4b-pt",
          model_size=ModelSize.MEDIUM,
          rtol=2e-3,
          atol=8e-2,
      ),
      dict(
          testcase_name="transformers/Qwen/Qwen3-0.6B",
          provider="transformers",
          module_name="Qwen/Qwen3-0.6B",
          model_size=ModelSize.SMALL,
          rtol=1e-3,
          atol=6e-2,
      ),
      dict(
          testcase_name="transformers/Qwen/Qwen-4B",
          provider="transformers",
          module_name="Qwen/Qwen3-4B",
          model_size=ModelSize.MEDIUM,
          rtol=2e-3,
          atol=6e-2,
      ),
      dict(
          testcase_name="transformers/meta-llama/Llama-3.2-1B",
          provider="transformers",
          module_name="meta-llama/Llama-3.2-1B",
          model_size=ModelSize.SMALL,
          rtol=5e-4,
          atol=1e-2,
      ),
  )
  def test_hf_transformers_inference(
      self,
      provider: str,
      module_name: str,
      model_size: ModelSize,
      rtol: float | None = None,
      atol: float | None = None,
  ) -> None:
    self._check_model_size(provider, module_name, model_size)

    # To reduce memory pressure, we first conduct inference on CPU and garbage
    # collect the model and other objects we no longer need before doing work
    # on the TPU.
    cpu_model, _ = self._create_model(provider, module_name)
    cpu_inputs = self._create_inputs(provider, module_name)
    with torch.inference_mode():
      cpu_out = cpu_model(**cpu_inputs)
    cpu_ppl = _calculate_perplexity(cpu_out.logits, cpu_inputs["input_ids"])
    del cpu_out, cpu_model
    gc.collect()

    _, tpu_compiled_model = self._create_model(
        provider, module_name, device=self.tpu_device
    )
    tpu_inputs = {k: v.to(self.tpu_device) for k, v in cpu_inputs.items()}
    with torch.inference_mode():
      tpu_out = tpu_compiled_model(**tpu_inputs)
    tpu_ppl = _calculate_perplexity(tpu_out.logits, tpu_inputs["input_ids"])

    self.assertLess(
        tpu_ppl,
        _PERPLEXITY_THRESHOLD,
        "TPU produced garbage results (high perplexity).",
    )
    utils.assert_close(
        actual=tpu_ppl,
        expected=cpu_ppl,
        rtol=rtol,
        atol=atol,
        preamble="TPU vs CPU perplexity",
        check_value=utils.CheckValueMode.LOOSE,
    )


if __name__ == "__main__":
  absltest.main()
