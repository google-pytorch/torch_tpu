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

"""Tests with torch.compile across PyTorch Image Models."""

import copy
import dataclasses
import random
from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from PIL import Image
import torch
import torch.accelerator
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.utils import utils
from tests import module_registry

_GOLDFISH_IMG_PATH = (
    epath.resource_path("torch_tpu") / "tests/compile/data/goldfish.jpg"
)


@dataclasses.dataclass
class ModelAndInputs:
  cpu_model: torch.nn.Module
  tpu_model: torch.nn.Module
  compiled_tpu_model: torch.nn.Module
  cpu_inputs: torch.Tensor
  tpu_inputs: torch.Tensor
  cpu_target: torch.Tensor | None
  tpu_target: torch.Tensor | None


def _get_logits(outputs):
  return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def _calculate_rmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
  """Calculates Root Mean Square Error."""
  return torch.sqrt(torch.mean((actual - expected) ** 2)).item()


def _load_real_image(
    img_path: epath.Path,
    transform: Callable[[Any], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
  """Load and preprocess test image for timm model."""
  with img_path.open("rb") as f:
    img = Image.open(f).convert("RGB")
    return transform(img).to(device)


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
  """Performs a single training step (Forward, Backward, Optimizer Step).

  Args:
    model: The model to train.
    optimizer: The optimizer to use.
    inputs: The input tensor.
    target: The target tensor.

  Returns:
    The calculated loss for the training step.
  """
  optimizer.zero_grad()

  outputs = model(inputs)
  logits = _get_logits(outputs)

  if target.dtype == torch.long:
    # For classification tasks (integer labels)
    loss = torch.nn.functional.cross_entropy(logits, target)
  else:
    # For feature extraction tasks (float targets)
    loss = torch.nn.functional.mse_loss(logits, target)

  loss.backward()
  optimizer.step()
  return loss


class ModelCoverageTimmTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.tpu_device = torch.accelerator.current_accelerator()
    cls.module_registry = module_registry.ModuleRegistry()

  def setUp(self):
    super().setUp()
    random.seed(42)
    torch.manual_seed(42)

  def _create_model_and_inputs(
      self,
      provider: str,
      module_name: str,
      *,
      is_training: bool = False,
      load_weights: bool = False,
      use_real_image: bool = False,
      img_path: epath.Path | None = None,
  ) -> ModelAndInputs:
    if use_real_image and img_path is None:
      raise ValueError("img_path must be provided when use_real_image is True.")

    module_spec = self.module_registry.get_module_spec(
        provider, module_name, load_weights=load_weights
    )
    cpu_model = module_spec.module_factory()

    if is_training:
      cpu_model.train()
    else:
      cpu_model.eval()

    tpu_model = copy.deepcopy(cpu_model).to(self.tpu_device)

    preprocessor = module_spec.preprocessor_factory()
    if use_real_image:
      if not preprocessor:
        raise ValueError(
            f"No preprocessor available for {module_name} to use real image."
        )
      cpu_inputs = _load_real_image(img_path, preprocessor, device="cpu")
    else:  # Default to random inputs
      args, _ = module_spec.sample_inputs_factory()
      cpu_inputs = args[0]

    tpu_inputs = cpu_inputs.to(self.tpu_device)

    cpu_target = None
    tpu_target = None

    # For training, generate targets based on the model's task
    if is_training:
      num_classes = getattr(cpu_model, "num_classes", 0)

      if num_classes > 0:
        # For classification tasks, generate integer labels
        cpu_target = torch.randint(0, num_classes, (cpu_inputs.shape[0],))
      else:
        # For feature extraction tasks, generate random float targets
        with torch.no_grad():
          dummy_output = _get_logits(cpu_model(cpu_inputs))
          out_shape = dummy_output.shape
        cpu_target = torch.randn(out_shape)

      tpu_target = cpu_target.to(self.tpu_device)

    compiled_tpu_model = torch.compile(
        tpu_model,
        dynamic=False,
        backend=torch_tpu_compile.TpuBackend(),
    )

    return ModelAndInputs(
        cpu_model=cpu_model,
        tpu_model=tpu_model,
        compiled_tpu_model=compiled_tpu_model,
        cpu_inputs=cpu_inputs,
        tpu_inputs=tpu_inputs,
        cpu_target=cpu_target,
        tpu_target=tpu_target,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="timm/convnext_small",
          provider="timm",
          module_name="convnext_small",
          rtol=1e-3,
          atol=1e-3,
      ),
      dict(
          testcase_name="timm/resnet50d",
          provider="timm",
          module_name="resnet50d",
          rtol=1e-3,
          atol=1e-3,
      ),
      dict(
          testcase_name="timm/vit_small_patch8_224",
          provider="timm",
          module_name="vit_small_patch8_224",
          rtol=1e-3,
          atol=1e-3,
      ),
  )
  def test_timm_training(
      self,
      provider: str,
      module_name: str,
      rtol: float | None = None,
      atol: float | None = None,
  ) -> None:
    model_and_inputs = self._create_model_and_inputs(
        provider, module_name, is_training=True
    )

    cpu_optimizer = torch.optim.SGD(
        model_and_inputs.cpu_model.parameters(), lr=0.01
    )
    tpu_optimizer = torch.optim.SGD(
        model_and_inputs.tpu_model.parameters(), lr=0.01
    )

    tpu_loss = _train_step(
        model_and_inputs.compiled_tpu_model,
        tpu_optimizer,
        model_and_inputs.tpu_inputs,
        model_and_inputs.tpu_target,
    )
    cpu_loss = _train_step(
        model_and_inputs.cpu_model,
        cpu_optimizer,
        model_and_inputs.cpu_inputs,
        model_and_inputs.cpu_target,
    )

    utils.assert_close(
        actual=tpu_loss,
        expected=cpu_loss,
        rtol=rtol,
        atol=atol,
        preamble="Timm Model TPU vs CPU Training Loss",
        check_value=utils.CheckValueMode.LOOSE,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="timm/convnext_small",
          provider="timm",
          module_name="convnext_small",
          rtol=1e-2,
          atol=1e-2,
      ),
      dict(
          testcase_name="timm/resnet50d",
          provider="timm",
          module_name="resnet50d",
          rtol=1e-3,
          atol=1e-3,
      ),
      dict(
          testcase_name="timm/vit_small_patch8_224",
          provider="timm",
          module_name="vit_small_patch8_224",
          rtol=1e-3,
          atol=3e-2,
      ),
  )
  def test_timm_inference(
      self,
      provider: str,
      module_name: str,
      rtol: float | None = None,
      atol: float | None = None,
  ) -> None:
    model_and_inputs = self._create_model_and_inputs(
        provider, module_name, is_training=False
    )

    cpu_out = model_and_inputs.cpu_model(model_and_inputs.cpu_inputs)
    tpu_out = model_and_inputs.compiled_tpu_model(model_and_inputs.tpu_inputs)

    cpu_logits = _get_logits(cpu_out)
    tpu_logits = _get_logits(tpu_out)

    utils.assert_close(
        actual=tpu_logits,
        expected=cpu_logits,
        rtol=rtol,
        atol=atol,
        preamble=f"Timm Model {module_name} TPU vs CPU Inference Logits",
        check_value=utils.CheckValueMode.LOOSE,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="convnext_small_e2e",
          provider="timm",
          module_name="convnext_small.in12k_ft_in1k",
          expected_class=1,  # Goldfish
          img_path=_GOLDFISH_IMG_PATH,
          rtol=1e-3,
          atol=4e-2,
          rmse_tol=1.1e-2,
          conf_tol=3e-3,
      ),
      dict(
          testcase_name="resnet50d_e2e",
          provider="timm",
          module_name="resnet50d.ra2_in1k",
          expected_class=1,  # Goldfish
          img_path=_GOLDFISH_IMG_PATH,
          rtol=1e-3,
          atol=1.3e-02,
          rmse_tol=4e-3,
          conf_tol=2e-3,
      ),
      dict(
          testcase_name="resnet50_e2e",
          provider="timm",
          module_name="resnet50",
          expected_class=1,  # Goldfish
          img_path=_GOLDFISH_IMG_PATH,
          rtol=2.1e-2,
          atol=1e-3,
          rmse_tol=4.5e-2,
          conf_tol=3e-3,
      ),
      dict(
          testcase_name="vgg16.tv_in1k_e2e",
          provider="timm",
          module_name="vgg16.tv_in1k",
          expected_class=1,  # Goldfish
          img_path=_GOLDFISH_IMG_PATH,
          rtol=1e-3,
          atol=5.4e-2,
          rmse_tol=1.3e-2,
          conf_tol=3e-3,
      ),
  )
  def test_timm_classification_inference_e2e(
      self,
      provider: str,
      module_name: str,
      expected_class: int | None = None,
      img_path: epath.Path | None = None,
      rtol: float | None = None,
      atol: float | None = None,
      rmse_tol: float | None = 1e-3,
      conf_tol: float | None = None,
  ) -> None:
    model_and_inputs = self._create_model_and_inputs(
        provider,
        module_name,
        is_training=False,
        load_weights=True,
        use_real_image=True,
        img_path=img_path,
    )

    cpu_out = model_and_inputs.cpu_model(model_and_inputs.cpu_inputs)
    tpu_out = model_and_inputs.compiled_tpu_model(model_and_inputs.tpu_inputs)

    cpu_logits = _get_logits(cpu_out)
    tpu_logits = _get_logits(tpu_out)

    with self.subTest("logits_check"):
      utils.assert_close(
          actual=tpu_logits,
          expected=cpu_logits,
          rtol=rtol,
          atol=atol,
          preamble=f"Model {module_name}: Logits Mismatch",
          check_value=utils.CheckValueMode.LOOSE,
      )

    with self.subTest("rmse_check"):
      rmse = _calculate_rmse(tpu_logits.cpu(), cpu_logits)
      self.assertLess(
          rmse,
          rmse_tol,
          f"Model {module_name}: RMSE={rmse:.6f} exceeded tolerance {rmse_tol}",
      )

    with self.subTest("prediction_accuracy_check"):
      cpu_pred = cpu_logits.argmax(dim=-1).item()
      tpu_pred = tpu_logits.argmax(dim=-1).item()
      self.assertEqual(
          cpu_pred,
          tpu_pred,
          f"Model {module_name}: TPU prediction ({tpu_pred}) does not match"
          f" CPU prediction ({cpu_pred}).",
      )
      if expected_class is not None:
        self.assertEqual(
            tpu_pred,
            expected_class,
            f"Model {module_name}: Expected class {expected_class},"
            f" got TPU predicted class {tpu_pred}",
        )

    with self.subTest("confidence_score_check"):
      cpu_conf = torch.nn.functional.softmax(cpu_logits, dim=-1).max().item()
      tpu_conf = torch.nn.functional.softmax(tpu_logits, dim=-1).max().item()
      self.assertAlmostEqual(
          tpu_conf,
          cpu_conf,
          delta=conf_tol,
          msg=(
              f"Model {module_name}: CPU confidence={cpu_conf:.6f},"
              f" TPU confidence={tpu_conf:.6f}"
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="vit_small_patch8_224_e2e",
          provider="timm",
          module_name="vit_small_patch8_224.dino",
          img_path=_GOLDFISH_IMG_PATH,
          rtol=1e-3,
          atol=9e-2,
          rmse_tol=1.7e-2,
          similarity_threshold=0.999992,
      ),
  )
  def test_timm_feature_extraction_inference_e2e(
      self,
      provider: str,
      module_name: str,
      img_path: epath.Path | None = None,
      rtol: float | None = None,
      atol: float | None = None,
      rmse_tol: float | None = 1e-3,
      similarity_threshold: float | None = 0.99,
  ) -> None:
    model_and_inputs = self._create_model_and_inputs(
        provider,
        module_name,
        is_training=False,
        load_weights=True,
        use_real_image=True,
        img_path=img_path,
    )

    cpu_out = model_and_inputs.cpu_model(model_and_inputs.cpu_inputs)
    tpu_out = model_and_inputs.compiled_tpu_model(model_and_inputs.tpu_inputs)

    cpu_logits = _get_logits(cpu_out)
    tpu_logits = _get_logits(tpu_out)

    with self.subTest("logits_check"):
      utils.assert_close(
          actual=tpu_logits,
          expected=cpu_logits,
          rtol=rtol,
          atol=atol,
          preamble=f"Model {module_name}: Logits Mismatch",
          check_value=utils.CheckValueMode.LOOSE,
      )

    with self.subTest("rmse_check"):
      rmse = _calculate_rmse(tpu_logits.cpu(), cpu_logits)
      self.assertLess(
          rmse,
          rmse_tol,
          f"Model {module_name}: RMSE={rmse:.6f} exceeded tolerance {rmse_tol}",
      )

    with self.subTest("shape_check"):
      self.assertEqual(tpu_logits.shape, cpu_logits.shape)

    with self.subTest(name="semantic_similarity"):
      tpu_feat = tpu_logits.cpu()
      cpu_feat = cpu_logits

      cos_sim = torch.nn.functional.cosine_similarity(tpu_feat, cpu_feat).item()
      print(f"[{module_name}] TPU vs CPU Cosine Similarity: {cos_sim:.6f}")
      self.assertGreater(
          cos_sim,
          similarity_threshold,
          f"Model {module_name}: Cosine similarity {cos_sim:.6f} below"
          f" threshold {similarity_threshold}",
      )


if __name__ == "__main__":
  absltest.main()
