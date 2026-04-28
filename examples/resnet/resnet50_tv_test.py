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

"""Test ResNet50 from torchvision.

Resnets were invented before bfloat16 was popular. The model is
typically run as FP32.
"""

import io

from absl import flags
from absl import logging
from absl.testing import absltest
from etils import epath
from PIL import Image
import torch
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import tracer_utils
from torch_tpu._internal.utils import utils
from examples import paths
import torchvision


log_utils.log_to_stderr()


_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["cuda", "tpu"],
    "Device to run the test on.",
    required=True,
)

WEIGHT_PATH = f"{paths.XM_HOME}weights/torchvision/resnet50_imagenet1k_v1.pt"


class ResNet50TVTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    seed = absltest.FLAGS.test_random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Safe to call even if not using CUDA.
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    self.device = torch.device(_DEVICE.value)
    # TODO(pganssle): Evaluate whether this assertion is still necessary.
    assert str(self.device).split(":", 1)[0] == _DEVICE.value

    torch.set_default_device(self.device)

  def test_resnet50_forward_returns_logits_of_shape_1x1000(self):
    # Arrange
    model = torchvision.models.resnet50()
    model.eval()
    model.to(self.device)
    img = torch.randn([1, 3, 224, 224]).to(self.device)

    # Act
    # Unlike keras models, torchvision models output raw logits.
    output = model(img)

    # Assert
    self.assertEqual(output.shape, (1, 1000))

  def _load_goldfish_image(self):
    """Internal helper to load the goldfish image."""
    img_path = epath.Path(__file__).parent / "goldfish.jpg"
    img = Image.open(io.BytesIO(img_path.read_bytes())).convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize([224, 224], antialias=True),
        torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ])
    return transform(img).unsqueeze(0).to(self.device)

  def test_load_weights(self):
    """Tests that resnet50 weights load correctly."""
    # Arrange
    model = torchvision.models.resnet50(weights=None)
    model.to(self.device)
    model.eval()
    weights_path = epath.Path(WEIGHT_PATH)
    img = self._load_goldfish_image()

    # Act
    with weights_path.open("rb") as f:
      state_dict = torch.load(f)
    model.load_state_dict(state_dict)

    predicted_class = model(img).argmax(dim=-1).squeeze().item()

    # Assert
    # Goldfish class is 1 in ImageNet.
    self.assertEqual(predicted_class, 1)

  def test_resnet50_fwd_numerics_tpu_to_cpu_are_close(self):
    """Tests that resnet50 forward pass numerics on TPU are close to CPU."""
    if _DEVICE.value not in ("tpu", "cuda"):
      self.skipTest("This test is only supported on tpu and cuda devices.")

    # Arrange
    model = torchvision.models.resnet50(weights=None)
    model.to(self.device)
    model.eval()
    weights_path = epath.Path(WEIGHT_PATH)
    with weights_path.open("rb") as f:
      state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    img = self._load_goldfish_image()

    # Act
    with torch.no_grad(), utils.ActivationTracer(model) as tracer:
      model(img)
    replayed_log = tracer_utils.replay_log(tracer.forward_log, device="cpu")

    # Print
    formatted_replay = tracer_utils.pformat_replay(
        tracer.forward_log, tracer.forward_pre_log, replayed_log
    )
    print(formatted_replay)

    # Assert
    for acc_event, cpu_event in zip(tracer.forward_log, replayed_log):
      print(
          f"Checking event from {acc_event['module']=}:\n"
          f"{utils.get_tensor_summary(acc_event['output'])=}\n"
          f"{utils.get_tensor_summary(cpu_event['output'])=}\n"
      )
      if isinstance(acc_event["module"], torch.nn.BatchNorm2d):
        # TODO: Investigate discrepancy in batchnorms
        continue
      try:
        utils.assert_close(
            acc_event["output"],
            cpu_event["output"],
            atol=1e-4,
            rtol=1e-2,
            check_value=utils.CheckValueMode.LOOSE,
        )
      except AssertionError as e:
        raise AssertionError(
            f"Unexpected diff in output for {acc_event['module']}"
        ) from e


if __name__ == "__main__":
  absltest.main()
