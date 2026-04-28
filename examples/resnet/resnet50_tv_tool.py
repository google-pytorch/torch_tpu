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

"""Tools to work with Resnet50.

These are not tests, but are structured as such to run on forge-ml.
"""

from absl import flags
from absl import logging
from absl.testing import absltest
from etils import epath
import torch
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import tracer_utils
from torch_tpu._internal.utils import utils
import torchvision


log_utils.log_to_stderr()


_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["cuda", "tpu", "cpu"],
    "Device to run the test on.",
    required=True,
)


class Resnet50TVToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    seed = absltest.FLAGS.test_random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Safe to call even if not using CUDA.
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    self.accelerator_device = torch.accelerator.current_accelerator()
    # TODO(jialeic): Investigate whether we need this flag at all
    assert str(self.accelerator_device) == _DEVICE.value

    torch.set_default_device(self.accelerator_device)

  def test_run_forward_with_op_tracer(self):
    r"""Runs OpTracer on ResNet50.

    Run the op tracer on CUDA:

    ```
    blaze test -c opt --config=cuda --test_output=all \
      //examples/resnet:resnet50_tv_tool_cuda \
      --test_filter=Resnet50TVToolTest.test_run_forward_with_op_tracer
    ```

    Run the op tracer on TPU:

    ```
    blaze test -c opt --test_output=all \
      //examples/resnet:resnet50_tv_tool_tpu \
      --test_filter=Resnet50TVToolTest.test_run_forward_with_op_tracer
    ```
    """
    with utils.OpTracer() as tracer:
      model = torchvision.models.resnet50()
      model.eval()
      model.to(self.accelerator_device)
      img = torch.randn([1, 3, 224, 224]).to(self.accelerator_device)
      model(img)

    # Log the output of the OpTracer
    op_trace_report = tracer_utils.pformat_op_tracer(tracer)
    logging.info("OpTracer Report:\n%s", op_trace_report)

  def test_store_weights(self):
    r"""Loads resnet50 weights from pytorch.org into ./weights/.

    For resnet50, run the command to save to local directory.

    ```sh
    blaze test -c opt --test_output=all \
      //examples/resnet:resnet50_tv_tool_cpu \
      --test_filter=Resnet50TVToolTest.test_store_weights \
      --test_strategy=local
    ```

    Then copy the weights into storage and make them globally readable:

    ```sh
    fileutil mkdir -p {paths.XM_HOME}weights/torchvision && \
    fileutil cp -R /tmp/weights.pt \
      {paths.XM_HOME}weights/torchvision/resnet50_imagenet1k_v1.pt && \
    fileutil chmod a+r {paths.XM_HOME}weights/torchvision/resnet50_imagenet1k_v1.pt

    ```
    """

    model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    path = epath.Path("/tmp/weights.pt")
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
  absltest.main()
