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

"""Smoke test of HuggingFace Wan2.2-TI2V-5B inference."""

import pathlib

from absl import flags
from absl import logging
from absl.testing import absltest
import diffusers
from PIL import Image
import torch
from torch_tpu._internal.utils import log_utils
from examples import paths

log_utils.log_to_stderr()


XM_HOME = paths.XM_HOME
_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cpu"],
    required=True,
    help="Accelerator to test.",
)
_MODEL_PATH = str(
    pathlib.Path(XM_HOME)
    / "weights"
    / "huggingface"
    / "Wan-AI"
    / "Wan2.2-TI2V-5B-Diffusers"
)
_D_TYPE = torch.bfloat16


class Wan2_2_TI2V_5B_SmokeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # This abstest flag will always be set to an int.
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError(f"absltest.FLAGS.test_random_seed not an int: {seed}")

    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    self.accelerator_device = torch.device(_DEVICE.value)
    # TODO(pganssle): Evaluate whether this assertion is still necessary.
    assert str(self.accelerator_device).split(":", 1)[0] == _DEVICE.value

  def tearDown(self):
    super().tearDown()

  def test_wan_inference(self):
    vae = diffusers.AutoencoderKLWan.from_pretrained(
        _MODEL_PATH, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = diffusers.WanPipeline.from_pretrained(
        _MODEL_PATH, vae=vae, torch_dtype=_D_TYPE
    ).to(self.accelerator_device)

    prompt = "A cat holding a sign that says hello world"

    if _DEVICE.value == "cpu":
      generator = torch.manual_seed(0)
    else:
      generator = torch.Generator(device=self.accelerator_device).manual_seed(0)

    height = 704
    width = 1280
    num_frames = 30  # Over 89 frames, we run out of memory on single-host TPU.
    num_inference_steps = 50
    guidance_scale = 5.0

    video = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]

    generated_video = video[0]
    logging.info(
        "Generated video floats (first frame, small slice): %s",
        generated_video[:2, :2, :2],
    )

    # Assert
    # Trivial assert, this smoke test just tests this doesn't crash.
    self.assertIsNotNone(generated_video)


if __name__ == "__main__":
  absltest.main()
