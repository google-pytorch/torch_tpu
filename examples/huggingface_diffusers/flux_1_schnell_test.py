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

"""Smoke tests of HuggingFace FLUX.1-schnell inference and training.

HuggingFace Model Card:
https://huggingface.co/black-forest-labs/FLUX.1-schnell

A guide to Flux model architecture: https://arxiv.org/pdf/2507.09595
"""

from absl import flags
from absl import logging
from absl.testing import absltest
import diffusers
import torch
import torch.nn.functional as F
from torch_tpu._internal.utils import log_utils
from examples import paths


log_utils.log_to_stderr()


XM_HOME = paths.XM_HOME
_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cuda", "cpu"],
    required=True,
    help="Accelerator to test.",
)


class Flux1SmokeTest(absltest.TestCase):

  MODEL_PATH = f"{XM_HOME}weights/huggingface/black-forest-labs/FLUX.1-schnell/"
  CODE_MODEL_PATH = "third_party/py/torch_tpu/examples/huggingface_diffusers/model_configs/black-forest-labs/FLUX.1-schnell"
  D_TYPE = torch.bfloat16

  def setUp(self):
    super().setUp()
    # This abstest flag will always be set to an int.
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)

    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    self.accelerator_device = torch.device(_DEVICE.value)
    # TODO(pganssle): Evaluate whether this assertion is still necessary.
    assert str(self.accelerator_device).split(":", 1)[0] == _DEVICE.value
    torch.set_default_device(self.accelerator_device)

  def tearDown(self):
    super().tearDown()
    if _DEVICE.value == "cuda":
      # Clear CUDA cache between tests.
      torch.cuda.empty_cache()

  def test_flux1_inference(self):
    # Act
    pipe = diffusers.FluxPipeline.from_pretrained(
        self.MODEL_PATH, torch_dtype=self.D_TYPE
    ).to(self.accelerator_device)

    prompt = "A cat holding a sign that says hello world"

    image = pipe(
        prompt,
        height=32,
        width=32,
        guidance_scale=3.5,
        num_inference_steps=2,
        max_sequence_length=32,
        generator=torch.Generator(self.accelerator_device).manual_seed(0),
    ).images[0]

    # Assert
    # Trivial assert, this smoke test just tests this doesn't crash.
    self.assertIsNotNone(image)

  # TODO(b/503472873): Fix the "tensor does not have a device" error
  # triggered by elementwise_affine=False in LayerNorm on TPU.
  @absltest.skip("Fails with tensor does not have a device")
  def test_flux1_training_mini(self):
    # Model miniaturization
    config = diffusers.FluxTransformer2DModel.load_config(
        self.CODE_MODEL_PATH,
        subfolder="transformer",
    )
    config["in_channels"] = 4
    config["joint_attention_dim"] = 32
    config["num_attention_heads"] = 2
    config["num_layers"] = 1
    config["num_single_layers"] = 2
    config["pooled_projection_dim"] = 32
    model = diffusers.FluxTransformer2DModel.from_config(config).to(
        device=self.accelerator_device
    )
    model = model.to(self.D_TYPE)

    # Create dummy inputs
    batch_size = 1
    in_channels = model.config.in_channels
    joint_dim = model.config.joint_attention_dim
    pooled_dim = model.config.pooled_projection_dim
    img_seq_len = 8
    text_seq_len = 16

    # hidden states - These would be image patch embeddings derived from the
    # VAE with noise added according to scheduler and timestep.
    hidden_states = torch.randn(
        batch_size,
        img_seq_len,  # The number of image patches.
        in_channels,  # Embedding dimension of each image patch.
        requires_grad=True,
        device=self.accelerator_device,
        dtype=self.D_TYPE,
    )
    # encoder hidden states - These would be per token text embeddings returned
    # by T5 (google/t5-v1_1-xxl) text encoder
    encoder_hidden_states = torch.randn(
        batch_size,
        text_seq_len,
        joint_dim,
        device=self.accelerator_device,
        dtype=self.D_TYPE,
    )
    # pooled projections - These would be pooled text embeddings for the entire
    # text prompt returned by CLIP (openai/clip-vit-large-patch14) text encoder
    pooled_projections = torch.randn(
        batch_size,
        pooled_dim,
        device=self.accelerator_device,
        dtype=self.D_TYPE,
    )
    # timestep - The current timestep of the diffusion process.
    timestep = torch.tensor(
        [999], dtype=torch.long, device=self.accelerator_device
    )

    # Number of axes for RoPE calculations; 3 for (time, height, width)
    num_axes = 3

    # Positional encodings for each text token embedding fed to
    # FluxTransformer2DModel for RoPE calculation in attention blocks.
    txt_ids = torch.randint(
        0,
        100,
        (text_seq_len, num_axes),
        dtype=self.D_TYPE,
        device=self.accelerator_device,
    )
    # Positional encodings for each image patch embedding fed to
    # FluxTransformer2DModel for RoPE calculations in attention blocks.
    img_ids = torch.randint(
        0,
        100,
        (img_seq_len, num_axes),
        dtype=self.D_TYPE,
        device=self.accelerator_device,
    )

    # TODO(jeremiahhsu): Consider adding noise scheduler instead of random
    # initialization for completeness
    noise = torch.randn_like(hidden_states).to(
        device=self.accelerator_device, dtype=self.D_TYPE
    )

    # -- Act --
    # Perform 3 backward passes.
    model.train()

    # Initialize optimizer with low LR for miniaturized model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    for _ in range(3):
      optimizer.zero_grad()

      out = model(
          hidden_states=hidden_states,
          timestep=timestep,
          encoder_hidden_states=encoder_hidden_states,
          pooled_projections=pooled_projections,
          txt_ids=txt_ids,
          img_ids=img_ids,
      )
      loss = F.mse_loss(out.sample, noise)
      losses.append(loss.item())

      loss.backward()
      optimizer.step()

    # Assert
    # We test that loss decreases after 3 backward passes.
    self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
  absltest.main()
