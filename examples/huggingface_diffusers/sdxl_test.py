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

"""Smoke tests of HuggingFace stable-diffusion-xl-base-1.0 inference and training.

HuggingFace Model Card:
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

SDXL Paper: https://arxiv.org/pdf/2307.01952
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


_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cuda", "cpu"],
    required=True,
    help="Accelerator to test.",
)


class SDXLSmokeTest(absltest.TestCase):
  """Tests for SDXL inference and training using HuggingFace."""

  # Model path with weights
  MODEL_PATH = f"{paths.XM_HOME}weights/huggingface/stabilityai/stable-diffusion-xl-base-1.0"
  CODE_MODEL_PATH = "third_party/py/torch_tpu/examples/huggingface_diffusers/model_configs/stabilityai/stable-diffusion-xl-base-1.0"
  # Model path on codebase with configs only
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

  def tearDown(self):
    super().tearDown()
    if _DEVICE.value == "cuda":
      # Clear CUDA cache between tests.
      torch.cuda.empty_cache()

  def test_sdxl_base_inference(self):
    """Test for SDXL inference using HuggingFace DiffusionPipeline and real weights.

    Adapted from reference:
    https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

    We use HuggingFace's Diffusion Pipeline to load the SDXL pipeline
    # consisting of:
    # 1. Two text encoders for per token and pooled text embeddings -
    #    (OpenCLIP-ViT/G & CLIP-ViT/L)
    # 2. SDXL UNet-Base (Denoising model)
    # 3. DDPMS Noise Scheduler
    # 4. VAE encoder and decoder
    # This is the SDXL base pipeline with no refiner.
    # TODO(jeremiahhsu): Add test with SDXL base + refiner pipeline.
    """

    # Act
    pipe = diffusers.DiffusionPipeline.from_pretrained(
        self.MODEL_PATH,
        torch_dtype=self.D_TYPE,
        use_safetensors=True,
        device=self.accelerator_device,
    )

    prompt = "An astronaut riding a green horse"

    out = pipe(prompt=prompt, height=8, width=8, num_inference_steps=1)
    image = out.images[0]

    # Assert
    # Trivial assert, this smoke test just tests this doesn't crash.
    self.assertIsNotNone(image)

  def test_sdxl_base_training(self):
    """Test backward pass of SDXL UNet diffusion model.

    This tests with random weights and dummy inputs for conditioning the
    unet expected in the SDXL pipeline.
    """

    # -- Arrange --
    # We minitaurize the model config for smoke tests.
    # Full training of larger diffusion models (i.e Flux.1-Schnell 12B) do not
    # fit on a single A100 accelerator. So, we miniaturize the architecture of
    # the denoising model and pass the expected latents and additional
    # conditioning to accurately simulate training steps.

    unet_config = diffusers.UNet2DConditionModel.load_config(
        self.CODE_MODEL_PATH, subfolder="unet"
    )
    unet_config["sample_size"] = 16
    unet_config["block_out_channels"] = (32, 64)
    unet_config["down_block_types"] = ("DownBlock2D", "CrossAttnDownBlock2D")
    unet_config["up_block_types"] = ("CrossAttnUpBlock2D", "UpBlock2D")
    unet_config["transformer_layers_per_block"] = (1, 1)
    unet_config["attention_head_dim"] = (4, 4)
    unet_config["cross_attention_dim"] = 32
    unet_config["addition_embed_type"] = "text_time"
    unet_config["addition_time_embed_dim"] = 4
    unet_config["projection_class_embeddings_input_dim"] = (
        unet_config["cross_attention_dim"]
        + 6 * unet_config["addition_time_embed_dim"]
    )

    # Load miniaturized unet model
    unet = diffusers.UNet2DConditionModel.from_config(unet_config).to(
        self.accelerator_device
    )
    # Load noise scheduler. The noise scheduler returns noisy latents for
    # each timestep.
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(
        self.CODE_MODEL_PATH, subfolder="scheduler"
    )

    # Create dummy inputs
    batch_size = 1
    latent_channels = unet.config.in_channels
    latent_height = unet.config.sample_size
    latent_width = unet.config.sample_size

    latents = torch.randn(
        (batch_size, latent_channels, latent_height, latent_width)
    ).to(self.accelerator_device)
    noise = torch.randn_like(latents)

    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
    ).to(self.accelerator_device)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Encoder hidden states - These would be per token text embeddings returned
    # by CLIP-ViT/L text encoder
    dummy_encoder_hidden_states = torch.randn((
        batch_size,
        77,  # Maximum text sequence length from CLIP `max_position_embeddings`
        unet.config.cross_attention_dim,
    )).to(self.accelerator_device)

    # Create dummy additional conditioning inputs which would be expected in an
    # SDXL pipeline.

    # Pooled text embeddings - A single embedding vector per batch representing
    # the entire text sequence returned by OpenCLIP-ViT/G
    dummy_pooled_embeds = torch.randn(
        (batch_size, unet.config.cross_attention_dim)
    ).to(self.accelerator_device)

    # time_ids - Tensor of shape [batch_size, 6] providing crop information
    # to the unet (nothing to do with timesteps).
    # Represents: [orig_height, orig_width, crops_coords_top, crops_coords_left,
    # target_height, target_width]
    # Here, we initialize random values.
    dummy_time_ids = torch.randn((batch_size, 6)).to(self.accelerator_device)
    added_cond_kwargs = {
        "text_embeds": dummy_pooled_embeds,
        "time_ids": dummy_time_ids,
    }

    # -- Act --
    # Perform 3 backward passes.
    unet.train()

    # Initialize optimizer with low LR for miniaturized model.
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-3)
    losses = []

    for _ in range(3):
      # Forward pass
      out = unet(
          noisy_latents,
          timesteps,
          dummy_encoder_hidden_states,
          added_cond_kwargs=added_cond_kwargs,
          return_dict=False,
      )[0]

      loss = F.mse_loss(out, noise)
      losses.append(loss.item())

      # Backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # -- Assert --
    # We test that loss decreases after 3 backward passes.
    self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
  absltest.main()
