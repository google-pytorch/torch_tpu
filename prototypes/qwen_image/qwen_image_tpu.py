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

#!/usr/bin/env python3
"""Qwen 20B Image Pure-TPU Inference Pipeline Wrapper."""

import gc
import logging  # PYTHON_LOGGING_OK=Standalone prototype script
import os
from typing import List
from PIL import Image
import torch
from prototypes.qwen_image.config import GenerationConfig, ModelConfig, TPUConfig
from prototypes.qwen_image.tpu_sharding import TPUShardingManager, mark_step_tpu

logger = logging.getLogger("Qwen20B_TPU_Pipeline")


class Qwen20BTPUPipeline:
  """Pure-TPU Inference Pipeline for Qwen 20B Image Model."""

  def __init__(self, tpu_config: TPUConfig, model_config: ModelConfig):
    self.tpu_config = tpu_config
    self.model_config = model_config
    self.sharding_manager = TPUShardingManager(tpu_config)
    self.primary_device = self.sharding_manager.primary_device
    self.dtype = tpu_config.precision

    logger.info(
        f"Initializing Qwen 20B TPU Pipeline on device: {self.primary_device}"
        f" (dtype={self.dtype})"
    )

    self.pipe = None
    self.transformer = None
    self.vae = None
    self._load_pipeline()

  def _load_pipeline(self) -> None:
    """Loads Diffusers QwenImagePipeline and applies TPU sharding and LoRA adapters."""
    logger.info(
        "[1/3] Loading base model components from"
        f" '{self.model_config.model_id}'..."
    )
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline

    try:
      from diffusers import QwenImagePipeline

      pipe_cls = QwenImagePipeline
    except ImportError:
      pipe_cls = DiffusionPipeline

    try:
      self.pipe = pipe_cls.from_pretrained(
          self.model_config.model_id,
          torch_dtype=self.dtype,
          use_safetensors=self.model_config.use_safetensors,
          low_cpu_mem_usage=True,
          device_map="cpu",
      )
      logger.info("Loaded QwenImagePipeline successfully!")
    except Exception as e:
      logger.warning(
          f"Could not load native QwenImagePipeline directly ({e}). Falling"
          " back to AutoPipeline."
      )
      self.pipe = AutoPipelineForText2Image.from_pretrained(
          self.model_config.model_id,
          torch_dtype=self.dtype,
          use_safetensors=self.model_config.use_safetensors,
          low_cpu_mem_usage=True,
          device_map="cpu",
      )

    if hasattr(self.pipe, "transformer"):
      self.transformer = self.pipe.transformer
    elif hasattr(self.pipe, "unet"):
      self.transformer = self.pipe.unet

    if self.model_config.lora_weights_path and os.path.exists(
        self.model_config.lora_weights_path
    ):
      logger.info(
          "[2/3] Injecting PEFT LoRA adapter from"
          f" '{self.model_config.lora_weights_path}'..."
      )
      try:
        diff_adapter = os.path.join(
            self.model_config.lora_weights_path,
            "clean_diffusers_adapter.safetensors",
        )
        if hasattr(self.pipe, "load_lora_weights") and os.path.exists(
            diff_adapter
        ):
          self.pipe.load_lora_weights(
              self.model_config.lora_weights_path,
              weight_name="clean_diffusers_adapter.safetensors",
          )
          logger.info("Loaded Diffusers LoRA adapter weights successfully!")
        else:
          from peft import PeftModel

          self.transformer = PeftModel.from_pretrained(
              self.pipe.transformer, self.model_config.lora_weights_path
          )
          if hasattr(self.pipe, "transformer"):
            self.pipe.transformer = self.transformer
          elif hasattr(self.pipe, "unet"):
            self.pipe.unet = self.transformer
          logger.info(
              "PEFT LoRA adapter injected into Qwen 20B Transformer"
              " successfully!"
          )
      except Exception as e:
        logger.error(f"Failed to inject LoRA adapter: {e}")

    logger.info("[3/3] Ensuring text encoders & VAE reside on host CPU RAM...")
    if (
        hasattr(self.pipe, "text_encoder")
        and self.pipe.text_encoder is not None
    ):
      self.pipe.text_encoder.to("cpu")
    if (
        hasattr(self.pipe, "text_encoder_2")
        and self.pipe.text_encoder_2 is not None
    ):
      self.pipe.text_encoder_2.to("cpu")
    if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
      self.pipe.vae.to("cpu")

    if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
      orig_decode = self.pipe.vae.decode

      def safe_decode(latents, *args, **kwargs):
        vae_dev = next(self.pipe.vae.parameters()).device
        if vae_dev.type != "meta":
          latents = latents.to(vae_dev)
        return orig_decode(latents, *args, **kwargs)

      self.pipe.vae.decode = safe_decode

    self.sharding_manager.shard_model(self.pipe)  # pytype: disable=bad-argument-type

    gc.collect()
    mark_step_tpu()
    logger.info("Qwen 20B TPU Pipeline initialization complete!")

  def generate(self, gen_config: GenerationConfig) -> List[Image.Image]:
    """Generates images on TPU using configured prompts and parameters."""
    prompts = gen_config.prompts
    logger.info(
        f"Starting Qwen 20B TPU generation for {len(prompts)} prompts..."
    )

    os.makedirs(gen_config.output_dir, exist_ok=True)
    generated_images = []

    with torch.no_grad():
      for idx, prompt in enumerate(prompts, 1):
        logger.info(f"Generating image [{idx}/{len(prompts)}]: '{prompt}'...")

        mark_step_tpu()
        gc.collect()

        output = self.pipe(  # pytype: disable=not-callable
            prompt=prompt,
            negative_prompt=(
                "3d render, plastic toy, photograph, realistic photo, glossy"
                " plastic, low quality"
            ),
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            height=gen_config.height,
            width=gen_config.width,
            generator=torch.Generator(device="cpu").manual_seed(
                gen_config.seed + idx
            ),
        )
        image = output.images[0]
        output_path = os.path.join(
            gen_config.output_dir, f"qwen20b_tpu_prompt_{idx}.png"
        )
        image.save(output_path)
        logger.info(f"Saved generated image to '{output_path}'")
        generated_images.append(image)

        gc.collect()
        mark_step_tpu()

    return generated_images
