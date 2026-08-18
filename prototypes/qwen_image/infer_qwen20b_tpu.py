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
"""Qwen 20B Image Pure-TPU Inference Execution Engine."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import argparse
import logging  # PYTHON_LOGGING_OK=Standalone prototype script
import torch

from prototypes.qwen_image.config import GenerationConfig, ModelConfig, TPUConfig
from prototypes.qwen_image.qwen_image_tpu import Qwen20BTPUPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Qwen20B_TPU_Main")


def parse_args():
  parser = argparse.ArgumentParser(
      description="Qwen 20B Image Pure-TPU Inference"
  )

  # TPU Hardware & Sharding Args
  parser.add_argument(
      "--sharding_strategy",
      type=str,
      default="spmd",
      choices=["spmd", "layer", "fsdp", "offload", "auto", "single"],
      help=(
          "TPU model sharding strategy: spmd (Mesh Sharding), layer (Pipeline"
          " Placement), fsdp, auto, or single"
      ),
  )
  parser.add_argument(
      "--num_devices",
      type=int,
      default=4,
      help="Number of TPU devices/chips to utilize",
  )
  parser.add_argument(
      "--tpu_accelerator_type",
      type=str,
      default="v6e-4",
      help="GCP TPU accelerator type (e.g. v6e-1, v6e-4, v6e-8)",
  )
  parser.add_argument(
      "--worker_hostnames",
      type=str,
      default="127.0.0.1",
      help="TPU worker hostnames/IPs",
  )

  # Model & Adapter Args
  parser.add_argument(
      "--model_id",
      type=str,
      default="Qwen/Qwen-Image",
      help="Hugging Face model ID or path",
  )
  parser.add_argument(
      "--lora_weights_path",
      type=str,
      default="./output_qwen_image_tpu",
      help="Path to trained PEFT LoRA adapter directory",
  )

  # Generation Args
  parser.add_argument(
      "--output_dir",
      type=str,
      default="./qwen_image_tpu_results",
      help="Directory to save generated images",
  )
  parser.add_argument(
      "--num_inference_steps",
      type=int,
      default=20,
      help="Number of diffusion denoising steps",
  )
  parser.add_argument(
      "--guidance_scale",
      type=float,
      default=7.5,
      help="Classifier-free guidance scale",
  )
  parser.add_argument(
      "--resolution",
      type=int,
      default=512,
      help="Image resolution height and width",
  )
  parser.add_argument(
      "--seed", type=int, default=42, help="Random seed for generation"
  )
  parser.add_argument(
      "--custom_prompts",
      type=str,
      default=None,
      help="Semicolon-separated list of custom prompts",
  )
  parser.add_argument(
      "--use_float32",
      action="store_true",
      default=False,
      help="Use float32 instead of default bfloat16",
  )

  return parser.parse_args()


def main():
  args = parse_args()

  tpu_config = TPUConfig(
      sharding_strategy=args.sharding_strategy,
      num_devices=args.num_devices,
      tpu_accelerator_type=args.tpu_accelerator_type,
      worker_hostnames=args.worker_hostnames,
      precision=torch.float32 if args.use_float32 else torch.bfloat16,
  )

  model_config = ModelConfig(
      model_id=args.model_id,
      lora_weights_path=args.lora_weights_path,
  )

  if args.custom_prompts:
    prompts = [p.strip() for p in args.custom_prompts.split(";") if p.strip()]
  else:
    prompts = [
        (
            "1girl, solo, anime girl face, blue eyes, long silver hair,"
            " detailed portrait, masterpiece"
        ),
        (
            "1boy, solo, anime boy face, red hair, heroic smile, portrait, high"
            " quality"
        ),
    ]

  gen_config = GenerationConfig(
      num_inference_steps=args.num_inference_steps,
      guidance_scale=args.guidance_scale,
      height=args.resolution,
      width=args.resolution,
      seed=args.seed,
      prompts=prompts,
      output_dir=args.output_dir,
  )

  pipeline = Qwen20BTPUPipeline(tpu_config, model_config)
  pipeline.generate(gen_config)


if __name__ == "__main__":
  main()
