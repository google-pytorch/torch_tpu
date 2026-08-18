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
"""Configuration dataclasses for Qwen 20B Image Pure-TPU Inference."""

from dataclasses import dataclass, field
import os
from typing import List, Optional, Tuple
import torch

# Disable Flax and TensorFlow backends in diffusers/transformers so only the
# PyTorch backend is loaded (also avoids importing flax, which fails on
# Python 3.14 with KeyError: '__annotations__').
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")


@dataclass
class TPUConfig:
  """TPU hardware and sharding environment configuration."""

  device_type: str = "tpu"  # "tpu" or "cpu"
  sharding_strategy: str = "spmd"  # "spmd", "layer", "fsdp", or "auto"
  num_devices: int = 4
  mesh_shape: Tuple[int, ...] = (1, 4)
  precision: torch.dtype = torch.bfloat16
  skip_mds_query: bool = True
  tpu_accelerator_type: str = "v6e-4"
  worker_hostnames: str = "127.0.0.1"
  worker_id: int = 0


@dataclass
class ModelConfig:
  """Qwen 20B Image model loading and adapter configuration."""

  model_id: str = "Qwen/Qwen-Image"
  lora_weights_path: Optional[str] = None
  use_safetensors: bool = True
  offload_text_encoder_to_cpu: bool = False  # False = Pure TPU execution
  offload_vae_to_cpu: bool = False  # False = Pure TPU execution


@dataclass
class GenerationConfig:
  """Diffusion generation parameters for Qwen 20B Image model."""

  num_inference_steps: int = 50
  guidance_scale: float = 7.5
  height: int = 512
  width: int = 512
  seed: int = 42
  prompts: List[str] = field(
      default_factory=lambda: [
          "a green and yellow toy with a red nose",
          "a picture of a cartoon character with a sword",
          "a drawing of a cartoon character doing a kick",
      ]
  )
  output_dir: str = "./qwen_image_tpu_results"
