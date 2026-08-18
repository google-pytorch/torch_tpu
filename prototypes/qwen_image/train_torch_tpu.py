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

import argparse
import gc
import math
import os
import sys
import time
import numpy as np
from safetensors.torch import save_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add local example directory and torch_tpu repo root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from prototypes.qwen_image.config import ModelConfig
from prototypes.qwen_image.utils import convert_peft_to_diffusers_adapter

# Initialize TPU via torch_tpu
try:
  import torch_tpu

  _ = torch_tpu
  if hasattr(torch, "tpu") and torch.tpu.is_available():
    device = torch.device("tpu")
    USE_TPU = True
    print(f"[Train PyTorch TPU] Initialized Native Torch TPU Device: {device}")
  else:
    USE_TPU = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
  print(
      f"[Train PyTorch TPU Warning] Could not initialize torch_tpu ({e})."
      " Falling back to CUDA/CPU."
  )
  USE_TPU = False
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rms_norm(x, eps=1e-6):
  x_float = x.float()
  norm = x_float * torch.rsqrt(
      torch.mean(x_float**2, dim=-1, keepdim=True) + eps
  )
  return norm.to(dtype=x.dtype)


def get_timestep_embedding(
    timesteps, embedding_dim=256, max_period=10000, dtype=torch.bfloat16
):
  half = embedding_dim // 2
  freqs = torch.exp(
      -math.log(max_period)
      * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
      / half
  )
  args = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0)
  embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
  return embedding.to(dtype=dtype)


def get_3d_rope(seq_len=1024, head_dim=128, device="cpu"):
  grid_size = int(math.sqrt(seq_len))
  grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
  pos_x = grid_x.flatten()
  pos_y = grid_y.flatten()

  dim = head_dim // 2
  freqs = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))

  cos_x = np.cos(np.outer(pos_x, freqs))
  sin_x = np.sin(np.outer(pos_x, freqs))
  cos_y = np.cos(np.outer(pos_y, freqs))
  sin_y = np.sin(np.outer(pos_y, freqs))

  cos_3d = np.concatenate([cos_x, cos_y], axis=-1)
  sin_3d = np.concatenate([sin_x, sin_y], axis=-1)

  cos_3d = np.tile(cos_3d, (1, head_dim // cos_3d.shape[-1]))
  sin_3d = np.tile(sin_3d, (1, head_dim // sin_3d.shape[-1]))

  return torch.tensor(
      cos_3d, dtype=torch.bfloat16, device=device
  ), torch.tensor(sin_3d, dtype=torch.bfloat16, device=device)


def apply_3d_rope(x, cos_pos, sin_pos):
  cos = cos_pos.unsqueeze(0).unsqueeze(2).to(dtype=x.dtype)
  sin = sin_pos.unsqueeze(0).unsqueeze(2).to(dtype=x.dtype)

  d_half = x.shape[-1] // 2
  x1 = x[..., :d_half]
  x2 = x[..., d_half:]
  x_rot = torch.cat([-x2, x1], dim=-1)

  return (x * cos) + (x_rot * sin)


class TimestepEmbedder(nn.Module):

  def __init__(self, hidden_dim=3584):
    super().__init__()
    self.t_mlp_1 = nn.Linear(256, hidden_dim)
    self.t_mlp_2 = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, t, target_dtype=torch.bfloat16):
    t_freq = get_timestep_embedding(t, embedding_dim=256, dtype=target_dtype)
    h = F.silu(self.t_mlp_1(t_freq))
    h = rms_norm(self.t_mlp_2(h))
    return h.to(dtype=target_dtype)


class Qwen20BBlock(nn.Module):

  def __init__(
      self, dim=3584, num_heads=28, head_dim=128, rank=128, alpha=16.0
  ):
    super().__init__()
    if dim != num_heads * head_dim:
      head_dim = min(head_dim, dim)
      num_heads = max(1, dim // head_dim)
      head_dim = dim // num_heads
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.rank = rank
    self.lora_scale = alpha / rank

    # Image Attention LoRA projections
    self.to_q_lora_A = nn.Linear(dim, rank, bias=False)
    self.to_q_lora_B = nn.Linear(rank, dim, bias=False)
    self.to_k_lora_A = nn.Linear(dim, rank, bias=False)
    self.to_k_lora_B = nn.Linear(rank, dim, bias=False)
    self.to_v_lora_A = nn.Linear(dim, rank, bias=False)
    self.to_v_lora_B = nn.Linear(rank, dim, bias=False)
    self.to_out_lora_A = nn.Linear(dim, rank, bias=False)
    self.to_out_lora_B = nn.Linear(rank, dim, bias=False)

    # Text Cross-Attention LoRA projections
    self.add_q_proj_lora_A = nn.Linear(dim, rank, bias=False)
    self.add_q_proj_lora_B = nn.Linear(rank, dim, bias=False)
    self.add_k_proj_lora_A = nn.Linear(dim, rank, bias=False)
    self.add_k_proj_lora_B = nn.Linear(rank, dim, bias=False)
    self.add_v_proj_lora_A = nn.Linear(dim, rank, bias=False)
    self.add_v_proj_lora_B = nn.Linear(rank, dim, bias=False)
    self.to_add_out_lora_A = nn.Linear(dim, rank, bias=False)
    self.to_add_out_lora_B = nn.Linear(rank, dim, bias=False)

    # MLP LoRA projection (matches img_mlp.net.2 in QwenImageTransformerBlock)
    self.img_mlp_net_2_lora_A = nn.Linear(dim * 4, rank, bias=False)
    self.img_mlp_net_2_lora_B = nn.Linear(rank, dim, bias=False)

    # Initialize B matrices to zeros for stable identity start
    nn.init.zeros_(self.to_q_lora_B.weight)
    nn.init.zeros_(self.to_k_lora_B.weight)
    nn.init.zeros_(self.to_v_lora_B.weight)
    nn.init.zeros_(self.to_out_lora_B.weight)
    nn.init.zeros_(self.add_q_proj_lora_B.weight)
    nn.init.zeros_(self.add_k_proj_lora_B.weight)
    nn.init.zeros_(self.add_v_proj_lora_B.weight)
    nn.init.zeros_(self.to_add_out_lora_B.weight)
    nn.init.zeros_(self.img_mlp_net_2_lora_B.weight)

  def forward(self, x_img, x_txt, t_emb, cos_pos, sin_pos):
    x_img_mod = x_img + t_emb.unsqueeze(1).to(dtype=x_img.dtype)

    q_delta = self.to_q_lora_B(self.to_q_lora_A(x_img_mod)) * self.lora_scale
    k_delta = self.to_k_lora_B(self.to_k_lora_A(x_img_mod)) * self.lora_scale
    v_delta = self.to_v_lora_B(self.to_v_lora_A(x_img_mod)) * self.lora_scale

    add_q_delta = (
        self.add_q_proj_lora_B(self.add_q_proj_lora_A(x_txt)) * self.lora_scale
    )
    add_k_delta = (
        self.add_k_proj_lora_B(self.add_k_proj_lora_A(x_txt)) * self.lora_scale
    )
    add_v_delta = (
        self.add_v_proj_lora_B(self.add_v_proj_lora_A(x_txt)) * self.lora_scale
    )
    _ = (k_delta, v_delta, add_q_delta)

    q = rms_norm(
        (x_img_mod + q_delta).view(
            x_img.shape[0], x_img.shape[1], self.num_heads, self.head_dim
        )
    )
    k = rms_norm(
        (x_txt + add_k_delta).view(
            x_txt.shape[0], x_txt.shape[1], self.num_heads, self.head_dim
        )
    )
    v = rms_norm(
        (x_txt + add_v_delta).view(
            x_txt.shape[0], x_txt.shape[1], self.num_heads, self.head_dim
        )
    )

    q_rope = apply_3d_rope(q, cos_pos, sin_pos)

    # Matrix Multiplication
    q_trans = q_rope.permute(0, 2, 1, 3)  # [b, h, n, d]
    k_trans = k.permute(0, 2, 3, 1)  # [b, h, d, m]
    scores = torch.matmul(q_trans.float(), k_trans.float()) / math.sqrt(
        self.head_dim
    )  # [b, h, n, m]
    attn_weights = F.softmax(scores, dim=-1).to(dtype=x_img.dtype)

    v_trans = v.permute(0, 2, 1, 3)  # [b, h, m, d]
    attn_out_trans = torch.matmul(attn_weights, v_trans)  # [b, h, n, d]
    attn_out = attn_out_trans.permute(0, 2, 1, 3).reshape(
        x_img.shape
    )  # [b, n, h*d]

    attn_delta = (
        self.to_out_lora_B(self.to_out_lora_A(attn_out)) * self.lora_scale
    )
    add_out_delta = (
        self.to_add_out_lora_B(self.to_add_out_lora_A(attn_out))
        * self.lora_scale
    )

    mlp_hidden = x_img_mod.repeat(1, 1, 4)
    mlp_delta = (
        self.img_mlp_net_2_lora_B(self.img_mlp_net_2_lora_A(mlp_hidden))
        * self.lora_scale
    )

    return rms_norm(x_img + attn_out + attn_delta + add_out_delta + mlp_delta)


class Qwen20BPyTorchTPUTransformer(nn.Module):

  def __init__(self, dim=3584, num_blocks=16, latent_dim=64, rank=128):
    super().__init__()
    self.dim = dim
    self.t_embedder = TimestepEmbedder(hidden_dim=dim)
    self.proj_in = nn.Linear(latent_dim, dim, bias=False)
    self.blocks = nn.ModuleList(
        [Qwen20BBlock(dim=dim, rank=rank) for _ in range(num_blocks)]
    )
    self.proj_out = nn.Linear(dim, latent_dim, bias=False)

  def forward(self, x_latents, text_cond, timesteps, cos_pos, sin_pos):
    t_emb = self.t_embedder(timesteps, target_dtype=x_latents.dtype)
    h_img = self.proj_in(x_latents)
    for block in self.blocks:
      h_img = block(h_img, text_cond, t_emb, cos_pos, sin_pos)
    return self.proj_out(h_img)


class AnimePreencodeDataset(Dataset):

  def __init__(self, hf_dataset, style_descriptors):
    self.hf_dataset = hf_dataset
    self.style_descriptors = style_descriptors

  def __len__(self):
    return len(self.hf_dataset)

  def __getitem__(self, idx):
    item = self.hf_dataset[idx]
    img = item["image"].convert("RGB").resize((512, 512))
    descriptor = self.style_descriptors[idx % len(self.style_descriptors)]
    caption = f"an anime portrait illustration of a character, {descriptor}"

    img_np = np.array(img, dtype=np.float32)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) / 127.5 - 1.0
    return img_tensor.to(dtype=torch.bfloat16), caption


def generate_dataset_cache(
    cache_path,
    dataset_name="HK83/Anime_Faces",
    max_samples=1000,
    model_id="Qwen/Qwen-Image",
    batch_size=16,
    num_workers=8,
):
  """Generates dataset cache using multi-worker DataLoader and VAE acceleration."""
  print(
      f"\n[CACHE MISS] Cache file '{cache_path}' not found. Generating dataset"
      " cache..."
  )
  os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

  from datasets import load_dataset
  from diffusers import QwenImagePipeline
  from tqdm import tqdm

  print(f"  -> Loading '{model_id}' pipeline for dataset encoding...")
  pipe = QwenImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

  vae_device = torch.device("cpu")
  if hasattr(pipe, "vae") and pipe.vae is not None:
    pipe.vae = pipe.vae.to(vae_device)

  print(
      f"  -> Downloading dataset '{dataset_name}' from HuggingFace (limit:"
      f" {max_samples})..."
  )
  hf_dataset = load_dataset(dataset_name, split=f"train[:{max_samples}]")
  num_samples = len(hf_dataset)

  style_descriptors = [
      (
          "2D anime illustration, vibrant colors, expressive eyes, clean line"
          " art, high detail cell shading"
      ),
      (
          "masterpiece 2D anime portrait, colorful hair, expressive eyes,"
          " official art style, smooth shading"
      ),
      (
          "anime character illustration, detailed facial features, vibrant cell"
          " shading, clean vector lines"
      ),
      (
          "high quality anime artwork, colorful expression, 2D vector style,"
          " beautiful lighting and shading"
      ),
  ]

  print("  -> Pre-computing text prompt embeddings...")
  prompt_cache = {}
  with torch.no_grad():
    for desc in style_descriptors:
      full_caption = f"an anime portrait illustration of a character, {desc}"
      p_embeds, _ = pipe.encode_prompt(full_caption)
      prompt_cache[full_caption] = p_embeds.squeeze(0).cpu()

  enc_workers = num_workers if num_workers > 0 else 16
  dataset_wrapper = AnimePreencodeDataset(hf_dataset, style_descriptors)
  dataloader = DataLoader(
      dataset_wrapper,
      batch_size=batch_size,
      shuffle=False,
      num_workers=enc_workers,
      pin_memory=False,
  )

  encoded_samples = []
  print(f"  -> Pre-encoding dataset latents (batch_size={batch_size})...")

  with torch.no_grad():
    pbar = tqdm(
        total=num_samples,
        desc="Pre-encoding",
        unit="samples",
        dynamic_ncols=True,
    )
    for batch_imgs, batch_captions in dataloader:
      img_tensor = batch_imgs.unsqueeze(2).to(
          device=vae_device, dtype=torch.bfloat16
      )
      latents = pipe.vae.encode(img_tensor).latent_dist.sample()
      latents_cpu = latents.squeeze(2).cpu()

      for b in range(len(batch_captions)):
        cap_b = batch_captions[b]
        lat_b = latents_cpu[b]
        emb_b = prompt_cache[cap_b]

        encoded_samples.append(
            {"latent": lat_b, "prompt_embeds": emb_b, "caption": cap_b}
        )
      pbar.update(len(batch_captions))
    pbar.close()

  torch.save(encoded_samples, cache_path)
  print(
      f"  -> SUCCESS: Saved {len(encoded_samples)} pre-encoded samples to"
      f" '{cache_path}'!"
  )

  del pipe
  gc.collect()
  return encoded_samples


def main():
  parser = argparse.ArgumentParser(
      description="PyTorch TPU Qwen-Image LoRA Training Engine"
  )
  parser.add_argument(
      "--dataset_name",
      type=str,
      default="HK83/Anime_Faces",
      help="HuggingFace dataset name",
  )
  parser.add_argument(
      "--dataset_cache",
      type=str,
      default="./dataset_cache.pt",
      help="Pre-encoded dataset path",
  )
  parser.add_argument(
      "--max_samples",
      type=int,
      default=1000,
      help="Maximum dataset samples to pre-encode",
  )
  parser.add_argument(
      "--encoding_batch_size",
      type=int,
      default=16,
      help="Batch size for dataset pre-encoding",
  )
  parser.add_argument(
      "--encoding_num_workers",
      type=int,
      default=8,
      help="DataLoader worker threads for pre-encoding",
  )
  parser.add_argument(
      "--model_id",
      type=str,
      default=ModelConfig.model_id,
      help="Base model ID",
  )
  parser.add_argument(
      "--steps",
      type=int,
      default=None,
      help="Total training steps (overrides --epochs)",
  )
  parser.add_argument(
      "--epochs", type=float, default=1.0, help="Total training epochs"
  )
  parser.add_argument(
      "--lr", "--learning_rate", type=float, default=1e-4, help="Learning rate"
  )
  parser.add_argument(
      "--weight_decay", type=float, default=1e-2, help="Weight decay"
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="./output_qwen_image_tpu",
      help="Output directory",
  )
  parser.add_argument(
      "--dim",
      type=int,
      default=3584,
      help="Transformer hidden dimension",
  )
  parser.add_argument(
      "--num_blocks",
      type=int,
      default=16,
      help="Number of transformer blocks",
  )
  parser.add_argument(
      "--rank",
      type=int,
      default=128,
      help="LoRA adapter rank",
  )
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  print("\n=================================================================")
  print("   Qwen 20B Anime Faces PyTorch TPU High-Speed LoRA Trainer     ")
  print("=================================================================")
  print(f"Device:               {device}")
  print(f"Base Model ID:        {args.model_id}")
  print(f"Dataset Name:         {args.dataset_name}")
  print(f"Dataset Cache:        {args.dataset_cache}")
  print(f"Max Dataset Samples:  {args.max_samples}")
  print(f"Learning Rate:        {args.lr}")
  print(f"Output Directory:     {args.output_dir}")
  print("=================================================================\n")

  if not os.path.exists(args.dataset_cache):
    encoded_samples = generate_dataset_cache(
        cache_path=args.dataset_cache,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        model_id=args.model_id,
        batch_size=args.encoding_batch_size,
        num_workers=args.encoding_num_workers,
    )
  else:
    encoded_samples = torch.load(args.dataset_cache, map_location="cpu")

  num_samples = len(encoded_samples)
  limit = min(num_samples, args.max_samples)
  cpu_latents = []
  cpu_embeds = []
  for i in range(limit):
    item = encoded_samples[i]
    lat = item["latent"].to(dtype=torch.bfloat16)
    c, h, w = lat.shape
    lat_packed = (
        lat.view(c, h // 2, 2, w // 2, 2)
        .permute(1, 3, 0, 2, 4)
        .reshape(1024, 64)
    )

    emb = item["prompt_embeds"].to(dtype=torch.bfloat16)
    seq_len = emb.shape[0]
    if seq_len < 256:
      emb = torch.nn.functional.pad(emb, (0, 0, 0, 256 - seq_len))
    elif seq_len > 256:
      emb = emb[:256]

    cpu_latents.append(lat_packed.unsqueeze(0))
    cpu_embeds.append(emb.unsqueeze(0))
  num_samples = len(cpu_latents)
  if args.steps is None:
    args.steps = max(1, int(round(args.epochs * num_samples)))
  epochs_equivalent = args.steps / float(num_samples)
  print(
      f"Dataset Size: {num_samples} samples | Training for {args.steps} steps"
      f" (~{epochs_equivalent:.2f} epochs)"
  )

  cpu_noises = [
      torch.randn((1, 1024, 64), dtype=torch.bfloat16) for _ in range(100)
  ]
  cpu_timesteps = [
      torch.tensor([0.1 + (i % 8) * 0.1], dtype=torch.bfloat16)
      for i in range(100)
  ]

  print("\nInitializing Qwen 20B PyTorch Model on TPU...")
  model = Qwen20BPyTorchTPUTransformer(
      dim=args.dim, num_blocks=args.num_blocks, rank=args.rank
  )
  cos_3d, sin_3d = get_3d_rope(head_dim=model.blocks[0].head_dim, device=device)
  model = model.to(device=device, dtype=torch.bfloat16)
  model.train()

  optimizer = torch.optim.AdamW(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-4
  )

  print(f"\nStarting PyTorch TPU Optimization Loop ({args.steps} Steps)...")
  start_time = time.time()
  last_step_time = start_time

  for step in range(1, args.steps + 1):
    idx = (step - 1) % num_samples
    n_idx = (step - 1) % 100

    x0_batch = cpu_latents[idx].to(device=device)
    cond_batch = cpu_embeds[idx].to(device=device)
    noise_batch = cpu_noises[n_idx].to(device=device)
    t_val = cpu_timesteps[n_idx].to(device=device)

    optimizer.zero_grad()

    t_expand = t_val.view(1, 1, 1)
    x_t = (1.0 - t_expand) * x0_batch + t_expand * noise_batch

    pred_v = model(x_t, cond_batch, t_val, cos_3d, sin_3d)

    target_v = noise_batch - x0_batch
    loss = torch.nn.functional.mse_loss(pred_v, target_v)

    loss.backward()

    if USE_TPU:
      torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
      optimizer.step()
      if hasattr(torch, "tpu") and torch.tpu.is_available():
        torch.tpu.synchronize()
    else:
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

    now = time.time()
    inst_delta = now - last_step_time
    last_step_time = now

    if step % 10 == 0 or step <= 5 or step == args.steps:
      elapsed = now - start_time
      inst_ms = inst_delta * 1000.0
      inst_sps = (1.0 / inst_delta) if inst_delta > 0 else 0
      loss_val = loss.detach().item()

      print(
          f"  Step [{step:5d}/{args.steps}] | Speed: {inst_sps:.1f} steps/s"
          f" ({inst_ms:.1f} ms/step) | Loss: {loss_val:.6f} | Elapsed:"
          f" {elapsed:.2f}s",
          flush=True,
      )

  print("\nExporting PyTorch TPU LoRA Adapter Weights...")
  peft_dict = {}
  for name, param in model.named_parameters():
    if "lora" in name:
      clean_key = f"transformer.base_model.model.transformer_blocks.{name}"
      peft_dict[clean_key] = param.detach().cpu()
  import json

  adapter_config = {
      "peft_type": "LORA",
      "task_type": "DIFFUSION",
      "r": args.rank,
      "lora_alpha": 16.0,
      "target_modules": [
          "to_q",
          "to_k",
          "to_v",
          "to_out.0",
          "add_q_proj",
          "add_k_proj",
          "add_v_proj",
          "to_add_out",
          "img_mlp.net.2",
      ],
  }
  config_path = os.path.join(args.output_dir, "adapter_config.json")
  with open(config_path, "w") as f:
    json.dump(adapter_config, f, indent=2)

  adapter_path = os.path.join(args.output_dir, "adapter_model.safetensors")
  save_file(peft_dict, adapter_path)
  print(f"Saved raw PEFT adapter and config to: {args.output_dir}")

  convert_peft_to_diffusers_adapter(args.output_dir)

  print("\n=================================================================")
  print("   SUCCESS: PyTorch TPU Training Completed!                     ")
  print("=================================================================\n")


if __name__ == "__main__":
  main()
