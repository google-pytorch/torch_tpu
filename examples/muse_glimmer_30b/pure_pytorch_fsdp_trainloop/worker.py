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

"""Per TPU worker entrypoint for running Muse Glimmer 30B trainloop."""

import os
import time

import torch

from . import model_fsdp

QUICK_REDUCED_MODEL = False  # Run a smaller variant for quick testing.


def run_worker_trainloop(device="tpu"):
  """Full training loop."""
  rank = int(os.getenv("RANK", default="0"))
  compile_backend = "tpu" if device == "tpu" else "inductor"

  # NOTE: rank-dependent seed will cause replicated weights to be initialized
  # differently across ranks. This is just a trainloop demonstration, and for
  # real training, we should broadcast the weights or load from checkpoint.
  torch.manual_seed(42 + rank)
  torch.set_default_dtype(torch.bfloat16)

  config = model_fsdp.Config()
  if QUICK_REDUCED_MODEL:
    config.vocab_size = 9216
    config.num_layers = 4

  with torch.device(device):
    m = model_fsdp.MuseGlimmer30B(config)
    m.apply_torch_compile(compile_backend)

    lm_head_loss = model_fsdp.MuseGlimmerChunkwiseFusedLMHeadLoss(config)
    lm_head_loss.apply_torch_compile(compile_backend)

  model_fsdp.init_nn_module(m)
  model_fsdp.init_nn_module(lm_head_loss)
  torch.accelerator.synchronize()

  if rank == 0:
    print_module(m)
    print_module(lm_head_loss)

  profile_trainloop(m, lm_head_loss, config, device, compile_backend)


def profile_trainloop(
    m: torch.nn.Module,
    head: torch.nn.Module,
    c: model_fsdp.Config,
    device: str,
    compile_backend: str,
):
  rank = int(os.getenv("RANK", default="0"))
  world_size = int(os.getenv("WORLD_SIZE", default="1"))

  trainable_params = [p for p in m.parameters() if p.requires_grad]
  trainable_params.extend([p for p in head.parameters() if p.requires_grad])
  optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

  non_fsdp_params = [
      *model_fsdp.get_non_fsdp_params(m),
      *model_fsdp.get_non_fsdp_params(head),
  ]

  # Put many small collective all-reduce ops under compile for optimizations.
  non_fsdp_all_reducer = torch.compile(
      model_fsdp.all_reduce_grads, dynamic=False, backend=compile_backend
  )

  # Fake dataloader:
  tokens = torch.randint(
      0,
      c.vocab_size,
      (c.batch_size, c.seq_len + 1),  # Note: local batch size here.
      device=device,
  )
  inputs = tokens[..., 0 : c.seq_len].contiguous()
  targets = tokens[..., 1:].contiguous()
  torch.accelerator.synchronize()  # Exclude from measurements.

  # Function to be profiled
  def run():
    # Forward pass:
    h = m(inputs)

    # Chunkwise fused LM head + cross-entropy loss:
    h_grad = head(h.detach(), targets)

    # Backward pass (for model backbone)
    h.backward(gradient=h_grad)

    if world_size > 1:
      # NOTE: FSDP params were already reduced during backward pass.
      non_fsdp_all_reducer(non_fsdp_params)

    optimizer.step()
    optimizer.zero_grad()

  # Run a few times and report step times from rank 0:
  for i in range(5):
    start_time = time.perf_counter()
    run()
    torch.accelerator.synchronize()
    duration = time.perf_counter() - start_time

    if rank == 0:
      tps = (c.batch_size * c.seq_len) / duration  # Local TPS
      print(
          f"Run {i}: {duration:.3f} seconds, {tps:.2f} TPS (per rank)",
          flush=True,
      )


def print_module(m: torch.nn.Module):
  """Print all buffer and weight shapes and dtypes."""

  print("Model buffers:", flush=True)
  for n, p in m.named_buffers(recurse=True):
    print(f" {n}: {p.shape}, {p.dtype}", flush=True)

  print("Model parameters:", flush=True)
  for n, p in m.named_parameters(recurse=True):
    print(
        f" {n}: {p.shape}, {p.dtype}, trainable={p.requires_grad}", flush=True
    )
