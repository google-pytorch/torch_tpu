# Copyright 2026 The TorchTPU Authors
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

"""Functional Autoregressive Generation Engine for DeepSeek-V4-Flash.

Provides:
1. Out-of-place token sampling without in-place tensor mutations.
2. Functional token generation loop without tensor slice assignments.
3. TTFT (Time-to-First-Token), TPOT (Time-Per-Output-Token), and throughput measurement.
"""

import json
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

try:
  from .model import ModelArgs, Transformer
except ImportError:
  from model import ModelArgs, Transformer


def get_inference_device() -> torch.device:
  """Selects Cloud TPU if available, falling back to CPU for testing."""
  try:
    import torch_tpu  # pylint: disable=unused-import
    return torch.device("tpu")
  except ImportError:
    pass

  try:
    import torch_xla.core.xla_model as xm
    return xm.xla_device()
  except ImportError:
    pass

  if torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def sample_functional(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
  """Out-of-place Gumbel-max sampling eliminating in-place mutations."""
  if temperature <= 0.0:
    return logits.argmax(dim=-1)
  scaled_logits = logits / max(temperature, 1e-5)
  probs = F.softmax(scaled_logits, dim=-1, dtype=torch.float32)
  noise = torch.empty_like(probs).exponential_(1)
  return (probs / torch.clamp(noise, min=1e-8)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int = 32,
    eos_id: int = 1,
    temperature: float = 0.7,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
  """Generates tokens using functional PyTorch operations on Cloud TPU."""
  if device is None:
    device = get_inference_device()

  bsz = len(prompt_tokens)
  prompt_lens = [len(p) for p in prompt_tokens]
  min_prompt_len = min(prompt_lens)
  max_prompt_len = max(prompt_lens)

  # Convert prompt list to padded tensor on target device
  prompt_tensor = torch.full(
      (bsz, max_prompt_len), -1, dtype=torch.long, device=device
  )
  for i, p in enumerate(prompt_tokens):
    prompt_tensor[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)

  # 1. Prefill Phase
  t_start = time.perf_counter()
  logits = model(prompt_tensor[:, :min_prompt_len], start_pos=0)
  if hasattr(torch, "xla") and hasattr(torch.xla, "sync"):
    torch.xla.sync()
  t_prefill = time.perf_counter()
  ttft = t_prefill - t_start

  # Sample first token
  next_token = sample_functional(logits, temperature)
  generated_tokens = [next_token.unsqueeze(-1)]
  decode_step_times = []

  # 2. Decode Phase (Step-by-step autoregression)
  cur_pos = min_prompt_len
  total_steps = max_new_tokens

  for _ in range(1, total_steps):
    t_step_start = time.perf_counter()
    logits = model(next_token.unsqueeze(1), start_pos=cur_pos)
    if hasattr(torch, "xla") and hasattr(torch.xla, "sync"):
      torch.xla.sync()
    t_step_end = time.perf_counter()
    decode_step_times.append(t_step_end - t_step_start)

    next_token = sample_functional(logits, temperature)
    generated_tokens.append(next_token.unsqueeze(-1))
    cur_pos += 1

  t_end = time.perf_counter()
  total_latency = t_end - t_start
  tpot = sum(decode_step_times) / max(1, len(decode_step_times))
  total_tokens_generated = bsz * total_steps
  throughput = total_tokens_generated / max(1e-6, total_latency)

  all_generated = torch.cat(generated_tokens, dim=-1).cpu().tolist()

  return {
      "device": str(device),
      "batch_size": bsz,
      "input_seq_len": min_prompt_len,
      "generated_tokens_count": total_tokens_generated,
      "ttft_sec": round(ttft, 5),
      "tpot_sec": round(tpot, 5),
      "total_latency_sec": round(total_latency, 5),
      "throughput_tokens_per_sec": round(throughput, 2),
      "output_tokens": all_generated,
  }


if __name__ == "__main__":
  print("Initializing DeepSeek-V4-Flash functional generation harness...")
  args = ModelArgs(
      n_layers=2,
      dim=256,
      moe_inter_dim=128,
      head_dim=64,
      q_lora_rank=64,
      o_lora_rank=64,
      n_heads=4,
      n_routed_experts=4,
      n_activated_experts=2,
      vocab_size=1000,
      max_batch_size=2,
      max_seq_len=256,
      window_size=32,
      compress_ratios=(0, 0),
      hc_mult=2,
  )
  target_device = get_inference_device()
  print(f"Target device: {target_device}")
  test_model = Transformer(args).to(target_device)
  for param in test_model.parameters():
    if param.is_floating_point():
      if param.dim() > 1:
        torch.nn.init.xavier_uniform_(param)
      else:
        torch.nn.init.zeros_(param)
    else:
      param.data.zero_()

  sample_prompts = [[10, 20, 30, 40], [50, 60, 70, 80]]
  results = generate(test_model, sample_prompts, max_new_tokens=8, device=target_device)
  print("Generation test completed successfully:")
  print(json.dumps(results, indent=2))
