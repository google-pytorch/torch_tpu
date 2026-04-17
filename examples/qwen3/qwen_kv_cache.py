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

# This is a development script for testing Qwen3 KV cache on TPU.
# It is not intended to be a production-quality example.

import time
from typing import Sequence

from absl import app
from absl import flags
import torch
from transformers.cache_utils import StaticCache
from transformers.models.qwen3 import modeling_qwen3

USE_CACHE = flags.DEFINE_boolean(
    "use_cache", True, "Whether to use the KV cache."
)


def model_generate(
    model, initial_inputs: torch.Tensor, use_cache: bool, decode_steps: int = 10
):
  """Generate from a model."""
  native_device = model.device
  model_kwargs = {}
  past_key_values = None
  cache_position_index = None

  if use_cache:
    print("Using static cache", flush=True)
    cache_position = torch.arange(0, initial_inputs.size(1)).to(native_device)
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=128,
        device=native_device,
        dtype=model.dtype,
    )
    cache_position_index = initial_inputs.size()[1]
    model_kwargs = {
        "past_key_values": past_key_values,
        "cache_position": cache_position,
    }
  else:
    print("Not using static cache", flush=True)

  with torch.no_grad():
    # Prefill
    start_time = time.time()
    output = model(input_ids=initial_inputs, **model_kwargs)
    end_time = time.time()
    logits = output.logits
    if use_cache:
      past_key_values = output.past_key_values

    prefill_time = end_time - start_time
    print(f"Prefill time: {prefill_time * 1000:.2f} ms")

    # Decode
    input_ids = initial_inputs
    start_time = time.time()
    output_tokens = []
    for i in range(decode_steps):
      print(
          f"Decode step {i}, logits size: {logits.size()}, logits device:"
          f" {logits.device}"
      )
      # greedy sampling
      logits = logits.to("cpu")
      next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
      output_tokens.append(next_token)
      next_token_device = next_token.to(native_device)

      if use_cache:
        decode_input_ids = next_token_device
        cache_position = torch.arange(
            cache_position_index,
            cache_position_index + next_token.size()[1],
        ).to(native_device)
        cache_position_index += next_token.size()[1]
        model_kwargs = {
            "past_key_values": past_key_values,
            "cache_position": cache_position,
        }
      else:
        input_ids = torch.cat([input_ids, next_token_device], dim=1)
        decode_input_ids = input_ids
        model_kwargs = {}

      output = model(input_ids=decode_input_ids, **model_kwargs)
      if use_cache:
        past_key_values = output.past_key_values
      logits = output.logits

    end_time = time.time()
    decode_time = end_time - start_time
    print(f"Decode time per token: {decode_time * 1000 / decode_steps:.2f} ms")

  output_ids = torch.cat(output_tokens, dim=1)
  output_list = output_ids.tolist()
  return output_list


def main(argv: Sequence[str]) -> None:
  tpu_device = torch.device("tpu")
  torch.manual_seed(42)
  config = modeling_qwen3.Qwen3Config(
      attention_bias=False,
      attention_dropout=0.0,
      bos_token_id=151643,
      eos_token_id=151645,
      head_dim=128,
      hidden_act="silu",
      hidden_size=1024,
      initializer_range=0.02,
      intermediate_size=3072,
      max_position_embeddings=40960,
      max_window_layers=28,
      num_attention_heads=16,
      num_hidden_layers=28,
      num_key_value_heads=8,
      rms_norm_eps=1e-06,
      rope_scaling=None,
      rope_theta=1000000,
      sliding_window=None,
      tie_word_embeddings=True,
      torch_dtype="bfloat16",
      # TODO(bbahl): Changing this to false fails during causal mask creation.
      use_cache=True,
      use_sliding_window=False,
      vocab_size=151936,
  )
  model_cpu = modeling_qwen3.Qwen3ForCausalLM(config)
  model_tpu = modeling_qwen3.Qwen3ForCausalLM(config)
  model_tpu.load_state_dict(model_cpu.state_dict())
  model_tpu.to(tpu_device)
  inputs_cpu = torch.randint(0, 100, (1, 10))
  output_text_cpu = model_generate(
      model_cpu, inputs_cpu, use_cache=USE_CACHE.value, decode_steps=4
  )
  print("-" * 10, "Model Output", "-" * 10)
  print(output_text_cpu, flush=True)

  inputs_tpu = inputs_cpu.to(tpu_device)
  output_text_tpu = model_generate(
      model_tpu, inputs_tpu, use_cache=USE_CACHE.value, decode_steps=4
  )
  print("-" * 10, "Model Output", "-" * 10)
  print(output_text_tpu, flush=True)
  assert (
      output_text_tpu == output_text_cpu
  ), f"{output_text_tpu=}\n{output_text_cpu=}"


if __name__ == "__main__":
  app.run(main)
