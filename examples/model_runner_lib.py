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

"""Model runner library for benchmarking PyTorch models."""

import time

from absl import logging
import torch


def _run_decode_step(model, logits, input_ids, use_cache, device):
  """Runs a single decode step.

  Args:
    model: The language model.
    logits: The logits from the previous step.
    input_ids: The input token IDs.
    use_cache: Whether the model uses a KV cache.
    device: The device the model is running on.

  Returns:
    The logits from the current decode step.
  """
  # greedy sampling
  if device.type == 'tpu':
    # TODO: support for numel-changing as_strided is not implemented
    # Otherwise we could avoid this transfer.
    logits = logits.to('cpu')

  next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1).to(device)

  if use_cache:
    logits = model(next_token)
  else:
    input_ids = torch.cat([input_ids, next_token], dim=1)
    logits = model(input_ids)

  return logits, input_ids


def model_runner(
    model,
    device,
    prompt_len=256,
    decode_steps=10,
    prefill_warmup_steps=1,
    decode_warmup_steps=1,
    batch_size=1,
):
  """Runs a model benchmark with prefill and decode phases.

  Args:
    model: The model to benchmark.
    device: The device to run the model on (e.g., 'tpu', 'cpu').
    prompt_len: The length of the prefill prompt.
    decode_steps: The number of decode steps to run.
    prefill_warmup_steps: The number of prefill warmup steps to run.
    decode_warmup_steps: The number of decode warmup steps to run.
    batch_size: The batch size for the model.

  Returns:
    A dictionary containing benchmark results, including prefill time and decode
    step times.
  """
  model.to(device)
  model.eval()
  dtype = model.cfg['dtype']

  logging.info('Using device: %s, dtype: %s', device, dtype)
  logging.info(
      'Config: prompt_len=%d, decode_steps=%d, batch_size=%d',
      prompt_len,
      decode_steps,
      batch_size,
  )

  with torch.no_grad():
    # --- Prefill Phase ---
    logging.info('Running Prefill with prompt length %d...', prompt_len)
    input_ids = torch.randint(
        0, model.cfg['vocab_size'], (batch_size, prompt_len), device=device
    ).to(device)

    # Warmup
    for _ in range(prefill_warmup_steps):
      _ = model(input_ids)
    model.reset_cache()

    # Run the prefill to be measured
    start_time = time.time()
    logits = model(input_ids)
    end_time = time.time()

    prefill_time = end_time - start_time
    logging.info('Prefill time: %.2f ms', prefill_time * 1000)

    # --- Decode Phase ---
    logging.info('Running Decode for %d steps...', decode_steps)
    use_cache = 'use_cache' in model.cfg and model.cfg['use_cache']

    # Warmup
    for _ in range(decode_warmup_steps):
      logits, input_ids = _run_decode_step(
          model, logits, input_ids, use_cache, device
      )

    start_time = time.time()

    for i in range(decode_steps):
      logging.info('Decode step %d', i)
      logits, input_ids = _run_decode_step(
          model, logits, input_ids, use_cache, device
      )

    end_time = time.time()
    total_decode_time = end_time - start_time
    avg_decode_time = total_decode_time / decode_steps
    logging.info(
        'Avg decode step time: %.2f ms (steps 1 to %d)',
        avg_decode_time * 1000,
        decode_steps,
    )
