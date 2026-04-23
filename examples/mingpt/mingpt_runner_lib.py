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

"""Library for running mingpt examples."""

import time

import torch
from examples.mingpt.impl import mingpt

GPT = mingpt.GPT


def run_model(model_tpu, x_input_tpu):
  with torch.no_grad():
    start_time_tpu = time.time()
    logits_from_tpu, _ = model_tpu(x_input_tpu)
    end_time_tpu = time.time()
    tpu_time = end_time_tpu - start_time_tpu

  return logits_from_tpu, tpu_time


def run_mingpt(
    config,
    common_dtype,
    tpu_device,
    num_preheat_runs: int,
    num_real_runs: int,
) -> tuple[list[float], list[float]]:
  """Runs the mingpt example."""
  model_tpu = GPT(config, device="cpu").to(common_dtype)
  model_tpu.to(tpu_device)
  model_tpu.eval()

  input_cpu = torch.randint(
      0, config.vocab_size, (2, config.block_size), dtype=torch.long
  )
  input_tpu = input_cpu.to(tpu_device)

  device_str = str(tpu_device)

  # preheat
  preheat_times = []
  for _ in range(num_preheat_runs):
    logits_from_tpu, tpu_time = run_model(model_tpu, input_tpu)
    assert logits_from_tpu.cpu() is not None
    preheat_times.append(tpu_time)

  # real runs
  real_times = []
  for _ in range(num_real_runs):
    logits_from_tpu, tpu_time = run_model(model_tpu, input_tpu)
    assert logits_from_tpu.cpu() is not None
    real_times.append(tpu_time)

  return preheat_times, real_times


def run_mingpt_with_default_params():
  """Runs the mingpt example with a set of default parameters.

  This function initializes a GPT model with a specific configuration,
  runs it on a TPU device with preheating and real runs, and returns
  the average times for both phases.

  Returns:
      A tuple containing:
      - avg_preheat_time: The average time taken for the preheat runs in
        seconds.
      - avg_real_time: The average time taken for the real runs in seconds.
  """
  config = GPT.get_default_config()
  config.model_type = "tpu-optimal-l-hd128"
  config.vocab_size = 100
  config.block_size = 1024
  common_dtype = torch.float32
  torch.manual_seed(42)
  device = torch.accelerator.current_accelerator()
  preheat_times, real_times = run_mingpt(
      config,
      common_dtype,
      device,
      num_preheat_runs=2,
      num_real_runs=32,
  )
  print(f"preheat_times={preheat_times}")
  print(f"real_times={real_times}")
  avg_preheat_time = sum(preheat_times) / len(preheat_times)
  avg_real_time = sum(real_times) / len(real_times)
  return avg_preheat_time, avg_real_time
