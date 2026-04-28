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

"""Helper routines to assist in micro-benchmarking."""

import time

from absl import flags
import torch


_ACC = flags.DEFINE_enum(
    "acc",
    "tpu",
    ["tpu", "cuda"],
    required=False,
    help="Accelerator device under test (must be 'tpu' or 'cuda').",
)


def _get_device() -> torch.device:
  """Initialize and return the accelerator runtime."""

  if _ACC.value == "tpu":

    return torch.device("tpu")
  if _ACC.value == "cuda":
    return torch.device("cuda:0")
  raise RuntimeError(f"Unexpected device: {_ACC.value}")


def _run_and_time_model(
    model: torch.nn.Module = None,
    input_tensor: torch.Tensor | None = None,
    warmup: int = 1,
    iterations: int = 1,
) -> tuple[float, torch.Tensor]:
  """Run and time the model.

  Args:
    model: The model to run.
    input_tensor: The model's input tensors.
    warmup: Number of warmup runs.
    iterations: How often to execute the model.

  Returns:
    Time spent in seconds as a float.
  """
  for _ in range(warmup):
    with torch.no_grad():
      _ = model(input_tensor)

  total_time = 0
  for _ in range(iterations):
    with torch.no_grad():
      start_time = time.time()
      output = model(input_tensor)
      total_time += time.time() - start_time

  return total_time / iterations, output.to("cpu")


def run_model_cpu_acc(
    cfg,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 1,
    iterations: int = 10,
    desc: str | None = None,
) -> None:
  """Run the model on CPU and accelerator device."""

  if desc:
    print(f"--- {desc} ---")
  acc_device = _get_device()
  print(f"Accelerator: {acc_device}")
  print(f"Iterations:  {iterations}")
  print(f"Warmup:      {warmup}")
  print(f'Emb dim:     {cfg["emb_dim"]}')
  print(f'Hidden dim:  {cfg["hidden_dim"]}')

  model.eval()
  t, out_cpu = _run_and_time_model(
      model=model,
      input_tensor=input_tensor,
      iterations=iterations,
      warmup=warmup,
  )
  print(f"CPU time:    {t*iterations:.3}ms")

  model.to(acc_device)
  model.eval()
  input_acc = input_tensor.to(acc_device)
  t, out_acc = _run_and_time_model(
      model=model, input_tensor=input_acc, iterations=iterations
  )
  print(f"ACC time:    {t*iterations:.3}ms")

  print("CPU:", out_cpu)
  print("ACC:", out_acc)
