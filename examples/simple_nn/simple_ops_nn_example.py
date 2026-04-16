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

from typing import Final
from absl import app
import torch
import torch.nn as nn
from torch_tpu._internal.utils import utils

TPU_DEVICE_TYPE: Final[str] = "tpu"


class SimpleNN(nn.Module):

  def __init__(self, in_features, out_features, bias=True):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features, bias=bias)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.1)

  def forward(self, x):
    x = self.linear(x)
    x = self.relu(x)
    x = self.dropout(x)
    return x


def main(argv):
  tpu_d: torch.device | None = torch.device("tpu")
  if tpu_d is None:
    return
  print(
      f"Device type: {tpu_d.type}, Device index:"
      f" {tpu_d.index if tpu_d.index is not None else 'default'}"
  )

  print("\n--- Test: Tensor creation (torch.ones) on TPU device ---")
  tpu_ones = torch.ones(2, 3, device=tpu_d, dtype=torch.float32)
  assert isinstance(tpu_ones, torch.Tensor)
  assert tpu_ones.device.type == TPU_DEVICE_TYPE

  cpu_ones_val = tpu_ones.cpu()
  assert isinstance(cpu_ones_val, torch.Tensor)
  assert cpu_ones_val.device.type == "cpu"
  utils.assert_close(cpu_ones_val, torch.ones(2, 3, dtype=torch.float32))
  print("Tensor creation (torch.ones) PASSED.")
  del tpu_ones, cpu_ones_val

  print("\n--- Test: Moving CPU tensor to TPU device ---")
  cpu_tensor = torch.randn(2, 2, dtype=torch.float32)
  tpu_from_cpu = cpu_tensor.to(tpu_d)
  assert isinstance(tpu_from_cpu, torch.Tensor)
  assert tpu_from_cpu.device.type == TPU_DEVICE_TYPE
  utils.assert_close(tpu_from_cpu.cpu(), cpu_tensor)
  print("CPU -> TPU .to() PASSED.")
  del cpu_tensor, tpu_from_cpu

  print("\n--- Test: Binary op (add) on TPU tensors ---")
  a_tpu = torch.ones((2, 2), device=tpu_d, dtype=torch.float32)
  b_tpu = torch.ones((2, 2), device=tpu_d, dtype=torch.float32)
  a_cpu = torch.ones((2, 2), dtype=torch.float32)
  b_cpu = torch.ones((2, 2), dtype=torch.float32)

  print(f"DEBUG Python: BEFORE add, a_tpu.cpu() = {a_cpu}")
  print(f"DEBUG Python: BEFORE add, b_tpu.cpu() = {b_cpu}")
  c_tpu = a_tpu + b_tpu
  assert isinstance(c_tpu, torch.Tensor)
  assert c_tpu.device.type == TPU_DEVICE_TYPE
  c_cpu = c_tpu.cpu()
  print(f"c_cpu: {c_cpu}")
  utils.assert_close(c_cpu, a_cpu + b_cpu)
  print("Binary op (add) PASSED.")
  del a_tpu, b_tpu, c_tpu

  print("\n --- Test: SimpleNN ---")
  torch.manual_seed(42)
  sm_ref_cpu = SimpleNN(2, 2, bias=True)
  sm_ref_cpu.eval()

  sm_tpu = SimpleNN(2, 2, bias=True)
  sm_tpu.load_state_dict(sm_ref_cpu.state_dict())
  sm_tpu.to(tpu_d)
  sm_tpu.eval()

  x_tpu = torch.ones((2, 2), device=tpu_d, dtype=torch.float32)
  y_from_tpu = sm_tpu(x_tpu)
  assert y_from_tpu.device.type == TPU_DEVICE_TYPE
  y_tpu_to_cpu = y_from_tpu.cpu()

  x_ref_cpu = torch.ones((2, 2), dtype=torch.float32)
  y_ref_cpu = sm_ref_cpu(x_ref_cpu)

  print(f"y_tpu_to_cpu: {y_tpu_to_cpu}")
  print(f"y_ref_cpu: {y_ref_cpu}")

  utils.assert_close(y_tpu_to_cpu, y_ref_cpu, rtol=5e-3, atol=5e-3)

  print("\n --- Test: SimpleNN Backward ---")
  sm_tpu.train()
  x_tpu_grad = torch.ones(
      (2, 2), device=tpu_d, dtype=torch.float32, requires_grad=True
  )
  y_train = sm_tpu(x_tpu_grad)
  loss = y_train.sum()
  print("Running backward pass...")
  loss.backward()
  print("Backward pass PASSED.")


if __name__ == "__main__":
  app.run(main)
