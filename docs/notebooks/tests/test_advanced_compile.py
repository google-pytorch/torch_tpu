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
# pylint: skip-file

"""Tests for advanced_compile.py — verifies each tutorial concept with strong assertions."""

import torch
import torch.nn as nn
import torch.optim as optim


def test_device_init(device):
  """Verifies that the TPU device is correctly initialized and registered."""
  import torch_tpu

  # Assert device type is correct
  assert device.type == "tpu"

  # Assert api call works
  tpu_device = torch.device("tpu")
  assert tpu_device.type == "tpu"


def test_efficient_loading(device):
  """Verifies direct HBM allocation using torch.device context manager."""

  class MyLargeModel(nn.Module):

    def __init__(self):
      super().__init__()
      self.net = nn.Sequential(
          nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 10)
      )

    def forward(self, x):
      return self.net(x)

  with torch.device(device):
    model = MyLargeModel()
    input_data = torch.randn(20, 10)

  # Assert model parameters are on TPU HBM
  assert model.net[0].weight.device.type == "tpu"
  assert input_data.device.type == "tpu"

  # Verify forward pass on compiled model
  compiled_model = torch.compile(model, backend="tpu", dynamic=False)
  output = compiled_model(input_data)

  assert output.shape == (20, 10)
  assert output.device.type == "tpu"


def test_training_step(device):
  """Verifies a full compiled training step with loss reduction and weight updates."""

  class SimpleModel(nn.Module):

    def __init__(self):
      super().__init__()
      self.net = nn.Linear(10, 10)

    def forward(self, x):
      return self.net(x)

  with torch.device(device):
    model = SimpleModel()
    # NOTE: capturable=True is mandatory for stateful optimizers on TPU
    optimizer = optim.Adam(model.parameters(), lr=1e-2, capturable=True)
    criterion = nn.MSELoss()

  @torch.compile(backend="tpu", dynamic=False)
  def train_step(data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.detach()

  data = torch.randn(16, 10, device=device)
  target = torch.randn(16, 10, device=device)

  # Initial weights
  initial_weights = model.net.weight.clone().detach()

  # Run several steps to see loss reduction
  initial_loss = train_step(data, target).item()
  for _ in range(5):
    current_loss = train_step(data, target).item()

  # Strong Assertions
  assert (
      current_loss < initial_loss
  ), f"Loss didn't decrease: {current_loss} >= {initial_loss}"

  final_weights = model.net.weight.detach()
  assert not torch.allclose(
      initial_weights, final_weights
  ), "Model weights did not update"


def test_explain_graph(device):
  """Verifies that dynamo.explain correctly identifies graph breaks."""
  from torch._dynamo import explain

  class DiagnosticModel(nn.Module):

    def forward(self, x):
      x = x + 1
      # This 'print' causes a graph break
      print(f"Internal Mean: {x.mean().item()}")
      return x * 2

  model = DiagnosticModel().to(device)
  input_data = torch.randn(10, 10, device=device)

  explanation = explain(model, input_data)

  # Assert that graph breaks were detected as intended
  assert (
      explanation.graph_count > 1
  ), f"Expected graph break not detected. Graphs: {explanation.graph_count}"
