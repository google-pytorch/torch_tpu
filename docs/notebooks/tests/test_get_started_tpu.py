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

"""Tests for get_started_tpu.py — verifies each executable cell runs without error."""

# pylint: skip-file  # ruff: noqa
import torch


def test_cell_init(device):
  """Cell: import torch, safe_init() — verifies device is returned."""
  assert device is not None
  assert "tpu" in str(device)


def test_cell_create_tensors(device):
  """Cell: create ones and randn tensors on device."""
  a = torch.ones((1024, 1024), device=device, dtype=torch.bfloat16)
  b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
  assert a.shape == (1024, 1024)
  assert b.shape == (1024, 1024)
  assert a.device.type == "tpu"


def test_cell_matmul(device):
  """Cell: perform matrix multiplication (deferred)."""
  a = torch.ones((1024, 1024), device=device, dtype=torch.bfloat16)
  b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
  c = torch.matmul(a, b)
  assert c.shape == (1024, 1024)


def test_cell_materialize(device):
  """Cell: call .cpu() to trigger XLA compilation and execution."""
  a = torch.ones((1024, 1024), device=device, dtype=torch.bfloat16)
  b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
  c = torch.matmul(a, b)
  final_result = c.cpu()
  assert final_result.device.type == "cpu"
  assert final_result.shape == (1024, 1024)
