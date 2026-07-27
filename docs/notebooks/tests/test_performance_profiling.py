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
# pylint: skip-file  # ruff: noqa

"""Tests for performance_profiling.py — verifies each executable cell runs without error."""

import os
import torch


def test_profiler_init(device):
  """Cell: import profiler and sync, initialize device."""
  from torch_tpu._internal import profiler
  from torch_tpu._internal import sync

  assert profiler is not None
  assert sync is not None


def test_capture_profile(device):
  """Cell: run training loop inside profiler context (reduced to 10 steps)."""
  from torch_tpu._internal import profiler
  from torch_tpu._internal import sync

  model = (
      torch.nn.Sequential(
          torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
      )
      .to(device)
      .to(torch.bfloat16)
  )

  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  data = torch.randn(32, 128, device=device, dtype=torch.bfloat16)

  log_dir = "/tmp/test_profiler_output"

  with profiler.profile(
      activities=[
          profiler.ProfilerActivity.CPU,
          profiler.ProfilerActivity.TPU,  # type: ignore
      ],
      on_trace_ready=profiler.xprof_trace_handler(dir_name=log_dir),
  ):
    for step in range(10):  # Reduced from 100 for test speed
      optimizer.zero_grad()
      loss = model(data).sum()
      loss.backward()
      optimizer.step()
      sync.synchronize(loss)

  assert os.path.exists(log_dir)


def test_verify_traces():
  """Cell: check that trace files exist in the output directory."""
  log_dir = "/tmp/test_profiler_output"

  if not os.path.exists(log_dir):
    # If previous test didn't create the dir, skip
    import pytest

    pytest.skip("Profiler output directory not found")

  trace_files = os.listdir(log_dir)
  assert len(trace_files) > 0, "No trace files found in profiler output"
