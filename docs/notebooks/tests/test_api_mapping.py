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

"""Tests for api_mapping.py — verifies each executable cell runs without error."""

import torch


def test_device_queries(device):
  """Cell: device_queries — verifies torch.tpu API methods work."""
  assert torch.tpu.is_available() is True
  assert torch.tpu.device_count() >= 1
  current = torch.tpu.current_device()
  assert current is not None


def test_seeding(device):
  """Cell: seeding_example — seeds all TPU cores."""
  torch.tpu.manual_seed_all(42)


def test_amp_supported_dtypes(device):
  """Cell: amp_example — gets AMP supported dtypes."""
  supported_dtypes = torch.tpu.get_amp_supported_dtype()
  assert isinstance(supported_dtypes, list)
  assert len(supported_dtypes) > 0
