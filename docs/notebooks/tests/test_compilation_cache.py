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
"""Tests for compilation_cache.py — verifies each executable cell runs without error."""

import torch


def test_cache_stats_demo(device):
  """Cell: cache_stats_demo — runs operations to populate and hit the cache."""
  # Run some operations to populate the cache
  x = torch.randn(32, 128, device=device)
  y = x @ x.T
  _ = y.cpu()  # Trigger materialization

  # Run the same graph again (should be a cache hit)
  x2 = torch.randn(32, 128, device=device)
  y2 = x2 @ x2.T
  _ = y2.cpu()
