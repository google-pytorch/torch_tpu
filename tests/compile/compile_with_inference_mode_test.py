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

"""Ensure we error out at torch.compile() when "dynamic" is applyed"""

import os
import unittest

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import utils


class TestModule(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.range = torch.arange(1, 9, dtype=torch.float32)

  @torch.inference_mode()
  def forward(self, start_pos: int):
    res = self.range[start_pos : start_pos + 3]
    return res


class EnsureNoDynamicTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    torch.compiler.reset()

  # Likely a bug in torch.compile() when inference_mode is applied. See
  # github issue: https://github.com/pytorch/pytorch/issues/169477
  @absltest.expectedFailure
  def test_compile_with_inference_mode_tpu(self):
    m = TestModule()
    compiled = torch.compile(m, backend="tpu")
    compiled(1).to("cpu")

  # Added the CPU test to demonstrate it is not a TPU specific issue.
  # TODO: Delete the CPU test once it is fixed.
  @absltest.expectedFailure
  def test_compile_with_inference_mode_cpu(self):
    m = TestModule()
    compiled = torch.compile(m)
    compiled(1)


if __name__ == "__main__":
  absltest.main()
