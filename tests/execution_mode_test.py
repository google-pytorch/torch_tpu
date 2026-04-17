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

"""Unit tests for ops."""

import concurrent
import random

from absl.testing import absltest
import torch
import torch.accelerator
from torch_tpu._internal import execution_mode


class ExecutionModeTest(absltest.TestCase):
  """Unit tests for setting compiler options."""

  def setUp(self):
    super().setUp()
    self.tpu = torch.accelerator.current_accelerator()

  def test_eager_mode_in_threads(self):
    """Tests setting compiler options in nested contexts concurrently."""
    eager_modes = [
        execution_mode.EagerMode.DEFER_AND_FUSE,
        execution_mode.EagerMode.DEFER_NEVER,
        execution_mode.EagerMode.INTERNAL_DEFER_ALL,
    ]

    def _change_eager_mode(new_mode):
      with execution_mode.eager_mode(new_mode):
        self.assertEqual(execution_mode.get_eager_mode(), new_mode)

    # Start 100 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
      executor.submit(_change_eager_mode, random.choice(eager_modes))

  def test_fallback_mode(self):
    fallback_modes = [True, False]

    def _change_fallback_mode(new_mode):
      with execution_mode.cpu_fallback_mode(new_mode):
        self.assertEqual(execution_mode.is_cpu_fallback_enabled(), new_mode)

    # Start 100 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
      executor.submit(_change_fallback_mode, random.choice(fallback_modes))


if __name__ == "__main__":
  absltest.main()
