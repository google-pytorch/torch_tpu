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

"""Unit tests for execution mode."""

import concurrent
import random
import threading

from absl.testing import absltest
from torch_tpu._internal import execution_mode

_EAGER_MODES = [
    execution_mode.EagerMode.DEFER_AND_FUSE,
    execution_mode.EagerMode.DEFER_NEVER,
    execution_mode.EagerMode.INTERNAL_DEFER_ALL,
]


class ExecutionModeTest(absltest.TestCase):
  """Unit tests for setting compiler options."""

  def test_eager_mode_nested_context(self):
    """Tests that nested context managers override each other correctly."""
    prev_mode = execution_mode.get_eager_mode()
    try:
      execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE)
      self.assertEqual(
          execution_mode.get_eager_mode(),
          execution_mode.EagerMode.DEFER_AND_FUSE,
      )

      with execution_mode.eager_mode(execution_mode.EagerMode.DEFER_NEVER):
        self.assertEqual(
            execution_mode.get_eager_mode(),
            execution_mode.EagerMode.DEFER_NEVER,
        )

        with execution_mode.eager_mode(
            execution_mode.EagerMode.INTERNAL_DEFER_ALL
        ):
          self.assertEqual(
              execution_mode.get_eager_mode(),
              execution_mode.EagerMode.INTERNAL_DEFER_ALL,
          )

        self.assertEqual(
            execution_mode.get_eager_mode(),
            execution_mode.EagerMode.DEFER_NEVER,
        )

      self.assertEqual(
          execution_mode.get_eager_mode(),
          execution_mode.EagerMode.DEFER_AND_FUSE,
      )
    finally:
      execution_mode.set_eager_mode(prev_mode)

  def test_eager_mode_global_override(self):
    """Tests interaction between global setting and context managers."""
    prev_mode = execution_mode.get_eager_mode()
    try:
      execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE)
      with execution_mode.eager_mode(execution_mode.EagerMode.DEFER_NEVER):
        self.assertEqual(
            execution_mode.get_eager_mode(),
            execution_mode.EagerMode.DEFER_NEVER,
        )
        # Changing global setting should not affect the current state, which is
        # determined once at the time of context entry and remains unchanged
        # within the context.
        execution_mode.set_eager_mode(
            execution_mode.EagerMode.INTERNAL_DEFER_ALL
        )
        self.assertEqual(
            execution_mode.get_eager_mode(),
            execution_mode.EagerMode.DEFER_NEVER,
        )

      # Upon exiting the context, the updated global setting takes effect.
      self.assertEqual(
          execution_mode.get_eager_mode(),
          execution_mode.EagerMode.INTERNAL_DEFER_ALL,
      )
    finally:
      execution_mode.set_eager_mode(prev_mode)

  def test_eager_mode_concurrent_nested_context(self):
    """Tests setting eager mode in nested contexts concurrently."""

    def _random_nested_context():
      mode1 = random.choice(_EAGER_MODES)
      mode2 = random.choice(_EAGER_MODES)
      # Nested context managers with random modes.
      with execution_mode.eager_mode(mode1):
        self.assertEqual(execution_mode.get_eager_mode(), mode1)
        with execution_mode.eager_mode(mode2):
          self.assertEqual(execution_mode.get_eager_mode(), mode2)
        self.assertEqual(execution_mode.get_eager_mode(), mode1)

    # Start 100 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
      futures = [executor.submit(_random_nested_context) for _ in range(100)]
      for future in futures:
        future.result()  # Ensure no exceptions were raised.

  def test_cpu_fallback_mode(self):
    """Tests that CPU fallback mode is a global setting shared across threads."""
    event = threading.Event()

    def _wait_and_check():
      event.wait()
      return execution_mode.enable_cpu_fallback

    expected_value = random.choice([True, False])

    # Start 100 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
      futures = [executor.submit(_wait_and_check) for _ in range(100)]
      # Update the global fallback mode randomly in the main thread.
      execution_mode.enable_cpu_fallback = expected_value
      event.set()
      # Ensure that all worker threads see the updated value.
      for future in futures:
        self.assertIs(future.result(), expected_value)

    # Reset
    execution_mode.enable_cpu_fallback = False


if __name__ == "__main__":
  absltest.main()
