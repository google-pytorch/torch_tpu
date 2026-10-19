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

from absl.testing import absltest
import torch
from tests import seed_test_utils


class TpuStreamsTest(seed_test_utils.RepeatableTest):
  """Tests for torch.tpu.Stream and torch.tpu.Event.

  This is the TPU-specific class implementation, which may have additional
  functionality beyond the torch.Stream and torch.Event specification.
  """

  def test_stream_unimplemented_method(self):

    expected_msg = (
        'Streams and Events are not fully implemented in TorchTPU. Please file'
        ' a feature request describing your use case.'
    )
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      default_stream = torch.tpu.default_stream()
      default_stream.query()

  def test_stream_partially_implemented_method(self):
    """Tests that unimplemented stream methods raise NotImplementedError."""

    s = torch.tpu.Stream()
    expected_msg = (
        'Streams and Events are not fully implemented in TorchTPU. Please file'
        ' a feature request describing your use case.'
    )
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      s.query()
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      s.priority_range()

  def test_event_record_invalid_type(self):
    event = torch.tpu.Event()
    with self.assertRaisesRegex(
        TypeError, 'expected torch.Stream, but got str'
    ):
      event.record('invalid')

  def test_event_partially_implemented(self):
    """Tests that ipc event methods remain unimplemented."""

    e = torch.tpu.Event()
    expected_msg = (
        'Streams and Events are not fully implemented in TorchTPU. Please file'
        ' a feature request describing your use case.'
    )
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      e.ipc_handle()


if __name__ == '__main__':
  absltest.main()
