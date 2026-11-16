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

"""Tests for the TPU-dependent helpers in test_utils."""

from absl.testing import absltest
import torch
from torch._logging._internal import trace_log
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


def _plus_one(x: torch.Tensor) -> torch.Tensor:
  return x + 1


class CountTpuCompilationsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()
    self.x = torch.ones(8, device="tpu")
    torch.accelerator.synchronize()

  def test_counts_first_compilation(self):
    compiled = torch.compile(_plus_one, backend="tpu")

    with test_utils.count_tpu_compilations() as compilations:
      out = compiled(self.x)

    self.assertEqual(compilations.count, 1)
    test_utils.assert_close(out.cpu(), torch.full((8,), 2.0))

  def test_does_not_count_reuse_of_compiled_program(self):
    compiled = torch.compile(_plus_one, backend="tpu")
    compiled(self.x)

    with test_utils.count_tpu_compilations() as compilations:
      out = compiled(self.x)

    self.assertEqual(compilations.count, 0)
    test_utils.assert_close(out.cpu(), torch.full((8,), 2.0))

  def test_running_count_attributes_compilations_to_calls(self):
    compiled = torch.compile(_plus_one, backend="tpu", dynamic=False)
    y = torch.ones(16, device="tpu")
    torch.accelerator.synchronize()

    per_call = []
    with test_utils.count_tpu_compilations() as compilations:
      for inp in (self.x, y, self.x):
        before = compilations.count
        compiled(inp)
        per_call.append(compilations.count - before)

    self.assertEqual(per_call, [1, 1, 0])
    self.assertEqual(compilations.count, 2)

  def test_count_is_frozen_on_exit(self):
    with test_utils.count_tpu_compilations() as compilations:
      pass
    torch.compile(_plus_one, backend="tpu")(self.x)

    self.assertEqual(compilations.count, 0)

  def test_clear_cache_makes_count_independent_of_earlier_compilations(self):
    torch.compile(_plus_one, backend="tpu")(self.x)

    torch.compiler.reset()
    with test_utils.count_tpu_compilations(clear_cache=False) as warm:
      torch.compile(_plus_one, backend="tpu")(self.x)
    torch.compiler.reset()
    with test_utils.count_tpu_compilations() as cleared:
      torch.compile(_plus_one, backend="tpu")(self.x)

    self.assertEqual(warm.count, 0)
    self.assertEqual(cleared.count, 1)


class CaptureTraceArtifactsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()

  def test_captures_tpu_compile_artifacts(self):
    x = torch.ones(8, device="tpu")

    with test_utils.capture_trace_artifacts() as artifacts:
      torch.compile(_plus_one, backend="tpu")(x)

    payloads = {a.name: a.payload for a in artifacts}
    self.assertIn("add", payloads["torchtpu_fx_graph"])
    self.assertIn("stablehlo.add", payloads["torchtpu_stablehlo_graph"])

  def test_removes_handler_on_exit(self):
    handlers_before = list(trace_log.handlers)

    with test_utils.capture_trace_artifacts():
      self.assertLen(trace_log.handlers, len(handlers_before) + 1)

    self.assertEqual(trace_log.handlers, handlers_before)


if __name__ == "__main__":
  absltest.main()
