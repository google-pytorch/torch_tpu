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

"""End-to-end test for torch_tpu's eager-mode tlparse trace pipeline.

Exercises the full chain: ATen op dispatch -> StructuredLogBuffer Push ->
Python daemon drain -> torch._logging.trace_structured -> dedicated_log
file. Asserts on the FX format, the chromium_event timeline entry,
artifact metadata (cache_hit, reason, op_name), and source-location
attribution.

Skipped when TORCH_TRACE is not set at process startup. To run locally:
  TORCH_TRACE=/tmp/trace python -m unittest torch_tpu/tests/tracing_test.py
"""

import glob
import json
import math
import os
from typing import Any
import unittest

# Set TORCH_TRACE before importing torch/torch_tpu to ensure it is picked up
# by C++ initializers and avoids clashes with concurrent test runs.
if "TORCH_TRACE" not in os.environ:
  test_tmpdir = os.environ.get("TEST_TMPDIR", "/tmp")
  os.environ["TORCH_TRACE"] = os.path.join(
      test_tmpdir, "torchtpu_trace_test_dir"
  )

# Whether registered unbounded dynamic kernels may serve eager batches, as
# they do unless TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS is "none".
_UNBOUNDED_DYNAMIC_KERNELS_ENABLED = (
    os.environ.get("TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS") != "none"
)

# pylint: disable=g-import-not-at-top
from absl.testing import absltest
import torch
from torch_tpu._internal import dynamism
from torch_tpu._internal import execution_mode
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal import tracing
from tests import seed_test_utils


def _trace_dir() -> str:
  d = os.environ.get("TORCH_TRACE")
  assert d, "TORCH_TRACE must be set"
  return d


def _read_latest_log() -> str:
  log_files = glob.glob(os.path.join(_trace_dir(), "dedicated_log_*.log"))
  assert log_files, f"No dedicated_log files in {_trace_dir()}"
  latest = max(log_files, key=os.path.getctime)
  with open(latest, "r") as fh:
    return fh.read()


def _parse_artifacts(log_content: str) -> list[tuple[dict[str, Any], str]]:
  """Returns [(metadata, payload)] tuples from artifact log entries.

  PyTorch's structured log format is a metadata JSON line followed by an
  indented payload block ending at the next top-level line.

  Args:
    log_content: The raw log content to parse.
  """
  out = []
  lines = log_content.splitlines()
  i = 0
  while i < len(lines):
    line = lines[i]
    idx = line.find("{")
    if idx < 0:
      i += 1
      continue
    try:
      meta_outer = json.loads(line[idx:])
    except json.JSONDecodeError:
      i += 1
      continue
    artifact = meta_outer.get("artifact")
    if not artifact:
      i += 1
      continue
    payload_lines = []
    j = i + 1
    while j < len(lines) and (
        lines[j].startswith("\t") or not lines[j].strip()
    ):
      payload_lines.append(lines[j].lstrip("\t"))
      j += 1
    out.append((artifact, "\n".join(payload_lines).strip()))
    i = j
  return out


def _aten_graph_artifacts(op_name: str) -> list[tuple[dict[str, Any], str]]:
  """Returns the logged torchtpu_eager_aten_graph artifacts of `op_name`."""
  return [
      (meta, payload)
      for meta, payload in _parse_artifacts(_read_latest_log())
      if meta.get("name") == "torchtpu_eager_aten_graph"
      and meta.get("op_name") == op_name
  ]


def _nonzero_input(
    shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
  """Returns a materialized F32 input with zeros and more than two nonzeros."""
  x = torch.arange(1, math.prod(shape) + 1, dtype=torch.float32).reshape(shape)
  x[x % 3 == 0] = 0
  return x.to(device)


class TracingTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.assertTrue(
        tracing._tpu_torch_tracing._structured_log_enabled(),
        "TORCH_TRACE not enabled. This test requires TORCH_TRACE to be set.",
    )
    os.makedirs(os.environ["TORCH_TRACE"], exist_ok=True)
    # All trace tests run with deferred materialization so reasons reflect
    # the user's API call, not kDebugMode.
    self._old_mode = execution_mode.eager_mode
    self.addCleanup(setattr, execution_mode, "eager_mode", self._old_mode)
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    self._dev = torch.accelerator.current_accelerator()

  def test_daemon_started(self):
    self.assertIsNotNone(tracing._daemon)

  def test_fx_format_with_source_location(self):
    a = torch.randn(4, 8, device=self._dev, dtype=torch.bfloat16)
    b = torch.randn(8, 16, device=self._dev, dtype=torch.bfloat16)
    # Keep referenced: synchronize() only materializes live tensors.
    _ = torch.mm(a, b)
    torch.accelerator.synchronize()
    tracing._flush()

    artifacts = _parse_artifacts(_read_latest_log())
    aten_payloads = [
        payload
        for (meta, payload) in artifacts
        if meta.get("name") == "torchtpu_eager_aten_graph" and payload
    ]
    self.assertNotEmpty(
        aten_payloads, "no torchtpu_eager_aten_graph artifact with payload"
    )

    sample = aten_payloads[-1]  # the mm one
    # FX shape.
    self.assertIn("# Graph:", sample)
    self.assertIn("= input\n", sample)
    self.assertIn("(%", sample)  # operand list
    self.assertIn("return %", sample)
    # Source location on at least one op row.
    self.assertTrue(
        any(
            x in sample
            for x in (
                os.path.basename(__file__),
                "_launcher",
                "absltest",
                "unittest",
            )
        ),
        f"missing expected source-location in:\n{sample}",
    )

  def test_multi_output_op(self):
    # topk() returns (values, indices).
    # Renders as "%a, %b: ..., ... = topk(...)".
    a = torch.randn(64, 64, device=self._dev, dtype=torch.bfloat16)
    # Keep referenced: synchronize() only materializes live tensors.
    _ = torch.topk(a, k=5)
    torch.accelerator.synchronize()
    tracing._flush()

    artifacts = _parse_artifacts(_read_latest_log())
    payloads = [
        payload
        for (meta, payload) in artifacts
        if meta.get("name") == "torchtpu_eager_aten_graph" and payload
    ]
    multi = [p for p in payloads if "topk" in p and ", %" in p]
    self.assertNotEmpty(multi, "no multi-output FX row found")

  def test_reasons_cpu_and_synchronize(self):
    a = torch.ones(64, 64, device=self._dev)
    b = torch.ones(64, 64, device=self._dev)
    (a + b).cpu()  # should tag .cpu()
    # Keep referenced: synchronize() only materializes live tensors.
    _ = a + b
    torch.accelerator.synchronize()  # should tag synchronize()
    tracing._flush()

    reasons = {
        meta.get("reason")
        for meta, _ in _parse_artifacts(_read_latest_log())
        if meta.get("name") == "torchtpu_eager_aten_graph"
    }
    self.assertContainsSubset(
        {".cpu()", "synchronize()"},
        reasons,
        f"expected both .cpu() and synchronize() in {reasons}",
    )

  def test_chromium_event_emitted(self):
    a = torch.ones(32, 32, device=self._dev)
    b = torch.ones(32, 32, device=self._dev)
    # Keep referenced: synchronize() only materializes live tensors.
    _ = a + b
    torch.accelerator.synchronize()
    tracing._flush()

    content = _read_latest_log()
    self.assertIn('"chromium_event"', content)
    self.assertIn('"name":"torchtpu_eager', content)
    self.assertIn('"cat":"torchtpu_eager"', content)

  def test_cache_hit_after_repeat(self):
    # Create inputs (materialized, no deferred ops).
    a = torch.ones(32, 32, device="cpu").to(self._dev)
    b = torch.ones(32, 32, device="cpu").to(self._dev)

    # First call, cache miss.
    (a + b).cpu()
    tracing._flush()

    # Second call: same shape, same op, cache populated.
    (a + b).cpu()
    tracing._flush()

    hits = [
        meta.get("cache_hit")
        for meta, _ in _parse_artifacts(_read_latest_log())
        if meta.get("name") == "torchtpu_eager_aten_graph"
        and meta.get("op_name", "").startswith("torchtpu_eager/")
    ]
    self.assertGreaterEqual(len(hits), 2)
    self.assertTrue(
        hits[-1], f"expected last event to be a cache hit; got {hits}"
    )

  def test_static_kernel_type(self):
    torch.tpu._clear_cache()  # pylint: disable=protected-access
    a = torch.ones(24, 24, device="cpu").to(self._dev)
    (a - a).cpu()
    tracing._flush()

    events = [
        (meta, graph)
        for meta, graph in _parse_artifacts(_read_latest_log())
        if meta.get("name") == "torchtpu_eager_aten_graph"
        and meta.get("op_name", "").startswith("torchtpu_eager/sub")
    ]
    self.assertNotEmpty(events)
    meta, _ = events[-1]
    self.assertFalse(meta["cache_hit"])
    self.assertEqual(meta["kernel_type"], "static")
    self.assertIn('"kernel_type":"static"', _read_latest_log())

  def test_bounded_dynamic_kernel_type(self):
    torch.tpu._clear_cache()  # pylint: disable=protected-access
    x = torch.rand(5, 3).to(self._dev)
    dynamism.mark_dynamic(x, 0, 2, 20)
    torch.abs(x).cpu()
    tracing._flush()

    events = [
        (meta, graph)
        for meta, graph in _parse_artifacts(_read_latest_log())
        if meta.get("name") == "torchtpu_eager_aten_graph"
        and meta.get("op_name", "").startswith("torchtpu_eager/abs")
    ]
    self.assertNotEmpty(events)
    meta, _ = events[-1]
    self.assertFalse(meta["cache_hit"])
    self.assertEqual(meta["kernel_type"], "bounded_dynamic")

  @unittest.skipUnless(
      _UNBOUNDED_DYNAMIC_KERNELS_ENABLED,
      "requires unbounded dynamic kernels to be enabled",
  )
  def test_unbounded_dynamic_kernel_miss_then_hit(self):
    torch.tpu._clear_cache()  # pylint: disable=protected-access
    # The registered nonzero kernel serves both shapes from one cache entry:
    # the first batch builds it (a miss), the second reuses it (a hit).
    for shape in ((5, 5), (5, 1025)):
      torch.nonzero(_nonzero_input(shape, self._dev)).cpu()
    tracing._flush()

    events = _aten_graph_artifacts("torchtpu_eager/nonzero")
    self.assertGreaterEqual(len(events), 2)
    (miss_meta, miss_graph), (hit_meta, hit_graph) = events[-2:]
    self.assertFalse(miss_meta["cache_hit"])
    self.assertIn("= nonzero(", miss_graph)
    self.assertTrue(hit_meta["cache_hit"])
    self.assertEmpty(hit_graph)
    for meta in (miss_meta, hit_meta):
      self.assertEqual(meta["kernel_type"], "unbounded_dynamic")
    self.assertIn('"kernel_type":"unbounded_dynamic"', _read_latest_log())


if __name__ == "__main__":
  absltest.main()
