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

import glob
import json
import os
import shutil
from absl.testing import absltest
import torch
from torch_tpu import api
from torch_tpu._internal import execution_mode
from torch_tpu._internal import tracing
from torch_tpu._internal.sync import _tpu_torch_sync
from torch_tpu._internal.sync import sync


class TracingTest(absltest.TestCase):

  def test_disabled_by_default(self):
    if _tpu_torch_sync._structured_log_enabled():
      self.skipTest("TORCH_TRACE is set, skipping disabled test")
    self.assertFalse(_tpu_torch_sync._structured_log_enabled())
    self.assertEqual(_tpu_torch_sync._drain_structured_log_buffer(), [])

  def test_enabled_by_env(self):
    if not _tpu_torch_sync._structured_log_enabled():
      self.skipTest("TORCH_TRACE not set")

    self.assertTrue(tracing._started)
    dev = api.tpu_device()
    a = torch.randn(32, 32, device=dev)
    b = torch.randn(32, 32, device=dev)
    (a + b).cpu()

    tracing.flush()

    log_dir = os.environ.get("TORCH_TRACE")
    self.assertTrue(log_dir)

    log_files = glob.glob(os.path.join(log_dir, "dedicated_log_*.log"))
    self.assertGreaterEqual(len(log_files), 1)

    latest_log = max(log_files, key=os.path.getctime)
    with open(latest_log, "r") as f:
      content = f.read()
      self.assertIn("torchtpu_eager_aten_graph", content)
      self.assertIn("torchtpu_eager_stablehlo", content)
      self.assertIn("torchtpu_eager_sync", content)

  def test_all_reasons(self):
    if not _tpu_torch_sync._structured_log_enabled():
      self.skipTest("TORCH_TRACE not set")

    # Ensure we defer execution to test materialization reasons
    old_mode = execution_mode.get_eager_mode()
    self.addCleanup(execution_mode.set_eager_mode, old_mode)
    execution_mode.set_eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL)

    dev = api.tpu_device()

    a = torch.ones(64, 64, device=dev)
    b = torch.ones(64, 64, device=dev)
    (a + b).cpu()

    (a.mean() * 2.0).item()

    d = torch.ones(128, 128, device=dev)
    e = torch.ones(128, 128, device=dev)
    sync.synchronize(d + e, wait=True)

    tracing.flush()

    log_dir = os.environ.get("TORCH_TRACE")
    log_files = glob.glob(os.path.join(log_dir, "dedicated_log_*.log"))
    self.assertNotEmpty(log_files, f"No log files found in {log_dir}")
    latest_log = max(log_files, key=os.path.getctime)

    # Use try...finally to ensure the log file is copied to test outputs
    # even if the assertion fails, aiding in remote debugging on Forge.
    try:
      reasons = set()
      with open(latest_log, "r") as fh:
        for line in fh:
          idx = line.find("{")
          if idx < 0:
            continue
          try:
            meta = json.loads(line[idx:])
          except json.JSONDecodeError:
            continue
          art = meta.get("artifact") or {}
          if art.get("name") == "torchtpu_eager_aten_graph":
            reasons.add(art.get("reason"))

      expected = {".cpu()", "synchronize()"}
      self.assertContainsSubset(expected, reasons)
    finally:
      # Copy log file to undeclared outputs so it gets fetched from Forge
      undeclared_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
      if undeclared_outputs_dir:
        shutil.copy(latest_log, undeclared_outputs_dir)

  def test_cache_hit(self):
    if not _tpu_torch_sync._structured_log_enabled():
      self.skipTest("TORCH_TRACE not set")

    old_mode = execution_mode.get_eager_mode()
    self.addCleanup(execution_mode.set_eager_mode, old_mode)
    execution_mode.set_eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL)

    dev = api.tpu_device()
    a = torch.ones(32, 32, device=dev)
    b = torch.ones(32, 32, device=dev)

    # First execution (Miss)
    (a + b).cpu()
    tracing.flush()

    # Second execution (Hit)
    (a + b).cpu()
    tracing.flush()

    log_dir = os.environ.get("TORCH_TRACE")
    log_files = glob.glob(os.path.join(log_dir, "dedicated_log_*.log"))
    self.assertNotEmpty(log_files)
    latest_log = max(log_files, key=os.path.getctime)

    hits = []
    with open(latest_log, "r") as fh:
      for line in fh:
        idx = line.find("{")
        if idx < 0:
          continue
        try:
          meta = json.loads(line[idx:])
        except json.JSONDecodeError:
          continue
        art = meta.get("artifact") or {}
        op_name = art.get("op_name")
        if op_name and op_name.startswith("torchtpu_eager/"):
          hits.append(art.get("cache_hit"))

    self.assertGreaterEqual(len(hits), 2)
    self.assertTrue(
        hits[-1], f"Expected last event to be a cache hit, got {hits}"
    )


if __name__ == "__main__":
  absltest.main()
