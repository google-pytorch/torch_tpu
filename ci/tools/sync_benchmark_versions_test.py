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

"""Unit tests for sync_benchmark_versions.py."""

import pathlib
import sys
import tempfile
import unittest

_CI_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CI_TOOLS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools import sync_benchmark_versions

_LOCK = """\
libtpu==0.0.49 \\
    --hash=sha256:aaaa
torch==2.14.0+cpu \\
    --hash=sha256:bbbb
torchvision==0.29.0 \\
    --hash=sha256:cccc
"""

_REGISTRY = """\
benchmarks {
  environment_configs {
    id: "tpu-v7"
    workload_action_inputs { key: "torch_spec" value: "torch==2.13.0" }
    workload_action_inputs { key: "libtpu_spec" value: "libtpu==0.0.47" }
  }
  environment_configs {
    id: "gpu-b200"
    workload_action_inputs { key: "torch_spec" value: "torch==2.13.0" }
  }
}
"""

_PINS = sync_benchmark_versions.Pins(torch="2.14.0", libtpu="0.0.49")


class SyncBenchmarkVersionsTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):

  def test_read_pins_strips_local_version(self):
    self.assertEqual(sync_benchmark_versions.read_pins(_LOCK), _PINS)

  def test_read_pins_ignores_prefixed_packages(self):
    lock = "torchvision==0.29.0 \\\ntorch==2.14.0 \\\nlibtpu==0.0.49 \\\n"
    self.assertEqual(sync_benchmark_versions.read_pins(lock), _PINS)

  def test_read_pins_missing_package_raises(self):
    with self.assertRaisesRegex(ValueError, "libtpu"):
      sync_benchmark_versions.read_pins("torch==2.14.0 \\\n")

  def test_sync_registry_updates_all_pins(self):
    synced = sync_benchmark_versions.sync_registry(_REGISTRY, _PINS)
    self.assertEqual(synced.count('value: "torch==2.14.0"'), 2)
    self.assertEqual(synced.count('value: "libtpu==0.0.49"'), 1)
    self.assertNotIn("2.13.0", synced)
    self.assertNotIn("0.0.47", synced)
    self.assertEqual(sync_benchmark_versions.find_drift(synced, _PINS), [])

  def test_sync_registry_leaves_unpinned_specs(self):
    registry = 'workload_action_inputs { key: "torch_spec" value: "" }\n'
    self.assertEqual(
        sync_benchmark_versions.sync_registry(registry, _PINS), registry
    )

  def test_find_drift_reports_stale_pins(self):
    self.assertEqual(
        sync_benchmark_versions.find_drift(_REGISTRY, _PINS),
        [
            "torch_spec: torch==2.13.0 (expected torch==2.14.0)",
            "libtpu_spec: libtpu==0.0.47 (expected libtpu==0.0.49)",
        ],
    )

  def test_find_registry_in_parent_directory(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      source_root = pathlib.Path(temp_dir)
      registry = source_root / sync_benchmark_versions._REGISTRY_RELATIVE_PATH
      registry.parent.mkdir(parents=True)
      registry.write_text(_REGISTRY)
      build_root = source_root / "oss"
      build_root.mkdir()
      self.assertEqual(
          sync_benchmark_versions.find_registry(build_root), registry
      )

  def test_repository_registry_matches_latest_lock(self):
    lock = sync_benchmark_versions.find_latest_lock(_REPO_ROOT)
    registry = sync_benchmark_versions.find_registry(_REPO_ROOT)
    pins = sync_benchmark_versions.read_pins(lock.read_text())
    drift = sync_benchmark_versions.find_drift(registry.read_text(), pins)
    self.assertEqual(
        drift,
        [],
        f"{registry.name} is out of sync with {lock.name}; run"
        " `python3 ci/tools/sync_benchmark_versions.py` to update it.",
    )


if __name__ == "__main__":
  unittest.main()
