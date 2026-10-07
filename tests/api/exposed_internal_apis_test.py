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

"""Verifies that `exposed_internal_apis.yaml` matches the discovered exposed internal APIs.

This test guards TorchTPU's internal API exposure: any addition, deletion, or
modification of internal Python APIs or environment variables exposed in
documentation, tutorials, or examples must be reflected in
`exposed_internal_apis.yaml`, which requires approval from the API owners.
"""

import os
from typing import Any

from absl.testing import absltest
from tests import repo_paths
from tests import seed_test_utils
from tests.api import scan_exposed_internal_apis
import yaml

# Relative to the TorchTPU repository root.
_OWNERS_PATH = "tests/api/OWNERS"

_UPDATE_INSTRUCTIONS = f"""\
Any change (addition, deletion, or modification) to TorchTPU internal APIs or
environment variables exposed in documentation and examples needs approval from
the `exposed_internal_apis.yaml` owners listed in `{_OWNERS_PATH}`.

{scan_exposed_internal_apis.UPDATE_INSTRUCTIONS}"""


def _parse_exposed_internal_apis_yaml(
    yaml_text: str,
) -> dict[str, dict[str, Any]]:
  """Parses `exposed_internal_apis.yaml` text into a dictionary of symbol -> attributes."""
  entries: dict[str, dict[str, Any]] = {}
  for line in yaml_text.splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
      continue
    entry = yaml.safe_load(stripped)
    if isinstance(entry, dict):
      for name, attributes in entry.items():
        entries[str(name)] = attributes
  return entries


def _describe_diff(
    expected: dict[str, dict[str, Any]],
    actual: dict[str, dict[str, Any]],
) -> list[str]:
  """Returns human-readable lines describing differences between baseline and actual."""
  lines: list[str] = []
  added = sorted(actual.keys() - expected.keys())
  removed = sorted(expected.keys() - actual.keys())
  modified = sorted(
      key
      for key in expected.keys() & actual.keys()
      if expected[key] != actual[key]
  )

  if added:
    lines.append(f"Newly exposed internal APIs / env vars ({len(added)}):")
    for key in added:
      used_in = actual[key].get("used_in", [])
      lines.append(f"  + {key} (used in {len(used_in)} locations)")
  if removed:
    lines.append(
        f"No longer exposed internal APIs / env vars ({len(removed)}):"
    )
    for key in removed:
      lines.append(f"  - {key}")
  if modified:
    lines.append(f"Modified usage locations ({len(modified)}):")
    for key in modified:
      old_used = set(expected[key].get("used_in", []))
      new_used = set(actual[key].get("used_in", []))
      added_locs = sorted(new_used - old_used)
      removed_locs = sorted(old_used - new_used)
      lines.append(f"  ~ {key}")
      for loc in added_locs:
        lines.append(f"      + {loc}")
      for loc in removed_locs:
        lines.append(f"      - {loc}")
  return lines


class ExposedInternalApisTest(seed_test_utils.RepeatableTest):

  def test_exposed_internal_apis_matches_scanned_code(self):
    root = repo_paths.get_repo_root_in_runfiles()
    baseline_path = os.path.join(
        root, repo_paths.EXPOSED_INTERNAL_APIS_YAML_PATH
    )
    tt_apis_path = os.path.join(root, repo_paths.TT_APIS_YAML_PATH)
    # The loader silently ignores a missing `tt_apis.yaml`.
    self.assertTrue(
        os.path.exists(tt_apis_path), f"tt_apis.yaml not found: {tt_apis_path}"
    )
    for subdir in scan_exposed_internal_apis.DEFAULT_SCAN_SUBDIRS:
      scan_dir = os.path.join(root, subdir)
      self.assertTrue(
          os.path.exists(scan_dir), f"Scan directory not found: {scan_dir}"
      )

    # `used_in` paths are relative to the repository root.
    internal_py, internal_env = scan_exposed_internal_apis.scan_directories(
        [
            os.path.join(root, subdir)
            for subdir in scan_exposed_internal_apis.DEFAULT_SCAN_SUBDIRS
        ],
        tt_apis_path,
        repository_root=root,
    )
    actual_text = scan_exposed_internal_apis.format_exposed_internal_apis_yaml(
        internal_py, internal_env
    )
    # Export up-to-date baseline so it can be downloaded from Sponge.
    scan_exposed_internal_apis.export_sponge_artifact(actual_text)

    with open(baseline_path, "r", encoding="utf-8") as f:
      expected_text = f.read()

    if actual_text.strip() == expected_text.strip():
      return

    diff_lines = _describe_diff(
        _parse_exposed_internal_apis_yaml(expected_text),
        _parse_exposed_internal_apis_yaml(actual_text),
    )
    if not diff_lines:
      diff_lines = [
          "No API entries differ, but the formatting of"
          " exposed_internal_apis.yaml is out of date."
      ]

    self.fail(
        "exposed_internal_apis.yaml does not match the internal APIs / env vars"
        " discovered across documentation and examples.\n\n"
        + "\n".join(diff_lines)
        + "\n\n"
        + _UPDATE_INSTRUCTIONS
    )


if __name__ == "__main__":
  absltest.main()
