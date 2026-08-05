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

import os
import tomllib
import zipfile

from absl.testing import absltest


def normalize_req(req_string):
  """Strips spaces, parentheses, and quotes to normalize dependency strings."""
  return (
      req_string.replace(" ", "")
      .replace("(", "")
      .replace(")", "")
      .replace('"', "")
      .replace("'", "")
  )


def get_wheel_dependencies(wheel_path, pyproject_path, is_local_torch):
  """Parses pyproject.toml and the built wheel to extract dependencies."""
  # 1. Parse expected dependencies from pyproject.toml.
  with open(pyproject_path, "rb") as f:
    pyproject = tomllib.load(f)
  expected_deps = {
      normalize_req(d) for d in pyproject["project"]["dependencies"]
  }
  if is_local_torch:
    expected_deps = {
        req for req in expected_deps if not req.startswith("torch")
    }

  # 2. Extract actual dependencies from the wheel.
  actual_deps = set()
  with zipfile.ZipFile(wheel_path, "r") as z:
    # Strictly look for the core METADATA file in the dist-info folder
    metadata_files = [
        n for n in z.namelist() if n.endswith(".dist-info/METADATA")
    ]
    if not metadata_files:
      raise FileNotFoundError(
          "Could not find .dist-info/METADATA file in wheel."
      )

    with z.open(metadata_files[0]) as f:
      for line in f.read().decode("utf-8").splitlines():
        if line.lower().startswith("requires-dist:"):
          # Extract the raw string after the colon.
          req = line.split(":", 1)[1]
          # Drop environment markers.
          req = req.split(";")[0]
          actual_deps.add(normalize_req(req))

  return expected_deps, actual_deps


class WheelSingleSourceTest(absltest.TestCase):  # ABSLTEST_OK=Wheel test

  def test_dependencies_match(self):
    expected_deps, actual_deps = get_wheel_dependencies(
        os.environ.get("WHEEL_PATH"),
        os.environ.get("PYPROJECT_PATH"),
        os.environ.get("LOCAL_TORCH") == "1",
    )
    mismatched_deps = expected_deps - actual_deps
    error_msg = ""
    if mismatched_deps:
      error_msg = "FAIL: The wheel dependencies do not match pyproject.toml.\n"
      error_msg += "\nMismatched dependencies:\n"
      for m in mismatched_deps:
        error_msg += f"  - {m}\n"
      error_msg += f"\nActually found in wheel: {actual_deps}"

    self.assertSetEqual(
        expected_deps,
        actual_deps,
        msg=error_msg,
    )


if __name__ == "__main__":
  absltest.main()
