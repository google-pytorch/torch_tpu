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

import email.message
import email.parser
import os
import pathlib
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


def get_wheel_metadata_files(wheel_path: str) -> list[str]:
  """Returns all archive member paths containing 'METADATA'."""
  with zipfile.ZipFile(wheel_path, "r") as z:
    return [n for n in z.namelist() if "METADATA" in n]


def read_wheel_metadata(wheel_path: str) -> email.message.Message:
  """Parses the .dist-info/METADATA file from the built wheel."""
  with zipfile.ZipFile(wheel_path, "r") as z:
    metadata_files = [
        n for n in z.namelist() if n.endswith(".dist-info/METADATA")
    ]
    if not metadata_files:
      raise FileNotFoundError(
          "Could not find .dist-info/METADATA file in wheel."
      )
    return email.parser.BytesParser().parsebytes(z.read(metadata_files[0]))


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
  msg = read_wheel_metadata(wheel_path)
  actual_deps = {
      normalize_req(req.split(";")[0])
      for req in msg.get_all("Requires-Dist") or ()
  }

  return expected_deps, actual_deps


class WheelSingleSourceTest(absltest.TestCase):  # ABSLTEST_OK=Wheel test

  def test_single_metadata_file_in_wheel(self):
    metadata_files = get_wheel_metadata_files(os.environ["WHEEL_PATH"])
    self.assertLen(
        metadata_files,
        1,
        "Expected exactly one file matching '*METADATA*' in the wheel (Twine"
        " searches all entries containing 'METADATA'), but found:"
        f" {metadata_files}",
    )
    self.assertTrue(
        metadata_files[0].endswith(".dist-info/METADATA"),
        f"Expected .dist-info/METADATA, got {metadata_files[0]}",
    )

  def test_description_and_summary_match(self):
    pyproject_path = pathlib.Path(os.environ["PYPROJECT_PATH"])
    with pyproject_path.open("rb") as f:
      pyproject = tomllib.load(f)
    readme_path = pyproject_path.parent / pyproject["project"]["readme"]
    expected_readme = readme_path.read_text(encoding="utf-8").strip()

    msg = read_wheel_metadata(os.environ["WHEEL_PATH"])
    self.assertEqual(msg["Summary"], pyproject["project"]["description"])
    self.assertEqual(msg["Description-Content-Type"], "text/markdown")
    self.assertEqual(msg.get_payload().strip(), expected_readme)

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
