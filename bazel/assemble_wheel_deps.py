#!/usr/bin/env python3
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

"""Assembles extra test dependencies into a ZIP file for wheel testing."""

import pathlib
import sys
import zipfile


def get_zip_relpath(path: pathlib.Path) -> str:
  """Calculates the relative path for a file in the output ZIP.

  Standardizes and strips paths to ensure they are relative to 'site-packages/'
  if that marker exists, or strips leading 'src/' if present.

  Args:
    path: The original path of the file.

  Returns:
    The relative path string to be used inside the ZIP file.
  """
  parts = path.parts
  if "site-packages" in parts:
    return str(pathlib.Path(*parts[parts.index("site-packages") + 1 :]))
  if parts and parts[0] == "src":
    return str(pathlib.Path(*parts[1:]))
  return str(path)


def main() -> None:
  if len(sys.argv) < 3:
    sys.exit("Usage: assemble_wheel_deps.py <output_zip> <input_files...>")

  _, output_zip, *input_files = sys.argv

  with zipfile.ZipFile(output_zip, "w") as z:
    for f in map(pathlib.Path, input_files):
      relative_path = get_zip_relpath(f)

      # Avoid duplicates (skip if already exists)
      try:
        z.getinfo(relative_path)
      except KeyError:
        z.write(f, arcname=relative_path)


if __name__ == "__main__":
  main()
