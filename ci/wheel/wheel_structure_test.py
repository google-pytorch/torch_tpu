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
import pathlib
import subprocess
import zipfile
from absl.testing import absltest
from absl.testing import parameterized

# A concrete, non-inline XLA symbol we can look for in the .so.
XLA_SYMBOL = "Reindent"

XLA_LIB = "torch_tpu/common/libpywrap_torch_tpu_xla.so"
COMMON_LIB = "torch_tpu/common/libpywrap_torch_tpu_common.so"

# Some representative thin C++ glue libraries.
PYTORCH_VERSIONS = ["2_10_0", "2_11_0", "2_12_0"]

GLUE_BASES = [
    "torch_tpu/_internal/env",
    "torch_tpu/_internal/testing",
    "torch_tpu/_internal/distributed/tpu_distributed",
    "torch_tpu/_internal/dynamism/_tpu_torch_dynamism",
]

GLUE_LIBS = []
for _base in GLUE_BASES:
  for _version in PYTORCH_VERSIONS:
    GLUE_LIBS.append(f"{_base}_actual_{_version}.so")


def _names(wheel: pathlib.Path) -> list[str]:
  with zipfile.ZipFile(wheel) as zf:
    return zf.namelist()


def _size(wheel: pathlib.Path, member: str) -> int:
  with zipfile.ZipFile(wheel) as zf:
    return zf.getinfo(member).file_size


def _extract(
    wheel: pathlib.Path, member: str, dest: pathlib.Path
) -> pathlib.Path:
  with zipfile.ZipFile(wheel) as zf:
    zf.extract(member, dest)
  return dest / member


def _defined_dynamic_symbols(so: pathlib.Path) -> str:
  return subprocess.run(
      ["nm", "-D", "--defined-only", str(so)],
      capture_output=True,
      text=True,
      check=True,
  ).stdout


def _needed_libraries(so: pathlib.Path) -> str:
  return subprocess.run(
      ["readelf", "-d", str(so)],
      capture_output=True,
      text=True,
      check=True,
  ).stdout


class WheelStructureTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    wheel_env = os.environ.get("WHEEL_PATH")
    if not wheel_env:
      self.fail("WHEEL_PATH environment variable is not set.")
    self.wheel_path = pathlib.Path(wheel_env)
    if not self.wheel_path.exists():
      self.fail(f"Wheel file not found at {self.wheel_path}")

  def test_common_lib_and_glues_present(self):
    names = _names(self.wheel_path)
    with self.subTest(msg="Checking XLA common library"):
      self.assertIn(XLA_LIB, names)
    with self.subTest(msg="Checking PyTorch common library"):
      self.assertIn(COMMON_LIB, names)
    for glue in GLUE_LIBS:
      with self.subTest(msg=f"Checking glue: {glue}"):
        self.assertIn(glue, names)

  def test_common_lib_holds_xla(self):
    dest_dir = pathlib.Path(self.create_tempdir().full_path)
    common_so = _extract(self.wheel_path, XLA_LIB, dest_dir)
    symbols = _defined_dynamic_symbols(common_so)
    self.assertIn(XLA_SYMBOL, symbols)

  @parameterized.parameters(GLUE_LIBS)
  def test_glue_is_thin_and_links_common(self, glue):
    dest_dir = pathlib.Path(self.create_tempdir().full_path)
    glue_so = _extract(self.wheel_path, glue, dest_dir)

    # 1. XLA symbols should NOT be duplicated in the thin glues.
    with self.subTest(msg=f"XLA symbols should not be in glue {glue}"):
      symbols = _defined_dynamic_symbols(glue_so)
      self.assertNotIn(XLA_SYMBOL, symbols)

    # 2. The thin glues should link against the common libraries.
    needed = _needed_libraries(glue_so)
    with self.subTest(msg=f"Glue {glue} should link PyTorch common library"):
      self.assertIn("libpywrap_torch_tpu_common.so", needed)
    with self.subTest(msg=f"Glue {glue} should link XLA common library"):
      self.assertIn("libpywrap_torch_tpu_xla.so", needed)
    with self.subTest(msg=f"Glue {glue} should have RUNPATH"):
      self.assertIn("RUNPATH", needed)
    with self.subTest(msg=f"Glue {glue} should have $ORIGIN in RUNPATH"):
      self.assertIn("$ORIGIN", needed)

  @parameterized.parameters(GLUE_LIBS)
  def test_glue_size_is_thin(self, glue):
    glue_size = _size(self.wheel_path, glue)
    common_size = _size(self.wheel_path, XLA_LIB)

    # The glue libraries should be tiny (typically < 50KB),
    # while the common library is huge (> 100MB).
    with self.subTest(
        msg=f"XLA common library should be larger than glue {glue}"
    ):
      self.assertGreater(common_size, glue_size * 10)
    with self.subTest(msg=f"Glue {glue} size should be reasonably small"):
      self.assertLess(glue_size, 2000 * 1024)

  def test_common_lib_has_no_pytorch_symbols(self):
    dest_dir = pathlib.Path(self.create_tempdir().full_path)
    common_so = _extract(self.wheel_path, XLA_LIB, dest_dir)
    symbols = _defined_dynamic_symbols(common_so)

    pytorch_symbols = []
    for line in symbols.splitlines():
      parts = line.strip().split()
      if len(parts) < 3:
        continue
      sym_type = parts[1]
      sym_name = parts[2]
      # Filter out weak symbols and undefined ones.
      if sym_type in ("W", "w", "V", "v", "U"):
        continue
      # Check if mangled name indicates c10, at, or torch namespace.
      for ns in ("N3c10", "N2at", "N5torch"):
        if ns in sym_name:
          pytorch_symbols.append((sym_type, sym_name))

    self.assertEmpty(
        pytorch_symbols,
        "Found strong defined PyTorch symbols in common library:"
        f" {pytorch_symbols}",
    )


if __name__ == "__main__":
  absltest.main()
