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
from typing import Final, Sequence
import zipfile
from absl.testing import absltest

# A concrete, non-inline XLA symbol we can look for in the .so.
# If this symbol is no longer present, you can find a replacement by running
# `nm -D --defined-only libpywrap_torch_tpu_deps.so | grep ' T '`
# and picking a stable-looking symbol that is definitely part of XLA.
XLA_SYMBOL: Final[str] = "Reindent"


def _names(wheel: pathlib.Path) -> Sequence[str]:
  with zipfile.ZipFile(wheel) as zf:
    return zf.namelist()


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


class WheelStructureTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    wheel_env = os.environ.get("WHEEL_PATH")
    if not wheel_env:
      raise ValueError("WHEEL_PATH environment variable is not set.")
    cls.wheel_path = pathlib.Path(wheel_env)
    if not cls.wheel_path.exists():
      raise FileNotFoundError(f"Wheel file not found at {cls.wheel_path}")

    names = _names(cls.wheel_path)
    # Find the common dependency library inside the wheel
    xla_lib = None
    for name in names:
      if name.endswith("libpywrap_torch_tpu_deps.so") or name.endswith(
          "libpywrap_torch_tpu_xla.so"
      ):
        xla_lib = name
        break

    if not xla_lib:
      raise FileNotFoundError(
          f"Could not find XLA library in wheel. Names: {names}"
      )

    # Use absltest's create_tempdir on the class requires a trick or we can just
    # use tempfile.
    # We will use the test runner's temp directory if available.
    cls.temp_dir_obj = absltest.get_default_test_tmpdir()
    dest_dir = pathlib.Path(cls.temp_dir_obj)
    common_so = _extract(cls.wheel_path, xla_lib, dest_dir)
    cls.symbols = _defined_dynamic_symbols(common_so)

  def test_xla_lib_has_xla_symbols(self):
    # Check XLA symbols exist
    self.assertIn(XLA_SYMBOL, self.symbols)

  def test_xla_lib_has_no_pytorch_symbols(self):
    pytorch_symbols = []
    for line in self.symbols.splitlines():
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
