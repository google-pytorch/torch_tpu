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

"""Structural checks on the multi-PyTorch-ABI wheel.

The separated wheel bundles, for every supported PyTorch version, a thin glue
`.so` per C++ extension plus a small per-version "torch common" library, and
shares a single large XLA/MLIR base library (`libxla_base.so`) across all
versions. This test opens the built wheel (`WHEEL_PATH`) and asserts:

  * exactly one shared `libxla_base.so`;
  * versioned glue and a per-version torch common exist for each built version;
  * the shared base holds the XLA backend and defines no PyTorch symbols, so it
    is genuinely version-independent (it is the same file for every version);
  * no libtorch/libc10 is bundled (torch is resolved from the user's install).
"""

from collections.abc import Sequence, Set
import functools
import os
import pathlib
import re
import subprocess
import tempfile
from typing import Final
import zipfile

from absl.testing import absltest

# The single shared, version-independent XLA/MLIR base library.
_XLA_BASE: Final[str] = "torch_tpu/common/libxla_base.so"

# A concrete, non-inline XLA symbol that lives in the shared base library.
_XLA_SYMBOL: Final[str] = "ShapeUtil"

# Demangled-name prefixes for symbols owned by the c10 / at / torch namespaces.
# The trailing "::" excludes torch_tpu's own functions, which merely mention
# these types in their signatures.
_TORCH_NAMESPACE_PREFIXES: Final[Sequence[str]] = ("c10::", "at::", "torch::")

# nm types that are not strong-defined: undefined (U) or weak/absolute
# (w/W/v/V). Only strong-defined symbols mean the code lives in the library.
_NON_STRONG_TYPES: Final[Set[str]] = frozenset({"U", "w", "W", "v", "V"})

# Each glue file is <module>_<major>_<minor>_<patch>.so (the nightly channel's
# glue carries the release triple its snapshot leads up to, like any other);
# the per-version torch common is lib<...>_<version>_common.so.
_GLUE_RE: Final[re.Pattern[str]] = re.compile(r"_(\d+_\d+_\d+)\.so$")
_TORCH_COMMON_RE: Final[re.Pattern[str]] = re.compile(
    r"_(\d+_\d+_\d+)_common\.so$"
)


def _wheel_path() -> pathlib.Path:
  path = os.environ.get("WHEEL_PATH")
  assert path, "WHEEL_PATH environment variable is not set"
  return pathlib.Path(path)


def _demangle(names: Sequence[str]) -> Sequence[str]:
  if not names:
    return []
  out = subprocess.run(
      ["c++filt"],
      input="\n".join(names),
      capture_output=True,
      text=True,
      check=True,
  ).stdout
  return out.splitlines()


def _strong_defined_symbols(so: pathlib.Path) -> Sequence[str]:
  out = subprocess.run(
      ["nm", "-D", "--defined-only", str(so)],
      capture_output=True,
      text=True,
      check=True,
  ).stdout
  mangled = []
  for line in out.splitlines():
    parts = line.split()
    if len(parts) < 3 or parts[1] in _NON_STRONG_TYPES:
      continue
    mangled.append(parts[2])
  return _demangle(mangled)


def _dt_needed(so: pathlib.Path) -> Sequence[str]:
  out = subprocess.run(
      ["readelf", "-d", str(so)],
      capture_output=True,
      text=True,
      check=True,
  ).stdout
  return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


class WheelStructureTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # The wheel is large (for non-optimized builds, the base alone is ~1.8 GB),
    # so unzip it once for the whole class. We cannot use create_tempdir
    # because there is no classmethod version of it.
    cls._wheel_root = pathlib.Path(
        cls.enter_context(tempfile.TemporaryDirectory())
    )

    with zipfile.ZipFile(_wheel_path()) as z:
      z.extractall(cls._wheel_root)

  @functools.cached_property
  def _names(self) -> Sequence[str]:
    return tuple(
        p.relative_to(self._wheel_root).as_posix()
        for p in self._wheel_root.rglob("*")
        if p.is_file()
    )

  @functools.cached_property
  def _xla_base_symbols(self) -> Sequence[str]:
    return _strong_defined_symbols(self._wheel_root / _XLA_BASE)

  @functools.cached_property
  def _xla_base_needed(self) -> Sequence[str]:
    return _dt_needed(self._wheel_root / _XLA_BASE)

  def _built_versions(self) -> Set[str]:
    versions = set()
    for name in self._names:
      m = _GLUE_RE.search(pathlib.PurePosixPath(name).name)
      if m and "_common.so" not in name:
        versions.add(m.group(1))
    return versions

  def test_exactly_one_shared_xla_base(self):
    bases = [n for n in self._names if n == _XLA_BASE]
    self.assertLen(
        bases,
        1,
        f"Expected exactly one {_XLA_BASE}; found: {bases}",
    )

  def test_each_version_has_glues_and_a_torch_common(self):
    versions = self._built_versions()
    self.assertNotEmpty(versions, "No versioned glue .so files found in wheel.")

    torch_commons = {
        m.group(1)
        for n in self._names
        if (m := _TORCH_COMMON_RE.search(pathlib.PurePosixPath(n).name))
    }
    for version in versions:
      # The env extension is present in every build; use it as a representative.
      self.assertIn(
          f"torch_tpu/_internal/env_{version}.so",
          self._names,
          f"Missing env glue for version {version}",
      )
      self.assertIn(
          version,
          torch_commons,
          f"Missing per-version torch common for {version}; found"
          f" {torch_commons}",
      )

  def test_no_libtorch_bundled(self):
    bundled = [
        n
        for n in self._names
        if re.search(r"(^|/)(libc10|libtorch)", pathlib.PurePosixPath(n).name)
    ]
    self.assertEmpty(
        bundled,
        "The wheel must not bundle libtorch/libc10 (resolved from the user's"
        f" installed torch), but found: {bundled}",
    )

  def test_shared_base_is_xla_and_torch_free(self):
    symbols = self._xla_base_symbols
    self.assertTrue(
        any(_XLA_SYMBOL in name for name in symbols),
        f"Expected XLA symbol {_XLA_SYMBOL!r} in the shared base library.",
    )
    pytorch_symbols = [
        name for name in symbols if name.startswith(_TORCH_NAMESPACE_PREFIXES)
    ]
    self.assertEmpty(
        pytorch_symbols,
        "The shared XLA base must define no PyTorch symbols (so it is identical"
        f" for every version), but defines: {pytorch_symbols[:20]}",
    )

  # TODO(pganssle): Fails today -- pywrap_library force-adds every dynamic dep
  # (libtorch) to all common-lib partitions, including this base. Strip the
  # stray entries in packaging, then remove the expectedFailure.
  @absltest.expectedFailure
  def test_shared_base_has_no_libtorch_dt_needed(self):
    """The shared XLA base must record no libtorch DT_NEEDED entries.

    The base defines no torch symbols, so it should carry no link-time
    dependency on libc10/libtorch either; that is what lets a single base
    library serve every packaged PyTorch version.
    """
    torch_needed = [
        n for n in self._xla_base_needed if re.match(r"lib(c10|torch)", n)
    ]
    self.assertEmpty(
        torch_needed,
        f"The shared XLA base must not DT_NEEDED any libtorch: {torch_needed}",
    )


if __name__ == "__main__":
  absltest.main()
