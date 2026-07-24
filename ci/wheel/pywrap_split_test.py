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

"""Tests the XLA/PyTorch split of the built wheel's shared libraries.

Two things are checked:

  1. The pywrapped "glue" extension modules actually load. Importing
     `torch_tpu` and a couple of the heavier PyTorch-touching extensions
     dlopens the version-matched glue `.so` files, which resolve their
     libtorch symbols from the installed PyTorch. A broken split (or a glue
     built against the wrong PyTorch ABI) raises an undefined-symbol
     `ImportError` here.

  2. The dynamic symbol tables are cleanly partitioned. `libxla_base.so` holds
     the XLA/MLIR backend and defines no PyTorch (`c10`/`at`/`torch`)
     symbols, and the thin glues do not duplicate the XLA symbols they share via
     it. Libraries are located by reading `/proc/self/maps`, so this reflects
     what was actually dlopened.
"""

import collections
from collections.abc import Mapping, Sequence, Set
import importlib
import pathlib
import re
import subprocess
from typing import Final

from absl.testing import absltest
from absl.testing import parameterized

# A concrete, non-inline XLA symbol that lives in the shared base and never in a
# correctly-split thin glue.
_XLA_SYMBOL: Final[str] = "ShapeUtil"

# The single shared, version-independent XLA/MLIR base library.
_XLA_BASE: Final[str] = "libxla_base.so"

# Demangled-name prefixes for symbols *owned* by the c10 / at / torch
# namespaces.  Note the trailing "::" — this deliberately excludes torch_tpu's
# own functions (e.g. "torch_tpu::AtenAddOut(at::Tensor const&, ...)"), which
# merely mention these types in their signatures.
_TORCH_NAMESPACE_PREFIXES: Final[Sequence[str]] = ("c10::", "at::", "torch::")

# C++ ABI descriptors that carry the *owned* namespace *after* the descriptor,
# e.g. "vtable for c10::TensorImpl" or "non-virtual thunk to
# torch::autograd::...".  These RTTI/vtable/thunk symbols are the ones most
# likely to leak across the shared-library boundary (duplicate vtables and RTTI
# mismatches are a classic split-library failure), yet a plain startswith on
# the namespace would miss them because the namespace no longer sits at the
# front of the demangled name.
_ABI_DESCRIPTOR_PREFIXES: Final[Sequence[str]] = (
    "vtable for ",
    "VTT for ",
    "construction vtable for ",
    "typeinfo for ",
    "typeinfo name for ",
    "non-virtual thunk to ",
    "virtual thunk to ",
    "covariant return thunk to ",
    "guard variable for ",
)

# nm types that are not strong-defined: undefined (U) or weak/absolute
# (w/W/v/V).  Only strong-defined symbols mean the code actually lives in the
# library; PyTorch inline/template symbols emitted against torch headers are
# weak.
_NON_STRONG_TYPES: Final[Set[str]] = frozenset({"U", "w", "W", "v", "V"})


def _owns_pytorch_symbol(name: str) -> bool:
  """Whether `name` is a symbol *defined by* the c10 / at / torch namespaces.

  Strips a leading C++ ABI descriptor (so RTTI/vtable/thunk symbols are judged
  by the type they belong to) before checking the namespace, and relies on the
  trailing "::" to exclude torch_tpu's own symbols, which merely mention these
  types in signatures or template arguments.

  Args:
    name: The name of the symbol

  Returns:
    Whether or not the symbol is defined by the relevant namespaces.
  """
  for prefix in _ABI_DESCRIPTOR_PREFIXES:
    if name.startswith(prefix):
      name = name[len(prefix) :]
      break
  return name.startswith(_TORCH_NAMESPACE_PREFIXES)


def _loaded_object_paths() -> Mapping[str, pathlib.Path]:
  """Returns a basename -> path map of shared objects loaded by this process."""
  paths = {}
  maps = pathlib.Path("/proc/self/maps").read_text()
  # The " (deleted)" suffix appears when a mapped .so has been unlinked on disk
  # (e.g. a temp venv cleaned up mid-run); match it so those libraries still
  # map.  e.g. given a maps line like:
  #
  # 7f1234500000-7f1234600000 r-xp 00012000 08:01 5563991  /some/path/libfoo.so
  #
  # this grabs "/some/path/libfoo.so".
  pattern = r"\s(/\S+\.so(?:\.\S+)?)(?: \(deleted\))?$"
  for match in re.finditer(pattern, maps, flags=re.MULTILINE):
    path = pathlib.Path(match.group(1))
    paths.setdefault(path.name, path)
  return paths


def _demangle(names: Sequence[str]) -> Sequence[str]:
  if not names:
    return []
  return subprocess.run(
      ["c++filt"],
      input="\n".join(names),
      capture_output=True,
      text=True,
      check=True,
  ).stdout.splitlines()


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


class PywrapSplitTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    # Importing pytorch should dlopen the extensions and resolve their
    # libtorch symbols.
    super().setUpClass()
    import torch  # pylint: disable=unused-import, g-import-not-at-top,  # noqa: F401

  @parameterized.parameters(
      "torch_tpu._internal.compile",
      "torch_tpu._internal.sync",
  )
  def test_extensions_load(self, module: str) -> None:
    # Reaching here means torch_tpu imported, i.e. the extensions dlopen'd and
    # resolved their libtorch symbols. Importing the heaviest PyTorch-touching
    # extensions exercises that resolution.
    module = importlib.import_module(module)
    self.assertIsNotNone(module)

  def test_shared_base_defines_no_pytorch_symbols(self):
    loaded = _loaded_object_paths()
    self.assertIn(
        _XLA_BASE,
        loaded,
        f"{_XLA_BASE} was not dlopened; loaded torch_tpu objects:"
        f" {sorted(n for n, p in loaded.items() if 'torch_tpu' in str(p))}",
    )
    symbols = _strong_defined_symbols(loaded[_XLA_BASE])

    # Sanity: the XLA compiler backend really is in the shared library.
    self.assertTrue(
        any(_XLA_SYMBOL in name for name in symbols),
        f"Expected XLA symbol {_XLA_SYMBOL!r} in {_XLA_BASE}",
    )

    pytorch_symbols = [name for name in symbols if _owns_pytorch_symbol(name)]
    self.assertEmpty(
        pytorch_symbols,
        f"{_XLA_BASE} is shared across every PyTorch version, so it must define"
        " no PyTorch symbols (they are satisfied by libtorch at load), but it"
        f" defines: {pytorch_symbols[:20]}",
    )

  def test_thin_glues_do_not_duplicate_xla_symbols(self):
    loaded = _loaded_object_paths()
    # Glues are the module extensions -- neither the shared XLA base nor a
    # per-version torch common (lib..._<major>_<minor>_<patch>_common.so).
    glues = {
        name: path
        for name, path in loaded.items()
        if "torch_tpu" in str(path)
        and name != _XLA_BASE
        and not name.endswith("_common.so")
    }
    self.assertNotEmpty(
        glues, f"No thin glue .so files found among: {sorted(loaded)}"
    )

    offenders = collections.defaultdict(list)
    for name, path in glues.items():
      for sym in _strong_defined_symbols(path):
        if _XLA_SYMBOL in sym:
          offenders[name].append(sym)
    self.assertEmpty(
        offenders,
        "Thin glues must share XLA symbols via libxla_base.so, not define their"
        f" own copy, but found: {dict(offenders)}",
    )


if __name__ == "__main__":
  absltest.main()
