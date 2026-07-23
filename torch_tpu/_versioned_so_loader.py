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

"""Runtime dispatch of PyTorch-version-specific torch_tpu C++ extensions.

torch_tpu bundles a thin `.so` glue per supported PyTorch build, each named
`<module>_<major>_<minor>_<patch>.so` (e.g. `env_2_12_0.so`). A
`MetaPathFinder` installed at import time rewrites imports of
`torch_tpu._internal.<module>` to the glue that matches the installed PyTorch,
and an `ExtensionProxyLoader` bridges the artifact's version-suffixed filename
to the unversioned `PyInit_<module>` symbol the C++ code actually exports.

Selection policy: a new glue is added to the wheel only when a PyTorch release
breaks ABI compatibility with the previous glue, so each glue serves every
release from its own version up to (but not including) the next glue's -- and
the newest glue serves everything after it, including releases newer than any
we built for. The loader therefore picks the newest glue whose version is <=
the installed PyTorch. Loading eagerly binds its symbols (the glues are linked
with `-z now`), so an ABI-incompatible glue fails loudly at import rather
than crashing later. If the installed PyTorch version cannot be determined
while versioned glues exist, or predates every built glue, that is an error.
When no versioned glues are present at all (e.g. an unversioned developer
build) the hook is not installed and imports resolve normally.

A dev build's version compares by the release triple it leads up to
(`2.14.0.dev20260711` -> `2.14.0`), so the glue the nightly-channel wheel
bundles -- built against a pinned dev snapshot and named by that snapshot's
release triple, like any other glue -- serves it whenever present.
"""

from collections.abc import Collection
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import pathlib
import sys
from typing import Final

_INTERNAL_PREFIX: Final[str] = "torch_tpu._internal."

# Resolved once by install_hook(): the suffix of the glue variant this process
# loads, or None when dispatch is disabled.
_VERSION_SUFFIX: str | None = None


def _parse(suffix: str) -> tuple[int, ...]:
  return tuple(int(part) for part in suffix.split("_"))


def installed_torch_version() -> str | None:
  """Returns the installed torch's full version string, or None.

  Prefers `torch.__version__` -- it reflects the torch actually loaded in
  this process, which is what the glue must match. torch is deliberately never
  imported from here: torch_tpu is itself mid-import and importing torch would
  re-enter it through torch's backend autoload. When torch is not already in
  `sys.modules` (torch_tpu imported first), the install metadata answers
  instead.
  """
  version = getattr(sys.modules.get("torch"), "__version__", None)
  if version is not None:
    return str(version)
  try:
    return importlib.metadata.version("torch")
  except importlib.metadata.PackageNotFoundError:
    return None


def version_suffix(version: str) -> str | None:
  """Turns a torch version into a glue suffix ("2.13.0+cpu" -> "2_13_0")."""
  # Cut any local tag first so it cannot ride on the patch component
  # ("2.13.0+cpu" would otherwise split into ("2", "13", "0+cpu")); a ".dev"
  # tag is a component of its own and falls off the triple.
  parts = version.split("+")[0].split(".")[:3]
  if len(parts) < 3 or not all(part.isdigit() for part in parts):
    return None
  return "_".join(parts)


def torch_version_suffix() -> str | None:
  """Returns the installed torch's "<major>_<minor>_<patch>", or None."""
  version = installed_torch_version()
  if version is None:
    return None
  return version_suffix(version)


def _glue_suffix(so_name: str) -> str | None:
  """Extracts the version suffix of a versioned glue filename.

  A glue artifact is named `<module>_<major>_<minor>_<patch>.so`, e.g.
  `env_2_12_0.so` -> `2_12_0`.

  Args:
    so_name: the artifact's filename (not its path), e.g. `env_2_12_0.so`.

  Returns:
    The version suffix (e.g. `2_12_0`), or None if the filename carries no
    version suffix (an unversioned artifact).
  """
  parts = so_name.removesuffix(".so").split("_")
  if len(parts) < 4 or not all(part.isdigit() for part in parts[-3:]):
    return None
  return "_".join(parts[-3:])


def discover_built_suffixes(
    package_dir: pathlib.Path,
) -> frozenset[str]:
  """Returns the set of glue suffixes bundled under `package_dir`."""
  return frozenset(
      suffix
      for so in package_dir.rglob("*.so")
      if (suffix := _glue_suffix(so.name)) is not None
  )


def resolve_suffix(
    running: str | None,
    built: Collection[str],
) -> str | None:
  """Chooses the glue suffix to load, or None to disable dispatch.

  Glues are built only for the PyTorch releases that broke ABI compatibility
  with the previous glue, so each glue serves every release from its own
  version up to the next glue's -- and the newest glue serves everything after
  it. The choice is therefore the newest built glue whose version is <= the
  running PyTorch.

  Args:
    running: the installed PyTorch's version suffix (e.g. `2_12_0`), or None if
      it could not be determined.
    built: the version suffixes of the glues bundled in the wheel.

  Returns:
    The suffix of the glue to load, or None when no versioned glues are
    bundled and dispatch should stay disabled.

  Raises:
    ImportError: if versioned glues exist but the installed PyTorch version
      cannot be determined, or predates every built glue.
  """
  if not built:
    # Unversioned build; leave imports to the normal finders.
    return None

  supported = ", ".join(s.replace("_", ".") for s in sorted(built, key=_parse))
  if running is None:
    raise ImportError(
        "torch_tpu could not determine the installed PyTorch version, so it "
        "cannot select a compatible extension module. Extensions are available "
        f"for PyTorch: {supported}."
    )

  candidates = [b for b in built if _parse(b) <= _parse(running)]
  if not candidates:
    raise ImportError(
        "torch_tpu has no extension module compatible with PyTorch "
        f"{running.replace('_', '.')}: it predates the oldest supported "
        f"release. Extensions are available for PyTorch: {supported}."
    )
  return max(candidates, key=_parse)


class ExtensionProxyLoader(importlib.abc.Loader):
  """Loads a version-suffixed glue under its unversioned module name.

  The C++ artifact is renamed to carry a version suffix but still exports
  `PyInit_<unversioned>`. CPython derives the expected init symbol from
  `spec.name`, so we present the unversioned name while the real
  ExtensionFileLoader creates the module.
  """

  def __init__(self, real_loader, unversioned_name, versioned_name):
    self._real_loader = real_loader
    self._unversioned_name = unversioned_name
    self._versioned_name = versioned_name

  def create_module(self, spec):
    if not hasattr(self._real_loader, "create_module"):
      return None
    original_name = spec.name
    # Safe: `spec` is freshly minted per find_spec() call (never shared), and
    # CPython holds the per-module import lock across create_module().
    try:
      spec.name = self._unversioned_name
      return self._real_loader.create_module(spec)
    finally:
      spec.name = original_name

  def exec_module(self, module):
    if hasattr(self._real_loader, "exec_module"):
      self._real_loader.exec_module(module)

  def __getattr__(self, item):
    return getattr(self._real_loader, item)


class VersionDispatchFinder(importlib.abc.MetaPathFinder):
  """Redirects `torch_tpu._internal.<m>` imports to the resolved glue."""

  def find_spec(self, fullname, path, target=None):
    """Redirects a `torch_tpu._internal` import to its versioned artifact.

    Args:
      fullname: the fully qualified name of the module being imported.
      path: the parent package's `__path__`, as supplied by the import system.
      target: an existing module the import updates (only set for reloads);
        forwarded to the underlying PathFinder lookup.

    Returns:
      A spec whose loader creates the module from the version-suffixed
      artifact under its unversioned name, or None to let the normal finders
      handle the import (dispatch disabled, a name outside
      `torch_tpu._internal`, or no versioned artifact for this module).
    """
    if _VERSION_SUFFIX is None or not fullname.startswith(_INTERNAL_PREFIX):
      return None
    # Avoid infinite recursion when we look up the versioned name ourselves.
    if fullname.endswith(f"_{_VERSION_SUFFIX}"):
      return None

    versioned_name = f"{fullname}_{_VERSION_SUFFIX}"
    spec = importlib.machinery.PathFinder.find_spec(
        versioned_name, path, target
    )
    if spec is None:
      # No versioned artifact for this name (e.g. a pure-Python submodule);
      # let the normal finders resolve it.
      return None

    is_pkg = getattr(spec, "submodule_search_locations", None) is not None
    new_spec = importlib.util.spec_from_loader(
        fullname,
        ExtensionProxyLoader(spec.loader, fullname, versioned_name),
        origin=spec.origin,
        is_package=is_pkg,
    )
    new_spec.has_location = getattr(spec, "has_location", False)
    new_spec.submodule_search_locations = getattr(
        spec, "submodule_search_locations", None
    )
    return new_spec


def install_hook(package_dir: pathlib.Path) -> None:
  """Selects the matching glue variant and installs the import hook.

  Args:
    package_dir: the torch_tpu package directory to scan for versioned glues.
  """
  global _VERSION_SUFFIX
  resolved = resolve_suffix(
      torch_version_suffix(),
      discover_built_suffixes(package_dir),
  )
  if resolved is None:
    return
  _VERSION_SUFFIX = resolved
  if not any(isinstance(f, VersionDispatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, VersionDispatchFinder())
