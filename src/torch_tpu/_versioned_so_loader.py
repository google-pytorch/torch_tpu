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
`PathEntryFinder` installed on `sys.path_hooks` at import time rewrites imports
and `pkgutil` module discovery inside `torch_tpu` to the glue that matches the
installed PyTorch, and an `ExtensionProxyLoader` bridges the artifact's
version-suffixed filename to the unversioned `PyInit_<module>` symbol the C++
code actually exports. Version-suffixed glue filenames are hidden from
`pkgutil.iter_modules` and `pkgutil.walk_packages` and cannot be imported
directly by their suffixed names.

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
(`2.14.0.dev20260711` and `2.14.0a0` -> `2.14.0`), so the glue the
nightly-channel wheel bundles -- built against a pinned dev snapshot and named
by that snapshot's release triple, like any other glue -- serves it whenever
present.
"""

from collections.abc import Collection, Iterator, Set
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import pathlib
import sys
import types

from packaging import version as packaging_version


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
  try:
    parsed = packaging_version.Version(version)
  except packaging_version.InvalidVersion:
    return None
  if parsed.epoch != 0 or len(parsed.release) < 3:
    return None
  return "_".join(str(part) for part in parsed.release[:3])


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


def _is_within(entry: str, package_dir: pathlib.Path) -> bool:
  """Returns whether `entry` is `package_dir` or a directory inside it."""
  return pathlib.Path(entry).resolve().is_relative_to(package_dir.resolve())


def _is_package_dir(path: pathlib.Path) -> bool:
  """Returns whether `path` is a package directory (has an `__init__` module)."""
  return (
      path.is_dir()
      and "." not in path.name
      and any(
          inspect.getmodulename(child.name) == "__init__"
          for child in path.iterdir()
      )
  )


def _default_path_entry_finder(path: str) -> importlib.abc.PathEntryFinder:
  """Returns the finder that standard `sys.path_hooks` would create for `path`."""
  for hook in sys.path_hooks:
    if isinstance(hook, _VersionDispatchPathHook):
      continue
    try:
      finder = hook(path)
    except ImportError:
      continue
    if isinstance(finder, importlib.abc.PathEntryFinder):
      return finder
  raise ImportError("no default path entry finder for directory", path=path)


class VersionDispatchPathEntryFinder(importlib.abc.PathEntryFinder):
  """Path entry finder that dispatches unversioned names to the active glue.

  Wraps a standard `FileFinder` by composition (rather than subclassing, since
  `pkgutil.iter_importer_modules` is a `@singledispatch` function registered on
  `FileFinder` that bypasses `iter_modules` overrides on subclasses) for a
  single directory inside `torch_tpu`. Unversioned extension names resolve to
  `<name>_<resolved_suffix>.so` via `ExtensionProxyLoader`, direct imports of
  `<name>_<any_built_suffix>` are hidden, and `iter_modules` yields each
  versioned extension once under its unversioned name.
  """

  def __init__(
      self,
      path: str,
      resolved_suffix: str,
      built_suffixes: Set[str],
  ) -> None:
    self.path = path
    self._dir = pathlib.Path(path)
    self._resolved_suffix = resolved_suffix
    self._built_suffixes = built_suffixes
    self._file_finder = _default_path_entry_finder(path)

  def _is_glue_module(self, modname: str) -> bool:
    """Returns whether `modname` is a versioned glue in `self.path`."""
    suffix = _glue_suffix(f"{modname}.so")
    return (
        suffix is not None
        and suffix in self._built_suffixes
        and (self._dir / f"{modname}.so").is_file()
    )

  def find_spec(
      self,
      fullname: str,
      target: types.ModuleType | None = None,
  ) -> importlib.machinery.ModuleSpec | None:
    tail = fullname.rpartition(".")[2]
    if self._is_glue_module(tail):
      return None

    # Match FileFinder and iter_modules precedence: a package directory
    # (`tail/__init__.*`) wins over a same-named extension module.
    if _is_package_dir(self._dir / tail):
      return self._file_finder.find_spec(fullname, target)

    versioned_tail = f"{tail}_{self._resolved_suffix}"
    if self._is_glue_module(versioned_tail):
      versioned_name = f"{fullname}_{self._resolved_suffix}"
      spec = self._file_finder.find_spec(versioned_name, target)
      if spec is not None and spec.loader is not None:
        is_pkg = spec.submodule_search_locations is not None
        new_spec = importlib.util.spec_from_loader(
            fullname,
            ExtensionProxyLoader(spec.loader, fullname, versioned_name),
            origin=spec.origin,
            is_package=is_pkg,
        )
        if new_spec is not None:
          new_spec.has_location = spec.has_location
          new_spec.submodule_search_locations = spec.submodule_search_locations
        return new_spec

    return self._file_finder.find_spec(fullname, target)

  def _iter_raw_modules(self) -> Iterator[tuple[str, bool]]:
    """Yields `(modname, ispkg)` for entries in `self.path`, like `FileFinder`.

    This mirrors `pkgutil._iter_file_finder_modules` rather than calling
    `pkgutil.iter_importer_modules(self._file_finder)` directly because
    `iter_importer_modules` is undocumented, excluded from `pkgutil.__all__`,
    and absent from typeshed.
    """
    for entry_path in sorted(self._dir.iterdir()):
      fn = entry_path.name
      modname = inspect.getmodulename(fn)
      if modname == "__init__":
        continue

      ispkg = False
      if not modname and _is_package_dir(entry_path):
        modname = fn
        ispkg = True

      if modname and "." not in modname:
        yield modname, ispkg

  def iter_modules(self, prefix: str = "") -> Iterator[tuple[str, bool]]:
    seen: dict[str, bool] = {}
    active_tail = f"_{self._resolved_suffix}"
    for modname, ispkg in self._iter_raw_modules():
      if not ispkg and self._is_glue_module(modname):
        if not modname.endswith(active_tail):
          continue
        modname = modname.removesuffix(active_tail)
      seen.setdefault(modname, ispkg)
    for modname in sorted(seen):
      yield f"{prefix}{modname}", seen[modname]

  def invalidate_caches(self) -> None:
    self._file_finder.invalidate_caches()


class _VersionDispatchPathHook:
  """`sys.path_hooks` callable for directories inside `package_dir`."""

  def __init__(
      self,
      package_dir: pathlib.Path,
      resolved_suffix: str,
      built_suffixes: Set[str],
  ) -> None:
    self.package_dir = package_dir
    self.resolved_suffix = resolved_suffix
    self.built_suffixes = built_suffixes

  def __call__(self, path: str) -> VersionDispatchPathEntryFinder:
    if not pathlib.Path(path).is_dir():
      raise ImportError("only directories are supported", path=path)
    if not _is_within(path, self.package_dir):
      raise ImportError("not inside torch_tpu package", path=path)
    return VersionDispatchPathEntryFinder(
        path,
        self.resolved_suffix,
        self.built_suffixes,
    )


def install_hook(package_dir: pathlib.Path) -> None:
  """Selects the matching glue variant and installs the path import hook.

  Args:
    package_dir: the torch_tpu package directory to scan for versioned glues.
  """
  built_suffixes = discover_built_suffixes(package_dir)
  resolved = resolve_suffix(torch_version_suffix(), built_suffixes)
  if resolved is None:
    return

  package_dir_resolved = package_dir.resolve()
  hook = _VersionDispatchPathHook(package_dir, resolved, built_suffixes)
  sys.path_hooks[:] = [
      existing
      for existing in sys.path_hooks
      if not (
          isinstance(existing, _VersionDispatchPathHook)
          and existing.package_dir.resolve() == package_dir_resolved
      )
  ]
  # Must precede the default FileFinder path hook, which would otherwise claim
  # package_dir first.
  sys.path_hooks.insert(0, hook)

  # Evict any cached FileFinder for package_dir or its subdirectories (for
  # example, torch_tpu/__init__.py imports _tcmalloc_check before calling
  # install_hook, which populates sys.path_importer_cache[str(package_dir)]).
  for cached_path in list(sys.path_importer_cache):
    if _is_within(cached_path, package_dir):
      del sys.path_importer_cache[cached_path]
