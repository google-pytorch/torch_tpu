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

torch_tpu bundles a thin ``.so`` glue per supported PyTorch build, each named
``<module>_<major>_<minor>_<patch>.so`` (e.g. ``env_2_12_0.so``). A
``MetaPathFinder`` installed at import time rewrites imports of
``torch_tpu._internal.<module>`` to the glue that matches the installed PyTorch,
and an ``ExtensionProxyLoader`` bridges the artifact's version-suffixed filename
to the unversioned ``PyInit_<module>`` symbol the C++ code actually exports.

Selection policy: there is no ABI-stability guarantee across PyTorch releases --
even patch releases -- so an exact ``major.minor.patch`` glue is preferred. If
none was built for the installed PyTorch, the newest glue we shipped is used as
a best effort (with a warning); loading it eagerly binds its symbols, so an
ABI-incompatible glue fails loudly at import rather than crashing later. If the
installed PyTorch version cannot be determined while versioned glues exist, that
is an error. When no versioned glues are present at all (e.g. an unversioned
developer build) the hook is not installed and imports resolve normally.
"""

import collections.abc
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import pathlib
import re
import sys
import warnings

_INTERNAL_PREFIX = "torch_tpu._internal."

# Matches a versioned extension artifact, capturing the "<major>_<minor>_<patch>"
# suffix, e.g. "env_2_12_0.so" -> "2_12_0".
_VERSIONED_SO = re.compile(r"_(\d+_\d+_\d+)\.so$")

# Resolved once by install_hook(): the suffix of the glue variant this process
# loads, or None when dispatch is disabled.
_VERSION_SUFFIX: str | None = None


def _parse(suffix: str) -> tuple[int, ...]:
  return tuple(int(part) for part in suffix.split("_"))


def torch_version_suffix() -> str | None:
  """Returns the installed torch's "<major>_<minor>_<patch>", or None."""
  try:
    version = importlib.metadata.version("torch")
  except importlib.metadata.PackageNotFoundError:
    return None
  match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
  if match is None:
    return None
  return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"


def discover_built_suffixes(
    package_dir: pathlib.Path,
) -> frozenset[str]:
  """Returns the set of glue suffixes bundled under ``package_dir``."""
  suffixes = set()
  for so in package_dir.rglob("*.so"):
    match = _VERSIONED_SO.search(so.name)
    if match is not None:
      suffixes.add(match.group(1))
  return frozenset(suffixes)


def resolve_suffix(
    running: str | None,
    built: collections.abc.Collection[str],
) -> str | None:
  """Chooses the glue suffix to load, or None to disable dispatch.

  Prefers an exact ``major.minor.patch`` match, then the newest patch built for
  the same ``(major, minor)`` -- PyTorch does not guarantee ABI stability across
  patch releases, but in practice it holds within a minor -- then, as a last
  resort for an unsupported minor, the newest glue built at all.

  Raises:
    ImportError: if versioned glues exist but the installed PyTorch version
      cannot be determined.
  """
  if not built:
    # Unversioned build; leave imports to the normal finders.
    return None
  if running is not None and running in built:
    return running

  supported = ", ".join(s.replace("_", ".") for s in sorted(built, key=_parse))
  if running is None:
    raise ImportError(
        "torch_tpu could not determine the installed PyTorch version, so it "
        "cannot select a compatible extension module. Extensions are available "
        f"for PyTorch: {supported}."
    )

  running_minor = _parse(running)[:2]
  same_minor = [b for b in built if _parse(b)[:2] == running_minor]
  if same_minor:
    chosen = max(same_minor, key=_parse)
    reason = "the same minor"
  else:
    chosen = max(built, key=_parse)
    reason = "the newest available"
  warnings.warn(
      f"torch_tpu has no extension module built for PyTorch "
      f"{running.replace('_', '.')}; falling back to {reason} build "
      f"({chosen.replace('_', '.')}). This may fail to load or misbehave if the "
      f"PyTorch ABI has changed. Built for: {supported}.",
      RuntimeWarning,
      stacklevel=3,
  )
  return chosen


class ExtensionProxyLoader(importlib.abc.Loader):
  """Loads a version-suffixed glue under its unversioned module name.

  The C++ artifact is renamed to carry a version suffix but still exports
  ``PyInit_<unversioned>``. CPython derives the expected init symbol from
  ``spec.name``, so we present the unversioned name while the real
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
    # Safe: ``spec`` is freshly minted per find_spec() call (never shared), and
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
  """Redirects ``torch_tpu._internal.<m>`` imports to the resolved glue."""

  def find_spec(self, fullname, path, target=None):
    if _VERSION_SUFFIX is None or not fullname.startswith(_INTERNAL_PREFIX):
      return None
    # Avoid infinite recursion when we look up the versioned name ourselves.
    if fullname.endswith(f"_{_VERSION_SUFFIX}"):
      return None

    versioned_name = f"{fullname}_{_VERSION_SUFFIX}"
    spec = importlib.machinery.PathFinder.find_spec(versioned_name, path, target)
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
      torch_version_suffix(), discover_built_suffixes(package_dir)
  )
  if resolved is None:
    return
  _VERSION_SUFFIX = resolved
  if not any(isinstance(f, VersionDispatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, VersionDispatchFinder())
