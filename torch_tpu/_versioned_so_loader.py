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

"""Dynamically dispatches import requests to versioned C++ extensions."""

import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import sys

_VERSION_SUFFIX = ""


class ExtensionProxyLoader(importlib.abc.Loader):
  """Wraps ExtensionFileLoader to cleanly handle the PyInit_ mismatch."""

  def __init__(self, real_loader, unversioned_name, versioned_name):
    self.real_loader = real_loader
    self.unversioned_name = unversioned_name
    self.versioned_name = versioned_name

  def create_module(self, spec):
    if not hasattr(self.real_loader, "create_module"):
      return None

    original_name = spec.name
    try:
      # The C-code exported PyInit_unversioned_name because
      # we just rename the artifact to include the version suffix.
      spec.name = self.unversioned_name
      return self.real_loader.create_module(spec)
    finally:
      spec.name = original_name

  def exec_module(self, module):
    if hasattr(self.real_loader, "exec_module"):
      self.real_loader.exec_module(module)

  def __getattr__(self, item):
    return getattr(self.real_loader, item)


class VersionDispatchFinder(importlib.abc.MetaPathFinder):
  """Finder that delegates import requests to versioned extensions."""

  def find_spec(self, fullname, path, target=None):
    """Finds a spec for the specified module."""
    if fullname.startswith("torch_tpu._internal."):
      # Ensure we don't infinitely recurse if searching for the
      # versioned name itself.
      if _VERSION_SUFFIX and fullname.endswith(_VERSION_SUFFIX):
        return None

      versioned_name = f"{fullname}_{_VERSION_SUFFIX}"

      # Ask the standard PathFinder to search for the versioned module.
      spec = importlib.machinery.PathFinder.find_spec(
          versioned_name, path, target
      )

      if spec is not None:
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

    return None


def install_hook(version_suffix):
  """Installs the import hook to resolve _internal modules."""
  global _VERSION_SUFFIX
  _VERSION_SUFFIX = version_suffix

  if not any(isinstance(f, VersionDispatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, VersionDispatchFinder())
