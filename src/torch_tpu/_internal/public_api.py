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

"""Declarative table and lazy module builder for TorchTPU's `torch.tpu` APIs.

This module is the single source of truth for the `torch.tpu` public module
hierarchy. `install()` turns `PUBLIC_MODULES` into real `types.ModuleType`
instances registered in `sys.modules` and attached to their parent modules,
resolving exported attributes lazily from `torch_tpu._internal` on first access
(PEP 562) so `import torch` stays fast and internal imports never leak onto the
public namespace.
"""

from collections.abc import Mapping, Sequence, Set as AbstractSet
import dataclasses
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from typing import Any, Final, TypeAlias


@dataclasses.dataclass(frozen=True)
class Attr:
  """An attribute exported by a public module.

  Attributes:
    target: Dotted module path and colon-separated attribute path, e.g.
      `"torch_tpu._internal.profiler.profiler_config:TpuProfilerConfig"`.
    cache: Whether to store the resolved value in the module's `__dict__` after
      first access. Set to `False` for attributes whose lookup must run on every
      access (such as descriptors or methods that tests patch on a class).
  """

  target: str
  cache: bool = True


@dataclasses.dataclass(frozen=True)
class Property:
  """A property descriptor installed on the public module's type.

  Attributes:
    target: Dotted module path and colon-separated attribute path resolving to a
      `property` object, e.g.
      `"torch_tpu._internal.device._device_module:_DeviceModuleMeta.window_num_stripes"`.
  """

  target: str


Export: TypeAlias = str | Attr | Property


@dataclasses.dataclass(frozen=True)
class Module:
  """Specification of a public module.

  Attributes:
    exports: Mapping from public attribute name to its `Export` specification.
    private_fallback: Optional `pkg.mod:Qual.name` target whose
      single-underscore (`_x`, non-dunder) attributes are served live when not
      in `exports`.
    stages: Optional `pkg.mod:attr` target resolving to a `__tt_api_stages__`
      dict attached to the module for API scanner inspection.
    main: Optional internal module name to delegate `python -m <public_module>`
      execution to via a lazy loader on `__spec__`.
    doc: Optional module docstring (`__doc__`).
  """

  exports: Mapping[str, Export] = dataclasses.field(default_factory=dict)
  private_fallback: str | None = None
  stages: str | None = None
  main: str | None = None
  doc: str | None = None


_DEVICE_MODULE: Final[str] = "torch_tpu._internal.device._device_module"
_TPU_DEVICE_CLASS: Final[str] = f"{_DEVICE_MODULE}:TpuDeviceModule"


def _tpu_attr(name: str) -> Attr:
  return Attr(f"{_TPU_DEVICE_CLASS}.{name}", cache=False)


# Purpose & Format:
# -----------------
# `PUBLIC_MODULES` defines every public Python module under `torch.tpu` and the
# symbols each module exports:
# - Keys are fully qualified public module paths (`"torch.tpu"`,
#   `"torch.tpu.<submodule>"`, etc.). Every ancestor module must either also
#   appear in `PUBLIC_MODULES` or already exist in `sys.modules` (e.g.
#   `"torch"`). Child modules are automatically attached as attributes on their
#   parent module and included in the parent's `__all__` / `dir()`.
# - Values are `Module(...)` entries whose `exports` dict maps each public
#   symbol name to one of:
#   * `"torch_tpu._internal.pkg.mod:qualname"` (shorthand for
#     `Attr(target, cache=True)`): imported lazily on first access and cached in
#     the module's `__dict__`.
#   * `Attr(target, cache=False)`: resolved lazily on every access without
#     caching in the module's `__dict__` (used for `TpuDeviceModule` methods and
#     descriptors).
#   * `Property(target)`: a `property` descriptor installed on a custom
#     `types.ModuleType` subclass so `torch.tpu.<prop> = value` invokes the
#     setter.
#
# Procedure for Adding or Updating Public Modules / APIs:
# -------------------------------------------------------
# 1. Implement the module, class, or function under `torch_tpu._internal/...`
#    and annotate public symbols with a lifecycle stage (`@experimental`,
#    `@deprecated`, etc. from `torch_tpu._internal.utils.annotations`).
# 2. Add or update the corresponding `Module` entry in `PUBLIC_MODULES` below
#    (keeping module keys and `exports` keys sorted). If adding a nested
#    submodule (e.g. `"torch.tpu.foo.bar"`), ensure its parent module
#    (`"torch.tpu.foo"`) also has an entry in `PUBLIC_MODULES`.
# 3. Add the internal library target to `deps` of `:public_api` in
#    `_internal/BUILD` (with `# build_cleaner: keep`, since imports happen
#    dynamically via string targets) and to `WHEEL_PY_TARGETS` in
#    `build_files/wheel.bzl`.
# 4. Regenerate `google/tools/tt_apis.yaml` (via `google/tools:scan_tt_apis`)
#    and add unit tests in `tests/api_registry_test.py`.
PUBLIC_MODULES: Mapping[str, Module] = {
    "torch.tpu": Module(
        exports={
            # go/keep-sorted start
            "Event": _tpu_attr("Event"),
            "Precision": _tpu_attr("Precision"),
            "Stream": _tpu_attr("Stream"),
            "allow_excess_precision": Property(
                f"{_DEVICE_MODULE}:_DeviceModuleMeta.allow_excess_precision"
            ),
            "current_device": _tpu_attr("current_device"),
            "current_stream": _tpu_attr("current_stream"),
            "default_generators": _tpu_attr("default_generators"),
            "default_stream": _tpu_attr("default_stream"),
            "device": _tpu_attr("device"),
            "device_count": _tpu_attr("device_count"),
            "device_of": _tpu_attr("device_of"),
            "get_amp_supported_dtype": _tpu_attr("get_amp_supported_dtype"),
            "get_device_name": _tpu_attr("get_device_name"),
            "get_device_properties": _tpu_attr("get_device_properties"),
            "get_rng_state": _tpu_attr("get_rng_state"),
            "initial_seed": _tpu_attr("initial_seed"),
            "is_available": _tpu_attr("is_available"),
            "is_initialized": _tpu_attr("is_initialized"),
            "manual_seed": _tpu_attr("manual_seed"),
            "manual_seed_all": _tpu_attr("manual_seed_all"),
            "precision": _tpu_attr("precision"),
            "set_device": _tpu_attr("set_device"),
            "set_rng_state": _tpu_attr("set_rng_state"),
            "stream": _tpu_attr("stream"),
            "synchronize": _tpu_attr("synchronize"),
            "topology_aware_mesh": _tpu_attr("topology_aware_mesh"),
            "window_num_stripes": Property(
                f"{_DEVICE_MODULE}:_DeviceModuleMeta.window_num_stripes"
            ),
            "window_stripe_chunk_mb": Property(
                f"{_DEVICE_MODULE}:_DeviceModuleMeta.window_stripe_chunk_mb"
            ),
            # go/keep-sorted end
        },
        private_fallback=_TPU_DEVICE_CLASS,
        stages=f"{_DEVICE_MODULE}:__tt_api_stages__",
        doc="Public TorchTPU device module (`torch.tpu`).",
    ),
    "torch.tpu.distributed": Module(
        doc="Public TorchTPU distributed namespace (`torch.tpu.distributed`).",
    ),
    "torch.tpu.distributed.environment": Module(
        exports={
            "set_tpu_launch_env": (
                "torch_tpu._internal.distributed.environment:set_tpu_launch_env"
            ),
        },
        main="torch_tpu._internal.distributed.environment",
        doc="Public TorchTPU distributed environment utilities.",
    ),
    "torch.tpu.pallas": Module(
        exports={
            "jax_op": "torch_tpu._internal.pallas.pallas:jax_op",
        },
        doc="Public Pallas API for TorchTPU (`torch.tpu.pallas`).",
    ),
    "torch.tpu.profiler": Module(
        exports={
            "TpuProfilerConfig": (
                "torch_tpu._internal.profiler.profiler_config:TpuProfilerConfig"
            ),
        },
        doc="Public profiler API for TorchTPU (`torch.tpu.profiler`).",
    ),
}


def _import_target(target: str) -> Any:
  """Imports and resolves a `pkg.mod:Qual.name` target."""
  module_path, qualname = target.split(":", 1)
  obj: Any = importlib.import_module(module_path)
  for attr in qualname.split("."):
    obj = getattr(obj, attr)
  return obj


class _PublicApiLoader(importlib.abc.Loader):
  """Loader that configures a virtual `torch.tpu` module from a `Module` spec."""

  def __init__(
      self,
      module_name: str,
      spec: Module,
      child_names: AbstractSet[str],
      is_package: bool,
  ) -> None:
    self._module_name = module_name
    self._spec = spec
    self._child_names = frozenset(child_names)
    self._is_package = is_package

  def create_module(
      self, spec: importlib.machinery.ModuleSpec
  ) -> types.ModuleType | None:
    existing = sys.modules.get(spec.name)
    if isinstance(existing, types.ModuleType):
      # Reuse the existing module object in-place so prior `from torch import tpu`
      # bindings stay valid across repeated `install()` calls. Reset standard
      # PEP 451 attributes so `module_from_spec` re-initializes them cleanly.
      existing.__spec__ = None
      existing.__loader__ = None
      existing.__package__ = None
      existing.__dict__.pop("__path__", None)
      return existing
    return None

  def is_package(self, fullname: str) -> bool:
    if fullname != self._module_name:
      raise ImportError(
          f"loader for {self._module_name} cannot handle {fullname}",
          name=fullname,
      )
    return self._is_package

  def exec_module(self, module: types.ModuleType) -> None:
    attr_exports: dict[str, Attr] = {}
    prop_exports: dict[str, property] = {}
    for export_name, export_spec in self._spec.exports.items():
      if isinstance(export_spec, str):
        attr_exports[export_name] = Attr(export_spec)
      elif isinstance(export_spec, Attr):
        attr_exports[export_name] = export_spec
      elif isinstance(export_spec, Property):
        prop = _import_target(export_spec.target)
        if not isinstance(prop, property):
          raise TypeError(
              f"Property target {export_spec.target!r} for"
              f" {self._module_name}.{export_name} resolved to {type(prop)!r},"
              " expected property."
          )
        prop_exports[export_name] = prop
      else:
        raise TypeError(f"Unsupported export spec: {export_spec!r}")

    if self._spec.stages is not None:
      attr_exports["__tt_api_stages__"] = Attr(self._spec.stages, cache=True)

    # Evict any attributes cached in `module.__dict__` by a prior `install()` or
    # `importlib.reload()` so they do not shadow the new `__getattr__`.
    for stale_key in getattr(module, "_tt_cached_attrs", ()):
      module.__dict__.pop(stale_key, None)
    cached_attrs: set[str] = set(self._child_names)
    setattr(module, "_tt_cached_attrs", cached_attrs)

    # `property` descriptors only intercept attribute access and assignment on a
    # module instance when defined on `type(module)`. Because `types.ModuleType`
    # itself cannot be mutated, assign a dedicated `ModuleType` subclass when
    # `prop_exports` is non-empty, or reset `__class__` back to `ModuleType`.
    if prop_exports:
      cls_name = f"_{self._module_name.replace('.', '_')}_Module"
      module.__class__ = type(cls_name, (types.ModuleType,), prop_exports)
    elif type(module) is not types.ModuleType:
      module.__class__ = types.ModuleType

    all_names = tuple(
        sorted(set(self._spec.exports.keys()) | self._child_names)
    )
    module.__doc__ = self._spec.doc
    setattr(module, "__all__", list(all_names))

    module_name = self._module_name
    child_names = self._child_names
    fallback_target = self._spec.private_fallback

    def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
      attr_spec = attr_exports.get(name)
      if attr_spec is not None:
        val = _import_target(attr_spec.target)
        if attr_spec.cache:
          setattr(module, name, val)
          cached_attrs.add(name)
        return val

      if name in child_names:
        child_mod = importlib.import_module(f"{module_name}.{name}")
        setattr(module, name, child_mod)
        cached_attrs.add(name)
        return child_mod

      if (
          fallback_target is not None
          and name.startswith("_")
          and not (name.startswith("__") and name.endswith("__"))
      ):
        fb_obj = _import_target(fallback_target)
        try:
          return getattr(fb_obj, name)
        except AttributeError:
          pass

      raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

    def __dir__() -> Sequence[str]:  # pylint: disable=invalid-name
      return all_names

    setattr(module, "__getattr__", __getattr__)
    setattr(module, "__dir__", __dir__)


class _ExecutablePublicApiLoader(
    _PublicApiLoader, importlib.abc.ExecutionLoader
):
  """Loader that also delegates `python -m` execution to an internal module."""

  def __init__(
      self,
      module_name: str,
      spec: Module,
      child_names: AbstractSet[str],
      is_package: bool,
      target_module: str,
  ) -> None:
    super().__init__(module_name, spec, child_names, is_package)
    self._target_module = target_module

  def _find_target_loader(self) -> importlib.abc.Loader:
    target_spec = importlib.util.find_spec(self._target_module)
    if target_spec is None or target_spec.loader is None:
      raise ImportError(
          f"Cannot find target module {self._target_module!r} for"
          f" {self._module_name!r}",
          name=self._module_name,
      )
    return target_spec.loader

  def get_filename(self, fullname: str) -> str:
    del fullname
    loader = self._find_target_loader()
    if not hasattr(loader, "get_filename"):
      raise ImportError(
          f"Loader for {self._target_module!r} does not support get_filename()",
          name=self._module_name,
      )
    # Pass `self._target_module` rather than `fullname`: stdlib `FileLoader`
    # validates that the requested name matches the loader's own `.name`.
    return loader.get_filename(self._target_module)

  def get_source(self, fullname: str) -> str | None:
    del fullname
    loader = self._find_target_loader()
    if not hasattr(loader, "get_source"):
      return None
    return loader.get_source(self._target_module)

  def get_code(self, fullname: str) -> types.CodeType | None:
    del fullname
    loader = self._find_target_loader()
    if not hasattr(loader, "get_code"):
      return None
    return loader.get_code(self._target_module)


class _PublicApiFinder(importlib.abc.MetaPathFinder):
  """MetaPathFinder serving virtual public modules registered via `install()`."""

  def __init__(self) -> None:
    self._loaders: dict[str, _PublicApiLoader] = {}

  def register(self, loaders: Mapping[str, _PublicApiLoader]) -> None:
    self._loaders.update(loaders)

  def find_spec(
      self,
      fullname: str,
      path: Sequence[str] | None = None,
      target: types.ModuleType | None = None,
  ) -> importlib.machinery.ModuleSpec | None:
    del path, target
    loader = self._loaders.get(fullname)
    if loader is None:
      return None
    is_pkg = loader.is_package(fullname)
    if isinstance(loader, _ExecutablePublicApiLoader):
      target_spec = importlib.util.find_spec(loader._target_module)
      origin = target_spec.origin if target_spec is not None else None
      spec = importlib.util.spec_from_loader(
          fullname, loader, origin=origin, is_package=is_pkg
      )
      if target_spec is not None and spec is not None:
        spec.has_location = target_spec.has_location
        spec.cached = target_spec.cached
      return spec
    return importlib.util.spec_from_loader(
        fullname, loader, origin=None, is_package=is_pkg
    )


_FINDER: Final[_PublicApiFinder] = _PublicApiFinder()


def _load_or_update_module(module_name: str) -> types.ModuleType:
  """Materializes or updates `module_name` via `_FINDER` and binds it to its parent."""
  spec = _FINDER.find_spec(module_name, None)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = mod
  spec.loader.exec_module(mod)
  parent_name, sep, leaf = module_name.rpartition(".")
  if sep and parent_name in sys.modules:
    parent_mod = sys.modules[parent_name]
    if not hasattr(parent_mod, "__path__"):
      setattr(parent_mod, "__path__", [])
    setattr(parent_mod, leaf, mod)
  return mod


def install(table: Mapping[str, Module] = PUBLIC_MODULES) -> None:
  """Registers and initializes the public modules defined in `table`."""
  children_by_parent: dict[str, set[str]] = {name: set() for name in table}
  package_names: set[str] = set()
  for name in table:
    parent, sep, leaf = name.rpartition(".")
    if sep:
      if parent in children_by_parent:
        children_by_parent[parent].add(leaf)
        package_names.add(parent)
      elif parent not in sys.modules:
        raise ValueError(
            f"Parent module {parent!r} for {name!r} is neither in `table` nor"
            " in `sys.modules`."
        )

  loaders: dict[str, _PublicApiLoader] = {}
  for module_name, mod_spec in table.items():
    is_pkg = module_name in package_names
    children = children_by_parent[module_name]
    if mod_spec.main is not None:
      loaders[module_name] = _ExecutablePublicApiLoader(
          module_name, mod_spec, children, is_pkg, mod_spec.main
      )
    else:
      loaders[module_name] = _PublicApiLoader(
          module_name, mod_spec, children, is_pkg
      )

  _FINDER.register(loaders)
  if _FINDER not in sys.meta_path:
    sys.meta_path.insert(0, _FINDER)

  # Only eagerly materialize root entries (whose parent is outside `table`, e.g.
  # `torch.tpu` attached to `torch`) and any submodules already present in
  # `sys.modules`. Unloaded child submodules are resolved on demand by `_FINDER`
  # upon `import` or parent attribute access, avoiding premature `sys.modules`
  # population that triggers `runpy` RuntimeWarnings on `python -m`.
  for module_name in sorted(table, key=lambda n: (n.count("."), n)):
    parent_name, sep, _ = module_name.rpartition(".")
    is_root = not sep or parent_name not in table
    if is_root or module_name in sys.modules:
      _load_or_update_module(module_name)
