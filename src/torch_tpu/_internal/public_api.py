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
hierarchy. `install()` turns `_PUBLIC_MODULES` into real `types.ModuleType`
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

from torch_tpu._internal.utils import annotations


@dataclasses.dataclass(frozen=True)
class Property:
  """A property descriptor installed on the public module's type.

  Attributes:
    target: Dotted module path and colon-separated attribute path resolving to a
      `property` object, e.g.
      `"torch_tpu._internal.device._device_module:_DeviceModuleMeta.allow_excess_precision"`.
  """

  target: str


Export: TypeAlias = str | Property


@dataclasses.dataclass(frozen=True)
class ModuleSpec:
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
_DISTRIBUTED_MODULE: Final[str] = (
    "torch_tpu._internal.distributed._distributed_module"
)
_DISTRIBUTED_PROPERTIES_CLASS: Final[str] = (
    f"{_DISTRIBUTED_MODULE}:DistributedModuleProperties"
)


def _tpu_attr(name: str) -> str:
  return f"{_TPU_DEVICE_CLASS}.{name}"


# Purpose & Format:
# -----------------
# `_PUBLIC_MODULES` defines every public Python module under `torch.tpu` and the
# symbols each module exports:
# - Keys are fully qualified public module paths (`"torch.tpu"`,
#   `"torch.tpu.<submodule>"`, etc.). Every ancestor module must either also
#   appear in `_PUBLIC_MODULES` or already exist in `sys.modules` (e.g.
#   `"torch"`). Child modules are automatically attached as attributes on their
#   parent module and included in the parent's `__all__` / `dir()`.
# - Values are `ModuleSpec(...)` entries whose `exports` dict maps each public
#   symbol name to one of:
#   * `"torch_tpu._internal.pkg.mod:qualname"`: imported lazily on first access
#     and cached in the module's `__dict__` (unless registered in the module's
#     `stages` map, in which case lookup delegates to the target so stage
#     warnings and descriptors run on access).
#   * `Property(target)`: a `property` descriptor installed on a custom
#     `types.ModuleType` subclass so `torch.tpu.<prop> = value` invokes the
#     setter.
#
# Procedure for Adding or Updating Public Modules / APIs:
# -------------------------------------------------------
# 1. Implement the module, class, or function under `torch_tpu._internal/...`
#    and annotate public symbols with a lifecycle stage (`@experimental`,
#    `@deprecated`, etc. from `torch_tpu._internal.utils.annotations`).
# 2. Add or update the corresponding `ModuleSpec` entry in `_PUBLIC_MODULES`
#    below (keeping module keys and `exports` keys sorted). If adding a nested
#    submodule (e.g. `"torch.tpu.foo.bar"`), ensure its parent module
#    (`"torch.tpu.foo"`) also has an entry in `_PUBLIC_MODULES`.
# 3. Add the internal library target to `deps` of `:public_api` in
#    `_internal/BUILD` (with `# build_cleaner: keep`, since imports happen
#    dynamically via string targets) and to `WHEEL_PY_TARGETS` in
#    `build_files/wheel.bzl`.
# 4. Regenerate `tests/api/tt_apis.yaml` (from the artifacts of
#    `tests/api:tt_apis_test`) and add unit tests in
#    `tests/api_registry_test.py`.
#
# TODO(wan): Consider a more local registration mechanism (e.g. a decorator in
# each internal module) so public symbols do not all need to be enumerated in
# this central table.
_PUBLIC_MODULES: Mapping[str, ModuleSpec] = {
    "torch.tpu": ModuleSpec(
        exports={
            # go/keep-sorted start
            "EagerMode": _tpu_attr("EagerMode"),
            "Event": _tpu_attr("Event"),
            "LayoutContext": _tpu_attr("LayoutContext"),
            "Precision": _tpu_attr("Precision"),
            "Stream": _tpu_attr("Stream"),
            "TpuLayout": _tpu_attr("TpuLayout"),
            "__version__": _tpu_attr("__version__"),
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
            "eager_mode": Property(
                f"{_DEVICE_MODULE}:_DeviceModuleMeta.eager_mode"
            ),
            "get_amp_supported_dtype": _tpu_attr("get_amp_supported_dtype"),
            "get_device_layout_if_materialized": _tpu_attr(
                "get_device_layout_if_materialized"
            ),
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
            "set_eager_mode": _tpu_attr("set_eager_mode"),
            "set_rng_state": _tpu_attr("set_rng_state"),
            "set_stream": _tpu_attr("set_stream"),
            "stream": _tpu_attr("stream"),
            "synchronize": _tpu_attr("synchronize"),
            "topology_aware_mesh": _tpu_attr("topology_aware_mesh"),
            # go/keep-sorted end
        },
        private_fallback=_TPU_DEVICE_CLASS,
        stages=f"{_DEVICE_MODULE}:__tt_api_stages__",
        doc="Public TorchTPU device module (`torch.tpu`).",
    ),
    "torch.tpu.compiler": ModuleSpec(
        exports={
            # go/keep-sorted start
            "custom_compiler_options": (
                "torch_tpu._internal.compiler_options:custom_compiler_options"
            ),
            # go/keep-sorted end
        },
        doc="Public compiler API for TorchTPU (`torch.tpu.compiler`).",
    ),
    "torch.tpu.distributed": ModuleSpec(
        exports={
            # go/keep-sorted start
            "set_tpu_launch_env": (
                "torch_tpu._internal.distributed.distributed:set_tpu_launch_env"
            ),
            "spmd_safe": "torch_tpu._internal.distributed.spmd_util:spmd_safe",
            "topology_aware_mesh": f"{_DISTRIBUTED_MODULE}:topology_aware_mesh",
            "window_num_stripes": Property(
                f"{_DISTRIBUTED_PROPERTIES_CLASS}.window_num_stripes"
            ),
            "window_stripe_chunk_mb": Property(
                f"{_DISTRIBUTED_PROPERTIES_CLASS}.window_stripe_chunk_mb"
            ),
            # go/keep-sorted end
        },
        main="torch_tpu._internal.distributed.distributed",
        doc="Public TorchTPU distributed namespace (`torch.tpu.distributed`).",
    ),
    "torch.tpu.ops": ModuleSpec(
        exports={
            # go/keep-sorted start
            "align_jagged_to_multiple": (
                "torch_tpu._internal.ops.jagged:align_jagged_to_multiple"
            ),
            "all_gather_dim": (
                "torch_tpu._internal.ops.collectives:all_gather_dim"
            ),
            "max_pool1d": "torch_tpu._internal.ops.ops:max_pool1d",
            "max_pool1d_backward": (
                "torch_tpu._internal.ops.ops:max_pool1d_backward"
            ),
            "max_pool2d": "torch_tpu._internal.ops.ops:max_pool2d",
            "max_pool2d_backward": (
                "torch_tpu._internal.ops.ops:max_pool2d_backward"
            ),
            "max_pool3d": "torch_tpu._internal.ops.ops:max_pool3d",
            "max_pool3d_backward": (
                "torch_tpu._internal.ops.ops:max_pool3d_backward"
            ),
            "optimization_barrier": (
                "torch_tpu._internal.ops.ops:optimization_barrier"
            ),
            "ragged_all_to_all": (
                "torch_tpu._internal.ops.ops:ragged_all_to_all"
            ),
            "ragged_dot": "torch_tpu._internal.ops.ops:ragged_dot",
            "ragged_dot_weight_grad": (
                "torch_tpu._internal.ops.ops:ragged_dot_weight_grad"
            ),
            "recv": "torch_tpu._internal.ops.ops:recv",
            "reduce_scatter_dim": (
                "torch_tpu._internal.ops.collectives:reduce_scatter_dim"
            ),
            "send": "torch_tpu._internal.ops.ops:send",
            "sparse_dense_matmul": (
                "torch_tpu._internal.ops.ops:sparse_dense_matmul"
            ),
            "sparse_dense_matmul_activation_unstack": (
                "torch_tpu._internal.ops.ops:sparse_dense_matmul_activation_unstack"
            ),
            "sparse_dense_matmul_grad_with_adagrad": (
                "torch_tpu._internal.ops.ops:sparse_dense_matmul_grad_with_adagrad"
            ),
            "sparse_dense_matmul_grad_with_adam": (
                "torch_tpu._internal.ops.ops:sparse_dense_matmul_grad_with_adam"
            ),
            "sparse_dense_matmul_grad_with_sgd": (
                "torch_tpu._internal.ops.ops:sparse_dense_matmul_grad_with_sgd"
            ),
            "sparse_dense_matmul_gradient_stack": (
                "torch_tpu._internal.ops.ops:sparse_dense_matmul_gradient_stack"
            ),
            "sparse_gather": "torch_tpu._internal.ops.ops:sparse_gather",
            "sparse_gather_backward": (
                "torch_tpu._internal.ops.ops:sparse_gather_backward"
            ),
            "sparse_iota": "torch_tpu._internal.ops.ops:sparse_iota",
            "sparse_map_row": "torch_tpu._internal.ops.ops:sparse_map_row",
            "splash_sdpa": (
                "torch_tpu._internal.ops.splash_attention:splash_sdpa"
            ),
            "strip_jagged_padding": (
                "torch_tpu._internal.ops.jagged:strip_jagged_padding"
            ),
            # go/keep-sorted end
        },
        doc="Public custom ops API for TorchTPU (`torch.tpu.ops`).",
    ),
    "torch.tpu.optimizers": ModuleSpec(
        exports={
            # go/keep-sorted start
            "AdamW": "torch_tpu._internal.optimizers.adamw:AdamW",
            "Muon": "torch_tpu._internal.optimizers.muon:Muon",
            "Optimizer": "torch_tpu._internal.optimizers.optim:Optimizer",
            "Sgd": "torch_tpu._internal.optimizers.sgd:Sgd",
            "SingleTraceTrainer": (
                "torch_tpu._internal.optimizers.single_trace_trainer:SingleTraceTrainer"
            ),
            # go/keep-sorted end
        },
        doc="Public optimizers API for TorchTPU (`torch.tpu.optimizers`).",
    ),
    "torch.tpu.pallas": ModuleSpec(
        exports={
            # go/keep-sorted start
            "get_pallas_mesh": (
                "torch_tpu._internal.pallas.mesh:get_pallas_mesh"
            ),
            "jax_op": "torch_tpu._internal.pallas.pallas:jax_op",
            # go/keep-sorted end
        },
        doc="Public Pallas API for TorchTPU (`torch.tpu.pallas`).",
    ),
    "torch.tpu.profiler": ModuleSpec(
        exports={
            # go/keep-sorted start
            "TpuProfilerConfig": (
                "torch_tpu._internal.profiler.profiler_config:TpuProfilerConfig"
            ),
            # go/keep-sorted end
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


def _is_dunder(name: str) -> bool:
  """Returns whether `name` is a dunder name, e.g. `__version__`."""
  return name.startswith("__") and name.endswith("__")


class _PublicApiModule(types.ModuleType):
  """Lazy `torch.tpu` module backed by a `ModuleSpec`."""

  def __init__(
      self,
      module_name: str,
      spec: ModuleSpec,
      child_names: AbstractSet[str],
      attr_exports: Mapping[str, str],
  ) -> None:
    super().__init__(module_name, spec.doc)
    self._module_name = module_name
    self._spec = spec
    self._child_names = frozenset(child_names)
    self._attr_exports = dict(attr_exports)
    # Like `torch.__all__`, `__all__` doesn't list dunder exports such as
    # `__version__`, and `dir()` lists the same names as `__all__`.
    self._all_names = tuple(
        sorted(
            {name for name in spec.exports if not _is_dunder(name)}
            | self._child_names
        )
    )
    self.__all__ = list(self._all_names)

  def _get_stages_map(self) -> Mapping[str, annotations.ApiStageInfo]:
    """Lazily builds and caches `__tt_api_stages__` on this module."""
    if "__tt_api_stages__" in self.__dict__:
      return self.__dict__["__tt_api_stages__"]
    stages_map: Mapping[str, annotations.ApiStageInfo] = (
        {} if self._spec.stages is None else _import_target(self._spec.stages)
    )
    setattr(self, "__tt_api_stages__", stages_map)
    return stages_map

  def __getattr__(self, name: str) -> Any:
    if name == "__tt_api_stages__" and self._spec.stages is not None:
      return self._get_stages_map()

    target = self._attr_exports.get(name)
    if target is not None:
      val = _import_target(target)
      if self._spec.stages is None or name not in self._get_stages_map():
        setattr(self, name, val)
      return val

    if name in self._child_names:
      child_mod = importlib.import_module(f"{self._module_name}.{name}")
      setattr(self, name, child_mod)
      return child_mod

    fallback_target = self._spec.private_fallback
    if (
        fallback_target is not None
        and name.startswith("_")
        and not _is_dunder(name)
    ):
      fb_obj = _import_target(fallback_target)
      try:
        return getattr(fb_obj, name)
      except AttributeError:
        pass

    raise AttributeError(
        f"module '{self._module_name}' has no attribute '{name}'"
    )

  def __dir__(self) -> Sequence[str]:
    return self._all_names


class _PublicApiLoader(importlib.abc.Loader):
  """Loader that creates and configures a `_PublicApiModule` from a `ModuleSpec`."""

  def __init__(
      self,
      module_name: str,
      spec: ModuleSpec,
      child_names: AbstractSet[str],
      is_package: bool,
  ) -> None:
    self._module_name = module_name
    self._spec = spec
    self._child_names = frozenset(child_names)
    self._is_package = is_package

  def is_package(self, fullname: str) -> bool:
    if fullname != self._module_name:
      raise ImportError(
          f"loader for {self._module_name} cannot handle {fullname}",
          name=fullname,
      )
    return self._is_package

  def create_module(
      self, spec: importlib.machinery.ModuleSpec
  ) -> _PublicApiModule:
    del spec
    attr_exports: dict[str, str] = {}
    prop_exports: dict[str, property] = {}
    for export_name, export_spec in self._spec.exports.items():
      if isinstance(export_spec, str):
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

    # `property` descriptors only intercept attribute access and assignment on a
    # module instance when defined on `type(module)`. When `prop_exports` is
    # non-empty, instantiate a dedicated `_PublicApiModule` subclass carrying
    # those descriptors.
    module_cls: type[_PublicApiModule] = _PublicApiModule
    if prop_exports:
      cls_name = f"_{self._module_name.replace('.', '_')}_Module"
      module_cls = type(cls_name, (_PublicApiModule,), prop_exports)

    return module_cls(
        self._module_name,
        self._spec,
        self._child_names,
        attr_exports,
    )

  def exec_module(self, module: types.ModuleType) -> None:
    for child_name in self._child_names:
      if (
          child_mod := sys.modules.get(f"{self._module_name}.{child_name}")
      ) is not None:
        setattr(module, child_name, child_mod)


class _ExecutablePublicApiLoader(
    _PublicApiLoader, importlib.abc.ExecutionLoader
):
  """Loader that also delegates `python -m` execution to an internal module."""

  def __init__(
      self,
      module_name: str,
      spec: ModuleSpec,
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


def install(table: Mapping[str, ModuleSpec] = _PUBLIC_MODULES) -> None:
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
  # `torch.tpu` attached to `torch`). Unloaded child submodules are resolved on
  # demand by `_FINDER` upon `import` or parent attribute access, avoiding
  # premature `sys.modules` population that triggers `runpy` RuntimeWarnings on
  # `python -m`.
  for module_name in sorted(table, key=lambda n: (n.count("."), n)):
    parent_name, sep, leaf = module_name.rpartition(".")
    is_root = not sep or parent_name not in table
    if is_root and module_name not in sys.modules:
      spec = _FINDER.find_spec(module_name)
      assert spec is not None and spec.loader is not None
      mod = importlib.util.module_from_spec(spec)
      sys.modules[module_name] = mod
      spec.loader.exec_module(mod)
      if sep and parent_name in sys.modules:
        setattr(sys.modules[parent_name], leaf, mod)
