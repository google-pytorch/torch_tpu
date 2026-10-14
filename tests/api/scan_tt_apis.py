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

r"""TorchTPU API Scanner.

This tool inventories all TorchTPU API surfaces and exports the results to
`tt_apis.yaml`. It also updates the user-facing lists of stable, experimental,
deprecated and internal APIs in `docs/markdown/public_apis.md`.

It uses a dual-strategy approach to discover and categorize APIs:

1. Runtime Memory Reflection:
   - Public Python APIs: Imports the public submodules of `torch_tpu`, then
     dynamically inspects runtime memory of loaded modules (`torch.tpu` and
     `torch_tpu`) for exported public symbols,
     recursing into submodules, classes, and factory return types. Every public
     path to a TorchTPU object is reported, but the members of a TorchTPU module
     or class are listed only at its home (where it's defined), and any other
     path to it gets a single entry. Modules, classes and other callables that
     a module imports from other libraries are skipped. The `__version__` of
     each TorchTPU module walked is public too, and its stage must come from
     the module's `__tt_api_stages__` map.
   - Accelerator Python APIs: Dynamically inspects `torch.accelerator` (and its
     public submodules) and probes each public API at runtime on TPU to include
     only APIs implemented for TPU.
   - ABSL Python Flags: Inspects `absl.flags.FLAGS` for live process flags that
     were defined within `torch_tpu` Python files.

2. Source Code Scanning:
   - Environment Variables: Parses C++ headers (e.g. `common/env_vars.h`) using
     regex for `inline constexpr char` declarations and resolves stages from
     `kEnvVarToStage`.
   - ABSL C++ Flags: Parses C++ source and header files for `ABSL_FLAG` macros.
   - Custom Ops: Inspects C++ registration blocks for custom op stages,
     skipping underscore-prefixed (internal) custom ops.

`tt_apis_test` compares the results with the checked-in files and exports the
up-to-date files as Sponge test artifacts.

The source code scan and both files are relative to the TorchTPU repository
root, which `repo_paths.get_repo_root_in_runfiles` locates in the runfiles.

TO UPDATE BASELINE (tt_apis.yaml and public_apis.md), `blaze run` the
`tests/api:scan_tt_apis` target of TorchTPU on a TPU host. It writes both files
directly into your workspace. Alternatively, copy the files from the "Artifacts"
tab of the Sponge link of `tt_apis_test`.
"""

import ast
import collections
import dataclasses
import enum
import importlib
import inspect
import os
import pkgutil
import re
import sys
import textwrap
import types
import typing
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import warnings

from absl import app
from absl import flags
from absl import logging
import torch
import torch_tpu
from torch_tpu._internal.utils import annotations
from tests import repo_paths
import yaml

Stage = annotations.Stage

# The label of `tt_apis_test`, whose Sponge artifacts are the up-to-date
# `tt_apis.yaml` and `public_apis.md`.
TT_APIS_TEST_LABEL = (
    f"//{repo_paths.G3_REPO_PATH}/{repo_paths.SCANNER_PACKAGE}:tt_apis_test"
)

# The label of the binary that updates `tt_apis.yaml` and `public_apis.md`.
SCAN_TT_APIS_LABEL = (
    f"//{repo_paths.G3_REPO_PATH}/{repo_paths.SCANNER_PACKAGE}:scan_tt_apis"
)

# Exclude build output, test, documentation, and tooling directories
EXCLUDED_DIRS: Set[str] = {
    # go/keep-sorted start
    "_build",
    "blaze-bin",
    "blaze-genfiles",
    "blaze-out",
    "build",
    "build_files",
    "docker",
    "docs",
    "examples",
    "experimental",
    "g3doc",
    "google",
    "oss",
    "test_data",
    "tests",
    "tools",
    # go/keep-sorted end
}

# Internal-only and test-only Python packages that are not part of the public
# API. These can appear as attributes of `torch_tpu` once imported (e.g. when
# this scanner is imported as a library by a test).
_EXCLUDED_PYTHON_API_PREFIXES: Tuple[str, ...] = (
    "torch_tpu.google",
    "torch_tpu.tests",
)


def _is_excluded_python_api(full_name: str) -> bool:
  """Returns whether `full_name` belongs to an internal-only Python package."""
  return any(
      full_name == prefix or full_name.startswith(f"{prefix}.")
      for prefix in _EXCLUDED_PYTHON_API_PREFIXES
  )


# Prefixes of the names of the modules that TorchTPU defines.
_TORCH_TPU_NAMESPACES: Tuple[str, ...] = (
    "torch.tpu.",
    "torch_tpu.",
)

# The paths whose members are public Python APIs, in the order that
# `discover_python_apis` walks them. `torch.tpu` goes first, so that the
# TorchTPU objects without a public home that it exposes are listed under it.
_PYTHON_API_ROOTS: Tuple[str, ...] = (
    "torch.tpu",
    "torch_tpu",
)

# The namespaces that public TorchTPU Python APIs live in.
PUBLIC_NAMESPACES: Tuple[str, ...] = (
    "torch.tpu",
    "torch.accelerator",
)

_PUBLIC_PYTHON_API_CATEGORY = "Public Python API"

# The attribute that holds a module's version, e.g. `torch.tpu.__version__`.
MODULE_VERSION_ATTR = "__version__"

# The `tt_apis.yaml` category of the groups of public paths to the same object.
DUPLICATED_EXPORT_CATEGORY = "Duplicated Export"


def is_in_public_namespace(path: str) -> bool:
  """Returns whether `path` is in one of `PUBLIC_NAMESPACES`."""
  return any(
      path == namespace or path.startswith(f"{namespace}.")
      for namespace in PUBLIC_NAMESPACES
  )


def _is_torch_tpu_name(name: Any) -> bool:
  """Returns whether `name` names a TorchTPU module or one of its members."""
  return isinstance(name, str) and f"{name}.".startswith(_TORCH_TPU_NAMESPACES)


def _is_torch_tpu_object(value: Any) -> bool:
  """Returns whether `value`, e.g. a module or class, is defined by TorchTPU."""
  if isinstance(value, types.ModuleType):
    return _is_torch_tpu_name(getattr(value, "__name__", None))
  return _is_torch_tpu_name(getattr(value, "__module__", None))


# ==============================================================================
# Data Structures & YAML Serialization
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class ApiEntry:
  """Represents an inventory item for a TorchTPU API surface.

  Attributes:
    name: The full identifier name of the API surface or flag, e.g.
      "torch.tpu.precision".
    category: The API category classification, e.g. "Public Python API",
      "Environment Variable".
    stage: The lifecycle stage annotation for this API, or None if unannotated,
      e.g. Stage.EXPERIMENTAL.
    defined_in: Relative source code or runtime definition path, e.g.
      "common/env_vars.h".
    deprecated_in: The TorchTPU version that deprecated this API (e.g. "2.13"),
      or None if it isn't deprecated or the version is unknown.
    prefer: The preferred alternative to this internal API (e.g.
      "torch.accelerator.synchronize"), or None if it isn't internal or has no
      preferred alternative.
  """

  name: str
  category: str
  stage: Optional[Stage]
  defined_in: str
  deprecated_in: Optional[str] = None
  prefer: Optional[str] = None


class FlowDict(dict):
  """A dict that forces inline/flow style mapping in PyYAML."""


def flow_dict_representer(dumper, data):
  sorted_data = dict(sorted(data.items()))
  return dumper.represent_mapping(
      "tag:yaml.org,2002:map", sorted_data, flow_style=True
  )


yaml.add_representer(FlowDict, flow_dict_representer, Dumper=yaml.SafeDumper)


# ==============================================================================
# C++ Source Code Scanning Regexes & Constants
# ==============================================================================

# Matches C++ environment variable constant declarations.
# Examples:
#   inline constexpr char kVarName[] = "ENV_VAR_NAME";
# Regex Breakdown:
# - \bconstexpr\s+char\s+ : Matches 'constexpr char' preceded by a word boundary
#                           (handles optional 'inline').
# - (\w+)                 : Group 1 captures C++ variable identifier (e.g.
#                           'kVarName').
# - \[\]\s*=\s*           : Matches array brackets '[]' and assignment op '='.
# - "([^"]+)"             : Group 2 captures environment variable string inside
#                           quotes (e.g. 'ENV_VAR_NAME').
# - ;                     : Matches trailing semicolon ';'.
ENV_PATTERN = re.compile(
    r'\bconstexpr\s+char\s+(\w+)\[\]\s*=\s*"([^"]+)";',
    re.DOTALL,
)

# Matches kEnvVarToStage definition block in C++ headers.
# Example:
#   inline constexpr auto kEnvVarToStage = MakeConstexprMap<...>( { ... } );
ENV_VAR_TO_STAGE_MAP_PATTERN = re.compile(
    r"\bkEnvVarToStage\b[^=]*=\s*([^;]+);",
    re.DOTALL,
)

# Matches an entry within kEnvVarToStage.
# Examples:
#   {kAcceleratorTypeEnvVar, SymbolStage::InternalImplementation()}
#   {kLibtpuInitArgsEnvVar, std::nullopt}
#   {kWorldSizeEnvVar, SymbolStage::Stable()}
#   {kDeprecatedEnvVar, SymbolStage::Deprecated("2.13")}
ENV_STAGE_ENTRY_PATTERN = re.compile(
    r"\{\s*([a-zA-Z0-9_]+|\"[^\"]+\")\s*,\s*([^}]+?)\s*\}",
    re.DOTALL,
)

_STAGE_INTERNAL_IMPLEMENTATION = "INTERNAL_IMPLEMENTATION"

# Matches the version in a deprecated stage expression.
# Example:
#   SymbolStage::Deprecated("2.13")
_DEPRECATED_VERSION_PATTERN = re.compile(r'\bDeprecated\s*\(\s*"([^"]+)"')

# Matches ABSL C++ flag definition macros.
# Example:
#   ABSL_FLAG(std::map<int, int>, flag_name, default_val, "help string");
# Regex Breakdown:
# - ABSL_FLAG\s*\(\s*     : Matches 'ABSL_FLAG' and opening parenthesis '('.
# - (.*?)                 : Group 1 captures C++ type (supports templated types
#                           with commas like 'std::map<int, int>').
# - \s*,\s*(\w+)          : Group 2 captures flag name identifier .
# - (?:\s*,\s*([^,]+?))?  : Group 3 (optional) captures default value.
# - (?:\s*,\s*(.*?))?     : Group 4 (optional) captures help string.
# - \s*\);                : Matches closing parenthesis ')' and trailing
#                           semicolon ';'.
FLAG_PATTERN = re.compile(
    r"ABSL_FLAG\s*\(\s*(.*?)\s*,\s*(\w+)(?:\s*,\s*([^,]+?))?(?:\s*,\s*(.*?))?\s*\);",
    re.DOTALL,
)

# Matches TORCH_LIBRARY_IMPL(tpu, PrivateUse1, ...) block body.
TORCH_LIBRARY_IMPL_TPU_PATTERN = re.compile(
    r"TORCH_LIBRARY_IMPL\(\s*tpu\s*,\s*PrivateUse1\b[^)]*\)\s*\{([^}]+)\}",
    re.DOTALL,
)

# Matches custom op stage helper calls in C++ registration blocks.
# Examples:
#   ImplExperimental<OpName::kSparseDenseMatmul>(m, ...);
#   ImplStable<OpName::kRaggedDot>(m, ...);
#   ImplDeprecated<OpName::kOldOp>(m, ...);
#   ImplInternal<OpName::kInternalOp>(m, ...);
OP_STAGE_PATTERN = re.compile(
    r"\b(ImplExperimental|ImplStable|ImplDeprecated|ImplInternal)"
    r"\s*<\s*OpName::(k\w+)\s*>",
    re.DOTALL,
)

# Matches OpName enum to string mapping in op_names.h.
# Example:
#   case OpName::kSparseDenseMatmul:
#     return "sparse_dense_matmul";
OP_NAME_STR_PATTERN = re.compile(
    r'case\s+OpName::(k\w+):\s*return\s*"([^"]+)";'
)

_IMPL_STAGE_MAP: Dict[str, Stage] = {
    "ImplExperimental": Stage.EXPERIMENTAL,
    "ImplStable": Stage.STABLE,
    "ImplDeprecated": Stage.DEPRECATED,
    "ImplInternal": Stage.INTERNAL,
}


# ==============================================================================
# Phase 1: Python Runtime Memory Reflection
# ==============================================================================


# A Python API's stage, the version that deprecated it if it's deprecated, and
# its preferred alternative if it's internal.
_StageInfo = Tuple[Optional[Stage], Optional[str], Optional[str]]

_NO_STAGE_INFO: _StageInfo = (None, None, None)


def _get_python_api_stage_info(
    attr_name: str, attribute_value: Any, current_object: Any
) -> _StageInfo:
  """Resolves the lifecycle stage, deprecation version and alternative of an API.

  Checks in order:
  1. Python decorator metadata (__tt_api_stage__) directly on the attribute.
  2. Module/class level stage map (__tt_api_stages__) on the parent object.
  3. Naming heuristic fallback: Internal attributes starting with '_' or
     containing 'internal' default to Stage.INTERNAL.
  4. Defaults to None (unannotated / unknown stage).

  Args:
      attr_name: The name of the attribute in `current_object`.
      attribute_value: The attribute's value.
      current_object: The module or class that holds the attribute.

  Returns:
      A (stage, deprecated_in, prefer) tuple, where `deprecated_in` is the
      version that deprecated the API, or None, and `prefer` is the preferred
      alternative to the API, or None.
  """
  # 1. Check direct decorator tag on the attribute value (function, class, fget, fset)
  stage_attr = getattr(attribute_value, annotations.TT_API_STAGE, None)
  if stage_attr is not None:
    return (
        Stage(stage_attr),
        getattr(attribute_value, annotations.TT_API_DEPRECATED_VERSION, None),
        getattr(attribute_value, annotations.TT_API_PREFER, None),
    )

  # 2. Check __tt_api_stages__ map on the parent object
  map_stage_info = _get_stages_map_stage_info(attr_name, current_object)
  if map_stage_info is not None:
    return map_stage_info

  # 3. Check naming heuristic fallback for unannotated internal APIs
  if attr_name.startswith("_") or "internal" in attr_name.lower():
    return Stage.INTERNAL, None, None

  return _NO_STAGE_INFO


def _get_stages_map_stage_info(
    attr_name: str, current_object: Any
) -> Optional[_StageInfo]:
  """Returns an API's stage info from the `__tt_api_stages__` map of its holder.

  Args:
      attr_name: The name of the attribute in `current_object`.
      current_object: The module or class that holds the attribute.

  Returns:
      A (stage, deprecated_in, prefer) tuple, or None if the map doesn't give
      the attribute a stage.
  """
  stages_map = getattr(current_object, annotations.TT_API_STAGES_MAP, {})
  if isinstance(stages_map, dict) and attr_name in stages_map:
    api_stage_info = stages_map[attr_name]
    stage = getattr(api_stage_info, "stage", None)
    if stage is not None:
      return Stage(stage), getattr(api_stage_info, "version", None), None
  return None


def _get_python_api_stage(
    attr_name: str, attribute_value: Any, current_object: Any
) -> Optional[Stage]:
  """Resolves the lifecycle stage for a Python API attribute or function.

  See `_get_python_api_stage_info`.

  Args:
      attr_name: The name of the attribute in `current_object`.
      attribute_value: The attribute's value.
      current_object: The module or class that holds the attribute.

  Returns:
      The stage, or None if it's unknown.
  """
  return _get_python_api_stage_info(attr_name, attribute_value, current_object)[
      0
  ]


def _infer_return_type_from_ast(func: Any) -> Optional[type]:
  """Statically inspects the AST of a function's return statements to infer the returned class within TorchTPU.

  Args:
      func: The callable function or method to inspect.

  Returns:
      The return class object if inferred and belongs to TorchTPU, or None.
  """
  try:
    # Unwrap decorated functions (e.g. @annotations.experimental wrappers) to
    # reach the original function definition and its true module globals.
    raw_func = inspect.unwrap(func)
    func_globals = getattr(raw_func, "__globals__", None) or getattr(
        getattr(raw_func, "__func__", None), "__globals__", {}
    )

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    func_node = next(
        (
            n
            for n in ast.iter_child_nodes(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        tree,
    )

    def _get_top_level_returns(node: ast.AST):
      if isinstance(node, ast.Return):
        yield node
      elif not isinstance(
          node,
          (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
      ):
        for child in ast.iter_child_nodes(node):
          yield from _get_top_level_returns(child)

    for stmt in getattr(func_node, "body", [func_node]):
      for ret_node in _get_top_level_returns(stmt):
        if isinstance(ret_node.value, ast.Call):
          call_func = ret_node.value.func
          if isinstance(call_func, ast.Attribute) and isinstance(
              call_func.value, ast.Name
          ):
            mod_obj = func_globals.get(call_func.value.id)
            cls_obj = getattr(mod_obj, call_func.attr, None)
          elif isinstance(call_func, ast.Name):
            cls_obj = func_globals.get(call_func.id)
          else:
            cls_obj = None

          if inspect.isclass(cls_obj):
            return cls_obj
  except Exception:  # pylint: disable=broad-except
    pass
  return None


def _import_public_submodules(package: types.ModuleType) -> None:
  """Recursively imports the public submodules of `package`.

  Reflection only finds the submodules that have been imported, and a package
  doesn't necessarily import its public subpackages itself (e.g. `torch_tpu`
  doesn't import `torch_tpu.ops`).

  Args:
      package: The package whose public submodules to import.
  """
  for module_info in pkgutil.iter_modules(
      getattr(package, "__path__", ()), f"{package.__name__}."
  ):
    name = module_info.name
    if name.rpartition(".")[2].startswith("_") or _is_excluded_python_api(name):
      continue
    submodule = importlib.import_module(name)
    if module_info.ispkg:
      _import_public_submodules(submodule)


def _get_home(value: Any) -> Optional[Tuple[str, Any]]:
  """Returns the public path where a TorchTPU module or class is defined.

  The home of a module is its `__name__`, and the home of a class is its
  `__qualname__` in its `__module__`. The home only counts if it's a public
  path (no part starts with "_") that resolves to `value`.

  Args:
      value: The module or class.

  Returns:
      A (home, container) pair, where `container` is the object that holds
      `value` at its home (None for a module), or None if `value` has no public
      home.
  """
  if isinstance(value, types.ModuleType):
    module_name, qualname = getattr(value, "__name__", None), ""
  else:
    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
  if not isinstance(module_name, str) or not isinstance(qualname, str):
    return None
  home = f"{module_name}.{qualname}" if qualname else module_name
  if (
      not _is_torch_tpu_name(home)
      or _is_excluded_python_api(home)
      or not all(
          part.isidentifier() and not part.startswith("_")
          for part in home.split(".")
      )
  ):
    return None

  container = None
  resolved = sys.modules.get(module_name)
  for part in qualname.split(".") if qualname else []:
    container = resolved
    try:
      resolved = getattr(container, part)
    except Exception:  # pylint: disable=broad-except
      return None
  return (home, container) if resolved is value else None


def _extract_python_apis(roots: Sequence[Tuple[Any, str]]) -> List[ApiEntry]:
  """Recursively inspects Python objects and their members for public APIs.

  Every public path to a TorchTPU object gets an entry, but the members of a
  TorchTPU module or class are listed only once: at its home (see `_get_home`),
  or at the first path where it's found if it has no public home. Modules,
  classes and other callables that a module imports from other libraries are
  skipped.

  Args:
      roots: The (object, prefix) pairs to inspect in the given order, e.g.
        `[(torch.tpu, "torch.tpu")]`.

  Returns:
      A list of ApiEntry objects representing discovered APIs with stages.
  """
  discovered: List[ApiEntry] = []
  # The path where the members of each walked module or class are listed, keyed
  # by the object's id.
  listed_at: Dict[int, str] = {}

  def _add(name: str, stage_info: _StageInfo) -> None:
    stage, deprecated_in, prefer = stage_info
    discovered.append(
        ApiEntry(
            name=name,
            category=_PUBLIC_PYTHON_API_CATEGORY,
            stage=stage,
            defined_in="runtime",
            deprecated_in=(
                deprecated_in if stage == Stage.DEPRECATED else None
            ),
            prefer=prefer if stage == Stage.INTERNAL else None,
        )
    )

  def _list_members(value: Any, path: str, stage_info: _StageInfo) -> None:
    """Lists a TorchTPU module or class, and its members, at its home.

    Args:
        value: The module or class.
        path: The path to list `value` at if it has no public home.
        stage_info: The stage info of `value` at `path`.
    """
    home = _get_home(value)
    if home is not None:
      path, container = home
      stage_info = _get_python_api_stage_info(
          path.rpartition(".")[2], value, container
      )
    # A module gets no entry at the path where its members are listed.
    if not isinstance(value, types.ModuleType):
      _add(path, stage_info)
    _traverse(value, path)

  def _traverse(current_object: Any, current_prefix: str) -> None:
    """Recursively traverses modules/classes to discover public APIs/members.

    Args:
        current_object: The current object to inspect.
        current_prefix: The current prefix string.
    """
    listed_at[id(current_object)] = current_prefix

    # A module's `__version__` is public although it starts with "_", and
    # `dir()` doesn't necessarily list it (e.g. `dir(torch.tpu)` doesn't). A
    # `str` can't carry a stage decorator, and the naming heuristic doesn't
    # apply, so its stage comes from the module's `__tt_api_stages__` map only.
    if isinstance(current_object, types.ModuleType):
      try:
        getattr(current_object, MODULE_VERSION_ATTR)
      except Exception:  # pylint: disable=broad-except
        pass
      else:
        _add(
            f"{current_prefix}.{MODULE_VERSION_ATTR}",
            _get_stages_map_stage_info(MODULE_VERSION_ATTR, current_object)
            or _NO_STAGE_INFO,
        )

    stages_map = getattr(current_object, annotations.TT_API_STAGES_MAP, {})
    attr_names = set(dir(current_object))
    # When current_object is a class, `dir(current_object)` only lists
    # attributes on the class and its base classes, omitting attributes and
    # properties defined on its custom metaclass (`type(current_object)`).
    # Explicitly add all attribute names from custom TorchTPU metaclasses to `attr_names`,
    # excluding builtin metaclasses (`type`, `object`) and external metaclasses.
    if isinstance(current_object, type):
      for cls in type(current_object).__mro__:
        if cls not in (type, object) and _is_torch_tpu_name(
            getattr(cls, "__module__", None)
        ):
          attr_names.update(cls.__dict__.keys())
    if isinstance(stages_map, dict):
      attr_names.update(stages_map.keys())
    # `dir` only lists the dataclass fields that have a class-level default.
    # Fields without one (or with a `default_factory`) exist only on instances,
    # so add all of them explicitly.
    dataclass_fields = (
        {field.name for field in dataclasses.fields(current_object)}
        if isinstance(current_object, type)
        and dataclasses.is_dataclass(current_object)
        else set()
    )
    attr_names.update(dataclass_fields)

    for attr_name in sorted(attr_names):
      if attr_name.startswith("_"):
        continue

      # Skip built-in Enum property descriptors ('name', 'value') when scanning Enum classes
      # (both Python standard enum.Enum and Pybind11 C++ enums containing __members__).
      is_enum_builtin_attr = attr_name in ("name", "value")
      is_standard_py_enum_class = isinstance(
          current_object, type
      ) and issubclass(current_object, enum.Enum)
      is_pybind11_cpp_enum_class = hasattr(current_object, "__members__")
      if is_enum_builtin_attr and (
          is_standard_py_enum_class or is_pybind11_cpp_enum_class
      ):
        continue

      full_name = f"{current_prefix}.{attr_name}"
      if _is_excluded_python_api(full_name):
        continue

      # 1. Resolve the parent class object: If current_object is an instance,
      # get its type; if it is a class object, use it directly.
      parent_class = (
          current_object
          if isinstance(current_object, type)
          else type(current_object)
      )

      # Skip attributes inherited from upstream base classes unless explicitly
      # annotated.
      if not isinstance(current_object, types.ModuleType):
        defining_class = next(
            (cls for cls in parent_class.__mro__ if attr_name in cls.__dict__),
            None,
        )
        if defining_class is not None and defining_class is not parent_class:
          if not _is_torch_tpu_name(
              getattr(defining_class, "__module__", None)
          ):
            if not (isinstance(stages_map, dict) and attr_name in stages_map):
              continue

      # 2. Retrieve the raw attribute descriptor from the class or metaclass:
      # If it's a @property, returns the un-evaluated `property` descriptor object
      # (containing .fget and .fset), avoiding invoking the getter.
      # inspect.getattr_static avoids executing property getters or dynamic __getattr__.
      property_desc = None
      if isinstance(current_object, type):
        meta_desc = inspect.getattr_static(
            type(current_object), attr_name, None
        )
        if isinstance(meta_desc, property):
          property_desc = meta_desc
      if property_desc is None:
        raw_desc = inspect.getattr_static(parent_class, attr_name, None)
        if isinstance(raw_desc, property):
          property_desc = raw_desc

      # 3. Check if the attribute is a property descriptor:
      if isinstance(property_desc, property):
        # A property's getter and setter can be in different lifecycle stages.
        # Inspect both underlying functions and track separate entries for them.
        for fn, suffix in (
            (property_desc.fget, "getter"),
            (property_desc.fset, "setter"),
        ):
          if fn is not None:
            _add(
                f"{full_name} {suffix}",
                _get_python_api_stage_info(attr_name, fn, current_object),
            )
        continue

      # 4. A dataclass field is an instance data member whose class-level
      # value, if any, is only its default, so list it without inspecting it.
      if attr_name in dataclass_fields:
        _add(
            full_name,
            _get_python_api_stage_info(attr_name, None, current_object),
        )
        continue

      # 5. For non-property attributes, safely retrieve the attribute value from the current object.
      try:
        attribute_value = getattr(current_object, attr_name)
      except Exception:  # pylint: disable=broad-except
        continue

      stage_info = _get_python_api_stage_info(
          attr_name, attribute_value, current_object
      )

      # 6. List the members of TorchTPU modules and classes only once (see
      # `_list_members`); any other path to them gets a single entry. Modules
      # and classes from other libraries (e.g. torch.Tensor) are skipped.
      if isinstance(attribute_value, (types.ModuleType, type)):
        if _is_torch_tpu_object(attribute_value):
          if id(attribute_value) not in listed_at:
            _list_members(attribute_value, full_name, stage_info)
          if listed_at[id(attribute_value)] != full_name:
            _add(full_name, stage_info)
        continue

      # Skip functions and other callables that a module imports from other
      # libraries (e.g. `from typing import Optional`). A callable instance,
      # e.g. a `functools.partial` object, is judged by its class.
      if isinstance(current_object, types.ModuleType) and callable(
          attribute_value
      ):
        owner = getattr(attribute_value, "__module__", None)
        if isinstance(owner, str) and not _is_torch_tpu_name(owner):
          continue
      # `from __future__ import annotations` binds a feature, not an API.
      if type(attribute_value).__module__ == "__future__":
        continue

      _add(full_name, stage_info)
      # Check if callable has a return-type annotation or AST return statement
      # pointing to a class within TorchTPU
      if callable(attribute_value):
        try:
          hints = typing.get_type_hints(attribute_value)
          return_type = hints.get("return")
        except Exception:  # pylint: disable=broad-except
          return_type = None

        if not inspect.isclass(return_type):
          return_type = _infer_return_type_from_ast(attribute_value)

        if (
            inspect.isclass(return_type)
            and _is_torch_tpu_object(return_type)
            and id(return_type) not in listed_at
        ):
          if _get_home(return_type) is None:
            # List the members of a returned class without a public home under
            # this callable.
            _traverse(return_type, full_name)
          else:
            _list_members(return_type, full_name, _NO_STAGE_INFO)

  for root, prefix in roots:
    if id(root) not in listed_at:
      _traverse(root, prefix)
  return discovered


_UNSUPPORTED_MESSAGE_SUBSTRINGS: Tuple[str, ...] = (
    "not implemented",
    "not supported",
    "doesn't support",
    "does not support",
    "unsupported",
)


def _is_unsupported_message(message: str) -> bool:
  """Returns True if an exception or warning message indicates an unsupported API."""
  lowered = message.lower()
  return any(substr in lowered for substr in _UNSUPPORTED_MESSAGE_SUBSTRINGS)


def _get_default_probe_arg(param_name: str) -> Any:
  """Returns a safe default argument for probing a required parameter."""
  if param_name in ("device", "device_index"):
    return 0
  if param_name == "stream":
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
      return torch.accelerator.current_stream()
    return None
  if param_name == "path":
    return ""
  return None


class _CppWarningBridge:
  """Helper whose `__torch_function__` executes inside C++ `HANDLE_TH_ERRORS`."""

  def __init__(self, callback: Any):
    self._callback = callback
    self.result = None

  @classmethod
  def __torch_function__(
      cls,
      func: Any,
      types_tuple: Any,
      args: Sequence[Any] = (),
      kwargs: Optional[Dict[str, Any]] = None,
  ) -> Any:
    del func, types_tuple, kwargs
    runner = args[0]
    runner.result = runner._callback()
    return None


def _invoke_with_cpp_warning_capture(callback: Any) -> Any:
  """Executes `callback()` inside a PyTorch C++ `PyWarningHandler` scope.

  Several `torch._C._accelerator_*` bindings are defined with raw pybind11
  `m.def` rather than `HANDLE_TH_ERRORS`, so C++ `TORCH_WARN` /
  `TORCH_WARN_ONCE` messages would otherwise bypass Python's `warnings` module.
  Dispatching through `torch.abs(_CppWarningBridge(callback))` enters
  `THPVariable_abs`'s `HANDLE_TH_ERRORS` block, installing
  `torch::PyWarningHandler` on the current thread so all C++ `TORCH_WARN`
  messages are surfaced as Python warnings.
  """
  bridge: Any = _CppWarningBridge(callback)
  torch.abs(bridge)
  return bridge.result


def _probe_accelerator_callable(
    callable_obj: Any, probe_cache: Dict[int, bool]
) -> bool:
  """Probes a `torch.accelerator` callable or class at runtime to check TPU support.

  Invokes `callable_obj` with zero-arg defaults (or safe defaults for required
  `device` / `stream` parameters) and returns False if it raises
  `NotImplementedError`, raises an error containing a "not implemented /
  not supported / backend doesn't support" message, or emits a warning with
  such a message.

  Args:
      callable_obj: The function or class to probe.
      probe_cache: Memoization map keyed by `id(callable_obj)` to avoid
        re-probing identical callables re-exported across submodules.

  Returns:
      True if the API is implemented for TPU, False otherwise.
  """
  obj_id = id(callable_obj)
  if obj_id in probe_cache:
    return probe_cache[obj_id]

  pos_args: List[Any] = []
  kw_args: Dict[str, Any] = {}
  try:
    sig = inspect.signature(callable_obj)
    for param in sig.parameters.values():
      if param.default is not inspect.Parameter.empty or param.kind in (
          inspect.Parameter.VAR_POSITIONAL,
          inspect.Parameter.VAR_KEYWORD,
      ):
        continue
      arg_val = _get_default_probe_arg(param.name)
      if param.kind == inspect.Parameter.KEYWORD_ONLY:
        kw_args[param.name] = arg_val
      else:
        pos_args.append(arg_val)
  except (ValueError, TypeError):
    pass

  def _run_probe():
    res = callable_obj(*pos_args, **kw_args)
    # For context-manager classes (e.g. torch.accelerator.device_index),
    # __init__ only stores arguments; enter and exit the context to exercise
    # the underlying C++ accelerator hook.
    if (
        isinstance(callable_obj, type)
        and hasattr(res, "__enter__")
        and hasattr(res, "__exit__")
    ):
      with res:
        pass
    return res

  supported = True
  with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    try:
      _invoke_with_cpp_warning_capture(_run_probe)
    except NotImplementedError:
      supported = False
    except Exception as exc:  # pylint: disable=broad-except
      if _is_unsupported_message(str(exc)):
        supported = False
      else:
        logging.warning(
            "Unexpected exception probing %r: %s", callable_obj, exc
        )
        supported = False

  if supported:
    for warn_item in caught_warnings:
      if _is_unsupported_message(str(warn_item.message)):
        supported = False
        break

  probe_cache[obj_id] = supported
  return supported


def _is_deprecated_by_pytorch(api: Any) -> bool:
  """Returns whether PyTorch deprecated an upstream API.

  PyTorch deprecates APIs with `typing_extensions.deprecated`, which, like
  `warnings.deprecated` (PEP 702), sets `__deprecated__` to the deprecation
  message.

  Args:
      api: The upstream PyTorch function or class.
  """
  return bool(getattr(api, "__deprecated__", None))


def discover_accelerator_apis(
    accelerator_module: Optional[Any] = None,
) -> List[ApiEntry]:
  """Discovers public Python APIs in `torch.accelerator` implemented for TPU.

  Args:
      accelerator_module: Optional module object to inspect (defaults to
        `torch.accelerator`).

  Returns:
      A sorted list of `ApiEntry` objects with category `Public Python API`
      and stage `Stage.DEPRECATED` if PyTorch deprecated the API, or
      `Stage.STABLE` otherwise.
  """
  if accelerator_module is None:
    accelerator_module = getattr(torch, "accelerator", None)
    assert (
        accelerator_module is not None
    ), "Failed to find 'torch.accelerator' module in runtime memory."

  # The discovered APIs that TPU supports, by path.
  discovered: Dict[str, Any] = {}
  visited_modules: Set[int] = set()
  probe_cache: Dict[int, bool] = {}

  # Force C++ TORCH_WARN_ONCE to emit warnings on every invocation during probing
  # and preserve device/stream state across probes.
  prev_warn_always = None
  if hasattr(torch._C, "_get_warnAlways") and hasattr(
      torch._C, "_set_warnAlways"
  ):
    prev_warn_always = torch._C._get_warnAlways()
    torch._C._set_warnAlways(True)

  initial_device_idx = None
  initial_stream = None
  if (
      accelerator_module is getattr(torch, "accelerator", None)
      and torch.accelerator.is_available()
  ):
    initial_device_idx = torch.accelerator.current_device_index()
    initial_stream = torch.accelerator.current_stream()

  def _traverse_module(current_mod: Any, current_prefix: str) -> None:
    mod_id = id(current_mod)
    if mod_id in visited_modules:
      return
    visited_modules.add(mod_id)

    explicit_all = getattr(current_mod, "__all__", None)
    if explicit_all is not None:
      candidate_names = set(explicit_all)
    else:
      candidate_names = {
          name for name in dir(current_mod) if not name.startswith("_")
      }

    # Always include public submodules belonging to torch.accelerator (such as
    # torch.accelerator.memory and torch.accelerator.random), even if not listed
    # in __all__.
    for attr_name in dir(current_mod):
      if attr_name.startswith("_"):
        continue
      try:
        attr_val = getattr(current_mod, attr_name)
      except Exception:  # pylint: disable=broad-except
        continue
      if isinstance(attr_val, types.ModuleType):
        submod_name = getattr(attr_val, "__name__", "") or ""
        if f"{submod_name}.".startswith(
            (f"{current_prefix}.", "torch.accelerator.")
        ):
          candidate_names.add(attr_name)

    for attr_name in sorted(candidate_names):
      if attr_name.startswith("_"):
        continue
      try:
        attribute_value = getattr(current_mod, attr_name)
      except Exception:  # pylint: disable=broad-except
        continue

      full_name = f"{current_prefix}.{attr_name}"

      if isinstance(attribute_value, types.ModuleType):
        submod_name = getattr(attribute_value, "__name__", "") or ""
        if f"{submod_name}.".startswith(
            (f"{current_prefix}.", "torch.accelerator.")
        ):
          _traverse_module(attribute_value, full_name)
        continue

      # When __all__ is not explicitly specified, skip imported external symbols
      if explicit_all is None:
        defining_mod = getattr(attribute_value, "__module__", "") or ""
        if defining_mod and not f"{defining_mod}.".startswith(
            (f"{current_prefix}.", "torch.accelerator.")
        ):
          continue

      if callable(attribute_value):
        if _probe_accelerator_callable(attribute_value, probe_cache):
          discovered[full_name] = attribute_value

  try:
    _traverse_module(accelerator_module, "torch.accelerator")
  finally:
    if initial_stream is not None:
      try:
        torch.accelerator.set_stream(initial_stream)
      except Exception:  # pylint: disable=broad-except
        pass
    if initial_device_idx is not None:
      try:
        torch.accelerator.set_device_index(initial_device_idx)
      except Exception:  # pylint: disable=broad-except
        pass
    if prev_warn_always is not None:
      torch._C._set_warnAlways(prev_warn_always)

  # `deprecated()` also sets `__deprecated__` on the function that a deprecated
  # alias wraps, e.g. `current_device_index` for `current_device_idx`, although
  # that function isn't deprecated.
  deprecated_alias_targets = {
      id(api.__wrapped__)
      for api in discovered.values()
      if _is_deprecated_by_pytorch(api) and hasattr(api, "__wrapped__")
  }
  # The APIs are controlled in upstream PyTorch and we don't have an easy way to
  # annotate them. Mark them stable as we cannot generate an "experimental"
  # warning for upstream APIs, unless PyTorch deprecated them.
  return [
      ApiEntry(
          name=full_name,
          category=_PUBLIC_PYTHON_API_CATEGORY,
          stage=(
              Stage.DEPRECATED
              if _is_deprecated_by_pytorch(api)
              and id(api) not in deprecated_alias_targets
              else Stage.STABLE
          ),
          defined_in="runtime",
      )
      for full_name, api in sorted(discovered.items())
  ]


def discover_python_apis() -> Tuple[List[ApiEntry], List[ApiEntry]]:
  """Dynamically inspects Python modules for exported APIs and Flags.

  Returns:
      A tuple of (public_python_apis, python_flags).
  """
  _import_public_submodules(torch_tpu)
  roots: List[Tuple[Any, str]] = []
  for path in _PYTHON_API_ROOTS:
    root = _resolve_module_level_path(path)
    assert root is not None, f"Failed to find '{path}' in runtime memory."
    roots.append((root, path))
  apis = _extract_python_apis(roots)

  # Deduplicate by full API name
  api_dict: Dict[str, ApiEntry] = {}
  for item in apis:
    api_dict[item.name] = item

  public_python_apis = sorted(api_dict.values(), key=lambda x: x.name)

  # ABSL Python Flags
  python_flags: List[ApiEntry] = []
  for flag_name in flags.FLAGS:
    flag_object = flags.FLAGS[flag_name]
    filename = getattr(flag_object, "filename", "") or ""
    if "torch_tpu" in filename or flag_name.startswith("torch_tpu"):
      stage = (
          Stage.INTERNAL if flag_name.startswith("torch_tpu_internal") else None
      )
      python_flags.append(
          ApiEntry(
              name=flag_name,
              category="ABSL Python Flag",
              stage=stage,
              defined_in="runtime",
          )
      )

  return (
      public_python_apis,
      sorted(python_flags, key=lambda x: x.name),
  )


def find_module_versions_without_stage(apis: Sequence[ApiEntry]) -> List[str]:
  """Returns the public module `__version__`s in `apis` that have no stage.

  A `str` can't carry a stage decorator, so the stage of a module's
  `__version__` must come from the module's `__tt_api_stages__` map.

  Args:
      apis: The discovered APIs, e.g. from `discover_python_apis`.

  Returns:
      The sorted names of the unannotated `__version__` APIs, e.g.
      `["torch.tpu.__version__"]`.
  """
  return sorted(
      api.name
      for api in apis
      if api.category == _PUBLIC_PYTHON_API_CATEGORY
      and api.name.endswith(f".{MODULE_VERSION_ATTR}")
      and api.stage is None
  )


def _resolve_module_level_path(path: str) -> Any:
  """Returns the object at `path` if `path` is a module or a module member.

  An object registered in `sys.modules` counts as a module even if it isn't a
  `types.ModuleType`. So do the paths in `_PYTHON_API_ROOTS`.

  Args:
      path: A dotted path, e.g. "torch.tpu.synchronize".

  Returns:
      The object, or None if `path` isn't a module or a member of one.
  """
  value = sys.modules.get(path)
  parent_path, _, name = path.rpartition(".")
  if value is not None or not parent_path:
    return value
  parent = sys.modules.get(parent_path)
  if parent is None:
    parent = _resolve_module_level_path(parent_path)
    if parent is None or (
        not isinstance(parent, types.ModuleType)
        and parent_path not in _PYTHON_API_ROOTS
    ):
      return None
  try:
    return getattr(parent, name)
  except Exception:  # pylint: disable=broad-except
    return None


def split_property_accessor(name: str) -> Tuple[str, str]:
  """Splits the name of an API entry into its path and property accessor.

  Args:
      name: The name of an API entry, e.g. "torch.tpu.num_stripes setter".

  Returns:
      The path and the accessor, e.g. ("torch.tpu.num_stripes", "setter"). The
      accessor is "" if the entry isn't a property accessor.
  """
  path, _, accessor = name.partition(" ")
  return path, accessor


def find_duplicate_public_paths(apis: Sequence[ApiEntry]) -> List[List[str]]:
  """Returns the public paths that lead to the same TorchTPU object.

  Only paths to modules and module members count; class members are listed
  once, under their class. A module gets no entry at the path where its members
  are listed, so the prefixes of each path count too. Deprecated paths and
  property accessors don't count.

  Args:
      apis: Public Python API entries, e.g. from `discover_python_apis`.

  Returns:
      The groups of two or more paths that lead to the same object, sorted.
  """
  paths: Set[str] = set()
  deprecated_paths: Set[str] = set()
  for api in apis:
    path, accessor = split_property_accessor(api.name)
    parts = path.split(".")
    paths.update(".".join(parts[:i]) for i in range(1, len(parts)))
    if accessor:
      continue
    if api.stage == Stage.DEPRECATED:
      deprecated_paths.add(path)
    else:
      paths.add(path)

  # Maps a key for each object to the object, which keeps the ids in the key
  # from being reused, and to its paths.
  groups: Dict[Any, Tuple[Any, List[str]]] = {}
  for path in sorted(paths - deprecated_paths):
    value = _resolve_module_level_path(path)
    if not _is_torch_tpu_object(value):
      continue
    # Each lookup of a method creates a new bound method.
    if inspect.ismethod(value):
      key = (id(value.__self__), id(value.__func__))
    else:
      key = id(value)
    _, group = groups.setdefault(key, (value, []))
    group.append(path)
  return sorted(group for _, group in groups.values() if len(group) > 1)


def find_duplicated_exports(apis: Sequence[ApiEntry]) -> List[List[str]]:
  """Returns the duplicate public paths that `tt_apis.yaml` tracks.

  These are the groups from `find_duplicate_public_paths` that include a path in
  `PUBLIC_NAMESPACES`. The other groups consist of legacy paths outside
  `PUBLIC_NAMESPACES`, which are tracked as such.

  Args:
      apis: API entries, e.g. from `discover_all_apis`. Only public Python API
        entries count.

  Returns:
      The groups of two or more paths that lead to the same object, sorted.
  """
  public_python_apis = [
      api for api in apis if api.category == _PUBLIC_PYTHON_API_CATEGORY
  ]
  return [
      group
      for group in find_duplicate_public_paths(public_python_apis)
      if any(is_in_public_namespace(path) for path in group)
  ]


def _resolve_public_path(path: str) -> Any:
  """Returns the object at a public API path, e.g. a nested class.

  Unlike `_resolve_module_level_path`, this also resolves class members.

  Args:
      path: A dotted path, e.g. "torch.tpu.optimizers.Sgd.ParamGroup".

  Returns:
      The object, or None if `path` doesn't resolve.
  """
  parts = path.split(".")
  for i in range(len(parts), 0, -1):
    value = _resolve_module_level_path(".".join(parts[:i]))
    if value is None:
      continue
    for part in parts[i:]:
      try:
        value = getattr(value, part)
      except Exception:  # pylint: disable=broad-except
        return None
    return value
  return None


def _assigned_self_attributes(
    function: Any,
) -> List[Tuple[str, str]]:
  """Returns the attributes that `function` assigns on its first argument.

  Covers `self.x = ...`, `self.x: T = ...`, `self.x += ...`, tuple and list
  targets, `for self.x in ...` and `with ... as self.x`, including in nested
  functions that capture `self`.

  Args:
      function: A function whose first parameter is the instance.

  Returns:
      (attribute name, "file:line") pairs, or an empty list if the source isn't
      available (e.g. for C extensions or generated dataclass methods).
  """
  function = inspect.unwrap(function)
  try:
    source_lines, first_line = inspect.getsourcelines(function)
    source_file = inspect.getsourcefile(function) or "<unknown>"
    tree = ast.parse(textwrap.dedent("".join(source_lines)))
  except (OSError, TypeError, SyntaxError):
    return []
  function_node = next(
      (
          node
          for node in ast.iter_child_nodes(tree)
          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
      ),
      None,
  )
  if function_node is None:
    return []
  positional = function_node.args.posonlyargs + function_node.args.args
  if not positional:
    return []
  self_name = positional[0].arg

  marker = f"{repo_paths.G3_REPO_PATH}/"
  if marker in source_file:
    source_file = source_file[source_file.rindex(marker) + len(marker) :]

  def _flatten(target: ast.AST) -> List[ast.AST]:
    if isinstance(target, (ast.Tuple, ast.List)):
      return [leaf for elt in target.elts for leaf in _flatten(elt)]
    if isinstance(target, ast.Starred):
      return _flatten(target.value)
    return [target]

  found: List[Tuple[str, str]] = []
  for node in ast.walk(function_node):
    if isinstance(node, ast.Assign):
      targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For)):
      targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
      targets = [node.optional_vars]
    else:
      continue
    for target in targets:
      for leaf in _flatten(target):
        if (
            isinstance(leaf, ast.Attribute)
            and isinstance(leaf.value, ast.Name)
            and leaf.value.id == self_name
        ):
          line = first_line + leaf.lineno - 1
          found.append((leaf.attr, f"{source_file}:{line}"))
  return found


def find_public_instance_attributes_of_class(
    cls: type,
) -> Dict[str, str]:
  """Returns the public instance attributes that a TorchTPU class assigns.

  Scans the methods (including property accessors) of `cls` and of its
  TorchTPU base classes for assignments to public attributes of `self`.
  Methods of base classes from other libraries are ignored. So are assignments
  to properties and other data descriptors (they call a setter, which the
  scanner lists), and dataclass fields, which the scanner lists from
  `dataclasses.fields`.

  Attributes set with `setattr`, by code outside the class, or by functions
  that aren't methods of the class aren't detected.

  Args:
      cls: The class.

  Returns:
      A map from each public instance attribute to the "file:line" of its first
      assignment.
  """
  dataclass_fields = (
      {field.name for field in dataclasses.fields(cls)}
      if dataclasses.is_dataclass(cls)
      else set()
  )
  found: Dict[str, str] = {}
  for owner in cls.__mro__:
    if owner is object or not _is_torch_tpu_name(
        getattr(owner, "__module__", None)
    ):
      continue
    for member in vars(owner).values():
      if isinstance(member, property):
        functions = [member.fget, member.fset, member.fdel]
      elif inspect.isfunction(member):
        functions = [member]
      else:
        # Static methods have no instance, and class methods assign class
        # attributes, which `dir` lists.
        continue
      for function in functions:
        if function is None:
          continue
        for name, location in _assigned_self_attributes(function):
          if name.startswith("_") or name in dataclass_fields:
            continue
          if hasattr(inspect.getattr_static(cls, name, None), "__set__"):
            continue
          found.setdefault(name, location)
  return found


def find_public_instance_attributes(
    apis: Sequence[ApiEntry],
) -> List[Tuple[str, str]]:
  """Returns the public instance attributes of public TorchTPU classes.

  The scanner discovers APIs with `dir`, which doesn't list instance
  attributes, so they would be public without being tracked. Public TorchTPU
  classes must keep their instance attributes private and expose state through
  properties instead. See `find_public_instance_attributes_of_class`.

  Args:
      apis: Public Python API entries, e.g. from `discover_python_apis`.

  Returns:
      Sorted (API path of the attribute, "file:line") pairs.
  """
  classes: Dict[int, Tuple[str, type]] = {}
  for api in apis:
    path, accessor = split_property_accessor(api.name)
    if accessor:
      continue
    value = _resolve_public_path(path)
    if not inspect.isclass(value) or not _is_torch_tpu_object(value):
      continue
    home = _get_home(value)
    if home is not None:
      path = home[0]
    if id(value) not in classes or path < classes[id(value)][0]:
      classes[id(value)] = (path, value)
  found: List[Tuple[str, str]] = []
  for path, cls in classes.values():
    for name, location in find_public_instance_attributes_of_class(cls).items():
      found.append((f"{path}.{name}", location))
  return sorted(found)


# ==============================================================================
# Phase 2: C++ Source Code Scanning
# ==============================================================================


def _strip_cpp_comments(content: str) -> str:
  """Strips C++ single-line (//) and block (/* */) comments while preserving string literals.

  Args:
      content: The C++ source code content.

  Returns:
      The C++ source code with comments removed.
  """
  pattern = re.compile(
      r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^\\"])*"',
      re.DOTALL,
  )

  def _replacer(match: re.Match[str]) -> str:
    token = match.group(0)
    if token.startswith("/"):
      return " "
    return token

  return pattern.sub(_replacer, content)


def _parse_env_stage_expr(expr: str) -> Optional[Any]:
  """Parses a C++ stage expression from kEnvVarToStage to a Stage enum or sentinel."""
  expr = expr.strip()
  if "InternalImplementation" in expr:
    return _STAGE_INTERNAL_IMPLEMENTATION
  if "InternalApi" in expr:
    return Stage.INTERNAL
  if "Experimental" in expr:
    return Stage.EXPERIMENTAL
  if "Stable" in expr:
    return Stage.STABLE
  if "Deprecated" in expr:
    return Stage.DEPRECATED
  return None


def _parse_env_deprecated_version(expr: str) -> Optional[str]:
  """Returns the version in a `SymbolStage::Deprecated("2.13")` expression."""
  match = _DEPRECATED_VERSION_PATTERN.search(expr)
  return match.group(1) if match else None


def _normalize_cpp_op_name(op_enum_name: str) -> str:
  """Converts C++ OpName enum string (e.g.

  kSparseDenseMatmul) to snake_case (e.g. sparse_dense_matmul).
  """
  name = op_enum_name.removeprefix("k").removeprefix("TorchTpu")
  s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
  return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def discover_cpp_source_surfaces(
    repository_root: str,
) -> Tuple[List[ApiEntry], List[ApiEntry], List[ApiEntry]]:
  """Scans C++ headers/sources for ABSL Flags, Env Vars, and Custom Ops.

  Args:
      repository_root: The root directory path of the repository.

  Returns:
      A tuple of (env_vars, cpp_flags, custom_ops).
  """
  env_vars: List[ApiEntry] = []
  cpp_flags: List[ApiEntry] = []
  custom_ops: List[ApiEntry] = []
  op_name_map: Dict[str, str] = {}
  raw_op_matches: List[Tuple[str, Optional[Stage], str]] = []

  seen_envs: Set[str] = set()
  seen_flags: Set[str] = set()
  seen_ops: Set[str] = set()

  env_stages: Dict[str, Any] = {}
  env_deprecated_versions: Dict[str, Optional[str]] = {}
  pending_envs: List[Tuple[str, str, str]] = (
      []
  )  # (const_name, env_name, relative_path)

  # Single Pass: Read files from disk, collect stages, and extract env/flag/op matches.
  # File contents (raw_content) are immediately discarded after processing each file.
  for root, dirs, files in os.walk(repository_root):
    dirs[:] = [
        d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")
    ]

    for file in files:
      if file.endswith((".h", ".cc", ".cpp")) and not file.endswith("_test.cc"):
        full_path = os.path.join(root, file)
        relative_path = os.path.relpath(full_path, repository_root)
        try:
          with open(full_path, "r", encoding="utf-8") as source_file:
            raw_content = source_file.read()

          # Strip C++ comments safely without truncating string literals containing '//'
          clean_content = _strip_cpp_comments(raw_content)

          # Extract official OpName string mappings from op_names.h
          if file == "op_names.h":
            for match in OP_NAME_STR_PATTERN.finditer(clean_content):
              op_enum_name = match.group(1)
              op_str = match.group(2)
              op_name_map[op_enum_name] = op_str

          # Extract kEnvVarToStage map from env_vars.h (or any header)
          for map_match in ENV_VAR_TO_STAGE_MAP_PATTERN.finditer(clean_content):
            map_body = map_match.group(1)
            for entry_match in ENV_STAGE_ENTRY_PATTERN.finditer(map_body):
              key = entry_match.group(1).strip().strip('"')
              val = entry_match.group(2).strip()
              env_stages[key] = _parse_env_stage_expr(val)
              env_deprecated_versions[key] = _parse_env_deprecated_version(val)

          # Extract Custom Ops from TORCH_LIBRARY_IMPL(tpu, PrivateUse1, ...) blocks
          for block_match in TORCH_LIBRARY_IMPL_TPU_PATTERN.finditer(
              clean_content
          ):
            block_body = block_match.group(1)
            for stage_match in OP_STAGE_PATTERN.finditer(block_body):
              impl_type, op_enum_name = (
                  stage_match.group(1),
                  stage_match.group(2),
              )
              stage = _IMPL_STAGE_MAP.get(impl_type)
              if stage is None:
                raise ValueError(
                    f"Unknown impl type: {impl_type} in {relative_path}"
                )
              raw_op_matches.append((op_enum_name, stage, relative_path))

          # Extract Environment Variables (defer stage resolution until all files are processed)
          for match in ENV_PATTERN.finditer(clean_content):
            const_name = match.group(1)
            env_name = match.group(2)
            if env_name in seen_envs:
              continue
            seen_envs.add(env_name)
            pending_envs.append((const_name, env_name, relative_path))

          # Extract ABSL C++ Flags
          for match in FLAG_PATTERN.finditer(clean_content):
            flag_name = match.group(2).strip()
            if flag_name in seen_flags:
              continue
            seen_flags.add(flag_name)

            stage = (
                Stage.INTERNAL
                if flag_name.startswith("torch_tpu_internal_")
                else None
            )
            cpp_flags.append(
                ApiEntry(
                    name=flag_name,
                    category="ABSL C++ Flag",
                    stage=stage,
                    defined_in=relative_path,
                )
            )

        except Exception as error:  # pylint: disable=broad-except
          logging.warning("Error reading file %s: %s", full_path, error)

  # Build custom ops ApiEntry list
  for op_enum_name, stage, relative_path in raw_op_matches:
    op_name = op_name_map.get(
        op_enum_name, _normalize_cpp_op_name(op_enum_name)
    )
    # Underscore-prefixed custom ops are internal implementation details (e.g.
    # backing a public Python wrapper), so they are not part of the API
    # surface.
    if op_name.startswith("_"):
      continue
    full_op_name = f"torch.ops.tpu.{op_name}"
    if full_op_name in seen_ops:
      continue
    seen_ops.add(full_op_name)

    custom_ops.append(
        ApiEntry(
            name=full_op_name,
            category="Custom Op",
            stage=stage,
            defined_in=relative_path,
        )
    )

  # Resolve final stages for discovered environment variables
  for const_name, env_name, relative_path in pending_envs:
    stage_info = env_stages.get(const_name, env_stages.get(env_name))
    if stage_info == _STAGE_INTERNAL_IMPLEMENTATION:
      continue

    env_vars.append(
        ApiEntry(
            name=env_name,
            category="Environment Variable",
            stage=stage_info,
            defined_in=relative_path,
            deprecated_in=env_deprecated_versions.get(
                const_name, env_deprecated_versions.get(env_name)
            ),
        )
    )

  return (
      sorted(env_vars, key=lambda x: x.name),
      sorted(cpp_flags, key=lambda x: x.name),
      sorted(custom_ops, key=lambda x: x.name),
  )


# ==============================================================================
# Phase 3: Formatting, Serialization & Artifact Export
# ==============================================================================


def _format_api_attributes(item: ApiEntry) -> FlowDict:
  """Returns the `tt_apis.yaml` attributes of an API item."""
  attributes = FlowDict({
      "category": item.category,
      "stage": item.stage.value if item.stage else "Unknown",
      "defined_in": item.defined_in,
  })
  if item.deprecated_in:
    attributes["deprecated_in"] = item.deprecated_in
  if item.prefer:
    attributes["prefer"] = item.prefer
  return attributes


# The license header of `tt_apis.yaml`, which the OSS lint requires.
_TT_APIS_YAML_HEADER = """\
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

"""


def format_tt_apis_yaml(
    all_items: List[ApiEntry],
    duplicated_exports: Sequence[Sequence[str]] = (),
) -> str:
  """Formats API items into the `tt_apis.yaml` baseline text.

  Args:
      all_items: A list of ApiEntry objects, each representing an API item.
      duplicated_exports: Groups of public paths to the same object, e.g. from
        `find_duplicated_exports`. Each group is listed under its first path in
        `PUBLIC_NAMESPACES`, with the other paths as `also_exported_as`.

  Returns:
      The YAML text, grouped into one commented section per category.
  """
  grouped: Dict[str, List[ApiEntry]] = collections.defaultdict(list)
  for item in all_items:
    grouped[item.category].append(item)

  categories: List[Tuple[str, Dict[str, FlowDict]]] = []
  for category, items in grouped.items():
    categories.append((
        category,
        {
            item.name: _format_api_attributes(item)
            for item in sorted(items, key=lambda x: x.name)
        },
    ))

  if duplicated_exports:
    duplicated_data: Dict[str, FlowDict] = {}
    for group in sorted(sorted(group) for group in duplicated_exports):
      name = next(
          (path for path in group if is_in_public_namespace(path)), group[0]
      )
      duplicated_data[name] = FlowDict({
          "category": DUPLICATED_EXPORT_CATEGORY,
          "also_exported_as": [path for path in group if path != name],
      })
    categories.append((DUPLICATED_EXPORT_CATEGORY, duplicated_data))

  sections: List[str] = []
  for category, formatted_data in categories:
    section_header = f"# --- {category} ---"
    section_yaml = yaml.safe_dump(
        formatted_data,
        width=float("inf"),
        sort_keys=False,
    )
    sections.append(f"{section_header}\n{section_yaml.strip()}")

  full_yaml_text = _TT_APIS_YAML_HEADER + "\n\n".join(sections) + "\n"
  return full_yaml_text


# The name of the generated list of public APIs, in `docs/markdown`.
PUBLIC_APIS_MD = "public_apis.md"

# The stages that `public_apis.md` lists, in order.
_PUBLIC_API_STAGES: Tuple[Stage, ...] = (
    Stage.STABLE,
    Stage.EXPERIMENTAL,
    Stage.DEPRECATED,
)

# The categories that `public_apis.md` lists, in order, with their headings.
_PUBLIC_API_MD_CATEGORIES: Tuple[Tuple[str, str], ...] = (
    (_PUBLIC_PYTHON_API_CATEGORY, "Python APIs"),
    ("Environment Variable", "Environment Variables"),
)


def _public_apis_md_markers(stage: Stage) -> Tuple[str, str]:
  """Returns the comments around the generated API list of a stage."""
  return (
      f"<!-- BEGIN GENERATED: {stage.value} APIs -->",
      f"<!-- END GENERATED: {stage.value} APIs -->",
  )


def _wrap_md_list_item(line: str) -> str:
  """Wraps a `public_apis.md` list item like mdformat."""
  return textwrap.fill(
      line,
      width=80,
      subsequent_indent="    ",
      break_long_words=False,
      break_on_hyphens=False,
  )


def _format_public_api_md_item(item: ApiEntry) -> str:
  """Returns the `public_apis.md` list item for a public API."""
  name, accessor = split_property_accessor(item.name)
  line = f"*   `{name}`"
  if accessor:
    line += f" ({accessor})"
  if item.stage == Stage.DEPRECATED:
    if item.category == _PUBLIC_PYTHON_API_CATEGORY and not _is_torch_tpu_name(
        item.name
    ):
      # An upstream PyTorch API, e.g. in `torch.accelerator`, that PyTorch
      # deprecated.
      line += " (deprecated by PyTorch)"
      return _wrap_md_list_item(line)
    if not item.deprecated_in:
      raise ValueError(
          f"{item.name} is deprecated, but the scanner can't tell in which"
          " TorchTPU version."
      )
    line += f" (deprecated in TorchTPU {item.deprecated_in})"
  return _wrap_md_list_item(line)


def _format_internal_apis_md_list(all_items: Sequence[ApiEntry]) -> str:
  """Formats the generated `public_apis.md` list of internal APIs.

  It lists the internal Python APIs in `PUBLIC_NAMESPACES`, i.e. the ones that
  are exposed for PyTorch to find but aren't meant to be called by users, with
  their preferred alternatives.

  Args:
      all_items: API entries, e.g. from `discover_all_apis`.

  Returns:
      The Markdown list.
  """
  items = sorted(
      (
          item
          for item in all_items
          if item.category == _PUBLIC_PYTHON_API_CATEGORY
          and item.stage == Stage.INTERNAL
          and is_in_public_namespace(item.name)
      ),
      key=lambda item: item.name,
  )
  if not items:
    return "There are no internal APIs in public namespaces at this time."
  lines: List[str] = []
  for item in items:
    name, accessor = split_property_accessor(item.name)
    line = f"*   `{name}`"
    if accessor:
      line += f" ({accessor})"
    if item.prefer:
      line += f": use `{item.prefer}` instead"
    lines.append(_wrap_md_list_item(line))
  return "\n".join(lines)


def _format_public_apis_md_list(
    all_items: Sequence[ApiEntry], stage: Stage
) -> str:
  """Formats the generated `public_apis.md` API list of a stage."""
  lines: List[str] = []
  for category, heading in _PUBLIC_API_MD_CATEGORIES:
    items = sorted(
        (
            item
            for item in all_items
            if item.category == category
            and item.stage == stage
            and (
                category != _PUBLIC_PYTHON_API_CATEGORY
                or is_in_public_namespace(item.name)
            )
        ),
        key=lambda item: item.name,
    )
    if items:
      if lines:
        lines.append("")
      lines.extend([f"### {heading}", ""])
      lines.extend(_format_public_api_md_item(item) for item in items)
  if not lines:
    lines.append(f"There are no {stage.value.lower()} APIs at this time.")
  return "\n".join(lines)


def update_public_apis_md(md_text: str, all_items: Sequence[ApiEntry]) -> str:
  """Updates the generated API lists in the `public_apis.md` text.

  The rest of `public_apis.md` is written by hand. Each stable, experimental,
  deprecated and internal API list is between a
  `<!-- BEGIN GENERATED: <Stage> APIs -->` and an
  `<!-- END GENERATED: <Stage> APIs -->` comment. The stable, experimental and
  deprecated lists have the Python APIs in `PUBLIC_NAMESPACES` and the
  environment variables in that stage. The internal list has the internal
  Python APIs in `PUBLIC_NAMESPACES` and their preferred alternatives.

  Args:
      md_text: The current `public_apis.md` text.
      all_items: API entries, e.g. from `discover_all_apis`.

  Returns:
      The `public_apis.md` text with up-to-date API lists.

  Raises:
      ValueError: If `md_text` doesn't contain exactly one API list per stage,
        or if the TorchTPU version that deprecated a deprecated TorchTPU API is
        unknown.
  """
  for stage in (*_PUBLIC_API_STAGES, Stage.INTERNAL):
    begin, end = _public_apis_md_markers(stage)
    pattern = re.compile(
        rf"^{re.escape(begin)}$.*?^{re.escape(end)}$", re.MULTILINE | re.DOTALL
    )
    if stage == Stage.INTERNAL:
      api_list = _format_internal_apis_md_list(all_items)
    else:
      api_list = _format_public_apis_md_list(all_items, stage)
    replacement = f"{begin}\n\n{api_list}\n\n{end}"
    md_text, count = pattern.subn(lambda _, r=replacement: r, md_text)
    if count != 1:
      raise ValueError(
          f"{PUBLIC_APIS_MD} must contain exactly one `{begin}` comment"
          f" followed by an `{end}` comment, but found {count}."
      )
  return md_text


def export_tt_apis_artifact(text: str, filename: str = "tt_apis.yaml") -> None:
  """Writes a generated baseline file to the Sponge test artifacts, if available.

  Args:
      text: The formatted text to export, e.g. from `format_tt_apis_yaml`.
      filename: The artifact's file name.
  """
  test_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
  if test_outputs_dir:
    artifact_path = os.path.join(test_outputs_dir, filename)
    try:
      with open(artifact_path, "w", encoding="utf-8") as artifact_file:
        artifact_file.write(text)
      print(f"Exported Sponge Test Artifact: {artifact_path}")
    except Exception as error:  # pylint: disable=broad-except
      logging.warning("Failed to export Sponge artifact: %s", error)


# How to update `tt_apis.yaml` and `public_apis.md`.
UPDATE_INSTRUCTIONS = f"""\
To update `{repo_paths.TT_APIS_YAML_PATH}` and
`{repo_paths.PUBLIC_APIS_MD_PATH}`, run this on a TPU host, which writes them
into your workspace:
  blaze run {SCAN_TT_APIS_LABEL}
Alternatively, open the Sponge link of `{TT_APIS_TEST_LABEL}`, go to the
"Artifacts" tab, download `{os.path.basename(repo_paths.TT_APIS_YAML_PATH)}`
and `{PUBLIC_APIS_MD}`, and copy them to the paths above, relative to the
TorchTPU repository root."""


def find_out_of_date_baseline_files(
    repository_root: str, all_items: Sequence[ApiEntry]
) -> Dict[str, str]:
  """Returns the up-to-date `tt_apis.yaml` and `public_apis.md`, if they changed.

  Args:
      repository_root: The TorchTPU repository root that has the current files.
      all_items: API entries, e.g. from `discover_all_apis`.

  Returns:
      The up-to-date text of each out-of-date file, keyed by its path relative
      to `repository_root`. A file that is already up to date is omitted.
  """
  items = list(all_items)

  def _read(relative_path: str) -> str:
    with open(
        os.path.join(repository_root, relative_path), encoding="utf-8"
    ) as current_file:
      return current_file.read()

  current = {
      relative_path: _read(relative_path)
      for relative_path in (
          repo_paths.TT_APIS_YAML_PATH,
          repo_paths.PUBLIC_APIS_MD_PATH,
      )
  }
  generated = {
      repo_paths.TT_APIS_YAML_PATH: format_tt_apis_yaml(
          items, find_duplicated_exports(items)
      ),
      repo_paths.PUBLIC_APIS_MD_PATH: update_public_apis_md(
          current[repo_paths.PUBLIC_APIS_MD_PATH], items
      ),
  }
  return {
      relative_path: text
      for relative_path, text in generated.items()
      if current[relative_path] != text
  }


def discover_all_apis(repository_root: str) -> List[ApiEntry]:
  """Discovers all TorchTPU API surfaces.

  Args:
      repository_root: The TorchTPU repository root used for source scanning.

  Returns:
      All discovered API items, ordered by category.
  """
  # 1. Run Runtime Reflection
  public_python_apis, python_flags = discover_python_apis()
  accelerator_apis = discover_accelerator_apis()

  # 2. Run Source Code Scanning
  env_vars, cpp_flags, custom_ops = discover_cpp_source_surfaces(
      repository_root
  )

  return (
      public_python_apis
      + accelerator_apis
      + python_flags
      + env_vars
      + cpp_flags
      + custom_ops
  )


def main(argv: Sequence[str]) -> None:
  """Writes the up-to-date `tt_apis.yaml` and `public_apis.md` into the workspace.

  Args:
      argv: The command line arguments, which must only be the program name.

  Raises:
      app.UsageError: If there are extra arguments, or if this doesn't run with
        `blaze run`, which tells the workspace to update.
  """
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  try:
    root = repo_paths.get_repo_root_in_workspace()
  except FileNotFoundError as error:
    raise app.UsageError(
        f"scan_tt_apis writes into the workspace: {error}"
    ) from error
  out_of_date = find_out_of_date_baseline_files(root, discover_all_apis(root))
  for relative_path, text in out_of_date.items():
    with open(
        os.path.join(root, relative_path), "w", encoding="utf-8"
    ) as output_file:
      output_file.write(text)
  if out_of_date:
    print("Updated:\n" + "\n".join(f"  {path}" for path in out_of_date))
  else:
    print("tt_apis.yaml and public_apis.md are up to date.")


if __name__ == "__main__":
  app.run(main)
