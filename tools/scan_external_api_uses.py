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

r"""Scans arbitrary GitHub or local repositories for TorchTPU API uses.

For each repository, list TorchTPU APIs in these categories:
- Internal: `_internal` modules, private names, or `stage: Internal` in
  `tt_apis.yaml`.
- Deprecated: `stage: Deprecated` in `tt_apis.yaml`.
- Unsupported: not in `tt_apis.yaml`.
- Experimental: public APIs with `stage: Experimental`.

A use is found:
- directly, e.g. `torch.tpu.synchronize()` or `getattr(torch.tpu, "name")`,
  also in strings and non-Python files;
- through `torch.get_device_module()`, a helper that returns it, or
  `getattr(torch, device)`;
- through a variable or `self` attribute that holds one of the above, e.g.
  `self.device_module = torch.get_device_module(d)`.

Environment variables and C++ flags in `tt_apis.yaml` are found as words in
any scanned file.

Each `--repo` is either:
- a GitHub (or any Git) URL, e.g. `https://github.com/vllm-project/vllm-torchtpu`,
  which is shallow-cloned into a temporary directory that is deleted
  afterwards. This needs network access, so run the tool directly, not as a
  test. `--ref` selects a branch, tag or commit SHA; by default it's the HEAD
  of the default branch.
- a local directory, e.g. `~/src/vllm`, relative to the directory the tool was
  started in (or absolute). It's scanned as is.

USAGE:
  scan_external_api_uses \
    --repo=https://github.com/vllm-project/vllm-torchtpu \
    --output=/tmp/torch_tpu_api_uses.yaml

  scan_external_api_uses \
    --repo=https://github.com/sgl-project/sglang-torchtpu --ref=<branch|tag|sha>
"""

import ast
import collections
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import enum
import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import types
from typing import Any
import warnings

from absl import app
from absl import flags
from absl import logging
import torch_tpu
from torch_tpu._internal.utils import annotations
from torch_tpu.tests import repo_paths
from torch_tpu.tests.api import scan_tt_apis
import yaml

ApiEntry = scan_tt_apis.ApiEntry
Stage = annotations.Stage

_REPO = flags.DEFINE_multi_string(  # PYTHON_FLAG_OK=Stand-alone CLI tool.
    "repo",
    [],
    "Required. GitHub (or any Git) URL, or local directory, to scan. Repeat"
    " to scan several repositories.",
)
_REF = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone CLI tool.
    "ref",
    None,
    "Branch, tag or commit SHA to check out. Only with a single Git URL;"
    " defaults to the HEAD of the default branch.",
)
_OUTPUT = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone CLI tool.
    "output",
    os.path.join(tempfile.gettempdir(), "torch_tpu_api_uses.yaml"),
    "YAML file to write the report to.",
)


class Category(enum.Enum):
  """How a TorchTPU API use is reported, in report order."""

  INTERNAL = "Internal"
  DEPRECATED = "Deprecated"
  UNSUPPORTED = "Unsupported (not in tt_apis.yaml)"
  EXPERIMENTAL = "Experimental"


@dataclasses.dataclass(frozen=True)
class Finding:
  """A TorchTPU API, environment variable or C++ flag that a repository uses.

  Attributes:
    name: The API or variable, e.g. "torch_tpu._internal.compile".
    kind: "Python API", "Environment Variable" or "ABSL C++ Flag".
    category: How it's reported.
    reason: Why it's reported, e.g. "Internal in tt_apis.yaml".
    prefer: The API to use instead, e.g. "torch.accelerator.synchronize", or
      None.
    files: The files that use it, relative to the repository root. Lines aren't
      listed, so that reports don't change with unrelated edits.
  """

  name: str
  kind: str
  category: Category
  reason: str
  prefer: str | None
  files: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RepoScanResult:
  """The result of scanning one repository."""

  repo: str
  commit: str | None
  findings: list[Finding]
  num_files: int
  num_files_using_non_stable_apis: int
  parse_errors: list[str]


# How a use is reported: its category, the reason, and the API to use instead,
# if any.
Classification = tuple[Category, str, str | None]


# Namespaces of TorchTPU Python APIs.
_TORCH_TPU_NAMESPACES = (
    "torch.ops.tpu",
    "torch.tpu",
    "torch_tpu",
)
_INTERNAL_ENV_PREFIX = "TORCH_TPU_INTERNAL_"
# TorchTPU API names in non-Python files and Python strings.
_API_TEXT_RE = re.compile(
    r"(?<![\w/.-])(?:torch_tpu|torch\.tpu|torch\.ops\.tpu)" r"(?:\.\w+)+"
)
_SCANNED_EXTENSIONS = (
    ".py",
    ".ipynb",
    ".md",
    ".rst",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
)
_EXCLUDED_DIR_NAMES = frozenset({
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
})
_GH_URL_PREFIXES = ("https://", "http://", "ssh://", "git@", "file://")
_GH_URL_RE = re.compile(
    r"^(?:https://|ssh://git@|git@)github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$"
)
# Device-generic code gets the module of the current accelerator from
# `torch.get_device_module()`, or from a helper of the same name that wraps it.
_DEVICE_MODULE_FUNCTION = "get_device_module"
_DEVICE_MODULE = "torch.tpu"
# The scopes of variables: functions and the module. A `self` attribute is
# scoped to its class.
_SCOPE_NODES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
# Every TorchTPU API name contains "tpu", or is reached through the device
# module; files without any of these aren't parsed.
_MAY_USE_TORCH_TPU_RE = re.compile(
    rf"tpu|{_DEVICE_MODULE_FUNCTION}|getattr\(\s*torch\s*,"
)
# `getattr(torch, <expression>)` is the device module if the expression is
# device-like, e.g. `getattr(torch, device)`, not e.g. `getattr(torch, dtype)`.
_DEVICE_LIKE_RE = re.compile(r"device|platform|accelerator|backend", re.I)
# Keys of the categories in the YAML report, in report order.
_CATEGORY_KEYS = {category: category.name.lower() for category in Category}


def load_tt_apis(
    path: str,
) -> tuple[dict[str, ApiEntry], dict[str, ApiEntry], dict[str, ApiEntry]]:
  """Loads a tuple of (python_apis, env_vars, cpp_flags) mapping names to their entries from `tt_apis.yaml`."""
  python_apis: dict[str, ApiEntry] = {}
  env_vars: dict[str, ApiEntry] = {}
  cpp_flags: dict[str, ApiEntry] = {}
  stages = {stage.value: stage for stage in Stage}
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      for name, info in (yaml.safe_load(line) or {}).items():
        if not isinstance(info, dict):
          continue
        category = str(info.get("category", ""))
        if category == "Environment Variable":
          apis = env_vars
        elif category == "ABSL C++ Flag":
          apis = cpp_flags
        else:
          apis = python_apis
        name = re.sub(r" (getter|setter)$", "", str(name))
        apis[name] = ApiEntry(
            name=name,
            category=category,
            stage=stages.get(str(info.get("stage", ""))),
            defined_in=info.get("defined_in", ""),
            deprecated_in=info.get("deprecated_in"),
            prefer=info.get("prefer"),
        )
  return python_apis, env_vars, cpp_flags


def _namespace_of(name: str) -> str | None:
  """Returns the TorchTPU namespace that `name` is in, or None."""
  for namespace in _TORCH_TPU_NAMESPACES:
    if name == namespace or name.startswith(f"{namespace}."):
      return namespace
  return None


def api_symbol(name: str) -> str:
  """Returns the TorchTPU API that `name` uses, without attributes of it.

  For example, `torch_tpu._internal.pallas.pallas.JAX_TO_TORCH_DTYPE_MAP.get`
  uses `torch_tpu._internal.pallas.pallas.JAX_TO_TORCH_DTYPE_MAP` (a dict), and
  `torch_tpu._internal.execution_mode.eager_mode.name` uses
  `torch_tpu._internal.execution_mode.eager_mode` (an enum value).

  Dunder attributes, e.g. `__call__`, are dropped. A `torch_tpu` name is
  resolved in the installed TorchTPU and cut after the first object that isn't
  a module or class. If it can't be resolved, e.g. TorchTPU removed it, only
  dunder attributes are dropped.

  Args:
    name: A TorchTPU Python API name, e.g. "torch_tpu._internal.sync.sync".

  Returns:
    The TorchTPU API name, without attributes of it.
  """
  parts = name.split(".")
  dunders = [i for i, p in enumerate(parts) if i and p.startswith("__")]
  if dunders:
    parts = parts[: dunders[0]]
  if parts[0] != "torch_tpu":
    return ".".join(parts)
  obj = torch_tpu
  for end in range(2, len(parts) + 1):
    try:
      if hasattr(obj, parts[end - 1]):
        obj = getattr(obj, parts[end - 1])
      else:  # A submodule that isn't imported yet.
        obj = importlib.import_module(".".join(parts[:end]))
    except Exception:  # pylint: disable=broad-except
      break
    if not isinstance(obj, (types.ModuleType, type)):
      return ".".join(parts[:end])
  return ".".join(parts)


def _classify_by_stage(entry: ApiEntry) -> Classification | None:
  """Classifies an API by its stage, or returns None if it's public."""
  if entry.stage == Stage.DEPRECATED:
    since = f" in {entry.deprecated_in}" if entry.deprecated_in else ""
    return Category.DEPRECATED, f"Deprecated{since}", entry.prefer
  if entry.stage == Stage.INTERNAL:
    return Category.INTERNAL, "Internal in tt_apis.yaml", entry.prefer
  if entry.stage == Stage.EXPERIMENTAL:
    return Category.EXPERIMENTAL, "Experimental", entry.prefer
  return None


def classify_python_api(
    name: str, python_apis: Mapping[str, ApiEntry]
) -> Classification | None:
  """Classifies a use of a TorchTPU Python API, or returns None if fine."""
  namespace = _namespace_of(name)
  if namespace is None or name == namespace:
    return None
  parts = name[len(namespace) + 1 :].split(".")
  if "_internal" in parts:
    return Category.INTERNAL, "`_internal` module", None
  if any(p.startswith("_") and not p.startswith("__") for p in parts):
    return Category.INTERNAL, "private name", None
  if name in python_apis:
    return _classify_by_stage(python_apis[name])
  # A module or class that contains public APIs.
  if any(key.startswith(f"{name}.") for key in python_apis):
    return None
  # A public `torch.accelerator` API of the same name, e.g.
  # `torch.tpu.empty_cache` -> `torch.accelerator.empty_cache`.
  accelerator_api = f"torch.accelerator.{'.'.join(parts)}"
  entry = python_apis.get(accelerator_api)
  if (
      namespace == "torch.tpu"
      and entry
      and entry.stage not in (Stage.DEPRECATED, Stage.INTERNAL)
  ):
    return Category.UNSUPPORTED, "not a public API", accelerator_api
  return Category.UNSUPPORTED, "not a public API", None


def classify_env_var(
    name: str, env_vars: Mapping[str, ApiEntry]
) -> Classification | None:
  """Classifies a use of an environment variable, or returns None if fine.

  - A variable in `tt_apis.yaml`, which lists every variable that TorchTPU
    reads, is classified by its stage.
  - A `TORCH_TPU_INTERNAL_*` variable that isn't in it is Unsupported:
    TorchTPU removed or renamed it.
  - Other variables aren't TorchTPU's, e.g. a downstream repository's own
    `TORCH_TPU_VERSION`, so they aren't reported.
  """
  entry = env_vars.get(name)
  if entry is None:
    if name.startswith(_INTERNAL_ENV_PREFIX):
      return (
          Category.UNSUPPORTED,
          "removed or renamed; TorchTPU doesn't read it",
          None,
      )
    return None
  return _classify_by_stage(entry)


def _is_device_module_call(
    node: ast.AST | None, functions: Container[str]
) -> bool:
  """Returns whether `node` is a call that returns the device module.

  E.g. `torch.get_device_module()`, `torch.get_device_module(x.device)` or
  `_device_module()`, but not `torch.get_device_module("cpu")`.

  Args:
    node: The expression to check, e.g. `torch.get_device_module()`.
    functions: Names of functions that return the device module, e.g.
      `{"get_device_module", "_device_module"}`.
  """
  if not isinstance(node, ast.Call):
    return False
  func = node.func
  if isinstance(func, ast.Attribute):
    name = func.attr
  elif isinstance(func, ast.Name):
    name = func.id
  else:
    return False
  device = _str_constant(node.args[0]) if node.args else None
  return name in functions and (device is None or device.startswith("tpu"))


def _getattr_args(node: ast.AST) -> tuple[ast.expr, ast.expr] | None:
  """Returns `(obj, name)` of `getattr(obj, name, ...)`, or None."""
  if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id in ("getattr", "hasattr")
      and len(node.args) >= 2
  ):
    return node.args[0], node.args[1]
  return None


def _str_constant(node: ast.AST) -> str | None:
  if isinstance(node, ast.Constant) and isinstance(node.value, str):
    return node.value
  return None


def _assignments(
    nodes: Iterable[ast.AST],
) -> Iterator[tuple[ast.expr, ast.expr]]:
  """Yields `(target, value)` of the `x = v` and `x: T = v` in `nodes`."""
  for node in nodes:
    if isinstance(node, ast.Assign):
      for target in node.targets:
        yield target, node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
      yield node.target, node.value


def _nodes_in_scope(scope: ast.AST) -> Iterator[ast.AST]:
  """Yields the nodes of a scope, without those of nested scopes."""
  stack = list(ast.iter_child_nodes(scope))
  while stack:
    node = stack.pop()
    yield node
    if not isinstance(node, (*_SCOPE_NODES, ast.ClassDef)):
      stack.extend(ast.iter_child_nodes(node))


class _ApiNameCollector(ast.NodeVisitor):
  """Collects the dotted names that Python code uses.

  A name is resolved to its full name through:

  - imports, e.g. `tt._internal.x` after `import torch_tpu as tt`;
  - the device module, e.g. `torch.get_device_module().x` is `torch.tpu.x`;
  - variables and `self` attributes assigned one of these, if all of their
    assignments in their scope agree;
  - `getattr(obj, "x")`, and API names in strings.

  Other objects, e.g. what a function returns, aren't tracked.
  """

  def __init__(self, tree: ast.AST):
    self.names: list[str] = []
    self.aliases: dict[str, str] = {}
    for node in ast.walk(tree):
      if isinstance(node, (ast.Import, ast.ImportFrom)):
        self._add_import_aliases(node)
    # Functions that return the device module: `get_device_module`, and
    # helpers in the file that only return what it returns, e.g.
    # `def _device_module(): return torch.get_device_module(device)`.
    self._device_module_functions = {_DEVICE_MODULE_FUNCTION}
    for node in ast.walk(tree):
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        returns = [
            n.value for n in _nodes_in_scope(node) if isinstance(n, ast.Return)
        ]
        if returns and all(
            _is_device_module_call(value, self._device_module_functions)
            for value in returns
        ):
          self._device_module_functions.add(node.name)
    # The enclosing scopes while visiting, each with the TorchTPU module that
    # its names hold (None if a name holds something else).
    self._scopes: list[tuple[ast.AST, dict[str, str | None]]] = []

  def _add_import_aliases(self, node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
      for alias in node.names:
        # `import a.b` binds `a`.
        name = alias.name if alias.asname else alias.name.split(".")[0]
        self.aliases[alias.asname or name] = name
    elif node.module and not node.level:  # Relative imports aren't TorchTPU.
      for alias in node.names:
        self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

  # Scopes: which TorchTPU module each variable holds, per function or class.

  def visit(self, node: ast.AST):
    if not isinstance(node, (*_SCOPE_NODES, ast.ClassDef)):
      return super().visit(node)
    self._scopes.append((node, self._bindings(node)))
    try:
      return super().visit(node)
    finally:
      self._scopes.pop()

  def _bindings(self, scope: ast.AST) -> dict[str, str | None]:
    """Returns the TorchTPU module that each name bound in `scope` holds.

    Assigned values are resolved in the enclosing scopes, so `b = a` after
    `a = torch.tpu` in the same scope isn't resolved.
    """
    names: dict[str, str | None] = {}

    def bind(name: str, value: ast.AST | None) -> None:
      held = self._resolve(value) if value is not None else None
      if held is not None and _namespace_of(held) is None:
        held = None  # Not a TorchTPU name.
      # A name holds a module only if all of its assignments agree.
      names[name] = held if names.get(name, held) == held else None

    if isinstance(scope, ast.ClassDef):
      # `self.x = v` in any method.
      for target, value in _assignments(ast.walk(scope)):
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
          bind(f"self.{target.attr}", value)
      return names
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
      # Parameters hold unknown objects.
      for arg in ast.walk(scope.args):
        if isinstance(arg, ast.arg):
          bind(arg.arg, None)
    nodes = list(_nodes_in_scope(scope))
    values = {id(target): value for target, value in _assignments(nodes)}
    # Every name that the scope stores, e.g. in `x = v`, `x, y = v`,
    # `for x in ...` or `with ... as x`. `a.b = v` (e.g. a monkey-patch)
    # doesn't rebind `a`.
    for node in nodes:
      if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        bind(node.id, values.get(id(node)))
    return names

  def _lookup(self, name: str) -> str | None:
    """Returns the TorchTPU module that `name` holds in its innermost scope."""
    is_self_attribute = name.startswith("self.")
    for scope, names in reversed(self._scopes):
      if isinstance(scope, ast.ClassDef) != is_self_attribute:
        # Variables are function-scoped, `self` attributes class-scoped.
        continue
      if name in names:
        return names[name]
    return None

  # Resolution: the full name that an expression refers to.

  def _resolve(self, node: ast.AST) -> str | None:
    """Returns the full name that an expression refers to, if it's known.

    E.g. `torch_tpu._internal.x.y` for `tt._internal.x.y` after
    `import torch_tpu as tt`, or `torch.tpu.synchronize` for
    `torch.get_device_module().synchronize`.
    """
    parts = []
    while isinstance(node, ast.Attribute):
      if isinstance(node.value, ast.Name) and node.value.id == "self":
        if held := self._lookup(f"self.{node.attr}"):
          return ".".join([held, *reversed(parts)])
      parts.append(node.attr)
      node = node.value
    if isinstance(node, ast.Name):
      base = self._lookup(node.id) or self.aliases.get(node.id)
    elif _is_device_module_call(node, self._device_module_functions):
      base = _DEVICE_MODULE
    elif args := _getattr_args(node):
      obj, attr = args
      if (attr_name := _str_constant(attr)) is not None:
        base = self._resolve(obj)
        base = base and f"{base}.{attr_name}"
      elif self._resolve(obj) == "torch" and _DEVICE_LIKE_RE.search(
          ast.unparse(attr)
      ):
        base = _DEVICE_MODULE
      else:
        base = None
    else:
      base = None
    return ".".join([base, *reversed(parts)]) if base else None

  # Visitors: collect the names, by node type.

  def visit_Import(self, node: ast.Import):
    for alias in node.names:
      self.names.append(alias.name)

  def visit_ImportFrom(self, node: ast.ImportFrom):
    if node.module and not node.level:
      for alias in node.names:
        self.names.append(f"{node.module}.{alias.name}")

  def visit_Attribute(self, node: ast.Attribute):
    # Only the outermost attribute of a chain, e.g. `a.b.c`, not `a.b`.
    if (name := self._resolve(node)) is not None:
      self.names.append(name)
      return
    self.generic_visit(node)

  def visit_Call(self, node: ast.Call):
    if _getattr_args(node) and (name := self._resolve(node)) is not None:
      self.names.append(name)
    self.generic_visit(node)

  def visit_Constant(self, node: ast.Constant):
    if isinstance(node.value, str):
      self.names += _API_TEXT_RE.findall(node.value)


def _api_names_in_python(code: str) -> list[str]:
  """Returns the TorchTPU API names in Python code."""
  with warnings.catch_warnings():
    # E.g. invalid escape sequences in other repositories' code.
    warnings.simplefilter("ignore", SyntaxWarning)
    tree = ast.parse(code)
  collector = _ApiNameCollector(tree)
  collector.visit(tree)
  return [name for name in collector.names if _namespace_of(name)]


def _api_names_in_text(text: str) -> list[str]:
  """Returns the TorchTPU API names in text, e.g. Markdown."""
  return _API_TEXT_RE.findall(text)


def _api_names_in_file(rel_path: str, text: str) -> list[str]:
  """Returns the TorchTPU API names in a file.

  Args:
    rel_path: The file's path relative to the repository root.
    text: The file's contents.

  Raises:
    SyntaxError: If a Python file or notebook code cell can't be parsed.
    ValueError: If a notebook isn't valid JSON.
  """
  if rel_path.endswith(".py"):
    return _api_names_in_python(text)
  if not rel_path.endswith(".ipynb"):
    return _api_names_in_text(text)
  names = []
  notebook = json.loads(text)
  cells = notebook.get("cells", []) if isinstance(notebook, dict) else None
  if not isinstance(cells, list) or not all(isinstance(c, dict) for c in cells):
    raise ValueError("not a Jupyter notebook")
  for cell in cells:
    source = cell.get("source", "")
    source = "".join(source) if isinstance(source, list) else source
    if cell.get("cell_type") == "code":
      # Comment out IPython magics and shell commands.
      source = re.sub(r"^(\s*[%!])", r"#\1", source, flags=re.M)
      names += _api_names_in_python(source)
    else:
      names += _api_names_in_text(source)
  return names


def _env_var_re(env_vars: Mapping[str, ApiEntry]) -> re.Pattern[str]:
  """Matches the variables in `tt_apis.yaml`, and internal ones that aren't.

  A wildcard such as `TORCH_TPU_INTERNAL_TIER*` in documentation isn't a
  variable, so it isn't matched.
  """
  patterns = [f"{_INTERNAL_ENV_PREFIX}[A-Z0-9_]+"]
  patterns += [re.escape(name) for name in env_vars]
  return re.compile(rf"\b(?:{'|'.join(patterns)})\b(?!\*)")


def _cpp_flag_re(cpp_flags: Mapping[str, ApiEntry]) -> re.Pattern[str]:
  """Matches the C++ flags in `tt_apis.yaml`, e.g. in `--flag=1`."""
  # `(?!)` never matches, if there are no flags.
  names = "|".join(map(re.escape, cpp_flags)) or "(?!)"
  return re.compile(rf"\b(?:{names})\b")


def _iter_files(root: str) -> Iterator[str]:
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d not in _EXCLUDED_DIR_NAMES)
    for filename in sorted(filenames):
      if filename.endswith(_SCANNED_EXTENSIONS) or filename == "Dockerfile":
        yield os.path.join(dirpath, filename)


def scan_repo(
    root: str,
    python_apis: Mapping[str, ApiEntry],
    env_vars: Mapping[str, ApiEntry],
    cpp_flags: Mapping[str, ApiEntry],
    repo: str = "",
    commit: str | None = None,
) -> RepoScanResult:
  """Scans a checked-out repository for uses of TorchTPU APIs.

  Args:
    root: The repository root.
    python_apis: The Python APIs in `tt_apis.yaml`, from `load_tt_apis`.
    env_vars: The environment variables in `tt_apis.yaml`.
    cpp_flags: The C++ flags in `tt_apis.yaml`.
    repo: The repository name or URL, for the result.
    commit: The scanned commit, for the result.

  Returns:
    The findings and scan statistics.
  """
  # Environment variables and C++ flags are matched as words in every file,
  # e.g. in `export NAME=1` or `--name=1`.
  word_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
      ("Environment Variable", _env_var_re(env_vars)),
      ("ABSL C++ Flag", _cpp_flag_re(cpp_flags)),
  )
  # (kind, name) -> files.
  uses: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
  num_files = 0
  parse_errors: list[str] = []

  for path in _iter_files(root):
    rel_path = os.path.relpath(path, root)
    try:
      with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    except OSError as e:
      parse_errors.append(f"{rel_path}: {e}")
      continue
    num_files += 1

    api_names = []
    # Other files can't use TorchTPU APIs, so they aren't parsed.
    if _MAY_USE_TORCH_TPU_RE.search(text):
      try:
        api_names = _api_names_in_file(rel_path, text)
      except (SyntaxError, ValueError) as e:
        parse_errors.append(f"{rel_path}: {type(e).__name__}: {e}")
        # Still report the API names in the file.
        api_names = _api_names_in_text(text)
    for name in api_names:
      uses[("Python API", api_symbol(name))].add(rel_path)
    for kind, pattern in word_patterns:
      for name in pattern.findall(text):
        uses[(kind, name)].add(rel_path)

  findings = []
  for (kind, name), files in uses.items():
    if kind == "Python API":
      classification = classify_python_api(name, python_apis)
    elif kind == "Environment Variable":
      classification = classify_env_var(name, env_vars)
    else:
      classification = _classify_by_stage(cpp_flags[name])
    if classification is None:
      continue
    category, reason, prefer = classification
    findings.append(
        Finding(
            name=name,
            kind=kind,
            category=category,
            reason=reason,
            prefer=prefer,
            files=tuple(sorted(files)),
        )
    )
  order = list(Category)
  findings.sort(key=lambda f: (order.index(f.category), f.kind, f.name))
  return RepoScanResult(
      repo=repo,
      commit=commit,
      findings=findings,
      num_files=num_files,
      num_files_using_non_stable_apis=len(
          {file for f in findings for file in f.files}
      ),
      parse_errors=parse_errors,
  )


def _git(*args: str) -> str:
  """Runs `git` and returns its stdout.

  Raises:
    RuntimeError: If `git` fails, with its stderr.
  """
  logging.info("Running: git %s", " ".join(args))
  result = subprocess.run(
      ["git", *args], capture_output=True, text=True, check=False
  )
  if result.returncode != 0:
    raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
  return result.stdout.strip()


def _head_commit(path: str) -> str | None:
  """Returns the checked-out commit of a Git repository, or None."""
  if not os.path.exists(os.path.join(path, ".git")):
    return None
  try:
    return _git("-C", path, "rev-parse", "HEAD")
  except RuntimeError:
    return None


@contextlib.contextmanager
def checkout_repo(
    repo: str, ref: str | None = None
) -> Iterator[tuple[str, str | None]]:
  """Yields `(path, commit)` of `repo` checked out at `ref`.

  A Git URL is shallow-cloned into a temporary directory, which is deleted on
  exit. A local directory is used as is.

  Args:
    repo: A Git URL or a local directory.
    ref: A branch, tag or commit SHA, for a Git URL only.

  Raises:
    FileNotFoundError: If a local `repo` isn't a directory.
    ValueError: If `ref` is given for a local directory.
    RuntimeError: If cloning or checking out fails.
  """
  if not repo.startswith(_GH_URL_PREFIXES):
    if ref:
      raise ValueError(
          "--ref only applies to a Git URL, not a local directory."
      )
    path = _from_working_dir(repo)
    if not os.path.isdir(path):
      raise FileNotFoundError(f"Not a directory: {path}")
    yield path, _head_commit(path)
    return

  with tempfile.TemporaryDirectory(prefix="scan_external_api_uses_") as tmp:
    path = os.path.join(tmp, "repo")
    if not ref:
      _git("clone", "--depth", "1", repo, path)
    else:
      try:
        _git("clone", "--depth", "1", "--branch", ref, repo, path)
      except RuntimeError:
        # `--branch` doesn't accept a commit SHA: fetch the history without
        # file contents, then check out the commit.
        shutil.rmtree(path, ignore_errors=True)
        _git("clone", "--filter=blob:none", "--no-checkout", repo, path)
        _git("-C", path, "checkout", ref)
    yield path, _head_commit(path)


def _counts(result: RepoScanResult) -> dict[Category, tuple[int, int]]:
  """Returns `(symbols, files)` per category, counting each file once."""
  findings = {category: [] for category in Category}
  for f in result.findings:
    findings[f.category].append(f)
  return {
      category: (len(fs), len({file for f in fs for file in f.files}))
      for category, fs in findings.items()
  }


def format_summary(results: Sequence[RepoScanResult]) -> str:
  """Formats the symbols / files per category as a Markdown table."""
  header = " | ".join(c.value for c in Category)
  lines = [
      f"| Repository | Commit | Files using non-Stable APIs | {header} |",
      "| :--- | :--- | ---: |" + " ---: |" * len(Category),
  ]
  for result in results:
    counts = _counts(result)
    cells = " | ".join(f"{s} / {u}" for s, u in counts.values())
    lines.append(
        f"| {result.repo} | {(result.commit or 'n/a')[:12]} |"
        f" {result.num_files_using_non_stable_apis} | {cells} |"
    )
  return "\n".join(lines)


def _finding_attributes(f: Finding) -> scan_tt_apis.FlowDict:
  """Returns the YAML attributes of a finding."""
  attributes = scan_tt_apis.FlowDict({"kind": f.kind, "reason": f.reason})
  if f.prefer:
    attributes["prefer"] = f.prefer
  attributes["used_in"] = list(f.files)
  return attributes


def format_yaml(results: Sequence[RepoScanResult]) -> str:
  """Formats the scan results as a YAML report, one entry per repository."""
  report = []
  for result in results:
    counts = _counts(result)
    entry: dict[str, Any] = {
        "repo": result.repo,
        "commit": result.commit,
        "files_scanned": result.num_files,
        "files_using_non_stable_apis": result.num_files_using_non_stable_apis,
        "summary": scan_tt_apis.FlowDict({
            _CATEGORY_KEYS[category]: scan_tt_apis.FlowDict(
                {"symbols": symbols, "files": files}
            )
            for category, (symbols, files) in counts.items()
        }),
    }
    for category in Category:
      entry[_CATEGORY_KEYS[category]] = {
          f.name: _finding_attributes(f)
          for f in result.findings
          if f.category == category
      }
    entry["parse_errors"] = list(result.parse_errors)
    report.append(entry)
  header = (
      "# TorchTPU API uses, reported by scan_external_api_uses; see its\n"
      "# module docstring for the categories. `used_in` lists the files,\n"
      "# relative to the repository root at `commit`.\n"
  )
  return header + yaml.safe_dump(report, width=float("inf"), sort_keys=False)


def _from_working_dir(path: str) -> str:
  """Resolves `path` relative to the directory the tool was started in."""
  return os.path.join(
      os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()),
      os.path.expanduser(path),
  )


def _tt_apis_path() -> str:
  """Returns `tt_apis.yaml` in the runfiles, or else in the workspace.

  `blaze run` doesn't set `RUNFILES_DIR`, so the runfiles aren't found there;
  it sets `BUILD_WORKSPACE_DIRECTORY` instead.
  """
  try:
    root = repo_paths.get_repo_root_in_runfiles()
  except FileNotFoundError:
    root = repo_paths.get_repo_root_in_workspace()
  return os.path.join(root, repo_paths.TT_APIS_YAML_PATH)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not _REPO.value:
    raise app.UsageError("--repo is required.")
  if _REF.value and len(_REPO.value) > 1:
    raise app.UsageError("--ref only applies to a single --repo.")

  python_apis, env_vars, cpp_flags = load_tt_apis(_tt_apis_path())
  results = []
  for repo in _REPO.value:
    with checkout_repo(repo, _REF.value) as (path, commit):
      results.append(
          scan_repo(
              path, python_apis, env_vars, cpp_flags, repo=repo, commit=commit
          )
      )

  output = _from_working_dir(_OUTPUT.value)
  with open(output, "w", encoding="utf-8") as f:
    f.write(format_yaml(results))
  print(format_summary(results))
  print(f"\nWrote the report to {output}")


if __name__ == "__main__":
  app.run(main)
