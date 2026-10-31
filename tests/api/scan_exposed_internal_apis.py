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

r"""TorchTPU Internal API Scanner.

This tool audits usages of TorchTPU internal APIs and internal environment
variables across user-facing documentation (e.g. `docs/notebooks/` of TorchTPU),
examples (e.g. `examples/`), and their associated tests.

It uses AST parsing, regex text scanning, and runtime stage reflection to
discover:
1. Internal Python APIs/modules (e.g. `torch_tpu._internal.*`, `torch.tpu._*`).
2. Internal environment variables (e.g. `TORCH_TPU_INTERNAL_*`, or marked
Internal in `tt_apis.yaml`).

Discovered symbols are cross-referenced with `tt_apis.yaml` and runtime
lifecycle
stage metadata to track their call sites and ensure internal interfaces are not
inappropriately surfaced.

USAGE:
  `blaze test` the `tests/api:exposed_internal_apis_test` target of TorchTPU.

TO UPDATE BASELINE (exposed_internal_apis.yaml):
  1. `blaze run` the `tests/api:scan_exposed_internal_apis` target of TorchTPU
     with `--update_baseline`.
  2. Alternatively, download `exposed_internal_apis.yaml` from the Sponge
     artifacts of the test above and copy it to `tests/api/` of TorchTPU in your
     workspace.
"""

import ast
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import importlib
import inspect
import json
import os
import re
from typing import Any

from absl import app
from absl import flags
from absl import logging
from tests import repo_paths
import yaml


def _get_pytorch_native_attrs() -> frozenset[str]:
  """Dynamically reflects native attributes and methods from PyTorch classes."""
  try:
    import torch  # pylint: disable=g-import-not-at-top

    attrs: set[str] = set()
    submodules = [
        torch,
        getattr(torch, "nn", None),
        getattr(torch, "library", None),
        getattr(torch, "autograd", None),
        getattr(getattr(torch, "autograd", None), "function", None),
    ]
    for mod in submodules:
      if mod is None:
        continue
      for obj in mod.__dict__.values():
        if inspect.isclass(obj):
          attrs.update(dir(obj))
    return frozenset(attrs)
  except ImportError:
    return frozenset()


_PYTORCH_NATIVE_ATTRS: frozenset[str] = _get_pytorch_native_attrs()


def _resolve_runtime_object(target_name: str) -> Any | None:
  """Attempts to dynamically resolve a symbol path to a live runtime object."""
  try:
    parts = target_name.split(".")
    for i in range(len(parts), 0, -1):
      mod_name = ".".join(parts[:i])
      try:
        obj = importlib.import_module(mod_name)
        for attr in parts[i:]:
          obj = getattr(obj, attr)
        return obj
      except (ImportError, AttributeError):
        continue
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  return None


class Stage(str, enum.Enum):
  """Lifecycle stages for TorchTPU APIs."""

  INTERNAL = "Internal"
  EXPERIMENTAL = "Experimental"
  STABLE = "Stable"
  DEPRECATED = "Deprecated"


# The OSS label of the binary that updates `exposed_internal_apis.yaml`.
SCAN_EXPOSED_INTERNAL_APIS_LABEL = (
    f"//{repo_paths.SCANNER_PACKAGE}:scan_exposed_internal_apis"
)

# The google3 label of the same binary.
_GOOGLE3_SCAN_EXPOSED_INTERNAL_APIS_LABEL = (
    f"//{repo_paths.G3_REPO_PATH}{SCAN_EXPOSED_INTERNAL_APIS_LABEL[1:]}"
)

# The directories to scan, relative to the TorchTPU repository root. Keep in
# sync with the `exposed_internal_apis_data` filegroup in BUILD.
DEFAULT_SCAN_SUBDIRS: tuple[str, ...] = (
    "docs/notebooks",
    "docs/source",
    "examples",
)

# How to update `exposed_internal_apis.yaml`.
UPDATE_INSTRUCTIONS = f"""\
To update `{repo_paths.EXPOSED_INTERNAL_APIS_YAML_PATH}`, run this, which writes
it into your workspace:
  In google3: blaze run {_GOOGLE3_SCAN_EXPOSED_INTERNAL_APIS_LABEL} -- --update_baseline
  In OSS: bazel run {SCAN_EXPOSED_INTERNAL_APIS_LABEL} -- --update_baseline
Alternatively, open the Sponge link of the test, go to the "Artifacts" tab,
download `{os.path.basename(repo_paths.EXPOSED_INTERNAL_APIS_YAML_PATH)}`, and
copy it to the path above, relative to the TorchTPU repository root."""

_TORCH_TPU_NAMESPACES: tuple[str, ...] = (
    "torch.tpu.",
    "torch_tpu.",
    "torch.ops.tpu.",
    "torch.ops.torch_tpu.",
)
_ROOT_NAMESPACES: tuple[str, ...] = (
    "torch",
    "torch.tpu",
    "torch_tpu",
    "torch.ops",
    "torch.ops.tpu",
    "torch.ops.torch_tpu",
)

# Internal implementation environment variables from csrc/common/env_vars.h
_INTERNAL_IMPLEMENTATION_ENV_VARS: frozenset[str] = frozenset({
    "ACCELERATOR_TYPE",
    "ALLOW_MULTIPLE_LIBTPU_LOAD",
    "CLOUD_TPU_TASK_ID",
    "MEGASCALE_DEBUG_PORT",
    "MEGASCALE_PORT",
    "SKIP_MEGASCALE_PJRT_CLIENT",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_BOUNDS",
    "TPU_PROCESS_PORT",
    "TPU_VISIBLE_CHIPS",
})


def _is_torch_tpu_name(name: str) -> bool:
  """Returns whether `name` names a TorchTPU module or one of its members."""
  return any(
      name == ns[:-1] or name.startswith(ns) for ns in _TORCH_TPU_NAMESPACES
  )


_UPDATE_BASELINE = flags.DEFINE_boolean(
    "update_baseline",
    False,
    "Whether to update the exposed_internal_apis.yaml golden file in the"
    " workspace directory.",
)


# ==============================================================================
# Data Structures & YAML Serialization
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class InternalApiEntry:
  """Represents an internal API or environment variable referenced in scanned files.

  Attributes:
    name: The full identifier name of the API surface or env var, e.g.
      "torch_tpu._internal.execution_mode" or
      "TORCH_TPU_INTERNAL_HANDSHAKE_PORT".
    category: Category classification, e.g. "Internal Python Module",
      "Environment Variable".
    stage: The lifecycle stage annotation from tt_apis.yaml, e.g.
      Stage.INTERNAL.
    used_in: List of relative file paths where the symbol is referenced (e.g.
      "docs/notebooks/eager_mode.py").
  """

  name: str
  category: str
  stage: Stage
  used_in: list[str]


class FlowDict(dict):
  """A dict that forces inline/flow style mapping in PyYAML."""


def flow_dict_representer(dumper, data):
  sorted_data = dict(sorted(data.items()))
  return dumper.represent_mapping(
      "tag:yaml.org,2002:map", sorted_data, flow_style=True
  )


yaml.add_representer(FlowDict, flow_dict_representer, Dumper=yaml.SafeDumper)


def _parse_stage(stage_val: Any) -> Stage | None:
  """Safely parses a stage value into an Stage | None."""
  if isinstance(stage_val, Stage):
    return stage_val
  if isinstance(stage_val, str):
    try:
      return Stage(stage_val)
    except ValueError:
      pass
  return None


def load_tt_apis_stages(
    tt_apis_path: str,
) -> tuple[dict[str, Stage | None], dict[str, Stage | None]]:
  """Loads Python API and Environment Variable stages dynamically from tt_apis.yaml.

  Args:
      tt_apis_path: Filesystem path to the `tt_apis.yaml` source of truth.

  Returns:
      A tuple of (python_api_stages, env_var_stages) mapping symbol names to
      their lifecycle stages (e.g. Stage.STABLE, Stage.EXPERIMENTAL,
      Stage.INTERNAL).
  """
  py_stages: dict[str, Stage | None] = {}
  env_stages: dict[str, Stage | None] = {}
  if os.path.exists(tt_apis_path):
    with open(tt_apis_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
          continue
        # Each entry is parsed on its own, since the same name can appear in
        # multiple sections (e.g. Duplicated Export), which a whole-file load
        # would collapse.
        for name, info in (yaml.safe_load(line) or {}).items():
          if not isinstance(info, dict):
            continue
          cat = info.get("category", "")
          stage = _parse_stage(info.get("stage", "Unknown"))
          if "Python API" in cat or "Custom Op" in cat or "Flag" in cat:
            py_stages[name] = stage
            # Normalize properties: strip " getter" and " setter" suffixes
            if name.endswith(" getter") or name.endswith(" setter"):
              prop_name = name.rsplit(" ", 1)[0]
              py_stages[prop_name] = stage
          elif cat == "Environment Variable":
            env_stages[name] = stage
  return py_stages, env_stages


# ==============================================================================
# AST & Source Code Scanning
# ==============================================================================


class InternalApiVisitor(ast.NodeVisitor):
  """AST and text visitor to extract internal API and internal env var usage."""

  def __init__(
      self,
      rel_path: str,
      py_stages: Mapping[str, Any],
      env_stages: Mapping[str, Any],
  ):
    self.rel_path = rel_path
    self.py_stages: dict[str, Stage | None] = {
        k: _parse_stage(v) for k, v in py_stages.items()
    }
    self.env_stages: dict[str, Stage | None] = {
        k: _parse_stage(v) for k, v in env_stages.items()
    }

    self.aliases: dict[str, str] = {}
    self.local_instances: dict[str, str] = {}
    self._skipped_nodes: set[ast.AST] = set()

    self.internal_py: dict[str, set[str]] = collections.defaultdict(set)
    self.internal_envs: dict[str, set[str]] = collections.defaultdict(set)

  def _record(self, bucket: dict[str, set[str]], symbol: str):
    bucket[symbol].add(self.rel_path)

  def _check_env(self, var_name: str):
    stage = self.env_stages.get(var_name)
    if (
        var_name.startswith(("TORCH_TPU_INTERNAL_", "TPU_INTERNAL_"))
        or var_name in _INTERNAL_IMPLEMENTATION_ENV_VARS
        or stage == Stage.INTERNAL
    ):
      self._record(self.internal_envs, var_name)

  def _is_known_public(self, name: str) -> bool:
    stage = self.py_stages.get(name)
    return (name in self.py_stages and stage != Stage.INTERNAL) or any(
        (k.startswith(f"{name}.") or k.startswith(f"{name} "))
        and self.py_stages[k] != Stage.INTERNAL
        for k in self.py_stages
    )

  def _check_py_symbol(self, name: str):
    # Ignore root namespace prefixes themselves (e.g. 'torch.tpu', 'torch_tpu').
    if name in _ROOT_NAMESPACES:
      return

    if not _is_torch_tpu_name(name) and name not in self.py_stages:
      return

    parts = name.split(".")
    stage = self.py_stages.get(name)

    # 1. Package boundary: implementation under `torch_tpu._internal.*` or `torch_tpu.google._internal.*`.
    is_internal_pkg = "_internal" in parts

    # 2. Lexical privacy: leading underscore on attributes after the namespace prefix
    prefix_len = 0
    for ns in _TORCH_TPU_NAMESPACES:
      prefix = ns[:-1]
      if name == prefix or name.startswith(f"{prefix}."):
        prefix_len = len(prefix.split("."))
        break
    is_private_attr = any(
        p.startswith("_") and p != "_internal" for p in parts[prefix_len:]
    )

    # 3. Explicit lifecycle annotation: marked `Internal` in tt_apis.yaml.
    is_explicit_internal = stage == Stage.INTERNAL

    # 4. Check whether this symbol is a verified, exposed public API in tt_apis.yaml.
    is_known_public = self._is_known_public(name)

    if is_known_public and not (
        is_internal_pkg or is_private_attr or is_explicit_internal
    ):
      return

    # 5. Unexposed surface: accessed on any _TORCH_TPU_NAMESPACES but absent from tt_apis.yaml whitelist.
    is_intra_example = name == "torch_tpu.examples" or name.startswith(
        "torch_tpu.examples."
    )
    is_unexposed = (
        any(name.startswith(ns) for ns in _TORCH_TPU_NAMESPACES)
        and not is_intra_example
        and not is_known_public
    )

    if (
        is_internal_pkg
        or is_private_attr
        or is_explicit_internal
        or is_unexposed
    ):
      self._record(self.internal_py, name)

  def _get_expr_base_name(self, expr: ast.AST) -> str | None:
    curr = expr
    while isinstance(curr, ast.Attribute):
      curr = curr.value
    if isinstance(curr, ast.Name):
      return curr.id
    return None

  def _resolve_expr_name(self, expr: ast.AST) -> str | None:
    parts = []
    curr = expr
    while isinstance(curr, ast.Attribute):
      parts.append(curr.attr)
      curr = curr.value
    if isinstance(curr, ast.Name):
      base_id = curr.id
      parts.append(base_id)
      parts.reverse()
      if base_id in self.local_instances:
        return ".".join([self.local_instances[base_id]] + parts[1:])
      if base_id in self.aliases:
        return ".".join([self.aliases[base_id]] + parts[1:])
      return ".".join(parts)
    return None

  def visit_Import(self, node: ast.Import):
    for alias in node.names:
      self._check_py_symbol(alias.name)
      asname = alias.asname or alias.name
      self.aliases[asname] = alias.name
    self.generic_visit(node)

  def visit_ImportFrom(self, node: ast.ImportFrom):
    mod = node.module or ""
    for alias in node.names:
      full = f"{mod}.{alias.name}" if mod else alias.name
      self._check_py_symbol(full)
      asname = alias.asname or alias.name
      self.aliases[asname] = full
    self.generic_visit(node)

  def visit_Name(self, node: ast.Name):
    if node in self._skipped_nodes:
      return
    if isinstance(node.ctx, ast.Load) and node.id in self.aliases:
      target = self.aliases[node.id]
      self._check_py_symbol(target)
    self.generic_visit(node)

  def _check_member_access(
      self,
      base_name: str | None,
      attrs: Sequence[str],
      is_call: bool = False,
  ) -> None:
    """Checks attribute access on an instance or call result for internal API leakage."""
    if not attrs:
      return

    has_private_attr = any(
        p.startswith("_") and not (p.startswith("__") and p.endswith("__"))
        for p in attrs
    )
    has_pytorch_native_attr = any(p in _PYTORCH_NATIVE_ATTRS for p in attrs)

    # 1. Private attribute access: flag references like obj._secret or call()._secret
    if has_private_attr:
      for p in attrs:
        if p.startswith("_") and not (p.startswith("__") and p.endswith("__")):
          if base_name and (
              "_internal" in base_name or _is_torch_tpu_name(base_name)
          ):
            self._record(self.internal_py, f"{base_name}.{p}")
          else:
            self._record(self.internal_py, f"<call>.{p}")
      return

    # 2. Check accesses on call results (return values of a function call: func().attr)
    if is_call:
      # Native PyTorch Tensor/Module/CustomOp methods on return values are standard operations.
      if has_pytorch_native_attr:
        return
      # If the callable is a known public API, method calls on its return value are valid
      # unless the attribute is an actual attribute of the callable itself.
      if base_name and self._is_known_public(base_name):
        base_obj = _resolve_runtime_object(base_name)
        if base_obj is not None and not hasattr(base_obj, attrs[0]):
          return
        full_name = ".".join([base_name] + list(attrs))
        if full_name not in self.py_stages:
          return
        self._check_py_symbol(full_name)
        return
      if base_name:
        full_name = ".".join([base_name] + list(attrs))
        self._check_py_symbol(full_name)
      return

    # 3. Check accesses on class instances (obj.attr where obj = SomeClass())
    if base_name:
      is_internal_base = (
          "_internal" in base_name
          or self.py_stages.get(base_name) == Stage.INTERNAL
      )
      # If an internal class is instantiated, any member access (even standard PyTorch
      # methods like forward/train) is a usage of an internal class.
      if is_internal_base:
        full_name = ".".join([base_name] + list(attrs))
        self._check_py_symbol(full_name)
        return

      # For public classes, native PyTorch methods not defined on the class itself are safe.
      if has_pytorch_native_attr:
        base_obj = _resolve_runtime_object(base_name)
        if base_obj is not None and not hasattr(base_obj, attrs[0]):
          return

      # Verify attribute against public API whitelist in tt_apis.yaml.
      full_name = ".".join([base_name] + list(attrs))
      self._check_py_symbol(full_name)

  def _is_class_target(self, target_name: str) -> bool:
    """Returns True if target_name resolves to a class or follows CapWords naming."""
    obj = _resolve_runtime_object(target_name)
    if obj is not None:
      return inspect.isclass(obj)
    leaf = target_name.split(".")[-1]
    return bool(leaf and leaf[0].isupper() and not leaf.isupper())

  def visit_Assign(self, node: ast.Assign):
    # Track local instances of classes (e.g. tracer = utils.OpTracer()).
    # Non-class callables (e.g. res = splash_sdpa(...)) return values, not instances.
    if isinstance(node.value, ast.Call):
      base_id = self._get_expr_base_name(node.value.func)
      if not (base_id and base_id in self.local_instances):
        call_target = self._resolve_expr_name(node.value.func)
        if call_target and self._is_class_target(call_target):
          for target in node.targets:
            if isinstance(target, ast.Name):
              self.local_instances[target.id] = call_target
    self.generic_visit(node)

  def visit_Attribute(self, node: ast.Attribute):
    if node in self._skipped_nodes:
      return

    # Unwind the full attribute chain
    parts = []
    curr = node
    while isinstance(curr, ast.Attribute):
      self._skipped_nodes.add(curr)
      parts.append(curr.attr)
      curr = curr.value

    if isinstance(curr, ast.Name):
      self._skipped_nodes.add(curr)
      base_id = curr.id
      parts.append(base_id)
      parts.reverse()
      full_name = ".".join(parts)
      self._check_py_symbol(full_name)

      # Check if base_id is a local instance or alias
      if base_id in self.local_instances:
        self._check_member_access(
            self.local_instances[base_id], parts[1:], is_call=False
        )
      elif base_id in self.aliases:
        aliased_base = self.aliases[base_id]
        aliased_full_name = ".".join([aliased_base] + parts[1:])
        if aliased_full_name != full_name:
          self._check_py_symbol(aliased_full_name)
    elif isinstance(curr, ast.Call):
      # Attribute on a call result: a.b()._c
      parts.reverse()
      call_func_name = self._resolve_expr_name(curr.func)
      self._check_member_access(call_func_name, parts, is_call=True)
      self.generic_visit(curr)
      return

    self.generic_visit(curr)

  def visit_Subscript(self, node: ast.Subscript):
    # Detects os.environ["VAR"]
    if isinstance(node.value, ast.Attribute) and isinstance(
        node.value.value, ast.Name
    ):
      if node.value.value.id == "os" and node.value.attr == "environ":
        if isinstance(node.slice, ast.Constant) and isinstance(
            node.slice.value, str
        ):
          self._check_env(node.slice.value)
    self.generic_visit(node)

  def visit_Call(self, node: ast.Call):
    func = node.func
    if isinstance(func, ast.Attribute):
      # Detects os.environ.get(...) / setdefault / pop
      if isinstance(func.value, ast.Attribute) and isinstance(
          func.value.value, ast.Name
      ):
        if (
            func.value.value.id == "os"
            and func.value.attr == "environ"
            and func.attr in ("get", "setdefault", "pop")
        ):
          if (
              node.args
              and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
          ):
            self._check_env(node.args[0].value)
      # Detects os.getenv(...)
      elif (
          isinstance(func.value, ast.Name)
          and func.value.id == "os"
          and func.attr == "getenv"
      ):
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
          self._check_env(node.args[0].value)
      # Detects env.get_*_env_once(...)
      elif (
          isinstance(func.value, ast.Name)
          and func.value.id in ("env", "env_utils")
          and "env" in func.attr
      ):
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
          self._check_env(node.args[0].value)

    # Detects getattr(obj, "attr") / hasattr(obj, "attr")
    if isinstance(func, ast.Name) and func.id in ("getattr", "hasattr"):
      if (
          len(node.args) >= 2
          and isinstance(node.args[1], ast.Constant)
          and isinstance(node.args[1].value, str)
      ):
        attr_name = node.args[1].value
        base_name = self._resolve_expr_name(node.args[0])
        if base_name:
          full_name = f"{base_name}.{attr_name}"
          self._check_py_symbol(full_name)

    self.generic_visit(node)

  def scan_text(self, source_code: str, is_python_source: bool = False):
    """Scans raw source text and markdown cells for internal modules and env vars.

    Args:
        source_code: The raw text of the file or cell.
        is_python_source: Whether the source code is a Python script or code
          cell.
    """
    lines = source_code.splitlines()
    for line in lines:
      # Skip import statements when scanning python code lines (handled by AST)
      if is_python_source and re.match(r"^\s*(?:from|import)\s+", line):
        continue

      # 1. Internal module references, ops, or flags in strings, shell scripts, or markdown
      # A flag is matched without its leading `--`, as listed in tt_apis.yaml.
      for m in re.finditer(
          r"\b((?:torch_tpu(?:\.google)?\._internal|torch\.tpu\._|torch\.ops\.tpu\._|torch\.ops\.torch_tpu)[a-zA-Z0-9_.]*)\b"
          r"|--(torch_tpu_internal_[a-zA-Z0-9_]+)\b",
          line,
      ):
        sym = m.group(1) or m.group(2)
        if sym:
          self._check_py_symbol(sym)

      # 2. Environment variables referenced in backticks or explicit internal prefix
      for m in re.finditer(
          r"`([A-Za-z0-9_]+)`|\b((?:TORCH_TPU_INTERNAL_|TPU_INTERNAL_)[A-Z0-9_]+)\b",
          line,
      ):
        var = m.group(1) or m.group(2)
        if var:
          if (
              var.startswith(("TORCH_TPU_INTERNAL_", "TPU_INTERNAL_"))
              or var in _INTERNAL_IMPLEMENTATION_ENV_VARS
              or self.env_stages.get(var) == Stage.INTERNAL
          ):
            self._check_env(var)

      # 3. Known internal implementation environment variables
      for var in _INTERNAL_IMPLEMENTATION_ENV_VARS:
        if re.search(rf"\b{re.escape(var)}\b", line):
          self._check_env(var)


# ==============================================================================
# File Discovery & Scanning Orchestration
# ==============================================================================


def discover_python_files(
    target_dir: str, include_tests: bool = True
) -> list[str]:
  """Recursively discovers all Python scripts, notebooks, markdown tutorials, and shell scripts."""
  if not os.path.exists(target_dir):
    logging.info("Target directory does not exist, skipping: %s", target_dir)
    return []

  scannable_extensions = (".py", ".ipynb", ".md", ".sh")
  discovered_files: list[str] = []
  for root, _, files in os.walk(target_dir):
    if "docs/developers" in root:
      continue
    for filename in sorted(files):
      if filename.startswith("."):
        continue
      if not filename.endswith(scannable_extensions):
        continue
      if not include_tests and (
          filename.startswith("test_") or filename.endswith("_test.py")
      ):
        continue
      discovered_files.append(os.path.join(root, filename))

  return sorted(discovered_files)


discover_scannable_files = discover_python_files


def scan_file(
    file_path: str,
    visitor: InternalApiVisitor,
) -> None:
  """Scans a single Python file, notebook, markdown document, or shell script."""
  try:
    if file_path.endswith(".ipynb"):
      with open(file_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

      for cell in nb_data.get("cells", []):
        source = cell.get("source", [])
        cell_text = "".join(source) if isinstance(source, list) else str(source)
        if not cell_text.strip():
          continue

        if cell.get("cell_type") == "code":
          clean_lines = [
              line if not line.strip().startswith(("%", "!")) else f"# {line}"
              for line in cell_text.splitlines()
          ]
          clean_code = "\n".join(clean_lines)
          tree = ast.parse(clean_code, filename=file_path)
          visitor.visit(tree)
          visitor.scan_text(clean_code, is_python_source=True)
        else:
          visitor.scan_text(cell_text, is_python_source=False)

    elif file_path.endswith(".py"):
      with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
      tree = ast.parse(code, filename=file_path)
      visitor.visit(tree)
      visitor.scan_text(code, is_python_source=True)

    elif file_path.endswith((".md", ".sh")):
      with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
      visitor.scan_text(content, is_python_source=False)

  except Exception as e:
    logging.error("Failed to parse %s: %s", file_path, e)
    raise RuntimeError(f"Failed to scan {file_path}: {e}") from e


def scan_directories(
    scan_dirs: Sequence[str],
    tt_apis_path: str,
    repository_root: str = "",
    include_tests: bool = True,
) -> tuple[list[InternalApiEntry], list[InternalApiEntry]]:
  """Scans documentation and examples directories and returns internal API entries.

  Args:
      scan_dirs: Sequence of directory paths to scan.
      tt_apis_path: Path to tt_apis.yaml source of truth.
      repository_root: Optional repository root for relative path normalization.
      include_tests: Whether to include test files (*_test.py and test_*.py).

  Returns:
      A 2-tuple of (internal_py, internal_env) entries.
  """
  py_stages, env_stages = load_tt_apis_stages(tt_apis_path)

  all_internal_py: dict[str, set[str]] = collections.defaultdict(set)
  all_internal_env: dict[str, set[str]] = collections.defaultdict(set)

  for scan_dir in scan_dirs:
    if not os.path.exists(scan_dir):
      logging.info("Scan directory does not exist, skipping: %s", scan_dir)
      continue

    discovered_files = discover_python_files(
        scan_dir, include_tests=include_tests
    )
    for file_path in discovered_files:
      if repository_root and file_path.startswith(repository_root):
        rel_path = os.path.relpath(file_path, repository_root)
      elif repo_paths.G3_REPO_PATH in file_path:
        idx = file_path.find(repo_paths.G3_REPO_PATH)
        rel_path = file_path[idx:]
      else:
        rel_path = file_path

      visitor = InternalApiVisitor(rel_path, py_stages, env_stages)
      scan_file(file_path, visitor)

      for k, v in visitor.internal_py.items():
        all_internal_py[k].update(v)
      for k, v in visitor.internal_envs.items():
        all_internal_env[k].update(v)

  internal_py_entries = [
      InternalApiEntry(
          name=name,
          category="Internal Python Module",
          stage=py_stages.get(name) or Stage.INTERNAL,
          used_in=sorted(locs),
      )
      for name, locs in sorted(all_internal_py.items())
  ]

  internal_env_entries = [
      InternalApiEntry(
          name=name,
          category="Environment Variable",
          stage=env_stages.get(name) or Stage.INTERNAL,
          used_in=sorted(locs),
      )
      for name, locs in sorted(all_internal_env.items())
  ]

  return internal_py_entries, internal_env_entries


# ==============================================================================
# Formatting, Serialization & Artifact Export
# ==============================================================================


_EXPOSED_INTERNAL_APIS_YAML_HEADER = """\
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


def format_exposed_internal_apis_yaml(
    internal_py: Sequence[InternalApiEntry],
    internal_env: Sequence[InternalApiEntry],
) -> str:
  """Formats categorized internal API items into YAML baseline text.

  Args:
      internal_py: Internal Python API entries.
      internal_env: Internal environment variable entries.

  Returns:
      The formatted YAML text grouped by section.
  """
  groups = [
      ("Internal Python API", internal_py),
      ("Internal Environment Variable", internal_env),
  ]

  sections: list[str] = []
  for header, items in groups:
    section_header = f"# --- {header} ---"
    if not items:
      sections.append(section_header)
      continue

    formatted_data = {
        item.name: FlowDict({
            "used_in": item.used_in,
        })
        for item in sorted(items, key=lambda x: x.name)
    }
    section_yaml = yaml.safe_dump(
        formatted_data,
        width=float("inf"),
        sort_keys=False,
    )
    sections.append(f"{section_header}\n{section_yaml.strip()}")

  return _EXPOSED_INTERNAL_APIS_YAML_HEADER + "\n\n".join(sections) + "\n"


format_internal_apis_yaml = format_exposed_internal_apis_yaml


def export_sponge_artifact(
    yaml_text: str, filename: str = "exposed_internal_apis.yaml"
) -> None:
  """Writes the baseline YAML text to Sponge test artifacts, if available.

  Args:
      yaml_text: The formatted YAML text to export.
      filename: The artifact output filename.
  """
  test_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
  if test_outputs_dir:
    artifact_path = os.path.join(test_outputs_dir, filename)
    try:
      with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
      print(f"Exported Sponge Test Artifact: {artifact_path}")
    except Exception as error:  # pylint: disable=broad-except
      logging.warning("Failed to export Sponge artifact: %s", error)


# ==============================================================================
# Main Execution Entrypoint
# ==============================================================================


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  try:
    root = repo_paths.get_repo_root_in_workspace()
  except FileNotFoundError as error:
    raise app.UsageError(
        f"scan_exposed_internal_apis scans the workspace: {error}"
    ) from error
  baseline_path = os.path.join(root, repo_paths.EXPOSED_INTERNAL_APIS_YAML_PATH)

  # `used_in` paths are relative to the repository root.
  internal_py, internal_env = scan_directories(
      [os.path.join(root, subdir) for subdir in DEFAULT_SCAN_SUBDIRS],
      os.path.join(root, repo_paths.TT_APIS_YAML_PATH),
      repository_root=root,
  )

  yaml_text = format_exposed_internal_apis_yaml(internal_py, internal_env)

  export_sponge_artifact(yaml_text)

  if _UPDATE_BASELINE.value:
    with open(baseline_path, "w", encoding="utf-8") as f:
      f.write(yaml_text)
    print(f"Successfully updated baseline: {baseline_path}")

  print("=" * 70)
  print("  TorchTPU Internal API Scanner")
  print("=" * 70)
  print(f"  Internal Python APIs (Violations):   {len(internal_py)}")
  print(f"  Internal Env Vars (Violations):      {len(internal_env)}")
  print("=" * 70)

  if internal_py or internal_env:
    print(
        "\n[!] WARNING: Found Internal APIs / Env Vars referenced in scanned"
        " code:"
    )
    for item in internal_py:
      print(f"    - {item.name} (used in {len(item.used_in)} locations)")
    for item in internal_env:
      print(f"    - {item.name} (used in {len(item.used_in)} locations)")
    print(
        "\nAction required: Scrub from docs/examples or promote to public"
        " API stage."
    )


if __name__ == "__main__":
  app.run(main)
