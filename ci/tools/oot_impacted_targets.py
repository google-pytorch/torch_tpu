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

"""Computes `//tests/torch:*` targets impacted by PyTorch `test/` diffs.

Each `torch_tpu_torch_test` target depends on a
`@pytorch_test//:<module>_module`
filegroup whose inputs are declared statically in `bazel/pytorch_test_repo.bzl`
(own `test/<module>.py`, `_EXTRA_MODULE_GLOBS`, the `<subdir>_support` glob and
`_EXTRA_MODULE_SUPPORT_TARGETS`). Because the CI pre-filter holds the TorchTPU
workspace and the `TORCH_SOURCE` wheel fixed and only varies `test/**` between
a base and head `pytorch/pytorch` commit, the impacted set is exactly the set
of targets whose declared inputs intersect `git diff --name-only base head --
test/`. No Bazel analysis or `bazel-diff` is needed.
"""

import argparse
import ast
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import functools
import json
import os
import pathlib
import re
import subprocess
import sys

_SUPPORT_SUFFIX = "_support"
_SELECTION_SCOPE = "test_files_only"
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclasses.dataclass(frozen=True)
class TorchTestTarget:
  """A `torch_tpu_torch_test` target parsed from `//tests/torch:BUILD`.

  Attributes:
    module: Upstream PyTorch test module path relative to `test/`, without the
      `.py` suffix (e.g. `"test_autograd"` or `"dynamo/test_misc"`).
    shard_count: Number of Bazel test shards configured on the target (defaults
      to 1 when omitted in BUILD).
    enabled: Whether the target runs in OSS CI (False when `notap`/`nobuild` is
      set to a non-empty string, `notest_oss`/`nobuild_oss` is True, or `tags`
      includes `"notest"` or `"nobuild"`).
  """

  module: str
  shard_count: int
  enabled: bool


@dataclasses.dataclass(frozen=True)
class ModuleInputs:
  """Declared `@pytorch_test` inputs of one test module (mirrors the .bzl).

  Attributes:
    include: Repo-relative Bazel glob patterns under `test/` included directly
      by the module's filegroup (own `test/<module>.py` plus extra globs).
    exclude: Repo-relative Bazel glob patterns excluded from `include`.
    support_subdirs: Top-level `test/<subdir>` names whose `<subdir>_support`
      filegroup is pulled in by this module.
  """

  include: tuple[str, ...]
  exclude: tuple[str, ...]
  support_subdirs: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class DepsTables:
  """`_EXTRA_MODULE_GLOBS` / `_EXTRA_MODULE_SUPPORT_TARGETS` from the .bzl.

  Attributes:
    extra_module_globs: Mapping from `torch_test_module` name to extra
      repo-relative glob patterns included in its filegroup.
    extra_module_support_targets: Mapping from `torch_test_module` name to
      additional `@pytorch_test` `:<subdir>_support` target labels.
  """

  extra_module_globs: Mapping[str, Sequence[str]]
  extra_module_support_targets: Mapping[str, Sequence[str]]


# --------------------------------------------------------------------------
# Parsing inputs (BUILD file and repo-rule tables).
# --------------------------------------------------------------------------


def _parse_optional_reason(
    kwargs: Mapping[str, ast.expr],
    key: str,
    rule_name: str,
    lineno: int,
) -> bool:
  """Returns True if `key` is a non-empty string literal; fails closed."""
  node = kwargs.get(key)
  if node is None or (isinstance(node, ast.Constant) and node.value is None):
    return False
  if (
      isinstance(node, ast.Constant)
      and isinstance(node.value, str)
      and node.value
  ):
    return True
  raise ValueError(
      f"torch_tpu_torch_test({rule_name!r}): {key!r} must be a non-empty "
      f"string literal or None (line {lineno})."
  )


def _parse_optional_bool(
    kwargs: Mapping[str, ast.expr],
    key: str,
    rule_name: str,
    lineno: int,
) -> bool:
  """Returns the bool literal value of `key` (False when omitted)."""
  node = kwargs.get(key)
  if node is None:
    return False
  if isinstance(node, ast.Constant) and isinstance(node.value, bool):
    return node.value
  raise ValueError(
      f"torch_tpu_torch_test({rule_name!r}): {key!r} must be a boolean "
      f"literal (line {lineno})."
  )


# LINT.IfChange(torch_test_module_format)
def parse_torch_test_targets(build_text: str) -> Mapping[str, TorchTestTarget]:
  """Parses `torch_tpu_torch_test` rules from `//tests/torch:BUILD` via AST.

  Args:
    build_text: Contents of `tests/torch/BUILD`.

  Returns:
    Mapping from full Bazel label (`//tests/torch:<name>`) to `TorchTestTarget`.

  Raises:
    ValueError: If no `torch_tpu_torch_test` rules are parsed from `build_text`
      or a rule uses unsupported non-literal `name`, `torch_test_module`,
      `tags`, `notap`, `nobuild`, `notest_oss`, `nobuild_oss`, or `**kwargs`
      expressions.
  """
  targets: dict[str, TorchTestTarget] = {}
  for node in ast.parse(build_text).body:
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
      continue
    call = node.value
    if not (
        isinstance(call.func, ast.Name)
        and call.func.id == "torch_tpu_torch_test"
    ):
      continue
    if call.args or any(kw.arg is None for kw in call.keywords):
      raise ValueError(
          "Unsupported positional or **kwargs arguments in "
          f"torch_tpu_torch_test at line {call.lineno}."
      )
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    name_node = kwargs.get("name")
    mod_node = kwargs.get("torch_test_module")
    if not (
        isinstance(name_node, ast.Constant)
        and isinstance(name_node.value, str)
        and isinstance(mod_node, ast.Constant)
        and isinstance(mod_node.value, str)
    ):
      raise ValueError(
          "torch_tpu_torch_test requires literal string 'name' and "
          f"'torch_test_module' at line {call.lineno}."
      )
    shard_node = kwargs.get("shard_count")
    if (
        isinstance(shard_node, ast.Constant)
        and isinstance(shard_node.value, int)
        and not isinstance(shard_node.value, bool)
    ):
      shard_count = shard_node.value
    else:
      shard_count = 1
    tags: set[str] = set()
    tags_node = kwargs.get("tags")
    if tags_node is not None and not (
        isinstance(tags_node, ast.Constant) and tags_node.value is None
    ):
      if not isinstance(tags_node, (ast.List, ast.Tuple)):
        raise ValueError(
            f"torch_tpu_torch_test({name_node.value!r}): 'tags' must be a "
            f"literal list/tuple of strings or None (line {call.lineno})."
        )
      for element in tags_node.elts:
        if not (
            isinstance(element, ast.Constant) and isinstance(element.value, str)
        ):
          raise ValueError(
              f"torch_tpu_torch_test({name_node.value!r}): 'tags' elements must"
              f" be string literals (line {call.lineno})."
          )
        tags.add(element.value)
    disabled_by_kwarg = (
        _parse_optional_reason(kwargs, "notap", name_node.value, call.lineno)
        or _parse_optional_reason(
            kwargs, "nobuild", name_node.value, call.lineno
        )
        or _parse_optional_bool(
            kwargs, "notest_oss", name_node.value, call.lineno
        )
        or _parse_optional_bool(
            kwargs, "nobuild_oss", name_node.value, call.lineno
        )
    )
    enabled = not (disabled_by_kwarg or tags & {"notest", "nobuild"})
    label = f"//tests/torch:{name_node.value}"
    targets[label] = TorchTestTarget(
        module=mod_node.value,
        shard_count=shard_count,
        enabled=enabled,
    )
  if not targets:
    raise ValueError("No torch_tpu_torch_test targets parsed from BUILD text.")
  return targets


# LINT.ThenChange(
#     gen_torch_tests.py:torch_test_module_format,
#     ../../bazel/pytorch_test_repo.bzl:torch_test_module_format,
# )


def parse_deps_tables(bzl_text: str) -> DepsTables:
  """Extracts the two dict literals from `pytorch_test_repo.bzl`.

  Starlark dict/list/string literals are a strict subset of Python literals, so
  `ast.literal_eval` on the assignment RHS keeps the .bzl as the single source
  of truth without duplicating the tables here.

  Args:
    bzl_text: Contents of `bazel/pytorch_test_repo.bzl`.

  Returns:
    `DepsTables` containing `_EXTRA_MODULE_GLOBS` and
    `_EXTRA_MODULE_SUPPORT_TARGETS`.

  Raises:
    ValueError: If either required dict assignment is missing from `bzl_text`.
  """
  found: dict[str, Mapping[str, Sequence[str]]] = {}
  for node in ast.parse(bzl_text).body:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
      continue
    target = node.targets[0]
    if isinstance(target, ast.Name) and target.id in (
        "_EXTRA_MODULE_GLOBS",
        "_EXTRA_MODULE_SUPPORT_TARGETS",
    ):
      found[target.id] = ast.literal_eval(node.value)
  missing = {"_EXTRA_MODULE_GLOBS", "_EXTRA_MODULE_SUPPORT_TARGETS"} - set(
      found
  )
  if missing:
    raise ValueError(f"pytorch_test_repo.bzl: missing {sorted(missing)}")
  return DepsTables(
      extra_module_globs=found["_EXTRA_MODULE_GLOBS"],
      extra_module_support_targets=found["_EXTRA_MODULE_SUPPORT_TARGETS"],
  )


def validate_deps_tables(
    targets: Mapping[str, TorchTestTarget], tables: DepsTables
) -> None:
  """Verifies every module key in `tables` exists in `tests/torch/BUILD`.

  Args:
    targets: Parsed `torch_tpu_torch_test` targets from `tests/torch/BUILD`.
    tables: Dependency tables parsed from `bazel/pytorch_test_repo.bzl`.

  Raises:
    ValueError: If `tables` references any module not present in `targets`.
  """
  known_modules = {target.module for target in targets.values()}
  stale = sorted(
      (
          set(tables.extra_module_globs)
          | set(tables.extra_module_support_targets)
      )
      - known_modules
  )
  if stale:
    raise ValueError(
        "pytorch_test_repo.bzl contains stale module entries not in "
        f"tests/torch/BUILD: {stale}"
    )


# --------------------------------------------------------------------------
# Bazel glob semantics (subset sufficient for the repo rule's patterns).
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _glob_to_regex(pattern: str) -> re.Pattern[str]:
  """Compiles a Bazel `glob()` pattern (`*`, `**`) into an anchored regex."""
  out: list[str] = []
  index = 0
  while index < len(pattern):
    char = pattern[index]
    if pattern.startswith("**/", index):
      out.append("(?:[^/]+/)*")
      index += 3
    elif pattern.startswith("**", index):
      out.append(".*")
      index += 2
    elif char == "*":
      out.append("[^/]*")
      index += 1
    else:
      out.append(re.escape(char))
      index += 1
  return re.compile("^" + "".join(out) + "$")


def glob_matches(pattern: str, path: str) -> bool:
  """Returns whether `path` matches a Bazel `glob()` pattern (`*`, `**`).

  Args:
    pattern: Bazel-style glob pattern supporting `*` and `**`.
    path: Repo-relative POSIX path to test against `pattern`.

  Returns:
    True if `path` matches `pattern` in full, False otherwise.
  """
  return _glob_to_regex(pattern).match(path) is not None


# LINT.IfChange(pytorch_test_glob_rules)
_MODULE_EXCLUDES = (
    "test/**/.*",
    "test/**/__pycache__/**",
)


def support_patterns(subdir: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """Returns include/exclude globs for `@pytorch_test//:<subdir>_support`.

  Args:
    subdir: Top-level directory name under `test/` (e.g. `"dynamo"`).

  Returns:
    `(include_globs, exclude_globs)` matching `pytorch_test_repo.bzl`.
  """
  return (
      (f"test/{subdir}/**",),
      (
          f"test/{subdir}/**/.*",
          f"test/{subdir}/**/__pycache__/**",
          f"test/{subdir}/**/test_*.py",
      ),
  )


def module_inputs(module: str, tables: DepsTables) -> ModuleInputs:
  """Returns the declared inputs of `@pytorch_test//:<module>_module`.

  Args:
    module: Upstream PyTorch test module path relative to `test/` without `.py`.
    tables: Dependency tables parsed from `bazel/pytorch_test_repo.bzl`.

  Returns:
    `ModuleInputs` describing include globs, exclude globs, and support subdirs.

  Raises:
    ValueError: If an entry in `extra_module_support_targets` is not a
      `:<subdir>_support` label.
  """
  include = [f"test/{module}.py"] + list(
      tables.extra_module_globs.get(module, [])
  )
  subdirs: list[str] = []
  if "/" in module:
    subdirs.append(module.split("/", 1)[0])
  for label in tables.extra_module_support_targets.get(module, []):
    if not (label.startswith(":") and label.endswith(_SUPPORT_SUFFIX)):
      raise ValueError(f"{module}: unsupported support target {label!r}")
    subdir = label[1 : -len(_SUPPORT_SUFFIX)]
    if subdir not in subdirs:
      subdirs.append(subdir)
  return ModuleInputs(
      include=tuple(include),
      exclude=_MODULE_EXCLUDES,
      support_subdirs=tuple(subdirs),
  )


def path_is_input(path: str, inputs: ModuleInputs) -> bool:
  """Returns whether `path` feeds the module's `@pytorch_test` filegroup.

  Args:
    path: Repo-relative POSIX path under `test/`.
    inputs: Declared `ModuleInputs` for a test module.

  Returns:
    True if `path` matches `inputs` after exclusions and repo-rule deletions.
  """
  # The repo rule deletes upstream BUILD files.
  if os.path.basename(path) in ("BUILD", "BUILD.bazel"):
    return False
  if any(glob_matches(pattern, path) for pattern in inputs.include) and not any(
      glob_matches(pattern, path) for pattern in inputs.exclude
  ):
    return True
  for subdir in inputs.support_subdirs:
    include_patterns, exclude_patterns = support_patterns(subdir)
    if any(
        glob_matches(pattern, path) for pattern in include_patterns
    ) and not any(glob_matches(pattern, path) for pattern in exclude_patterns):
      return True
  return False


# LINT.ThenChange(../../bazel/pytorch_test_repo.bzl:pytorch_test_glob_rules)


# --------------------------------------------------------------------------
# Impact computation.
# --------------------------------------------------------------------------


def compute_impacted(
    changed_paths: Iterable[str],
    targets: Mapping[str, TorchTestTarget],
    tables: DepsTables,
) -> Mapping[str, Sequence[str]]:
  """Maps each impacted enabled target label to the changed paths that hit it.

  Args:
    changed_paths: Repo-relative paths changed between base and head commits.
    targets: All `torch_tpu_torch_test` targets parsed from `tests/torch/BUILD`.
    tables: Dependency tables parsed from `bazel/pytorch_test_repo.bzl`.

  Returns:
    Sorted mapping from impacted enabled target label to matching changed paths.
  """
  changed = sorted(set(changed_paths))
  impacted: dict[str, tuple[str, ...]] = {}
  for label, target in sorted(targets.items()):
    if not target.enabled:
      continue
    inputs = module_inputs(target.module, tables)
    hits = tuple(path for path in changed if path_is_input(path, inputs))
    if hits:
      impacted[label] = hits
  return impacted


def _validate_commit_sha(sha: str) -> str:
  """Validates a full 40-char commit SHA to prevent Git option injection."""
  cleaned = sha.strip()
  if not _COMMIT_SHA_RE.fullmatch(cleaned):
    raise ValueError(
        f"Invalid commit SHA {sha!r}: expected 40 hexadecimal characters."
    )
  return cleaned


def _git(pytorch_dir: pathlib.Path, *args: str) -> str:
  """Runs a Git command in `pytorch_dir` and returns UTF-8 stdout."""
  result = subprocess.run(
      ["git", "-C", os.fspath(pytorch_dir), *args],
      capture_output=True,
      text=True,
      encoding="utf-8",
      errors="replace",
      check=False,
  )
  if result.returncode != 0:
    raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
  return result.stdout


def ensure_commit(pytorch_dir: pathlib.Path, sha: str) -> None:
  """Fetches `sha` (trees only, depth 1) when it is not present locally.

  Args:
    pytorch_dir: Path to the local PyTorch Git repository.
    sha: Full 40-character hexadecimal commit SHA to ensure is available locally
      (required by Git `upload-pack` for uncached fetches).

  Raises:
    ValueError: If `sha` is not a full 40-character hexadecimal commit SHA.
    RuntimeError: If `git fetch` exits with a non-zero status.
  """
  valid_sha = _validate_commit_sha(sha)
  if (
      subprocess.run(
          [
              "git",
              "-C",
              os.fspath(pytorch_dir),
              "rev-parse",
              "--verify",
              "--quiet",
              "--end-of-options",
              f"{valid_sha}^{{commit}}",
          ],
          capture_output=True,
          text=True,
          encoding="utf-8",
          errors="replace",
          check=False,
      ).returncode
      == 0
  ):
    return
  _git(
      pytorch_dir,
      "fetch",
      "--no-tags",
      "--depth=1",
      "--filter=blob:none",
      "--end-of-options",
      "origin",
      valid_sha,
  )


def changed_test_paths(
    pytorch_dir: pathlib.Path, base_sha: str, head_sha: str
) -> Sequence[str]:
  """Returns paths under `test/` that differ between the two commit trees.

  `--no-renames` reports a rename as delete+add so both the old and the new
  module are selected, matching what Bazel's glob sees. `-z` emits NUL-delimited
  raw paths without octal escaping. Tree-only (blobless) fetches are sufficient
  because no content is compared.

  Args:
    pytorch_dir: Path to the local PyTorch Git repository.
    base_sha: Base 40-character hexadecimal commit SHA.
    head_sha: Head 40-character hexadecimal commit SHA.

  Returns:
    Sequence of repo-relative paths under `test/` modified between `base_sha`
    and `head_sha`.

  Raises:
    ValueError: If `base_sha` or `head_sha` is not a valid hex commit SHA.
    RuntimeError: If `git diff` exits with a non-zero status.
  """
  valid_base = _validate_commit_sha(base_sha)
  valid_head = _validate_commit_sha(head_sha)
  out = _git(
      pytorch_dir,
      "diff",
      "--name-only",
      "--no-renames",
      "-z",
      "--end-of-options",
      valid_base,
      valid_head,
      "--",
      "test/",
  )
  return tuple(path for path in out.split("\0") if path)


# --------------------------------------------------------------------------
# Outputs.
# --------------------------------------------------------------------------


def render_markdown(
    *,
    base_sha: str,
    head_sha: str,
    changed: Sequence[str],
    impacted: Mapping[str, Sequence[str]],
    targets: Mapping[str, TorchTestTarget],
) -> str:
  """Renders a Markdown job summary of impacted `//tests/torch:*` targets.

  Args:
    base_sha: Base upstream PyTorch commit SHA.
    head_sha: Head upstream PyTorch commit SHA.
    changed: Changed repo-relative paths under `test/`.
    impacted: Mapping from impacted target label to matching changed paths.
    targets: All `torch_tpu_torch_test` targets parsed from `tests/torch/BUILD`.

  Returns:
    Formatted Markdown summary text.
  """
  enabled = [target for target in targets.values() if target.enabled]
  total_shards = sum(target.shard_count for target in enabled)
  impacted_shards = sum(targets[label].shard_count for label in impacted)
  lines = [
      "# PyTorch Out-of-Tree TPU Impacted Tests",
      "",
      f"* **Commit Window**: `{base_sha[:12]} -> {head_sha[:12]}`",
      (
          f"* **Selection Scope**: `{_SELECTION_SCOPE}` (upstream `test/` diffs"
          " only; daily `scope=all` runs cover wheel/library changes)"
      ),
      f"* **Changed `test/` paths**: {len(changed)}",
      (
          f"* **Impacted Enabled Targets**: **{len(impacted)} /"
          f" {len(enabled)}** (**{impacted_shards} / {total_shards}** shards)"
      ),
      "",
  ]
  if impacted:
    lines += ["## Impacted Targets", ""]
    for label, hits in impacted.items():
      lines.append(f"* `{label}` (`test/{targets[label].module}.py`)")
      for hit in hits[:5]:
        lines.append(f"  * `{hit}`")
      if len(hits) > 5:
        lines.append(f"  * … {len(hits) - 5} more")
  else:
    lines.append("No enabled `//tests/torch:*` TPU test targets were impacted.")
  lines.append("")
  return "\n".join(lines)


def _opt_path(raw: str) -> pathlib.Path | None:
  return pathlib.Path(raw) if raw.strip() else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses CLI arguments for computing impacted OOT PyTorch test targets.

  Args:
    argv: Optional argument sequence; defaults to `sys.argv[1:]` when None.

  Returns:
    Parsed `argparse.Namespace` containing paths, SHAs, and output destinations.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--workspace-dir", type=pathlib.Path, default=pathlib.Path.cwd()
  )
  parser.add_argument("--pytorch-dir", type=pathlib.Path, required=True)
  parser.add_argument("--base-sha", required=True)
  parser.add_argument("--head-sha", required=True)
  parser.add_argument("--impacted-targets-output", type=_opt_path, default=None)
  parser.add_argument("--markdown-output", type=_opt_path, default=None)
  parser.add_argument("--json-output", type=_opt_path, default=None)
  parser.add_argument("--github-output", type=_opt_path, default=None)
  return parser.parse_args(argv)


def _write_outputs(
    *,
    args: argparse.Namespace,
    base_sha: str,
    head_sha: str,
    changed: Sequence[str],
    impacted: Mapping[str, Sequence[str]],
    targets: Mapping[str, TorchTestTarget],
) -> None:
  """Writes impacted targets, `$GITHUB_OUTPUT`, Markdown, and JSON artifacts."""
  labels = list(impacted)
  if args.impacted_targets_output is not None:
    args.impacted_targets_output.parent.mkdir(parents=True, exist_ok=True)
    args.impacted_targets_output.write_text(
        "".join(f"{label}\n" for label in labels), encoding="utf-8"
    )
  gh_lines = [
      f"has_impacted_targets={str(bool(labels)).lower()}",
      f"impacted_count={len(labels)}",
      f"selection_scope={_SELECTION_SCOPE}",
  ]
  for line in gh_lines:
    print(f"[ci-output] {line}")
  if args.github_output is not None:
    args.github_output.parent.mkdir(parents=True, exist_ok=True)
    with args.github_output.open("a", encoding="utf-8") as output_file:
      output_file.write("".join(f"{line}\n" for line in gh_lines))

  markdown = render_markdown(
      base_sha=base_sha,
      head_sha=head_sha,
      changed=changed,
      impacted=impacted,
      targets=targets,
  )
  if args.markdown_output is not None:
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    with args.markdown_output.open("a", encoding="utf-8") as output_file:
      output_file.write(markdown)
  else:
    print(markdown)
  if args.json_output is not None:
    payload = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "selection_scope": _SELECTION_SCOPE,
        "changed_test_paths": list(changed),
        "impacted_targets": {
            label: list(hits) for label, hits in impacted.items()
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point for computing and emitting impacted OOT test targets.

  Args:
    argv: Optional CLI argument sequence; defaults to `sys.argv[1:]` when None.

  Returns:
    Process exit code (`0` on success).
  """
  args = parse_args(argv)
  base_sha = _validate_commit_sha(args.base_sha)
  head_sha = _validate_commit_sha(args.head_sha)

  workspace_dir = args.workspace_dir.resolve()
  targets = parse_torch_test_targets(
      (workspace_dir / "tests/torch/BUILD").read_text(encoding="utf-8")
  )
  tables = parse_deps_tables(
      (workspace_dir / "bazel/pytorch_test_repo.bzl").read_text(
          encoding="utf-8"
      )
  )
  validate_deps_tables(targets, tables)

  pytorch_dir = args.pytorch_dir.resolve()
  ensure_commit(pytorch_dir, base_sha)
  ensure_commit(pytorch_dir, head_sha)
  changed = changed_test_paths(pytorch_dir, base_sha, head_sha)
  impacted = compute_impacted(changed, targets, tables)
  _write_outputs(
      args=args,
      base_sha=base_sha,
      head_sha=head_sha,
      changed=changed,
      impacted=impacted,
      targets=targets,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
