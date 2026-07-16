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

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Syncs MODULE.bazel's hand-copied version constants from their sources.

MODULE.bazel cannot `load()` Starlark, so the values whose single source of
truth lives in a `.bzl` file -- the PyTorch version lists and the Python
versions -- are duplicated there by hand. This script rewrites MODULE.bazel in
place so each copy matches its source; it is idempotent, and a no-op when
everything is already in sync.

Run it with `uv run --no-project ci/tools/sync_module_bazel_versions.py`
(--no-project keeps uv from resolving the full torch_tpu project) after
changing one of the source files. The repository root may be passed as an
argument and defaults to the nearest parent directory of this file named
`torch_tpu`. CI runs the script and fails if it produced a diff.

All the files involved are Python-syntax-compatible, so both sides are read
the same way: as top-level `NAME = <literal>` assignments.
"""

import argparse
import ast
from collections.abc import Collection, Mapping, Sequence
import pathlib
import sys
from typing import Final

# The constants MODULE.bazel duplicates, keyed by the repo-relative
# source-of-truth file each is copied from -- shaped like the load() statement
# MODULE.bazel cannot have.
_SYNCED_CONSTANTS: Final[Mapping[pathlib.Path, Sequence[str]]] = {
    pathlib.Path("bazel/pytorch_versions.bzl"): (
        "EXTRA_PYTORCH_VERSIONS",
        "NIGHTLY_TORCH_VERSION",
    ),
    pathlib.Path("bazel/supported_python_versions.bzl"): (
        "SUPPORTED_PYTHON_VERSIONS",
        "DEFAULT_PYTHON_VERSION",
    ),
}


class SyncError(Exception):
  """A version constant cannot be synced."""


def _find_repo_root() -> pathlib.Path:
  """Returns the nearest parent directory of this file named `torch_tpu`."""
  path = pathlib.Path(__file__).resolve()
  for parent in path.parents:
    if parent.name == "torch_tpu":
      return parent
  raise SyncError(
      f"no parent directory of {path} is named torch_tpu; pass the repository"
      " root explicitly"
  )


def _literal_assignments(
    source: str, names: Collection[str]
) -> Mapping[str, ast.Assign]:
  """Returns the top-level `NAME = <literal>` assignments for `names`.

  Only the requested names are collected, and parsing stops as soon as all of
  them have been found. A requested name whose value is not a literal (e.g. a
  concatenation of other constants) is left out.
  """
  wanted = set(names)
  found: dict[str, ast.Assign] = {}
  for node in ast.parse(source).body:
    if not wanted:
      break
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      if isinstance(target, ast.Name) and target.id in wanted:
        try:
          ast.literal_eval(node.value)
        except ValueError:
          continue
        found[target.id] = node
        wanted.discard(target.id)
  return found


def _render(name: str, value: object) -> str:
  """Renders an assignment the way buildifier formats it in MODULE.bazel."""
  if isinstance(value, str):
    return f'{name} = "{value}"'
  if isinstance(value, list):
    items = "".join(f'    "{item}",\n' for item in value)
    return f"{name} = [\n{items}]"
  raise SyncError(f"{name}: unsupported value type {type(value).__name__}")


def _sync_lines(
    lines: Sequence[str], values: Mapping[str, object]
) -> Sequence[str]:
  """Returns `lines` with each constant's assignment synced to `values`.

  `lines` is a MODULE.bazel's contents split with `splitlines(keepends=True)`;
  the result joins back with `"".join()`. An assignment is only rewritten when
  its parsed value differs from the target value, and anything trailing it on
  its last line (an inline comment, the line ending) is preserved, so the
  function is idempotent and leaves in-sync input untouched.

  Raises:
    SyncError: if a constant has no literal assignment in `lines`.
  """
  nodes = _literal_assignments("".join(lines), values.keys())
  missing = sorted(values.keys() - nodes.keys())
  if missing:
    raise SyncError(
        f"{', '.join(missing)} not defined (as a literal assignment) in"
        " MODULE.bazel; add it by hand, with a comment pointing at its source,"
        " so this script can keep it in sync"
    )

  # Splice bottom-up so the line numbers of the nodes above stay valid.
  ordered = sorted(values, key=lambda name: nodes[name].lineno, reverse=True)
  synced = list(lines)
  for name in ordered:
    node = nodes[name]
    if ast.literal_eval(node.value) == values[name]:
      continue
    start, end = node.lineno - 1, node.end_lineno
    tail = synced[end - 1][node.end_col_offset :]
    synced[start:end] = [_render(name, values[name]) + tail]
  return synced


def _source_values(repo_root: pathlib.Path) -> Mapping[str, object]:
  """Reads every synced constant's target value from its source file.

  Raises:
    SyncError: if a source file does not define one of its constants.
  """
  values: dict[str, object] = {}
  for source, names in _SYNCED_CONSTANTS.items():
    nodes = _literal_assignments((repo_root / source).read_text(), names)
    missing = [name for name in names if name not in nodes]
    if missing:
      raise SyncError(
          f"{', '.join(missing)} not defined (as a literal assignment) in"
          f" {source}"
      )
    values |= {name: ast.literal_eval(nodes[name].value) for name in names}
  return values


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
      "repo_root",
      nargs="?",
      type=pathlib.Path,
      help=(
          "Repository root to sync (default: the nearest parent directory of"
          " this script named torch_tpu)."
      ),
  )
  args = parser.parse_args(argv)

  try:
    repo_root = args.repo_root if args.repo_root is not None else _find_repo_root()
    module_bazel = repo_root / "MODULE.bazel"
    lines = module_bazel.read_text().splitlines(keepends=True)
    synced = _sync_lines(lines, _source_values(repo_root))
  except (SyncError, OSError) as error:
    print(f"ERROR: {error}", file=sys.stderr)
    return 1

  if list(synced) != lines:
    module_bazel.write_text("".join(synced))
    print("MODULE.bazel updated; review the diff and commit it.")
  else:
    print("MODULE.bazel is in sync.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
