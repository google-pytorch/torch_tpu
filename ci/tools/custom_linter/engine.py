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

"""Custom presubmit convention checker for TorchTPU OSS repository.

This tool validates coding conventions (such as banned Bazel rules or C++
patterns). The rules are read from the same METADATA presubmit configuration
that the internal monorepo uses (`custom_lint_rules.METADATA`, exported to OSS
as `ci/custom_lint_rules.textproto`); see metadata.py for the supported
fields.

It supports incremental checking (`check_delta_only: true`) by parsing unified
diffs from `git diff -U0` (in Git/GitHub Actions CI), `hg diff -U0`, or
`jj diff` (in Mercurial/Jujutsu workspaces), ensuring developers and PRs are
only alerted to violations on new or modified lines. For Git, the base is
resolved via `git merge-base <base> HEAD` (falling back to `<base>` in shallow
CI checkouts) and diffed against the working tree so uncommitted changes are
included. If diff computation fails in diff mode, it exits with an error rather
than silently falling back to a full scan; pass `--all` to check all files.

Rules with `disable_tags` can be suppressed by adding `TAG=<reason>` to the
commit/CL description. In OSS GitHub Actions CI, description tags only work for
commits authored on GitHub (read from `<base>..HEAD^2` on PR merge commits or
`<base>..HEAD`); Copybara-exported PRs only carry the public commit text, so
inline suppression comments (`suppression_regexp`, e.g. `# FOO_OK=...`) are the
portable mechanism across both the internal monorepo and OSS.
"""

import argparse
import bisect
import codecs
from collections.abc import Iterable, Mapping, Sequence
import concurrent.futures
import dataclasses
import functools
import json
import multiprocessing
import os
import pathlib
import posixpath
import re
import sys
import threading
import time
from typing import Any, Final, TextIO

from ci.tools.custom_linter import metadata as custom_lint_metadata
from ci.tools.custom_linter import vcs as custom_lint_vcs

# Root directory name of the internal monorepo; its rules use canonical
# `//depot/<name>/<project path>/...` paths. This is the only monorepo-specific
# value in the linter; Copybara rewrites it on export.
MONOREPO_DIR_NAME: Final[str] = 'monorepo'
_DEPOT_ROOT: Final[str] = f'//depot/{MONOREPO_DIR_NAME}'

Rule = custom_lint_metadata.Rule
RuleSet = custom_lint_metadata.RuleSet
PathMapper = custom_lint_metadata.PathMapper
LoadError = custom_lint_metadata.LoadError

Vcs = custom_lint_vcs.Vcs
GitVcs = custom_lint_vcs.GitVcs
HgVcs = custom_lint_vcs.HgVcs
JjVcs = custom_lint_vcs.JjVcs
VcsError = custom_lint_vcs.VcsError
VCS_TYPES = custom_lint_vcs.VCS_TYPES
VCS_AUTO = custom_lint_vcs.VCS_AUTO
VCS_ANY = custom_lint_vcs.VCS_ANY
_resolve_path = custom_lint_vcs.resolve_path
find_vcs_root = custom_lint_vcs.find_vcs_root

BINARY_SNIFF_BYTES: Final[int] = 8192

# Rules file of the monorepo layout. Its `path_*` fields are canonical
# `_DEPOT_ROOT/<project path>/...` paths, where the project path is the
# project root's location under `MONOREPO_DIR_NAME` (see
# `_monorepo_canonical_root`).
MONOREPO_RULES_FILE: Final[str] = 'custom_lint_rules.METADATA'

# In OSS, Copybara rewrites the monorepo canonical root in the rules' `path_*`
# fields to `torch_tpu`, and moves files around. These maps undo the moves so
# that OSS paths can be matched against the same rules.
OSS_CANONICAL_ROOT: Final[str] = 'torch_tpu/oss'
OSS_RULES_FILE: Final[str] = 'ci/custom_lint_rules.textproto'
OSS_PATH_MAPS: Final[tuple[tuple[str, str], ...]] = (
    (OSS_RULES_FILE, 'torch_tpu/custom_lint_rules.textproto'),
    ('src/torch_tpu', 'torch_tpu'),
    ('LICENSE', 'torch_tpu/LICENSE'),
    ('build_files', 'torch_tpu/build_files'),
    ('csrc', 'torch_tpu/csrc'),
    ('docker', 'torch_tpu/docker'),
    ('docs', 'torch_tpu/docs'),
    ('examples', 'torch_tpu/examples'),
    ('perflab', 'torch_tpu/perflab'),
    ('tests', 'torch_tpu/tests'),
    # Files that Copybara moves from oss/torch_tpu/ over their monorepo version.
    (
        'csrc/common/contain.cc',
        'torch_tpu/oss/torch_tpu/csrc/common/contain.cc',
    ),
    (
        'csrc/common/environment.cc',
        'torch_tpu/oss/torch_tpu/csrc/common/environment.cc',
    ),
    (
        'csrc/common/environment_test.cc',
        'torch_tpu/oss/torch_tpu/csrc/common/environment_test.cc',
    ),
    (
        'src/torch_tpu/_internal/distributed/multiprocessing.py',
        'torch_tpu/oss/torch_tpu/_internal/distributed/multiprocessing.py',
    ),
    (
        'src/torch_tpu/_internal/benchmarks/export_adapter.py',
        'torch_tpu/oss/torch_tpu/_internal/benchmarks/export_adapter.py',
    ),
    (
        'src/torch_tpu/_internal/benchmarks/xprof_adapter.py',
        'torch_tpu/oss/torch_tpu/_internal/benchmarks/xprof_adapter.py',
    ),
)


def _monorepo_rel(path: pathlib.Path) -> str | None:
  """Returns path relative to its enclosing `MONOREPO_DIR_NAME` dir, or None."""
  for parent in (path, *path.parents):
    if parent.name == MONOREPO_DIR_NAME:
      return path.relative_to(parent).as_posix()
  return None


@functools.lru_cache(maxsize=1)
def _tool_project_rel() -> str | None:
  """Returns this tool's project path inside the monorepo, if any.

  Derived from this file's location (source tree or build runfiles): the
  nearest parent with the monorepo rules file, relative to the enclosing
  `MONOREPO_DIR_NAME` directory. None outside the monorepo (e.g. in OSS).
  """
  tool = pathlib.Path(__file__).absolute()
  for parent in tool.parents:
    if (parent / MONOREPO_RULES_FILE).is_file():
      return _monorepo_rel(parent)
  return None


def _monorepo_canonical_root(root_dir: pathlib.Path) -> str:
  """Returns the canonical `_DEPOT_ROOT/...` path of the project at root_dir.

  The project path is root_dir's location inside the monorepo. If root_dir is
  not inside it (e.g. a copy of the project elsewhere), the tool's own project
  path is used, so the copy is matched as the project the rules were written
  for.

  Args:
    root_dir: Project root containing `MONOREPO_RULES_FILE`.
  """
  for candidate in (root_dir.absolute(), _resolve_path(root_dir)):
    rel = _monorepo_rel(candidate)
    if rel is not None:
      break
  else:
    rel = _tool_project_rel() or root_dir.name
  return _DEPOT_ROOT if rel in ('', '.') else posixpath.join(_DEPOT_ROOT, rel)


def default_path_mapper(root_dir: pathlib.Path | None = None) -> PathMapper:
  """Returns the mapper from project-relative to canonical rule paths."""
  target_root = root_dir if root_dir is not None else find_torch_tpu_root()
  if (target_root / MONOREPO_RULES_FILE).is_file():
    return PathMapper(_monorepo_canonical_root(target_root))
  if (target_root / OSS_RULES_FILE).is_file():
    return PathMapper(OSS_CANONICAL_ROOT, OSS_PATH_MAPS)
  fallback_root = find_torch_tpu_root()
  if (fallback_root / OSS_RULES_FILE).is_file():
    return PathMapper(OSS_CANONICAL_ROOT, OSS_PATH_MAPS)
  return PathMapper(_monorepo_canonical_root(target_root))


@dataclasses.dataclass(frozen=True)
class CandidateFile:
  """A candidate file paired with matching rules to avoid redundant evaluation."""

  file_path: pathlib.Path
  rel_path: str
  matching_rules: Sequence[Rule]


@dataclasses.dataclass(frozen=True)
class Violation:
  """Represents a rule violation in a specific file and line."""

  rule_id: str
  file_path: str
  # 1-based line; None for whole-file findings (e.g. a missing required
  # pattern, or a file operation).
  line_number: int | None
  line_content: str
  message: str
  severity: str = 'ERROR'

  @property
  def location(self) -> str:
    if self.line_number is None:
      return self.file_path
    return f'{self.file_path}:{self.line_number}'


class ModifiedLines(dict[str, set[int]]):
  """Maps project-relative paths to their added or modified line numbers.

  Attributes:
    operations: Maps paths to the `FileOperation` (ADD, EDIT, BRANCH) that the
      change performs on them.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.operations: dict[str, str] = {}


def normalize_path(path: pathlib.Path, root_dir: pathlib.Path) -> str:
  """Returns path relative to root_dir as POSIX string, or str(path) if not relative."""
  try:
    return path.relative_to(root_dir).as_posix()
  except ValueError:
    pass
  try:
    return path.resolve().relative_to(_resolve_path(root_dir)).as_posix()
  except ValueError:
    return path.as_posix()


_DIFF_HEADER_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
    r'\t|\s{2,}|\s+\d{4}-\d{2}-\d{2}'
)
_DIFF_HUNK_RE: Final[re.Pattern[str]] = re.compile(
    r'^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'
)
_DEV_NULL: Final[tuple[str, ...]] = ('/dev/null', 'dev/null', '')


def normalize_diff_path(
    path: str,
    root_dir: pathlib.Path | None = None,
    *,
    prefix: str = '',
) -> str:
  """Normalizes a VCS diff file path to be relative to the project root.

  Strips leading `a/` or `b/` diff prefixes and `prefix` (the project's path
  under the VCS root). When `prefix` is non-empty, paths outside `prefix`
  return `''` so callers omit them.

  Args:
    path: Raw file path from a unified diff header.
    root_dir: Optional project root directory used to normalize absolute paths.
    prefix: Optional POSIX directory prefix of the project under the VCS root.

  Returns:
    The project-relative POSIX path, or `''` if the file lies outside the
    project.
  """
  if path in _DEV_NULL:
    return ''
  if path.startswith('b/') or path.startswith('a/'):
    path = path[2:]
  if prefix:
    norm_prefix = prefix if prefix.endswith('/') else f'{prefix}/'
    if not path.startswith(norm_prefix):
      return ''
    return path[len(norm_prefix) :]
  if root_dir is not None and path.startswith('/'):
    resolved_root = _resolve_path(root_dir)
    resolved_path = _resolve_path(pathlib.Path(path))
    try:
      return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
      return ''
  return path


def _diff_header_path(header: str) -> str:
  """Returns the path of a `--- `/`+++ ` header without trailing metadata."""
  # Unified diff headers append metadata (timestamp, file mode) after the file
  # path, separated by a tab (\t), 2+ spaces (\s{2,}), or a space before an ISO
  # date (\s+\d{4}-\d{2}-\d{2}).
  return _DIFF_HEADER_SPLIT_RE.split(header.strip(), maxsplit=1)[0].strip()


def parse_unified_diff(
    diff_text: str,
    root_dir: pathlib.Path | None = None,
    *,
    prefix: str = '',
) -> ModifiedLines:
  """Parses a unified diff and maps file relative paths to modified lines.

  Also records the file operation of each file in `operations`: ADD for new
  files, BRANCH for unmodified copies and renames (ADD if modified afterwards),
  and EDIT otherwise. Deleted files and files outside `prefix` are omitted.

  Args:
    diff_text: Output of `git diff`, `hg diff` or `jj diff --git`.
    root_dir: Project root used to normalize absolute paths.
    prefix: POSIX directory prefix of the project under the VCS root.

  Returns:
    The modified lines per file.
  """
  modified_lines = ModifiedLines()
  added: set[str] = set()
  copied: set[str] = set()
  deleted: set[str] = set()
  current_file: str | None = None
  # Path of the latest `--- ` header; names the file a `+++ /dev/null` deletes.
  old_path = ''
  # Per-file header state, reset by `diff ...` lines.
  pending_added = pending_copied = pending_deleted = False
  # Remaining old/new lines of the current hunk; hunk bodies are skipped so
  # that content lines starting with `--- `/`+++ ` are not taken as headers.
  old_left = new_left = 0

  def register(path: str) -> str | None:
    if path in _DEV_NULL:
      return None
    normalized = normalize_diff_path(path, root_dir=root_dir, prefix=prefix)
    if normalized in _DEV_NULL:
      return None
    modified_lines.setdefault(normalized, set())
    for flag, paths in (
        (pending_added, added),
        (pending_copied, copied),
        (pending_deleted, deleted),
    ):
      if flag:
        paths.add(normalized)
    return normalized

  for line in diff_text.split('\n'):
    line = line.removesuffix('\r')
    if old_left > 0 or new_left > 0:
      if line.startswith('+'):
        new_left -= 1
        continue
      if line.startswith('-'):
        old_left -= 1
        continue
      if line.startswith(' '):
        old_left -= 1
        new_left -= 1
        continue
      if line.startswith('\\'):
        continue
      old_left = new_left = 0  # Truncated hunk; resume header parsing.
    if line.startswith('\\'):
      continue  # `\ No newline at end of file`.
    if line.startswith('diff '):
      current_file = None
      old_path = ''
      pending_added = pending_copied = pending_deleted = False
      if line.startswith('diff --git ') and ' b/' in line:
        current_file = register(line[line.rindex(' b/') + 1 :])
    elif line.startswith('new file mode'):
      pending_added = True
      if current_file is not None:
        added.add(current_file)
    elif line.startswith('deleted file mode'):
      pending_deleted = True
      if current_file is not None:
        deleted.add(current_file)
    elif line.startswith(('rename to ', 'copy to ')):
      pending_copied = True
      current_file = register(line.split(' ', 2)[2])
    elif line.startswith('--- '):
      # Starts a file section (plain unified diffs have no `diff` line).
      # Git-style `new file mode`/`rename to` lines were already recorded.
      current_file = None
      old_path = _diff_header_path(line[4:])
      pending_added = old_path in _DEV_NULL
      pending_copied = pending_deleted = False
    elif line.startswith('+++ '):
      path_part = _diff_header_path(line[4:])
      if path_part in _DEV_NULL:
        # The deleted file is the one named by the preceding `--- ` header
        # (git diffs also record it via `deleted file mode`).
        pending_deleted = True
        if old_path not in _DEV_NULL:
          norm_old = normalize_diff_path(
              old_path, root_dir=root_dir, prefix=prefix
          )
          if norm_old:
            deleted.add(norm_old)
        current_file = None
      else:
        current_file = register(path_part)
    elif line.startswith('@@ ') and current_file is not None:
      # Hunk header: @@ -old_start[,old_count] +new_start[,new_count] @@
      match = _DIFF_HUNK_RE.match(line)
      if match:
        old_left = int(match.group(1)) if match.group(1) is not None else 1
        start_line = int(match.group(2))
        count = int(match.group(3)) if match.group(3) is not None else 1
        new_left = count
        if count > 0:
          modified_lines[current_file].update(
              range(start_line, start_line + count)
          )

  for path in deleted:
    modified_lines.pop(path, None)
  for path in modified_lines:
    if path in copied:
      op = 'ADD' if modified_lines[path] else 'BRANCH'
    elif path in added:
      op = 'ADD'
    else:
      op = 'EDIT'
    modified_lines.operations[path] = op
  return modified_lines


def get_modified_lines_from_vcs(
    base_sha: str | None = None,
    root_dir: pathlib.Path | None = None,
    *,
    vcs: str = VCS_AUTO,
) -> ModifiedLines:
  """Returns the modified lines per file from the VCS diff.

  Args:
    base_sha: git base to diff against.
    root_dir: Project root; defaults to the current working directory.
    vcs: Which VCS to ask; see VCS_AUTO, VCS_ANY and VCS_TYPES.

  Returns:
    The modified lines per file.

  Raises:
    VcsError: If no VCS diff command succeeded.
  """
  resolved_root = _resolve_path(root_dir or pathlib.Path.cwd())
  backend = custom_lint_vcs.for_name(vcs, resolved_root)
  diff_text = backend.diff(base_sha)
  prefix = backend.prefix(resolved_root)
  return parse_unified_diff(diff_text, root_dir=resolved_root, prefix=prefix)


def get_commit_message(
    base_sha: str | None,
    root_dir: pathlib.Path | None = None,
    *,
    vcs: str = VCS_AUTO,
) -> str:
  """Retrieves current CL/commit messages to check for disable tags.

  Args:
    base_sha: git base; commits after it are included.
    root_dir: Directory used to detect the VCS; defaults to the cwd.
    vcs: Which VCS to ask; see VCS_AUTO, VCS_ANY and VCS_TYPES.

  Returns:
    The commit message(s), or '' if the message is empty.

  Raises:
    VcsError: If no VCS command succeeded.
  """
  backend = custom_lint_vcs.for_name(vcs, root_dir)
  return backend.log(base_sha)


def is_rule_disabled(rule: Rule, commit_message: str) -> bool:
  """Returns True if a `TAG=...` line in commit_message disables the rule."""
  if not commit_message or not rule.disable_tags:
    return False
  tags = custom_lint_metadata.parse_tags(commit_message)
  return any(tag in tags for tag in rule.disable_tags)


DEFAULT_EXCLUDED_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r'(^|/)(custom_linter/([^/]+/)*[^/]+\.py|'
    r'custom_lint[^/]*\.py|custom_lint_rules\.(METADATA|textproto)|'
    r'copy\.bara\.sky)$|'
    r'.*\.(orig|bak|gz|tgz|bz2|xz|zst|zip|tar|whl|so|a|o|dylib|dll|'
    r'pyc|pyo|png|jpg|jpeg|gif|ico|webp|pdf|woff|woff2|ttf|otf|'
    r'npz|npy|pt|pth|ckpt|safetensors|parquet|onnx|pb)$|'
    r'.*\.del\..*'
)


def is_path_excluded(rel_path: str) -> bool:
  """Returns True if the file path is globally excluded from custom lint checks."""
  return DEFAULT_EXCLUDED_PATH_RE.search(rel_path) is not None


def find_rules_file(root_dir: pathlib.Path) -> pathlib.Path:
  """Returns the default rules file for the project rooted at root_dir."""
  monorepo_rules = root_dir / MONOREPO_RULES_FILE
  if monorepo_rules.is_file():
    return monorepo_rules
  oss_rules = root_dir / OSS_RULES_FILE
  if oss_rules.is_file():
    return oss_rules
  return monorepo_rules


def load_rules(
    rules_file: pathlib.Path | str | None = None,
    root_dir: pathlib.Path | None = None,
    mapper: PathMapper | None = None,
) -> RuleSet:
  """Loads the custom lint rules.

  Relative rule paths are resolved against the canonical directory of the rules
  file when it lies inside the project, and against the canonical root
  otherwise.

  Args:
    rules_file: METADATA-format rules file; defaults to the project's.
    root_dir: Project root; defaults to the torch_tpu root.
    mapper: Maps project-relative paths to canonical paths.

  Returns:
    The loaded rules.
  """
  root_dir = root_dir or find_torch_tpu_root()
  mapper = mapper or default_path_mapper(root_dir)
  if rules_file is None:
    path = find_rules_file(root_dir)
    rel = normalize_path(path, root_dir)
  else:
    path = pathlib.Path(rules_file)
    rel = normalize_path(path, root_dir)
    if pathlib.Path(rel).is_absolute() or rel.startswith('../'):
      rel = None
  base_dir = (
      posixpath.dirname(mapper.to_canonical(rel))
      if rel
      else mapper.canonical_root
  )
  return custom_lint_metadata.load_rules_file(path, base_dir)


@functools.lru_cache(maxsize=1)
def load_default_rules() -> RuleSet:
  """Loads and caches the project's rules."""
  return load_rules()


def _read_text_file(path: pathlib.Path) -> str | None:
  """Reads a UTF-8 text file, returning None if unreadable or binary."""
  try:
    with open(path, 'rb') as f:
      chunk = f.read(BINARY_SNIFF_BYTES)
      if b'\x00' in chunk:
        return None
      if len(chunk) < BINARY_SNIFF_BYTES:
        text = chunk.decode('utf-8')
      else:
        codecs.utf_8_decode(chunk, 'strict', False)
        rest = f.read()
        if b'\x00' in rest:
          return None
        text = (chunk + rest).decode('utf-8')
      if '\r' in text:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
      return text
  except (OSError, UnicodeDecodeError):
    return None


def _compute_line_starts(content: str) -> list[int]:
  """Computes 0-based character start offsets for each line in content."""
  line_starts = [0]
  pos = 0
  n = len(content)
  while True:
    idx = content.find('\n', pos)
    if idx == -1 or idx + 1 >= n:
      break
    line_starts.append(idx + 1)
    pos = idx + 1
  return line_starts


class TextFile:
  r"""A file's text with lazily computed lines and line offsets.

  Lines are split on `\n` only (unlike str.splitlines(), which also splits on
  form feeds and other separators), so line numbers agree with diffs.
  """

  __slots__ = ('text', '_lines', '_line_starts')

  def __init__(self, text: str):
    self.text = text
    self._lines: list[str] | None = None
    self._line_starts: list[int] | None = None

  @property
  def lines(self) -> list[str]:
    if self._lines is None:
      lines = self.text.split('\n')
      if lines and not lines[-1]:
        lines.pop()
      self._lines = lines
    return self._lines

  @property
  def line_starts(self) -> list[int]:
    if self._line_starts is None:
      self._line_starts = _compute_line_starts(self.text)
    return self._line_starts

  @property
  def line_count(self) -> int:
    return len(self.lines)

  def line_of(self, offset: int) -> int:
    """Returns the 1-based line containing character `offset`."""
    return bisect.bisect_right(self.line_starts, offset)

  def span(self, first: int, last: int) -> tuple[int, int]:
    """Returns the character range of lines first..last, without the final newline."""
    starts = self.line_starts
    start = starts[first - 1]
    if last < len(starts):
      end = starts[last] - 1  # The newline that ends line `last`.
    else:
      end = len(self.text)
      if end > start and self.text[end - 1] == '\n':
        end -= 1
    return start, max(start, end)


def _literals_present(rule: Rule, content: str) -> bool:
  """Fast literal pre-filter: False if rule.prohibited_re cannot match content."""
  singletons = rule.required_singletons
  if singletons and not all(s in content for s in singletons):
    return False
  disjunctions = rule.required_disjunctions
  return not disjunctions or all(
      any(s in content for s in clause) for clause in disjunctions
  )


# A finding: 1-based line (or None) and an optional message override.
_Finding = tuple[int | None, str | None]


def _check_lines(
    rule: Rule,
    prohibited: re.Pattern[str],
    text: TextFile,
    changed: set[int] | None,
    findings: list[_Finding],
) -> None:
  """Line mode: reports lines matching prohibited but not suppression."""
  if changed is not None:
    n = text.line_count
    candidates = sorted(i for i in changed if 1 <= i <= n)
  elif rule.prohibited_is_line_local:
    # Narrow to lines overlapping a whole-file match (see is_line_local).
    cand: set[int] = set()
    multiline = rule.prohibited_multiline_re
    assert multiline is not None
    for m in multiline.finditer(text.text):
      first = text.line_of(m.start())
      last = text.line_of(max(m.start(), m.end() - 1))
      cand.update(range(first, last + 1))
    if not cand:
      return
    n = text.line_count
    candidates = sorted(i for i in cand if 1 <= i <= n)
  else:
    candidates = range(1, text.line_count + 1)
  lines = text.lines
  supp = rule.suppression_re
  for i in candidates:
    line = lines[i - 1]
    if prohibited.search(line) and not (supp and supp.search(line)):
      findings.append((i, None))


def _chunks(changed: set[int] | None, line_count: int) -> list[list[int]]:
  """Groups line numbers into runs of consecutive lines, clamped to the file."""
  if line_count == 0:
    return []
  if changed is None:
    return [[1, line_count]]
  out: list[list[int]] = []
  for n in sorted(changed):
    if not 1 <= n <= line_count:
      continue
    if out and out[-1][1] + 1 == n:
      out[-1][1] = n
    else:
      out.append([n, n])
  return out


def _check_chunks(
    rule: Rule,
    prohibited: re.Pattern[str],
    text: TextFile,
    changed: set[int] | None,
    findings: list[_Finding],
) -> None:
  """Modified-chunks mode: matches runs of changed lines as one text."""
  supp = rule.suppression_re
  for first, last in _chunks(changed, text.line_count):
    start, end = text.span(first, last)
    chunk = text.text[start:end]
    if supp and supp.search(chunk):
      continue
    for m in prohibited.finditer(chunk):
      findings.append((text.line_of(start + m.start()), None))


def _any_line_matches(rule: Rule, text: TextFile) -> bool:
  """Returns whether rule.required_re matches at least one line."""
  required = rule.required_re
  assert required is not None
  if rule.required_is_line_local:
    multiline = rule.required_multiline_re
    assert multiline is not None
    m = multiline.search(text.text)
    if m is None:
      return False
    if '\n' not in m.group():
      return True
  return any(required.search(line) for line in text.lines)


def _evaluate(
    rule: Rule, text: TextFile, changed: set[int] | None
) -> list[_Finding]:
  """Evaluates a check_contents rule against one file."""
  findings: list[_Finding] = []
  content = text.text
  prohibited = rule.prohibited_re
  if prohibited is not None and not _literals_present(rule, content):
    prohibited = None

  if rule.mode == custom_lint_metadata.MODE_ENTIRE_FILE:
    limit = custom_lint_metadata.MAX_ENTIRE_FILE_BYTES
    if len(content) > limit // 4 and len(content.encode('utf-8')) > limit:
      if not rule.ignore_large_files:
        findings.append((
            None,
            (
                f'File is larger than {limit} bytes and cannot be matched by'
                f' {rule.id}.'
            ),
        ))
      return findings
    if prohibited is not None:
      supp = rule.suppression_re
      if supp is None or not supp.search(content):
        findings.extend(
            (text.line_of(m.start()), None)
            for m in prohibited.finditer(content)
        )
    if rule.required_re is not None and not rule.required_re.search(content):
      findings.append((None, None))
    return findings

  if prohibited is not None:
    if rule.mode == custom_lint_metadata.MODE_MODIFIED_CHUNKS:
      _check_chunks(rule, prohibited, text, changed, findings)
    else:
      _check_lines(rule, prohibited, text, changed, findings)
  if rule.required_re is not None and not _any_line_matches(rule, text):
    findings.append((None, None))
  return findings


def _file_operation(
    modified_lines: Mapping[str, set[int]] | None, rel_path: str
) -> str | None:
  operations = getattr(modified_lines, 'operations', None)
  return operations.get(rel_path) if operations else None


def _rule_applies(
    rule: Rule,
    rel_path: str,
    modified_lines: Mapping[str, set[int]] | None,
    operation: str | None,
) -> bool:
  """Applies the rule's operation and delta-only filters."""
  if not rule.applies_to(operation):
    return False
  if (
      rule.check_delta_only
      and modified_lines is not None
      and rel_path not in modified_lines
  ):
    return False
  return True


def _violations(
    rule: Rule,
    rel_path: str,
    text: TextFile | None,
    findings: Sequence[_Finding],
) -> list[Violation]:
  """Builds Violation objects from raw findings."""
  out = []
  for line_number, message in findings:
    line_content = ''
    if line_number is not None and text is not None:
      lines = text.lines
      if 1 <= line_number <= len(lines):
        line_content = lines[line_number - 1].strip()
    out.append(
        Violation(
            rule_id=rule.id,
            file_path=rel_path,
            line_number=line_number,
            line_content=line_content,
            message=message or rule.failure_message,
            severity=rule.severity,
        )
    )
  return out


def check_file_content(
    content: str,
    rel_path: str,
    rule: Rule,
    modified_lines: Mapping[str, set[int]] | None = None,
    *,
    canonical_path: str | None = None,
    file_operation: str | None = None,
    _paths_prechecked: bool = False,
    _text: TextFile | None = None,
) -> Sequence[Violation]:
  """Checks file content against a rule and returns any violations.

  Args:
    content: The file's text.
    rel_path: Project-relative path, used in reports and to look up
      modified_lines.
    rule: The rule to check.
    modified_lines: Changed lines per file, or None without a change context
      (every line is checked).
    canonical_path: Path matched against the rule's path filters; defaults to
      rel_path mapped by default_path_mapper().
    file_operation: How the change touched the file; defaults to
      `modified_lines.operations[rel_path]` when available.
    _paths_prechecked: Skip the path filters (already applied by the caller).
    _text: Precomputed TextFile for content.

  Returns:
    The violations found.
  """
  if not _paths_prechecked:
    if is_path_excluded(rel_path):
      return []
    if canonical_path is None:
      canonical_path = default_path_mapper().to_canonical(rel_path)
    if not rule.matches_path(canonical_path):
      return []
  if file_operation is None:
    file_operation = _file_operation(modified_lines, rel_path)
  if not _rule_applies(rule, rel_path, modified_lines, file_operation):
    return []
  if rule.kind == custom_lint_metadata.CHECK_FILE_OPERATION:
    return _violations(rule, rel_path, None, [(None, None)])
  text = _text if _text is not None else TextFile(content)
  changed = (
      modified_lines.get(rel_path)
      if rule.check_delta_only and modified_lines is not None
      else None
  )
  return _violations(rule, rel_path, text, _evaluate(rule, text, changed))


def _build_path_matcher(
    rules: Sequence[Rule],
) -> list[
    tuple[
        re.Pattern[str],
        list[tuple[re.Pattern[str] | None, list[tuple[int, Rule]]]],
    ]
]:
  """Groups rules by (path_regex, path_exclusion_regex) to avoid duplicate regex checks."""
  by_path: dict[str, list[tuple[int, Rule]]] = {}
  for idx, r in enumerate(rules):
    by_path.setdefault(r.path_regex, []).append((idx, r))
  compiled_groups = []
  for _, indexed_rules in by_path.items():
    path_re = indexed_rules[0][1].path_re
    by_excl: dict[str | None, list[tuple[int, Rule]]] = {}
    for idx, r in indexed_rules:
      by_excl.setdefault(r.path_exclusion_regex, []).append((idx, r))
    excl_groups = [
        (irules[0][1].path_exclusion_re, irules)
        for _, irules in by_excl.items()
    ]
    compiled_groups.append((path_re, excl_groups))
  return compiled_groups


def _match_rules_grouped(
    canonical_path: str,
    compiled_groups: Sequence[
        tuple[
            re.Pattern[str],
            Sequence[tuple[re.Pattern[str] | None, Sequence[tuple[int, Rule]]]],
        ]
    ],
) -> list[Rule]:
  """Returns rules matching canonical_path in original rule sequence order."""
  matched: list[tuple[int, Rule]] = []
  for path_re, excl_groups in compiled_groups:
    if path_re.match(canonical_path):
      for excl_re, irules in excl_groups:
        if excl_re is None or not excl_re.match(canonical_path):
          matched.extend(irules)
  if len(matched) > 1:
    matched.sort(key=lambda x: x[0])
  return [r for _, r in matched]


# Directory names skipped when walking. Build-tool convenience symlinks (e.g.
# `bazel-bin`) are never followed anyway, since the walker does not descend into
# symlinked directories; they are listed for checkouts that materialize them.
DEFAULT_IGNORE_DIRS: Final[frozenset[str]] = frozenset({
    '.git',
    '.hg',
    '.jj',
    '.nox',
    '__pycache__',
    'node_modules',
    'venv',
    '.venv',
    'bazel-bin',
    'bazel-out',
    'bazel-testlogs',
})


def find_candidate_files(
    root_dir: pathlib.Path,
    rules: Sequence[Rule],
    *,
    max_workers: int = 16,
    mapper: PathMapper | None = None,
    rel_prefix: str = '',
    ignore_dirs: frozenset[str] | None = None,
    no_default_excludes: bool = False,
) -> Sequence[CandidateFile]:
  """Finds all files under root_dir that match at least one rule's path filters.

  Args:
    root_dir: Directory to scan.
    rules: Rules whose path filters select candidates.
    max_workers: Directory-walk threads.
    mapper: Maps project-relative paths to canonical paths; rel paths are
      matched as-is when None.
    rel_prefix: Project-relative path of root_dir ('' for the project root).
    ignore_dirs: Exact directory names to skip when walking; defaults to
      `DEFAULT_IGNORE_DIRS` unless `no_default_excludes` is True.
    no_default_excludes: If True and `ignore_dirs` is None, do not apply default
      directory exclusions.

  Returns:
    The candidates, sorted by rel_path.
  """
  if not rules:
    return []
  if ignore_dirs is None:
    ignore_dirs = frozenset() if no_default_excludes else DEFAULT_IGNORE_DIRS
  elif not isinstance(ignore_dirs, frozenset):
    ignore_dirs = frozenset(ignore_dirs)
  prefixes: list[str] = []
  if '.git' in ignore_dirs:
    prefixes.append('.')
  if 'bazel-bin' in ignore_dirs:
    prefixes.append('bazel-')
  ignore_prefixes = tuple(prefixes)

  compiled_groups = _build_path_matcher(rules)
  to_canonical = mapper.to_canonical if mapper is not None else str
  rel_prefix = rel_prefix.strip('/')

  def scan_entries(
      dirpath: str, rel_dir: str
  ) -> tuple[list[CandidateFile], list[tuple[str, str]]]:
    local_candidates: list[CandidateFile] = []
    subdirs: list[tuple[str, str]] = []
    try:
      with os.scandir(dirpath) as it:
        for entry in it:
          name = entry.name
          if entry.is_dir(follow_symlinks=False):
            if name not in ignore_dirs and (
                not ignore_prefixes or not name.startswith(ignore_prefixes)
            ):
              sub_rel = f'{rel_dir}/{name}' if rel_dir else name
              subdirs.append((entry.path, sub_rel))
          else:
            rel_path = f'{rel_dir}/{name}' if rel_dir else name
            if is_path_excluded(rel_path):
              continue
            matching = _match_rules_grouped(
                to_canonical(rel_path), compiled_groups
            )
            if matching:
              local_candidates.append(
                  CandidateFile(
                      file_path=pathlib.Path(entry.path),
                      rel_path=rel_path,
                      matching_rules=matching,
                  )
              )
    except OSError:
      pass
    return local_candidates, subdirs

  def scan_dir_sync(dirpath: str, rel_dir: str) -> list[CandidateFile]:
    local_candidates, subdirs = scan_entries(dirpath, rel_dir)
    for sub_path, sub_rel in subdirs:
      local_candidates.extend(scan_dir_sync(sub_path, sub_rel))
    return local_candidates

  if max_workers <= 1:
    candidates = scan_dir_sync(str(root_dir), rel_prefix)
    candidates.sort(key=lambda c: c.rel_path)
    return candidates

  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:

    def scan_dir_async(
        dirpath: str, rel_dir: str
    ) -> tuple[list[CandidateFile], list[concurrent.futures.Future[Any]]]:
      local_candidates, subdirs = scan_entries(dirpath, rel_dir)
      child_futs = [
          pool.submit(scan_dir_async, sub_path, sub_rel)
          for sub_path, sub_rel in subdirs
      ]
      return local_candidates, child_futs

    def collect(fut: concurrent.futures.Future[Any]) -> list[CandidateFile]:
      out, child_futs = fut.result()
      for cf in child_futs:
        out.extend(collect(cf))
      return out

    candidates = collect(pool.submit(scan_dir_async, str(root_dir), rel_prefix))

  candidates.sort(key=lambda c: c.rel_path)
  return candidates


def find_torch_tpu_root(
    start_path: pathlib.Path | None = None,
) -> pathlib.Path:
  """Finds the root directory of the project repository for tests and scripts."""

  def is_project_root(p: pathlib.Path) -> bool:
    return (
        (p / MONOREPO_RULES_FILE).is_file()
        or (p / OSS_RULES_FILE).is_file()
        or (p / 'pyproject.toml').is_file()
    )

  if start_path:
    resolved_start = _resolve_path(start_path)
    for p in (resolved_start, *resolved_start.parents):
      if is_project_root(p):
        return p

  for tool_path in (
      pathlib.Path(__file__).absolute(),
      _resolve_path(pathlib.Path(__file__)),
  ):
    for parent in tool_path.parents:
      if is_project_root(parent):
        return parent

  candidate = _resolve_path(start_path or pathlib.Path.cwd())
  for p in (candidate, *candidate.parents):
    if is_project_root(p):
      return p

  return candidate


def _check_candidate(
    task: tuple[CandidateFile, Mapping[str, set[int]] | None],
) -> list[Violation]:
  """Worker function to evaluate all matching rules on a single CandidateFile."""
  candidate, modified_lines = task
  rel_path = candidate.rel_path
  operation = _file_operation(modified_lines, rel_path)
  violations: list[Violation] = []
  content_rules: list[Rule] = []
  for rule in candidate.matching_rules:
    if not _rule_applies(rule, rel_path, modified_lines, operation):
      continue
    if rule.kind == custom_lint_metadata.CHECK_FILE_OPERATION:
      violations.extend(_violations(rule, rel_path, None, [(None, None)]))
    else:
      content_rules.append(rule)
  if not content_rules:
    return violations

  content = _read_text_file(candidate.file_path)
  if content is None:
    return violations
  text = TextFile(content)
  for rule in content_rules:
    # Cheap literal pre-filter before building lines or running regexes.
    if rule.required_re is None and not _literals_present(rule, content):
      continue
    changed = (
        modified_lines.get(rel_path)
        if rule.check_delta_only and modified_lines is not None
        else None
    )
    violations.extend(
        _violations(rule, rel_path, text, _evaluate(rule, text, changed))
    )
  return violations


_WORKER_CANDIDATES: Sequence[CandidateFile] = ()
_WORKER_MODIFIED_LINES: Mapping[str, set[int]] | None = None


def _check_candidate_slice(bounds: tuple[int, int]) -> list[Violation]:
  """Evaluates a contiguous slice of _WORKER_CANDIDATES inherited via fork CoW."""
  start_idx, end_idx = bounds
  out: list[Violation] = []
  candidates = _WORKER_CANDIDATES
  modified_lines = _WORKER_MODIFIED_LINES
  for idx in range(start_idx, end_idx):
    out.extend(_check_candidate((candidates[idx], modified_lines)))
  return out


_CompiledPathGroups = Sequence[
    tuple[
        re.Pattern[str],
        Sequence[tuple[re.Pattern[str] | None, Sequence[tuple[int, Rule]]]],
    ]
]


@dataclasses.dataclass(frozen=True)
class CheckResult:
  """Collected violations and execution metrics from a lint run.

  Attributes:
    violations: List of rule violations found across checked files.
    files_checked: Number of candidate files evaluated.
    rules_count: Number of active rules evaluated.
    effective_jobs: Number of worker processes/threads used.
    cpu_count: Number of CPUs detected on the host.
    t_walk: Wall time in seconds spent selecting candidate files.
    t_eval: Wall time in seconds spent evaluating rules and filtering tags.
    t_total: Total wall time in seconds (`t_walk` + `t_eval`).
  """

  violations: list[Violation]
  files_checked: int
  rules_count: int
  effective_jobs: int
  cpu_count: int
  t_walk: float
  t_eval: float
  t_total: float

  @property
  def has_errors(self) -> bool:
    """Returns True if any violation has `ERROR` severity."""
    return any(v.severity == 'ERROR' for v in self.violations)

  def timing_summary(self) -> str:
    """Formats the one-line `INFO:` timing and worker summary."""
    return (
        f'INFO: Checked {self.files_checked} files across {self.rules_count}'
        f' rules using {self.effective_jobs}/{self.cpu_count} CPU worker(s) in'
        f' {self.t_total:.3f}s (walk={self.t_walk:.3f}s,'
        f' eval={self.t_eval:.3f}s)'
    )


def _candidates_for_paths(
    files: Sequence[str],
    root_dir: pathlib.Path,
    rules: Sequence[Rule],
    compiled_groups: _CompiledPathGroups,
    mapper: PathMapper,
    walk_workers: int,
    ignore_dirs: frozenset[str] | None = None,
) -> list[CandidateFile]:
  """Resolves explicit file and directory paths into matching CandidateFiles.

  Args:
    files: File or directory paths (absolute, CWD-relative, or root-relative).
    root_dir: Resolved project root directory.
    rules: Active rules used when walking directory arguments.
    compiled_groups: Pre-grouped path matchers from `_build_path_matcher`.
    mapper: `PathMapper` translating project-relative paths to canonical paths.
    walk_workers: Maximum threads for directory walks.
    ignore_dirs: Optional `frozenset` of directory names to exclude when walking
      directory arguments.

  Returns:
    List of `CandidateFile` entries matching at least one active rule.
  """
  candidate_files: list[CandidateFile] = []
  for f in files:
    p = pathlib.Path(f)
    if not p.is_absolute():
      cwd_path = pathlib.Path.cwd() / p
      root_path = root_dir / p
      p = (
          root_path
          if not cwd_path.exists() and root_path.exists()
          else cwd_path
      )
    p = p.resolve()
    rel_path = normalize_path(p, root_dir)
    if p.is_dir():
      candidate_files.extend(
          find_candidate_files(
              p,
              rules,
              max_workers=walk_workers,
              mapper=mapper,
              rel_prefix='' if rel_path == '.' else rel_path,
              ignore_dirs=ignore_dirs,
          )
      )
    elif p.is_file():
      if is_path_excluded(rel_path):
        continue
      matching = _match_rules_grouped(
          mapper.to_canonical(rel_path), compiled_groups
      )
      if matching:
        candidate_files.append(
            CandidateFile(
                file_path=p,
                rel_path=rel_path,
                matching_rules=matching,
            )
        )
  return candidate_files


def _candidates_for_diff(
    modified_lines: Mapping[str, set[int]],
    root_dir: pathlib.Path,
    compiled_groups: _CompiledPathGroups,
    mapper: PathMapper,
) -> list[CandidateFile]:
  """Selects existing modified files from a VCS diff that match at least one rule.

  Args:
    modified_lines: Mapping of project-relative paths to modified line numbers.
    root_dir: Resolved project root directory.
    compiled_groups: Pre-grouped path matchers from `_build_path_matcher`.
    mapper: `PathMapper` translating project-relative paths to canonical paths.

  Returns:
    Sorted list of `CandidateFile` entries for modified files on disk.
  """
  candidate_files: list[CandidateFile] = []
  for rel_path in sorted(modified_lines):
    if is_path_excluded(rel_path):
      continue
    full_path = root_dir / rel_path
    if not full_path.is_file():
      continue
    matching = _match_rules_grouped(
        mapper.to_canonical(rel_path), compiled_groups
    )
    if matching:
      candidate_files.append(
          CandidateFile(
              file_path=full_path,
              rel_path=rel_path,
              matching_rules=matching,
          )
      )
  return candidate_files


def collect_violations(
    root_dir: pathlib.Path,
    *,
    files: Sequence[str] | None = None,
    base_sha: str | None = None,
    check_all: bool = False,
    jobs: int | None = None,
    rules_file: str | None = None,
    only_rules: Sequence[str] | None = None,
    mapper: PathMapper | None = None,
    vcs: str = VCS_AUTO,
    exclude_dirs: Iterable[str] | None = None,
    no_default_excludes: bool = False,
) -> CheckResult:
  """Selects candidate files, evaluates convention rules, and filters by tags.

  Candidate selection operates in one of three modes:
  - Explicit files (`files` is non-empty, `check_all=False`): checks all lines
    of the specified files and directories without querying the VCS.
  - VCS diff (`files` is None/empty, `check_all=False`): queries the VCS diff
    (against `base_sha` for Git) to check only modified files and modified lines
    for `check_delta_only` rules, and lazily checks commit messages for
    `disable_tags` when findings occur.
  - Full scan (`check_all=True`): walks `root_dir` and checks all lines of every
    matching non-excluded file without querying the VCS.

  Args:
    root_dir: Project root directory (or a path inside it) to resolve and walk.
    files: Optional sequence of explicit file or directory paths to check.
    base_sha: Git base ref/SHA to diff and read commit messages against in diff
      mode.
    check_all: If True, scan all files and all lines under `root_dir` instead of
      computing a VCS diff.
    jobs: Number of parallel worker processes/threads (`None` or `<= 0`
      auto-detects available CPUs; `1` runs serially in-process).
    rules_file: Optional path to the `.METADATA` / `.textproto` rules file;
      defaults to the repository's standard rules file.
    only_rules: Optional sequence of rule IDs to run; if provided, only those
      rules are evaluated.
    mapper: Optional `PathMapper` translating between canonical depot paths and
      project-relative paths; defaults to `default_path_mapper()`.
    vcs: VCS selection (`'auto'`, `'any'`, `'git'`, `'hg'`, or `'jj'`).
    exclude_dirs: Optional iterable of exact directory names to exclude when
      walking the directory tree.
    no_default_excludes: If True, do not apply `DEFAULT_IGNORE_DIRS` when
      walking the directory tree.

  Returns:
    A `CheckResult` containing the filtered violations and timing/worker stats.

  Raises:
    LoadError: If the rules file cannot be parsed or validated.
    ValueError: If `only_rules` contains unknown rule IDs or `vcs` is invalid.
    VcsError: If diff computation or commit-message retrieval fails in VCS diff
      mode.
  """
  root_dir = _resolve_path(root_dir)
  mapper = mapper or default_path_mapper(root_dir)
  rule_set = load_rules(rules_file, root_dir=root_dir, mapper=mapper)
  rules = rule_set.rules
  if only_rules:
    unknown = sorted(set(only_rules) - {r.id for r in rules})
    if unknown:
      raise ValueError(f'unknown rule id(s): {", ".join(unknown)}')
    rules = [r for r in rules if r.id in only_rules]
  active_rules = list(rules)

  if not check_all and not files:
    try:
      modified_lines = get_modified_lines_from_vcs(
          base_sha, root_dir=root_dir, vcs=vcs
      )
    except VcsError as e:
      target = f' against {base_sha}' if base_sha else ''
      raise VcsError(
          f'could not compute the diff{target}: {e}; pass --all to scan'
          ' everything'
      ) from e
  else:
    modified_lines = None

  base_ignores: frozenset[str] = (
      frozenset() if no_default_excludes else DEFAULT_IGNORE_DIRS
  )
  ignore_dirs: frozenset[str] = (
      base_ignores | frozenset(exclude_dirs) if exclude_dirs else base_ignores
  )

  detected_cpus = os.cpu_count() or 1
  walk_workers = 1 if jobs == 1 else min(16, detected_cpus * 2)

  t0 = time.perf_counter()
  if files:
    compiled_groups = _build_path_matcher(active_rules)
    candidate_files = _candidates_for_paths(
        files,
        root_dir,
        active_rules,
        compiled_groups,
        mapper,
        walk_workers,
        ignore_dirs=ignore_dirs,
    )
  elif modified_lines is not None:
    compiled_groups = _build_path_matcher(active_rules)
    candidate_files = _candidates_for_diff(
        modified_lines, root_dir, compiled_groups, mapper
    )
  else:
    candidate_files = list(
        find_candidate_files(
            root_dir,
            active_rules,
            max_workers=walk_workers,
            mapper=mapper,
            ignore_dirs=ignore_dirs,
        )
    )
  t_walk = time.perf_counter() - t0

  effective_jobs = (
      jobs
      if jobs is not None
      else (detected_cpus if len(candidate_files) > 32 else 1)
  )

  t_eval_start = time.perf_counter()
  all_violations: list[Violation] = []
  if effective_jobs > 1 and len(candidate_files) > 1:
    for rule in active_rules:
      rule.prewarm()
    global _WORKER_CANDIDATES, _WORKER_MODIFIED_LINES
    _WORKER_CANDIDATES = candidate_files
    _WORKER_MODIFIED_LINES = modified_lines
    is_free_threaded = not getattr(sys, '_is_gil_enabled', lambda: True)()
    n_chunks = min(
        len(candidate_files),
        effective_jobs * (8 if is_free_threaded else 4),
    )
    step = (len(candidate_files) + n_chunks - 1) // n_chunks
    slices = [
        (i, min(len(candidate_files), i + step))
        for i in range(0, len(candidate_files), step)
    ]
    start_methods = multiprocessing.get_all_start_methods()
    if is_free_threaded:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=effective_jobs
      ) as executor:
        for slice_violations in executor.map(_check_candidate_slice, slices):
          all_violations.extend(slice_violations)
    elif 'fork' in start_methods and threading.active_count() == 1:
      # fork is the fastest option: workers inherit _WORKER_CANDIDATES and the
      # compiled rules copy-on-write, with nothing to pickle. On a 24-core
      # `--all` run it took 1.3 s vs 1.5 s (forkserver) and 1.7 s (spawn).
      # It is only safe without other live threads (the walk's thread pool
      # has already exited here); embedders with threads take the path below.
      mp_ctx = multiprocessing.get_context('fork')
      with concurrent.futures.ProcessPoolExecutor(
          max_workers=effective_jobs, mp_context=mp_ctx
      ) as executor:
        for slice_violations in executor.map(_check_candidate_slice, slices):
          all_violations.extend(slice_violations)
    else:
      # Explicit non-fork context: the default is fork on Linux (< 3.14).
      mp_ctx = multiprocessing.get_context(
          'forkserver' if 'forkserver' in start_methods else 'spawn'
      )
      tasks = [(c, modified_lines) for c in candidate_files]
      with concurrent.futures.ProcessPoolExecutor(
          max_workers=effective_jobs, mp_context=mp_ctx
      ) as executor:
        for file_violations in executor.map(
            _check_candidate, tasks, chunksize=step
        ):
          all_violations.extend(file_violations)
  else:
    for candidate in candidate_files:
      all_violations.extend(_check_candidate((candidate, modified_lines)))

  if all_violations and not check_all and not files:
    rule_by_id = {r.id: r for r in rules}
    if any(rule_by_id[v.rule_id].disable_tags for v in all_violations):
      commit_message = get_commit_message(base_sha, root_dir=root_dir, vcs=vcs)
      if commit_message:
        disabled_ids = {
            r.id for r in rules if is_rule_disabled(r, commit_message)
        }
        if disabled_ids:
          all_violations = [
              v for v in all_violations if v.rule_id not in disabled_ids
          ]

  t_eval = time.perf_counter() - t_eval_start
  t_total = time.perf_counter() - t0
  return CheckResult(
      violations=all_violations,
      files_checked=len(candidate_files),
      rules_count=len(active_rules),
      effective_jobs=effective_jobs,
      cpu_count=detected_cpus,
      t_walk=t_walk,
      t_eval=t_eval,
      t_total=t_total,
  )


def format_summary_report(
    violations: Sequence[Violation],
    files_checked: int,
) -> str:
  """Formats violations into a grouped summary report matching custom_linter_report.txt."""
  by_rule: dict[str, list[Violation]] = {}
  for v in violations:
    by_rule.setdefault(v.rule_id, []).append(v)

  sorted_groups = sorted(
      by_rule.values(), key=lambda g: (-len(g), g[0].rule_id)
  )
  errors = sum(1 for v in violations if v.severity == 'ERROR')
  warnings = len(violations) - errors
  out_lines = [
      '=' * 60,
      f' SUMMARY: {errors} errors, {warnings} warnings',
      (
          f' Files checked: {files_checked}   Findings: {len(violations)}  '
          f' Rules with findings: {len(sorted_groups)}'
      ),
      '=' * 60,
      '',
      '   FINDINGS  RULE',
  ]
  for g in sorted_groups:
    out_lines.append(f'  {len(g):9d}  {g[0].rule_id}')
  out_lines.append('')

  for g in sorted_groups:
    by_path: dict[str, list[Violation]] = {}
    for v in g:
      by_path.setdefault(v.file_path, []).append(v)
    out_lines.extend([
        '#' * 60,
        (
            f'# {g[0].rule_id}  [{g[0].severity}]  {len(g)} findings in'
            f' {len(by_path)} files'
        ),
        '#' * 60,
        'Message:',
    ])
    for msg_line in g[0].message.split('\n'):
      out_lines.append(f'  {msg_line}')
    out_lines.append('')
    for path, path_violations in sorted(by_path.items()):
      out_lines.append(f'  {path}  ({len(path_violations)})')
      for pv in path_violations:
        out_lines.append(f'      {pv.location}')
    out_lines.append('')

  return '\n'.join(out_lines) + '\n'


def write_text_report(
    result: CheckResult, stream: TextIO | None = None
) -> None:
  """Writes the human-readable violation report and timing summary to `stream`.

  Args:
    result: Collected `CheckResult` from `collect_violations`.
    stream: Output text stream; defaults to `sys.stdout`.
  """
  out = sys.stdout if stream is None else stream
  timing_summary = result.timing_summary()
  if not result.violations:
    print(timing_summary, file=out)
    print(
        'INFO: Custom TorchTPU convention checks passed successfully.',
        file=out,
    )
    return

  print('=' * 65, file=out)
  print(
      f'ERROR: {len(result.violations)} custom convention check violation(s)'
      ' found:',
      file=out,
  )
  print('=' * 65, file=out)
  for v in result.violations:
    severity = '' if v.severity == 'ERROR' else f' ({v.severity})'
    print(f'[{v.rule_id}]{severity} {v.location}', file=out)
    if v.line_content:
      print(f'  >   {v.line_content}', file=out)
    print(f'  {v.message}', file=out)
    print('-' * 65, file=out)
  print(f'Total violations: {len(result.violations)}', file=out)
  print(timing_summary, file=out)
  print('=' * 65, file=out)


def write_json(result: CheckResult, stream: TextIO | None = None) -> None:
  """Writes `result.violations` serialized as a JSON array to `stream`.

  Args:
    result: Collected `CheckResult` from `collect_violations`.
    stream: Output text stream; defaults to `sys.stdout`.
  """
  out = sys.stdout if stream is None else stream
  print(
      json.dumps([dataclasses.asdict(v) for v in result.violations], indent=2),
      file=out,
  )


def run_checks(
    root_dir: pathlib.Path,
    files: Sequence[str] | None = None,
    base_sha: str | None = None,
    check_all: bool = False,
    jobs: int | None = None,
    report_file: str | None = None,
    output_format: str = 'text',
    rules_file: str | None = None,
    only_rules: Sequence[str] | None = None,
    mapper: PathMapper | None = None,
    vcs: str = VCS_AUTO,
    exclude_dirs: Iterable[str] | None = None,
    no_default_excludes: bool = False,
) -> int:
  """Runs convention checks across candidate files and reports violations.

  Candidate selection operates in one of three modes:
  - Explicit files (`files` is non-empty, `check_all=False`): checks only the
    specified files and directories (all lines, without querying VCS diff).
  - VCS diff (`files` is None/empty, `check_all=False`): queries the VCS diff
    (against `base_sha` for Git) to check only modified files and modified lines
    for `check_delta_only` rules, and lazily checks commit messages for
    `disable_tags` when findings occur.
  - Full scan (`check_all=True`): walks `root_dir` and checks all lines of every
    matching non-excluded file (`disable_tags` and file-operation rules are not
    applied).

  Args:
    root_dir: Project root directory (or a path inside it) to resolve and walk.
    files: Optional sequence of explicit file or directory paths to check.
    base_sha: Git base ref/SHA to diff and read commit messages against in diff
      mode.
    check_all: If True, scan all files and all lines under `root_dir` instead of
      computing a VCS diff.
    jobs: Number of parallel worker processes/threads (`None` or `<= 0`
      auto-detects available CPUs; `1` runs serially in-process).
    report_file: Optional file path to write a grouped human-readable summary
      report.
    output_format: Output format written to stdout (`'text'` or `'json'`).
    rules_file: Optional path to the `.METADATA` / `.textproto` rules file;
      defaults to the repository's standard rules file.
    only_rules: Optional sequence of rule IDs to run; if provided, only those
      rules are evaluated.
    mapper: Optional `PathMapper` translating between canonical depot paths and
      project-relative paths; defaults to `default_path_mapper()`.
    vcs: VCS selection (`'auto'`, `'any'`, `'git'`, `'hg'`, or `'jj'`).
    exclude_dirs: Optional iterable of exact directory names to exclude when
      walking the directory tree.
    no_default_excludes: If True, do not apply `DEFAULT_IGNORE_DIRS` when
      walking the directory tree.

  Returns:
    0 if no `ERROR`-severity findings were reported (warnings are allowed),
    1 if one or more `ERROR`-severity findings were reported, or
    2 if the rules file failed to load or an unknown rule ID was requested
    (note: `VcsError` is raised on VCS failure and converted to exit code 2 by
    `main()`).

  Raises:
    VcsError: If diff computation or commit-message retrieval fails in VCS diff
      mode.
  """
  try:
    result = collect_violations(
        root_dir,
        files=files,
        base_sha=base_sha,
        check_all=check_all,
        jobs=jobs,
        rules_file=rules_file,
        only_rules=only_rules,
        mapper=mapper,
        vcs=vcs,
        exclude_dirs=exclude_dirs,
        no_default_excludes=no_default_excludes,
    )
  except (LoadError, ValueError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    return 2

  if report_file:
    report_path = pathlib.Path(report_file)
    report_path.write_text(
        format_summary_report(result.violations, result.files_checked),
        encoding='utf-8',
    )
    if output_format != 'json':
      print(f'Report written to {report_path} (view with: less {report_path})')

  if output_format == 'json':
    write_json(result, sys.stdout)
    print(result.timing_summary(), file=sys.stderr)
  else:
    write_text_report(result, sys.stdout)

  return 1 if result.has_errors else 0


def _default_project_root(cwd: pathlib.Path) -> pathlib.Path | None:
  """Returns the nearest dir with a rules file, from cwd up to the VCS root.

  In a monorepo checkout the VCS root (e.g. a workspace directory above
  `MONOREPO_DIR_NAME`) is not the project root, so the search walks up from
  `cwd` and stops at the VCS root (or the filesystem root if there is no VCS).
  For each directory it also tries the tool's own project path inside the
  monorepo, so `bazel run`-style invocations (whose working directory is the
  monorepo root) find the project.

  Args:
    cwd: Directory the user ran the tool from.

  Returns:
    The project root, or None if no rules file was found.
  """

  def has_rules(p: pathlib.Path) -> bool:
    return (p / MONOREPO_RULES_FILE).is_file() or (p / OSS_RULES_FILE).is_file()

  start = _resolve_path(cwd)
  vcs_root = find_vcs_root(start)
  rel = _tool_project_rel()
  for p in (start, *start.parents):
    if has_rules(p):
      return p
    if rel and p.name == MONOREPO_DIR_NAME and has_rules(p / rel):
      return p / rel
    if vcs_root is not None and p == vcs_root:
      break
  return None


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description='Custom TorchTPU OSS presubmit convention checker.'
  )
  parser.add_argument(
      'files',
      nargs='*',
      help='Optional specific files or directories to check.',
  )
  parser.add_argument(
      '--root',
      default=None,
      help='Optional root directory of the torch_tpu repository.',
  )
  default_base = os.environ.get('BASE_SHA') or 'origin/main'
  parser.add_argument(
      '--base-sha',
      default=default_base,
      help=(
          'Base git SHA or tracking branch for incremental diffs (default:'
          ' $BASE_SHA if set, falling back to "origin/main" in git; resolved'
          ' via git merge-base against HEAD, falling back to base itself in'
          ' shallow CI checkouts; compared against the working tree to include'
          ' uncommitted edits).'
      ),
  )
  parser.add_argument(
      '--all',
      action='store_true',
      dest='check_all',
      help='Check all lines regardless of git/hg/jj diffs.',
  )
  parser.add_argument(
      '-j',
      '--jobs',
      type=int,
      default=None,
      help=(
          'Number of parallel worker processes (default: auto-scales to CPU'
          ' count when checking >32 files).'
      ),
  )
  parser.add_argument(
      '-o',
      '--report-file',
      default=None,
      help=(
          'Optional path to write a grouped summary report (e.g.'
          ' /tmp/custom_linter_report.txt).'
      ),
  )
  parser.add_argument(
      '--format',
      choices=['text', 'json'],
      default='text',
      dest='output_format',
      help='Output format (text or json).',
  )
  parser.add_argument(
      '--rules',
      default=None,
      dest='rules_file',
      help=(
          f'METADATA-format rules file (default: {MONOREPO_RULES_FILE} or'
          f' {OSS_RULES_FILE}).'
      ),
  )
  parser.add_argument(
      '--rule',
      action='append',
      default=None,
      dest='only_rules',
      metavar='ID',
      help='Only run the rule with this id (repeatable).',
  )
  parser.add_argument(
      '--vcs',
      choices=(VCS_AUTO, VCS_ANY) + VCS_TYPES,
      default=VCS_AUTO,
      help=(
          'VCS used for the diff and the commit message: detected from the'
          f' filesystem ({VCS_AUTO}), tried in order ({VCS_ANY}), or a'
          ' specific one.'
      ),
  )
  parser.add_argument(
      '--exclude-dir',
      action='append',
      default=None,
      dest='exclude_dirs',
      metavar='NAME',
      help='Directory name to exclude when walking the tree (repeatable).',
  )
  parser.add_argument(
      '--no-default-excludes',
      action='store_true',
      default=False,
      dest='no_default_excludes',
      help='Do not apply the default directory ignore list when walking.',
  )
  args = parser.parse_args(argv)

  working_dir = os.environ.get('BUILD_WORKING_DIRECTORY')
  cwd = pathlib.Path(working_dir) if working_dir else pathlib.Path.cwd()
  if args.root:
    root_path = pathlib.Path(args.root)
    root_dir = (cwd / root_path) if not root_path.is_absolute() else root_path
  else:
    root_dir = _default_project_root(cwd)
    if root_dir is None:
      if args.rules_file is None:
        print(
            f'ERROR: could not find {MONOREPO_RULES_FILE} or {OSS_RULES_FILE}'
            f' in {_resolve_path(cwd)} or its parents (up to the VCS root);'
            ' pass --root <project_root>',
            file=sys.stderr,
        )
        return 2
      vcs_root = find_vcs_root(cwd)
      root_dir = vcs_root if vcs_root is not None else _resolve_path(cwd)
  try:
    return run_checks(
        root_dir=root_dir,
        files=args.files,
        base_sha=args.base_sha,
        check_all=args.check_all,
        jobs=args.jobs,
        report_file=args.report_file,
        output_format=args.output_format,
        rules_file=args.rules_file,
        only_rules=args.only_rules,
        vcs=args.vcs,
        exclude_dirs=args.exclude_dirs,
        no_default_excludes=args.no_default_excludes,
    )
  except VcsError as e:
    print(f'ERROR: {e}', file=sys.stderr)
    return 2


if __name__ == '__main__':
  sys.exit(main())
