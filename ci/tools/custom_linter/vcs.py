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

"""Version-control abstraction for the custom convention linter.

Provides a small polymorphic `Vcs` interface with concrete implementations for
Git (`GitVcs`), Mercurial (`HgVcs`), and Jujutsu (`JjVcs`), plus filesystem
detection (`detect`) and name lookup (`for_name`).
"""

import abc
from collections.abc import Iterator, Mapping, Sequence, Set
import dataclasses
import enum
import functools
import pathlib
import re
import subprocess
from typing import Final

from ci.tools.custom_linter import metadata as custom_lint_metadata


class VcsMode(enum.StrEnum):
  """Values for `--vcs` selection.

  - GIT, HG, JJ: use only that VCS backend.
  - AUTO: detect from the filesystem, falling back to ANY.
  - ANY: try every VCS backend in VCS_TYPES order.
  """

  AUTO = 'auto'
  ANY = 'any'
  GIT = 'git'
  HG = 'hg'
  JJ = 'jj'


VCS_TYPES: Final[Sequence[VcsMode]] = (VcsMode.GIT, VcsMode.HG, VcsMode.JJ)

_JJ_REVSET: Final[str] = 'heads(::@ & immutable())..@'


class VcsError(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when a VCS diff or log command fails."""


@dataclasses.dataclass(frozen=True)
class ModifiedLines(Mapping[str, Set[int]]):
  """Maps project-relative paths to their modified line numbers and operations.

  Attributes:
    lines: Maps project-relative paths to their 1-based added or modified line
      numbers.
    operations: Maps project-relative paths to the `FileOperation` (ADD, EDIT,
      DELETE, BRANCH) that the change performs on them.
  """

  lines: Mapping[str, Set[int]] = dataclasses.field(default_factory=dict)
  operations: Mapping[str, custom_lint_metadata.FileOperation] = (
      dataclasses.field(default_factory=dict)
  )

  def __getitem__(self, key: str) -> Set[int]:
    return self.lines[key]

  def __iter__(self) -> Iterator[str]:
    return iter(self.lines)

  def __len__(self) -> int:
    return len(self.lines)


@dataclasses.dataclass(frozen=True)
class Commit:
  """Represents a VCS commit or CL description and its parsed metadata tags."""

  message: str

  @functools.cached_property
  def tags(self) -> Set[str]:
    """Returns the `KEY=value` tag names parsed from `self.message`."""
    return custom_lint_metadata.parse_tags(self.message)


def resolve_path(path: pathlib.Path) -> pathlib.Path:
  """Returns `path.resolve()`."""
  return path.resolve()


def _find_marker_root(
    candidate: pathlib.Path, marker: str
) -> pathlib.Path | None:
  for p in (candidate, *candidate.parents):
    if (p / marker).exists():
      return p
  return None


def find_vcs_root(
    start_path: pathlib.Path | None = None,
) -> pathlib.Path | None:
  """Finds the root directory of the VCS repository (git, hg, or jj)."""
  candidate = resolve_path(start_path or pathlib.Path.cwd())
  found = _detect_at(candidate)
  return found[1] if found is not None else None


def _clean_base(base: str) -> str:
  """Strips range dots (`..` or `...`) and whitespace from a git base ref."""
  return base.strip().split('..')[0].rstrip('.')


def _run_vcs_cmd(cmd: Sequence[str], cwd: str | None = None) -> str:
  """Runs `cmd` in `cwd` and returns stdout, raising `VcsError` on failure."""
  cmd_str = ' '.join(cmd)
  try:
    result = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
        cwd=cwd,
    )
  except OSError as exc:
    raise VcsError(f'{cmd_str}: {exc}') from exc
  if result.returncode == 0:
    return result.stdout
  stderr = (result.stderr or '').strip() or f'exit code {result.returncode}'
  raise VcsError(f'{cmd_str}: {stderr}')


class Vcs(abc.ABC):
  """Abstract interface for a version-control backend."""

  name: str = ''
  marker: str = ''

  def __init__(
      self,
      start: pathlib.Path | None = None,
      *,
      root_path: pathlib.Path | None = None,
  ) -> None:
    self._start = start
    self._root_path = root_path

  @property
  def _cwd(self) -> str | None:
    """Directory to run VCS commands in (the process cwd if unset).

    Commands must run inside the checkout: under `bazel run` the process cwd
    is the runfiles tree, which is not a repository.
    """
    return str(self._start) if self._start is not None else None

  def root(self) -> pathlib.Path:
    """Returns the root directory of the VCS repository."""
    if self._root_path is not None:
      return self._root_path
    candidate = resolve_path(self._start or pathlib.Path.cwd())
    if self.marker:
      found = _find_marker_root(candidate, self.marker)
    else:
      detected = _detect_at(candidate)
      found = detected[1] if detected is not None else None
    return found if found is not None else candidate

  def prefix(self, project_root: pathlib.Path) -> str:
    """Returns the POSIX prefix (with trailing '/') of `project_root` under `root()`."""
    vcs_root = self.root()
    try:
      rel = resolve_path(project_root).relative_to(resolve_path(vcs_root))
    except ValueError:
      return ''
    posix = rel.as_posix()
    return '' if posix in ('', '.') else f'{posix}/'

  @abc.abstractmethod
  def diff(self, base: str | None = None) -> str:
    """Returns unified diff text with `-U0` and git-style headers."""

  @abc.abstractmethod
  def get_commit_message(
      self, base: str | None = None, head: str | None = None
  ) -> str:
    """Returns the commit/CL description text for the current change."""

  def get_modified_lines(
      self,
      base: str | None = None,
      root_dir: pathlib.Path | None = None,
  ) -> ModifiedLines:
    """Returns the modified lines and file operations from the VCS diff."""
    resolved_root = resolve_path(root_dir or self._start or pathlib.Path.cwd())
    diff_text = self.diff(base)
    prefix = self.prefix(resolved_root)
    return parse_unified_diff(diff_text, root_dir=resolved_root, prefix=prefix)

  def get_commit(
      self, base: str | None = None, head: str | None = None
  ) -> Commit:
    """Returns a `Commit` for the current change."""
    return Commit(self.get_commit_message(base, head=head))


class GitVcs(Vcs):
  """Git VCS backend."""

  name: str = VcsMode.GIT
  marker: str = '.git'

  def diff(self, base: str | None = None) -> str:
    git_cmd = [
        'git',
        'diff',
        '-U0',
        '--no-color',
        '--no-ext-diff',
        '--src-prefix=a/',
        '--dst-prefix=b/',
    ]
    if base:
      clean_base = _clean_base(base)
      if clean_base:
        resolved_base = _run_vcs_cmd(
            ['git', 'merge-base', clean_base, 'HEAD'], self._cwd
        ).strip()
        git_cmd.append(resolved_base)
    return _run_vcs_cmd(git_cmd, self._cwd)

  def get_commit_message(
      self, base: str | None = None, head: str | None = None
  ) -> str:
    clean_base = _clean_base(base) if base else ''
    clean_head = _clean_base(head) if head else ''
    if clean_base:
      head_ref = clean_head or 'HEAD'
      _run_vcs_cmd(['git', 'merge-base', clean_base, head_ref], self._cwd)
      cmd = ['git', 'log', f'{clean_base}..{head_ref}', '--format=%B']
    elif clean_head:
      cmd = ['git', 'log', '-1', clean_head, '--format=%B']
    else:
      cmd = ['git', 'log', '-1', '--format=%B']
    out = _run_vcs_cmd(cmd, self._cwd)
    return out if out.strip() else ''


class HgVcs(Vcs):
  """Mercurial VCS backend."""

  name: str = VcsMode.HG
  marker: str = '.hg'

  def diff(self, base: str | None = None) -> str:
    del base
    # `present(p4base)` resolves `p4base` in Fig/CitC workspaces and evaluates
    # to the empty revset in standard Mercurial repositories, where
    # `first(... + .)` falls back to `.` (matching `hg diff` without `-r`).
    return _run_vcs_cmd(
        ['hg', 'diff', '--git', '-U0', '-r', 'first(present(p4base) + .)'],
        self._cwd,
    )

  def get_commit_message(
      self, base: str | None = None, head: str | None = None
  ) -> str:
    del base, head
    out = _run_vcs_cmd(
        ['hg', 'log', '-r', '.', '--template', '{desc}'], self._cwd
    )
    return out if out.strip() else ''


class JjVcs(Vcs):
  """Jujutsu VCS backend."""

  name: str = VcsMode.JJ
  marker: str = '.jj'

  def diff(self, base: str | None = None) -> str:
    del base
    return _run_vcs_cmd(
        [
            'jj',
            'diff',
            '--git',
            '--context',
            '0',
            '--color',
            'never',
            '--no-pager',
            '-r',
            _JJ_REVSET,
        ],
        self._cwd,
    )

  def get_commit_message(
      self, base: str | None = None, head: str | None = None
  ) -> str:
    del base, head
    out = _run_vcs_cmd(
        [
            'jj',
            'log',
            '--ignore-working-copy',
            '--no-graph',
            '--color',
            'never',
            '--no-pager',
            '-r',
            _JJ_REVSET,
            '-T',
            'description ++ "\n"',
        ],
        self._cwd,
    )
    return out if out.strip() else ''


class _AnyVcs(Vcs):
  """Composite VCS backend that tries Git, Mercurial, and Jujutsu in order."""

  name: str = VcsMode.ANY
  marker: str = ''

  def __init__(
      self,
      start: pathlib.Path | None = None,
      *,
      root_path: pathlib.Path | None = None,
  ) -> None:
    super().__init__(start, root_path=root_path)
    self._active: Vcs | None = None

  def _backends(self) -> Sequence[Vcs]:
    return (GitVcs(self._start), HgVcs(self._start), JjVcs(self._start))

  def root(self) -> pathlib.Path:
    if self._active is not None:
      return self._active.root()
    return super().root()

  def diff(self, base: str | None = None) -> str:
    if self._active is not None:
      return self._active.diff(base)
    errors: list[str] = []
    for backend in self._backends():
      try:
        out = backend.diff(base)
        self._active = backend
        return out
      except VcsError as exc:
        errors.append(str(exc))
    raise VcsError('could not compute the diff (' + '; '.join(errors) + ')')

  def get_commit_message(
      self, base: str | None = None, head: str | None = None
  ) -> str:
    if self._active is not None:
      return self._active.get_commit_message(base, head=head)
    errors: list[str] = []
    for backend in self._backends():
      try:
        out = backend.get_commit_message(base, head=head)
        self._active = backend
        return out
      except VcsError as exc:
        errors.append(str(exc))
    raise VcsError(
        'could not retrieve commit message (' + '; '.join(errors) + ')'
    )


def _detect_at(candidate: pathlib.Path) -> tuple[VcsMode, pathlib.Path] | None:
  for p in (candidate, *candidate.parents):
    # Check `.jj` before `.git`/`.hg` so colocated Jujutsu checkouts use `jj`.
    if (p / '.jj').exists():
      return (VcsMode.JJ, p)
    if (p / '.git').exists():
      return (VcsMode.GIT, p)
    if (p / '.hg').exists():
      return (VcsMode.HG, p)
  return None


_VCS_BY_NAME: Final[Mapping[VcsMode, type[Vcs]]] = {
    VcsMode.GIT: GitVcs,
    VcsMode.HG: HgVcs,
    VcsMode.JJ: JjVcs,
}


def detect(start: pathlib.Path | None = None) -> Vcs | None:
  """Detects the nearest `.git`, `.jj`, or `.hg` repository without subprocesses."""
  resolved = resolve_path(start or pathlib.Path.cwd())
  found = _detect_at(resolved)
  if found is None:
    return None
  kind, root_path = found
  return _VCS_BY_NAME[kind](resolved, root_path=root_path)


def for_name(name: VcsMode, start: pathlib.Path | None = None) -> Vcs:
  """Returns the `Vcs` backend for `name`."""
  if name == VcsMode.AUTO:
    detected = detect(start)
    return detected if detected is not None else _AnyVcs(start)
  if name == VcsMode.ANY:
    return _AnyVcs(start)
  cls = _VCS_BY_NAME.get(name)
  if cls is None:
    expected = ', '.join((VcsMode.AUTO, VcsMode.ANY, *VCS_TYPES))
    raise ValueError(f'Unknown vcs {name!r}; expected one of {expected}')
  return cls(start)


_DIFF_HEADER_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
    r'\t|\s{2,}|\s+\d{4}-\d{2}-\d{2}'
)
_DIFF_HUNK_RE: Final[re.Pattern[str]] = re.compile(
    r'^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'
)
_DEV_NULL: Final[Set[str]] = frozenset({'/dev/null', 'dev/null', ''})


def normalize_diff_path(
    path: str,
    root_dir: pathlib.Path | None = None,
    *,
    prefix: str = '',
) -> str | None:
  """Normalizes a VCS diff file path to be relative to the project root.

  Strips leading `a/` or `b/` diff prefixes and `prefix` (the project's path
  under the VCS root). When `prefix` is non-empty, paths outside `prefix`
  return `None` so callers omit them.

  Args:
    path: Raw file path from a unified diff header.
    root_dir: Optional project root directory used to normalize absolute paths.
    prefix: Optional POSIX directory prefix of the project under the VCS root.

  Returns:
    The project-relative POSIX path, or `None` if `path` is `/dev/null` or lies
    outside the project.
  """
  if path in _DEV_NULL:
    return None
  if path.startswith('b/') or path.startswith('a/'):
    path = path[2:]
  if prefix:
    norm_prefix = prefix if prefix.endswith('/') else f'{prefix}/'
    if not path.startswith(norm_prefix):
      return None
    normalized = path[len(norm_prefix) :]
    return normalized or None
  if root_dir is not None and path.startswith('/'):
    resolved_root = resolve_path(root_dir)
    resolved_path = resolve_path(pathlib.Path(path))
    try:
      rel = resolved_path.relative_to(resolved_root).as_posix()
      return None if rel == '.' else rel
    except ValueError:
      return None
  return path or None


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
  files, DELETE for deleted files, BRANCH for unmodified copies and renames
  (ADD if modified afterwards), and EDIT otherwise. Deleted files are omitted
  from `lines` (which tracks existing modified files) and files outside `prefix`
  are omitted altogether.

  Args:
    diff_text: Output of `git diff`, `hg diff` or `jj diff --git`.
    root_dir: Project root used to normalize absolute paths.
    prefix: POSIX directory prefix of the project under the VCS root.

  Returns:
    The modified lines and file operations per file.
  """
  lines: dict[str, set[int]] = {}
  operations: dict[str, custom_lint_metadata.FileOperation] = {}
  added: set[str] = set()
  copied: set[str] = set()
  current_file: str | None = None
  # Path of the latest `--- ` header; names the file a `+++ /dev/null` deletes.
  old_path = ''
  # Per-file header state, reset by `diff ...` lines.
  pending_added = pending_copied = pending_deleted = False
  # Remaining old/new lines of the current hunk; hunk bodies are skipped so
  # that content lines starting with `--- `/`+++ ` are not taken as headers.
  old_left = new_left = 0

  def mark_deleted(norm_path: str | None) -> None:
    if norm_path is None:
      return
    lines.pop(norm_path, None)
    added.discard(norm_path)
    copied.discard(norm_path)
    operations[norm_path] = custom_lint_metadata.FileOperation.DELETE

  def register(path: str) -> str | None:
    normalized = normalize_diff_path(path, root_dir=root_dir, prefix=prefix)
    if normalized is None:
      return None
    if pending_deleted:
      mark_deleted(normalized)
      return None
    operations.pop(normalized, None)
    lines.setdefault(normalized, set())
    if pending_added:
      added.add(normalized)
    if pending_copied:
      copied.add(normalized)
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
        mark_deleted(current_file)
        current_file = None
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
        mark_deleted(
            normalize_diff_path(old_path, root_dir=root_dir, prefix=prefix)
        )
        current_file = None
      else:
        current_file = register(path_part)
    elif line.startswith('@@ '):
      # Hunk header: @@ -old_start[,old_count] +new_start[,new_count] @@
      match = _DIFF_HUNK_RE.match(line)
      if match:
        old_left = int(match.group(1)) if match.group(1) is not None else 1
        start_line = int(match.group(2))
        count = int(match.group(3)) if match.group(3) is not None else 1
        new_left = count
        if current_file is not None and count > 0:
          lines[current_file].update(range(start_line, start_line + count))

  for path in lines:
    if path in copied:
      op = (
          custom_lint_metadata.FileOperation.ADD
          if lines[path]
          else custom_lint_metadata.FileOperation.BRANCH
      )
    elif path in added:
      op = custom_lint_metadata.FileOperation.ADD
    else:
      op = custom_lint_metadata.FileOperation.EDIT
    operations[path] = op
  return ModifiedLines(lines=lines, operations=operations)
