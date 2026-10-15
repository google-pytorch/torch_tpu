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

"""Version-control abstraction for the TorchTPU custom convention linter.

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


VCS_TYPES: Final[tuple[VcsMode, ...]] = (VcsMode.GIT, VcsMode.HG, VcsMode.JJ)
VCS_AUTO: Final[VcsMode] = VcsMode.AUTO
VCS_ANY: Final[VcsMode] = VcsMode.ANY

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
      BRANCH) that the change performs on them.
  """

  lines: Mapping[str, Set[int]] = dataclasses.field(default_factory=dict)
  operations: Mapping[str, custom_lint_metadata.FileOperation | str] = (
      dataclasses.field(default_factory=dict)
  )

  def __getitem__(self, key: str) -> Set[int]:
    return self.lines[key]

  def __iter__(self) -> Iterator[str]:
    return iter(self.lines)

  def __len__(self) -> int:
    return len(self.lines)


class Commit:
  """Represents a VCS commit or CL description and its parsed metadata tags."""

  def __init__(
      self,
      message: str | None = None,
      *,
      vcs: 'Vcs | None' = None,
      base: str | None = None,
      head: str | None = None,
  ) -> None:
    self._message = message
    self._vcs = vcs
    self._base = base
    self._head = head

  @functools.cached_property
  def message(self) -> str:
    """Returns the commit/CL description text, querying the VCS lazily if needed."""
    if self._message is not None:
      return self._message
    if self._vcs is not None:
      return self._vcs.get_commit_message(self._base, head=self._head)
    return ''

  @functools.cached_property
  def tags(self) -> frozenset[str]:
    """Returns the `KEY=value` tag names parsed from `self.message`."""
    return custom_lint_metadata.parse_tags(self.message)


def resolve_path(path: pathlib.Path) -> pathlib.Path:
  """Returns `path.resolve()`, cached by absolute path.

  Relative paths are made absolute first (no syscall beyond `getcwd`) so that
  the cache key does not depend on the current working directory.

  Args:
    path: The path to resolve.

  Returns:
    The resolved `pathlib.Path`.
  """
  return _resolve_absolute(path if path.is_absolute() else path.absolute())


@functools.lru_cache(maxsize=256)
def _resolve_absolute(path: pathlib.Path) -> pathlib.Path:
  return path.resolve()


@functools.lru_cache(maxsize=64)
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
        check=False,
        cwd=cwd,
    )
  except FileNotFoundError as exc:
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
  def log(self, base: str | None = None, head: str | None = None) -> str:
    """Returns commit message text for the current change."""

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

  def get_commit_message(
      self, base: str | None = None, head: str | None = None
  ) -> str:
    """Returns the commit/CL description text for the current change."""
    return self.log(base, head=head)

  def get_commit(
      self, base: str | None = None, head: str | None = None
  ) -> Commit:
    """Returns a `Commit` for the current change."""
    return Commit(vcs=self, base=base, head=head)


class GitVcs(Vcs):
  """Git VCS backend."""

  name: str = 'git'
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
        try:
          mb_result = subprocess.run(
              ['git', 'merge-base', clean_base, 'HEAD'],
              stdout=subprocess.PIPE,
              stderr=subprocess.PIPE,
              text=True,
              check=False,
              cwd=self._cwd,
          )
        except FileNotFoundError as exc:
          raise VcsError(f'git merge-base {clean_base} HEAD: {exc}') from exc
        if mb_result.returncode == 0 and mb_result.stdout.strip():
          resolved_base = mb_result.stdout.strip()
        else:
          resolved_base = clean_base
        git_cmd.append(resolved_base)
    return _run_vcs_cmd(git_cmd, self._cwd)

  def _shallow_boundary_exclusions(
      self, clean_base: str, clean_head: str
  ) -> list[str]:
    """Returns `^<root>` exclusions when `base` and `head` are disconnected."""
    try:
      mb_result = subprocess.run(
          ['git', 'merge-base', clean_base, clean_head],
          capture_output=True,
          text=True,
          check=False,
          cwd=self._cwd,
      )
      if mb_result.returncode == 0:
        return []
      head_rp = subprocess.run(
          ['git', 'rev-parse', '--verify', clean_head],
          capture_output=True,
          text=True,
          check=False,
          cwd=self._cwd,
      )
      roots_result = subprocess.run(
          ['git', 'rev-list', '--max-parents=0', clean_head],
          capture_output=True,
          text=True,
          check=False,
          cwd=self._cwd,
      )
    except FileNotFoundError:
      return []
    if head_rp.returncode != 0 or roots_result.returncode != 0:
      return []
    resolved_head = head_rp.stdout.strip()
    roots = [
        r.strip()
        for r in roots_result.stdout.splitlines()
        if r.strip() and r.strip() != resolved_head
    ]
    return [f'^{r}' for r in roots]

  def log(self, base: str | None = None, head: str | None = None) -> str:
    clean_base = _clean_base(base) if base else ''
    clean_head = _clean_base(head) if head else ''
    if clean_base:
      extra_exclusions: list[str] = []
      if clean_head:
        head_ref = clean_head
        extra_exclusions = self._shallow_boundary_exclusions(
            clean_base, clean_head
        )
      else:
        head_ref = 'HEAD'
        try:
          rp_result = subprocess.run(
              ['git', 'rev-parse', '--verify', 'HEAD^2'],
              capture_output=True,
              text=True,
              check=False,
              cwd=self._cwd,
          )
          if rp_result.returncode == 0:
            head_ref = 'HEAD^2'
        except FileNotFoundError:
          pass
      cmd = [
          'git',
          'log',
          f'{clean_base}..{head_ref}',
          *extra_exclusions,
          '--format=%B',
      ]
    elif clean_head:
      cmd = ['git', 'log', '-1', clean_head, '--format=%B']
    else:
      cmd = ['git', 'log', '-1', '--format=%B']
    out = _run_vcs_cmd(cmd, self._cwd)
    return out if out.strip() else ''


class HgVcs(Vcs):
  """Mercurial VCS backend."""

  name: str = 'hg'
  marker: str = '.hg'

  def diff(self, base: str | None = None) -> str:
    del base
    try:
      return _run_vcs_cmd(
          ['hg', 'diff', '--git', '-U0', '-r', 'p4base'], self._cwd
      )
    except VcsError as exc:
      if 'unknown identifier: p4base' in str(
          exc
      ) or "unknown revision 'p4base'" in str(exc):
        return _run_vcs_cmd(['hg', 'diff', '--git', '-U0'], self._cwd)
      raise

  def log(self, base: str | None = None, head: str | None = None) -> str:
    del base, head
    out = _run_vcs_cmd(
        ['hg', 'log', '-r', '.', '--template', '{desc}'], self._cwd
    )
    return out if out.strip() else ''


class JjVcs(Vcs):
  """Jujutsu VCS backend."""

  name: str = 'jj'
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

  def log(self, base: str | None = None, head: str | None = None) -> str:
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

  name: str = VCS_ANY
  marker: str = ''

  def __init__(
      self,
      start: pathlib.Path | None = None,
      *,
      root_path: pathlib.Path | None = None,
  ) -> None:
    super().__init__(start, root_path=root_path)
    self._active: Vcs | None = None

  def _backends(self) -> tuple[Vcs, ...]:
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

  def log(self, base: str | None = None, head: str | None = None) -> str:
    if self._active is not None:
      return self._active.log(base, head=head)
    errors: list[str] = []
    for backend in self._backends():
      try:
        out = backend.log(base, head=head)
        self._active = backend
        return out
      except VcsError as exc:
        errors.append(str(exc))
    raise VcsError(
        'could not retrieve commit message (' + '; '.join(errors) + ')'
    )


@functools.lru_cache(maxsize=64)
def _detect_at(candidate: pathlib.Path) -> tuple[str, pathlib.Path] | None:
  for p in (candidate, *candidate.parents):
    if (p / '.jj').exists():
      return ('jj', p)
    if (p / '.git').exists():
      return ('git', p)
    if (p / '.hg').exists():
      return ('hg', p)
  return None


_VCS_BY_NAME: Final[dict[str, type[Vcs]]] = {
    'git': GitVcs,
    'hg': HgVcs,
    'jj': JjVcs,
}


def detect(start: pathlib.Path | None = None) -> Vcs | None:
  """Detects the nearest `.git`, `.jj`, or `.hg` repository without subprocesses."""
  resolved = resolve_path(start or pathlib.Path.cwd())
  found = _detect_at(resolved)
  if found is None:
    return None
  kind, root_path = found
  return _VCS_BY_NAME[kind](resolved, root_path=root_path)


def for_name(name: VcsMode | str, start: pathlib.Path | None = None) -> Vcs:
  """Returns the `Vcs` backend for `name` (`'git'`, `'hg'`, `'jj'`, `'any'`, `'auto'`)."""
  if name == VcsMode.AUTO:
    detected = detect(start)
    return detected if detected is not None else _AnyVcs(start)
  if name == VcsMode.ANY:
    return _AnyVcs(start)
  cls = _VCS_BY_NAME.get(name)
  if cls is None:
    raise ValueError(
        f'Unknown vcs {name!r}; expected one of'
        f' {", ".join((VcsMode.AUTO, VcsMode.ANY) + VCS_TYPES)}'
    )
  return cls(start)


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
    resolved_root = resolve_path(root_dir)
    resolved_path = resolve_path(pathlib.Path(path))
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
    The modified lines and file operations per file.
  """
  lines: dict[str, set[int]] = {}
  operations: dict[str, custom_lint_metadata.FileOperation] = {}
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
    lines.setdefault(normalized, set())
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
          lines[current_file].update(range(start_line, start_line + count))

  for path in deleted:
    lines.pop(path, None)
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


def get_modified_lines_from_vcs(
    base_sha: str | None = None,
    root_dir: pathlib.Path | None = None,
    *,
    vcs: VcsMode | str = VCS_AUTO,
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
  resolved_root = resolve_path(root_dir or pathlib.Path.cwd())
  return for_name(vcs, resolved_root).get_modified_lines(
      base_sha, root_dir=resolved_root
  )


def get_commit_message(
    base_sha: str | None,
    root_dir: pathlib.Path | None = None,
    *,
    vcs: VcsMode | str = VCS_AUTO,
    head_sha: str | None = None,
) -> str:
  """Retrieves current CL/commit messages to check for disable tags.

  Args:
    base_sha: git base; commits after it are included.
    root_dir: Directory used to detect the VCS; defaults to the cwd.
    vcs: Which VCS to ask; see VCS_AUTO, VCS_ANY and VCS_TYPES.
    head_sha: Optional git head SHA/ref for shallow PR checkouts where `HEAD` is
      a depth-1 synthetic merge commit and `HEAD^2` does not resolve.

  Returns:
    The commit message(s), or '' if the message is empty.

  Raises:
    VcsError: If no VCS command succeeded.
  """
  return for_name(vcs, root_dir).get_commit_message(base_sha, head=head_sha)
