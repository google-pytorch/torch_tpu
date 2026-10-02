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
from collections.abc import Sequence
import functools
import pathlib
import subprocess
from typing import Final

# Values for `--vcs`: a name from VCS_TYPES uses only that VCS, VCS_AUTO
# detects it from the filesystem (trying every VCS in VCS_TYPES order if none is
# found), and VCS_ANY tries every VCS in VCS_TYPES order.
VCS_TYPES: Final[tuple[str, ...]] = ('git', 'hg', 'jj')
VCS_AUTO: Final[str] = 'auto'
VCS_ANY: Final[str] = 'any'

_JJ_REVSET: Final[str] = 'heads(::@ & immutable())..@'


class VcsError(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when a VCS diff or log command fails."""


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
def _find_vcs_root_cached(candidate: pathlib.Path) -> pathlib.Path | None:
  for p in (candidate, *candidate.parents):
    if (p / '.git').exists() or (p / '.hg').exists() or (p / '.jj').exists():
      return p
  return None


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
  return _find_vcs_root_cached(candidate)


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
      found = _find_vcs_root_cached(candidate)
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
  def log(self, base: str | None = None) -> str:
    """Returns commit message text for the current change."""


class GitVcs(Vcs):
  """Git VCS backend."""

  name: Final[str] = 'git'
  marker: Final[str] = '.git'

  def diff(self, base: str | None = None) -> str:
    git_cmd = ['git', 'diff', '-U0', '--no-color']
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

  def log(self, base: str | None = None) -> str:
    clean_base = _clean_base(base) if base else ''
    if clean_base:
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
      cmd = ['git', 'log', f'{clean_base}..{head_ref}', '--format=%B']
    else:
      cmd = ['git', 'log', '-1', '--format=%B']
    out = _run_vcs_cmd(cmd, self._cwd)
    return out if out.strip() else ''


class HgVcs(Vcs):
  """Mercurial VCS backend."""

  name: Final[str] = 'hg'
  marker: Final[str] = '.hg'

  def diff(self, base: str | None = None) -> str:
    del base
    return _run_vcs_cmd(['hg', 'diff', '--git', '-U0'], self._cwd)

  def log(self, base: str | None = None) -> str:
    del base
    out = _run_vcs_cmd(
        ['hg', 'log', '-l', '1', '--template', '{desc}'], self._cwd
    )
    return out if out.strip() else ''


class JjVcs(Vcs):
  """Jujutsu VCS backend."""

  name: Final[str] = 'jj'
  marker: Final[str] = '.jj'

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

  def log(self, base: str | None = None) -> str:
    del base
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

  name: Final[str] = VCS_ANY
  marker: Final[str] = ''

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
    errors: list[str] = []
    for backend in self._backends():
      try:
        out = backend.diff(base)
        self._active = backend
        return out
      except VcsError as exc:
        errors.append(str(exc))
    raise VcsError('could not compute the diff (' + '; '.join(errors) + ')')

  def log(self, base: str | None = None) -> str:
    errors: list[str] = []
    for backend in self._backends():
      try:
        return backend.log(base)
      except VcsError as exc:
        errors.append(str(exc))
    raise VcsError(
        'could not retrieve commit message (' + '; '.join(errors) + ')'
    )


@functools.lru_cache(maxsize=64)
def _detect_at(candidate: pathlib.Path) -> tuple[str, pathlib.Path] | None:
  for p in (candidate, *candidate.parents):
    if (p / '.git').exists():
      return ('git', p)
    if (p / '.jj').exists():
      return ('jj', p)
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


def for_name(name: str, start: pathlib.Path | None = None) -> Vcs:
  """Returns the `Vcs` backend for `name` (`'git'`, `'hg'`, `'jj'`, `'any'`, `'auto'`)."""
  if name == VCS_AUTO:
    detected = detect(start)
    return detected if detected is not None else _AnyVcs(start)
  if name == VCS_ANY:
    return _AnyVcs(start)
  cls = _VCS_BY_NAME.get(name)
  if cls is None:
    raise ValueError(
        f'Unknown vcs {name!r}; expected one of'
        f' {", ".join((VCS_AUTO, VCS_ANY) + VCS_TYPES)}'
    )
  return cls(start)
