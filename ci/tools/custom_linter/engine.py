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

"""Custom presubmit convention checker.

Validates coding conventions defined in a `METADATA`-style presubmit
configuration file (such as `custom_lint_rules.METADATA`); see `metadata.py`
for supported rule fields.

Supports incremental checking (`check_delta_only: true`) by parsing unified
diffs from `git diff -U0`, `hg diff -U0`, or `jj diff`, reporting violations
only on new or modified lines. For Git, the base is resolved via
`git merge-base <base> HEAD` and diffed against the working tree so uncommitted
changes are included. If diff computation fails in diff mode, execution exits
with an error rather than silently falling back to a full scan; pass `--all` to
check all files.

Rules with `disable_tags` can be suppressed by adding `TAG=<reason>` to the
commit or CL description (read from `<base>..<head>` in Git or the active
revision in Mercurial/Jujutsu), and rules with `suppression_regexp` can be
suppressed inline (for example, `# FOO_OK=<reason>`).
"""

import argparse
import bisect
from collections.abc import Iterable, Mapping, Sequence, Set
import concurrent.futures
import dataclasses
import enum
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
import tomllib
import traceback
import types
from typing import Any, Final, TextIO, TypeAlias

from ci.tools.custom_linter import metadata as custom_lint_metadata
from ci.tools.custom_linter import vcs as custom_lint_vcs

BINARY_SNIFF_BYTES: Final[int] = 8192

_MAX_WALK_WORKERS: Final[int] = 16
_PARALLEL_CANDIDATE_THRESHOLD: Final[int] = 32

DEFAULT_CANONICAL_ROOT: Final[str] = '//depot'
DEFAULT_RULES_FILE: Final[str] = 'custom_lint_rules.METADATA'

DEFAULT_EXCLUDED_PATH_REGEX: Final[str] = (
    r'(^|/)custom_lint_rules\.METADATA$|'
    r'.*\.(orig|bak|gz|tgz|bz2|xz|zst|zip|tar|whl|so|a|o|dylib|dll|'
    r'pyc|pyo|png|jpg|jpeg|gif|ico|webp|pdf|woff|woff2|ttf|otf|'
    r'npz|npy|pt|pth|ckpt|safetensors|parquet|onnx|pb)$|'
    r'.*\.del\..*'
)


class ConfigError(ValueError):
  """Raised when a configuration file or CLI argument is invalid."""


@dataclasses.dataclass(frozen=True)
class LinterConfig:
  """Project-level configuration loaded from `[tool.custom_linter]` in TOML."""

  canonical_root: str = DEFAULT_CANONICAL_ROOT
  rules_base_dir: str | None = None
  rules_file: str = DEFAULT_RULES_FILE
  path_maps: Mapping[str, str] = dataclasses.field(default_factory=dict)
  max_entire_file_bytes: int = custom_lint_metadata.MAX_ENTIRE_FILE_BYTES
  excluded_path_regex: str = DEFAULT_EXCLUDED_PATH_REGEX

  def __post_init__(self) -> None:
    if self.rules_base_dir is not None:
      object.__setattr__(
          self, 'rules_base_dir', self.rules_base_dir.rstrip('/')
      )
    object.__setattr__(
        self, 'path_maps', types.MappingProxyType(dict(self.path_maps))
    )

  @functools.cached_property
  def excluded_path_re(self) -> re.Pattern[str]:
    return re.compile(self.excluded_path_regex)


_CONFIG_KEYS: Final[Set[str]] = frozenset(
    f.name for f in dataclasses.fields(LinterConfig)
)


def _find_config_file(root_dir: pathlib.Path) -> pathlib.Path | None:
  """Finds `pyproject.toml` in `root_dir`."""
  candidate = root_dir / 'pyproject.toml'
  return candidate if candidate.is_file() else None


def load_config(
    root_dir: pathlib.Path | None = None,
    config_file: os.PathLike[str] | str | None = None,
) -> LinterConfig:
  """Loads `[tool.custom_linter]` from `pyproject.toml` or `config_file`."""
  if config_file is not None:
    path = pathlib.Path(config_file)
    if not path.is_file():
      raise ConfigError(f'config file not found: {path}')
  else:
    target_root = root_dir if root_dir is not None else find_project_root()
    found = _find_config_file(target_root)
    if found is None:
      return LinterConfig()
    path = found

  try:
    data = tomllib.loads(path.read_text(encoding='utf-8'))
  except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
    raise ConfigError(f'{path}: invalid config file: {e}') from e

  tool_table = data.get('tool')
  if isinstance(tool_table, Mapping) and 'custom_linter' in tool_table:
    section = tool_table['custom_linter']
  elif config_file is not None:
    raise ConfigError(f'{path}: missing [tool.custom_linter] section')
  else:
    return LinterConfig()

  if not isinstance(section, Mapping):
    raise ConfigError(f'{path}: [tool.custom_linter] must be a table')
  unknown = sorted(set(section) - _CONFIG_KEYS)
  if unknown:
    raise ConfigError(
        f'{path}: unknown [tool.custom_linter] key(s): {", ".join(unknown)}'
    )

  kwargs: dict[str, Any] = {}
  for str_key in (
      'canonical_root',
      'rules_base_dir',
      'rules_file',
      'excluded_path_regex',
  ):
    if str_key in section:
      val = section[str_key]
      if not isinstance(val, str):
        raise ConfigError(
            f'{path}: {str_key} must be a string, got {type(val).__name__}'
        )
      kwargs[str_key] = val

  if 'max_entire_file_bytes' in section:
    val = section['max_entire_file_bytes']
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
      raise ConfigError(
          f'{path}: max_entire_file_bytes must be a positive integer, got'
          f' {type(val).__name__}'
      )
    kwargs['max_entire_file_bytes'] = val

  if 'path_maps' in section:
    raw_maps = section['path_maps']
    if not isinstance(raw_maps, Mapping):
      raise ConfigError(
          f'{path}: path_maps must be a table of string pairs, got'
          f' {type(raw_maps).__name__}'
      )
    for k, v in raw_maps.items():
      if not isinstance(k, str) or not isinstance(v, str):
        raise ConfigError(
            f'{path}: path_maps entries must be string pairs, got key'
            f' {type(k).__name__} and value {type(v).__name__}'
        )
    kwargs['path_maps'] = raw_maps

  cfg = LinterConfig(**kwargs)
  try:
    _ = cfg.excluded_path_re
  except re.error as e:
    raise ConfigError(f'{path}: invalid excluded_path_regex: {e}') from e
  return cfg


def _has_rules_file(
    path: pathlib.Path, config: LinterConfig | None = None
) -> bool:
  """Returns True if path has a project rules file."""
  if config is not None:
    return (path / config.rules_file).is_file()
  if (path / 'pyproject.toml').is_file():
    cfg = load_config(path)
    return (path / cfg.rules_file).is_file()
  rules = path / DEFAULT_RULES_FILE
  if not rules.is_file():
    return False
  if (path.parent / 'pyproject.toml').is_file():
    parent_cfg = load_config(path.parent)
    if rules == path.parent / parent_cfg.rules_file:
      return False
  return True


def default_path_mapper(
    root_dir: pathlib.Path | None = None,
    config: LinterConfig | None = None,
) -> custom_lint_metadata.PathMapper:
  """Returns the mapper from project-relative to canonical rule paths."""
  target_root = (
      root_dir if root_dir is not None else find_project_root(config=config)
  )
  cfg = config or load_config(target_root)
  return custom_lint_metadata.PathMapper(
      cfg.canonical_root, tuple(cfg.path_maps.items())
  )


@dataclasses.dataclass(frozen=True)
class CandidateFile:
  """A candidate file paired with matching rules to avoid redundant evaluation."""

  file_path: pathlib.Path
  rel_path: str
  matching_rules: Sequence[custom_lint_metadata.Rule]


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
  severity: custom_lint_metadata.Severity = custom_lint_metadata.Severity.ERROR

  @property
  def location(self) -> str:
    if self.line_number is None:
      return self.file_path
    return f'{self.file_path}:{self.line_number}'


def normalize_path(path: pathlib.Path, root_dir: pathlib.Path) -> str:
  """Returns path relative to root_dir as POSIX string, or str(path) if not relative."""
  try:
    return path.relative_to(root_dir).as_posix()
  except ValueError:
    pass
  try:
    return (
        path.resolve()
        .relative_to(custom_lint_vcs.resolve_path(root_dir))
        .as_posix()
    )
  except ValueError:
    return path.as_posix()


# TODO(b/581426770): Migrate description-only disable_tags to in-code
# justifications so rules do not depend on exported commit-message tags.
def is_rule_disabled(
    rule: custom_lint_metadata.Rule,
    commit: custom_lint_vcs.Commit,
) -> bool:
  """Returns True if `commit` disables `rule` via `disable_tags`."""
  if not rule.disable_tags or not commit.message:
    return False
  return bool(commit.tags & rule.disable_tags)


def is_path_excluded(
    rel_path: str,
    excluded_path_re: re.Pattern[str] | None = None,
) -> bool:
  """Returns True if the file path is globally excluded from custom lint checks."""
  compiled = (
      excluded_path_re
      if excluded_path_re is not None
      else re.compile(DEFAULT_EXCLUDED_PATH_REGEX)
  )
  return compiled.search(rel_path) is not None


def find_rules_file(
    root_dir: pathlib.Path,
    config: LinterConfig | None = None,
) -> pathlib.Path:
  """Returns the default rules file for the project rooted at root_dir."""
  cfg = config or load_config(root_dir)
  return root_dir / cfg.rules_file


def load_rules(
    rules_file: os.PathLike[str] | str | None = None,
    root_dir: pathlib.Path | None = None,
    mapper: custom_lint_metadata.PathMapper | None = None,
    config: LinterConfig | None = None,
) -> custom_lint_metadata.RuleSet:
  """Loads the custom lint rules.

  Relative rule paths are resolved against the canonical directory of the rules
  file when it is a subdirectory rules file inside the project, and against
  `config.rules_base_dir` (or `mapper.canonical_root` if unset) otherwise.

  Args:
    rules_file: METADATA-format rules file; defaults to the project's.
    root_dir: Project root; defaults to `find_project_root()`.
    mapper: Maps project-relative paths to canonical paths.
    config: Optional `LinterConfig` overriding rules filenames and path maps.

  Returns:
    The loaded rules.
  """
  root_dir = root_dir or find_project_root()
  cfg = config or load_config(root_dir)
  mapper = mapper or default_path_mapper(root_dir, cfg)
  path = (
      find_rules_file(root_dir, cfg)
      if rules_file is None
      else pathlib.Path(rules_file)
  )
  rel = normalize_path(path, root_dir)
  in_project = (
      bool(rel)
      and not pathlib.Path(rel).is_absolute()
      and not rel.startswith('../')
  )
  if in_project and rel != cfg.rules_file:
    base_dir = posixpath.dirname(mapper.to_canonical(rel))
  elif cfg.rules_base_dir is not None:
    base_dir = cfg.rules_base_dir
  else:
    base_dir = mapper.canonical_root
  return custom_lint_metadata.load_rules_file(path, base_dir)


def _read_text_file(path: pathlib.Path) -> str | None:
  """Reads a text file with UTF-8 replacement, returning None if binary.

  Broad rules (e.g. `path_expression: "//depot/.../my_project/..."`) match files
  regardless of extension. Opening in binary mode to inspect the first
  `BINARY_SNIFF_BYTES` bytes for a NUL byte (`b'\\x00'`, the standard Git/grep
  binary heuristic) lets us skip binary files without reading the rest of a
  large binary blob or accepting NUL bytes (which are valid UTF-8 in
  `open(path, encoding='utf-8')`). Invalid UTF-8 bytes in non-binary files are
  replaced with `U+FFFD` so RE2 rules can still evaluate the file; `OSError`
  propagates to the caller.
  """
  with open(path, 'rb') as f:
    chunk = f.read(BINARY_SNIFF_BYTES)
    if b'\x00' in chunk:
      return None
    if len(chunk) < BINARY_SNIFF_BYTES:
      raw = chunk
    else:
      rest = f.read()
      if b'\x00' in rest:
        return None
      raw = chunk + rest
  text = raw.decode('utf-8', errors='replace')
  if '\r' in text:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
  return text


def _compute_line_starts(content: str) -> Sequence[int]:
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

  def __init__(self, text: str) -> None:
    self.text = text
    self._lines: Sequence[str] | None = None
    self._line_starts: Sequence[int] | None = None

  @property
  def lines(self) -> Sequence[str]:
    if self._lines is None:
      lines = self.text.split('\n')
      if lines and not lines[-1]:
        lines.pop()
      self._lines = lines
    return self._lines

  @property
  def line_starts(self) -> Sequence[int]:
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
    """Returns the character range of lines first..last, including trailing newline."""
    starts = self.line_starts
    start = starts[first - 1]
    end = starts[last] if last < len(starts) else len(self.text)
    return start, max(start, end)


# A finding: 1-based line (or None) and the violation message.
_Finding: TypeAlias = tuple[int | None, str]


def _check_lines(
    rule: custom_lint_metadata.Rule,
    prohibited: custom_lint_metadata.RegexPattern,
    text: TextFile,
    changed: Set[int] | None,
) -> Sequence[_Finding]:
  """Line mode: returns lines matching prohibited but not suppression."""
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
      return ()
    n = text.line_count
    candidates = sorted(i for i in cand if 1 <= i <= n)
  else:
    candidates = range(1, text.line_count + 1)
  lines = text.lines
  supp = rule.suppression_re
  msg = rule.failure_message
  findings: list[_Finding] = []
  for i in candidates:
    line = lines[i - 1]
    if prohibited.search(line) and not (supp and supp.search(line)):
      findings.append((i, msg))
  return findings


def _chunks(
    changed: Set[int] | None, line_count: int
) -> Sequence[tuple[int, int]]:
  """Groups line numbers into runs of consecutive lines, clamped to the file."""
  if line_count == 0:
    return ()
  if changed is None:
    return ((1, line_count),)
  out: list[tuple[int, int]] = []
  for n in sorted(changed):
    if not 1 <= n <= line_count:
      continue
    if out and out[-1][1] + 1 == n:
      out[-1] = (out[-1][0], n)
    else:
      out.append((n, n))
  return out


def _check_chunks(
    rule: custom_lint_metadata.Rule,
    prohibited: custom_lint_metadata.RegexPattern,
    text: TextFile,
    changed: Set[int] | None,
) -> Sequence[_Finding]:
  """Modified-chunks mode: matches runs of changed lines as one text."""
  supp = rule.suppression_re
  msg = rule.failure_message
  findings: list[_Finding] = []
  for first, last in _chunks(changed, text.line_count):
    start, end = text.span(first, last)
    chunk = text.text[start:end]
    limit = max(start, end - 1)
    match_lines = [
        text.line_of(min(start + m.start(), limit))
        for m in prohibited.finditer(chunk)
    ]
    if not match_lines:
      continue
    if supp is None:
      for line_no in match_lines:
        findings.append((line_no, msg))
      continue
    for i, seg_first in enumerate(match_lines):
      seg_last = next(
          (ln - 1 for ln in match_lines[i + 1 :] if ln > seg_first),
          last,
      )
      seg_start, seg_end = text.span(seg_first, seg_last)
      if not supp.search(text.text[seg_start:seg_end]):
        findings.append((seg_first, msg))
  return findings


def _any_line_matches(rule: custom_lint_metadata.Rule, text: TextFile) -> bool:
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
    rule: custom_lint_metadata.Rule,
    text: TextFile,
    changed: Set[int] | None,
    *,
    max_entire_file_bytes: int = custom_lint_metadata.MAX_ENTIRE_FILE_BYTES,
) -> Sequence[_Finding]:
  """Evaluates a check_contents rule against one file."""
  findings: list[_Finding] = []
  content = text.text
  msg = rule.failure_message
  prohibited = rule.prohibited_re

  if rule.mode == custom_lint_metadata.MatchMode.ENTIRE_FILE:
    limit = max_entire_file_bytes
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
            (text.line_of(m.start()), msg) for m in prohibited.finditer(content)
        )
    if rule.required_re is not None and not rule.required_re.search(content):
      findings.append((None, msg))
    return findings

  if prohibited is not None:
    if rule.mode == custom_lint_metadata.MatchMode.MODIFIED_CHUNKS:
      findings.extend(_check_chunks(rule, prohibited, text, changed))
    else:
      findings.extend(_check_lines(rule, prohibited, text, changed))
  if rule.required_re is not None and not _any_line_matches(rule, text):
    findings.append((None, msg))
  return findings


def _file_operation(
    modified_lines: custom_lint_vcs.ModifiedLines | None,
    rel_path: str,
) -> custom_lint_metadata.FileOperation | None:
  if modified_lines is not None:
    return modified_lines.operations.get(rel_path)
  return None


def _rule_applies(
    rule: custom_lint_metadata.Rule,
    rel_path: str,
    modified_lines: custom_lint_vcs.ModifiedLines | None,
    operation: custom_lint_metadata.FileOperation | None,
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
    rule: custom_lint_metadata.Rule,
    rel_path: str,
    text: TextFile | None,
    findings: Sequence[_Finding],
) -> Sequence[Violation]:
  """Builds Violation objects from raw findings."""
  out: list[Violation] = []
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
            message=message,
            severity=rule.severity,
        )
    )
  return out


def check_file_content(
    content: str,
    rel_path: str,
    rule: custom_lint_metadata.Rule,
    modified_lines: custom_lint_vcs.ModifiedLines | None = None,
    *,
    canonical_path: str,
    file_operation: custom_lint_metadata.FileOperation | None = None,
    max_entire_file_bytes: int = custom_lint_metadata.MAX_ENTIRE_FILE_BYTES,
    excluded_path_re: re.Pattern[str] | None = None,
) -> Sequence[Violation]:
  """Checks file content against a rule and returns any violations.

  Args:
    content: The file's text.
    rel_path: Project-relative path, used in reports and to look up
      modified_lines.
    rule: The rule to check.
    modified_lines: Changed lines and operations per file, or None without a
      change context (every line is checked).
    canonical_path: Path matched against the rule's path filters.
    file_operation: How the change touched the file; defaults to
      `modified_lines.operations[rel_path]` when available.
    max_entire_file_bytes: Maximum file size in bytes matched by entire-file
      rules.
    excluded_path_re: Compiled regex of globally excluded relative paths.

  Returns:
    The violations found.
  """
  if is_path_excluded(rel_path, excluded_path_re):
    return []
  if not rule.matches_path(canonical_path):
    return []
  if file_operation is None:
    file_operation = _file_operation(modified_lines, rel_path)
  if not _rule_applies(rule, rel_path, modified_lines, file_operation):
    return []
  if rule.kind == custom_lint_metadata.CheckKind.CHECK_FILE_OPERATION:
    return _violations(rule, rel_path, None, [(None, rule.failure_message)])
  text = TextFile(content)
  changed = (
      modified_lines.get(rel_path)
      if rule.check_delta_only and modified_lines is not None
      else None
  )
  return _violations(
      rule,
      rel_path,
      text,
      _evaluate(
          rule, text, changed, max_entire_file_bytes=max_entire_file_bytes
      ),
  )


@dataclasses.dataclass(frozen=True, slots=True)
class _ExclusionGroup:
  """Rules sharing both `path_regex` and `path_exclusion_regex`."""

  exclusion_re: custom_lint_metadata.RegexPattern | None
  indexed_rules: Sequence[tuple[int, custom_lint_metadata.Rule]]


@dataclasses.dataclass(frozen=True, slots=True)
class _CompiledPathGroup:
  """Compiled `path_re` paired with its per-exclusion rule groups."""

  path_re: custom_lint_metadata.RegexPattern
  exclusion_groups: Sequence[_ExclusionGroup]


_CompiledPathGroups: TypeAlias = Sequence[_CompiledPathGroup]


def _build_path_matcher(
    rules: Sequence[custom_lint_metadata.Rule],
) -> _CompiledPathGroups:
  """Groups rules by (path_regex, path_exclusion_regex) to avoid duplicate regex checks."""
  by_path: dict[str, list[tuple[int, custom_lint_metadata.Rule]]] = {}
  for idx, r in enumerate(rules):
    by_path.setdefault(r.path_regex, []).append((idx, r))
  compiled_groups: list[_CompiledPathGroup] = []
  for indexed_rules in by_path.values():
    path_re = indexed_rules[0][1].path_re
    by_excl: dict[str | None, list[tuple[int, custom_lint_metadata.Rule]]] = {}
    for idx, r in indexed_rules:
      by_excl.setdefault(r.path_exclusion_regex, []).append((idx, r))
    excl_groups = tuple(
        _ExclusionGroup(irules[0][1].path_exclusion_re, tuple(irules))
        for irules in by_excl.values()
    )
    compiled_groups.append(_CompiledPathGroup(path_re, excl_groups))
  return tuple(compiled_groups)


def _match_rules_grouped(
    canonical_path: str,
    compiled_groups: _CompiledPathGroups,
) -> Sequence[custom_lint_metadata.Rule]:
  """Returns rules matching canonical_path in original rule sequence order."""
  matched: list[tuple[int, custom_lint_metadata.Rule]] = []
  for group in compiled_groups:
    if group.path_re.match(canonical_path):
      for excl_group in group.exclusion_groups:
        excl_re = excl_group.exclusion_re
        if excl_re is None or not excl_re.match(canonical_path):
          matched.extend(excl_group.indexed_rules)
  if len(matched) > 1:
    matched.sort(key=lambda x: x[0])
  return [r for _, r in matched]


# Directory names skipped when walking. Symlinked directories (e.g. `bazel-bin`)
# are not descended into by `scan_entries` (`is_dir(follow_symlinks=False)`);
# they are also listed here for checkouts that materialize them as real dirs.
DEFAULT_IGNORE_DIRS: Final[Set[str]] = frozenset({
    '.git',
    '.hg',
    '.jj',
    '.mypy_cache',
    '.nox',
    '.pytest_cache',
    '.pytype',
    '.ruff_cache',
    '.tox',
    '.venv',
    '__pycache__',
    'bazel-bin',
    'bazel-genfiles',
    'bazel-out',
    'bazel-testlogs',
    'node_modules',
    'venv',
})


def find_candidate_files(
    root_dir: pathlib.Path,
    rules: Sequence[custom_lint_metadata.Rule],
    *,
    max_workers: int = _MAX_WALK_WORKERS,
    mapper: custom_lint_metadata.PathMapper | None = None,
    rel_prefix: str = '',
    ignore_dirs: Set[str] | None = None,
    use_default_excludes: bool = True,
    excluded_path_re: re.Pattern[str] | None = None,
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
      `DEFAULT_IGNORE_DIRS` when `use_default_excludes` is True.
    use_default_excludes: If True and `ignore_dirs` is None, apply default
      directory exclusions (`DEFAULT_IGNORE_DIRS`).
    excluded_path_re: Compiled regex of globally excluded relative paths.

  Returns:
    The candidates, sorted by rel_path.
  """
  if not rules:
    return []
  if ignore_dirs is None:
    ignore_dirs = DEFAULT_IGNORE_DIRS if use_default_excludes else frozenset()
  elif not isinstance(ignore_dirs, frozenset):
    ignore_dirs = frozenset(ignore_dirs)

  compiled_groups = _build_path_matcher(rules)
  to_canonical = mapper.to_canonical if mapper is not None else str
  rel_prefix = rel_prefix.strip('/')

  def scan_entries(
      dirpath: str, rel_dir: str
  ) -> tuple[list[CandidateFile], list[tuple[str, str]]]:
    local_candidates: list[CandidateFile] = []
    subdirs: list[tuple[str, str]] = []
    with os.scandir(dirpath) as it:
      for entry in it:
        name = entry.name
        if entry.is_dir(follow_symlinks=False):
          if name not in ignore_dirs:
            sub_rel = f'{rel_dir}/{name}' if rel_dir else name
            subdirs.append((entry.path, sub_rel))
        elif entry.is_file(follow_symlinks=False) or (
            entry.is_symlink() and pathlib.Path(entry.path).is_file()
        ):
          rel_path = f'{rel_dir}/{name}' if rel_dir else name
          if is_path_excluded(rel_path, excluded_path_re):
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


def _default_project_root(
    cwd: pathlib.Path,
    config: LinterConfig | None = None,
) -> pathlib.Path | None:
  """Returns the nearest dir with a rules file, from cwd up to the VCS root."""
  for raw_start in dict.fromkeys(
      (cwd.absolute(), custom_lint_vcs.resolve_path(cwd))
  ):
    start = raw_start.parent if raw_start.is_file() else raw_start
    vcs_root = custom_lint_vcs.find_vcs_root(start)
    for p in (start, *start.parents):
      if _has_rules_file(p, config):
        return p
      if vcs_root is not None and p == vcs_root:
        break
  return None


def _fallback_project_root(target: pathlib.Path) -> pathlib.Path:
  """Returns the nearest `pyproject.toml` or VCS root above `target`."""
  vcs_root: pathlib.Path | None = None
  for raw_start in dict.fromkeys(
      (target.absolute(), custom_lint_vcs.resolve_path(target))
  ):
    candidate = raw_start.parent if raw_start.is_file() else raw_start
    vcs_root = custom_lint_vcs.find_vcs_root(candidate)
    for p in (candidate, *candidate.parents):
      if (p / 'pyproject.toml').is_file():
        return p
      if vcs_root is not None and p == vcs_root:
        break
  resolved = custom_lint_vcs.resolve_path(target)
  return (
      vcs_root
      if vcs_root is not None
      else (resolved.parent if resolved.is_file() else resolved)
  )


def find_project_root(
    start_path: pathlib.Path | None = None,
    config: LinterConfig | None = None,
) -> pathlib.Path:
  """Finds the root directory of the project starting at `start_path` or CWD."""
  target = start_path or pathlib.Path.cwd()
  found = _default_project_root(target, config=config)
  if found is not None:
    return found
  return _fallback_project_root(target)


def _check_candidate(
    task: tuple[
        CandidateFile,
        custom_lint_vcs.ModifiedLines | None,
        int,
    ],
) -> Sequence[Violation]:
  """Worker function to evaluate all matching rules on a single CandidateFile."""
  candidate, modified_lines, max_entire_file_bytes = task
  rel_path = candidate.rel_path
  operation = _file_operation(modified_lines, rel_path)
  violations: list[Violation] = []
  content_rules: list[custom_lint_metadata.Rule] = []
  for rule in candidate.matching_rules:
    if not _rule_applies(rule, rel_path, modified_lines, operation):
      continue
    if rule.kind == custom_lint_metadata.CheckKind.CHECK_FILE_OPERATION:
      violations.extend(
          _violations(rule, rel_path, None, [(None, rule.failure_message)])
      )
    else:
      content_rules.append(rule)
  if not content_rules:
    return violations

  try:
    content = _read_text_file(candidate.file_path)
  except FileNotFoundError:
    return violations
  if content is None:
    return violations
  text = TextFile(content)
  for rule in content_rules:
    changed = (
        modified_lines.get(rel_path)
        if rule.check_delta_only and modified_lines is not None
        else None
    )
    violations.extend(
        _violations(
            rule,
            rel_path,
            text,
            _evaluate(
                rule,
                text,
                changed,
                max_entire_file_bytes=max_entire_file_bytes,
            ),
        )
    )
  return violations


_WORKER_CANDIDATES: Sequence[CandidateFile] = ()
_WORKER_MODIFIED_LINES: custom_lint_vcs.ModifiedLines | None = None
_WORKER_MAX_ENTIRE_FILE_BYTES: int = custom_lint_metadata.MAX_ENTIRE_FILE_BYTES


def _check_candidate_slice(
    bounds: tuple[int, int],
    candidates: Sequence[CandidateFile] | None = None,
    modified_lines: custom_lint_vcs.ModifiedLines | None = None,
    max_entire_file_bytes: int | None = None,
) -> Sequence[Violation]:
  """Evaluates a contiguous slice of candidates (or _WORKER_CANDIDATES via fork CoW)."""
  start_idx, end_idx = bounds
  out: list[Violation] = []
  if candidates is None:
    candidates = _WORKER_CANDIDATES
    modified_lines = _WORKER_MODIFIED_LINES
    max_entire_file_bytes = _WORKER_MAX_ENTIRE_FILE_BYTES
  elif max_entire_file_bytes is None:
    max_entire_file_bytes = custom_lint_metadata.MAX_ENTIRE_FILE_BYTES
  for idx in range(start_idx, end_idx):
    out.extend(
        _check_candidate(
            (candidates[idx], modified_lines, max_entire_file_bytes)
        )
    )
  return out


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

  violations: Sequence[Violation]
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
    return any(
        v.severity == custom_lint_metadata.Severity.ERROR
        for v in self.violations
    )

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
    rules: Sequence[custom_lint_metadata.Rule],
    compiled_groups: _CompiledPathGroups,
    mapper: custom_lint_metadata.PathMapper,
    walk_workers: int,
    ignore_dirs: Set[str] | None = None,
    excluded_path_re: re.Pattern[str] | None = None,
    working_dir: pathlib.Path | None = None,
) -> Sequence[CandidateFile]:
  """Resolves explicit file and directory paths into matching CandidateFiles.

  Args:
    files: File or directory paths (absolute, CWD-relative, or root-relative).
    root_dir: Resolved project root directory.
    rules: Active rules used when walking directory arguments.
    compiled_groups: Pre-grouped path matchers from `_build_path_matcher`.
    mapper: `PathMapper` translating project-relative paths to canonical paths.
    walk_workers: Maximum threads for directory walks.
    ignore_dirs: Optional set of directory names to exclude when walking
      directory arguments.
    excluded_path_re: Compiled regex of globally excluded relative paths.
    working_dir: Effective working directory for resolving relative paths;
      defaults to `pathlib.Path.cwd()`.

  Returns:
    Sequence of `CandidateFile` entries matching at least one active rule.
  """
  base_cwd = working_dir if working_dir is not None else pathlib.Path.cwd()
  candidate_files: list[CandidateFile] = []
  for f in files:
    p = pathlib.Path(f)
    if not p.is_absolute():
      cwd_path = (base_cwd / p).resolve()
      root_path = (root_dir / p).resolve()
      p = (
          cwd_path
          if cwd_path.exists() and cwd_path.is_relative_to(root_dir)
          else (root_path if root_path.exists() else cwd_path)
      )
    else:
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
              excluded_path_re=excluded_path_re,
          )
      )
    elif p.is_file():
      if is_path_excluded(rel_path, excluded_path_re):
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
    else:
      raise FileNotFoundError(f'Path does not exist: {f}')
  return candidate_files


def _candidates_for_diff(
    modified_lines: custom_lint_vcs.ModifiedLines,
    root_dir: pathlib.Path,
    compiled_groups: _CompiledPathGroups,
    mapper: custom_lint_metadata.PathMapper,
    excluded_path_re: re.Pattern[str] | None = None,
) -> Sequence[CandidateFile]:
  """Selects changed files from a VCS diff that match at least one rule.

  Args:
    modified_lines: `ModifiedLines` mapping project-relative paths to modified
      line numbers and file operations.
    root_dir: Resolved project root directory.
    compiled_groups: Pre-grouped path matchers from `_build_path_matcher`.
    mapper: `PathMapper` translating project-relative paths to canonical paths.
    excluded_path_re: Compiled regex of globally excluded relative paths.

  Returns:
    Sorted sequence of `CandidateFile` entries for changed files.
  """
  candidate_files: list[CandidateFile] = []
  all_paths = set(modified_lines) | set(modified_lines.operations)
  for rel_path in sorted(all_paths):
    if is_path_excluded(rel_path, excluded_path_re):
      continue
    op = _file_operation(modified_lines, rel_path)
    full_path = root_dir / rel_path
    if op == custom_lint_metadata.FileOperation.DELETE:
      matching = [
          r
          for r in _match_rules_grouped(
              mapper.to_canonical(rel_path), compiled_groups
          )
          if r.kind == custom_lint_metadata.CheckKind.CHECK_FILE_OPERATION
          and r.applies_to(op)
      ]
    else:
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
    head_sha: str | None = None,
    check_all: bool = False,
    jobs: int | None = None,
    rules_file: os.PathLike[str] | str | None = None,
    only_rules: Sequence[str] | None = None,
    mapper: custom_lint_metadata.PathMapper | None = None,
    vcs: custom_lint_vcs.VcsMode | custom_lint_vcs.Vcs = (
        custom_lint_vcs.VcsMode.AUTO
    ),
    exclude_dirs: Iterable[str] | None = None,
    use_default_excludes: bool = True,
    config: LinterConfig | None = None,
    working_dir: pathlib.Path | None = None,
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
    head_sha: Git head ref/SHA for reading commit messages (defaults to HEAD).
    check_all: If True, scan all files and all lines under `root_dir` instead of
      computing a VCS diff.
    jobs: Number of parallel worker processes/threads (`None` or `<= 0`
      auto-detects available CPUs; `1` runs serially in-process).
    rules_file: Optional path to the METADATA-format rules file; defaults to the
      repository's standard rules file.
    only_rules: Optional sequence of rule IDs to run; if provided, only those
      rules are evaluated.
    mapper: Optional `PathMapper` translating between canonical depot paths and
      project-relative paths; defaults to `default_path_mapper()`.
    vcs: VCS selection (`VcsMode` or a prebuilt `Vcs` backend instance).
    exclude_dirs: Optional iterable of exact directory names to exclude when
      walking the directory tree.
    use_default_excludes: If True, apply `DEFAULT_IGNORE_DIRS` when walking the
      directory tree.
    config: Optional `LinterConfig` instance overriding `pyproject.toml` values.
    working_dir: Optional effective working directory for resolving relative
      entries in `files`; defaults to `pathlib.Path.cwd()`.

  Returns:
    A `CheckResult` containing the filtered violations and timing/worker stats.

  Raises:
    LoadError: If the rules file cannot be parsed or validated.
    ConfigError: If `only_rules` contains unknown rule IDs or `pyproject.toml`
      is invalid.
    FileNotFoundError: If an explicit entry in `files` or `root_dir` does not
      exist.
    OSError: If reading a candidate file or walking `root_dir` fails (other than
      a candidate file vanishing mid-run).
    VcsError: If diff computation or commit-message retrieval fails in VCS diff
      mode.
  """
  root_dir = custom_lint_vcs.resolve_path(root_dir)
  cfg = config or load_config(root_dir)
  mapper = mapper or default_path_mapper(root_dir, cfg)
  rule_set = load_rules(
      rules_file, root_dir=root_dir, mapper=mapper, config=cfg
  )
  rules = rule_set.rules
  if only_rules:
    unknown = sorted(set(only_rules) - {r.id for r in rules})
    if unknown:
      raise ConfigError(f'unknown rule id(s): {", ".join(unknown)}')
    rules = [r for r in rules if r.id in only_rules]
  active_rules = list(rules)

  vcs_backend: custom_lint_vcs.Vcs | None = None
  if not check_all and not files:
    vcs_backend = (
        vcs
        if isinstance(vcs, custom_lint_vcs.Vcs)
        else custom_lint_vcs.for_name(vcs, root_dir)
    )
    try:
      modified_lines = vcs_backend.get_modified_lines(
          base_sha, root_dir=root_dir
      )
    except custom_lint_vcs.VcsError as e:
      target = f' against {base_sha}' if base_sha else ''
      raise custom_lint_vcs.VcsError(
          f'could not compute the diff{target}: {e}; pass --all to scan'
          ' everything'
      ) from e
  else:
    modified_lines = None

  base_ignores: Set[str] = (
      DEFAULT_IGNORE_DIRS if use_default_excludes else frozenset()
  )
  ignore_dirs: Set[str] = (
      base_ignores | frozenset(exclude_dirs) if exclude_dirs else base_ignores
  )
  excluded_path_re = cfg.excluded_path_re
  max_entire_file_bytes = cfg.max_entire_file_bytes

  detected_cpus = os.cpu_count() or 1
  walk_workers = 1 if jobs == 1 else min(_MAX_WALK_WORKERS, detected_cpus * 2)

  t0 = time.perf_counter()
  if files:
    compiled_groups = _build_path_matcher(active_rules)
    candidate_files = list(
        _candidates_for_paths(
            files,
            root_dir,
            active_rules,
            compiled_groups,
            mapper,
            walk_workers,
            ignore_dirs=ignore_dirs,
            excluded_path_re=excluded_path_re,
            working_dir=working_dir,
        )
    )
  elif modified_lines is not None:
    compiled_groups = _build_path_matcher(active_rules)
    candidate_files = list(
        _candidates_for_diff(
            modified_lines,
            root_dir,
            compiled_groups,
            mapper,
            excluded_path_re=excluded_path_re,
        )
    )
  else:
    candidate_files = list(
        find_candidate_files(
            root_dir,
            active_rules,
            max_workers=walk_workers,
            mapper=mapper,
            ignore_dirs=ignore_dirs,
            excluded_path_re=excluded_path_re,
        )
    )
  t_walk = time.perf_counter() - t0

  effective_jobs = (
      jobs
      if jobs is not None and jobs > 0
      else (
          detected_cpus
          if len(candidate_files) > _PARALLEL_CANDIDATE_THRESHOLD
          else 1
      )
  )

  t_eval_start = time.perf_counter()
  all_violations: list[Violation] = []
  if effective_jobs > 1 and len(candidate_files) > 1:
    for rule in active_rules:
      rule.prewarm()
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
        for slice_violations in executor.map(
            lambda b: _check_candidate_slice(
                b, candidate_files, modified_lines, max_entire_file_bytes
            ),
            slices,
        ):
          all_violations.extend(slice_violations)
    elif 'fork' in start_methods and threading.active_count() == 1:
      # fork is the fastest option: workers inherit _WORKER_CANDIDATES and the
      # compiled rules copy-on-write, with nothing to pickle. On a 24-core
      # `--all` run it took 1.3 s vs 1.5 s (forkserver) and 1.7 s (spawn).
      # It is only safe without other live threads (the walk's thread pool
      # has already exited here); embedders with threads take the path below.
      global _WORKER_CANDIDATES, _WORKER_MODIFIED_LINES
      global _WORKER_MAX_ENTIRE_FILE_BYTES
      _WORKER_CANDIDATES = candidate_files
      _WORKER_MODIFIED_LINES = modified_lines
      _WORKER_MAX_ENTIRE_FILE_BYTES = max_entire_file_bytes
      try:
        mp_ctx = multiprocessing.get_context('fork')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_jobs, mp_context=mp_ctx
        ) as executor:
          for slice_violations in executor.map(_check_candidate_slice, slices):
            all_violations.extend(slice_violations)
      finally:
        _WORKER_CANDIDATES = ()
        _WORKER_MODIFIED_LINES = None
        _WORKER_MAX_ENTIRE_FILE_BYTES = (
            custom_lint_metadata.MAX_ENTIRE_FILE_BYTES
        )
    else:
      # Explicit non-fork context: the default is fork on Linux (< 3.14).
      mp_ctx = multiprocessing.get_context(
          'forkserver' if 'forkserver' in start_methods else 'spawn'
      )
      tasks = [
          (c, modified_lines, max_entire_file_bytes) for c in candidate_files
      ]
      with concurrent.futures.ProcessPoolExecutor(
          max_workers=effective_jobs, mp_context=mp_ctx
      ) as executor:
        for file_violations in executor.map(
            _check_candidate, tasks, chunksize=step
        ):
          all_violations.extend(file_violations)
  else:
    for candidate in candidate_files:
      all_violations.extend(
          _check_candidate((candidate, modified_lines, max_entire_file_bytes))
      )

  if all_violations and vcs_backend is not None:
    rule_by_id = {r.id: r for r in rules}
    if any(rule_by_id[v.rule_id].disable_tags for v in all_violations):
      commit = vcs_backend.get_commit(base_sha, head=head_sha)
      if commit.message:
        disabled_ids = {r.id for r in rules if is_rule_disabled(r, commit)}
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
  """Formats violations into a grouped human-readable summary report."""
  by_rule: dict[str, list[Violation]] = {}
  for v in violations:
    by_rule.setdefault(v.rule_id, []).append(v)

  sorted_groups = sorted(
      by_rule.values(), key=lambda g: (-len(g), g[0].rule_id)
  )
  errors = sum(
      1 for v in violations if v.severity == custom_lint_metadata.Severity.ERROR
  )
  warnings = sum(
      1
      for v in violations
      if v.severity == custom_lint_metadata.Severity.WARNING
  )
  notices = sum(
      1
      for v in violations
      if v.severity == custom_lint_metadata.Severity.NOTICE
  )
  summary_parts = [f'{errors} errors', f'{warnings} warnings']
  if notices:
    summary_parts.append(f'{notices} notices')
  out_lines = [
      '=' * 60,
      f' SUMMARY: {", ".join(summary_parts)}',
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
        'INFO: Custom convention checks passed successfully.',
        file=out,
    )
    return

  banner_level = 'ERROR' if result.has_errors else 'WARNING'
  print('=' * 65, file=out)
  print(
      f'{banner_level}: {len(result.violations)} custom convention check'
      ' violation(s) found:',
      file=out,
  )
  print('=' * 65, file=out)
  for v in result.violations:
    severity = (
        ''
        if v.severity == custom_lint_metadata.Severity.ERROR
        else f' ({v.severity})'
    )
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


class OutputFormat(enum.StrEnum):
  """Supported stdout output formats for `run_checks` and `--format`."""

  TEXT = 'text'
  JSON = 'json'


def run_checks(
    root_dir: pathlib.Path,
    *,
    files: Sequence[str] | None = None,
    base_sha: str | None = None,
    head_sha: str | None = None,
    check_all: bool = False,
    jobs: int | None = None,
    report_file: os.PathLike[str] | str | None = None,
    output_format: OutputFormat = OutputFormat.TEXT,
    rules_file: os.PathLike[str] | str | None = None,
    only_rules: Sequence[str] | None = None,
    mapper: custom_lint_metadata.PathMapper | None = None,
    vcs: custom_lint_vcs.VcsMode | custom_lint_vcs.Vcs = (
        custom_lint_vcs.VcsMode.AUTO
    ),
    exclude_dirs: Iterable[str] | None = None,
    use_default_excludes: bool = True,
    config: LinterConfig | None = None,
    working_dir: pathlib.Path | None = None,
) -> int:
  """Runs `collect_violations`, writes output reports, and returns an exit code.

  See `collect_violations` for candidate selection modes and shared arguments.

  Args:
    root_dir: Project root directory (or a path inside it) to resolve and walk.
    files: Forwarded to `collect_violations`.
    base_sha: Forwarded to `collect_violations`.
    head_sha: Forwarded to `collect_violations`.
    check_all: Forwarded to `collect_violations`.
    jobs: Forwarded to `collect_violations`.
    report_file: Optional file path to write a grouped human-readable summary
      report.
    output_format: Output format written to stdout (`OutputFormat.TEXT` or
      `OutputFormat.JSON`).
    rules_file: Forwarded to `collect_violations`.
    only_rules: Forwarded to `collect_violations`.
    mapper: Forwarded to `collect_violations`.
    vcs: Forwarded to `collect_violations`.
    exclude_dirs: Forwarded to `collect_violations`.
    use_default_excludes: Forwarded to `collect_violations`.
    config: Forwarded to `collect_violations`.
    working_dir: Forwarded to `collect_violations`.

  Returns:
    0 if no `ERROR`-severity findings were reported (warnings are allowed),
    1 if one or more `ERROR`-severity findings were reported, or
    2 if rule loading, configuration, or file I/O failed (`VcsError` is raised
    on VCS failure and converted to exit code 2 by `main()`).

  Raises:
    VcsError: If diff computation or commit-message retrieval fails in VCS diff
      mode.
  """
  try:
    result = collect_violations(
        root_dir,
        files=files,
        base_sha=base_sha,
        head_sha=head_sha,
        check_all=check_all,
        jobs=jobs,
        rules_file=rules_file,
        only_rules=only_rules,
        mapper=mapper,
        vcs=vcs,
        exclude_dirs=exclude_dirs,
        use_default_excludes=use_default_excludes,
        config=config,
        working_dir=working_dir,
    )
  except (custom_lint_metadata.LoadError, ConfigError, OSError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    return 2

  if report_file:
    report_path = pathlib.Path(report_file)
    report_path.write_text(
        format_summary_report(result.violations, result.files_checked),
        encoding='utf-8',
    )
    if output_format != OutputFormat.JSON:
      print(f'Report written to {report_path} (view with: less {report_path})')

  if output_format == OutputFormat.JSON:
    write_json(result, sys.stdout)
    print(result.timing_summary(), file=sys.stderr)
  else:
    write_text_report(result, sys.stdout)

  return 1 if result.has_errors else 0


def _resolve_config_path(raw_path: str, cwd: pathlib.Path) -> pathlib.Path:
  """Resolves `--config` against `cwd` or the process working directory."""
  cfg_path = pathlib.Path(raw_path)
  if cfg_path.is_absolute():
    return cfg_path
  for base in (cwd, pathlib.Path.cwd()):
    if base.is_dir() and (base / cfg_path).is_file():
      return base / cfg_path
  return cfg_path


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description='Custom presubmit convention checker.'
  )
  parser.add_argument(
      'files',
      nargs='*',
      help='Optional specific files or directories to check.',
  )
  parser.add_argument(
      '--root',
      default=None,
      help='Optional root directory of the repository.',
  )
  parser.add_argument(
      '--config',
      default=None,
      dest='config_file',
      help='Optional path to pyproject.toml or custom_linter TOML config file.',
  )
  default_base = os.environ.get('BASE_SHA') or 'origin/main'
  parser.add_argument(
      '--base-sha',
      default=default_base,
      help=(
          'Base git SHA or tracking branch for incremental diffs (default:'
          ' $BASE_SHA if set, falling back to "origin/main" in git; resolved'
          ' via git merge-base against HEAD and compared against the working'
          ' tree to include uncommitted edits).'
      ),
  )
  default_head = os.environ.get('HEAD_SHA') or None
  parser.add_argument(
      '--head-sha',
      default=default_head,
      help=(
          'Head git SHA or ref for reading PR commit messages (default:'
          ' $HEAD_SHA if set, falling back to HEAD).'
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
          f' count when checking >{_PARALLEL_CANDIDATE_THRESHOLD} files).'
      ),
  )
  parser.add_argument(
      '-o',
      '--report-file',
      default=None,
      help=(
          'Optional path to write a grouped text summary report regardless of'
          ' --format (e.g. /tmp/custom_linter_report.txt).'
      ),
  )
  parser.add_argument(
      '--format',
      type=OutputFormat,
      choices=tuple(OutputFormat),
      default=OutputFormat.TEXT,
      dest='output_format',
      help='Stdout output format (text or json).',
  )
  parser.add_argument(
      '--rules',
      default=None,
      dest='rules_file',
      help=(
          f'METADATA-format rules file (default: {DEFAULT_RULES_FILE} or'
          ' from config).'
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
      type=custom_lint_vcs.VcsMode,
      choices=(
          custom_lint_vcs.VcsMode.AUTO,
          custom_lint_vcs.VcsMode.ANY,
          *custom_lint_vcs.VCS_TYPES,
      ),
      default=custom_lint_vcs.VcsMode.AUTO,
      help=(
          'VCS used for the diff and the commit message: detected from the'
          f' filesystem ({custom_lint_vcs.VcsMode.AUTO}), tried in order'
          f' ({custom_lint_vcs.VcsMode.ANY}), or a specific one.'
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
      action='store_false',
      default=True,
      dest='use_default_excludes',
      help='Do not apply the default directory ignore list when walking.',
  )
  args = parser.parse_args(argv)

  try:
    working_dir = os.environ.get('BUILD_WORKING_DIRECTORY')
    cwd = pathlib.Path(working_dir) if working_dir else pathlib.Path.cwd()
    if args.config_file:
      cfg_path = _resolve_config_path(args.config_file, cwd)
      cfg = load_config(cwd, cfg_path)
    else:
      cfg = None

    if args.root:
      root_path = pathlib.Path(args.root)
      root_dir = (cwd / root_path) if not root_path.is_absolute() else root_path
    else:
      root_dir = _default_project_root(cwd, config=cfg)
      if root_dir is None:
        if args.rules_file is None:
          expected_rules = cfg.rules_file if cfg else DEFAULT_RULES_FILE
          print(
              f'ERROR: could not find {expected_rules} in'
              f' {custom_lint_vcs.resolve_path(cwd)} or its parents (up to the'
              ' VCS root); pass --root <project_root>',
              file=sys.stderr,
          )
          return 2
        root_dir = _fallback_project_root(cwd)
    return run_checks(
        root_dir,
        files=args.files,
        base_sha=args.base_sha,
        head_sha=args.head_sha,
        check_all=args.check_all,
        jobs=args.jobs,
        report_file=args.report_file,
        output_format=args.output_format,
        rules_file=args.rules_file,
        only_rules=args.only_rules,
        vcs=args.vcs,
        exclude_dirs=args.exclude_dirs,
        use_default_excludes=args.use_default_excludes,
        config=cfg,
        working_dir=cwd,
    )
  except (
      ConfigError,
      custom_lint_metadata.LoadError,
      custom_lint_vcs.VcsError,
      OSError,
  ) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    return 2
  except Exception as e:  # pylint: disable=broad-exception-caught
    traceback.print_exc()
    print(
        f'ERROR: unexpected internal error ({type(e).__name__}): {e}',
        file=sys.stderr,
    )
    return 2


if __name__ == '__main__':
  sys.exit(main())
