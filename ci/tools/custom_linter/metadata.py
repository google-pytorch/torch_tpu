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

"""Loads custom lint rules from METADATA-style presubmit configuration.

Rules are read from a `METADATA` presubmit configuration file (for example,
`custom_lint_rules.METADATA`) using the `presubmit { ... }` text format.

Supported entries:

- `presubmit { check_contents { ... } }`: `prohibited_regexp`,
  `required_regexp`, `suppression_regexp`, `check_delta_only`,
  `match_entire_file`, `match_multi_line`, `ignore_large_files`, `operation`.
- `presubmit { check_file_operation { ... } }`: reports files changed by one of
  the listed `operation`s.

Other checks (`check_description`, `check_lint`, `include_presubmit`,
`check_contents` with `check_deleted_only`, ...) are recorded in
`RuleSet.skipped` and ignored during evaluation. Rule paths are matched against
canonical paths (e.g. `//depot/repo/proj/foo.py` or `my_project/foo.py`);
`PathMapper` converts project-relative paths to that form. Rules can be
suppressed per change via `disable_tags` in commit/CL descriptions or per line
via inline `suppression_regexp` comments.
"""

from collections.abc import Iterator, Mapping, Sequence, Set
import dataclasses
import enum
import functools
import os
import pathlib
import re
import string
from typing import Any, Final, Protocol, TypeAlias

import re2


class RegexPattern(Protocol):
  """Protocol matching both standard `re.Pattern` and `re2` compiled regexes."""

  def match(self, text: str) -> Any:
    ...

  def search(self, text: str) -> Any:
    ...

  def finditer(self, text: str) -> Iterator[Any]:
    ...


# ---------------------------------------------------------------------------
# Text-format parser.
# ---------------------------------------------------------------------------


class ParseError(ValueError):
  """A text-format syntax error with a 1-based source position."""

  def __init__(self, line: int, column: int, message: str) -> None:
    super().__init__(f'{line}:{column}: {message}')
    self.line = line
    self.column = column
    self.message = message


@dataclasses.dataclass(frozen=True)
class Ident:
  """An unquoted scalar token: identifier, enum value, boolean or number."""

  text: str


FieldValue: TypeAlias = 'str | Ident | Message'


@dataclasses.dataclass(frozen=True)
class Field:
  """A single `name: value` entry; repeated fields appear as multiple entries."""

  name: str
  value: FieldValue
  line: int


@dataclasses.dataclass(frozen=True)
class Message:
  """An ordered sequence of fields, as written in the source."""

  fields: Sequence[Field] = ()

  def fields_named(self, name: str) -> Iterator[Field]:
    return (f for f in self.fields if f.name == name)

  def messages(self, name: str) -> Iterator[tuple['Message', int]]:
    """Returns all nested messages named `name`, with their lines."""
    for f in self.fields_named(name):
      if isinstance(f.value, Message):
        yield f.value, f.line

  def message(self, name: str) -> 'Message | None':
    return next((m for m, _ in self.messages(name)), None)

  def strings(self, name: str) -> Sequence[str]:
    return [
        f.value for f in self.fields_named(name) if isinstance(f.value, str)
    ]

  def string(self, name: str) -> str | None:
    """Returns the last string value of `name` (proto "last one wins")."""
    values = self.strings(name)
    return values[-1] if values else None

  def idents(self, name: str) -> Sequence[str]:
    return [
        f.value.text
        for f in self.fields_named(name)
        if isinstance(f.value, Ident)
    ]

  def ident(self, name: str) -> str | None:
    values = self.idents(name)
    return values[-1] if values else None

  def get_bool(self, name: str) -> bool | None:
    """Returns the boolean value of `name`; raises ValueError if malformed."""
    value = self.ident(name)
    if value is None:
      return None
    if value in ('true', 'True', 't', '1'):
      return True
    if value in ('false', 'False', 'f', '0'):
      return False
    raise ValueError(f'field `{name}`: expected a boolean, got `{value}`')


_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?P<newline>\n)
    | (?P<space>[^\S\n]+)
    | (?P<comment>\#[^\n]*)
    | (?P<string>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')
    | (?P<punct>[{}\[\]<>:,;/])
    | (?P<ident>[A-Za-z0-9_\-+.]+)
    """,
    re.VERBOSE,
)

_SIMPLE_ESCAPES: Final[Mapping[str, bytes]] = {
    'n': b'\n',
    't': b'\t',
    'r': b'\r',
    'a': b'\x07',
    'b': b'\x08',
    'f': b'\x0c',
    'v': b'\x0b',
    '\\': b'\\',
    "'": b"'",
    '"': b'"',
    '?': b'?',
}


def _unescape(body: str, line: int, column: int) -> str:
  """Decodes C-style escapes in a quoted string body."""
  if '\\' not in body:
    return body
  out = bytearray()
  i, n = 0, len(body)
  while i < n:
    ch = body[i]
    i += 1
    if ch != '\\':
      out += ch.encode('utf-8')
      continue
    if i >= n:
      raise ParseError(line, column, 'trailing backslash')
    esc = body[i]
    i += 1
    if esc in _SIMPLE_ESCAPES:
      out += _SIMPLE_ESCAPES[esc]
    elif esc in string.octdigits:
      digits = esc
      while len(digits) < 3 and i < n and body[i] in string.octdigits:
        digits += body[i]
        i += 1
      value = int(digits, 8)
      if value > 0xFF:
        raise ParseError(line, column, 'octal escape out of range')
      out.append(value)
    elif esc == 'x':
      digits = ''
      while len(digits) < 2 and i < n and body[i] in string.hexdigits:
        digits += body[i]
        i += 1
      if not digits:
        raise ParseError(line, column, 'invalid \\x escape')
      out.append(int(digits, 16))
    else:
      raise ParseError(line, column, f'unknown escape `\\{esc}`')
  return out.decode('utf-8', errors='replace')


class _TokenKind(enum.StrEnum):
  """Token kinds emitted by `_tokenize`."""

  STR = 'str'
  IDENT = 'ident'
  PUNCT = 'punct'


@dataclasses.dataclass(frozen=True, slots=True)
class _Token:
  """A lexical token in a textproto document."""

  kind: _TokenKind
  value: str
  line: int
  column: int


def _tokenize(text: str) -> Sequence[_Token]:
  """Returns `_Token` instances for `text`."""
  tokens: list[_Token] = []
  pos, line, line_start = 0, 1, 0
  n = len(text)
  while pos < n:
    m = _TOKEN_RE.match(text, pos)
    column = pos - line_start + 1
    if m is None:
      ch = text[pos]
      if ch in '"\'':
        end = text.find('\n', pos)
        message = 'unterminated string' if end < 0 else 'newline in string'
        raise ParseError(line, column, message)
      raise ParseError(line, column, f'unexpected character `{ch}`')
    kind = m.lastgroup
    value = m.group()
    pos = m.end()
    if kind == 'newline':
      line += 1
      line_start = pos
    elif kind == 'string':
      decoded = _unescape(value[1:-1], line, column)
      # Adjacent string literals are concatenated.
      if tokens and tokens[-1].kind == _TokenKind.STR:
        prev = tokens[-1]
        tokens[-1] = _Token(
            _TokenKind.STR, prev.value + decoded, prev.line, prev.column
        )
      else:
        tokens.append(_Token(_TokenKind.STR, decoded, line, column))
    elif kind == 'punct':
      tokens.append(_Token(_TokenKind.PUNCT, value, line, column))
    elif kind == 'ident':
      tokens.append(_Token(_TokenKind.IDENT, value, line, column))
  return tokens


class _Parser:
  """Recursive-descent parser over the token sequence."""

  def __init__(self, tokens: Sequence[_Token]) -> None:
    self._tokens = tokens
    self._pos = 0

  def _error(self, message: str) -> ParseError:
    tok = (
        self._tokens[self._pos]
        if self._pos < len(self._tokens)
        else (self._tokens[-1] if self._tokens else None)
    )
    line, column = (tok.line, tok.column) if tok else (1, 1)
    return ParseError(line, column, message)

  def _peek_punct(self, ch: str) -> bool:
    if self._pos < len(self._tokens):
      tok = self._tokens[self._pos]
      return tok.kind == _TokenKind.PUNCT and tok.value == ch
    return False

  def _eat_punct(self, ch: str) -> bool:
    if self._peek_punct(ch):
      self._pos += 1
      return True
    return False

  def parse_fields(self, close: str | None) -> Message:
    fields: list[Field] = []
    while True:
      if self._pos >= len(self._tokens):
        if close is None:
          return Message(tuple(fields))
        raise self._error(f'expected `{close}`')
      if close is not None and self._eat_punct(close):
        return Message(tuple(fields))
      self._parse_field(fields)
      if not self._eat_punct(','):
        self._eat_punct(';')

  def _parse_field_name(self) -> tuple[str, int]:
    """Parses an identifier or bracketed extension field name."""
    if self._pos >= len(self._tokens):
      raise self._error('expected field name')
    tok = self._tokens[self._pos]
    self._pos += 1
    if tok.kind == _TokenKind.IDENT:
      return tok.value, tok.line
    if tok.kind == _TokenKind.PUNCT and tok.value == '[':
      # Extension or Any type URL: `[foo.bar]` or `[type.googleapis.com/x.Y]`.
      parts = ['[']
      while True:
        if self._pos >= len(self._tokens):
          raise self._error('malformed extension name')
        sub = self._tokens[self._pos]
        self._pos += 1
        if sub.kind == _TokenKind.IDENT or (
            sub.kind == _TokenKind.PUNCT and sub.value == '/'
        ):
          parts.append(sub.value)
        elif sub.kind == _TokenKind.PUNCT and sub.value == ']':
          break
        else:
          raise self._error('malformed extension name')
      parts.append(']')
      return ''.join(parts), tok.line
    raise ParseError(tok.line, tok.column, 'expected field name')

  def _parse_field(self, fields: list[Field]) -> None:
    """Parses one field or repeated list of values into `fields`."""
    name, line = self._parse_field_name()
    has_colon = self._eat_punct(':')
    if self._eat_punct('['):
      # List syntax: `name: [v1, v2]` or `name [{..}, {..}]`.
      if self._eat_punct(']'):
        return
      while True:
        fields.append(Field(name, self._parse_value(has_colon), line))
        if self._eat_punct(']'):
          return
        if not self._eat_punct(','):
          raise self._error('expected `,` or `]` in list')
    fields.append(Field(name, self._parse_value(has_colon), line))

  def _parse_value(self, has_colon: bool) -> FieldValue:
    """Parses a scalar or nested message field value."""
    if self._eat_punct('{'):
      return self.parse_fields('}')
    if self._eat_punct('<'):
      return self.parse_fields('>')
    if not has_colon:
      raise self._error('expected `:` before scalar value')
    if self._pos < len(self._tokens):
      tok = self._tokens[self._pos]
      if tok.kind == _TokenKind.STR:
        self._pos += 1
        return tok.value
      if tok.kind == _TokenKind.IDENT:
        self._pos += 1
        return Ident(tok.value)
    raise self._error('expected a value')


def parse_textproto(text: str) -> Message:
  """Parses a protobuf text-format document without a schema."""
  return _Parser(_tokenize(text)).parse_fields(None)


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------


def is_rooted(base_dir: str, path: str) -> bool:
  """Returns whether `path` is rooted rather than relative to `base_dir`.

  A rule path or regex is already rooted if it begins with a depot root (`//`),
  a wildcard prefix (`.../`), or with `base_dir` itself.

  When `base_dir` is a relative name (such as `"my_project"` after rewriting a
  monorepo prefix to `"my_project"`), any rule path starting with
  `"my_project/"` is treated as already rooted at `base_dir`.
  """
  base = base_dir.rstrip('/')
  if path.startswith('//') or path.startswith('.../'):
    return True
  if base and path.startswith(base):
    rest = path[len(base) :]
    # A rule rooted at `base_dir` may be followed by `/` (e.g. `my_project/src/...`)
    # or an unescaped regex quantifier/wildcard like `.*` (e.g. `my_project.*`).
    return not rest or rest[0] in '/.'
  return False


def resolve_rule_path(base_dir: str, path: str) -> str:
  """Prefixes relative rule paths with the directory that declares them."""
  if is_rooted(base_dir, path):
    return path
  return f'{base_dir.rstrip("/")}/{path}'


def path_expression_to_regex(expression: str) -> str:
  """Converts a depot path expression (`...` and `*` wildcards) to a regex."""
  out: list[str] = []
  i, n = 0, len(expression)
  while i < n:
    if expression.startswith('...', i):
      out.append('.*')
      i += 3
    elif expression[i] == '*':
      out.append('[^/]*')
      i += 1
    else:
      out.append(re.escape(expression[i]))
      i += 1
  return ''.join(out)


def anchor(regex: str) -> str:
  """Wraps a regex so that it must match the entire input."""
  return f'^(?:{regex})$'


@dataclasses.dataclass(frozen=True)
class PathMapper:
  """Maps project-relative paths to the canonical paths rules are written for.

  By default `a/b` maps to `<canonical_root>/a/b`. Prefix maps override this for
  subtrees whose layout differs (for example, mapping `src/my_project` to
  `my_project`). The longest matching prefix wins, and prefixes only match whole
  path components.
  """

  canonical_root: str
  prefix_maps: Sequence[tuple[str, str]] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, 'canonical_root', self.canonical_root.rstrip('/'))
    maps = [(r.strip('/'), c.rstrip('/')) for r, c in self.prefix_maps]
    maps.sort(key=lambda m: len(m[0]), reverse=True)
    object.__setattr__(self, 'prefix_maps', tuple(maps))

  def to_canonical(self, rel_path: str) -> str:
    """Converts a project-relative POSIX path to its canonical form."""
    for prefix, canonical in self.prefix_maps:
      if not prefix:
        return f'{canonical}/{rel_path}'
      if rel_path == prefix:
        return canonical
      if rel_path.startswith(prefix) and rel_path[len(prefix)] == '/':
        return f'{canonical}{rel_path[len(prefix):]}'
    return f'{self.canonical_root}/{rel_path}'


# ---------------------------------------------------------------------------
# Change-description tags.
# ---------------------------------------------------------------------------

_TAG_KEY_RE: Final[re.Pattern[str]] = re.compile(r'[A-Z][A-Z0-9_]*')


def parse_tags(description: str) -> Set[str]:
  """Returns the tag keys of `KEY=value` lines in a change description.

  `KEY` must be an upper-case identifier at the start of the line (leading
  whitespace allowed). Mentions in prose do not count.

  Args:
    description: Commit or changelist description text.

  Returns:
    Set of parsed upper-case tag keys.
  """
  keys: set[str] = set()
  for line in description.splitlines():
    key, sep, _ = line.strip().partition('=')
    key = key.rstrip()
    if sep and _TAG_KEY_RE.fullmatch(key):
      keys.add(key)
  return frozenset(keys)


# ---------------------------------------------------------------------------
# Regex helpers.
# ---------------------------------------------------------------------------


_RE2_OPTIONS: Final[Any] = re2.Options()
_RE2_OPTIONS.log_errors = False
_RE2_OPTIONS.never_capture = True


@functools.cache
def _compile_re(pattern: str, multiline: bool = False) -> RegexPattern:
  """Compiles and caches an RE2 regular expression pattern."""
  if multiline:
    pattern = f'(?m:{pattern})'
  return re2.compile(pattern, options=_RE2_OPTIONS)


_TEXT_ANCHOR_RE: Final[re.Pattern[str]] = re.compile(r'(?<!\\)(?:\\\\)*\\[Az]')


@functools.cache
def is_line_local(pattern: str) -> bool:
  r"""Returns whether whole-file MULTILINE matching finds all per-line matches.

  If a valid RE2 pattern matches some line, the same pattern compiled with
  `(?m:...)` matches the file text overlapping that line, unless it uses
  `\A` or `\z` (which only match at the start/end of the whole text).
  """
  return not _TEXT_ANCHOR_RE.search(pattern)


# ---------------------------------------------------------------------------
# Rule model.
# ---------------------------------------------------------------------------


class CheckKind(enum.StrEnum):
  """Kinds of checks supported by `Rule`."""

  CHECK_CONTENTS = 'check_contents'
  CHECK_FILE_OPERATION = 'check_file_operation'


class MatchMode(enum.StrEnum):
  """Match modes of `check_contents` rules.

  - LINE: each line separately; suppressions on the same line.
  - ENTIRE_FILE: whole file; suppression anywhere.
  - MODIFIED_CHUNKS: runs of consecutive changed lines; each match is
    suppressed within the chunk segment from its start line through the line
    before the next match (or the end of the chunk).
  """

  LINE = 'line'
  ENTIRE_FILE = 'entire-file'
  MODIFIED_CHUNKS = 'modified-chunks'


class Severity(enum.StrEnum):
  """`failure_status` values of the METADATA presubmit schema."""

  ERROR = 'ERROR'
  WARNING = 'WARNING'
  NOTICE = 'NOTICE'


class FileOperation(enum.StrEnum):
  """`FileOperation` values of the METADATA presubmit schema."""

  ADD = 'ADD'
  EDIT = 'EDIT'
  DELETE = 'DELETE'
  BRANCH = 'BRANCH'
  INTEGRATE = 'INTEGRATE'


# Files larger than this are not matched in entire-file mode.
MAX_ENTIRE_FILE_BYTES: Final[int] = 1 << 20


@dataclasses.dataclass(frozen=True)
class Rule:
  """A single check loaded from a `presubmit` block.

  Attributes:
    id: The check's `base.id`.
    path_regex: Anchored regex over canonical paths selecting files.
    prohibited_regex: Pattern that must not appear.
    failure_message: Message shown for each violation.
    suppression_regex: Pattern that suppresses a `prohibited_regex` match.
    path_exclusion_regex: Anchored regex over canonical paths excluding files.
    check_delta_only: Only check lines changed by the change under review.
    disable_tags: Description tags (`TAG=...`) that disable this rule.
    required_regex: Pattern that must appear at least once in the file.
    mode: `MatchMode` controlling how lines/chunks/files are matched.
    operations: File operations the rule applies to; empty means all.
    kind: `CheckKind` (`CHECK_CONTENTS` or `CHECK_FILE_OPERATION`).
    severity: `Severity` (`ERROR`, `WARNING`, or `NOTICE`).
    ignore_large_files: Skip (instead of failing on) files too large for
      entire-file matching.
    origin: `file:line` where the rule is defined.
  """

  id: str
  path_regex: str
  prohibited_regex: str | None = None
  failure_message: str = ''
  suppression_regex: str | None = None
  path_exclusion_regex: str | None = None
  check_delta_only: bool = False
  disable_tags: Set[str] = frozenset()
  required_regex: str | None = None
  mode: MatchMode = MatchMode.LINE
  operations: Set[FileOperation] = frozenset()
  kind: CheckKind = CheckKind.CHECK_CONTENTS
  severity: Severity = Severity.ERROR
  ignore_large_files: bool = False
  origin: str = ''

  def __post_init__(self) -> None:
    object.__setattr__(self, 'disable_tags', frozenset(self.disable_tags))
    object.__setattr__(self, 'operations', frozenset(self.operations))
    if not self.failure_message:
      object.__setattr__(self, 'failure_message', f'Check {self.id} failed.')

  def prewarm(self) -> None:
    """Pre-warms cached properties before multi-worker execution."""
    _ = self.path_re
    _ = self.path_exclusion_re
    _ = self.prohibited_re
    _ = self.prohibited_multiline_re
    _ = self.prohibited_is_line_local
    _ = self.suppression_re
    _ = self.required_re
    _ = self.required_multiline_re
    _ = self.required_is_line_local

  @functools.cached_property
  def path_re(self) -> RegexPattern:
    """Compiled `path_regex`."""
    return _compile_re(self.path_regex)

  @functools.cached_property
  def path_exclusion_re(self) -> RegexPattern | None:
    """Compiled `path_exclusion_regex`, if set."""
    if not self.path_exclusion_regex:
      return None
    return _compile_re(self.path_exclusion_regex)

  @functools.cached_property
  def prohibited_re(self) -> RegexPattern | None:
    """Compiled `prohibited_regex`, if set."""
    if self.prohibited_regex is None:
      return None
    return _compile_re(self.prohibited_regex)

  @functools.cached_property
  def prohibited_multiline_re(self) -> RegexPattern | None:
    """`prohibited_re` with `^`/`$` matching at line boundaries."""
    if self.prohibited_regex is None:
      return None
    return _compile_re(self.prohibited_regex, multiline=True)

  @functools.cached_property
  def prohibited_is_line_local(self) -> bool:
    """Whether `prohibited_regex` can be evaluated via whole-file MULTILINE."""
    return self.prohibited_regex is not None and is_line_local(
        self.prohibited_regex
    )

  @functools.cached_property
  def suppression_re(self) -> RegexPattern | None:
    """Compiled `suppression_regex`, if set."""
    if self.suppression_regex is None:
      return None
    return _compile_re(self.suppression_regex)

  @functools.cached_property
  def required_re(self) -> RegexPattern | None:
    """Compiled `required_regex`, if set."""
    if self.required_regex is None:
      return None
    return _compile_re(self.required_regex)

  @functools.cached_property
  def required_multiline_re(self) -> RegexPattern | None:
    """`required_re` with `^`/`$` matching at line boundaries."""
    if self.required_regex is None:
      return None
    return _compile_re(self.required_regex, multiline=True)

  @functools.cached_property
  def required_is_line_local(self) -> bool:
    """Whether `required_regex` can be evaluated via whole-file MULTILINE."""
    return self.required_regex is not None and is_line_local(
        self.required_regex
    )

  def matches_path(self, canonical_path: str) -> bool:
    """Returns True if the rule applies to `canonical_path`."""
    if not self.path_re.match(canonical_path):
      return False
    excl = self.path_exclusion_re
    return excl is None or not excl.match(canonical_path)

  def applies_to(self, operation: FileOperation | None) -> bool:
    """Returns whether the rule applies to a file changed by `operation`.

    `None` means there is no change context (explicit files, `--all`). Rules
    restricted to specific operations are about changes, so they do not apply.
    `check_contents` rules never apply to deleted files.

    Args:
      operation: A `FileOperation`, or None.
    """
    if (
        self.kind == CheckKind.CHECK_CONTENTS
        and operation == FileOperation.DELETE
    ):
      return False
    if not self.operations:
      return True
    return operation is not None and operation in self.operations


@dataclasses.dataclass(frozen=True)
class Skipped:
  """A check found in a rule file that this tool does not implement."""

  kind: str
  id: str | None
  origin: str


@dataclasses.dataclass(frozen=True)
class RuleSet:
  """The result of loading a rule file."""

  rules: Sequence[Rule] = ()
  skipped: Sequence[Skipped] = ()


class LoadError(ValueError):
  """An error loading a rule file, with its location when known."""

  def __init__(self, file: str, line: int | None, message: str) -> None:
    location = f'{file}:{line}' if line is not None else file
    super().__init__(f'{location}: {message}')
    self.file = file
    self.line = line
    self.message = message


# ---------------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------------


def _check_regex(field: str, pattern: str) -> str:
  try:
    _compile_re(pattern)
  except re2.error as e:
    raise ValueError(f'invalid regex in `{field}` ({pattern!r}): {e}') from e
  return pattern


def _join_alternatives(regexes: Sequence[str]) -> str | None:
  if not regexes:
    return None
  if len(regexes) == 1:
    return regexes[0]
  return '|'.join(regexes)


def _build_path_filter(
    presubmit: Message, base_dir: str
) -> tuple[str, str | None]:
  """Returns (include, exclude) anchored regexes over canonical paths."""

  def expression(field: str, expr: str) -> str:
    return _check_regex(
        field,
        anchor(path_expression_to_regex(resolve_rule_path(base_dir, expr))),
    )

  def regexp(field: str, pattern: str) -> str:
    if not is_rooted(base_dir, pattern):
      pattern = f'{re.escape(base_dir.rstrip("/"))}/{pattern}'
    return _check_regex(field, anchor(pattern))

  include = [
      expression('path_expression', e)
      for e in presubmit.strings('path_expression')
  ]
  include += [
      regexp('path_regexp', r) for r in presubmit.strings('path_regexp')
  ]
  if not include:
    include.append(expression('path_expression', '...'))
  exclude = [
      expression('path_expression_exclusion', e)
      for e in presubmit.strings('path_expression_exclusion')
  ]
  exclude += [
      regexp('path_regexp_exclusion', r)
      for r in presubmit.strings('path_regexp_exclusion')
  ]
  include_regex = _join_alternatives(include)
  assert include_regex is not None
  return include_regex, _join_alternatives(exclude)


def _operations(message: Message, name: str) -> Set[FileOperation]:
  out: list[FileOperation] = []
  for op in message.idents(name):
    try:
      out.append(FileOperation(op))
    except ValueError as e:
      raise ValueError(f'unknown `{name}` `{op}`') from e
  return frozenset(out)


_DEFAULT_RULE_NAMES: Final[Mapping[CheckKind, str]] = {
    CheckKind.CHECK_CONTENTS: 'CheckContents',
    CheckKind.CHECK_FILE_OPERATION: 'CheckFileOperation',
}


def _build_rule(
    kind: CheckKind,
    check: Message,
    paths: tuple[str, str | None],
    block_ops: Set[FileOperation],
    origin: str,
    line: int,
) -> Rule | Skipped:
  """Builds a Rule (or Skipped entry) from a check message."""
  base = check.message('base')
  rule_id = (
      base.string('id') if base else None
  ) or f'{_DEFAULT_RULE_NAMES[kind]}@{line}'

  def flag(name: str) -> bool:
    return bool(check.get_bool(name))

  if kind == CheckKind.CHECK_CONTENTS and flag('check_deleted_only'):
    return Skipped(f'{kind}.check_deleted_only', rule_id, origin)

  raw_status = check.ident('failure_status') or Severity.ERROR.value
  try:
    status = Severity(raw_status)
  except ValueError as e:
    raise ValueError(f'unknown `failure_status` `{raw_status}`') from e
  check_ops = _operations(check, 'operation')
  failure_message = check.string('failure_message') or ''
  disable_tags = (
      frozenset(base.strings('disable_tags')) if base else frozenset()
  )
  operations = check_ops or block_ops
  if kind == CheckKind.CHECK_FILE_OPERATION:
    return Rule(
        id=rule_id,
        path_regex=paths[0],
        path_exclusion_regex=paths[1],
        failure_message=failure_message,
        severity=status,
        disable_tags=disable_tags,
        operations=operations,
        origin=origin,
        kind=CheckKind.CHECK_FILE_OPERATION,
    )

  def optional_regex(name: str) -> str | None:
    pattern = check.string(name)
    return None if pattern is None else _check_regex(name, pattern)

  prohibited = optional_regex('prohibited_regexp')
  required = optional_regex('required_regexp')
  suppression = optional_regex('suppression_regexp')
  if prohibited is None and required is None:
    raise ValueError(
        'check_contents needs `prohibited_regexp` or `required_regexp`'
    )
  delta_only = flag('check_delta_only')
  entire_file = flag('match_entire_file')
  if delta_only and entire_file:
    raise ValueError(
        '`check_delta_only` is incompatible with `match_entire_file`'
    )
  if entire_file:
    mode = MatchMode.ENTIRE_FILE
  elif delta_only and flag('match_multi_line'):
    mode = MatchMode.MODIFIED_CHUNKS
  else:
    # `match_multi_line` is documented to apply only with `check_delta_only`.
    mode = MatchMode.LINE
  return Rule(
      id=rule_id,
      path_regex=paths[0],
      path_exclusion_regex=paths[1],
      failure_message=failure_message,
      severity=status,
      disable_tags=disable_tags,
      operations=operations,
      origin=origin,
      kind=CheckKind.CHECK_CONTENTS,
      prohibited_regex=prohibited,
      required_regex=required,
      suppression_regex=suppression,
      check_delta_only=delta_only,
      mode=mode,
      ignore_large_files=(
          flag('ignore_large_files')
          or flag('match_entire_file_ignore_large_files')
      ),
  )


def parse_metadata(text: str, file: str, base_dir: str) -> RuleSet:
  """Parses METADATA `text`; relative rule paths resolve against `base_dir`."""
  try:
    root = parse_textproto(text)
  except ParseError as e:
    raise LoadError(
        file, e.line, f'syntax error at column {e.column}: {e.message}'
    ) from e

  rules: list[Rule] = []
  skipped: list[Skipped] = []
  seen_ids: set[str] = set()
  for presubmit, presubmit_line in root.messages('presubmit'):
    # Path filters are built lazily: blocks without supported checks may use
    # patterns that are never evaluated (e.g. RE2-only syntax).
    paths: tuple[str, str | None] | None = None
    block_ops: Set[FileOperation] = frozenset()
    for field in presubmit.fields:
      origin = f'{file}:{field.line}'
      value = field.value
      if isinstance(value, Message) and field.name in (
          CheckKind.CHECK_CONTENTS,
          CheckKind.CHECK_FILE_OPERATION,
      ):
        check_kind = CheckKind(field.name)
        if paths is None:
          try:
            paths = _build_path_filter(presubmit, base_dir)
            block_ops = _operations(presubmit, 'file_operation')
          except ValueError as e:
            raise LoadError(file, presubmit_line, str(e)) from e
        try:
          built = _build_rule(
              check_kind, value, paths, block_ops, origin, field.line
          )
        except ValueError as e:
          base = value.message('base')
          rule_id = (
              base.string('id') if base else None
          ) or _DEFAULT_RULE_NAMES[check_kind]
          raise LoadError(file, field.line, f'{rule_id}: {e}') from e
        if isinstance(built, Skipped):
          skipped.append(built)
        elif built.id in seen_ids:
          raise LoadError(file, field.line, f'duplicate rule id `{built.id}`')
        else:
          seen_ids.add(built.id)
          rules.append(built)
      elif isinstance(value, Message) and field.name.startswith('check_'):
        base = value.message('base')
        skipped.append(
            Skipped(field.name, base.string('id') if base else None, origin)
        )
      elif isinstance(value, str) and field.name == 'include_presubmit':
        skipped.append(Skipped('include_presubmit', value, origin))
  return RuleSet(rules=tuple(rules), skipped=tuple(skipped))


def load_rules_file(path: os.PathLike[str] | str, base_dir: str) -> RuleSet:
  """Reads and parses a METADATA rule file."""
  file_path = pathlib.Path(path)
  try:
    text = file_path.read_text(encoding='utf-8')
  except (OSError, UnicodeDecodeError) as e:
    raise LoadError(str(file_path), None, f'cannot read rules file: {e}') from e
  return parse_metadata(text, str(file_path), base_dir)
