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

The rules live in `custom_lint_rules.METADATA` (exported to OSS under the same
name as `ci/custom_lint_rules.METADATA`) and use the `presubmit { ... }` text
format of the METADATA presubmit schema, so the same file drives both the
monorepo presubmit and this linter.

Supported entries:

- `presubmit { check_contents { ... } }`: `prohibited_regexp`,
  `required_regexp`, `suppression_regexp`, `check_delta_only`,
  `match_entire_file`, `match_multi_line`, `ignore_large_files`, `operation`.
- `presubmit { check_file_operation { ... } }`: reports files changed by one of
  the listed `operation`s.

Other checks (`check_description`, `check_lint`, `include_presubmit`, ...) are
reported as skipped. Rule paths are matched against canonical depot-style paths
(e.g. `//depot/repo/proj/foo.py` or `torch_tpu/foo.py`); `PathMapper` converts
project-relative paths to that form. Note that `disable_tags` in commit
descriptions only work in OSS CI for commits authored on GitHub
(Copybara-exported PRs only carry public commit text); inline
`suppression_regexp` comments are the portable suppression mechanism.
"""

from collections.abc import Iterable, Iterator, Sequence, Set
import dataclasses
import enum
import functools
import pathlib
import re
import string
from typing import Any, Final

# CPython 3.11+ exposes the regex parser and opcode constants as `re._parser`
# and `re._constants`. We use them to inspect parsed regex ASTs for two optional
# fast paths:
#   1. `_extract_required_literals`: extracts literal substrings that must
#      appear in a file before running `prohibited_re`.
#   2. `is_line_local`: checks whether a line-mode regex can be pre-filtered
#      with a single whole-file `re.MULTILINE` scan instead of matching every
#      line individually.
# Because these modules are CPython-internal (and were named `sre_parse` /
# `sre_constants` prior to Python 3.11), they may be absent on non-CPython
# runtimes or future Python releases; when unavailable, both helpers fall back
# to `()` / `False` and the engine evaluates regexes directly on each line.
try:
  import re._constants as _sre_constants  # type: ignore[import-not-found]  # pylint: disable=protected-access,g-import-not-at-top
  import re._parser as _sre_parse  # type: ignore[import-not-found]  # pylint: disable=protected-access,g-import-not-at-top
except ImportError:  # pragma: no cover
  _sre_constants: Any = None  # type: ignore[no-redef]
  _sre_parse: Any = None  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Text-format parser.
# ---------------------------------------------------------------------------


class ParseError(ValueError):
  """A text-format syntax error with a 1-based source position."""

  def __init__(self, line: int, column: int, message: str):
    super().__init__(f'{line}:{column}: {message}')
    self.line = line
    self.column = column
    self.message = message


@dataclasses.dataclass(frozen=True)
class Ident:
  """An unquoted scalar token: identifier, enum value, boolean or number."""

  text: str


@dataclasses.dataclass(frozen=True)
class Field:
  """A single `name: value` entry; repeated fields appear as multiple entries."""

  name: str
  value: 'str | Ident | Message'
  line: int


@dataclasses.dataclass
class Message:
  """An ordered list of fields, as written in the source."""

  fields: list[Field] = dataclasses.field(default_factory=list)

  def fields_named(self, name: str) -> Iterator[Field]:
    return (f for f in self.fields if f.name == name)

  def messages(self, name: str) -> Iterator[tuple['Message', int]]:
    """Returns all nested messages named `name`, with their lines."""
    for f in self.fields_named(name):
      if isinstance(f.value, Message):
        yield f.value, f.line

  def message(self, name: str) -> 'Message | None':
    return next((m for m, _ in self.messages(name)), None)

  def strings(self, name: str) -> list[str]:
    return [
        f.value for f in self.fields_named(name) if isinstance(f.value, str)
    ]

  def string(self, name: str) -> str | None:
    """Returns the last string value of `name` (proto "last one wins")."""
    values = self.strings(name)
    return values[-1] if values else None

  def idents(self, name: str) -> list[str]:
    return [
        f.value.text
        for f in self.fields_named(name)
        if isinstance(f.value, Ident)
    ]

  def ident(self, name: str) -> str | None:
    values = self.idents(name)
    return values[-1] if values else None

  def bool(self, name: str) -> bool | None:
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

_SIMPLE_ESCAPES: Final[dict[str, bytes]] = {
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


class _TokenKind(enum.Enum):
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

  def __init__(self, tokens: Sequence[_Token]):
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
    message = Message()
    while True:
      if self._pos >= len(self._tokens):
        if close is None:
          return message
        raise self._error(f'expected `{close}`')
      if close is not None and self._eat_punct(close):
        return message
      self._parse_field(message)
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

  def _parse_field(self, message: Message) -> None:
    """Parses one field or repeated list of values into `message`."""
    name, line = self._parse_field_name()
    has_colon = self._eat_punct(':')
    if self._eat_punct('['):
      # List syntax: `name: [v1, v2]` or `name [{..}, {..}]`.
      if self._eat_punct(']'):
        return
      while True:
        message.fields.append(Field(name, self._parse_value(has_colon), line))
        if self._eat_punct(']'):
          return
        if not self._eat_punct(','):
          raise self._error('expected `,` or `]` in list')
    message.fields.append(Field(name, self._parse_value(has_colon), line))

  def _parse_value(self, has_colon: bool) -> 'str | Ident | Message':
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
  """Returns whether `path` is rooted rather than relative to `base_dir`."""
  base = base_dir.rstrip('/')
  if path.startswith('//') or path.startswith('.../'):
    return True
  if base and path.startswith(base):
    rest = path[len(base) :]
    return not rest or rest[0] in '/.'
  return False


def resolve_path(base_dir: str, path: str) -> str:
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


class PathMapper:
  """Maps project-relative paths to the canonical paths rules are written for.

  By default `a/b` maps to `<canonical_root>/a/b`. Prefix maps override this for
  subtrees whose layout differs, e.g. the OSS export where Copybara moves the
  Python package to `src/torch_tpu/`. The longest matching prefix wins, and
  prefixes only match whole path components.
  """

  def __init__(
      self,
      canonical_root: str,
      prefix_maps: Iterable[tuple[str, str]] = (),
  ):
    self.canonical_root = canonical_root.rstrip('/')
    maps = [(r.strip('/'), c.rstrip('/')) for r, c in prefix_maps]
    maps.sort(key=lambda m: len(m[0]), reverse=True)
    self.prefix_maps: tuple[tuple[str, str], ...] = tuple(maps)

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

  def __repr__(self) -> str:
    return f'PathMapper({self.canonical_root!r}, {list(self.prefix_maps)!r})'


# ---------------------------------------------------------------------------
# Change-description tags.
# ---------------------------------------------------------------------------

_TAG_KEY_RE: Final[re.Pattern[str]] = re.compile(r'[A-Z][A-Z0-9_]*')


def parse_tags(description: str) -> frozenset[str]:
  """Returns the tag keys of `KEY=value` lines in a change description.

  `KEY` must be an upper-case identifier at the start of the line (leading
  whitespace allowed). Mentions in prose do not count.

  Args:
    description: Commit or changelist description text.

  Returns:
    Frozenset of parsed upper-case tag keys.
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


@functools.lru_cache(maxsize=None)
def _compile_re(pattern: str, flags: int = 0) -> re.Pattern[str]:
  """Compiles and caches a regular expression pattern with RE2-like ASCII semantics."""
  return re.compile(pattern, flags | re.ASCII)


_RE2_END_OF_TEXT: Final[re.Pattern[str]] = re.compile(r'(?<!\\)((?:\\\\)*)\\z')


def re2_to_python(pattern: str) -> str:
  """Translates the RE2-only syntax used in METADATA files to Python `re`."""
  if '\\z' in pattern:
    pattern = _RE2_END_OF_TEXT.sub(r'\1\\Z', pattern)
  return pattern


def _simplify_literal_set(literals: set[str]) -> set[str]:
  """Simplifies a disjunctive literal set by dropping strings that contain shorter members."""
  items = sorted(literals, key=len)
  kept: list[str] = []
  for item in items:
    if not any(existing in item for existing in kept):
      kept.append(item)
  return set(kept)


def _exact_branch_strings(
    parsed: Any, max_branches: int = 80
) -> set[str] | None:
  """Returns exact literal strings matched by a regex AST node if finite and small."""
  cur: set[str] = {''}
  for op, av in parsed:
    if op == _sre_constants.LITERAL:
      ch = chr(av)
      cur = {prefix + ch for prefix in cur}
    elif op == _sre_constants.AT:
      continue
    elif op == _sre_constants.SUBPATTERN:
      _, add_flags, _, sub = av
      if add_flags & re.IGNORECASE:
        return None
      sub_s = _exact_branch_strings(sub, max_branches)
      if sub_s is None:
        return None
      cur = {p + s for p in cur for s in sub_s}  # pylint: disable=g-complex-comprehension
      if len(cur) > max_branches:
        return None
    elif op == _sre_constants.BRANCH:
      _, branches = av
      b_union: set[str] = set()
      for b in branches:
        sub_s = _exact_branch_strings(b, max_branches)
        if sub_s is None:
          return None
        b_union |= sub_s
        if len(b_union) > max_branches:
          return None
      cur = {p + s for p in cur for s in b_union}  # pylint: disable=g-complex-comprehension
      if len(cur) > max_branches:
        return None
    else:
      return None
  return cur


def _extract_clauses_from_ast(parsed: Any) -> list[tuple[str, ...]]:
  """Extracts conjunctive clauses of disjunctive literal substrings from a regex AST."""
  clauses: list[set[str]] = []
  n = len(parsed)
  i = 0
  while i < n:
    j = i
    cur_exact: set[str] = {''}
    while j < n:
      op, av = parsed[j]
      if op == _sre_constants.LITERAL:
        ch = chr(av)
        cur_exact = {p + ch for p in cur_exact}
        j += 1
      elif op in (_sre_constants.SUBPATTERN, _sre_constants.BRANCH):
        sub_s = _exact_branch_strings(parsed[j : j + 1], max_branches=80)
        if sub_s is not None:
          nc = {p + s for p in cur_exact for s in sub_s}  # pylint: disable=g-complex-comprehension
          if len(nc) <= 80:
            cur_exact = nc
            j += 1
            continue
        break
      else:
        break
    if j > i:
      valid = {s for s in cur_exact if len(s) >= 2}
      if len(valid) == len(cur_exact):
        simp = _simplify_literal_set(valid)
        if len(simp) <= 12:
          clauses.append(simp)
          i = j
          continue
    op, av = parsed[i]
    if op == _sre_constants.SUBPATTERN:
      _, add_flags, _, sub = av
      if not (add_flags & re.IGNORECASE):
        for c in _extract_clauses_from_ast(sub):
          clauses.append(set(c))
    elif op == _sre_constants.BRANCH:
      _, branches = av
      b_sets: list[set[str]] = []
      ok = True
      for b in branches:
        sub_c = _extract_clauses_from_ast(b)
        if not sub_c:
          ok = False
          break
        best_b = min(sub_c, key=lambda c: (len(c), -min(len(x) for x in c)))
        b_sets.append(set(best_b))
      if ok and b_sets:
        u = _simplify_literal_set(set().union(*b_sets))
        if len(u) <= 12:
          clauses.append(u)
    i += 1
  unique_clauses: list[tuple[str, ...]] = []
  seen: set[tuple[str, ...]] = set()
  for c in clauses:
    t = tuple(sorted(c))
    if t not in seen:
      seen.add(t)
      unique_clauses.append(t)
  unique_clauses.sort(key=lambda t: (len(t), -min(len(x) for x in t)))
  return unique_clauses


@functools.lru_cache(maxsize=None)
def _extract_required_literals(
    pattern: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
  """Returns (singletons, disjunctions) that must appear in text for pattern to match."""
  if _sre_parse is None or _sre_constants is None:
    return (), ()
  try:
    parsed = _sre_parse.parse(pattern)
  except re.error:
    return (), ()
  if parsed.state.flags & re.IGNORECASE:
    return (), ()
  clauses = _extract_clauses_from_ast(parsed)
  singletons = tuple(c[0] for c in clauses if len(c) == 1)
  disjunctions = tuple(c for c in clauses if len(c) > 1)
  return singletons, disjunctions


def _walk_ast(parsed: Any) -> Iterator[tuple[Any, Any]]:
  """Yields every (op, arg) node of a parsed regex, recursively."""
  for op, av in parsed:
    yield op, av
    if op == _sre_constants.SUBPATTERN:
      yield from _walk_ast(av[3])
    elif op == _sre_constants.BRANCH:
      for b in av[1]:
        yield from _walk_ast(b)
    elif op in (
        _sre_constants.MAX_REPEAT,
        _sre_constants.MIN_REPEAT,
        getattr(_sre_constants, 'POSSESSIVE_REPEAT', None),
    ):
      yield from _walk_ast(av[2])
    elif op in (_sre_constants.ASSERT, _sre_constants.ASSERT_NOT):
      yield from _walk_ast(av[1])
    elif op == getattr(_sre_constants, 'ATOMIC_GROUP', None):
      yield from _walk_ast(av)
    elif op == _sre_constants.GROUPREF_EXISTS:
      yield from _walk_ast(av[1])
      if av[2] is not None:
        yield from _walk_ast(av[2])


@functools.lru_cache(maxsize=None)
def is_line_local(pattern: str) -> bool:
  r"""Returns whether whole-file MULTILINE matching finds all per-line matches.

  If a pattern matches some line, the same pattern compiled with re.MULTILINE
  matches the file text overlapping that line, unless it depends on what lies
  beyond the line: lookarounds, `\A`/`\Z` (which only match at the start/end of
  the whole text), or possessive quantifiers / atomic groups (which can
  possessively consume `\n` without backtracking to `$`).
  """
  if _sre_parse is None or _sre_constants is None:
    return False
  try:
    parsed = _sre_parse.parse(pattern)
  except re.error:
    return False
  text_anchors = (
      _sre_constants.AT_BEGINNING_STRING,
      _sre_constants.AT_END_STRING,
  )
  non_backtracking_ops = (
      getattr(_sre_constants, 'POSSESSIVE_REPEAT', None),
      getattr(_sre_constants, 'ATOMIC_GROUP', None),
  )
  for op, av in _walk_ast(parsed):
    if op in (_sre_constants.ASSERT, _sre_constants.ASSERT_NOT):
      return False
    if op == _sre_constants.AT and av in text_anchors:
      return False
    if op == _sre_constants.GROUPREF_EXISTS:
      return False
    if op is not None and op in non_backtracking_ops:
      return False
  return True


# ---------------------------------------------------------------------------
# Rule model.
# ---------------------------------------------------------------------------


class CheckKind(enum.StrEnum):
  """Kinds of checks supported by `Rule`."""

  CHECK_CONTENTS = 'check_contents'
  CHECK_FILE_OPERATION = 'check_file_operation'


CHECK_CONTENTS: Final[CheckKind] = CheckKind.CHECK_CONTENTS
CHECK_FILE_OPERATION: Final[CheckKind] = CheckKind.CHECK_FILE_OPERATION


class MatchMode(enum.StrEnum):
  """Match modes of `check_contents` rules.

  - LINE: each line separately; suppressions on the same line.
  - ENTIRE_FILE: whole file; suppression anywhere.
  - MODIFIED_CHUNKS: runs of consecutive changed lines.
  """

  LINE = 'line'
  ENTIRE_FILE = 'entire-file'
  MODIFIED_CHUNKS = 'modified-chunks'


MODE_LINE: Final[MatchMode] = MatchMode.LINE
MODE_ENTIRE_FILE: Final[MatchMode] = MatchMode.ENTIRE_FILE
MODE_MODIFIED_CHUNKS: Final[MatchMode] = MatchMode.MODIFIED_CHUNKS


class Severity(enum.StrEnum):
  """`failure_status` values of the METADATA presubmit schema."""

  ERROR = 'ERROR'
  WARNING = 'WARNING'
  NOTICE = 'NOTICE'


SEVERITIES: Final[tuple[Severity, ...]] = tuple(Severity)


class FileOperation(enum.StrEnum):
  """`FileOperation` values of the METADATA presubmit schema."""

  ADD = 'ADD'
  EDIT = 'EDIT'
  DELETE = 'DELETE'
  BRANCH = 'BRANCH'
  INTEGRATE = 'INTEGRATE'


FILE_OPERATIONS: Final[tuple[FileOperation, ...]] = tuple(FileOperation)

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
  operations: tuple[FileOperation, ...] = ()
  kind: CheckKind = CheckKind.CHECK_CONTENTS
  severity: Severity = Severity.ERROR
  ignore_large_files: bool = False
  origin: str = ''

  def __post_init__(self):
    object.__setattr__(self, 'disable_tags', frozenset(self.disable_tags))
    object.__setattr__(
        self, 'operations', tuple(FileOperation(op) for op in self.operations)
    )
    object.__setattr__(self, 'mode', MatchMode(self.mode))
    object.__setattr__(self, 'kind', CheckKind(self.kind))
    object.__setattr__(self, 'severity', Severity(self.severity))
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
    _ = self._prohibited_singletons
    _ = self._prohibited_disjunctions

  @functools.cached_property
  def path_re(self) -> re.Pattern[str]:
    """Compiled `path_regex`."""
    return _compile_re(self.path_regex)

  @functools.cached_property
  def path_exclusion_re(self) -> re.Pattern[str] | None:
    """Compiled `path_exclusion_regex`, if set."""
    if not self.path_exclusion_regex:
      return None
    return _compile_re(self.path_exclusion_regex)

  @functools.cached_property
  def prohibited_re(self) -> re.Pattern[str] | None:
    """Compiled `prohibited_regex`, if set."""
    if self.prohibited_regex is None:
      return None
    return _compile_re(self.prohibited_regex)

  @functools.cached_property
  def prohibited_multiline_re(self) -> re.Pattern[str] | None:
    """`prohibited_re` with `^`/`$` matching at line boundaries."""
    if self.prohibited_regex is None:
      return None
    return _compile_re(self.prohibited_regex, re.MULTILINE)

  @functools.cached_property
  def prohibited_is_line_local(self) -> bool:
    """Whether `prohibited_regex` can be evaluated via whole-file MULTILINE."""
    return self.prohibited_regex is not None and is_line_local(
        self.prohibited_regex
    )

  @functools.cached_property
  def suppression_re(self) -> re.Pattern[str] | None:
    """Compiled `suppression_regex`, if set."""
    if self.suppression_regex is None:
      return None
    return _compile_re(self.suppression_regex)

  @functools.cached_property
  def required_re(self) -> re.Pattern[str] | None:
    """Compiled `required_regex`, if set."""
    if self.required_regex is None:
      return None
    return _compile_re(self.required_regex)

  @functools.cached_property
  def required_multiline_re(self) -> re.Pattern[str] | None:
    """`required_re` with `^`/`$` matching at line boundaries."""
    if self.required_regex is None:
      return None
    return _compile_re(self.required_regex, re.MULTILINE)

  @functools.cached_property
  def required_is_line_local(self) -> bool:
    """Whether `required_regex` can be evaluated via whole-file MULTILINE."""
    return self.required_regex is not None and is_line_local(
        self.required_regex
    )

  @functools.cached_property
  def _prohibited_singletons(self) -> tuple[str, ...]:
    """Literals that must all appear in a file for `prohibited_re` to match."""
    if self.prohibited_regex is None:
      return ()
    return _extract_required_literals(self.prohibited_regex)[0]

  @functools.cached_property
  def _prohibited_disjunctions(self) -> tuple[tuple[str, ...], ...]:
    """Clauses of which one literal must appear for `prohibited_re` to match."""
    if self.prohibited_regex is None:
      return ()
    return _extract_required_literals(self.prohibited_regex)[1]

  def matches_path(self, canonical_path: str) -> bool:
    """Returns True if the rule applies to `canonical_path`."""
    if not self.path_re.match(canonical_path):
      return False
    excl = self.path_exclusion_re
    return excl is None or not excl.match(canonical_path)

  def applies_to(self, operation: FileOperation | str | None) -> bool:
    """Returns whether the rule applies to a file changed by `operation`.

    `None` means there is no change context (explicit files, `--all`). Rules
    restricted to specific operations are about changes, so they do not apply.

    Args:
      operation: A `FileOperation`, or None.
    """
    if not self.operations:
      return True
    return operation is not None and operation in self.operations


@dataclasses.dataclass(frozen=True)
class Skipped:
  """A check found in a rule file that this tool does not implement."""

  kind: str
  id: str | None
  origin: str


@dataclasses.dataclass
class RuleSet:
  """The result of loading a rule file."""

  rules: list[Rule] = dataclasses.field(default_factory=list)
  skipped: list[Skipped] = dataclasses.field(default_factory=list)


class LoadError(ValueError):
  """An error loading a rule file, with its location when known."""

  def __init__(self, file: str, line: int | None, message: str):
    location = f'{file}:{line}' if line is not None else file
    super().__init__(f'{location}: {message}')
    self.file = file
    self.line = line
    self.message = message


# ---------------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------------


def _check_regex(field: str, pattern: str) -> str:
  pattern = re2_to_python(pattern)
  try:
    _compile_re(pattern)
  except re.error as e:
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
        field, anchor(path_expression_to_regex(resolve_path(base_dir, expr)))
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


def _operations(message: Message, name: str) -> tuple[FileOperation, ...]:
  out: list[FileOperation] = []
  for op in message.idents(name):
    try:
      out.append(FileOperation(op))
    except ValueError as e:
      raise ValueError(f'unknown `{name}` `{op}`') from e
  return tuple(out)


def _build_rule(
    kind: CheckKind,
    check: Message,
    paths: tuple[str, str | None],
    block_ops: tuple[FileOperation, ...],
    origin: str,
    line: int,
) -> Rule | Skipped:
  """Builds a Rule (or Skipped entry) from a check message."""
  base = check.message('base')
  rule_id = (base.string('id') if base else None) or f'CheckContents@{line}'

  def flag(name: str) -> bool:
    return bool(check.bool(name))

  if kind == CheckKind.CHECK_CONTENTS and flag('check_deleted_only'):
    return Skipped(f'{kind}.check_deleted_only', rule_id, origin)

  raw_status = check.ident('failure_status') or Severity.ERROR.value
  try:
    status = Severity(raw_status)
  except ValueError as e:
    raise ValueError(f'unknown `failure_status` `{raw_status}`') from e
  check_ops = _operations(check, 'operation')
  common = dict(
      id=rule_id,
      path_regex=paths[0],
      path_exclusion_regex=paths[1],
      failure_message=check.string('failure_message') or '',
      severity=status,
      disable_tags=frozenset(base.strings('disable_tags'))
      if base
      else frozenset(),
      operations=check_ops or block_ops,
      origin=origin,
  )
  if kind == CheckKind.CHECK_FILE_OPERATION:
    return Rule(kind=CheckKind.CHECK_FILE_OPERATION, **common)

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
      **common,
  )


def parse_metadata(text: str, file: str, base_dir: str) -> RuleSet:
  """Parses METADATA `text`; relative rule paths resolve against `base_dir`."""
  try:
    root = parse_textproto(text)
  except ParseError as e:
    raise LoadError(
        file, e.line, f'syntax error at column {e.column}: {e.message}'
    ) from e

  rule_set = RuleSet()
  seen_ids: set[str] = set()
  for presubmit, presubmit_line in root.messages('presubmit'):
    # Path filters are built lazily: blocks without supported checks may use
    # patterns that are never evaluated (e.g. RE2-only syntax).
    paths: tuple[str, str | None] | None = None
    block_ops: tuple[FileOperation, ...] = ()
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
          rule_id = (base.string('id') if base else None) or 'CheckContents'
          raise LoadError(file, field.line, f'{rule_id}: {e}') from e
        if isinstance(built, Skipped):
          rule_set.skipped.append(built)
        elif built.id in seen_ids:
          raise LoadError(file, field.line, f'duplicate rule id `{built.id}`')
        else:
          seen_ids.add(built.id)
          rule_set.rules.append(built)
      elif isinstance(value, Message) and field.name.startswith('check_'):
        base = value.message('base')
        rule_set.skipped.append(
            Skipped(field.name, base.string('id') if base else None, origin)
        )
      elif isinstance(value, str) and field.name == 'include_presubmit':
        rule_set.skipped.append(Skipped('include_presubmit', value, origin))
  return rule_set


def load_rules_file(path: pathlib.Path | str, base_dir: str) -> RuleSet:
  """Reads and parses a METADATA rule file."""
  try:
    text = pathlib.Path(path).read_text(encoding='utf-8')
  except OSError as e:
    raise LoadError(str(path), None, f'cannot read rules file: {e}') from e
  return parse_metadata(text, str(path), base_dir)
