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

"""Helper library for C++ style linter."""

import os
import pathlib
import re
from typing import Generator


def _find_repo_root() -> pathlib.Path:
  """Returns the root directory of the torch_tpu repository."""
  rf_dir = os.environ.get("TEST_SRCDIR") or os.environ.get("RUNFILES_DIR")
  if rf_dir:
    for candidate in [
        # Blaze (internal Google3).
        pathlib.Path(rf_dir) / "google3" / "third_party" / "py" / "torch_tpu",
        # Bazel (OSS with Bzlmod names the root workspace '_main').
        pathlib.Path(rf_dir) / "_main",
        # Bazel (classic WORKSPACE or when torch_tpu is used as an external dependency).
        pathlib.Path(rf_dir) / "torch_tpu",
    ]:
      if (candidate / "tests" / "cpp_style_test_data.cc").exists():
        return candidate

  raise FileNotFoundError(
      "Could not find torch_tpu repository root in runfiles."
  )


_RUNFILES_DIR = _find_repo_root()

_NAMESPACE_START_RE = re.compile(r"^namespace\s*([\w:]*)\s*{")
_NAMESPACE_END_RE = re.compile(r"^}  // namespace\s*([\w:]*)\s*")

_NON_REPO_DIR_NAMES = frozenset(
    {"site-packages", "dist-packages", "venv", ".venv"}
)


def _is_ignored_dir(name: str) -> bool:
  """Returns True if a directory should be skipped during file traversal.

  In OSS Bazel builds, rules_python creates virtual environments inside the
  test runfiles directory (e.g. tests/_<target>.venv/lib/.../site-packages/).
  These directories pull in external third-party C/C++ files (such as
  lxml/etree.h) that are not part of the TorchTPU repository and must be
  skipped. Wheel tests unpack the built torch_tpu wheel into a
  <target>_unpacked_wheel/ directory; the public headers it ships are copies
  of source files that are linted at their source location.
  Hidden directories (starting with '.') are also ignored.
  """
  return (
      name.endswith(".venv")
      or name.endswith("_unpacked_wheel")
      or name in _NON_REPO_DIR_NAMES
      or name.startswith(".")
  )


class CppFile:
  """Represents a C++ file with its path."""

  def __init__(
      self,
      path: pathlib.Path,
      repo_root: pathlib.Path | None = None,
  ):
    self._path = path
    self._repo_root = repo_root or pathlib.Path(_RUNFILES_DIR)

  @property
  def path(self) -> pathlib.Path:
    return self._path

  @property
  def torch_tpu_path(self) -> str:
    """Returns the torch_tpu relative path as a string."""
    return str(self._path.relative_to(self._repo_root))

  @property
  def is_header(self) -> bool:
    """Returns True if the file is a C++ header file."""
    return self._path.suffix == ".h"

  @property
  def is_test_data(self) -> bool:
    """Returns True if the file is a test data file."""
    return "test_data" in self._path.name

  @property
  def is_repo_file(self) -> bool:
    """Returns True if the file is part of the repository (not virtualenv/external)."""
    if self._path.name.startswith("."):
      return False
    try:
      rel_path = self._path.relative_to(self._repo_root)
      parts_to_check = rel_path.parts[:-1]
    except ValueError:
      parts_to_check = self._path.parts[:-1]

    for part in parts_to_check:
      if _is_ignored_dir(part):
        return False
    return True

  @property
  def expected_header_guard(self) -> str:
    """Returns the expected header guard macro name."""
    rel_path = self.torch_tpu_path
    # In OSS builds, Copybara moves the C++ source tree under a 'torch_tpu/'
    # subdirectory within the workspace root ('_main'). Header guards in TorchTPU
    # are defined relative to the 'torch_tpu/' package root (starting with
    # 'TORCH_TPU_' followed by the relative path under torch_tpu/). Strip any
    # leading 'torch_tpu/' prefix to prevent a duplicate 'TORCH_TPU_TORCH_TPU_' guard.
    if rel_path.startswith("torch_tpu/"):
      rel_path = rel_path.removeprefix("torch_tpu/")
    guard = re.sub(r"[^A-Za-z0-9]", "_", rel_path).upper() + "_"
    return f"TORCH_TPU_{guard}"

  def __repr__(self) -> str:
    return self.torch_tpu_path

  def get_namespace(self) -> Generator[tuple[list[str], str], None, None]:
    """Yields ((active_namespace...), line) for each line in the file.

    Example:
    c++```
    namespace foo {
    namespace bar{
    namespace {
    void Baz() {}
    }  // namespace
    }  // namespace bar
    }  // namespace foo
    ```

    Will generate result containing (["foo", "bar", ""], "void Baz() {}").
    """
    for _, active_namespaces, line in self.get_namespace_with_line_number():
      yield active_namespaces, line

  def get_namespace_with_line_number(
      self,
  ) -> Generator[tuple[int, list[str], str], None, None]:
    """Like `get_namespace`, but also yields the 1-based line number first.

    For lines that open or close a namespace, the yielded namespaces are the
    ones active *before* the line takes effect.
    """

    namespace_stack = []

    for line_number, line in enumerate(
        self.path.read_text().splitlines(), start=1
    ):
      if not line:
        continue

      match = _NAMESPACE_START_RE.match(line)
      if match:
        name = match.group(1)
        segments = name.split("::")
        yield (line_number, list(namespace_stack), line)
        namespace_stack.extend(segments)
        continue

      match = _NAMESPACE_END_RE.match(line)
      if match:
        name = match.group(1)
        segments = name.split("::")
        yield (line_number, list(namespace_stack), line)
        for _ in reversed(segments):
          namespace_stack.pop()
        continue

      yield (line_number, list(namespace_stack), line)


class CppCode:
  """Provides access to C++ files in torch_tpu."""

  def __init__(
      self,
      torch_tpu_path: str = "",
      repo_root: pathlib.Path | None = None,
  ):
    self._repo_root = repo_root or pathlib.Path(_RUNFILES_DIR)
    self._torch_tpu_path = torch_tpu_path
    assert self.path.exists()

  @property
  def path(self) -> pathlib.Path:
    return self._repo_root / self._torch_tpu_path

  def files(
      self,
      exclude_test_data: bool = False,
      exclude_non_repo: bool = True,
  ) -> Generator[CppFile, None, None]:
    """Yields CppFile objects for the selected files."""
    for root, dirs, filenames in os.walk(self.path):
      if exclude_non_repo:
        dirs[:] = [d for d in dirs if not _is_ignored_dir(d)]
      for filename in filenames:
        if filename.endswith((".cc", ".cpp", ".h")):
          f = CppFile(pathlib.Path(root) / filename, repo_root=self._repo_root)
          if exclude_non_repo and not f.is_repo_file:
            continue
          if exclude_test_data and f.is_test_data:
            continue
          yield f

  def headers(
      self,
      exclude_test_data: bool = False,
      exclude_non_repo: bool = True,
  ) -> Generator[CppFile, None, None]:
    """Yields CppFile objects for header files."""
    for f in self.files(
        exclude_test_data=exclude_test_data,
        exclude_non_repo=exclude_non_repo,
    ):
      if f.is_header:
        yield f

  def find_file(self, filename: str) -> CppFile | None:
    """Finds a file by its name in the selected files."""
    for f in self.files():
      if f.path.name == filename:
        return f
    return None


_IFNDEF_RE = re.compile(r"^#ifndef\s+(\w+)")
_DEFINE_RE = re.compile(r"^#define\s+(\w+)")
_ENDIF_RE = re.compile(r"^#endif(?:\s+(?://|/\*)\s*(\w+))?")


def check_header_guard(file: CppFile) -> list[str]:
  """Checks if a CppFile has the correct header guard.

  Args:
    file: The CppFile to check.

  Returns:
    A list of strings, where each string is an error message. Empty results mean
    no header guard violations.
  """
  if not file.is_header:
    return []

  expected = file.expected_header_guard
  errors = []
  lines = file.path.read_text().splitlines()

  # Find the opening #ifndef
  ifndef_idx = None
  actual_ifndef = None
  for idx, line in enumerate(lines):
    match = _IFNDEF_RE.match(line.strip())
    if match:
      ifndef_idx = idx
      actual_ifndef = match.group(1)
      break

  if ifndef_idx is None:
    return [
        f"{file.torch_tpu_path}: Missing header guard #ifndef. Expected"
        f" '#ifndef {expected}'"
    ]

  if actual_ifndef != expected:
    errors.append(
        f"{file.torch_tpu_path}:{ifndef_idx + 1}: Expected '#ifndef"
        f" {expected}', got '#ifndef {actual_ifndef}'"
    )

  # Check #define immediately following #ifndef (skipping empty lines)
  define_idx = None
  actual_define = None
  for idx in range(ifndef_idx + 1, len(lines)):
    line_stripped = lines[idx].strip()
    if not line_stripped:
      continue
    match = _DEFINE_RE.match(line_stripped)
    if match:
      define_idx = idx
      actual_define = match.group(1)
    break

  if define_idx is None:
    errors.append(
        f"{file.torch_tpu_path}: Missing header guard #define. Expected"
        f" '#define {expected}'"
    )
  elif actual_define != expected:
    errors.append(
        f"{file.torch_tpu_path}:{define_idx + 1}: Expected '#define"
        f" {expected}', got '#define {actual_define}'"
    )

  # Check closing #endif (the last #endif in the file)
  endif_idx = None
  actual_endif_comment = None
  for idx in reversed(range(len(lines))):
    line_stripped = lines[idx].strip()
    if line_stripped.startswith("#endif"):
      endif_idx = idx
      match = _ENDIF_RE.match(line_stripped)
      if match:
        actual_endif_comment = match.group(1)
      break

  if endif_idx is None:
    errors.append(
        f"{file.torch_tpu_path}: Missing closing #endif for header guard."
    )
  elif actual_endif_comment != expected:
    errors.append(
        f"{file.torch_tpu_path}:{endif_idx + 1}: Expected '#endif  //"
        f" {expected}', got '{lines[endif_idx].strip()}'"
    )

  return errors


_APPROVED_NESTED_NAMESPACES = frozenset({"error", "internal"})

# Exempts one namespace opening from `check_namespace_root`, and from the
# "torch_tpu must be outermost" part of `check_unapproved_namespace`, when placed
# in a trailing comment on the line that opens it.
_NON_TORCH_TPU_MARKER = "NON_TORCH_TPU_OK"
_NON_TORCH_TPU_MARKER_RE = re.compile(
    rf"//.*\b{_NON_TORCH_TPU_MARKER}\b(?:=(?P<reason>.*))?"
)


def _has_marker_with_reason(line: str) -> bool:
  """Returns True if `line` has a `NON_TORCH_TPU_OK` marker with a reason."""
  marker = _NON_TORCH_TPU_MARKER_RE.search(line)
  return bool(marker and (marker.group("reason") or "").strip())


def _format_namespace(segments: list[str]) -> str:
  """Formats namespace segments for error messages, e.g. `foo::(anonymous)`."""
  return "::".join(segment or "(anonymous)" for segment in segments)


def _nested_namespace_error(active_namespaces: list[str]) -> str | None:
  """Returns why `active_namespaces` is an invalid nesting, or None if valid."""
  if "torch_tpu" in active_namespaces[1:]:
    return (
        "torch_tpu must only be used as the outermost namespace, got"
        f" {_format_namespace(active_namespaces)}"
    )
  if not active_namespaces or active_namespaces[0] != "torch_tpu":
    return None
  namespaces_to_check = active_namespaces[1:]
  # We only permit the inner-most namespace to be anonymous.
  if namespaces_to_check and not namespaces_to_check[-1]:
    namespaces_to_check.pop(-1)
  if all(n in _APPROVED_NESTED_NAMESPACES for n in namespaces_to_check):
    return None
  return (
      "Only approved named namespaces allowed under torch_tpu:"
      f" {sorted(_APPROVED_NESTED_NAMESPACES)}, got {namespaces_to_check}"
  )


def check_unapproved_namespace(file: CppFile) -> list[str]:
  """Checks if a CppFile uses any unapproved nested namespaces.

  Namespaces nested under torch_tpu must be one of the approved names, and only
  the inner-most one may be anonymous. torch_tpu itself must only appear as the
  outermost namespace (e.g. `mlir::torch_tpu` is an error), unless the line that
  opens the offending namespace has a `// NON_TORCH_TPU_OK=<reason>` marker.

  Args:
    file: The CppFile to check.

  Returns:
    A list of strings, where each string is an error message. Empty results mean
    no improper usage.
  """
  errors = []
  for (
      line_number,
      active_namespaces,
      line,
  ) in file.get_namespace_with_line_number():
    match = _NAMESPACE_START_RE.match(line)
    if not match:
      continue
    opened_namespaces = active_namespaces + match.group(1).split("::")
    error = _nested_namespace_error(opened_namespaces)
    # Only report the namespace that introduces the problem, not every
    # namespace nested inside of it.
    if not error or _nested_namespace_error(active_namespaces):
      continue
    if "torch_tpu" in opened_namespaces[1:] and _has_marker_with_reason(line):
      continue
    errors.append(f"{file.torch_tpu_path}:{line_number}: {error}")

  return errors


def check_namespace_root(file: CppFile) -> list[str]:
  """Checks that every top-level namespace in a CppFile is rooted at torch_tpu.

  Every `namespace ... {` that is not nested in another namespace must start
  with `torch_tpu`, so e.g. `namespace mlir::torch_tpu {`, `namespace xla {` and
  a top-level anonymous `namespace {` are errors.

  Extending a third-party namespace is occasionally unavoidable (e.g. for
  registration macros or forward declarations that must not pull in external
  headers). Such a namespace block can be exempted by adding a trailing comment
  on the line that opens it, in the form `// NON_TORCH_TPU_OK=<reason>`. The
  reason must be non-empty.

  Args:
    file: The CppFile to check.

  Returns:
    A list of strings, where each string is an error message. Empty results mean
    no improper usage.
  """
  errors = []
  for (
      line_number,
      active_namespaces,
      line,
  ) in file.get_namespace_with_line_number():
    match = _NAMESPACE_START_RE.match(line)
    if not match or active_namespaces:
      continue
    segments = match.group(1).split("::")
    if segments[0] == "torch_tpu":
      continue
    location = f"{file.torch_tpu_path}:{line_number}"
    if _has_marker_with_reason(line):
      continue
    if _NON_TORCH_TPU_MARKER_RE.search(line):
      errors.append(
          f"{location}: {_NON_TORCH_TPU_MARKER} requires a reason, e.g."
          f" `// {_NON_TORCH_TPU_MARKER}=<reason>`, got: {line.strip()}"
      )
      continue
    errors.append(
        f"{location}: Namespaces must be rooted at torch_tpu, got"
        f" {_format_namespace(segments)}. If extending this namespace is"
        " unavoidable, add a trailing"
        f" `// {_NON_TORCH_TPU_MARKER}=<reason>` comment on the same line."
    )

  return errors
