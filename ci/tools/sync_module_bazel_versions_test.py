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

"""Tests for sync_module_bazel_versions.

Run with `uv run --no-project --with "pytest>=9" pytest ci/tools/`.
"""

import ast
import itertools
from collections.abc import Mapping, Sequence
import pathlib
from typing import Final

import pytest

import sync_module_bazel_versions as sync

_IN_SYNC_VALUES: Final[Mapping[str, object]] = {
    "SUPPORTED_PYTHON_VERSIONS": ["3.11", "3.12"],
    "DEFAULT_PYTHON_VERSION": "3.12",
    "EXTRA_PYTORCH_VERSIONS": ["2.12.0", "2.13.0"],
    "NIGHTLY_TORCH_VERSION": "2.14.0",
}

# One drifted replacement per constant, covering both value shapes: a longer
# list, a shorter list, and changed strings.
_DRIFTED_VALUES: Final[Mapping[str, object]] = {
    "SUPPORTED_PYTHON_VERSIONS": ["3.12", "3.13", "3.14"],
    "DEFAULT_PYTHON_VERSION": "3.13",
    "EXTRA_PYTORCH_VERSIONS": ["2.13.0"],
    "NIGHTLY_TORCH_VERSION": "2.15.0",
}

_ALL_CONSTANTS: Final[Sequence[str]] = tuple(_IN_SYNC_VALUES)

# Every subset of constants, for the drift-combination tests.
_DRIFT_COMBINATIONS: Final[Sequence[tuple[str, ...]]] = tuple(
    combo
    for size in range(len(_ALL_CONSTANTS) + 1)
    for combo in itertools.combinations(_ALL_CONSTANTS, size)
)


def _assignment(name: str, value: object) -> str:
  if isinstance(value, str):
    return f'{name} = "{value}"'
  items = "".join(f'    "{item}",\n' for item in value)
  return f"{name} = [\n{items}]"


def _module_text(values: Mapping[str, object]) -> str:
  """A miniature MODULE.bazel with the synced constants and their trimmings."""
  return (
      "module(\n"
      '    name = "torch_tpu",\n'
      ")\n"
      "\n"
      "# The supported Python versions -- hand-synced from\n"
      "# bazel/supported_python_versions.bzl.\n"
      f"{_assignment('SUPPORTED_PYTHON_VERSIONS', values['SUPPORTED_PYTHON_VERSIONS'])}\n"
      "\n"
      "# The default Python version, likewise hand-synced.\n"
      f"{_assignment('DEFAULT_PYTHON_VERSION', values['DEFAULT_PYTHON_VERSION'])}\n"
      "\n"
      "# The extra PyTorch versions -- hand-synced from\n"
      "# bazel/pytorch_versions.bzl.\n"
      f"{_assignment('EXTRA_PYTORCH_VERSIONS', values['EXTRA_PYTORCH_VERSIONS'])}\n"
      "\n"
      f"{_assignment('NIGHTLY_TORCH_VERSION', values['NIGHTLY_TORCH_VERSION'])}\n"
      "\n"
      "# Derived, deliberately not a literal: the parser must skip it.\n"
      "EXTRA_GLUE_TORCH_VERSIONS = EXTRA_PYTORCH_VERSIONS + [NIGHTLY_TORCH_VERSION]\n"
  )


def _lines(text: str) -> list[str]:
  return text.splitlines(keepends=True)


def _parsed_values(text: str) -> Mapping[str, object]:
  nodes = sync._literal_assignments(text, _ALL_CONSTANTS)
  return {name: ast.literal_eval(node.value) for name, node in nodes.items()}


def _write_repo(
    root: pathlib.Path,
    *,
    module_values: Mapping[str, object],
    source_values: Mapping[str, object],
) -> None:
  (root / "bazel").mkdir(parents=True)
  (root / "bazel" / "pytorch_versions.bzl").write_text(
      '"""PyTorch versions."""\n'
      "\n"
      'DEFAULT_TORCH_VERSION = "2.11.0"\n'
      "\n"
      f"{_assignment('EXTRA_PYTORCH_VERSIONS', source_values['EXTRA_PYTORCH_VERSIONS'])}\n"
      "\n"
      f"{_assignment('NIGHTLY_TORCH_VERSION', source_values['NIGHTLY_TORCH_VERSION'])}\n"
      "\n"
      "WHEEL_TORCH_VERSIONS = [DEFAULT_TORCH_VERSION] + EXTRA_PYTORCH_VERSIONS\n"
  )
  (root / "bazel" / "supported_python_versions.bzl").write_text(
      '"""Python versions."""\n'
      "\n"
      f"{_assignment('SUPPORTED_PYTHON_VERSIONS', source_values['SUPPORTED_PYTHON_VERSIONS'])}\n"
      "\n"
      f"{_assignment('DEFAULT_PYTHON_VERSION', source_values['DEFAULT_PYTHON_VERSION'])}\n"
  )
  (root / "MODULE.bazel").write_text(_module_text(module_values))


class TestMain:
  """End-to-end tests over a fake repository root."""

  def test_in_sync_is_a_noop(self, tmp_path, capsys):
    _write_repo(
        tmp_path, module_values=_IN_SYNC_VALUES, source_values=_IN_SYNC_VALUES
    )
    before = (tmp_path / "MODULE.bazel").read_text()

    assert sync.main([str(tmp_path)]) == 0

    assert (tmp_path / "MODULE.bazel").read_text() == before
    assert "in sync" in capsys.readouterr().out

  def test_updates_drift_and_is_idempotent(self, tmp_path, capsys):
    _write_repo(
        tmp_path, module_values=_IN_SYNC_VALUES, source_values=_DRIFTED_VALUES
    )

    assert sync.main([str(tmp_path)]) == 0
    first = (tmp_path / "MODULE.bazel").read_text()
    assert _parsed_values(first) == _DRIFTED_VALUES
    assert "updated" in capsys.readouterr().out

    assert sync.main([str(tmp_path)]) == 0
    assert (tmp_path / "MODULE.bazel").read_text() == first
    assert "in sync" in capsys.readouterr().out

  def test_missing_constant_in_source_errors(self, tmp_path, capsys):
    _write_repo(
        tmp_path, module_values=_IN_SYNC_VALUES, source_values=_IN_SYNC_VALUES
    )
    bzl = tmp_path / "bazel" / "pytorch_versions.bzl"
    bzl.write_text(bzl.read_text().replace("NIGHTLY_TORCH_VERSION", "RENAMED"))
    before = (tmp_path / "MODULE.bazel").read_text()

    assert sync.main([str(tmp_path)]) == 1

    error = capsys.readouterr().err
    assert "NIGHTLY_TORCH_VERSION" in error
    assert "pytorch_versions.bzl" in error
    assert (tmp_path / "MODULE.bazel").read_text() == before

  def test_missing_constant_in_module_errors(self, tmp_path, capsys):
    _write_repo(
        tmp_path, module_values=_IN_SYNC_VALUES, source_values=_IN_SYNC_VALUES
    )
    module_bazel = tmp_path / "MODULE.bazel"
    module_bazel.write_text(
        module_bazel.read_text().replace("DEFAULT_PYTHON_VERSION", "RENAMED")
    )

    assert sync.main([str(tmp_path)]) == 1

    error = capsys.readouterr().err
    assert "DEFAULT_PYTHON_VERSION" in error
    assert "MODULE.bazel" in error

  def test_missing_source_file_errors(self, tmp_path, capsys):
    _write_repo(
        tmp_path, module_values=_IN_SYNC_VALUES, source_values=_IN_SYNC_VALUES
    )
    (tmp_path / "bazel" / "supported_python_versions.bzl").unlink()

    assert sync.main([str(tmp_path)]) == 1
    assert "supported_python_versions.bzl" in capsys.readouterr().err

  def test_missing_module_bazel_errors(self, tmp_path, capsys):
    _write_repo(
        tmp_path, module_values=_IN_SYNC_VALUES, source_values=_IN_SYNC_VALUES
    )
    (tmp_path / "MODULE.bazel").unlink()

    assert sync.main([str(tmp_path)]) == 1
    assert "MODULE.bazel" in capsys.readouterr().err


class TestFindRepoRoot:

  def test_finds_the_nearest_torch_tpu_parent(self, tmp_path, monkeypatch):
    script = tmp_path / "torch_tpu" / "ci" / "tools" / "sync.py"
    script.parent.mkdir(parents=True)
    monkeypatch.setattr(sync, "__file__", str(script))

    assert sync._find_repo_root() == tmp_path / "torch_tpu"

  def test_no_torch_tpu_parent_raises(self, tmp_path, monkeypatch):
    script = tmp_path / "elsewhere" / "sync.py"
    script.parent.mkdir(parents=True)
    monkeypatch.setattr(sync, "__file__", str(script))

    with pytest.raises(sync.SyncError, match="torch_tpu"):
      sync._find_repo_root()


class TestSyncLines:

  def test_no_drift_returns_lines_unchanged(self):
    lines = _lines(_module_text(_IN_SYNC_VALUES))
    assert list(sync._sync_lines(lines, _IN_SYNC_VALUES)) == lines

  def test_is_idempotent_after_a_change(self):
    lines = _lines(_module_text(_IN_SYNC_VALUES))
    once = sync._sync_lines(lines, _DRIFTED_VALUES)
    assert list(once) != lines

    twice = sync._sync_lines(once, _DRIFTED_VALUES)
    assert list(twice) == list(once)

  def test_is_idempotent_after_no_change(self):
    lines = _lines(_module_text(_IN_SYNC_VALUES))
    once = sync._sync_lines(lines, _IN_SYNC_VALUES)
    twice = sync._sync_lines(once, _IN_SYNC_VALUES)
    assert list(twice) == list(once) == lines

  @pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
  def test_is_idempotent_under_line_endings(self, newline):
    text = _module_text(_IN_SYNC_VALUES).replace("\n", newline)
    lines = _lines(text)

    in_sync = sync._sync_lines(lines, _IN_SYNC_VALUES)
    assert list(in_sync) == lines

    once = sync._sync_lines(lines, _DRIFTED_VALUES)
    twice = sync._sync_lines(once, _DRIFTED_VALUES)
    assert list(twice) == list(once)
    assert _parsed_values("".join(twice)) == _DRIFTED_VALUES

  @pytest.mark.parametrize(
      "drifted",
      _DRIFT_COMBINATIONS,
      ids=["+".join(combo) or "none" for combo in _DRIFT_COMBINATIONS],
  )
  def test_updates_exactly_the_drifted_constants(self, drifted):
    values = dict(_IN_SYNC_VALUES) | {
        name: _DRIFTED_VALUES[name] for name in drifted
    }
    lines = _lines(_module_text(_IN_SYNC_VALUES))

    synced = "".join(sync._sync_lines(lines, values))

    assert _parsed_values(synced) == values
    # The untouched trimmings survive: rendering the same values from scratch
    # must reproduce the result exactly, comments and derived assignment
    # included.
    assert synced == _module_text(values)

  def test_preserves_surrounding_comments_and_blank_lines(self):
    lines = _lines(_module_text(_IN_SYNC_VALUES))
    synced = "".join(sync._sync_lines(lines, _DRIFTED_VALUES))

    for line in _lines(_module_text(_IN_SYNC_VALUES)):
      if line.startswith("#") or line == "\n":
        assert line in synced

  def test_preserves_a_trailing_inline_comment(self):
    text = _module_text(_IN_SYNC_VALUES).replace(
        'DEFAULT_PYTHON_VERSION = "3.12"',
        'DEFAULT_PYTHON_VERSION = "3.12"  # Keep in sync.',
    )

    synced = "".join(sync._sync_lines(_lines(text), _DRIFTED_VALUES))

    assert 'DEFAULT_PYTHON_VERSION = "3.13"  # Keep in sync.\n' in synced

  def test_missing_constant_raises(self):
    lines = _lines("OTHER_CONSTANT = 1\n")
    with pytest.raises(sync.SyncError, match="DEFAULT_PYTHON_VERSION"):
      sync._sync_lines(lines, {"DEFAULT_PYTHON_VERSION": "3.12"})

  def test_non_literal_assignment_raises(self):
    lines = _lines("DEFAULT_PYTHON_VERSION = SOMETHING_ELSE\n")
    with pytest.raises(sync.SyncError, match="DEFAULT_PYTHON_VERSION"):
      sync._sync_lines(lines, {"DEFAULT_PYTHON_VERSION": "3.12"})

  def test_unsupported_value_type_raises(self):
    lines = _lines(_module_text(_IN_SYNC_VALUES))
    with pytest.raises(sync.SyncError, match="unsupported value type"):
      sync._sync_lines(lines, dict(_IN_SYNC_VALUES) | {"NIGHTLY_TORCH_VERSION": 2.14})
