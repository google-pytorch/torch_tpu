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

"""Verifies that the published Markdown files use GitHub-compatible tables.

google3's Markdown formatter (`mdformat`, run by `hg fix`) can rewrite wide
tables into g3doc's multi-line table syntax, where a table row continues on
lines that start and end with `:`. GitHub doesn't support that syntax and
renders such tables as garbage, so this test fails if any Markdown file
published on GitHub contains it.
"""

from collections.abc import Sequence
import os
import re

from absl.testing import absltest
from tests import repo_paths

# Directories (relative to the repository root) that aren't published on
# GitHub, and so may use g3doc-only syntax.
_INTERNAL_DIRS = ("google", "g3doc")

# A file that must be scanned, to catch a test that silently scans nothing.
_EXPECTED_FILE = "docs/api_reference.md"

_CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_CONTINUATION_ROW_RE = re.compile(r"^\s*:.*:\s*$")

_FIX_INSTRUCTIONS = """\
Rewrite each table with standard single-line GFM rows, and keep `hg fix` from
re-wrapping it by surrounding it with these lines (each preceded by a blank
line):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
...table...
<!-- mdformat on -->
"""


def _find_published_markdown_files(root: str) -> Sequence[str]:
  """Returns the published Markdown files, relative to `root`."""
  paths = []
  for dirpath, dirnames, filenames in os.walk(root):
    rel_dir = os.path.relpath(dirpath, root)
    dirnames[:] = [
        d
        for d in dirnames
        if not repo_paths.is_ignored_dir(d)
        and not (rel_dir == "." and d in _INTERNAL_DIRS)
    ]
    for filename in filenames:
      if filename.endswith(".md"):
        paths.append(os.path.normpath(os.path.join(rel_dir, filename)))
  return sorted(paths)


def _find_multiline_table_rows(text: str) -> Sequence[int]:
  """Returns the 1-based line numbers of g3doc multi-line table rows."""
  lines = text.splitlines()
  in_fence = False
  line_numbers = []
  for i, line in enumerate(lines):
    if _CODE_FENCE_RE.match(line):
      in_fence = not in_fence
      continue
    if (
        not in_fence
        and i > 0
        and _TABLE_CONTINUATION_ROW_RE.match(line)
        and lines[i - 1].lstrip().startswith(("|", ":"))
    ):
      line_numbers.append(i + 1)
  return line_numbers


class MarkdownTablesTest(absltest.TestCase):  # ABSLTEST_OK=tooling

  def test_find_multiline_table_rows(self):
    text = (
        "| A   | B      |\n"
        "| --- | ------ |\n"
        "| 1   | first  |\n"
        ":     : second :\n"
        "\n"
        "```\n"
        "| x |\n"
        ": y :\n"
        "```\n"
    )

    self.assertEqual(_find_multiline_table_rows(text), [4])

  def test_published_markdown_tables_are_github_compatible(self):
    root = repo_paths.get_repo_root_in_runfiles()
    paths = _find_published_markdown_files(root)
    self.assertIn(_EXPECTED_FILE, paths)

    errors = []
    for path in paths:
      with open(os.path.join(root, path), encoding="utf-8") as f:
        for line_number in _find_multiline_table_rows(f.read()):
          errors.append(f"  {path}:{line_number}")

    if errors:
      self.fail(
          "These Markdown files use g3doc multi-line tables, which GitHub"
          " can't render:\n"
          + "\n".join(errors)
          + "\n\n"
          + _FIX_INSTRUCTIONS
      )


if __name__ == "__main__":
  absltest.main()
