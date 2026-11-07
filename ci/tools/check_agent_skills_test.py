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

"""Unit tests for check_agent_skills.py."""

import io
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_CI_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CI_TOOLS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools import check_agent_skills


class CheckAgentSkillsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.repo_root = pathlib.Path(self.temp_dir.name)
    self.mock_stderr = self.enterContext(
        mock.patch("sys.stderr", new_callable=io.StringIO)
    )

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def _add_skill(self, directory: str, frontmatter: str) -> None:
    skill_dir = self.repo_root / ".agents" / "skills" / directory
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n# Body\n")

  def _assert_invalid(self, message: str) -> None:
    self.assertFalse(check_agent_skills.validate_agent_skills(self.repo_root))
    self.assertIn(message, self.mock_stderr.getvalue())

  def test_valid_skills_pass(self):
    self._add_skill("tt-write-docs", "name: tt-write-docs\ndescription: x")
    self._add_skill("tt-fix-op-v2", "name: tt-fix-op-v2\ndescription: x")
    self.assertTrue(check_agent_skills.validate_agent_skills(self.repo_root))
    self.assertEqual(self.mock_stderr.getvalue(), "")

  def test_no_skills_directory_passes(self):
    self.assertTrue(check_agent_skills.validate_agent_skills(self.repo_root))

  def test_missing_prefix_fails(self):
    self._add_skill("write-docs", "name: write-docs")
    self._assert_invalid("must be `tt-` followed by")

  def test_underscore_in_name_fails(self):
    self._add_skill("tt_write_docs", "name: tt_write_docs")
    self._assert_invalid("must be `tt-` followed by")

  def test_uppercase_in_name_fails(self):
    self._add_skill("tt-Write-Docs", "name: tt-Write-Docs")
    self._assert_invalid("must be `tt-` followed by")

  def test_gerund_name_fails(self):
    self._add_skill("tt-writing-docs", "name: tt-writing-docs")
    self._assert_invalid("must start with an imperative verb")

  def test_directory_not_matching_name_fails(self):
    self._add_skill("docs-review", "name: tt-write-docs")
    self._assert_invalid("must be named after the skill")

  def test_missing_skill_md_fails(self):
    skill_dir = self.repo_root / ".agents" / "skills" / "tt-write-docs"
    skill_dir.mkdir(parents=True)
    (skill_dir / "README.md").write_text("Not a skill.")
    self._assert_invalid("has no SKILL.md")

  def test_empty_directory_ignored(self):
    (self.repo_root / ".agents" / "skills" / "docs-review").mkdir(parents=True)
    self.assertTrue(check_agent_skills.validate_agent_skills(self.repo_root))

  def test_missing_name_fails(self):
    self._add_skill("tt-write-docs", "description: x")
    self._assert_invalid("has no `name`")

  def test_main_returns_nonzero_on_error(self):
    self._add_skill("docs-review", "name: docs-review")
    self.assertEqual(
        check_agent_skills.main(["--repo-root", str(self.repo_root)]), 1
    )


if __name__ == "__main__":
  unittest.main()
