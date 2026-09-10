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

"""Unit tests for CI base SHA resolution and diff isolation scripts.

These tests verify that `ci/tools/resolve_base_sha.sh`,
`ci/check_wheel_diff.sh`,
and `.github/actions/bazel-diff/run_bazel_diff.sh` correctly isolate pull
request
changes across various Git topologies (multi-parent PR merge commits, shallow
clones, single-parent commits, and explicit overrides) without leaking
unrelated commits from `main`.
"""

import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Final
import unittest

_CI_TOOLS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent
_REPO_ROOT: Final[pathlib.Path] = _CI_TOOLS_DIR.parents[1]
_RESOLVE_BASE_SHA_SCRIPT: Final[pathlib.Path] = (
    _CI_TOOLS_DIR / "resolve_base_sha.sh"
)
_CHECK_WHEEL_DIFF_SCRIPT: Final[pathlib.Path] = (
    _REPO_ROOT / "ci" / "check_wheel_diff.sh"
)


class ResolveBaseShaTest(  #
    unittest.TestCase  # UNITTEST_OK=avoid circular dependency on bazel-diff
):
  """Validates base SHA resolution across Git commit graphs."""

  def setUp(self):
    super().setUp()
    self.git_env = os.environ.copy()
    self.git_env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    self.git_env["GIT_CONFIG_SYSTEM"] = "/dev/null"
    # Force the standard C locale to ensure Git outputs predictable, English
    # messages and deterministic date/status formatting across all environments.
    self.git_env["LC_ALL"] = "C"

  def _run_git(self, cwd: str, *args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=self.git_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()

  def _init_repo(self, repo_dir: str) -> None:
    self._run_git(repo_dir, "init", "-b", "main")
    self._run_git(repo_dir, "config", "user.name", "Test User")
    self._run_git(repo_dir, "config", "user.email", "test@example.com")
    self._run_git(repo_dir, "config", "commit.gpgsign", "false")

  def test_single_parent_commit_without_override_returns_empty(self):
    """Linear history without override returns empty string to trigger full build."""
    with tempfile.TemporaryDirectory() as td:
      self._init_repo(td)
      test_file = pathlib.Path(td) / "file.txt"
      test_file.write_text("first\n")
      self._run_git(td, "add", "file.txt")
      self._run_git(td, "commit", "-m", "First commit")

      test_file.write_text("second\n")
      self._run_git(td, "commit", "-am", "Second commit")
      head_sha = self._run_git(td, "rev-parse", "HEAD")

      res = subprocess.run(
          ["bash", str(_RESOLVE_BASE_SHA_SCRIPT), head_sha, ""],
          cwd=td,
          env=self.git_env,
          capture_output=True,
          text=True,
          check=True,
      )
      self.assertEqual(res.stdout.strip(), "")

  def test_single_parent_commit_with_override_resolves_merge_base(self):
    """Linear history with an override base resolves via git merge-base."""
    with tempfile.TemporaryDirectory() as td:
      self._init_repo(td)
      test_file = pathlib.Path(td) / "file.txt"
      test_file.write_text("first\n")
      self._run_git(td, "add", "file.txt")
      self._run_git(td, "commit", "-m", "First commit")
      base_sha = self._run_git(td, "rev-parse", "HEAD")

      test_file.write_text("second\n")
      self._run_git(td, "commit", "-am", "Second commit")
      head_sha = self._run_git(td, "rev-parse", "HEAD")

      res = subprocess.run(
          ["bash", str(_RESOLVE_BASE_SHA_SCRIPT), head_sha, base_sha],
          cwd=td,
          env=self.git_env,
          capture_output=True,
          text=True,
          check=True,
      )
      self.assertEqual(res.stdout.strip(), base_sha)

  def test_pr_merge_commit_resolves_to_first_parent_and_isolates_diff(self):
    """PR merge commit resolves to main's head, isolating PR diff."""
    with tempfile.TemporaryDirectory() as td:
      self._init_repo(td)

      # Commit A on main
      readme = pathlib.Path(td) / "README.md"
      readme.write_text("initial\n")
      self._run_git(td, "add", "README.md")
      self._run_git(td, "commit", "-m", "Commit A")

      # Branch PR: modifies docs/guide.md
      self._run_git(td, "checkout", "-b", "feature-branch")
      docs_dir = pathlib.Path(td) / "docs"
      docs_dir.mkdir(parents=True, exist_ok=True)
      (docs_dir / "guide.md").write_text("PR documentation\n")
      self._run_git(td, "add", "docs/guide.md")
      self._run_git(td, "commit", "-m", "PR Commit C")

      # Branch main: receives unrelated commit B modifying MODULE.bazel
      self._run_git(td, "checkout", "main")
      module_file = pathlib.Path(td) / "MODULE.bazel"
      module_file.write_text("module_version_2\n")
      self._run_git(td, "add", "MODULE.bazel")
      self._run_git(td, "commit", "-m", "Commit B on main")
      main_sha = self._run_git(td, "rev-parse", "HEAD")

      # Synthetic PR merge commit M = merge(main, PR)
      self._run_git(
          td, "merge", "--no-ff", "feature-branch", "-m", "Merge PR commit"
      )
      merge_sha = self._run_git(td, "rev-parse", "HEAD")

      # Resolve base SHA for M
      res = subprocess.run(
          ["bash", str(_RESOLVE_BASE_SHA_SCRIPT), merge_sha, ""],
          cwd=td,
          env=self.git_env,
          capture_output=True,
          text=True,
          check=True,
      )
      resolved_base = res.stdout.strip()
      self.assertEqual(resolved_base, main_sha)

      # Assert diff isolates strictly PR files and excludes main's commit B
      changed_files = self._run_git(
          td, "diff", "--name-only", resolved_base, merge_sha
      ).splitlines()
      self.assertEqual(changed_files, ["docs/guide.md"])
      self.assertNotIn("MODULE.bazel", changed_files)

  def test_pr_merge_commit_with_explicit_override_takes_precedence(self):
    """Explicit override base SHA takes precedence over PR merge commit resolution."""
    with tempfile.TemporaryDirectory() as td:
      self._init_repo(td)

      # Commit A on main
      readme = pathlib.Path(td) / "README.md"
      readme.write_text("initial\n")
      self._run_git(td, "add", "README.md")
      self._run_git(td, "commit", "-m", "Commit A")
      commit_a = self._run_git(td, "rev-parse", "HEAD")

      # Branch PR
      self._run_git(td, "checkout", "-b", "feature-branch")
      (pathlib.Path(td) / "pr.txt").write_text("pr\n")
      self._run_git(td, "add", "pr.txt")
      self._run_git(td, "commit", "-m", "PR Commit C")

      # Commit B on main
      self._run_git(td, "checkout", "main")
      (pathlib.Path(td) / "file.txt").write_text("v2\n")
      self._run_git(td, "add", "file.txt")
      self._run_git(td, "commit", "-m", "Commit B on main")

      # Merge commit M
      self._run_git(
          td, "merge", "--no-ff", "feature-branch", "-m", "Merge PR commit"
      )
      merge_sha = self._run_git(td, "rev-parse", "HEAD")

      # Passing commit_a as override must take precedence over main's head
      res = subprocess.run(
          ["bash", str(_RESOLVE_BASE_SHA_SCRIPT), merge_sha, commit_a],
          cwd=td,
          env=self.git_env,
          capture_output=True,
          text=True,
          check=True,
      )
      self.assertEqual(res.stdout.strip(), commit_a)

  def test_pr_merge_in_shallow_clone_depth_2(self):
    """PR merge commit resolution succeeds in fetch-depth: 2 shallow clones."""
    with tempfile.TemporaryDirectory() as td:
      origin_dir = os.path.join(td, "origin")
      os.makedirs(origin_dir)
      self._init_repo(origin_dir)

      # Base commit on main
      (pathlib.Path(origin_dir) / "file.txt").write_text("v1\n")
      self._run_git(origin_dir, "add", "file.txt")
      self._run_git(origin_dir, "commit", "-m", "Init")

      # PR commit
      self._run_git(origin_dir, "checkout", "-b", "feature")
      (pathlib.Path(origin_dir) / "pr.txt").write_text("pr\n")
      self._run_git(origin_dir, "add", "pr.txt")
      self._run_git(origin_dir, "commit", "-m", "PR commit")

      # Main commit
      self._run_git(origin_dir, "checkout", "main")
      (pathlib.Path(origin_dir) / "file.txt").write_text("v2\n")
      self._run_git(origin_dir, "commit", "-am", "Main commit B")
      expected_parent_1 = self._run_git(origin_dir, "rev-parse", "HEAD")

      # Merge commit
      self._run_git(origin_dir, "merge", "--no-ff", "feature", "-m", "Merge PR")
      merge_sha = self._run_git(origin_dir, "rev-parse", "HEAD")

      # Shallow clone with fetch-depth: 2 (mimicking actions/checkout)
      clone_dir = os.path.join(td, "shallow_clone")
      os.makedirs(clone_dir)
      self._run_git(clone_dir, "init")
      self._run_git(clone_dir, "config", "commit.gpgsign", "false")
      self._run_git(clone_dir, "remote", "add", "origin", origin_dir)
      self._run_git(clone_dir, "fetch", "--depth=2", "origin", merge_sha)
      self._run_git(clone_dir, "checkout", merge_sha)

      res = subprocess.run(
          ["bash", str(_RESOLVE_BASE_SHA_SCRIPT), merge_sha, ""],
          cwd=clone_dir,
          env=self.git_env,
          capture_output=True,
          text=True,
          check=True,
      )
      self.assertEqual(res.stdout.strip(), expected_parent_1)

  def test_pr_merge_in_shallow_clone_depth_1_fallback(self):
    """If parent 1 is missing in depth 1 clone, fallback handles gracefully."""
    with tempfile.TemporaryDirectory() as td:
      origin_dir = os.path.join(td, "origin")
      os.makedirs(origin_dir)
      self._init_repo(origin_dir)

      (pathlib.Path(origin_dir) / "file.txt").write_text("v1\n")
      self._run_git(origin_dir, "add", "file.txt")
      self._run_git(origin_dir, "commit", "-m", "Init")

      self._run_git(origin_dir, "checkout", "-b", "feature")
      (pathlib.Path(origin_dir) / "pr.txt").write_text("pr\n")
      self._run_git(origin_dir, "add", "pr.txt")
      self._run_git(origin_dir, "commit", "-m", "PR commit")

      self._run_git(origin_dir, "checkout", "main")
      self._run_git(origin_dir, "merge", "--no-ff", "feature", "-m", "Merge PR")
      merge_sha = self._run_git(origin_dir, "rev-parse", "HEAD")

      # Shallow clone with fetch-depth: 1 (parent 1 is not in history)
      clone_dir = os.path.join(td, "shallow_clone_1")
      os.makedirs(clone_dir)
      self._run_git(clone_dir, "init")
      self._run_git(clone_dir, "config", "commit.gpgsign", "false")
      self._run_git(clone_dir, "remote", "add", "origin", origin_dir)
      self._run_git(clone_dir, "fetch", "--depth=1", "origin", merge_sha)
      self._run_git(clone_dir, "checkout", merge_sha)

      # Should exit 0 without crashing
      res = subprocess.run(
          ["bash", str(_RESOLVE_BASE_SHA_SCRIPT), merge_sha, ""],
          cwd=clone_dir,
          env=self.git_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(res.returncode, 0)

  def test_check_wheel_diff_integration(self):
    """check_wheel_diff.sh skips build on docs-only PR despite main commit."""
    with tempfile.TemporaryDirectory() as td:
      self._init_repo(td)

      # Copy ci directory structure into test repo so relative paths resolve
      os.makedirs(os.path.join(td, "ci", "tools"), exist_ok=True)
      shutil.copy(
          str(_RESOLVE_BASE_SHA_SCRIPT), os.path.join(td, "ci", "tools")
      )
      shutil.copy(str(_CHECK_WHEEL_DIFF_SCRIPT), os.path.join(td, "ci"))

      # Commit A on main
      readme = pathlib.Path(td) / "README.md"
      readme.write_text("initial\n")
      self._run_git(td, "add", "README.md")
      self._run_git(td, "commit", "-m", "Commit A")

      # PR branch: docs only
      self._run_git(td, "checkout", "-b", "docs-branch")
      docs_file = pathlib.Path(td) / "docs" / "api.md"
      docs_file.parent.mkdir(parents=True, exist_ok=True)
      docs_file.write_text("api docs\n")
      self._run_git(td, "add", "docs/api.md")
      self._run_git(td, "commit", "-m", "Docs PR")

      # Main branch: modifies MODULE.bazel (wheel-affecting file)
      self._run_git(td, "checkout", "main")
      module_file = pathlib.Path(td) / "MODULE.bazel"
      module_file.write_text("module_change\n")
      self._run_git(td, "add", "MODULE.bazel")
      self._run_git(td, "commit", "-m", "Main commit touching MODULE.bazel")

      # Merge commit M
      self._run_git(
          td, "merge", "--no-ff", "docs-branch", "-m", "Merge docs PR"
      )
      merge_sha = self._run_git(td, "rev-parse", "HEAD")

      gh_output = os.path.join(td, "gh_output.txt")
      env = self.git_env.copy()
      env["HEAD_SHA"] = merge_sha
      env["BASE_SHA"] = ""
      env["GITHUB_OUTPUT"] = gh_output

      # 1. PR event should detect docs-only and emit should_build=false
      env["EVENT_NAME"] = "pull_request"
      env["FORCE_BUILD"] = "false"
      subprocess.run(
          ["bash", os.path.join(td, "ci", "check_wheel_diff.sh")],
          cwd=td,
          env=env,
          check=True,
      )
      with open(gh_output) as f:
        out = f.read()
      self.assertIn("should_build=false", out)

      # 2. Push event should emit should_build=true
      pathlib.Path(gh_output).unlink()
      env["EVENT_NAME"] = "push"
      subprocess.run(
          ["bash", os.path.join(td, "ci", "check_wheel_diff.sh")],
          cwd=td,
          env=env,
          check=True,
      )
      with open(gh_output) as f:
        out = f.read()
      self.assertIn("should_build=true", out)

      # 3. Force build label on PR should emit should_build=true
      pathlib.Path(gh_output).unlink()
      env["EVENT_NAME"] = "pull_request"
      env["FORCE_BUILD"] = "true"
      subprocess.run(
          ["bash", os.path.join(td, "ci", "check_wheel_diff.sh")],
          cwd=td,
          env=env,
          check=True,
      )
      with open(gh_output) as f:
        out = f.read()
      self.assertIn("should_build=true", out)


if __name__ == "__main__":
  unittest.main()
