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

"""Tests for custom_linter.vcs and custom_linter.engine VCS integration."""

from collections.abc import Sequence
import io
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
from unittest import mock

import pytest
from ci.tools.custom_linter import engine as custom_lint
from ci.tools.custom_linter import vcs as custom_lint_vcs


@pytest.fixture(name="rules_repo")
def _rules_repo_fixture(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
  """Creates a minimal project root with a single `.txt` rule."""
  root_path = tmp_path.resolve()
  (root_path / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = root_path / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.txt" check_contents { base { id: "R" }'
      ' prohibited_regexp: "x" } }\n',
      encoding="utf-8",
  )
  return root_path, rules_file


def test_detect_and_root_and_prefix(tmp_path: pathlib.Path) -> None:
  repo = tmp_path.resolve()
  (repo / ".git").mkdir()
  sub = repo / "pkg" / "sub" / "proj"
  sub.mkdir(parents=True)

  detected = custom_lint_vcs.detect(sub)
  assert isinstance(detected, custom_lint_vcs.GitVcs)
  assert detected.name == "git"
  assert detected.root() == repo
  assert detected.prefix(sub) == "pkg/sub/proj/"
  assert not detected.prefix(repo)


def test_detect_prefers_jj_when_colocated_with_hg(
    tmp_path: pathlib.Path,
) -> None:
  repo = tmp_path.resolve()
  (repo / ".hg").mkdir()
  (repo / ".jj").mkdir()
  detected = custom_lint_vcs.detect(repo)
  assert isinstance(detected, custom_lint_vcs.JjVcs)
  assert detected.name == "jj"
  assert detected.root() == repo


def test_detect_prefers_jj_when_colocated_with_git(
    tmp_path: pathlib.Path,
) -> None:
  repo = tmp_path.resolve()
  (repo / ".git").mkdir()
  (repo / ".jj").mkdir()
  detected = custom_lint_vcs.detect(repo)
  assert isinstance(detected, custom_lint_vcs.JjVcs)
  assert detected.name == "jj"
  assert detected.root() == repo
  assert custom_lint_vcs.find_vcs_root(repo) == repo


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("git", custom_lint_vcs.GitVcs),
        ("hg", custom_lint_vcs.HgVcs),
        ("jj", custom_lint_vcs.JjVcs),
    ],
)
def test_for_name_returns_expected_backend(
    name: str, expected_cls: type[custom_lint_vcs.Vcs]
) -> None:
  backend = custom_lint_vcs.for_name(name)
  assert isinstance(backend, expected_cls)
  assert backend.name == name


def test_for_name_any() -> None:
  assert custom_lint_vcs.for_name("any").name == "any"


def test_any_vcs_reuses_active_backend_across_diff_and_log() -> None:
  any_vcs = custom_lint_vcs.for_name("any")
  calls: list[str] = []

  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    calls.append(cmd[0])
    if cmd[0] == "git":
      return mock.Mock(returncode=1, stdout="", stderr="not git")
    if cmd[0] == "hg":
      return mock.Mock(returncode=0, stdout="hg output\n", stderr="")
    return mock.Mock(returncode=1, stdout="", stderr="unexpected")

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    assert any_vcs.diff() == "hg output\n"
    assert any_vcs.log() == "hg output\n"
    assert any_vcs.diff() == "hg output\n"
  # First diff tries git then hg; subsequent log() and diff() go straight to hg.
  assert calls == ["git", "hg", "hg", "hg"]

  any_vcs_log_first = custom_lint_vcs.for_name("any")
  calls_log_first: list[str] = []

  def fake_run_log_first(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    calls_log_first.append(cmd[0])
    if cmd[0] == "git":
      return mock.Mock(returncode=1, stdout="", stderr="not git")
    if cmd[0] == "hg":
      return mock.Mock(returncode=0, stdout="hg desc\n", stderr="")
    return mock.Mock(returncode=1, stdout="", stderr="unexpected")

  with mock.patch(
      "subprocess.run", side_effect=fake_run_log_first, autospec=True
  ):
    assert any_vcs_log_first.log() == "hg desc\n"
    assert any_vcs_log_first.diff() == "hg desc\n"
  assert calls_log_first == ["git", "hg", "hg"]


_GIT_DIFF_FLAGS = [
    "git",
    "diff",
    "-U0",
    "--no-color",
    "--no-ext-diff",
    "--src-prefix=a/",
    "--dst-prefix=b/",
]


def test_get_modified_lines_from_vcs_git_merge_base_success(
    tmp_path: pathlib.Path,
) -> None:
  repo = tmp_path.resolve()
  (repo / ".git").mkdir()
  sub = repo / "sub" / "proj"
  sub.mkdir(parents=True)
  git_diff_output = textwrap.dedent("""\
      +++ b/sub/proj/foo/BUILD
      @@ -10,0 +11,2 @@
      +cc_library(
      +    name = "lib",
      +)
      +++ b/other/ignored.py
      @@ -1 +1 @@
      -old
      +new
      """)

  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd[:3] == ["git", "merge-base", "origin/main"]:
      return mock.Mock(returncode=0, stdout="abc1234\n", stderr="")
    if cmd == [*_GIT_DIFF_FLAGS, "abc1234"]:
      return mock.Mock(returncode=0, stdout=git_diff_output, stderr="")
    return mock.Mock(returncode=1, stdout="", stderr="")

  with mock.patch(
      "subprocess.run", side_effect=fake_run, autospec=True
  ) as mock_run:
    modified = custom_lint_vcs.get_modified_lines_from_vcs(
        "origin/main", root_dir=sub, vcs="git"
    )
    assert isinstance(modified, custom_lint_vcs.ModifiedLines)
    assert set(modified) == {"foo/BUILD"}
    assert modified["foo/BUILD"] == {11, 12}
    assert "sub/proj/foo/BUILD" not in modified
    assert "other/ignored.py" not in modified
    assert [c.args[0] for c in mock_run.call_args_list] == [
        ["git", "merge-base", "origin/main", "HEAD"],
        [*_GIT_DIFF_FLAGS, "abc1234"],
    ]


def test_get_modified_lines_from_vcs_git_merge_base_failure_fallback(
    tmp_path: pathlib.Path,
) -> None:
  repo = tmp_path.resolve()
  (repo / ".git").mkdir()
  git_diff_output = textwrap.dedent("""\
      +++ b/foo/BUILD
      @@ -10,0 +11,2 @@
      +cc_library(
      +    name = "lib",
      +)
      """)

  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd[:3] == ["git", "merge-base", "origin/main"]:
      return mock.Mock(returncode=128, stdout="", stderr="fatal: no merge base")
    if cmd == [*_GIT_DIFF_FLAGS, "origin/main"]:
      return mock.Mock(returncode=0, stdout=git_diff_output, stderr="")
    return mock.Mock(returncode=1, stdout="", stderr="")

  with mock.patch(
      "subprocess.run", side_effect=fake_run, autospec=True
  ) as mock_run:
    modified = custom_lint_vcs.get_modified_lines_from_vcs(
        "origin/main", root_dir=repo, vcs="git"
    )
    assert isinstance(modified, custom_lint_vcs.ModifiedLines)
    assert "foo/BUILD" in modified
    assert modified["foo/BUILD"] == {11, 12}
    assert [c.args[0] for c in mock_run.call_args_list] == [
        ["git", "merge-base", "origin/main", "HEAD"],
        [*_GIT_DIFF_FLAGS, "origin/main"],
    ]


@pytest.mark.parametrize(
    "raw_base", ["origin/main...", "origin/main..HEAD", "origin/main.."]
)
def test_get_modified_lines_from_vcs_git_with_existing_dots(
    tmp_path: pathlib.Path,
    raw_base: str,
) -> None:
  repo = tmp_path.resolve()
  (repo / ".git").mkdir()

  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd[:2] == ["git", "merge-base"]:
      return mock.Mock(returncode=0, stdout="def5678\n", stderr="")
    if cmd[:7] == _GIT_DIFF_FLAGS:
      return mock.Mock(returncode=0, stdout="", stderr="")
    return mock.Mock(returncode=1, stdout="", stderr="")

  with mock.patch(
      "subprocess.run", side_effect=fake_run, autospec=True
  ) as mock_run:
    custom_lint_vcs.get_modified_lines_from_vcs(
        raw_base, root_dir=repo, vcs="git"
    )
    assert [c.args[0] for c in mock_run.call_args_list] == [
        ["git", "merge-base", "origin/main", "HEAD"],
        [*_GIT_DIFF_FLAGS, "def5678"],
    ]


def test_get_commit_message() -> None:
  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd == ["git", "rev-parse", "--verify", "HEAD^2"]:
      return mock.Mock(
          returncode=128, stdout="", stderr="fatal: Needed a single revision"
      )
    if cmd == ["git", "log", "origin/main..HEAD", "--format=%B"]:
      return mock.Mock(returncode=0, stdout="feat: new feature\n", stderr="")
    return mock.Mock(returncode=1, stdout="", stderr="")

  with mock.patch(
      "subprocess.run", side_effect=fake_run, autospec=True
  ) as mock_run:
    msg = custom_lint_vcs.get_commit_message("origin/main...", vcs="git")
    assert msg == "feat: new feature\n"
    assert [c.args[0] for c in mock_run.call_args_list] == [
        ["git", "rev-parse", "--verify", "HEAD^2"],
        ["git", "log", "origin/main..HEAD", "--format=%B"],
    ]


def test_get_commit_message_merge_commit() -> None:
  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd == ["git", "rev-parse", "--verify", "HEAD^2"]:
      return mock.Mock(returncode=0, stdout="deadbeef\n", stderr="")
    if cmd == ["git", "log", "origin/main..HEAD^2", "--format=%B"]:
      return mock.Mock(
          returncode=0, stdout="PR commit\n\nALLOW_FOO=reason\n", stderr=""
      )
    return mock.Mock(returncode=1, stdout="", stderr="")

  with mock.patch(
      "subprocess.run", side_effect=fake_run, autospec=True
  ) as mock_run:
    msg = custom_lint_vcs.get_commit_message("origin/main", vcs="git")
    assert msg == "PR commit\n\nALLOW_FOO=reason\n"
    assert [c.args[0] for c in mock_run.call_args_list] == [
        ["git", "rev-parse", "--verify", "HEAD^2"],
        ["git", "log", "origin/main..HEAD^2", "--format=%B"],
    ]


def test_get_commit_message_empty_stdout_returns_empty() -> None:
  mock_result = mock.Mock(returncode=0, stdout="", stderr="")
  with mock.patch("subprocess.run", return_value=mock_result, autospec=True):
    assert not custom_lint_vcs.get_commit_message(None, vcs="hg")


def test_get_commit_message_all_fail_raises_vcs_error() -> None:
  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd[:2] == ["git", "rev-parse"]:
      return mock.Mock(returncode=0, stdout="deadbeef\n", stderr="")
    return mock.Mock(
        returncode=128, stdout="", stderr=f"fatal: {cmd[0]} failed"
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    with pytest.raises(custom_lint_vcs.VcsError) as exc_info:
      custom_lint_vcs.get_commit_message(
          "origin/main", vcs=custom_lint_vcs.VCS_ANY
      )
    err_msg = str(exc_info.value)
    assert "git log origin/main..HEAD^2 --format=%B" in err_msg
    assert "fatal: git failed" in err_msg
    assert "hg log -r . --template {desc}" in err_msg
    assert "fatal: hg failed" in err_msg


def test_get_modified_lines_from_vcs_hg(tmp_path: pathlib.Path) -> None:
  repo = tmp_path.resolve()
  (repo / ".hg").mkdir()
  sub = repo / "ws" / "pkg"
  sub.mkdir(parents=True)
  hg_diff_output = textwrap.dedent("""\
      +++ b/ws/pkg/common/BUILD
      @@ -5 +5 @@
      -old
      +new
      +++ b/ws/other/BUILD
      @@ -1 +1 @@
      -a
      +b
      """)

  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd[0] == "git":
      return mock.Mock(returncode=1, stdout="", stderr="fatal: not a git repo")
    if cmd == ["hg", "diff", "--git", "-U0", "-r", "p4base"]:
      return mock.Mock(returncode=0, stdout=hg_diff_output, stderr="")
    return mock.Mock(returncode=1, stdout="", stderr="")

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    modified = custom_lint_vcs.get_modified_lines_from_vcs(
        root_dir=sub, vcs=custom_lint_vcs.VCS_ANY
    )
    assert isinstance(modified, custom_lint_vcs.ModifiedLines)
    assert set(modified) == {"common/BUILD"}
    assert modified["common/BUILD"] == {5}
    assert "ws/pkg/common/BUILD" not in modified
    assert "ws/other/BUILD" not in modified


def test_get_modified_lines_from_vcs_jj(tmp_path: pathlib.Path) -> None:
  repo = tmp_path.resolve()
  (repo / ".jj").mkdir()
  sub = repo / "sub" / "pkg"
  sub.mkdir(parents=True)
  jj_diff_output = textwrap.dedent("""\
      +++ b/sub/pkg/csrc/BUILD
      @@ -8,0 +9,2 @@
      +cc_library(
      +    name = "jj_lib",
      +)
      """)

  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd[0] in ("git", "hg"):
      return mock.Mock(returncode=1, stdout="", stderr="not a git/hg repo")
    if cmd[0] == "jj":
      return mock.Mock(returncode=0, stdout=jj_diff_output, stderr="")
    return mock.Mock(returncode=1, stdout="", stderr="")

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    modified = custom_lint_vcs.get_modified_lines_from_vcs(
        root_dir=sub, vcs=custom_lint_vcs.VCS_ANY
    )
    assert isinstance(modified, custom_lint_vcs.ModifiedLines)
    assert set(modified) == {"csrc/BUILD"}
    assert modified["csrc/BUILD"] == {9, 10}


def test_explicit_vcs_runs_only_that_vcs() -> None:
  mock_result = mock.Mock(returncode=1, stdout="", stderr="abort: no repo")
  with mock.patch(
      "subprocess.run", return_value=mock_result, autospec=True
  ) as mock_run:
    with pytest.raises(custom_lint_vcs.VcsError, match="abort: no repo"):
      custom_lint_vcs.get_modified_lines_from_vcs(vcs="hg")
    with pytest.raises(custom_lint_vcs.VcsError, match="abort: no repo"):
      custom_lint_vcs.get_commit_message(None, vcs="hg")
  assert [c.args[0][0] for c in mock_run.call_args_list] == ["hg", "hg"]


def test_get_modified_lines_from_vcs_error_contains_backend_stderr() -> None:
  mock_result = mock.Mock(
      returncode=128, stdout="", stderr="fatal: bad revision 'origin/main'"
  )
  with mock.patch("subprocess.run", return_value=mock_result, autospec=True):
    with pytest.raises(custom_lint_vcs.VcsError) as exc_info:
      custom_lint_vcs.get_modified_lines_from_vcs("origin/main", vcs="git")
  err_msg = str(exc_info.value)
  assert (
      "git diff -U0 --no-color --no-ext-diff --src-prefix=a/ --dst-prefix=b/"
      " origin/main"
      in err_msg
  )
  assert "fatal: bad revision 'origin/main'" in err_msg


def test_auto_uses_detected_vcs() -> None:
  mock_result = mock.Mock(returncode=0, stdout="desc\n", stderr="")
  with (
      mock.patch.object(
          custom_lint_vcs,
          "detect",
          return_value=custom_lint_vcs.JjVcs(),
          autospec=True,
      ),
      mock.patch(
          "subprocess.run", return_value=mock_result, autospec=True
      ) as mock_run,
  ):
    assert custom_lint_vcs.get_commit_message(None) == "desc\n"
  mock_run.assert_called_once()
  assert mock_run.call_args.args[0][0] == "jj"


def test_auto_without_detected_vcs_tries_all() -> None:
  mock_result = mock.Mock(returncode=1, stdout="", stderr="backend failure")
  with (
      mock.patch.object(
          custom_lint_vcs, "detect", return_value=None, autospec=True
      ),
      mock.patch(
          "subprocess.run", return_value=mock_result, autospec=True
      ) as mock_run,
  ):
    with pytest.raises(custom_lint_vcs.VcsError, match="backend failure"):
      custom_lint_vcs.get_modified_lines_from_vcs()
  assert [c.args[0][0] for c in mock_run.call_args_list] == ["git", "hg", "jj"]


def test_unknown_vcs_raises() -> None:
  with pytest.raises(ValueError):
    custom_lint_vcs.get_commit_message(None, vcs="svn")


def test_resolve_path_follows_cwd_changes(tmp_path: pathlib.Path) -> None:
  dir_a = tmp_path / "a"
  dir_b = tmp_path / "b"
  dir_a.mkdir()
  dir_b.mkdir()
  old_cwd = os.getcwd()
  try:
    os.chdir(dir_a)
    first = custom_lint_vcs.resolve_path(pathlib.Path("x"))
    os.chdir(dir_b)
    second = custom_lint_vcs.resolve_path(pathlib.Path("x"))
  finally:
    os.chdir(old_cwd)
  assert first == dir_a.resolve() / "x"
  assert second == dir_b.resolve() / "x"


def test_all_vcs_fail_raises_vcs_error_and_exit_2(
    rules_repo: tuple[pathlib.Path, pathlib.Path],
) -> None:
  root_path, rules_file = rules_repo
  mock_fail = mock.Mock(returncode=1, stdout="", stderr="fatal: error")
  with mock.patch(
      "subprocess.run", return_value=mock_fail, autospec=True
  ) as mock_run:
    with pytest.raises(custom_lint_vcs.VcsError) as exc_info:
      custom_lint.run_checks(
          root_dir=root_path,
          check_all=False,
          files=None,
          base_sha="origin/main",
          rules_file=str(rules_file),
      )
    err_str = str(exc_info.value)
    assert "could not compute the diff against origin/main: " in err_str
    assert "fatal: error" in err_str
    assert "; pass --all to scan everything" in err_str

    stderr_io = io.StringIO()
    with (
        mock.patch("sys.stderr", stderr_io),
        mock.patch(
            "sys.argv",
            [
                "custom_lint",
                "--root",
                str(root_path),
                "--base-sha",
                "origin/main",
                "--rules",
                str(rules_file),
            ],
        ),
    ):
      code = custom_lint.main()
      assert code == 2
      stderr_text = stderr_io.getvalue()
      assert "could not compute the diff against origin/main: " in stderr_text
      assert "fatal: error" in stderr_text
      assert "; pass --all to scan everything" in stderr_text

    stderr_env = io.StringIO()
    with (
        mock.patch("sys.stderr", stderr_env),
        mock.patch.dict(os.environ, {"BASE_SHA": "env_base_sha"}),
    ):
      code = custom_lint.main([
          "--root",
          str(root_path),
          "--rules",
          str(rules_file),
      ])
      assert code == 2
      stderr_env_text = stderr_env.getvalue()
      assert (
          "could not compute the diff against env_base_sha: " in stderr_env_text
      )
      assert "fatal: error" in stderr_env_text
      assert "; pass --all to scan everything" in stderr_env_text

    mock_run.reset_mock()
    with mock.patch("builtins.print", autospec=True):
      code = custom_lint.run_checks(
          root_dir=root_path,
          check_all=True,
          files=None,
          rules_file=str(rules_file),
      )
      assert code == 0
    mock_run.assert_not_called()

    test_file = root_path / "test.txt"
    test_file.write_text("clean\n", encoding="utf-8")
    with mock.patch("builtins.print", autospec=True):
      code = custom_lint.run_checks(
          root_dir=root_path,
          check_all=False,
          files=[str(test_file)],
          rules_file=str(rules_file),
      )
      assert code == 0
    mock_run.assert_not_called()


@pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)
def test_git_integration_uncommitted_edits_and_shallow_fallback(
    tmp_path: pathlib.Path,
) -> None:
  repo_dir = tmp_path.resolve()
  env = dict(os.environ)
  env.update({
      "GIT_AUTHOR_NAME": "Test",
      "GIT_AUTHOR_EMAIL": "test@example.com",
      "GIT_COMMITTER_NAME": "Test",
      "GIT_COMMITTER_EMAIL": "test@example.com",
  })

  def run_git(
      *args: str, check: bool = True
  ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )

  run_git("init", "-b", "main")
  run_git("config", "user.name", "Test")
  run_git("config", "user.email", "test@example.com")
  run_git("config", "diff.noprefix", "true")
  run_git("config", "diff.mnemonicPrefix", "true")

  (repo_dir / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = repo_dir / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.py" check_contents { base { id:'
      ' "NoBanned" } prohibited_regexp: "BANNED_TOKEN" failure_status: ERROR }'
      " }\n",
      encoding="utf-8",
  )
  file_a = repo_dir / "a.py"
  file_a.write_text("# initial clean file\n", encoding="utf-8")
  run_git("add", ".")
  run_git("commit", "-m", "initial commit")
  base_sha = run_git("rev-parse", "HEAD").stdout.strip()

  file_a.write_text(
      "# initial clean file\nx = 'BANNED_TOKEN'\n", encoding="utf-8"
  )

  old_cwd = os.getcwd()
  try:
    os.chdir(repo_dir)

    with mock.patch("builtins.print", autospec=True):
      exit_code = custom_lint.run_checks(
          root_dir=repo_dir,
          base_sha=base_sha,
          check_all=False,
          jobs=1,
          rules_file=str(rules_file),
          vcs="git",
      )
    assert exit_code == 1

    modified = custom_lint_vcs.get_modified_lines_from_vcs(
        base_sha, root_dir=repo_dir, vcs="git"
    )
    assert isinstance(modified, custom_lint_vcs.ModifiedLines)
    assert "a.py" in modified
    assert modified["a.py"] == {2}

    run_git("add", ".")
    run_git("commit", "-m", "commit with edit on main")

    run_git("checkout", "--orphan", "orphan_base")
    run_git("rm", "-rf", ".")
    file_orphan = repo_dir / "orphan.py"
    file_orphan.write_text("# unrelated root\n", encoding="utf-8")
    run_git("add", "orphan.py")
    run_git("commit", "-m", "orphan base commit")
    orphan_base_sha = run_git("rev-parse", "HEAD").stdout.strip()

    run_git("checkout", "main")

    mb_proc = run_git("merge-base", orphan_base_sha, "HEAD", check=False)
    assert mb_proc.returncode != 0

    modified_shallow = custom_lint_vcs.get_modified_lines_from_vcs(
        orphan_base_sha, root_dir=repo_dir, vcs="git"
    )
    assert isinstance(modified_shallow, custom_lint_vcs.ModifiedLines)
    assert "a.py" in modified_shallow

    with mock.patch("builtins.print", autospec=True):
      exit_code_shallow = custom_lint.run_checks(
          root_dir=repo_dir,
          base_sha=orphan_base_sha,
          check_all=False,
          jobs=1,
          rules_file=str(rules_file),
          vcs="git",
      )
    assert exit_code_shallow == 1
  finally:
    os.chdir(old_cwd)


@pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)
def test_git_shallow_pr_merge_commit_with_head_sha(
    tmp_path: pathlib.Path,
) -> None:
  origin_dir = (tmp_path / "origin").resolve()
  shallow_dir = (tmp_path / "shallow").resolve()
  origin_dir.mkdir()
  shallow_dir.mkdir()
  env = dict(os.environ)
  env.update({
      "GIT_AUTHOR_NAME": "Test",
      "GIT_AUTHOR_EMAIL": "test@example.com",
      "GIT_COMMITTER_NAME": "Test",
      "GIT_COMMITTER_EMAIL": "test@example.com",
  })

  def run_git(
      cwd: pathlib.Path, *args: str, check: bool = True
  ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )

  run_git(origin_dir, "init", "-b", "main")
  (origin_dir / "pyproject.toml").write_text("", encoding="utf-8")
  rules_file = origin_dir / "rules.METADATA"
  rules_file.write_text(
      'presubmit { path_regexp: ".*\\\\.py" check_contents { base { id:'
      ' "NoBanned" disable_tags: "ALLOW_BANNED" } prohibited_regexp:'
      ' "BANNED_TOKEN" failure_status: ERROR } }\n',
      encoding="utf-8",
  )
  (origin_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
  run_git(origin_dir, "add", ".")
  run_git(origin_dir, "commit", "-m", "initial main commit")
  base_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # Advance main with another commit so PR merge commit is non-fast-forward.
  (origin_dir / "other.py").write_text("y = 2\n", encoding="utf-8")
  run_git(origin_dir, "add", "other.py")
  run_git(origin_dir, "commit", "-m", "advance main")

  # Create PR branch from base_sha with a disable tag in the commit message.
  run_git(origin_dir, "checkout", "-b", "pr", base_sha)
  (origin_dir / "a.py").write_text("x = 'BANNED_TOKEN'\n", encoding="utf-8")
  run_git(origin_dir, "add", "a.py")
  run_git(
      origin_dir, "commit", "-m", "Add token\n\nALLOW_BANNED=needed for test"
  )
  head_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # Create synthetic PR merge commit on top of main (like refs/pull/N/merge).
  run_git(origin_dir, "checkout", "main")
  run_git(origin_dir, "merge", "--no-ff", "-m", "Merge PR into main", head_sha)
  merge_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # Simulate GitHub Actions depth-1 checkout of the synthetic merge commit,
  # followed by lint.yml's shallow fetch of head_sha and base_sha.
  run_git(shallow_dir, "init", "-b", "main")
  run_git(shallow_dir, "remote", "add", "origin", str(origin_dir))
  run_git(shallow_dir, "fetch", "--no-tags", "--depth=1", "origin", merge_sha)
  run_git(shallow_dir, "checkout", "--detach", "FETCH_HEAD")
  run_git(shallow_dir, "fetch", "--no-tags", "--depth=2", "origin", head_sha)
  run_git(shallow_dir, "fetch", "--no-tags", "--depth=1", "origin", base_sha)

  # Verify the shallow-clone failure mode without head_sha: HEAD^2 is missing
  # or `base..HEAD` only sees the synthetic merge commit without the PR tag.
  msg_without_head = custom_lint_vcs.get_commit_message(
      base_sha, root_dir=shallow_dir, vcs="git"
  )
  assert "ALLOW_BANNED" not in msg_without_head

  # With head_sha, `git log base..head_sha` sees the PR commit's disable tag.
  msg_with_head = custom_lint_vcs.get_commit_message(
      base_sha, root_dir=shallow_dir, vcs="git", head_sha=head_sha
  )
  assert "ALLOW_BANNED=needed for test" in msg_with_head

  with (
      mock.patch("sys.stdout", io.StringIO()),
      mock.patch("sys.stderr", io.StringIO()),
      mock.patch.dict(os.environ, {"BASE_SHA": base_sha, "HEAD_SHA": head_sha}),
  ):
    rc = custom_lint.main([
        "--root",
        str(shallow_dir),
        "--rules",
        str(shallow_dir / "rules.METADATA"),
        "--vcs",
        "git",
        "--jobs",
        "1",
    ])
  assert rc == 0


def test_commit_cached_message_and_tags() -> None:
  direct = custom_lint_vcs.Commit("Summary\n\nALLOW_FOO=bar\nROLLBACK_OF=123\n")
  assert direct.message == "Summary\n\nALLOW_FOO=bar\nROLLBACK_OF=123\n"
  assert direct.tags == frozenset({"ALLOW_FOO", "ROLLBACK_OF"})

  backend = custom_lint_vcs.HgVcs()
  mock_result = mock.Mock(
      returncode=0, stdout="CL desc\n\nALLOW_BAZ=ok\n", stderr=""
  )
  with mock.patch(
      "subprocess.run", return_value=mock_result, autospec=True
  ) as mock_run:
    commit = backend.get_commit()
    mock_run.assert_not_called()
    assert commit.tags == frozenset({"ALLOW_BAZ"})
    assert commit.message == "CL desc\n\nALLOW_BAZ=ok\n"
    # Second access to .tags hits the cached message without re-running VCS.
    assert commit.tags == frozenset({"ALLOW_BAZ"})
    mock_run.assert_called_once()


def test_hg_diff_falls_back_when_p4base_unknown() -> None:
  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if "-r" in cmd and "p4base" in cmd:
      return mock.Mock(
          returncode=255,
          stdout="",
          stderr="hg: parse error: unknown identifier: p4base\n",
      )
    return mock.Mock(
        returncode=0,
        stdout="+++ b/foo.py\n@@ -0,0 +1 @@\n+x\n",
        stderr="",
    )

  backend = custom_lint_vcs.HgVcs()
  with mock.patch(
      "subprocess.run", side_effect=fake_run, autospec=True
  ) as mock_run:
    diff_out = backend.diff()
    assert "+++ b/foo.py" in diff_out
    assert [c.args[0] for c in mock_run.call_args_list] == [
        ["hg", "diff", "--git", "-U0", "-r", "p4base"],
        ["hg", "diff", "--git", "-U0"],
    ]


def test_git_shallow_boundary_exclusions_mocked() -> None:
  def fake_run(cmd: Sequence[str], **kwargs: object) -> mock.Mock:
    del kwargs
    if cmd[:2] == ["git", "merge-base"]:
      return mock.Mock(returncode=1, stdout="", stderr="")
    if cmd == ["git", "rev-parse", "--verify", "head123"]:
      return mock.Mock(returncode=0, stdout="head123\n", stderr="")
    if cmd == ["git", "rev-list", "--max-parents=0", "head123"]:
      return mock.Mock(returncode=0, stdout="branchpoint000\n", stderr="")
    if cmd == [
        "git",
        "log",
        "base456..head123",
        "^branchpoint000",
        "--format=%B",
    ]:
      return mock.Mock(
          returncode=0, stdout="External PR\n\nALLOW_X=1\n", stderr=""
      )
    return mock.Mock(returncode=1, stdout="", stderr="")

  backend = custom_lint_vcs.GitVcs()
  with mock.patch(
      "subprocess.run", side_effect=fake_run, autospec=True
  ) as mock_run:
    msg = backend.log(base="base456", head="head123")
    assert msg == "External PR\n\nALLOW_X=1\n"
    assert [c.args[0] for c in mock_run.call_args_list] == [
        ["git", "merge-base", "base456", "head123"],
        ["git", "rev-parse", "--verify", "head123"],
        ["git", "rev-list", "--max-parents=0", "head123"],
        [
            "git",
            "log",
            "base456..head123",
            "^branchpoint000",
            "--format=%B",
        ],
    ]


@pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)
def test_git_shallow_pr_when_main_advances_excludes_branch_point_commit(
    tmp_path: pathlib.Path,
) -> None:
  origin_dir = tmp_path / "origin"
  shallow_dir = tmp_path / "shallow"
  origin_dir.mkdir()
  shallow_dir.mkdir()

  def run_git(
      cwd: pathlib.Path, *args: str
  ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )

  run_git(origin_dir, "init", "-b", "main")
  (origin_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
  run_git(origin_dir, "add", "a.py")
  # Branch-point commit M on main was exported by Copybara ("Generated from cl/100").
  run_git(origin_dir, "commit", "-m", "Initial sync\n\nGenerated from cl/100")
  branch_point_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # Advance main by 2 commits so base_sha (tip of main) at --depth=1 cannot reach M.
  (origin_dir / "other.py").write_text("y = 2\n", encoding="utf-8")
  run_git(origin_dir, "add", "other.py")
  run_git(origin_dir, "commit", "-m", "main commit 1")
  (origin_dir / "other.py").write_text("y = 3\n", encoding="utf-8")
  run_git(origin_dir, "add", "other.py")
  run_git(origin_dir, "commit", "-m", "main commit 2")
  base_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # External PR branches from M (branch_point_sha) without Copybara marker.
  run_git(origin_dir, "checkout", "-b", "pr", branch_point_sha)
  (origin_dir / "a.py").write_text("x = 2\n", encoding="utf-8")
  run_git(origin_dir, "add", "a.py")
  run_git(origin_dir, "commit", "-m", "External contributor PR\n\nALLOW_X=1")
  head_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  run_git(origin_dir, "checkout", "main")
  run_git(origin_dir, "merge", "--no-ff", "-m", "Merge PR", head_sha)
  merge_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  run_git(shallow_dir, "init", "-b", "main")
  run_git(shallow_dir, "remote", "add", "origin", str(origin_dir))
  run_git(shallow_dir, "fetch", "--no-tags", "--depth=1", "origin", merge_sha)
  run_git(shallow_dir, "checkout", "--detach", "FETCH_HEAD")
  # PR_COMMITS=1 -> fetch head_sha with --depth=2 (fetches head_sha + M)
  run_git(shallow_dir, "fetch", "--no-tags", "--depth=2", "origin", head_sha)
  # Fetch base_sha (tip of main) with --depth=1
  run_git(shallow_dir, "fetch", "--no-tags", "--depth=1", "origin", base_sha)

  msg = custom_lint_vcs.get_commit_message(
      base_sha, root_dir=shallow_dir, vcs="git", head_sha=head_sha
  )
  assert "External contributor PR" in msg
  assert "ALLOW_X=1" in msg
  assert "Generated from cl/100" not in msg


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
