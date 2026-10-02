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

"""Tests for custom_linter.vcs."""

from collections.abc import Sequence
import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
from typing import Final
from unittest import mock

import pytest
from ci.tools.custom_linter import vcs as custom_lint_vcs

# ---------------------------------------------------------------------------
# Unified diff parsing tests
# ---------------------------------------------------------------------------


def test_parse_unified_diff() -> None:
  diff_text = textwrap.dedent("""\
      +++ b/foo/BUILD.bazel\t2026-08-26 23:52:54.000000000 +0000
      @@ -10,0 +11,3 @@
      +cc_library(
      +    name = "lib",
      +)
      +++ b/bar/BUILD
      @@ -5 +5 @@
      -py_library(name = "x")
      +cc_test(name = "x")
      +++ b/baz/file.txt   2026-08-26 18:14:30.000000000 -0600
      @@ -1 +1 @@
      -old
      +new
      +++ b/simple/main.py
      @@ -10 +10,2 @@
      -# old code
      +# new code 1
      +# new code 2
      +++ b/deletion/file.py
      @@ -10,3 +10,0 @@
      -line 10
      -line 11
      -line 12
      +++ /dev/null\tWed Aug 26 23:52:54 2026
      @@ -1,5 +0,0 @@
      -deleted_line
      """)
  modified = custom_lint_vcs.parse_unified_diff(diff_text)
  assert "foo/BUILD.bazel" in modified
  assert modified["foo/BUILD.bazel"] == {11, 12, 13}
  assert "bar/BUILD" in modified
  assert modified["bar/BUILD"] == {5}
  assert "baz/file.txt" in modified
  assert modified["baz/file.txt"] == {1}
  assert "simple/main.py" in modified
  assert modified["simple/main.py"] == {10, 11}
  assert "deletion/file.py" in modified
  assert modified["deletion/file.py"] == set()
  assert "/dev/null" not in modified
  assert "dev/null" not in modified


def test_file_operations_from_git_diff() -> None:
  diff_text = textwrap.dedent("""\
      diff --git a/p/BUILD b/p/BUILD
      --- a/p/BUILD
      +++ b/p/BUILD
      @@ -43,0 +44,1 @@
      +a
      diff --git a/new.py b/new.py
      new file mode 100644
      --- /dev/null
      +++ b/new.py
      @@ -0,0 +1,2 @@
      +x
      +--- not a header
      diff --git a/gone.cc b/gone.cc
      deleted file mode 100644
      --- a/gone.cc
      +++ /dev/null
      @@ -1,2 +0,0 @@
      -a
      -b
      diff --git a/old.cc b/moved.cc
      similarity index 100%
      rename from old.cc
      rename to moved.cc
      diff --git a/src.h b/copy.h
      similarity index 90%
      copy from src.h
      copy to copy.h
      --- a/src.h
      +++ b/copy.h
      @@ -3 +3 @@
      -old
      +new
      """)
  modified = custom_lint_vcs.parse_unified_diff(diff_text)
  assert modified.operations == {
      "p/BUILD": "EDIT",
      "new.py": "ADD",
      "gone.cc": "DELETE",
      "moved.cc": "BRANCH",
      "copy.h": "ADD",
  }
  assert modified["new.py"] == {1, 2}
  assert modified["moved.cc"] == set()
  assert "gone.cc" not in modified
  assert "not a header" not in "".join(modified)


def test_parse_unified_diff_delete_then_recreate() -> None:
  diff_text = textwrap.dedent("""\
      diff --git a/recreated.py b/recreated.py
      deleted file mode 100644
      --- a/recreated.py
      +++ /dev/null
      @@ -1,2 +0,0 @@
      -old_a
      ---- looks_like_header_in_deleted_hunk
      diff --git a/recreated.py b/recreated.py
      new file mode 100644
      --- /dev/null
      +++ b/recreated.py
      @@ -0,0 +1,2 @@
      +new_a
      +new_b
      """)
  modified = custom_lint_vcs.parse_unified_diff(diff_text)
  assert modified["recreated.py"] == {1, 2}
  assert modified.operations == {"recreated.py": "ADD"}


def test_new_file_in_plain_diff() -> None:
  diff_text = "--- /dev/null\n+++ b/a.py\n@@ -0,0 +1 @@\n+x\n"
  modified = custom_lint_vcs.parse_unified_diff(diff_text)
  assert modified.operations == {"a.py": "ADD"}


def test_parse_unified_diff_with_prefix_strips_and_filters() -> None:
  prefix = "sub/proj/"
  diff_text = textwrap.dedent(f"""\
      diff --git a/{prefix}foo/BUILD.bazel b/{prefix}foo/BUILD.bazel
      --- a/{prefix}foo/BUILD.bazel
      +++ b/{prefix}foo/BUILD.bazel
      @@ -10,0 +11,3 @@
      +cc_library(
      +    name = "lib",
      +)
      +++ {prefix}bar/BUILD
      @@ -5 +5 @@
      -py_library(name = "x")
      +cc_test(name = "x")
      diff --git a/other_pkg/outside.py b/other_pkg/outside.py
      new file mode 100644
      --- /dev/null
      +++ b/other_pkg/outside.py
      @@ -0,0 +1,2 @@
      +x = 1
      +y = 2
      """)
  modified = custom_lint_vcs.parse_unified_diff(diff_text, prefix=prefix)
  assert set(modified) == {"foo/BUILD.bazel", "bar/BUILD"}
  assert modified["foo/BUILD.bazel"] == {11, 12, 13}
  assert modified["bar/BUILD"] == {5}
  assert "other_pkg/outside.py" not in modified
  assert "outside.py" not in modified


def test_normalize_diff_path(tmp_path: pathlib.Path) -> None:
  assert (
      custom_lint_vcs.normalize_diff_path("b/pkg/BUILD.bazel")
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint_vcs.normalize_diff_path("a/pkg/BUILD.bazel")
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint_vcs.normalize_diff_path(
          "b/sub/proj/pkg/BUILD.bazel", prefix="sub/proj/"
      )
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint_vcs.normalize_diff_path(
          "a/sub/proj/pkg/BUILD.bazel", prefix="sub/proj"
      )
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint_vcs.normalize_diff_path(
          "b/other/pkg/BUILD.bazel", prefix="sub/proj/"
      )
      is None
  )
  assert custom_lint_vcs.normalize_diff_path("/dev/null") is None
  assert custom_lint_vcs.normalize_diff_path("dev/null") is None
  assert custom_lint_vcs.normalize_diff_path("") is None
  root = tmp_path.resolve()
  assert (
      custom_lint_vcs.normalize_diff_path(
          str(root / "pkg/BUILD.bazel"), root_dir=root
      )
      == "pkg/BUILD.bazel"
  )
  assert (
      custom_lint_vcs.normalize_diff_path(
          "/outside/root/BUILD.bazel", root_dir=root
      )
      is None
  )


# ---------------------------------------------------------------------------
# VCS backend & detection tests
# ---------------------------------------------------------------------------


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
        (custom_lint_vcs.VcsMode.GIT, custom_lint_vcs.GitVcs),
        (custom_lint_vcs.VcsMode.HG, custom_lint_vcs.HgVcs),
        (custom_lint_vcs.VcsMode.JJ, custom_lint_vcs.JjVcs),
    ],
)
def test_for_name_returns_expected_backend(
    name: custom_lint_vcs.VcsMode, expected_cls: type[custom_lint_vcs.Vcs]
) -> None:
  backend = custom_lint_vcs.for_name(name)
  assert isinstance(backend, expected_cls)
  assert backend.name == name


def test_for_name_any() -> None:
  assert (
      custom_lint_vcs.for_name(custom_lint_vcs.VcsMode.ANY).name
      == custom_lint_vcs.VcsMode.ANY
  )


def test_any_vcs_reuses_active_backend_across_diff_and_commit_message() -> None:
  any_vcs = custom_lint_vcs.for_name(custom_lint_vcs.VcsMode.ANY)
  calls: list[str] = []

  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    calls.append(cmd[0])
    if cmd[0] == "git":
      return subprocess.CompletedProcess(
          args=cmd, returncode=1, stdout="", stderr="not git"
      )
    if cmd[0] == "hg":
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="hg output\n", stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr="unexpected"
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    assert any_vcs.diff() == "hg output\n"
    assert any_vcs.get_commit_message() == "hg output\n"
    assert any_vcs.diff() == "hg output\n"
  # First diff tries git then hg; subsequent calls go straight to hg.
  assert calls == ["git", "hg", "hg", "hg"]

  any_vcs_msg_first = custom_lint_vcs.for_name(custom_lint_vcs.VcsMode.ANY)
  calls_msg_first: list[str] = []

  def fake_run_msg_first(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    calls_msg_first.append(cmd[0])
    if cmd[0] == "git":
      return subprocess.CompletedProcess(
          args=cmd, returncode=1, stdout="", stderr="not git"
      )
    if cmd[0] == "hg":
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="hg desc\n", stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr="unexpected"
    )

  with mock.patch(
      "subprocess.run", side_effect=fake_run_msg_first, autospec=True
  ):
    assert any_vcs_msg_first.get_commit_message() == "hg desc\n"
    assert any_vcs_msg_first.diff() == "hg desc\n"
  assert calls_msg_first == ["git", "hg", "hg"]


_GIT_DIFF_FLAGS: Final[Sequence[str]] = (
    "git",
    "diff",
    "-U0",
    "--no-color",
    "--no-ext-diff",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


def test_get_modified_lines_git_merge_base_success(
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

  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd == ["git", "merge-base", "origin/main", "HEAD"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="abc1234\n", stderr=""
      )
    if cmd == [*_GIT_DIFF_FLAGS, "abc1234"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=git_diff_output, stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    modified = custom_lint_vcs.for_name(
        custom_lint_vcs.VcsMode.GIT, sub
    ).get_modified_lines("origin/main", root_dir=sub)
    assert isinstance(modified, custom_lint_vcs.ModifiedLines)
    assert set(modified) == {"foo/BUILD"}
    assert modified["foo/BUILD"] == {11, 12}
    assert "sub/proj/foo/BUILD" not in modified
    assert "other/ignored.py" not in modified


def test_get_modified_lines_git_merge_base_failure_raises_vcs_error(
    tmp_path: pathlib.Path,
) -> None:
  repo = tmp_path.resolve()
  (repo / ".git").mkdir()
  calls: list[Sequence[str]] = []

  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    calls.append(cmd)
    if cmd == ["git", "merge-base", "origin/main", "HEAD"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=128, stdout="", stderr="fatal: no merge base"
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    with pytest.raises(
        custom_lint_vcs.VcsError, match="git merge-base.*fatal: no merge base"
    ):
      custom_lint_vcs.for_name(
          custom_lint_vcs.VcsMode.GIT, repo
      ).get_modified_lines("origin/main", root_dir=repo)
  assert calls == [["git", "merge-base", "origin/main", "HEAD"]]


@pytest.mark.parametrize(
    "raw_base", ["origin/main...", "origin/main..HEAD", "origin/main.."]
)
def test_get_modified_lines_git_with_existing_dots(
    tmp_path: pathlib.Path,
    raw_base: str,
) -> None:
  repo = tmp_path.resolve()
  (repo / ".git").mkdir()

  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd == ["git", "merge-base", "origin/main", "HEAD"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="def5678\n", stderr=""
      )
    if cmd == [*_GIT_DIFF_FLAGS, "def5678"]:
      return subprocess.CompletedProcess(
          args=cmd,
          returncode=0,
          stdout="+++ b/a.py\n@@ -0,0 +1 @@\n+x\n",
          stderr="",
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    modified = custom_lint_vcs.for_name(
        custom_lint_vcs.VcsMode.GIT, repo
    ).get_modified_lines(raw_base, root_dir=repo)
    assert modified["a.py"] == {1}


def test_get_commit_message() -> None:
  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd == ["git", "merge-base", "origin/main", "HEAD"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="abc1234\n", stderr=""
      )
    if cmd == ["git", "log", "origin/main..HEAD", "--format=%B"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="feat: new feature\n", stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    msg = custom_lint_vcs.for_name(
        custom_lint_vcs.VcsMode.GIT
    ).get_commit_message("origin/main...")
    assert msg == "feat: new feature\n"


def test_get_commit_message_local_merge_uses_head_not_second_parent() -> None:
  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd == ["git", "merge-base", "origin/main", "HEAD"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="abc1234\n", stderr=""
      )
    if cmd == ["git", "log", "origin/main..HEAD", "--format=%B"]:
      return subprocess.CompletedProcess(
          args=cmd,
          returncode=0,
          stdout="PR commit\n\nALLOW_FOO=reason\n",
          stderr="",
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    msg = custom_lint_vcs.for_name(
        custom_lint_vcs.VcsMode.GIT
    ).get_commit_message("origin/main")
    assert msg == "PR commit\n\nALLOW_FOO=reason\n"


def test_get_commit_message_empty_stdout_returns_empty() -> None:
  empty_result = subprocess.CompletedProcess(
      args=[], returncode=0, stdout="", stderr=""
  )
  with mock.patch("subprocess.run", return_value=empty_result, autospec=True):
    assert not custom_lint_vcs.for_name(
        custom_lint_vcs.VcsMode.HG
    ).get_commit_message()


def test_get_commit_message_all_fail_raises_vcs_error() -> None:
  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    return subprocess.CompletedProcess(
        args=cmd, returncode=128, stdout="", stderr=f"fatal: {cmd[0]} failed"
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    with pytest.raises(custom_lint_vcs.VcsError) as exc_info:
      custom_lint_vcs.for_name(custom_lint_vcs.VcsMode.ANY).get_commit_message(
          "origin/main"
      )
    err_msg = str(exc_info.value)
    assert "git merge-base origin/main HEAD" in err_msg
    assert "fatal: git failed" in err_msg
    assert "hg log -r . --template {desc}" in err_msg
    assert "fatal: hg failed" in err_msg


def test_get_modified_lines_hg(tmp_path: pathlib.Path) -> None:
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

  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd[0] == "git":
      return subprocess.CompletedProcess(
          args=cmd, returncode=1, stdout="", stderr="fatal: not a git repo"
      )
    if cmd == [
        "hg",
        "diff",
        "--git",
        "-U0",
        "-r",
        "first(present(p4base) + .)",
    ]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=hg_diff_output, stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    modified = custom_lint_vcs.for_name(
        custom_lint_vcs.VcsMode.ANY, sub
    ).get_modified_lines(root_dir=sub)
    assert isinstance(modified, custom_lint_vcs.ModifiedLines)
    assert set(modified) == {"common/BUILD"}
    assert modified["common/BUILD"] == {5}
    assert "ws/pkg/common/BUILD" not in modified
    assert "ws/other/BUILD" not in modified


def test_get_modified_lines_jj(tmp_path: pathlib.Path) -> None:
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

  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd[0] in ("git", "hg"):
      return subprocess.CompletedProcess(
          args=cmd, returncode=1, stdout="", stderr="not a git/hg repo"
      )
    if cmd[0] == "jj":
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=jj_diff_output, stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    modified = custom_lint_vcs.for_name(
        custom_lint_vcs.VcsMode.ANY, sub
    ).get_modified_lines(root_dir=sub)
    assert isinstance(modified, custom_lint_vcs.ModifiedLines)
    assert set(modified) == {"csrc/BUILD"}
    assert modified["csrc/BUILD"] == {9, 10}


def test_explicit_vcs_runs_only_that_vcs() -> None:
  fail_result = subprocess.CompletedProcess(
      args=[], returncode=1, stdout="", stderr="abort: no repo"
  )
  with mock.patch(
      "subprocess.run", return_value=fail_result, autospec=True
  ) as mock_run:
    with pytest.raises(custom_lint_vcs.VcsError, match="abort: no repo"):
      custom_lint_vcs.for_name(custom_lint_vcs.VcsMode.HG).get_modified_lines()
    with pytest.raises(custom_lint_vcs.VcsError, match="abort: no repo"):
      custom_lint_vcs.for_name(custom_lint_vcs.VcsMode.HG).get_commit_message()
  assert [c.args[0][0] for c in mock_run.call_args_list] == ["hg", "hg"]


def test_get_modified_lines_error_contains_backend_stderr() -> None:
  fail_result = subprocess.CompletedProcess(
      args=[],
      returncode=128,
      stdout="",
      stderr="fatal: bad revision 'origin/main'",
  )
  with mock.patch("subprocess.run", return_value=fail_result, autospec=True):
    with pytest.raises(custom_lint_vcs.VcsError) as exc_info:
      custom_lint_vcs.for_name(custom_lint_vcs.VcsMode.GIT).get_modified_lines(
          "origin/main"
      )
  err_msg = str(exc_info.value)
  assert "git merge-base origin/main HEAD" in err_msg
  assert "fatal: bad revision 'origin/main'" in err_msg

  def fake_diff_fail(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd == ["git", "merge-base", "origin/main", "HEAD"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="abc1234\n", stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=128, stdout="", stderr="fatal: unable to read tree"
    )

  with mock.patch("subprocess.run", side_effect=fake_diff_fail, autospec=True):
    with pytest.raises(custom_lint_vcs.VcsError) as diff_exc:
      custom_lint_vcs.for_name(custom_lint_vcs.VcsMode.GIT).get_modified_lines(
          "origin/main"
      )
  diff_err = str(diff_exc.value)
  assert (
      "git diff -U0 --no-color --no-ext-diff --src-prefix=a/ --dst-prefix=b/"
      " abc1234"
      in diff_err
  )
  assert "fatal: unable to read tree" in diff_err


def test_auto_uses_detected_vcs() -> None:
  ok_result = subprocess.CompletedProcess(
      args=[], returncode=0, stdout="desc\n", stderr=""
  )
  with (
      mock.patch.object(
          custom_lint_vcs,
          "detect",
          return_value=custom_lint_vcs.JjVcs(),
          autospec=True,
      ),
      mock.patch(
          "subprocess.run", return_value=ok_result, autospec=True
      ) as mock_run,
  ):
    assert (
        custom_lint_vcs.for_name(
            custom_lint_vcs.VcsMode.AUTO
        ).get_commit_message()
        == "desc\n"
    )
  mock_run.assert_called_once()
  assert mock_run.call_args.args[0][0] == "jj"


def test_auto_without_detected_vcs_tries_all() -> None:
  fail_result = subprocess.CompletedProcess(
      args=[], returncode=1, stdout="", stderr="backend failure"
  )
  with (
      mock.patch.object(
          custom_lint_vcs, "detect", return_value=None, autospec=True
      ),
      mock.patch(
          "subprocess.run", return_value=fail_result, autospec=True
      ) as mock_run,
  ):
    with pytest.raises(custom_lint_vcs.VcsError, match="backend failure"):
      custom_lint_vcs.for_name(
          custom_lint_vcs.VcsMode.AUTO
      ).get_modified_lines()
  assert [c.args[0][0] for c in mock_run.call_args_list] == ["git", "hg", "jj"]


def test_unknown_vcs_raises() -> None:
  with pytest.raises(ValueError):
    custom_lint_vcs.for_name("svn")  # type: ignore[arg-type]


def test_resolve_path_follows_cwd_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  dir_a = tmp_path / "a"
  dir_b = tmp_path / "b"
  dir_a.mkdir()
  dir_b.mkdir()
  monkeypatch.chdir(dir_a)
  first = custom_lint_vcs.resolve_path(pathlib.Path("x"))
  monkeypatch.chdir(dir_b)
  second = custom_lint_vcs.resolve_path(pathlib.Path("x"))
  assert first == dir_a.resolve() / "x"
  assert second == dir_b.resolve() / "x"


@pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)
def test_git_integration_uncommitted_edits_and_disconnected_base(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
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

  file_a = repo_dir / "a.py"
  file_a.write_text("# initial clean file\n", encoding="utf-8")
  run_git("add", ".")
  run_git("commit", "-m", "initial commit")
  base_sha = run_git("rev-parse", "HEAD").stdout.strip()

  file_a.write_text(
      "# initial clean file\nx = 'BANNED_TOKEN'\n", encoding="utf-8"
  )

  monkeypatch.chdir(repo_dir)
  modified = custom_lint_vcs.GitVcs(repo_dir).get_modified_lines(
      base_sha, root_dir=repo_dir
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

  with pytest.raises(custom_lint_vcs.VcsError, match="git merge-base"):
    custom_lint_vcs.GitVcs(repo_dir).get_modified_lines(
        orphan_base_sha, root_dir=repo_dir
    )


@pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)
def test_git_pr_merge_commit_with_head_sha(
    tmp_path: pathlib.Path,
) -> None:
  origin_dir = (tmp_path / "origin").resolve()
  checkout_dir = (tmp_path / "checkout").resolve()
  origin_dir.mkdir()
  checkout_dir.mkdir()
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
  (origin_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
  run_git(origin_dir, "add", ".")
  run_git(origin_dir, "commit", "-m", "initial main commit")
  branch_point_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # Advance main with another commit so PR merge commit is non-fast-forward.
  (origin_dir / "other.py").write_text("y = 2\n", encoding="utf-8")
  run_git(origin_dir, "add", "other.py")
  run_git(origin_dir, "commit", "-m", "advance main")
  base_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # Create PR branch from branch_point_sha with a disable tag on the first
  # commit and a follow-up commit without the tag.
  run_git(origin_dir, "checkout", "-b", "pr", branch_point_sha)
  (origin_dir / "a.py").write_text("x = 'BANNED_TOKEN'\n", encoding="utf-8")
  run_git(origin_dir, "add", "a.py")
  run_git(
      origin_dir, "commit", "-m", "Add token\n\nALLOW_BANNED=needed for test"
  )
  (origin_dir / "a.py").write_text(
      "x = 'BANNED_TOKEN'\ny = 'clean'\n", encoding="utf-8"
  )
  run_git(origin_dir, "add", "a.py")
  run_git(origin_dir, "commit", "-m", "Follow-up cleanup")
  head_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # Create synthetic PR merge commit on top of main (like refs/pull/N/merge).
  run_git(origin_dir, "checkout", "main")
  run_git(
      origin_dir,
      "merge",
      "--no-ff",
      "-m",
      "Merge PR into main\n\nSYNTHETIC_MERGE_TAG=ignore",
      head_sha,
  )
  merge_sha = run_git(origin_dir, "rev-parse", "HEAD").stdout.strip()

  # Simulate GitHub Actions full-history checkout (`fetch-depth: 0`) of the
  # synthetic merge commit, plus head_sha and base_sha.
  run_git(checkout_dir, "init", "-b", "main")
  run_git(checkout_dir, "remote", "add", "origin", str(origin_dir))
  run_git(checkout_dir, "fetch", "--no-tags", "origin", merge_sha)
  run_git(checkout_dir, "checkout", "--detach", "FETCH_HEAD")
  run_git(checkout_dir, "fetch", "--no-tags", "origin", head_sha)
  run_git(checkout_dir, "fetch", "--no-tags", "origin", base_sha)

  git_vcs = custom_lint_vcs.GitVcs(checkout_dir)
  # Without head_sha, `base..HEAD` includes GitHub's synthetic merge commit.
  commit_without_head = git_vcs.get_commit(base_sha)
  assert "SYNTHETIC_MERGE_TAG" in commit_without_head.tags

  # With head_sha, `git log base..head_sha` aggregates all PR commits while
  # excluding GitHub's synthetic merge commit.
  commit_with_head = git_vcs.get_commit(base_sha, head=head_sha)
  assert "ALLOW_BANNED=needed for test" in commit_with_head.message
  assert "Follow-up cleanup" in commit_with_head.message
  assert "Merge PR into main" not in commit_with_head.message
  assert commit_with_head.tags == frozenset({"ALLOW_BANNED"})
  modified = git_vcs.get_modified_lines(base_sha, root_dir=checkout_dir)
  assert set(modified) == {"a.py"}
  assert modified["a.py"] == {1, 2}


def test_commit_cached_message_and_tags() -> None:
  direct = custom_lint_vcs.Commit("Summary\n\nALLOW_FOO=bar\nROLLBACK_OF=123\n")
  assert direct.message == "Summary\n\nALLOW_FOO=bar\nROLLBACK_OF=123\n"
  assert direct.tags == frozenset({"ALLOW_FOO", "ROLLBACK_OF"})
  with pytest.raises(dataclasses.FrozenInstanceError):
    direct.message = "mutated"  # type: ignore[misc]

  backend = custom_lint_vcs.HgVcs()
  ok_result = subprocess.CompletedProcess(
      args=[], returncode=0, stdout="CL desc\n\nALLOW_BAZ=ok\n", stderr=""
  )
  with mock.patch(
      "subprocess.run", return_value=ok_result, autospec=True
  ) as mock_run:
    commit = backend.get_commit()
    mock_run.assert_called_once()
    assert commit.message == "CL desc\n\nALLOW_BAZ=ok\n"
    first_tags = commit.tags
    assert first_tags == frozenset({"ALLOW_BAZ"})
    # Second access to .tags returns the cached object without re-parsing.
    assert commit.tags is first_tags


def test_hg_diff_uses_present_p4base_revset() -> None:
  def fake_run(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd == [
        "hg",
        "diff",
        "--git",
        "-U0",
        "-r",
        "first(present(p4base) + .)",
    ]:
      return subprocess.CompletedProcess(
          args=cmd,
          returncode=0,
          stdout="+++ b/foo.py\n@@ -0,0 +1 @@\n+x\n",
          stderr="",
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  backend = custom_lint_vcs.HgVcs()
  with mock.patch("subprocess.run", side_effect=fake_run, autospec=True):
    diff_out = backend.diff()
    assert "+++ b/foo.py" in diff_out


@pytest.mark.skipif(
    shutil.which("hg") is None, reason="hg executable not available"
)
def test_hg_plain_repo_without_p4base_falls_back_to_working_copy_parent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("HGRCPATH", os.devnull)
  repo_dir = tmp_path.resolve()
  subprocess.run(["hg", "init"], cwd=repo_dir, check=True, capture_output=True)
  file_a = repo_dir / "a.py"
  file_a.write_text("x = 1\n", encoding="utf-8")
  subprocess.run(
      ["hg", "add", "a.py"], cwd=repo_dir, check=True, capture_output=True
  )
  subprocess.run(
      ["hg", "commit", "-u", "Test <test@example.com>", "-m", "init"],
      cwd=repo_dir,
      check=True,
      capture_output=True,
  )
  file_a.write_text("x = 2\n", encoding="utf-8")

  modified = custom_lint_vcs.HgVcs(repo_dir).get_modified_lines(
      root_dir=repo_dir
  )
  assert set(modified) == {"a.py"}
  assert modified["a.py"] == {1}


def test_git_missing_binary_raises_vcs_error() -> None:
  backend = custom_lint_vcs.GitVcs()
  for exc in (
      FileNotFoundError(2, "No such file or directory", "git"),
      PermissionError(13, "Permission denied", "git"),
  ):
    with mock.patch(
        "subprocess.run",
        side_effect=exc,
        autospec=True,
    ):
      with pytest.raises(custom_lint_vcs.VcsError, match="git merge-base"):
        backend.diff("origin/main")
      with pytest.raises(custom_lint_vcs.VcsError, match="git merge-base"):
        backend.get_commit_message(base="origin/main", head="abc123")
      with pytest.raises(custom_lint_vcs.VcsError, match="git merge-base"):
        backend.get_commit_message(base="origin/main")
      with pytest.raises(custom_lint_vcs.VcsError, match="git log"):
        backend.get_commit_message(head="abc123")
      with pytest.raises(custom_lint_vcs.VcsError, match="git log"):
        backend.get_commit_message()


@pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)
def test_git_diff_and_commit_message_tolerate_non_utf8_bytes(
    tmp_path: pathlib.Path,
) -> None:
  repo_dir = tmp_path.resolve()

  def run_git(*args: str) -> subprocess.CompletedProcess[bytes]:
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
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

  run_git("init", "-b", "main")
  file_a = repo_dir / "a.cc"
  file_a.write_bytes(b"// line 1\nint x = 1;\n")
  run_git("add", "a.cc")
  run_git("commit", "-m", "initial")
  base_sha = (
      run_git("rev-parse", "HEAD")
      .stdout.decode("utf-8", errors="replace")
      .strip()
  )

  # Commit non-UTF-8 bytes in both file diff and commit message.
  file_a.write_bytes(b"// line 1\n// caf\xe9 \xff\nint x = 2;\n")
  msg_file = tmp_path / "commit_msg.bin"
  msg_file.write_bytes(b"Update caf\xe9\n\nALLOW_NON_UTF8=reason\n")
  run_git("add", "a.cc")
  run_git("commit", "-F", str(msg_file))

  git_vcs = custom_lint_vcs.GitVcs(repo_dir)
  modified = git_vcs.get_modified_lines(base_sha, root_dir=repo_dir)
  assert set(modified) == {"a.cc"}
  assert modified["a.cc"] == {2, 3}

  commit = git_vcs.get_commit(base_sha)
  assert "\ufffd" in commit.message
  assert commit.tags == frozenset({"ALLOW_NON_UTF8"})


@pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)
def test_git_local_merge_from_main_preserves_feature_disable_tag(
    tmp_path: pathlib.Path,
) -> None:
  repo_dir = tmp_path.resolve()

  def run_git(*args: str) -> subprocess.CompletedProcess[str]:
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
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )

  run_git("init", "-b", "main")
  (repo_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
  run_git("add", "a.py")
  run_git("commit", "-m", "initial")

  run_git("checkout", "-b", "feature")
  (repo_dir / "a.py").write_text("x = 2\n", encoding="utf-8")
  run_git("add", "a.py")
  run_git("commit", "-m", "Feature work\n\nALLOW_FEATURE=needed")

  run_git("checkout", "main")
  (repo_dir / "other.py").write_text("y = 1\n", encoding="utf-8")
  run_git("add", "other.py")
  run_git("commit", "-m", "Upstream sync\n\nGenerated from cl/999")

  run_git("checkout", "feature")
  run_git("merge", "--no-ff", "-m", "Merge branch 'main' into feature", "main")

  msg = custom_lint_vcs.GitVcs(repo_dir).get_commit_message("main")
  assert "ALLOW_FEATURE=needed" in msg
  assert "Generated from cl/999" not in msg


def test_git_get_commit_message_with_head_and_error() -> None:
  def fake_run_ok(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd == ["git", "merge-base", "base456", "head123"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="mb000\n", stderr=""
      )
    if cmd == ["git", "log", "base456..head123", "--format=%B"]:
      return subprocess.CompletedProcess(
          args=cmd,
          returncode=0,
          stdout="External PR\n\nALLOW_X=1\n",
          stderr="",
      )
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr="unexpected command"
    )

  backend = custom_lint_vcs.GitVcs()
  with mock.patch("subprocess.run", side_effect=fake_run_ok, autospec=True):
    msg = backend.get_commit_message(base="base456", head="head123")
    assert msg == "External PR\n\nALLOW_X=1\n"

  no_mb_calls: list[Sequence[str]] = []

  def fake_no_merge_base(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    no_mb_calls.append(cmd)
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr=""
    )

  with mock.patch(
      "subprocess.run", side_effect=fake_no_merge_base, autospec=True
  ):
    with pytest.raises(
        custom_lint_vcs.VcsError, match="git merge-base base456 head123"
    ):
      backend.get_commit_message(base="base456", head="head123")
  assert no_mb_calls == [["git", "merge-base", "base456", "head123"]]

  def fake_log_err(
      cmd: Sequence[str], **kwargs: object
  ) -> subprocess.CompletedProcess[str]:
    del kwargs
    if cmd == ["git", "merge-base", "base456", "head123"]:
      return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout="mb000\n", stderr=""
      )
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=128,
        stdout="",
        stderr="fatal: bad revision 'base456..head123'",
    )

  with mock.patch("subprocess.run", side_effect=fake_log_err, autospec=True):
    with pytest.raises(
        custom_lint_vcs.VcsError, match="git log.*fatal: bad revision"
    ):
      backend.get_commit_message(base="base456", head="head123")


@pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)
def test_git_pr_when_main_advances_excludes_main_and_rejects_shallow_disconnect(
    tmp_path: pathlib.Path,
) -> None:
  origin_dir = (tmp_path / "origin").resolve()
  full_dir = (tmp_path / "full").resolve()
  shallow_dir = (tmp_path / "shallow").resolve()
  origin_dir.mkdir()
  full_dir.mkdir()
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

  # Advance main by 2 commits touching other.py after the PR branched.
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

  # Full history checkout of the PR head (matching `fetch-depth: 0`):
  # merge-base resolves to branch_point_sha so untouched `other.py` on main is
  # not flagged, and `base_sha..head_sha` excludes `Generated from cl/100`.
  run_git(full_dir, "init", "-b", "main")
  run_git(full_dir, "remote", "add", "origin", str(origin_dir))
  run_git(full_dir, "fetch", "--no-tags", "origin", head_sha)
  run_git(full_dir, "checkout", "--detach", "FETCH_HEAD")
  run_git(full_dir, "fetch", "--no-tags", "origin", base_sha)

  full_vcs = custom_lint_vcs.GitVcs(full_dir)
  msg = full_vcs.get_commit_message(base_sha, head=head_sha)
  assert "External contributor PR" in msg
  assert "ALLOW_X=1" in msg
  assert "Generated from cl/100" not in msg
  modified = full_vcs.get_modified_lines(base_sha, root_dir=full_dir)
  assert set(modified) == {"a.py"}
  assert modified["a.py"] == {1}

  # Disconnected shallow checkout (`--depth=2` includes branch_point_sha's
  # `Generated from cl/100`, while `base_sha` at `--depth=1` cannot reach it):
  # both `get_modified_lines` and `get_commit_message` raise `VcsError` rather
  # than diffing `base_sha` directly or leaking pre-branch commit messages.
  run_git(shallow_dir, "init", "-b", "main")
  run_git(shallow_dir, "remote", "add", "origin", str(origin_dir))
  run_git(shallow_dir, "fetch", "--no-tags", "--depth=2", "origin", head_sha)
  run_git(shallow_dir, "checkout", "--detach", "FETCH_HEAD")
  run_git(shallow_dir, "fetch", "--no-tags", "--depth=1", "origin", base_sha)

  shallow_vcs = custom_lint_vcs.GitVcs(shallow_dir)
  with pytest.raises(custom_lint_vcs.VcsError, match="git merge-base"):
    shallow_vcs.get_modified_lines(base_sha, root_dir=shallow_dir)
  with pytest.raises(custom_lint_vcs.VcsError, match="git merge-base"):
    shallow_vcs.get_commit_message(base_sha, head=head_sha)


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
