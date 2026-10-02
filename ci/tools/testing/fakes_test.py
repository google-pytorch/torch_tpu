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

"""Unit tests for the shared tool test fakes."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

try:
  from ci.tools.testing import fakes  # pylint: disable=g-import-not-at-top
except ImportError:
  from torch_tpu.oss.ci.tools.testing import fakes  # pylint: disable=g-import-not-at-top


class FakeSubprocessRunnerTest(unittest.TestCase):  # UNITTEST_OK=CI tools test
  """Checks that the runner honors the `subprocess.run` contract."""

  def test_canned_result_is_returned_as_captured_bytes(self):
    runner = fakes.FakeSubprocessRunner()
    runner.on(["git", "rev-parse"], stdout="abc\n", stderr="warn\n")
    proc = runner(["git", "rev-parse", "HEAD"], capture_output=True)
    self.assertEqual(proc.args, ["git", "rev-parse", "HEAD"])
    self.assertEqual(proc.returncode, 0)
    self.assertEqual(proc.stdout, b"abc\n")
    self.assertEqual(proc.stderr, b"warn\n")

  def test_text_mode_returns_str(self):
    runner = fakes.FakeSubprocessRunner()
    runner.on(["tool"], stdout="out")
    proc = runner(["tool"], stdout=subprocess.PIPE, text=True)
    self.assertEqual(proc.stdout, "out")
    self.assertIsNone(proc.stderr)

  def test_uncaptured_output_is_none(self):
    runner = fakes.FakeSubprocessRunner()
    runner.on(["tool"], stdout="out", stderr="err")
    proc = runner(["tool"])
    self.assertIsNone(proc.stdout)
    self.assertIsNone(proc.stderr)

  def test_stderr_can_be_merged_into_stdout(self):
    runner = fakes.FakeSubprocessRunner()
    runner.on(["tool"], stdout="out,", stderr="err")
    proc = runner(
        ["tool"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    self.assertEqual(proc.stdout, "out,err")

  def test_check_raises_on_failure(self):
    runner = fakes.FakeSubprocessRunner()
    runner.on(["tool"], returncode=2, stderr="boom")
    with self.assertRaises(subprocess.CalledProcessError) as ctx:
      runner(["tool"], check=True, capture_output=True, text=True)
    self.assertEqual(ctx.exception.returncode, 2)
    self.assertEqual(ctx.exception.stderr, "boom")
    # Without `check`, the failure is only reported through the exit code.
    self.assertEqual(runner(["tool"]).returncode, 2)

  def test_longest_prefix_wins(self):
    runner = fakes.FakeSubprocessRunner()
    runner.on(["git"], stdout="generic")
    runner.on(["git", "status"], stdout="specific")
    self.assertEqual(
        runner(["git", "status"], text=True, capture_output=True).stdout,
        "specific",
    )
    self.assertEqual(
        runner(["git", "log"], text=True, capture_output=True).stdout, "generic"
    )

  def test_unexpected_command_raises(self):
    runner = fakes.FakeSubprocessRunner()
    with self.assertRaises(fakes.UnexpectedCommandError):
      runner(["rm", "-rf", "/"])

  def test_handler_side_effects_and_call_recording(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      out_file = pathlib.Path(temp_dir) / "out.txt"

      def write_output(call: fakes.FakeCall) -> fakes.FakeResult:
        out_file.write_text(call.env["GREETING"], encoding="utf-8")
        return fakes.FakeResult(returncode=3)

      runner = fakes.FakeSubprocessRunner()
      runner.on(["greet"], write_output)
      proc = runner(
          [pathlib.Path("greet"), "--loud"],
          env={"GREETING": "hi"},
          cwd=pathlib.Path(temp_dir),
      )
      self.assertEqual(proc.returncode, 3)
      self.assertEqual(out_file.read_text(encoding="utf-8"), "hi")
      self.assertEqual(
          runner.calls,
          [
              fakes.FakeCall(
                  ("greet", "--loud"), env={"GREETING": "hi"}, cwd=temp_dir
              )
          ],
      )
      self.assertEqual(runner.commands("greet"), [("greet", "--loud")])
      self.assertEqual(runner.commands("other"), [])


class FakeGcsTest(unittest.TestCase):  # UNITTEST_OK=CI tools test
  """Checks uploads through `gcloud storage cp`."""

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    self.local = pathlib.Path(temp_dir.name) / "local"
    self.local.mkdir()
    self.gcs = fakes.FakeGcs(
        pathlib.Path(temp_dir.name) / "gcs", read_only_buckets=["locked"]
    )
    self.runner = fakes.FakeSubprocessRunner()
    self.gcs.install(self.runner)

  def test_upload_into_prefix(self):
    (self.local / "a.gz").write_bytes(b"A")
    (self.local / "b.gz").write_bytes(b"B")
    self.runner(
        [
            "gcloud",
            "storage",
            "cp",
            str(self.local / "a.gz"),
            str(self.local / "b.gz"),
            "gs://bucket/dir/",
        ],
        check=True,
    )
    self.assertEqual(
        self.gcs.list_objects("gs://bucket/dir"),
        ["gs://bucket/dir/a.gz", "gs://bucket/dir/b.gz"],
    )
    self.assertEqual(self.gcs.read("gs://bucket/dir/b.gz"), b"B")

  def test_upload_to_object_name(self):
    (self.local / "a.gz").write_bytes(b"A")
    self.runner(
        ["gcloud", "storage", "cp", str(self.local / "a.gz"), "gs://b/x.gz"],
        check=True,
    )
    self.assertTrue(self.gcs.exists("gs://b/x.gz"))
    self.assertFalse(self.gcs.exists("gs://b/a.gz"))

  def test_upload_to_read_only_bucket_fails(self):
    (self.local / "a.gz").write_bytes(b"A")
    proc = self.runner(
        ["gcloud", "storage", "cp", str(self.local / "a.gz"), "gs://locked/"],
        capture_output=True,
        text=True,
    )
    self.assertEqual(proc.returncode, 1)
    self.assertIn("AccessDeniedException", proc.stderr)
    self.assertEqual(self.gcs.list_objects("gs://locked"), [])


class FakeGithubTest(unittest.TestCase):  # UNITTEST_OK=CI tools test
  """Checks the in-memory GitHub state."""

  def setUp(self):
    super().setUp()
    self.gh = fakes.FakeGithub()
    self.gh.add_user("member", name="A Member")
    self.gh.add_user("outsider")
    self.team = self.gh.add_organization("org", members=["member"]).add_team(
        "team"
    )

  def test_add_membership_adds_org_members_and_invites_others(self):
    team = self.gh.get_organization("org").get_team_by_slug("team")
    team.add_membership(self.gh.get_user("member"), role="member")
    team.add_membership(self.gh.get_user("outsider"), role="member")
    self.assertEqual([u.login for u in team.get_members()], ["member"])
    self.assertEqual([u.login for u in team.invitations()], ["outsider"])

  def test_user_attributes(self):
    self.assertEqual(self.gh.get_user("member").name, "A Member")
    self.assertIsNone(self.gh.get_user("outsider").name)

  def test_unknown_entities_raise(self):
    with self.assertRaises(fakes.FakeGithubError):
      self.gh.get_user("nobody")
    with self.assertRaises(fakes.FakeGithubError):
      self.gh.get_organization("nowhere")
    with self.assertRaises(fakes.FakeGithubError):
      self.gh.get_organization("org").get_team_by_slug("missing")

  def test_invalid_role_raises(self):
    with self.assertRaises(ValueError):
      self.team.add_membership(self.gh.get_user("member"), role="owner")


if __name__ == "__main__":
  unittest.main(argv=sys.argv[:1])
