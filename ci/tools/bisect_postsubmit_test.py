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

"""Unit tests for the bisect_postsubmit CI bisection driver tool."""

import json
import subprocess
from typing import Any
import unittest

from ci.tools import bisect_postsubmit


class BisectPostsubmitTest(unittest.TestCase):
  """Tests for bisect_postsubmit bisection logic and helpers."""

  def test_build_bazel_args_fail_fast(self) -> None:
    build_args, test_args = bisect_postsubmit.build_bazel_args(
        bazel_config="ci_tpu_v7",
        fail_fast=True,
        seed=None,
        repo_root="/repo",
    )
    self.assertIn("--config=ci_tpu_v7", build_args)
    self.assertIn("--nokeep_going", build_args)
    self.assertIn("--config=ci_tpu_v7", test_args)
    self.assertIn("--nokeep_going", test_args)
    self.assertIn("--test_runner_fail_fast", test_args)
    self.assertIn(
        "--run_under=/repo/ci/tools/parallel_accelerator_execute.sh",
        test_args,
    )

  def test_build_bazel_args_seed_and_cpu(self) -> None:
    build_args, test_args = bisect_postsubmit.build_bazel_args(
        bazel_config="ci_cpu",
        fail_fast=False,
        seed="4242",
        repo_root="/repo",
    )
    self.assertEqual(build_args, ["--config=ci_cpu"])
    self.assertNotIn("--nokeep_going", test_args)
    self.assertNotIn("--test_runner_fail_fast", test_args)
    self.assertIn("--test_arg=--test_random_seed=4242", test_args)
    self.assertIn("--test_env=PYTHONHASHSEED=4242", test_args)
    self.assertFalse(any("--run_under" in a for a in test_args))

  def test_check_prior_run_no_current_id(self) -> None:
    skip, reason = bisect_postsubmit.check_prior_run(
        current_run_id=None,
        bazel_config="ci_tpu_v7",
        runner="linux-x86-tpu7x-224-4tpu",
    )
    self.assertFalse(skip)
    self.assertIn("Manual dispatch", reason)

  def test_check_prior_run_failure_isolated_to_other_config(self) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:3] == ["gh", "run", "view"]:
        data = {
            "jobs": [
                {
                    "name": "Build and Test (linux-x86-n2-64)",
                    "conclusion": "failure",
                },
                {
                    "name": "Build and Test (linux-x86-tpu7x-224-4tpu)",
                    "conclusion": "success",
                },
            ]
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data))
      return subprocess.CompletedProcess(cmd, 0, stdout="")

    skip, reason = bisect_postsubmit.check_prior_run(
        current_run_id=200,
        bazel_config="ci_tpu_v7",
        runner="linux-x86-tpu7x-224-4tpu",
        run_cmd_fn=fake_run,
    )
    self.assertTrue(skip)
    self.assertIn("isolated to other configs", reason)

  def test_check_prior_run_main_already_failing(self) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:3] == ["gh", "run", "view"]:
        run_id = cmd[3]
        if run_id == "200":
          data = {
              "jobs": [{
                  "name": "Build and Test (linux-x86-tpu7x-224-4tpu)",
                  "conclusion": "failure",
              }]
          }
        elif run_id == "100":
          data = {
              "jobs": [{
                  "name": "Build and Test (linux-x86-tpu7x-224-4tpu)",
                  "conclusion": "failure",
                  "steps": [{
                      "name": "Run Bazel Test",
                      "conclusion": "failure",
                  }],
              }]
          }
        else:
          data = {"jobs": []}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data))

      if cmd[:3] == ["gh", "run", "list"]:
        runs = [
            {"databaseId": 200, "status": "completed", "conclusion": "failure"},
            {"databaseId": 100, "status": "completed", "conclusion": "failure"},
        ]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(runs))

      return subprocess.CompletedProcess(cmd, 0, stdout="")

    skip, reason = bisect_postsubmit.check_prior_run(
        current_run_id=200,
        bazel_config="ci_tpu_v7",
        runner="linux-x86-tpu7x-224-4tpu",
        run_cmd_fn=fake_run,
    )
    self.assertTrue(skip)
    self.assertIn("Main branch was already failing", reason)

  def test_check_prior_run_prior_superseded_and_healthy(self) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:3] == ["gh", "run", "view"]:
        run_id = cmd[3]
        if run_id == "200":
          data = {
              "jobs": [{
                  "name": "Build and Test (linux-x86-tpu7x-224-4tpu)",
                  "conclusion": "failure",
              }]
          }
        elif run_id == "150":
          # Run 150 was superseded so tests were skipped
          data = {
              "jobs": [{
                  "name": "Build and Test (linux-x86-tpu7x-224-4tpu)",
                  "conclusion": "success",
                  "steps": [{
                      "name": "Run Bazel Test",
                      "conclusion": "skipped",
                  }],
              }]
          }
        elif run_id == "100":
          # Run 100 ran tests and succeeded
          data = {
              "jobs": [{
                  "name": "Build and Test (linux-x86-tpu7x-224-4tpu)",
                  "conclusion": "success",
                  "steps": [{
                      "name": "Run Bazel Test",
                      "conclusion": "success",
                  }],
              }]
          }
        else:
          data = {"jobs": []}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data))

      if cmd[:3] == ["gh", "run", "list"]:
        runs = [
            {"databaseId": 200, "status": "completed"},
            {"databaseId": 150, "status": "completed"},
            {"databaseId": 100, "status": "completed"},
        ]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(runs))

      return subprocess.CompletedProcess(cmd, 0, stdout="")

    skip, reason = bisect_postsubmit.check_prior_run(
        current_run_id=200,
        bazel_config="ci_tpu_v7",
        runner="linux-x86-tpu7x-224-4tpu",
        run_cmd_fn=fake_run,
    )
    self.assertFalse(skip)
    self.assertIn("Prior run 100 succeeded", reason)

  def test_find_good_sha_skips_superseded_and_finds_valid_ancestor(
      self,
  ) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:3] == ["gh", "run", "list"]:
        runs = [
            {"databaseId": 300, "headSha": "sha_newest_skipped"},
            {"databaseId": 250, "headSha": "sha_not_ancestor"},
            {"databaseId": 200, "headSha": "sha_valid_ancestor"},
        ]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(runs))

      if cmd[:2] == ["git", "merge-base"]:
        cand = cmd[3]
        if cand == "sha_not_ancestor":
          return subprocess.CompletedProcess(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)

      if cmd[:3] == ["gh", "run", "view"]:
        run_id = cmd[3]
        if run_id == "300":
          # Tests were skipped
          data = {
              "jobs": [{
                  "name": "Build and Test (linux-x86-tpu7x-224-4tpu)",
                  "steps": [{
                      "name": "Run Bazel Test",
                      "conclusion": "skipped",
                  }],
              }]
          }
        elif run_id == "200":
          data = {
              "jobs": [{
                  "name": "Build and Test (linux-x86-tpu7x-224-4tpu)",
                  "steps": [{
                      "name": "Run Bazel Test",
                      "conclusion": "success",
                  }],
              }]
          }
        else:
          data = {"jobs": []}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data))

      return subprocess.CompletedProcess(cmd, 0, stdout="")

    good = bisect_postsubmit.find_good_sha(
        bad_sha="sha_bad",
        bazel_config="ci_tpu_v7",
        runner="linux-x86-tpu7x-224-4tpu",
        run_cmd_fn=fake_run,
    )
    self.assertEqual(good, "sha_valid_ancestor")

  def test_run_bisection_not_reproducible_on_bad(self) -> None:
    # BAD_SHA tests pass -> skips without checking GOOD_SHA
    ran_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
      del kwargs
      ran_cmds.append(cmd)
      return subprocess.CompletedProcess(cmd, 0, stdout="")

    res = bisect_postsubmit.run_bisection(
        bad_sha="bad123",
        good_sha="good123",
        build_args=["--config=ci_tpu_v7"],
        test_args=["--config=ci_tpu_v7"],
        test_targets=["//..."],
        run_cmd_fn=fake_run,
    )
    self.assertTrue(res.skipped)
    self.assertIn("not reproduced", res.skip_reason)
    # Check that GOOD_SHA was never checked out
    self.assertNotIn(["git", "checkout", "good123", "--quiet"], ran_cmds)

  def test_run_bisection_baseline_failure_on_good(self) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:2] == ["bazel", "test"]:
        # Both bad and good fail
        return subprocess.CompletedProcess(cmd, 1, stdout="fail")
      return subprocess.CompletedProcess(cmd, 0, stdout="")

    with self.assertRaises(RuntimeError) as ctx:
      bisect_postsubmit.run_bisection(
          bad_sha="bad123",
          good_sha="good123",
          build_args=["--config=ci_tpu_v7"],
          test_args=["--config=ci_tpu_v7"],
          test_targets=["//..."],
          run_cmd_fn=fake_run,
      )
    self.assertIn("Baseline failed on GOOD_SHA", str(ctx.exception))

  def test_run_bisection_1_commit_short_circuit(self) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:3] == ["git", "checkout", "bad123"]:
        return subprocess.CompletedProcess(cmd, 0)
      if cmd[:3] == ["git", "checkout", "good123"]:
        return subprocess.CompletedProcess(cmd, 0)
      if cmd[:2] == ["bazel", "test"]:
        # If currently at bad123, fail; if at good123, pass
        # Look at last git checkout
        return subprocess.CompletedProcess(cmd, 0)
      if cmd[:3] == ["git", "rev-list", "--count"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="1\n")
      return subprocess.CompletedProcess(cmd, 0, stdout="")

    # Custom test caller that fails on bad, passes on good
    current_checked_out = [""]

    def custom_run(
        cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:2] == ["git", "checkout"]:
        current_checked_out[0] = cmd[2]
      if cmd[:2] == ["bazel", "test"]:
        if current_checked_out[0] == "bad123":
          return subprocess.CompletedProcess(cmd, 1)  # failed on bad
        return subprocess.CompletedProcess(cmd, 0)  # passed on good
      if cmd[:3] == ["git", "rev-list", "--count"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="1\n")
      return subprocess.CompletedProcess(cmd, 0, stdout="")

    res = bisect_postsubmit.run_bisection(
        bad_sha="bad123",
        good_sha="good123",
        build_args=["--config=ci_tpu_v7"],
        test_args=["--config=ci_tpu_v7"],
        test_targets=["//..."],
        run_cmd_fn=custom_run,
    )
    self.assertFalse(res.skipped)
    self.assertEqual(res.culprit_sha, "bad123")
    self.assertEqual(res.iterations, 0)

  def test_run_bisection_multi_commit(self) -> None:
    current_checked_out = [""]

    def custom_run(
        cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:2] == ["git", "checkout"]:
        current_checked_out[0] = cmd[2]
      if cmd[:2] == ["bazel", "test"]:
        if current_checked_out[0] in ("bad123", "candidate_1"):
          return subprocess.CompletedProcess(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)
      if cmd[:3] == ["git", "rev-list", "--count"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="5\n")
      if cmd[:2] == ["git", "bisect"] and cmd[2] == "bad":
        out = "candidate_1 is the first bad commit\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out)
      if cmd[:2] == ["git", "rev-parse"] and cmd[2] == "HEAD":
        return subprocess.CompletedProcess(cmd, 0, stdout="candidate_1\n")
      return subprocess.CompletedProcess(cmd, 0, stdout="")

    res = bisect_postsubmit.run_bisection(
        bad_sha="bad123",
        good_sha="good123",
        build_args=["--config=ci_tpu_v7"],
        test_args=["--config=ci_tpu_v7"],
        test_targets=["//..."],
        max_iterations=5,
        run_cmd_fn=custom_run,
    )
    self.assertFalse(res.skipped)
    self.assertEqual(res.culprit_sha, "candidate_1")
    self.assertEqual(res.iterations, 1)

  def test_run_bisection_max_iterations(self) -> None:
    current_checked_out = [""]

    def custom_run(
        cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
      del kwargs
      if cmd[:2] == ["git", "checkout"]:
        current_checked_out[0] = cmd[2]
      if cmd[:2] == ["bazel", "test"]:
        if current_checked_out[0] == "bad123":
          return subprocess.CompletedProcess(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)
      if cmd[:3] == ["git", "rev-list", "--count"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="10\n")
      if cmd[:2] == ["git", "bisect"] and cmd[2] == "good":
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Bisecting: 3 revisions left to test after this\n"
        )
      if cmd[:2] == ["git", "rev-parse"] and cmd[2] == "HEAD":
        return subprocess.CompletedProcess(cmd, 0, stdout="cand_sha\n")
      if cmd[:2] == ["git", "rev-parse"] and cmd[2] == "refs/bisect/bad":
        return subprocess.CompletedProcess(
            cmd, 0, stdout="narrowed_bad_commit\n"
        )
      return subprocess.CompletedProcess(cmd, 0, stdout="")

    res = bisect_postsubmit.run_bisection(
        bad_sha="bad123",
        good_sha="good123",
        build_args=["--config=ci_tpu_v7"],
        test_args=["--config=ci_tpu_v7"],
        test_targets=["//..."],
        max_iterations=2,
        run_cmd_fn=custom_run,
    )
    self.assertFalse(res.skipped)
    self.assertIsNone(res.culprit_sha)
    self.assertEqual(res.iterations, 2)
    self.assertEqual(res.narrowed_sha, "narrowed_bad_commit")


if __name__ == "__main__":
  unittest.main()
