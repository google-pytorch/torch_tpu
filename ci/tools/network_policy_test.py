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

"""Tests for the CI network policy (see ci/README.md).

* WorkflowEnforcementTest: every job that checks out the repository
  activates the policy right after its first `actions/checkout` step, or is
  listed in `_JOBS_WITHOUT_CHECKOUT` with a reason.
* NoLocalOverridesTest: nothing under `.github/`, `ci/`, `tools/` or
  `noxfile.py` hard-codes a retry or timeout number or bypasses `.curlrc`.
* ActivateScriptTest: runs `activate.sh` and checks what it exports to
  `GITHUB_ENV` and renders into `.curlrc`.
"""

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_POLICY_DIR = _REPO_ROOT / "ci" / "tools" / "network_policy"
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
_POLICY_ACTION = "./.github/actions/network-policy"
_ACTIVATE_SCRIPT = "ci/tools/network_policy/activate.sh"
_BASH = shutil.which("bash")

# Jobs that never check out the repository, so they have nothing to activate
# the policy from. Key: "<workflow file>:<job id>"; value: why that is fine.
_JOBS_WITHOUT_CHECKOUT = {
    "build_wheel.yml:wheel_tests_passed": "Aggregates results; no network.",
    "daily_project_management.yml:stale": "Runs actions/stale only.",
    "label-pr.yml:label": "Runs actions/labeler only.",
    "on_pr_close.yaml:cancel_runs": "Runs a cancellation action only.",
    "on_pr_ready.yaml:post_review_process": (
        "pull_request_target job that deliberately never checks out the PR."
    ),
}

# Where hard-coded network numbers are looked for, relative to the repository
# root. The policy directory and this file are exempt.
_SCAN_ROOTS = (".github", "ci", "tools", "noxfile.py")
_SCAN_EXEMPT = (_POLICY_DIR, pathlib.Path(__file__).resolve())
_FORBIDDEN_PATTERNS = {
    "hard-codes a policy value (edit policy.env instead)": re.compile(
        r"\b(UV_HTTP_RETRIES|UV_HTTP_TIMEOUT|PIP_RETRIES|PIP_TIMEOUT"
        r"|DOWNLOAD_RETRIES|DOWNLOAD_RETRY_DELAY|DOWNLOAD_RETRY_MAX_TIME"
        r"|DOWNLOAD_CONNECT_TIMEOUT)\s*[:=]\s*[\"']?\d"
    ),
    "passes a per-call retry or timeout flag": re.compile(
        r"(?<![\w-])--(retry|retries|retry-delay|retry-max-time"
        r"|retry-all-errors|retry-connrefused|connect-timeout)\b"
    ),
    "disables curl's configuration file (it carries the policy)": re.compile(
        r"\bcurl\b[^\n|]*\s-(q|-disable)\b"
    ),
}


def _file_policy() -> dict[str, str]:
  """The policy as written in policy.env (KEY=VALUE lines, # comments)."""
  policy = {}
  for raw_line in (_POLICY_DIR / "policy.env").read_text().splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if line:
      key, _, value = line.partition("=")
      policy[key] = value
  return policy


def _jobs():
  for path in sorted(_WORKFLOWS_DIR.iterdir()):
    if path.suffix in (".yml", ".yaml"):
      workflow = yaml.safe_load(path.read_text()) or {}
      for job_id, job in (workflow.get("jobs") or {}).items():
        yield f"{path.name}:{job_id}", job


def _first_checkout(steps) -> int | None:
  for index, step in enumerate(steps):
    if str(step.get("uses", "")).startswith("actions/checkout@"):
      return index
  return None


def _scanned_files():
  for root in _SCAN_ROOTS:
    path = _REPO_ROOT / root
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file in files:
      if not file.is_file() or file.suffix in (".md", ".pyc"):
        continue
      parents = file.relative_to(path).parts[:-1]  # Skip generated caches.
      if any(p == "__pycache__" or p.startswith(".") for p in parents):
        continue
      if any(file == e or e in file.parents for e in _SCAN_EXEMPT):
        continue
      yield file


class WorkflowEnforcementTest(unittest.TestCase):

  def test_every_job_activates_the_policy_right_after_checkout(self):
    allowlisted = set()
    for key, job in _jobs():
      steps = job.get("steps")
      if steps is None:
        continue  # Calls a reusable workflow; no steps of its own.
      with self.subTest(job=key):
        checkout = _first_checkout(steps)
        if checkout is None:
          self.assertIn(
              key,
              _JOBS_WITHOUT_CHECKOUT,
              f"{key} has no actions/checkout step: add one followed by"
              f" `- uses: {_POLICY_ACTION}`, or list the job in"
              " _JOBS_WITHOUT_CHECKOUT with a reason.",
          )
          allowlisted.add(key)
          continue
        self.assertNotIn(key, _JOBS_WITHOUT_CHECKOUT, f"{key} checks out")
        self.assertLess(checkout + 1, len(steps), f"{key}: bare checkout")
        following = steps[checkout + 1]
        checkout_path = (steps[checkout].get("with") or {}).get("path")
        if checkout_path:
          expected = f"{str(checkout_path).rstrip('/')}/{_ACTIVATE_SCRIPT}"
          actual = str(following.get("run", "")).strip()
        else:
          expected = _POLICY_ACTION
          actual = following.get("uses")
        self.assertEqual(
            actual,
            expected,
            f"{key}: the step right after the first actions/checkout has to"
            f" activate the CI network policy (expected {expected!r}), found"
            f" {following!r}.",
        )
    self.assertEqual(
        allowlisted, set(_JOBS_WITHOUT_CHECKOUT), "Stale allowlist entries."
    )
    for key, reason in _JOBS_WITHOUT_CHECKOUT.items():
      self.assertTrue(reason.strip(), f"{key} needs a reason")

  def test_sparse_checkouts_include_the_policy(self):
    for key, job in _jobs():
      steps = job.get("steps") or []
      checkout = _first_checkout(steps)
      if checkout is None:
        continue
      sparse = (steps[checkout].get("with") or {}).get("sparse-checkout")
      if sparse is None:
        continue
      entries = [e.strip() for e in str(sparse).splitlines() if e.strip()]
      for required in ("ci/tools", ".github/actions/network-policy"):
        with self.subTest(job=key, entry=required):
          self.assertIn(required, entries)

  def test_policy_action_only_runs_the_activation_script(self):
    action = yaml.safe_load(
        (_REPO_ROOT / _POLICY_ACTION / "action.yml").read_text()
    )
    self.assertEqual(action["runs"]["using"], "composite")
    (step,) = action["runs"]["steps"]
    self.assertEqual(step["shell"], "bash")
    self.assertTrue(step["run"].strip().endswith(_ACTIVATE_SCRIPT))


class NoLocalOverridesTest(unittest.TestCase):

  def test_no_file_hard_codes_a_network_number(self):
    for file in _scanned_files():
      text = file.read_text(errors="replace")
      for what, pattern in _FORBIDDEN_PATTERNS.items():
        with self.subTest(file=str(file.relative_to(_REPO_ROOT)), what=what):
          match = pattern.search(text)
          self.assertIsNone(
              match,
              f"{file.relative_to(_REPO_ROOT)} {what}:"
              f" {match.group(0) if match else ''!r}. Every network client"
              " inherits the policy from ci/tools/network_policy/policy.env;"
              " see ci/README.md.",
          )

  def test_scan_covers_the_expected_files(self):
    scanned = {str(f.relative_to(_REPO_ROOT)) for f in _scanned_files()}
    self.assertIn("noxfile.py", scanned)
    self.assertIn(".github/workflows/presubmit.yml", scanned)
    self.assertIn("ci/tools/run_bazel_diff.py", scanned)
    self.assertIn("tools/turn_tracker.py", scanned)
    self.assertNotIn("ci/tools/network_policy/policy.env", scanned)


@unittest.skipIf(_BASH is None, "bash is required")
class ActivateScriptTest(unittest.TestCase):
  """Runs activate.sh in a clean, minimal environment."""

  def setUp(self):
    super().setUp()
    self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
    self.script = str(_POLICY_DIR / "activate.sh")
    self.github_env = self.tmp / "github_env"
    self.env = {
        "PATH": os.environ["PATH"],
        "HOME": str(self.tmp),
        "TMPDIR": str(self.tmp),
        "LC_ALL": "C",
        "GITHUB_ENV": str(self.github_env),
        "RUNNER_TEMP": str(self.tmp / "runner_temp"),
    }

  def run_shell(self, args, **env):
    result = subprocess.run(
        args, env={**self.env, **env}, capture_output=True, text=True
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    return result

  def exported(self) -> dict[str, str]:
    lines = self.github_env.read_text().splitlines()
    self.assertEqual(len(lines), len(set(lines)), lines)
    return dict(line.split("=", 1) for line in lines)

  def test_populates_github_env_and_renders_curlrc(self):
    policy = _file_policy()
    runtime = self.tmp / "runner_temp" / "torch_tpu_network_policy"

    result = self.run_shell([self.script], GITHUB_ACTIONS="true")

    self.assertIn("network-policy: activated", result.stdout)
    exported = self.exported()
    self.assertEqual(set(exported), set(policy) | {"CURL_HOME"})
    for key, value in policy.items():
      self.assertEqual(exported[key], value, key)
    self.assertEqual(exported["CURL_HOME"], str(runtime))
    curlrc = (runtime / ".curlrc").read_text()
    self.assertIn("retry-all-errors\n", curlrc)
    self.assertNotIn("retry-delay", curlrc)
    for option, key in (
        ("retry", "DOWNLOAD_RETRIES"),
        ("retry-max-time", "DOWNLOAD_RETRY_MAX_TIME"),
        ("connect-timeout", "DOWNLOAD_CONNECT_TIMEOUT"),
    ):
      self.assertIn(f"{option} = {policy[key]}\n", curlrc)


if __name__ == "__main__":
  unittest.main()
