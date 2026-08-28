#!/usr/bin/env python3
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

"""Tests for the Spot TPU relay scripts.

Everything here exercises paths that bail out before the first SSH hop, so the
suite needs no TPU, no network, and no gcloud credentials.
"""

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELAY_RUNNER = os.path.join(REPO_ROOT, "ci", "tools", "relay_test_runner.sh")
REMOTE_EXECUTOR = os.path.join(
    REPO_ROOT, "ci", "tools", "remote_tpu_executor.sh"
)
RELAY_DRIVER = os.path.join(REPO_ROOT, "scripts", "relay_presubmit_pr.sh")

SESSION_KEYS = (
    "TPU_IP",
    "SSH_USER",
    "SSH_CONTROL_PATH",
    "TPU_NAME",
    "TPU_ZONE",
)


class RelayRunnerTestCase(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Shared setup: a scratch dir plus a helper that runs the relay runner."""

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.xml_path = os.path.join(self.tmp.name, "test.xml")

  def write_session(self, **values):
    """Writes a session env file and returns its path."""
    path = os.path.join(self.tmp.name, "session.env")
    with open(path, "w", encoding="utf-8") as f:
      for key, value in values.items():
        f.write(f'{key}="{value}"\n')
    return path

  def run_relay(self, args=("/bin/true",), session_env=None, **env_overrides):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": self.tmp.name,
        "XML_OUTPUT_FILE": self.xml_path,
        "TEST_TARGET": "//tests:relay_smoke_test",
        "TPU_SESSION_ENV": (
            session_env or os.path.join(self.tmp.name, "absent.env")
        ),
    }
    env.update(env_overrides)
    return subprocess.run(
        ["bash", RELAY_RUNNER, *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def read_xml(self):
    self.assertTrue(
        os.path.exists(self.xml_path), "No JUnit report was written"
    )
    return ET.parse(self.xml_path).getroot().find("testsuite")


class TestRelayRunnerFailureReports(RelayRunnerTestCase):
  """Every early exit has to leave Bazel a readable JUnit report.

  Without one, Bazel reports the target as failing with no explanation at all.
  """

  def test_missing_test_binary_is_a_usage_error(self):
    result = self.run_relay(args=())
    self.assertEqual(result.returncode, 1)
    self.assertIn("No test binary specified", result.stderr)

  def test_missing_session_file_reports_how_to_fix_it(self):
    result = self.run_relay()
    self.assertEqual(result.returncode, 1)
    self.assertIn("spot_tpu_manager.sh up", result.stderr)

    suite = self.read_xml()
    self.assertEqual(suite.get("failures"), "1")
    self.assertEqual(
        suite.find("testcase/failure").get("message"), "No active TPU session"
    )

  def test_incomplete_session_file_names_the_missing_variables(self):
    session = self.write_session(TPU_IP="10.0.0.1")
    result = self.run_relay(session_env=session)
    self.assertEqual(result.returncode, 1)

    failure = self.read_xml().find("testcase/failure")
    self.assertEqual(failure.get("message"), "Incomplete TPU session")
    self.assertIn("SSH_USER", failure.text)

  def test_report_names_the_bazel_target_not_the_binary(self):
    self.run_relay()
    # Bazel labels can't appear verbatim in an XML name attribute, so `/` and
    # `:` are flattened.
    self.assertEqual(self.read_xml().get("name"), "__tests_relay_smoke_test")

  def test_first_failure_wins_the_report(self):
    """The exit trap also writes a report, and must not clobber the real one."""
    self.run_relay()
    self.assertEqual(
        len(ET.parse(self.xml_path).getroot().findall("testsuite")), 1
    )
    self.assertNotIn(
        "interrupted by signal",
        self.read_xml().find("testcase/failure").get("message"),
    )

  def test_runs_without_an_xml_output_file(self):
    """Bazel only sets XML_OUTPUT_FILE for test actions, not for `bazel run`."""
    env = dict(os.environ)
    env.pop("XML_OUTPUT_FILE", None)
    result = subprocess.run(
        ["bash", RELAY_RUNNER, "/bin/true"],
        env={
            "PATH": os.environ["PATH"],
            "HOME": self.tmp.name,
            "TPU_SESSION_ENV": os.path.join(self.tmp.name, "absent.env"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(result.returncode, 1)


class TestRelayRunnerProjectBoundary(RelayRunnerTestCase):
  """The relay must never touch a project other than rbe-tpu-oss."""

  def test_rejects_an_explicit_foreign_project(self):
    result = self.run_relay(TPU_PROJECT="some-other-project")
    self.assertEqual(result.returncode, 1)
    self.assertIn("Project boundary violation", result.stderr)
    self.assertEqual(
        self.read_xml().find("testcase/failure").get("message"),
        "Project Boundary Violation",
    )

  def test_rejects_a_foreign_project_from_the_gcloud_environment(self):
    for var in [
        "CLOUDSDK_CORE_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "GCLOUD_PROJECT",
    ]:
      with self.subTest(var=var):
        self.setUp()
        result = self.run_relay(**{var: "some-other-project"})
        self.assertEqual(result.returncode, 1)
        self.assertIn(var, result.stderr)

  def test_allows_the_project_it_owns(self):
    result = self.run_relay(
        TPU_PROJECT="rbe-tpu-oss", CLOUDSDK_CORE_PROJECT="rbe-tpu-oss"
    )
    # Still fails, but on the missing session rather than the boundary check.
    self.assertNotIn("boundary", result.stderr)


class RelayPoolTestCase(RelayRunnerTestCase):
  """Shared helpers for the tests that exercise a fleet of session files."""

  def setUp(self):
    super().setUp()
    self.pool = os.path.join(self.tmp.name, "pool")
    os.makedirs(self.pool)

  def add_pool_vm(self, name):
    """Adds a deliberately incomplete session so the run stops after leasing.

    The failure message names the session file it leased, which is how these
    tests observe which VM was picked without needing real hardware.
    """
    path = os.path.join(self.pool, f"{name}.env")
    with open(path, "w", encoding="utf-8") as f:
      f.write('TPU_IP="10.0.0.1"\n')
    return path

  def hold_lock(self, session_path):
    """Locks a pool entry from outside, standing in for a busy VM."""
    holder = subprocess.Popen(
        ["flock", f"{session_path}.lock", "-c", "sleep 30"],
    )

    def stop_holder():
      holder.kill()
      holder.wait()

    self.addCleanup(stop_holder)
    # Give flock a moment to actually take the lock.
    for _ in range(50):
      probe = subprocess.run(
          ["flock", "-n", f"{session_path}.lock", "-c", "true"], check=False
      )
      if probe.returncode != 0:
        return holder
      time.sleep(0.1)
    self.fail(f"Could not take the lock on {session_path}")

  def leased_session(self):
    """Returns the pool entry named in the failure report."""
    return self.read_xml().find("testcase/failure").text


class TestRelayRunnerPoolLeasing(RelayPoolTestCase):
  """One Bazel run spreads tests over a fleet by leasing a VM per test."""

  def test_leases_the_only_free_vm(self):
    self.add_pool_vm("vm_0")
    result = self.run_relay(TPU_SESSION_POOL=self.pool)
    self.assertEqual(result.returncode, 1)
    self.assertIn("vm_0.env", self.leased_session())

  def test_skips_a_vm_another_test_is_using(self):
    busy = self.add_pool_vm("vm_0")
    self.add_pool_vm("vm_1")
    self.hold_lock(busy)

    self.run_relay(TPU_SESSION_POOL=self.pool)
    self.assertIn("vm_1.env", self.leased_session())

  def test_releases_the_lock_when_the_test_exits(self):
    self.add_pool_vm("vm_0")
    for attempt in range(2):
      with self.subTest(attempt=attempt):
        self.xml_path = os.path.join(self.tmp.name, f"test_{attempt}.xml")
        self.run_relay(TPU_SESSION_POOL=self.pool)
        self.assertIn("vm_0.env", self.leased_session())

  def test_reports_a_busy_fleet_instead_of_hanging(self):
    busy = self.add_pool_vm("vm_0")
    self.hold_lock(busy)

    result = self.run_relay(TPU_SESSION_POOL=self.pool, TEST_TIMEOUT="2")
    self.assertEqual(result.returncode, 1)
    self.assertIn("No free TPU VM", result.stderr)
    self.assertEqual(
        self.read_xml().find("testcase/failure").get("message"),
        "No free TPU in pool",
    )

  def test_an_empty_pool_fails_fast(self):
    result = self.run_relay(TPU_SESSION_POOL=self.pool, TEST_TIMEOUT="2")
    self.assertEqual(result.returncode, 1)
    self.assertIn("No free TPU VM", result.stderr)

  def test_without_a_pool_the_single_session_file_is_used(self):
    session = self.write_session(TPU_IP="10.0.0.9")
    self.run_relay(session_env=session)
    self.assertIn("session.env", self.leased_session())


class TestRelayScriptsAreWellFormed(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Cheap guards that catch the mistakes these scripts have hit before."""

  SCRIPTS = [
      RELAY_RUNNER,
      REMOTE_EXECUTOR,
      os.path.join(REPO_ROOT, "scripts", "spot_tpu_manager.sh"),
      os.path.join(REPO_ROOT, "scripts", "run_presubmit_v5_relay.sh"),
      os.path.join(REPO_ROOT, "scripts", "spot_tpu_fleet.sh"),
      os.path.join(REPO_ROOT, "ci", "tools", "presubmit_job_matrix.sh"),
  ]

  def test_scripts_parse(self):
    for script in self.SCRIPTS:
      with self.subTest(script=os.path.basename(script)):
        subprocess.run(["bash", "-n", script], check=True)

  def test_scripts_are_executable(self):
    for script in self.SCRIPTS:
      with self.subTest(script=os.path.basename(script)):
        self.assertTrue(os.access(script, os.X_OK))

  def test_no_developer_home_directories_are_baked_in(self):
    for script in self.SCRIPTS:
      with self.subTest(script=os.path.basename(script)):
        with open(script, "r", encoding="utf-8") as f:
          self.assertNotIn("/usr/local/google/home/", f.read())

  def test_no_pinned_python_version_in_remote_paths(self):
    """The TPU VM image picks its own Python; nothing may assume a version.

    The image shipped 3.10 while these scripts assumed a 3.12 venv, which broke
    every remote run.
    """
    for script in [
        RELAY_RUNNER,
        REMOTE_EXECUTOR,
        os.path.join(REPO_ROOT, "scripts", "spot_tpu_manager.sh"),
    ]:
      with self.subTest(script=os.path.basename(script)):
        with open(script, "r", encoding="utf-8") as f:
          body = f.read()
        self.assertNotIn("tpu_venv", body)
        self.assertNotRegex(body, r"python3\.\d+")

  def test_only_rbe_tpu_oss_is_hardcoded(self):
    """Catches a literal project name other than the one we're allowed to use.

    Variables and `--project=*)` parse patterns are fine; a baked-in project id
    is what would quietly send compute somewhere it doesn't belong.
    """
    literal_project = re.compile(r"--project=([a-z][a-z0-9-]{4,})")
    for script in self.SCRIPTS:
      with self.subTest(script=os.path.basename(script)):
        with open(script, "r", encoding="utf-8") as f:
          for lineno, line in enumerate(f, 1):
            for match in literal_project.finditer(line):
              self.assertEqual(
                  match.group(1),
                  "rbe-tpu-oss",
                  f"{script}:{lineno} hardcodes a project this relay"
                  " doesn't own",
              )


STAGE_SCRIPT = os.path.join(REPO_ROOT, "ci", "tools", "stage_relay_base.sh")
RELAY_SSH = os.path.join(REPO_ROOT, "ci", "tools", "relay_ssh.sh")


class StageRelayBaseTestCase(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Builds a throwaway workspace so staging can be exercised without a TPU.

  The layout mirrors what bazel leaves behind: a bazel-bin symlink, runfiles
  trees underneath it, dependency repositories at the runfiles root, and shared
  libraries under _main/_solib_x86_64.
  """

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.root = self._tmp.dir if False else self._tmp.name
    self.addCleanup(self._tmp.cleanup)

    tools = os.path.join(self.root, "ci", "tools")
    os.makedirs(tools)
    for src in (STAGE_SCRIPT, RELAY_SSH):
      dst = os.path.join(tools, os.path.basename(src))
      with open(src, "rb") as fh:
        data = fh.read()
      with open(dst, "wb") as fh:
        fh.write(data)
      os.chmod(dst, 0o755)
    self.script = os.path.join(tools, "stage_relay_base.sh")

    self.lock = os.path.join(self.root, "MODULE.bazel.lock")
    self._write(self.lock, "lockfile v1")

    self.out = os.path.join(self.root, "out")
    os.makedirs(self.out)
    os.symlink(self.out, os.path.join(self.root, "bazel-bin"))

    self.tarballs = os.path.join(self.root, "tarballs")

  def _write(self, path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
      fh.write(text)

  def add_runfiles_tree(self, name, deps=(), solibs=(), csrc=()):
    tree = os.path.join(self.out, "tests", name + ".runfiles")
    for dep in deps:
      self._write(os.path.join(tree, dep, "site-packages", "mod.py"), "x = 1\n")
    for solib in solibs:
      self._write(
          os.path.join(tree, "_main", "_solib_x86_64", solib, "lib.so"),
          "binary-" + solib,
      )
    for module in csrc:
      self._write(
          os.path.join(tree, "_main", "csrc", "common", module),
          "binary-" + module,
      )
    return tree

  def session_file(self, name="vm_0.env"):
    """A session missing TPU_IP, so staging stops before any SSH hop."""
    path = os.path.join(self.root, name)
    self._write(
        path, 'export TPU_NAME="fake-vm"\nexport TPU_ZONE="europe-west4-b"\n'
    )
    return path

  def run_stage(self, *args, env=None):
    environ = dict(os.environ)
    environ["TORCH_TPU_BASE_TARBALL_DIR"] = self.tarballs
    environ.pop("CLOUDSDK_CORE_PROJECT", None)
    if env:
      environ.update(env)
    return subprocess.run(
        [self.script, *args],
        capture_output=True,
        text=True,
        env=environ,
        timeout=120,
    )

  @staticmethod
  def stamp_for(layer, output):
    match = re.search(
        rf"^\[stage_relay_base\] {layer} stamp (\w+)$", output, re.M
    )
    return match.group(1) if match else None

  def layer_members(self, layer, stamp):
    """The log carries a 12-character prefix; the file name carries the rest."""
    matches = [
        f for f in os.listdir(self.tarballs) if f.startswith(f"{layer}_{stamp}")
    ]
    self.assertEqual(
        len(matches), 1, f"expected one {layer} tarball, got {matches}"
    )
    with tarfile.open(os.path.join(self.tarballs, matches[0])) as tf:
      return sorted(tf.getnames())


class TestStageRelayBaseDiscovery(StageRelayBaseTestCase):

  def test_collects_dependency_repositories_and_solibs(self):
    self.add_runfiles_tree(
        "a",
        deps=["rules_python++pip+torch_tpu_pypi_312_absl_py"],
        solibs=["_U_libtorch"],
    )
    result = self.run_stage("--session", self.session_file())
    self.assertIn(
        "found 1 dependency repositories and 1 solib directories", result.stdout
    )

  def test_same_repository_in_two_trees_is_shipped_once(self):
    shared = "rules_python++pip+torch_tpu_pypi_312_torch"
    self.add_runfiles_tree("a", deps=[shared], solibs=["_U_shared"])
    self.add_runfiles_tree("b", deps=[shared], solibs=["_U_shared"])
    result = self.run_stage("--session", self.session_file())
    self.assertIn(
        "found 1 dependency repositories and 1 solib directories", result.stdout
    )

  def test_unions_repositories_across_trees(self):
    self.add_runfiles_tree(
        "a", deps=["rules_python++pip+torch_tpu_pypi_312_torch"]
    )
    self.add_runfiles_tree(
        "b", deps=["rules_python++pip+torch_tpu_pypi_312_numpy"]
    )
    result = self.run_stage("--session", self.session_file())
    self.assertIn("found 2 dependency repositories", result.stdout)

  def test_ignores_directories_that_are_not_dependency_repositories(self):
    tree = self.add_runfiles_tree(
        "a", deps=["rules_python++pip+torch_tpu_pypi_312_torch"]
    )
    self._write(os.path.join(tree, "bazel_tools", "something.py"), "\n")
    self._write(os.path.join(tree, "_main", "tests", "a.py"), "\n")
    result = self.run_stage("--session", self.session_file())
    self.assertIn("found 1 dependency repositories", result.stdout)

  def test_reports_a_useful_error_without_a_build(self):
    result = self.run_stage("--session", self.session_file())
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("build the test targets first", result.stderr)

  def test_extension_modules_ride_the_solib_layer(self):
    """_main/csrc sits outside _solib_x86_64 and is the same build every time.

    Missing it put a 493 MB shared object in every per-test payload.
    """
    self.add_runfiles_tree(
        "a",
        deps=["rules_python++pip+torch_tpu_pypi_312_torch"],
        csrc=["libpywrap_torch_tpu_common.so"],
    )
    result = self.run_stage("--session", self.session_file())
    self.assertIn("and 1 solib directories", result.stdout)
    self.assertIn(
        "csrc/common/libpywrap_torch_tpu_common.so",
        self.layer_members("solib", self.stamp_for("solib", result.stdout)),
    )

  def test_extension_modules_are_collected_once_across_trees(self):
    for name in ("a", "b"):
      self.add_runfiles_tree(
          name,
          deps=["rules_python++pip+torch_tpu_pypi_312_torch"],
          solibs=["_U_libtorch"],
          csrc=["libpywrap_torch_tpu_common.so"],
      )
    result = self.run_stage("--session", self.session_file())
    self.assertIn("and 2 solib directories", result.stdout)


class TestStageRelayBaseStamps(StageRelayBaseTestCase):

  def setUp(self):
    super().setUp()
    self.add_runfiles_tree(
        "a",
        deps=["rules_python++pip+torch_tpu_pypi_312_torch"],
        solibs=["_U_libtorch"],
        csrc=["libpywrap_torch_tpu_common.so"],
    )
    self.session = self.session_file()
    first = self.run_stage("--session", self.session)
    self.deps_stamp = self.stamp_for("deps", first.stdout)
    self.solib_stamp = self.stamp_for("solib", first.stdout)
    self.assertIsNotNone(self.deps_stamp, first.stdout + first.stderr)
    self.assertIsNotNone(self.solib_stamp, first.stdout + first.stderr)

  def test_a_code_change_moves_only_the_solib_stamp(self):
    lib = os.path.join(
        self.out,
        "tests",
        "a.runfiles",
        "_main",
        "_solib_x86_64",
        "_U_libtorch",
        "lib.so",
    )
    self._write(lib, "binary-rebuilt-and-longer")
    result = self.run_stage("--session", self.session)
    self.assertEqual(self.deps_stamp, self.stamp_for("deps", result.stdout))
    self.assertNotEqual(
        self.solib_stamp, self.stamp_for("solib", result.stdout)
    )

  def test_a_rebuilt_extension_module_moves_the_solib_stamp(self):
    """Otherwise the VMs keep serving an old .so out of the base cache."""
    lib = os.path.join(
        self.out,
        "tests",
        "a.runfiles",
        "_main",
        "csrc",
        "common",
        "libpywrap_torch_tpu_common.so",
    )
    self._write(lib, "binary-rebuilt-and-longer")
    result = self.run_stage("--session", self.session)
    self.assertEqual(self.deps_stamp, self.stamp_for("deps", result.stdout))
    self.assertNotEqual(
        self.solib_stamp, self.stamp_for("solib", result.stdout)
    )

  def test_a_lockfile_change_moves_only_the_deps_stamp(self):
    self._write(self.lock, "lockfile v2")
    result = self.run_stage("--session", self.session)
    self.assertNotEqual(self.deps_stamp, self.stamp_for("deps", result.stdout))
    self.assertEqual(self.solib_stamp, self.stamp_for("solib", result.stdout))

  def test_a_new_dependency_moves_the_deps_stamp(self):
    self.add_runfiles_tree(
        "b", deps=["rules_python++pip+torch_tpu_pypi_312_numpy"]
    )
    result = self.run_stage("--session", self.session)
    self.assertNotEqual(self.deps_stamp, self.stamp_for("deps", result.stdout))

  def test_rebuilding_an_extension_module_moves_the_solib_stamp(self):
    """Otherwise a code change gets served out of a stale base cache."""
    lib = os.path.join(
        self.out,
        "tests",
        "a.runfiles",
        "_main",
        "csrc",
        "common",
        "libpywrap.so",
    )
    self._write(lib, "binary")
    first = self.run_stage("--session", self.session)
    self._write(lib, "binary-rebuilt-and-longer")
    second = self.run_stage("--session", self.session)
    self.assertNotEqual(
        self.stamp_for("solib", first.stdout),
        self.stamp_for("solib", second.stdout),
    )

  def test_an_unchanged_workspace_reuses_both_tarballs(self):
    result = self.run_stage("--session", self.session)
    self.assertEqual(self.deps_stamp, self.stamp_for("deps", result.stdout))
    self.assertEqual(self.solib_stamp, self.stamp_for("solib", result.stdout))
    self.assertNotIn("packing deps layer", result.stderr)
    self.assertNotIn("packing solib layer", result.stderr)


class TestStageRelayBaseGuards(StageRelayBaseTestCase):

  def test_requires_a_pool_or_a_session(self):
    result = self.run_stage()
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("pass --pool or --session", result.stderr)

  def test_rejects_an_unknown_argument(self):
    result = self.run_stage("--session", self.session_file(), "--turbo")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unknown argument", result.stderr)

  def test_rejects_a_non_numeric_job_count(self):
    result = self.run_stage("--session", self.session_file(), "--jobs", "lots")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("--jobs must be a positive integer", result.stderr)

  def test_refuses_to_leave_the_allowed_project(self):
    self.add_runfiles_tree(
        "a", deps=["rules_python++pip+torch_tpu_pypi_312_torch"]
    )
    result = self.run_stage(
        "--session",
        self.session_file(),
        env={"CLOUDSDK_CORE_PROJECT": "some-other-project"},
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("rbe-tpu-oss", result.stderr)

  def test_skips_a_session_that_never_finished_provisioning(self):
    self.add_runfiles_tree(
        "a",
        deps=["rules_python++pip+torch_tpu_pypi_312_torch"],
        solibs=["_U_libtorch"],
    )
    result = self.run_stage("--session", self.session_file())
    self.assertIn("incomplete session file", result.stderr)
    self.assertIn("staged 0/1", result.stdout)


class TestRelaySshHelper(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):

  def source_and_run(self, snippet):
    return subprocess.run(
        ["bash", "-c", f'source "{RELAY_SSH}"\n{snippet}'],
        capture_output=True,
        text=True,
        timeout=60,
    )

  def test_defines_the_functions_both_callers_rely_on(self):
    result = self.source_and_run(
        "declare -F relay_ssh_ensure_master relay_ssh_master_alive"
        " relay_ssh_identity"
    )
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_publishes_shared_ssh_options(self):
    result = self.source_and_run('printf "%s\\n" "${RELAY_SSH_OPTS[@]}"')
    self.assertIn("BatchMode=yes", result.stdout)
    self.assertIn("ConnectTimeout=10", result.stdout)

  def test_identity_defaults_to_the_gce_key_and_honours_an_override(self):
    default = self.source_and_run("relay_ssh_identity")
    self.assertTrue(
        default.stdout.strip().endswith("/.ssh/google_compute_engine")
    )
    override = self.source_and_run(
        "SSH_IDENTITY=/tmp/other-key relay_ssh_identity"
    )
    self.assertEqual(override.stdout.strip(), "/tmp/other-key")

  def test_reports_a_missing_master_socket_as_dead(self):
    result = self.source_and_run(
        "relay_ssh_master_alive /tmp/definitely-not-a-socket-12345"
        ' user@203.0.113.1; echo "rc=$?"'
    )
    self.assertIn("rc=1", result.stdout)

  def test_master_outlives_its_parent_and_persists(self):
    """setsid and ControlPersist are what keep the master alive after `up`."""
    with open(RELAY_SSH) as fh:
      body = fh.read()
    self.assertIn("setsid ssh -M -N -f", body)
    self.assertIn("ControlPersist=4h", body)


class TestProvisioningScriptsSupportOnDemand(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Spot v5e gets preempted within minutes, so on-demand has to be reachable."""

  def read(self, relpath):
    with open(os.path.join(REPO_ROOT, relpath)) as fh:
      return fh.read()

  def test_manager_only_passes_spot_when_not_on_demand(self):
    body = self.read("scripts/spot_tpu_manager.sh")
    self.assertIn(
        '[[ "$CLI_ON_DEMAND" == "true" ]] || create_args+=(--spot)', body
    )

  def test_manager_accepts_the_on_demand_flag(self):
    body = self.read("scripts/spot_tpu_manager.sh")
    self.assertEqual(
        body.count("--on-demand)"), 2, "both subcommands should accept it"
    )

  def test_fleet_forwards_the_on_demand_flag(self):
    body = self.read("scripts/spot_tpu_fleet.sh")
    self.assertIn("up_args+=(--on-demand)", body)

  def test_manager_no_longer_installs_its_own_python(self):
    """The image ships 3.10; bazel builds against a hermetic 3.12."""
    body = self.read("scripts/spot_tpu_manager.sh")
    self.assertNotIn("pip install", body)
    self.assertNotIn("libtpu==", body)


class StreamPayloadTestCase(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Runs the real stream_payload against a stand-in runfiles tree.

  The function is lifted out of relay_test_runner.sh rather than reimplemented,
  so these tests fail if the archive layout drifts.
  """

  VENV_REL = "_main/tests/pkg.venv"
  SITE_REL = VENV_REL + "/lib/python3.12/site-packages"

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.root = self._tmp.name

    self.runfiles = os.path.join(self.root, "pkg.runfiles")
    self.host_only = os.path.join(self.root, "host_only")

    # Bazel points these at the local output base. Their bytes are real and
    # only exist here, so the payload has to dereference them.
    self._write(os.path.join(self.host_only, "pyvenv.cfg"), "")
    self._write(
        os.path.join(self.host_only, "bazel.pth"), "import _bazel_site_init\n"
    )

    self._write(
        os.path.join(self.runfiles, "_main/src/torch_tpu/__init__.py"),
        "x = 1\n",
    )
    self._write(
        os.path.join(
            self.runfiles, "repo_torch/site-packages/torch/__init__.py"
        ),
        "y = 2\n",
    )

    self._link("/pyvenv.cfg", self.VENV_REL + "/pyvenv.cfg", host_only=True)
    self._link("/bazel.pth", self.SITE_REL + "/bazel.pth", host_only=True)

    # Resolves against the base cache once the executor links it in.
    self._link(
        "../../../../../../repo_torch/site-packages/torch",
        self.SITE_REL + "/torch",
    )
    # Names a repository the executor repoints; must stay a link or the
    # payload balloons to gigabytes.
    self._link(
        os.path.join(
            self.root, "output_base/external/repo_torch/site-packages/extra.py"
        ),
        self.SITE_REL + "/extra.py",
    )

  def _write(self, path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
      fh.write(text)

  def _link(self, target, relpath, host_only=False):
    path = os.path.join(self.runfiles, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.symlink(self.host_only + target if host_only else target, path)

  def _harness(self):
    """Extracts PAYLOAD_EXCLUDES and stream_payload into a runnable script."""
    with open(RELAY_RUNNER) as fh:
      lines = fh.read().splitlines()

    start = next(
        i
        for i, l in enumerate(lines)
        if l.startswith("readonly PAYLOAD_EXCLUDES=(")
    )
    body_start = next(
        i for i, l in enumerate(lines) if l.startswith("stream_payload() {")
    )
    end = next(i for i in range(body_start, len(lines)) if lines[i] == "}")

    script = os.path.join(self.root, "harness.sh")
    with open(script, "w") as fh:
      fh.write('#!/usr/bin/env bash\nset -uo pipefail\nRUNFILES_ROOT="$1"\n')
      fh.write("\n".join(lines[start : end + 1]))
      fh.write("\nstream_payload\n")
    os.chmod(script, 0o755)
    return script

  def listing(self):
    """Every member of the concatenated payload, in the order tar sees them."""
    payload = subprocess.run(
        [self._harness(), self.runfiles], capture_output=True, timeout=120
    ).stdout
    out = subprocess.run(
        ["tar", "-tzvif", "-"], input=payload, capture_output=True, timeout=120
    ).stdout.decode()

    members = []
    for line in out.splitlines():
      fields = line.split(None, 5)
      if len(fields) < 6:
        continue
      name = fields[5].split(" -> ")[0]
      members.append((name.lstrip("./").rstrip("/"), line[0]))
    return members

  def kind_of(self, members, relpath):
    matches = [kind for name, kind in members if name == relpath]
    self.assertEqual(
        len(matches), 1, f"{relpath} should appear exactly once, got {matches}"
    )
    return matches[0]


class TestStreamPayloadArchivesAreDisjoint(StreamPayloadTestCase):
  """GNU tar 1.34 keeps an existing symlink instead of overwriting it.

  The TPU image ships 1.34, so the first archive to write a name wins. Any
  duplicate name across the three archives is a silent correctness bug.
  """

  def test_no_member_is_written_twice(self):
    members = self.listing()
    names = [name for name, _ in members if name]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    self.assertEqual(
        duplicates,
        [],
        f"payload writes these names more than once: {duplicates}",
    )

  def test_host_only_venv_links_arrive_as_real_files(self):
    """pyvenv.cfg and bazel.pth decide whether the venv activates at all."""
    members = self.listing()
    self.assertEqual(self.kind_of(members, self.VENV_REL + "/pyvenv.cfg"), "-")
    self.assertEqual(self.kind_of(members, self.SITE_REL + "/bazel.pth"), "-")

  def test_repository_links_stay_links(self):
    """Dereferencing these would put gigabytes on the wire per test action."""
    members = self.listing()
    self.assertEqual(self.kind_of(members, self.SITE_REL + "/torch"), "l")
    self.assertEqual(self.kind_of(members, self.SITE_REL + "/extra.py"), "l")

  def test_tree_outside_the_venv_is_dereferenced(self):
    members = self.listing()
    self.assertEqual(
        self.kind_of(members, "_main/src/torch_tpu/__init__.py"), "-"
    )


class TestStreamPayloadLeavesSharedBuildOutputsOut(StreamPayloadTestCase):
  """The base cache already holds these, and they are the bulk of the bytes.

  libpywrap_torch_tpu_common.so under _main/csrc is 493 MB of a 499 MB payload
  and is the same build for every test, so shipping it per action cost ~15s of
  gzip on every payload cache miss.
  """

  def setUp(self):
    super().setUp()
    self._write(
        os.path.join(
            self.runfiles, "_main/csrc/common/libpywrap_torch_tpu_common.so"
        ),
        "a very large binary",
    )
    self._write(
        os.path.join(self.runfiles, "_main/_solib_x86_64/_U_libtorch/lib.so"),
        "another large binary",
    )
    # Not ours and not in the base cache: only _main/csrc is excluded.
    self._write(
        os.path.join(
            self.runfiles, "repo_torch/site-packages/torch/csrc/api.h"
        ),
        "#pragma once\n",
    )

  def names(self):
    return {name for name, _ in self.listing()}

  def test_extension_modules_stay_out(self):
    names = self.names()
    self.assertNotIn("_main/csrc/common/libpywrap_torch_tpu_common.so", names)
    self.assertNotIn("_main/csrc", names)

  def test_shared_libraries_stay_out(self):
    self.assertNotIn("_main/_solib_x86_64/_U_libtorch/lib.so", self.names())

  def test_other_directories_named_csrc_still_ship(self):
    self.assertIn("repo_torch/site-packages/torch/csrc/api.h", self.names())


class TestPayloadKeyIsRunScoped(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """The cache key decides whether a stale tree can ever be served.

  It is lifted out of relay_test_runner.sh rather than reimplemented, so these
  tests fail if the derivation changes.
  """

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)

    with open(RELAY_RUNNER, encoding="utf-8") as fh:
      lines = fh.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l == 'payload_key=""')
    end = next(i for i in range(start, len(lines)) if lines[i] == "fi")

    self.script = os.path.join(self._tmp.name, "payload_key.sh")
    with open(self.script, "w", encoding="utf-8") as fh:
      fh.write(
          "#!/usr/bin/env bash\nset -uo"
          ' pipefail\nRUNFILES_ROOT="$1"\nremote_env=""\n'
      )
      fh.write("\n".join(lines[start : end + 1]))
      fh.write('\nprintf "key=%s\\nenv=%s\\n" "$payload_key" "$remote_env"\n')

  def derive(self, runfiles_root, run_id=None):
    env = {"PATH": os.environ["PATH"]}
    if run_id is not None:
      env["TORCH_TPU_RELAY_RUN_ID"] = run_id
    out = subprocess.run(
        ["bash", self.script, runfiles_root],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    fields = dict(line.split("=", 1) for line in out.splitlines())
    return fields["key"], fields["env"]

  def test_key_names_the_run_and_the_tree(self):
    key, remote_env = self.derive("/tmp/pkg.runfiles", run_id="r1789p42")
    self.assertRegex(key, r"^r1789p42_[0-9a-f]{16}$")
    self.assertIn(f"TORCH_TPU_PAYLOAD_KEY={key}", remote_env)

  def test_shards_of_one_target_share_a_key(self):
    """This is the whole point: shards 2..50 reuse shard 1's prepared tree."""
    first, _ = self.derive("/tmp/pkg.runfiles", run_id="r1789p42")
    second, _ = self.derive("/tmp/pkg.runfiles", run_id="r1789p42")
    self.assertEqual(first, second)

  def test_different_targets_get_different_keys(self):
    one, _ = self.derive("/tmp/a.runfiles", run_id="r1789p42")
    two, _ = self.derive("/tmp/b.runfiles", run_id="r1789p42")
    self.assertNotEqual(one, two)

  def test_a_later_run_cannot_reuse_an_earlier_tree(self):
    old, _ = self.derive("/tmp/pkg.runfiles", run_id="r1789p42")
    new, _ = self.derive("/tmp/pkg.runfiles", run_id="r1790p43")
    self.assertNotEqual(old, new)

  def test_no_run_id_disables_caching(self):
    """A bare `bazel test --run_under` has no run id, and no build barrier."""
    key, remote_env = self.derive("/tmp/pkg.runfiles")
    self.assertEqual(key, "")
    self.assertNotIn("TORCH_TPU_PAYLOAD_KEY", remote_env)


class RemoteExecutorCacheTestCase(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Runs the real executor against a stand-in payload, no TPU involved.

  The test binary is a shell script, so the executor never reaches the
  hermetic interpreter it would normally find in the base cache.
  """

  RUN_ID = "r1789p42"

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.root = self._tmp.name

    self.base_cache = os.path.join(self.root, "base")
    self.payload_cache = os.path.join(self.root, "payloads")
    os.makedirs(os.path.join(self.base_cache, "repo_fake"))
    os.makedirs(self.payload_cache)
    with open(os.path.join(self.base_cache, "repo_fake", "dep.py"), "w") as fh:
      fh.write("z = 3\n")

    tree = os.path.join(self.root, "tree")
    os.makedirs(os.path.join(tree, "_main", "tests"))
    self.test_bin = os.path.join(tree, "_main", "tests", "fake_test.sh")
    with open(self.test_bin, "w") as fh:
      # cwd is the workspace root, so this lands beside it in the sandbox.
      fh.write("#!/usr/bin/env bash\ntouch ../ran\n")
    os.chmod(self.test_bin, 0o755)

    # Aimed at the host's output base, which means nothing here. The executor
    # has to repoint it at the base cache.
    os.symlink(
        os.path.join(self.root, "output_base/external/repo_fake/dep.py"),
        os.path.join(tree, "_main", "tests", "dep.py"),
    )

    self.payload = subprocess.run(
        ["tar", "-czf", "-", "-C", tree, "."], capture_output=True, check=True
    ).stdout

  def run_executor(self, key=None, payload=None, sandbox=None):
    """Runs one action. Passing no payload is what a cache hit looks like."""
    sandbox = sandbox or os.path.join(
        self.root, f"sandbox_{time.monotonic_ns()}"
    )
    env = {
        "PATH": os.environ["PATH"],
        "HOME": self.root,
        "TORCH_TPU_BASE_CACHE": self.base_cache,
        "TORCH_TPU_PAYLOAD_CACHE": self.payload_cache,
    }
    if key is not None:
      env["TORCH_TPU_PAYLOAD_KEY"] = key
    result = subprocess.run(
        ["bash", REMOTE_EXECUTOR, sandbox, "tests/fake_test.sh"],
        env=env,
        input=payload if payload is not None else b"",
        capture_output=True,
        timeout=120,
        check=False,
    )
    return sandbox, result

  def key(self, suffix="aaaaaaaaaaaaaaaa", run_id=None):
    return f"{run_id or self.RUN_ID}_{suffix}"

  def cached_trees(self):
    return sorted(os.listdir(self.payload_cache))


class TestRemoteExecutorPayloadCache(RemoteExecutorCacheTestCase):
  """A miss prepares the tree once; every later shard clones it."""

  def test_miss_publishes_the_prepared_tree(self):
    key = self.key()
    sandbox, result = self.run_executor(key=key, payload=self.payload)
    self.assertEqual(result.returncode, 0, result.stderr.decode())
    self.assertTrue(os.path.exists(os.path.join(sandbox, "ran")))
    self.assertEqual(self.cached_trees(), [key])
    self.assertTrue(
        os.path.exists(
            os.path.join(self.payload_cache, key, "_main/tests/fake_test.sh")
        )
    )

  def test_the_cache_holds_no_test_output(self):
    """Publishing happens before the test runs, or shard 2 inherits shard 1."""
    key = self.key()
    self.run_executor(key=key, payload=self.payload)
    published = os.listdir(os.path.join(self.payload_cache, key))
    self.assertNotIn("ran", published)
    self.assertNotIn("test.xml", published)

  def test_hit_runs_with_no_payload_at_all(self):
    key = self.key()
    self.run_executor(key=key, payload=self.payload)

    sandbox, result = self.run_executor(key=key)
    self.assertEqual(result.returncode, 0, result.stderr.decode())
    self.assertTrue(os.path.exists(os.path.join(sandbox, "ran")))
    self.assertTrue(
        os.path.exists(os.path.join(sandbox, "_main/tests/fake_test.sh"))
    )

  def test_only_a_miss_repairs_links(self):
    """The whole-tree link walk is the slow part a hit is meant to skip."""
    key = self.key()
    _, miss = self.run_executor(key=key, payload=self.payload)
    _, hit = self.run_executor(key=key)
    self.assertIn("Repointed", miss.stderr.decode())
    self.assertNotIn("Repointed", hit.stderr.decode())

  def test_cached_links_survive_a_differently_named_sandbox(self):
    """Links have to name the base cache, not the sandbox that prepared them."""
    key = self.key()
    self.run_executor(key=key, payload=self.payload)

    cached_link = os.path.join(self.payload_cache, key, "_main/tests/dep.py")
    self.assertEqual(
        os.readlink(cached_link),
        os.path.join(self.base_cache, "repo_fake/dep.py"),
    )

    sandbox, _ = self.run_executor(key=key)
    self.assertTrue(os.path.exists(os.path.join(sandbox, "_main/tests/dep.py")))

  def test_no_key_caches_nothing(self):
    sandbox, result = self.run_executor(payload=self.payload)
    self.assertEqual(result.returncode, 0, result.stderr.decode())
    self.assertTrue(os.path.exists(os.path.join(sandbox, "ran")))
    self.assertEqual(self.cached_trees(), [])


class TestRemoteExecutorCacheEviction(RemoteExecutorCacheTestCase):
  """Trees from an earlier run can never be reused, so they get reclaimed."""

  def test_earlier_runs_are_dropped_and_this_run_is_kept(self):
    stale = self.key(suffix="bbbbbbbbbbbbbbbb", run_id="r1700p1")
    sibling = self.key(suffix="cccccccccccccccc")
    for name in (stale, sibling):
      os.makedirs(os.path.join(self.payload_cache, name))

    key = self.key()
    _, result = self.run_executor(key=key, payload=self.payload)
    self.assertEqual(result.returncode, 0, result.stderr.decode())
    self.assertEqual(self.cached_trees(), sorted([key, sibling]))


class TestRemoteExecutorBaseCacheLinks(RemoteExecutorCacheTestCase):
  """What the payload leaves out has to come back from the base cache."""

  def setUp(self):
    super().setUp()
    self.csrc_lib = os.path.join(
        self.base_cache, "csrc", "common", "libpywrap.so"
    )
    os.makedirs(os.path.dirname(self.csrc_lib))
    with open(self.csrc_lib, "w") as fh:
      fh.write("extension module\n")

    self.solib = os.path.join(
        self.base_cache, "_solib_x86_64", "_U_libtorch", "lib.so"
    )
    os.makedirs(os.path.dirname(self.solib))
    with open(self.solib, "w") as fh:
      fh.write("shared library\n")

  def test_a_miss_links_both_into_the_workspace_root(self):
    sandbox, result = self.run_executor(key=self.key(), payload=self.payload)
    self.assertEqual(result.returncode, 0, result.stderr.decode())
    for name, expected in (
        ("csrc", self.csrc_lib),
        ("_solib_x86_64", self.solib),
    ):
      link = os.path.join(sandbox, "_main", name)
      self.assertTrue(os.path.islink(link), f"{name} should be a symlink")
      self.assertEqual(os.readlink(link), os.path.join(self.base_cache, name))
      self.assertTrue(os.path.exists(expected))

  def test_a_cache_hit_resolves_the_same_files(self):
    key = self.key()
    self.run_executor(key=key, payload=self.payload)
    sandbox, result = self.run_executor(key=key)
    self.assertEqual(result.returncode, 0, result.stderr.decode())
    self.assertTrue(
        os.path.exists(os.path.join(sandbox, "_main/csrc/common/libpywrap.so"))
    )

  def test_nothing_is_linked_when_the_base_cache_has_no_extension_modules(self):
    shutil.rmtree(os.path.join(self.base_cache, "csrc"))
    sandbox, result = self.run_executor(key=self.key(), payload=self.payload)
    self.assertEqual(result.returncode, 0, result.stderr.decode())
    self.assertFalse(os.path.lexists(os.path.join(sandbox, "_main", "csrc")))


class TestRelayRunnerQuarantine(RelayPoolTestCase):
  """A VM that has gone bad must stop taking work.

  A broken VM fails in milliseconds, so it frees its lock sooner than a healthy
  VM finishes a test and wins the next lease race. One rebooted VM took 64
  shards that way while seven healthy VMs sat idle.
  """

  def quarantine(self, session_path, reason="bad vm"):
    with open(f"{session_path}.quarantine", "w", encoding="utf-8") as fh:
      fh.write(f"2026-09-11T00:00:00Z\t{reason}\n")

  def test_a_quarantined_vm_is_passed_over(self):
    bad = self.add_pool_vm("vm_0")
    self.add_pool_vm("vm_1")
    self.quarantine(bad)

    self.run_relay(TPU_SESSION_POOL=self.pool)
    self.assertIn("vm_1.env", self.leased_session())

  def test_a_quarantined_vm_is_skipped_even_when_it_is_free(self):
    """The bad VM is unlocked and first in the listing; it must still lose."""
    bad = self.add_pool_vm("vm_0")
    busy = self.add_pool_vm("vm_1")
    self.add_pool_vm("vm_2")
    self.quarantine(bad)
    self.hold_lock(busy)

    self.run_relay(TPU_SESSION_POOL=self.pool)
    self.assertIn("vm_2.env", self.leased_session())

  def test_a_fully_quarantined_pool_reports_no_free_vm(self):
    for name in ("vm_0", "vm_1"):
      self.quarantine(self.add_pool_vm(name))

    result = self.run_relay(TPU_SESSION_POOL=self.pool, TEST_TIMEOUT="2")
    self.assertEqual(result.returncode, 1)
    self.assertEqual(
        self.read_xml().find("testcase/failure").get("message"),
        "No free TPU in pool",
    )


class TestCheckPreemption(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Tells "the VM is gone" apart from "I could not ask".

  Collapsing the two reported 64 shards as preempted against a VM that was
  READY the whole time, which pointed the investigation at Google Cloud
  instead of at the guest reboot we caused ourselves.
  """

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.root = self._tmp.name

    self.bin_dir = os.path.join(self.root, "bin")
    os.makedirs(self.bin_dir)

    with open(RELAY_RUNNER, encoding="utf-8") as fh:
      lines = fh.read().splitlines()
    start = next(
        i for i, l in enumerate(lines) if l.startswith("check_preemption() {")
    )
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")

    self.script = os.path.join(self.root, "check_preemption.sh")
    with open(self.script, "w", encoding="utf-8") as fh:
      fh.write(
          "#!/usr/bin/env bash\nset -uo"
          ' pipefail\nALLOWED_PROJECT="rbe-tpu-oss"\nTPU_NAME="vm-under-test"\nTPU_ZONE="europe-west4-b"\nTPU_IP="10.0.0.1"\nwrite_stub_xml()'
          ' { printf "STUB:%s\\n" "$1"; }\n'
      )
      fh.write("\n".join(lines[start : end + 1]))
      fh.write("\ncheck_preemption\nprintf 'rc=%s\\n' \"$?\"\n")

  def fake_gcloud(self, stdout="", exit_code=0):
    path = os.path.join(self.bin_dir, "gcloud")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(
          f'#!/usr/bin/env bash\nprintf "%s" "{stdout}"\nexit {exit_code}\n'
      )
    os.chmod(path, 0o755)

  def verdict(self):
    out = subprocess.run(
        ["bash", self.script],
        env={"PATH": f"{self.bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    ).stdout
    stub = next(
        (l[5:] for l in out.splitlines() if l.startswith("STUB:")), None
    )
    rc = next((l[3:] for l in out.splitlines() if l.startswith("rc=")), None)
    return stub, rc

  def test_a_failed_query_is_not_evidence_of_preemption(self):
    self.fake_gcloud(exit_code=1)
    stub, rc = self.verdict()
    self.assertEqual(stub, "TPU State Unknown")
    self.assertEqual(rc, "1")

  def test_an_empty_answer_is_not_evidence_of_preemption(self):
    """gcloud can exit 0 and print nothing when it is being rate limited."""
    self.fake_gcloud(stdout="")
    stub, rc = self.verdict()
    self.assertEqual(stub, "TPU State Unknown")

  def test_a_real_preemption_is_still_reported(self):
    self.fake_gcloud(stdout="PREEMPTED")
    stub, rc = self.verdict()
    self.assertEqual(stub, "Spot TPU Preempted")
    self.assertEqual(rc, "0")

  def test_a_deleted_vm_is_reported_as_preemption(self):
    self.fake_gcloud(stdout="TERMINATED")
    stub, _ = self.verdict()
    self.assertEqual(stub, "Spot TPU Preempted")

  def test_a_live_node_that_wont_answer_ssh_is_a_connection_failure(self):
    """What a guest reboot looks like: node READY, control master dead."""
    self.fake_gcloud(stdout="READY")
    stub, rc = self.verdict()
    self.assertEqual(stub, "SSH Connection Failed")
    self.assertEqual(rc, "1")


FLEET_SCRIPT = os.path.join(REPO_ROOT, "scripts", "spot_tpu_fleet.sh")
SPOT_MANAGER = os.path.join(REPO_ROOT, "scripts", "spot_tpu_manager.sh")


class FleetTeardownTestCase(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Runs spot_tpu_fleet.sh against a fake gcloud that records what it was asked."""

  ZONE = "europe-west4-b"

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.root = self._tmp.name
    self.pool = os.path.join(self.root, "pool")
    os.makedirs(self.pool)

    self.bin_dir = os.path.join(self.root, "bin")
    os.makedirs(self.bin_dir)
    self.gcloud_log = os.path.join(self.root, "gcloud.log")

  def fake_gcloud(self, listed_names):
    """Installs a gcloud that logs every call and lists the given TPU names."""
    listing_file = os.path.join(self.root, "listing.txt")
    with open(listing_file, "w", encoding="utf-8") as fh:
      fh.write("".join(f"{name}\n" for name in listed_names))

    path = os.path.join(self.bin_dir, "gcloud")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "{self.gcloud_log}"
# argv is: compute tpus tpu-vm <verb> ...
[[ "$4" == "list" ]] && cat "{listing_file}"
exit 0
""")
    os.chmod(path, 0o755)

  def run_fleet(self, *args):
    return subprocess.run(
        ["bash", FLEET_SCRIPT, *args],
        env={"PATH": f"{self.bin_dir}:{os.environ['PATH']}", "HOME": self.root},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

  def gcloud_calls(self):
    if not os.path.exists(self.gcloud_log):
      return []
    with open(self.gcloud_log, encoding="utf-8") as fh:
      return [line.strip() for line in fh if line.strip()]

  def delete_calls(self):
    return [c for c in self.gcloud_calls() if " delete " in f" {c} "]


class TestFleetOrphanSweep(FleetTeardownTestCase):
  """`down` has to clear VMs that no session file ever recorded.

  Provisioning has created a node and then failed before writing its session.
  The pool never learned about it, so teardown walked straight past it and it
  billed for hours.
  """

  def test_down_deletes_a_vm_with_no_session_file(self):
    self.fake_gcloud(["spot-tpu-v5e-111-1", "spot-tpu-v5e-222-2"])
    proc = self.run_fleet("down", "--pool", self.pool, "--zone", self.ZONE)
    self.assertEqual(proc.returncode, 0, proc.stderr)

    deleted = self.delete_calls()
    self.assertEqual(len(deleted), 2, deleted)
    for name in ("spot-tpu-v5e-111-1", "spot-tpu-v5e-222-2"):
      self.assertTrue(
          any(name in call and self.ZONE in call for call in deleted),
          f"{name} was never deleted: {deleted}",
      )

  def test_the_sweep_only_looks_at_vms_this_tooling_named(self):
    """Without the name filter the sweep would delete other people's TPUs."""
    self.fake_gcloud([])
    self.run_fleet("down", "--pool", self.pool, "--zone", self.ZONE)

    listings = [c for c in self.gcloud_calls() if " list " in f" {c} "]
    self.assertTrue(listings, self.gcloud_calls())
    for call in listings:
      self.assertIn("--filter=name~spot-tpu-v5e-", call)

  def test_an_empty_zone_deletes_nothing(self):
    self.fake_gcloud([])
    proc = self.run_fleet("down", "--pool", self.pool, "--zone", self.ZONE)
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertEqual(self.delete_calls(), [])


class TestFleetDeadline(FleetTeardownTestCase):
  """The deadline is the only thing standing between a killed orchestrator and

  a fleet that bills until someone notices.
  """

  def test_the_deadline_tears_the_fleet_down_when_it_lands(self):
    self.fake_gcloud(["spot-tpu-v5e-333-3"])
    proc = self.run_fleet(
        "deadline",
        "--pool",
        self.pool,
        "--zone",
        self.ZONE,
        "--deadline-minutes",
        "0",
    )
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertEqual(len(self.delete_calls()), 1, self.gcloud_calls())

  def test_down_cancels_an_armed_deadline(self):
    """A reaper left running would delete whatever fleet is up when it fires."""
    self.fake_gcloud([])
    sleeper = subprocess.Popen(["sleep", "300"])
    self.addCleanup(sleeper.terminate)
    pid_file = os.path.join(self.pool, "reaper.pid")
    with open(pid_file, "w", encoding="utf-8") as fh:
      fh.write(str(sleeper.pid))

    self.run_fleet("down", "--pool", self.pool, "--zone", self.ZONE)

    self.assertFalse(os.path.exists(pid_file))
    self.assertEqual(sleeper.wait(timeout=30), -signal.SIGTERM)

  def test_a_deadline_of_zero_arms_nothing(self):
    self.fake_gcloud([])
    self.run_fleet("down", "--pool", self.pool, "--deadline-minutes", "0")
    self.assertFalse(os.path.exists(os.path.join(self.pool, "reaper.pid")))

  def test_a_non_numeric_deadline_is_rejected(self):
    self.fake_gcloud([])
    proc = self.run_fleet(
        "down", "--pool", self.pool, "--deadline-minutes", "2h"
    )
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("--deadline-minutes", proc.stderr)


class TestNoGuestSideShutdown(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Nothing may schedule a halt inside the guest.

  `shutdown -h` does not release a TPU node. The service restarts the guest, so
  it keeps billing and comes back with /tmp wiped and the SSH control master
  dead. One firing mid-run took out 145 shards.
  """

  BANNED = ("shutdown -h", "shutdown -c", "poweroff", "halt -p")
  SCRIPTS = (SPOT_MANAGER, FLEET_SCRIPT, RELAY_RUNNER, REMOTE_EXECUTOR)

  def test_no_script_schedules_a_halt_on_the_vm(self):
    for script in self.SCRIPTS:
      with open(script, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

      for number, line in enumerate(lines, start=1):
        # The comment standing in for the removed watchdog names these on
        # purpose, so only code counts.
        if line.lstrip().startswith("#"):
          continue
        for banned in self.BANNED:
          self.assertNotIn(
              banned,
              line,
              f"{os.path.basename(script)}:{number} runs '{banned}':"
              f" {line.strip()}",
          )


class TestResolveOutcome(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """The mapping from "what came back off the VM" to "what bazel is told".

  Every branch has to leave a JUnit report behind. Bazel prints a bare FAILED
  with no explanation when a test action produces none, which is how a fleet
  problem ends up looking like a broken test.
  """

  PASSING_XML = (
      '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
      ' failures="0"><testcase name="c"/></testsuite></testsuites>'
  )

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.root = self._tmp.name
    self.xml_path = os.path.join(self.root, "test.xml")

    with open(RELAY_RUNNER, encoding="utf-8") as fh:
      lines = fh.read().splitlines()
    for name in ("write_stub_xml", "resolve_outcome"):
      start = next(
          i for i, l in enumerate(lines) if l.startswith(f"{name}() {{")
      )
      end = next(i for i in range(start, len(lines)) if lines[i] == "}")
      setattr(self, f"_{name}_src", "\n".join(lines[start : end + 1]))

  def resolve(self, ssh_rc, meta_rc=0, remote_exitcode="0", remote_xml=""):
    script = os.path.join(self.root, "resolve.sh")
    with open(script, "w", encoding="utf-8") as fh:
      fh.write(
          "#!/usr/bin/env bash\nset -uo pipefail\n"
          f'CLEAN_TARGET="tests_relay_smoke_test"\nTIMEOUT_SEC=900\nTPU_NAME="vm-under-test"\n'
          f"ssh_rc={ssh_rc}\nmeta_rc={meta_rc}\n"
          f'remote_exitcode="{remote_exitcode}"\nremote_xml={shlex.quote(remote_xml)}\n'
          'check_preemption() { write_stub_xml "Spot TPU Preempted" "stub";'
          " return 0; }\n"
          f"{self._write_stub_xml_src}\n{self._resolve_outcome_src}\n"
          "resolve_outcome\nprintf 'rc=%s\\n' \"$?\"\n"
      )
    proc = subprocess.run(
        ["bash", script],
        env={"PATH": os.environ["PATH"], "XML_OUTPUT_FILE": self.xml_path},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    rc = next(
        int(l[3:]) for l in proc.stdout.splitlines() if l.startswith("rc=")
    )
    return rc, self.read_report()

  def read_report(self):
    if not os.path.exists(self.xml_path):
      return None
    suite = ET.parse(self.xml_path).getroot().find("testsuite")
    failure = suite.find(".//failure")
    return {
        "failures": int(suite.get("failures", "0")),
        "message": failure.get("message") if failure is not None else None,
    }

  def test_a_real_report_is_passed_through_untouched(self):
    rc, report = self.resolve(ssh_rc=0, remote_xml=self.PASSING_XML)
    self.assertEqual(rc, 0)
    self.assertEqual(report["failures"], 0)

  def test_a_clean_exit_with_no_report_passes_when_the_fetch_worked(self):
    rc, report = self.resolve(ssh_rc=0, meta_rc=0, remote_exitcode="0")
    self.assertEqual(rc, 0)
    self.assertEqual(report["failures"], 0)

  def test_a_clean_exit_is_not_a_pass_when_the_report_could_not_be_read(self):
    """Otherwise a broken control master reads as a green test."""
    rc, report = self.resolve(ssh_rc=0, meta_rc=255, remote_exitcode="")
    self.assertEqual(rc, 1)
    self.assertEqual(report["message"], "Test Report Not Retrieved")

  def test_a_silent_vm_is_not_a_pass_even_if_the_fetch_returned_zero(self):
    rc, report = self.resolve(ssh_rc=0, meta_rc=0, remote_exitcode="")
    self.assertEqual(rc, 1)
    self.assertEqual(report["message"], "Test Report Not Retrieved")

  def test_a_timeout_is_named_as_one(self):
    rc, report = self.resolve(ssh_rc=124)
    self.assertEqual(rc, 124)
    self.assertEqual(report["message"], "Test Timeout")

  def test_ssh_255_with_a_matching_remote_code_is_the_test_failing(self):
    rc, report = self.resolve(ssh_rc=255, remote_exitcode="255")
    self.assertEqual(rc, 255)
    self.assertEqual(report["message"], "Test failed with exit code 255")

  def test_ssh_255_without_a_matching_remote_code_is_a_transport_fault(self):
    """255 from ssh alone says nothing about the test; ask the VM's state."""
    rc, report = self.resolve(ssh_rc=255, remote_exitcode="")
    self.assertEqual(rc, 255)
    self.assertEqual(report["message"], "Spot TPU Preempted")

  def test_an_ordinary_failure_keeps_its_exit_code(self):
    rc, report = self.resolve(ssh_rc=3, remote_exitcode="3")
    self.assertEqual(rc, 3)
    self.assertEqual(report["message"], "Test failed with exit code 3")

  def test_a_failing_run_that_produced_a_report_keeps_that_report(self):
    """The VM's own report says more than a synthesized stub ever could."""
    failing = self.PASSING_XML.replace('failures="0"', 'failures="1"')
    rc, report = self.resolve(ssh_rc=1, remote_exitcode="1", remote_xml=failing)
    self.assertEqual(rc, 1)
    self.assertEqual(report["failures"], 1)
    self.assertIsNone(report["message"])


FLEET_SCRIPT = os.path.join(REPO_ROOT, "scripts", "spot_tpu_fleet.sh")
DRIVER_SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_presubmit_v5_relay.sh")


class TestDeadlineReaperOutlivesItsParent(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """The reaper is the only thing standing between a crash and a billing leak.

  It used to launch under plain `nohup`, which blocks SIGHUP and nothing else.
  A process-group kill of the orchestrator took the reaper with it and eight
  VMs billed for 21 hours before anyone noticed.
  """

  def setUp(self):
    self.pool = tempfile.mkdtemp(prefix="reaper_test_")
    self.addCleanup(shutil.rmtree, self.pool, ignore_errors=True)
    self.pid_file = os.path.join(self.pool, "reaper.pid")

  def test_the_reaper_records_its_own_pid_not_its_launchers(self):
    """`arm_deadline` cannot use `$!`, because with setsid that is the launcher.

    The pid file has to name the process that actually sleeps, otherwise
    `cancel_deadline` kills a pid that has already exited and the real reaper
    survives to delete somebody else's fleet.
    """
    proc = subprocess.Popen(
        [
            "bash",
            FLEET_SCRIPT,
            "deadline",
            "--pool",
            self.pool,
            "--deadline-minutes",
            "9999",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    self.addCleanup(proc.wait)
    self.addCleanup(proc.kill)

    deadline = time.time() + 10
    while time.time() < deadline:
      if os.path.exists(self.pid_file) and os.path.getsize(self.pid_file):
        break
      time.sleep(0.05)

    self.assertTrue(os.path.exists(self.pid_file), "reaper wrote no pid file")
    with open(self.pid_file, "r", encoding="utf-8") as f:
      self.assertEqual(int(f.read().strip()), proc.pid)

  def test_the_reaper_lands_outside_its_launchers_process_group(self):
    """This is the bug that leaked eight VMs, reproduced as a test.

    `up` skips any slot that already has a session file, so a pool with one
    pre-made session arms the deadline without calling gcloud at all. The
    reaper has to end up in its own process group, because the thing that
    killed it was a group kill aimed at the orchestrator.
    """
    with open(os.path.join(self.pool, "vm_0.env"), "w", encoding="utf-8") as f:
      f.write("TPU_NAME=already-up\n")

    launcher = subprocess.Popen(
        [
            "bash",
            FLEET_SCRIPT,
            "up",
            "--size",
            "1",
            "--pool",
            self.pool,
            "--deadline-minutes",
            "9999",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    self.assertEqual(launcher.wait(timeout=60), 0)

    with open(self.pid_file, "r", encoding="utf-8") as f:
      reaper_pid = int(f.read().strip())
    self.addCleanup(self._kill, reaper_pid)

    os.kill(reaper_pid, 0)
    self.assertEqual(os.getpgid(reaper_pid), reaper_pid)
    self.assertNotEqual(os.getpgid(reaper_pid), os.getpgid(0))

  def test_arming_a_second_deadline_kills_the_first_reaper(self):
    """Two live reapers means the fleet dies at the earlier of two deadlines.

    `arm_deadline` used to `rm -f` the pid file and spawn. That forgets the old
    reaper's pid without stopping it, so it keeps sleeping and tears down
    whatever fleet is up when its own deadline lands. Seen in production with
    two reapers armed 38 minutes apart.
    """
    with open(os.path.join(self.pool, "vm_0.env"), "w", encoding="utf-8") as f:
      f.write("TPU_NAME=already-up\n")

    first_pid = self._arm_via_up()
    second_pid = self._arm_via_up()

    self.assertNotEqual(first_pid, second_pid)
    self._assert_dead(first_pid, "the first reaper outlived the second arm")
    os.kill(second_pid, 0)

  def test_running_deadline_by_hand_retires_the_previous_reaper(self):
    """Operators run `deadline` directly, so the reaper enforces this itself."""
    first = self._spawn_deadline()
    first_pid = self._await_pid_file()
    self.assertEqual(first_pid, first.pid)

    second = self._spawn_deadline()
    self.addCleanup(second.wait)
    self.addCleanup(second.kill)

    self.assertEqual(first.wait(timeout=30), -signal.SIGTERM)
    self.assertEqual(self._await_pid_file(expected=second.pid), second.pid)

  def _spawn_deadline(self):
    proc = subprocess.Popen(
        [
            "bash",
            FLEET_SCRIPT,
            "deadline",
            "--pool",
            self.pool,
            "--deadline-minutes",
            "9999",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    self.addCleanup(self._kill, proc.pid)
    return proc

  def _arm_via_up(self):
    launcher = subprocess.Popen(
        [
            "bash",
            FLEET_SCRIPT,
            "up",
            "--size",
            "1",
            "--pool",
            self.pool,
            "--deadline-minutes",
            "9999",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    self.assertEqual(launcher.wait(timeout=60), 0)
    pid = self._await_pid_file()
    self.addCleanup(self._kill, pid)
    return pid

  def _await_pid_file(self, expected=None):
    deadline = time.time() + 15
    last = None
    while time.time() < deadline:
      try:
        with open(self.pid_file, "r", encoding="utf-8") as f:
          last = int(f.read().strip())
      except (OSError, ValueError):
        last = None
      if last is not None and (expected is None or last == expected):
        return last
      time.sleep(0.05)
    self.fail(f"pid file never settled (last saw {last}, wanted {expected})")

  def _assert_dead(self, pid, message):
    deadline = time.time() + 15
    while time.time() < deadline:
      try:
        os.kill(pid, 0)
      except ProcessLookupError:
        return
      time.sleep(0.05)
    self.fail(message)

  @staticmethod
  def _kill(pid):
    try:
      os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
      pass


class TestSandboxDeleteDoesNotHoldTheLease(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Deleting an unpacked runfiles tree takes seconds the next shard could use.

  The action still owns the VM lease while the metadata fetch runs, so the
  delete is renamed out of the way and detached instead of waited on.
  """

  def test_the_fetch_renames_the_sandbox_and_detaches_the_delete(self):
    with open(RELAY_RUNNER, "r", encoding="utf-8") as f:
      body = f.read()
    fetch = next(
        line
        for line in body.splitlines()
        if "test.exitcode" in line and "remote_meta" in line
    )
    self.assertIn(".trash", fetch)
    self.assertIn("setsid rm -rf", fetch)
    self.assertNotRegex(fetch, r"rm -rf '\$\{REMOTE_SANDBOX\}'")

  def test_the_detached_delete_idiom_actually_removes_the_tree(self):
    root = tempfile.mkdtemp(prefix="sandbox_test_")
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    sandbox = os.path.join(root, "sandbox")
    os.makedirs(os.path.join(sandbox, "nested"))
    with open(os.path.join(sandbox, "nested", "f"), "w", encoding="utf-8") as f:
      f.write("x" * 1024)

    subprocess.run(
        [
            "bash",
            "-c",
            (
                f"mv '{sandbox}' '{sandbox}.trash' 2>/dev/null && {{ setsid rm"
                f" -rf '{sandbox}.trash' </dev/null >/dev/null 2>&1 & }}"
            ),
        ],
        check=True,
    )
    self.assertFalse(os.path.exists(sandbox), "rename did not happen")

    deadline = time.time() + 10
    while os.path.exists(f"{sandbox}.trash") and time.time() < deadline:
      time.sleep(0.05)
    self.assertFalse(os.path.exists(f"{sandbox}.trash"))


class TestPhaseTimingIsOptIn(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """`TORCH_TPU_RELAY_TIMING=1` has to reach the test action and the VM.

  Bazel scrubs the environment, so setting the variable in the driver's own
  shell does nothing; it has to be forwarded explicitly at both hops. Two
  earlier probe runs produced no timing at all because of exactly this.
  """

  def test_the_driver_forwards_timing_on_every_path(self):
    with open(DRIVER_SCRIPT, "r", encoding="utf-8") as f:
      body = f.read()
    forwards = re.findall(r"--test_env=TORCH_TPU_RELAY_TIMING=1", body)
    self.assertEqual(
        len(forwards),
        2,
        "both the single-VM and the pool branch rebuild relay_env from scratch",
    )

  def test_the_relay_forwards_timing_to_the_vm(self):
    with open(RELAY_RUNNER, "r", encoding="utf-8") as f:
      body = f.read()
    self.assertRegex(body, r'remote_env\+="TORCH_TPU_RELAY_TIMING=1 "')

  def test_marks_print_nothing_unless_timing_is_on(self):
    for script, marker, label in (
        (RELAY_RUNNER, "mark", "relay-timing"),
        (REMOTE_EXECUTOR, "vmark", "vm-timing"),
    ):
      with self.subTest(script=os.path.basename(script)):
        with open(script, "r", encoding="utf-8") as f:
          body = f.read()
        start = body.index(f"{marker}() {{")
        fn = body[start : body.index("\n}\n", start)]
        self.assertIn("TORCH_TPU_RELAY_TIMING", fn)
        self.assertIn(label, fn)


class TestStageRelayBaseIsolatesRuns(StageRelayBaseTestCase):
  """Two runs sharing a VM pool must not unpack over each other.

  The base directory used to be a fixed path, so a second run with different
  C++ content replaced the shared objects underneath the first run's tests.
  """

  BASE_DIR_RE = re.compile(r"^/tmp/torch_tpu_relay/base-[0-9a-f]{16}$")

  def emit_base_dir(self, **kwargs):
    emit = os.path.join(self.root, "base_dir.txt")
    proc = self.run_stage(
        "--session",
        self.session_file(),
        "--emit-base-dir",
        emit,
        **kwargs,
    )
    self.assertTrue(
        os.path.exists(emit), f"nothing emitted\n{proc.stdout}\n{proc.stderr}"
    )
    with open(emit, encoding="utf-8") as fh:
      return fh.read().strip()

  def test_the_emitted_path_is_content_addressed(self):
    self.add_runfiles_tree("a", deps=["rules_python++pip+x"], solibs=["_U_a"])
    self.assertRegex(self.emit_base_dir(), self.BASE_DIR_RE)

  def test_identical_content_reuses_the_same_directory(self):
    self.add_runfiles_tree("a", deps=["rules_python++pip+x"], solibs=["_U_a"])
    first = self.emit_base_dir()
    shutil.rmtree(self.tarballs)
    self.assertEqual(first, self.emit_base_dir())

  def test_a_changed_shared_object_moves_the_directory(self):
    tree = self.add_runfiles_tree(
        "a", deps=["rules_python++pip+x"], solibs=["_U_a"]
    )
    before = self.emit_base_dir()

    solib = os.path.join(tree, "_main", "_solib_x86_64", "_U_a", "lib.so")
    with open(solib, "w", encoding="utf-8") as fh:
      fh.write("a different build of the extension module")
    os.utime(solib, (1_700_000_000, 1_700_000_000))

    self.assertNotEqual(before, self.emit_base_dir())

  def test_a_changed_lockfile_moves_the_directory(self):
    self.add_runfiles_tree("a", deps=["rules_python++pip+x"], solibs=["_U_a"])
    before = self.emit_base_dir()
    self._write(self.lock, "lockfile v2")
    self.assertNotEqual(before, self.emit_base_dir())


class StageRelayBaseSshTestCase(StageRelayBaseTestCase):
  """Lets staging run all the way through against a recording fake ssh."""

  REMOTE_IP = "10.0.0.9"

  def setUp(self):
    super().setUp()
    self.bin_dir = os.path.join(self.root, "bin")
    os.makedirs(self.bin_dir)
    self.ssh_log = os.path.join(self.root, "ssh.log")

    # relay_ssh_master_alive insists on a real socket before it will reuse a
    # control path, so give it one instead of letting it fork a master.
    self.control_path = os.path.join(self.root, "cm.sock")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(self.control_path)
    self.addCleanup(sock.close)

    script = os.path.join(self.bin_dir, "ssh")
    with open(script, "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
printf '%s\\n' "${{@: -1}}" >> "{self.ssh_log}"
cmd="${{@: -1}}"
# A cache probe on a layer nobody has pushed yet has to report a miss.
[[ "$cmd" == *".complete' ]"* ]] && exit 1
[[ "$cmd" == tar* ]] && cat > /dev/null
exit 0
""")
    os.chmod(script, 0o755)

  def connected_session(self, name="vm_0.env"):
    path = os.path.join(self.root, name)
    self._write(
        path,
        'export TPU_NAME="fake-vm"\nexport TPU_ZONE="europe-west4-b"\n'
        f'export TPU_IP="{self.REMOTE_IP}"\nexport SSH_USER="ci"\n'
        f'export SSH_CONTROL_PATH="{self.control_path}"\n',
    )
    return path

  def run_staged(self, *args):
    return self.run_stage(
        "--session",
        self.connected_session(),
        *args,
        env={"PATH": f"{self.bin_dir}:{os.environ['PATH']}"},
    )

  def remote_commands(self):
    with open(self.ssh_log, encoding="utf-8") as fh:
      return [line.rstrip("\n") for line in fh]


class TestStageRelayBaseRemoteLayout(StageRelayBaseSshTestCase):

  def setUp(self):
    super().setUp()
    self.add_runfiles_tree("a", deps=["rules_python++pip+x"], solibs=["_U_a"])
    self.emit = os.path.join(self.root, "base_dir.txt")
    proc = self.run_staged("--emit-base-dir", self.emit)
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    with open(self.emit, encoding="utf-8") as fh:
      self.base_dir = fh.read().strip()
    self.commands = self.remote_commands()

  def test_each_layer_unpacks_into_its_own_stamped_directory(self):
    extracts = [c for c in self.commands if c.startswith("tar -xzf")]
    self.assertTrue(extracts, self.commands)
    for cmd in extracts:
      self.assertRegex(cmd, r"-C '/tmp/torch_tpu_relay/store/\w+-\w+'")

  def test_the_complete_marker_lands_after_the_archive(self):
    for layer in ("deps", "solib"):
      extract = self._index(rf"tar -xzf - -C '\S*/{layer}-")
      marker = self._index(rf"touch '\S*/{layer}-\S*/\.complete'")
      self.assertLess(extract, marker, f"{layer} marked complete too early")

  def test_the_base_directory_is_a_view_over_the_layers(self):
    link = self._command(r"^set -e; rm -rf '/tmp/torch_tpu_relay/base-")
    self.assertIn(f"mkdir -p '{self.base_dir}'", link)
    self.assertIn("ln -sfn", link)
    self.assertIn("/tmp/torch_tpu_relay/store/", link)

  def test_live_layers_are_kept_out_of_reach_of_the_sweep(self):
    link = self._command(r"^set -e; rm -rf '/tmp/torch_tpu_relay/base-")
    self.assertRegex(link, r"touch '/tmp/torch_tpu_relay/store/\w+-\w+'")
    sweep = self._command(r"^find '/tmp/torch_tpu_relay/store'")
    self.assertIn("-mtime +", sweep)

  def _index(self, pattern):
    for i, cmd in enumerate(self.commands):
      if re.search(pattern, cmd):
        return i
    self.fail(f"no remote command matched {pattern}: {self.commands}")

  def _command(self, pattern):
    return self.commands[self._index(pattern)]


class TestStageRelayBaseSkipsLayersAlreadyThere(StageRelayBaseSshTestCase):

  def test_a_complete_layer_is_not_pushed_again(self):
    self.add_runfiles_tree("a", deps=["rules_python++pip+x"], solibs=["_U_a"])
    # Report every probe as a hit.
    with open(os.path.join(self.bin_dir, "ssh"), "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
printf '%s\\n' "${{@: -1}}" >> "{self.ssh_log}"
exit 0
""")
    os.chmod(os.path.join(self.bin_dir, "ssh"), 0o755)

    proc = self.run_staged()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertEqual(
        [c for c in self.remote_commands() if c.startswith("tar -xzf")], []
    )
    self.assertIn("already current", proc.stdout)


class TestBaseDirectoryReachesTheRemoteExecutor(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """The staged path is worthless if the three hops don't pass it along."""

  def test_the_driver_forwards_the_staged_path_to_the_test_action(self):
    with open(DRIVER_SCRIPT, encoding="utf-8") as fh:
      body = fh.read()
    self.assertIn("--emit-base-dir", body)
    self.assertRegex(
        body, r'--test_env=TORCH_TPU_RELAY_BASE_DIR="\$remote_base_dir"'
    )

  def test_the_relay_honours_the_staged_path(self):
    with open(RELAY_RUNNER, encoding="utf-8") as fh:
      body = fh.read()
    self.assertRegex(body, r'REMOTE_BASE_CACHE="\$\{TORCH_TPU_RELAY_BASE_DIR:-')
    self.assertRegex(body, r"TORCH_TPU_BASE_CACHE=\$\{REMOTE_BASE_CACHE\}")

  def test_the_executor_reads_it_from_the_environment(self):
    with open(REMOTE_EXECUTOR, encoding="utf-8") as fh:
      body = fh.read()
    self.assertRegex(body, r'BASE_CACHE="\$\{TORCH_TPU_BASE_CACHE:-')


class FleetAttachTestCase(FleetTeardownTestCase):
  """A fake gcloud that can also answer `describe` and `ssh`, which attach needs."""

  def fake_gcloud_with_addresses(
      self, listed_names, ip="10.0.0.5", whoami="ci"
  ):
    listing_file = os.path.join(self.root, "listing.txt")
    with open(listing_file, "w", encoding="utf-8") as fh:
      fh.write("".join(f"{name}\n" for name in listed_names))

    path = os.path.join(self.bin_dir, "gcloud")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "{self.gcloud_log}"
# argv is: compute tpus tpu-vm <verb> ...
case "$4" in
  list) cat "{listing_file}" ;;
  describe)
    # Only the external-IP format resolves; the fallback must stay unused.
    [[ "$*" == *"externalIp"* ]] && printf '%s\\n' "{ip}"
    ;;
  ssh) printf '%s\\n' "{whoami}" ;;
esac
exit 0
""")
    os.chmod(path, 0o755)

  def sessions(self):
    return sorted(f for f in os.listdir(self.pool) if f.endswith(".env"))

  def read_session(self, name):
    values = {}
    with open(os.path.join(self.pool, name), encoding="utf-8") as fh:
      for line in fh:
        match = re.match(r'^export (\w+)="(.*)"$', line.strip())
        if match:
          values[match.group(1)] = match.group(2)
    return values


class TestFleetAttach(FleetAttachTestCase):
  """`attach` lets a second operator drive a fleet somebody else brought up.

  CI cannot provision its own v5e capacity, so it has to borrow the standing
  fleet. That only works if attaching never creates or deletes hardware.
  """

  def test_it_writes_one_session_per_ready_vm(self):
    self.fake_gcloud_with_addresses(
        ["spot-tpu-v5e-111-1", "spot-tpu-v5e-111-2"]
    )
    proc = self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertEqual(self.sessions(), ["vm_0.env", "vm_1.env"])

  def test_a_session_carries_everything_the_relay_reads(self):
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"])
    self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)

    session = self.read_session("vm_0.env")
    for key in (
        "TPU_NAME",
        "TPU_ZONE",
        "TPU_PROJECT",
        "TPU_IP",
        "SSH_CONTROL_PATH",
        "SSH_USER",
        "SSH_IDENTITY",
    ):
      self.assertIn(key, session)
    self.assertEqual(session["TPU_IP"], "10.0.0.5")
    self.assertEqual(session["TPU_ZONE"], self.ZONE)
    self.assertEqual(session["TPU_PROJECT"], "rbe-tpu-oss")
    self.assertEqual(session["SSH_CONTROL_PATH"], "/tmp/tpu_cm_10.0.0.5_22_ci")

  def test_it_creates_and_deletes_nothing(self):
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"])
    self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)

    calls = self.gcloud_calls()
    self.assertEqual(self.delete_calls(), [])
    self.assertEqual([c for c in calls if " create " in f" {c} "], [])

  def test_it_only_looks_at_ready_vms_this_tooling_named(self):
    self.fake_gcloud_with_addresses([])
    self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)

    listings = [c for c in self.gcloud_calls() if " list " in f" {c} "]
    self.assertTrue(listings, self.gcloud_calls())
    for call in listings:
      self.assertIn(
          "--filter=name~spot-tpu-v5e- AND acceleratorType~v5litepod-1$"
          " AND state:READY",
          call,
      )

  def test_attaching_twice_does_not_hand_out_a_vm_under_two_slots(self):
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"])
    self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)
    second = self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)

    self.assertEqual(self.sessions(), ["vm_0.env"])
    self.assertIn("already in the pool", second.stdout + second.stderr)

  def test_a_new_vm_lands_in_a_free_slot_next_to_the_existing_ones(self):
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"])
    self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)
    self.fake_gcloud_with_addresses(
        ["spot-tpu-v5e-111-1", "spot-tpu-v5e-111-2"]
    )
    self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)

    self.assertEqual(self.sessions(), ["vm_0.env", "vm_1.env"])
    names = {self.read_session(f)["TPU_NAME"] for f in self.sessions()}
    self.assertEqual(names, {"spot-tpu-v5e-111-1", "spot-tpu-v5e-111-2"})

  def test_it_records_the_identity_ci_will_connect_with(self):
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"])
    key = os.path.join(self.root, "ephemeral_key")
    self.run_fleet(
        "attach",
        "--pool",
        self.pool,
        "--zone",
        self.ZONE,
        "--ssh-identity",
        key,
    )
    self.assertEqual(self.read_session("vm_0.env")["SSH_IDENTITY"], key)

  def test_no_key_push_and_no_claim_skip_the_ssh_hop_entirely(self):
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"], whoami="somebody")
    self.run_fleet(
        "attach",
        "--pool",
        self.pool,
        "--zone",
        self.ZONE,
        "--ssh-user",
        "runner",
        "--no-key-push",
        "--no-claim",
    )
    self.assertEqual(self.read_session("vm_0.env")["SSH_USER"], "runner")
    self.assertEqual(
        [c for c in self.gcloud_calls() if " ssh " in f" {c} "], []
    )

  def test_no_key_push_still_claims_but_keeps_the_given_user(self):
    """Claiming needs a hop even when the key does not.

    The hop is still worth making: without the claim two people attaching at
    once both take every VM. What --no-key-push turns off is trusting the
    remote's answer about who we log in as.
    """
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"], whoami="somebody")
    self.run_fleet(
        "attach",
        "--pool",
        self.pool,
        "--zone",
        self.ZONE,
        "--ssh-user",
        "runner",
        "--no-key-push",
    )
    self.assertEqual(self.read_session("vm_0.env")["SSH_USER"], "runner")
    self.assertTrue([c for c in self.gcloud_calls() if " ssh " in f" {c} "])

  def test_a_vm_with_no_address_never_enters_the_pool(self):
    """A half-provisioned node in the pool loses every test that leases it."""
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"], ip="")
    proc = self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)
    self.assertNotEqual(proc.returncode, 0)
    self.assertEqual(self.sessions(), [])

  def test_an_empty_fleet_fails_loudly(self):
    self.fake_gcloud_with_addresses([])
    proc = self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("Bring a fleet up first", proc.stdout + proc.stderr)


class TestFleetAttachToReservedCapacity(FleetAttachTestCase):
  """`attach` has to reach reserved VMs, which nobody named `spot-tpu-v5e-`.

  The fleet this tooling creates is on-demand or Spot and gets a generated
  name. A permanent reservation is named by whoever reserved it, so a hardcoded
  prefix makes the whole fleet invisible to `attach` and the job reports "no
  VMs" while the hardware sits idle.
  """

  def listing_filters(self):
    return [
        call
        for call in self.gcloud_calls()
        if " list " in f" {call} " and "--filter=" in call
    ]

  def test_a_custom_prefix_replaces_the_default_one(self):
    self.fake_gcloud_with_addresses(["reserved-v5e-alpha"])
    proc = self.run_fleet(
        "attach",
        "--pool",
        self.pool,
        "--zone",
        self.ZONE,
        "--name-prefix",
        "reserved-v5e-",
    )

    self.assertEqual(proc.returncode, 0, proc.stderr)
    filters = self.listing_filters()
    self.assertTrue(filters, self.gcloud_calls())
    for call in filters:
      self.assertIn("name~reserved-v5e-", call)
      self.assertNotIn("name~spot-tpu-v5e-", call)

  def test_an_empty_prefix_drops_the_name_clause_entirely(self):
    self.fake_gcloud_with_addresses(["anything-at-all"])
    proc = self.run_fleet(
        "attach", "--pool", self.pool, "--zone", self.ZONE, "--name-prefix", ""
    )

    self.assertEqual(proc.returncode, 0, proc.stderr)
    for call in self.listing_filters():
      self.assertNotIn("name~", call)
      self.assertIn("state:READY", call)

  def test_the_accelerator_guard_survives_a_widened_name(self):
    """Widening the name must not let a v5p or an 8-chip host in.

    The relay leases one chip per test action. A multi-chip host accepts the
    lease and then fails every test on it, which reads as a broken change
    rather than a wrong VM.
    """
    self.fake_gcloud_with_addresses(["anything-at-all"])
    self.run_fleet(
        "attach", "--pool", self.pool, "--zone", self.ZONE, "--name-prefix", ""
    )

    for call in self.listing_filters():
      self.assertIn("acceleratorType~v5litepod-1$", call)

  def test_the_accelerator_match_is_anchored(self):
    """`v5litepod-1` must not also match `v5litepod-16`."""
    self.fake_gcloud_with_addresses(["reserved-1"])
    self.run_fleet(
        "attach",
        "--pool",
        self.pool,
        "--zone",
        self.ZONE,
        "--name-prefix",
        "reserved-",
    )

    for call in self.listing_filters():
      self.assertIn("v5litepod-1$", call)

  def test_the_accelerator_type_is_overridable(self):
    self.fake_gcloud_with_addresses(["reserved-1"])
    self.run_fleet(
        "attach",
        "--pool",
        self.pool,
        "--zone",
        self.ZONE,
        "--accelerator-type",
        "v5litepod-4",
    )

    for call in self.listing_filters():
      self.assertIn("acceleratorType~v5litepod-4$", call)

  def test_widening_attach_never_widens_what_down_deletes(self):
    """The one combination that could destroy somebody else's reservation.

    `down` sweeps orphans by name. If --name-prefix reached that sweep, an
    operator who attached with an empty prefix and then ran `down` would delete
    every TPU VM in the project.
    """
    proc = self.run_fleet(
        "down", "--pool", self.pool, "--zone", self.ZONE, "--name-prefix", ""
    )

    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("only applies to 'attach'", proc.stdout + proc.stderr)
    self.assertEqual(self.delete_calls(), [])

  def test_down_still_sweeps_on_the_creation_prefix(self):
    self.fake_gcloud(["spot-tpu-v5e-111-1"])
    self.run_fleet("down", "--pool", self.pool, "--zone", self.ZONE)

    listings = [c for c in self.gcloud_calls() if " list " in f" {c} "]
    self.assertTrue(listings, self.gcloud_calls())
    for call in listings:
      self.assertIn("name~spot-tpu-v5e-", call)


class TestFleetDetach(FleetAttachTestCase):
  """`detach` is what CI runs at the end.

  `down` there would delete somebody else's VMs.
  """

  def test_it_clears_the_pool_without_touching_the_hardware(self):
    self.fake_gcloud_with_addresses(["spot-tpu-v5e-111-1"])
    self.run_fleet("attach", "--pool", self.pool, "--zone", self.ZONE)
    self.assertEqual(self.sessions(), ["vm_0.env"])

    proc = self.run_fleet("detach", "--pool", self.pool)
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertEqual(self.sessions(), [])
    self.assertEqual(self.delete_calls(), [])

  def test_it_clears_leases_and_quarantine_markers_too(self):
    for name in ("vm_0.env", "vm_0.env.lock", "vm_0.quarantine"):
      with open(os.path.join(self.pool, name), "w", encoding="utf-8") as fh:
        fh.write("x")
    self.run_fleet("detach", "--pool", self.pool)
    self.assertEqual(os.listdir(self.pool), [])

  def test_detaching_from_nothing_is_not_an_error(self):
    proc = self.run_fleet(
        "detach", "--pool", os.path.join(self.root, "never-existed")
    )
    self.assertEqual(proc.returncode, 0, proc.stderr)


class TestTestRuleEnvReachesTheVm(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """A test rule's `env = {...}` has to survive the SSH hop.

  It did not, and three targets that are green on the upstream ct5lp runner
  went red in the relay: errors_test_tpu stopped seeing
  TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS=1 and its error cases stopped raising,
  and pallas_test stopped seeing TPU_PREMAPPED_BUFFER_SIZE=0.
  """

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.script = self._harness()

  def _harness(self):
    """Lifts the forwarding block out of the runner instead of copying it."""
    with open(RELAY_RUNNER, encoding="utf-8") as fh:
      lines = fh.read().splitlines()

    start = next(
        i
        for i, l in enumerate(lines)
        if l.startswith("readonly FORWARDED_ENV_PREFIXES=(")
    )
    end = next(
        i for i in range(start, len(lines)) if lines[i] == "done < <(env -0)"
    )

    path = os.path.join(self._tmp.name, "harness.sh")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write('#!/usr/bin/env bash\nset -uo pipefail\nremote_env=""\n')
      fh.write("\n".join(lines[start : end + 1]))
      fh.write('\nprintf "%s" "$remote_env"\n')
    os.chmod(path, 0o755)
    return path

  def forwarded(self, **env):
    out = subprocess.run(
        ["bash", self.script],
        env={"PATH": os.environ["PATH"], **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    return dict(pair.split("=", 1) for pair in shlex.split(out))

  def test_it_forwards_the_variables_those_three_targets_need(self):
    forwarded = self.forwarded(
        TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS="1",
        TPU_PREMAPPED_BUFFER_SIZE="0",
        IS_OSS="1",
    )
    self.assertEqual(forwarded["TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS"], "1")
    self.assertEqual(forwarded["TPU_PREMAPPED_BUFFER_SIZE"], "0")
    self.assertEqual(forwarded["IS_OSS"], "1")

  def test_it_covers_every_env_key_the_test_build_files_set(self):
    """The allowlist goes stale the moment somebody adds a new prefix."""
    declared = set()
    for build in (
        "tests/BUILD",
        "tests/pallas/BUILD",
        "tests/compile/BUILD",
        "tests/distributed/BUILD",
    ):
      path = os.path.join(REPO_ROOT, build)
      if not os.path.exists(path):
        continue
      with open(path, encoding="utf-8") as fh:
        body = fh.read()
      for block in re.findall(r"env\s*=\s*\{(.*?)\}", body, re.S):
        declared.update(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', block))
    self.assertTrue(declared, "no env attributes found; the scrape is broken")

    forwarded = self.forwarded(**{name: "probe" for name in sorted(declared)})
    missing = sorted(declared - set(forwarded))
    self.assertEqual(missing, [], f"these never reach the VM: {missing}")

  def test_the_host_side_of_the_hop_stays_on_the_host(self):
    """TPU_NAME means something to libtpu, and the session file's value is the

    name of the VM as gcloud sees it, not anything the runtime should act on.
    """
    forwarded = self.forwarded(
        TPU_NAME="spot-tpu-v5e-1",
        TPU_ZONE="europe-west4-b",
        TPU_IP="10.0.0.1",
        TPU_PROJECT="rbe-tpu-oss",
    )
    self.assertEqual(forwarded, {})

  def test_the_relay_does_not_forward_its_own_bookkeeping(self):
    forwarded = self.forwarded(
        TORCH_TPU_RELAY_RUN_ID="r1p2",
        TORCH_TPU_RELAY_BASE_DIR="/tmp/torch_tpu_relay/base-deadbeef",
        TORCH_TPU_BASE_CACHE="/tmp/torch_tpu_relay/base",
        TORCH_TPU_PAYLOAD_KEY="r1p2_abc",
    )
    self.assertEqual(forwarded, {})

  def test_unrelated_shell_variables_do_not_ride_along(self):
    forwarded = self.forwarded(EDITOR="vim", GOPATH="/home/x/go")
    self.assertEqual(forwarded, {})

  def test_a_value_with_shell_metacharacters_survives_intact(self):
    forwarded = self.forwarded(TORCH_LOGS_FORMAT="%(message)s; rm -rf /")
    self.assertEqual(forwarded["TORCH_LOGS_FORMAT"], "%(message)s; rm -rf /")


class RelayDriverTestCase(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests shell scripts.
):
  """Harness for scripts/relay_presubmit_pr.sh.

  Every external command the driver shells out to is replaced with a recorder:
  `gh`, the fleet script, the relay runner, and corp-ssh-helper. Nothing here
  touches GitHub, GCP, or a TPU.
  """

  HEAD_SHA = "a" * 40

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.bin = os.path.join(self.tmp.name, "bin")
    self.pool = os.path.join(self.tmp.name, "pool")
    self.out = os.path.join(self.tmp.name, "out")
    os.makedirs(self.bin)
    os.makedirs(self.pool)
    self.calls = os.path.join(self.tmp.name, "calls.log")
    # The preflight refuses to run anywhere the SSH proxy is missing.
    self.write_stub("corp-ssh-helper", "exit 0")
    self.write_stub("gcloud", "exit 0")

  def write_stub(self, name, body):
    """Installs an executable that logs its arguments, then runs `body`."""
    path = os.path.join(self.bin, name)
    lines = [
        "#!/usr/bin/env bash",
        'printf "%s %s\\n" "' + name + '" "$*" >> "$CALL_LOG"',
        body,
    ]
    with open(path, "w", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path

  def fake_gh(self, labels=("ci:replace-tpu-v5",), head_sha=None):
    """A `gh` that answers the queries the driver makes and records the rest."""
    label_output = "".join(f"{label}\n" for label in labels)
    body = "\n".join([
        'case "$*" in',
        f'  *"pulls/"*".head.sha"*) echo "{head_sha or self.HEAD_SHA}" ;;',
        f"  *\"pulls/\"*\".labels\"*) printf '%s' '{label_output}' ;;",
        '  *"comments?per_page"*) echo "" ;;',
        "  *) ;;",
        "esac",
        "exit 0",
    ])
    return self.write_stub("gh", body)

  def fake_fleet(self, vms=2):
    """A fleet script that writes session files on attach and clears on detach."""
    body = "\n".join([
        'pool=""',
        'prev=""',
        'for arg in "$@"; do',
        '  if [[ "$prev" == "--pool" ]]; then pool="$arg"; fi',
        '  prev="$arg"',
        "done",
        'if [[ "$1" == "attach" ]]; then',
        '  mkdir -p "$pool"',
        (
            f"  for i in $(seq 1 {vms}); do echo TPU_NAME=vm$i >"
            ' "$pool/vm_$i.env"; done'
        ),
        'elif [[ "$1" == "detach" ]]; then',
        '  rm -f "$pool"/*.env',
        "fi",
        "exit 0",
    ])
    return self.write_stub("spot_tpu_fleet.sh", body)

  def fake_relay(
      self, status="PASSED", failed=0, exit_code=0, write_summary=True
  ):
    """A relay runner that drops a summary where the driver looks for one."""
    summary = json.dumps({
        "status": status,
        "total_targets": 57,
        "passed_targets": 57 - failed,
        "failed_targets": failed,
        "duration_seconds": 450.8,
    })
    lines = []
    if write_summary:
      lines += [
          'out=""',
          'prev=""',
          'for arg in "$@"; do',
          '  if [[ "$prev" == "--output-dir" ]]; then out="$arg"; fi',
          '  prev="$arg"',
          "done",
          'mkdir -p "$out"',
          "cat > \"$out/presubmit_summary.json\" <<'JSON'",
          summary,
          "JSON",
          'echo "report body" > "$out/presubmit_report.md"',
      ]
    lines.append(f"exit {exit_code}")
    return self.write_stub("run_presubmit_v5_relay.sh", "\n".join(lines))

  def run_driver(self, *args, **env_overrides):
    env = {
        "PATH": self.bin + os.pathsep + os.environ["PATH"],
        "HOME": self.tmp.name,
        "USER": "tester",
        "CALL_LOG": self.calls,
        "GH_BIN": os.path.join(self.bin, "gh"),
        "SPOT_TPU_FLEET_BIN": os.path.join(self.bin, "spot_tpu_fleet.sh"),
        "RUN_PRESUBMIT_V5_RELAY_BIN": os.path.join(
            self.bin, "run_presubmit_v5_relay.sh"
        ),
    }
    env.update(env_overrides)
    return subprocess.run(
        ["bash", RELAY_DRIVER, *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def recorded(self):
    if not os.path.exists(self.calls):
      return []
    with open(self.calls, "r", encoding="utf-8") as f:
      return [line.rstrip("\n") for line in f]

  def gh_calls(self):
    return [call for call in self.recorded() if call.startswith("gh ")]

  def fleet_calls(self):
    return [
        call
        for call in self.recorded()
        if call.startswith("spot_tpu_fleet.sh ")
    ]

  def relay_call(self):
    return next(
        call
        for call in self.recorded()
        if call.startswith("run_presubmit_v5_relay.sh ")
    )

  def posted_statuses(self):
    """Returns (state, context) for every status the driver published."""
    statuses = []
    for call in self.gh_calls():
      if "/statuses/" not in call:
        continue
      state = re.search(r"state=(\S+)", call)
      context = re.search(r"context=(.+?)(?= -f |$)", call)
      statuses.append((
          state.group(1) if state else None,
          context.group(1) if context else None,
      ))
    return statuses

  def default_args(self, mode="shadow"):
    return [
        "--pr",
        "3730",
        "--mode",
        mode,
        "--pool",
        self.pool,
        "--output-dir",
        self.out,
        "--skip-head-check",
    ]

  def stub_everything(self, **relay_kwargs):
    self.fake_gh()
    self.fake_fleet()
    self.fake_relay(**relay_kwargs)


class TestRelayDriverArguments(RelayDriverTestCase):
  """Argument validation, before anything external gets touched."""

  def test_it_requires_a_pull_request(self):
    result = self.run_driver("--mode", "shadow")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("--pr is required", result.stderr)

  def test_it_rejects_a_non_numeric_pull_request(self):
    result = self.run_driver("--pr", "twelve")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("--pr must be a number", result.stderr)

  def test_it_rejects_an_unknown_mode(self):
    result = self.run_driver("--pr", "1", "--mode", "gating")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("--mode must be", result.stderr)

  def test_it_rejects_a_malformed_repo(self):
    result = self.run_driver("--pr", "1", "--repo", "torch_tpu")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("OWNER/NAME", result.stderr)

  def test_it_rejects_a_non_numeric_job_count(self):
    result = self.run_driver("--pr", "1", "--jobs", "lots")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("--jobs must be a number", result.stderr)

  def test_it_rejects_an_unknown_flag(self):
    result = self.run_driver("--pr", "1", "--turbo")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unknown argument", result.stderr)

  def test_a_dry_run_touches_nothing(self):
    self.stub_everything()
    result = self.run_driver(
        *self.default_args(), "--dry-run", "--sha", self.HEAD_SHA
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(self.recorded(), [])

  def test_a_dry_run_names_the_context_it_would_report_under(self):
    result = self.run_driver(
        *self.default_args(mode="replacement"),
        "--dry-run",
        "--sha",
        self.HEAD_SHA,
    )
    self.assertIn("Presubmit on linux-x86-ct5lp-224-8tpu", result.stdout)


class TestRelayDriverPreflight(RelayDriverTestCase):
  """Refusals that have to happen before any hardware is leased."""

  def test_it_stops_when_the_ssh_proxy_is_missing(self):
    """Without corp-ssh-helper every SSH hangs until it times out, which is a

    slow and confusing way to discover the firewall.
    """
    self.stub_everything()
    result = self.run_driver(
        *self.default_args(),
        CORP_SSH_HELPER_BIN=os.path.join(self.bin, "no-such-proxy"),
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("corp-ssh-helper", result.stderr)
    self.assertEqual(self.fleet_calls(), [])

  def test_it_refuses_to_report_on_a_tree_that_is_not_the_pr_head(self):
    self.stub_everything()
    result = self.run_driver(
        "--pr", "3730", "--pool", self.pool, "--output-dir", self.out
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("Reporting a verdict for code you did not run", result.stderr)
    self.assertEqual(self.posted_statuses(), [])

  def test_a_replacement_run_needs_the_replace_label(self):
    """Without it presubmit.yml still schedules the real ct5lp runner, and both

    would report under the same check name.
    """
    self.fake_gh(labels=("ci:relay-tpu-v5",))
    self.fake_fleet()
    self.fake_relay()
    result = self.run_driver(*self.default_args(mode="replacement"))
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("ci:replace-tpu-v5", result.stderr)
    self.assertEqual(self.posted_statuses(), [])

  def test_add_label_supplies_the_missing_label(self):
    self.fake_gh(labels=())
    self.fake_fleet()
    self.fake_relay()
    result = self.run_driver(
        *self.default_args(mode="replacement"), "--add-label"
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertTrue(
        any(
            "issues/3730/labels" in call and "ci:replace-tpu-v5" in call
            for call in self.gh_calls()
        ),
        self.gh_calls(),
    )

  def test_a_shadow_run_does_not_need_any_label(self):
    self.fake_gh(labels=())
    self.fake_fleet()
    self.fake_relay()
    result = self.run_driver(*self.default_args())
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_it_stops_when_no_vms_are_available(self):
    self.fake_gh()
    self.fake_fleet(vms=0)
    self.fake_relay()
    result = self.run_driver(*self.default_args())
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("does not create hardware", result.stderr)


class TestRelayDriverReporting(RelayDriverTestCase):
  """What lands on the pull request."""

  SHADOW = "TPU v5e relay (shadow)"
  GATING = "Presubmit on linux-x86-ct5lp-224-8tpu"

  def test_a_passing_shadow_run_reports_under_the_advisory_context(self):
    self.stub_everything()
    result = self.run_driver(*self.default_args())
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(
        self.posted_statuses(),
        [("pending", self.SHADOW), ("success", self.SHADOW)],
    )

  def test_a_passing_replacement_run_resolves_the_required_check(self):
    self.stub_everything()
    result = self.run_driver(*self.default_args(mode="replacement"))
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(
        self.posted_statuses(),
        [("pending", self.GATING), ("success", self.GATING)],
    )

  def test_a_failing_run_reports_failure_and_exits_non_zero(self):
    self.stub_everything(status="FAILED", failed=3, exit_code=1)
    result = self.run_driver(*self.default_args(mode="replacement"))
    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(self.posted_statuses()[-1], ("failure", self.GATING))

  def test_a_run_that_leaves_no_summary_reports_error_not_success(self):
    """An interrupted run has verified nothing.

    Reporting green would be a lie.
    """
    self.stub_everything(exit_code=2, write_summary=False)
    result = self.run_driver(*self.default_args(mode="replacement"))
    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(self.posted_statuses()[-1][0], "error")

  def test_a_nonzero_exit_with_no_failing_target_is_an_error_not_a_failure(
      self,
  ):
    """A build break or a dead fleet is not the same as a test failing."""
    self.stub_everything(status="PASSED", failed=0, exit_code=1)
    self.run_driver(*self.default_args())
    self.assertEqual(self.posted_statuses()[-1][0], "error")

  def test_the_verdict_carries_the_target_counts(self):
    self.stub_everything()
    self.run_driver(*self.default_args())
    final = [call for call in self.gh_calls() if "/statuses/" in call][-1]
    self.assertIn("57/57 targets passed", final)

  def test_it_comments_the_report_on_the_pull_request(self):
    self.stub_everything()
    self.run_driver(*self.default_args())
    self.assertTrue(
        any("issues/3730/comments" in call for call in self.gh_calls()),
        self.gh_calls(),
    )

  def test_no_report_leaves_the_pull_request_alone(self):
    self.stub_everything()
    result = self.run_driver(*self.default_args(), "--no-report")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(self.posted_statuses(), [])


class TestRelayDriverFleetHandling(RelayDriverTestCase):
  """The driver borrows hardware. It must never own it."""

  def fleet_verbs(self):
    return [call.split()[1] for call in self.fleet_calls()]

  def test_it_attaches_and_detaches(self):
    self.stub_everything()
    self.run_driver(*self.default_args())
    self.assertEqual(self.fleet_verbs(), ["attach", "detach"])

  def test_it_detaches_even_when_the_relay_fails(self):
    """A run that dies holding every session file starves the next one."""
    self.stub_everything(status="FAILED", failed=1, exit_code=1)
    self.run_driver(*self.default_args())
    self.assertIn("detach", self.fleet_verbs())

  def test_it_never_creates_or_deletes_hardware(self):
    self.stub_everything()
    self.run_driver(*self.default_args())
    self.assertNotIn("up", self.fleet_verbs())
    self.assertNotIn("down", self.fleet_verbs())

  def test_it_passes_every_zone_through(self):
    self.stub_everything()
    self.run_driver(
        *self.default_args(), "--zone", "us-west1-c", "--zone", "us-west4-a"
    )
    attach = next(call for call in self.fleet_calls() if " attach " in call)
    self.assertIn("--zone us-west1-c", attach)
    self.assertIn("--zone us-west4-a", attach)

  def test_it_defaults_to_europe_west4_b(self):
    self.stub_everything()
    self.run_driver(*self.default_args())
    attach = next(call for call in self.fleet_calls() if " attach " in call)
    self.assertIn("--zone europe-west4-b", attach)

  def test_it_defaults_the_job_count_to_the_number_of_vms(self):
    """More jobs than VMs leaves tests spinning in the lease retry loop."""
    self.fake_gh()
    self.fake_fleet(vms=4)
    self.fake_relay()
    self.run_driver(*self.default_args())
    self.assertIn("--jobs 4", self.relay_call())

  def test_an_explicit_job_count_wins(self):
    self.fake_gh()
    self.fake_fleet(vms=4)
    self.fake_relay()
    self.run_driver(*self.default_args(), "--jobs", "2")
    self.assertIn("--jobs 2", self.relay_call())

  def test_it_builds_with_the_config_that_keeps_runfiles_on_disk(self):
    """--config=ci alone sets --remote_download_minimal, and the base cache is

    built by reading the runfiles trees off bazel-bin.
    """
    self.stub_everything()
    self.run_driver(*self.default_args())
    self.assertIn("--bazel-config ci_tpu_v5_relay", self.relay_call())


class TestRelayDriverBazelFlags(RelayDriverTestCase):
  """What the driver hands the relay runner on the bazel side.

  The ci configs point the build event upload at an instance only the CI
  service account can write to. When anyone else runs the relay the upload
  fails after the suite has already passed, bazel exits non-zero, and the
  driver reports a green run as an error. So the driver switches the upload
  off unless asked for it.
  """

  def relay_args(self):
    return self.relay_call().split(" ", 1)[1]

  def test_it_turns_the_resultstore_upload_off_by_default(self):
    self.stub_everything()
    result = self.run_driver(*self.default_args())
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--bazel-flag --bes_backend=", self.relay_args())

  def test_upload_results_leaves_the_upload_alone(self):
    self.stub_everything()
    result = self.run_driver(*self.default_args(), "--upload-results")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertNotIn("--bes_backend=", self.relay_args())

  def test_it_forwards_an_extra_flag(self):
    self.stub_everything()
    result = self.run_driver(
        *self.default_args(), "--bazel-flag", "--verbose_failures"
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--bazel-flag --verbose_failures", self.relay_args())

  def test_an_extra_flag_does_not_displace_the_default(self):
    self.stub_everything()
    self.run_driver(*self.default_args(), "--bazel-flag=--verbose_failures")
    args = self.relay_args()
    self.assertIn("--bes_backend=", args)
    self.assertIn("--verbose_failures", args)

  def test_the_config_still_goes_through(self):
    self.stub_everything()
    self.run_driver(*self.default_args())
    self.assertIn("--bazel-config ci_tpu_v5_relay", self.relay_args())

  def test_a_missing_flag_value_is_an_error(self):
    result = self.run_driver("--pr", "1", "--bazel-flag")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("--bazel-flag requires an argument", result.stderr)


class TestStageRelayBaseRelaxesOvercommit(StageRelayBaseSshTestCase):
  """The OOM error test needs the host allocator to get out of the way.

  tests/tpu_errors_test.py copies a 4 TB result back and asserts the TPU is
  what reports the OOM. A 47 GB v5litepod-1 refuses that malloc under the
  kernel's default heuristic, so the test sees the host allocator's error
  instead and fails. A ct5lp CI runner has the RAM and never hits it.
  """

  def setUp(self):
    super().setUp()
    self.add_runfiles_tree("a", deps=["rules_python++pip+x"], solibs=["_U_a"])
    proc = self.run_staged()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.commands = self.remote_commands()

  def test_it_allows_the_overcommit(self):
    self.assertIn("sudo sysctl -w vm.overcommit_memory=1", self.commands)

  def test_it_happens_before_anything_is_pushed(self):
    tweak = self.commands.index("sudo sysctl -w vm.overcommit_memory=1")
    pushes = [
        i for i, cmd in enumerate(self.commands) if cmd.startswith("tar -xzf")
    ]
    self.assertTrue(pushes, self.commands)
    self.assertLess(tweak, pushes[0])


class TestStageRelayBaseSurvivesAReadOnlyHost(StageRelayBaseSshTestCase):
  """A VM that refuses the sysctl still gets staged.

  Only one test out of 45,000 cares about the overcommit setting. Failing the
  whole run over it would trade one bad result for fifty-seven.
  """

  def setUp(self):
    super().setUp()
    script = os.path.join(self.bin_dir, "ssh")
    with open(script, "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
printf '%s\\n' "${{@: -1}}" >> "{self.ssh_log}"
cmd="${{@: -1}}"
[[ "$cmd" == sudo* ]] && exit 1
[[ "$cmd" == *".complete' ]"* ]] && exit 1
[[ "$cmd" == tar* ]] && cat > /dev/null
exit 0
""")
    os.chmod(script, 0o755)
    self.add_runfiles_tree("a", deps=["rules_python++pip+x"], solibs=["_U_a"])

  def test_staging_still_succeeds(self):
    proc = self.run_staged()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertIn("ready", proc.stdout)

  def test_it_says_which_test_will_suffer(self):
    proc = self.run_staged()
    self.assertIn("tpu_errors_test", proc.stderr)


class FleetClaimTestCase(FleetAttachTestCase):
  """Runs the real claim script against a fake per-VM filesystem.

  The fake gcloud executes whatever `--command=` it is handed, so these
  exercise the script attach actually sends rather than a paraphrase of it.
  TORCH_TPU_REMOTE_CLAIM_PATH aims the claim at scratch space, and the stub
  suffixes it with the VM name so two VMs do not share one claim the way they
  would if they really shared a disk.
  """

  VM = "spot-tpu-v5e-111-1"

  def setUp(self):
    super().setUp()
    self.claim_root = os.path.join(self.root, "vms", "claim")
    self.user = "tester"

  def claim_path(self, name=None):
    return f"{self.claim_root}-{name or self.VM}"

  def fake_gcloud_running_commands(self, listed_names, ip="10.0.0.5"):
    listing_file = os.path.join(self.root, "listing.txt")
    with open(listing_file, "w", encoding="utf-8") as fh:
      fh.write("".join(f"{name}\n" for name in listed_names))

    path = os.path.join(self.bin_dir, "gcloud")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "{self.gcloud_log}"
case "$4" in
  list) cat "{listing_file}" ;;
  describe)
    [[ "$*" == *"externalIp"* ]] && printf '%s\\n' "{ip}"
    ;;
  ssh)
    vm="$5"
    for arg in "$@"; do
      if [[ "$arg" == --command=* ]]; then
        cmd="${{arg#--command=}}"
        printf '%s' "$cmd" | sed "s|{self.claim_root}|{self.claim_root}-$vm|g" | bash
      fi
    done
    ;;
esac
exit 0
""")
    os.chmod(path, 0o755)

  def run_fleet(self, *args, user=None):
    return subprocess.run(
        ["bash", FLEET_SCRIPT, *args],
        env={
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "HOME": self.root,
            "USER": user or self.user,
            "TORCH_TPU_REMOTE_CLAIM_PATH": self.claim_root,
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

  def attach(self, *extra, pool=None, user=None):
    return self.run_fleet(
        "attach",
        "--pool",
        pool or self.pool,
        "--zone",
        self.ZONE,
        *extra,
        user=user,
    )

  def owner_of(self, name=None):
    with open(
        os.path.join(self.claim_path(name), "owner"), encoding="utf-8"
    ) as fh:
      return fh.read().strip()

  def mine(self):
    return f"{self.user}:{os.path.basename(self.pool)}"

  def write_claim(self, owner, name=None, age_seconds=0):
    claim = self.claim_path(name)
    os.makedirs(claim, exist_ok=True)
    with open(os.path.join(claim, "owner"), "w") as fh:
      fh.write(owner + "\n")
    with open(os.path.join(claim, "at"), "w") as fh:
      fh.write(str(int(time.time()) - age_seconds) + "\n")


class TestFleetClaim(FleetClaimTestCase):
  """Two people attaching at once must not land on the same chips.

  `attach` takes every READY VM it can see. Without a claim, a second engineer
  attaching while the first is mid-run gets the same 28 VMs, and every chip
  then runs two tests that fight over it.
  """

  def test_attaching_leaves_a_claim_naming_the_pool(self):
    self.fake_gcloud_running_commands([self.VM])
    proc = self.attach()
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertEqual(self.owner_of(), self.mine())

  def test_a_vm_somebody_else_holds_is_skipped(self):
    self.fake_gcloud_running_commands([self.VM])
    self.write_claim("someone-else:their-pool")
    proc = self.attach()
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("held by someone-else:their-pool", proc.stdout)
    self.assertEqual(self.sessions(), [])

  def test_the_holder_keeps_the_claim_it_had(self):
    self.fake_gcloud_running_commands([self.VM])
    self.write_claim("someone-else:their-pool")
    self.attach()
    self.assertEqual(self.owner_of(), "someone-else:their-pool")

  def test_a_held_vm_does_not_stop_the_rest_of_the_fleet(self):
    """One busy VM is a smaller fleet, not a failed attach."""
    second = "spot-tpu-v5e-111-2"
    self.fake_gcloud_running_commands([self.VM, second])
    self.write_claim("someone-else:their-pool")
    proc = self.attach()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertEqual(len(self.sessions()), 1)
    self.assertEqual(self.owner_of(second), self.mine())

  def test_re_attaching_the_same_pool_is_allowed(self):
    """Attach is meant to be safe to repeat; its own claim is not a conflict."""
    self.fake_gcloud_running_commands([self.VM])
    self.assertEqual(self.attach().returncode, 0)
    os.remove(os.path.join(self.pool, "vm_0.env"))
    proc = self.attach()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertEqual(self.sessions(), ["vm_0.env"])

  def test_another_pool_of_the_same_user_is_still_a_conflict(self):
    """Two runs by one person are two runs. The chips cannot be shared."""
    self.fake_gcloud_running_commands([self.VM])
    self.write_claim(f"{self.user}:some-other-pool")
    self.assertNotEqual(self.attach().returncode, 0)

  def test_a_claim_older_than_the_ttl_is_taken_over(self):
    """A run that died still holds its claim. Nothing else will ever free it."""
    self.fake_gcloud_running_commands([self.VM])
    self.write_claim("ghost:dead-pool", age_seconds=14401)
    proc = self.attach()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertEqual(self.owner_of(), self.mine())

  def test_a_claim_inside_the_ttl_is_left_alone(self):
    self.fake_gcloud_running_commands([self.VM])
    self.write_claim("busy:live-pool", age_seconds=14399)
    self.assertNotEqual(self.attach().returncode, 0)

  def test_the_ttl_is_configurable(self):
    self.fake_gcloud_running_commands([self.VM])
    self.write_claim("ghost:dead-pool", age_seconds=120)
    proc = self.attach("--claim-ttl", "60")
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

  def test_force_claim_takes_a_live_claim(self):
    self.fake_gcloud_running_commands([self.VM])
    self.write_claim("busy:live-pool")
    proc = self.attach("--force-claim")
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertEqual(self.owner_of(), self.mine())

  def test_no_claim_leaves_no_marker(self):
    self.fake_gcloud_running_commands([self.VM])
    proc = self.attach("--no-claim")
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertFalse(os.path.exists(self.claim_path()))


class TestFleetDetachReleasesTheClaim(FleetClaimTestCase):
  """A claim nobody releases holds the VM until it ages out four hours later.

  `detach` is the hand-back, so it is where the marker has to come off. It
  only removes claims this pool owns: a crashed run's leftovers are somebody
  else's business, and stealing them here would defeat the whole point.
  """

  def fake_ssh(self, vm=None):
    """An ssh that runs the command against the fake VM's claim directory."""
    path = os.path.join(self.bin_dir, "ssh")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
cmd="${{@: -1}}"
printf '%s' "$cmd" \\
  | sed "s|{self.claim_root}|{self.claim_path(vm)}|g" \\
  | bash
""")
    os.chmod(path, 0o755)

  def test_detach_removes_a_claim_this_pool_owns(self):
    self.fake_gcloud_running_commands([self.VM])
    self.assertEqual(self.attach().returncode, 0)
    self.assertTrue(os.path.exists(self.claim_path()))

    self.fake_ssh()
    proc = self.run_fleet("detach", "--pool", self.pool)
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertFalse(os.path.exists(self.claim_path()))

  def test_detach_leaves_somebody_elses_claim_alone(self):
    self.fake_gcloud_running_commands([self.VM])
    self.attach()
    self.write_claim("someone-else:their-pool")

    self.fake_ssh()
    self.run_fleet("detach", "--pool", self.pool)
    self.assertEqual(self.owner_of(), "someone-else:their-pool")

  def test_detach_still_clears_the_pool_when_the_vm_is_unreachable(self):
    """The VM may be gone. The session files must go regardless."""
    self.fake_gcloud_running_commands([self.VM])
    self.attach()

    path = os.path.join(self.bin_dir, "ssh")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write("#!/usr/bin/env bash\nexit 255\n")
    os.chmod(path, 0o755)

    proc = self.run_fleet("detach", "--pool", self.pool)
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertEqual(self.sessions(), [])


class FleetQueueTestCase(FleetClaimTestCase):
  """Runs the real waiting line against a fake fleet.

  The claim tests only need one VM. These need two, because the point of the
  queue is what happens when a run can have some of the fleet but not all of
  it, so the stubs here give each VM its own address and its own claim.
  """

  VMS = ("spot-tpu-v5e-111-1", "spot-tpu-v5e-111-2")

  def setUp(self):
    super().setUp()
    self.queue_root = os.path.join(self.root, "vms", "queue")
    self.hosts_file = os.path.join(self.root, "hosts.txt")

  def ip_of(self, name):
    return f"10.0.0.{name.rsplit('-', 1)[1]}"

  def fake_fleet(self, names=None):
    """A gcloud whose VMs have separate addresses, claims and one shared queue."""
    names = names or list(self.VMS)
    listing_file = os.path.join(self.root, "listing.txt")
    with open(listing_file, "w", encoding="utf-8") as fh:
      fh.write("".join(f"{name}\n" for name in names))
    with open(self.hosts_file, "w", encoding="utf-8") as fh:
      fh.write("".join(f"{self.ip_of(name)} {name}\n" for name in names))

    path = os.path.join(self.bin_dir, "gcloud")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "{self.gcloud_log}"
case "$4" in
  list) cat "{listing_file}" ;;
  describe)
    [[ "$*" == *"externalIp"* ]] && printf '10.0.0.%s\\n' "${{5##*-}}"
    ;;
  ssh)
    vm="$5"
    for arg in "$@"; do
      if [[ "$arg" == --command=* ]]; then
        cmd="${{arg#--command=}}"
        printf '%s' "$cmd" | sed "s|{self.claim_root}|{self.claim_root}-$vm|g" | bash
      fi
    done
    ;;
esac
exit 0
""")
    os.chmod(path, 0o755)

    # release_claims goes over plain ssh, so the rollback path needs a stub
    # that can tell the VMs apart by address.
    ssh_path = os.path.join(self.bin_dir, "ssh")
    with open(ssh_path, "w", encoding="utf-8") as fh:
      fh.write(f"""#!/usr/bin/env bash
cmd="${{@: -1}}"
dest=""
for arg in "$@"; do
  [[ "$arg" == *@* ]] && dest="${{arg#*@}}"
done
vm=$(awk -v ip="$dest" '$1==ip {{print $2}}' "{self.hosts_file}")
printf '%s' "$cmd" | sed "s|{self.claim_root}|{self.claim_root}-$vm|g" | bash
""")
    os.chmod(ssh_path, 0o755)

  def wait_attach(self, *extra, timeout="2", poll="1", **kwargs):
    return self.attach(
        "--wait",
        "--wait-timeout",
        timeout,
        "--wait-poll",
        poll,
        *extra,
        **kwargs,
    )

  def run_fleet(self, *args, user=None):
    proc = subprocess.run(
        ["bash", FLEET_SCRIPT, *args],
        env={
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "HOME": self.root,
            "USER": user or self.user,
            "TORCH_TPU_REMOTE_CLAIM_PATH": self.claim_root,
            "TORCH_TPU_REMOTE_QUEUE_PATH": self.queue_root,
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return proc

  def tickets(self):
    if not os.path.isdir(self.queue_root):
      return []
    return sorted(os.listdir(self.queue_root))

  def plant_ticket(self, name="000000000001.99.1", owner="other:pool", age=0):
    """Puts somebody else in the line ahead of us."""
    ticket = os.path.join(self.queue_root, name)
    os.makedirs(ticket, exist_ok=True)
    with open(os.path.join(ticket, "owner"), "w", encoding="utf-8") as fh:
      fh.write(owner + "\n")
    stamp = time.time() - age
    os.utime(ticket, (stamp, stamp))
    return ticket

  def queue_hosts(self):
    """The VMs that were sent queue commands.

    A record in the log is one gcloud invocation, and the queue script is
    several lines long, so records have to be split on their own first token
    rather than on newlines.
    """
    with open(self.gcloud_log, encoding="utf-8") as fh:
      text = fh.read()
    records = re.split(r"^(?=compute tpus tpu-vm )", text, flags=re.MULTILINE)
    return {
        record.split()[4]
        for record in records
        if self.queue_root in record
        and record.startswith("compute tpus tpu-vm ssh ")
    }


class TestFleetAttachWaitsItsTurn(FleetQueueTestCase):
  """One run gets the whole fleet; the next one queues instead of splitting it.

  Twenty-eight chips and a seven-minute suite mean the fastest way through two
  runs is back to back on everything, not side by side on half each. Without a
  line, two waiters also deadlock: each takes the VMs the other needs and
  neither can start.
  """

  def test_taking_the_whole_fleet_leaves_nobody_in_the_line(self):
    self.fake_fleet()
    proc = self.wait_attach()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertEqual(len(self.sessions()), 2)
    self.assertEqual(self.tickets(), [])

  def test_the_line_lives_on_the_first_vm_by_name(self):
    """Every caller has to pick the same host or there are two lines."""
    self.fake_fleet(list(reversed(self.VMS)))
    self.wait_attach()
    self.assertEqual(self.queue_hosts(), {self.VMS[0]})

  def test_the_queue_host_can_be_named(self):
    self.fake_fleet()
    self.wait_attach("--queue-host", self.VMS[1])
    self.assertEqual(self.queue_hosts(), {self.VMS[1]})

  def test_a_half_free_fleet_is_handed_back_rather_than_held(self):
    """Keeping the free half is what turns two waiters into a deadlock."""
    self.fake_fleet()
    self.write_claim("someone-else:their-pool", name=self.VMS[0])

    proc = self.wait_attach()
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("The fleet is busy", proc.stdout)
    self.assertEqual(self.sessions(), [])
    self.assertFalse(os.path.exists(self.claim_path(self.VMS[1])))

  def test_somebody_ahead_of_us_keeps_us_off_the_hardware(self):
    self.fake_fleet()
    self.plant_ticket()

    proc = self.wait_attach()
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("1 run(s) ahead of you", proc.stdout)
    for vm in self.VMS:
      self.assertFalse(os.path.exists(self.claim_path(vm)))

  def test_a_ticket_nobody_touches_loses_its_place(self):
    """A killed run must not hold the line until somebody notices."""
    self.fake_fleet()
    stale = self.plant_ticket(age=600)

    proc = self.wait_attach()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertFalse(os.path.exists(stale))
    self.assertEqual(len(self.sessions()), 2)

  def test_a_live_ticket_ahead_of_us_is_left_alone(self):
    self.fake_fleet()
    ticket = self.plant_ticket(age=0)
    self.wait_attach()
    self.assertTrue(os.path.exists(ticket))

  def test_waiting_ends_at_the_timeout(self):
    self.fake_fleet()
    self.write_claim("someone-else:their-pool", name=self.VMS[0])
    proc = self.wait_attach(timeout="1")
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("Gave up waiting", proc.stderr)

  def test_giving_up_takes_our_ticket_out_of_the_line(self):
    self.fake_fleet()
    self.write_claim("someone-else:their-pool", name=self.VMS[0])
    self.wait_attach(timeout="1")
    self.assertEqual(self.tickets(), [])

  def test_the_fleet_is_taken_once_the_holder_lets_go(self):
    """The wait has to end on its own, not just report the queue correctly."""
    self.fake_fleet()
    self.write_claim("someone-else:their-pool", name=self.VMS[0])

    proc = self.wait_attach("--claim-ttl", "3", timeout="20", poll="2")
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertIn("The fleet is busy", proc.stdout)
    self.assertEqual(len(self.sessions()), 2)
    self.assertEqual(self.owner_of(self.VMS[0]), self.mine())

  def test_without_wait_a_busy_fleet_still_gives_you_what_is_free(self):
    """--wait changes the deal; plain attach keeps working as it always did."""
    self.fake_fleet()
    self.write_claim("someone-else:their-pool", name=self.VMS[0])

    proc = self.attach()
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
    self.assertEqual(len(self.sessions()), 1)
    self.assertEqual(self.queue_hosts(), set())


class TestRelayDriverQueuesForTheFleet(RelayDriverTestCase):
  """A PR run should stand in line rather than fight for chips.

  Two runs sharing 28 VMs each get half the parallelism and twice the wall
  clock, and with claims in place the second one simply fails. Queueing turns
  that into "wait seven minutes, then get everything".
  """

  def attach_call(self):
    return next(
        call
        for call in self.fleet_calls()
        if call.startswith("spot_tpu_fleet.sh attach")
    )

  def test_a_run_waits_its_turn_by_default(self):
    self.stub_everything()
    result = self.run_driver(*self.default_args(), "--sha", self.HEAD_SHA)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--wait", self.attach_call())
    self.assertIn("--wait-timeout 3600", self.attach_call())

  def test_no_wait_asks_for_the_old_fail_fast_behaviour(self):
    self.stub_everything()
    result = self.run_driver(
        *self.default_args(), "--sha", self.HEAD_SHA, "--no-wait"
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertNotIn("--wait", self.attach_call())

  def test_the_wait_can_be_shortened(self):
    self.stub_everything()
    self.run_driver(
        *self.default_args(), "--sha", self.HEAD_SHA, "--wait-timeout", "120"
    )
    self.assertIn("--wait-timeout 120", self.attach_call())

  def test_the_plan_says_whether_it_will_queue(self):
    self.stub_everything()
    waiting = self.run_driver(
        *self.default_args(), "--sha", self.HEAD_SHA, "--dry-run"
    )
    self.assertIn("wait for all of it", waiting.stdout)
    impatient = self.run_driver(
        *self.default_args(), "--sha", self.HEAD_SHA, "--dry-run", "--no-wait"
    )
    self.assertIn("take what is free", impatient.stdout)
