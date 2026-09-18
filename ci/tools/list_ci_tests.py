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

"""Tool to inspect and list test targets executed by OSS TorchTPU CI jobs.

Parses .bazelrc test tag filters and invokes Bazel query to inspect which test
targets run in each CI job configuration.

Example usage:
    # List test targets grouped by CI job:
    $ ci/tools/list_ci_tests.py

    # List test targets in count format:
    $ ci/tools/list_ci_tests.py --format=count

    # Check which CI jobs run a specific test target:
    $ ci/tools/list_ci_tests.py --test=ops_test_tpu-v5lite

    # List only the test targets matching one or more globs:
    $ ci/tools/list_ci_tests.py --tests=ops_test*,//tests:*_tpu-v5lite
"""

import argparse
from collections.abc import Mapping, Sequence
import fnmatch
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.request
import xml.etree.ElementTree

import yaml

# Bazelisk release used when bootstrapping. Kept in sync with the version the
# README tells contributors to install. Unlike a pinned Bazel version this is
# not a correctness constraint: whichever Bazelisk we end up with reads the
# repository's `.bazelversion`, so this only affects the launcher itself.
_BAZELISK_VERSION = "1.27.0"


# Matches a Bazel config line that sets --test_tag_filters, e.g.
#   test:ci_cpu --test_tag_filters=-nobuild,-notest
# The filter list may be bare, single-quoted, or double-quoted.
_TEST_TAG_FILTERS_RE = re.compile(
    r"""
    ^(?:common|test|build):        # Bazel command the config applies to.
    (?P<job>[a-zA-Z0-9_-]+)\s+     # CI job (Bazel config) name.
    .*?                            # Any flags preceding --test_tag_filters.
    --test_tag_filters=
    (?:
        "(?P<qfilters>[^"]*)"      # Double-quoted filter list.
      | '(?P<sqfilters>[^']*)'     # Single-quoted filter list.
      | (?P<filters>[^\s#]+)       # Bare filter list, up to space or comment.
    )
    """,
    re.VERBOSE,
)


def find_repo_root(
    start_path: pathlib.Path | None = None,
) -> pathlib.Path:
  """Locates the OSS repository root containing .bazelrc.

  Resolution strategy:
    1. Checks the TORCH_TPU_REPO_DIR environment variable for an explicit path.
    2. Traverses upward from `start_path` (or this script's directory) towards
       the filesystem root until a directory containing `.bazelrc` is found.
    3. Falls back to the current working directory if no parent contains
       `.bazelrc`.

  Args:
    start_path: Optional path to start searching upwards from. Defaults to the
      location of this script or the current working directory.

  Returns:
    Path to the detected repository root directory.
  """
  env_override = os.environ.get("TORCH_TPU_REPO_DIR")
  if env_override:
    p = pathlib.Path(env_override).resolve()
    if p.is_dir():
      return p

  curr = (start_path or pathlib.Path(__file__)).resolve()
  if curr.is_file():
    curr = curr.parent

  while curr != curr.parent:
    if (curr / ".bazelrc").is_file():
      return curr
    curr = curr.parent

  return pathlib.Path.cwd().resolve()


def get_bazelrc_path(repo_root: pathlib.Path) -> pathlib.Path:
  """Returns the path to the .bazelrc file defining CI test filters.

  Args:
    repo_root: Repository root expected to contain .bazelrc.

  Returns:
    Path to the .bazelrc file.

  Raises:
    FileNotFoundError: If repo_root does not contain a .bazelrc file.
  """
  bazelrc_path = repo_root / ".bazelrc"
  if bazelrc_path.is_file():
    return bazelrc_path
  raise FileNotFoundError(
      f"Could not find .bazelrc defining CI test filters under {repo_root}."
  )


def parse_bazelrc_tag_filters(
    bazelrc_path: pathlib.Path,
) -> Mapping[str, Sequence[str]]:
  """Parses --test_tag_filters configurations for CI jobs from .bazelrc.

  Looks for lines of the form:
    test:<job_name> --test_tag_filters=<filters>

  Args:
    bazelrc_path: Path to the .bazelrc file.

  Returns:
    A mapping from CI job name (e.g. 'ci_cpu_presubmit') to its filter list
    (e.g. ['-nobuild', '-notest', '-requires-gpu', '-requires-tpu']).
  """
  ci_configs: dict[str, list[str]] = {}

  with open(bazelrc_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      m = _TEST_TAG_FILTERS_RE.match(line)
      if m:
        job = m.group("job")
        filters_str = (
            m.group("qfilters")
            or m.group("sqfilters")
            or m.group("filters")
            or ""
        )
        filters = [
            f.strip("\t '\"")
            for f in filters_str.split(",")
            if f.strip("\t '\"")
        ]
        existing = ci_configs.setdefault(job, [])
        for f in filters:
          if f not in existing:
            existing.append(f)

  return ci_configs


def matches_tag_filters(
    target_tags: set[str],
    filter_list: Sequence[str],
) -> bool:
  """Evaluates whether a target's tags satisfy Bazel --test_tag_filters rules.

  Bazel filter semantics reference:
    https://bazel.build/reference/command-line-reference#flag--test_tag_filters

  Semantics summary:
    - Filters can be negative (prefixed with '-', e.g. '-requires-gpu') or
      positive (e.g. 'presubmit-v5').
    - Negative filters: If any excluded tag in `filter_list` is present in
      `target_tags`, the target is excluded immediately (evaluates to False).
    - Positive filters: If any positive tags are specified in `filter_list`,
      the target must possess at least one of them to be included. If no
      positive tags match, the target is excluded (evaluates to False).
    - If no positive filters are specified, the target is included as long as
      it does not match any negative filters.
    - Order does not matter: negative exclusions take precedence regardless of
      position in the filter list.

  Args:
    target_tags: A set of string tag identifiers defined on the test target
      (e.g., {'requires-tpu', 'presubmit-v5'}).
    filter_list: A sequence of tag filter expressions, positive or negative
      (e.g., ['requires-tpu', '-flaky', '-nopresubmit']).

  Returns:
    True if the target satisfies all negative and positive filters; False
    otherwise.
  """
  positive_filters = []
  for f in filter_list:
    if f.startswith("-"):
      excluded_tag = f[1:]
      if excluded_tag in target_tags:
        return False
    else:
      positive_filters.append(f)

  if positive_filters:
    if not any(pos_tag in target_tags for pos_tag in positive_filters):
      return False

  return True


class TestTarget:
  """Represents a discovered test target with its metadata and tags."""

  def __init__(
      self,
      name: str,
      package: str,
      tags: set[str],
      source_file: str,
  ):
    """Initializes a TestTarget instance.

    Args:
      name: The short rule name of the test target (e.g.,
        'ops_test_tpu-v5lite').
      package: The package path containing the rule (e.g., 'tests' or
        'csrc/ops'), without leading or trailing slashes.
      tags: A set of string tags declared on the test rule (e.g.,
        {'requires-tpu', 'presubmit-v5'}).
      source_file: The path or location of the BUILD file where the target was
        declared (e.g., '/workspace/tests/BUILD').
    """
    self.name = name
    self.package = package
    self.tags = tags
    self.source_file = source_file

  @property
  def label(self) -> str:
    pkg = self.package.strip("/")
    if pkg:
      return f"//{pkg}:{self.name}"
    return f"//:{self.name}"

  def __repr__(self) -> str:
    return f"<TestTarget {self.label} tags={sorted(self.tags)}>"


class JobMachineInfo:
  """Information about the machine type used to execute a CI job."""

  def __init__(self, machine_type: str, runner: str = ""):
    self.machine_type = machine_type or "Unknown"
    self.runner = runner or ""

  @property
  def description(self) -> str:
    """Returns formatted description, e.g. 'TPU v5e (linux-x86-ct5lp)'."""
    if self.machine_type and self.runner and self.runner != "Unknown":
      return f"{self.machine_type} ({self.runner})"
    return self.machine_type or self.runner or "Unknown"

  def __repr__(self) -> str:
    return (
        f"JobMachineInfo(machine_type={self.machine_type!r},"
        f" runner={self.runner!r})"
    )


def decode_runner_machine_type(runner: str) -> str:
  """Infers human-readable hardware machine type from runner label."""
  r = runner.strip().lower()
  if "gpu" in r or "h100" in r or "a100" in r:
    return "GPU"
  if "ct5lp" in r or "v5" in r:
    return "TPU v5e"
  if "ct6e" in r or "v6" in r:
    return "TPU v6e"
  if "tpu7x-56-1tpu" in r:
    return "TPU v7x 1-Chip"
  if "tpu7x" in r or "v7" in r:
    return "TPU v7x"
  if "n4" in r or "n2" in r or "cpu" in r:
    return "CPU"
  if "ubuntu" in r:
    return "Ubuntu (GitHub-hosted)"
  return "Unknown"


def _resolve_machine_type(machine_type: str, name: str, runner: str) -> str:
  """Resolves the machine type from metadata or infers from runner label.

  Args:
    machine_type: Machine type explicitly declared by the workflow definition
      (e.g. 'TPU v6e'), or the empty string if it declares none.
    name: Display name of the job or matrix entry (e.g. 'CPU'), or the empty
      string if it has none.
    runner: Runner label string (e.g. 'linux-x86-ct5lp-224-8tpu').

  Returns:
    The canonical human-readable machine type string (e.g. 'TPU v5e', 'CPU').
  """
  if machine_type:
    return machine_type
  if name.upper() == "CPU":
    return name
  return decode_runner_machine_type(runner)


def get_workflows_dir(repo_root: pathlib.Path) -> pathlib.Path | None:
  """Returns the .github/workflows directory, or None if it does not exist."""
  workflows_dir = repo_root / ".github" / "workflows"
  if workflows_dir.is_dir():
    return workflows_dir
  return None


# Captures the Bazel config a `bazel test` step command selects, e.g.
#   run: bazel test --config=ci_cpu //tests:ops_test
# The workflows themselves are parsed as YAML, but a step's `run` value is an
# opaque shell command, so pulling the flag back out of it still needs a regex.
_BAZEL_TEST_CONFIG_RE = re.compile(
    r"""
    bazel\s+test\s+
    [^\n]*?                    # Flags preceding --config on the command line.
    --config=(?P<config>[a-zA-Z0-9_-]+)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Matrix entry keys naming the Bazel config (== CI job) an entry selects. The
# key varies across our workflows, and a single entry may set several of them,
# e.g. a job that tests both the sources and the wheel built from them.
_CONFIG_KEYS = (
    "bazel_config",
    "source_test_config",
    "wheel_test_config",
    "config",
)

# GitHub Actions only expands expressions at workflow run time, so a value
# containing one is a template rather than a label we can report.
_EXPRESSION_MARKER = "${{"


def _load_workflow(path: pathlib.Path) -> dict[str, Any]:
  """Loads a workflow file, returning an empty mapping if it is unusable."""
  try:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
  except (OSError, yaml.YAMLError):
    return {}
  return document if isinstance(document, dict) else {}


def _get_jobs(workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
  """Returns the job definitions declared by a parsed workflow."""
  jobs = workflow.get("jobs")
  if not isinstance(jobs, dict):
    return []
  return [job for job in jobs.values() if isinstance(job, dict)]


def _get_matrix_entries(job: Mapping[str, Any]) -> list[dict[str, Any]]:
  """Returns every mapping-valued matrix entry declared by a job.

  Our workflows spell the matrix either as a custom list of mappings (usually
  `job_info`) or as the built-in `include` list, so every list-valued matrix
  field is inspected rather than a fixed set of field names.

  Args:
    job: A single job definition taken from a workflow's `jobs` mapping.

  Returns:
    The matrix entries, in declaration order.
  """
  strategy = job.get("strategy")
  if not isinstance(strategy, dict):
    return []
  matrix = strategy.get("matrix")
  if not isinstance(matrix, dict):
    return []
  entries = []
  for value in matrix.values():
    if not isinstance(value, list):
      continue
    entries.extend(entry for entry in value if isinstance(entry, dict))
  return entries


def _get_literal(mapping: Mapping[str, Any], key: str) -> str:
  """Returns mapping[key] if it is a literal string, or '' if it is not."""
  value = mapping.get(key)
  if not isinstance(value, str) or _EXPRESSION_MARKER in value:
    return ""
  return value.strip()


def _get_job_runner(job: Mapping[str, Any]) -> str:
  """Returns a concrete runner label for a job, or '' if it has none.

  `runs-on` is usually a template referencing the matrix (e.g.
  `${{ matrix.job_info.runner }}`), in which case the first runner pinned by a
  matrix entry stands in for the job as a whole.

  Args:
    job: A single job definition taken from a workflow's `jobs` mapping.

  Returns:
    The runner label, or the empty string.
  """
  runner = _get_literal(job, "runs-on")
  if runner:
    return runner
  for entry in _get_matrix_entries(job):
    runner = _get_literal(entry, "runner")
    if runner:
      return runner
  return ""


def _get_step_configs(job: Mapping[str, Any]) -> list[str]:
  """Returns the Bazel configs selected by the job's `run` step commands."""
  steps = job.get("steps")
  if not isinstance(steps, list):
    return []
  configs = []
  for step in steps:
    if not isinstance(step, dict):
      continue
    command = step.get("run")
    if isinstance(command, str):
      configs.extend(_BAZEL_TEST_CONFIG_RE.findall(command))
  return configs


def parse_workflow_machine_types(
    workflows_dir: pathlib.Path | None,
) -> dict[str, JobMachineInfo]:
  """Parses CI job runner and machine type definitions from workflows.

  Extracts the runner and machine type mapping directly from the actual CI
  workflow definitions to guarantee the tool remains synchronized with CI.

  Args:
    workflows_dir: Path to the .github/workflows directory, if available.

  Returns:
    Dictionary mapping CI config name -> JobMachineInfo.
  """
  job_machines: dict[str, JobMachineInfo] = {}
  if not workflows_dir or not workflows_dir.is_dir():
    return job_machines

  for wf_path in sorted(workflows_dir.glob("*.y*ml")):
    jobs = _get_jobs(_load_workflow(wf_path))

    # Matrix entries are authoritative: each one pairs a config with the exact
    # runner that config is dispatched to.
    for job in jobs:
      for entry in _get_matrix_entries(job):
        runner = _get_literal(entry, "runner")
        machine_type = _resolve_machine_type(
            _get_literal(entry, "machine_type"),
            _get_literal(entry, "name"),
            runner,
        )
        for key in _CONFIG_KEYS:
          config = _get_literal(entry, key)
          if config:
            job_machines[config] = JobMachineInfo(machine_type, runner)

    # A config hardcoded in a step command only tells us which job runs it, so
    # it falls back to the job's runner and never overrides a matrix entry.
    for job in jobs:
      runner = _get_job_runner(job)
      if not runner:
        continue
      machine_type = decode_runner_machine_type(runner)
      for config in _get_step_configs(job):
        job_machines.setdefault(config, JobMachineInfo(machine_type, runner))

  return job_machines


def _get_bazelisk_download_url(version: str = _BAZELISK_VERSION) -> str:
  """Returns the GitHub release download URL for Bazelisk on this machine.

  Args:
    version: Bazelisk release version, without the leading 'v'.

  Returns:
    The download URL for the current OS and CPU architecture.

  Raises:
    RuntimeError: If the OS or CPU architecture is not published by Bazelisk.
  """
  os_names = {"linux": "linux", "darwin": "darwin", "win32": "windows"}
  arch_names = {
      "x86_64": "amd64",
      "amd64": "amd64",
      "aarch64": "arm64",
      "arm64": "arm64",
  }

  os_name = os_names.get(sys.platform)
  if os_name is None:
    raise RuntimeError(f"Unsupported operating system: {sys.platform}.")
  arch_name = arch_names.get(platform.machine().lower())
  if arch_name is None:
    raise RuntimeError(f"Unsupported CPU architecture: {platform.machine()}.")

  suffix = ".exe" if os_name == "windows" else ""
  return (
      "https://github.com/bazelbuild/bazelisk/releases/download/"
      f"v{version}/bazelisk-{os_name}-{arch_name}{suffix}"
  )


def _get_bazelisk_cache_path(
    version: str = _BAZELISK_VERSION,
) -> pathlib.Path:
  """Returns the path of the user-space cached Bazelisk binary."""
  cache_dir = pathlib.Path.home() / ".cache" / "torch_tpu" / "bin"
  return cache_dir / f"bazelisk-{version}"


def _verify_bazel_binary(
    bazel_bin: pathlib.Path,
    repo_root: pathlib.Path | None = None,
) -> bool:
  """Verifies that a Bazel binary is executable and runs successfully.

  Args:
    bazel_bin: Path to the candidate Bazel binary executable.
    repo_root: Optional repository root directory. If provided, `bazel
      --version` is executed with `cwd=repo_root`, ensuring any wrapper tools
      (e.g., Bazelisk) respect the repository's `.bazelversion`. If None, `cwd`
      is not set (the current working directory is used).

  Returns:
    True if the binary exists, is executable, and returns exit code 0 when
    invoked with `--version`; False otherwise.
  """
  if not bazel_bin.is_file() or not os.access(bazel_bin, os.X_OK):
    return False
  try:
    proc = subprocess.run(
        [str(bazel_bin), "--version"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode == 0
  except (OSError, subprocess.SubprocessError):
    return False


def _download_bazelisk(
    target_path: pathlib.Path,
    version: str = _BAZELISK_VERSION,
) -> pathlib.Path:
  """Downloads Bazelisk from GitHub Releases into target_path.

  Args:
    target_path: Destination path for the downloaded executable.
    version: Bazelisk release version, without the leading 'v'.

  Returns:
    target_path.

  Raises:
    RuntimeError: If the platform is unsupported or the download fails.
  """
  download_url = _get_bazelisk_download_url(version)
  target_path.parent.mkdir(parents=True, exist_ok=True)
  sys.stderr.write(f"Downloading Bazelisk {version} into {target_path}...\n")
  try:
    with urllib.request.urlopen(download_url, timeout=60) as resp:
      if resp.status != 200:
        raise RuntimeError(
            f"Failed to download Bazelisk: HTTP {resp.status} {resp.reason}"
        )
      # Download to a temporary path and rename, so that an interrupted
      # download cannot leave a truncated binary in the cache.
      temp_path = target_path.with_suffix(".tmp")
      with open(temp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
      temp_path.chmod(0o755)
      temp_path.replace(target_path)
  except (urllib.error.URLError, OSError, RuntimeError) as e:
    raise RuntimeError(
        f"Failed to download Bazelisk {version} from {download_url}: {e}"
    ) from e
  return target_path


def find_bazel_binary(
    repo_root: pathlib.Path,
    explicit_bazel: pathlib.Path | None = None,
) -> pathlib.Path:
  """Locates a Bazel launcher that can run the version pinned by the repo.

  Bootstrapping installs Bazelisk rather than a specific Bazel release, so
  this tool never has to parse `.bazelversion` or reproduce Bazel's per-OS and
  per-CPU release naming. Note that a plain `bazel` on PATH is not necessarily
  sufficient: the apt-installed wrapper dispatches on `.bazelversion` but
  cannot fetch a missing version, which is why every candidate is verified by
  running `bazel --version` from `repo_root`.

  Checks in order:
    1. Explicit path passed as argument.
    2. Environment variable $BAZEL_BIN.
    3. `bazel` on PATH.
    4. Cached Bazelisk in `~/.cache/torch_tpu/bin/`.
    5. Bazelisk downloaded from its official GitHub releases.

  Args:
    repo_root: Path to repository root. Candidates are verified with this as the
      working directory so that `.bazelversion` is honoured.
    explicit_bazel: Optional explicit path provided by the caller.

  Returns:
    Path to the verified Bazel executable.

  Raises:
    FileNotFoundError: If no working Bazel binary can be located or
      bootstrapped.
  """
  if explicit_bazel:
    candidate = explicit_bazel.resolve()
    if _verify_bazel_binary(candidate, repo_root):
      return candidate
    raise FileNotFoundError(
        f"Specified Bazel binary at {candidate} is not executable or failed"
        " verification."
    )

  env_bin = os.environ.get("BAZEL_BIN")
  if env_bin:
    candidate = pathlib.Path(env_bin).resolve()
    if _verify_bazel_binary(candidate, repo_root):
      return candidate

  which_path = shutil.which("bazel")
  if which_path:
    candidate = pathlib.Path(which_path)
    if _verify_bazel_binary(candidate, repo_root):
      return candidate

  cached_path = _get_bazelisk_cache_path()
  if _verify_bazel_binary(cached_path, repo_root):
    return cached_path

  try:
    downloaded = _download_bazelisk(cached_path)
  except RuntimeError as e:
    raise FileNotFoundError(
        "Bazel is required to discover test targets, but no working `bazel`"
        f" was found and Bazelisk could not be downloaded:\n{e}\nInstall"
        " Bazelisk as described in the repository README, or point $BAZEL_BIN"
        " at a Bazel binary."
    ) from e

  if not _verify_bazel_binary(downloaded, repo_root):
    raise FileNotFoundError(
        f"Downloaded Bazelisk at {downloaded} failed verification. Install"
        " Bazelisk as described in the repository README, or point $BAZEL_BIN"
        " at a Bazel binary."
    )
  return downloaded


def parse_bazel_query_xml(xml_content: str) -> list[TestTarget]:
  """Parses Bazel query XML output into TestTarget objects.

  Bazel query XML output format (relevant subsets):
    The root element is `<query version="2">` containing `<rule>` elements for
    matching build targets.
    - Each `<rule>` has attributes:
        - `class`: The rule class type (e.g., `py_test`, `cc_test`).
        - `name`: The full target label (e.g., `//tests:ops_test_tpu-v5lite`).
        - `location`: The declaration source file and line (e.g.,
          `/workspace/tests/BUILD:10:1`).
    - Inside `<rule>`, tag metadata is stored in a `<list name="tags">` element
      containing zero or more `<string value="...">` elements.

  Example XML:
    ```xml
    <query version="2">
      <rule class="py_test" location="/workspace/tests/BUILD:10:1"
            name="//tests:my_test">
        <string name="name" value="my_test"/>
        <list name="tags">
          <string value="requires-tpu"/>
          <string value="presubmit-v5"/>
        </list>
      </rule>
    </query>
    ```

  Args:
    xml_content: String containing the raw XML output from `bazel query
      --output=xml`.

  Returns:
    A list of TestTarget objects extracted from the XML, sorted by target label.

  Raises:
    ValueError: If `xml_content` cannot be parsed as valid XML.
  """
  targets: list[TestTarget] = []
  try:
    root = xml.etree.ElementTree.fromstring(xml_content)
  except xml.etree.ElementTree.ParseError as e:
    raise ValueError(f"Failed to parse Bazel query XML output: {e}") from e

  for rule in root.findall("rule"):
    label = rule.get("name")
    if not label:
      continue

    if ":" in label:
      pkg_part, name = label.split(":", 1)
      package = pkg_part.lstrip("/")
    else:
      package = ""
      name = label.lstrip("/")

    tags: set[str] = set()
    for list_elem in rule.findall("list"):
      if list_elem.get("name") == "tags":
        for str_elem in list_elem.findall("string"):
          val = str_elem.get("value")
          if val:
            tags.add(val)

    location = rule.get("location", "")
    source_file = location.split(":")[0] if location else ""

    targets.append(
        TestTarget(
            name=name,
            package=package,
            tags=tags,
            source_file=source_file,
        )
    )

  targets.sort(key=lambda t: t.label)
  return targets


def discover_test_targets(
    repo_root: pathlib.Path,
    bazel_bin: pathlib.Path | None = None,
) -> list[TestTarget]:
  """Queries all test targets and tags defined in repo_root using Bazel query.

  Args:
    repo_root: Root directory of the repository where the Bazel query will be
      executed.
    bazel_bin: Optional explicit path to the Bazel binary executable. If None,
      the binary is located via `find_bazel_binary(repo_root)`.

  Returns:
    A list of TestTarget objects discovered in the repository.

  Raises:
    RuntimeError: If Bazel query fails or exits with a non-zero returncode.
    FileNotFoundError: If no working Bazel binary can be located.
  """
  resolved_bazel = find_bazel_binary(repo_root, explicit_bazel=bazel_bin)
  cmd = [
      str(resolved_bazel),
      "query",
      'kind(".*_test rule", //...)',
      "--output=xml",
  ]
  proc = subprocess.run(
      cmd,
      cwd=repo_root,
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    err_msg = proc.stderr.strip() or proc.stdout.strip()
    raise RuntimeError(
        f"Bazel query failed (exit code {proc.returncode}):\n{err_msg}"
    )

  return parse_bazel_query_xml(proc.stdout)


def filter_targets_by_patterns(
    targets: Sequence[TestTarget],
    patterns: Sequence[str],
) -> tuple[list[TestTarget], list[str]]:
  """Filters test targets down to those matching any of the given globs.

  Args:
    targets: Sequence of TestTarget objects to filter.
    patterns: Sequence of glob patterns (e.g. 'ops_test*' or
      '//tests:*_tpu-v5lite'). A target matches when any pattern matches its
      short name, its ':name' form, or its full label.

  Returns:
    A tuple of:
      - The matching TestTarget objects, in the order they were given.
      - The patterns that matched no target, so that callers can report
        typos rather than silently listing nothing.
  """
  matched: list[TestTarget] = []
  unmatched_patterns = set(patterns)

  for t in targets:
    candidates = (t.name, f":{t.name}", t.label)
    matching_patterns = [
        p
        for p in patterns
        if any(fnmatch.fnmatchcase(c, p) for c in candidates)
    ]
    if matching_patterns:
      matched.append(t)
      unmatched_patterns.difference_update(matching_patterns)

  return matched, [p for p in patterns if p in unmatched_patterns]


def map_jobs_to_tests(
    targets: Sequence[TestTarget],
    ci_configs: Mapping[str, Sequence[str]],
    selected_jobs: Sequence[str] | None = None,
) -> dict[str, list[TestTarget]]:
  """Maps each CI job to the test targets it will execute.

  Args:
    targets: Sequence of candidate TestTarget objects to evaluate.
    ci_configs: Mapping from CI job name (e.g. 'ci_cpu_presubmit') to its
      sequence of tag filters (e.g. ['-nobuild', '-notest', '-requires-tpu']).
    selected_jobs: Optional sequence of CI job names to restrict evaluation to.
      If None or empty, all jobs in `ci_configs` are evaluated in sorted order.

  Returns:
    A dictionary mapping each evaluated CI job name to the list of TestTarget
    objects matching that job's filter configuration.
  """
  job_to_targets: dict[str, list[TestTarget]] = {}
  jobs_to_eval = selected_jobs if selected_jobs else sorted(ci_configs.keys())

  for job in jobs_to_eval:
    if job not in ci_configs:
      continue
    filters = ci_configs[job]
    matched = [t for t in targets if matches_tag_filters(t.tags, filters)]
    job_to_targets[job] = matched

  return job_to_targets


def map_test_to_jobs(
    target_query: str,
    targets: Sequence[TestTarget],
    ci_configs: Mapping[str, Sequence[str]],
    selected_jobs: Sequence[str] | None = None,
) -> tuple[TestTarget | None, list[str]]:
  """Queries which CI jobs execute a specific test target.

  Args:
    target_query: The test target label or name to search for (e.g.,
      'ops_test_tpu-v5lite', '//tests:ops_test_tpu-v5lite', or
      ':ops_test_tpu-v5lite').
    targets: Sequence of available TestTarget objects to match against.
    ci_configs: Mapping from CI job name to its sequence of tag filters.
    selected_jobs: Optional sequence of CI job names to restrict evaluation to.
      If None or empty, all jobs in `ci_configs` are evaluated in sorted order.

  Returns:
    A tuple of:
      - The matched TestTarget object, or None if no match was found.
      - A list of CI job names that will execute this test target based on tag
        filters.
  """
  clean_query = target_query.strip()
  matched_target: TestTarget | None = None

  for t in targets:
    if clean_query in (t.name, t.label, f":{t.name}"):
      matched_target = t
      break

  if not matched_target:
    return None, []

  jobs_to_eval = selected_jobs if selected_jobs else sorted(ci_configs.keys())
  matching_jobs: list[str] = []
  for job in jobs_to_eval:
    if job not in ci_configs:
      continue
    filters = ci_configs[job]
    if matches_tag_filters(matched_target.tags, filters):
      matching_jobs.append(job)

  return matched_target, matching_jobs


def format_text_output(
    job_to_targets: Mapping[str, Sequence[TestTarget]],
    job_machines: Mapping[str, JobMachineInfo] | None = None,
) -> str:
  """Formats test targets grouped by CI job in human-readable text."""
  job_machines = job_machines or {}
  lines = []
  for job, targets in sorted(job_to_targets.items()):
    machine_desc = ""
    if job in job_machines:
      machine_desc = f" [Machine: {job_machines[job].description}]"
    lines.append(f"CI Job: {job}{machine_desc} ({len(targets)} tests)")
    if targets:
      for t in targets:
        lines.append(f"  {t.label}")
    else:
      lines.append("  (No tests match this job's filter)")
    lines.append("")
  return "\n".join(lines).rstrip()


def format_count_output(
    job_to_targets: Mapping[str, Sequence[TestTarget]],
    job_machines: Mapping[str, JobMachineInfo] | None = None,
) -> str:
  """Formats test count summary grouped by CI job."""
  job_machines = job_machines or {}
  col1_width = max([len(j) for j in job_to_targets] + [len("CI Job Name")]) + 4
  machine_descs = [
      job_machines[j].description if j in job_machines else "Unknown"
      for j in job_to_targets
  ]
  col2_width = max([len(m) for m in machine_descs] + [len("Machine Type")]) + 4
  header = (
      "CI Job Name".ljust(col1_width)
      + "Machine Type".ljust(col2_width)
      + "Test Count"
  )
  lines = [header, "-" * (col1_width + col2_width + 12)]
  for job, targets in sorted(job_to_targets.items()):
    machine_desc = (
        job_machines[job].description if job in job_machines else "Unknown"
    )
    lines.append(
        f"{job.ljust(col1_width)}{machine_desc.ljust(col2_width)}{len(targets)}"
    )
  return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
  """Builds CLI argument parser for list_ci_tests."""
  parser = argparse.ArgumentParser(
      description="List test targets included in TorchTPU CI jobs.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--jobs",
      type=str,
      default=None,
      help=(
          "Comma-separated list of CI job names to include (e.g."
          " --jobs=ci_cpu_presubmit,ci_tpu_v5_presubmit). Defaults to all jobs."
      ),
  )
  parser.add_argument(
      "--test",
      type=str,
      default=None,
      help=(
          "Specify a test target name or label to list all CI jobs that execute"
          " it (e.g. --test=ops_test_tpu-v5lite or"
          " --test=//tests:ops_test_tpu-v5lite)."
      ),
  )
  parser.add_argument(
      "--tests",
      type=str,
      metavar="PATTERNS",
      default=None,
      help=(
          "Comma-separated list of test target globs limiting the output to"
          " the matching tests (e.g. --tests=ops_test*,//tests:*_tpu-v5lite)."
          " Each glob is matched against the target name and its full label."
      ),
  )
  parser.add_argument(
      "--format",
      choices=["text", "count"],
      default="text",
      help="Output format: 'text' (default) or 'count'.",
  )
  parser.add_argument(
      "--list_jobs",
      action="store_true",
      help="List all defined CI jobs and their tag filters, then exit.",
  )
  parser.add_argument(
      "--bazel_bin",
      type=str,
      default=None,
      help="Explicit path to the Bazel binary executable to use.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = _build_arg_parser()
  args = parser.parse_args(argv)

  repo_root = find_repo_root()
  try:
    bazelrc_path = get_bazelrc_path(repo_root)
  except FileNotFoundError as e:
    sys.stderr.write(f"Error: {e}\n")
    return 1

  ci_configs = parse_bazelrc_tag_filters(bazelrc_path)
  if not ci_configs:
    sys.stderr.write(
        f"Error: No CI test_tag_filters found in {bazelrc_path}.\n"
    )
    return 1

  workflows_dir = get_workflows_dir(repo_root)
  job_machines = parse_workflow_machine_types(workflows_dir)

  if args.list_jobs:
    print(f"Available CI jobs from {bazelrc_path}:")
    for job, filters in sorted(ci_configs.items()):
      machine_desc = ""
      if job in job_machines:
        machine_desc = f" [Machine: {job_machines[job].description}]"
      print(f"  {job}{machine_desc}: {','.join(filters)}")
    return 0

  selected_jobs: list[str] | None = None
  if args.jobs:
    selected_jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    invalid_jobs = [j for j in selected_jobs if j not in ci_configs]
    if invalid_jobs:
      sys.stderr.write(
          f"Warning: Unknown CI job(s): {', '.join(invalid_jobs)}.\n"
          f"Available jobs: {', '.join(sorted(ci_configs.keys()))}\n\n"
      )

  bazel_path = pathlib.Path(args.bazel_bin) if args.bazel_bin else None
  try:
    targets = discover_test_targets(repo_root, bazel_bin=bazel_path)
  except Exception as e:  # pylint: disable=broad-exception-caught
    sys.stderr.write(f"Error discovering test targets: {e}\n")
    return 1

  if args.tests:
    patterns = [p.strip() for p in args.tests.split(",") if p.strip()]
    targets, unmatched_patterns = filter_targets_by_patterns(targets, patterns)
    if unmatched_patterns:
      sys.stderr.write(
          "Warning: No test target matches:"
          f" {', '.join(unmatched_patterns)}.\n\n"
      )

  if args.test:
    matched_target, matching_jobs = map_test_to_jobs(
        args.test, targets, ci_configs, selected_jobs
    )
    if not matched_target:
      sys.stderr.write(
          f"Error: Test target '{args.test}' was not found in {repo_root}.\n"
      )
      return 1

    print(f"Test Target: {matched_target.label}")
    print(f"Tags: {', '.join(sorted(matched_target.tags))}")
    print(f"Source: {matched_target.source_file}")
    print(f"\nRuns in {len(matching_jobs)} CI job(s):")
    if matching_jobs:
      for j in sorted(matching_jobs):
        machine_desc = ""
        if j in job_machines:
          machine_desc = f" [{job_machines[j].description}]"
        print(f"  - {j}{machine_desc}")
    else:
      print("  (None: excluded from all queried CI jobs)")
    return 0

  job_to_targets = map_jobs_to_tests(targets, ci_configs, selected_jobs)

  if args.format == "count":
    print(format_count_output(job_to_targets, job_machines))
  else:
    print(format_text_output(job_to_targets, job_machines))

  return 0


if __name__ == "__main__":
  sys.exit(main())
