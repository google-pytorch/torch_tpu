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

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Calculates impacted Bazel tests using bazel-diff and runs them."""

import argparse
from collections.abc import Collection, Mapping, Sequence
import dataclasses
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Final
import xml.etree.ElementTree

_BAZEL_DIFF_VERSION: Final[str] = "v46.1.0"
_BAZEL_DIFF_URL: Final[str] = (
    f"https://github.com/Tinder/bazel-diff/releases/download/{_BAZEL_DIFF_VERSION}/"
    "bazel-diff_deploy.jar"
)
_BAZEL_DIFF_JAR: Final[pathlib.Path] = pathlib.Path("/tmp/bazel-diff.jar")

# Global Bazel configuration file patterns (one regex per entry). Files matching
# these patterns represent Bazel configuration; if these change, the whole cache
# is invalidated.
#
# `requirements/` holds the resolved dependency locks that `MODULE.bazel` feeds
# to `pip.parse`, so a change there moves the inputs of essentially every Python
# target and selecting targets would cost more than running everything. It is
# also what makes it safe for a job to repin a lock in its checkout before
# calling this script: the `git checkout` calls below abort on a dirty file
# whose content differs between the two commits, and matching here returns
# before any checkout happens. Narrowing this entry would break such a job.
GLOBAL_BAZEL_CONFIGS: Final[Sequence[str]] = (
    # go/keep-sorted start
    r"MODULE\.bazel",
    r"REPO\.bazel",
    r"WORKSPACE",
    r"WORKSPACE\.bzlmod",
    r"\.bazelrc",
    r"\.bazelversion",
    r"requirements/.*",
    # go/keep-sorted end
)
_GLOBAL_BAZEL_CONFIGS_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(" + "|".join(GLOBAL_BAZEL_CONFIGS) + r")$"
)

# CI definitions shared by every job (one regex per entry), plus, handled
# separately below, the workflow file that defines the running job itself.
#
# bazel-diff cannot see any of this: these files are not in the build graph, so
# changing them leaves every target's hash identical and the job reports zero
# impacted targets and passes without running a single test. That is the worst
# possible outcome for a change to CI, which is precisely the kind of change
# that needs its own test suite run to be believed.
#
# TODO: make the trigger logic more precise to avoid unnecessary full runs.
SHARED_CI_PATHS: Final[Sequence[str]] = (
    # go/keep-sorted start
    r"\.github/actions/.*",
    # go/keep-sorted end
)
_SHARED_CI_PATHS_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(" + "|".join(SHARED_CI_PATHS) + r")$"
)

# Documentation and repository metadata (one regex per entry). These files don't
# change any code, so changes to them should not trigger tests to run.
DOC_OR_METADATA_PATTERNS: Final[Sequence[str]] = (
    # go/keep-sorted start
    r"(.*/)?OWNERS",
    r".*\.gif",
    r".*\.jpeg",
    r".*\.jpg",
    r".*\.md",
    r".*\.png",
    r".*\.svg",
    r".*\.webp",
    r"LICENSE.*",
    r"\.clang.*",
    r"\.gitignore",
    r"\.vscode/.*",
    r"docs/.*",
    # go/keep-sorted end
)
_DOC_OR_METADATA_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(" + "|".join(DOC_OR_METADATA_PATTERNS) + r")$"
)

# The OpInfo matrix tests (tagged as "ops-test") are by far the most expensive
# tests in CI: they sweep the whole op_db across dtypes and TPU generations.
# They also depend on the monolithic pywrap_torch_tpu library, so bazel-diff
# marks them impacted by almost any change. Run them only for changes to the
# operator lowerings and the harness they actually exercise.
# One regex per entry.
OPS_TEST_PATHS: Final[Sequence[str]] = (
    # go/keep-sorted start
    r"csrc/ops/.*",
    r"src/torch_tpu/ops/.*",
    r"tests/dynamism_test_utils\.py",
    r"tests/op_testing\.py",
    r"tests/ops_test\.py",
    r"tests/ops_test_data\.py",
    r"tests/ops_test_gpu_golden.*",
    # go/keep-sorted end
)
_OPS_TEST_PATHS_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(" + "|".join(OPS_TEST_PATHS) + r")$"
)

# Seconds of test work per target (summed over its Bazel shards) on the 8-chip
# v5e runner, used to split a job's tests across several runners. Taken from
# the test summary of a full ci_tpu_v5_presubmit run: a sharded test's total is
# its "Stats over N runs" count times the average. The numbers only steer the
# split, so a stale table makes shards uneven but never drops a test.
_TEST_DURATIONS_FILE: Final[pathlib.Path] = (
    pathlib.Path(__file__).parent / "test_durations.json"
)
_DEFAULT_TEST_SECONDS: Final[float] = 30.0

_BAZELRC_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:common|build|test)(?::(?P<config>[\w.-]+))?\s+(?P<flags>.*)$"
)


@dataclasses.dataclass(frozen=True)
class BazelDiffOptions:
  """Parsed command-line options for running bazel-diff."""

  workspace_dir: pathlib.Path
  base_sha: str
  current_sha: str
  bazel_config: str
  extra_bazel_flags: str
  test_all: bool
  workflow_ref: str
  shard_index: int = 0
  shard_count: int = 1


def parse_args(argv: Sequence[str]) -> BazelDiffOptions:
  """Parses command-line arguments."""
  default_ws = os.environ.get("GITHUB_WORKSPACE", ".")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--workspace-dir",
      default=default_ws,
      help="Path to the repository workspace directory.",
  )
  parser.add_argument(
      "--base-sha",
      default="",
      help="Base commit SHA for diff comparison.",
  )
  parser.add_argument(
      "--current-sha",
      default="HEAD",
      help="Current commit SHA (defaults to HEAD).",
  )
  parser.add_argument(
      "--bazel-config",
      default="",
      help="Bazel --config flag to use.",
  )
  parser.add_argument(
      "--extra-bazel-flags",
      default="",
      help="Additional flags for the Bazel test command.",
  )
  parser.add_argument(
      "--test-all",
      default="false",
      help=(
          "Set to 'true' to bypass bazel-diff target selection and run all"
          " tests."
      ),
  )
  parser.add_argument(
      "--workflow-ref",
      default=os.environ.get("GITHUB_WORKFLOW_REF", ""),
      help=(
          "Ref of the workflow defining the running job, in the form"
          " '<owner>/<repo>/.github/workflows/<file>@<ref>'. Defaults to"
          " $GITHUB_WORKFLOW_REF, which GitHub Actions sets automatically."
      ),
  )
  parser.add_argument(
      "--shard-index",
      type=int,
      default=0,
      help="Zero-based index of this job among --shard-count jobs.",
  )
  parser.add_argument(
      "--shard-count",
      type=int,
      default=1,
      help=(
          "Number of jobs splitting the selected tests. Each job runs the"
          " tests assigned to --shard-index."
      ),
  )
  ns = parser.parse_args(argv)
  if ns.shard_count < 1 or not 0 <= ns.shard_index < ns.shard_count:
    parser.error(
        f"--shard-index={ns.shard_index} must be in [0, --shard-count) and"
        f" --shard-count={ns.shard_count} must be at least 1."
    )
  return BazelDiffOptions(
      workspace_dir=pathlib.Path(ns.workspace_dir),
      base_sha=ns.base_sha,
      current_sha=ns.current_sha or "HEAD",
      bazel_config=ns.bazel_config,
      extra_bazel_flags=ns.extra_bazel_flags,
      test_all=(str(ns.test_all).lower() == "true"),
      workflow_ref=ns.workflow_ref,
      shard_index=ns.shard_index,
      shard_count=ns.shard_count,
  )


def is_global_bazel_config_modified(changed_files: Sequence[str]) -> bool:
  """Returns True if any changed file matches GLOBAL_BAZEL_CONFIGS."""
  return any(_GLOBAL_BAZEL_CONFIGS_REGEX.search(f) for f in changed_files)


def workflow_path_from_ref(workflow_ref: str) -> str:
  """Extracts the repo-relative workflow path from a GITHUB_WORKFLOW_REF value.

  Args:
    workflow_ref: Value of the form
      `<owner>/<repo>/.github/workflows/<file>@<git-ref>`, or the empty string
      when running outside GitHub Actions.

  Returns:
    The repo-relative path (e.g. `.github/workflows/presubmit.yml`), or the
    empty string if `workflow_ref` does not contain one. The owner and repo
    names are stripped by searching for the `.github/` segment rather than by
    splitting on a fixed number of slashes, since a repo name may itself
    contain a slash-like structure in forks.
  """
  path = workflow_ref.split("@", 1)[0]
  marker = "/.github/"
  index = path.find(marker)
  if index == -1:
    return ""
  return path[index + len("/") :]


def is_ci_definition_modified(
    changed_files: Sequence[str], workflow_ref: str
) -> bool:
  """Returns True if a changed file defines how the running job tests.

  Args:
    changed_files: Repo-relative paths changed by the PR.
    workflow_ref: `GITHUB_WORKFLOW_REF` for the running job, used to identify
      the job's own workflow file. Workflow files *other* than that one are
      deliberately ignored: an edit to the nightly workflow says nothing about
      whether the presubmit jobs still work, and forcing every job to a full run
      on any workflow edit would be needlessly expensive.

  Returns:
    True if the job's own definition, or CI infrastructure shared by all jobs,
    was modified.
  """
  if any(_SHARED_CI_PATHS_REGEX.search(f) for f in changed_files):
    return True
  workflow_path = workflow_path_from_ref(workflow_ref)
  return bool(workflow_path) and workflow_path in changed_files


def is_doc_or_metadata_file(path: str) -> bool:
  """Returns True if path is a documentation, image, or repo metadata file."""
  return bool(_DOC_OR_METADATA_REGEX.search(path))


def are_all_files_doc_or_metadata(changed_files: Sequence[str]) -> bool:
  """Returns True if all paths in changed_files are doc/metadata."""
  return all(is_doc_or_metadata_file(f) for f in changed_files)


def should_run_ops_tests(changed_files: Sequence[str]) -> bool:
  """Returns True if any changed file matches OPS_TEST_PATHS."""
  return any(_OPS_TEST_PATHS_REGEX.search(f) for f in changed_files)


def filter_ops_test_targets(
    impacted_targets: Sequence[str],
    ops_test_targets: Sequence[str],
) -> Sequence[str]:
  """Subtracts ops-test targets from impacted_targets, preserving order."""
  # bazel-diff under Bzlmod outputs canonical labels prefixed with "@@//"
  # (e.g. "@@//tests:ops_test"), whereas `bazel query` outputs apparent labels
  # starting with "//" (e.g. "//tests:ops_test"). Normalize both before matching.
  ops_set = {t.removeprefix("@@") for t in ops_test_targets}
  return [t for t in impacted_targets if t.removeprefix("@@") not in ops_set]


def read_targets_file(path: pathlib.Path) -> Sequence[str]:
  """Reads a newline-delimited targets file, stripping empty lines."""
  return list(
      filter(
          None, map(str.strip, path.read_text(encoding="utf-8").splitlines())
      )
  )


def resolve_base_sha(
    workspace_dir: pathlib.Path, current_sha: str, base_sha: str
) -> str:
  """Resolves effective base SHA via ci/tools/resolve_base_sha.sh."""
  resolve_script = workspace_dir / "ci" / "tools" / "resolve_base_sha.sh"
  res = subprocess.run(
      ["bash", os.fspath(resolve_script), current_sha, base_sha],
      capture_output=True,
      text=True,
      check=True,
  )
  return res.stdout.strip()


def get_changed_files(base_sha: str, current_sha: str) -> Sequence[str]:
  """Fetches base_sha if needed, prints diff summary, and returns changed files."""
  subprocess.run(
      ["git", "fetch", "--depth=1", "origin", base_sha],
      stderr=subprocess.DEVNULL,
      check=False,
  )
  changed_files_out = subprocess.check_output(
      ["git", "diff", "--name-only", base_sha, current_sha],
      text=True,
  )
  changed_files = list(
      filter(None, map(str.strip, changed_files_out.splitlines()))
  )
  print(f"BASE_SHA={base_sha}")
  print(f"CURRENT_SHA={current_sha}")
  print("CHANGED_FILES:")
  status_res = subprocess.run(
      ["git", "diff", "--name-status", base_sha, current_sha],
      check=False,
  )
  if status_res.returncode != 0:
    for f in changed_files:
      print(f)
  print("")
  print("Diff stat:")
  subprocess.run(
      ["git", "diff", "--stat", base_sha, current_sha],
      check=False,
  )
  print("")
  return changed_files


def run_bazel_diff(
    *args: str | os.PathLike[str],
) -> subprocess.CompletedProcess[bytes]:
  """Executes bazel-diff jar with the given CLI arguments."""
  cmd = [
      "java",
      "-jar",
      os.fspath(_BAZEL_DIFF_JAR),
      *(os.fspath(a) for a in args),
  ]
  return subprocess.run(cmd, check=True)


def effective_flag(
    bazelrc_text: str, command_line: Sequence[str], flag: str
) -> str | None:
  """Returns the last value `--<flag>` takes once --config flags are expanded.

  Applies the config-less `common`, `build` and `test` lines of `.bazelrc`,
  then the command line, expanding each `--config=<name>` in place with the
  flags from its `common:`, `build:` and `test:` lines, in file order and
  recursively, the way `bazel test` does. Later values win.

  Args:
    bazelrc_text: Contents of the repository `.bazelrc`.
    command_line: Flags as passed to `bazel test`, e.g. `["--config=ci"]`.
    flag: Flag name without dashes, e.g. `test_tag_filters`.

  Returns:
    The flag's final value, or None if nothing sets it.
  """
  config_flags: dict[str, list[str]] = {}
  for line in bazelrc_text.splitlines():
    m = _BAZELRC_LINE_RE.match(line.strip())
    if not m:
      continue
    try:
      flags = shlex.split(m.group("flags"), comments=True)
    except ValueError:
      flags = m.group("flags").split()
    config_flags.setdefault(m.group("config") or "", []).extend(flags)

  prefix = f"--{flag}="
  value = None

  def expand(flags: Sequence[str], active: tuple[str, ...]) -> None:
    nonlocal value
    for f in flags:
      if f.startswith("--config="):
        name = f.removeprefix("--config=")
        if name and name not in active:
          expand(config_flags.get(name, []), active + (name,))
      elif f.startswith(prefix):
        value = f.removeprefix(prefix)

  expand([*config_flags.get("", []), *command_line], ())
  return value


def matches_tag_filters(tags: Collection[str], filters: Sequence[str]) -> bool:
  """Returns True if a test with `tags` passes Bazel's --test_tag_filters.

  A test passes if it has none of the `-tag` entries and, when there are
  positive entries, at least one of them.
  """
  positive = [f for f in filters if not f.startswith("-")]
  if any(f.removeprefix("-") in tags for f in filters if f.startswith("-")):
    return False
  return not positive or any(f in tags for f in positive)


@dataclasses.dataclass(frozen=True)
class QueriedTest:
  """A test target and its tags, as reported by `bazel query`."""

  label: str
  tags: frozenset[str]


def tests_query_expression(patterns: Sequence[str]) -> str:
  """Returns a `bazel query` expression for the tests `patterns` would run.

  Wildcard patterns such as `//...` skip `manual` targets in `bazel test`, so
  those are subtracted; explicitly listed labels are kept either way.
  """
  labels = [p.removeprefix("@@") for p in patterns]
  wildcards = [p for p in labels if p.endswith("...") or p.endswith(":all")]
  expression = f"tests(set({' '.join(labels)}))"
  if wildcards:
    expression += (
        f' except attr(tags, "\\bmanual\\b", set({" ".join(wildcards)}))'
    )
  return expression


def parse_query_xml(xml_text: str) -> list[QueriedTest]:
  """Returns the label and tags of each rule in `bazel query --output=xml`."""
  tests = []
  for rule in xml.etree.ElementTree.fromstring(xml_text).iter("rule"):
    tags = frozenset(
        s.get("value", "")
        for lst in rule.findall("list")
        if lst.get("name") == "tags"
        for s in lst.findall("string")
    )
    tests.append(QueriedTest(rule.get("name", ""), tags))
  return tests


def load_test_durations(path: pathlib.Path) -> dict[str, float]:
  """Reads per-test seconds from `path`, keyed by full `//` label.

  The file leaves off the leading `//` because Copybara rewrites labels under
  examples/ and perflab/ on export and can't map them back.
  """
  raw = json.loads(path.read_text(encoding="utf-8"))
  return {f"//{label.removeprefix('//')}": secs for label, secs in raw.items()}


def estimated_seconds(
    test: QueriedTest, durations: Mapping[str, float], local_test_jobs: int
) -> float:
  """Returns the seconds `test` adds to a runner's wall time.

  An `exclusive` test runs alone, so it costs its full duration. Other tests
  share the runner's `local_test_jobs` accelerators, so they cost a fraction.
  """
  seconds = durations.get(test.label, _DEFAULT_TEST_SECONDS)
  if "exclusive" in test.tags:
    return seconds
  return seconds / max(local_test_jobs, 1)


def partition_tests(
    tests: Sequence[QueriedTest],
    shard_count: int,
    durations: Mapping[str, float],
    local_test_jobs: int,
) -> list[list[str]]:
  """Splits tests into `shard_count` groups with similar estimated run time.

  Greedy longest-first: each test goes to the group with the least estimated
  time so far. Ties break on label so every shard computes the same split.

  Args:
    tests: Tests the job would run.
    shard_count: Number of groups.
    durations: Seconds of test work per label; missing labels get
      `_DEFAULT_TEST_SECONDS`.
    local_test_jobs: Tests Bazel runs at once on one runner.

  Returns:
    For each shard, its sorted test labels.
  """
  loads = [0.0] * shard_count
  shards: list[list[str]] = [[] for _ in range(shard_count)]
  costs = {
      t.label: estimated_seconds(t, durations, local_test_jobs) for t in tests
  }
  for label in sorted(costs, key=lambda label: (-costs[label], label)):
    i = min(range(shard_count), key=lambda i: (loads[i], i))
    loads[i] += costs[label]
    shards[i].append(label)
  return [sorted(s) for s in shards]


def shard_target_patterns(
    patterns: Sequence[str], shards: Sequence[Sequence[str]], shard_index: int
) -> list[str]:
  """Returns the target patterns shard `shard_index` should pass to Bazel.

  Shard 0 keeps the original patterns minus the tests the other shards took,
  so anything the split missed (e.g. a test the query didn't see) still runs
  somewhere. The other shards run only their own tests.
  """
  if shard_index > 0:
    return list(shards[shard_index])
  moved = sorted(label for shard in shards[1:] for label in shard)
  return list(patterns) + [f"-{label}" for label in moved]


def _query_tests(patterns: Sequence[str]) -> list[QueriedTest] | None:
  """Runs `bazel query` for the tests in `patterns`, or None on failure."""
  with tempfile.NamedTemporaryFile(
      "w", suffix=".query", encoding="utf-8", delete=False
  ) as f:
    f.write(tests_query_expression(patterns))
  res = subprocess.run(
      [
          "bazel",
          "query",
          "--keep_going",
          "--output=xml",
          f"--query_file={f.name}",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  os.unlink(f.name)
  # Exit code 3 means some patterns failed to load; the rest are still listed,
  # and shard 0 runs anything missing.
  if res.returncode not in (0, 3):
    print(res.stderr)
    return None
  return parse_query_xml(res.stdout)


def select_shard_patterns(
    opts: BazelDiffOptions, patterns: Sequence[str]
) -> list[str]:
  """Returns the target patterns for this job's shard of `patterns`."""
  command_line = [f"--config={opts.bazel_config}"] if opts.bazel_config else []
  command_line += shlex.split(opts.extra_bazel_flags)
  bazelrc = (opts.workspace_dir / ".bazelrc").read_text(encoding="utf-8")
  tag_filters = effective_flag(bazelrc, command_line, "test_tag_filters")
  local_test_jobs = effective_flag(bazelrc, command_line, "local_test_jobs")

  tests = _query_tests(patterns)
  if tests is None:
    print(
        "WARNING: could not list tests to split them. Shard 0 runs everything"
        " and the other shards run nothing."
    )
    return list(patterns) if opts.shard_index == 0 else []

  filters = [f for f in (tag_filters or "").split(",") if f]
  tests = [t for t in tests if matches_tag_filters(t.tags, filters)]
  durations = load_test_durations(_TEST_DURATIONS_FILE)
  jobs = int(local_test_jobs) if (local_test_jobs or "").isdigit() else 1
  shards = partition_tests(tests, opts.shard_count, durations, jobs)

  by_label = {t.label: t for t in tests}
  print(f"--- Test split across {opts.shard_count} shards ---")
  for i, shard in enumerate(shards):
    seconds = sum(
        estimated_seconds(by_label[l], durations, jobs) for l in shard
    )
    marker = " (this job)" if i == opts.shard_index else ""
    print(f"Shard {i}: {len(shard)} test(s), ~{seconds / 60:.1f} min{marker}")
  for label in shards[opts.shard_index]:
    print(f"  {label}")
  return shard_target_patterns(patterns, shards, opts.shard_index)


def _run_bazel_test(
    bazel_config: str,
    extra_bazel_flags: str,
    target_args: Sequence[str],
) -> int:
  """Runs bazel test via bash so extra_bazel_flags shell expansions are preserved.

  Args:
    bazel_config: Bazel --config name to pass (or empty string for none).
    extra_bazel_flags: Additional raw flags string passed to `bazel test`.
    target_args: Target patterns or `--target_pattern_file` flags to test.

  Returns:
    The exit code of the Bazel test command (0 if successful or if exit code 4
    'no tests found' is returned).
  """
  target_str = " ".join(shlex.quote(arg) for arg in target_args)
  config_flag = f"--config={shlex.quote(bazel_config)} " if bazel_config else ""
  cmd = f"bazel test {config_flag}{extra_bazel_flags} {target_str}"
  sys.stdout.flush()
  res = subprocess.run(["bash", "-c", cmd], check=False)
  if res.returncode == 4:
    print("Bazel returned Exit Code 4 (No tests found). Treating as success!")
    return 0
  if res.returncode != 0:
    print(f"Bazel test failed with exit code {res.returncode}.")
  return res.returncode


def _run_tests(
    opts: BazelDiffOptions,
    patterns: Sequence[str],
    patterns_file: pathlib.Path | None = None,
) -> int:
  """Runs `patterns`, or only this job's share of them when sharded.

  Args:
    opts: Parsed options, with `workspace_dir` resolved.
    patterns: Target patterns to test.
    patterns_file: A file already holding `patterns`, if there is one.

  Returns:
    The exit code of the Bazel test command.
  """
  if opts.shard_count > 1:
    patterns = select_shard_patterns(opts, patterns)
    if not patterns:
      print("No tests assigned to this shard. Skipping Bazel entirely!")
      return 0
    patterns_file = opts.workspace_dir / "shard_targets.txt"
    patterns_file.write_text("\n".join(patterns) + "\n", encoding="utf-8")
  if patterns_file is None:
    return _run_bazel_test(opts.bazel_config, opts.extra_bazel_flags, patterns)
  # bazel-diff and the shard split feed explicit labels, and Bazel treats an
  # explicitly requested target whose target_compatible_with is unsatisfied as
  # a hard error -- unlike the wildcard //... runs everywhere else, which skip
  # it. --skip_incompatible_explicit_targets restores the wildcard behavior, so
  # config-gated tests are skipped rather than failing presubmit.
  return _run_bazel_test(
      opts.bazel_config,
      opts.extra_bazel_flags,
      [
          "--skip_incompatible_explicit_targets",
          f"--target_pattern_file={os.fspath(patterns_file)}",
      ],
  )


def main(argv: Sequence[str]) -> int:
  sys.stdout.reconfigure(line_buffering=True)
  opts = parse_args(argv)

  workspace_dir = opts.workspace_dir
  if not workspace_dir.is_dir():
    raise ValueError(f"Workspace directory does not exist: {workspace_dir}")
  workspace_dir = workspace_dir.resolve()
  opts = dataclasses.replace(opts, workspace_dir=workspace_dir)
  os.chdir(workspace_dir)

  # Resolve effective base SHA via centralized helper. If CURRENT_SHA is a merge
  # commit (synthetic PR merge), resolves to CURRENT_SHA^1 to isolate PR
  # changes.
  base_sha = resolve_base_sha(workspace_dir, opts.current_sha, opts.base_sha)
  current_sha = opts.current_sha
  test_all = opts.test_all
  run_ops_tests = True

  # Check if global Bazel configurations were modified. If so, bypass bazel-diff
  # to guarantee a full build validation.
  #
  # We intentionally do not use bazel-diff's native `--seed-filepaths` flag for
  # handling changes to global configuration files. It forces bazel-diff to
  # calculate the entire repository graph, serialize all targets (5000+) to
  # impacted_targets.txt, and feed them back to Bazel. This introduces a slight
  # performance regression.
  if base_sha and not test_all:
    changed_files = get_changed_files(base_sha, current_sha)

    if is_global_bazel_config_modified(changed_files):
      print("Global Bazel configurations modified.")
      print("Forcing a full test run to ensure global configuration validity.")
      test_all = True

    if is_ci_definition_modified(changed_files, opts.workflow_ref):
      print("CI definitions for this job modified.")
      print(
          "Forcing a full test run: bazel-diff sees no impacted targets for a"
          " CI-only change, so this job would otherwise pass without running"
          " any tests."
      )
      test_all = True

    # Check if all modified files are documentation, images, or repo metadata.
    # If so, they do not affect any build or test targets, so we can skip Bazel
    # entirely.
    if are_all_files_doc_or_metadata(changed_files):
      print("All modified files are documentation or repository metadata.")
      print(
          "No test targets impacted by these changes. Skipping Bazel entirely!"
      )
      return 0

    run_ops_tests = should_run_ops_tests(changed_files)
    if run_ops_tests:
      print("Operator lowering or OpInfo harness modified; keeping ops-test.")

  # If BASE_SHA is empty (e.g., workflow_dispatch, postsubmit, nightly on main)
  # or label "ci:disable-bazel-diff" is applied, diff-based testing is invalid.
  # Fall back to running all tests.
  if not base_sha or test_all:
    print(
        "No BASE_SHA provided (not a PR), bazel-diff disabled via PR label or"
        " global configuration files modified. Skipping bazel-diff and running"
        " all tests."
    )
    return _run_tests(opts, ["//..."])

  print("Downloading bazel-diff...")
  subprocess.run(
      [
          "curl",
          "-fLo",
          os.fspath(_BAZEL_DIFF_JAR),
          "--retry",
          "5",
          "--retry-connrefused",
          _BAZEL_DIFF_URL,
      ],
      check=True,
  )

  print(f"Computing impacted targets between {base_sha} and {current_sha}...")

  cquery_args = (
      (f"--cqueryCommandOptions=--config={opts.bazel_config}",)
      if opts.bazel_config
      else ()
  )

  print("--- Generating Base Hashes ---")
  print(f"PR detected. Fetching exact base SHA: {base_sha}")
  subprocess.run(
      ["git", "fetch", "--depth=1", "origin", base_sha],
      stderr=subprocess.DEVNULL,
      check=False,
  )
  subprocess.run(["git", "checkout", base_sha], check=True)
  base_hashes = workspace_dir / "base_hashes.json"
  run_bazel_diff(
      "generate-hashes",
      "-w",
      workspace_dir,
      base_hashes,
      "--useCquery",
      *cquery_args,
  )

  print("--- Generating PR Hashes ---")
  print(f"Checking out current SHA: {current_sha}")
  subprocess.run(["git", "checkout", current_sha], check=True)
  pr_hashes = workspace_dir / "pr_hashes.json"
  run_bazel_diff(
      "generate-hashes",
      "-w",
      workspace_dir,
      pr_hashes,
      "--useCquery",
      *cquery_args,
  )

  print("--- Determining Impacted Targets ---")
  impacted_targets_file = workspace_dir / "impacted_targets.txt"
  run_bazel_diff(
      "get-impacted-targets",
      "-sh",
      base_hashes,
      "-fh",
      pr_hashes,
      "-w",
      workspace_dir,
      "-o",
      impacted_targets_file,
  )

  impacted_targets = read_targets_file(impacted_targets_file)

  # When no files under OPS_TEST_PATHS were modified, query Bazel for all
  # targets tagged "ops-test" and subtract them from the impacted target list.
  # Their generated *_build_test targets remain in impacted_targets.txt so the
  # test binaries are still compiled on every PR.
  if not run_ops_tests and impacted_targets:
    print(
        "No operator lowering or OpInfo harness changes; dropping ops-test"
        " targets."
    )
    query_res = subprocess.run(
        ["bazel", "query", r'attr(tags, "\bops-test\b", //tests/...)'],
        capture_output=True,
        text=True,
        check=False,
    )
    ops_test_targets = list(
        filter(None, map(str.strip, query_res.stdout.splitlines()))
    )
    if ops_test_targets:
      impacted_targets = filter_ops_test_targets(
          impacted_targets, ops_test_targets
      )
      impacted_targets_file.write_text(
          "\n".join(impacted_targets) + ("\n" if impacted_targets else ""),
          encoding="utf-8",
      )

  print("--- Running Impacted Bazel Tests ---")
  if impacted_targets:
    print(f"Found {len(impacted_targets)} impacted target(s).")
    print("--- Impacted Targets List ---")
    for target in sorted(impacted_targets):
      print(target)
    print("-----------------------------")
    return _run_tests(opts, impacted_targets, impacted_targets_file)
  else:
    print("No tests impacted by these changes. Skipping Bazel entirely!")
    return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
