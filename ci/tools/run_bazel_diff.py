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
# dependencies = ["pyyaml"]
# ///

"""Calculates impacted Bazel tests using bazel-diff and runs them."""

import argparse
from collections.abc import Collection, Sequence
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

# Reused for the .bazelrc `--test_tag_filters` parsing and Bazel's tag filter
# semantics, so the per-config skip decision cannot drift from `list_ci_tests`.
from ci.tools import list_ci_tests

_BAZEL_DIFF_VERSION: Final[str] = "v46.1.0"
_BAZEL_DIFF_URL: Final[str] = (
    f"https://github.com/Tinder/bazel-diff/releases/download/{_BAZEL_DIFF_VERSION}/"
    "bazel-diff_deploy.jar"
)
_BAZEL_DIFF_JAR: Final[pathlib.Path] = pathlib.Path("/tmp/bazel-diff.jar")

# Execution modes. `calculate` and `test` split the work across two GitHub
# Actions jobs so the expensive diff is computed once on a cheap CPU runner,
# and scarce accelerator runners are only claimed when there is work for them.
# `all` keeps the original single-job behavior for local runs and postsubmits.
MODE_CALCULATE: Final[str] = "calculate"
MODE_TEST: Final[str] = "test"
MODE_ALL: Final[str] = "all"

# Config used for `cquery` graph hashing when no runner-specific config is
# given (i.e. the central `calculate` job). `ci` only sets execution, caching
# and output flags -- it adds no `--build_tag_filters`/`--test_tag_filters` --
# so the configured target graph stays un-pruned while still reusing the shared
# remote analysis cache.
_DEFAULT_CQUERY_CONFIG: Final[str] = "ci"

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
    r"\.github/workflows/presubmit_runner\.yml",
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
  mode: str = MODE_ALL
  impacted_targets_file: str = ""
  candidate_configs: Sequence[str] = ()
  workflow_file: str = ""


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
      "--mode",
      default=MODE_ALL,
      choices=(MODE_CALCULATE, MODE_TEST, MODE_ALL),
      help=(
          "Execution mode: 'calculate' only computes impacted targets and"
          " reports them via $GITHUB_OUTPUT, 'test' only runs tests from a"
          " precomputed targets file, and 'all' does both in one invocation."
      ),
  )
  parser.add_argument(
      "--impacted-targets-file",
      default="",
      help=(
          "Path (relative to the workspace) of a precomputed impacted targets"
          " file to test instead of recomputing the diff."
      ),
  )
  parser.add_argument(
      "--candidate-configs",
      default="",
      help=(
          "Comma-separated list of Bazel configs to evaluate when computing"
          " impacted_configs."
      ),
  )
  parser.add_argument(
      "--workflow-file",
      default="",
      help=(
          "Path (relative to workspace) to a GitHub Actions workflow YAML file"
          " from which candidate Bazel configs are extracted via"
          " list_ci_tests.extract_workflow_configs()."
      ),
  )
  ns = parser.parse_args(argv)
  candidate_configs = tuple(
      c.strip() for c in str(ns.candidate_configs).split(",") if c.strip()
  )
  return BazelDiffOptions(
      workspace_dir=pathlib.Path(ns.workspace_dir),
      base_sha=ns.base_sha,
      current_sha=ns.current_sha or "HEAD",
      bazel_config=ns.bazel_config,
      extra_bazel_flags=ns.extra_bazel_flags,
      test_all=(str(ns.test_all).lower() == "true"),
      workflow_ref=ns.workflow_ref,
      mode=ns.mode,
      impacted_targets_file=ns.impacted_targets_file,
      candidate_configs=candidate_configs,
      workflow_file=ns.workflow_file,
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


def normalize_label(label: str) -> str:
  """Strips bazel-diff's main-repo prefix so the label is query-friendly.

  bazel-diff emits fully canonical labels such as `@@//ci/tools:foo`, which
  `bazel query` in the main repository expects as plain `//ci/tools:foo`.

  Args:
    label: A target label as emitted by bazel-diff.

  Returns:
    The label without a leading main-repository prefix.
  """
  for prefix in ("@@//", "@//"):
    if label.startswith(prefix):
      return label[len(prefix) - 2 :]
  return label


def query_impacted_tests(
    impacted_targets: Sequence[str],
) -> Sequence[list_ci_tests.TestTarget] | None:
  """Narrows impacted targets to test rules and reads their tags.

  bazel-diff reports every impacted target, including source files, filegroups
  and binaries, so the raw count says nothing about whether any test would
  actually run. `tests(...)` keeps only test rules (expanding test suites).

  This is a loading-phase query: it needs no configuration, no analysis and no
  accelerator, so it is safe and cheap to run on the CPU runner.

  Args:
    impacted_targets: Labels emitted by `bazel-diff get-impacted-targets`.

  Returns:
    The impacted test targets with their tags (possibly empty if none of the
    impacted targets is a test), or None if the query could not be completed.
  """
  if not impacted_targets:
    return []
  labels = " ".join(f'"{normalize_label(t)}"' for t in impacted_targets)
  # The query goes through --query_file rather than argv: a large PR can impact
  # thousands of targets, and Linux caps a single argument at 128 KiB
  # (MAX_ARG_STRLEN), which the joined label list would blow past.
  with tempfile.TemporaryDirectory() as td:
    query_file = pathlib.Path(td) / "query.txt"
    query_file.write_text(f"tests(set({labels}))", encoding="utf-8")
    res = subprocess.run(
        [
            "bazel",
            "query",
            f"--query_file={os.fspath(query_file)}",
            "--output=xml",
            "--keep_going",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
  # Exit code 3 means "partial success with errors" under --keep_going (e.g. a
  # target was deleted in the PR); the XML emitted so far is still usable. Any
  # target that failed to resolve has no test left to run anyway.
  if res.returncode not in (0, 3):
    print(f"bazel query failed with exit code {res.returncode}.")
    if res.stderr:
      print(res.stderr)
    return None
  if not res.stdout.strip():
    return []
  try:
    return list_ci_tests.parse_bazel_query_xml(res.stdout)
  except ValueError as e:
    print(f"Could not parse bazel query output: {e}")
    return None


def resolve_impacted_configs(
    impacted_targets: Sequence[str],
    bazelrc_path: pathlib.Path,
    allowed_configs: Collection[str] | None = None,
    impacted_tests: Sequence[list_ci_tests.TestTarget] | None = None,
) -> Sequence[str] | None:
  """Determines which CI configurations have at least one test to run.

  Each presubmit runner applies its own `--test_tag_filters`, so an impacted
  target set can be non-empty overall while a given runner has nothing to do.
  Resolving that up front lets the workflow skip the job before it claims a
  scarce accelerator.

  This relies on every CI config declaring its own `--test_tag_filters` line in
  `.bazelrc` rather than inheriting one via `--config=`, because
  `parse_bazelrc_tag_filters` does not resolve config inheritance. A config
  that only inherits its filters is absent from the mapping and would never be
  reported as impacted. `list_ci_tests` already depends on the same invariant.

  Args:
    impacted_targets: Labels emitted by `bazel-diff get-impacted-targets`.
    bazelrc_path: Path to the repository `.bazelrc`.
    allowed_configs: The subset of configurations to evaluate (e.g. active
      presubmit configs extracted from a workflow). Non-matching configs in
      `.bazelrc` are ignored. If None, evaluates all configs in `.bazelrc`.
    impacted_tests: Pre-queried test targets with their tags, or None to query
      `impacted_targets` via `query_impacted_tests`.

  Returns:
    The sorted config names with at least one impacted test, or None if the
    set could not be determined and callers must conservatively run everything.
  """
  if impacted_tests is None:
    impacted_tests = query_impacted_tests(impacted_targets)
  if impacted_tests is None:
    print("Could not determine impacted tests; assuming all configs impacted.")
    return None

  print(f"Impacted test target(s): {len(impacted_tests)}")
  if not impacted_tests:
    return []

  try:
    tag_filters = list_ci_tests.parse_bazelrc_tag_filters(bazelrc_path)
  except OSError as e:
    print(
        f"Could not read {bazelrc_path} ({e}); assuming all configs impacted."
    )
    return None

  impacted_configs = sorted(
      config
      for config, filters in tag_filters.items()
      if (allowed_configs is None or config in allowed_configs)
      and any(
          list_ci_tests.matches_tag_filters(test.tags, filters)
          for test in impacted_tests
      )
  )
  print(f"Configurations with impacted tests: {impacted_configs}")
  return impacted_configs


def resolve_mode(opts: BazelDiffOptions) -> str:
  """Derives the effective execution mode from the requested options.

  An explicit `--mode=calculate` always wins. Otherwise a missing
  `--bazel-config` means we cannot run tests at all, so we degrade to
  `calculate`. An explicit `--mode=test` runs tests directly. A precomputed
  targets file (and no `--test-all` override) means the diff was already
  computed upstream, so we only need to run the tests.

  Args:
    opts: The parsed command-line options.

  Returns:
    One of MODE_CALCULATE, MODE_TEST or MODE_ALL.
  """
  if opts.mode == MODE_CALCULATE or not opts.bazel_config:
    return MODE_CALCULATE
  if opts.mode == MODE_TEST:
    return MODE_TEST
  if opts.impacted_targets_file and not opts.test_all:
    return MODE_TEST
  return MODE_ALL


def write_github_output(
    has_impacted_targets: bool,
    is_full_run: bool,
    impacted_configs: Sequence[str] | None = None,
) -> None:
  """Reports the diff outcome to downstream jobs via `$GITHUB_OUTPUT`.

  `has_impacted_targets` lets the workflow skip claiming accelerator runners
  entirely, and `is_full_run` tells them to test `//...` instead of downloading
  a targets artifact that was never uploaded.

  `impacted_configs` refines this per runner. It is deliberately left unset
  when the value is None, because an unset output reads back as the empty
  string in a workflow expression, which the gate treats as "unknown, run
  everything". That keeps the conservative fallback in one place instead of
  encoding a sentinel value that both sides must agree on.

  Args:
    has_impacted_targets: Whether any target needs to be tested.
    is_full_run: Whether downstream runners must test everything.
    impacted_configs: Bazel configs with at least one impacted test, or None if
      the set could not be determined.
  """
  github_output = os.environ.get("GITHUB_OUTPUT")
  print(
      f"has_impacted_targets={str(has_impacted_targets).lower()},"
      f" is_full_run={str(is_full_run).lower()},"
      f" impacted_configs={impacted_configs}"
  )
  if not github_output:
    return
  with open(github_output, "a", encoding="utf-8") as f:
    f.write(f"has_impacted_targets={str(has_impacted_targets).lower()}\n")
    f.write(f"is_full_run={str(is_full_run).lower()}\n")
    if impacted_configs is not None:
      f.write(f"impacted_configs={json.dumps(list(impacted_configs))}\n")


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


def _run_precomputed_targets(
    opts: BazelDiffOptions, workspace_dir: pathlib.Path
) -> int:
  """Runs tests from a targets file computed by an upstream `calculate` job.

  Args:
    opts: The parsed command-line options.
    workspace_dir: The resolved repository workspace directory.

  Returns:
    The exit code of the Bazel test command, or 0 if there is nothing to test.
  """
  if opts.test_all:
    print("Full test run requested in test mode. Running all tests.")
    return _run_bazel_test(opts.bazel_config, opts.extra_bazel_flags, ["//..."])

  if not opts.impacted_targets_file:
    print(
        "No impacted targets file specified in test mode. Skipping Bazel"
        " entirely!"
    )
    return 0

  targets_path = workspace_dir / opts.impacted_targets_file
  impacted_targets = (
      read_targets_file(targets_path) if targets_path.is_file() else []
  )
  if not impacted_targets:
    print(
        f"Precomputed impacted targets file ({opts.impacted_targets_file}) is"
        " empty or missing. Skipping Bazel entirely!"
    )
    return 0

  print(
      f"Using precomputed impacted targets from {opts.impacted_targets_file}"
      f" ({len(impacted_targets)} target(s))."
  )
  print("--- Impacted Targets List ---")
  for target in impacted_targets:
    print(target)
  print("-----------------------------")
  return _run_bazel_test(
      opts.bazel_config,
      opts.extra_bazel_flags,
      [
          "--skip_incompatible_explicit_targets",
          f"--target_pattern_file={os.fspath(targets_path)}",
      ],
  )


def main(argv: Sequence[str]) -> int:
  sys.stdout.reconfigure(line_buffering=True)
  opts = parse_args(argv)

  workspace_dir = opts.workspace_dir
  if not workspace_dir.is_dir():
    raise ValueError(f"Workspace directory does not exist: {workspace_dir}")
  workspace_dir = workspace_dir.resolve()
  os.chdir(workspace_dir)

  mode = resolve_mode(opts)
  print(f"Running bazel-diff in '{mode}' mode.")

  # The diff was already computed by an upstream job, so skip straight to
  # testing the targets it handed us.
  if mode == MODE_TEST:
    return _run_precomputed_targets(opts, workspace_dir)

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

    # if is_ci_definition_modified(changed_files, opts.workflow_ref):
    #   print("CI definitions for this job modified.")
    #   print(
    #       "Forcing a full test run: bazel-diff sees no impacted targets for a"
    #       " CI-only change, so this job would otherwise pass without running"
    #       " any tests."
    #   )
    #   test_all = True

    # Check if all modified files are documentation, images, or repo metadata.
    # If so, they do not affect any build or test targets, so we can skip Bazel
    # entirely.
    if are_all_files_doc_or_metadata(changed_files):
      print("All modified files are documentation or repository metadata.")
      print(
          "No test targets impacted by these changes. Skipping Bazel entirely!"
      )
      if mode == MODE_CALCULATE:
        write_github_output(has_impacted_targets=False, is_full_run=False)
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
    if mode == MODE_CALCULATE:
      print("Signaling a full test run to downstream runners.")
      write_github_output(has_impacted_targets=True, is_full_run=True)
      return 0
    return _run_bazel_test(opts.bazel_config, opts.extra_bazel_flags, ["//..."])

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

  # The central `calculate` job runs without a runner-specific config, so fall
  # back to the shared `ci` base config for graph hashing.
  cquery_config = opts.bazel_config or _DEFAULT_CQUERY_CONFIG
  cquery_args = (f"--cqueryCommandOptions=--config={cquery_config}",)

  print("--- Generating Base Hashes ---")
  print(f"PR detected. Fetching exact base SHA: {base_sha}")
  subprocess.run(
      ["git", "fetch", "--depth=1", "origin", base_sha],
      stderr=subprocess.DEVNULL,
      check=False,
  )
  subprocess.run(["git", "checkout", base_sha], check=True)
  base_hashes = workspace_dir / "base_hashes.json"
  try:
    run_bazel_diff(
        "generate-hashes",
        "-w",
        workspace_dir,
        base_hashes,
        "--useCquery",
        *cquery_args,
    )
  finally:
    # Never leave the checkout parked on the base SHA; a later job step (or a
    # local run) would otherwise operate on the wrong tree.
    print(f"Checking out current SHA: {current_sha}")
    subprocess.run(["git", "checkout", current_sha], check=True)

  print("--- Generating PR Hashes ---")
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

  if impacted_targets:
    print(f"Found {len(impacted_targets)} impacted target(s).")
    print("--- Impacted Targets List ---")
    for target in sorted(impacted_targets):
      print(target)
    print("-----------------------------")

  if mode == MODE_CALCULATE:
    # Query for the test rules among impacted targets to drop non-test targets
    # (source files, cc_library, py_library) and deleted targets (exit code 3
    # under --keep_going). If the query succeeds, update impacted_targets.txt
    # on disk so downstream runners receive only valid, resolved test targets.
    impacted_tests = query_impacted_tests(impacted_targets)
    if impacted_tests is not None:
      test_labels = [t.label for t in impacted_tests]
      impacted_targets_file.write_text(
          "\n".join(test_labels) + ("\n" if test_labels else ""),
          encoding="utf-8",
      )
    # bazel-diff reports source files, filegroups and binaries alongside test
    # rules, so a non-empty target set does not mean every runner has work.
    # Narrowing to the configs that actually match lets the workflow skip a
    # runner before it claims an accelerator it would hold for seconds.
    assume_all_impacted: bool = False
    if opts.candidate_configs:
      allowed_configs = frozenset(opts.candidate_configs)
    elif opts.workflow_file:
      wf_path = workspace_dir / opts.workflow_file
      extracted = (
          list_ci_tests.extract_workflow_configs(wf_path)
          if wf_path.is_file()
          else []
      )
      if not extracted:
        print(
            f"Could not extract candidate configs from {opts.workflow_file};"
            " assuming all configs impacted."
        )
        assume_all_impacted = True
        allowed_configs = None
      else:
        allowed_configs = frozenset(extracted)
    else:
      allowed_configs = None

    if assume_all_impacted:
      impacted_configs = None
    else:
      impacted_configs = resolve_impacted_configs(
          impacted_targets,
          workspace_dir / ".bazelrc",
          allowed_configs=allowed_configs,
          impacted_tests=impacted_tests,
      )
    if impacted_configs is not None and not impacted_configs:
      print("No impacted target is a test in any CI configuration.")
    write_github_output(
        has_impacted_targets=impacted_configs is None or bool(impacted_configs),
        is_full_run=False,
        impacted_configs=impacted_configs,
    )
    return 0

  if not impacted_targets:
    print("No tests impacted by these changes. Skipping Bazel entirely!")
    return 0

  print("--- Running Impacted Bazel Tests ---")
  # bazel-diff feeds the impacted targets as explicit labels, and Bazel treats
  # an explicitly requested target whose target_compatible_with is unsatisfied
  # as a hard error -- unlike the wildcard //... runs everywhere else, which
  # skip it. --skip_incompatible_explicit_targets restores the wildcard
  # behavior, so config-gated tests are skipped rather than failing presubmit.
  return _run_bazel_test(
      opts.bazel_config,
      opts.extra_bazel_flags,
      [
          "--skip_incompatible_explicit_targets",
          f"--target_pattern_file={os.fspath(impacted_targets_file)}",
      ],
  )


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
