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
from collections.abc import Sequence
import dataclasses
import os
import pathlib
import re
import shlex
import subprocess
import sys
from typing import Final

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
# TODO(b/568391187): add a unit test to ensure every entry must match at least
# one file.
OPS_TEST_PATHS: Final[Sequence[str]] = (
    # go/keep-sorted start
    r"csrc/ops/.*",
    r"src/torch_tpu/ops/.*",
    r"tests/dynamism_test_utils\.py",
    r"tests/goldens\.pin",
    r"tests/op_testing\.py",
    r"tests/ops_test\.py",
    r"tests/ops_test_data\.py",
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
  ns = parser.parse_args(argv)
  return BazelDiffOptions(
      workspace_dir=pathlib.Path(ns.workspace_dir),
      base_sha=ns.base_sha,
      current_sha=ns.current_sha or "HEAD",
      bazel_config=ns.bazel_config,
      extra_bazel_flags=ns.extra_bazel_flags,
      test_all=(str(ns.test_all).lower() == "true"),
      workflow_ref=ns.workflow_ref,
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
  changed_files_out = subprocess.run(
      ["git", "diff", "--name-only", base_sha, current_sha],
      capture_output=True,
      text=True,
      check=True,
  ).stdout
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


def _bazel_analysis_command(bazel_config: str) -> list[str]:
  """Builds the `bazel cquery` command used for full-repo analysis checks."""
  cmd = ["bazel", "cquery"]
  if bazel_config:
    cmd.append(f"--config={bazel_config}")
  cmd.extend(["--output=label", "/..."])
  cmd[-1] = "/" + cmd[-1]
  return cmd


def _run_bazel_analysis_check(bazel_config: str) -> int:
  """Analyzes every target the way bazel-diff's `generate-hashes` does.

  A full `bazel test` run is not a substitute for this check: the CI configs
  set `--build_tests_only` and, for the wheel tests, `--test_lang_filters`,
  which drop non-matching targets, such as Starlark analysis tests, before they
  are analyzed. bazel-diff's cquery applies no such filters, so a target that
  fails analysis would pass the PR that introduced it and then break
  bazel-diff's base hash generation for every later PR.

  Args:
    bazel_config: Bazel --config name to pass (or empty string for none). As
      with bazel-diff's `--cqueryCommandOptions`, no other flags are passed.

  Returns:
    The exit code of the `bazel cquery` command.
  """
  cmd = _bazel_analysis_command(bazel_config)
  print("--- Analyzing All Targets (as bazel-diff does) ---")
  sys.stdout.flush()
  # Only failures matter; the label of every configured target is noise.
  res = subprocess.run(cmd, stdout=subprocess.DEVNULL, check=False)
  if res.returncode != 0:
    print(
        f"Bazel analysis failed with exit code {res.returncode}. bazel-diff"
        " analyzes every target when computing hashes, so merging this change"
        " would break bazel-diff for all subsequent PRs."
    )
  return res.returncode


def _bazel_test_command(
    bazel_config: str,
    extra_bazel_flags: str,
    target_args: Sequence[str],
) -> str:
  """Builds the `bazel test` shell command string."""
  target_str = " ".join(shlex.quote(arg) for arg in target_args)
  config_flag = f"--config={shlex.quote(bazel_config)} " if bazel_config else ""
  return f"bazel test {config_flag}{extra_bazel_flags} {target_str}"


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
  cmd = _bazel_test_command(bazel_config, extra_bazel_flags, target_args)
  sys.stdout.flush()
  res = subprocess.run(["bash", "-c", cmd], check=False)
  if res.returncode == 4:
    print("Bazel returned Exit Code 4 (No tests found). Treating as success!")
    return 0
  if res.returncode != 0:
    print(f"Bazel test failed with exit code {res.returncode}.")
  return res.returncode


def main(argv: Sequence[str]) -> int:
  sys.stdout.reconfigure(line_buffering=True)
  opts = parse_args(argv)

  workspace_dir = opts.workspace_dir
  if not workspace_dir.is_dir():
    raise ValueError(f"Workspace directory does not exist: {workspace_dir}")
  workspace_dir = workspace_dir.resolve()
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
    # bazel-diff is skipped here, so nothing else would analyze the targets
    # that `bazel test` filters out. Catch analysis failures now rather than
    # in the base hash generation of the next PR.
    analysis_rc = _run_bazel_analysis_check(opts.bazel_config)
    if analysis_rc != 0:
      return analysis_rc
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
  else:
    print("No tests impacted by these changes. Skipping Bazel entirely!")
    return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
