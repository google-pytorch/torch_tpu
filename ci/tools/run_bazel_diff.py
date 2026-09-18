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
GLOBAL_BAZEL_CONFIGS: Final[Sequence[str]] = (
    # go/keep-sorted start
    r"MODULE\.bazel",
    r"REPO\.bazel",
    r"WORKSPACE",
    r"WORKSPACE\.bzlmod",
    r"\.bazelrc",
    r"\.bazelversion",
    # go/keep-sorted end
)
_GLOBAL_BAZEL_CONFIGS_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(" + "|".join(GLOBAL_BAZEL_CONFIGS) + r")$"
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


@dataclasses.dataclass(frozen=True)
class BazelDiffOptions:
  """Parsed command-line options for running bazel-diff."""

  workspace_dir: pathlib.Path
  base_sha: str
  current_sha: str
  bazel_config: str
  extra_bazel_flags: str
  test_all: bool


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
  ns = parser.parse_args(argv)
  return BazelDiffOptions(
      workspace_dir=pathlib.Path(ns.workspace_dir),
      base_sha=ns.base_sha,
      current_sha=ns.current_sha or "HEAD",
      bazel_config=ns.bazel_config,
      extra_bazel_flags=ns.extra_bazel_flags,
      test_all=(str(ns.test_all).lower() == "true"),
  )


def is_global_bazel_config_modified(changed_files: Sequence[str]) -> bool:
  """Returns True if any changed file matches GLOBAL_BAZEL_CONFIGS."""
  return any(_GLOBAL_BAZEL_CONFIGS_REGEX.search(f) for f in changed_files)


def is_doc_or_metadata_file(path: str) -> bool:
  """Returns True if path is a documentation, image, or repo metadata file."""
  return bool(_DOC_OR_METADATA_REGEX.search(path))


def are_all_files_doc_or_metadata(changed_files: Sequence[str]) -> bool:
  """Returns True if all paths in changed_files are doc/metadata."""
  return all(is_doc_or_metadata_file(f) for f in changed_files)


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

    # Check if all modified files are documentation, images, or repo metadata.
    # If so, they do not affect any build or test targets, so we can skip Bazel
    # entirely.
    if are_all_files_doc_or_metadata(changed_files):
      print("All modified files are documentation or repository metadata.")
      print(
          "No test targets impacted by these changes. Skipping Bazel entirely!"
      )
      return 0

  # If BASE_SHA is empty (e.g., workflow_dispatch, postsubmit, nightly on main)
  # or label "ci:disable-bazel-diff" is applied, diff-based testing is invalid.
  # Fall back to running all tests.
  if not base_sha or test_all:
    print(
        "No BASE_SHA provided (not a PR), bazel-diff disabled via PR label or"
        " global configuration files modified. Skipping bazel-diff and running"
        " all tests."
    )
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
  print("--- Running Impacted Bazel Tests ---")
  if impacted_targets:
    print(f"Found {len(impacted_targets)} impacted target(s).")
    print("--- Impacted Targets List ---")
    for target in impacted_targets:
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
