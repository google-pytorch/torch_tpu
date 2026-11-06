# Copyright 2025 Google LLC
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

"""Measures error message test coverage for torch_tpu.

Scans the torch_tpu C++ sources for error macros, runs the error tests in
coverage mode, and computes the per-file coverage. If it's infeasible to test an
error macro (e.g. because the function never returns an error Status), please
add an `// ERROR_COV_INFEASIBLE=<why>` comment on the same line as the error
macro so that it will be excluded from the coverage report.

This works the same way in google3 (Blaze) and in OSS (Bazel). All file paths
are in the canonical form returned by `canonicalize_path()`, so that both
builds compute the same keys.
"""

import bisect
import collections
from collections.abc import Mapping, Sequence
import concurrent.futures
import dataclasses
import os
import re
import subprocess

from absl import logging
from tests import repo_paths
import yaml

# The error tests to run in coverage mode, relative to the repository root.
# They must be data dependencies of the target that runs them.
# TODO(wan): Add *_cov versions of tests/distributed:device_mesh_errors_test and
# tests/distributed:handshake_errors_test, and run them in error_coverage_test.
COV_TEST_BINARIES = (
    # go/keep-sorted start
    "tests/distributed/collective_errors_test_cov",
    "tests/errors_test_cov",
    "tests/tpu_errors_test_cov",
    # go/keep-sorted end
)

# The environment variables for running the error tests in coverage mode. They
# ensure the needed info in stdout while skipping addr2line symbolization
# (coverage only parses the `C++ error trace` and `Exception raised from`
# header, not the symbolized C++ call stack frames).
COV_ENV = {
    "TORCH_DISABLE_ADDR2LINE": "1",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "TORCH_TPU_ENABLE_DEBUG_CHECKS": "1",
}

# The environment variables that blaze/bazel set for the test that runs the
# error tests, which the error tests must not act on: the test filter
# (`--test_filter`) and sharding variables would make them run only some of
# their test cases, and the output files belong to the parent test.
_PARENT_TEST_RUNNER_ENV_VARS = (
    # go/keep-sorted start
    "TESTBRIDGE_TEST_ONLY",
    "TESTBRIDGE_TEST_RUNNER_FAIL_FAST",
    "TEST_PREMATURE_EXIT_FILE",
    "TEST_SHARD_INDEX",
    "TEST_SHARD_STATUS_FILE",
    "TEST_TOTAL_SHARDS",
    "XML_OUTPUT_FILE",
    # go/keep-sorted end
)

# These macros create a new error from scratch (as opposed to
# transforming/plumbing an existing error). We focus on coverage for such macros
# for now.
_NEED_TO_COVER_MACROS = (
    "TT_ERROR",
    "TT_RET_CHECK",
    "TT_CHECK_THROW",
)

# The prefix of paths in google3, e.g. in `__FILE__`.
_G3_PREFIX = repo_paths.G3_REPO_PATH + "/"

# The prefix of the internal files, which Copybara doesn't export. Their error
# macros aren't checked, so that google3 and OSS can share the baseline.
_INTERNAL_PREFIX = "google/"

# The internal C++ directory that Copybara moves to `csrc/` in OSS.
_OSS_OVERLAY_CSRC_PREFIX = "oss/torch_tpu/csrc/"


def canonicalize_path(path: str) -> str:
  """Returns the path relative to the google3 package root.

  This equals the OSS path for `csrc/`, which Copybara doesn't move: google3
  paths (`third_party/py/torch_tpu/csrc/...`) and OSS paths (`csrc/...` or
  `./csrc/...`) of the same file have the same canonical form. So do the
  internal `oss/torch_tpu/csrc/X` and its exported path `csrc/X`.

  Args:
    path: A path relative to the google3 root or to the repository root.
  """
  # In OSS, `__FILE__` of a header is relative to the current directory, e.g.
  # `./csrc/common/aten_utils.h`.
  path = path.removeprefix("./")
  path = path.removeprefix(_G3_PREFIX)
  if path.startswith(_OSS_OVERLAY_CSRC_PREFIX):
    path = "csrc/" + path.removeprefix(_OSS_OVERLAY_CSRC_PREFIX)
  return path


def _get_cpp_files(repo_root: str) -> list[str]:
  """Returns the paths of the C++ files in the repository."""
  cpp_files = []
  for root, dirs, files in os.walk(repo_root):
    # Prune directories that aren't part of the repository (e.g. virtual
    # environments) and build output symlinks (e.g. blaze-*, bazel-*) in-place
    # so os.walk does not recurse into them.
    dirs[:] = [
        d
        for d in dirs
        if not repo_paths.is_ignored_dir(d)
        and not d.startswith(("blaze-", "bazel-"))
    ]
    for file in files:
      if file.endswith((".cc", ".h")):
        cpp_files.append(os.path.join(root, file))
  return cpp_files


class ErrorMacroLine:
  """Represents a line containing an error macro."""

  def __init__(self, line_num: int, cov_feasible: bool, macro: str):
    """Creates an ErrorMacroLine object.

    Args:
      line_num: The 1-based line number of the error macro.
      cov_feasible: Whether the error macro is feasible to test.
      macro: The name of the error macro.
    """

    self.line_num = line_num
    self.cov_feasible = cov_feasible
    self.macro = macro


def get_file_to_locs(repo_root: str) -> dict[str, list[ErrorMacroLine]]:
  """Scans all torch_tpu C++ files for error macros.

  Args:
    repo_root: The repository root, e.g. from
      `repo_paths.get_repo_root_in_runfiles()`.

  Returns:
    The error macros in each C++ file that has any, keyed by the canonical
    path of the file. Internal files (under `google/`) are skipped.

  Raises:
    ValueError: If two files have the same canonical path.
  """
  cpp_files = _get_cpp_files(repo_root)
  file_to_locs: dict[str, list[ErrorMacroLine]] = {}
  canonical_to_file: dict[str, str] = {}
  with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(lambda f: (f, get_error_macro_locs(f)), cpp_files)
    for cpp_file, locs in results:
      if not locs:
        continue
      path = canonicalize_path(os.path.relpath(cpp_file, repo_root))
      if path.startswith(_INTERNAL_PREFIX):
        continue
      if path in canonical_to_file:
        raise ValueError(
            f"{canonical_to_file[path]} and {cpp_file} have the same canonical"
            f" path {path}."
        )
      canonical_to_file[path] = cpp_file
      file_to_locs[path] = locs
  return file_to_locs


class SourceLoc:
  """Represents a source location in a C++ file."""

  def __init__(self, path: str, line_number: int):
    self.path = path
    self.line_number = line_number

  def __str__(self):
    return f"{self.path}:{self.line_number}"

  def __repr__(self):
    return self.__str__()

  def __lt__(self, other):
    """Compares two SourceLoc objects."""
    if not isinstance(other, SourceLoc):
      raise TypeError("Can only compare SourceLoc objects.")
    return self.path < other.path or (
        self.path == other.path and self.line_number < other.line_number
    )

  def __eq__(self, other):
    """Compares two SourceLoc objects."""
    if not isinstance(other, SourceLoc):
      raise TypeError("Can only compare SourceLoc objects.")
    return self.path == other.path and self.line_number == other.line_number

  def __hash__(self):
    return hash((self.path, self.line_number))


# Matches one of the torch_tpu error macros.
_TT_ERROR_API_RE = re.compile(
    r"\b(TT_ERROR|TT_RET_CHECK|TT_ASSIGN_OR_RETURN|TT_ASSIGN_OR_THROW|"
    r"TT_RETURN_IF_ERROR|TT_THROW_IF_ERROR|TT_CHECK_THROW)\("
)


# Matches an ERROR_COV_INFEASIBLE comment.
_COV_INFEASIBLE_RE = re.compile(r".*\bERROR_COV_INFEASIBLE\s*=.+")


def get_error_macro_locs(cpp_file: str) -> list[ErrorMacroLine]:
  """Scans a C++ file for error macros.

  Args:
    cpp_file: The C++ filepath to scan.

  Returns:
    The 1-based line numbers of the error macros found in the file, or an empty
    list if the file is a test file. The numbers are 1-based and sorted in
    ascending order.
  """

  if cpp_file.endswith(("_test.cc", "_test_helper.h")):
    return []
  macro_lines = []
  in_preprocessor = False
  with open(cpp_file, "r") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()

      # Ignore comments and preprocessor directives.
      if in_preprocessor or line.startswith("#"):
        in_preprocessor = line.endswith("\\")
        continue
      if line.startswith(("//", '"')):
        continue

      m = _TT_ERROR_API_RE.search(line)
      if m:
        # If we cannot find the ERROR_COV_INFEASIBLE comment, we assume the
        # error macro is feasible to test.
        cov_feasible = _COV_INFEASIBLE_RE.match(line) is None
        macro_lines.append(ErrorMacroLine(line_num, cov_feasible, m.group(1)))
  return macro_lines


# Matches the covered locations output by the errors_test_cov target.
# Group 1: file path.
# Group 2: line number.
_COVERED_LOCS_RE = re.compile(r"Covered ([/\w\.]+):(\d+)")


def parse_cov_output(cov_output: str) -> set[SourceLoc]:
  """Parses the coverage output from the coverage target.

  Args:
    cov_output: The stdout of the coverage target.

  Returns:
    The set of covered source locations, with canonical paths.
  """
  return {
      SourceLoc(canonicalize_path(path), int(line_num))
      for path, line_num in _COVERED_LOCS_RE.findall(cov_output)
  }


def run_cov_tests(repo_root: str) -> str:
  """Runs the error tests in coverage mode and returns their stdout.

  Args:
    repo_root: The repository root in the runfiles, which contains
      `COV_TEST_BINARIES`.

  Returns:
    The combined stdout of all the error tests.

  Raises:
    FileNotFoundError: If an error test isn't in the runfiles.
    RuntimeError: If an error test fails.
  """
  env = {
      name: value
      for name, value in os.environ.items()
      if name not in _PARENT_TEST_RUNNER_ENV_VARS
  }
  env.update(COV_ENV)
  outputs = []
  for binary in COV_TEST_BINARIES:
    test_path = os.path.join(repo_root, binary)
    if not os.path.exists(test_path):
      raise FileNotFoundError(
          f"Test executable not found: {test_path!r}. Check the data"
          " dependencies of the target."
      )
    logging.info("Running test executable: %s", test_path)
    ret = subprocess.run(
        [test_path, "--test_mode=cov"],
        capture_output=True,
        check=False,
        env=env,
    )
    stdout = ret.stdout.decode("utf-8")
    if ret.returncode != 0:
      raise RuntimeError(
          f"{test_path} failed with return code {ret.returncode}.\n"
          f"STDOUT:\n{stdout}\nSTDERR:\n{ret.stderr.decode('utf-8')}"
      )
    logging.info(
        "%s printed %d bytes to stdout and %d bytes to stderr.",
        test_path,
        len(ret.stdout),
        len(ret.stderr),
    )
    outputs.append(stdout)
  return "".join(outputs)


class CoverageForFile:
  """Represents the coverage for a single C++ file."""

  def __init__(
      self,
      cpp_file: str,
      locs: list[int],
      covered_locs: list[int],
      infeasible_locs: list[int],
  ):
    """Initializes the coverage for a single C++ file.

    Args:
      cpp_file: The C++ filepath.
      locs: The 1-based line numbers of the error macros found in the file.
      covered_locs: The 1-based line numbers of the error macros covered by the
        error tests.
      infeasible_locs: The 1-based line numbers of the error macros that are
        infeasible to test.
    """

    self.cpp_file = cpp_file
    self.locs = sorted(locs)
    self.covered_locs = sorted(covered_locs)
    self.infeasible_locs = sorted(infeasible_locs)
    # Locations of error macros that are not covered by the error tests.
    self.not_covered_locs = sorted(set(locs) - set(covered_locs))
    # Locations of error macros that we should add tests for.
    self.to_cover_locs = sorted(
        set(self.not_covered_locs) - set(infeasible_locs)
    )
    # Locations of error macros that are marked as infeasible to test but
    # actually tested.
    self.infeasible_and_covered_locs = sorted(
        set(self.infeasible_locs) & set(self.covered_locs)
    )

  def ratio(self) -> float:
    """Returns the coverage ratio as a float in the range [0, 1]."""
    return len(self.covered_locs) / len(self.locs) if self.locs else 1.0

  def ratio_upper_bound(self) -> float:
    """Returns the maximum possible coverage ratio.

    Since some error macros are infeasible to test, maximum possible
    coverage may be < 1.
    """
    return (
        len(set(self.locs) - set(self.infeasible_locs)) / len(self.locs)
        if self.locs
        else 1.0
    )

  def __str__(self):
    msg = (
        f"{self.cpp_file}:"
        f" {len(self.covered_locs)}/{len(self.locs)} ({100*self.ratio():.1f}%)"
        " error macros covered"
    )
    if self.infeasible_locs:
      msg += f" ({len(self.infeasible_locs)} error macros infeasible to cover)"
    return msg

  def __repr__(self):
    return self.__str__()

  def __lt__(self, other):
    """Compares two CoverageForFile objects.

    Sorts by the number of to-be-covered locations in ascending order.
    Breaks ties by the number of covered locations in descending order.

    With this ordering, smaller CoverageForFile objects in general need less
    work.

    Args:
      other: The other CoverageForFile object to compare to.

    Returns:
      True if self is less than other, False otherwise.
    """
    if not isinstance(other, CoverageForFile):
      raise TypeError("Can only compare CoverageForFile objects.")

    return (len(self.to_cover_locs), -len(self.covered_locs)) < (
        len(other.to_cover_locs),
        -len(other.covered_locs),
    )

  def to_dict(self) -> Mapping[str, int]:
    """Returns the serialized statistics map for the file."""
    return {
        "infeasible": len(self.infeasible_locs),
        "remaining": len(self.to_cover_locs),
        "total": len(self.locs),
    }


def sub_coverage_report(file_path: str, lines: list[int], status: str) -> str:
  """Generates a human-readable report for lines with a given status.

  Args:
    file_path: The C++ file path.
    lines: A list of 1-based line numbers to report.
    status: The status of the lines (e.g. "covered").

  Returns:
    A human-readable report for the given lines.
  """
  report = []
  for loc in lines:
    report.append(f"  {file_path}:{loc}: {status}")
  return "\n".join(report)


def coverage_report(coverage: CoverageForFile) -> str:
  """Generates a human-readable report for a single file's error coverage."""
  report = [
      str(coverage),
      sub_coverage_report(coverage.cpp_file, coverage.covered_locs, "covered"),
      sub_coverage_report(
          coverage.cpp_file, coverage.infeasible_locs, "infeasible to cover"
      ),
      sub_coverage_report(
          coverage.cpp_file, coverage.to_cover_locs, "TO BE COVERED"
      ),
  ]
  return "\n".join(report)


def _need_to_cover(loc: ErrorMacroLine) -> bool:
  """Returns whether the loc.macro needs to be covered."""
  return loc.macro in _NEED_TO_COVER_MACROS


def get_per_file_coverage(
    file_to_locs: dict[str, list[ErrorMacroLine]],
    covered_raw_locs: set[SourceLoc],
) -> list[CoverageForFile]:
  """Returns the covered error macro locations for each file."""
  file_to_covered_raw_locs: dict[str, set[int]] = collections.defaultdict(set)
  for raw_loc in covered_raw_locs:
    file_to_covered_raw_locs[raw_loc.path].add(raw_loc.line_number)

  result = []
  for cpp_file, locs in file_to_locs.items():
    covered_locs = []
    covered_raw_lines = file_to_covered_raw_locs.get(cpp_file)
    if covered_raw_lines:
      loc_line_nums = [loc.line_num for loc in locs]
      for line_num in covered_raw_lines:
        # Find the largest line number in locs that is smaller than or equal to
        # the covered line number. This is necessary as the line numbers in locs
        # are for the lines containing the *start* of the error macros, while the
        # line numbers in covered_raw_locs are for the lines containing the
        # *end* (i.e. the closing parenthesis) of the error macros.
        idx = bisect.bisect_right(loc_line_nums, line_num) - 1
        if idx >= 0:
          loc = locs[idx]
          if _need_to_cover(loc):
            covered_locs.append(loc.line_num)

    need_to_cover_locs = [l.line_num for l in locs if _need_to_cover(l)]
    if not need_to_cover_locs:
      continue

    result.append(
        CoverageForFile(
            cpp_file,
            locs=need_to_cover_locs,
            covered_locs=covered_locs,
            infeasible_locs=[
                l.line_num
                for l in locs
                if not l.cov_feasible and _need_to_cover(l)
            ],
        )
    )
  return result


# The baseline coverage data, relative to the repository root.
COVERAGE_DATA_PATH = "tests/error_coverage_data.yaml"


# The order of the per-file statistics in the coverage data file, from the most
# to the least important.
_STATS_ORDER = ("remaining", "total", "infeasible")


class _FlowDict(dict):
  """A dict that PyYAML dumps in the inline (flow) style."""


def _stats_sort_key(key: str) -> tuple[int, str]:
  """Orders the keys by _STATS_ORDER, then any unknown keys alphabetically."""
  if key in _STATS_ORDER:
    return _STATS_ORDER.index(key), ""
  return len(_STATS_ORDER), key


def _represent_flow_dict(
    dumper: yaml.SafeDumper, data: _FlowDict
) -> yaml.MappingNode:
  return dumper.represent_mapping(
      "tag:yaml.org,2002:map",
      sorted(data.items(), key=lambda item: _stats_sort_key(item[0])),
      flow_style=True,
  )


yaml.add_representer(_FlowDict, _represent_flow_dict, Dumper=yaml.SafeDumper)


def load_coverage_data(path: str) -> dict[str, Mapping[str, int]]:
  """Loads a coverage data file.

  Args:
    path: The coverage data file.

  Returns:
    The coverage statistics of each file, keyed by its canonical path.

  Raises:
    FileNotFoundError: If the coverage data file doesn't exist.
    ValueError: If the coverage data file isn't a mapping.
  """
  with open(path) as f:
    data = yaml.safe_load(f) or {}
  if not isinstance(data, dict):
    raise ValueError(f"{path} must be a mapping, got {type(data).__name__}.")
  return data


def dump_coverage_data(data: Mapping[str, Mapping[str, int]]) -> str:
  """Returns the coverage data as YAML, with one line per file, sorted."""
  return yaml.safe_dump(
      {path: _FlowDict(stats) for path, stats in sorted(data.items())},
      # Prevent line breaks - we want one line per entry.
      width=float("inf"),
  )


# The license header of the coverage data files, which the OSS lint requires.
_COVERAGE_DATA_HEADER = """\
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

"""


def format_coverage_data_file(data: Mapping[str, Mapping[str, int]]) -> str:
  """Returns the contents of a coverage data file, with the license header."""
  return _COVERAGE_DATA_HEADER + dump_coverage_data(data)


def update_command(repo_root: str) -> str:
  """Returns the command that prints the new coverage data.

  Args:
    repo_root: The repository root in the runfiles, which tells whether this
      runs in google3 (Blaze) or in OSS (Bazel).
  """
  if repo_root.endswith(repo_paths.G3_REPO_PATH):
    return (
        "blaze test -c opt"
        f" //{repo_paths.G3_REPO_PATH}/tests:error_coverage_test"
        " --test_arg=--print_coverage_data --test_output=all"
    )
  # Like the OSS presubmit, run with the newest supported dependencies on 8 TPU
  # v5e chips.
  return (
      "cp requirements/requirements_3_14_latest.txt"
      " requirements/requirements_3_14.txt && bazel test"
      " --config=ci_tpu_v5_newest_deps"
      ' --run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh"'
      " //tests:error_coverage_test --test_arg=--print_coverage_data"
      " --test_output=all"
  )


def _update_instructions(repo_root: str) -> str:
  """Returns how to update the coverage data file.

  Args:
    repo_root: The repository root in the runfiles, which tells whether this
      runs in google3 (Blaze) or in OSS (Bazel).
  """
  if repo_root.endswith(repo_paths.G3_REPO_PATH):
    # The updater is internal, so it can't be the update_command() for OSS.
    updater = (
        f"//{repo_paths.G3_REPO_PATH}/google/tools:update_error_coverage_data"
    )
    return f"""\
To update {COVERAGE_DATA_PATH}, run:

blaze run -c opt {updater}"""
  return f"""\
To update {COVERAGE_DATA_PATH}, run this command, which prints the new coverage \
data between BEGIN_YAML and END_YAML, and paste the data into \
{COVERAGE_DATA_PATH}, below the license header:

{update_command(repo_root)}"""


@dataclasses.dataclass(frozen=True)
class CoverageCheckResult:
  """The result of checking the coverage against the coverage data file.

  Attributes:
    regressions: Files with more uncovered error macros than the coverage data
      file allows. The error tests need to cover them.
    outdated: Outdated entries of the coverage data file: entries that allow
      more uncovered error macros than there are, so that they could hide a
      future regression, and entries of files that no longer have error macros
      to cover. The coverage data file needs to be updated.
    infeasible_but_covered: `file:line` of the error macros that are marked as
      infeasible to test but actually tested. If their tests were removed,
      nothing would notice, so their ERROR_COV_INFEASIBLE comments need to be
      removed.
    warnings: Other differences, which can't hide a regression.
  """

  regressions: list[str]
  outdated: list[str]
  infeasible_but_covered: list[str]
  warnings: list[str]


def check_coverage(
    expected_coverages: Mapping[str, Mapping[str, int]],
    actual_coverages: Sequence[CoverageForFile],
) -> CoverageCheckResult:
  """Checks the actual coverage against the coverage data file.

  The `remaining` (uncovered and feasible to cover) count of each file must
  match the coverage data file exactly:

  - If a file has more uncovered error macros than its entry allows (or has any
    while it has no entry), it's a coverage regression.
  - If a file has fewer, its entry is outdated: a future regression of up to
    that many macros would go unnoticed.

  Entries of files that no longer have error macros to cover (e.g. deleted
  files) are outdated too. This assumes that every file in the coverage data
  file exists in both google3 and OSS: the internal files under `google/`,
  which Copybara doesn't export, are never in it.

  Error macros marked as ERROR_COV_INFEASIBLE but covered are reported too, as
  their tests could be removed unnoticed.

  As the check compares counts, a change that covers one macro and uncovers
  another in the same file passes.

  Differences in the `total` and `infeasible` counts, and files that are
  missing from the coverage data file but fully covered, are only warnings, as
  they can't hide a regression.

  Args:
    expected_coverages: The coverage data file contents, keyed by canonical
      path.
    actual_coverages: The actual coverage of each file that has error macros to
      cover, from `get_per_file_coverage()`.

  Returns:
    The regressions, outdated entries, infeasible but covered error macros, and
    warnings.
  """
  regressions = []
  outdated = []
  infeasible_but_covered = []
  warnings = []
  for actual_coverage in sorted(actual_coverages, key=lambda c: c.cpp_file):
    file_path = actual_coverage.cpp_file
    actual_remaining = len(actual_coverage.to_cover_locs)
    report = sub_coverage_report(
        file_path, actual_coverage.to_cover_locs, "TO BE COVERED"
    )

    infeasible_but_covered.extend(
        f"{file_path}:{line}"
        for line in actual_coverage.infeasible_and_covered_locs
    )

    if file_path not in expected_coverages:
      if actual_remaining > 0:
        regressions.append(
            f"{file_path} has an error coverage regression.\nThis file or its"
            " error macros may be newly added, and"
            f" {actual_remaining} such macros in it are"
            f" uncovered:\n{report}\nMake sure all {actual_remaining} of them"
            " are covered."
        )
      else:
        warnings.append(
            f"{file_path!r} is missing from {COVERAGE_DATA_PATH}. All its error"
            " macros are covered."
        )
      continue

    expected_file_data = expected_coverages[file_path]
    expected_remaining = expected_file_data.get("remaining", 0)
    if actual_remaining > expected_remaining:
      regressions.append(
          f"""\
{file_path} has an error coverage regression.
Before, {expected_remaining} such macros in the file were uncovered.
With this CL, {actual_remaining} such macros in the file are uncovered:
{report}
Make sure at least {actual_remaining - expected_remaining} of them are covered."""
      )
    elif actual_remaining < expected_remaining:
      outdated.append(
          f"{file_path}: {COVERAGE_DATA_PATH} expects"
          f" {expected_remaining} uncovered error macros, but only"
          f" {actual_remaining} are uncovered now. Lower its `remaining` from"
          f" {expected_remaining} to {actual_remaining}."
      )

    expected_total = expected_file_data.get("total", 0)
    expected_infeasible = expected_file_data.get("infeasible", 0)
    actual_total = len(actual_coverage.locs)
    actual_infeasible = len(actual_coverage.infeasible_locs)
    if (
        actual_total != expected_total
        or actual_infeasible != expected_infeasible
    ):
      warnings.append(
          f"In {file_path!r}: statistics out of sync! Expected"
          f" total={expected_total}, infeasible={expected_infeasible}. Actual"
          f" total={actual_total}, infeasible={actual_infeasible}."
      )

  actual_paths = {coverage.cpp_file for coverage in actual_coverages}
  for file_path in sorted(expected_coverages):
    if file_path not in actual_paths:
      outdated.append(
          f"{file_path}: the file no longer exists or has no error macros to"
          " cover. Delete its entry."
      )

  return CoverageCheckResult(
      regressions=regressions,
      outdated=outdated,
      infeasible_but_covered=infeasible_but_covered,
      warnings=warnings,
  )


def failure_message(result: CoverageCheckResult, repo_root: str) -> str | None:
  """Returns the test failure message for `result`, or None if it passes.

  Args:
    result: The result of `check_coverage()`.
    repo_root: The repository root in the runfiles, which tells whether this
      runs in google3 (Blaze) or in OSS (Bazel).
  """
  sections = []
  if result.regressions:
    joined_regressions = "\n\n".join(result.regressions)
    sections.append(f"""\
TT_ERROR, TT_RET_CHECK, and TT_CHECK_THROW macros need to be covered in \
tests/errors_test.py (preferred), tests/tpu_errors_test.py (if the error \
occurs on TPU only) or tests/distributed/collective_errors_test.py (for \
collectives).

{joined_regressions}

You can ignore regressions caused by other people. However, please avoid \
introducing new regressions. If it's infeasible to test an error macro (e.g. \
because the function never returns an error Status), please add an \
`// ERROR_COV_INFEASIBLE=<why>` comment on the same line as the macro.

If this failure is caused by recent changes by others, you may update the \
coverage data in {COVERAGE_DATA_PATH} to match the reality so that \
error_coverage_test passes.""")
  if result.outdated:
    joined_outdated = "\n".join(f"- {entry}" for entry in result.outdated)
    sections.append(f"""\
{COVERAGE_DATA_PATH} is outdated, so it could hide a future error coverage \
regression. Please update it:

{joined_outdated}""")
  if result.infeasible_but_covered:
    joined_locs = "\n".join(f"- {loc}" for loc in result.infeasible_but_covered)
    sections.append(f"""\
These error macros are marked as infeasible to test, but the error tests cover \
them. If their tests were removed, nothing would notice. Please remove the \
ERROR_COV_INFEASIBLE comment of each, and update {COVERAGE_DATA_PATH}:

{joined_locs}""")
  if not sections:
    return None
  sections.append(_update_instructions(repo_root))
  return "\n\n".join(sections)
