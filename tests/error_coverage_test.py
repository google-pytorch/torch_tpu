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

r"""Checks the coverage of error macros in the torch_tpu codebase.

This test fails if the error tests cover fewer error macros in a C++ file than
the baseline in error_coverage_data.yaml says. google3 and OSS share the
baseline, so the internal C++ files (under `google/`) aren't checked.
"""

import os

from absl import flags
from absl import logging
from absl.testing import absltest
from tests import error_coverage_lib
from tests import repo_paths

_PRINT_COVERAGE_DATA = flags.DEFINE_bool(
    "print_coverage_data",
    False,
    "If true, print the actual coverage data to stdout between BEGIN_YAML and"
    " END_YAML, in the format of error_coverage_data.yaml, instead of checking"
    " it.",
)


def _print_coverage_data(
    error_coverage_data: dict[str, error_coverage_lib.CoverageForFile],
) -> None:
  """Prints the error coverage data to stdout."""

  # Insert prefix and suffix for easier parsing.
  print("BEGIN_YAML", flush=True)
  # The dump already ends with a line break.
  print(
      error_coverage_lib.dump_coverage_data({
          path: coverage.to_dict()
          for path, coverage in error_coverage_data.items()
      }),
      end="",
      flush=True,
  )
  print("END_YAML", flush=True)


class ErrorCoverageTest(
    absltest.TestCase  # ABSLTEST_OK=runs error coverage runner
):
  """Tests that the error macro test coverage is within the threshold.

  Ensures that the error macros within the `torch_tpu` codebase are
  adequately covered by tests. It identifies error macros, runs coverage
  tests, and compares the actual coverage against expected thresholds
  defined in `error_coverage_data.yaml`.
  """

  def test_error_macro_coverage(self):
    """Tests that no C++ file has more uncovered error macros than expected.

    Fails if a file has more uncovered error macros than its baseline. Only
    warns if a file has fewer, or if its total or infeasible counts changed.
    """
    repo_root = repo_paths.get_repo_root_in_runfiles()
    # file_to_locs["csrc/round/round.cc"] = [loc1, loc2, ...]
    file_to_locs = error_coverage_lib.get_file_to_locs(repo_root)
    stdout_str = error_coverage_lib.run_cov_tests(repo_root)
    covered_locs = error_coverage_lib.parse_cov_output(stdout_str)
    per_file_coverage = error_coverage_lib.get_per_file_coverage(
        file_to_locs, covered_locs
    )

    if _PRINT_COVERAGE_DATA.value:
      # We skip the coverage data check when dumping the data. So, there's no
      # need to read the old data.
      expected_coverages = {}
    else:
      expected_coverages = error_coverage_lib.load_coverage_data(
          os.path.join(repo_root, error_coverage_lib.COVERAGE_DATA_PATH)
      )

    actual_coverages = {}
    error_messages = []
    for actual_coverage in per_file_coverage:
      logging.info("Check coverage for %s", actual_coverage.cpp_file)
      # The path relative to the repository root.
      file_path = actual_coverage.cpp_file

      actual_to_be_covered = len(actual_coverage.to_cover_locs)
      actual_coverages[file_path] = actual_coverage

      if _PRINT_COVERAGE_DATA.value:
        # Skip the coverage check if we are dumping the coverage data.
        continue

      report = error_coverage_lib.sub_coverage_report(
          actual_coverage.cpp_file,
          actual_coverage.to_cover_locs,
          "TO BE COVERED",
      )

      if file_path not in expected_coverages and actual_to_be_covered > 0:
        error_messages.append(
            f"{file_path} has an error coverage regression.\nThis"
            " file or its error macros may be newly added, and"
            f" {actual_to_be_covered} such macros in it are"
            f" uncovered:\n{report}\nMake sure all {actual_to_be_covered} of"
            " them are covered."
        )
        continue

      expected_file_data = expected_coverages.get(file_path, {})
      expected_to_be_covered = expected_file_data.get("remaining", 0)
      expected_total = expected_file_data.get("total", 0)
      expected_infeasible = expected_file_data.get("infeasible", 0)

      regression = actual_to_be_covered - expected_to_be_covered
      if regression > 0:
        error_messages.append(f"""\
{file_path} has an error coverage regression.
Before, {expected_to_be_covered} such macros in the file were uncovered.
With this CL, {actual_to_be_covered} such macros in the file are uncovered:
{report}
Make sure at least {regression} of them are covered.""")

      if actual_to_be_covered < expected_to_be_covered:
        logging.warning(
            "In %(file_path)r: expected %(expected)d error macros uncovered,"
            " got %(actual)d. Please reduce the expected uncovered count in"
            " error_coverage_data.yaml.\n%(report)s",
            {
                "file_path": file_path,
                "expected": expected_to_be_covered,
                "actual": actual_to_be_covered,
                "report": error_coverage_lib.coverage_report(actual_coverage),
            },
        )

      # Warn developers if total or infeasible counts drift out of sync.
      actual_total = len(actual_coverage.locs)
      actual_infeasible = len(actual_coverage.infeasible_locs)
      if (
          actual_total != expected_total
          or actual_infeasible != expected_infeasible
      ):
        logging.warning(
            "In %r: statistics out of sync! Expected total=%d, "
            "infeasible=%d. Actual total=%d, infeasible=%d. "
            "Please update the baseline:\n%s",
            file_path,
            expected_total,
            expected_infeasible,
            actual_total,
            actual_infeasible,
            error_coverage_lib.update_command(repo_root),
        )

      if actual_coverage.infeasible_and_covered_locs:
        logging.warning(
            "In %r: line %s are marked as infeasible to test but actually "
            "tested. Please remove the ERROR_COV_INFEASIBLE comment.",
            file_path,
            actual_coverage.infeasible_and_covered_locs,
        )

    if _PRINT_COVERAGE_DATA.value:
      _print_coverage_data(actual_coverages)
    elif error_messages:
      joined_message = "\n\n".join(error_messages)
      raise ValueError(f"""\
TT_ERROR, TT_RET_CHECK, and TT_CHECK_THROW macros need to be covered in \
tests/errors_test.py (preferred), tests/tpu_errors_test.py (if the error \
occurs on TPU only) or tests/distributed/collective_errors_test.py (for \
collectives).

{joined_message}

You can ignore regressions caused by other people. However, please avoid \
introducing new regressions. If it's infeasible to test an error macro (e.g. \
because the function never returns an error Status), please add an \
`// ERROR_COV_INFEASIBLE=<why>` comment on the same line as the macro.

If this failure is caused by recent changes by others, you may update the \
coverage data in tests/error_coverage_data.yaml to match the reality so that \
error_coverage_test passes. This command prints the new coverage data between \
BEGIN_YAML and END_YAML; paste it into tests/error_coverage_data.yaml, below \
the license header:

{error_coverage_lib.update_command(repo_root)}
""")


if __name__ == "__main__":
  absltest.main()
