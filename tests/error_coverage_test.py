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

This test fails if error_coverage_data.yaml doesn't match the coverage of the
error macros in the C++ files, in a way that matters:

- A file has more uncovered error macros than the baseline allows (a coverage
  regression).
- A file has fewer uncovered error macros than the baseline allows, or the
  baseline has an entry for a file that no longer has error macros to cover. The
  outdated baseline could hide a future regression.
- An error macro marked as ERROR_COV_INFEASIBLE is covered.

google3 and OSS share the baseline, so the internal C++ files (under `google/`)
aren't checked.
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
    per_file_coverage: list[error_coverage_lib.CoverageForFile],
) -> None:
  """Prints the error coverage data to stdout."""

  # Insert prefix and suffix for easier parsing.
  print("BEGIN_YAML", flush=True)
  # The dump already ends with a line break.
  print(
      error_coverage_lib.dump_coverage_data({
          coverage.cpp_file: coverage.to_dict()
          for coverage in per_file_coverage
      }),
      end="",
      flush=True,
  )
  print("END_YAML", flush=True)


class ErrorCoverageTest(
    absltest.TestCase  # ABSLTEST_OK=runs error coverage runner
):
  """Tests that the error macro coverage matches error_coverage_data.yaml."""

  def test_error_macro_coverage(self):
    """Tests the coverage of the error macros against the baseline.

    See `error_coverage_lib.check_coverage()` for what fails the test.
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
      # Skip the check when dumping the data.
      _print_coverage_data(per_file_coverage)
      return

    expected_coverages = error_coverage_lib.load_coverage_data(
        os.path.join(repo_root, error_coverage_lib.COVERAGE_DATA_PATH)
    )
    result = error_coverage_lib.check_coverage(
        expected_coverages, per_file_coverage
    )
    for warning in result.warnings:
      logging.warning("%s", warning)
    message = error_coverage_lib.failure_message(result, repo_root)
    if message is not None:
      raise ValueError(message)


if __name__ == "__main__":
  absltest.main()
