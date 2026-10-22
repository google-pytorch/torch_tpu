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

"""Unit tests for error_coverage_lib, which don't need a TPU."""

import os
from unittest import mock

from absl.testing import absltest
from tests import error_coverage_lib


class GetErrorMacroLocsTest(absltest.TestCase):  # ABSLTEST_OK=No RNG used.

  def test_skips_test_files(self):
    self.assertEqual(error_coverage_lib.get_error_macro_locs("foo_test.cc"), [])
    self.assertEqual(
        error_coverage_lib.get_error_macro_locs("foo_test_helper.h"), []
    )

  def test_finds_macros_outside_comments_and_preprocessor_directives(self):
    cpp_file = self.create_tempfile(
        "sample.cc",
        content="""\
// TT_ERROR(error::kInternal) in single-line comment
#define MY_MACRO(x) \\
  if (x) \\
    return TT_ERROR(error::kInternal)

absl::Status Foo(int x) {
  if (x < 0) {
    return TT_ERROR(error::kInvalidArgument);
  }
  static_assert(
      true,
      "  return TT_ERROR(error::kInternal) << \\"message\\";");
  TT_RET_CHECK(x != 0, error::kInternal);  // ERROR_COV_INFEASIBLE=reason
  return absl::OkStatus();
}
""",
    )
    locs = error_coverage_lib.get_error_macro_locs(cpp_file.full_path)
    self.assertEqual(
        [(loc.line_num, loc.cov_feasible, loc.macro) for loc in locs],
        [
            (8, True, "TT_ERROR"),
            (13, False, "TT_RET_CHECK"),
        ],
    )


class CanonicalizePathTest(absltest.TestCase):  # ABSLTEST_OK=No RNG used.

  def test_strips_google3_repo_path(self):
    self.assertEqual(
        error_coverage_lib.canonicalize_path(
            "third_party/py/torch_tpu/csrc/eager/foo.cc"
        ),
        "csrc/eager/foo.cc",
    )

  def test_maps_oss_overlay_to_its_exported_path(self):
    self.assertEqual(
        error_coverage_lib.canonicalize_path(
            "oss/torch_tpu/csrc/common/environment.cc"
        ),
        "csrc/common/environment.cc",
    )
    self.assertEqual(
        error_coverage_lib.canonicalize_path(
            "third_party/py/torch_tpu/oss/torch_tpu/csrc/common/environment.cc"
        ),
        "csrc/common/environment.cc",
    )

  def test_strips_leading_current_directory(self):
    self.assertEqual(
        error_coverage_lib.canonicalize_path("./csrc/common/aten_utils.h"),
        "csrc/common/aten_utils.h",
    )

  def test_keeps_oss_overlay_paths_outside_csrc(self):
    # Copybara moves only oss/torch_tpu/csrc/ to the repository root.
    self.assertEqual(
        error_coverage_lib.canonicalize_path("oss/torch_tpu/_internal/foo.cc"),
        "oss/torch_tpu/_internal/foo.cc",
    )

  def test_keeps_canonical_paths(self):
    self.assertEqual(
        error_coverage_lib.canonicalize_path("csrc/eager/foo.cc"),
        "csrc/eager/foo.cc",
    )
    self.assertEqual(
        error_coverage_lib.canonicalize_path("src/torch_tpu/foo.cc"),
        "src/torch_tpu/foo.cc",
    )


class ParseCovOutputTest(absltest.TestCase):  # ABSLTEST_OK=No RNG used.

  def test_canonicalizes_google3_and_oss_paths(self):
    output = """\
Covered third_party/py/torch_tpu/csrc/eager/foo.cc:12
some other line
Covered csrc/eager/bar.cc:34
Covered third_party/py/torch_tpu/oss/torch_tpu/csrc/common/environment.cc:5
Covered csrc/eager/bar.cc:34
"""
    self.assertEqual(
        error_coverage_lib.parse_cov_output(output),
        {
            error_coverage_lib.SourceLoc("csrc/eager/foo.cc", 12),
            error_coverage_lib.SourceLoc("csrc/eager/bar.cc", 34),
            error_coverage_lib.SourceLoc("csrc/common/environment.cc", 5),
        },
    )


class GetFileToLocsTest(absltest.TestCase):  # ABSLTEST_OK=No RNG used.

  def _create_file(self, repo_root: str, path: str, content: str) -> None:
    full_path = os.path.join(repo_root, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
      f.write(content)

  def test_scans_repo_with_canonical_keys_and_skips_non_repo_dirs(self):
    repo_root = self.create_tempdir().full_path
    macro = "absl::Status F() { return TT_ERROR(error::kInternal); }\n"
    self._create_file(repo_root, "csrc/eager/foo.cc", macro)
    self._create_file(repo_root, "csrc/eager/foo.h", "// No macros.\n")
    self._create_file(repo_root, "csrc/eager/foo_test.cc", macro)
    self._create_file(
        repo_root, "oss/torch_tpu/csrc/common/environment.cc", macro
    )
    # Directories that aren't part of the repository.
    self._create_file(
        repo_root, "tests/_foo.venv/lib/site-packages/x/y.h", macro
    )
    self._create_file(repo_root, "site-packages/z.h", macro)
    self._create_file(repo_root, "bazel-out/k8/bin/csrc/gen.cc", macro)
    self._create_file(repo_root, "blaze-bin/csrc/gen.cc", macro)
    self._create_file(repo_root, ".hidden/foo.cc", macro)

    file_to_locs = error_coverage_lib.get_file_to_locs(repo_root)

    self.assertCountEqual(
        file_to_locs.keys(),
        ["csrc/eager/foo.cc", "csrc/common/environment.cc"],
    )
    self.assertEqual(
        [loc.line_num for loc in file_to_locs["csrc/eager/foo.cc"]], [1]
    )

  def test_rejects_paths_with_the_same_canonical_form(self):
    repo_root = self.create_tempdir().full_path
    macro = "absl::Status F() { return TT_ERROR(error::kInternal); }\n"
    self._create_file(repo_root, "csrc/common/environment.cc", macro)
    self._create_file(
        repo_root, "oss/torch_tpu/csrc/common/environment.cc", macro
    )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Tests the tool's own errors.
        ValueError, "csrc/common/environment.cc"
    ):
      error_coverage_lib.get_file_to_locs(repo_root)


class GetPerFileCoverageTest(absltest.TestCase):  # ABSLTEST_OK=No RNG used.

  def test_maps_covered_lines_to_macros(self):
    file_to_locs = {
        "csrc/foo.cc": [
            error_coverage_lib.ErrorMacroLine(10, True, "TT_ERROR"),
            error_coverage_lib.ErrorMacroLine(20, False, "TT_RET_CHECK"),
            error_coverage_lib.ErrorMacroLine(30, True, "TT_ERROR"),
            error_coverage_lib.ErrorMacroLine(40, True, "TT_RETURN_IF_ERROR"),
        ],
    }
    # A multi-line macro is reported at its last line.
    covered = {error_coverage_lib.SourceLoc("csrc/foo.cc", 11)}

    [coverage] = error_coverage_lib.get_per_file_coverage(file_to_locs, covered)

    self.assertEqual(coverage.cpp_file, "csrc/foo.cc")
    self.assertEqual(
        coverage.to_dict(), {"infeasible": 1, "remaining": 1, "total": 3}
    )


class RunCovTestsTest(absltest.TestCase):  # ABSLTEST_OK=No RNG used.

  def test_runs_each_binary_with_cov_env_and_concatenates_stdout(self):
    repo_root = self.create_tempdir().full_path
    for i, binary in enumerate(error_coverage_lib.COV_TEST_BINARIES):
      path = os.path.join(repo_root, binary)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, "w") as f:
        f.write(f"""\
#!/bin/sh
echo "Covered csrc/f{i}.cc:$TORCH_TPU_ENABLE_DEBUG_CHECKS $1"
""")
      os.chmod(path, 0o755)

    output = error_coverage_lib.run_cov_tests(repo_root)

    self.assertEqual(
        output,
        "".join(
            f"Covered csrc/f{i}.cc:1 --test_mode=cov\n"
            for i in range(len(error_coverage_lib.COV_TEST_BINARIES))
        ),
    )

  def test_does_not_pass_the_test_runner_env_to_binaries(self):
    repo_root = self.create_tempdir().full_path
    for binary in error_coverage_lib.COV_TEST_BINARIES:
      path = os.path.join(repo_root, binary)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, "w") as f:
        f.write("""\
#!/bin/sh
echo "filter=${TESTBRIDGE_TEST_ONLY-unset}" \\
    "shards=${TEST_TOTAL_SHARDS-unset}" \\
    "index=${TEST_SHARD_INDEX-unset}" \\
    "status=${TEST_SHARD_STATUS_FILE-unset}" \\
    "xml=${XML_OUTPUT_FILE-unset}" \\
    "premature_exit=${TEST_PREMATURE_EXIT_FILE-unset}" \\
    "fail_fast=${TESTBRIDGE_TEST_RUNNER_FAIL_FAST-unset}"
""")
      os.chmod(path, 0o755)

    with mock.patch.dict(
        os.environ,
        {
            "TESTBRIDGE_TEST_ONLY": "ErrorCoverageTest.test_foo",
            "TEST_TOTAL_SHARDS": "2",
            "TEST_SHARD_INDEX": "1",
            "TEST_SHARD_STATUS_FILE": "/tmp/status",
            "XML_OUTPUT_FILE": "/tmp/test.xml",
            "TEST_PREMATURE_EXIT_FILE": "/tmp/premature_exit",
            "TESTBRIDGE_TEST_RUNNER_FAIL_FAST": "1",
        },
    ):
      output = error_coverage_lib.run_cov_tests(repo_root)

    self.assertEqual(
        output,
        "filter=unset shards=unset index=unset status=unset xml=unset"
        " premature_exit=unset fail_fast=unset\n"
        * len(error_coverage_lib.COV_TEST_BINARIES),
    )

  def test_raises_if_a_binary_fails(self):
    repo_root = self.create_tempdir().full_path
    for binary in error_coverage_lib.COV_TEST_BINARIES:
      path = os.path.join(repo_root, binary)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, "w") as f:
        f.write("#!/bin/sh\necho oops >&2\nexit 3\n")
      os.chmod(path, 0o755)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Tests the tool's own errors.
        RuntimeError, "oops"
    ):
      error_coverage_lib.run_cov_tests(repo_root)

  def test_raises_if_a_binary_is_missing(self):
    repo_root = self.create_tempdir().full_path

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Tests the tool's own errors.
        FileNotFoundError, "errors_test_cov"
    ):
      error_coverage_lib.run_cov_tests(repo_root)


if __name__ == "__main__":
  absltest.main()
