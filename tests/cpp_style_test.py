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

import pathlib
from absl.testing import absltest
from tests import cpp_style_lib as style


class CppStyleTest(absltest.TestCase):  # ABSLTEST_OK=avoid heavy dependencies

  def test_find_test_data(self):
    cpp_code = style.CppCode("tests")

    self.assertIsNotNone(cpp_code.find_file("cpp_style_test_data.cc"))

  def test_namespace(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")

    output = list(file.get_namespace())
    self.assertIn(
        (["torch_tpu"], "void Foo() { torch_tpu::TorchTpuClass(); }"), output
    )

  def test_torch_tpu_path(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")
    self.assertIsNotNone(file, "Could not find cpp_style_test_data.cc")
    self.assertEqual(file.torch_tpu_path, "tests/cpp_style_test_data.cc")

  def test_nested_namespace(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")

    output = list(file.get_namespace())
    self.assertIn(
        (
            ["outer", "torch_tpu"],
            "void Foo3() { ::torch_tpu::TorchTpuClass(); }",
        ),
        output,
    )

  def test_unnamed_namespace(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")

    output = list(file.get_namespace())
    self.assertIn(([""], "void Foo4() { torch_tpu::TorchTpuClass(); }"), output)

  def test_multi_segment_namespace(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")

    output = list(file.get_namespace())
    self.assertIn(
        (["foo", "bar"], "void Foo5() { torch_tpu::TorchTpuClass(); }"), output
    )

  def test_unapproved_nested_namespace_in_test_data(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")
    self.assertIsNotNone(file, "Could not find cpp_style_test_data.cc")

    errors = style.check_unapproved_namespace(file)

    expected_error = (
        "}  // namespace torch_tpu::bar: Only approved named namespaces allowed"
        " under torch_tpu: ['error', 'internal'], got ['bar']"
    )
    expected_error_2 = (
        "void Baz2() { torch_tpu::TorchTpuClass(); }: Only approved named"
        " namespaces allowed under torch_tpu: ['error', 'internal'], got ['',"
        " 'internal']"
    )
    self.assertIn(expected_error, errors)
    self.assertIn(expected_error_2, errors)
    self.assertFalse(any("Baz3()" in e for e in errors))

  def test_expected_header_guard(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    file1 = style.CppFile(temp_dir / "csrc" / "common" / "env_vars.h", temp_dir)
    self.assertEqual(
        file1.expected_header_guard, "TORCH_TPU_CSRC_COMMON_ENV_VARS_H_"
    )

    file2 = style.CppFile(temp_dir / "ops" / "foo-bar" / "baz.h", temp_dir)
    self.assertEqual(
        file2.expected_header_guard, "TORCH_TPU_OPS_FOO_BAR_BAZ_H_"
    )

    # OSS layout with torch_tpu prefix:
    file3 = style.CppFile(
        temp_dir / "torch_tpu" / "csrc" / "common" / "env_vars.h", temp_dir
    )
    self.assertEqual(
        file3.expected_header_guard, "TORCH_TPU_CSRC_COMMON_ENV_VARS_H_"
    )

  def test_is_repo_file(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    repo_file = style.CppFile(
        temp_dir / "csrc" / "common" / "env_vars.h", temp_dir
    )
    self.assertTrue(repo_file.is_repo_file)

    venv_file = style.CppFile(
        temp_dir
        / "tests"
        / "_header_guard_lint_test.venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "lxml"
        / "etree.h",
        temp_dir,
    )
    self.assertFalse(venv_file.is_repo_file)

    hidden_file = style.CppFile(temp_dir / ".git" / "foo.h", temp_dir)
    self.assertFalse(hidden_file.is_repo_file)

  def test_is_header_and_test_data(self):
    cpp_code = style.CppCode("tests")
    file = cpp_code.find_file("cpp_style_test_data.cc")
    self.assertIsNotNone(file)
    self.assertFalse(file.is_header)
    self.assertTrue(file.is_test_data)

  def test_headers_and_exclude_test_data(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    (temp_dir / "a.h").write_text("// header")
    (temp_dir / "b.cc").write_text("// source")
    (temp_dir / "c.cpp").write_text("// source cpp")
    (temp_dir / "cpp_style_test_data.h").write_text("// test data header")

    venv_dir = (
        temp_dir
        / "tests"
        / "_header_guard_lint_test.venv"
        / "lib"
        / "site-packages"
    )
    venv_dir.mkdir(parents=True, exist_ok=True)
    (venv_dir / "external.h").write_text("// external header")

    code = style.CppCode("", repo_root=temp_dir)
    all_files = [f.path.name for f in code.files(exclude_non_repo=False)]
    self.assertCountEqual(
        all_files,
        ["a.h", "b.cc", "c.cpp", "cpp_style_test_data.h", "external.h"],
    )

    repo_files = [f.path.name for f in code.files()]
    self.assertCountEqual(
        repo_files, ["a.h", "b.cc", "c.cpp", "cpp_style_test_data.h"]
    )

    filtered_files = [f.path.name for f in code.files(exclude_test_data=True)]
    self.assertCountEqual(filtered_files, ["a.h", "b.cc", "c.cpp"])

    all_headers = [f.path.name for f in code.headers(exclude_non_repo=False)]
    self.assertCountEqual(
        all_headers, ["a.h", "cpp_style_test_data.h", "external.h"]
    )

    repo_headers = [f.path.name for f in code.headers()]
    self.assertCountEqual(repo_headers, ["a.h", "cpp_style_test_data.h"])

    filtered_headers = [
        f.path.name for f in code.headers(exclude_test_data=True)
    ]
    self.assertCountEqual(filtered_headers, ["a.h"])

  def test_check_header_guard_valid(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    header = temp_dir / "csrc" / "test.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        "/* Copyright */\n"
        "\n"
        "#ifndef TORCH_TPU_CSRC_TEST_H_\n"
        "#define TORCH_TPU_CSRC_TEST_H_\n"
        "\n"
        "namespace torch_tpu {}\n"
        "\n"
        "#endif  // TORCH_TPU_CSRC_TEST_H_\n"
    )
    cpp_file = style.CppFile(header, repo_root=temp_dir)
    errors = style.check_header_guard(cpp_file)
    self.assertEmpty(errors)

  def test_check_header_guard_missing_ifndef(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    header = temp_dir / "csrc" / "test.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("namespace torch_tpu {}\n")
    cpp_file = style.CppFile(header, repo_root=temp_dir)
    errors = style.check_header_guard(cpp_file)
    self.assertLen(errors, 1)
    self.assertIn("Missing header guard #ifndef", errors[0])

  def test_check_header_guard_mismatched_ifndef(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    header = temp_dir / "csrc" / "test.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        "#ifndef WRONG_GUARD_H_\n"
        "#define WRONG_GUARD_H_\n"
        "#endif  // WRONG_GUARD_H_\n"
    )
    cpp_file = style.CppFile(header, repo_root=temp_dir)
    errors = style.check_header_guard(cpp_file)
    self.assertTrue(
        any("Expected '#ifndef TORCH_TPU_CSRC_TEST_H_'" in e for e in errors)
    )
    self.assertTrue(
        any("Expected '#define TORCH_TPU_CSRC_TEST_H_'" in e for e in errors)
    )
    self.assertTrue(
        any("Expected '#endif  // TORCH_TPU_CSRC_TEST_H_'" in e for e in errors)
    )

  def test_check_header_guard_missing_define(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    header = temp_dir / "csrc" / "test.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        "#ifndef TORCH_TPU_CSRC_TEST_H_\n"
        "namespace torch_tpu {}\n"
        "#endif  // TORCH_TPU_CSRC_TEST_H_\n"
    )
    cpp_file = style.CppFile(header, repo_root=temp_dir)
    errors = style.check_header_guard(cpp_file)
    self.assertLen(errors, 1)
    self.assertIn("Missing header guard #define", errors[0])

  def test_check_header_guard_missing_endif(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    header = temp_dir / "csrc" / "test.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        "#ifndef TORCH_TPU_CSRC_TEST_H_\n"
        "#define TORCH_TPU_CSRC_TEST_H_\n"
        "namespace torch_tpu {}\n"
    )
    cpp_file = style.CppFile(header, repo_root=temp_dir)
    errors = style.check_header_guard(cpp_file)
    self.assertLen(errors, 1)
    self.assertIn("Missing closing #endif", errors[0])

  def test_check_header_guard_non_header(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    source = temp_dir / "csrc" / "test.cc"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("namespace torch_tpu {}\n")
    cpp_file = style.CppFile(source, repo_root=temp_dir)
    errors = style.check_header_guard(cpp_file)
    self.assertEmpty(errors)


if __name__ == "__main__":
  absltest.main()
