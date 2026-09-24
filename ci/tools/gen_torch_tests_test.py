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

"""Unit tests for ci/tools/gen_torch_tests.py."""

import os
import pathlib
import tempfile
import textwrap
from typing import Final
import unittest
from unittest import mock

from ci.tools import gen_torch_tests

_SAMPLE_BUILD_FILE_TEMPLATE: Final[str] = textwrap.dedent(f"""\
    # Copyright 2026 Google LLC
    load(
        "//torch_tpu/build_files:build_files:build_defs.bzl",
        "torch_tpu_oss_torch_test",
        "torch_tpu_package_end",
    )

    pytype_strict_binary(
        name = "torch_test_runner",
        srcs = ["torch_test_runner.py"],
    )

    {gen_torch_tests.BEGIN_MARKER}
    __text__
    {gen_torch_tests.END_MARKER}

    torch_tpu_package_end()
    """)


class GenTorchTestsTest(unittest.TestCase):  # UNITTEST_OK=testing tools
  """Unit tests for PyTorch test target generator."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.test_root = pathlib.Path(self.temp_dir.name)

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def _create_mock_pytorch_tree(self, root: pathlib.Path) -> None:
    submodule_dir = root / "dynamo"
    submodule_dir.mkdir(parents=True, exist_ok=True)
    (submodule_dir / "test_debug_utils.py").write_text(
        "instantiate_device_type_tests(TestDebugUtils, globals())\n"
    )
    (root / "test_ops.py").write_text(
        "common_device_type.instantiate_device_type_tests(TestOps, globals())\n"
    )
    (root / "test_cuda.py").write_text("class TestCuda: pass\n")

  def test_discover_device_generic_test_modules(self):
    pytorch_dir = self.test_root / "test"
    pytorch_dir.mkdir(parents=True, exist_ok=True)
    self._create_mock_pytorch_tree(pytorch_dir)

    discovered = gen_torch_tests.discover_device_generic_test_modules(
        pytorch_dir
    )

    self.assertEqual(
        discovered,
        frozenset({"dynamo/test_debug_utils", "test_ops"}),
    )

  def test_torch_test_target(self):
    target_def = gen_torch_tests.torch_test_target("dynamo/test_debug_utils")
    expected = textwrap.dedent("""\
        torch_tpu_oss_torch_test(
            name = "dynamo_debug_utils_test",
            torch_test_module = "dynamo/test_debug_utils",
        )""")
    self.assertEqual(target_def, expected)

  def test_torch_test_target_disabled_module(self):
    target_def = gen_torch_tests.torch_test_target("test_binary_ufuncs")
    expected = textwrap.dedent("""\
        torch_tpu_oss_torch_test(
            name = "binary_ufuncs_test",
            notap = "b/563402379",  # NOTAP_OK=torch tests
            torch_test_module = "test_binary_ufuncs",
        )""")
    self.assertEqual(target_def, expected)

  def test_build_file_path_default_with_bazel_workspace_directory(self):
    workspace_dir = self.test_root / "workspace"
    expected_build = workspace_dir / "tests" / "torch" / "BUILD"

    with mock.patch.dict(
        os.environ, {"BUILD_WORKSPACE_DIRECTORY": str(workspace_dir)}
    ):
      self.assertEqual(gen_torch_tests.build_file_path(None), expected_build)

  def test_build_file_path_default_fallback_to_cwd(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      expected_build = pathlib.Path.cwd() / "tests" / "torch" / "BUILD"
      self.assertEqual(gen_torch_tests.build_file_path(None), expected_build)

  def test_custom_build_file_path(self):
    build_file = self.test_root / "__torch__" / "BUILD"
    self.assertEqual(
        gen_torch_tests.build_file_path(str(build_file)),
        build_file.resolve(),
    )
    self.assertEqual(
        gen_torch_tests.build_file_path(build_file),
        build_file.resolve(),
    )

  def test_write_to_build_file_success(self):
    build_file = self.test_root / "BUILD"
    build_file.write_text(_SAMPLE_BUILD_FILE_TEMPLATE)

    new_targets = "torch_tpu_oss_torch_test(name = 'ops_test')"
    gen_torch_tests.write_to_build_file(build_file, new_targets)

    content = build_file.read_text()
    self.assertIn(gen_torch_tests.BEGIN_MARKER, content)
    self.assertIn(gen_torch_tests.END_MARKER, content)
    self.assertIn(new_targets, content)
    self.assertNotIn("__text__", content)
    self.assertIn("torch_test_runner", content)
    self.assertIn("torch_tpu_package_end()", content)

  def test_write_to_build_file_without_markers_exits(self):
    build_file = self.test_root / "BUILD"
    build_file.write_text("__text__")

    with self.assertRaises(SystemExit) as cm:
      gen_torch_tests.write_to_build_file(build_file, "targets")
    self.assertIn("No markers are found", str(cm.exception))


if __name__ == "__main__":
  unittest.main()
