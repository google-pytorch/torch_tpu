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

"""Runs a PyTorch test module with TorchTPU.

This script is a test runner that exercises TorchTPU against test modules
defined in https://github.com/pytorch/pytorch/tree/main/test. By utilizing the
native PrivateUse1TestBase APIs, it instantiates, filters and executes TPU
variants of device-generic tests for a given test module.

The test runner is intended to be invoked via a custom Bazel rule, which sets up
required dependencies, environment variables etc. See also tests/torch/BUILD.

It takes no command-line arguments and requires following environment variables:
  1. TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR: The path to the test/ directory of a
     PyTorch checkout supplied via --test_env at runtime.
  2. TORCH_TPU_INTERNAL_TORCH_TEST_MODULE: The name of the test module to run
     set by the Bazel rule.

The test runner exits with 0 on success, and non-zero on failure.
"""

from collections.abc import Sequence
import os
import pathlib
import runpy
import sys
from typing import Any, Final
import unittest

from absl import app
from absl.testing import absltest

import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from torch.testing._internal import common_device_type  # pylint: disable=g-bad-import-order

_TORCH_TEST_DIR_ENV_VAR: Final = "TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR"
_TORCH_TEST_MODULE_ENV_VAR: Final = "TORCH_TPU_INTERNAL_TORCH_TEST_MODULE"


def _set_private_use1_test_configs() -> None:
  """Declares TorchTPU test configuration on PrivateUse1TestBase."""
  base = common_device_type.PrivateUse1TestBase
  if base not in common_device_type.device_type_test_bases:
    raise RuntimeError(
        "PrivateUse1TestBase is not a registered device test base. "
        "The TPU backend was unavailable when "
        "`torch.testing._internal.common_device_type` was imported.\n"
        "Ensure `import torch_tpu` precedes any imports from "
        "`torch.testing._internal` in the test runner."
    )

  # Resolve the base's device type ("tpu") up front so every generated class
  # has "TPU" suffix.
  base.setUpClass()
  # TODO(b/523293280): extend test configs with TPU-specific overrides.
  base.set_test_configs()


def _filter_device_generic_tests() -> None:
  """Restricts unittest's loader to device-generic test classes only."""
  original = unittest.TestLoader.getTestCaseNames

  def device_generic_tests(
      self, test_case_cls: type[unittest.TestCase]  # UNITTEST_OK=torch testing
  ) -> list[str]:
    """Returns test method names for device-generic test classes only."""
    # Keep classes materialized by `instantiate_device_type_tests` only,
    # i.e. `DeviceTypeTestBase` subclasses.
    if not (
        isinstance(test_case_cls, type)
        and issubclass(test_case_cls, common_device_type.DeviceTypeTestBase)
    ):
      return []
    return original(self, test_case_cls)

  unittest.TestLoader.getTestCaseNames = device_generic_tests


def _parse_test_env_vars() -> tuple[pathlib.Path, pathlib.Path]:
  """Parses required test directory and module from environment variables."""
  test_module = os.environ.get(_TORCH_TEST_MODULE_ENV_VAR)
  if not test_module:
    raise EnvironmentError(f"{_TORCH_TEST_MODULE_ENV_VAR} is not set.")

  test_dir = os.environ.get(_TORCH_TEST_DIR_ENV_VAR)
  if not test_dir:
    raise EnvironmentError(
        f"{_TORCH_TEST_DIR_ENV_VAR} is not set. Please supply it via "
        f"--test_env={_TORCH_TEST_DIR_ENV_VAR}=/path/to/pytorch/test"
    )

  test_dir_path = pathlib.Path(test_dir)
  test_module_path = test_dir_path / f"{test_module}.py"
  if not test_module_path.is_file():
    raise FileNotFoundError(
        f"{test_module_path} cannot be found. Please set"
        f" {_TORCH_TEST_DIR_ENV_VAR} to the test/ directory of a"
        " https://github.com/pytorch/pytorch checkout."
    )

  return test_dir_path, test_module_path


def _setup_test_args(test_dir: pathlib.Path, test_module: pathlib.Path) -> None:
  """Rewrites sys.path and sys.argv the way upstream test modules expect."""
  # Mirror what `python test/.../test_module.py` does to the interpreter state.
  #
  # The CPython launcher prepends the script's own directory, while
  # `runpy.run_path` leaves `sys.path` untouched. Upstream test modules rely on
  # the launcher behavior to import their sibling helpers, and nested modules
  # additionally import helpers relative to the test/ root. Inserting test/
  # first and the module's own directory second keeps the latter ahead, e.g.
  #   test/test_torch.py      -> ["<checkout>/test", ...]
  #   test/nn/test_pooling.py -> ["<checkout>/test/nn", "<checkout>/test", ...]
  sys.path.insert(0, str(test_dir))
  if test_module.parent != test_dir:
    sys.path.insert(0, str(test_module.parent))
  # Under Bazel, sys.argv[0] points to the runner binary, not a .py file.
  # Upstream records argv[0] as the running script, asserts every loaded
  # TestCase is defined in it, rebuilds the unittest argv from it, and names
  # report directories after it. Thus, point it at the test module itself, e.g.
  #   blaze-bin/tests/torch/test_torch -> <checkout>/test/test_torch.py
  sys.argv[0] = str(test_module)


def _use_absltest_main() -> None:
  """Routes test execution to absltest with better Bazel testing protocol support."""

  def absltest_main(*args: Any, **kwargs: Any) -> None:
    """Drop-in replacement for unittest.main that delegates to absltest."""
    # Always run verbose to print progress output.
    kwargs.setdefault("verbosity", 2)
    absltest.main(*args, **kwargs)

  unittest.main = absltest_main


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Early exit on older PyTorch versions where PrivateUse1TestBase APIs are not
  # available.
  base = getattr(common_device_type, "PrivateUse1TestBase", None)
  if not hasattr(base, "set_test_configs"):
    sys.exit(
        "ERROR: PrivateUse1TestBase.set_test_configs API is not available in"
        f" torch version {torch.__version__}.\n"
        "ERROR: Please use v2.13+ to run device-generic tests with TorchTPU."
    )

  test_dir_path, test_module_path = _parse_test_env_vars()
  _setup_test_args(test_dir_path, test_module_path)

  # Read by torch.testing._internal.common_device_type to
  # restrict instantiate_device_type_tests to the TPU backend.
  os.environ["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "tpu"
  _set_private_use1_test_configs()
  _filter_device_generic_tests()

  # Execute the test module as a program via absltest. Each test module has its
  # own entry point invoking `run_tests()` from its own namespace.
  _use_absltest_main()
  runpy.run_path(str(test_module_path), run_name="__main__")


if __name__ == "__main__":
  app.run(main)
