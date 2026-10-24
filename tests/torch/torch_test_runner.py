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
     PyTorch source tree, set by the Bazel rule or supplied via --test_env at
     runtime. Relative paths are resolved against the initial working directory.
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
from torch.testing._internal import common_utils
from torch_tpu._internal.device import _device_ops_backend
from torch_tpu._internal.utils import annotations
from tests.torch import torch_test_configs

_TORCH_TEST_DIR_ENV_VAR: Final = "TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR"
_TORCH_TEST_MODULE_ENV_VAR: Final = "TORCH_TPU_INTERNAL_TORCH_TEST_MODULE"

# Intentionally keep it close to the latest PyTorch version to leverage
# newly introduced accelerator testing APIs.
_MINIMAL_TORCH_VERSION: Final = "2.14"


def _set_private_use1_test_configs(test_module: str) -> None:
  """Declares TorchTPU configuration on PrivateUse1TestBase per test module."""
  base = common_device_type.PrivateUse1TestBase
  if base not in common_device_type.device_type_test_bases:
    raise RuntimeError(
        "PrivateUse1TestBase is not a registered device test base. "
        "The TPU backend was unavailable when "
        "`torch.testing._internal.common_device_type` was imported.\n"
        "Ensure `import torch_tpu` precedes any imports from "
        "`torch.testing._internal` in the test runner."
    )

  # Initialize the TPU runtime before any test is instantiated.
  # `instantiate_device_type_tests` calls `get_all_devices()` for
  # `@deviceCountAtLeast` tests at import time, i.e. via `device_count()`. Before
  # initialization, `device_count()` counts physical TPU PCI devices, which can
  # exceed the devices addressable by this process (b/577219302). Initializing
  # first makes it return the addressable device count.
  _device_ops_backend._get_device_count()  # pylint: disable=protected-access

  # Resolve the base's device type ("tpu") up front so every generated class
  # has "TPU" suffix.
  base.setUpClass()
  # TODO(b/523293280): extend test configs with TPU-specific overrides.
  base.set_test_configs(
      test_exclusions=torch_test_configs.TEST_EXCLUSIONS.get(test_module)
  )


def _filter_device_generic_tests() -> None:
  """Restricts unittest's loader to device-generic test classes for TPU."""
  original = unittest.TestLoader.getTestCaseNames

  def device_generic_tests(
      self: unittest.TestLoader,
      testCaseClass: type[unittest.TestCase],  # UNITTEST_OK=torch testing  # pylint: disable=invalid-name
  ) -> Sequence[str]:
    """Returns test method names for device-generic test classes only."""
    # Keep classes materialized by `instantiate_device_type_tests` only,
    # i.e. `DeviceTypeTestBase` subclasses.
    if not (
        isinstance(testCaseClass, type)
        and issubclass(testCaseClass, common_device_type.DeviceTypeTestBase)
    ):
      return []
    # Drop classes labelled for other hardware types, e.g. CUDA, or as GENERIC,
    # which are plain tests covered by PyTorch CPU CI. Keep unlabelled classes
    # until the entire test suite is labelled.
    if getattr(testCaseClass, "hw_classification", None) not in (
        None,
        common_utils.HardwareClassification.ACCELERATOR,
    ):
      return []
    return original(self, testCaseClass)

  unittest.TestLoader.getTestCaseNames = device_generic_tests


def _parse_test_env_vars() -> tuple[str, pathlib.Path, pathlib.Path]:
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

  # Absolute paths stay valid even if the working directory changes later,
  # e.g. when the test module is run by a test framework that changes it.
  # Unlike resolve(), absolute() does not follow symlinks out of the runfiles.
  test_dir_path = pathlib.Path(test_dir).absolute()
  test_module_path = test_dir_path / f"{test_module}.py"
  if not test_module_path.is_file():
    raise FileNotFoundError(
        f"{test_module_path} cannot be found. Please set"
        f" {_TORCH_TEST_DIR_ENV_VAR} to the test/ directory of a"
        " https://github.com/pytorch/pytorch checkout."
    )

  return test_module, test_dir_path, test_module_path


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

  # `unittest.main` is declared as `type[TestProgram]`, so a drop-in function
  # cannot satisfy the annotation even though it is call-compatible.
  unittest.main = absltest_main  # pyrefly: ignore[bad-assignment]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Early exit on older PyTorch versions where the accelerator testing APIs
  # used by this runner are not available.
  if torch.__version__ < _MINIMAL_TORCH_VERSION:
    sys.exit(
        f"ERROR: Please use v{_MINIMAL_TORCH_VERSION}+ to run device-generic"
        " tests with TorchTPU."
    )

  test_module, test_dir_path, test_module_path = _parse_test_env_vars()
  _setup_test_args(test_dir_path, test_module_path)

  # PYTORCH_TESTING_DEVICE_* environment variables are read by
  # `torch.testing._internal.common_device_type` to pick the device test bases
  # of each `instantiate_device_type_tests` invocation in the following steps:
  #   1. Filter by `only_for` / `except_for` fields at the call site.
  #   2. Add back the bases named in PYTORCH_TESTING_DEVICE_FOR_CUSTOM.
  #   3. Filter by PYTORCH_TESTING_DEVICE_ONLY_FOR / PYTORCH_TESTING_DEVICE_EXCEPT_FOR.  # pylint: disable=line-too-long
  # Setting both to "tpu" overrides upstream settings and yields exactly one
  # TPU variant per instantiation.
  os.environ["PYTORCH_TESTING_DEVICE_FOR_CUSTOM"] = "tpu"
  os.environ["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "tpu"
  _set_private_use1_test_configs(test_module)
  _filter_device_generic_tests()

  # TODO(b/571605100): fix upstream tests to use public APIs only and then
  # remove API warning suppression.
  annotations._internal_apis_allowed_for_testing = True  # pylint: disable=protected-access

  # Execute the test module as a program. Each test module has its own entry
  # point invoking `run_tests()` from its own namespace. Google-internal build
  # patches `run_tests()` to run with pytest.
  _use_absltest_main()
  runpy.run_path(str(test_module_path), run_name="__main__")


if __name__ == "__main__":
  app.run(main)
