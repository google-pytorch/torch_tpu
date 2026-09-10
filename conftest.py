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

"""Pytest configuration for TorchTPU tests."""

from collections.abc import Sequence
import os
from typing import Any


def _set_flag_value(flag: Any, candidates: Sequence[str]) -> None:
  """Sets the flag to the first candidate value accepted by its parser."""
  for candidate in candidates:
    try:
      flag.value = flag._parse(candidate)  # pylint: disable=protected-access
      return
    except (ValueError, TypeError, AttributeError):
      continue


def _configure_absl_flags(configure_test_mode: bool = False) -> None:
  """Marks ABSL flags as parsed and optionally sets default test_mode."""
  try:
    from absl import flags  # pylint: disable=g-import-not-at-top
  except ImportError:
    return

  if (
      configure_test_mode
      and os.environ.get("TORCH_TPU_DISABLE_TEST_MODE_OVERRIDE") != "true"
      and "test_mode" in flags.FLAGS
  ):
    test_mode_flag = flags.FLAGS["test_mode"]
    module_name = getattr(test_mode_flag, "module_name", "")
    candidates = (
        ("tpu",)
        if "error_testing" in module_name
        else ("torch_tpu_vs_cpu", "cpu", "tpu")
    )
    _set_flag_value(test_mode_flag, candidates)

  flags.FLAGS.mark_as_parsed()


def pytest_configure(config: Any) -> None:
  """Hook called after command line options have been parsed."""
  del config
  _configure_absl_flags()


def pytest_collection_modifyitems(items: list[Any]) -> None:
  """Hook to initialize required ABSL flags after modules are imported."""
  del items
  _configure_absl_flags(configure_test_mode=True)


def pytest_runtest_setup(item: Any) -> None:
  """Hook called before running each test item."""
  del item
  _configure_absl_flags()
