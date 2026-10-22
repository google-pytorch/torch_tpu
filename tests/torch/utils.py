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

"""Test utilities for running PyTorch device-generic tests with TorchTPU."""

import importlib
import os
from typing import Any
import unittest

import pytest

_pytest_xfail_patched = False


def patch_pytest_xfail_for_unittest() -> None:
  """Patches pytest.xfail to raise unittest.SkipTest when running outside pytest.

  Decorators like @xfailIfNoAcceleratorTriton call pytest.xfail(...)
  imperatively
  inside test wrappers. When upstream tests are run with unittest/absltest
  rather
  than pytest, _pytest.outcomes.XFailed (which inherits from BaseException) is
  not
  caught as an expected failure or skip, aborting the test runner with ERROR.
  """
  global _pytest_xfail_patched
  if _pytest_xfail_patched:
    return

  try:
    outcomes = importlib.import_module("_pytest.outcomes")
    original_xfail = outcomes.xfail
  except ImportError:
    outcomes = None
    original_xfail = getattr(pytest, "xfail", None)

  def _unittest_skip_xfail(reason: str = "", *args: Any, **kwargs: Any) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ and original_xfail is not None:
      return original_xfail(reason, *args, **kwargs)
    raise unittest.SkipTest(f"pytest.xfail: {reason}")

  if original_xfail is not None:
    _unittest_skip_xfail.Exception = getattr(  # pyrefly: ignore[missing-attribute]
        original_xfail, "Exception", unittest.SkipTest
    )

  setattr(pytest, "xfail", _unittest_skip_xfail)
  if outcomes is not None:
    setattr(outcomes, "xfail", _unittest_skip_xfail)

  _pytest_xfail_patched = True
