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

"""Utilities for OSS/GCP environment detection and testing."""

from collections.abc import Callable
import functools
import os
from typing import Any, TypeVar
import unittest

from torch import torch_version
from torch_tpu._internal.utils import utils

_T = TypeVar("_T", bound=Callable[..., Any])


def running_in_cloud() -> bool:
  """Returns True if the codebase is running in GCP."""
  try:
    import libtpu  # pylint: disable=g-import-not-at-top,unused-import # # ruff: ignore[unused-import]  # pyrefly: ignore[missing-import]

    return True
  except ImportError:
    return False


def is_oss() -> bool:
  """Returns True if running in Open Source (OSS / GCP / Cloud) environment."""
  return running_in_cloud() or os.getenv("IS_OSS", "0") == "1"


def is_internal() -> bool:
  """Returns True if running in the Google internal environment."""
  return not is_oss()


def skip_in_oss(
    reason: str = "Skipping in Open Source (OSS) environment due to numerics.",
):
  """Decorator to skip test in Open Source (OSS/GCP) environment."""

  def decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
      if is_oss():
        self.skipTest(reason)
      return func(self, *args, **kwargs)

    return wrapper

  return decorator


def skip_if_libtpu_version(
    pred: Callable[[torch_version.TorchVersion], bool],
    message: str,
) -> Callable[[_T], _T]:
  """Decorator to skip a test class, method, or function in OSS when `pred(libtpu_version)` is true."""
  return unittest.skipIf(
      is_oss() and (ver := utils.libtpu_version()) is not None and pred(ver),
      message,
  )
