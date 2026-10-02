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

"""Compatibility utilities for different Python versions."""

import pathlib
import sys

from torch_tpu._internal import warnings
from torch_tpu._internal.utils import tt_warnings


def warn_deprecation_with_skip(message: str, skip_dir: pathlib.Path):
  """Issues a deprecation warning, skipping files in skip_dir if Python >= 3.12.

  This handles the `skip_file_prefixes` argument which was added to
  `warnings.warn` in Python 3.12. For older versions, it falls back to
  `stacklevel=2` to skip the immediate caller frame.

  Args:
    message: The deprecation message to issue.
    skip_dir: The directory path to skip in stack traces.
  """
  if sys.version_info >= (3, 12):
    tt_warnings.warn(
        message,
        warnings.DEPRECATED_API,
        skip_file_prefixes=(str(skip_dir),),
    )
  else:
    tt_warnings.warn(
        message,
        warnings.DEPRECATED_API,
        stacklevel=3,
    )
