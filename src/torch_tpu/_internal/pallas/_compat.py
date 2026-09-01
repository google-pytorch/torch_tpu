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
import warnings


def warn_deprecation_with_skip(message: str, skip_dir: pathlib.Path):
  """Issues a DeprecationWarning, skipping files in skip_dir if Python >= 3.12.

  This handles the `skip_file_prefixes` argument which was added to
  `warnings.warn` in Python 3.12. For older versions, it falls back to
  `stacklevel=2` to skip the immediate caller frame.

  Args:
    message: The deprecation message to issue.
    skip_dir: The directory path to skip in stack traces.
  """
  kwargs = {"category": DeprecationWarning}
  if sys.version_info >= (3, 12):
    kwargs["skip_file_prefixes"] = (str(skip_dir),)  # pyrefly: ignore[bad-assignment]
  else:
    kwargs["stacklevel"] = 3

  warnings.warn(message, **kwargs)
