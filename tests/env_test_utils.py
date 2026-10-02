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

"""Test helpers for environment variables read by TorchTPU."""

from collections.abc import Iterator, Mapping
import contextlib
import os
from unittest import mock

from torch_tpu._internal import env_utils
from torch_tpu._internal import testing as tt_testing


@contextlib.contextmanager
def scoped_env(
    values: Mapping[str, str], *, clear: bool = False
) -> Iterator[None]:
  """Sets environment variables as TorchTPU observes them, temporarily.

  On entry and on exit, this clears the values memoized by the
  `get_*_env_once()` readers in `torch_tpu._internal.env` and by
  `env_utils.get_and_set_str_env_once()`. TorchTPU code therefore observes
  `values` inside the context, and the original environment afterwards.

  Args:
    values: The environment variables to set.
    clear: Whether to unset all other environment variables first.

  Yields:
    None.
  """
  try:
    with (
        mock.patch.dict(os.environ, values, clear=clear),
        mock.patch.dict(env_utils._resolved_values, clear=True),  # pylint: disable=protected-access
    ):
      tt_testing.reset_env_cache_for_testing()
      yield
  finally:
    tt_testing.reset_env_cache_for_testing()
