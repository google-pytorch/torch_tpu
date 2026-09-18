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

"""Python-side helpers for environment variables.

The `torch_tpu._internal.env` module exposes the C++ `GetEnvOnce()` family for
*reading* environment variables. This module complements it with helpers that
also *write* them, which cannot live on the C++ side: `SetEnv()` calls
`setenv()`, which updates the C-level environment but leaves Python's
`os.environ` mapping stale.
"""

import os
from typing import Callable

# Maps an environment variable name to the value resolved for it by
# get_and_set_str_env_once().
_resolved_values: dict[str, str] = {}


def get_and_set_str_env_once(
    name: str, value_factory: Callable[[], str]
) -> str:
  """Returns the value of an environment variable, generating it if unset.

  If the environment variable is not set, `value_factory()` is called and the
  environment variable is set to the result, so that child processes inherit
  it. A variable set to the empty string counts as set.

  `value_factory` is called at most once per variable name, even if the
  variable is later unset. The environment variable is guaranteed to be set
  when this function returns.

  Unlike the readers in `torch_tpu._internal.env`, this function goes through
  `os.environ` rather than C++. Writing via the C++ `SetEnv()` would leave
  Python's `os.environ` mapping stale, and reading via `GetEnvOnce()` before
  writing would permanently memoize the pre-write value on the C++ side, which
  C++ readers of the same variable would then observe.

  Args:
    name: The name of the environment variable.
    value_factory: Called to produce the value when the variable is unset. It is
      not called when the variable is already set, so it is a suitable place for
      expensive work such as picking unused ports.

  Returns:
    The value of the environment variable.
  """
  value = os.environ.get(name)
  if value is not None:
    return value

  # The variable is unset. It may still have been resolved by an earlier call
  # and unset since, for example by a test restoring `os.environ`, so reuse the
  # earlier value rather than generating a second one.
  value = _resolved_values.get(name)
  if value is None:
    value = value_factory()
    _resolved_values[name] = value
  os.environ[name] = value
  return value
