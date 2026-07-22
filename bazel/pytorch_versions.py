#!/usr/bin/env python3
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

"""Print a PyTorch-version value from the canonical `pytorch_versions.bzl`.

`//bazel:pytorch_versions.bzl` is the single source of truth for the PyTorch
versions the multi-ABI wheel targets. It is data-only Starlark (plain
assignments), which is also valid Python, so shell tooling -- namely
`requirements/lock_environments.sh` -- can read the same lists from it through
this helper instead of hardcoding the versions a second time.

Each element is printed on its own line (a bare string value prints as a single
line), which is convenient for a shell `for` loop.
"""

import argparse
from collections.abc import Sequence, Set
import pathlib
from typing import Final

# The canonical Starlark file sits next to this script under the same stem.
_BZL_PATH: Final[pathlib.Path] = pathlib.Path(__file__).with_suffix(".bzl")

# The names a caller may ask for, matching the assignments in the .bzl.
_KNOWN_NAMES: Final[Set[str]] = frozenset({
    "DEFAULT_TORCH_VERSION",
    "EXTRA_PYTORCH_VERSIONS",
    "GLUE_TORCH_VERSIONS",
    "NIGHTLY_TORCH_VERSION",
    "SENTINEL_TORCH_VERSION",
    "WHEEL_TORCH_VERSIONS",
})


def read_value(name: str, bzl_path: pathlib.Path = _BZL_PATH) -> Sequence[str]:
  """Reads the value bound to `name` in the canonical `.bzl` file.

  Args:
    name: the constant to read, one of `_KNOWN_NAMES`.
    bzl_path: the data-only `.bzl` file to read it from.

  Returns:
    The value as a sequence of strings. A scalar (e.g.
    `DEFAULT_TORCH_VERSION`) is returned as a one-element sequence so every
    caller can treat the result uniformly.

  Raises:
    TypeError: if the value bound to `name` is not a string or a list.
  """
  namespace: dict[str, object] = {}
  exec(compile(bzl_path.read_text(), str(bzl_path), "exec"), namespace)  # pylint: disable=exec-used
  value = namespace[name]
  if isinstance(value, str):
    return (value,)
  if isinstance(value, list):
    return tuple(value)
  raise TypeError(f"{name} in {bzl_path} is not a string or list: {value!r}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "name",
      nargs="?",
      default="EXTRA_PYTORCH_VERSIONS",
      choices=sorted(_KNOWN_NAMES),
      help="Which value to print (default: EXTRA_PYTORCH_VERSIONS).",
  )
  args = parser.parse_args()

  for line in read_value(args.name):
    print(line)


if __name__ == "__main__":
  main()
