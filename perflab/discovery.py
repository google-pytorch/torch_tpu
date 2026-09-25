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

"""Imports every module in a package so its @register_benchmark decorators run.

Import failures are collected and returned; the caller is expected to turn
them into visible failures. One broken module does not prevent the others from
loading; the full suite is loaded and the user gets an explicit report of what
failed and why.
"""

import dataclasses
import importlib
import pkgutil
import types


@dataclasses.dataclass(frozen=True)
class ImportFailure:
  """A module that could not be imported, and why."""

  module: str
  error: Exception

  def __str__(self) -> str:
    return f"{self.module}: {type(self.error).__name__}: {self.error}"


def import_submodules(package: types.ModuleType) -> list[ImportFailure]:
  """Imports every submodule of `package` (non-recursive, one level).

  Benchmark registration happens as a side effect of import; the decorators run
  and populate registry.

  Args:
    package: The Python package module whose immediate submodules to import.

  Returns:
    A list of ImportFailure instances for any submodules that failed to import;
    an empty list means all submodules imported successfully.
  """
  failures: list[ImportFailure] = []
  for info in pkgutil.iter_modules(package.__path__):
    full = f"{package.__name__}.{info.name}"
    try:
      importlib.import_module(full)
    except Exception as exc:  # pylint: disable=broad-exception-caught,broad-except
      failures.append(ImportFailure(module=full, error=exc))
  return failures
