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

"""Public distributed API for TorchTPU.

Exposes `environment` under `torch.tpu.distributed` for bootstrapping TPU
distributed launch environment variables, while delegating the underlying
implementation to `_internal.distributed.environment`.
"""

import importlib.util
import types
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from torch_tpu._internal.distributed import (
      environment as environment,  # noqa: F401
  )

__all__ = ["environment"]
__path__: list[str] = []


class _EnvironmentApiModule(types.ModuleType):
  """Public API proxy module for `torch.tpu.distributed.environment`."""

  __all__ = ["set_tpu_launch_env"]

  def __getattr__(self, name: str) -> Any:
    if name == "set_tpu_launch_env":
      from torch_tpu._internal.distributed.environment import (  # pylint: disable=g-import-not-at-top
          set_tpu_launch_env,
      )

      return set_tpu_launch_env
    raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

  def __dir__(self) -> list[str]:
    return self.__all__


def _create_environment_module() -> types.ModuleType:
  from torch_tpu._internal.distributed import (  # pylint: disable=g-import-not-at-top
      environment as _internal_environment,
  )

  mod = _EnvironmentApiModule(
      "torch.tpu.distributed.environment",
      _internal_environment.__doc__,
  )
  mod.__file__ = _internal_environment.__file__
  mod.__package__ = "torch.tpu.distributed"
  if _internal_environment.__file__ is not None:
    mod.__spec__ = importlib.util.spec_from_file_location(
        "torch.tpu.distributed.environment",
        _internal_environment.__file__,
    )
  return mod


_environment_module: types.ModuleType | None = None


def __getattr__(name: str) -> Any:
  """Lazily loads public distributed symbols upon attribute access (PEP 562)."""
  global _environment_module
  if name == "environment":
    if _environment_module is None:
      _environment_module = _create_environment_module()
    return _environment_module
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
  return __all__
