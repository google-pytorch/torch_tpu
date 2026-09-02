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

"""Public server API for TorchTPU.

Exposes `start_server` and `stop_server` under `torch.tpu.server` and
`torch_tpu.server` for managing the background gRPC profiler server on demand,
while delegating the underlying implementation to `_internal.profiler._impl`.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  # pylint: disable=g-import-not-at-top,g-multiple-import
  from torch_tpu._internal.profiler._impl import (
      start_server,
      stop_server,
  )

__all__ = ["start_server", "stop_server"]

# Module-level __getattr__ and __dir__ implement PEP 562.
# pylint: disable=invalid-name


def __getattr__(name: str) -> Any:
  """Lazily loads public server symbols upon attribute access (PEP 562).

  Lazy loading avoids loading C++ gRPC and profiler backend libraries during
  backend initialization and defers importing internal server dependencies until
  public symbols are actually referenced.

  Args:
    name: The name of the attribute being accessed.

  Returns:
    The requested attribute from the internal server module.

  Raises:
    AttributeError: If the requested attribute is not part of the public API.
  """
  if name == "start_server":
    # Lazily import start_server from _internal.
    from torch_tpu._internal.profiler._impl import (  # pylint: disable=g-import-not-at-top
        start_server as internal_start_server,
    )

    return internal_start_server
  elif name == "stop_server":
    # Lazily import stop_server from _internal.
    from torch_tpu._internal.profiler._impl import (  # pylint: disable=g-import-not-at-top
        stop_server as internal_stop_server,
    )

    return internal_stop_server
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
  return __all__
