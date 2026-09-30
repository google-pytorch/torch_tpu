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

"""Public compiler API for TorchTPU.

Exposes `custom_compiler_options` under `torch.tpu.compiler` for configuring
XLA compiler options within a context manager, while delegating the underlying
implementation to `_internal.compiler_options`.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from torch_tpu._internal.compiler_options import (
      custom_compiler_options as custom_compiler_options,  # noqa: F401
  )

__all__ = ["custom_compiler_options"]


def __getattr__(name: str) -> Any:
  """Lazily loads public compiler symbols upon attribute access (PEP 562).

  Lazy loading avoids circular imports during backend initialization and defers
  importing internal dependencies until public symbols are actually referenced.

  Args:
    name: The name of the attribute being accessed.

  Returns:
    The requested attribute from the internal compiler_options module.

  Raises:
    AttributeError: If the requested attribute is not part of the public API.
  """
  if name == "custom_compiler_options":
    from torch_tpu._internal.compiler_options import (  # pylint: disable=g-import-not-at-top
        custom_compiler_options,
    )

    return custom_compiler_options
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
  return __all__
