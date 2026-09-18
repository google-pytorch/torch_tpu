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

"""Public Pallas API for TorchTPU.

Exposes `jax_op` under `torch.tpu.pallas` and `torch_tpu.pallas` for registering
and calling JAX/Pallas custom operations from PyTorch, while delegating the
underlying implementation to `_internal.pallas.pallas`.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from torch_tpu._internal.pallas.pallas import (
      jax_op as jax_op,  # noqa: F401
  )

__all__ = ["jax_op"]


def __getattr__(name: str) -> Any:
  """Lazily loads public Pallas symbols upon attribute access (PEP 562).

  Lazy loading avoids circular imports during backend initialization and defers
  importing JAX and Pallas dependencies until public symbols are actually
  referenced.

  Args:
    name: The name of the attribute being accessed.

  Returns:
    The requested attribute from the internal pallas module.

  Raises:
    AttributeError: If the requested attribute is not part of the public API.
  """
  if name == "jax_op":
    from torch_tpu._internal.pallas.pallas import (  # pylint: disable=g-import-not-at-top
        jax_op,
    )

    return jax_op
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
  return __all__
