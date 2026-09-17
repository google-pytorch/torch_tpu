# Copyright 2025 Google LLC
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
"""Pallas API for Torch-TPU."""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from torch_tpu._internal.pallas.pallas import (
      PallasMesh as PallasMesh,  # noqa: F401
      custom_jax_kernel as custom_jax_kernel,  # noqa: F401
      get_default_pallas_mesh as get_default_pallas_mesh,  # noqa: F401
      get_pallas_mesh as get_pallas_mesh,  # noqa: F401
      jax_op as jax_op,  # noqa: F401
  )


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "PallasMesh",
    "custom_jax_kernel",
    "get_default_pallas_mesh",
    "get_pallas_mesh",
    "jax_op",
    # go/keep-sorted end
]


def __getattr__(name: str) -> Any:
  """Lazily loads pallas symbols and submodules upon attribute access (PEP 562).

  Lazy loading avoids importing JAX/Pallas dependencies when importing
  submodules (such as `torch_tpu._internal.pallas.api` during backend
  initialization) until Pallas symbols are actually referenced.
  """
  if name in (
      "PallasMesh",
      "custom_jax_kernel",
      "get_default_pallas_mesh",
      "get_pallas_mesh",
      "jax_op",
  ):
    from torch_tpu._internal.pallas import pallas  # pylint: disable=g-import-not-at-top

    return getattr(pallas, name)
  try:
    import importlib  # pylint: disable=g-import-not-at-top

    return importlib.import_module(f".{name}", __name__)
  except (ImportError, AttributeError):
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'"
    ) from None


def __dir__() -> list[str]:
  return __all__
