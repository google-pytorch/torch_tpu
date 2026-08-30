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

"""Public profiler API for TorchTPU.

Exposes `TpuProfilerConfig` under `torch.tpu.profiler` and `torch_tpu.profiler`
for configuring TPU-specific execution tracing options with PyTorch's native
profiler (`torch.profiler.profile`), while delegating the underlying
implementation to `_internal.profiler.profiler_config`.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from torch_tpu._internal.profiler.profiler_config import (
      TpuProfilerConfig as TpuProfilerConfig,  # noqa: F401
  )

__all__ = ["TpuProfilerConfig"]


def __getattr__(name: str) -> Any:
  """Lazily loads public profiler symbols upon attribute access (PEP 562).

  Lazy loading avoids circular imports during backend initialization and defers
  importing internal profiler dependencies until public symbols are actually
  referenced.

  Args:
    name: The name of the attribute being accessed.

  Returns:
    The requested attribute from the internal profiler module.

  Raises:
    AttributeError: If the requested attribute is not part of the public API.
  """
  if name == "TpuProfilerConfig":
    # Lazily import the concrete configuration class from _internal.
    from torch_tpu._internal.profiler.profiler_config import (  # pylint: disable=g-import-not-at-top
        TpuProfilerConfig,
    )

    return TpuProfilerConfig
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
  return __all__
