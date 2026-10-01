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

"""Public exceptions and warnings API for TorchTPU.

Exposes TorchTPU warning categories under `torch.tpu.exceptions` with
lifecycle stage annotations, while delegating the underlying definitions to
`_internal.exceptions`.
"""

from typing import Any, TYPE_CHECKING

from torch_tpu._internal import exceptions as _exceptions
from torch_tpu._internal.utils import annotations as _annotations

if TYPE_CHECKING:
  from torch_tpu._internal.exceptions import (
      ApiStageWarning as ApiStageWarning,  # noqa: F401
      CorrectnessWarning as CorrectnessWarning,  # noqa: F401
      DeprecatedApiWarning as DeprecatedApiWarning,  # noqa: F401
      ExperimentalApiWarning as ExperimentalApiWarning,  # noqa: F401
      PerformanceWarning as PerformanceWarning,  # noqa: F401
      TcMallocWarning as TcMallocWarning,  # noqa: F401
      TorchTpuWarning as TorchTpuWarning,  # noqa: F401
      UnoptimizedBuildWarning as UnoptimizedBuildWarning,  # noqa: F401
  )

__all__ = [
    # go/keep-sorted start
    "ApiStageWarning",
    "CorrectnessWarning",
    "DeprecatedApiWarning",
    "ExperimentalApiWarning",
    "PerformanceWarning",
    "TcMallocWarning",
    "TorchTpuWarning",
    "UnoptimizedBuildWarning",
    # go/keep-sorted end
]

__tt_api_stages__: dict[str, _annotations.ApiStageInfo] = {
    # go/keep-sorted start
    "ApiStageWarning": _annotations.ApiStageInfo(
        stage=_annotations.Stage.EXPERIMENTAL,
        reason="Warning category for APIs that are not in the STABLE stage.",
        value=_exceptions.ApiStageWarning,
    ),
    "CorrectnessWarning": _annotations.ApiStageInfo(
        stage=_annotations.Stage.EXPERIMENTAL,
        reason="Warning category for potential correctness issues.",
        value=_exceptions.CorrectnessWarning,
    ),
    "DeprecatedApiWarning": _annotations.ApiStageInfo(
        stage=_annotations.Stage.EXPERIMENTAL,
        reason="Warning category for APIs that are in the DEPRECATED stage.",
        value=_exceptions.DeprecatedApiWarning,
    ),
    "ExperimentalApiWarning": _annotations.ApiStageInfo(
        stage=_annotations.Stage.EXPERIMENTAL,
        reason="Warning category for APIs that are in the EXPERIMENTAL stage.",
        value=_exceptions.ExperimentalApiWarning,
    ),
    "PerformanceWarning": _annotations.ApiStageInfo(
        stage=_annotations.Stage.EXPERIMENTAL,
        reason="Warning category for potential performance issues.",
        value=_exceptions.PerformanceWarning,
    ),
    "TcMallocWarning": _annotations.ApiStageInfo(
        stage=_annotations.Stage.EXPERIMENTAL,
        reason="Warning category for missing tcmalloc allocator.",
        value=_exceptions.TcMallocWarning,
    ),
    "TorchTpuWarning": _annotations.ApiStageInfo(
        stage=_annotations.Stage.EXPERIMENTAL,
        reason="Base warning category for all warnings emitted by TorchTPU.",
        value=_exceptions.TorchTpuWarning,
    ),
    "UnoptimizedBuildWarning": _annotations.ApiStageInfo(
        stage=_annotations.Stage.EXPERIMENTAL,
        reason="Warning category for unoptimized TorchTPU builds.",
        value=_exceptions.UnoptimizedBuildWarning,
    ),
    # go/keep-sorted end
}


def __getattr__(name: str) -> Any:
  """Resolves public exception and warning symbols upon attribute access (PEP 562)."""
  return _annotations._resolve_module_attribute(  # pylint: disable=protected-access
      __tt_api_stages__,
      name,
      "torch.tpu.exceptions",
      tt_api_globals=globals(),
  )


def __dir__() -> list[str]:
  return __all__
