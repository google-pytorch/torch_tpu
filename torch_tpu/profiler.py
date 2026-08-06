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

"""Public proxy module for the `torch.tpu.profiler` namespace.

This module exposes public TorchTPU profiling APIs under `torch.tpu.profiler`
using standard PEP-562 lazy attribute loading.

Purpose of this file:
1. Namespace Resolution: Serves as the canonical python module file mapped to
   `sys.modules["torch.tpu.profiler"]`, enabling standard PyTorch import
   idioms (e.g. `from torch.tpu.profiler import TpuProfilerConfig`).
2. Boot Safety: Uses PEP-562 `__getattr__` to defer importing heavy internal C++
   profiler runtime bindings until actual attribute access, preventing circular
   import deadlocks during PyTorch device backend startup.
3. Tooling Compatibility: Provides an `if TYPE_CHECKING:` branch so IDEs,
   Pytype, Mypy, and autocompletion tools resolve public symbols statically.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from torch_tpu._internal.profiler.profiler_config import TpuProfilerConfig  # pylint: disable=unused-import

  __all__ = ["TpuProfilerConfig"]

if not TYPE_CHECKING:
  __all__ = ["TpuProfilerConfig"]

  def __getattr__(name: str) -> Any:
    if name == "TpuProfilerConfig":
      from torch_tpu._internal.profiler import profiler_config  # pylint: disable=g-import-not-at-top

      target = profiler_config.TpuProfilerConfig
      globals()[name] = target
      return target

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

  def __dir__() -> list[str]:
    return __all__
