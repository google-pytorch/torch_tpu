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

"""Profiler module helper for lazy loading and deprecation warnings.

This module acts as the public `torch.tpu.profiler` namespace. It implements
PEP-562 (__getattr__ and __dir__) to lazily load the internal profiling
components at runtime. This avoids importing heavy profiler dependencies
during early device initialization, and allows us to emit experimental warnings
when users access the public profiler API.
"""

import warnings


def __getattr__(name: str):  # pylint: disable=invalid-name
  if name == "TpuProfilerConfig":
    warnings.warn(
        "torch.tpu.profiler.TpuProfilerConfig is experimental and subject to"
        " change.",
        category=UserWarning,
        stacklevel=2,
    )
    # pylint: disable=g-import-not-at-top
    from torch_tpu._internal.profiler import profiler_config
    # pylint: enable=g-import-not-at-top
    globals()[name] = profiler_config.TpuProfilerConfig
    return profiler_config.TpuProfilerConfig

  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():  # pylint: disable=invalid-name
  return sorted(set(globals().keys()) | {"TpuProfilerConfig"})
