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

"""Central list of every C++ Python extension in torch_tpu.

Any new pybind_extension target created in the codebase MUST be added here so it
is linked into both the bazel-only aggregator (//torch_tpu/csrc/common:pywrap_torch_tpu)
and every per-version wheel pywrap_library (//torch_tpu/csrc/common/glue_<suffix>).
"""

PYWRAP_EXTENSIONS = [
    "//torch_tpu/csrc/internal:env",
    "//torch_tpu/csrc/internal:testing",
    "//torch_tpu/csrc/internal/batch_transfer:batch_transfer_impl",
    "//torch_tpu/csrc/internal/compile:tpu_torch_compile",
    "//torch_tpu/csrc/internal/compiler_options:compiler_options_impl",
    "//torch_tpu/csrc/internal/device:_device_ops_backend",
    "//torch_tpu/csrc/internal/device_utils:annotations_py",
    "//torch_tpu/csrc/internal/distributed:tpu_distributed",
    "//torch_tpu/csrc/internal/dynamism:_tpu_torch_dynamism",
    "//torch_tpu/csrc/internal/execution_mode:execution_mode_impl",
    "//torch_tpu/csrc/internal/pallas:tpu_torch_pallas",
    "//torch_tpu/csrc/internal/precision:precision_impl",
    "//torch_tpu/csrc/internal/profiler:_profiler_backend",
    "//torch_tpu/csrc/internal/sync:_tpu_torch_sync",
    "//torch_tpu/csrc/internal/tracing:_tpu_torch_tracing",
]
