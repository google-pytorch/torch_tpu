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

Any new pybind_extension target created in the codebase MUST be added here so
it is linked into the aggregating pywrap_library
(//torch_tpu/common:pywrap_torch_tpu) and its backend probe feeds the shared
XLA base filter (//torch_tpu/common:_xla_base_agg).
"""

PYWRAP_EXTENSIONS = [
    "//torch_tpu/_internal:env",
    "//torch_tpu/_internal:testing",
    "//torch_tpu/_internal/batch_transfer:batch_transfer_impl",
    "//torch_tpu/_internal/compile:tpu_torch_compile",
    "//torch_tpu/_internal/compiler_options:compiler_options_impl",
    "//torch_tpu/_internal/device:_device_ops_backend",
    "//torch_tpu/_internal/device_utils:annotations_py",
    "//torch_tpu/_internal/distributed:tpu_distributed",
    "//torch_tpu/_internal/dynamism:_tpu_torch_dynamism",
    "//torch_tpu/_internal/execution_mode:execution_mode_impl",
    "//torch_tpu/_internal/pallas:tpu_torch_pallas",
    "//torch_tpu/_internal/precision:precision_impl",
    "//torch_tpu/_internal/profiler:_profiler_backend",
    "//torch_tpu/_internal/sync:_tpu_torch_sync",
    "//torch_tpu/_internal/tracing:_tpu_torch_tracing",
]
