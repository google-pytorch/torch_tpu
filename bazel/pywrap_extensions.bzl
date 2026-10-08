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

Core extensions loaded during `import torch_tpu` are listed in
`CORE_PYWRAP_EXTENSIONS` and linked into `//csrc/common:pywrap_torch_tpu`.
Subsystem-scoped extensions that are not loaded at `import torch_tpu` time are
listed in `SCOPED_PYWRAP_EXTENSIONS` and linked into dedicated per-subsystem
`pywrap_library` targets (dynamically linking against
`libpywrap_torch_tpu_common.so` to preserve C++ ODR safety) so C++ changes in an
isolated subsystem do not invalidate `pywrap_torch_tpu` and re-trigger the
entire Python test suite in `bazel-diff`.

`PYWRAP_EXTENSIONS` remains the union of all extensions for the multi-version
pip wheel (`//csrc/common/glue_<suffix>` and `//csrc/common:xla_base_filter`).
"""

CORE_PYWRAP_EXTENSIONS = [
    "//src/torch_tpu/_internal:env",
    "//src/torch_tpu/_internal:testing",
    "//src/torch_tpu/_internal:warnings_impl",
    "//src/torch_tpu/_internal/compile:tpu_torch_compile",
    "//src/torch_tpu/_internal/compiler_options:compiler_options_impl",
    "//src/torch_tpu/_internal/device:_device_ops_backend",
    "//src/torch_tpu/_internal/distributed:tpu_distributed",
    "//src/torch_tpu/_internal/execution_mode:execution_mode_impl",
    "//src/torch_tpu/_internal/pallas:tpu_torch_pallas",
    "//src/torch_tpu/_internal/precision:precision_impl",
    "//src/torch_tpu/_internal/profiler:_profiler_backend",
    "//src/torch_tpu/_internal/sync:_tpu_torch_sync",
    "//src/torch_tpu/_internal/tracing:_tpu_torch_tracing",
]

SCOPED_PYWRAP_EXTENSIONS = {
    "pywrap_annotations": "//src/torch_tpu/_internal/device_utils:annotations_py",
    "pywrap_batch_transfer": "//src/torch_tpu/_internal/batch_transfer:batch_transfer_impl",
    "pywrap_dynamism": "//src/torch_tpu/_internal/dynamism:_tpu_torch_dynamism",
}

PYWRAP_EXTENSIONS = sorted(
    CORE_PYWRAP_EXTENSIONS + list(SCOPED_PYWRAP_EXTENSIONS.values()),
)
