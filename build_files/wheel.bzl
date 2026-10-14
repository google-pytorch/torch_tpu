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

"""Shared list of the Python targets that the torch_tpu wheel ships."""

WHEEL_PY_TARGETS = [
    # go/keep-sorted start
    "//src/torch_tpu",
    "//src/torch_tpu/_internal/compile/dynamic:view_decomposition",
    "//src/torch_tpu/_internal/compile/fx_passes:clone_mutated_returned_placeholders",
    "//src/torch_tpu/_internal/compile/fx_passes:propagate_symints",
    "//src/torch_tpu/_internal/compile/fx_passes:reorder_symints",
    "//src/torch_tpu/_internal/compile/fx_passes:sink_get_attr_constants",
    "//src/torch_tpu/_internal/compile:tpu_backend",
    "//src/torch_tpu/_internal/compiler_options",
    "//src/torch_tpu/_internal/device:_device_module",
    "//src/torch_tpu/_internal/device_utils:annotations",
    "//src/torch_tpu/_internal/distributed:_distributed_module",
    "//src/torch_tpu/_internal/distributed:distributed",
    "//src/torch_tpu/_internal/distributed:gpu_env",
    "//src/torch_tpu/_internal/distributed:handshake",
    "//src/torch_tpu/_internal/distributed:process_group_utils",
    "//src/torch_tpu/_internal/distributed:spmd_util",
    "//src/torch_tpu/_internal/dynamism",
    "//src/torch_tpu/_internal/execution_mode",
    "//src/torch_tpu/_internal/export",
    "//src/torch_tpu/_internal/ops",
    "//src/torch_tpu/_internal/ops/scaled_mm/kernels:blockwise128_scaled_mm_kernel",
    "//src/torch_tpu/_internal/ops/scaled_mm/kernels:blockwise_scaled_mm_kernel",
    "//src/torch_tpu/_internal/ops:ops_lib",
    "//src/torch_tpu/_internal/ops:splash_attention",
    "//src/torch_tpu/_internal/optim",
    "//src/torch_tpu/_internal/optimizers:adamw",
    "//src/torch_tpu/_internal/optimizers:muon",
    "//src/torch_tpu/_internal/optimizers:optim",
    "//src/torch_tpu/_internal/optimizers:sgd",
    "//src/torch_tpu/_internal/optimizers:single_trace_trainer",
    "//src/torch_tpu/_internal/pallas",
    "//src/torch_tpu/_internal/pallas:mesh",
    "//src/torch_tpu/_internal/performance/micro:utils",
    "//src/torch_tpu/_internal/profiler",
    "//src/torch_tpu/_internal/stream:streams",
    "//src/torch_tpu/_internal/sync",
    "//src/torch_tpu/_internal/tracing",
    "//src/torch_tpu/_internal/utils",
    "//src/torch_tpu/_internal/utils:hardware",
    "//src/torch_tpu/_internal/utils:log_utils",
    "//src/torch_tpu/_internal/utils:tracer_utils",
    "//src/torch_tpu/_internal:env_utils",
    "//src/torch_tpu/_internal:public_api",
    "//src/torch_tpu/_internal:version",
    "//src/torch_tpu/_internal:warnings",
    "//src/torch_tpu/_internal/benchmarks:export_adapter",
    "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
    "//src/torch_tpu/_internal/distributed:multiprocessing",
    "//src/torch_tpu:_loader",
    # go/keep-sorted end
]
