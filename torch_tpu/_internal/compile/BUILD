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

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        ":debug",
        ":torch_tpu_compiled_executable",
        ":tpu_torch_compile",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/compile/fx_passes:fold_gqa",
        "//torch_tpu/_internal/compile/fx_passes:mark_activation_checkpoints",
        "//torch_tpu/_internal/compile/fx_passes:mark_embedded_constants",
        "//torch_tpu/_internal/export",
    ],
)

torch_tpu_py_library(
    name = "split_compiler",
    srcs = ["split_compiler.py"],
    deps = [
        ":compiler",
        ":torch_tpu_compiled_executable",
        ":tpu_torch_compile",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile/fx_passes:clone_mutated_returned_placeholders",
        "//torch_tpu/_internal/compile/fx_passes:force_collectives_output",
        "//torch_tpu/_internal/compile/fx_passes:mark_embedded_constants",
        "//torch_tpu/_internal/compile/fx_passes:propagate_symints",
        "//torch_tpu/_internal/compile/fx_passes:reorder_symints",
        "//torch_tpu/_internal/compile/fx_passes:sink_get_attr_constants",
        "//torch_tpu/_internal/distributed:collective_ops",
        "//torch_tpu/_internal/distributed:handshake",
        "//torch_tpu/_internal/distributed:process_group_utils",
        "//torch_tpu/_internal/distributed:spmd_util",
    ],
)

torch_tpu_py_library(
    name = "torch_tpu_compiled_executable",
    srcs = ["torch_tpu_compiled_executable.py"],
    deps = [
        ":tpu_torch_compile",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "compile",
    srcs = [
        "__init__.py",
    ],
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
    deps = [
        ":debug",
        ":scan",
        ":tpu_backend",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "tpu_backend",
    srcs = [
        "_backend.py",
    ],
    tags = [
        # Avoid a circular dependency. Imports torch_tpu for package version.
        "ignore_for_dep=third_party.py.torch_tpu",
    ],
    deps = [
        ":compiler",
        ":debug",
        ":split_compiler",
        ":torch_tpu_compiled_executable",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/compile/dynamic:dynamic_compiler",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_library(
    name = "scan",
    srcs = ["scan.py"],
    deps = [
        ":tpu_torch_compile",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/export",
    ],
)

pybind_extension(
    name = "tpu_torch_compile",
    srcs = ["//csrc/internal/compile:tpu_torch_compile.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:compilation_cache",
        "//csrc/common:compilation_spec",
        "//csrc/common:compile_options_key",
        "//csrc/common:context_manager",
        "//csrc/common:context_states",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:pybind_error_utils",
        "//csrc/common:shape",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/distributed:handshake",
        "//csrc/eager:device_buffer",
        "//csrc/eager:device_gen_impl",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/internal/compile:compiled_mode",
        "//csrc/internal/compile:dispatch_scan",
        "//csrc/internal/compile:torch_compile_utils",
        "//csrc/internal/dynamism:dynamism_ops",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//csrc/ops:scan_builder",
        "//csrc/ops/view_decomposition:contiguous_to_view",
        "//csrc/ops/view_decomposition:decomposition",
        "//csrc/ops/view_decomposition:strided_layout",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@tsl//tsl/platform:fingerprint",
        "@xla//xla:shape_util",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@xla//xla/mlir/utils:error_util",
        "@xla//xla/pjrt:maybe_owning_mlir_module",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/service:computation_placer",
        "@xla//xla/service:device_assignment",
    ],
)

torch_tpu_py_library(
    name = "debug",
    srcs = ["debug.py"],
)

torch_tpu_package_end()
