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
    "torch_tpu_cc_library",
    "torch_tpu_cc_test",
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
        ":torch_tpu_compiled_executable",
        ":tpu_torch_compile",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile/fx_passes:mark_activation_checkpoints",
        "//torch_tpu/_internal/compile/fx_passes:mark_embedded_constants",
        "//torch_tpu/_internal/export",
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
        ":scan",
        ":tpu_backend",
        ":tpu_torch_compile",
        "//shims/absl_py/logging",
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
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile/dynamic:compiler",
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

torch_tpu_cc_library(
    name = "compiled_mode",
    srcs = [
        "compiled_mode.cc",
    ],
    hdrs = [
        "compiled_mode.h",
    ],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:compilation_spec",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "//torch_tpu/common:to_string",
        "//torch_tpu/common:utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:device_buffer_utils",
        "//torch_tpu/eager:materialize",
        "//torch_tpu/eager:structured_log_buffer",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/eager:traversal",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "//torch_tpu/ops/view_decomposition:decomposition",
        "//torch_tpu/ops/view_decomposition:strided_layout",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:stablehlo_serialization",
        "@tsl//tsl/platform:path",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:maybe_owning_mlir_module",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",  # buildcleaner: keep
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that tpu_torch_compile.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "tpu_torch_compile_lib",
    srcs = ["tpu_torch_compile.cc"],
    deps = [
        ":compiled_mode",
        ":dispatch_scan",
        "//shims/pybind11",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/_internal/dynamism:dynamism_ops",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:compilation_cache",
        "//torch_tpu/common:compilation_spec",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:env_vars",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "//torch_tpu/common:to_string",
        "//torch_tpu/common:utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:device_gen_impl",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "//torch_tpu/ops:scan_builder",
        "//torch_tpu/ops/view_decomposition:contiguous_to_view",
        "//torch_tpu/ops/view_decomposition:decomposition",
        "//torch_tpu/ops/view_decomposition:strided_layout",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@xla//xla/mlir/utils:error_util",
        "@xla//xla/pjrt:maybe_owning_mlir_module",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_layout",
    ],
)

pybind_extension(
    name = "tpu_torch_compile",
    srcs = ["tpu_torch_compile.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        ":compiled_mode",
        ":dispatch_scan",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/_internal/dynamism:dynamism_ops",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:compilation_cache",
        "//torch_tpu/common:compilation_spec",
        "//torch_tpu/common:compile_options_key",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:env_vars",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "//torch_tpu/common:to_string",
        "//torch_tpu/common:utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:device_gen_impl",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "//torch_tpu/ops:scan_builder",
        "//torch_tpu/ops/view_decomposition:contiguous_to_view",
        "//torch_tpu/ops/view_decomposition:decomposition",
        "//torch_tpu/ops/view_decomposition:strided_layout",
        "//torch_tpu/pjrt:pjrt_state",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/translate:register",
        "@xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@xla//xla/mlir/utils:error_util",
        "@xla//xla/pjrt:maybe_owning_mlir_module",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_layout",
    ],
)

torch_tpu_cc_library(
    name = "dispatch_scan",
    srcs = ["dispatch_scan.cc"],
    hdrs = ["dispatch_scan.h"],
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "//torch_tpu/common:utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "//torch_tpu/ops:scan_builder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
    ],
)

torch_tpu_cc_test(
    name = "dispatch_scan_test",
    srcs = ["dispatch_scan_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    notest_oss = "Requires TPU PJRT plugin (transitively via eager:tensor_to_buffer -> pjrt:pjrt_state), which fails due to missing TPU_LIBRARY_PATH.",
    deps = [
        ":dispatch_scan",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:context_states",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:scan_builder",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/mlir/utils:error_util",
        "@xla//xla/tsl/platform:statusor",
    ],
)

torch_tpu_package_end()
