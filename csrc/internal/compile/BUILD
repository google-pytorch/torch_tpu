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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files(["tpu_torch_compile.cc"])

torch_tpu_cc_library(
    name = "compiled_mode",
    srcs = [
        "compiled_mode.cc",
    ],
    hdrs = [
        "compiled_mode.h",
    ],
    deps = [
        "//csrc/common:compilation",
        "//csrc/common:compilation_spec",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:device_buffer_utils",
        "//csrc/eager:materialize",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/eager:traversal",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//csrc/ops/view_decomposition:decomposition",
        "//csrc/ops/view_decomposition:strided_layout",
        "//csrc/pjrt:pjrt_state",
        "//shims/pybind11",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/algorithm:container",
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
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:maybe_owning_mlir_module",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",  # buildcleaner: keep
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_cc_library(
    name = "torch_compile_utils",
    srcs = [
        "torch_compile_utils.cc",
    ],
    hdrs = [
        "torch_compile_utils.h",
    ],
    deps = ["//csrc/common:env_vars"],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that tpu_torch_compile.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "tpu_torch_compile_lib",
    srcs = ["tpu_torch_compile.cc"],
    deps = [
        ":compiled_mode",
        ":dispatch_scan",
        ":torch_compile_utils",
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:compilation_cache",
        "//csrc/common:compilation_spec",
        "//csrc/common:context_manager",
        "//csrc/common:context_states",
        "//csrc/common:device_type",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:pybind_error_utils",
        "//csrc/common:shape",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/distributed:handshake",
        "//csrc/eager:device_buffer",
        "//csrc/eager:device_gen_impl",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/internal/dynamism:dynamism_ops",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//csrc/ops:scan_builder",
        "//csrc/ops/view_decomposition:contiguous_to_view",
        "//csrc/ops/view_decomposition:decomposition",
        "//csrc/ops/view_decomposition:strided_layout",
        "//csrc/pjrt:pjrt_state",
        "//shims/pybind11",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@tsl//tsl/platform:fingerprint",
        "@xla//xla:shape_util",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@xla//xla/mlir/utils:error_util",
        "@xla//xla/pjrt:maybe_owning_mlir_module",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",  # buildcleaner: keep
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
        "@xla//xla/service:device_assignment",
    ],
)

torch_tpu_cc_library(
    name = "dispatch_scan",
    srcs = ["dispatch_scan.cc"],
    hdrs = ["dispatch_scan.h"],
    deps = [
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/common:utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//csrc/ops:scan_builder",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
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
    requires_libtpu = True,
    deps = [
        ":dispatch_scan",
        "//csrc/common:cache_key",
        "//csrc/common:compilation",
        "//csrc/common:context_states",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/common:fingerprint_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:status_test_utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:eager_mode",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:scan_builder",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
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
        "@xla//xla/mlir/utils:error_util",
    ],
)

torch_tpu_package_end()
