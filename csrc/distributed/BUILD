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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "window_tpu",
    srcs = ["window_tpu.cc"],
    hdrs = ["window_tpu.h"],
    deps = [
        "//csrc/common:device_type",
        "//csrc/common:error_utils",
        "//csrc/common:macro_utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/pjrt:pjrt_utils",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "@com_google_absl//absl/status",
        "@xla//xla:future",
    ],
    alwayslink = True,
)

torch_tpu_cc_library(
    name = "process_group_tpu",
    srcs = ["process_group_tpu.cc"],
    hdrs = ["process_group_tpu.h"],
    deps = [
        ":allgather",
        ":allreduce",
        ":alltoall",
        ":reduce_scatter",
        ":types",
        ":utils",
        ":window_tpu",
        "//csrc/common:aten_utils",
        "//csrc/common:cache_key",
        "//csrc/common:device_type",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:fingerprint_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/common:flags",
        "//csrc/common:macro_utils",
        "//csrc/common:shape",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:materialize",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/internal/compile:torch_compile_utils",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/runtime:device_id",
    ],
    alwayslink = True,
)

torch_tpu_cc_test(
    name = "process_group_tpu_test",
    srcs = ["process_group_tpu_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    deps = [
        ":process_group_tpu",
        "//csrc/common:oss_test_main",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

torch_tpu_cc_library(
    name = "types",
    srcs = [
        "types.cc",
    ],
    hdrs = [
        "types.h",
    ],
    deps = [
        "//csrc/common:fingerprint_utils",
        "//csrc/common:to_string",
    ],
)

torch_tpu_cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    deps = [
        ":types",
        "//csrc/common:fingerprint_utils",
        "//csrc/common:oss_test_main",
    ],
)

torch_tpu_cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        ":types",
        "//csrc/common:aten_utils",
        "//csrc/common:context_manager",
        "//csrc/common:context_states",
        "//csrc/common:error_utils",
        "//csrc/common:to_string",
        "//csrc/eager:repeated_ops_heuristic",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_library(
    name = "allgather",
    srcs = [
        "allgather.cc",
    ],
    hdrs = [
        "allgather.h",
    ],
    deps = [
        ":types",
        ":utils",
        "//csrc/common:aten_utils",
        "//csrc/common:dimension_types",
        "//csrc/ops:op_builder_utils",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "allreduce",
    srcs = [
        "allreduce.cc",
    ],
    hdrs = [
        "allreduce.h",
    ],
    deps = [
        ":types",
        ":utils",
        "//csrc/common:to_string",
        "//csrc/ops:op_builder_utils",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "reduce_scatter",
    srcs = [
        "reduce_scatter.cc",
    ],
    hdrs = [
        "reduce_scatter.h",
    ],
    deps = [
        ":types",
        ":utils",
        "//csrc/common:aten_utils",
        "//csrc/common:dimension_types",
        "//csrc/ops:op_builder_utils",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "alltoall",
    srcs = [
        "alltoall.cc",
    ],
    hdrs = [
        "alltoall.h",
    ],
    deps = [
        ":types",
        ":utils",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/common:utils",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops/experimental/ragged_all_to_all:ragged_all_to_all_builder",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "handshake",
    srcs = ["handshake.cc"],
    hdrs = ["handshake.h"],
    deps = [
        "//csrc/common:env_vars",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

torch_tpu_cc_test(
    name = "handshake_test",
    srcs = ["handshake_test.cc"],
    deps = [
        ":handshake",
        "//csrc/common:env_vars",
        "//csrc/common:oss_test_main",
    ],
)

torch_tpu_package_end()
