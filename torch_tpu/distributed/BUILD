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
        "//shims/torch",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:device_type",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:env_vars",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:fingerprint_utils",
        "//torch_tpu/common:flags",
        "//torch_tpu/common:shape",
        "//torch_tpu/common:to_string",
        "//torch_tpu/common:utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:materialize",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:structured_log_buffer",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/macros:kernel",
        "//torch_tpu/pjrt:pjrt_state",
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

torch_tpu_cc_library(
    name = "types",
    srcs = [
        "types.cc",
    ],
    hdrs = [
        "types.h",
    ],
    deps = [
        "//torch_tpu/common:fingerprint_utils",
        "//torch_tpu/common:to_string",
    ],
)

torch_tpu_cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    deps = [
        ":types",
        "//torch_tpu/common:fingerprint_utils",
        "//torch_tpu/common:oss_test_main",
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
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/eager:repeated_ops_heuristic",
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
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/ops:op_builder_utils",
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
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
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
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/ops:op_builder_utils",
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
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_package_end()
