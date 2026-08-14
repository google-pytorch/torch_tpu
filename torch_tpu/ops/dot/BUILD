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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "dot",
    srcs = ["dot.cc"],
    hdrs = ["dot.h"],
    deps = [
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_library(
    name = "dot_checks",
    srcs = ["dot_checks.cc"],
    hdrs = ["dot_checks.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_library(
    name = "dot_aten_kernels",
    srcs = ["dot_aten_kernels.cc"],
    hdrs = ["dot_aten_kernels.h"],
    deps = [
        ":dot",
        ":dot_checks",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:fixed_size_span",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:precision_context",
        "//torch_tpu/ops/macros:kernel",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:mlir_builder",
    ],
)

torch_tpu_cc_library(
    name = "vdot_aten_kernels",
    srcs = ["vdot_aten_kernels.cc"],
    hdrs = ["vdot_aten_kernels.h"],
    deps = [
        ":dot",
        ":dot_checks",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:fixed_size_span",
        "//torch_tpu/common:to_string",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:precision_context",
        "//torch_tpu/ops:unary",
        "//torch_tpu/ops/macros:kernel",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
