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
    name = "nonzero_aten_kernels",
    srcs = ["nonzero_aten_kernels.cc"],
    hdrs = ["nonzero_aten_kernels.h"],
    deps = [
        ":nonzero",
        "//csrc/common:cache_key",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:unary_aten_kernels",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_library(
    name = "nonzero",
    srcs = ["nonzero.cc"],
    hdrs = ["nonzero.h"],
    deps = [
        "//csrc/common:aten_utils",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops/reductions",
        "//csrc/ops/sort",
        "//csrc/ops/stride:stride_helper",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
