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
    name = "aten_vector_norm_kernels",
    srcs = ["aten_vector_norm_kernels.cc"],
    hdrs = ["aten_vector_norm_kernels.h"],
    deps = [
        ":pnorm",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:unary_aten_kernels",
        "//torch_tpu/ops/macros:kernel",
        "//torch_tpu/ops/reductions",
        "//torch_tpu/ops/reductions:reduction_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
    ],
)

torch_tpu_cc_library(
    name = "pnorm",
    srcs = ["pnorm.cc"],
    hdrs = ["pnorm.h"],
    deps = [
        "//torch_tpu/common:error_utils",
        "//torch_tpu/ops:binary",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops/reductions",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
