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
    name = "all_any_aten_kernels",
    srcs = ["all_any_aten_kernels.cc"],
    hdrs = ["all_any_aten_kernels.h"],
    deps = [
        ":all_any",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/csrc/common:cache_key",
        "//torch_tpu/csrc/common:dimension_types",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/ops:op_builder_utils",
        "//torch_tpu/csrc/ops:op_names",
        "//torch_tpu/csrc/ops:unary_aten_kernels",
        "//torch_tpu/csrc/ops/macros:kernel",
        "//torch_tpu/csrc/ops/reductions",
        "//torch_tpu/csrc/ops/reductions:reduction_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
    ],
)

torch_tpu_cc_library(
    name = "all_any",
    srcs = ["all_any.cc"],
    hdrs = ["all_any.h"],
    deps = [
        "//torch_tpu/csrc/ops:op_builder_utils",
        "//torch_tpu/csrc/ops/reductions",
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
