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
    name = "cumsum_aten_kernels",
    srcs = ["cumsum_aten_kernels.cc"],
    hdrs = ["cumsum_aten_kernels.h"],
    deps = [
        "cumsum",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:unary_aten_kernels",
        "//torch_tpu/ops/macros:kernel",
        "//torch_tpu/ops/resize:resize_aten_kernels",
        "@com_google_absl//absl/functional:bind_front",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_library(
    name = "cumsum",
    srcs = ["cumsum.cc"],
    hdrs = ["cumsum.h"],
    deps = [
        "//torch_tpu/common:error_utils",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:scan_builder",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
