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
    name = "triangular_aten_kernels",
    srcs = ["triangular_aten_kernels.cc"],
    hdrs = ["triangular_aten_kernels.h"],
    deps = [
        ":triangular",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/ops:op_names",
        "//torch_tpu/csrc/ops:unary_aten_kernels",
        "//torch_tpu/csrc/ops/macros:kernel",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

torch_tpu_cc_library(
    name = "triangular",
    srcs = ["triangular.cc"],
    hdrs = ["triangular.h"],
    deps = [
        "//torch_tpu/csrc/common:aten_utils",
        "//torch_tpu/csrc/common:dimension_types",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/ops:op_builder_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
