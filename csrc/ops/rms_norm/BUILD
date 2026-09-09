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
    name = "rms_norm",
    srcs = ["rms_norm.cc"],
    hdrs = ["rms_norm.h"],
    deps = [
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/common:to_string",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops/layer_norm",
        "//csrc/ops/reductions",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_library(
    name = "rms_norm_aten_kernels",
    srcs = ["rms_norm_aten_kernels.cc"],
    hdrs = ["rms_norm_aten_kernels.h"],
    deps = [
        ":rms_norm",
        "//csrc/common:aten_utils",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:to_string",
        "//csrc/eager:device_buffer",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/layer_norm",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
    ],
    alwayslink = True,
)

torch_tpu_package_end()
