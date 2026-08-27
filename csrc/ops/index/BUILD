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
    name = "index_aten_kernels",
    srcs = ["index_aten_kernels.cc"],
    hdrs = ["index_aten_kernels.h"],
    deps = [
        ":index",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "//torch_tpu/csrc/common:aten_utils",
        "//torch_tpu/csrc/common:cache_key",
        "//torch_tpu/csrc/common:dimension_types",
        "//torch_tpu/csrc/common:dtype",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/common:to_string",
        "//torch_tpu/csrc/eager:device_buffer",
        "//torch_tpu/csrc/eager:op_dispatcher",
        "//torch_tpu/csrc/eager:tensor_to_buffer",
        "//torch_tpu/csrc/ops:index_utils",
        "//torch_tpu/csrc/ops:op_builder_utils",
        "//torch_tpu/csrc/ops:op_names",
        "//torch_tpu/csrc/ops/macros:kernel",
        "//torch_tpu/csrc/ops/masked_select:masked_select_aten_kernels",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
    ],
)

torch_tpu_cc_library(
    name = "index",
    srcs = ["index.cc"],
    hdrs = ["index.h"],
    deps = [
        "//torch_tpu/csrc/common:aten_utils",
        "//torch_tpu/csrc/common:dimension_types",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_broadcast_lowering",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
