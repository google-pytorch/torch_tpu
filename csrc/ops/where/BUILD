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
    name = "where_aten_kernels",
    srcs = ["where_aten_kernels.cc"],
    hdrs = ["where_aten_kernels.h"],
    deps = [
        ":where",
        "//csrc/common:cache_key",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/common:to_string",
        "//csrc/eager:device_buffer",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//csrc/ops/resize:resize_aten_kernels",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_library(
    name = "where",
    srcs = ["where.cc"],
    hdrs = ["where.h"],
    deps = [
        "//csrc/common:error_utils",
        "//csrc/ops:op_builder_utils",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_package_end()
