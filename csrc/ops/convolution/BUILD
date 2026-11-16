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

# Domain experts for this package:
# cbasile-g
# salkan0

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "convolution_checks",
    srcs = ["convolution_checks.cc"],
    hdrs = ["convolution_checks.h"],
    deps = [
        "//csrc/common:error_utils",
        "//csrc/common:to_string",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "convolution",
    srcs = ["convolution.cc"],
    hdrs = ["convolution.h"],
    deps = [
        ":convolution_checks",
        "//csrc/common:aten_utils",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/common:to_string",
        "//csrc/ops:op_builder_utils",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_broadcast_lowering",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
        "@stablehlo//stablehlo/integrations/cpp/builder:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "convolution_aten_kernels",
    srcs = ["convolution_aten_kernels.cc"],
    hdrs = ["convolution_aten_kernels.h"],
    deps = [
        ":convolution",
        ":convolution_checks",
        "//csrc/common:aten_utils",
        "//csrc/common:cache_key",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:eager_mode",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:precision_context",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
    ],
)

torch_tpu_package_end()
