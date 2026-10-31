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
# willfroom
# tudorpip
# ghpvnist

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "scaled_dot_product_attention_aten_kernels",
    srcs = ["scaled_dot_product_attention_aten_kernels.cc"],
    hdrs = ["scaled_dot_product_attention_aten_kernels.h"],
    deps = [
        ":flash_attention_backward_kernel",
        ":flash_attention_kernel",
        ":flash_attention_kernel_builder",
        ":helpers",
        ":scaled_dot_product_attention_shlo",
        "//csrc/common:cache_key",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/common:flags",
        "//csrc/common:warnings",
        "//csrc/eager:device_buffer",
        "//csrc/eager:eager_mode",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/internal/mosaic:op_builders",
        "//csrc/ops:binary",
        "//csrc/ops:custom_kernels",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//csrc/ops/reductions",
        "//csrc/ops/reductions:sum",
        "//csrc/ops/view_decomposition:contiguous_to_view",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
        "@stablehlo//stablehlo/integrations/cpp/builder:stablehlo_builder",
        "@xla//xla/pjrt:mlir_to_hlo",
    ],
)

torch_tpu_cc_library(
    name = "helpers",
    srcs = ["helpers.cc"],
    hdrs = ["helpers.h"],
    visibility = ["//visibility:private"],  # Internal use only.
    deps = [
        "//csrc/common:dimension_types",
        "//csrc/common:utils",
        "//csrc/ops:op_builder_utils",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/util:base_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
        "@stablehlo//stablehlo/integrations/cpp/builder:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "scaled_dot_product_attention_shlo",
    srcs = ["scaled_dot_product_attention_shlo.cc"],
    hdrs = ["scaled_dot_product_attention_shlo.h"],
    deps = [
        ":helpers",
        "//csrc/common:cache_key",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/common:utils",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:precision_context",
        "//csrc/ops/view_decomposition:contiguous_to_view",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":flash_attention_config",
        "//csrc/common:error_utils",
        "//csrc/internal/mosaic/kiln",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
    ],
)

torch_tpu_cc_library(
    name = "flash_attention_config",
    hdrs = ["flash_attention_config.h"],
    deps = [
        "//csrc/internal/mosaic/kiln",
        "@llvm-project//llvm:Support",
    ],
)

torch_tpu_cc_library(
    name = "flash_attention_kernel",
    srcs = ["flash_attention_kernel.cc"],
    hdrs = ["flash_attention_kernel.h"],
    deps = [
        ":flash_attention_config",
        ":util",
        "//csrc/common:error_utils",
        "//csrc/internal/mosaic/kiln",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
    ],
)

torch_tpu_cc_library(
    name = "flash_attention_backward_kernel",
    srcs = ["flash_attention_backward_kernel.cc"],
    hdrs = ["flash_attention_kernel.h"],
    deps = [
        ":flash_attention_config",
        ":flash_attention_kernel",
        ":util",
        "//csrc/common:error_utils",
        "//csrc/internal/mosaic/kiln",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
    ],
)

torch_tpu_cc_library(
    name = "flash_attention_kernel_builder",
    srcs = ["flash_attention_kernel_builder.cc"],
    hdrs = ["flash_attention_kernel_builder.h"],
    deps = [
        ":flash_attention_backward_kernel",  # buildcleaner: keep
        ":flash_attention_config",
        ":flash_attention_kernel",
        ":helpers",
        "//csrc/common:cache_key",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/common:utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/internal/mosaic:kiln_custom_call",
        "//csrc/internal/mosaic/kiln",
        "//csrc/ops:binary",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:precision_context",
        "//csrc/ops:rng_utils",
        "//csrc/ops/reductions",
        "//csrc/ops/reductions:sum",
        "//csrc/ops/view_decomposition:contiguous_to_view",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
        "@stablehlo//stablehlo/integrations/cpp/builder:stablehlo_builder",
    ],
)

torch_tpu_package_end()
