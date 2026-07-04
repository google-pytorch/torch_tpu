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
    name = "scaled_dot_product_attention_aten_kernels",
    srcs = ["scaled_dot_product_attention_aten_kernels.cc"],
    hdrs = ["scaled_dot_product_attention_aten_kernels.h"],
    deps = [
        ":helpers",
        ":scaled_dot_product_attention_shlo",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:fixed_size_span",
        "//torch_tpu/common:flags",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:custom_kernels",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/macros:kernel",
        "//torch_tpu/ops/scaled_dot_product_attention/kernels:sdpa_bwd_bf16_causal_mlir_embed",
        "//torch_tpu/ops/scaled_dot_product_attention/kernels:sdpa_bwd_bf16_non_causal_mlir_embed",
        "//torch_tpu/ops/scaled_dot_product_attention/kernels:sdpa_bwd_f32_causal_mlir_embed",
        "//torch_tpu/ops/scaled_dot_product_attention/kernels:sdpa_bwd_f32_non_causal_mlir_embed",
        "//torch_tpu/ops/scaled_dot_product_attention/kernels:sdpa_fwd_bf16_causal_mlir_embed",
        "//torch_tpu/ops/scaled_dot_product_attention/kernels:sdpa_fwd_bf16_non_causal_mlir_embed",
        "//torch_tpu/ops/scaled_dot_product_attention/kernels:sdpa_fwd_f32_causal_mlir_embed",
        "//torch_tpu/ops/scaled_dot_product_attention/kernels:sdpa_fwd_f32_non_causal_mlir_embed",
        "//torch_tpu/ops/view_decomposition:contiguous_to_view",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/pjrt:mlir_to_hlo",
    ],
)

torch_tpu_cc_library(
    name = "helpers",
    srcs = ["helpers.cc"],
    hdrs = ["helpers.h"],
    visibility = ["//visibility:private"],  # Internal use only.
    deps = [
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "scaled_dot_product_attention_shlo",
    srcs = ["scaled_dot_product_attention_shlo.cc"],
    hdrs = ["scaled_dot_product_attention_shlo.h"],
    deps = [
        ":helpers",
        "//shims/torch:aten_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:fixed_size_span",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops/view_decomposition:contiguous_to_view",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
