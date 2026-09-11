# Copyright 2026 Google LLC
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

load(
    "//build_files:build_defs.bzl",
    "if_oss",
    "torch_tpu_cc_library",
    "torch_tpu_cc_test",
    "torch_tpu_package_end",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "scaled_mm_aten_kernels",
    srcs = ["scaled_mm_aten_kernels.cc"],
    hdrs = ["scaled_mm_aten_kernels.h"],
    deps = [
        "//csrc/common:aten_utils",
        "//csrc/common:cache_key",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:macro_utils",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:custom_kernels",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:precision_context",
        "//csrc/ops/macros:kernel",
        "//csrc/ops/scaled_mm/kernels:blockwise_scaled_mm_e4m3_embed",
        "//csrc/ops/scaled_mm/kernels:blockwise_scaled_mm_e5m2_embed",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
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

torch_tpu_cc_test(
    name = "scaled_mm_aten_kernels_test",
    srcs = ["scaled_mm_aten_kernels_test.cc"],
    exec_properties = {"cpp_link.mem": "20g"},
    tags = if_oss(["requires-tpu-v5lite"]),
    deps = [
        ":scaled_mm_aten_kernels",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:status_test_utils",
        "//csrc/ops:op_builder_utils",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch",  # buildcleaner: keep
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:func_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_test(
    name = "scaled_mm_aten_kernels_tpu_test",
    srcs = ["scaled_mm_aten_kernels_test.cc"],
    exec_properties = {"cpp_link.mem": "20g"},
    nobuild_oss = True,  # NOBUILD_OSS_OK=scaled_mm_aten_kernels_test already builds and runs on TPU in OSS.
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":scaled_mm_aten_kernels",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:status_test_utils",
        "//csrc/ops:op_builder_utils",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch",  # buildcleaner: keep
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:func_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
