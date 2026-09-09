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
    name = "normal_aten_kernels",
    srcs = ["normal_aten_kernels.cc"],
    hdrs = ["normal_aten_kernels.h"],
    deps = [
        "//csrc/common:aten_utils",
        "//csrc/common:cache_key",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:device_gen_impl",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//csrc/ops:rng_utils",
        "//csrc/ops:xla_utils",
        "//csrc/ops/macros:kernel",
        "//csrc/ops/min_max",
        "//csrc/ops/min_max:min_max_aten_kernels",
        "//csrc/ops/resize:resize_aten_kernels",
        "//csrc/ops/uniform",
        "//csrc/ops/view:view_aten_kernels",
        "//csrc/pjrt:pjrt_utils",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:chlo_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_package_end()
