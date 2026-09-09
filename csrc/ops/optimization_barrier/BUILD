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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "optimization_barrier_kernels",
    srcs = ["optimization_barrier_kernels.cc"],
    hdrs = ["optimization_barrier_kernels.h"],
    deps = [
        "//csrc/common:cache_key",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//csrc/ops/view_decomposition:contiguous_to_view",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_package_end()
