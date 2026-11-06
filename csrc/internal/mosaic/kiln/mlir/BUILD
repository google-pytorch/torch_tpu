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

# Escape hatches from Kiln into raw MLIR; not part of the main Kiln API (kiln.h).

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "stablehlo_splice",
    srcs = ["stablehlo_splice.cc"],
    hdrs = ["stablehlo_splice.h"],
    deps = [
        "//csrc/common:error_utils",
        "//csrc/internal/mosaic/kiln/core:context",
        "//csrc/internal/mosaic/kiln/core:dtype",
        "//csrc/internal/mosaic/kiln/core:tile",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
    ],
)

torch_tpu_package_end()
