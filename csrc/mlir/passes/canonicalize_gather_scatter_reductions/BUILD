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
    "torch_tpu_cc_library",
    "torch_tpu_cc_test",
    "torch_tpu_package_end",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "canonicalize_gather_scatter_reductions",
    srcs = ["canonicalize_gather_scatter_reductions.cc"],
    hdrs = ["canonicalize_gather_scatter_reductions.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_test(
    name = "canonicalize_gather_scatter_reductions_test",
    srcs = ["canonicalize_gather_scatter_reductions_test.cc"],
    deps = [
        ":canonicalize_gather_scatter_reductions",
        "//csrc/common:oss_test_main",
        "//csrc/mlir:pipeline",
        "@com_google_absl//absl/status:status_matchers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:reference_api",
        "@stablehlo//:reference_configuration",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
