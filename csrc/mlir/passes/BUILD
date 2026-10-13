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
    name = "dummy_pass",
    srcs = ["dummy_pass.cc"],
    hdrs = ["dummy_pass.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

torch_tpu_cc_library(
    name = "fold_concat_softmax",
    srcs = ["fold_concat_softmax.cc"],
    hdrs = ["fold_concat_softmax.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_library(
    name = "fuse_rotary_and_view_transpose",
    srcs = ["fuse_rotary_and_view_transpose.cc"],
    hdrs = ["fuse_rotary_and_view_transpose.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_test(
    name = "fuse_rotary_and_view_transpose_test",
    srcs = ["fuse_rotary_and_view_transpose_test.cc"],
    deps = [
        ":fuse_rotary_and_view_transpose",
        "//csrc/common:oss_test_main",
        "//csrc/mlir:pipeline",
        "@com_google_absl//absl/status:status_matchers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_library(
    name = "lower_separable_gather_to_dot",
    srcs = ["lower_separable_gather_to_dot.cc"],
    hdrs = ["lower_separable_gather_to_dot.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_cc_test(
    name = "lower_separable_gather_to_dot_test",
    srcs = ["lower_separable_gather_to_dot_test.cc"],
    deps = [
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

torch_tpu_cc_library(
    name = "reassociate_norm_weights",
    srcs = ["reassociate_norm_weights.cc"],
    hdrs = ["reassociate_norm_weights.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
