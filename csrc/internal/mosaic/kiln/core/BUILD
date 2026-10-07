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

# Core building blocks of the KILN kernel language.

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//csrc/internal/mosaic/kiln:__subpackages__"],
)

torch_tpu_cc_library(
    name = "context",
    srcs = ["context.cc"],
    hdrs = ["context.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":dtype",
        ":literal",
        ":value_base",
        "@com_google_absl//absl/log:absl_check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_library(
    name = "value_base",
    hdrs = ["value_base.h"],
    deps = [":dtype"],
)

torch_tpu_cc_library(
    name = "dtype",
    hdrs = ["dtype.h"],
    deps = ["@llvm-project//llvm:Support"],
)

torch_tpu_cc_library(
    name = "literal",
    hdrs = ["literal.h"],
)

torch_tpu_cc_library(
    name = "cond",
    srcs = ["cond.cc"],
    hdrs = ["cond.h"],
    deps = [
        ":context",
        ":dtype",
        ":value_base",
        "@com_google_absl//absl/log:absl_check",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_library(
    name = "elementwise",
    hdrs = ["elementwise.h"],
    deps = [":dtype"],
)

# Shared definitions of the elementwise ops. Only for the .cc files of handle
# types that explicitly instantiate `internal::ElementwiseOps`.
torch_tpu_cc_library(
    name = "elementwise_impl",
    srcs = ["elementwise_impl.cc"],
    hdrs = ["elementwise_impl.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":context",
        ":dtype",
        ":elementwise",
        "//csrc/common:libtpu_version",
        "//csrc/common:version",
        "@com_google_absl//absl/log:absl_check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

torch_tpu_cc_library(
    name = "scalar",
    srcs = ["scalar.cc"],
    hdrs = ["scalar.h"],
    deps = [
        ":cond",
        ":context",
        ":dtype",
        ":elementwise",
        ":elementwise_impl",
        ":literal",
        ":value_base",
        "@com_google_absl//absl/log:absl_check",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_library(
    name = "index_arg",
    hdrs = ["index_arg.h"],
    deps = [
        ":scalar",
        "@com_google_absl//absl/log:absl_check",
    ],
)

torch_tpu_cc_library(
    name = "control_flow",
    srcs = ["control_flow.cc"],
    hdrs = ["control_flow.h"],
    deps = [
        ":cond",
        ":context",
        ":dtype",
        ":index_arg",
        ":scalar",
        "@com_google_absl//absl/functional:function_ref",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
    ],
)

torch_tpu_cc_library(
    name = "tile",
    srcs = ["tile.cc"],
    hdrs = ["tile.h"],
    deps = [
        ":context",
        ":dtype",
        ":elementwise",
        ":elementwise_impl",
        ":index_arg",
        ":literal",
        ":scalar",
        ":value_base",
        "//csrc/internal/mosaic:op_builders",
        "@com_google_absl//absl/log:absl_check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:VectorDialect",
        "@xla//xla/mosaic:tpu_dialect",
    ],
)

torch_tpu_cc_library(
    name = "ref",
    srcs = ["ref.cc"],
    hdrs = ["ref.h"],
    deps = [
        ":context",
        ":dtype",
        ":index_arg",
        ":scalar",
        ":tile",
        ":value_base",
        "@com_google_absl//absl/log:absl_check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:VectorDialect",
        "@xla//xla/mosaic:tpu_dialect",
    ],
)

torch_tpu_cc_library(
    name = "index_map",
    hdrs = ["index_map.h"],
    deps = [
        ":index_arg",
        "@llvm-project//llvm:Support",
    ],
)

torch_tpu_cc_library(
    name = "specs",
    hdrs = ["specs.h"],
    deps = [
        ":dtype",
        ":index_map",
        "@llvm-project//llvm:Support",
    ],
)

torch_tpu_cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    deps = [
        ":context",
        ":dtype",
        ":index_arg",
        ":index_map",
        ":ref",
        ":scalar",
        ":specs",
        "//csrc/common:error_utils",
        "//csrc/internal/mosaic:op_builders",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_package_end()
