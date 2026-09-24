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
    name = "instrumentation",
    srcs = ["instrumentation.cc"],
    hdrs = ["instrumentation.h"],
    deps = [
        "//csrc/common:env_vars",
        "//csrc/common:utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

torch_tpu_cc_library(
    name = "pipeline",
    srcs = ["pipeline.cc"],
    hdrs = ["pipeline.h"],
    deps = [
        ":instrumentation",
        "//csrc/common:error_utils",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_passes_optimization",
        "@xla//xla/mlir/utils:error_util",
    ],
)

torch_tpu_cc_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.cc"],
    deps = [
        ":pipeline",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "@com_google_absl//absl/status:status_matchers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_library(
    name = "registration",
    srcs = ["registration.cc"],
    hdrs = ["registration.h"],
    deps = [
        "//csrc/mlir/passes:dummy_pass",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@stablehlo//:register",
    ],
)

torch_tpu_package_end()
