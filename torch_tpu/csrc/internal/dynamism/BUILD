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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files(["dynamism_py.cc"])

torch_tpu_cc_library(
    name = "dynamism_cc",
    srcs = [
        "dynamism.cc",
    ],
    hdrs = [
        "dynamism.h",
    ],
    deps = [
        "//shims/torch:aten_headers",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/common:shape",
        "//torch_tpu/csrc/eager:device_buffer",
        "//torch_tpu/csrc/eager:tensor_to_buffer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "dynamism_py_cc",
    srcs = ["dynamism_py.cc"],
    deps = [
        ":dynamism_cc",
        "//shims/pybind11",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/csrc/common:device_type",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/common:shape",
    ],
)

torch_tpu_cc_library(
    name = "dynamism_ops",
    srcs = ["dynamism_ops.cc"],
    hdrs = ["dynamism_ops.h"],
    deps = [
        "//shims/torch:torch_headers",
        "//torch_tpu/csrc/common:aten_utils",
        "//torch_tpu/csrc/common:dimension_types",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/common:shape",
        "//torch_tpu/csrc/common:to_string",
        "//torch_tpu/csrc/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:func_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_broadcast_lowering",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_test(
    name = "dynamism_ops_test",
    srcs = ["dynamism_ops_test.cc"],
    deps = [
        ":dynamism_ops",
        "//torch_tpu/csrc/common:dimension_types",
        "//torch_tpu/csrc/common:oss_test_main",
        "//torch_tpu/csrc/common:shape",
        "//torch_tpu/csrc/common:status_test_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:register",
    ],
)

torch_tpu_package_end()
