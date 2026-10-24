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

# Domain experts for this package:
# mkhanna15
# arfaian

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
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/eager:device_buffer",
        "//csrc/eager:eager_mode",
        "//csrc/eager:materialize",
        "//csrc/eager:split_traversal",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//shims/torch:aten_headers",
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
        "//csrc/common:device_type",
        "//csrc/common:error_utils",
        "//csrc/common:pybind_error_utils",
        "//csrc/common:shape",
        "//shims/pybind11",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
    ],
)

torch_tpu_cc_library(
    name = "dynamism_ops",
    srcs = ["dynamism_ops.cc"],
    hdrs = ["dynamism_ops.h"],
    deps = [
        "//csrc/common:aten_utils",
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/common:to_string",
        "//csrc/ops:op_builder_utils",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_broadcast_lowering",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@stablehlo//stablehlo/integrations/cpp/builder:func_builder",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
        "@stablehlo//stablehlo/integrations/cpp/builder:stablehlo_builder",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_test(
    name = "dynamism_ops_test",
    srcs = ["dynamism_ops_test.cc"],
    deps = [
        ":dynamism_ops",
        "//csrc/common:dimension_types",
        "//csrc/common:oss_test_main",
        "//csrc/common:shape",
        "//csrc/common:status_test_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@stablehlo//:register",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
    ],
)

torch_tpu_package_end()
