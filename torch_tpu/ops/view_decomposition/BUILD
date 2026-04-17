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
    name = "strided_layout",
    srcs = ["strided_layout.cc"],
    hdrs = ["strided_layout.h"],
    deps = [
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:to_string",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_test(
    name = "strided_layout_test",
    srcs = ["strided_layout_test.cc"],
    deps = [
        ":strided_layout",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_library(
    name = "view_primitives",
    srcs = [
        "bitcast_primitive.cc",
        "broadcast_primitive.cc",
        "pad_primitive.cc",
        "reshape_primitive.cc",
        "slice_primitive.cc",
        "transpose_primitive.cc",
        "unfold_primitive.cc",
        "view_sequence.cc",
    ],
    hdrs = [
        "bitcast_primitive.h",
        "broadcast_primitive.h",
        "conj_primitive.h",
        "pad_primitive.h",
        "reshape_primitive.h",
        "slice_primitive.h",
        "transpose_primitive.h",
        "unfold_primitive.h",
        "view_sequence.h",
    ],
    deps = [
        ":strided_layout",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:unary",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_test(
    name = "reshape_primitive_test",
    srcs = ["reshape_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_test(
    name = "transpose_primitive_test",
    srcs = ["transpose_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_test(
    name = "broadcast_primitive_test",
    srcs = ["broadcast_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_test(
    name = "slice_primitive_test",
    srcs = ["slice_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_test(
    name = "pad_primitive_test",
    srcs = ["pad_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_test(
    name = "bitcast_primitive_test",
    srcs = ["bitcast_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_test(
    name = "unfold_primitive_test",
    srcs = ["unfold_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_cc_test(
    name = "view_sequence_test",
    srcs = ["view_sequence_test.cc"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//shims/torch:aten_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla/tsl/platform:statusor",
    ],
)

torch_tpu_cc_library(
    name = "decomposition",
    srcs = ["decomposition.cc"],
    hdrs = ["decomposition.h"],
    deps = [
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_test(
    name = "decomposition_test",
    srcs = ["decomposition_test.cc"],
    deps = [
        ":decomposition",
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_library(
    name = "inversion",
    srcs = ["inversion.cc"],
    hdrs = ["inversion.h"],
    deps = [
        ":decomposition",
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_test(
    name = "inversion_test",
    srcs = ["inversion_test.cc"],
    deps = [
        ":inversion",
        ":strided_layout",
        ":view_primitives",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:shape",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_package_end()
