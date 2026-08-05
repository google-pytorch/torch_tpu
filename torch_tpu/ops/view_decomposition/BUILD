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
        "//shims/torch:aten_headers",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:to_string",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_test(
    name = "strided_layout_test",
    srcs = ["strided_layout_test.cc"],
    deps = [
        ":strided_layout",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "view_primitive_error_utils",
    hdrs = ["view_primitive_error_utils.h"],
    deps = [
        ":strided_layout",
        "//torch_tpu/common:to_string",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_test(
    name = "view_primitive_error_utils_test",
    srcs = ["view_primitive_error_utils_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    deps = [
        ":reshape_primitive",
        ":strided_layout",
        ":view_primitive_error_utils",
        ":view_sequence",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_library(
    name = "reshape_primitive",
    srcs = ["reshape_primitive.cc"],
    hdrs = ["reshape_primitive.h"],
    deps = [
        ":strided_layout",
        ":view_primitive_error_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@stablehlo//:mlir_builder",
    ],
)

torch_tpu_cc_test(
    name = "reshape_primitive_test",
    srcs = ["reshape_primitive_test.cc"],
    deps = [
        ":reshape_primitive",
        ":strided_layout",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "transpose_primitive",
    srcs = ["transpose_primitive.cc"],
    hdrs = ["transpose_primitive.h"],
    deps = [
        ":strided_layout",
        ":view_primitive_error_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:to_string",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_test(
    name = "transpose_primitive_test",
    srcs = ["transpose_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":transpose_primitive",
        "//torch_tpu/common:oss_test_main",
    ],
)

torch_tpu_cc_library(
    name = "broadcast_primitive",
    srcs = ["broadcast_primitive.cc"],
    hdrs = ["broadcast_primitive.h"],
    deps = [
        ":strided_layout",
        ":view_primitive_error_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
    ],
)

torch_tpu_cc_test(
    name = "broadcast_primitive_test",
    srcs = ["broadcast_primitive_test.cc"],
    deps = [
        ":broadcast_primitive",
        ":strided_layout",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "slice_primitive",
    srcs = ["slice_primitive.cc"],
    hdrs = ["slice_primitive.h"],
    deps = [
        ":strided_layout",
        ":view_primitive_error_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_broadcast_lowering",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_test(
    name = "slice_primitive_test",
    srcs = ["slice_primitive_test.cc"],
    deps = [
        ":slice_primitive",
        ":strided_layout",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "pad_primitive",
    srcs = ["pad_primitive.cc"],
    hdrs = ["pad_primitive.h"],
    deps = [
        ":strided_layout",
        ":view_primitive_error_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_test(
    name = "pad_primitive_test",
    srcs = ["pad_primitive_test.cc"],
    deps = [
        ":pad_primitive",
        ":strided_layout",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_cc_library(
    name = "conj_primitive",
    hdrs = ["conj_primitive.h"],
    deps = [
        ":strided_layout",
        "@com_google_absl//absl/status:statusor",
    ],
)

torch_tpu_cc_library(
    name = "bitcast_primitive",
    srcs = ["bitcast_primitive.cc"],
    hdrs = ["bitcast_primitive.h"],
    deps = [
        ":strided_layout",
        ":view_primitive_error_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:static_shape_check",
        "//torch_tpu/common:to_string",
        "//torch_tpu/common:utils",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_test(
    name = "bitcast_primitive_test",
    srcs = ["bitcast_primitive_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    deps = [
        ":bitcast_primitive",
        ":strided_layout",
        "//torch_tpu/common:oss_test_main",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_builder",
        "@xla//xla/hlo/testlib:filecheck",
    ],
)

torch_tpu_cc_library(
    name = "unfold_primitive",
    srcs = ["unfold_primitive.cc"],
    hdrs = ["unfold_primitive.h"],
    deps = [
        ":strided_layout",
        ":view_primitive_error_utils",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_test(
    name = "unfold_primitive_test",
    srcs = ["unfold_primitive_test.cc"],
    deps = [
        ":strided_layout",
        ":unfold_primitive",
        "//torch_tpu/common:oss_test_main",
    ],
)

torch_tpu_cc_library(
    name = "view_sequence",
    srcs = ["view_sequence.cc"],
    hdrs = ["view_sequence.h"],
    deps = [
        ":bitcast_primitive",
        ":broadcast_primitive",
        ":conj_primitive",
        ":pad_primitive",
        ":reshape_primitive",
        ":slice_primitive",
        ":strided_layout",
        ":transpose_primitive",
        ":unfold_primitive",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:unary",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@stablehlo//:mlir_builder",
    ],
)

torch_tpu_cc_test(
    name = "view_sequence_test",
    srcs = ["view_sequence_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    deps = [
        ":bitcast_primitive",
        ":broadcast_primitive",
        ":conj_primitive",
        ":pad_primitive",
        ":reshape_primitive",
        ":slice_primitive",
        ":strided_layout",
        ":transpose_primitive",
        ":unfold_primitive",
        ":view_sequence",
        "//shims/torch:aten_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:fingerprint_utils",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla/tsl/platform:statusor",
    ],
)

torch_tpu_cc_library(
    name = "decomposition",
    srcs = ["decomposition.cc"],
    hdrs = ["decomposition.h"],
    deps = [
        ":bitcast_primitive",
        ":broadcast_primitive",
        ":conj_primitive",
        ":pad_primitive",
        ":reshape_primitive",
        ":slice_primitive",
        ":strided_layout",
        ":transpose_primitive",
        ":unfold_primitive",
        ":view_sequence",
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
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    deps = [
        ":decomposition",
        ":strided_layout",
        ":view_sequence",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:oss_test_main",
        "//torch_tpu/common:to_string",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_library(
    name = "inversion",
    srcs = ["inversion.cc"],
    hdrs = ["inversion.h"],
    deps = [
        ":bitcast_primitive",
        ":broadcast_primitive",
        ":conj_primitive",
        ":decomposition",
        ":pad_primitive",
        ":reshape_primitive",
        ":slice_primitive",
        ":strided_layout",
        ":transpose_primitive",
        ":unfold_primitive",
        ":view_primitive_error_utils",
        ":view_sequence",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
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
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    deps = [
        ":bitcast_primitive",
        ":conj_primitive",
        ":inversion",
        ":pad_primitive",
        ":reshape_primitive",
        ":slice_primitive",
        ":strided_layout",
        ":transpose_primitive",
        ":view_sequence",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:oss_test_main",
        "//torch_tpu/common:shape",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
    ],
)

torch_tpu_cc_library(
    name = "contiguous_to_view",
    srcs = ["contiguous_to_view.cc"],
    hdrs = ["contiguous_to_view.h"],
    deps = [
        ":decomposition",
        ":strided_layout",
        "//shims/torch:aten_headers",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/common:utils",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:device_buffer_utils",
        "//torch_tpu/eager:tensor_to_buffer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_package_end()
