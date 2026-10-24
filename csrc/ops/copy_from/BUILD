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
# benawilson
# basioli-k

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
    name = "copy_from_aten_kernels",
    srcs = ["copy_from_aten_kernels.cc"],
    hdrs = ["copy_from_aten_kernels.h"],
    deps = [
        ":copy_common",
        ":cpu_to_tpu",
        ":tpu_to_cpu",
        ":tpu_to_tpu",
        "//csrc/common:aten_utils",
        "//csrc/common:cache_key",
        "//csrc/common:device_type",
        "//csrc/common:error_utils",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//csrc/ops/resize:resize_aten_kernels",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

torch_tpu_cc_library(
    name = "tpu_to_tpu",
    srcs = ["tpu_to_tpu.cc"],
    hdrs = ["tpu_to_tpu.h"],
    deps = [
        ":copy_common",
        "//csrc/common:cache_key",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
        "@stablehlo//stablehlo/integrations/cpp/builder:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "cpu_to_tpu",
    srcs = ["cpu_to_tpu.cc"],
    hdrs = ["cpu_to_tpu.h"],
    deps = [
        ":copy_common",
        "//csrc/common:error_utils",
        "//csrc/common:layout_utils",
        "//csrc/common:to_string",
        "//csrc/eager:device_buffer",
        "//csrc/eager:events_queue",
        "//csrc/eager:materialize",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/pjrt:pjrt_transfer",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_library(
    name = "tpu_to_cpu",
    srcs = ["tpu_to_cpu.cc"],
    hdrs = ["tpu_to_cpu.h"],
    deps = [
        ":copy_common",
        ":tpu_to_tpu",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:reroute_warnings",
        "//csrc/common:status_builder",
        "//csrc/common:to_string",
        "//csrc/common:utils",
        "//csrc/eager:current_stream",
        "//csrc/eager:device_buffer",
        "//csrc/eager:events_queue",
        "//csrc/eager:materialize",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:split_traversal",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//csrc/ops:xla_utils",
        "//csrc/ops/as_strided:as_strided_aten_kernels",
        "//csrc/pjrt:pjrt_transfer",
        "//csrc/pjrt:pjrt_utils",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:future",
    ],
)

torch_tpu_cc_library(
    name = "copy_common",
    srcs = ["copy_common.cc"],
    hdrs = ["copy_common.h"],
    deps = [
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/eager:eager_mode",
        "//csrc/ops/as_strided:as_strided_aten_kernels",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_cc_test(
    name = "copy_common_test",
    srcs = ["copy_common_test.cc"],
    requires_libtpu = True,
    deps = [
        ":copy_common",
        "//csrc/common:oss_test_main",
        "//csrc/eager:eager_mode",
    ],
)

torch_tpu_package_end()
