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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "copy_from_aten_kernels",
    srcs = ["copy_from_aten_kernels.cc"],
    hdrs = ["copy_from_aten_kernels.h"],
    deps = [
        ":cpu_to_tpu",
        ":tpu_to_cpu",
        ":tpu_to_tpu",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:aten_utils",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:device_type",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops/macros:kernel",
        "//torch_tpu/ops/resize:resize_aten_kernels",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

torch_tpu_cc_library(
    name = "tpu_to_tpu",
    srcs = ["tpu_to_tpu.cc"],
    hdrs = ["tpu_to_tpu.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/common:cache_key",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
    ],
)

torch_tpu_cc_library(
    name = "cpu_to_tpu",
    srcs = ["cpu_to_tpu.cc"],
    hdrs = ["cpu_to_tpu.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:device_buffer_utils",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/pjrt:pjrt_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:attr_type_builder_util",
        "@xla//xla:xla_data_proto_cc",
    ],
)

torch_tpu_cc_library(
    name = "tpu_to_cpu",
    srcs = ["tpu_to_cpu.cc"],
    hdrs = ["tpu_to_cpu.h"],
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:materialize",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:structured_log_buffer",
        "//torch_tpu/eager:tensor_to_buffer",
        "//torch_tpu/ops:op_builder_utils",
        "//torch_tpu/ops:op_names",
        "//torch_tpu/ops:python_context",
        "//torch_tpu/ops:xla_utils",
        "//torch_tpu/pjrt:pjrt_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@stablehlo//:attr_type_builder_util",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

torch_tpu_package_end()
