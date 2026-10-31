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
# haifeng-jin
# zhanyong-wan
# cbasile-g

# Defines a library that wraps the internal PJRT client.
# This is a workaround to avoid depending on internal components
# directly from the main torch_tpu library.

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

# This acts as an adapter between the PJRT client and the torch_tpu
# library. It allows us to use the PJRT client without depending on
# internal components directly, which are not open-sourced.
torch_tpu_cc_library(
    name = "pjrt",
    srcs = [
        "pjrt_client.cc",
    ],
    hdrs = [
        "pjrt_client.h",
    ],
    deps = [
        "//csrc/common:error_utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_cc_library(
    name = "pjrt_state",
    srcs = ["pjrt_state.cc"],
    hdrs = ["pjrt_state.h"],
    deps = [
        ":pjrt",
        "//csrc/common:contain",
        "//csrc/common:device_type",
        "//csrc/common:discovery",
        "//csrc/common:env_vars",
        "//csrc/common:environment",
        "//csrc/common:error_utils",
        "//csrc/common:utils",
        "//csrc/distributed/slicebuilder:discovery",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@xla//xla/backends/profiler/plugin:plugin_tracer",
        "@xla//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "@xla//xla/pjrt:host_memory_allocator",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_device_description",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "@xla//xla/pjrt/plugin:plugin_names",
        "@xla//xla/tsl/framework:allocator",
    ],
)

torch_tpu_cc_test(
    name = "pjrt_init_test",
    srcs = ["pjrt_init_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":pjrt_state",
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:status_test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

torch_tpu_cc_library(
    name = "pjrt_utils",
    srcs = [
        "pjrt_utils.cc",
    ],
    hdrs = [
        "pjrt_utils.h",
    ],
    deps = [
        ":pjrt_state",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_cc_library(
    name = "pjrt_execute",
    srcs = [
        "pjrt_execute.cc",
    ],
    hdrs = [
        "pjrt_execute.h",
    ],
    deps = [
        ":pjrt_utils",
        "//csrc/common:compilation",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:absl_vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:future",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/service:collective_ops_utils",
    ],
)

torch_tpu_cc_library(
    name = "pjrt_transfer",
    srcs = [
        "pjrt_transfer.cc",
    ],
    hdrs = [
        "pjrt_transfer.h",
    ],
    deps = [
        ":pjrt_utils",
        "//csrc/common:context_manager",
        "//csrc/common:context_states",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/common:utils",
        "//csrc/eager:current_stream",
        "//csrc/eager:device_buffer",
        "//csrc/eager:host_allocator",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:absl_vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@tsl//tsl/platform:path",
        "@xla//xla:future",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_status_utils",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
    ],
)

torch_tpu_cc_test(
    name = "pjrt_execute_test",
    srcs = ["pjrt_execute_test.cc"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    requires_libtpu = True,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":pjrt_execute",
        ":pjrt_state",
        "//csrc/common:compilation",
        "//csrc/common:error_utils",
        "//csrc/common:mlir_test_modules",
        "//csrc/common:oss_test_main",
        "//csrc/common:shape",
        "//csrc/common:status_test_utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
    ],
)

torch_tpu_package_end()
