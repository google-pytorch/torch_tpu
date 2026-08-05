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
        "//torch_tpu/common:error_utils",
        "//torch_tpu/distributed/slicebuilder:discovery",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_static_registration",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_cc_library(
    name = "pjrt_state",
    srcs = ["pjrt_state.cc"],
    hdrs = ["pjrt_state.h"],
    deps = [
        ":pjrt",
        "//torch_tpu/common:contain",
        "//torch_tpu/common:device_type",
        "//torch_tpu/common:env_vars",
        "//torch_tpu/common:environment",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/distributed/slicebuilder:discovery",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
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
    requires_libtpu = True,
    deps = [
        ":pjrt_state",
        "//torch_tpu/common:oss_test_main",
        "@com_google_absl//absl/status",
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
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/common:device_type",
        "//torch_tpu/common:dimension_types",
        "//torch_tpu/common:dtype",
        "//torch_tpu/common:env_vars",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:to_string",
        "//torch_tpu/eager:device_buffer",
        "//torch_tpu/eager:events_queue",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:absl_vlog_is_on",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@stablehlo//:attr_type_builder_util",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:future",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/service:collective_ops_utils",
    ],
)

torch_tpu_package_end()
