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
load("//shims/build_files:build_defs.bzl", "TT_FRIENDS")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ] + TT_FRIENDS,
)

exports_files(["profiler_api.cc"])

torch_tpu_cc_library(
    name = "xprof_callback_handler",
    srcs = ["xprof_callback_handler.cc"],
    hdrs = ["xprof_callback_handler.h"],
    deps = [
        "//shims/torch",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/csrc/common:context_manager",
        "//torch_tpu/csrc/common:context_states",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
    ],
)

torch_tpu_cc_library(
    name = "tpu_profiler_plugin_cc",
    srcs = ["tpu_profiler_plugin.cc"],
    hdrs = [
        "tpu_profiler_plugin.h",
    ],
    deps = [
        ":xprof_callback_handler",
        "//shims/torch:kineto",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//torch_tpu/csrc/common:device_type",
        "//torch_tpu/csrc/common:env_vars",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/common:utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:env",
    ],
)

# This library is not used directly, but is needed for go/cclean and Clangd.
# Without this rule, cclean complains that profiler.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "profiler_api_cc",
    srcs = [
        "profiler_api.cc",
    ],
    deps = [
        "//shims/pybind11",
        "//torch_tpu/csrc/common:context_manager",
        "//torch_tpu/csrc/common:context_states",
        "//torch_tpu/csrc/common:error_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/backends/profiler/cpu:host_tracer",
        "@xla//xla/backends/profiler/cpu:python_tracer",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform/cloud:gcs_file_system",
        "@xla//xla/tsl/profiler/rpc:profiler_server_impl",
    ],
)

torch_tpu_package_end()
