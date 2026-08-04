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
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/build_files:build_defs.bzl", "TT_FRIENDS")
load("//shims/py_rules:strict.bzl", "py_library_providing_imports_info")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ] + TT_FRIENDS,
)

torch_tpu_py_library(
    name = "profiler_config",
    srcs = ["profiler_config.py"],
    deps = ["//shims/torch:pytorch"],
)

# We use py_library_providing_imports_info with profiler.modspec.json to explicitly
# re-export the submodules provided by :profiler_lib. This prevents PyStrictDeps
# violations at downstream use sites that import `from torch_tpu._internal.profiler import profiler`
# without requiring them to directly depend on :profiler_lib.
py_library_providing_imports_info(
    name = "profiler",
    srcs = [
        "__init__.py",
    ],
    lib_rule = torch_tpu_py_library,
    module_specs = [
        "profiler.modspec.json",
    ],
    deps = [
        ":profiler_config",
        ":profiler_lib",
    ],
)

torch_tpu_py_library(
    name = "profiler_lib",
    srcs = ["profiler_api.py"],
    deps = [
        ":_impl",
        ":_profiler_backend",
    ],
)

torch_tpu_py_library(
    name = "_impl",
    srcs = ["_impl.py"],
    deps = [
        ":_profiler_backend",
        "//shims/absl_py/logging",
    ],
)

torch_tpu_cc_library(
    name = "xprof_callback_handler",
    srcs = ["xprof_callback_handler.cc"],
    hdrs = ["xprof_callback_handler.h"],
    deps = [
        "//shims/torch",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
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
        "//torch_tpu/common:device_type",
        "//torch_tpu/common:env_vars",
        "//torch_tpu/common:error_utils",
        "//torch_tpu/common:utils",
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
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/common:error_utils",
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

pybind_extension(
    name = "_profiler_backend",
    srcs = [
        "profiler_api.cc",
    ],
    # Torch_tpu uses rules_pywrap in XLA for OSS builds:
    # https://github.com/openxla/xla/tree/main/third_party/py/rules_pywrap.
    # These args are required to link extensions against common libs.
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        ":tpu_profiler_plugin_cc",
        ":xprof_callback_handler",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/common:error_utils",
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
