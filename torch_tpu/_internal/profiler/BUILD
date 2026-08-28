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
    "oss_target",
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
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils:annotations",
    ],
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

oss_target(
    torch_tpu_py_library,
    name = "xprof_adapter",
    srcs = ["xprof_adapter.py"],
)

pybind_extension(
    name = "_profiler_backend",
    srcs = ["//torch_tpu/csrc/internal/profiler:profiler_api.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    wrap_py_init = True,
    deps = [
        "//torch_tpu/csrc/common:context_manager",
        "//torch_tpu/csrc/common:context_states",
        "//torch_tpu/csrc/common:error_utils",
        "//torch_tpu/csrc/internal/profiler:tpu_profiler_plugin_cc",
        "//torch_tpu/csrc/internal/profiler:xprof_callback_handler",
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
