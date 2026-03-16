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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end", "torch_tpu_py_library")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ],
)

torch_tpu_py_library(
    name = "profiler",
    srcs = [
        "__init__.py",
        "_impl.py",
        "profiler_api.py",
    ],
    deps = [
        ":_profiler_backend",
        "//shims/absl_py/logging",
    ],
)

# This library is not used directly, but is needed for go/cclean and Clangd.
# Without this rule, cclean complains that profiler.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "profiler_cc",
    srcs = [
        "profiler.cc",
    ],
    hdrs = [
        "profiler.h",
    ],
    deps = [
        "//shims/pybind11",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/log:absl_log",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/backends/profiler/cpu:host_tracer",
        "@xla//xla/backends/profiler/cpu:python_tracer",
        "@xla//xla/backends/profiler/tpu:tpu_tracer",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform/cloud:gcs_file_system",
        "@xla//xla/tsl/profiler/rpc:profiler_server_impl",
    ],
)

pybind_extension(
    name = "_profiler_backend",
    srcs = [
        "profiler.cc",
        "profiler.h",
    ],
    # Torch_tpu uses rules_pywrap in XLA for OSS builds:
    # https://github.com/openxla/xla/tree/main/third_party/py/rules_pywrap.
    # These args are required to link extensions against common libs.
    common_lib_packages = [
        "//torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/backends/profiler/cpu:host_tracer",
        "@xla//xla/backends/profiler/cpu:python_tracer",
        "@xla//xla/backends/profiler/tpu:tpu_tracer",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform/cloud:gcs_file_system",
        "@xla//xla/tsl/profiler/rpc:profiler_server_impl",
    ],
)

torch_tpu_package_end()
