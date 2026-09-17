# Copyright 2026 Google LLC
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
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/build_files:py_relocated_library.bzl", "py_relocated_library")
load("//shims/build_files:py_relocated_pybind_extension.bzl", "py_relocated_pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

py_relocated_library(
    name = "tracing",
    srcs = ["__init__.py"],
    py_library = torch_tpu_py_library,
    deps = [
        ":_tpu_torch_tracing",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

py_relocated_pybind_extension(
    name = "_tpu_torch_tracing",
    srcs = ["//csrc/internal/tracing:tracing_py.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:pybind_error_utils",
        "//csrc/eager:structured_log_buffer",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/time",
    ],
)

torch_tpu_package_end()
