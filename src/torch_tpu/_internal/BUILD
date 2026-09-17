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
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/build_files:py_relocated_library.bzl", "py_relocated_library")
load("//shims/build_files:py_relocated_pybind_extension.bzl", "py_relocated_pybind_extension")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

py_relocated_pybind_extension(
    name = "env",
    srcs = ["//csrc/internal:env.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:libtpu_version",
        "//csrc/common:pybind_error_utils",
        "//csrc/common:utils",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/log:absl_log",
    ],
)

py_relocated_pybind_extension(
    # Internal testing utilities.
    name = "testing",
    srcs = ["//csrc/internal:testing.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:error_utils",
        "//csrc/common:pybind_error_utils",
        "//csrc/eager:device_gen_impl",
        "//csrc/eager:events_queue",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/flags:flag",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

py_relocated_library(
    name = "env_utils",
    srcs = ["env_utils.py"],
    py_library = torch_tpu_py_library,
)

torch_tpu_package_end()
