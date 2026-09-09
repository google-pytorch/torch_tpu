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
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "dynamism",
    srcs = [
        "__init__.py",
    ],
    visibility = [
        "//:__subpackages__",
    ],
    deps = [":_tpu_torch_dynamism"],
)

pybind_extension(
    name = "_tpu_torch_dynamism",
    srcs = ["//csrc/internal/dynamism:dynamism_py.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:device_type",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/internal/dynamism:dynamism_cc",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
    ],
)

torch_tpu_package_end()
