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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

py_relocated_library(
    name = "adamw",
    srcs = [
        "adamw.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":patch",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "patch",
    srcs = [
        "patch.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "optim",
    srcs = [
        "__init__.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":adamw",
        ":patch",
    ],
)

torch_tpu_package_end()
