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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "adamw",
    srcs = [
        "adamw.py",
    ],
    deps = [
        ":patch",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "patch",
    srcs = [
        "patch.py",
    ],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "optim",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":adamw",
        ":patch",
    ],
)

torch_tpu_package_end()
