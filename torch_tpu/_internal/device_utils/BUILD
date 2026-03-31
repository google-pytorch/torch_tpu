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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
    ],
)

pytype_strict_library(
    name = "device_utils",
    srcs = ["__init__.py"],
    deps = ["//shims/torch:pytorch"],
)

pytype_strict_library(
    name = "annotations",
    srcs = ["annotations.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_py_test(
    name = "annotations_test",
    srcs = ["annotations_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":annotations",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/sync",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "layout_hint_compile_test",
    srcs = ["layout_hint_compile_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":annotations",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/api",
    ],
)

torch_tpu_package_end()
