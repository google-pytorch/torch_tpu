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

load("@rules_license//rules:license.bzl", "license")
load(
    "//build_files:build_defs.bzl",
    "if_oss",
    "torch_tpu_package_end",
)
load("//build_files:entry_points.bzl", "ENTRY_POINTS")
load("//build_files:py_dist_info.bzl", "py_dist_info")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")
load("//shims/py_rules:strict.bzl", "py_library_providing_imports_info")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//:__subpackages__"],
)

license(
    name = "license",
    package_name = "torch_tpu",
)

exports_files(["LICENSE"])

py_dist_info(
    name = "torch_tpu_dist_info",
    package_name = "torch_tpu",
    entry_points = ENTRY_POINTS,
    version = "0.0.0",
)

pytype_strict_library(
    name = "_loader",
    srcs = ["_loader.py"],
    data = [
        ":torch_tpu_dist_info",
    ],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/device:_device_module",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/profiler",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:hardware",
    ],
)

py_library_providing_imports_info(
    name = "torch_tpu",
    srcs = [
        "__init__.py",
    ],
    lib_rule = pytype_strict_library,
    module_specs = ["torch_tpu.modspec.json"],
    deps = if_oss(
        ["//shims/libtpu"],
        [],
    ) + [
        ":_loader",
        "//torch_tpu/_internal:testing",  # buildcleaner: keep
    ],
)

torch_tpu_package_end()
