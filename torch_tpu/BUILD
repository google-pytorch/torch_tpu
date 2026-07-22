load("@rules_license//rules:license.bzl", "license")
load("@torch_tpu_version//:version.bzl", OSS_WHEEL_VERSION = "WHEEL_VERSION")  # copybara:comment(oss-only)
load(
    "//build_files:build_defs.bzl",
    "if_oss",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//build_files:entry_points.bzl", "ENTRY_POINTS")
load("//build_files:py_dist_info.bzl", "py_dist_info")
load("//shims/build_files:build_defs.bzl", "TT_FRIENDS")

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
load("//shims/py_rules:strict.bzl", "py_library_providing_imports_info")

WHEEL_VERSION = "0.0.0"

WHEEL_VERSION = OSS_WHEEL_VERSION  # copybara:comment(oss-only)

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
    version = WHEEL_VERSION,
)

torch_tpu_py_library(
    name = "_loader",
    srcs = ["_loader.py"],
    data = [
        ":torch_tpu_dist_info",
    ],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal:native_scan",
        "//torch_tpu/_internal/batch_transfer",  # buildcleaner: keep
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/device:_device_module",
        "//torch_tpu/_internal/device:_tpu_backend_config",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/precision",
        "//torch_tpu/_internal/profiler",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/tracing",
        "//torch_tpu/_internal/utils:hardware",
    ],
)

py_library_providing_imports_info(
    name = "torch_tpu",
    srcs = [
        "__init__.py",
    ],
    lib_rule = torch_tpu_py_library,
    module_specs = ["torch_tpu.modspec.json"],
    # Users of this library may rely on autoloading of the TPU backend, in which
    # case they'd import `torch` but not `torch_tpu` directly. However, we need
    # to ensure `torch_tpu` is linked in even if no symbols are explicitly
    # used by the user. This tag prevents build_cleaner from removing the dependency.
    tags = ["keep_dep"],
    visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
        "@pypi//helion:__subpackages__",
        "@pypi//tokamax:__subpackages__",
    ] + TT_FRIENDS,
    deps = if_oss(
        ["//shims/libtpu"],
        [],
    ) + [
        ":_loader",
        "//torch_tpu/_internal:testing",  # buildcleaner: keep
    ],
)

torch_tpu_package_end()
