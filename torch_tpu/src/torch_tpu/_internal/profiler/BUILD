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
load("//shims/build_files:py_relocated_library.bzl", "py_relocated_library")
load("//shims/py_rules:strict.bzl", "py_library_providing_imports_info")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ] + TT_FRIENDS,
)

py_relocated_library(
    name = "profiler_config",
    srcs = ["profiler_config.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/src/torch_tpu/_internal/utils:annotations",
    ],
)

# We use py_library_providing_imports_info with profiler.modspec.json to explicitly
# re-export the submodules provided by :profiler_lib. This prevents PyStrictDeps
# violations at downstream use sites that import `from torch_tpu._internal.profiler import profiler`
# without requiring them to directly depend on :profiler_lib.
py_relocated_library(
    name = "profiler",
    srcs = [
        "__init__.py",
    ],
    lib_rule = torch_tpu_py_library,
    module_specs = [
        "profiler.modspec.json",
    ],
    py_library = py_library_providing_imports_info,
    deps = [
        ":profiler_config",
        ":profiler_lib",
    ],
)

py_relocated_library(
    name = "profiler_lib",
    srcs = ["profiler_api.py"],
    py_library = torch_tpu_py_library,
    deps = [
        ":_impl",
        "//torch_tpu/csrc/internal/profiler:_profiler_backend",
    ],
)

py_relocated_library(
    name = "_impl",
    srcs = ["_impl.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/absl_py/logging",
        "//torch_tpu/csrc/internal/profiler:_profiler_backend",
    ],
)

oss_target(
    torch_tpu_py_library,
    name = "xprof_adapter",
    srcs = ["xprof_adapter.py"],
)

torch_tpu_package_end()
