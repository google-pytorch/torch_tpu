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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library")
load("//shims/py_rules:strict.bzl", "py_library_providing_imports_info")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# We use py_library_providing_imports_info with sync.modspec.json to explicitly
# re-export the submodules provided by :sync_lib. This prevents PyStrictDeps
# violations at downstream use sites that import `from torch_tpu._internal.sync import sync`
# without requiring them to directly depend on :sync_lib.
py_library_providing_imports_info(
    name = "sync",
    srcs = [
        "__init__.py",
    ],
    lib_rule = torch_tpu_py_library,
    module_specs = [
        "sync.modspec.json",
    ],
    deps = [
        ":sync_lib",
    ],
)

torch_tpu_py_library(
    name = "sync_lib",
    srcs = [
        "sync.py",
    ],
    deps = [
        ":_tpu_torch_sync",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/device:_device_ops_backend",
    ],
)

pybind_extension(
    name = "_tpu_torch_sync",
    srcs = ["//csrc/internal/sync:sync_py.cc"],
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
        "//csrc/eager:device_buffer",
        "//csrc/eager:materialize",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/internal/sync:sync_cc",
        "//csrc/ops:op_names",
        "//csrc/ops:python_context",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

torch_tpu_package_end()
