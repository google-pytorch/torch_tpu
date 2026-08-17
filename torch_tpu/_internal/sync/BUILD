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
        "//shims/torch:pytorch",
        "//torch_tpu/csrc/internal/device:_device_ops_backend",
        "//torch_tpu/csrc/internal/sync:_tpu_torch_sync",
    ],
)

torch_tpu_package_end()
