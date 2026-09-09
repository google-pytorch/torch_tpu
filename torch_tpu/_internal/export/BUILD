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

# We use py_library_providing_imports_info with export.modspec.json to explicitly
# re-export the submodules provided by :export_lib. This prevents PyStrictDeps
# violations at downstream use sites that import `from torch_tpu._internal.export import export`
# without requiring them to directly depend on :export_lib.
py_library_providing_imports_info(
    name = "export",
    srcs = [
        "__init__.py",
    ],
    lib_rule = torch_tpu_py_library,
    module_specs = [
        "export.modspec.json",
    ],
    deps = [
        ":export_lib",
    ],
)

torch_tpu_py_library(
    name = "export_lib",
    srcs = [
        "export.py",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/benchmarks:xprof_adapter",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/execution_mode",
    ],
)

torch_tpu_package_end()
