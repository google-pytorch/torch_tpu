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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end", "torch_tpu_py_library")
load("//shims/py_rules:strict.bzl", "py_library_providing_imports_info")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pybind_extension(
    name = "precision_impl",
    srcs = ["precision_impl.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/ops:precision_context",
        "@stablehlo//:stablehlo_ops",
    ],
)

# We use py_library_providing_imports_info with precision.modspec.json to explicitly
# re-export the submodules provided by :precision_lib. This prevents PyStrictDeps
# violations at downstream use sites that import `from torch_tpu._internal.precision import precision`
# without requiring them to directly depend on :precision_lib.
py_library_providing_imports_info(
    name = "precision",
    srcs = [
        "__init__.py",
    ],
    lib_rule = torch_tpu_py_library,
    module_specs = [
        "precision.modspec.json",
    ],
    deps = [
        ":precision_lib",
    ],
)

torch_tpu_py_library(
    name = "precision_lib",
    srcs = [
        "precision.py",
    ],
    deps = [
        ":precision_impl",
        "//torch_tpu/_internal/utils:annotations",
    ],
)

# This library is not used directly, but is needed for go/cclean.
torch_tpu_cc_library(
    name = "precision_impl_build",
    srcs = ["precision_impl.cc"],
    deps = [
        "//shims/pybind11",
        "//torch_tpu/ops:precision_context",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
