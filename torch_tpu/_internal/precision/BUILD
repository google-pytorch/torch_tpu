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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")
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
        "//torch_tpu/ops:precision_context",
        "@stablehlo//:stablehlo_ops",
    ],
)

pytype_strict_library(
    name = "precision",
    srcs = [
        "__init__.py",
        "precision.py",
    ],
    type_checking = "pyrefly",
    deps = [":precision_impl"],
)

# This library is not used directly, but is needed for go/cclean.
torch_tpu_cc_library(
    name = "precision_impl_build",
    srcs = ["precision_impl.cc"],
    deps = [
        "//shims/pybind11",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/ops:precision_context",
        "@stablehlo//:stablehlo_ops",
    ],
)

torch_tpu_package_end()
