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
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ],
)

torch_tpu_py_library(
    name = "execution_mode",
    srcs = [
        "__init__.py",
    ],
    deps = [":execution_mode_impl"],
)

pybind_extension(
    name = "execution_mode_impl",
    srcs = ["//torch_tpu/csrc/internal/execution_mode:execution_mode_impl.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    wrap_py_init = True,
    deps = [
        "//torch_tpu/csrc/common:context_manager",
        "//torch_tpu/csrc/common:context_states",
        "//torch_tpu/csrc/eager:eager_mode",
        "//torch_tpu/csrc/eager:tpu_aten_kernels",
    ],
)

torch_tpu_package_end()
