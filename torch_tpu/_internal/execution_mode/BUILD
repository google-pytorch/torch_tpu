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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end", "torch_tpu_py_library")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
    ],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that execution_mode_impl.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "execution_mode_impl_build",
    srcs = ["execution_mode_impl.cc"],
    deps = [
        "//shims/pybind11",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

pybind_extension(
    # "private" python module for internal use.
    name = "execution_mode_impl",
    srcs = ["execution_mode_impl.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:context_manager",
        "//torch_tpu/common:context_states",
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_library(
    name = "execution_mode",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":execution_mode_impl",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_package_end()
