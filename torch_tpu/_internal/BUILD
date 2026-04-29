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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")
load("//shims/pybind11:pybind.bzl", "pybind_extension")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
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
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/eager:op_dispatcher",
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
        "//torch_tpu/eager:eager_mode",
        "//torch_tpu/eager:op_dispatcher",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that compiler_options_impl.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "compiler_options_impl_build",
    srcs = ["compiler_options_impl.cc"],
    deps = [
        "//shims/pybind11",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:compilation",
    ],
)

pybind_extension(
    # "private" python module for internal use.
    name = "compiler_options_impl",
    srcs = ["compiler_options_impl.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:compilation",
    ],
)

# This library is not used directly, but is needed to satisfy go/cclean
torch_tpu_cc_library(
    name = "env_build",
    srcs = ["env.cc"],
    deps = [
        "//shims/pybind11",
        "//torch_tpu/common:utils",
    ],
)

pybind_extension(
    name = "env",
    srcs = ["env.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch:torch_headers",
        "//torch_tpu/common:utils",
    ],
)

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that testing.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "testing_build",
    srcs = ["testing.cc"],
    deps = [
        "//shims/pybind11",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/eager:op_dispatcher",
    ],
)

pybind_extension(
    # Internal testing utilities.
    name = "testing",
    srcs = ["testing.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/eager:op_dispatcher",
        "@com_google_absl//absl/flags:flag",
    ],
)


# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that raw_transfer_impl.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "raw_transfer_impl_build",
    srcs = ["raw_transfer_impl.cc"],
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/pjrt:raw_transfer",
        "//shims/pybind11",
    ],
)

pybind_extension(
    name = "raw_transfer_impl",
    srcs = ["raw_transfer_impl.cc"],
    common_lib_packages = [
        "torch_tpu/common",
    ],
    wrap_py_init = True,
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/pjrt:raw_transfer",
    ],
)

torch_tpu_package_end()
