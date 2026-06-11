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

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
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
        "//torch_tpu/common:compilation_spec",
        "//torch_tpu/common:error_utils",
    ],
)

torch_tpu_pybind_extension(
    # "private" python module for internal use.
    name = "compiler_options_impl",
    srcs = ["compiler_options_impl.cc"],
    deps = [
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
        "//torch_tpu/common:compilation",
        "//torch_tpu/common:compilation_spec",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/status",
    ],
)

torch_tpu_py_library(
    name = "compiler_options",
    srcs = [
        "__init__.py",
    ],
    deps = [":compiler_options_impl"],
)

torch_tpu_package_end()
