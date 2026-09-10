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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
    ],
)

exports_files(["compiler_options_impl.cc"])

# This library is not used directly, but is needed for go/cclean. Without
# this rule, cclean complains that compiler_options_impl.cc has no rule depending on it.
torch_tpu_cc_library(
    name = "compiler_options_impl_build",
    srcs = ["compiler_options_impl.cc"],
    deps = [
        "//csrc/common:compilation",
        "//csrc/common:compilation_spec",
        "//csrc/common:error_utils",
        "//shims/pybind11",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:torch_headers",
    ],
)

torch_tpu_package_end()
