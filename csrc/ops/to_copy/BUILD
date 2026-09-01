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

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_cc_library",
    "torch_tpu_package_end",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "to_copy",
    srcs = ["to_copy_aten_kernels.cc"],
    hdrs = ["to_copy_aten_kernels.h"],
    deps = [
        "//csrc/common:cache_key",
        "//csrc/ops:op_names",
        "//csrc/ops/copy_from:copy_from_aten_kernels",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/torch/headeronly:torch_headeronly",
    ],
)

torch_tpu_package_end()
