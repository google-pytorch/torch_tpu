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
# See the License for the doxygen comments.

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files(["batch_transfer_impl.cc"])

# This library is not used directly, but is needed for go/cclean.
torch_tpu_cc_library(
    name = "batch_transfer_impl_build",
    srcs = ["batch_transfer_impl.cc"],
    deps = [
        "//csrc/common:dimension_types",
        "//csrc/common:error_utils",
        "//csrc/experimental/batch_transfer",
        "//shims/pybind11",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
    ],
)

torch_tpu_package_end()
