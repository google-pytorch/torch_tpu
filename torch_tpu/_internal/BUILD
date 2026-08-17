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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "native_scan",
    srcs = ["native_scan.py"],
    visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "native_scan_test",
    srcs = ["native_scan_test.py"],
    deps = [
        ":native_scan",
        "//shims/absl_py/testing:absltest",
        "//tests:seed_test_utils",
        "//torch_tpu",
    ],
)

torch_tpu_package_end()
