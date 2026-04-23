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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "strictly_ordering",
    srcs = ["strictly_ordering.py"],
    notest_oss = "Disabled during switch from opt-in to opt-out",
    strict = True,
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/sync",
    ],
)

torch_tpu_package_end()
