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

load("//build_files:build_defs.bzl", "torch_tpu_package_end")
load("//shims/pynball:build_defs.bzl", "ipynb_deps")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

ipynb_deps(
    name = "precision_tutorial_deps",
    srcs = ["precision_tutorial.ipynb"],
    pytype = False,
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/api",
    ],
)

torch_tpu_package_end()
