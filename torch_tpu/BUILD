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

load("@rules_license//rules:license.bzl", "license")
load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//:__subpackages__"],
)

license(
    name = "license",
    package_name = "torch_tpu",
)

exports_files(["LICENSE"])

torch_tpu_py_library(
    name = "torch_tpu",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//shims/libtpu",  # copybara:comment(oss-only)
        "//torch_tpu/_internal:testing",  # buildcleaner: keep
    ],
)

torch_tpu_package_end()
