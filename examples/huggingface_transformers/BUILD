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

load("//build_files:build_defs.bzl", "torch_tpu_package_end")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

filegroup(
    name = "model_configs_data",
    srcs = glob(["model_configs/**/config.json"]),
)

pytype_strict_library(
    name = "model_configs",
    srcs = ["model_configs.py"],
    data = [":model_configs_data"],
    type_checking = "pyrefly",
    deps = [
        "//shims/etils/epath",
        "@pypi//transformers",
    ],
)

# TODO: targets call

torch_tpu_package_end()
