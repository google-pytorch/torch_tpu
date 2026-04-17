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
    default_visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
        "//shims/torchprime:__subpackages__",
        "//shims/torchtitan:__subpackages__",
        "@pypi//helion:__subpackages__",
    ],
)

pytype_strict_library(
    name = "api",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu:_loader",
    ],
)

torch_tpu_package_end()
