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

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "ops",
    srcs = ["__init__.py"],
    deps = [
        ":jagged",
    ],
)

torch_tpu_py_library(
    name = "jagged",
    srcs = ["jagged.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_library(
    name = "splash_attention",
    srcs = ["splash_attention.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/jax:pallas_tpu_ops",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/pallas",
        "@pypi//jax",
        "@pypi//torchax",  # build_cleaner: keep
    ],
)

torch_tpu_package_end()
