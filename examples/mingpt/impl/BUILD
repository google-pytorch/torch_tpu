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

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//:__subpackages__"],
)

license(
    name = "license",
    package_name = "torch_tpu_impl",
)

licenses(["notice"])

torch_tpu_py_library(
    name = "mingpt_lib",
    srcs = ["mingpt.py"],
    tags = [
        "ignore_for_dep=third_party.py.transformers",
        "nofixdeps",
    ],
    deps = [
        ":utils",
        "//shims/torch:pytorch",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    tags = [
        "ignore_for_dep=third_party.py.transformers",
        "nofixdeps",
    ],
    deps = [
        ":utils",
        "//shims/torch:pytorch",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "utils",
    srcs = ["utils.py"],
    tags = [
        "ignore_for_dep=third_party.py.transformers",
        "nofixdeps",
    ],
    deps = [
        "//shims/torch:pytorch",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "mingpt",
    srcs = [],
    tags = [
        "ignore_for_dep=third_party.py.transformers",
        "nofixdeps",
    ],
    deps = [
        ":mingpt_lib",
        ":trainer",
        ":utils",
    ],
)

torch_tpu_py_library(
    name = "mingpt_jax",
    srcs = [
        "mingpt_jax.py",
    ],
    deps = [
        "@pypi//jax",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
    ],
)

torch_tpu_package_end()
