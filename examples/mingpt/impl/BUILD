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
load("//shims/py_rules:strict.bzl", "py_library_providing_imports_info")

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
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//shims/torch:pytorch",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        ":utils",  # build_cleaner: keep - relative import
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "mingpt_lib",
    srcs = ["mingpt.py"],
    tags = [
        # Prevent build_cleaner from adding @pypi//transformers:v5, which is a lazy import
        # that is invisible to this package. See go/build-cleaner/customize-behavior#ignore-for-dep
        "nofixdeps",
        "ignore_for_dep=third_party.py.transformers",
    ],
    deps = [
        ":utils",
        "//shims/torch:pytorch",
    ],
)

# We use py_library_providing_imports_info with mingpt.modspec.json to explicitly
# re-export the submodules provided by :mingpt_lib, :trainer, and :utils. This prevents
# PyStrictDeps violations at downstream use sites that import `from examples.mingpt.impl import mingpt`
# or trainer/utils without requiring them to directly depend on the sub-targets.
py_library_providing_imports_info(
    name = "mingpt",
    srcs = [
        "__init__.py",
    ],
    lib_rule = torch_tpu_py_library,
    module_specs = [
        "mingpt.modspec.json",
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
