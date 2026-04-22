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
load("//shims/py_rules:pytype.bzl", "pytype_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_library(
    name = "symbol_bounds",
    srcs = ["symbol_bounds.py"],
    type_checking = "pyrefly",
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

pytype_library(
    name = "sym_shape_manager",
    srcs = [
        "__init__.py",
        "sym_shape_manager.py",
    ],
    type_checking = "pyrefly",
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
    deps = [
        ":symbol_bounds",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

pytype_library(
    name = "graph_transformations",
    srcs = [
        "graph_transformations.py",
    ],
    type_checking = "pyrefly",
    deps = [
        ":sym_shape_manager",
        ":symbol_bounds",
        "//shims/torch:pytorch",
    ],
)

pytype_library(
    name = "compiler",
    srcs = [
        "__init__.py",
        "compiler.py",
    ],
    type_checking = "pyrefly",
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
    deps = [
        ":graph_transformations",
        ":sym_shape_manager",
        ":symbol_bounds",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_package_end()
