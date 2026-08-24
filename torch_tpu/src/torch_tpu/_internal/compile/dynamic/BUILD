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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library")
load("//shims/build_files:py_relocated_library.bzl", "py_relocated_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

py_relocated_library(
    name = "symbol_bounds",
    srcs = ["symbol_bounds.py"],
    py_library = torch_tpu_py_library,
    deps = [
        ":sym_utils",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

py_relocated_library(
    name = "sym_utils",
    srcs = ["sym_utils.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

py_relocated_library(
    name = "init",
    srcs = ["__init__.py"],
    py_library = torch_tpu_py_library,
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
)

py_relocated_library(
    name = "sym_shape_manager",
    srcs = [
        "sym_shape_manager.py",
    ],
    py_library = torch_tpu_py_library,
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
    deps = [
        ":sym_utils",
        ":symbol_bounds",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

py_relocated_library(
    name = "view_ops_transformations",
    srcs = [
        "view_ops_transformations.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":sym_shape_manager",
        ":sym_utils",
        ":symbol_bounds",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "generative_ops_transformation",
    srcs = [
        "generative_ops_transformation.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":sym_shape_manager",
        ":sym_utils",
        ":symbol_bounds",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "graph_transformations",
    srcs = [
        "graph_transformations.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":generative_ops_transformation",
        ":sym_shape_manager",
        ":sym_utils",
        ":symbol_bounds",
        ":view_ops_transformations",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "compiler",
    srcs = [
        "compiler.py",
    ],
    py_library = torch_tpu_py_library,
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
    deps = [
        ":graph_transformations",
        ":sym_shape_manager",
        ":symbol_bounds",
        ":view_ops_transformations",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/csrc/internal/compile:tpu_torch_compile",
        "//torch_tpu/src/torch_tpu/_internal/compile:compiler",
    ],
)

torch_tpu_package_end()
