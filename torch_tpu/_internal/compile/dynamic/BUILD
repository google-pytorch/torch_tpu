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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "symbol_bounds",
    srcs = ["symbol_bounds.py"],
    deps = [
        ":sym_utils",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

torch_tpu_py_library(
    name = "sym_utils",
    srcs = ["sym_utils.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

torch_tpu_py_library(
    name = "init",
    srcs = ["__init__.py"],
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
)

torch_tpu_py_library(
    name = "sym_shape_manager",
    srcs = [
        "sym_shape_manager.py",
    ],
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

torch_tpu_py_library(
    name = "view_ops_passes",
    srcs = [
        "view_ops_passes.py",
    ],
    deps = [
        ":sym_shape_manager",
        ":sym_utils",
        ":symbol_bounds",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "generative_ops_pass",
    srcs = [
        "generative_ops_pass.py",
    ],
    deps = [
        ":sym_shape_manager",
        ":sym_utils",
        ":symbol_bounds",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "graph_transformations",
    srcs = [
        "graph_transformations.py",
    ],
    deps = [
        ":generative_ops_pass",
        ":sym_shape_manager",
        ":sym_utils",
        ":symbol_bounds",
        ":view_ops_passes",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "view_decomposition",
    srcs = [
        "view_decomposition.py",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

torch_tpu_py_library(
    name = "dynamic_compiler",
    srcs = [
        "dynamic_compiler.py",
    ],
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
    deps = [
        ":graph_transformations",
        ":sym_shape_manager",
        ":view_ops_passes",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:compiler",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_package_end()
