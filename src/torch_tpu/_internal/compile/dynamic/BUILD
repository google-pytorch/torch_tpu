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
    name = "output_sym_shape",
    srcs = [
        "output_sym_shape.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":sym_utils",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
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
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "view_ops_passes",
    srcs = [
        "view_ops_passes.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":sym_shape_manager",
        ":sym_utils",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "generative_ops_pass",
    srcs = [
        "generative_ops_pass.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":sym_shape_manager",
        ":sym_utils",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "convert_constant_symints_pass",
    srcs = [
        "convert_constant_symints_pass.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":sym_utils",
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
        ":convert_constant_symints_pass",
        ":dynamic_view_pass",
        ":generative_ops_pass",
        ":sym_shape_manager",
        ":sym_utils",
        ":symbol_bounds",
        ":view_ops_passes",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "view_decomposition",
    srcs = [
        "view_decomposition.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//sympy",
    ],
)

py_relocated_library(
    name = "dynamic_view_pass",
    srcs = [
        "dynamic_view_pass.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":view_decomposition",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "sym_bounds_manager",
    srcs = [
        "sym_bounds_manager.py",
    ],
    py_library = torch_tpu_py_library,
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

py_relocated_library(
    name = "dynamic_executable",
    srcs = [
        "dynamic_executable.py",
    ],
    py_library = torch_tpu_py_library,
    deps = [
        ":output_sym_shape",
        ":sym_utils",
        ":view_decomposition",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

py_relocated_library(
    name = "dynamic_compiler",
    srcs = [
        "dynamic_compiler.py",
    ],
    py_library = torch_tpu_py_library,
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
    deps = [
        ":dynamic_executable",
        ":graph_transformations",
        ":output_sym_shape",
        ":sym_bounds_manager",
        ":sym_utils",
        ":symbol_bounds",
        ":view_ops_passes",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/utils",
    ],
)

torch_tpu_package_end()
