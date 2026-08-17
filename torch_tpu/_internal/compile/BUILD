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
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        ":torch_tpu_compiled_executable",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile/fx_passes:mark_activation_checkpoints",
        "//torch_tpu/_internal/compile/fx_passes:mark_embedded_constants",
        "//torch_tpu/_internal/export",
        "//torch_tpu/_internal/profiler:xprof_adapter",
        "//torch_tpu/csrc/internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_py_library(
    name = "collective_ops",
    srcs = ["collective_ops.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "split_compiler",
    srcs = ["split_compiler.py"],
    deps = [
        ":collective_ops",
        ":compiler",
        ":torch_tpu_compiled_executable",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile/fx_passes:clone_mutated_returned_placeholders",
        "//torch_tpu/_internal/compile/fx_passes:force_collectives_output",
        "//torch_tpu/_internal/compile/fx_passes:mark_embedded_constants",
        "//torch_tpu/_internal/compile/fx_passes:propagate_symints",
        "//torch_tpu/_internal/compile/fx_passes:reorder_symints",
        "//torch_tpu/_internal/compile/fx_passes:sink_get_attr_constants",
        "//torch_tpu/_internal/distributed:spmd_util",
        "//torch_tpu/csrc/internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_py_library(
    name = "torch_tpu_compiled_executable",
    srcs = ["torch_tpu_compiled_executable.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/csrc/internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_py_library(
    name = "compile",
    srcs = [
        "__init__.py",
    ],
    visibility = [
        "//:__subpackages__",
        "//learning/agents/tern/torch_agent:__subpackages__",
    ],
    deps = [
        ":scan",
        ":tpu_backend",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "tpu_backend",
    srcs = [
        "_backend.py",
    ],
    tags = [
        # Avoid a circular dependency. Imports torch_tpu for package version.
        "ignore_for_dep=third_party.py.torch_tpu",
    ],
    deps = [
        ":compiler",
        ":split_compiler",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile/dynamic:dynamic_compiler",
        "//torch_tpu/_internal/profiler:xprof_adapter",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_library(
    name = "scan",
    srcs = ["scan.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/export",
        "//torch_tpu/csrc/internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_package_end()
