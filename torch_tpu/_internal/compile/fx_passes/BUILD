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
    name = "mark_embedded_constants",
    srcs = ["mark_embedded_constants.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_py_library(
    name = "mark_activation_checkpoints",
    srcs = ["mark_activation_checkpoints.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "force_collectives_output",
    srcs = ["force_collectives_output.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed:collective_ops",
    ],
)

torch_tpu_py_library(
    name = "propagate_symints",
    srcs = ["propagate_symints.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile/dynamic:sym_utils",
    ],
)

torch_tpu_py_library(
    name = "clone_mutated_returned_placeholders",
    srcs = ["clone_mutated_returned_placeholders.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "reorder_symints",
    srcs = ["reorder_symints.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile/dynamic:sym_utils",
    ],
)

torch_tpu_py_library(
    name = "sink_get_attr_constants",
    srcs = ["sink_get_attr_constants.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "fold_gqa",
    srcs = ["fold_gqa.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_package_end()
