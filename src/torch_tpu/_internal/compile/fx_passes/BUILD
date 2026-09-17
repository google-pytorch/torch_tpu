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
    name = "detect_causal_mask",
    srcs = ["detect_causal_mask.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "mark_embedded_constants",
    srcs = ["mark_embedded_constants.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

py_relocated_library(
    name = "mark_activation_checkpoints",
    srcs = ["mark_activation_checkpoints.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "force_collectives_output",
    srcs = ["force_collectives_output.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/distributed:collective_ops",
    ],
)

py_relocated_library(
    name = "propagate_symints",
    srcs = ["propagate_symints.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile/dynamic:sym_utils",
    ],
)

py_relocated_library(
    name = "clone_mutated_returned_placeholders",
    srcs = ["clone_mutated_returned_placeholders.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "reorder_symints",
    srcs = ["reorder_symints.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/compile/dynamic:sym_utils",
    ],
)

py_relocated_library(
    name = "sink_get_attr_constants",
    srcs = ["sink_get_attr_constants.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
    ],
)

py_relocated_library(
    name = "reassociate_norm_weights",
    srcs = ["reassociate_norm_weights.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal:env",
    ],
)

py_relocated_library(
    name = "fold_gqa",
    srcs = ["fold_gqa.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_package_end()
