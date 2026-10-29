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

# Verification of Pallas kernels and the PyTorch programs that call them. The
# tests live in //tests/pallas.

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
    name = "findings",
    srcs = ["findings.py"],
)

torch_tpu_py_library(
    name = "fx_verifier",
    srcs = ["fx_verifier.py"],
    deps = [
        ":findings",
        "//shims/jax:pallas_tpu",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/pallas:pallas_lib",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "idl_solver",
    srcs = ["idl_solver.py"],
)

torch_tpu_py_library(
    name = "interpret_verifier",
    srcs = ["interpret_verifier.py"],
    deps = [
        ":findings",
        ":fx_verifier",
        "//shims/jax:pallas_tpu",
        "//shims/torch:pytorch",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "jaxpr_model",
    srcs = ["jaxpr_model.py"],
    deps = [
        ":findings",
        ":fx_verifier",
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/pallas:pallas_lib",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "smt_checker",
    srcs = ["smt_checker.py"],
    deps = [
        ":findings",
        ":idl_solver",
        ":jaxpr_model",
    ],
)

torch_tpu_py_library(
    name = "verifier",
    srcs = ["verifier.py"],
    deps = [
        ":findings",
        ":fx_verifier",
        ":interpret_verifier",
        ":smt_checker",
        "//shims/torch:pytorch",
        "@pypi//jax",
    ],
)

torch_tpu_package_end()
