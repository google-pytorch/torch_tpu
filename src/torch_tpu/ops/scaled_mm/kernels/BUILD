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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")
load("//shims/build_files:py_relocated_library.bzl", "py_relocated_library")
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# Shared build-time generator. Dispatches to a per-recipe generator registered
# in `_RECIPE_GENERATORS`, so every block-scaling recipe must be listed in deps.
py_relocated_library(
    # PY_LIBRARY_OK=Build-time kernel generator library.
    name = "scaled_mm_generate",
    srcs = ["scaled_mm_generate.py"],
    py_library = pytype_strict_library,
    deps = [
        ":blockwise128_scaled_mm_kernel",
        ":blockwise_scaled_mm_kernel",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//src/torch_tpu/_internal/pallas:pallas_kernel_generate_utils",
        "@pypi//jax",
    ],
)

pytype_strict_binary(
    name = "scaled_mm_generate_bin",
    srcs = ["scaled_mm_generate.py"],
    main = "scaled_mm_generate.py",
    deps = [
        ":blockwise128_scaled_mm_kernel",
        ":blockwise_scaled_mm_kernel",
        ":scaled_mm_generate",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//src/torch_tpu/_internal/pallas:pallas_kernel_generate_utils",
        "@pypi//jax",
    ],
)

py_relocated_library(
    name = "blockwise_scaled_mm_kernel",
    srcs = ["blockwise_scaled_mm_kernel.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "@pypi//jax",
    ],
)

# Numerics suite for block-scaled GEMM Pallas kernels.
torch_tpu_py_test(
    name = "blockwise_scaled_mm_kernel_test",
    srcs = ["blockwise_scaled_mm_kernel_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":blockwise_scaled_mm_kernel",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

# Host-side precondition and tile-selection tests. Pure Python logic that runs
# before pallas_call, so no TPU is required and it can run in presubmit.
torch_tpu_py_test(
    name = "blockwise_scaled_mm_kernel_host_test",
    srcs = ["blockwise_scaled_mm_kernel_host_test.py"],
    tags = [
        # Tests build-time generator scaled_mm_generate (not packaged in runtime wheel).
        "wheel_test_excluded",
    ],
    deps = [
        ":blockwise_scaled_mm_kernel",
        ":scaled_mm_generate",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

py_relocated_library(
    name = "blockwise128_scaled_mm_kernel",
    srcs = ["blockwise128_scaled_mm_kernel.py"],
    py_library = torch_tpu_py_library,
    deps = [
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "blockwise128_scaled_mm_kernel_test",
    srcs = ["blockwise128_scaled_mm_kernel_test.py"],
    main = "blockwise128_scaled_mm_kernel_test.py",
    run_on_accelerators = [
        "requires-tpu-v5lite",
        "requires-tpu-v6e",
        "requires-tpu-v7x",
    ],
    tags = [
        "nosan",  # *san builds are low-value for accelerator numerics tests.
    ],
    deps = [
        ":blockwise128_scaled_mm_kernel",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "blockwise128_scaled_mm_kernel_host_test",
    srcs = ["blockwise128_scaled_mm_kernel_host_test.py"],
    tags = [
        # Tests build-time generator scaled_mm_generate (not packaged in runtime wheel).
        "wheel_test_excluded",
    ],
    deps = [
        ":blockwise128_scaled_mm_kernel",
        ":scaled_mm_generate",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_package_end()
