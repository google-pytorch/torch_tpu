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

# Domain experts for this package:
# ryo-not-rio
# lukeboyer

load(
    "//build_files:accelerator_specification.bzl",
    "run_on_config",
    "tpu_config",
    "tpus",
)
load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)
load("//shims/py_rules:py_binary.bzl", "py_binary")
load("//shims/py_rules:py_library.bzl", "py_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "kernels",
    srcs = ["__init__.py"],
)

torch_tpu_py_library(
    name = "rowwise_scaled_grouped_mm_kernel",
    srcs = ["rowwise_scaled_grouped_mm_kernel.py"],
    deps = [
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "//src/torch_tpu/_internal/ops/scaled_mm/kernels:aot_target",
        "@pypi//jax",
    ],
)

# Build-time generator for the precompiled kernel variants.
py_library(
    # PY_LIBRARY_OK=Build-time kernel generator library.
    name = "scaled_grouped_mm_generate",
    srcs = ["scaled_grouped_mm_generate.py"],
    type_checking = True,
    deps = [
        ":rowwise_scaled_grouped_mm_kernel",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//src/torch_tpu/_internal/pallas:pallas_kernel_generate_utils",
        "@pypi//jax",
    ],
)

py_binary(
    name = "scaled_grouped_mm_generate_bin",
    srcs = ["scaled_grouped_mm_generate.py"],
    main = "scaled_grouped_mm_generate.py",
    type_checking = True,
    deps = [
        ":scaled_grouped_mm_generate",
    ],
)

# Schedule, argument checks, AOT export and generator tests, plus interpreted
# kernel numerics. Runs on CPU, so it can run in presubmit.
torch_tpu_py_test(
    name = "rowwise_scaled_grouped_mm_kernel_host_test",
    srcs = ["rowwise_scaled_grouped_mm_kernel_host_test.py"],
    shard_count = 4,  # Compile-heavy: keeps sanitizer runs within the timeout.
    tags = [
        "notsan",  # TODO(b/584257115): Fails under OSS TSan.
        # Tests build-time generator scaled_grouped_mm_generate (not packaged in runtime wheel).
        "wheel_test_excluded",
    ],
    deps = [
        ":rowwise_scaled_grouped_mm_kernel",
        ":scaled_grouped_mm_generate",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "rowwise_scaled_grouped_mm_kernel_test",
    srcs = ["rowwise_scaled_grouped_mm_kernel_test.py"],
    compatible_accelerators = tpus(exclude = ["v5p"]),
    main = "rowwise_scaled_grouped_mm_kernel_test.py",
    run_on = run_on_config(
        postsubmit = [
            tpu_config("v5e"),
            tpu_config("v6e"),
            tpu_config("v7x"),
        ],
        reason = "Keeps the pre-migration per-TPU test targets.",
    ),
    tags = [
        "nosan",  # *san builds are low-value for accelerator numerics tests.
    ],
    deps = [
        ":rowwise_scaled_grouped_mm_kernel",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/ops/scaled_mm/kernels:scaled_mm_test_utils",
        "//tests:seed_test_utils",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "rowwise_scaled_grouped_mm_benchmark_test",
    srcs = ["rowwise_scaled_grouped_mm_benchmark_test.py"],
    compatible_accelerators = tpus(only = ["v7x"]),
    main = "rowwise_scaled_grouped_mm_benchmark_test.py",
    notap = "Manual microbenchmark",
    deps = [
        ":rowwise_scaled_grouped_mm_kernel",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_package_end()
