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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")
load("//shims/build_defs_testing:bzl_library.bzl", "bzl_library")
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")
load(":defs.bzl", "define_sdpa_kernel")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "flash_attention",
    srcs = ["flash_attention.py"],
    tags = [
        "ignore_for_dep=third_party.py.jax._src.pallas.core",
    ],
    deps = [
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "@pypi//jax",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_py_library(
    name = "scaled_dot_product_attention_kernels_impl",
    srcs = ["scaled_dot_product_attention_kernels.py"],
    tags = [
        "ignore_for_dep=third_party.py.jax._src.pallas.core",
    ],
    deps = [
        ":flash_attention",
        ":splash_attention",
        "//shims/jax:pallas",
        "//shims/torch:pytorch",
        "@pypi//jax",
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_py_library(
    name = "splash_base",
    srcs = ["splash/base.py"],
    tags = [
        "ignore_for_dep=third_party.py.jax._src.pallas.core",
    ],
    deps = [
        ":splash_attention_mask_info",
        "@pypi//jax",
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_py_library(
    name = "splash_attention",
    srcs = ["splash/splash_attention.py"],
    tags = [
        "ignore_for_dep=third_party.py.jax._src.pallas.core",
    ],
    deps = [
        ":splash_attention_mask",
        ":splash_attention_mask_info",
        ":splash_base",
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "@pypi//jax",
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_py_library(
    name = "splash_attention_mask",
    srcs = ["splash/splash_attention_mask.py"],
    tags = [
        "ignore_for_dep=third_party.py.jax._src.pallas.core",
    ],
    deps = [
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_py_library(
    name = "splash_attention_mask_info",
    srcs = ["splash/splash_attention_mask_info.py"],
    tags = [
        "ignore_for_dep=third_party.py.jax._src.pallas.core",
    ],
    deps = [
        ":splash_attention_mask",
        "@pypi//jax",
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
    ],
)

torch_tpu_py_library(
    name = "scaled_dot_product_attention_kernels",
    srcs = [],
    tags = [
        "ignore_for_dep=third_party.py.jax._src.pallas.core",
    ],
    deps = [
        ":flash_attention",
        ":scaled_dot_product_attention_kernels_impl",
        ":splash_attention",
        ":splash_attention_mask",
        ":splash_attention_mask_info",
        ":splash_base",
    ],
)

pytype_strict_binary(
    name = "scaled_dot_product_attention_generate",
    srcs = ["scaled_dot_product_attention_generate.py"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    deps = [
        ":scaled_dot_product_attention_kernels",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//torch_tpu/_internal/pallas:pallas_kernel_generate_utils",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "scaled_dot_product_attention_kernels_test",
    srcs = ["scaled_dot_product_attention_kernels_test.py"],
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 10,
    tags = [
        # TODO(b/493270350): Investigate if the test should be enabled for asan.
        "noasan",
        # Disabling tsan for this test, see details in b/491065267.
        "notsan",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":scaled_dot_product_attention_kernels",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

# Forward Kernels
define_sdpa_kernel(
    name = "sdpa_fwd_f32_non_causal_mlir",
    dtype = "float32",
    forward = True,
    is_causal = False,
)

define_sdpa_kernel(
    name = "sdpa_fwd_f32_causal_mlir",
    dtype = "float32",
    forward = True,
    is_causal = True,
)

define_sdpa_kernel(
    name = "sdpa_fwd_bf16_non_causal_mlir",
    dtype = "bfloat16",
    forward = True,
    is_causal = False,
)

define_sdpa_kernel(
    name = "sdpa_fwd_bf16_causal_mlir",
    dtype = "bfloat16",
    forward = True,
    is_causal = True,
)

# Backward Kernels
define_sdpa_kernel(
    name = "sdpa_bwd_f32_non_causal_mlir",
    dtype = "float32",
    forward = False,
    is_causal = False,
)

define_sdpa_kernel(
    name = "sdpa_bwd_f32_causal_mlir",
    dtype = "float32",
    forward = False,
    is_causal = True,
)

define_sdpa_kernel(
    name = "sdpa_bwd_bf16_non_causal_mlir",
    dtype = "bfloat16",
    forward = False,
    is_causal = False,
)

define_sdpa_kernel(
    name = "sdpa_bwd_bf16_causal_mlir",
    dtype = "bfloat16",
    forward = False,
    is_causal = True,
)

bzl_library(
    name = "defs_bzl",
    srcs = ["defs.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
    deps = ["//build_files:build_defs_bzl"],
)

torch_tpu_package_end()
