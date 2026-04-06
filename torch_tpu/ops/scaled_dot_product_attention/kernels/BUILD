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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")
load(":defs.bzl", "define_sdpa_kernel")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_strict_library(
    name = "scaled_dot_product_attention_kernels",
    srcs = [
        "flash_attention.py",
        "scaled_dot_product_attention_kernels.py",
    ],
    tags = [
        "ignore_for_dep=third_party.py.jax._src.pallas.core",
    ],
    deps = [
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "//shims/torch:pytorch",
        "@pypi//jax",
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",  # buildcleaner: keep
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

torch_tpu_package_end()
