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

load("//build_files:build_defs.bzl", "torch_tpu_package_end")
load("//shims/build_defs_testing:bzl_library.bzl", "bzl_library")
load(":defs.bzl", "define_sdpa_kernel")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
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
