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

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_cc_library",
    "torch_tpu_package_end",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "sparse_dense_matmul",
    srcs = ["sparse_dense_matmul_aten_kernels.cc"],
    hdrs = ["sparse_dense_matmul_aten_kernels.h"],
    deps = [
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:side_effect_util",
    ],
)

torch_tpu_cc_library(
    name = "sparse_dense_matmul_grad_with_adagrad",
    srcs = ["sparse_dense_matmul_grad_with_adagrad_aten_kernels.cc"],
    hdrs = ["sparse_dense_matmul_grad_with_adagrad_aten_kernels.h"],
    deps = [
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:side_effect_util",
    ],
)

torch_tpu_cc_library(
    name = "sparse_dense_matmul_grad_with_adam",
    srcs = ["sparse_dense_matmul_grad_with_adam_aten_kernels.cc"],
    hdrs = ["sparse_dense_matmul_grad_with_adam_aten_kernels.h"],
    deps = [
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:side_effect_util",
    ],
)

torch_tpu_cc_library(
    name = "sparse_dense_matmul_grad_with_sgd",
    srcs = ["sparse_dense_matmul_grad_with_sgd_aten_kernels.cc"],
    hdrs = ["sparse_dense_matmul_grad_with_sgd_aten_kernels.h"],
    deps = [
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:fixed_size_span",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:side_effect_util",
    ],
)

torch_tpu_cc_library(
    name = "input_preprocessing",
    srcs = ["input_preprocessing_aten_kernels.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        # Note: Despite residing under jax_tpu_embedding, the core C++
        # SparseCore preprocessing library is framework-independent.
        "@jax_tpu_embedding//jax_tpu_embedding/sparsecore/lib/core:abstract_input_batch",
        "@jax_tpu_embedding//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_core",
        "@jax_tpu_embedding//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_util",
        "@jax_tpu_embedding//jax_tpu_embedding/sparsecore/lib/core:ragged_tensor_input_batch",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "//shims/torch/c10/util:base_headers",
        "//csrc/common:error_utils",
    ],
    alwayslink = 1,
)

torch_tpu_cc_library(
    name = "compute_stats_aten_kernels",
    srcs = ["compute_stats_aten_kernels.cc"],
    deps = [
        "//csrc/common:error_utils",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/core:base_headers",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
    alwayslink = 1,
)

torch_tpu_package_end()
