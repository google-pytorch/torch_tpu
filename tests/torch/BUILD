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
    "torch_tpu_oss_torch_test",
    "torch_tpu_package_end",
)
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")

package(default_applicable_licenses = ["//:license"])

pytype_strict_binary(
    name = "torch_test_runner",
    testonly = True,
    srcs = ["torch_test_runner.py"],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
    ],
)

# The test targets below are auto-generated. Please do not edit them manually.
# To re-generate test targets, run the following command:
#
#   $ bazel run //ci/tools:gen_torch_tests -- \
#       --pytorch_test_dir=~/.github/pytorch/test
#
# GENERATED_TEST_TARGETS_BEGIN
torch_tpu_oss_torch_test(
    name = "complex_tensor_complex_tensor_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "complex_tensor/test_complex_tensor",
)

torch_tpu_oss_torch_test(
    name = "distributed_checkpoint__experimental_staging_test",
    torch_test_module = "distributed/checkpoint/_experimental/test_staging",
)

torch_tpu_oss_torch_test(
    name = "dynamo_activation_checkpointing_test",
    torch_test_module = "dynamo/test_activation_checkpointing",
)

torch_tpu_oss_torch_test(
    name = "dynamo_after_aot_test",
    torch_test_module = "dynamo/test_after_aot",
)

torch_tpu_oss_torch_test(
    name = "dynamo_backends_test",
    torch_test_module = "dynamo/test_backends",
)

torch_tpu_oss_torch_test(
    name = "dynamo_ctx_manager_test",
    torch_test_module = "dynamo/test_ctx_manager",
)

torch_tpu_oss_torch_test(
    name = "dynamo_debug_utils_test",
    torch_test_module = "dynamo/test_debug_utils",
)

torch_tpu_oss_torch_test(
    name = "dynamo_dynamo_decompositions_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "dynamo/test_dynamo_decompositions",
)

torch_tpu_oss_torch_test(
    name = "dynamo_dynamo_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "dynamo/test_dynamo_ops",
)

torch_tpu_oss_torch_test(
    name = "dynamo_export_test",
    torch_test_module = "dynamo/test_export",
)

torch_tpu_oss_torch_test(
    name = "dynamo_minifier_test",
    torch_test_module = "dynamo/test_minifier",
)

torch_tpu_oss_torch_test(
    name = "dynamo_misc_test",
    shard_count = 8,
    torch_test_module = "dynamo/test_misc",
)

torch_tpu_oss_torch_test(
    name = "dynamo_model_output_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "dynamo/test_model_output",
)

torch_tpu_oss_torch_test(
    name = "dynamo_modules_test",
    torch_test_module = "dynamo/test_modules",
)

torch_tpu_oss_torch_test(
    name = "dynamo_unspec_test",
    torch_test_module = "dynamo/test_unspec",
)

torch_tpu_oss_torch_test(
    name = "dynamo_utils_test",
    torch_test_module = "dynamo/test_utils",
)

torch_tpu_oss_torch_test(
    name = "export_hop_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "export/test_hop",
)

torch_tpu_oss_torch_test(
    name = "inductor_analysis_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_analysis",
)

torch_tpu_oss_torch_test(
    name = "inductor_compiled_autograd_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_compiled_autograd",
)

torch_tpu_oss_torch_test(
    name = "inductor_compiled_optimizers_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "inductor/test_compiled_optimizers",
)

torch_tpu_oss_torch_test(
    name = "inductor_fp8_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_fp8",
)

torch_tpu_oss_torch_test(
    name = "inductor_inductor_scheduler_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_inductor_scheduler",
)

torch_tpu_oss_torch_test(
    name = "inductor_torchinductor_dynamic_shapes_test",
    shard_count = 8,
    torch_test_module = "inductor/test_torchinductor_dynamic_shapes",
)

torch_tpu_oss_torch_test(
    name = "inductor_torchinductor_opinfo_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 16,
    torch_test_module = "inductor/test_torchinductor_opinfo",
)

torch_tpu_oss_torch_test(
    name = "inductor_torchinductor_opinfo_properties_test",
    shard_count = 8,
    torch_test_module = "inductor/test_torchinductor_opinfo_properties",
)

torch_tpu_oss_torch_test(
    name = "inductor_unbacked_symints_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_unbacked_symints",
)

torch_tpu_oss_torch_test(
    name = "inductor_utils_test",
    torch_test_module = "inductor/test_utils",
)

torch_tpu_oss_torch_test(
    name = "nn_convolution_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "nn/test_convolution",
)

torch_tpu_oss_torch_test(
    name = "nn_dropout_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "nn/test_dropout",
)

torch_tpu_oss_torch_test(
    name = "nn_embedding_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 2,
    torch_test_module = "nn/test_embedding",
)

torch_tpu_oss_torch_test(
    name = "nn_init_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "nn/test_init",
)

torch_tpu_oss_torch_test(
    name = "nn_multihead_attention_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "nn/test_multihead_attention",
)

torch_tpu_oss_torch_test(
    name = "nn_parametrization_test",
    torch_test_module = "nn/test_parametrization",
)

torch_tpu_oss_torch_test(
    name = "nn_pooling_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "nn/test_pooling",
)

torch_tpu_oss_torch_test(
    name = "profiler_execution_trace_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "profiler/test_execution_trace",
)

torch_tpu_oss_torch_test(
    name = "profiler_memory_profiler_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "profiler/test_memory_profiler",
)

torch_tpu_oss_torch_test(
    name = "profiler_profiler_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "profiler/test_profiler",
)

torch_tpu_oss_torch_test(
    name = "quantization_core_experimental_bits_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "quantization/core/experimental/test_bits",
)

torch_tpu_oss_torch_test(
    name = "quantization_core_experimental_floatx_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "quantization/core/experimental/test_floatx",
)

torch_tpu_oss_torch_test(
    name = "accelerator_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_accelerator",
)

torch_tpu_oss_torch_test(
    name = "autograd_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_autograd",
)

torch_tpu_oss_torch_test(
    name = "binary_ufuncs_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "test_binary_ufuncs",
)

torch_tpu_oss_torch_test(
    name = "compile_benchmark_util_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_compile_benchmark_util",
)

torch_tpu_oss_torch_test(
    name = "complex_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_complex",
)

torch_tpu_oss_torch_test(
    name = "content_store_test",
    torch_test_module = "test_content_store",
)

torch_tpu_oss_torch_test(
    name = "dataloader_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_dataloader",
)

torch_tpu_oss_torch_test(
    name = "decomp_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 50,
    torch_test_module = "test_decomp",
)

torch_tpu_oss_torch_test(
    name = "dlpack_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_dlpack",
)

torch_tpu_oss_torch_test(
    name = "expanded_weights_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_expanded_weights",
)

torch_tpu_oss_torch_test(
    name = "foreach_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "test_foreach",
)

torch_tpu_oss_torch_test(
    name = "fx_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_fx",
)

torch_tpu_oss_torch_test(
    name = "fx_experimental_test",
    torch_test_module = "test_fx_experimental",
)

torch_tpu_oss_torch_test(
    name = "indexing_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "test_indexing",
)

torch_tpu_oss_torch_test(
    name = "legacy_vmap_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_legacy_vmap",
)

torch_tpu_oss_torch_test(
    name = "linalg_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "test_linalg",
)

torch_tpu_oss_torch_test(
    name = "masked_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_masked",
)

torch_tpu_oss_torch_test(
    name = "matmul_cuda_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_matmul_cuda",
)

torch_tpu_oss_torch_test(
    name = "meta_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 30,
    torch_test_module = "test_meta",
)

torch_tpu_oss_torch_test(
    name = "modules_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 2,
    torch_test_module = "test_modules",
)

torch_tpu_oss_torch_test(
    name = "multiprocessing_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_multiprocessing",
)

torch_tpu_oss_torch_test(
    name = "native_mha_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_native_mha",
)

torch_tpu_oss_torch_test(
    name = "nestedtensor_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 5,
    torch_test_module = "test_nestedtensor",
)

torch_tpu_oss_torch_test(
    name = "nn_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 2,
    torch_test_module = "test_nn",
)

torch_tpu_oss_torch_test(
    name = "numpy_interop_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_numpy_interop",
)

torch_tpu_oss_torch_test(
    name = "ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 50,
    torch_test_module = "test_ops",
)

torch_tpu_oss_torch_test(
    name = "ops_fwd_gradients_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 5,
    torch_test_module = "test_ops_fwd_gradients",
)

torch_tpu_oss_torch_test(
    name = "ops_gradients_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 20,
    torch_test_module = "test_ops_gradients",
)

torch_tpu_oss_torch_test(
    name = "ops_jit_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 6,
    torch_test_module = "test_ops_jit",
)

torch_tpu_oss_torch_test(
    name = "optim_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 2,
    torch_test_module = "test_optim",
)

torch_tpu_oss_torch_test(
    name = "overrides_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 15,
    torch_test_module = "test_overrides",
)

torch_tpu_oss_torch_test(
    name = "prims_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_prims",
)

torch_tpu_oss_torch_test(
    name = "python_dispatch_test",
    torch_test_module = "test_python_dispatch",
)

torch_tpu_oss_torch_test(
    name = "reductions_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 2,
    torch_test_module = "test_reductions",
)

torch_tpu_oss_torch_test(
    name = "scaled_matmul_cuda_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_scaled_matmul_cuda",
)

torch_tpu_oss_torch_test(
    name = "scatter_gather_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "test_scatter_gather_ops",
)

torch_tpu_oss_torch_test(
    name = "segment_reductions_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_segment_reductions",
)

torch_tpu_oss_torch_test(
    name = "serialization_test",
    torch_test_module = "test_serialization",
)

torch_tpu_oss_torch_test(
    name = "shape_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_shape_ops",
)

torch_tpu_oss_torch_test(
    name = "sort_and_select_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 2,
    torch_test_module = "test_sort_and_select",
)

torch_tpu_oss_torch_test(
    name = "sparse_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_sparse",
)

torch_tpu_oss_torch_test(
    name = "sparse_csr_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_sparse_csr",
)

torch_tpu_oss_torch_test(
    name = "spectral_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_spectral_ops",
)

torch_tpu_oss_torch_test(
    name = "tensor_creation_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "test_tensor_creation_ops",
)

torch_tpu_oss_torch_test(
    name = "tensor_iterator_test",
    torch_test_module = "test_tensor_iterator",
)

torch_tpu_oss_torch_test(
    name = "testing_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "test_testing",
)

torch_tpu_oss_torch_test(
    name = "torch_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "test_torch",
)

torch_tpu_oss_torch_test(
    name = "type_promotion_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "test_type_promotion",
)

torch_tpu_oss_torch_test(
    name = "unary_ufuncs_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 50,
    torch_test_module = "test_unary_ufuncs",
)

torch_tpu_oss_torch_test(
    name = "utils_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_utils",
)

torch_tpu_oss_torch_test(
    name = "view_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "test_view_ops",
)

torch_tpu_oss_torch_test(
    name = "xpu_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_xpu",
)
# GENERATED_TEST_TARGETS_END

torch_tpu_package_end()
