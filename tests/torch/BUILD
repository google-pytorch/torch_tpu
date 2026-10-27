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
# chizhang529
# jialei-chen

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)
load("//shims/build_files:torch_rules.bzl", "torch_tpu_torch_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary")

package(default_applicable_licenses = ["//:license"])

torch_tpu_py_library(
    name = "torch_test_configs",
    testonly = True,
    srcs = ["torch_test_configs.py"],
    deps = [
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
    ],
)

pytype_strict_binary(
    name = "torch_test_runner",
    testonly = True,
    srcs = ["torch_test_runner.py"],
    deps = [
        ":torch_test_configs",
        "//shims/absl_py:app",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

# The test targets below are auto-generated. Please do not edit them manually.
# To re-generate test targets, run the following command:
#
#   $ bazel run //ci/tools:gen_torch_tests -- \
#       --pytorch_test_dir=~/.github/pytorch/test
#
# GENERATED_TEST_TARGETS_BEGIN
torch_tpu_torch_test(
    name = "autograd_functional_test",
    torch_test_module = "autograd/test_functional",
)

torch_tpu_torch_test(
    name = "complex_tensor_complex_tensor_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "complex_tensor/test_complex_tensor",
)

torch_tpu_torch_test(
    name = "distributed__composable_fsdp_fully_shard_comm_test",
    torch_test_module = "distributed/_composable/fsdp/test_fully_shard_comm",
)

torch_tpu_torch_test(
    name = "distributed_checkpoint__experimental_staging_test",
    torch_test_module = "distributed/checkpoint/_experimental/test_staging",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_distributed_checkpoint_test",
    torch_test_module = "distributed/fsdp/test_distributed_checkpoint",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_apply_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_apply",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_checkpoint_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_checkpoint",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_clip_grad_norm_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_clip_grad_norm",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_comm_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_comm",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_core_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_core",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_dtensor_state_dict_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_dtensor_state_dict",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_exec_order_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_exec_order",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_fine_tune_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_fine_tune",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_input_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_input",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_multiple_forward_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_multiple_forward",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_multiple_wrapping_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_multiple_wrapping",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_overlap_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_overlap",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_pure_fp16_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_pure_fp16",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_traversal_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_traversal",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_fsdp_uneven_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_fsdp_uneven",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_hsdp_dtensor_state_dict_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/fsdp/test_hsdp_dtensor_state_dict",
)

torch_tpu_torch_test(
    name = "distributed_fsdp_utils_test",
    torch_test_module = "distributed/fsdp/test_utils",
)

torch_tpu_torch_test(
    name = "distributed_pipelining_backward_test",
    torch_test_module = "distributed/pipelining/test_backward",
)

torch_tpu_torch_test(
    name = "distributed_pipelining_microbatch_test",
    torch_test_module = "distributed/pipelining/test_microbatch",
)

torch_tpu_torch_test(
    name = "distributed_pipelining_schedule_test",
    torch_test_module = "distributed/pipelining/test_schedule",
)

torch_tpu_torch_test(
    name = "distributed_pipelining_transformer_test",
    torch_test_module = "distributed/pipelining/test_transformer",
)

torch_tpu_torch_test(
    name = "distributed_pipelining_unflatten_test",
    torch_test_module = "distributed/pipelining/test_unflatten",
)

torch_tpu_torch_test(
    name = "distributed_tensor_dtensor_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/tensor/test_dtensor_ops",
)

torch_tpu_torch_test(
    name = "distributed_backends_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/test_backends",
)

torch_tpu_torch_test(
    name = "distributed_c10d_torchcomms_test",
    torch_test_module = "distributed/test_c10d_torchcomms",
)

torch_tpu_torch_test(
    name = "distributed_dynamo_distributed_test",
    torch_test_module = "distributed/test_dynamo_distributed",
)

torch_tpu_torch_test(
    name = "distributed_functional_api_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/test_functional_api",
)

torch_tpu_torch_test(
    name = "distributed_functional_differentials_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributed/test_functional_differentials",
)

torch_tpu_torch_test(
    name = "distributions_distributions_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "distributions/test_distributions",
)

torch_tpu_torch_test(
    name = "dynamo_activation_checkpointing_test",
    torch_test_module = "dynamo/test_activation_checkpointing",
)

torch_tpu_torch_test(
    name = "dynamo_after_aot_test",
    torch_test_module = "dynamo/test_after_aot",
)

torch_tpu_torch_test(
    name = "dynamo_aot_autograd_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "dynamo/test_aot_autograd",
)

torch_tpu_torch_test(
    name = "dynamo_backends_test",
    torch_test_module = "dynamo/test_backends",
)

torch_tpu_torch_test(
    name = "dynamo_ctx_manager_test",
    torch_test_module = "dynamo/test_ctx_manager",
)

torch_tpu_torch_test(
    name = "dynamo_debug_utils_test",
    torch_test_module = "dynamo/test_debug_utils",
)

torch_tpu_torch_test(
    name = "dynamo_deviceguard_test",
    torch_test_module = "dynamo/test_deviceguard",
)

torch_tpu_torch_test(
    name = "dynamo_dynamo_decompositions_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "dynamo/test_dynamo_decompositions",
)

torch_tpu_torch_test(
    name = "dynamo_dynamo_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "dynamo/test_dynamo_ops",
)

torch_tpu_torch_test(
    name = "dynamo_export_test",
    torch_test_module = "dynamo/test_export",
)

torch_tpu_torch_test(
    name = "dynamo_fx_annotate_test",
    torch_test_module = "dynamo/test_fx_annotate",
)

torch_tpu_torch_test(
    name = "dynamo_higher_order_ops_test",
    torch_test_module = "dynamo/test_higher_order_ops",
)

torch_tpu_torch_test(
    name = "dynamo_hooks_test",
    torch_test_module = "dynamo/test_hooks",
)

torch_tpu_torch_test(
    name = "dynamo_minifier_test",
    torch_test_module = "dynamo/test_minifier",
)

torch_tpu_torch_test(
    name = "dynamo_misc_test",
    shard_count = 8,
    torch_test_module = "dynamo/test_misc",
)

torch_tpu_torch_test(
    name = "dynamo_model_output_test",
    torch_test_module = "dynamo/test_model_output",
)

torch_tpu_torch_test(
    name = "dynamo_modules_test",
    torch_test_module = "dynamo/test_modules",
)

torch_tpu_torch_test(
    name = "dynamo_python_dispatcher_test",
    torch_test_module = "dynamo/test_python_dispatcher",
)

torch_tpu_torch_test(
    name = "dynamo_repros_test",
    torch_test_module = "dynamo/test_repros",
)

torch_tpu_torch_test(
    name = "dynamo_streams_test",
    torch_test_module = "dynamo/test_streams",
)

torch_tpu_torch_test(
    name = "dynamo_tp_richcompare_test",
    torch_test_module = "dynamo/test_tp_richcompare",
)

torch_tpu_torch_test(
    name = "dynamo_unspec_test",
    torch_test_module = "dynamo/test_unspec",
)

torch_tpu_torch_test(
    name = "dynamo_utils_test",
    torch_test_module = "dynamo/test_utils",
)

torch_tpu_torch_test(
    name = "export_converter_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "export/test_converter",
)

torch_tpu_torch_test(
    name = "export_hop_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "export/test_hop",
)

torch_tpu_torch_test(
    name = "export_opinfo_schema_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "export/test_opinfo_schema",
)

torch_tpu_torch_test(
    name = "export_passes_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "export/test_passes",
)

torch_tpu_torch_test(
    name = "export_serialize_test",
    torch_test_module = "export/test_serialize",
)

torch_tpu_torch_test(
    name = "export_torchbind_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "export/test_torchbind",
)

torch_tpu_torch_test(
    name = "functorch_dim_split_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "functorch/dim/test_split",
)

torch_tpu_torch_test(
    name = "functorch_ac_test",
    torch_test_module = "functorch/test_ac",
)

torch_tpu_torch_test(
    name = "functorch_aot_joint_with_descriptors_test",
    torch_test_module = "functorch/test_aot_joint_with_descriptors",
)

torch_tpu_torch_test(
    name = "functorch_compile_to_python_test",
    torch_test_module = "functorch/test_compile_to_python",
)

torch_tpu_torch_test(
    name = "functorch_control_flow_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "functorch/test_control_flow",
)

torch_tpu_torch_test(
    name = "functorch_dims_test",
    torch_test_module = "functorch/test_dims",
)

torch_tpu_torch_test(
    name = "functorch_eager_transforms_test",
    shard_count = 2,
    torch_test_module = "functorch/test_eager_transforms",
)

torch_tpu_torch_test(
    name = "functorch_memory_efficient_fusion_test",
    torch_test_module = "functorch/test_memory_efficient_fusion",
)

torch_tpu_torch_test(
    name = "functorch_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "functorch/test_ops",
)

torch_tpu_torch_test(
    name = "functorch_vmap_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "functorch/test_vmap",
)

torch_tpu_torch_test(
    name = "higher_order_ops_invoke_subgraph_test",
    torch_test_module = "higher_order_ops/test_invoke_subgraph",
)

torch_tpu_torch_test(
    name = "higher_order_ops_print_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "higher_order_ops/test_print",
)

torch_tpu_torch_test(
    name = "higher_order_ops_with_effects_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "higher_order_ops/test_with_effects",
)

torch_tpu_torch_test(
    name = "inductor_analysis_test",
    torch_test_module = "inductor/test_analysis",
)

torch_tpu_torch_test(
    name = "inductor_compiled_autograd_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_compiled_autograd",
)

torch_tpu_torch_test(
    name = "inductor_compiled_optimizers_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "inductor/test_compiled_optimizers",
)

torch_tpu_torch_test(
    name = "inductor_custom_lowering_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_custom_lowering",
)

torch_tpu_torch_test(
    name = "inductor_decompose_mem_bound_mm_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_decompose_mem_bound_mm",
)

torch_tpu_torch_test(
    name = "inductor_flex_attention_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_flex_attention",
)

torch_tpu_torch_test(
    name = "inductor_flex_decoding_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_flex_decoding",
)

torch_tpu_torch_test(
    name = "inductor_fp8_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_fp8",
)

torch_tpu_torch_test(
    name = "inductor_fused_attention_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_fused_attention",
)

torch_tpu_torch_test(
    name = "inductor_inductor_scheduler_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_inductor_scheduler",
)

torch_tpu_torch_test(
    name = "inductor_mix_order_reduction_test",
    torch_test_module = "inductor/test_mix_order_reduction",
)

torch_tpu_torch_test(
    name = "inductor_mmdecomp_test",
    torch_test_module = "inductor/test_mmdecomp",
)

torch_tpu_torch_test(
    name = "inductor_op_dtype_prop_test",
    torch_test_module = "inductor/test_op_dtype_prop",
)

torch_tpu_torch_test(
    name = "inductor_selective_lowering_test",
    torch_test_module = "inductor/test_selective_lowering",
)

torch_tpu_torch_test(
    name = "inductor_slice_scatter_chunking_test",
    torch_test_module = "inductor/test_slice_scatter_chunking",
)

torch_tpu_torch_test(
    name = "inductor_torchinductor_codegen_dynamic_shapes_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_torchinductor_codegen_dynamic_shapes",
)

torch_tpu_torch_test(
    name = "inductor_torchinductor_dynamic_shapes_test",
    shard_count = 8,
    torch_test_module = "inductor/test_torchinductor_dynamic_shapes",
)

torch_tpu_torch_test(
    name = "inductor_torchinductor_opinfo_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 16,
    torch_test_module = "inductor/test_torchinductor_opinfo",
)

torch_tpu_torch_test(
    name = "inductor_torchinductor_opinfo_properties_test",
    shard_count = 8,
    torch_test_module = "inductor/test_torchinductor_opinfo_properties",
)

torch_tpu_torch_test(
    name = "inductor_unbacked_symints_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "inductor/test_unbacked_symints",
)

torch_tpu_torch_test(
    name = "inductor_utils_test",
    torch_test_module = "inductor/test_utils",
)

torch_tpu_torch_test(
    name = "lazy_ts_opinfo_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "lazy/test_ts_opinfo",
)

torch_tpu_torch_test(
    name = "nn_convolution_test",
    shard_count = 8,
    torch_test_module = "nn/test_convolution",
)

torch_tpu_torch_test(
    name = "nn_dropout_test",
    torch_test_module = "nn/test_dropout",
)

torch_tpu_torch_test(
    name = "nn_embedding_test",
    shard_count = 8,
    torch_test_module = "nn/test_embedding",
)

torch_tpu_torch_test(
    name = "nn_init_test",
    torch_test_module = "nn/test_init",
)

torch_tpu_torch_test(
    name = "nn_lazy_modules_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "nn/test_lazy_modules",
)

torch_tpu_torch_test(
    name = "nn_multihead_attention_test",
    torch_test_module = "nn/test_multihead_attention",
)

torch_tpu_torch_test(
    name = "nn_packed_sequence_test",
    torch_test_module = "nn/test_packed_sequence",
)

torch_tpu_torch_test(
    name = "nn_parametrization_test",
    torch_test_module = "nn/test_parametrization",
)

torch_tpu_torch_test(
    name = "nn_pooling_test",
    shard_count = 4,
    torch_test_module = "nn/test_pooling",
)

torch_tpu_torch_test(
    name = "profiler_execution_trace_test",
    torch_test_module = "profiler/test_execution_trace",
)

torch_tpu_torch_test(
    name = "profiler_memory_profiler_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "profiler/test_memory_profiler",
)

torch_tpu_torch_test(
    name = "profiler_profiler_test",
    shard_count = 4,
    torch_test_module = "profiler/test_profiler",
)

torch_tpu_torch_test(
    name = "profiler_trace_validator_test",
    torch_test_module = "profiler/test_trace_validator",
)

torch_tpu_torch_test(
    name = "python_native_tensor_utils_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "python_native/test_tensor_utils",
)

torch_tpu_torch_test(
    name = "quantization_core_experimental_bits_test",
    torch_test_module = "quantization/core/experimental/test_bits",
)

torch_tpu_torch_test(
    name = "quantization_core_experimental_floatx_test",
    torch_test_module = "quantization/core/experimental/test_floatx",
)

torch_tpu_torch_test(
    name = "accelerator_test",
    torch_test_module = "test_accelerator",
)

torch_tpu_torch_test(
    name = "ao_sparsity_test",
    torch_test_module = "test_ao_sparsity",
)

torch_tpu_torch_test(
    name = "autocast_test",
    torch_test_module = "test_autocast",
)

torch_tpu_torch_test(
    name = "autograd_test",
    torch_test_module = "test_autograd",
)

torch_tpu_torch_test(
    name = "binary_ufuncs_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "test_binary_ufuncs",
)

torch_tpu_torch_test(
    name = "bmm_outer_product_test",
    torch_test_module = "test_bmm_outer_product",
)

torch_tpu_torch_test(
    name = "comparison_utils_test",
    torch_test_module = "test_comparison_utils",
)

torch_tpu_torch_test(
    name = "compile_benchmark_util_test",
    torch_test_module = "test_compile_benchmark_util",
)

torch_tpu_torch_test(
    name = "complex_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_complex",
)

torch_tpu_torch_test(
    name = "content_store_test",
    torch_test_module = "test_content_store",
)

torch_tpu_torch_test(
    name = "custom_ops_test",
    torch_test_module = "test_custom_ops",
)

torch_tpu_torch_test(
    name = "dataloader_test",
    torch_test_module = "test_dataloader",
)

torch_tpu_torch_test(
    name = "decomp_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 50,
    torch_test_module = "test_decomp",
)

torch_tpu_torch_test(
    name = "dlpack_test",
    torch_test_module = "test_dlpack",
)

torch_tpu_torch_test(
    name = "dynamic_shapes_test",
    torch_test_module = "test_dynamic_shapes",
)

torch_tpu_torch_test(
    name = "expanded_weights_test",
    shard_count = 2,
    torch_test_module = "test_expanded_weights",
)

torch_tpu_torch_test(
    name = "fake_tensor_test",
    torch_test_module = "test_fake_tensor",
)

torch_tpu_torch_test(
    name = "foreach_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "test_foreach",
)

torch_tpu_torch_test(
    name = "fx_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_fx",
)

torch_tpu_torch_test(
    name = "fx_experimental_test",
    torch_test_module = "test_fx_experimental",
)

torch_tpu_torch_test(
    name = "indexing_test",
    shard_count = 8,
    torch_test_module = "test_indexing",
)

torch_tpu_torch_test(
    name = "legacy_vmap_test",
    torch_test_module = "test_legacy_vmap",
)

torch_tpu_torch_test(
    name = "linalg_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 4,
    torch_test_module = "test_linalg",
)

torch_tpu_torch_test(
    name = "masked_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_masked",
)

torch_tpu_torch_test(
    name = "maskedtensor_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_maskedtensor",
)

torch_tpu_torch_test(
    name = "matmul_cuda_test",
    torch_test_module = "test_matmul_cuda",
)

torch_tpu_torch_test(
    name = "meta_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 30,
    torch_test_module = "test_meta",
)

torch_tpu_torch_test(
    name = "modules_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 2,
    torch_test_module = "test_modules",
)

torch_tpu_torch_test(
    name = "multiprocessing_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_multiprocessing",
)

torch_tpu_torch_test(
    name = "native_mha_test",
    torch_test_module = "test_native_mha",
)

torch_tpu_torch_test(
    name = "nestedtensor_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 5,
    torch_test_module = "test_nestedtensor",
)

torch_tpu_torch_test(
    name = "nn_test",
    shard_count = 10,
    torch_test_module = "test_nn",
)

torch_tpu_torch_test(
    name = "numpy_interop_test",
    torch_test_module = "test_numpy_interop",
)

torch_tpu_torch_test(
    name = "opaque_obj_v2_test",
    torch_test_module = "test_opaque_obj_v2",
)

torch_tpu_torch_test(
    name = "ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 50,
    torch_test_module = "test_ops",
)

torch_tpu_torch_test(
    name = "ops_fwd_gradients_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 5,
    torch_test_module = "test_ops_fwd_gradients",
)

torch_tpu_torch_test(
    name = "ops_gradients_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 20,
    torch_test_module = "test_ops_gradients",
)

torch_tpu_torch_test(
    name = "ops_unbacked_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_ops_unbacked",
)

torch_tpu_torch_test(
    name = "optim_test",
    shard_count = 2,
    torch_test_module = "test_optim",
)

torch_tpu_torch_test(
    name = "out_dtype_op_test",
    torch_test_module = "test_out_dtype_op",
)

torch_tpu_torch_test(
    name = "overrides_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 15,
    torch_test_module = "test_overrides",
)

torch_tpu_torch_test(
    name = "precompile_test",
    torch_test_module = "test_precompile",
)

torch_tpu_torch_test(
    name = "prims_test",
    torch_test_module = "test_prims",
)

torch_tpu_torch_test(
    name = "proxy_tensor_test",
    torch_test_module = "test_proxy_tensor",
)

torch_tpu_torch_test(
    name = "python_dispatch_test",
    torch_test_module = "test_python_dispatch",
)

torch_tpu_torch_test(
    name = "reductions_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 2,
    torch_test_module = "test_reductions",
)

torch_tpu_torch_test(
    name = "scaled_matmul_cuda_test",
    torch_test_module = "test_scaled_matmul_cuda",
)

torch_tpu_torch_test(
    name = "scatter_gather_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "test_scatter_gather_ops",
)

torch_tpu_torch_test(
    name = "schema_check_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_schema_check",
)

torch_tpu_torch_test(
    name = "segment_reductions_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_segment_reductions",
)

torch_tpu_torch_test(
    name = "serialization_test",
    torch_test_module = "test_serialization",
)

torch_tpu_torch_test(
    name = "shape_ops_test",
    torch_test_module = "test_shape_ops",
)

torch_tpu_torch_test(
    name = "sort_and_select_test",
    shard_count = 4,
    torch_test_module = "test_sort_and_select",
)

torch_tpu_torch_test(
    name = "sparse_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_sparse",
)

torch_tpu_torch_test(
    name = "sparse_csr_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_sparse_csr",
)

torch_tpu_torch_test(
    name = "sparse_semi_structured_test",
    torch_test_module = "test_sparse_semi_structured",
)

torch_tpu_torch_test(
    name = "spectral_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_spectral_ops",
)

torch_tpu_torch_test(
    name = "stateless_test",
    torch_test_module = "test_stateless",
)

torch_tpu_torch_test(
    name = "stateless_rng_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    torch_test_module = "test_stateless_rng",
)

torch_tpu_torch_test(
    name = "tensor_creation_ops_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 8,
    torch_test_module = "test_tensor_creation_ops",
)

torch_tpu_torch_test(
    name = "tensor_iterator_test",
    torch_test_module = "test_tensor_iterator",
)

torch_tpu_torch_test(
    name = "tensorboard_test",
    torch_test_module = "test_tensorboard",
)

torch_tpu_torch_test(
    name = "testing_test",
    shard_count = 4,
    torch_test_module = "test_testing",
)

torch_tpu_torch_test(
    name = "torch_test",
    shard_count = 4,
    torch_test_module = "test_torch",
)

torch_tpu_torch_test(
    name = "transformers_test",
    torch_test_module = "test_transformers",
)

torch_tpu_torch_test(
    name = "type_promotion_test",
    shard_count = 4,
    torch_test_module = "test_type_promotion",
)

torch_tpu_torch_test(
    name = "unary_ufuncs_test",
    notap = "b/563402379",  # NOTAP_OK=torch tests
    shard_count = 50,
    torch_test_module = "test_unary_ufuncs",
)

torch_tpu_torch_test(
    name = "utils_test",
    torch_test_module = "test_utils",
)

torch_tpu_torch_test(
    name = "view_ops_test",
    shard_count = 4,
    torch_test_module = "test_view_ops",
)

torch_tpu_torch_test(
    name = "torch_np_basic_test",
    torch_test_module = "torch_np/test_basic",
)
# GENERATED_TEST_TARGETS_END

torch_tpu_package_end()
