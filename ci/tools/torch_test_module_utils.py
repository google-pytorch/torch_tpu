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

"""Utilities to work with upstream PyTorch test modules."""

from collections.abc import Mapping
from typing import Final

# TODO(b/563402379): triage failures and enable all test modules.
DISABLED_TEST_MODULES: Final[frozenset[str]] = frozenset({
    # go/keep-sorted start
    "complex_tensor/test_complex_tensor",
    "distributed/fsdp/test_fsdp_apply",
    "distributed/fsdp/test_fsdp_checkpoint",
    "distributed/fsdp/test_fsdp_clip_grad_norm",
    "distributed/fsdp/test_fsdp_comm",
    "distributed/fsdp/test_fsdp_core",
    "distributed/fsdp/test_fsdp_dtensor_state_dict",
    "distributed/fsdp/test_fsdp_exec_order",
    "distributed/fsdp/test_fsdp_fine_tune",
    "distributed/fsdp/test_fsdp_input",
    "distributed/fsdp/test_fsdp_multiple_forward",
    "distributed/fsdp/test_fsdp_multiple_wrapping",
    "distributed/fsdp/test_fsdp_overlap",
    "distributed/fsdp/test_fsdp_pure_fp16",
    "distributed/fsdp/test_fsdp_traversal",
    "distributed/fsdp/test_fsdp_uneven",
    "distributed/fsdp/test_hsdp_dtensor_state_dict",
    "distributed/pipelining/test_microbatch",
    "distributed/tensor/test_dtensor_ops",
    "distributed/test_backends",
    "distributed/test_functional_api",
    "distributed/test_functional_differentials",
    "distributions/test_distributions",
    "dynamo/test_aot_autograd",
    "dynamo/test_dynamo_decompositions",
    "dynamo/test_dynamo_ops",
    "dynamo/test_streams",
    "export/test_hop",
    "export/test_opinfo_schema",
    "export/test_torchbind",
    "functorch/test_control_flow",
    "functorch/test_eager_transforms",
    "functorch/test_ops",
    "functorch/test_vmap",
    "higher_order_ops/test_print",
    "higher_order_ops/test_with_effects",
    "inductor/test_compiled_autograd",
    "inductor/test_compiled_optimizers",
    "inductor/test_flex_attention",
    "inductor/test_flex_decoding",
    "inductor/test_fp8",
    "inductor/test_inductor_scheduler",
    "inductor/test_torchinductor_codegen_dynamic_shapes",
    "inductor/test_torchinductor_opinfo",
    "inductor/test_unbacked_symints",
    "lazy/test_ts_opinfo",
    "nn/test_convolution",
    "nn/test_dropout",
    "nn/test_embedding",
    "nn/test_init",
    "nn/test_lazy_modules",
    "nn/test_multihead_attention",
    "nn/test_pooling",
    "profiler/test_execution_trace",
    "profiler/test_memory_profiler",
    "profiler/test_profiler",
    "quantization/core/experimental/test_bits",
    "quantization/core/experimental/test_floatx",
    "test_accelerator",
    "test_ao_sparsity",
    "test_autograd",
    "test_binary_ufuncs",
    "test_bmm_outer_product",
    "test_complex",
    "test_decomp",
    "test_dlpack",
    "test_expanded_weights",
    "test_foreach",
    "test_fx",
    "test_indexing",
    "test_legacy_vmap",
    "test_linalg",
    "test_masked",
    "test_maskedtensor",
    "test_matmul_cuda",
    "test_meta",
    "test_modules",
    "test_multiprocessing",
    "test_native_mha",
    "test_nestedtensor",
    "test_nn",
    "test_numpy_interop",
    "test_opaque_obj_v2",
    "test_ops",
    "test_ops_fwd_gradients",
    "test_ops_gradients",
    "test_ops_jit",
    "test_ops_unbacked",
    "test_optim",
    "test_overrides",
    "test_precompile",
    "test_prims",
    "test_reductions",
    "test_scaled_matmul_cuda",
    "test_scatter_gather_ops",
    "test_schema_check",
    "test_segment_reductions",
    "test_serialization",
    "test_shape_ops",
    "test_sort_and_select",
    "test_sparse",
    "test_sparse_csr",
    "test_spectral_ops",
    "test_stateless_rng",
    "test_tensor_creation_ops",
    "test_testing",
    "test_torch",
    "test_transformers",
    "test_type_promotion",
    "test_unary_ufuncs",
    "test_utils",
    "test_view_ops",
    # go/keep-sorted end
})

# Mapping from test module names to number of shards.
# Per https://bazel.build/reference/be/common-definitions#common-attributes-tests,
# Bazel accepts shard counts in the range of [1, 50].
TEST_MODULE_SHARDS: Final[Mapping[str, int]] = {
    # go/keep-sorted start
    "dynamo/test_misc": 8,
    "inductor/test_compiled_optimizers": 8,
    "inductor/test_torchinductor_dynamic_shapes": 8,
    "inductor/test_torchinductor_opinfo": 16,
    "inductor/test_torchinductor_opinfo_properties": 8,
    "nn/test_convolution": 8,
    "nn/test_embedding": 2,
    "nn/test_pooling": 4,
    "profiler/test_profiler": 4,
    "test_binary_ufuncs": 8,
    "test_decomp": 50,
    "test_foreach": 4,
    "test_indexing": 4,
    "test_linalg": 4,
    "test_meta": 30,
    "test_modules": 2,
    "test_nestedtensor": 5,
    "test_nn": 2,
    "test_ops": 50,
    "test_ops_fwd_gradients": 5,
    "test_ops_gradients": 20,
    "test_ops_jit": 6,
    "test_optim": 2,
    "test_overrides": 15,
    "test_reductions": 2,
    "test_scatter_gather_ops": 8,
    "test_sort_and_select": 2,
    "test_tensor_creation_ops": 8,
    "test_testing": 4,
    "test_torch": 8,
    "test_type_promotion": 4,
    "test_unary_ufuncs": 50,
    "test_view_ops": 4,
    # go/keep-sorted end
}

# Test modules with no device-generic tests for TPU, including
#   1. Test modules specific to non-TPU accelerators, e.g. NCCL for Nvidia GPU.
#   2. Test modules where every test class is labelled for other hardware or
#      as GENERIC, so no TPU variant is instantiated.
_ACCELERATOR_SPECIFIC_TEST_MODULES: Final[frozenset[str]] = frozenset({
    # go/keep-sorted start
    "distributed/algorithms/ddp_comm_hooks/test_ddp_hooks",
    "distributed/launcher/test_run",
    "distributed/test_c10d_nccl2",
    "distributed/test_c10d_nccl_single_rank",
    "distributed/test_cupy_as_tensor",
    "distributed/test_data_parallel",
    "distributed/test_nccl",
    "distributed/test_overlap_bucketing_unit",
    "distributed/test_symmetric_memory",
    "dynamo/test_compiler_bisector",
    "export/test_export_opinfo",
    "functorch/test_aotdispatch",
    "higher_order_ops/test_inline_asm_elementwise",
    "inductor/test_aoti_torchbind_constants",
    "inductor/test_cpu_select_algorithm",
    "inductor/test_cudacodecache",
    "inductor/test_cudagraph_trees",
    "inductor/test_flex_flash",
    "inductor/test_flex_gemm",
    "inductor/test_flydsl_grouped_scheduler",
    "inductor/test_flydsl_template",
    "inductor/test_gpu_select_algorithm",
    "inductor/test_mkldnn_pattern_matcher",
    "inductor/test_reorder_for_locality_in_training",
    "inductor/test_strict_numerics",
    "inductor/test_symm_mem_registry",
    "inductor/test_symmetric_mm",
    "inductor/test_triton_helpers",
    "lazy/test_reuse_ir",
    "nn/attention/test_fa3",
    "nn/attention/test_fa4",
    "profiler/test_kineto",
    "test_cuda",
    "test_flop_counter",
    "test_functionalization_of_rng_ops",
    "test_jit_fuser_te",
    "test_jit_llga_fuser",
    "test_jiterator",
    "test_mkldnn",
    "test_mps",
    "test_varlen_attention",
    "test_xpu",
    "xpu/test_conv",
    "xpu/test_fusion",
    "xpu/test_gemm",
    "xpu/test_lstm",
    "xpu/test_transformers",
    # go/keep-sorted end
})

# Test modules that cannot run in TorchTPU test environment.
_UNRUNNABLE_TEST_MODULES: Final[frozenset[str]] = frozenset({
    # go/keep-sorted start
    # Tests a mock OpenReg device backend, which conflicts with TorchTPU
    # PrivateUse1 registration.
    "cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing",
    # Invokes pip and compiles a Python wheel dynamically during setUpClass.
    "cpp_extensions/python_agnostic_extension/test/test_python_agnostic",
    # Compiles and loads a C++ extension at import time.
    "cpp_extensions/test_libtorch_agnostic",
    # Only runnable via test_jit.py.
    "jit/test_dtype_analysis",
    # Requires onnxruntime to validate ONNX export operator.
    "onnx/test_op_consistency",
    # Requires onnxruntime and onnxscript reference runtimes.
    "onnx/torchlib/test_ops",
    # Compiles native C++ extension code.
    "profiler/test_cpp_thread",
    # go/keep-sorted end
})

EXCLUDED_TEST_MODULES: Final[frozenset[str]] = (
    _ACCELERATOR_SPECIFIC_TEST_MODULES | _UNRUNNABLE_TEST_MODULES
)
