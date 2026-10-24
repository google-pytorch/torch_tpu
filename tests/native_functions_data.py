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

"""Ops that are missing a required C++ registration.

This list contains ops from native_functions.yaml that are required to be
registered by the logic in native_functions_test.py, but do not have a
corresponding C++ registration in tpu_aten_kernels.cc.

This is a temporary list that should be emptied, either by adding a C++
registration for the op, or by adjusting the logic in native_functions_test.py
that determines which ops need a C++ registration.
"""

UNREGISTERED_OPS = frozenset([
    # go/keep-sorted start
    '_aminmax',
    '_aminmax.dim',
    '_amp_foreach_non_finite_check_and_unscale_',
    '_amp_update_scale_',
    '_batch_norm_with_update',
    '_batch_norm_with_update.out',
    '_cholesky_solve_helper',
    '_coalesce',
    '_coalesced_',
    '_compute_linear_combination',
    '_compute_linear_combination.out',
    '_conv_depthwise2d',
    '_conv_depthwise2d.out',
    '_convert_indices_from_coo_to_csr.out',
    '_convert_indices_from_csr_to_coo.out',
    '_convert_weight_to_int4pack',
    '_convert_weight_to_int4pack_for_cpu',
    '_cslt_compress',
    '_cslt_sparse_mm',
    '_cslt_sparse_mm_search',
    '_dimI',
    '_dimV',
    '_dirichlet_grad',
    '_dyn_quant_matmul_4bit',
    '_dyn_quant_pack_4bit_weight',
    '_efficient_attention_backward',
    '_efficient_attention_forward',
    '_embedding_bag_dense_backward',
    '_embedding_bag_per_sample_weights_backward',
    '_empty_affine_quantized',
    '_empty_per_channel_affine_quantized',
    '_fake_quantize_learnable_per_channel_affine',
    '_fake_quantize_learnable_per_channel_affine_backward',
    '_fake_quantize_learnable_per_tensor_affine',
    '_fake_quantize_learnable_per_tensor_affine_backward',
    '_fake_quantize_per_tensor_affine_cachemask_tensor_qparams',
    '_fill_mem_eff_dropout_mask_',
    '_flash_attention_backward',
    '_flash_attention_forward',
    '_flash_attention_forward.quantized',
    '_flash_attention_forward_no_dropout_inplace',
    '_functional_assert_async.msg',
    '_histogramdd_bin_edges',
    '_histogramdd_from_bin_cts',
    '_histogramdd_from_bin_tensors',
    '_indices',
    '_int_mm',
    '_int_mm.out',
    '_linalg_det.result',
    '_linalg_eigh.eigenvalues',
    '_linalg_eigvals',
    '_linalg_slogdet.sign',
    '_lstm_mps',
    '_make_dep_token',
    '_make_per_channel_quantized_tensor',
    '_make_per_tensor_quantized_tensor',
    '_masked_scale',
    '_mixed_dtypes_linear',
    '_mkldnn_reshape',
    '_mkldnn_transpose',
    '_mkldnn_transpose_',
    '_mps_convolution',
    '_mps_convolution_transpose',
    # Legacy NestedTensor 1.0 operators (buffer-based storage with
    # nested_sizes/strides metadata, superseded by modern torch.jagged):
    '_nested_compute_contiguous_strides_offsets',
    '_nested_from_padded',
    '_nested_from_padded_and_nested_example',
    '_nested_select_backward',
    '_nested_sum_backward',
    '_nested_tensor_from_mask',
    '_nested_tensor_from_mask_left_aligned',
    '_nested_tensor_size',
    '_nested_tensor_softmax_with_shape',
    '_nested_tensor_storage_offsets',
    '_nested_tensor_strides',
    '_nested_view_from_buffer',
    # Other unregistered operators:
    '_nnz',
    '_philox_key_fold_in',
    '_philox_key_fold_in.Tensor',
    '_philox_key_split',
    '_philox_normal_',
    '_philox_randint_',
    '_philox_uniform_',
    '_resize_output_',
    '_sample_dirichlet',
    '_scaled_addmm.out',
    '_scaled_dot_product_attention_math_for_mps',
    '_scaled_dot_product_efficient_attention',
    '_scaled_dot_product_efficient_attention_backward',
    '_scaled_dot_product_flash_attention',
    '_scaled_dot_product_flash_attention.quantized',
    '_scaled_dot_product_flash_attention_backward',
    '_scaled_dot_product_flash_attention_for_cpu',
    '_scaled_dot_product_flash_attention_for_cpu_backward',
    '_segment_reduce_backward',
    '_slow_conv2d_backward.grad_input',
    '_slow_conv2d_backward.output_mask',
    '_slow_conv2d_forward',
    '_slow_conv2d_forward.output',
    '_sparse_broadcast_to',
    '_sparse_coo_tensor_with_dims',
    '_sparse_coo_tensor_with_dims_and_tensors',
    '_sparse_csr_prod.dim_dtype',
    '_sparse_csr_sum.dim_dtype',
    '_sparse_log_softmax',
    '_sparse_log_softmax_backward_data',
    '_sparse_mask_projection',
    '_sparse_mm_reduce_impl',
    '_sparse_mm_reduce_impl_backward',
    '_sparse_semi_structured_addmm',
    '_sparse_semi_structured_apply',
    '_sparse_semi_structured_apply_dense',
    '_sparse_semi_structured_linear',
    '_sparse_semi_structured_mm',
    '_sparse_semi_structured_tile',
    '_sparse_softmax',
    '_sparse_softmax_backward_data',
    '_sparse_sparse_matmul',
    '_sparse_sum_backward',
    '_spdiags',
    '_spsolve',
    '_standard_gamma',
    '_standard_gamma_grad',
    '_to_dense',
    '_to_sparse',
    '_to_sparse.sparse_dim',
    '_to_sparse_bsc',
    '_to_sparse_bsr',
    '_to_sparse_csc',
    '_to_sparse_csr',
    '_to_sparse_semi_structured',
    '_transform_bias_rescale_qkv',
    '_triton_multi_head_attention',
    '_triton_scaled_dot_attention',
    '_unique',
    '_upsample_bicubic2d_aa_backward.grad_input',
    '_upsample_bilinear2d_aa_backward.grad_input',
    '_upsample_lanczos2d_aa.out',
    '_upsample_lanczos2d_aa_backward.grad_input',
    '_use_miopen_ctc_loss',
    '_use_miopen_ctc_loss.Tensor',
    '_validate_compressed_sparse_indices',
    '_values',
    '_weight_int4pack_mm',
    '_weight_int4pack_mm_for_cpu',
    '_weight_int4pack_mm_with_scales_and_zeros',
    '_weight_int8pack_mm',
    'adaptive_max_pool2d_backward.grad_input',
    'adaptive_max_pool3d.out',
    'adaptive_max_pool3d_backward.grad_input',
    'addbmm',
    'addbmm.out',
    'addbmm_',
    'batch_norm_backward',
    'batch_norm_backward_elemt',
    'batch_norm_backward_reduce',
    'batch_norm_elemt',
    'batch_norm_elemt.out',
    'batch_norm_gather_stats',
    'batch_norm_gather_stats_with_counts',
    'batch_norm_stats',
    'batch_norm_update_stats',
    'binomial',
    'cauchy_',
    'cholesky_inverse',
    'cholesky_inverse.out',
    'conv_depthwise3d',
    'copy_sparse_to_sparse_',
    'copysign.out',
    'dequantize.self',
    'dequantize.tensors',
    'empty_quantized',
    'erfc.out',
    'fake_quantize_per_channel_affine_cachemask',
    'fractional_max_pool2d.output',
    'fractional_max_pool2d_backward.grad_input',
    'fractional_max_pool3d.output',
    'fractional_max_pool3d_backward',
    'fractional_max_pool3d_backward.grad_input',
    'frexp.Tensor_out',
    'from_file',
    'gcd.out',
    'geometric_',
    'glu_backward_jvp',
    'glu_jvp',
    'hardshrink.out',
    'hardshrink_backward.grad_input',
    'hash_tensor.out',
    'heaviside.out',
    'histogram.bin_ct',
    'histogram.bin_ct_out',
    'histogram.bins_tensor',
    'histogram.bins_tensor_out',
    'hspmm',
    'hspmm.out',
    'huber_loss',
    'huber_loss.out',
    'huber_loss_backward.out',
    'hypot.out',
    'i0.out',
    'igamma.out',
    'igammac.out',
    'index_select.out',
    'int_repr',
    'is_set_to',
    'kthvalue.values',
    'linalg_cross.out',
    'linalg_eig',
    'linalg_eig.out',
    'linalg_eigvals.out',
    'linalg_householder_product',
    'linalg_householder_product.out',
    'linalg_ldl_factor_ex.out',
    'linalg_ldl_solve.out',
    'linalg_lstsq.out',
    'linalg_matrix_exp',
    'linalg_matrix_sqrth',
    'linalg_polar.out',
    'log_normal_',
    'logaddexp.out',
    'logaddexp2.out',
    'logspace.out',
    'lstm_mps_backward',
    'matmul_backward',  # No need to register as matmul is an optional op
    'max_pool2d_backward',
    'max_unpool2d',
    'max_unpool2d.out',
    'max_unpool3d',
    'max_unpool3d.out',
    'median',
    'median.dim_values',
    'miopen_batch_norm',
    'miopen_batch_norm_backward',
    'miopen_convolution',
    'miopen_convolution_add_relu',
    'miopen_convolution_relu',
    'miopen_convolution_transpose',
    'miopen_ctc_loss',
    'miopen_ctc_loss.Tensor',
    'miopen_depthwise_convolution',
    'miopen_rnn',
    'miopen_rnn_backward',
    'mish.out',
    'mkldnn_adaptive_avg_pool2d',
    'mkldnn_adaptive_avg_pool2d.out',
    'mkldnn_adaptive_avg_pool2d_backward',
    'mkldnn_linear',
    'mkldnn_linear_backward',
    'mkldnn_linear_backward_input',
    'mkldnn_linear_backward_weights',
    'mkldnn_max_pool2d',
    'mkldnn_max_pool2d_backward',
    'mkldnn_max_pool3d',
    'mkldnn_max_pool3d_backward',
    'mkldnn_reorder_conv2d_weight',
    'mkldnn_reorder_conv3d_weight',
    'mkldnn_rnn_layer',
    'mkldnn_rnn_layer_backward',
    'mode',
    'mps_convolution_backward',
    'mps_convolution_transpose_backward',
    'mse_loss_backward.grad_input',
    'multi_margin_loss',
    'multi_margin_loss.out',
    'multi_margin_loss_backward',
    'multi_margin_loss_backward.grad_input',
    'multilabel_margin_loss_backward',
    'multilabel_margin_loss_backward.grad_input',
    'multilabel_margin_loss_forward',
    'multilabel_margin_loss_forward.output',
    'mvlgamma.out',
    'nanmedian',
    'nanmedian.dim_values',
    'nansum',
    'nansum.out',
    'narrow_copy.out',
    'nextafter.out',
    'nonzero_static',
    'nonzero_static.out',
    'ormqr',
    'ormqr.out',
    'poisson',
    'q_per_channel_axis',
    'q_per_channel_scales',
    'q_per_channel_zero_points',
    'q_scale',
    'q_zero_point',
    'qscheme',
    'quantize_per_channel',
    'quantize_per_tensor',
    'quantize_per_tensor.tensor_qparams',
    'quantize_per_tensor.tensors',
    'quantize_per_tensor_dynamic',
    'quantized_batch_norm',
    'quantized_max_pool1d',
    'quantized_max_pool2d',
    'quantized_max_pool3d',
    'range.out',
    'record_stream',
    'renorm.out',
    'resize_as_sparse_',
    'rrelu_with_noise',
    'rrelu_with_noise.out',
    'rrelu_with_noise_',
    'sinc.out',
    'slow_conv3d_forward',
    'slow_conv3d_forward.output',
    'slow_conv_dilated2d',
    'slow_conv_dilated3d',
    'slow_conv_transpose2d.out',
    'slow_conv_transpose3d',
    'slow_conv_transpose3d.out',
    'smooth_l1_loss.out',
    'smooth_l1_loss_backward.grad_input',
    'softshrink.out',
    'softshrink_backward.grad_input',
    'sparse_mask',
    'sparse_resize_',
    'sparse_resize_and_clear_',
    'sparse_sampled_addmm',
    'sparse_sampled_addmm.out',
    'special_airy_ai.out',
    'special_bessel_j0.out',
    'special_bessel_j1.out',
    'special_bessel_y0.out',
    'special_bessel_y1.out',
    'special_chebyshev_polynomial_t.out',
    'special_chebyshev_polynomial_u.out',
    'special_chebyshev_polynomial_v.out',
    'special_chebyshev_polynomial_w.out',
    'special_entr.out',
    'special_erfcx.out',
    'special_hermite_polynomial_h.out',
    'special_hermite_polynomial_he.out',
    'special_i0e.out',
    'special_i1.out',
    'special_i1e.out',
    'special_laguerre_polynomial_l.out',
    'special_legendre_polynomial_p.out',
    'special_log_ndtr.out',
    'special_modified_bessel_i0.out',
    'special_modified_bessel_i1.out',
    'special_modified_bessel_k0.out',
    'special_modified_bessel_k1.out',
    'special_ndtri.out',
    'special_scaled_modified_bessel_k0.out',
    'special_scaled_modified_bessel_k1.out',
    'special_shifted_chebyshev_polynomial_t.out',
    'special_shifted_chebyshev_polynomial_u.out',
    'special_shifted_chebyshev_polynomial_v.out',
    'special_shifted_chebyshev_polynomial_w.out',
    'special_spherical_bessel_j0.out',
    'special_xlog1py.out',
    'special_zeta.out',
    'sspaddmm.out',
    'std_mean.correction',
    'to_mkldnn',
    'to_padded_tensor',
    'trace',
    'triangular_solve.X',
    'triu_indices',
    'unique_dim',
    'upsample_trilinear3d.out',
    'upsample_trilinear3d_backward.grad_input',
    # go/keep-sorted end
])

"""Ops that are registered in C++ but don't need to be as pytorch automatically
delegates them to other ops. Each op should be commented with why it needs to be
registered.

This list contains ops that have registered C++ kernels, but for which we don't
require a C++ registration, either because they are not defined, or because
the logic in native_functions_test.py that determines which ops need a C++
registration has filtered them out.
"""
REGISTRATION_OVERRIDES = frozenset([
    # go/keep-sorted start
    # _fused_rms_norm and _scaled_dot_product_fused_attention_overrideable are
    # registered for fused forward/backward lowerings.
    '_fused_rms_norm',
    '_grouped_mm',
    # This is a PrivateUse1 specific op that communicates shallow-copyability.
    # If it is missing, parameter tieing falls back to untied copies.
    '_has_compatible_shallow_copy_type',
    '_masked_softmax.out',
    '_masked_softmax_backward.out',
    '_scaled_dot_product_cudnn_attention',
    '_scaled_dot_product_cudnn_attention_backward',
    '_scaled_dot_product_fused_attention_overrideable',
    '_scaled_dot_product_fused_attention_overrideable_backward',
    '_scaled_grouped_mm_v2',
    '_scaled_mm_v2',
    '_thnn_fused_gru_cell.out',
    '_thnn_fused_lstm_cell.out',
    '_to_copy',
    # avg_pool1d and avg_pool1d.out are registered to provide a direct 1D
    # pooling lowering on TPU, overriding the default PyTorch decomposition.
    'avg_pool1d',
    'avg_pool1d.out',
    # constant_pad_nd is registered to lower directly to StableHLO Concatenate
    # instead of empty + fill_ + slice.copy_ (DynamicUpdateSlice).
    'constant_pad_nd',
    'convolution',
    'convolution.out',
    'convolution_backward',
    # copy_ is registered in copy_from_aten_kernels.cc to avoid an infinite
    # recursion when the input is a complex number with the conjugate bit set.
    # The loop is copy_ -> MathOpFallback::fallback_impl -> clone -> copy_...
    # Registering copy_ as an alias for _copy_from breaks this loop.
    'copy_',
    # cross_entropy_loss is registered to provide a fused StableHLO lowering
    # and autograd implementation on TPU, overriding the default decomposition.
    'cross_entropy_loss',
    # TODO: b/513607161 - revert once upstream PyTorch bug is fixed.
    # ctc_loss and ctc_loss.Tensor are registered because we provide a custom
    # StableHLO lowering and autograd implementation, bypassing the default
    # decomposition in order to workaround a bug in PyTorch core.
    'ctc_loss',
    'ctc_loss.Tensor',
    # einsum is registered directly on PrivateUse1 and AutogradPrivateUse1 to
    # lower contractions directly to stablehlo.dot_general instead of
    # decomposing into permute + reshape + bmm + view + permute.
    'einsum',
    # embedding, embedding.out, embedding_backward, and embedding_sparse_backward
    # are registered to provide direct StableHLO lowerings on TPU, overriding
    # the default PyTorch decomposition.
    'embedding',
    'embedding.out',
    'embedding_backward',
    'embedding_sparse_backward',
    # TODO: b/518595804 - PyTorch's promote_type_fft omits
    # device.is_privateuseone() from maybe_support_half.
    # Register FFT ops to allow manually casting BF16 to F32, so that
    # NotImplementedError is not thrown.
    'fft_fft',
    'fft_fft.out',
    'fft_ifft',
    'fft_ifft.out',
    'fft_irfft',
    'fft_irfft.out',
    'fft_rfft',
    'fft_rfft.out',
    # full and full_like are registered directly to lower to a single StableHLO
    # constant instead of the composite empty + fill_ double dispatch.
    'full',
    'full_like',
    'gather',
    # Gelu is registered to handle type promotion for integer and boolean
    # inputs to match PyTorch GPU behavior for functional calls.
    'gelu',
    # gru.input is registered to provide an optimized fused StableHLO kernel
    # on TPU, overriding the default PyTorch decomposition.
    'gru.input',
    # TODO(b/525550199): Remove ldexp overrides once we upgrade to PyTorch
    # v2.13.
    'ldexp.Tensor',
    'ldexp.out',
    'ldexp_',
    # linear and linear.out are registered to provide direct StableHLO lowerings
    # via DotGeneral on TPU, overriding the default PyTorch decomposition.
    'linear',
    'linear.out',
    # lstm.input is registered to provide an optimized fused StableHLO kernel
    # on TPU, overriding the default PyTorch decomposition.
    'lstm.input',
    # masked_fill.Scalar and masked_fill.Tensor are registered to lower directly
    # to StableHLO Select without decomposing into expand + clone +
    # masked_fill_.
    'masked_fill.Scalar',
    'masked_fill.Tensor',
    'native_layer_norm',  # pending removal
    # new_full, new_ones, new_zeros, ones, and ones_like are registered
    # directly to lower to a single StableHLO constant instead of the composite
    # empty + fill_/zero_ double dispatch.
    'new_full',
    'new_ones',
    'new_zeros',
    # nll_loss_nd is registered to provide a direct N-D StableHLO lowering and
    # autograd implementation on TPU, overriding the default decomposition.
    'nll_loss_nd',
    'ones',
    'ones_like',
    # repeat_interleave.Tensor_out is registered to write indices directly into
    # out via DispatchOpOut, overriding the autogen copy decomposition.
    'repeat_interleave.Tensor_out',
    # repeat_interleave.self_Tensor is registered to provide an optimized TPU
    # implementation, overriding the default decomposition.
    'repeat_interleave.self_Tensor',
    # repeat_interleave.self_int is registered to provide an optimized TPU
    # implementation, overriding the default decomposition.
    'repeat_interleave.self_int',
    # rms_norm is registered to provide a direct 1-output inference lowering
    # and mixed-dtype/non-contiguous support on TPU, overriding the default
    # CompositeImplicitAutograd decomposition.
    'rms_norm',
    # select_backward is registered to lower directly to a single StableHLO
    # DynamicUpdateSlice operation instead of zeros + select + copy_.
    'select_backward',
    # select_scatter is registered to lower directly to StableHLO
    # DynamicUpdateSlice or identity reshape, overriding the default
    # clone + select + copy_ decomposition.
    'select_scatter',
    'select_scatter.out',
    # silu_backward is registered to lower directly to a single fused StableHLO
    # kernel (BuildSiluBackwardShlo), overriding the default decomposition.
    'silu_backward',
    # slice_backward is registered to lower directly to a single StableHLO
    # DynamicUpdateSlice, Concatenate, or Pad operation instead of zeros +
    # slice + copy_.
    'slice_backward',
    # slice_scatter is registered to lower directly to StableHLO
    # DynamicUpdateSlice or Concatenate with algebraic strength reductions,
    # overriding the default clone + slice + copy_ decomposition.
    'slice_scatter',
    'slice_scatter.out',
    # split_with_sizes_copy.out is used by PyTorch's FSDP2 wrapper. The default
    # decomposition performs 2-d reshapes that are inefficient on TPU.
    'split_with_sizes_copy.out',
    # stack and stack.out are registered to lower directly to StableHLO
    # Reshape + Concatenate without decomposing into per-tensor unsqueeze view
    # creation + cat.
    'stack',
    'stack.out',
    # zeros and zeros_like are registered directly to lower to a single
    # StableHLO constant instead of the composite empty + zero_ double dispatch.
    'zeros',
    'zeros_like',
    # go/keep-sorted end
])

"""Regex version of REGISTRATION_OVERRIDES. The regex is matched against the
registered ops.
"""
REGISTRATION_OVERRIDES_REGEX = frozenset([
    # go/keep-sorted start
    '_foreach_.*',
    # go/keep-sorted end
])
