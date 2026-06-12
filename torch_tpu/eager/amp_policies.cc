/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torch_tpu/eager/amp_policies.h"

#include "ATen/autocast_mode.h"
#include "ATen/ops/_addmm_activation_ops.h"
#include "ATen/ops/_convolution_ops.h"
#include "ATen/ops/_grid_sampler_2d_cpu_fallback_ops.h"
#include "ATen/ops/_lu_with_info_ops.h"
#include "ATen/ops/_native_multi_head_attention_ops.h"
#include "ATen/ops/_scaled_dot_product_flash_attention_ops.h"
#include "ATen/ops/_thnn_fused_lstm_cell_ops.h"
#include "ATen/ops/adaptive_avg_pool3d_ops.h"
#include "ATen/ops/adaptive_max_pool3d_ops.h"
#include "ATen/ops/addbmm_ops.h"
#include "ATen/ops/addmm_ops.h"
#include "ATen/ops/addmv_ops.h"
#include "ATen/ops/addr_ops.h"
#include "ATen/ops/avg_pool3d_ops.h"
#include "ATen/ops/baddbmm_ops.h"
#include "ATen/ops/binary_cross_entropy_ops.h"
#include "ATen/ops/binary_cross_entropy_with_logits_ops.h"
#include "ATen/ops/bmm_ops.h"
#include "ATen/ops/cdist_ops.h"
#include "ATen/ops/chain_matmul_ops.h"
#include "ATen/ops/cholesky_inverse_ops.h"
#include "ATen/ops/cholesky_solve_ops.h"
#include "ATen/ops/conv1d_ops.h"
#include "ATen/ops/conv2d_ops.h"
#include "ATen/ops/conv3d_ops.h"
#include "ATen/ops/conv_tbc_ops.h"
#include "ATen/ops/conv_transpose1d_ops.h"
#include "ATen/ops/convolution_ops.h"
#include "ATen/ops/cosine_embedding_loss_ops.h"
#include "ATen/ops/cross_entropy_loss_ops.h"
#include "ATen/ops/einsum_ops.h"
#include "ATen/ops/fake_quantize_per_tensor_affine_ops.h"
#include "ATen/ops/fft_fft2_ops.h"
#include "ATen/ops/fft_fft_ops.h"
#include "ATen/ops/fft_fftn_ops.h"
#include "ATen/ops/fft_hfft_ops.h"
#include "ATen/ops/fft_ifft2_ops.h"
#include "ATen/ops/fft_ifft_ops.h"
#include "ATen/ops/fft_ifftn_ops.h"
#include "ATen/ops/fft_ihfft_ops.h"
#include "ATen/ops/fft_irfft2_ops.h"
#include "ATen/ops/fft_irfft_ops.h"
#include "ATen/ops/fft_irfftn_ops.h"
#include "ATen/ops/fft_rfft2_ops.h"
#include "ATen/ops/fft_rfft_ops.h"
#include "ATen/ops/fft_rfftn_ops.h"
#include "ATen/ops/fractional_max_pool2d_ops.h"
#include "ATen/ops/fractional_max_pool3d_ops.h"
#include "ATen/ops/geqrf_ops.h"
#include "ATen/ops/grid_sampler_2d_ops.h"
#include "ATen/ops/grid_sampler_3d_ops.h"
#include "ATen/ops/grid_sampler_ops.h"
#include "ATen/ops/gru_cell_ops.h"
#include "ATen/ops/hinge_embedding_loss_ops.h"
#include "ATen/ops/huber_loss_ops.h"
#include "ATen/ops/inverse_ops.h"
#include "ATen/ops/kl_div_ops.h"
#include "ATen/ops/l1_loss_ops.h"
#include "ATen/ops/linalg_cholesky_ex_ops.h"
#include "ATen/ops/linalg_cholesky_ops.h"
#include "ATen/ops/linalg_cond_ops.h"
#include "ATen/ops/linalg_eig_ops.h"
#include "ATen/ops/linalg_eigh_ops.h"
#include "ATen/ops/linalg_eigvals_ops.h"
#include "ATen/ops/linalg_eigvalsh_ops.h"
#include "ATen/ops/linalg_householder_product_ops.h"
#include "ATen/ops/linalg_inv_ex_ops.h"
#include "ATen/ops/linalg_inv_ops.h"
#include "ATen/ops/linalg_lstsq_ops.h"
#include "ATen/ops/linalg_matrix_rank_ops.h"
#include "ATen/ops/linalg_multi_dot_ops.h"
#include "ATen/ops/linalg_qr_ops.h"
#include "ATen/ops/linalg_solve_ops.h"
#include "ATen/ops/linalg_svd_ops.h"
#include "ATen/ops/linalg_svdvals_ops.h"
#include "ATen/ops/linalg_tensorinv_ops.h"
#include "ATen/ops/linalg_tensorsolve_ops.h"
#include "ATen/ops/linalg_vecdot_ops.h"
#include "ATen/ops/linear_ops.h"
#include "ATen/ops/lstm_cell_ops.h"
#include "ATen/ops/lu_solve_ops.h"
#include "ATen/ops/margin_ranking_loss_ops.h"
#include "ATen/ops/matmul_ops.h"
#include "ATen/ops/max_pool3d_ops.h"
#include "ATen/ops/max_unpool2d_ops.h"
#include "ATen/ops/max_unpool3d_ops.h"
#include "ATen/ops/mm_ops.h"
#include "ATen/ops/mse_loss_ops.h"
#include "ATen/ops/multi_margin_loss_ops.h"
#include "ATen/ops/multilabel_margin_loss_forward_ops.h"
#include "ATen/ops/multilabel_margin_loss_ops.h"
#include "ATen/ops/mv_ops.h"
#include "ATen/ops/nanquantile_ops.h"
#include "ATen/ops/nll_loss2d_ops.h"
#include "ATen/ops/nll_loss_ops.h"
#include "ATen/ops/orgqr_ops.h"
#include "ATen/ops/ormqr_ops.h"
#include "ATen/ops/pinverse_ops.h"
#include "ATen/ops/poisson_nll_loss_ops.h"
#include "ATen/ops/polar_ops.h"
#include "ATen/ops/prelu_ops.h"
#include "ATen/ops/prod_ops.h"
#include "ATen/ops/qr_ops.h"
#include "ATen/ops/quantile_ops.h"
#include "ATen/ops/reflection_pad1d_ops.h"
#include "ATen/ops/reflection_pad2d_ops.h"
#include "ATen/ops/replication_pad1d_ops.h"
#include "ATen/ops/replication_pad2d_ops.h"
#include "ATen/ops/replication_pad3d_ops.h"
#include "ATen/ops/rnn_relu_cell_ops.h"
#include "ATen/ops/rnn_tanh_cell_ops.h"
#include "ATen/ops/scaled_dot_product_attention_ops.h"
#include "ATen/ops/smooth_l1_loss_ops.h"
#include "ATen/ops/soft_margin_loss_ops.h"
#include "ATen/ops/stft_ops.h"
#include "ATen/ops/svd_ops.h"
#include "ATen/ops/trace_ops.h"
#include "ATen/ops/triangular_solve_ops.h"
#include "ATen/ops/triplet_margin_loss_ops.h"
#include "ATen/ops/view_as_complex_ops.h"
#include "torch/library.h"
#include "torch_tpu/common/macro_utils.h"

namespace torch_tpu {

// Register a fallback for automatic mixed precision (AMP).
// Anything not explicitly registered in another TORCH_LIBRARY_IMPL block will
// not force a dtype conversion (up or down).
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

// For formatting.
// The PyTorch "KERNEL_PRIVATEUSEONE" is required to register a policy for
// an autocast op, but this macro doesn't have a semicolon after its
// expansion, which breaks auto-formatting.
#define TT_AMP_POLICY(...) \
  KERNEL_PRIVATEUSEONE(__VA_ARGS__) TT_REQUIRE_SEMICOLON_

// When we apply autocast logic on a specific op, we should register it here.
// This can be used to force a lower precision for efficiency, or higher
// precision for accuracy, using:
//   `TT_AMP_POLICY(aten_op_name, lower_precision_fp);`
// or
//   `TT_AMP_POLICY(aten_op_name, fp32);`.
// or other CastPolicy values (see torch/aten/src/ATen/autocast_mode.h)
//
// Current policy (inferred from existing CPU/CUDA/MPS/etc. policies):
//  - Convolutions and matmul-type ops use lower_precision_fp
//  - Layer-specific "cell" ops (e.g. lstm_cell) match the convolution policy
//    for CUDA.
//  - We match the settings of the CPU autocast policy with regards to fp32.
//    Both CPU and TPU use bfloat16 by default, while CUDA uses float16, and
//    float16 is significantly more prone to overflow (5 exponent bits vs 8 for
//    bfloat16)
//    We prefer to use "fp32_set_opt_dtype" and "fp32_append_dtype" for kernels
//    which support them (see torch/aten/src/ATen/autocast_mode.h).
//  - We don't use "promote"; if an op would crash from having both a bfloat16
//    and a float32 input, we should fix that in the ops/ implementation, not
//    here.
//
// Note that we register policies for some kernels even when we just use the
// automatic fallthrough. For example, `mm` will automatically decompose to
// `mm.out + empty`; but, when using AMP, `mm` will infer an output dtype while
// `mm.out` will explicitly preserve the input dtype. This means that we need to
// specify the policy for `mm`, even though we implement `mm.out`.
// This means we can't use OpName (since that enum only has entries for the ops
// we actually implement).
TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // Convolution and matmul-type ops.
  // go/keep-sorted start
  TT_AMP_POLICY(_addmm_activation, lower_precision_fp);
  TT_AMP_POLICY(_convolution, deprecated, lower_precision_fp);
  TT_AMP_POLICY(_convolution, lower_precision_fp);
  TT_AMP_POLICY(_native_multi_head_attention, lower_precision_fp);
  TT_AMP_POLICY(_scaled_dot_product_flash_attention, lower_precision_fp);
  TT_AMP_POLICY(_thnn_fused_lstm_cell, lower_precision_fp);
  TT_AMP_POLICY(addbmm, lower_precision_fp);
  TT_AMP_POLICY(addmm, lower_precision_fp);
  TT_AMP_POLICY(addmv, lower_precision_fp);
  TT_AMP_POLICY(addr, lower_precision_fp);
  TT_AMP_POLICY(baddbmm, lower_precision_fp);
  TT_AMP_POLICY(bmm, lower_precision_fp);
  TT_AMP_POLICY(chain_matmul, lower_precision_fp);
  TT_AMP_POLICY(conv1d, lower_precision_fp);
  TT_AMP_POLICY(conv1d, padding, lower_precision_fp);
  TT_AMP_POLICY(conv2d, lower_precision_fp);
  TT_AMP_POLICY(conv2d, padding, lower_precision_fp);
  TT_AMP_POLICY(conv3d, lower_precision_fp);
  TT_AMP_POLICY(conv3d, padding, lower_precision_fp);
  TT_AMP_POLICY(conv_tbc, lower_precision_fp);
  TT_AMP_POLICY(conv_transpose1d, lower_precision_fp);
  TT_AMP_POLICY(conv_transpose2d, input, lower_precision_fp);
  TT_AMP_POLICY(conv_transpose3d, input, lower_precision_fp);
  TT_AMP_POLICY(convolution, lower_precision_fp);
  TT_AMP_POLICY(einsum, lower_precision_fp);
  TT_AMP_POLICY(gru_cell, lower_precision_fp);
  TT_AMP_POLICY(linalg_multi_dot, lower_precision_fp);
  TT_AMP_POLICY(linalg_vecdot, lower_precision_fp);
  TT_AMP_POLICY(linear, lower_precision_fp);
  TT_AMP_POLICY(lstm_cell, lower_precision_fp);
  TT_AMP_POLICY(matmul, lower_precision_fp);
  TT_AMP_POLICY(mm, lower_precision_fp);
  TT_AMP_POLICY(mv, lower_precision_fp);
  TT_AMP_POLICY(prelu, lower_precision_fp);
  TT_AMP_POLICY(rnn_relu_cell, lower_precision_fp);
  TT_AMP_POLICY(rnn_tanh_cell, lower_precision_fp);
  TT_AMP_POLICY(scaled_dot_product_attention, lower_precision_fp);
  // go/keep-sorted end

  // fp32 cast policy, matching the CPU autocast policy (aligning on bfloat16).
  // go/keep-sorted start
  TT_AMP_POLICY(_grid_sampler_2d_cpu_fallback, fp32);
  TT_AMP_POLICY(_lu_with_info, fp32);
  TT_AMP_POLICY(adaptive_avg_pool3d, fp32);
  TT_AMP_POLICY(adaptive_max_pool3d, fp32);
  TT_AMP_POLICY(avg_pool3d, fp32);
  TT_AMP_POLICY(binary_cross_entropy, fp32);
  TT_AMP_POLICY(binary_cross_entropy_with_logits, fp32);
  TT_AMP_POLICY(cdist, fp32);
  TT_AMP_POLICY(cholesky_inverse, fp32);
  TT_AMP_POLICY(cholesky_solve, fp32);
  TT_AMP_POLICY(cosine_embedding_loss, fp32);
  TT_AMP_POLICY(cross_entropy_loss, fp32);
  TT_AMP_POLICY(ctc_loss, IntList, fp32);
  TT_AMP_POLICY(ctc_loss, Tensor, fp32);
  TT_AMP_POLICY(fake_quantize_per_tensor_affine, fp32);
  TT_AMP_POLICY(fft_fft, fp32);
  TT_AMP_POLICY(fft_fft2, fp32);
  TT_AMP_POLICY(fft_fftn, fp32);
  TT_AMP_POLICY(fft_hfft, fp32);
  TT_AMP_POLICY(fft_ifft, fp32);
  TT_AMP_POLICY(fft_ifft2, fp32);
  TT_AMP_POLICY(fft_ifftn, fp32);
  TT_AMP_POLICY(fft_ihfft, fp32);
  TT_AMP_POLICY(fft_irfft, fp32);
  TT_AMP_POLICY(fft_irfft2, fp32);
  TT_AMP_POLICY(fft_irfftn, fp32);
  TT_AMP_POLICY(fft_rfft, fp32);
  TT_AMP_POLICY(fft_rfft2, fp32);
  TT_AMP_POLICY(fft_rfftn, fp32);
  TT_AMP_POLICY(fractional_max_pool2d, fp32);
  TT_AMP_POLICY(fractional_max_pool3d, fp32);
  TT_AMP_POLICY(geqrf, fp32);
  TT_AMP_POLICY(grid_sampler, fp32);
  TT_AMP_POLICY(grid_sampler_2d, fp32);
  TT_AMP_POLICY(grid_sampler_3d, fp32);
  TT_AMP_POLICY(hinge_embedding_loss, fp32);
  TT_AMP_POLICY(huber_loss, fp32);
  TT_AMP_POLICY(inverse, fp32);
  TT_AMP_POLICY(kl_div, fp32);
  TT_AMP_POLICY(l1_loss, fp32);
  TT_AMP_POLICY(linalg_cholesky, fp32);
  TT_AMP_POLICY(linalg_cholesky_ex, fp32);
  TT_AMP_POLICY(linalg_cond, fp32);
  TT_AMP_POLICY(linalg_cond, p_str, fp32);
  TT_AMP_POLICY(linalg_eig, fp32);
  TT_AMP_POLICY(linalg_eigh, fp32);
  TT_AMP_POLICY(linalg_eigvals, fp32);
  TT_AMP_POLICY(linalg_eigvalsh, fp32);
  TT_AMP_POLICY(linalg_householder_product, fp32);
  TT_AMP_POLICY(linalg_inv, fp32);
  TT_AMP_POLICY(linalg_inv_ex, fp32);
  TT_AMP_POLICY(linalg_lstsq, fp32);
  TT_AMP_POLICY(linalg_matrix_rank, atol_rtol_float, fp32);
  TT_AMP_POLICY(linalg_matrix_rank, atol_rtol_tensor, fp32);
  TT_AMP_POLICY(linalg_matrix_rank, fp32);
  TT_AMP_POLICY(linalg_matrix_rank, tol_tensor, fp32);
  TT_AMP_POLICY(linalg_qr, fp32);
  TT_AMP_POLICY(linalg_solve, fp32);
  TT_AMP_POLICY(linalg_svd, fp32);
  TT_AMP_POLICY(linalg_svdvals, fp32);
  TT_AMP_POLICY(linalg_tensorinv, fp32);
  TT_AMP_POLICY(linalg_tensorsolve, fp32);
  TT_AMP_POLICY(lu_solve, fp32);
  TT_AMP_POLICY(margin_ranking_loss, fp32);
  TT_AMP_POLICY(max_pool3d, fp32);
  TT_AMP_POLICY(max_unpool2d, fp32);
  TT_AMP_POLICY(max_unpool3d, fp32);
  TT_AMP_POLICY(mse_loss, fp32);
  TT_AMP_POLICY(multi_margin_loss, fp32);
  TT_AMP_POLICY(multilabel_margin_loss, fp32);
  TT_AMP_POLICY(multilabel_margin_loss_forward, fp32);
  TT_AMP_POLICY(nanquantile, fp32);
  TT_AMP_POLICY(nanquantile, scalar, fp32);
  TT_AMP_POLICY(nll_loss, fp32);
  TT_AMP_POLICY(nll_loss2d, fp32);
  TT_AMP_POLICY(orgqr, fp32);
  TT_AMP_POLICY(ormqr, fp32);
  TT_AMP_POLICY(pinverse, fp32);
  TT_AMP_POLICY(poisson_nll_loss, fp32);
  TT_AMP_POLICY(polar, fp32);
  // Note: CPU uses `fp32` for prod, but `prod` supports `fp32_set_opt_dtype`,
  // which is preferred.
  TT_AMP_POLICY(prod, dim_int, fp32_set_opt_dtype);
  TT_AMP_POLICY(prod, fp32_set_opt_dtype);
  TT_AMP_POLICY(qr, fp32);
  TT_AMP_POLICY(quantile, fp32);
  TT_AMP_POLICY(quantile, scalar, fp32);
  TT_AMP_POLICY(reflection_pad1d, fp32);
  TT_AMP_POLICY(reflection_pad2d, fp32);
  TT_AMP_POLICY(replication_pad1d, fp32);
  TT_AMP_POLICY(replication_pad2d, fp32);
  TT_AMP_POLICY(replication_pad3d, fp32);
  TT_AMP_POLICY(smooth_l1_loss, fp32);
  TT_AMP_POLICY(soft_margin_loss, fp32);
  TT_AMP_POLICY(stft, center, fp32);
  TT_AMP_POLICY(stft, fp32);
  TT_AMP_POLICY(svd, fp32);
  TT_AMP_POLICY(trace, fp32);
  TT_AMP_POLICY(triangular_solve, fp32);
  TT_AMP_POLICY(triplet_margin_loss, fp32);
  TT_AMP_POLICY(view_as_complex, fp32);
  // go/keep-sorted end
}

#undef TT_AMP_POLICY  // Should only be used inside this file.

}  // namespace torch_tpu
