/*
 * Copyright 2025 Google LLC
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

#include "torch_tpu/eager/tpu_aten_kernels.h"

#include <atomic>
#include <string>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/stack.h"
#include "ATen/native/CPUFallback.h"
#include "ATen/native/DispatchStub.h"
#include "ATen/native/transformers/attention.h"
#include "ATen/ops/empty.h"
#include "torch/library.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/a_min_max/a_min_max_aten_kernels.h"
#include "torch_tpu/ops/addcdiv/addcdiv_aten_kernels.h"
#include "torch_tpu/ops/addcmul/addcmul_aten_kernels.h"
#include "torch_tpu/ops/addmm/addmm_aten_kernels.h"
#include "torch_tpu/ops/addmv/addmv_aten_kernels.h"
#include "torch_tpu/ops/all_any/all_any_aten_kernels.h"
#include "torch_tpu/ops/arange/arange_aten_kernels.h"
#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"
#include "torch_tpu/ops/assertion_aten_kernels.h"
#include "torch_tpu/ops/baddbmm/baddbmm_aten_kernels.h"
#include "torch_tpu/ops/bernoulli/bernoulli_aten_kernels.h"
#include "torch_tpu/ops/binary_aten_kernels.h"  // IWYU pragma: keep for AtenMulTensor, etc
#include "torch_tpu/ops/bincount/bincount_aten_kernels.h"
#include "torch_tpu/ops/bmm/bmm_aten_kernels.h"
#include "torch_tpu/ops/bucketize/bucketize_aten_kernels.h"
#include "torch_tpu/ops/cat/cat_aten_kernels.h"
#include "torch_tpu/ops/clamp/clamp_aten_kernels.h"
#include "torch_tpu/ops/col2im/col2im_aten_kernels.h"
#include "torch_tpu/ops/compile/stateless_rng_kernels.h"
#include "torch_tpu/ops/convolution/convolution_aten_kernels.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/ctc_loss/ctc_loss_aten_kernels.h"
#include "torch_tpu/ops/cummax/cummax_aten_kernels.h"
#include "torch_tpu/ops/cummin/cummin_aten_kernels.h"
#include "torch_tpu/ops/cumprod/cumprod_aten_kernels.h"
#include "torch_tpu/ops/cumsum/cumsum_aten_kernels.h"
#include "torch_tpu/ops/distance/dist_aten_kernels.h"
#include "torch_tpu/ops/dot/dot_aten_kernels.h"
#include "torch_tpu/ops/dot/vdot_aten_kernels.h"
#include "torch_tpu/ops/dropout/dropout_aten_kernels.h"
#include "torch_tpu/ops/dynamic/dynamic_arange/dynamic_arange.h"
#include "torch_tpu/ops/dynamic/set_dimension_logical_size/set_dimension_logical_size.h"
#include "torch_tpu/ops/elu/elu_aten_kernels.h"
#include "torch_tpu/ops/embedding/embedding_aten_kernels.h"
#include "torch_tpu/ops/equal/equal_aten_kernels.h"
#include "torch_tpu/ops/experimental/ragged_dot_aten_kernels.h"
#include "torch_tpu/ops/experimental/send_recv_kernels.h"
#include "torch_tpu/ops/exponential/exponential_aten_kernels.h"
#include "torch_tpu/ops/eye/eye_aten_kernels.h"
#include "torch_tpu/ops/fake_quantize/fake_quantize_aten_kernels.h"
#include "torch_tpu/ops/fft/fft_aten_kernels.h"
#include "torch_tpu/ops/fill/fill_aten_kernels.h"
#include "torch_tpu/ops/flip/flip_aten_kernels.h"
#include "torch_tpu/ops/fmax/fmax_aten_kernels.h"
#include "torch_tpu/ops/fmin/fmin_aten_kernels.h"
#include "torch_tpu/ops/foreach_aten_kernels.h"
#include "torch_tpu/ops/gather/gather_aten_kernels.h"
#include "torch_tpu/ops/gelu/gelu_aten_kernels.h"
#include "torch_tpu/ops/glu/glu_aten_kernels.h"
#include "torch_tpu/ops/grid_sampler/grid_sampler_aten_kernels.h"
#include "torch_tpu/ops/group_norm/group_norm_aten_kernels.h"
#include "torch_tpu/ops/grouped_mm/grouped_mm_aten_kernels.h"
#include "torch_tpu/ops/hardsigmoid/hardsigmoid_aten_kernels.h"
#include "torch_tpu/ops/hardswish/hardswish_aten_kernels.h"
#include "torch_tpu/ops/hardtanh/hardtanh_aten_kernels.h"
#include "torch_tpu/ops/histc/histc_aten_kernels.h"
#include "torch_tpu/ops/im2col/im2col_aten_kernels.h"
#include "torch_tpu/ops/index/index_aten_kernels.h"
#include "torch_tpu/ops/index_add/index_add_aten_kernels.h"
#include "torch_tpu/ops/index_copy/index_copy_aten_kernels.h"
#include "torch_tpu/ops/index_put/index_put_aten_kernels.h"
#include "torch_tpu/ops/index_reduce/index_reduce_aten_kernels.h"
#include "torch_tpu/ops/index_select/index_select_aten_kernels.h"
#include "torch_tpu/ops/is/is_aten_kernels.h"
#include "torch_tpu/ops/isin/isin_aten_kernels.h"
#include "torch_tpu/ops/layer_norm/layer_norm_aten_kernels.h"
#include "torch_tpu/ops/leaky_relu/leaky_relu_aten_kernels.h"
#include "torch_tpu/ops/lerp/lerp_aten_kernels.h"
#include "torch_tpu/ops/linalg/linalg_kernels.h"
#include "torch_tpu/ops/linalg/lu/linalg_lu_kernels.h"
#include "torch_tpu/ops/linalg/qr/linalg_qr_kernels.h"
#include "torch_tpu/ops/linalg/solve_triangular/linalg_solve_triangular_kernels.h"
#include "torch_tpu/ops/linalg/vector_norm/aten_vector_norm_kernels.h"
#include "torch_tpu/ops/linspace/linspace_aten_kernels.h"
#include "torch_tpu/ops/logical/logical_aten_kernels.h"
#include "torch_tpu/ops/masked_fill/masked_fill_aten_kernels.h"  // IWYU pragma: keep for AtenMaskedFill
#include "torch_tpu/ops/masked_scatter/masked_scatter_aten_kernels.h"
#include "torch_tpu/ops/masked_select/masked_select_aten_kernels.h"
#include "torch_tpu/ops/min_max/min_max_aten_kernels.h"
#include "torch_tpu/ops/mm/mm_aten_kernels.h"
#include "torch_tpu/ops/mse_loss/mse_loss_aten_kernels.h"
#include "torch_tpu/ops/multinomial/multinomial_aten_kernels.h"
#include "torch_tpu/ops/native_batch_norm/native_batch_norm_aten_kernels.h"
#include "torch_tpu/ops/nll_loss/nll_loss_aten_kernels.h"
#include "torch_tpu/ops/nonzero/nonzero_aten_kernels.h"
#include "torch_tpu/ops/normal/normal_aten_kernels.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/optimization_barrier/optimization_barrier_kernels.h"
#include "torch_tpu/ops/pooling/adaptive_avg_pool_aten_kernels.h"
#include "torch_tpu/ops/pooling/avg_pool_aten_kernels.h"
#include "torch_tpu/ops/pooling/max_pool_aten_kernels.h"
#include "torch_tpu/ops/prod/prod_aten_kernels.h"
#include "torch_tpu/ops/random/random_aten_kernels.h"
#include "torch_tpu/ops/randperm/randperm_aten_kernels.h"
#include "torch_tpu/ops/reductions/mean_aten_kernels.h"
#include "torch_tpu/ops/reductions/sum_aten_kernels.h"
#include "torch_tpu/ops/reductions/var_aten_kernels.h"
#include "torch_tpu/ops/reflection_pad/reflection_pad_aten_kernels.h"
#include "torch_tpu/ops/repeat_interleave/repeat_interleave_aten_kernels.h"
#include "torch_tpu/ops/replication_pad/replication_pad_aten_kernels.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/ops/rms_norm/rms_norm_aten_kernels.h"
#include "torch_tpu/ops/roll/roll_aten_kernels.h"
#include "torch_tpu/ops/round/round_aten_kernels.h"
#include "torch_tpu/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.h"
#include "torch_tpu/ops/scaled_mm/scaled_mm_aten_kernels.h"
#include "torch_tpu/ops/scatter/scatter_aten_kernels.h"
#include "torch_tpu/ops/set/set_aten_kernels.h"
#include "torch_tpu/ops/sigmoid/sigmoid_aten_kernels.h"
#include "torch_tpu/ops/softmax/softmax_aten_kernels.h"
#include "torch_tpu/ops/softplus/softplus_aten_kernels.h"
#include "torch_tpu/ops/sort/sort_aten_kernels.h"
#include "torch_tpu/ops/split_with_sizes_copy/split_with_sizes_copy_aten_kernels.h"
#include "torch_tpu/ops/take/take_aten_kernels.h"
#include "torch_tpu/ops/tanh/tanh_aten_kernels.h"
#include "torch_tpu/ops/threshold/threshold_aten_kernels.h"
#include "torch_tpu/ops/to_copy/to_copy_aten_kernels.h"
#include "torch_tpu/ops/topk/topk_aten_kernels.h"
#include "torch_tpu/ops/triangular/triangular_aten_kernels.h"
#include "torch_tpu/ops/tril_indices/tril_indices_aten_kernels.h"
#include "torch_tpu/ops/unary_aten_kernels.h"
#include "torch_tpu/ops/unfold/unfold_aten_kernels.h"
#include "torch_tpu/ops/uniform/uniform_aten_kernels.h"
#include "torch_tpu/ops/unique/unique_aten_kernels.h"
#include "torch_tpu/ops/upsample/upsample_aten_kernels.h"
#include "torch_tpu/ops/view/view_aten_kernels.h"
#include "torch_tpu/ops/weight_norm/weight_norm_aten_kernels.h"
#include "torch_tpu/ops/where/where_aten_kernels.h"
#include "torch_tpu/ops/xlogy/xlogy_aten_kernels.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// Registers the kernel function for the given op name in the given library.
template <typename KernelFn>
void Impl(torch::Library& m, const OpName op_name, KernelFn kernel_fn) {
  // ToString() returns a string_view, but impl() requires a const char*.
  // We need to convert the string_view to a std::string so that we can
  // get a NUL-terminated const char* from it. Note that string_view::data()
  // is not guaranteed to be NUL-terminated.
  m.impl(  // M_IMPL_OK=implementing Impl().
      std::string(ToString(op_name)).c_str(), kernel_fn);
}

}  // namespace

// When the dispatch key set is {PrivateUse1} (i.e. for TPU tensors in the
// eager mode), pytorch will try this dispatch table first. If the op is not
// found here, pytorch will then try the (_, PrivateUse1, m) dispatch table
// defined later in this file.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // go/keep-sorted start
  Impl(m, OpName::kAbsOut, AtenAbsOut);
  Impl(m, OpName::kAcosOut, AtenAcosOut);
  Impl(m, OpName::kAcoshOut, AtenAcoshOut);
  Impl(m, OpName::kAdaptiveAvgPool2d, AtenAdaptiveAvgPool2d);
  Impl(m, OpName::kAdaptiveAvgPool2dBackward, AtenAdaptiveAvgPool2dBackward);
  Impl(m, OpName::kAdaptiveAvgPool2dOut, AtenAdaptiveAvgPool2dOut);
  Impl(m, OpName::kAdaptiveAvgPool3d, AtenAdaptiveAvgPool3d);
  Impl(m, OpName::kAdaptiveAvgPool3dBackward, AtenAdaptiveAvgPool3dBackward);
  Impl(m, OpName::kAdaptiveAvgPool3dBackwardGradInput,
       AtenAdaptiveAvgPool3dBackwardGradInput);
  Impl(m, OpName::kAdaptiveAvgPool3dOut, AtenAdaptiveAvgPool3dOut);
  Impl(m, OpName::kAddOut, AtenAddOut);
  Impl(m, OpName::kAddReluOut, AtenAddReluOut);
  Impl(m, OpName::kAddReluScalar, AtenAddReluScalar);
  Impl(m, OpName::kAddReluTensor, AtenAddReluTensor);
  Impl(m, OpName::kAddRelu_Scalar, AtenAddRelu_Scalar);
  Impl(m, OpName::kAddRelu_Tensor, AtenAddRelu_Tensor);
  Impl(m, OpName::kAddcdivOut, AtenAddcdivOut);
  Impl(m, OpName::kAddcmulOut, AtenAddcmulOut);
  Impl(m, OpName::kAddmmDtype, AtenAddmmDtype);
  Impl(m, OpName::kAddmmDtypeOut, AtenAddmmDtypeOut);
  Impl(m, OpName::kAddmmOut, AtenAddmmOut);
  Impl(m, OpName::kAddmvOut, AtenAddmvOut);
  Impl(m, OpName::kAllAllOut, AtenAllAllOut);
  Impl(m, OpName::kAllOut, AtenAllOut);
  Impl(m, OpName::kAmaxOut, AtenAmaxOut);
  Impl(m, OpName::kAminOut, AtenAminOut);
  Impl(m, OpName::kAminmaxOut, AtenAminmaxOut);
  Impl(m, OpName::kAnyAllOut, AtenAnyAllOut);
  Impl(m, OpName::kAnyOut, AtenAnyOut);
  Impl(m, OpName::kArangeStartOut, AtenArangeStartOut);
  Impl(m, OpName::kArgMaxOut, AtenArgmaxOut);
  Impl(m, OpName::kArgMinOut, AtenArgminOut);
  Impl(m, OpName::kAsStrided, AtenAsStrided);
  Impl(m, OpName::kAsinOut, AtenAsinOut);
  Impl(m, OpName::kAsinhOut, AtenAsinhOut);
  Impl(m, OpName::kAssertAsync, AtenAssertAsync);
  Impl(m, OpName::kAssertAsyncMsg, AtenAssertAsyncMsg);
  Impl(m, OpName::kAtan2Out, AtenAtan2Out);
  Impl(m, OpName::kAtanOut, AtenAtanOut);
  Impl(m, OpName::kAtanhOut, AtenAtanhOut);
  Impl(m, OpName::kAvgPool2dBackwardGradInput, AtenAvgPool2dBackwardGradInput);
  Impl(m, OpName::kAvgPool2dOut, AtenAvgPool2dOut);
  Impl(m, OpName::kAvgPool3dBackwardGradInput, AtenAvgPool3dBackwardGradInput);
  Impl(m, OpName::kAvgPool3dOut, AtenAvgPool3dOut);
  Impl(m, OpName::kBaddbmmDtype, AtenBaddbmmDtype);
  Impl(m, OpName::kBaddbmmDtypeOut, AtenBaddbmmDtypeOut);
  Impl(m, OpName::kBaddbmmOut, AtenBaddbmmOut);
  Impl(m, OpName::kBernoulli_Float, AtenBernoulli_Float);
  Impl(m, OpName::kBinCount, AtenBinCount);
  Impl(m, OpName::kBitwiseAndTensorOut, AtenBitwiseAndTensorOut);
  Impl(m, OpName::kBitwiseLeftShiftTensorOut, AtenBitwiseLeftShiftTensorOut);
  Impl(m, OpName::kBitwiseNotOut, AtenBitwiseNotOut);
  Impl(m, OpName::kBitwiseOrTensorOut, AtenBitwiseOrTensorOut);
  Impl(m, OpName::kBitwiseRightShiftTensorOut, AtenBitwiseRightShiftTensorOut);
  Impl(m, OpName::kBitwiseXorTensorOut, AtenBitwiseXorTensorOut);
  Impl(m, OpName::kBmmDtype, AtenBmmDtype);
  Impl(m, OpName::kBmmDtypeOut, AtenBmmDtypeOut);
  Impl(m, OpName::kBmmOut, AtenBmmOut);
  Impl(m, OpName::kBucketizeScalar, AtenBucketizeScalar);
  Impl(m, OpName::kBucketizeTensor, AtenBucketizeTensor);
  Impl(m, OpName::kBucketizeTensorOut, AtenBucketizeTensorOut);
  Impl(m, OpName::kCatOut, AtenCatOut);
  Impl(m, OpName::kCdistForward, AtenCdistForward);
  Impl(m, OpName::kCeilOut, AtenCeilOut);
  Impl(m, OpName::kClampMaxOut, AtenClampMaxOut);
  Impl(m, OpName::kClampMaxTensorOut, AtenClampMaxTensorOut);
  Impl(m, OpName::kClampMinOut, AtenClampMinOut);
  Impl(m, OpName::kClampMinTensorOut, AtenClampMinTensorOut);
  Impl(m, OpName::kClampOut, AtenClampOut);
  Impl(m, OpName::kClampTensorOut, AtenClampTensorOut);
  Impl(m, OpName::kCol2Im, AtenCol2Im);
  Impl(m, OpName::kCol2ImOut, AtenCol2ImOut);
  Impl(m, OpName::kComplexOut, AtenComplexOut);
  Impl(m, OpName::kConjPhysicalOut, AtenConjPhysicalOut);
  Impl(m, OpName::kConvolution, AtenConvolution);
  Impl(m, OpName::kConvolutionBackward, AtenConvolutionBackward);
  Impl(m, OpName::kConvolutionOut, AtenConvolutionOut);
  Impl(m, OpName::kCopyFrom, AtenCopyFrom);
  // per https://github.com/pytorch/xla/issues/2881, this function was added
  // to fix aten:cpufallback for pytorch_xla in 2021.
  // But it isn't registered for CPU and is not called from copy_ in Python.
  // TODO: Revisit if we can replace it with _copy_from. // NOLINT
  Impl(m, OpName::kCopyFromAndResize, AtenCopyFromAndResize);
  Impl(m, OpName::kCopy_, AtenCopy_);
  Impl(m, OpName::kCosOut, AtenCosOut);
  Impl(m, OpName::kCoshOut, AtenCoshOut);
  Impl(m, OpName::kCtcLoss, AtenCtcLoss);
  Impl(m, OpName::kCtcLossBackward, AtenCtcLossBackward);
  Impl(m, OpName::kCtcLossBackwardTensor, AtenCtcLossBackwardTensor);
  Impl(m, OpName::kCtcLossTensor, AtenCtcLossTensor);
  Impl(m, OpName::kCummaxHelper, AtenCummaxHelper);
  Impl(m, OpName::kCumminHelper, AtenCumminHelper);
  Impl(m, OpName::kCumprodOut, AtenCumprodOut);
  Impl(m, OpName::kCumsumOut, AtenCumsumOut);
  Impl(m, OpName::kDivOut, AtenDivOut);
  Impl(m, OpName::kDivOutMode, AtenDivOutMode);
  Impl(m, OpName::kDot, AtenDot);
  Impl(m, OpName::kEfficientZeroTensor, AtenEfficientZeroTensor);
  Impl(m, OpName::kEluBackwardGradInput, AtenEluBackwardGradInput);
  Impl(m, OpName::kEluOut, AtenEluOut);
  Impl(m, OpName::kEmbeddingBag, AtenEmbeddingBag);
  Impl(m, OpName::kEmbeddingBagBackward, AtenEmbeddingBagBackward);
  Impl(m, OpName::kEmbeddingBagForwardOnly, AtenEmbeddingBagForwardOnly);
  Impl(m, OpName::kEmbeddingDenseBackward, AtenEmbeddingDenseBackward);
  Impl(m, OpName::kEmbeddingRenorm_, AtenEmbeddingRenorm_);
  Impl(m, OpName::kEmptyMemoryFormat, AtenEmptyMemoryFormat);
  Impl(m, OpName::kEmptyStrided, AtenEmptyStrided);
  Impl(m, OpName::kEqScalarOut, AtenEqScalarOut);
  Impl(m, OpName::kEqTensorOut, AtenEqTensorOut);
  Impl(m, OpName::kEqual, AtenEqual);
  Impl(m, OpName::kErfInvOut, AtenErfInvOut);
  Impl(m, OpName::kErfOut, AtenErfOut);
  Impl(m, OpName::kExp2Out, AtenExp2Out);
  Impl(m, OpName::kExpM1Out, AtenExpm1Out);
  Impl(m, OpName::kExpOut, AtenExpOut);
  Impl(m, OpName::kExponential_, AtenExponential_);
  Impl(m, OpName::kEyeMOut, AtenEyeMOut);
  Impl(m, OpName::kEyeOut, AtenEyeOut);
  Impl(m, OpName::kFakeQuantizePerTensorAffineCachemask,
       FakeQuantizePerTensorAffineCachemask);
  Impl(m, OpName::kFftR2c, AtenFftR2c);
  Impl(m, OpName::kFftR2cOut, AtenFftR2cOut);
  Impl(m, OpName::kFill_Scalar, AtenFillScalar_);
  Impl(m, OpName::kFill_Tensor, AtenFillTensor_);
  Impl(m, OpName::kFlip, AtenFlip);
  Impl(m, OpName::kFloorDivide, AtenFloorDivide);
  Impl(m, OpName::kFloorDivideOut, AtenFloorDivideOut);
  Impl(m, OpName::kFloorDivide_Tensor, AtenFloorDivide_Tensor);
  Impl(m, OpName::kFloorOut, AtenFloorOut);
  Impl(m, OpName::kFmaxOut, AtenFmaxOut);
  Impl(m, OpName::kFminOut, AtenFminOut);
  Impl(m, OpName::kFmodTensorOut, AtenFmodTensorOut);
  Impl(m, OpName::kForeachAbs, AtenForeachAbs);
  Impl(m, OpName::kForeachAbs_, AtenForeachAbs_);
  Impl(m, OpName::kForeachAcos, AtenForeachAcos);
  Impl(m, OpName::kForeachAcos_, AtenForeachAcos_);
  Impl(m, OpName::kForeachAddList, AtenForeachAddList);
  Impl(m, OpName::kForeachAddScalar, AtenForeachAddScalar);
  Impl(m, OpName::kForeachAddScalarList, AtenForeachAddScalarList);
  Impl(m, OpName::kForeachAddTensor, AtenForeachAddTensor);
  Impl(m, OpName::kForeachAdd_List, AtenForeachAdd_List);
  Impl(m, OpName::kForeachAdd_Scalar, AtenForeachAdd_Scalar);
  Impl(m, OpName::kForeachAdd_ScalarList, AtenForeachAdd_ScalarList);
  Impl(m, OpName::kForeachAdd_Tensor, AtenForeachAdd_Tensor);
  Impl(m, OpName::kForeachAddcdivScalar, AtenForeachAddcdivScalar);
  Impl(m, OpName::kForeachAddcdivScalarList, AtenForeachAddcdivScalarList);
  Impl(m, OpName::kForeachAddcdivTensor, AtenForeachAddcdivTensor);
  Impl(m, OpName::kForeachAddcdiv_Scalar, AtenForeachAddcdiv_Scalar);
  Impl(m, OpName::kForeachAddcdiv_ScalarList, AtenForeachAddcdiv_ScalarList);
  Impl(m, OpName::kForeachAddcdiv_Tensor, AtenForeachAddcdiv_Tensor);
  Impl(m, OpName::kForeachAddcmulScalar, AtenForeachAddcmulScalar);
  Impl(m, OpName::kForeachAddcmulScalarList, AtenForeachAddcmulScalarList);
  Impl(m, OpName::kForeachAddcmulTensor, AtenForeachAddcmulTensor);
  Impl(m, OpName::kForeachAddcmul_Scalar, AtenForeachAddcmul_Scalar);
  Impl(m, OpName::kForeachAddcmul_ScalarList, AtenForeachAddcmul_ScalarList);
  Impl(m, OpName::kForeachAddcmul_Tensor, AtenForeachAddcmul_Tensor);
  Impl(m, OpName::kForeachAsin, AtenForeachAsin);
  Impl(m, OpName::kForeachAsin_, AtenForeachAsin_);
  Impl(m, OpName::kForeachAtan, AtenForeachAtan);
  Impl(m, OpName::kForeachAtan_, AtenForeachAtan_);
  Impl(m, OpName::kForeachCeil, AtenForeachCeil);
  Impl(m, OpName::kForeachCeil_, AtenForeachCeil_);
  Impl(m, OpName::kForeachClampMaxList, AtenForeachClampMaxList);
  Impl(m, OpName::kForeachClampMaxScalar, AtenForeachClampMaxScalar);
  Impl(m, OpName::kForeachClampMaxScalarList, AtenForeachClampMaxScalarList);
  Impl(m, OpName::kForeachClampMax_List, AtenForeachClampMax_List);
  Impl(m, OpName::kForeachClampMax_Scalar, AtenForeachClampMax_Scalar);
  Impl(m, OpName::kForeachClampMax_ScalarList, AtenForeachClampMax_ScalarList);
  Impl(m, OpName::kForeachClampMinList, AtenForeachClampMinList);
  Impl(m, OpName::kForeachClampMinScalar, AtenForeachClampMinScalar);
  Impl(m, OpName::kForeachClampMinScalarList, AtenForeachClampMinScalarList);
  Impl(m, OpName::kForeachClampMin_List, AtenForeachClampMin_List);
  Impl(m, OpName::kForeachClampMin_Scalar, AtenForeachClampMin_Scalar);
  Impl(m, OpName::kForeachClampMin_ScalarList, AtenForeachClampMin_ScalarList);
  Impl(m, OpName::kForeachCopy_, AtenForeachCopy_);
  Impl(m, OpName::kForeachCos, AtenForeachCos);
  Impl(m, OpName::kForeachCos_, AtenForeachCos_);
  Impl(m, OpName::kForeachCosh, AtenForeachCosh);
  Impl(m, OpName::kForeachCosh_, AtenForeachCosh_);
  Impl(m, OpName::kForeachDivList, AtenForeachDivList);
  Impl(m, OpName::kForeachDivScalar, AtenForeachDivScalar);
  Impl(m, OpName::kForeachDivScalarList, AtenForeachDivScalarList);
  Impl(m, OpName::kForeachDivTensor, AtenForeachDivTensor);
  Impl(m, OpName::kForeachDiv_List, AtenForeachDiv_List);
  Impl(m, OpName::kForeachDiv_Scalar, AtenForeachDiv_Scalar);
  Impl(m, OpName::kForeachDiv_ScalarList, AtenForeachDiv_ScalarList);
  Impl(m, OpName::kForeachDiv_Tensor, AtenForeachDiv_Tensor);
  Impl(m, OpName::kForeachErf, AtenForeachErf);
  Impl(m, OpName::kForeachErf_, AtenForeachErf_);
  Impl(m, OpName::kForeachErfc, AtenForeachErfc);
  Impl(m, OpName::kForeachErfc_, AtenForeachErfc_);
  Impl(m, OpName::kForeachExp, AtenForeachExp);
  Impl(m, OpName::kForeachExp_, AtenForeachExp_);
  Impl(m, OpName::kForeachExpm1, AtenForeachExpm1);
  Impl(m, OpName::kForeachExpm1_, AtenForeachExpm1_);
  Impl(m, OpName::kForeachFloor, AtenForeachFloor);
  Impl(m, OpName::kForeachFloor_, AtenForeachFloor_);
  Impl(m, OpName::kForeachFrac, AtenForeachFrac);
  Impl(m, OpName::kForeachFrac_, AtenForeachFrac_);
  Impl(m, OpName::kForeachLerpList, AtenForeachLerpList);
  Impl(m, OpName::kForeachLerpScalar, AtenForeachLerpScalar);
  Impl(m, OpName::kForeachLerpScalarList, AtenForeachLerpScalarList);
  Impl(m, OpName::kForeachLerp_List, AtenForeachLerp_List);
  Impl(m, OpName::kForeachLerp_Scalar, AtenForeachLerp_Scalar);
  Impl(m, OpName::kForeachLerp_ScalarList, AtenForeachLerp_ScalarList);
  Impl(m, OpName::kForeachLgamma, AtenForeachLgamma);
  Impl(m, OpName::kForeachLgamma_, AtenForeachLgamma_);
  Impl(m, OpName::kForeachLog, AtenForeachLog);
  Impl(m, OpName::kForeachLog10, AtenForeachLog10);
  Impl(m, OpName::kForeachLog10_, AtenForeachLog10_);
  Impl(m, OpName::kForeachLog1p, AtenForeachLog1p);
  Impl(m, OpName::kForeachLog1p_, AtenForeachLog1p_);
  Impl(m, OpName::kForeachLog2, AtenForeachLog2);
  Impl(m, OpName::kForeachLog2_, AtenForeachLog2_);
  Impl(m, OpName::kForeachLog_, AtenForeachLog_);
  Impl(m, OpName::kForeachMax, AtenForeachMax);
  Impl(m, OpName::kForeachMaximumList, AtenForeachMaximumList);
  Impl(m, OpName::kForeachMaximumScalar, AtenForeachMaximumScalar);
  Impl(m, OpName::kForeachMaximumScalarList, AtenForeachMaximumScalarList);
  Impl(m, OpName::kForeachMaximum_List, AtenForeachMaximum_List);
  Impl(m, OpName::kForeachMaximum_Scalar, AtenForeachMaximum_Scalar);
  Impl(m, OpName::kForeachMaximum_ScalarList, AtenForeachMaximum_ScalarList);
  Impl(m, OpName::kForeachMinimumList, AtenForeachMinimumList);
  Impl(m, OpName::kForeachMinimumScalar, AtenForeachMinimumScalar);
  Impl(m, OpName::kForeachMinimumScalarList, AtenForeachMinimumScalarList);
  Impl(m, OpName::kForeachMinimum_List, AtenForeachMinimum_List);
  Impl(m, OpName::kForeachMinimum_Scalar, AtenForeachMinimum_Scalar);
  Impl(m, OpName::kForeachMinimum_ScalarList, AtenForeachMinimum_ScalarList);
  Impl(m, OpName::kForeachMulList, AtenForeachMulList);
  Impl(m, OpName::kForeachMulScalar, AtenForeachMulScalar);
  Impl(m, OpName::kForeachMulScalarList, AtenForeachMulScalarList);
  Impl(m, OpName::kForeachMulTensor, AtenForeachMulTensor);
  Impl(m, OpName::kForeachMul_List, AtenForeachMul_List);
  Impl(m, OpName::kForeachMul_Scalar, AtenForeachMul_Scalar);
  Impl(m, OpName::kForeachMul_ScalarList, AtenForeachMul_ScalarList);
  Impl(m, OpName::kForeachMul_Tensor, AtenForeachMul_Tensor);
  Impl(m, OpName::kForeachNeg, AtenForeachNeg);
  Impl(m, OpName::kForeachNeg_, AtenForeachNeg_);
  Impl(m, OpName::kForeachNormScalar, AtenForeachNormScalar);
  Impl(m, OpName::kForeachPowList, AtenForeachPowList);
  Impl(m, OpName::kForeachPowScalar, AtenForeachPowScalar);
  Impl(m, OpName::kForeachPowScalarAndTensor, AtenForeachPowScalarAndTensor);
  Impl(m, OpName::kForeachPowScalarList, AtenForeachPowScalarList);
  Impl(m, OpName::kForeachPow_List, AtenForeachPow_List);
  Impl(m, OpName::kForeachPow_Scalar, AtenForeachPow_Scalar);
  Impl(m, OpName::kForeachPow_ScalarList, AtenForeachPow_ScalarList);
  Impl(m, OpName::kForeachReciprocal, AtenForeachReciprocal);
  Impl(m, OpName::kForeachReciprocal_, AtenForeachReciprocal_);
  Impl(m, OpName::kForeachRound, AtenForeachRound);
  Impl(m, OpName::kForeachRound_, AtenForeachRound_);
  Impl(m, OpName::kForeachRsqrt, AtenForeachRsqrt);
  Impl(m, OpName::kForeachRsqrt_, AtenForeachRsqrt_);
  Impl(m, OpName::kForeachSigmoid, AtenForeachSigmoid);
  Impl(m, OpName::kForeachSigmoid_, AtenForeachSigmoid_);
  Impl(m, OpName::kForeachSign, AtenForeachSign);
  Impl(m, OpName::kForeachSign_, AtenForeachSign_);
  Impl(m, OpName::kForeachSin, AtenForeachSin);
  Impl(m, OpName::kForeachSin_, AtenForeachSin_);
  Impl(m, OpName::kForeachSinh, AtenForeachSinh);
  Impl(m, OpName::kForeachSinh_, AtenForeachSinh_);
  Impl(m, OpName::kForeachSqrt, AtenForeachSqrt);
  Impl(m, OpName::kForeachSqrt_, AtenForeachSqrt_);
  Impl(m, OpName::kForeachSubList, AtenForeachSubList);
  Impl(m, OpName::kForeachSubScalar, AtenForeachSubScalar);
  Impl(m, OpName::kForeachSubScalarList, AtenForeachSubScalarList);
  Impl(m, OpName::kForeachSub_List, AtenForeachSub_List);
  Impl(m, OpName::kForeachSub_Scalar, AtenForeachSub_Scalar);
  Impl(m, OpName::kForeachSub_ScalarList, AtenForeachSub_ScalarList);
  Impl(m, OpName::kForeachTan, AtenForeachTan);
  Impl(m, OpName::kForeachTan_, AtenForeachTan_);
  Impl(m, OpName::kForeachTanh, AtenForeachTanh);
  Impl(m, OpName::kForeachTanh_, AtenForeachTanh_);
  Impl(m, OpName::kForeachTrunc, AtenForeachTrunc);
  Impl(m, OpName::kForeachTrunc_, AtenForeachTrunc_);
  Impl(m, OpName::kForeachZero_, AtenForeachZero_);
  Impl(m, OpName::kFusedRmsNorm, AtenFusedRmsNorm);
  Impl(m, OpName::kFusedRmsNormBackward, AtenFusedRmsNormBackward);
  Impl(m, OpName::kGather, AtenGather);
  Impl(m, OpName::kGatherOut, AtenGatherOut);
  Impl(m, OpName::kGeScalarOut, AtenGeScalarOut);
  Impl(m, OpName::kGeTensorOut, AtenGeTensorOut);
  Impl(m, OpName::kGelu, AtenGelu);
  Impl(m, OpName::kGeluBackwardGradInput, AtenGeluBackwardGradInput);
  Impl(m, OpName::kGeluOut, AtenGeluOut);
  Impl(m, OpName::kGeqrf, AtenGeqrf);
  Impl(m, OpName::kGeqrfA, AtenGeqrfA);
  Impl(m, OpName::kGluBackward, AtenGluBackward);
  Impl(m, OpName::kGluBackwardGradInput, AtenGluBackwardGradInput);
  Impl(m, OpName::kGluOut, AtenGluOut);
  Impl(m, OpName::kGridSampler2d, AtenGridSampler2d);
  Impl(m, OpName::kGridSampler2dBackward, AtenGridSampler2dBackward);
  Impl(m, OpName::kGridSampler3d, AtenGridSampler3d);
  Impl(m, OpName::kGridSampler3dBackward, AtenGridSampler3dBackward);
  Impl(m, OpName::kGroupedMm, AtenGroupedMm);
  Impl(m, OpName::kGtScalarOut, AtenGtScalarOut);
  Impl(m, OpName::kGtTensorOut, AtenGtTensorOut);
  Impl(m, OpName::kHardsigmoidBackwardGradInput,
       AtenHardsigmoidBackwardGradInput);
  Impl(m, OpName::kHardsigmoidOut, AtenHardsigmoidOut);
  Impl(m, OpName::kHardswish, AtenHardswish);
  Impl(m, OpName::kHardswishBackward, AtenHardswishBackward);
  Impl(m, OpName::kHardswishOut, AtenHardswishOut);
  Impl(m, OpName::kHardswish_, AtenHardswish_);
  Impl(m, OpName::kHardtanh, AtenHardtanh);
  Impl(m, OpName::kHardtanhOut, AtenHardtanhOut);
  Impl(m, OpName::kHardtanh_, AtenHardtanh_);
  Impl(m, OpName::kHistc, AtenHistc);
  Impl(m, OpName::kHistcOut, AtenHistcOut);
  Impl(m, OpName::kIlshiftScalar, AtenIlshiftScalar);
  Impl(m, OpName::kIlshiftTensor, AtenIlshiftTensor);
  Impl(m, OpName::kIm2Col, AtenIm2Col);
  Impl(m, OpName::kIm2ColOut, AtenIm2ColOut);
  Impl(m, OpName::kIndexAddOut, TpuAtenIndexAddOut);
  Impl(m, OpName::kIndexCopyOut, AtenIndexCopyOut);
  Impl(m, OpName::kIndexPutImpl_, TpuAtenIndexPutImpl_);
  Impl(m, OpName::kIndexReduceOut, TpuAtenIndexReduceOut);
  Impl(m, OpName::kIndexSelect, TpuAtenIndexSelect);
  Impl(m, OpName::kIndexTensorOut, AtenIndexTensorOut);
  Impl(m, OpName::kIrshiftScalar, AtenIrshiftScalar);
  Impl(m, OpName::kIrshiftTensor, AtenIrshiftTensor);
  Impl(m, OpName::kIsInScalarTensorOut, AtenIsInScalarTensorOut);
  Impl(m, OpName::kIsInTensorScalarOut, AtenIsInTensorScalarOut);
  Impl(m, OpName::kIsInTensorTensorOut, AtenIsInTensorTensorOut);
  Impl(m, OpName::kIsNan, AtenIsNan);
  Impl(m, OpName::kIsNegInfOut, AtenIsNegInfOut);
  Impl(m, OpName::kIsPosInfOut, AtenIsPosInfOut);
  Impl(m, OpName::kLeScalarOut, AtenLeScalarOut);
  Impl(m, OpName::kLeTensorOut, AtenLeTensorOut);
  Impl(m, OpName::kLeakyReluBackward, AtenLeakyReluBackwardGradInput);
  Impl(m, OpName::kLeakyReluOut, AtenLeakyReluOut);
  Impl(m, OpName::kLerpScalarOut, AtenLerpScalarOut);
  Impl(m, OpName::kLerpTensorOut, AtenLerpTensorOut);
  Impl(m, OpName::kLgammaOut, AtenLgammaOut);
  Impl(m, OpName::kLinalgInvExInverse, AtenLinalgInvExInverse);
  Impl(m, OpName::kLinalgLuFactorExOut, AtenLinalgLuFactorExOut);
  Impl(m, OpName::kLinalgLuOut, AtenLinalgLuOut);
  Impl(m, OpName::kLinalgLuSolveOut, AtenLinalgLuSolveOut);
  Impl(m, OpName::kLinalgQrOut, AtenLinalgQrOut);
  Impl(m, OpName::kLinalgSolveExOut, AtenLinalgSolveExOut);
  Impl(m, OpName::kLinalgSolveTriangular, AtenLinalgSolveTriangular);
  Impl(m, OpName::kLinalgSolveTriangularOut, AtenLinalgSolveTriangularOut);
  Impl(m, OpName::kLinalgVectorNormOut, AtenLinalgVectorNormOut);
  Impl(m, OpName::kLinspaceOut, AtenLinspaceOut);
  Impl(m, OpName::kLocalScalarDense, AtenLocalScalarDense);
  Impl(m, OpName::kLog10Out, AtenLog10Out);
  Impl(m, OpName::kLog1pOut, AtenLog1pOut);
  Impl(m, OpName::kLog2Out, AtenLog2Out);
  Impl(m, OpName::kLogOut, AtenLogOut);
  Impl(m, OpName::kLogSoftmaxBackwardDataOut, AtenLogSoftmaxBackwardDataOut);
  Impl(m, OpName::kLogSoftmaxOut, AtenLogSoftmaxOut);
  Impl(m, OpName::kLogicalAndOut, AtenLogicalAndOut);
  Impl(m, OpName::kLogicalNotOut, AtenLogicalNotOut);
  Impl(m, OpName::kLogicalOrOut, AtenLogicalOrOut);
  Impl(m, OpName::kLogicalXorOut, AtenLogicalXorOut);
  Impl(m, OpName::kLshiftScalar, AtenLshiftScalar);
  Impl(m, OpName::kLshiftTensor, AtenLshiftTensor);
  Impl(m, OpName::kLtScalarOut, AtenLtScalarOut);
  Impl(m, OpName::kLtTensorOut, AtenLtTensorOut);
  Impl(m, OpName::kLuUnpackOut, AtenLuUnpackOut);
  Impl(m, OpName::kMaskedFill_Scalar, AtenMaskedFill_Scalar);
  Impl(m, OpName::kMaskedFill_Tensor, AtenMaskedFill_Tensor);
  Impl(m, OpName::kMaskedScatter_, AtenMaskedScatter_);
  Impl(m, OpName::kMaskedSelect, AtenMaskedSelect);
  Impl(m, OpName::kMaskedSelectOut, AtenMaskedSelectOut);
  Impl(m, OpName::kMax, AtenMax);
  Impl(m, OpName::kMaxDimMax, AtenMaxDimMax);
  Impl(m, OpName::kMaxPool2dWithIndicesBackwardGradInput,
       AtenMaxPool2dWithIndicesBackwardGradInput);
  Impl(m, OpName::kMaxPool2dWithIndicesOut, AtenMaxPool2dWithIndicesOut);
  Impl(m, OpName::kMaxPool3dWithIndices, AtenMaxPool3dWithIndices);
  Impl(m, OpName::kMaxPool3dWithIndicesBackward,
       AtenMaxPool3dWithIndicesBackward);
  Impl(m, OpName::kMaxPool3dWithIndicesBackwardGradInput,
       AtenMaxPool3dWithIndicesBackwardGradInput);
  Impl(m, OpName::kMaxPool3dWithIndicesOut, AtenMaxPool3dWithIndicesOut);
  Impl(m, OpName::kMaxUnaryOut, AtenMaxUnaryOut);
  Impl(m, OpName::kMaximumOut, AtenMaximumOut);
  Impl(m, OpName::kMeanOut, AtenMeanOut);
  Impl(m, OpName::kMin, AtenMin);
  Impl(m, OpName::kMinDimMin, AtenMinDimMin);
  Impl(m, OpName::kMinUnaryOut, AtenMinUnaryOut);
  Impl(m, OpName::kMinimumOut, AtenMinimumOut);
  Impl(m, OpName::kMmDtype, AtenMmDtype);
  Impl(m, OpName::kMmDtypeOut, AtenMmDtypeOut);
  Impl(m, OpName::kMmOut, AtenMmOut);
  Impl(m, OpName::kMseLossBackward, AtenMseLossBackward);
  Impl(m, OpName::kMseLossOut, AtenMseLossOut);
  Impl(m, OpName::kMulOut, AtenMulOut);
  Impl(m, OpName::kMultinomial, AtenMultinomial);
  Impl(m, OpName::kMultinomialOut, AtenMultinomialOut);
  Impl(m, OpName::kNativeBatchNorm, AtenNativeBatchNorm);
  Impl(m, OpName::kNativeBatchNormBackward, AtenNativeBatchNormBackward);
  Impl(m, OpName::kNativeBatchNormLegit, AtenNativeBatchNormLegit);
  Impl(m, OpName::kNativeBatchNormLegitNoStats,
       AtenNativeBatchNormLegitNoStats);
  Impl(m, OpName::kNativeBatchNormLegitNoStatsOut,
       AtenNativeBatchNormLegitNoStatsOut);
  Impl(m, OpName::kNativeBatchNormLegitOut, AtenNativeBatchNormLegitOut);
  Impl(m, OpName::kNativeBatchNormOut, AtenNativeBatchNormOut);
  Impl(m, OpName::kNativeDropout, AtenDropout);
  Impl(m, OpName::kNativeDropoutBackward, AtenNativeDropoutBackward);
  Impl(m, OpName::kNativeGroupNormBackward, AtenNativeGroupNormBackward);
  Impl(m, OpName::kNativeLayerNorm, AtenNativeLayerNorm);
  Impl(m, OpName::kNativeLayerNormBackward, AtenLayerNormBackward);
  Impl(m, OpName::kNeScalarOut, AtenNeScalarOut);
  Impl(m, OpName::kNeTensorOut, AtenNeTensorOut);
  Impl(m, OpName::kNegOut, AtenNegOut);
  Impl(m, OpName::kNllLoss2dBackward, AtenNllLoss2dBackward);
  Impl(m, OpName::kNllLoss2dBackwardGradInput, AtenNllLoss2dBackwardGradInput);
  Impl(m, OpName::kNllLoss2dForward, AtenNllLoss2dForward);
  Impl(m, OpName::kNllLoss2dForwardOut, AtenNllLoss2dForwardOut);
  Impl(m, OpName::kNllLossBackwardGradInput, AtenNllLossBackwardGradInput);
  Impl(m, OpName::kNllLossForwardOut, AtenNllLossForwardOut);
  Impl(m, OpName::kNonzero, AtenNonzero);
  Impl(m, OpName::kNonzeroOut, AtenNonzeroOut);
  Impl(m, OpName::kNormalFloatTensor, AtenNormalFloatTensor);
  Impl(m, OpName::kNormalFloatTensorOut, AtenNormalFloatTensorOut);
  Impl(m, OpName::kNormalTensorFloat, AtenNormalTensorFloat);
  Impl(m, OpName::kNormalTensorFloatOut, AtenNormalTensorFloatOut);
  Impl(m, OpName::kNormalTensorTensor, AtenNormalTensorTensor);
  Impl(m, OpName::kNormalTensorTensorOut, AtenNormalTensorTensorOut);
  Impl(m, OpName::kNormal_, AtenNormal_);
  Impl(m, OpName::kPdistForward, AtenPdistForward);
  Impl(m, OpName::kPolarOut, AtenPolarOut);
  Impl(m, OpName::kPowScalarOut, AtenPowScalarOut);
  Impl(m, OpName::kPowTensorScalarOut, AtenPowTensorScalarOut);
  Impl(m, OpName::kPowTensorTensorOut, AtenPowTensorTensorOut);
  Impl(m, OpName::kProd, AtenProd);
  Impl(m, OpName::kProdDimOut, AtenProdDimOut);
  Impl(m, OpName::kRandom_, AtenRandom_);
  Impl(m, OpName::kRandom_From, AtenRandom_From);
  Impl(m, OpName::kRandom_To, AtenRandom_To);
  Impl(m, OpName::kRandpermGeneratorOut, AtenRandpermGeneratorOut);
  Impl(m, OpName::kReciprocalOut, AtenReciprocalOut);
  Impl(m, OpName::kReflectionPad1dBackwardGradInput,
       AtenReflectionPad1dBackwardGradInput);
  Impl(m, OpName::kReflectionPad1dOut, AtenReflectionPad1dOut);
  Impl(m, OpName::kReflectionPad2d, AtenReflectionPad2d);
  Impl(m, OpName::kReflectionPad2dBackward, AtenReflectionPad2dBackward);
  Impl(m, OpName::kReflectionPad2dBackwardGradInput,
       AtenReflectionPad2dBackwardGradInput);
  Impl(m, OpName::kReflectionPad2dOut, AtenReflectionPad2dOut);
  Impl(m, OpName::kReflectionPad3dBackwardGradInput,
       AtenReflectionPad3dBackwardGradInput);
  Impl(m, OpName::kReflectionPad3dOut, AtenReflectionPad3dOut);
  Impl(m, OpName::kRelu, AtenRelu);
  Impl(m, OpName::kRelu_, AtenRelu_);
  Impl(m, OpName::kRemainderScalarTensor, AtenRemainderScalarTensor);
  Impl(m, OpName::kRemainderTensorOut, AtenRemainderTensorOut);
  Impl(m, OpName::kRepeatInterleaveSelfTensor, AtenRepeatInterleave);
  Impl(m, OpName::kReplicationPad1dBackwardGradInput,
       AtenReplicationPad1dBackwardGradInput);
  Impl(m, OpName::kReplicationPad1dOut, AtenReplicationPad1dOut);
  Impl(m, OpName::kReplicationPad2dBackward, AtenReplicationPad2dBackward);
  Impl(m, OpName::kReplicationPad2dBackwardGradInput,
       AtenReplicationPad2dBackwardGradInput);
  Impl(m, OpName::kReplicationPad2dOut, AtenReplicationPad2dOut);
  Impl(m, OpName::kReplicationPad3dBackward, AtenReplicationPad3dBackward);
  Impl(m, OpName::kReplicationPad3dBackwardGradInput,
       AtenReplicationPad3dBackwardGradInput);
  Impl(m, OpName::kReplicationPad3dOut, AtenReplicationPad3dOut);
  Impl(m, OpName::kReshapeAlias, AtenReshapeAlias);
  Impl(m, OpName::kResize_, AtenResize_);
  Impl(m, OpName::kRoll, AtenRoll);
  Impl(m, OpName::kRoundDecimalsOut, AtenRoundDecimalsOut);
  Impl(m, OpName::kRoundOut, AtenRoundOut);
  Impl(m, OpName::kRshiftScalar, AtenRshiftScalar);
  Impl(m, OpName::kRshiftTensor, AtenRshiftTensor);
  Impl(m, OpName::kRsqrtOut, AtenRsqrtOut);
  Impl(m, OpName::kRsubTensor, AtenRsubTensor);
  Impl(m, OpName::kScaledDotProductEfficientAttention,
       AtenScaledDotProductEfficientAttention);
  Impl(m, OpName::kScaledDotProductFlashAttention,
       AtenScaledDotProductFlashAttention);
  Impl(m, OpName::kScaledDotProductFlashAttentionBackward,
       AtenScaledDotProductFlashAttentionBackward);
  Impl(m, OpName::kScaledDotProductFusedAttentionOverrideable,
       AtenScaledDotProductFusedAttentionOverrideable);
  Impl(m, OpName::kScaledDotProductFusedAttentionOverrideableBackward,
       AtenScaledDotProductFusedAttentionOverrideableBackward);
  Impl(m, OpName::kScaledMm, AtenScaledMm);
  Impl(m, OpName::kScaledMmOut, AtenScaledMmOut);
  Impl(m, OpName::kScatterAddOut, AtenScatterAddOut);
  Impl(m, OpName::kScatterReduceOut, AtenScatterReduceOut);
  Impl(m, OpName::kScatterReduceTwoOut, AtenScatterReduceTwoOut);
  Impl(m, OpName::kScatterSrcOut, AtenScatterSrcOut);
  Impl(m, OpName::kScatterValueOut, AtenScatterValueOut);
  Impl(m, OpName::kScatterValueReduceOut, AtenScatterValueReduceOut);
  Impl(m, OpName::kSet_, AtenSet_);
  Impl(m, OpName::kSet_SourceStorage, AtenSet_SourceStorage);
  Impl(m, OpName::kSet_SourceStorageOffset, AtenSet_SourceStorageOffset);
  Impl(m, OpName::kSet_SourceTensor, AtenSet_SourceTensor);
  Impl(m, OpName::kSgnOut, AtenSgnOut);
  Impl(m, OpName::kSigmoidBackwardGradInput, AtenSigmoidBackwardGradInput);
  Impl(m, OpName::kSigmoidOut, AtenSigmoidOut);
  Impl(m, OpName::kSignOut, AtenSignOut);
  Impl(m, OpName::kSignbitOut, AtenSignbitOut);
  Impl(m, OpName::kSiluOut, AtenSiluOut);
  Impl(m, OpName::kSinOut, AtenSinOut);
  Impl(m, OpName::kSinhOut, AtenSinhOut);
  Impl(m, OpName::kSoftmaxBackwardDataOut, AtenSoftmaxBackwardDataOut);
  Impl(m, OpName::kSoftmaxOut, AtenSoftmaxOut);
  Impl(m, OpName::kSoftplusBackwardGradInput, AtenSoftplusBackwardGradInput);
  Impl(m, OpName::kSoftplusOut, AtenSoftplusOut);
  Impl(m, OpName::kSortValuesStable, AtenSortValuesStable);
  Impl(m, OpName::kSplitWithSizesCopyOut, AtenSplitWithSizesCopyOut);
  Impl(m, OpName::kSqrtOut, AtenSqrtOut);
  Impl(m, OpName::kSubOut, AtenSubOut);
  Impl(m, OpName::kSumIntListOut, AtenSumIntListOut);
  Impl(m, OpName::kTake, AtenTake);
  Impl(m, OpName::kTakeOut, AtenTakeOut);
  Impl(m, OpName::kTanOut, AtenTanOut);
  Impl(m, OpName::kTanhBackwardGradInput, AtenTanhBackwardGradInput);
  Impl(m, OpName::kTanhOut, AtenTanhOut);
  Impl(m, OpName::kThresholdBackwardGradInput, AtenThresholdBackwardGradInput);
  Impl(m, OpName::kThresholdOut, AtenThresholdOut);
  Impl(m, OpName::kToCopy, AtenToCopy);
  Impl(m, OpName::kTopkValues, AtenTopKValues);
  Impl(m, OpName::kTrilIndices, AtenTrilIndices);
  Impl(m, OpName::kTrilOut, AtenTrilOut);
  Impl(m, OpName::kTriuOut, AtenTriuOut);
  Impl(m, OpName::kTruncOut, AtenTruncOut);
  Impl(m, OpName::kUnfold, AtenUnfold);
  Impl(m, OpName::kUnfoldBackward, AtenUnfoldBackward);
  Impl(m, OpName::kUniform_, AtenUniform_);
  Impl(m, OpName::kUnique2, AtenUnique2);
  Impl(m, OpName::kUpsampleBilinear2dBackwardGradInput,
       AtenUpsampleBilinear2dBackwardGradInput);
  Impl(m, OpName::kUpsampleBilinear2dOut, AtenUpsampleBilinear2dOut);
  Impl(m, OpName::kUpsampleNearest1dBackwardGradInput,
       AtenUpsampleNearest1dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearest1dOut, AtenUpsampleNearest1dOut);
  Impl(m, OpName::kUpsampleNearest2dBackwardGradInput,
       AtenUpsampleNearest2dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearest2dOut, AtenUpsampleNearest2dOut);
  Impl(m, OpName::kUpsampleNearest3dBackwardGradInput,
       AtenUpsampleNearest3dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearest3dOut, AtenUpsampleNearest3dOut);
  Impl(m, OpName::kUpsampleNearestExact1dBackwardGradInput,
       AtenUpsampleNearestExact1dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearestExact1dOut, AtenUpsampleNearestExact1dOut);
  Impl(m, OpName::kUpsampleNearestExact2dBackwardGradInput,
       AtenUpsampleNearestExact2dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearestExact2dOut, AtenUpsampleNearestExact2dOut);
  Impl(m, OpName::kUpsampleNearestExact3dBackwardGradInput,
       AtenUpsampleNearestExact3dBackwardGradInput);
  Impl(m, OpName::kUpsampleNearestExact3dOut, AtenUpsampleNearestExact3dOut);
  Impl(m, OpName::kVarCorrection, AtenVar);
  Impl(m, OpName::kVarCorrectionOut, AtenVarOut);
  Impl(m, OpName::kVdot, AtenVdot);
  Impl(m, OpName::kView, AtenView);
  Impl(m, OpName::kViewAsComplex, AtenViewAsComplex);
  Impl(m, OpName::kViewAsReal, AtenViewAsReal);
  Impl(m, OpName::kWeightNormInterface, AtenWeightNormInterface);
  Impl(m, OpName::kWhereSelf, AtenWhereSelf);
  Impl(m, OpName::kWhereSelfOut, AtenWhereSelfOut);
  Impl(m, OpName::kXlogyOutTensor, AtenXlogyOutTensor);
  Impl(m, OpName::kZero_, AtenZero_);
  // go/keep-sorted end
}

// Register the cpu_fallback for missing operators. This is mostly for testing
// and debugging. This way, as least we have models running, without being
// blocked by certain missing operators. By default, the fallback is disabled.
//
// To change fallback mode the intended way, use python context manager:
// `with fallback_mode(FallbackMode.ALLOW_FALLBACK):`
//
void TpuMissingOpFallback(const c10::OperatorHandle& op,
                          torch::jit::Stack* const stack) {
  const auto& op_name = op.schema().operator_name();
  if (!IsCpuFallbackEnabled()) {
    TT_CHECK_THROW(false, error::kUnimplemented)
        << "operator '" << op_name
        << "' is not implemented for TPU. Please file a feature request";
  } else {
    ABSL_LOG(WARNING) << "operator '" << op_name
                      << "' is not implemented for TPU. Falling back to CPU";
    at::native::cpu_fallback(op, stack);
  }
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&TpuMissingOpFallback>());
}

// TODO: (b/448113143) -- Once TPU is upstreamed, remove this fallthrough
//  and add an AutogradTPU entry to VariableFallbackKernel.cpp.
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

// MaxPool is by default a CompositeImplicitAutograd op. The decomposition
// replaces it with MaxPool2dWithIndices (even when return_indices=False).
// Intuitively, the with-indices variant would be helpful for training, because
// the indices are needed to compute the gradient. This is NOT the case on tpu
// however, re-computing the indices via SelectAndScatter is much faster.
//
// Therefore the optimal lowering of max_pool2d uses the non-indices variant
// for both forward and backward, the latter leveraging SelectAndScatter.
// SelectAndScatter does not support dilations however, so we fallback to the
// indices variant and preserve the default composite behavior in this case.
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  Impl(m, OpName::kMaxPool2d, AtenMaxPool2d);
}

TORCH_LIBRARY(torch_tpu, m) {
  m.def(
      "max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, "
      "int[] dilation, "
      "bool ceil_mode) "
      "-> Tensor");
  m.def(
      "max_pool2d_backward(Tensor grad_output, Tensor self, int[] kernel_size, "
      "int[] stride, int[] padding, int[] dilation, bool ceil_mode) "
      "-> Tensor");
  m.def("ragged_dot(Tensor lhs, Tensor rhs, Tensor group_sizes) -> Tensor");
  m.def(
      "ragged_dot.out(Tensor lhs, Tensor rhs, Tensor grop_sizes, *, "
      "Tensor(a!) out) -> Tensor(a!)");
  m.def("optimization_barrier(Tensor[] inputs) -> Tensor[]");
  m.def(
      "stateless_dropout(Tensor rng_state, Tensor input, float p, "
      "bool? train) -> (Tensor, Tensor, Tensor)");
  // This op is a torch_tpu custom op for use in torch.compile() mode to handle
  // dynamic tensor shapes on TPU. It lowers down to
  // stablehlo.set_dimension_size which XLA uses to determine the runtime size
  // of the padded tensor dimension.
  // Args:
  //   input: The input tensor to set the dimension size of.
  //   dim: The padded dimension to set the size of. Must be non-negative and
  //     less than the input tensor's dimension.
  //   size: The size tensor that contains the runtime size of the padded
  //     dimension. Must be a 0-D tensor. The size is a tensor so as to avoid
  //     re-compilation. TODO: Explore changing this to an int and
  //     make the op handle the int to tensor promotion.
  // Returns:
  //   The input tensor with the first `size` elements of the specified
  //   dimension being valid and the rest being undefined.
  m.def(
      "set_dimension_logical_size(Tensor input, int dim, Tensor size) -> "
      "Tensor");

  // This op is a torch_tpu custom op for use in torch.compile() mode to handle
  // dynamic arange operations on TPU. It generates a sequence of numbers
  // starting from `start`, up to (but not including) `end`, by `step`.
  // Since the length of the sequence can be dynamic, it takes a `max_length`
  // to allocate a static tensor of that size, and then uses
  // `stablehlo.set_dimension_size` (during lowering) to set the runtime size.
  // Args:
  //   start: The starting value of the sequence. Must be a 0-D tensor.
  //   end: The ending value of the sequence. Must be a 0-D tensor.
  //   step: The step size between consecutive elements. Must be a 0-D tensor.
  //   max_length: The maximum possible length of the sequence. This is used for
  //     static allocation.
  //   dtype: The desired data type of the output tensor.
  // Returns:
  //   A 1-D tensor containing the generated sequence, with a physical size of
  //   `max_length` and a logical size determined by `start`, `end`, and `step`.
  m.def(
      "dynamic_arange(Tensor start, Tensor end, Tensor step, "
      "int max_length, ScalarType dtype) -> Tensor");

  // Experimental P2P communication ops for ProcessGroupTpu.
  // Isolated from the public torch.distributed API to safely prototype new
  // behaviors.
  m.def("experimental_send(Tensor[] tensors, int dst, int tag) -> Any");
  m.def("experimental_recv(Tensor[] tensors, int src, int tag) -> Any");
}

TORCH_LIBRARY_IMPL(torch_tpu, Meta, m) {
  Impl(m, OpName::kMaxPool2d,
       [](const at::Tensor& self, at::IntArrayRef kernel_size,
          at::IntArrayRef stride, at::IntArrayRef padding,
          at::IntArrayRef dilation, bool ceil_mode) {
         return at::empty(
             GetMaxPoolOutputSize(self.sizes(), kernel_size, stride, padding,
                                  dilation, ceil_mode, 2),
             self.options());
       });
  Impl(m, OpName::kMaxPool2dBackward,
       [](const at::Tensor& grad_output, const at::Tensor& self,
          at::IntArrayRef kernel_size, at::IntArrayRef stride,
          at::IntArrayRef padding, at::IntArrayRef dilation,
          bool ceil_mode) { return at::empty(self.sizes(), self.options()); });
}

TORCH_LIBRARY_IMPL(torch_tpu, PrivateUse1, m) {
  Impl(m, OpName::kDistributedExperimentalSend, TorchTpuExperimentalSend);
  Impl(m, OpName::kDistributedExperimentalRecv, TorchTpuExperimentalRecv);
  Impl(m, OpName::kMaxPool2d, TpuMaxPool2d);
  Impl(m, OpName::kMaxPool2dBackward, TpuMaxPool2dBackward);
  Impl(m, OpName::kRaggedDot, AtenRaggedDot);
  Impl(m, OpName::kRaggedDotOut, AtenRaggedDotOut);
  Impl(m, OpName::kTorchTpuOptimizationBarrier, TorchTpuOptimizationBarrier);
  Impl(m, OpName::kTorchTpuStatelessDropout, TorchTpuStatelessDropout);
  Impl(m, OpName::kSetDimensionLogicalSize, SetDimensionLogicalSize);
  Impl(m, OpName::kDynamicArange, DynamicArange);
}

TORCH_LIBRARY_IMPL(torch_tpu, CPU, m) {
  Impl(m, OpName::kRaggedDot, AtenRaggedDot);
  Impl(m, OpName::kRaggedDotOut, AtenRaggedDotOut);
}

// Returns a mutable reference to the global CPU fallback mode (defaulted to
// false).
static std::atomic<bool>& GetMutableGlobalCpuFallbackMode() {
  static std::atomic<bool> cpu_fallback_enabled = false;
  return cpu_fallback_enabled;
}

void EnableCpuFallback(const bool enabled) {
  GetMutableGlobalCpuFallbackMode().store(enabled);
}

bool IsCpuFallbackEnabled() { return GetMutableGlobalCpuFallbackMode().load(); }

}  // namespace torch_tpu

namespace at::native {
// Per https://github.com/pytorch/pytorch/issues/162989, to override
// _fused_sdp_choice We will need to REGISTER_PRIVATEUSE1_DISPATCH against
// _fused_sdp_choice_stub. See the example here:
// https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegExtra.cpp
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub,
                              &torch_tpu::AtenFusedSdpChoice);
}  // namespace at::native
