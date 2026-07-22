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
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/stack.h"
#include "ATen/native/CPUFallback.h"
#include "ATen/native/DispatchStub.h"
#include "ATen/native/Resize.h"
#include "ATen/native/transformers/attention.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_like.h"
#include "ATen/ops/result_type.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "c10/util/Exception.h"
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
#include "torch_tpu/ops/convolution/convolution_aten_kernels.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/count_nonzero/count_nonzero_aten_kernels.h"
#include "torch_tpu/ops/ctc_loss/ctc_loss_aten_kernels.h"
#include "torch_tpu/ops/cummax/cummax_aten_kernels.h"
#include "torch_tpu/ops/cummin/cummin_aten_kernels.h"
#include "torch_tpu/ops/cumprod/cumprod_aten_kernels.h"
#include "torch_tpu/ops/cumsum/cumsum_aten_kernels.h"
#include "torch_tpu/ops/digamma/digamma_aten_kernels.h"
#include "torch_tpu/ops/distance/dist_aten_kernels.h"
#include "torch_tpu/ops/dot/dot_aten_kernels.h"
#include "torch_tpu/ops/dot/vdot_aten_kernels.h"
#include "torch_tpu/ops/dropout/dropout_aten_kernels.h"
#include "torch_tpu/ops/dynamic/dynamic_arange/dynamic_arange.h"
#include "torch_tpu/ops/dynamic/dynamic_broadcast/dynamic_broadcast.h"
#include "torch_tpu/ops/dynamic/dynamic_reshape/dynamic_reshape.h"
#include "torch_tpu/ops/dynamic/set_dimension_logical_size/set_dimension_logical_size.h"
#include "torch_tpu/ops/elu/elu_aten_kernels.h"
#include "torch_tpu/ops/embedding/embedding_aten_kernels.h"
#include "torch_tpu/ops/equal/equal_aten_kernels.h"
#include "torch_tpu/ops/experimental/ragged_all_to_all/ragged_all_to_all_aten_kernels.h"
#include "torch_tpu/ops/experimental/ragged_dot/ragged_dot_aten_kernels.h"
#include "torch_tpu/ops/experimental/send_recv/send_recv_kernels.h"
#include "torch_tpu/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_aten_kernels.h"
#include "torch_tpu/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_grad_with_adagrad_aten_kernels.h"
#include "torch_tpu/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_grad_with_adam_aten_kernels.h"
#include "torch_tpu/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_grad_with_sgd_aten_kernels.h"
#include "torch_tpu/ops/exponential/exponential_aten_kernels.h"
#include "torch_tpu/ops/eye/eye_aten_kernels.h"
#include "torch_tpu/ops/fake_quantize/fake_quantize_aten_kernels.h"
#include "torch_tpu/ops/fake_quantize/fused_moving_avg_obs_fq_helper_aten_kernels.h"
#include "torch_tpu/ops/fft/fft_aten_kernels.h"
#include "torch_tpu/ops/fill/fill_aten_kernels.h"
#include "torch_tpu/ops/flip/flip_aten_kernels.h"
#include "torch_tpu/ops/fmax/fmax_aten_kernels.h"
#include "torch_tpu/ops/fmin/fmin_aten_kernels.h"
#include "torch_tpu/ops/foreach_aten_kernels.h"
#include "torch_tpu/ops/fused_adagrad/fused_adagrad_aten_kernels.h"
#include "torch_tpu/ops/fused_adam/fused_adam_aten_kernels.h"
#include "torch_tpu/ops/fused_adamw/fused_adamw_aten_kernels.h"
#include "torch_tpu/ops/fused_sgd/fused_sgd_aten_kernels.h"
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
#include "torch_tpu/ops/index_fill/index_fill_aten_kernels.h"
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
#include "torch_tpu/ops/logcumsumexp/logcumsumexp_aten_kernels.h"
#include "torch_tpu/ops/logical/logical_aten_kernels.h"
#include "torch_tpu/ops/masked_fill/masked_fill_aten_kernels.h"  // IWYU pragma: keep for AtenMaskedFill
#include "torch_tpu/ops/masked_scatter/masked_scatter_aten_kernels.h"
#include "torch_tpu/ops/masked_select/masked_select_aten_kernels.h"
#include "torch_tpu/ops/min_max/min_max_aten_kernels.h"
#include "torch_tpu/ops/mm/mm_aten_kernels.h"
#include "torch_tpu/ops/mse_loss/mse_loss_aten_kernels.h"
#include "torch_tpu/ops/multinomial/multinomial_aten_kernels.h"
#include "torch_tpu/ops/nan_to_num/nan_to_num_aten_kernels.h"
#include "torch_tpu/ops/native_batch_norm/native_batch_norm_aten_kernels.h"
#include "torch_tpu/ops/native_norm/native_norm_aten_kernels.h"
#include "torch_tpu/ops/nll_loss/nll_loss_aten_kernels.h"
#include "torch_tpu/ops/nonzero/nonzero_aten_kernels.h"
#include "torch_tpu/ops/normal/normal_aten_kernels.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/optimization_barrier/optimization_barrier_kernels.h"
#include "torch_tpu/ops/polygamma/polygamma_aten_kernels.h"
#include "torch_tpu/ops/pooling/adaptive_avg_pool_aten_kernels.h"
#include "torch_tpu/ops/pooling/avg_pool_aten_kernels.h"
#include "torch_tpu/ops/pooling/max_pool_aten_kernels.h"
#include "torch_tpu/ops/prod/prod_aten_kernels.h"
#include "torch_tpu/ops/put/put_aten_kernels.h"
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
#include "torch_tpu/ops/searchsorted_aten_kernels.h"
#include "torch_tpu/ops/set/set_aten_kernels.h"
#include "torch_tpu/ops/sigmoid/sigmoid_aten_kernels.h"
#include "torch_tpu/ops/softmax/softmax_aten_kernels.h"
#include "torch_tpu/ops/softplus/softplus_aten_kernels.h"
#include "torch_tpu/ops/sort/sort_aten_kernels.h"
#include "torch_tpu/ops/split_with_sizes_copy/split_with_sizes_copy_aten_kernels.h"
#include "torch_tpu/ops/take/take_aten_kernels.h"
#include "torch_tpu/ops/tanh/tanh_aten_kernels.h"
#include "torch_tpu/ops/thnn_fused_gru_cell/thnn_fused_gru_cell_aten_kernels.h"
#include "torch_tpu/ops/thnn_fused_lstm_cell/thnn_fused_lstm_cell_aten_kernels.h"
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
#include "torch_tpu/ops/upsample/upsample_bicubic2d_aten_kernels.h"
#include "torch_tpu/ops/view/view_aten_kernels.h"
#include "torch_tpu/ops/weight_norm/weight_norm_aten_kernels.h"
#include "torch_tpu/ops/where/where_aten_kernels.h"
#include "torch_tpu/ops/xlogy/xlogy_aten_kernels.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// Registers the kernel function for the given op name in the given library.
// Use this on the following cases:
// 1. an aten op,
// 2. an op for the Meta backend, and
// 3. a TPU-specific op that has graduated from "experimental" to "stable".
template <OpName op_name, typename KernelFn>
void ImplStable(torch::Library& m, KernelFn kernel_fn) {
  m.impl(  // M_IMPL_OK=implementing ImplStable().
      std::string(ToString(op_name)).c_str(), std::move(kernel_fn));
}

// Helper to extract the concrete signature of the kernel function.
// This is necessary because torch::Library::impl() needs to inspect the
// signature of the registered callable at compile-time to verify it against
// the operator schema.
//
// If we were to use a generic lambda (e.g., `[](auto&&... args)`), its
// `operator()` would be a template, making the signature uninspectable
// and causing compilation errors in CppFunction template deduction.
//
// ImplWrapperHelper uses template specialization to unpack the
// return type (Ret) and argument types (Args...) of the function pointer,
// allowing us to construct a monomorphic lambda with an explicit signature.
template <typename Func>
struct ImplWrapperHelper;

template <typename Ret, typename... Args>
struct ImplWrapperHelper<Ret (*)(Args...)> {
  // Registers a kernel function with the PyTorch library, injecting a callback
  // to be executed on entry.
  //
  // This helper uses template specialization to unpack the concrete signature
  // of the kernel function pointer, allowing us to construct a monomorphic
  // lambda that PyTorch's m.impl() can inspect at compile-time.
  //
  // @param op_name The OpName of the operator (compile-time template
  // parameter).
  // @param func The concrete kernel function pointer.
  // @param on_entry_fn A callback executed when the operator is invoked.
  template <OpName op_name, typename OnEntryFn>
  static void RegisterOp(torch::Library& m, Ret (*func)(Args...),
                         OnEntryFn on_entry_fn) {
    const std::string_view name = ToString(op_name);
    m.impl(  // M_IMPL_OK=implementing Impl*().
        std::string(name).c_str(),
        [func = std::move(func),
         on_entry_fn = std::move(on_entry_fn)](Args... args) -> Ret {
          on_entry_fn();
          return func(std::forward<Args>(args)...);
        });
  }
};

// Registers the kernel function for the given op name in the given library,
// and injects a user-visible warning that the operator is experimental.
//
// NOTE: It is critical that `op_name` is a compile-time template parameter
// rather than a runtime parameter. `TORCH_WARN_ONCE` dedupes warnings by its
// macro expansion site using a local `static` flag.
//
// If `op_name` were a runtime parameter, the macro would be expanded in a
// single shared helper function. This would result in "once-per-program"
// behavior, meaning only the very first experimental operator called in the
// program would trigger a warning, and all others would remain silent.
//
// By making `op_name` a template parameter, `ImplExperimental<OpName::kFoo>`
// and `ImplExperimental<OpName::kBar>` become distinct function template
// instantiations. This forces the compiler to instantiate a unique lambda and
// a unique `TORCH_WARN_ONCE` static flag for each operator, achieving the
// desired "once-per-operator" warning semantics with zero runtime overhead
// (no mutexes or registries required).
template <OpName op_name, typename KernelFn>
void ImplExperimental(torch::Library& m, KernelFn kernel_fn) {
  ImplWrapperHelper<KernelFn>::template RegisterOp<op_name>(
      m, std::move(kernel_fn), []() {
        TORCH_WARN_ONCE("operator ", ToString(op_name),
                        " is experimental; its name, signature, and behavior "
                        "may change without notice; use at your own risk");
      });
}

// Registers the kernel function for the given op name in the given library,
// and injects a user-visible warning that the operator is deprecated and will
// be removed in the given TorchTPU version.
//
// NOTE: Like `ImplExperimental`, `op_name` must be a compile-time template
// parameter to ensure `TORCH_WARN_ONCE` generates a unique static flag per
// operator, guaranteeing correct "once-per-operator" warning semantics.
template <OpName op_name, typename KernelFn>
void ImplDeprecated(torch::Library& m, KernelFn kernel_fn, int major_version,
                    int minor_version) {
  ImplWrapperHelper<KernelFn>::template RegisterOp<op_name>(
      m, std::move(kernel_fn), [major_version, minor_version]() {
        TORCH_WARN_ONCE(
            "operator ", ToString(op_name),
            " is deprecated and will be removed in TorchTPU version ",
            major_version, ".", minor_version);
      });
}

}  // namespace

// When the dispatch key set is {PrivateUse1} (i.e. for TPU tensors in the
// eager mode), pytorch will try this dispatch table first. If the op is not
// found here, pytorch will then try the (_, PrivateUse1, m) dispatch table
// defined later in this file.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // go/keep-sorted start
  ImplStable<OpName::kAbsOut>(m, AtenAbsOut);
  ImplStable<OpName::kAcosOut>(m, AtenAcosOut);
  ImplStable<OpName::kAcoshOut>(m, AtenAcoshOut);
  ImplStable<OpName::kAdaptiveAvgPool2d>(m, AtenAdaptiveAvgPool2d);
  ImplStable<OpName::kAdaptiveAvgPool2dBackward>(m,
                                                 AtenAdaptiveAvgPool2dBackward);
  ImplStable<OpName::kAdaptiveAvgPool2dOut>(m, AtenAdaptiveAvgPool2dOut);
  ImplStable<OpName::kAdaptiveAvgPool3d>(m, AtenAdaptiveAvgPool3d);
  ImplStable<OpName::kAdaptiveAvgPool3dBackward>(m,
                                                 AtenAdaptiveAvgPool3dBackward);
  ImplStable<OpName::kAdaptiveAvgPool3dBackwardGradInput>(
      m, AtenAdaptiveAvgPool3dBackwardGradInput);
  ImplStable<OpName::kAdaptiveAvgPool3dOut>(m, AtenAdaptiveAvgPool3dOut);
  ImplStable<OpName::kAddOut>(m, AtenAddOut);
  ImplStable<OpName::kAddReluOut>(m, AtenAddReluOut);
  ImplStable<OpName::kAddReluScalar>(m, AtenAddReluScalar);
  ImplStable<OpName::kAddReluTensor>(m, AtenAddReluTensor);
  ImplStable<OpName::kAddRelu_Scalar>(m, AtenAddRelu_Scalar);
  ImplStable<OpName::kAddRelu_Tensor>(m, AtenAddRelu_Tensor);
  ImplStable<OpName::kAddcdivOut>(m, AtenAddcdivOut);
  ImplStable<OpName::kAddcmulOut>(m, AtenAddcmulOut);
  ImplStable<OpName::kAddmmActivationOut>(m, AtenAddmmActivationOut);
  ImplStable<OpName::kAddmmDtype>(m, AtenAddmmDtype);
  ImplStable<OpName::kAddmmDtypeOut>(m, AtenAddmmDtypeOut);
  ImplStable<OpName::kAddmmOut>(m, AtenAddmmOut);
  ImplStable<OpName::kAddmvOut>(m, AtenAddmvOut);
  ImplStable<OpName::kAllAllOut>(m, AtenAllAllOut);
  ImplStable<OpName::kAllOut>(m, AtenAllOut);
  ImplStable<OpName::kAmaxOut>(m, AtenAmaxOut);
  ImplStable<OpName::kAminOut>(m, AtenAminOut);
  ImplStable<OpName::kAminmaxOut>(m, AtenAminmaxOut);
  ImplStable<OpName::kAnyAllOut>(m, AtenAnyAllOut);
  ImplStable<OpName::kAnyOut>(m, AtenAnyOut);
  ImplStable<OpName::kArangeStartOut>(m, AtenArangeStartOut);
  ImplStable<OpName::kArgMaxOut>(m, AtenArgmaxOut);
  ImplStable<OpName::kArgMinOut>(m, AtenArgminOut);
  ImplStable<OpName::kAsStrided>(m, AtenAsStrided);
  ImplStable<OpName::kAsinOut>(m, AtenAsinOut);
  ImplStable<OpName::kAsinhOut>(m, AtenAsinhOut);
  ImplStable<OpName::kAssertAsync>(m, AtenAssertAsync);
  ImplStable<OpName::kAssertAsyncMsg>(m, AtenAssertAsyncMsg);
  ImplStable<OpName::kAtan2Out>(m, AtenAtan2Out);
  ImplStable<OpName::kAtanOut>(m, AtenAtanOut);
  ImplStable<OpName::kAtanhOut>(m, AtenAtanhOut);
  ImplStable<OpName::kAvgPool2dBackwardGradInput>(
      m, AtenAvgPool2dBackwardGradInput);
  ImplStable<OpName::kAvgPool2dOut>(m, AtenAvgPool2dOut);
  ImplStable<OpName::kAvgPool3dBackwardGradInput>(
      m, AtenAvgPool3dBackwardGradInput);
  ImplStable<OpName::kAvgPool3dOut>(m, AtenAvgPool3dOut);
  ImplStable<OpName::kBaddbmmDtype>(m, AtenBaddbmmDtype);
  ImplStable<OpName::kBaddbmmDtypeOut>(m, AtenBaddbmmDtypeOut);
  ImplStable<OpName::kBaddbmmOut>(m, AtenBaddbmmOut);
  ImplStable<OpName::kBernoulliOut>(m, AtenBernoulliOut);
  ImplStable<OpName::kBernoulli_Float>(m, AtenBernoulli_Float);
  ImplStable<OpName::kBernoulli_Tensor>(m, AtenBernoulli_Tensor);
  ImplStable<OpName::kBinCount>(m, AtenBinCount);
  ImplStable<OpName::kBitwiseAndTensorOut>(m, AtenBitwiseAndTensorOut);
  ImplStable<OpName::kBitwiseLeftShiftTensorOut>(m,
                                                 AtenBitwiseLeftShiftTensorOut);
  ImplStable<OpName::kBitwiseNotOut>(m, AtenBitwiseNotOut);
  ImplStable<OpName::kBitwiseOrTensorOut>(m, AtenBitwiseOrTensorOut);
  ImplStable<OpName::kBitwiseRightShiftTensorOut>(
      m, AtenBitwiseRightShiftTensorOut);
  ImplStable<OpName::kBitwiseXorTensorOut>(m, AtenBitwiseXorTensorOut);
  ImplStable<OpName::kBmmDtype>(m, AtenBmmDtype);
  ImplStable<OpName::kBmmDtypeOut>(m, AtenBmmDtypeOut);
  ImplStable<OpName::kBmmOut>(m, AtenBmmOut);
  ImplStable<OpName::kBucketizeScalar>(m, AtenBucketizeScalar);
  ImplStable<OpName::kBucketizeTensor>(m, AtenBucketizeTensor);
  ImplStable<OpName::kBucketizeTensorOut>(m, AtenBucketizeTensorOut);
  ImplStable<OpName::kCatOut>(m, AtenCatOut);
  ImplStable<OpName::kCdistBackward>(m, AtenCdistBackward);
  ImplStable<OpName::kCdistForward>(m, AtenCdistForward);
  ImplStable<OpName::kCeilOut>(m, AtenCeilOut);
  ImplStable<OpName::kClampMaxOut>(m, AtenClampMaxOut);
  ImplStable<OpName::kClampMaxTensorOut>(m, AtenClampMaxTensorOut);
  ImplStable<OpName::kClampMinOut>(m, AtenClampMinOut);
  ImplStable<OpName::kClampMinTensorOut>(m, AtenClampMinTensorOut);
  ImplStable<OpName::kClampOut>(m, AtenClampOut);
  ImplStable<OpName::kClampTensorOut>(m, AtenClampTensorOut);
  ImplStable<OpName::kCol2Im>(m, AtenCol2Im);
  ImplStable<OpName::kCol2ImOut>(m, AtenCol2ImOut);
  ImplStable<OpName::kComplexOut>(m, AtenComplexOut);
  ImplStable<OpName::kConjPhysicalOut>(m, AtenConjPhysicalOut);
  ImplStable<OpName::kConvolution>(m, AtenConvolution);
  ImplStable<OpName::kConvolutionBackward>(m, AtenConvolutionBackward);
  ImplStable<OpName::kConvolutionOut>(m, AtenConvolutionOut);
  ImplStable<OpName::kCopyFrom>(m, AtenCopyFrom);
  // per https://github.com/pytorch/xla/issues/2881, this function was added
  // to fix aten:cpufallback for pytorch_xla in 2021.
  // But it isn't registered for CPU and is not called from copy_ in Python.
  // TODO: Revisit if we can replace it with _copy_from. // NOLINT
  ImplStable<OpName::kCopyFromAndResize>(m, AtenCopyFromAndResize);
  ImplStable<OpName::kCopy_>(m, AtenCopy_);
  ImplStable<OpName::kCosOut>(m, AtenCosOut);
  ImplStable<OpName::kCoshOut>(m, AtenCoshOut);
  ImplStable<OpName::kCountNonzeroDimIntList>(m, AtenCountNonzeroDimIntList);
  ImplStable<OpName::kCtcLoss>(m, AtenCtcLoss);
  ImplStable<OpName::kCtcLossBackward>(m, AtenCtcLossBackward);
  ImplStable<OpName::kCtcLossBackwardTensor>(m, AtenCtcLossBackwardTensor);
  // TODO(b/513607161): remove CtcLossPublic overrides once upstream PyTorch bug
  // is fixed.
  // This is a workaround for a bug in PyTorch core's ctc_loss implementation
  // when used in combination with the privateuse1 backend and bf16 dtype. PT's
  // ctc_loss impl attempts to initialize a tensor on CPU with bf16 and triggers
  // a buggy codepath for non cpu/gpu backends. Overriding the "public" variant
  // of ctc_loss (rather than strictly the internal _ctc_loss variant that we
  // already override) is necessary to work around the bug.
  ImplStable<OpName::kCtcLossPublic>(m, AtenCtcLossPublic);
  ImplStable<OpName::kCtcLossPublicTensor>(m, AtenCtcLossPublicTensor);
  ImplStable<OpName::kCtcLossTensor>(m, AtenCtcLossTensor);
  ImplStable<OpName::kCummaxHelper>(m, AtenCummaxHelper);
  ImplStable<OpName::kCumminHelper>(m, AtenCumminHelper);
  ImplStable<OpName::kCumprodOut>(m, AtenCumprodOut);
  ImplStable<OpName::kCumsumOut>(m, AtenCumsumOut);
  ImplStable<OpName::kDigammaOut>(m, AtenDigammaOut);
  ImplStable<OpName::kDivOut>(m, AtenDivOut);
  ImplStable<OpName::kDivOutMode>(m, AtenDivOutMode);
  ImplStable<OpName::kDot>(m, AtenDot);
  ImplStable<OpName::kEfficientZeroTensor>(m, AtenEfficientZeroTensor);
  ImplStable<OpName::kEluBackwardGradInput>(m, AtenEluBackwardGradInput);
  ImplStable<OpName::kEluOut>(m, AtenEluOut);
  ImplStable<OpName::kEmbeddingBag>(m, AtenEmbeddingBag);
  ImplStable<OpName::kEmbeddingBagBackward>(m, AtenEmbeddingBagBackward);
  ImplStable<OpName::kEmbeddingBagForwardOnly>(m, AtenEmbeddingBagForwardOnly);
  ImplStable<OpName::kEmbeddingDenseBackward>(m, AtenEmbeddingDenseBackward);
  ImplStable<OpName::kEmbeddingRenorm_>(m, AtenEmbeddingRenorm_);
  ImplStable<OpName::kEmptyMemoryFormat>(m, AtenEmptyMemoryFormat);
  ImplStable<OpName::kEmptyStrided>(m, AtenEmptyStrided);
  ImplStable<OpName::kEqScalarOut>(m, AtenEqScalarOut);
  ImplStable<OpName::kEqTensorOut>(m, AtenEqTensorOut);
  ImplStable<OpName::kEqual>(m, AtenEqual);
  ImplStable<OpName::kErfInvOut>(m, AtenErfInvOut);
  ImplStable<OpName::kErfOut>(m, AtenErfOut);
  ImplStable<OpName::kExp2Out>(m, AtenExp2Out);
  ImplStable<OpName::kExpM1Out>(m, AtenExpm1Out);
  ImplStable<OpName::kExpOut>(m, AtenExpOut);
  ImplStable<OpName::kExponential_>(m, AtenExponential_);
  ImplStable<OpName::kEyeMOut>(m, AtenEyeMOut);
  ImplStable<OpName::kEyeOut>(m, AtenEyeOut);
  ImplStable<OpName::kFakeQuantizePerTensorAffineCachemask>(
      m, FakeQuantizePerTensorAffineCachemask);
  ImplStable<OpName::kFftC2c>(m, AtenFftC2c);
  ImplStable<OpName::kFftC2cOut>(m, AtenFftC2cOut);
  ImplStable<OpName::kFftC2r>(m, AtenFftC2r);
  ImplStable<OpName::kFftC2rOut>(m, AtenFftC2rOut);
  ImplStable<OpName::kFftR2c>(m, AtenFftR2c);
  ImplStable<OpName::kFftR2cOut>(m, AtenFftR2cOut);
  ImplStable<OpName::kFill_Scalar>(m, AtenFillScalar_);
  ImplStable<OpName::kFill_Tensor>(m, AtenFillTensor_);
  ImplStable<OpName::kFlip>(m, AtenFlip);
  ImplStable<OpName::kFloorDivide>(m, AtenFloorDivide);
  ImplStable<OpName::kFloorDivideOut>(m, AtenFloorDivideOut);
  ImplStable<OpName::kFloorDivide_Tensor>(m, AtenFloorDivide_Tensor);
  ImplStable<OpName::kFloorOut>(m, AtenFloorOut);
  ImplStable<OpName::kFmaxOut>(m, AtenFmaxOut);
  ImplStable<OpName::kFminOut>(m, AtenFminOut);
  ImplStable<OpName::kFmodTensorOut>(m, AtenFmodTensorOut);
  ImplStable<OpName::kForeachAbs>(m, AtenForeachAbs);
  ImplStable<OpName::kForeachAbs_>(m, AtenForeachAbs_);
  ImplStable<OpName::kForeachAcos>(m, AtenForeachAcos);
  ImplStable<OpName::kForeachAcos_>(m, AtenForeachAcos_);
  ImplStable<OpName::kForeachAddList>(m, AtenForeachAddList);
  ImplStable<OpName::kForeachAddScalar>(m, AtenForeachAddScalar);
  ImplStable<OpName::kForeachAddScalarList>(m, AtenForeachAddScalarList);
  ImplStable<OpName::kForeachAddTensor>(m, AtenForeachAddTensor);
  ImplStable<OpName::kForeachAdd_List>(m, AtenForeachAdd_List);
  ImplStable<OpName::kForeachAdd_Scalar>(m, AtenForeachAdd_Scalar);
  ImplStable<OpName::kForeachAdd_ScalarList>(m, AtenForeachAdd_ScalarList);
  ImplStable<OpName::kForeachAdd_Tensor>(m, AtenForeachAdd_Tensor);
  ImplStable<OpName::kForeachAddcdivScalar>(m, AtenForeachAddcdivScalar);
  ImplStable<OpName::kForeachAddcdivScalarList>(m,
                                                AtenForeachAddcdivScalarList);
  ImplStable<OpName::kForeachAddcdivTensor>(m, AtenForeachAddcdivTensor);
  ImplStable<OpName::kForeachAddcdiv_Scalar>(m, AtenForeachAddcdiv_Scalar);
  ImplStable<OpName::kForeachAddcdiv_ScalarList>(m,
                                                 AtenForeachAddcdiv_ScalarList);
  ImplStable<OpName::kForeachAddcdiv_Tensor>(m, AtenForeachAddcdiv_Tensor);
  ImplStable<OpName::kForeachAddcmulScalar>(m, AtenForeachAddcmulScalar);
  ImplStable<OpName::kForeachAddcmulScalarList>(m,
                                                AtenForeachAddcmulScalarList);
  ImplStable<OpName::kForeachAddcmulTensor>(m, AtenForeachAddcmulTensor);
  ImplStable<OpName::kForeachAddcmul_Scalar>(m, AtenForeachAddcmul_Scalar);
  ImplStable<OpName::kForeachAddcmul_ScalarList>(m,
                                                 AtenForeachAddcmul_ScalarList);
  ImplStable<OpName::kForeachAddcmul_Tensor>(m, AtenForeachAddcmul_Tensor);
  ImplStable<OpName::kForeachAsin>(m, AtenForeachAsin);
  ImplStable<OpName::kForeachAsin_>(m, AtenForeachAsin_);
  ImplStable<OpName::kForeachAtan>(m, AtenForeachAtan);
  ImplStable<OpName::kForeachAtan_>(m, AtenForeachAtan_);
  ImplStable<OpName::kForeachCeil>(m, AtenForeachCeil);
  ImplStable<OpName::kForeachCeil_>(m, AtenForeachCeil_);
  ImplStable<OpName::kForeachClampMaxList>(m, AtenForeachClampMaxList);
  ImplStable<OpName::kForeachClampMaxScalar>(m, AtenForeachClampMaxScalar);
  ImplStable<OpName::kForeachClampMaxScalarList>(m,
                                                 AtenForeachClampMaxScalarList);
  ImplStable<OpName::kForeachClampMax_List>(m, AtenForeachClampMax_List);
  ImplStable<OpName::kForeachClampMax_Scalar>(m, AtenForeachClampMax_Scalar);
  ImplStable<OpName::kForeachClampMax_ScalarList>(
      m, AtenForeachClampMax_ScalarList);
  ImplStable<OpName::kForeachClampMinList>(m, AtenForeachClampMinList);
  ImplStable<OpName::kForeachClampMinScalar>(m, AtenForeachClampMinScalar);
  ImplStable<OpName::kForeachClampMinScalarList>(m,
                                                 AtenForeachClampMinScalarList);
  ImplStable<OpName::kForeachClampMin_List>(m, AtenForeachClampMin_List);
  ImplStable<OpName::kForeachClampMin_Scalar>(m, AtenForeachClampMin_Scalar);
  ImplStable<OpName::kForeachClampMin_ScalarList>(
      m, AtenForeachClampMin_ScalarList);
  ImplStable<OpName::kForeachCopy_>(m, AtenForeachCopy_);
  ImplStable<OpName::kForeachCos>(m, AtenForeachCos);
  ImplStable<OpName::kForeachCos_>(m, AtenForeachCos_);
  ImplStable<OpName::kForeachCosh>(m, AtenForeachCosh);
  ImplStable<OpName::kForeachCosh_>(m, AtenForeachCosh_);
  ImplStable<OpName::kForeachDivList>(m, AtenForeachDivList);
  ImplStable<OpName::kForeachDivScalar>(m, AtenForeachDivScalar);
  ImplStable<OpName::kForeachDivScalarList>(m, AtenForeachDivScalarList);
  ImplStable<OpName::kForeachDivTensor>(m, AtenForeachDivTensor);
  ImplStable<OpName::kForeachDiv_List>(m, AtenForeachDiv_List);
  ImplStable<OpName::kForeachDiv_Scalar>(m, AtenForeachDiv_Scalar);
  ImplStable<OpName::kForeachDiv_ScalarList>(m, AtenForeachDiv_ScalarList);
  ImplStable<OpName::kForeachDiv_Tensor>(m, AtenForeachDiv_Tensor);
  ImplStable<OpName::kForeachErf>(m, AtenForeachErf);
  ImplStable<OpName::kForeachErf_>(m, AtenForeachErf_);
  ImplStable<OpName::kForeachErfc>(m, AtenForeachErfc);
  ImplStable<OpName::kForeachErfc_>(m, AtenForeachErfc_);
  ImplStable<OpName::kForeachExp>(m, AtenForeachExp);
  ImplStable<OpName::kForeachExp_>(m, AtenForeachExp_);
  ImplStable<OpName::kForeachExpm1>(m, AtenForeachExpm1);
  ImplStable<OpName::kForeachExpm1_>(m, AtenForeachExpm1_);
  ImplStable<OpName::kForeachFloor>(m, AtenForeachFloor);
  ImplStable<OpName::kForeachFloor_>(m, AtenForeachFloor_);
  ImplStable<OpName::kForeachFrac>(m, AtenForeachFrac);
  ImplStable<OpName::kForeachFrac_>(m, AtenForeachFrac_);
  ImplStable<OpName::kForeachLerpList>(m, AtenForeachLerpList);
  ImplStable<OpName::kForeachLerpScalar>(m, AtenForeachLerpScalar);
  ImplStable<OpName::kForeachLerpScalarList>(m, AtenForeachLerpScalarList);
  ImplStable<OpName::kForeachLerp_List>(m, AtenForeachLerp_List);
  ImplStable<OpName::kForeachLerp_Scalar>(m, AtenForeachLerp_Scalar);
  ImplStable<OpName::kForeachLerp_ScalarList>(m, AtenForeachLerp_ScalarList);
  ImplStable<OpName::kForeachLgamma>(m, AtenForeachLgamma);
  ImplStable<OpName::kForeachLgamma_>(m, AtenForeachLgamma_);
  ImplStable<OpName::kForeachLog10>(m, AtenForeachLog10);
  ImplStable<OpName::kForeachLog10_>(m, AtenForeachLog10_);
  ImplStable<OpName::kForeachLog1p>(m, AtenForeachLog1p);
  ImplStable<OpName::kForeachLog1p_>(m, AtenForeachLog1p_);
  ImplStable<OpName::kForeachLog2>(m, AtenForeachLog2);
  ImplStable<OpName::kForeachLog2_>(m, AtenForeachLog2_);
  ImplStable<OpName::kForeachLog>(m, AtenForeachLog);
  ImplStable<OpName::kForeachLog_>(m, AtenForeachLog_);
  ImplStable<OpName::kForeachMax>(m, AtenForeachMax);
  ImplStable<OpName::kForeachMaximumList>(m, AtenForeachMaximumList);
  ImplStable<OpName::kForeachMaximumScalar>(m, AtenForeachMaximumScalar);
  ImplStable<OpName::kForeachMaximumScalarList>(m,
                                                AtenForeachMaximumScalarList);
  ImplStable<OpName::kForeachMaximum_List>(m, AtenForeachMaximum_List);
  ImplStable<OpName::kForeachMaximum_Scalar>(m, AtenForeachMaximum_Scalar);
  ImplStable<OpName::kForeachMaximum_ScalarList>(m,
                                                 AtenForeachMaximum_ScalarList);
  ImplStable<OpName::kForeachMinimumList>(m, AtenForeachMinimumList);
  ImplStable<OpName::kForeachMinimumScalar>(m, AtenForeachMinimumScalar);
  ImplStable<OpName::kForeachMinimumScalarList>(m,
                                                AtenForeachMinimumScalarList);
  ImplStable<OpName::kForeachMinimum_List>(m, AtenForeachMinimum_List);
  ImplStable<OpName::kForeachMinimum_Scalar>(m, AtenForeachMinimum_Scalar);
  ImplStable<OpName::kForeachMinimum_ScalarList>(m,
                                                 AtenForeachMinimum_ScalarList);
  ImplStable<OpName::kForeachMulList>(m, AtenForeachMulList);
  ImplStable<OpName::kForeachMulScalar>(m, AtenForeachMulScalar);
  ImplStable<OpName::kForeachMulScalarList>(m, AtenForeachMulScalarList);
  ImplStable<OpName::kForeachMulTensor>(m, AtenForeachMulTensor);
  ImplStable<OpName::kForeachMul_List>(m, AtenForeachMul_List);
  ImplStable<OpName::kForeachMul_Scalar>(m, AtenForeachMul_Scalar);
  ImplStable<OpName::kForeachMul_ScalarList>(m, AtenForeachMul_ScalarList);
  ImplStable<OpName::kForeachMul_Tensor>(m, AtenForeachMul_Tensor);
  ImplStable<OpName::kForeachNeg>(m, AtenForeachNeg);
  ImplStable<OpName::kForeachNeg_>(m, AtenForeachNeg_);
  ImplStable<OpName::kForeachNormScalar>(m, AtenForeachNormScalar);
  ImplStable<OpName::kForeachPowList>(m, AtenForeachPowList);
  ImplStable<OpName::kForeachPowScalar>(m, AtenForeachPowScalar);
  ImplStable<OpName::kForeachPowScalarAndTensor>(m,
                                                 AtenForeachPowScalarAndTensor);
  ImplStable<OpName::kForeachPowScalarList>(m, AtenForeachPowScalarList);
  ImplStable<OpName::kForeachPow_List>(m, AtenForeachPow_List);
  ImplStable<OpName::kForeachPow_Scalar>(m, AtenForeachPow_Scalar);
  ImplStable<OpName::kForeachPow_ScalarList>(m, AtenForeachPow_ScalarList);
  ImplStable<OpName::kForeachReciprocal>(m, AtenForeachReciprocal);
  ImplStable<OpName::kForeachReciprocal_>(m, AtenForeachReciprocal_);
  ImplStable<OpName::kForeachRound>(m, AtenForeachRound);
  ImplStable<OpName::kForeachRound_>(m, AtenForeachRound_);
  ImplStable<OpName::kForeachRsqrt>(m, AtenForeachRsqrt);
  ImplStable<OpName::kForeachRsqrt_>(m, AtenForeachRsqrt_);
  ImplStable<OpName::kForeachSigmoid>(m, AtenForeachSigmoid);
  ImplStable<OpName::kForeachSigmoid_>(m, AtenForeachSigmoid_);
  ImplStable<OpName::kForeachSign>(m, AtenForeachSign);
  ImplStable<OpName::kForeachSign_>(m, AtenForeachSign_);
  ImplStable<OpName::kForeachSin>(m, AtenForeachSin);
  ImplStable<OpName::kForeachSin_>(m, AtenForeachSin_);
  ImplStable<OpName::kForeachSinh>(m, AtenForeachSinh);
  ImplStable<OpName::kForeachSinh_>(m, AtenForeachSinh_);
  ImplStable<OpName::kForeachSqrt>(m, AtenForeachSqrt);
  ImplStable<OpName::kForeachSqrt_>(m, AtenForeachSqrt_);
  ImplStable<OpName::kForeachSubList>(m, AtenForeachSubList);
  ImplStable<OpName::kForeachSubScalar>(m, AtenForeachSubScalar);
  ImplStable<OpName::kForeachSubScalarList>(m, AtenForeachSubScalarList);
  ImplStable<OpName::kForeachSub_List>(m, AtenForeachSub_List);
  ImplStable<OpName::kForeachSub_Scalar>(m, AtenForeachSub_Scalar);
  ImplStable<OpName::kForeachSub_ScalarList>(m, AtenForeachSub_ScalarList);
  ImplStable<OpName::kForeachTan>(m, AtenForeachTan);
  ImplStable<OpName::kForeachTan_>(m, AtenForeachTan_);
  ImplStable<OpName::kForeachTanh>(m, AtenForeachTanh);
  ImplStable<OpName::kForeachTanh_>(m, AtenForeachTanh_);
  ImplStable<OpName::kForeachTrunc>(m, AtenForeachTrunc);
  ImplStable<OpName::kForeachTrunc_>(m, AtenForeachTrunc_);
  ImplStable<OpName::kForeachZero_>(m, AtenForeachZero_);
  ImplStable<OpName::kFusedAdagrad>(m, AtenFusedAdagrad);
  ImplStable<OpName::kFusedAdagradTensorLr>(m, AtenFusedAdagradTensorLr);
  ImplStable<OpName::kFusedAdam>(m, AtenFusedAdam);
  ImplStable<OpName::kFusedAdamTensorLr>(m, AtenFusedAdamTensorLr);
  ImplStable<OpName::kFusedAdamw>(m, AtenFusedAdamw);
  ImplStable<OpName::kFusedAdamwTensorLr>(m, AtenFusedAdamwTensorLr);
  ImplStable<OpName::kFusedDropout>(m, AtenFusedDropout);
  ImplStable<OpName::kFusedMovingAvgObsFqHelper>(m, FusedMovingAvgObsFqHelper);
  ImplStable<OpName::kFusedRmsNorm>(m, AtenFusedRmsNorm);
  ImplStable<OpName::kFusedRmsNormBackward>(m, AtenFusedRmsNormBackward);
  ImplStable<OpName::kFusedSgd>(m, AtenFusedSgd);
  ImplStable<OpName::kFusedSgdTensorLr>(m, AtenFusedSgdTensorLr);
  ImplStable<OpName::kGather>(m, AtenGather);
  ImplStable<OpName::kGatherOut>(m, AtenGatherOut);
  ImplStable<OpName::kGeScalarOut>(m, AtenGeScalarOut);
  ImplStable<OpName::kGeTensorOut>(m, AtenGeTensorOut);
  ImplStable<OpName::kGelu>(m, AtenGelu);
  ImplStable<OpName::kGeluBackwardGradInput>(m, AtenGeluBackwardGradInput);
  ImplStable<OpName::kGeluOut>(m, AtenGeluOut);
  ImplStable<OpName::kGeqrf>(m, AtenGeqrf);
  ImplStable<OpName::kGeqrfA>(m, AtenGeqrfA);
  ImplStable<OpName::kGluBackward>(m, AtenGluBackward);
  ImplStable<OpName::kGluBackwardGradInput>(m, AtenGluBackwardGradInput);
  ImplStable<OpName::kGluOut>(m, AtenGluOut);
  ImplStable<OpName::kGridSampler2d>(m, AtenGridSampler2d);
  ImplStable<OpName::kGridSampler2dBackward>(m, AtenGridSampler2dBackward);
  ImplStable<OpName::kGridSampler3d>(m, AtenGridSampler3d);
  ImplStable<OpName::kGridSampler3dBackward>(m, AtenGridSampler3dBackward);
  ImplStable<OpName::kGroupedMm>(m, AtenGroupedMm);
  ImplStable<OpName::kGtScalarOut>(m, AtenGtScalarOut);
  ImplStable<OpName::kGtTensorOut>(m, AtenGtTensorOut);
  ImplStable<OpName::kHardsigmoidBackwardGradInput>(
      m, AtenHardsigmoidBackwardGradInput);
  ImplStable<OpName::kHardsigmoidOut>(m, AtenHardsigmoidOut);
  ImplStable<OpName::kHardswish>(m, AtenHardswish);
  ImplStable<OpName::kHardswishBackward>(m, AtenHardswishBackward);
  ImplStable<OpName::kHardswishOut>(m, AtenHardswishOut);
  ImplStable<OpName::kHardswish_>(m, AtenHardswish_);
  ImplStable<OpName::kHardtanh>(m, AtenHardtanh);
  ImplStable<OpName::kHardtanhBackward>(m, AtenHardtanhBackward);
  ImplStable<OpName::kHardtanhBackwardGradInput>(m,
                                                 AtenHardtanhBackwardGradInput);
  ImplStable<OpName::kHardtanhOut>(m, AtenHardtanhOut);
  ImplStable<OpName::kHardtanh_>(m, AtenHardtanh_);
  ImplStable<OpName::kHistc>(m, AtenHistc);
  ImplStable<OpName::kHistcOut>(m, AtenHistcOut);
  ImplStable<OpName::kIlshiftScalar>(m, AtenIlshiftScalar);
  ImplStable<OpName::kIlshiftTensor>(m, AtenIlshiftTensor);
  ImplStable<OpName::kIm2Col>(m, AtenIm2Col);
  ImplStable<OpName::kIm2ColOut>(m, AtenIm2ColOut);
  ImplStable<OpName::kIndexAddOut>(m, TpuAtenIndexAddOut);
  ImplStable<OpName::kIndexCopyOut>(m, AtenIndexCopyOut);
  ImplStable<OpName::kIndexFillIntScalar>(m, AtenIndexFillIntScalar_);
  ImplStable<OpName::kIndexFillIntTensor>(m, AtenIndexFillIntTensor_);
  ImplStable<OpName::kIndexPutImpl_>(m, TpuAtenIndexPutImpl_);
  ImplStable<OpName::kIndexReduceOut>(m, TpuAtenIndexReduceOut);
  ImplStable<OpName::kIndexSelect>(m, TpuAtenIndexSelect);
  ImplStable<OpName::kIndexTensorOut>(m, AtenIndexTensorOut);
  ImplStable<OpName::kIrshiftScalar>(m, AtenIrshiftScalar);
  ImplStable<OpName::kIrshiftTensor>(m, AtenIrshiftTensor);
  ImplStable<OpName::kIsInScalarTensorOut>(m, AtenIsInScalarTensorOut);
  ImplStable<OpName::kIsInTensorScalarOut>(m, AtenIsInTensorScalarOut);
  ImplStable<OpName::kIsInTensorTensorOut>(m, AtenIsInTensorTensorOut);
  ImplStable<OpName::kIsNan>(m, AtenIsNan);
  ImplStable<OpName::kIsNegInfOut>(m, AtenIsNegInfOut);
  ImplStable<OpName::kIsPosInfOut>(m, AtenIsPosInfOut);
  ImplStable<OpName::kLdexpOut>(m, AtenLdexpOut);
  ImplStable<OpName::kLdexpTensor>(m, AtenLdexpTensor);
  ImplStable<OpName::kLdexp_>(m, AtenLdexp_);
  ImplStable<OpName::kLeScalarOut>(m, AtenLeScalarOut);
  ImplStable<OpName::kLeTensorOut>(m, AtenLeTensorOut);
  ImplStable<OpName::kLeakyReluBackward>(m, AtenLeakyReluBackwardGradInput);
  ImplStable<OpName::kLeakyReluOut>(m, AtenLeakyReluOut);
  ImplStable<OpName::kLerpScalarOut>(m, AtenLerpScalarOut);
  ImplStable<OpName::kLerpTensorOut>(m, AtenLerpTensorOut);
  ImplStable<OpName::kLgammaOut>(m, AtenLgammaOut);
  ImplStable<OpName::kLinalgInvExInverse>(m, AtenLinalgInvExInverse);
  ImplStable<OpName::kLinalgLuFactorExOut>(m, AtenLinalgLuFactorExOut);
  ImplStable<OpName::kLinalgLuOut>(m, AtenLinalgLuOut);
  ImplStable<OpName::kLinalgLuSolveOut>(m, AtenLinalgLuSolveOut);
  ImplStable<OpName::kLinalgQrOut>(m, AtenLinalgQrOut);
  ImplStable<OpName::kLinalgSolveExOut>(m, AtenLinalgSolveExOut);
  ImplStable<OpName::kLinalgSolveTriangular>(m, AtenLinalgSolveTriangular);
  ImplStable<OpName::kLinalgSolveTriangularOut>(m,
                                                AtenLinalgSolveTriangularOut);
  ImplStable<OpName::kLinalgVectorNormOut>(m, AtenLinalgVectorNormOut);
  ImplStable<OpName::kLinspaceOut>(m, AtenLinspaceOut);
  ImplStable<OpName::kLocalScalarDense>(m, AtenLocalScalarDense);
  ImplStable<OpName::kLog10Out>(m, AtenLog10Out);
  ImplStable<OpName::kLog1pOut>(m, AtenLog1pOut);
  ImplStable<OpName::kLog2Out>(m, AtenLog2Out);
  ImplStable<OpName::kLogOut>(m, AtenLogOut);
  ImplStable<OpName::kLogSigmoidBackward>(m, AtenLogSigmoidBackward);
  ImplStable<OpName::kLogSigmoidBackwardGradInput>(
      m, AtenLogSigmoidBackwardGradInput);
  ImplStable<OpName::kLogSigmoidForward>(m, AtenLogSigmoidForward);
  ImplStable<OpName::kLogSigmoidForwardOut>(m, AtenLogSigmoidForwardOut);
  ImplStable<OpName::kLogSoftmaxBackwardDataOut>(m,
                                                 AtenLogSoftmaxBackwardDataOut);
  ImplStable<OpName::kLogSoftmaxOut>(m, AtenLogSoftmaxOut);
  ImplStable<OpName::kLogcumsumexp>(m, AtenLogcumsumexp);
  ImplStable<OpName::kLogcumsumexpOut>(m, AtenLogcumsumexpOut);
  ImplStable<OpName::kLogicalAndOut>(m, AtenLogicalAndOut);
  ImplStable<OpName::kLogicalNotOut>(m, AtenLogicalNotOut);
  ImplStable<OpName::kLogicalOrOut>(m, AtenLogicalOrOut);
  ImplStable<OpName::kLogicalXorOut>(m, AtenLogicalXorOut);
  ImplStable<OpName::kLshiftScalar>(m, AtenLshiftScalar);
  ImplStable<OpName::kLshiftTensor>(m, AtenLshiftTensor);
  ImplStable<OpName::kLtScalarOut>(m, AtenLtScalarOut);
  ImplStable<OpName::kLtTensorOut>(m, AtenLtTensorOut);
  ImplStable<OpName::kLuUnpackOut>(m, AtenLuUnpackOut);
  ImplStable<OpName::kMaskedFill_Scalar>(m, AtenMaskedFill_Scalar);
  ImplStable<OpName::kMaskedFill_Tensor>(m, AtenMaskedFill_Tensor);
  ImplStable<OpName::kMaskedScatter_>(m, AtenMaskedScatter_);
  ImplStable<OpName::kMaskedSelect>(m, AtenMaskedSelect);
  ImplStable<OpName::kMaskedSelectOut>(m, AtenMaskedSelectOut);
  ImplStable<OpName::kMax>(m, AtenMax);
  ImplStable<OpName::kMaxDimMax>(m, AtenMaxDimMax);
  ImplStable<OpName::kMaxPool2dWithIndicesBackwardGradInput>(
      m, AtenMaxPool2dWithIndicesBackwardGradInput);
  ImplStable<OpName::kMaxPool2dWithIndicesOut>(m, AtenMaxPool2dWithIndicesOut);
  ImplStable<OpName::kMaxPool3dWithIndices>(m, AtenMaxPool3dWithIndices);
  ImplStable<OpName::kMaxPool3dWithIndicesBackward>(
      m, AtenMaxPool3dWithIndicesBackward);
  ImplStable<OpName::kMaxPool3dWithIndicesBackwardGradInput>(
      m, AtenMaxPool3dWithIndicesBackwardGradInput);
  ImplStable<OpName::kMaxPool3dWithIndicesOut>(m, AtenMaxPool3dWithIndicesOut);
  ImplStable<OpName::kMaxUnaryOut>(m, AtenMaxUnaryOut);
  ImplStable<OpName::kMaximumOut>(m, AtenMaximumOut);
  ImplStable<OpName::kMeanOut>(m, AtenMeanOut);
  ImplStable<OpName::kMin>(m, AtenMin);
  ImplStable<OpName::kMinDimMin>(m, AtenMinDimMin);
  ImplStable<OpName::kMinUnaryOut>(m, AtenMinUnaryOut);
  ImplStable<OpName::kMinimumOut>(m, AtenMinimumOut);
  ImplStable<OpName::kMmDtype>(m, AtenMmDtype);
  ImplStable<OpName::kMmDtypeOut>(m, AtenMmDtypeOut);
  ImplStable<OpName::kMmOut>(m, AtenMmOut);
  ImplStable<OpName::kMseLossBackward>(m, AtenMseLossBackward);
  ImplStable<OpName::kMseLossOut>(m, AtenMseLossOut);
  ImplStable<OpName::kMulOut>(m, AtenMulOut);
  ImplStable<OpName::kMultinomial>(m, AtenMultinomial);
  ImplStable<OpName::kMultinomialOut>(m, AtenMultinomialOut);
  ImplStable<OpName::kNanToNumOut>(m, AtenNanToNumOut);
  ImplStable<OpName::kNativeBatchNorm>(m, AtenNativeBatchNorm);
  ImplStable<OpName::kNativeBatchNormBackward>(m, AtenNativeBatchNormBackward);
  ImplStable<OpName::kNativeBatchNormLegit>(m, AtenNativeBatchNormLegit);
  ImplStable<OpName::kNativeBatchNormLegitNoStats>(
      m, AtenNativeBatchNormLegitNoStats);
  ImplStable<OpName::kNativeBatchNormLegitNoStatsOut>(
      m, AtenNativeBatchNormLegitNoStatsOut);
  ImplStable<OpName::kNativeBatchNormLegitOut>(m, AtenNativeBatchNormLegitOut);
  ImplStable<OpName::kNativeBatchNormOut>(m, AtenNativeBatchNormOut);
  ImplStable<OpName::kNativeDropout>(m, AtenDropout);
  ImplStable<OpName::kNativeDropoutBackward>(m, AtenNativeDropoutBackward);
  ImplStable<OpName::kNativeGroupNormBackward>(m, AtenNativeGroupNormBackward);
  ImplStable<OpName::kNativeLayerNorm>(m, AtenNativeLayerNorm);
  ImplStable<OpName::kNativeLayerNormBackward>(m, AtenLayerNormBackward);
  ImplStable<OpName::kNativeNorm>(m, AtenNativeNormScalar);
  ImplStable<OpName::kNativeNormScalarOptDimDtype>(
      m, AtenNativeNormScalarOptDimDtype);
  ImplStable<OpName::kNeScalarOut>(m, AtenNeScalarOut);
  ImplStable<OpName::kNeTensorOut>(m, AtenNeTensorOut);
  ImplStable<OpName::kNegOut>(m, AtenNegOut);
  ImplStable<OpName::kNllLoss2dBackward>(m, AtenNllLoss2dBackward);
  ImplStable<OpName::kNllLoss2dBackwardGradInput>(
      m, AtenNllLoss2dBackwardGradInput);
  ImplStable<OpName::kNllLoss2dForward>(m, AtenNllLoss2dForward);
  ImplStable<OpName::kNllLoss2dForwardOut>(m, AtenNllLoss2dForwardOut);
  ImplStable<OpName::kNllLossBackwardGradInput>(m,
                                                AtenNllLossBackwardGradInput);
  ImplStable<OpName::kNllLossForwardOut>(m, AtenNllLossForwardOut);
  ImplStable<OpName::kNonzero>(m, AtenNonzero);
  ImplStable<OpName::kNonzeroOut>(m, AtenNonzeroOut);
  ImplStable<OpName::kNormalFloatTensor>(m, AtenNormalFloatTensor);
  ImplStable<OpName::kNormalFloatTensorOut>(m, AtenNormalFloatTensorOut);
  ImplStable<OpName::kNormalTensorFloat>(m, AtenNormalTensorFloat);
  ImplStable<OpName::kNormalTensorFloatOut>(m, AtenNormalTensorFloatOut);
  ImplStable<OpName::kNormalTensorTensor>(m, AtenNormalTensorTensor);
  ImplStable<OpName::kNormalTensorTensorOut>(m, AtenNormalTensorTensorOut);
  ImplStable<OpName::kNormal_>(m, AtenNormal_);
  ImplStable<OpName::kPdistBackward>(m, AtenPdistBackward);
  ImplStable<OpName::kPdistForward>(m, AtenPdistForward);
  ImplStable<OpName::kPolarOut>(m, AtenPolarOut);
  ImplStable<OpName::kPolygammaOut>(m, AtenPolygammaOut);
  ImplStable<OpName::kPowScalarOut>(m, AtenPowScalarOut);
  ImplStable<OpName::kPowTensorScalarOut>(m, AtenPowTensorScalarOut);
  ImplStable<OpName::kPowTensorTensorOut>(m, AtenPowTensorTensorOut);
  ImplStable<OpName::kProd>(m, AtenProd);
  ImplStable<OpName::kProdDimOut>(m, AtenProdDimOut);
  ImplStable<OpName::kPut_>(m, AtenPut_);
  ImplStable<OpName::kRandom_>(m, AtenRandom_);
  ImplStable<OpName::kRandom_From>(m, AtenRandom_From);
  ImplStable<OpName::kRandom_To>(m, AtenRandom_To);
  ImplStable<OpName::kRandpermGeneratorOut>(m, AtenRandpermGeneratorOut);
  ImplStable<OpName::kReciprocalOut>(m, AtenReciprocalOut);
  ImplStable<OpName::kReflectionPad1dBackwardGradInput>(
      m, AtenReflectionPad1dBackwardGradInput);
  ImplStable<OpName::kReflectionPad1dOut>(m, AtenReflectionPad1dOut);
  ImplStable<OpName::kReflectionPad2d>(m, AtenReflectionPad2d);
  ImplStable<OpName::kReflectionPad2dBackward>(m, AtenReflectionPad2dBackward);
  ImplStable<OpName::kReflectionPad2dBackwardGradInput>(
      m, AtenReflectionPad2dBackwardGradInput);
  ImplStable<OpName::kReflectionPad2dOut>(m, AtenReflectionPad2dOut);
  ImplStable<OpName::kReflectionPad3dBackwardGradInput>(
      m, AtenReflectionPad3dBackwardGradInput);
  ImplStable<OpName::kReflectionPad3dOut>(m, AtenReflectionPad3dOut);
  ImplStable<OpName::kRelu>(m, AtenRelu);
  ImplStable<OpName::kRelu_>(m, AtenRelu_);
  ImplStable<OpName::kRemainderScalarTensor>(m, AtenRemainderScalarTensor);
  ImplStable<OpName::kRemainderTensorOut>(m, AtenRemainderTensorOut);
  ImplStable<OpName::kRepeatInterleaveSelfTensor>(m, AtenRepeatInterleave);
  ImplStable<OpName::kReplicationPad1dBackwardGradInput>(
      m, AtenReplicationPad1dBackwardGradInput);
  ImplStable<OpName::kReplicationPad1dOut>(m, AtenReplicationPad1dOut);
  ImplStable<OpName::kReplicationPad2dBackward>(m,
                                                AtenReplicationPad2dBackward);
  ImplStable<OpName::kReplicationPad2dBackwardGradInput>(
      m, AtenReplicationPad2dBackwardGradInput);
  ImplStable<OpName::kReplicationPad2dOut>(m, AtenReplicationPad2dOut);
  ImplStable<OpName::kReplicationPad3dBackward>(m,
                                                AtenReplicationPad3dBackward);
  ImplStable<OpName::kReplicationPad3dBackwardGradInput>(
      m, AtenReplicationPad3dBackwardGradInput);
  ImplStable<OpName::kReplicationPad3dOut>(m, AtenReplicationPad3dOut);
  ImplStable<OpName::kReshapeAlias>(m, AtenReshapeAlias);
  ImplStable<OpName::kResize_>(m, AtenResize_);
  ImplStable<OpName::kRoll>(m, AtenRoll);
  ImplStable<OpName::kRoundDecimalsOut>(m, AtenRoundDecimalsOut);
  ImplStable<OpName::kRoundOut>(m, AtenRoundOut);
  ImplStable<OpName::kRshiftScalar>(m, AtenRshiftScalar);
  ImplStable<OpName::kRshiftTensor>(m, AtenRshiftTensor);
  ImplStable<OpName::kRsqrtOut>(m, AtenRsqrtOut);
  ImplStable<OpName::kRsubTensor>(m, AtenRsubTensor);
  ImplStable<OpName::kScaledDotProductEfficientAttention>(
      m, AtenScaledDotProductEfficientAttention);
  ImplStable<OpName::kScaledDotProductEfficientAttentionBackward>(
      m, AtenScaledDotProductEfficientAttentionBackward);
  ImplStable<OpName::kScaledDotProductFlashAttention>(
      m, AtenScaledDotProductFlashAttention);
  ImplStable<OpName::kScaledDotProductFlashAttentionBackward>(
      m, AtenScaledDotProductFlashAttentionBackward);
  ImplStable<OpName::kScaledDotProductFusedAttentionOverrideable>(
      m, AtenScaledDotProductFusedAttentionOverrideable);
  ImplStable<OpName::kScaledDotProductFusedAttentionOverrideableBackward>(
      m, AtenScaledDotProductFusedAttentionOverrideableBackward);
  ImplStable<OpName::kScaledMm>(m, AtenScaledMm);
  ImplStable<OpName::kScaledMmOut>(m, AtenScaledMmOut);
  ImplStable<OpName::kScatterAddOut>(m, AtenScatterAddOut);
  ImplStable<OpName::kScatterReduceOut>(m, AtenScatterReduceOut);
  ImplStable<OpName::kScatterReduceTwoOut>(m, AtenScatterReduceTwoOut);
  ImplStable<OpName::kScatterSrcOut>(m, AtenScatterSrcOut);
  ImplStable<OpName::kScatterValueOut>(m, AtenScatterValueOut);
  ImplStable<OpName::kScatterValueReduceOut>(m, AtenScatterValueReduceOut);
  ImplStable<OpName::kSearchsortedScalar>(m, AtenSearchsortedScalar);
  ImplStable<OpName::kSearchsortedScalarOut>(m, AtenSearchsortedScalarOut);
  ImplStable<OpName::kSearchsortedTensor>(m, AtenSearchsortedTensor);
  ImplStable<OpName::kSearchsortedTensorOut>(m, AtenSearchsortedTensorOut);
  ImplStable<OpName::kSet_>(m, AtenSet_);
  ImplStable<OpName::kSet_SourceStorage>(m, AtenSet_SourceStorage);
  ImplStable<OpName::kSet_SourceStorageOffset>(m, AtenSet_SourceStorageOffset);
  ImplStable<OpName::kSet_SourceTensor>(m, AtenSet_SourceTensor);
  ImplStable<OpName::kSgnOut>(m, AtenSgnOut);
  ImplStable<OpName::kSigmoidBackwardGradInput>(m,
                                                AtenSigmoidBackwardGradInput);
  ImplStable<OpName::kSigmoidOut>(m, AtenSigmoidOut);
  ImplStable<OpName::kSignOut>(m, AtenSignOut);
  ImplStable<OpName::kSignbitOut>(m, AtenSignbitOut);
  ImplStable<OpName::kSiluOut>(m, AtenSiluOut);
  ImplStable<OpName::kSinOut>(m, AtenSinOut);
  ImplStable<OpName::kSinhOut>(m, AtenSinhOut);
  ImplStable<OpName::kSoftmaxBackwardDataOut>(m, AtenSoftmaxBackwardDataOut);
  ImplStable<OpName::kSoftmaxOut>(m, AtenSoftmaxOut);
  ImplStable<OpName::kSoftplusBackwardGradInput>(m,
                                                 AtenSoftplusBackwardGradInput);
  ImplStable<OpName::kSoftplusOut>(m, AtenSoftplusOut);
  ImplStable<OpName::kSortValuesStable>(m, AtenSortValuesStable);
  ImplStable<OpName::kSplitWithSizesCopyOut>(m, AtenSplitWithSizesCopyOut);
  ImplStable<OpName::kSqrtOut>(m, AtenSqrtOut);
  ImplStable<OpName::kSubOut>(m, AtenSubOut);
  ImplStable<OpName::kSumIntListOut>(m, AtenSumIntListOut);
  ImplStable<OpName::kTake>(m, AtenTake);
  ImplStable<OpName::kTakeOut>(m, AtenTakeOut);
  ImplStable<OpName::kTanOut>(m, AtenTanOut);
  ImplStable<OpName::kTanhBackwardGradInput>(m, AtenTanhBackwardGradInput);
  ImplStable<OpName::kTanhOut>(m, AtenTanhOut);
  ImplStable<OpName::kThnnFusedGruCell>(m, AtenThnnFusedGruCell);
  ImplStable<OpName::kThnnFusedGruCellBackward>(m,
                                                AtenThnnFusedGruCellBackward);
  ImplStable<OpName::kThnnFusedGruCellOut>(m, AtenThnnFusedGruCellOut);
  ImplStable<OpName::kThnnFusedLstmCell>(m, AtenThnnFusedLstmCell);
  ImplStable<OpName::kThnnFusedLstmCellBackwardImpl>(
      m, AtenThnnFusedLstmCellBackwardImpl);
  ImplStable<OpName::kThnnFusedLstmCellOut>(m, AtenThnnFusedLstmCellOut);
  ImplStable<OpName::kThresholdBackwardGradInput>(
      m, AtenThresholdBackwardGradInput);
  ImplStable<OpName::kThresholdOut>(m, AtenThresholdOut);
  ImplStable<OpName::kToCopy>(m, AtenToCopy);
  ImplStable<OpName::kTopkValues>(m, AtenTopKValues);
  ImplStable<OpName::kTrilIndices>(m, AtenTrilIndices);
  ImplStable<OpName::kTrilOut>(m, AtenTrilOut);
  ImplStable<OpName::kTriuOut>(m, AtenTriuOut);
  ImplStable<OpName::kTruncOut>(m, AtenTruncOut);
  ImplStable<OpName::kUnfold>(m, AtenUnfold);
  ImplStable<OpName::kUnfoldBackward>(m, AtenUnfoldBackward);
  ImplStable<OpName::kUniform_>(m, AtenUniform_);
  ImplStable<OpName::kUnique2>(m, AtenUnique2);
  ImplStable<OpName::kUpsampleBicubic2dBackwardGradInput>(
      m, AtenUpsampleBicubic2dBackwardGradInput);
  ImplStable<OpName::kUpsampleBicubic2dOut>(m, AtenUpsampleBicubic2dOut);
  ImplStable<OpName::kUpsampleBilinear2dBackwardGradInput>(
      m, AtenUpsampleBilinear2dBackwardGradInput);
  ImplStable<OpName::kUpsampleBilinear2dOut>(m, AtenUpsampleBilinear2dOut);
  ImplStable<OpName::kUpsampleNearest1dBackwardGradInput>(
      m, AtenUpsampleNearest1dBackwardGradInput);
  ImplStable<OpName::kUpsampleNearest1dOut>(m, AtenUpsampleNearest1dOut);
  ImplStable<OpName::kUpsampleNearest2dBackwardGradInput>(
      m, AtenUpsampleNearest2dBackwardGradInput);
  ImplStable<OpName::kUpsampleNearest2dOut>(m, AtenUpsampleNearest2dOut);
  ImplStable<OpName::kUpsampleNearest3dBackwardGradInput>(
      m, AtenUpsampleNearest3dBackwardGradInput);
  ImplStable<OpName::kUpsampleNearest3dOut>(m, AtenUpsampleNearest3dOut);
  ImplStable<OpName::kUpsampleNearestExact1dBackwardGradInput>(
      m, AtenUpsampleNearestExact1dBackwardGradInput);
  ImplStable<OpName::kUpsampleNearestExact1dOut>(m,
                                                 AtenUpsampleNearestExact1dOut);
  ImplStable<OpName::kUpsampleNearestExact2dBackwardGradInput>(
      m, AtenUpsampleNearestExact2dBackwardGradInput);
  ImplStable<OpName::kUpsampleNearestExact2dOut>(m,
                                                 AtenUpsampleNearestExact2dOut);
  ImplStable<OpName::kUpsampleNearestExact3dBackwardGradInput>(
      m, AtenUpsampleNearestExact3dBackwardGradInput);
  ImplStable<OpName::kUpsampleNearestExact3dOut>(m,
                                                 AtenUpsampleNearestExact3dOut);
  ImplStable<OpName::kVarCorrection>(m, AtenVar);
  ImplStable<OpName::kVarCorrectionOut>(m, AtenVarOut);
  ImplStable<OpName::kVarMeanCorrection>(m, AtenVarMeanCorrection);
  ImplStable<OpName::kVdot>(m, AtenVdot);
  ImplStable<OpName::kView>(m, AtenView);
  ImplStable<OpName::kViewAsComplex>(m, AtenViewAsComplex);
  ImplStable<OpName::kViewAsReal>(m, AtenViewAsReal);
  ImplStable<OpName::kWeightNormInterface>(m, AtenWeightNormInterface);
  ImplStable<OpName::kWhereSelf>(m, AtenWhereSelf);
  ImplStable<OpName::kWhereSelfOut>(m, AtenWhereSelfOut);
  ImplStable<OpName::kXlogyOutTensor>(m, AtenXlogyOutTensor);
  ImplStable<OpName::kZero_>(m, AtenZero_);
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
    TT_CHECK_THROW(false, error::kPythonNotImplementedError)
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

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  // TODO(b/513607161): remove CtcLossPublic overrides once upstream PyTorch bug
  // is fixed.
  // Because we override the public ctc_loss in PrivateUse1, we bypass its
  // default decomposition into _ctc_loss. Since the public ctc_loss is normally
  // a composite op that relies on this decomposition for autograd, bypassing
  // it means PyTorch's autograd no longer knows how to differentiate it.
  // Therefore, we must explicitly provide this autograd override to map
  // ctc_loss to our custom forward/backward autograd implementation.
  ImplStable<OpName::kCtcLossPublic>(m, AtenCtcLossPublicAutograd);
  ImplStable<OpName::kCtcLossPublicTensor>(m, AtenCtcLossPublicTensorAutograd);
  // MaxPool is by default a CompositeImplicitAutograd op. The decomposition
  // replaces it with MaxPool2dWithIndices (even when return_indices=False).
  // Intuitively, the with-indices variant would be helpful for training,
  // because the indices are needed to compute the gradient. This is NOT the
  // case on tpu however, re-computing the indices via SelectAndScatter is much
  // faster.
  //
  // Therefore the optimal lowering of max_pool2d uses the non-indices variant
  // for both forward and backward, the latter leveraging SelectAndScatter.
  // SelectAndScatter does not support dilations however, so we fallback to the
  // indices variant and preserve the default composite behavior in this case.
  ImplStable<OpName::kMaxPool2d>(m, AtenMaxPool2d);
}

// Signatures of custom ops in torch.ops.tpu. Their C++ bindings are registered
// in TORCH_LIBRARY_IMPL(tpu, PrivateUse1, m) below.
TORCH_LIBRARY(tpu, m) {
  m.def(
      "max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, "
      "int[] dilation, bool ceil_mode) -> Tensor");
  m.def(
      "max_pool2d_backward(Tensor grad_output, Tensor self, int[] kernel_size, "
      "int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor");
  m.def("ragged_dot(Tensor lhs, Tensor rhs, Tensor group_sizes) -> Tensor");
  m.def(
      "ragged_dot.out(Tensor lhs, Tensor rhs, Tensor group_sizes, *, "
      "Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "ragged_all_to_all(Tensor operand, Tensor output, Tensor "
      "input_offsets, Tensor send_sizes, Tensor output_offsets, Tensor "
      "recv_sizes, str process_group_name) -> "
      "Tensor");
  m.def(
      "ragged_all_to_all.out(Tensor operand, Tensor output, Tensor "
      "input_offsets, Tensor send_sizes, Tensor output_offsets, Tensor "
      "recv_sizes, str process_group_name, *, "
      "Tensor(a!) out) "
      "-> Tensor(a!)");
  m.def("optimization_barrier(Tensor[] inputs) -> Tensor[]");
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

  // Broadcasts the input tensor to a dynamic output shape.
  //
  // This operator is used in torch.compile() mode to support broadcasting
  // tensors with bounded dynamism on TPU. It lowers to
  // stablehlo.broadcast_in_dim followed by stablehlo.set_dimension_size for
  // dynamic dimensions.
  //
  // Args:
  //   input: The input tensor to broadcast.
  //   shape: List of 0-D (scalar) int32 tensors containing the runtime sizes of
  //     the output dimensions.
  //   broadcast_dims: Specifies which dimensions of the output correspond to
  //     dimensions of the input.
  //   static_shape: The static upper bound for the output shape. Used for
  //     static allocation.
  //   is_dynamic: List of booleans indicating which output dimensions are
  //     dynamic.
  //
  // Returns:
  //   The broadcasted tensor, bounded to its dynamic shape.
  m.def(
      "dynamic_broadcast(Tensor input, Tensor[] shape, int[] broadcast_dims, "
      "int[] static_shape, bool[] is_dynamic) -> Tensor");

  // This op is a torch_tpu custom op for use in torch.compile() mode to handle
  // dynamic reshape operations on TPU. It reshapes the input tensor to the
  // specified shape.
  // Args:
  //   input: The input tensor to reshape.
  //   shape: List of 0-D (scalar) int32 tensors containing the runtime sizes of
  //     the output dimensions.
  //   static_shape: The static upper bound for the output shape. Used for
  //     static allocation.
  //   is_dynamic: List of booleans indicating which output dimensions are
  //     dynamic.
  // Returns:
  //   The reshaped tensor, bounded to its dynamic shape.
  m.def(
      "dynamic_reshape(Tensor input, Tensor[] shape, int[] static_shape, "
      "bool[] is_dynamic) -> Tensor");

  // Experimental P2P communication ops for ProcessGroupTpu.
  // Isolated from the public torch.distributed API to safely prototype new
  // behaviors.
  m.def("experimental_send(Tensor[] tensors, int dst, int tag) -> Any");
  m.def("experimental_recv(Tensor[] tensors, int src, int tag) -> Any");
  m.def(
      "sparse_dense_matmul(Tensor row_pointers, Tensor embedding_ids, Tensor "
      "sample_ids, Tensor gains, Tensor embedding_table, int "
      "device_batch_size, int max_ids_per_partition, int "
      "max_unique_ids_per_partition) -> Tensor");
  m.def(
      "sparse_dense_matmul_grad_with_adagrad(Tensor row_pointers, Tensor "
      "embedding_ids, Tensor sample_ids, Tensor gains, Tensor embedding_table, "
      "Tensor accumulator, Tensor activations_grad, Tensor learning_rate, "
      "float epsilon, int "
      "device_batch_size, int max_ids_per_partition, int "
      "max_unique_ids_per_partition, str computation_name) -> (Tensor, "
      "Tensor)");
  m.def(
      "sparse_dense_matmul_grad_with_sgd(Tensor row_pointers, Tensor "
      "embedding_ids, Tensor sample_ids, Tensor gains, Tensor embedding_table, "
      "Tensor activations_grad, Tensor learning_rate, int device_batch_size, "
      "int max_ids_per_partition, int max_unique_ids_per_partition, str "
      "computation_name) -> Tensor");
  m.def(
      "sparse_dense_matmul_grad_with_adam(Tensor row_pointers, Tensor "
      "embedding_ids, Tensor sample_ids, Tensor gains, Tensor embedding_table, "
      "Tensor momentum, Tensor velocity, Tensor activations_grad, Tensor "
      "alpha_t, float beta_1, float beta_2, float epsilon, int "
      "device_batch_size, int max_ids_per_partition, int "
      "max_unique_ids_per_partition, str computation_name) -> (Tensor, Tensor, "
      "Tensor)");
}

TORCH_LIBRARY_IMPL(tpu, Meta, m) {
  ImplStable<OpName::kMaxPool2d>(
      m, [](const at::Tensor& self, at::IntArrayRef kernel_size,
            at::IntArrayRef stride, at::IntArrayRef padding,
            at::IntArrayRef dilation, bool ceil_mode) {
        return at::empty(GetMaxPoolOutputSize(self.sizes(), kernel_size, stride,
                                              padding, dilation, ceil_mode, 2),
                         self.options());
      });
  ImplStable<OpName::kMaxPool2dBackward>(
      m, [](const at::Tensor& grad_output, const at::Tensor& self,
            at::IntArrayRef kernel_size, at::IntArrayRef stride,
            at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
        return at::empty(self.sizes(), self.options());
      });
  ImplStable<OpName::kSparseDenseMatmul>(
      m,
      [](const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
         const at::Tensor& sample_ids, const at::Tensor& gains,
         const at::Tensor& embedding_table, int64_t device_batch_size,
         int64_t max_ids_per_partition,
         int64_t max_unique_ids_per_partition) -> at::Tensor {
        TT_CHECK_THROW(embedding_table.dim() == 2, error::kInvalidArgument)
            << "embedding_table must be 2D";
        return at::empty({device_batch_size, embedding_table.size(1)},
                         embedding_table.options());
      });
  ImplStable<OpName::kSparseDenseMatmulGradWithSgd>(
      m,
      [](const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
         const at::Tensor& sample_ids, const at::Tensor& gains,
         const at::Tensor& embedding_table, const at::Tensor& activations_grad,
         const at::Tensor& learning_rate, int64_t device_batch_size,
         int64_t max_ids_per_partition, int64_t max_unique_ids_per_partition,
         std::string_view computation_name) {
        return at::empty_like(embedding_table);
      });
  ImplExperimental<OpName::kSparseDenseMatmulGradWithAdagrad>(
      m,
      +[](const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
          const at::Tensor& sample_ids, const at::Tensor& gains,
          const at::Tensor& embedding_table, const at::Tensor& accumulator,
          const at::Tensor& activations_grad, const at::Tensor& learning_rate,
          double epsilon, int64_t device_batch_size,
          int64_t max_ids_per_partition, int64_t max_unique_ids_per_partition,
          std::string_view computation_name) {
        return std::make_tuple(at::empty_like(embedding_table),
                               at::empty_like(accumulator));
      });
  ImplExperimental<OpName::kSparseDenseMatmulGradWithAdam>(
      m,
      +[](const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
          const at::Tensor& sample_ids, const at::Tensor& gains,
          const at::Tensor& embedding_table, const at::Tensor& momentum,
          const at::Tensor& velocity, const at::Tensor& activations_grad,
          const at::Tensor& alpha_t, double beta_1, double beta_2,
          double epsilon, int64_t device_batch_size,
          int64_t max_ids_per_partition, int64_t max_unique_ids_per_partition,
          std::string_view computation_name) {
        return std::make_tuple(at::empty_like(embedding_table),
                               at::empty_like(momentum),
                               at::empty_like(velocity));
      });
  ImplStable<OpName::kRaggedDot>(
      m, +[](const at::Tensor& lhs, const at::Tensor& rhs,
             const at::Tensor& group_sizes) {
        return at::empty({lhs.size(0), rhs.size(2)},
                         lhs.options().dtype(at::result_type(lhs, rhs)));
      });
  ImplStable<OpName::kRaggedDotOut>(
      m,
      +[](const at::Tensor& lhs, const at::Tensor& rhs,
          const at::Tensor& group_sizes, at::Tensor& out) -> at::Tensor& {
        at::native::resize_output(out, {lhs.size(0), rhs.size(2)});
        return out;
      });
  ImplStable<OpName::kRaggedAllToAll>(
      m, +[](const at::Tensor& operand, const at::Tensor& output,
             const at::Tensor& input_offsets, const at::Tensor& send_sizes,
             const at::Tensor& output_offsets, const at::Tensor& recv_sizes,
             std::string_view process_group_name) {
        return at::empty_like(output);
      });
  ImplStable<OpName::kRaggedAllToAllOut>(
      m,
      +[](const at::Tensor& operand, const at::Tensor& output,
          const at::Tensor& input_offsets, const at::Tensor& send_sizes,
          const at::Tensor& output_offsets, const at::Tensor& recv_sizes,
          std::string_view process_group_name, at::Tensor& out) -> at::Tensor& {
        at::native::resize_output(out, output.sizes());
        return out;
      });
}

TORCH_LIBRARY_IMPL(tpu, PrivateUse1, m) {
  ImplExperimental<OpName::kDistributedExperimentalSend>(
      m, TorchTpuExperimentalSend);
  ImplExperimental<OpName::kDistributedExperimentalRecv>(
      m, TorchTpuExperimentalRecv);
  ImplExperimental<OpName::kMaxPool2d>(m, TpuMaxPool2d);
  ImplExperimental<OpName::kMaxPool2dBackward>(m, TpuMaxPool2dBackward);
  ImplExperimental<OpName::kRaggedDot>(m, AtenRaggedDot);
  ImplExperimental<OpName::kRaggedDotOut>(m, AtenRaggedDotOut);
  ImplExperimental<OpName::kRaggedAllToAll>(m, AtenRaggedAllToAll);
  ImplExperimental<OpName::kRaggedAllToAllOut>(m, AtenRaggedAllToAllOut);
  ImplExperimental<OpName::kTorchTpuOptimizationBarrier>(
      m, TorchTpuOptimizationBarrier);
  ImplExperimental<OpName::kSetDimensionLogicalSize>(m,
                                                     SetDimensionLogicalSize);
  ImplExperimental<OpName::kDynamicArange>(m, DynamicArange);
  ImplExperimental<OpName::kDynamicBroadcast>(m, DynamicBroadcast);
  ImplExperimental<OpName::kDynamicReshape>(m, DynamicReshape);
  ImplExperimental<OpName::kSparseDenseMatmul>(m, AtenSparseDenseMatmul);
  ImplExperimental<OpName::kSparseDenseMatmulGradWithSgd>(
      m, AtenSparseDenseMatmulGradWithSgd);
  ImplExperimental<OpName::kSparseDenseMatmulGradWithAdagrad>(
      m, AtenSparseDenseMatmulGradWithAdagrad);
  ImplExperimental<OpName::kSparseDenseMatmulGradWithAdam>(
      m, AtenSparseDenseMatmulGradWithAdam);
}

TORCH_LIBRARY_IMPL(tpu, CPU, m) {
  ImplExperimental<OpName::kRaggedDot>(m, AtenRaggedDot);
  ImplExperimental<OpName::kRaggedDotOut>(m, AtenRaggedDotOut);
  ImplExperimental<OpName::kRaggedAllToAll>(m, AtenRaggedAllToAll);
  ImplExperimental<OpName::kRaggedAllToAllOut>(m, AtenRaggedAllToAllOut);
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
