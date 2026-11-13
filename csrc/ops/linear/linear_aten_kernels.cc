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

#include "csrc/ops/linear/linear_aten_kernels.h"

#include <array>
#include <cstdint>
#include <memory>  // IWYU pragma: keep
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/linear.h"
#include "ATen/ops/linear_native.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/DimVector.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/const_tensor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fake_tensor_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/VariableTypeUtils.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/generated/Functions.h"  // IWYU pragma: keep
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {
namespace {

// Builds a StableHLO `dot_general` operation with the given contracting
// dimensions and precision config. Casts 64-bit integer inputs to `f64` before
// the dot product and back to `i64` afterwards to work around XLA TPU backend
// precision/performance limitations on `i64` dot products.
mlir::MlirOp BuildDotGeneral(mlir::MlirOp lhs, mlir::MlirOp rhs,
                             absl::Span<const int64_t> lhs_contracting,
                             absl::Span<const int64_t> rhs_contracting,
                             mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  const bool is_any_i64 = lhs_type.getElementType().isInteger(64) ||
                          rhs_type.getElementType().isInteger(64);
  if (is_any_i64) {
    lhs = mlir::stablehlo::ConvertElementType(lhs, mlir::ElementType::F64);
    rhs = mlir::stablehlo::ConvertElementType(rhs, mlir::ElementType::F64);
  }

  auto dot_dimension_numbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &lhs.getContext(),
      /*lhs_batching_dimensions=*/{},
      /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/AsArrayRef(lhs_contracting),
      /*rhs_contracting_dimensions=*/AsArrayRef(rhs_contracting));
  const mlir::stablehlo::Precision precisions[2] = {precision, precision};
  auto precision_config =
      mlir::stablehlo::PrecisionConfigAttr::get(&lhs.getContext(), precisions);

  mlir::MlirOp dot_result = mlir::stablehlo::DotGeneral(
      lhs, rhs, dot_dimension_numbers, precision_config);
  if (is_any_i64) {
    return mlir::stablehlo::ConvertElementType(dot_result,
                                               mlir::ElementType::I64);
  }
  return dot_result;
}

// Builds the StableHLO graph for `aten::linear`: y = input @ weight^T + bias.
// Supports arbitrary input rank >= 1 and 1D or 2D weight without intermediate
// reshapes by contracting `input`'s last dimension with `weight`'s last
// dimension directly via `stablehlo.dot_general`.
absl::StatusOr<mlir::MlirOp> BuildLinearShlo(
    mlir::MlirOp input, mlir::MlirOp weight, std::optional<mlir::MlirOp> bias,
    mlir::stablehlo::Precision precision) {
  const int64_t input_rank = GetTensorTypeOrDie(input).getRank();
  const int64_t weight_rank = GetTensorTypeOrDie(weight).getRank();
  const int64_t lhs_contracting[1] = {input_rank - 1};
  const int64_t rhs_contracting[1] = {weight_rank - 1};

  mlir::MlirOp dot_result = BuildDotGeneral(input, weight, lhs_contracting,
                                            rhs_contracting, precision);
  if (bias.has_value()) {
    TT_ASSIGN_OR_RETURN(auto broadcasted,
                        ApplyBroadcastIfNeeded(dot_result, *bias));
    return mlir::stablehlo::Add(broadcasted[0], broadcasted[1]);
  }
  return dot_result;
}

absl::StatusOr<mlir::MlirOp> MaybeConjPhysicalShlo(mlir::MlirOp op) {
  if (!IsComplexType(GetTensorTypeOrDie(op))) {
    return op;
  }
  return BuildConjPhysicalShlo(op);
}

// Builds the StableHLO graph for `grad_input = grad_output @ conj(weight)`.
absl::StatusOr<mlir::MlirOp> BuildGradInputShlo(
    mlir::MlirOp grad_output, mlir::MlirOp weight,
    mlir::stablehlo::Precision precision) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp w_for_grad_input,
                      MaybeConjPhysicalShlo(weight));
  if (GetTensorTypeOrDie(weight).getRank() == 2) {
    const int64_t lhs_contracting[1] = {
        GetTensorTypeOrDie(grad_output).getRank() - 1};
    const int64_t rhs_contracting[1] = {0};
    return BuildDotGeneral(grad_output, w_for_grad_input, lhs_contracting,
                           rhs_contracting, precision);
  }
  return BuildDotGeneral(grad_output, w_for_grad_input, {}, {}, precision);
}

// Builds the StableHLO graph for `grad_weight = grad_output^T @ conj(input)`.
absl::StatusOr<mlir::MlirOp> BuildGradWeightShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input,
    mlir::stablehlo::Precision precision) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp x_for_grad_weight,
                      MaybeConjPhysicalShlo(input));
  Dimensions contracting_dims(GetTensorTypeOrDie(input).getRank() - 1);
  std::iota(contracting_dims.begin(), contracting_dims.end(), 0);
  return BuildDotGeneral(grad_output, x_for_grad_weight, contracting_dims,
                         contracting_dims, precision);
}

// Builds the StableHLO graph for
// `grad_bias = sum(grad_output, dims=[0..input_rank-2])`.
absl::StatusOr<mlir::MlirOp> BuildGradBiasShlo(mlir::MlirOp grad_output,
                                               int64_t input_rank) {
  if (input_rank == 1) {
    return grad_output;
  }
  Dimensions reduction_dims(input_rank - 1);
  std::iota(reduction_dims.begin(), reduction_dims.end(), 0);
  return BuildSumShlo(grad_output, reduction_dims, ReductionMode::kDropDims);
}

// Validates the input, weight, and optional bias tensors for `aten::linear` and
// returns the output shape `[..., Out]` (or `[...]` when `weight` is 1D).
// Takes `ConstTensor`s so that the validation does not specialize the symbolic
// shapes of the FakeTensors that torch.compile traces with.
absl::StatusOr<c10::SymDimVector> ValidateLinearInputs(
    const ConstTensor& input, const ConstTensor& weight,
    const std::optional<ConstTensor>& bias) {
  TT_ASSIGN_OR_RETURN(const int64_t input_dim, input.dim());
  TT_ASSIGN_OR_RETURN(const int64_t weight_dim, weight.dim());
  TT_RET_CHECK(input_dim >= 1 && weight_dim >= 1, error::kInvalidArgument)
      << "expected both arguments to linear to be at least 1D, got "
      << input_dim << "D and " << weight_dim << "D";
  TT_RET_CHECK(weight_dim <= 2, error::kInvalidArgument)
      << "expected the weight tensor to be at most 2D, got " << weight_dim
      << "D";
  TT_RET_CHECK(!IsBool(input.scalar_type()), error::kPythonNotImplementedError)
      << "not implemented for " << ToString(input.scalar_type());
  TT_RET_CHECK(input.scalar_type() == weight.scalar_type(),
               error::kInvalidArgument)
      << "expected the input and weight tensors to have the same dtype, got "
      << ToString(input.scalar_type()) << " and "
      << ToString(weight.scalar_type());
  if (bias.has_value() && bias->defined()) {
    TT_RET_CHECK(input.scalar_type() == bias->scalar_type(),
                 error::kInvalidArgument)
        << "expected the input and bias tensors to have the same dtype, got "
        << ToString(input.scalar_type()) << " and "
        << ToString(bias->scalar_type());
  }
  TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef input_sizes, input.sym_sizes());
  TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef weight_sizes,
                      weight.sym_sizes());
  // Like upstream, this guards only on the contracting dimensions being equal.
  TT_RET_CHECK(input_sizes.back() == weight_sizes.back(),
               error::kInvalidArgument)
      << "expected the input and weight tensors to have the same last "
         "dimension size, got "
      << input_sizes.back() << " and " << weight_sizes.back() << " for shapes "
      << ToString(input_sizes) << " and " << ToString(weight_sizes);

  c10::SymDimVector output_dims(input_sizes.begin(), input_sizes.end() - 1);
  if (weight_dim == 2) {
    output_dims.push_back(weight_sizes[0]);
  }

  if (bias.has_value() && bias->defined()) {
    TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef bias_sizes,
                        bias->sym_sizes());
    TT_RET_CHECK(at::is_expandable_to(bias_sizes, output_dims),
                 error::kInvalidArgument)
        << "expected the bias tensor to be broadcastable to the output shape "
        << ToString(c10::SymIntArrayRef(output_dims)) << ", got shape "
        << ToString(bias_sizes);
  }

  return output_dims;
}

// Validates the self (input), grad_output, and weight tensors for
// `aten::linear_backward`.
absl::Status ValidateLinearBackwardInputs(const at::Tensor& self,
                                          const at::Tensor& grad_output,
                                          const at::Tensor& weight) {
  TT_ASSIGN_OR_RETURN(const c10::SymDimVector sym_expected_output_dims,
                      ValidateLinearInputs(self, weight, std::nullopt));
  TT_ASSIGN_OR_RETURN(const Dimensions expected_output_dims,
                      ConcreteSizeToDimensions(sym_expected_output_dims));
  TT_RET_CHECK(grad_output.scalar_type() == self.scalar_type(),
               error::kInvalidArgument)
      << "expected the grad_output tensor to have dtype "
      << ToString(self.scalar_type()) << ", got "
      << ToString(grad_output.scalar_type());
  TT_RET_CHECK(grad_output.sizes() == at::IntArrayRef(expected_output_dims),
               error::kInvalidArgument)
      << "expected the grad_output tensor to have shape "
      << ToString(expected_output_dims) << ", got "
      << ToString(grad_output.sizes());
  return absl::OkStatus();
}

// Shared dispatcher helper for `Linear` and `LinearOut`. Dispatches a 3-input
// op when `bias` is defined, or a 2-input op otherwise.
template <typename ReturnType, typename Dispatch2Fn, typename Dispatch3Fn>
ReturnType DispatchLinearImpl(const at::Tensor& input, const at::Tensor& weight,
                              const std::optional<at::Tensor>& bias,
                              Dimensions output_dims,
                              OpParamCacheKeys param_keys,
                              Dispatch2Fn&& dispatch2_fn,
                              Dispatch3Fn&& dispatch3_fn) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype_mlir,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  const auto sanitized_bias = SanitizeOptionalTensor(bias);
  const bool has_bias = sanitized_bias.has_value();
  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  DispatchOpOptions<1> options{.out_dtype = output_dtype_mlir,
                               .out_dims = std::move(output_dims),
                               .op_param_cache_keys = std::move(param_keys)};

  if (has_bias) {
    auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [input_op, weight_op, bias_op] = inputs;
      return BuildLinearShlo(input_op, weight_op, bias_op, current_precision);
    };
    return dispatch3_fn(std::move(op_builder), std::move(options));
  } else {
    auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [input_op, weight_op] = inputs;
      return BuildLinearShlo(input_op, weight_op, std::nullopt,
                             current_precision);
    };
    return dispatch2_fn(std::move(op_builder), std::move(options));
  }
}

absl::StatusOr<DeviceBufferRef> Linear(const at::Tensor& input,
                                       const at::Tensor& weight,
                                       const std::optional<at::Tensor>& bias,
                                       OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const c10::SymDimVector sym_output_dims,
                      ValidateLinearInputs(input, weight, bias));
  TT_ASSIGN_OR_RETURN(Dimensions output_dims,
                      ConcreteSizeToDimensions(sym_output_dims));
  return DispatchLinearImpl<absl::StatusOr<DeviceBufferRef>>(
      input, weight, bias, std::move(output_dims), std::move(param_keys),
      [&]  // REF_CAPTURE_OK=called synchronously
      (auto op_builder, auto options) {
        return DispatchOp<2>(std::move(op_builder), {input, weight},
                             std::move(options));
      },
      [&]  // REF_CAPTURE_OK=called synchronously
      (auto op_builder, auto options) {
        return DispatchOp<3>(std::move(op_builder), {input, weight, *bias},
                             std::move(options));
      });
}

absl::Status LinearOut(const at::Tensor& input, const at::Tensor& weight,
                       const std::optional<at::Tensor>& bias, at::Tensor& out,
                       OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const c10::SymDimVector sym_output_dims,
                      ValidateLinearInputs(input, weight, bias));
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, input.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(Dimensions output_dims,
                      ConcreteSizeToDimensions(sym_output_dims));
  return DispatchLinearImpl<absl::Status>(
      input, weight, bias, std::move(output_dims), std::move(param_keys),
      [&]  // REF_CAPTURE_OK=called synchronously
      (auto op_builder, auto options) {
        return DispatchOpOut<2>(std::move(op_builder), {input, weight}, out,
                                std::move(options));
      },
      [&]  // REF_CAPTURE_OK=called synchronously
      (auto op_builder, auto options) {
        return DispatchOpOut<3>(std::move(op_builder), {input, weight, *bias},
                                out, std::move(options));
      });
}

// Computes the gradients of `aten::linear` enabled in `output_mask`, and
// returns undefined tensors for the disabled ones.
//
// Each gradient is dispatched as its own op. The outputs of one op share a
// `DeviceBufferList`, which frees its device memory only when all of them are
// dead, so a single op would keep the activation-sized `grad_input` allocated
// for as long as `grad_weight` or `grad_bias` lives in a parameter's `.grad`.
absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>> LinearBackward(
    const at::Tensor& self, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool, 3> output_mask,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateLinearBackwardInputs(self, grad_output, weight));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(grad_output.scalar_type()));
  const mlir::stablehlo::Precision precision = GetAndAddPrecisionTo(param_keys);

  at::Tensor grad_input;  // UNINITIALIZED_TENSOR_OK
  if (output_mask[0]) {
    OpParamCacheKeys grad_input_keys = param_keys.Clone();
    TT_RETURN_IF_ERROR(grad_input_keys.SetParam("gradient", "input"));
    auto op_builder = [precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [grad_output_op, weight_op] = inputs;
      return BuildGradInputShlo(grad_output_op, weight_op, precision);
    };
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef grad_input_buffer,
        DispatchOp<2>(std::move(op_builder), {grad_output, weight},
                      {.out_dtype = output_dtype,
                       .out_dims = self.sizes(),
                       .op_param_cache_keys = std::move(grad_input_keys)}));
    grad_input = MakeTensor(std::move(grad_input_buffer));
  }

  at::Tensor grad_weight;  // UNINITIALIZED_TENSOR_OK
  if (output_mask[1]) {
    OpParamCacheKeys grad_weight_keys = param_keys.Clone();
    TT_RETURN_IF_ERROR(grad_weight_keys.SetParam("gradient", "weight"));
    auto op_builder = [precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [grad_output_op, self_op] = inputs;
      return BuildGradWeightShlo(grad_output_op, self_op, precision);
    };
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef grad_weight_buffer,
        DispatchOp<2>(std::move(op_builder), {grad_output, self},
                      {.out_dtype = output_dtype,
                       .out_dims = weight.sizes(),
                       .op_param_cache_keys = std::move(grad_weight_keys)}));
    grad_weight = MakeTensor(std::move(grad_weight_buffer));
  }

  at::Tensor grad_bias;  // UNINITIALIZED_TENSOR_OK
  if (output_mask[2]) {
    OpParamCacheKeys grad_bias_keys = param_keys.Clone();
    TT_RETURN_IF_ERROR(grad_bias_keys.SetParam("gradient", "bias"));
    Dimensions bias_dims;
    if (weight.dim() == 2) {
      bias_dims = {weight.size(0)};
    }
    auto op_builder =
        [input_rank = self.dim()](
            mlir::MlirOp grad_output_op) -> absl::StatusOr<mlir::MlirOp> {
      return BuildGradBiasShlo(grad_output_op, input_rank);
    };
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef grad_bias_buffer,
        DispatchOp<1>(std::move(op_builder), grad_output,
                      {.out_dtype = output_dtype,
                       .out_dims = bias_dims,
                       .op_param_cache_keys = std::move(grad_bias_keys)}));
    grad_bias = MakeTensor(std::move(grad_bias_buffer));
  }

  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

// Returns true if `tensor` is a Meta tensor or a `FakeTensor` (whose
// underlying storage has `DeviceType::Meta`).
bool IsMetaOrFakeTensor(const at::Tensor& tensor) {
  return tensor.defined() &&
         (tensor.is_meta() ||
          (tensor.key_set().has(c10::DispatchKey::Python) &&
           tensor.has_storage() &&
           tensor.storage().device_type() == c10::DeviceType::Meta));
}

// Returns true if `tensor` is a Python wrapper tensor subclass (such as
// `DTensor`) not backed by Meta storage. Distinguishes wrapper subclasses
// (including storage-less subclasses) from `FakeTensor` (used by
// `torch.compile` and `torch.export`), whose underlying storage has
// `DeviceType::Meta`.
bool IsPythonWrapperSubclass(const at::Tensor& tensor) {
  return tensor.defined() && tensor.key_set().has(c10::DispatchKey::Python) &&
         (!tensor.has_storage() ||
          tensor.storage().device_type() != c10::DeviceType::Meta);
}

// Creates an autograd `Node` smart pointer (`c10::intrusive_ptr<T>` in PyTorch
// 2.13+ where `Node` inherits from `c10::intrusive_ptr_target`, or
// `std::shared_ptr<T>` with `torch::autograd::deleteNode` in earlier PyTorch
// versions used by OSS builds).
// TODO: b/585211199 - Simplify to `c10::make_intrusive<T>()` once minimum OSS
// PyTorch is >= 2.13.0, where `Node` is a `c10::intrusive_ptr_target`.
template <typename T>
auto MakeAutogradNode() {
  if constexpr (std::is_base_of_v<c10::intrusive_ptr_target, T>) {
    return c10::make_intrusive<T>();
  } else {
    return std::shared_ptr<T>(new T(), [](T* ptr) { deleteNode(ptr); });
  }
}

}  // namespace

at::Tensor AtenLinearAutograd(const at::Tensor& input, const at::Tensor& weight,
                              const std::optional<at::Tensor>& bias) {
  const auto sanitized_bias = SanitizeOptionalTensor(bias);
  const bool has_bias = sanitized_bias.has_value();
  const bool any_requires_grad =
      torch::autograd::compute_requires_grad(input, weight, bias);
  // Uses `sym_size()` so that FakeTensors with symbolic shapes are not
  // specialized.
  const bool is_non_standard_grad_bias =
      any_requires_grad && has_bias &&
      ((weight.dim() == 2 &&
        (sanitized_bias->dim() != 1 ||
         sanitized_bias->sym_size(0) != weight.sym_size(0))) ||
       (weight.dim() == 1 && sanitized_bias->dim() != 0)) &&
      torch::autograd::compute_requires_grad(*sanitized_bias) &&
      ValidateLinearInputs(input, weight, bias).ok();
  const bool is_meta_1d_weight =
      weight.dim() == 1 &&
      (IsMetaOrFakeTensor(input) || IsMetaOrFakeTensor(weight));
  if (is_non_standard_grad_bias || is_meta_1d_weight ||
      IsPythonWrapperSubclass(input) || IsPythonWrapperSubclass(weight) ||
      (has_bias && IsPythonWrapperSubclass(*sanitized_bias)) ||
      IsFunctorchActive() || torch::autograd::isFwGradDefined(input) ||
      torch::autograd::isFwGradDefined(weight) ||
      torch::autograd::isFwGradDefined(bias)) {
    if (weight.dim() == 1 && has_bias) {
      return at::native::linear(input, weight, std::nullopt) + *sanitized_bias;
    }
    return at::native::linear(input, weight, bias);
  }
  using LinearBackwardNode = torch::autograd::generated::LinearBackward0;
  decltype(MakeAutogradNode<LinearBackwardNode>()) grad_fn;
  if (any_requires_grad) {
    grad_fn = MakeAutogradNode<LinearBackwardNode>();
    grad_fn->set_next_edges(
        torch::autograd::collect_next_edges(input, weight, bias));
    grad_fn->input_ = torch::autograd::SavedVariable(input, false);
    grad_fn->weight_ = torch::autograd::SavedVariable(weight, false);
  }
  at::Tensor output = ([&]() {  // REF_CAPTURE_OK=called locally
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::linear(input, weight, bias);
  })();
  if (grad_fn) {
    torch::autograd::set_history(torch::autograd::flatten_tensor_args(output),
                                 grad_fn);
  }
  return output;
}

at::Tensor AtenLinear(const at::Tensor& input, const at::Tensor& weight,
                      const std::optional<at::Tensor>& bias) {
  TT_KERNEL(OpName::kLinear, param_keys, (input, weight, bias), {
    TT_ASSIGN_OR_THROW(auto result_buffer,
                       Linear(input, weight, bias, std::move(param_keys)));
    return MakeTensor(std::move(result_buffer));
  });
}

at::Tensor& AtenLinearOut(const at::Tensor& input, const at::Tensor& weight,
                          const std::optional<at::Tensor>& bias,
                          at::Tensor& out) {
  TT_KERNEL(OpName::kLinearOut, param_keys, (input, weight, bias, out), {
    TT_THROW_IF_ERROR(
        LinearOut(input, weight, bias, out, std::move(param_keys)));
    return out;
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenLinearBackward(
    const at::Tensor& self, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool, 3> output_mask) {
  TT_KERNEL(OpName::kLinearBackward, param_keys,
            (self, grad_output, weight,
             IgnoreInCacheKey(output_mask, "Each gradient is its own op")),
            {
              TT_ASSIGN_OR_THROW(
                  auto gradients,
                  LinearBackward(self, grad_output, weight, output_mask,
                                 std::move(param_keys)));
              return gradients;
            });
}

}  // namespace torch_tpu
