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

#include "csrc/ops/searchsorted_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "c10/util/string_view.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/cpu_to_tpu.h"
#include "csrc/ops/gather/gather.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace stablehlo = mlir::stablehlo;

namespace {

// PyTorch's dispatcher bypasses backend device guards for auxiliary keyword
// arguments (e.g. sorter) during out-variant and composite execution, allowing
// them to arrive on CPU. We use CopyCpuToTpuBuffer instead of ATen's .to() to
// transfer CPU tensors to TPU without triggering dispatcher re-entry or
// CompositeOpCheck failures during tracing.
absl::StatusOr<at::Tensor> ResolveAuxiliaryTensorDevice(
    const at::Tensor& tensor) {
  if (!tensor.defined() ||
      tensor.device().type() == GetPrivateUse1DeviceType()) {
    return tensor;
  }
  TT_ASSIGN_OR_RETURN(DeviceBufferRef tpu_buffer,
                      CopyCpuToTpuBuffer(tensor, /*non_blocking=*/false));
  return MakeTensor(std::move(tpu_buffer));
}

absl::StatusOr<c10::optional<at::Tensor>> ResolveAuxiliaryTensorDevice(
    const c10::optional<at::Tensor>& tensor) {
  if (!tensor.has_value() || !tensor.value().defined()) {
    return tensor;
  }
  TT_ASSIGN_OR_RETURN(at::Tensor resolved_tensor,
                      ResolveAuxiliaryTensorDevice(tensor.value()));
  return c10::make_optional(std::move(resolved_tensor));
}

absl::Status ValidateSearchsortedInputs(
    const at::Tensor& sorted, const at::Tensor& values,
    const c10::optional<at::Tensor>& sorter) {
  // Reject booleans for non-empty inputs to keep consistent with CUDA impl,
  // which short-circuits and returns success for empty inputs before
  // dispatching on the input dtype.
  TT_RET_CHECK(values.numel() == 0 || values.scalar_type() != at::kBool,
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(values.scalar_type());
  TT_RET_CHECK(sorted.dim() != 0, error::kInvalidArgument)
      << "expected sorted_sequence to have >0 dimension, got 0";

  if (values.dim() == 0 && sorted.dim() != 1) {
    TT_RET_CHECK(false, error::kInvalidArgument)
        << "expected values to not be a scalar when sorted_sequence dimension "
           "is "
           "not 1, got sorted_sequence dim "
        << sorted.dim() << " and values dim 0";
  }

  if (sorted.dim() != 1) {
    TT_RET_CHECK(sorted.dim() == values.dim(), error::kInvalidArgument)
        << "expected sorted_sequence to be 1-dimensional or have the same "
           "number of dimensions as values, got "
        << sorted.dim() << " and " << values.dim();
    for (int64_t i = 0; i < sorted.dim() - 1; ++i) {
      TT_RET_CHECK(sorted.size(i) == values.size(i), error::kInvalidArgument)
          << "expected sorted_sequence to have same shape as values except for "
             "the last dimension, got "
          << sorted.sizes() << " and " << values.sizes();
    }
  }

  if (sorter.has_value() && sorter.value().defined()) {
    TT_RET_CHECK(sorter.value().sizes() == sorted.sizes(),
                 error::kInvalidArgument)
        << "expected sorter and sorted_sequence to have the same shape, got "
        << sorter.value().sizes() << " and " << sorted.sizes();
    TT_RET_CHECK(sorter.value().scalar_type() == at::ScalarType::Long,
                 error::kInvalidArgument)
        << "expected sorter to have Long dtype, got "
        << sorter.value().scalar_type();
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> ResolveSearchsortedIsRight(
    bool right, c10::optional<c10::string_view> side) {
  if (!side.has_value()) {
    return right;
  }
  auto side_val = side.value();
  if (side_val != "left" && side_val != "right") {
    TT_RET_CHECK(false, error::kInvalidArgument)
        << "expected side to be 'left' or 'right', got '" << side_val << "'";
  }
  if (right && side_val == "left") {
    TT_RET_CHECK(false, error::kInvalidArgument)
        << "expected side and right to not be opposites, got side '" << side_val
        << "' and right True";
  }
  return side_val == "right";
}

mlir::MlirOp Broadcast1DSequenceToQueryDims(mlir::MlirOp seq_op,
                                            mlir::RankedTensorType query_type,
                                            int64_t seq_len) {
  auto seq_type = GetTensorTypeOrDie(seq_op);
  if (seq_type.getRank() == 1 && query_type.getRank() > 1) {
    llvm::SmallVector<int64_t, 4> broadcast_shape(query_type.getShape().begin(),
                                                  query_type.getShape().end());
    broadcast_shape.back() = seq_len;
    llvm::SmallVector<int64_t, 1> bcast_dims = {query_type.getRank() - 1};
    auto bcast_type = seq_type.clone(broadcast_shape);
    return stablehlo::BroadcastInDim(bcast_type, seq_op, bcast_dims);
  }
  return seq_op;
}

absl::StatusOr<mlir::MlirOp> ApplySorterIfPresent(
    mlir::MlirOp sorted, std::optional<mlir::MlirOp> sorter_op,
    mlir::RankedTensorType query_type, int64_t seq_len,
    mlir::ElementType sorted_seq_elem_type) {
  sorted = Broadcast1DSequenceToQueryDims(sorted, query_type, seq_len);
  if (!sorter_op.has_value()) {
    return sorted;
  }
  mlir::MlirOp s_op =
      Broadcast1DSequenceToQueryDims(sorter_op.value(), query_type, seq_len);
  auto sorted_type = GetTensorTypeOrDie(sorted);
  int64_t gather_dim =
      (query_type.getRank() == 0) ? 0 : (sorted_type.getRank() - 1);
  TT_ASSIGN_OR_RETURN(auto result, BuildGatherShlo(sorted, gather_dim, s_op,
                                                   /*sparse_grad=*/false,
                                                   sorted_seq_elem_type));
  return result;
}

absl::StatusOr<mlir::MlirOp> BuildSearchsortedLoopShlo(
    mlir::MlirOp sorted, mlir::MlirOp query, bool is_right, bool is_float,
    int64_t seq_len, int64_t num_iters, mlir::ElementType out_mlir_type,
    mlir::ElementType sorted_seq_elem_type, mlir::MlirBuilder& builder) {
  auto query_type = GetTensorTypeOrDie(query);
  mlir::RankedTensorType out_tensor_type = mlir::makeTensorType(
      builder.getContext(), query_type.getShape(), out_mlir_type);

  mlir::MlirOp low = MakeScalarConstant(builder, 0, out_mlir_type);
  low = stablehlo::BroadcastInDim(out_tensor_type, low, {});

  mlir::MlirOp init_high = MakeScalarConstant(builder, seq_len, out_mlir_type);
  init_high = stablehlo::BroadcastInDim(out_tensor_type, init_high, {});
  mlir::MlirOp high = init_high;

  mlir::MlirOp one = MakeScalarConstant(builder, 1, out_mlir_type);
  one = stablehlo::BroadcastInDim(out_tensor_type, one, {});

  mlir::MlirOp two = MakeScalarConstant(builder, 2, out_mlir_type);
  two = stablehlo::BroadcastInDim(out_tensor_type, two, {});

  mlir::MlirOp seq_len_minus_one =
      MakeScalarConstant(builder, seq_len - 1, out_mlir_type);
  seq_len_minus_one =
      stablehlo::BroadcastInDim(out_tensor_type, seq_len_minus_one, {});

  // Unrolled binary search loop: XLA/StableHLO requires bounded computation
  // loops to be unrolled at graph construction time. We execute `num_iters`
  // iterations, where `num_iters` is sufficient to converge on a sequence of
  // length `seq_len`.
  for (int64_t i = 0; i < num_iters; ++i) {
    auto diff = stablehlo::Subtract(high, low);
    auto half_diff = stablehlo::Div(diff, two);
    auto mid = stablehlo::Add(low, half_diff);
    auto clamped_mid = stablehlo::Min(mid, seq_len_minus_one);

    // When query is a 0D scalar tensor, `clamped_mid` is also 0D.
    // BuildGatherShlo expects a 1D index tensor when gathering along dim 0.
    // Therefore, we expand `clamped_mid` to 1D shape {1} for gathering, and
    // reshape the gathered result back to 0D {}.
    int64_t gather_dim =
        (query_type.getRank() == 0) ? 0 : (query_type.getRank() - 1);
    mlir::MlirOp gather_index = clamped_mid;
    if (query_type.getRank() == 0) {
      gather_index = stablehlo::Reshape(clamped_mid, {1});
    }
    TT_ASSIGN_OR_RETURN(
        auto val, BuildGatherShlo(sorted, gather_dim, gather_index,
                                  /*sparse_grad=*/false, sorted_seq_elem_type));
    if (query_type.getRank() == 0) {
      val = stablehlo::Reshape(val, {});
    }

    mlir::MlirOp mask =
        is_right
            ? stablehlo::Compare(val, query, stablehlo::ComparisonDirection::LE)
            : stablehlo::Compare(val, query,
                                 stablehlo::ComparisonDirection::LT);

    auto mid_plus_one = stablehlo::Add(mid, one);
    // Clamp mid_plus_one to init_high (seq_len) to prevent out-of-bounds XLA
    // gather clamping from incrementing low beyond seq_len on power-of-2
    // lengths.
    mid_plus_one = stablehlo::Min(mid_plus_one, init_high);
    low = stablehlo::Select(mask, mid_plus_one, low);
    high = stablehlo::Select(mask, high, mid);
  }

  mlir::MlirOp result = low;
  if (is_float) {
    // Floating-point NaN handling: according to PyTorch semantics, if a query
    // value is NaN, searchsorted must return the sequence length (the upper
    // bound).
    auto is_nan =
        stablehlo::Compare(query, query, stablehlo::ComparisonDirection::NE);
    result = stablehlo::Select(is_nan, init_high, result);
  }
  return result;
}

template <typename ReturnType, typename DispatchFn>
ReturnType SearchsortedTensorCommon(const at::Tensor& sorted_sequence,
                                    const at::Tensor& values, bool out_int32,
                                    bool right,
                                    c10::optional<c10::string_view> side,
                                    const c10::optional<at::Tensor>& sorter,
                                    OpParamCacheKeys&& param_keys,
                                    DispatchFn&& dispatch_fn) {
  TT_ASSIGN_OR_RETURN(at::Tensor resolved_sorted,
                      ResolveAuxiliaryTensorDevice(sorted_sequence));
  TT_ASSIGN_OR_RETURN(at::Tensor resolved_values,
                      ResolveAuxiliaryTensorDevice(values));
  TT_ASSIGN_OR_RETURN(c10::optional<at::Tensor> resolved_sorter,
                      ResolveAuxiliaryTensorDevice(sorter));

  TT_RETURN_IF_ERROR(ValidateSearchsortedInputs(
      resolved_sorted, resolved_values, resolved_sorter));
  TT_ASSIGN_OR_RETURN(bool is_right, ResolveSearchsortedIsRight(right, side));

  at::ScalarType out_dtype =
      out_int32 ? at::ScalarType::Int : at::ScalarType::Long;
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_mlir_type,
                      internal::ToElementType(out_dtype));
  Dimensions out_dims = CopyIntVector(resolved_values.sizes());

  std::vector<at::Tensor> input_vec = {resolved_sorted, resolved_values};
  input_vec.reserve(3);
  if (resolved_sorter.has_value() && resolved_sorter.value().defined()) {
    input_vec.push_back(resolved_sorter.value());
  }

  int64_t seq_len = resolved_sorted.size(-1);
  int64_t num_iters = 0;
  int64_t temp_seq_len = seq_len;
  while (temp_seq_len > 0) {
    num_iters++;
    temp_seq_len /= 2;
  }

  at::ScalarType common_aten_type = at::promoteTypes(
      resolved_sorted.scalar_type(), resolved_values.scalar_type());
  TT_ASSIGN_OR_RETURN(auto common_elem_type,
                      internal::ToElementType(common_aten_type));
  const bool is_float = at::isFloatingType(common_aten_type);

  auto build_shlo =
      [is_right, out_mlir_type, seq_len, num_iters,
       has_sorter =
           (resolved_sorter.has_value() && resolved_sorter.value().defined()),
       common_elem_type,
       is_float](absl::Span<mlir::MlirOp const> inputs,
                 mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp sorted = inputs[0];
    mlir::MlirOp query = inputs[1];
    std::optional<mlir::MlirOp> sorter_op;
    if (has_sorter) {
      sorter_op = inputs[2];
    }

    mlir::Type common_mlir_type =
        mlir::getElementType(builder.getContext(), common_elem_type);
    if (GetTensorTypeOrDie(sorted).getElementType() != common_mlir_type) {
      sorted = stablehlo::ConvertElementType(sorted, common_mlir_type);
    }
    if (GetTensorTypeOrDie(query).getElementType() != common_mlir_type) {
      query = stablehlo::ConvertElementType(query, common_mlir_type);
    }

    auto query_type = GetTensorTypeOrDie(query);
    TT_ASSIGN_OR_RETURN(
        sorted, ApplySorterIfPresent(sorted, sorter_op, query_type, seq_len,
                                     common_elem_type));
    return BuildSearchsortedLoopShlo(sorted, query, is_right, is_float, seq_len,
                                     num_iters, out_mlir_type, common_elem_type,
                                     builder);
  };

  return dispatch_fn(std::move(build_shlo), std::move(input_vec), out_mlir_type,
                     out_dims, std::move(param_keys),
                     resolved_values.numel() == 0);
}

absl::StatusOr<DeviceBufferRef> SearchsortedTensorInternal(
    const at::Tensor& sorted_sequence, const at::Tensor& values, bool out_int32,
    bool right, c10::optional<c10::string_view> side,
    const c10::optional<at::Tensor>& sorter, OpParamCacheKeys&& param_keys) {
  return SearchsortedTensorCommon<absl::StatusOr<DeviceBufferRef>>(
      sorted_sequence, values, out_int32, right, side, sorter,
      std::move(param_keys),
      [](auto build_shlo, auto input_vec, mlir::ElementType out_mlir_type,
         const Dimensions& out_dims, OpParamCacheKeys&& param_keys,
         bool is_empty) -> absl::StatusOr<DeviceBufferRef> {
        if (is_empty) {
          return CreateZeroSizeDeviceBufferRef(out_dims, out_mlir_type);
        }
        return DispatchOp<kDynamicSize>(
            std::move(build_shlo), input_vec,
            {.out_dtype = out_mlir_type,
             .out_dims = out_dims,
             .op_param_cache_keys = std::move(param_keys)});
      });
}

absl::Status SearchsortedTensorOut(const at::Tensor& sorted_sequence,
                                   const at::Tensor& values, bool out_int32,
                                   bool right,
                                   c10::optional<c10::string_view> side,
                                   const c10::optional<at::Tensor>& sorter,
                                   at::Tensor& out,
                                   OpParamCacheKeys&& param_keys) {
  return SearchsortedTensorCommon<absl::Status>(
      sorted_sequence, values, out_int32, right, side, sorter,
      std::move(param_keys),
      [&out](auto build_shlo, auto input_vec, mlir::ElementType out_mlir_type,
             const Dimensions& out_dims, OpParamCacheKeys&& param_keys,
             bool is_empty) -> absl::Status {
        if (is_empty) {
          TT_ASSIGN_OR_RETURN(
              DeviceBufferRef result_buf,
              CreateZeroSizeDeviceBufferRef(out_dims, out_mlir_type));
          TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, out_dims));
          return AssignBufferToAtTensor(std::move(result_buf), out);
        }
        return DispatchOpOut<kDynamicSize>(
            std::move(build_shlo), input_vec, out,
            {.out_dtype = out_mlir_type,
             .out_dims = out_dims,
             .op_param_cache_keys = std::move(param_keys)});
      });
}

}  // namespace

at::Tensor AtenSearchsortedTensor(const at::Tensor& sorted_sequence,
                                  const at::Tensor& values, bool out_int32,
                                  bool right,
                                  c10::optional<c10::string_view> side,
                                  const c10::optional<at::Tensor>& sorter) {
  TT_KERNEL(OpName::kSearchsortedTensor, param_keys,
            (sorted_sequence, values, out_int32, right, side, sorter), {
              TT_ASSIGN_OR_THROW(auto result,
                                 SearchsortedTensorInternal(
                                     sorted_sequence, values, out_int32, right,
                                     side, sorter, std::move(param_keys)));
              return MakeTensor(std::move(result));
            });
}

at::Tensor& AtenSearchsortedTensorOut(const at::Tensor& sorted_sequence,
                                      const at::Tensor& values, bool out_int32,
                                      bool right,
                                      c10::optional<c10::string_view> side,
                                      const c10::optional<at::Tensor>& sorter,
                                      at::Tensor& out) {
  TT_KERNEL(OpName::kSearchsortedTensorOut, param_keys,
            (sorted_sequence, values, out_int32, right, side, sorter, out), {
              TT_THROW_IF_ERROR(SearchsortedTensorOut(
                  sorted_sequence, values, out_int32, right, side, sorter, out,
                  std::move(param_keys)));
              return out;
            });
}

at::Tensor AtenSearchsortedScalar(const at::Tensor& sorted_sequence,
                                  const at::Scalar& value, bool out_int32,
                                  bool right,
                                  c10::optional<c10::string_view> side,
                                  const c10::optional<at::Tensor>& sorter) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(OpName::kSearchsortedScalar, param_keys,
            (sorted_sequence, promoted_value, out_int32, right, side, sorter), {
              TT_ASSIGN_OR_THROW(at::Tensor values, promoted_value.GetTensor());
              TT_ASSIGN_OR_THROW(auto result,
                                 SearchsortedTensorInternal(
                                     sorted_sequence, values, out_int32, right,
                                     side, sorter, std::move(param_keys)));
              return MakeTensor(std::move(result));
            });
}

at::Tensor& AtenSearchsortedScalarOut(const at::Tensor& sorted_sequence,
                                      const at::Scalar& value, bool out_int32,
                                      bool right,
                                      c10::optional<c10::string_view> side,
                                      const c10::optional<at::Tensor>& sorter,
                                      at::Tensor& out) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kSearchsortedScalarOut, param_keys,
      (sorted_sequence, promoted_value, out_int32, right, side, sorter, out), {
        TT_ASSIGN_OR_THROW(at::Tensor values, promoted_value.GetTensor());
        TT_THROW_IF_ERROR(SearchsortedTensorOut(sorted_sequence, values,
                                                out_int32, right, side, sorter,
                                                out, std::move(param_keys)));
        return out;
      });
}

}  // namespace torch_tpu
