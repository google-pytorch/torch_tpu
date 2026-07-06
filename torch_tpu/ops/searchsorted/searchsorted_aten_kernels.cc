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

#include "torch_tpu/ops/searchsorted/searchsorted_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/gather/gather.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
namespace {

enum class SearchsortedOutDtype { kInt64, kInt32 };
enum class SearchsortedBoundMode { kLeft, kRight };
enum class SearchsortedRightMode { kFalse, kTrue };

absl::Status ValidateSearchsortedArgs(
    const at::Tensor& sorted_sequence, const at::Tensor& self,
    SearchsortedRightMode right_mode,
    const std::optional<std::string_view> side_opt,
    const std::optional<at::Tensor>& sorter) {
  // 1. Validate 'side' vs 'right' rules
  if (side_opt.has_value()) {
    const std::string_view side = *side_opt;
    TT_RET_CHECK(side == "left" || side == "right", error::kInvalidArgument)
        << "expected side to be 'left' or 'right', got " << side;
    TT_RET_CHECK(right_mode == SearchsortedRightMode::kFalse || side == "right",
                 error::kInvalidArgument)
        << "expected side to not be 'left' when right is true, got side='left' "
           "and right=true";
  }

  // 2. Validate Dtypes (No Complex Dtypes)
  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType seq_dtype,
      ConvertTo<mlir::ElementType>(sorted_sequence.scalar_type()));
  TT_RET_CHECK(!IsComplex(seq_dtype), error::kInvalidArgument)
      << "expected a non-complex dtype for sorted_sequence, got "
      << ToString(sorted_sequence.scalar_type());

  TT_ASSIGN_OR_RETURN(const mlir::ElementType self_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_RET_CHECK(!IsComplex(self_dtype), error::kInvalidArgument)
      << "expected a non-complex dtype for self, got "
      << ToString(self.scalar_type());

  // 3. Validate Sorter Tensor (if defined)
  if (sorter.has_value() && sorter->defined()) {
    TT_RET_CHECK(sorter->sizes() == sorted_sequence.sizes(),
                 error::kInvalidArgument)
        << "expected sorter to have the same shape as sorted_sequence "
        << ToString(sorted_sequence.sizes()) << ", got "
        << ToString(sorter->sizes());
    TT_RET_CHECK(sorter->scalar_type() == at::ScalarType::Long,
                 error::kInvalidArgument)
        << "expected sorter to have int64 dtype, got "
        << ToString(sorter->scalar_type());
  }

  // 4. Validate Dimensional Compatibility
  TT_RET_CHECK(sorted_sequence.dim() > 0, error::kInvalidArgument)
      << "expected sorted_sequence to have > 0 dimensions, got 0";

  TT_RET_CHECK(self.dim() > 0 || (self.dim() == 0 && self.numel() == 1 &&
                                  sorted_sequence.dim() == 1),
               error::kInvalidArgument)
      << "expected sorted_sequence to be 1D when self is a scalar or 0D "
         "tensor, "
      << "got sorted_sequence with shape " << ToString(sorted_sequence.sizes())
      << " and self with shape " << ToString(self.sizes());

  if (sorted_sequence.dim() > 1) {
    TT_RET_CHECK(self.dim() == sorted_sequence.dim(), error::kInvalidArgument)
        << "expected self and sorted_sequence to have the same number of "
           "dimensions "
        << "when sorted_sequence is not 1D, got sorted_sequence with shape "
        << ToString(sorted_sequence.sizes()) << " and self with shape "
        << ToString(self.sizes());

    for (int64_t i = 0; i < sorted_sequence.dim() - 1; ++i) {
      TT_RET_CHECK(sorted_sequence.size(i) == self.size(i),
                   error::kInvalidArgument)
          << "expected self and sorted_sequence to have matching outer "
             "dimension sizes, "
          << "got sorted_sequence with shape "
          << ToString(sorted_sequence.sizes()) << " and self with shape "
          << ToString(self.sizes());
    }
  }

  return absl::OkStatus();
}

absl::Status PrepareOutTensor(const at::Tensor& result,
                              SearchsortedOutDtype out_dtype_mode,
                              at::Tensor& out) {
  const at::ScalarType expected_dtype =
      (out_dtype_mode == SearchsortedOutDtype::kInt32) ? at::ScalarType::Int
                                                       : at::ScalarType::Long;
  TT_RET_CHECK(out.scalar_type() == expected_dtype, error::kInvalidArgument)
      << "expected out to have " << ToString(expected_dtype)
      << " dtype when out_int32 is "
      << ((out_dtype_mode == SearchsortedOutDtype::kInt32) ? "true" : "false")
      << ", got " << ToString(out.scalar_type());

  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, result.sizes()));
  return absl::OkStatus();
}

// Builds the StableHLO graph for searchsorted using an unsqueezed broadcast
// reduction:
// 1. Unsqueezes self by 1 at the end and sorted_sequence by 1 at the
//    second-to-last dim so they broadcast together to shape (..., M, N).
// 2. Compares sorted_sequence against self using LE (right bound) or LT
//    (left bound) to generate a boolean mask of sequence elements preceding the
//    insertion value.
// 3. Reduces along the innermost dimension N with AddOp to count matching
//    elements, which directly equals the sorted insertion index.
absl::StatusOr<mlir::MlirOp> BuildSearchsortedShlo(
    mlir::MlirOp self_op, mlir::MlirOp sorted_seq_op,
    mlir::ElementType out_dtype, SearchsortedBoundMode bound_mode) {
  const int64_t self_rank = GetTensorTypeOrDie(self_op).getRank();
  const int64_t seq_rank = GetTensorTypeOrDie(sorted_seq_op).getRank();

  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_unsqueeze,
                      Unsqueeze(self_op, self_rank));
  mlir::MlirOp seq_bcast_input = sorted_seq_op;
  if (seq_rank > 1) {
    TT_ASSIGN_OR_RETURN(seq_bcast_input,
                        Unsqueeze(sorted_seq_op, seq_rank - 1));
  }

  TT_ASSIGN_OR_RETURN(const auto broadcasted,
                      ApplyBroadcastIfNeeded(self_unsqueeze, seq_bcast_input));
  mlir::MlirOp self_bcast = broadcasted[0];
  mlir::MlirOp seq_bcast = broadcasted[1];

  const auto direction = (bound_mode == SearchsortedBoundMode::kRight)
                             ? mlir::stablehlo::ComparisonDirection::LE
                             : mlir::stablehlo::ComparisonDirection::LT;
  const mlir::MlirOp compare_op = mlir::stablehlo::ConvertElementType(
      mlir::stablehlo::Compare(seq_bcast, self_bcast, direction), out_dtype);

  const int64_t reduction_dim = GetTensorTypeOrDie(compare_op).getRank() - 1;
  mlir::MlirBuilder& builder = self_op.getBuilder();

  const mlir::SmallVector<mlir::MlirOp> reduce_results =
      mlir::stablehlo::Reduce(
          builder, {compare_op}, {MakeScalarConstant(builder, 0, out_dtype)},
          [out_dtype](mlir::RegionBuilder& region_builder) {
            mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
                mlir::getElementType(region_builder.getContext(), out_dtype),
                region_builder.getRegion(), region_builder.getOpBuilder());
          },
          /*dimensions=*/{reduction_dim});

  return reduce_results[0];
}

absl::StatusOr<DeviceBufferRef> SearchsortedImpl(
    const at::Tensor& sorted_sequence, const at::Tensor& self,
    const std::optional<at::Tensor>& sorter,
    SearchsortedOutDtype out_dtype_mode, SearchsortedBoundMode bound_mode,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(
      const auto common_dtype,
      ConvertTo<mlir::ElementType>(at::result_type(sorted_sequence, self)));
  const auto out_dtype = (out_dtype_mode == SearchsortedOutDtype::kInt32)
                             ? mlir::ElementType::I32
                             : mlir::ElementType::I64;

  if (sorter.has_value() && sorter->defined()) {
    auto op_builder = [common_dtype, out_dtype, bound_mode,
                       dim = sorted_sequence.dim() - 1,
                       has_elements = sorted_sequence.size(-1) >
                                      0](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [sorted_seq_op, self_op, sorter_op] = inputs;

      TT_ASSIGN_OR_RETURN(const mlir::MlirOp cast_seq_op,
                          CastIfNeeded(sorted_seq_op, common_dtype));
      TT_ASSIGN_OR_RETURN(const mlir::MlirOp cast_self_op,
                          CastIfNeeded(self_op, common_dtype));
      mlir::MlirOp gathered_seq_op = cast_seq_op;
      if (has_elements) {
        TT_ASSIGN_OR_RETURN(
            gathered_seq_op,
            BuildGatherShlo(cast_seq_op, dim, sorter_op, false, common_dtype));
      }

      TT_ASSIGN_OR_RETURN(const mlir::MlirOp result_op,
                          BuildSearchsortedShlo(cast_self_op, gathered_seq_op,
                                                out_dtype, bound_mode));
      return result_op;
    };

    return DispatchOp<3>(std::move(op_builder),
                         {sorted_sequence, self, *sorter},
                         {.out_dtype = out_dtype,
                          .out_dims = CopyIntVector(self.sizes()),
                          .op_param_cache_keys = std::move(param_keys)});
  }

  auto op_builder = [common_dtype, out_dtype,
                     bound_mode](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [sorted_seq_op, self_op] = inputs;

    TT_ASSIGN_OR_RETURN(const mlir::MlirOp cast_seq_op,
                        CastIfNeeded(sorted_seq_op, common_dtype));
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp cast_self_op,
                        CastIfNeeded(self_op, common_dtype));

    TT_ASSIGN_OR_RETURN(const mlir::MlirOp result_op,
                        BuildSearchsortedShlo(cast_self_op, cast_seq_op,
                                              out_dtype, bound_mode));
    return result_op;
  };

  return DispatchOp<2>(std::move(op_builder), {sorted_sequence, self},
                       {.out_dtype = out_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}

struct SearchsortedKernelArgs {
  SearchsortedOutDtype out_dtype_mode;
  SearchsortedRightMode right_mode;
  SearchsortedBoundMode bound_mode;
  std::optional<at::Tensor> tpu_sorter;
};

SearchsortedKernelArgs PrepareSearchsortedKernelArgs(
    const at::Tensor& sorted_sequence, bool out_int32, bool right,
    const std::optional<std::string_view>& side,
    const std::optional<at::Tensor>& sorter) {
  const SearchsortedOutDtype out_dtype_mode =
      out_int32 ? SearchsortedOutDtype::kInt32 : SearchsortedOutDtype::kInt64;
  const SearchsortedRightMode right_mode =
      right ? SearchsortedRightMode::kTrue : SearchsortedRightMode::kFalse;
  const bool is_right_bound = (side.has_value() ? (*side == "right") : right);
  const SearchsortedBoundMode bound_mode = is_right_bound
                                               ? SearchsortedBoundMode::kRight
                                               : SearchsortedBoundMode::kLeft;
  std::optional<at::Tensor> tpu_sorter = sorter;
  if (tpu_sorter.has_value() && tpu_sorter->defined() &&
      tpu_sorter->device() != sorted_sequence.device()) {
    tpu_sorter = tpu_sorter->to(sorted_sequence.device());
  }
  return {out_dtype_mode, right_mode, bound_mode, std::move(tpu_sorter)};
}

}  // namespace

at::Tensor AtenSearchsortedTensor(const at::Tensor& sorted_sequence,
                                  const at::Tensor& self, bool out_int32,
                                  bool right,
                                  const std::optional<std::string_view> side,
                                  const std::optional<at::Tensor>& sorter) {
  auto args = PrepareSearchsortedKernelArgs(sorted_sequence, out_int32, right,
                                            side, sorter);
  auto& tpu_sorter = args.tpu_sorter;
  TT_KERNEL(OpName::kSearchsortedTensor, param_keys,
            (sorted_sequence, self, out_int32, right, side, tpu_sorter), {
              TT_THROW_IF_ERROR(ValidateSearchsortedArgs(
                  sorted_sequence, self, args.right_mode, side, tpu_sorter));

              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buffer,
                  SearchsortedImpl(sorted_sequence, self, tpu_sorter,
                                   args.out_dtype_mode, args.bound_mode,
                                   std::move(param_keys)));
              return MakeTensor(result_buffer);
            });
}

at::Tensor& AtenSearchsortedTensorOut(
    const at::Tensor& sorted_sequence, const at::Tensor& self, bool out_int32,
    bool right, const std::optional<std::string_view> side,
    const std::optional<at::Tensor>& sorter, at::Tensor& out) {
  auto args = PrepareSearchsortedKernelArgs(sorted_sequence, out_int32, right,
                                            side, sorter);
  auto& tpu_sorter = args.tpu_sorter;
  TT_KERNEL(OpName::kSearchsortedTensorOut, param_keys,
            (sorted_sequence, self, out_int32, right, side, tpu_sorter, out), {
              TT_THROW_IF_ERROR(ValidateSearchsortedArgs(
                  sorted_sequence, self, args.right_mode, side, tpu_sorter));

              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buffer,
                  SearchsortedImpl(sorted_sequence, self, tpu_sorter,
                                   args.out_dtype_mode, args.bound_mode,
                                   std::move(param_keys)));

              const at::Tensor temp_result = MakeTensor(result_buffer);
              TT_THROW_IF_ERROR(
                  PrepareOutTensor(temp_result, args.out_dtype_mode, out));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), out));
              return out;
            });
}

at::Tensor AtenSearchsortedScalar(const at::Tensor& sorted_sequence,
                                  const at::Scalar& self, bool out_int32,
                                  bool right,
                                  const std::optional<std::string_view> side,
                                  const std::optional<at::Tensor>& sorter) {
  auto args = PrepareSearchsortedKernelArgs(sorted_sequence, out_int32, right,
                                            side, sorter);
  auto& tpu_sorter = args.tpu_sorter;
  auto promoted_self = PromoteScalar(self);
  TT_KERNEL(
      OpName::kSearchsortedScalar, param_keys,
      (sorted_sequence, promoted_self, out_int32, right, side, tpu_sorter), {
        TT_ASSIGN_OR_THROW(
            at::Tensor self_tensor,
            promoted_self.GetTensor(sorted_sequence.scalar_type()));
        TT_THROW_IF_ERROR(ValidateSearchsortedArgs(
            sorted_sequence, self_tensor, args.right_mode, side, tpu_sorter));

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buffer,
            SearchsortedImpl(sorted_sequence, self_tensor, tpu_sorter,
                             args.out_dtype_mode, args.bound_mode,
                             std::move(param_keys)));
        return MakeTensor(result_buffer);
      });
}

at::Tensor& AtenSearchsortedScalarOut(
    const at::Tensor& sorted_sequence, const at::Scalar& self, bool out_int32,
    bool right, const std::optional<std::string_view> side,
    const std::optional<at::Tensor>& sorter, at::Tensor& out) {
  auto args = PrepareSearchsortedKernelArgs(sorted_sequence, out_int32, right,
                                            side, sorter);
  auto& tpu_sorter = args.tpu_sorter;
  auto promoted_self = PromoteScalar(self);
  TT_KERNEL(
      OpName::kSearchsortedScalarOut, param_keys,
      (sorted_sequence, promoted_self, out_int32, right, side, tpu_sorter, out),
      {
        TT_ASSIGN_OR_THROW(
            at::Tensor self_tensor,
            promoted_self.GetTensor(sorted_sequence.scalar_type()));
        TT_THROW_IF_ERROR(ValidateSearchsortedArgs(
            sorted_sequence, self_tensor, args.right_mode, side, tpu_sorter));

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buffer,
            SearchsortedImpl(sorted_sequence, self_tensor, tpu_sorter,
                             args.out_dtype_mode, args.bound_mode,
                             std::move(param_keys)));

        const at::Tensor temp_result = MakeTensor(result_buffer);
        TT_THROW_IF_ERROR(
            PrepareOutTensor(temp_result, args.out_dtype_mode, out));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

}  // namespace torch_tpu
