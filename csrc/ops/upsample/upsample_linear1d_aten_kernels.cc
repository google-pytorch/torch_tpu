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

#include "csrc/ops/upsample/upsample_linear1d_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

enum class AlignCorners : bool {
  kNo = false,
  kYes = true,
};

struct Linear1dGatherConfig {
  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers;
  Dimensions slice_sizes;
};

struct Linear1dDimInfo {
  mlir::MlirOp left_indices;
  mlir::MlirOp right_indices;
  mlir::MlirOp left_weight;
  mlir::MlirOp right_weight;
};

mlir::MlirOp Clamp(mlir::MlirOp min, mlir::MlirOp operand, mlir::MlirOp max) {
  return mlir::stablehlo::Clamp(min, operand, max);
}

mlir::MlirOp Gather(
    mlir::MlirOp operand, mlir::MlirOp start_indices,
    mlir::stablehlo::GatherDimensionNumbersAttr dimension_numbers,
    llvm::ArrayRef<int64_t> slice_sizes, bool indices_are_sorted = false) {
  return mlir::stablehlo::Gather(operand, start_indices, dimension_numbers,
                                 slice_sizes, indices_are_sorted);
}

mlir::MlirOp Reshape(mlir::Type result_type, mlir::MlirOp operand) {
  return mlir::stablehlo::Reshape(result_type, operand);
}

mlir::MlirOp BroadcastInDim(mlir::Type result_type, mlir::MlirOp operand,
                            llvm::ArrayRef<int64_t> broadcast_dimensions) {
  return mlir::stablehlo::BroadcastInDim(result_type, operand,
                                         broadcast_dimensions);
}

Linear1dGatherConfig GetLinear1dGatherConfig(mlir::MlirOp input) {
  const mlir::stablehlo::Dimensions input_dims = GetDimensions(input);
  Dimensions slice_sizes = {input_dims[0].size, input_dims[1].size, 1};

  const Dimensions offset_dims = {0, 1};
  const Dimensions collapsed_slice_dims = {2};
  const Dimensions start_index_map = {2};

  const mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      mlir::stablehlo::GatherDimensionNumbersAttr::get(
          &input.getContext(),
          /*offset_dims=*/AsArrayRef<int64_t>(offset_dims),
          /*collapsed_slice_dims=*/AsArrayRef<int64_t>(collapsed_slice_dims),
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/AsArrayRef<int64_t>(start_index_map),
          /*index_vector_dim=*/1);

  return {gather_dimension_numbers, slice_sizes};
}

absl::StatusOr<mlir::MlirOp> ComputeLinear1dSourceIndex(
    mlir::MlirBuilder& builder, int64_t out_size, int64_t in_size,
    std::optional<double> scale_opt, AlignCorners align_corners,
    mlir::Type calc_type) {
  const auto iota_type = mlir::RankedTensorType::get({out_size}, calc_type);
  const auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);

  if (align_corners == AlignCorners::kYes) {
    const double stride = (out_size > 1) ? static_cast<double>(in_size - 1) /
                                               static_cast<double>(out_size - 1)
                                         : 0.0;
    const auto stride_const = MakeConstantLike(iota, stride);
    return BuildMulShlo(iota, stride_const);
  }

  const double scale =
      (scale_opt.has_value() && *scale_opt > 0.0)
          ? (1.0 / *scale_opt)
          : (static_cast<double>(in_size) / static_cast<double>(out_size));
  const auto half_const = MakeConstantLike(iota, 0.5);
  TT_ASSIGN_OR_RETURN(const auto iota_plus_half,
                      BuildAddShlo(iota, half_const));

  const auto scale_const = MakeConstantLike(iota, scale);
  TT_ASSIGN_OR_RETURN(const auto scaled,
                      BuildMulShlo(iota_plus_half, scale_const));
  TT_ASSIGN_OR_RETURN(const auto src_idx, BuildSubShlo(scaled, half_const));

  const auto zero_const = MakeConstantLike(iota, 0.0);
  return BuildMaximumShlo(src_idx, zero_const);
}

absl::StatusOr<Linear1dDimInfo> ComputeLinear1dDimInfo(
    mlir::MlirBuilder& builder, mlir::MlirOp src_idx, int64_t out_size,
    int64_t in_size, mlir::Type calc_type) {
  const auto idx_floor_calc = mlir::stablehlo::Floor(src_idx);
  const auto one_calc = MakeConstantLike(src_idx, 1.0);

  TT_ASSIGN_OR_RETURN(const auto idx_ceil_calc,
                      BuildAddShlo(idx_floor_calc, one_calc));

  TT_ASSIGN_OR_RETURN(const auto right_weight,
                      BuildSubShlo(src_idx, idx_floor_calc));
  TT_ASSIGN_OR_RETURN(const auto left_weight,
                      BuildSubShlo(one_calc, right_weight));

  const auto i32_type = builder.getOpBuilder().getI32Type();
  const auto idx_floor_i32_unclamped =
      mlir::stablehlo::ConvertElementType(idx_floor_calc, i32_type);
  const auto idx_ceil_i32_unclamped =
      mlir::stablehlo::ConvertElementType(idx_ceil_calc, i32_type);

  const auto zero_i32 = MakeConstantLike(idx_floor_i32_unclamped, 0);
  const auto max_idx_i32 = MakeConstantLike(idx_floor_i32_unclamped,
                                            std::max<int64_t>(0, in_size - 1));

  const auto idx_floor_i32 =
      Clamp(idx_floor_i32_unclamped, zero_i32, max_idx_i32);
  const auto idx_ceil_i32 =
      Clamp(idx_ceil_i32_unclamped, zero_i32, max_idx_i32);

  const auto index_tensor_type =
      mlir::RankedTensorType::get({out_size, 1}, i32_type);
  const auto left_indices = Reshape(index_tensor_type, idx_floor_i32);
  const auto right_indices = Reshape(index_tensor_type, idx_ceil_i32);

  return Linear1dDimInfo{left_indices, right_indices, left_weight,
                         right_weight};
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleLinear1dShlo(
    mlir::MlirOp input, const Dimensions& output_shape,
    AlignCorners align_corners, std::optional<double> scales,
    mlir::Type calc_type) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::stablehlo::Dimensions input_dims = GetDimensions(input);

  const int64_t in_width = input_dims[2].size;
  const int64_t out_width = output_shape[2];
  const auto element_type = GetTensorTypeOrDie(input).getElementType();

  TT_ASSIGN_OR_RETURN(
      const auto src_idx,
      ComputeLinear1dSourceIndex(builder, out_width, in_width, scales,
                                 align_corners, calc_type));

  TT_ASSIGN_OR_RETURN(
      const auto dim_info,
      ComputeLinear1dDimInfo(builder, src_idx, out_width, in_width, calc_type));

  const Linear1dGatherConfig gather_config = GetLinear1dGatherConfig(input);

  const auto left_gathered = Gather(
      input, dim_info.left_indices, gather_config.gather_dimension_numbers,
      AsArrayRef<int64_t>(gather_config.slice_sizes),
      /*indices_are_sorted=*/true);

  const auto right_gathered = Gather(
      input, dim_info.right_indices, gather_config.gather_dimension_numbers,
      AsArrayRef<int64_t>(gather_config.slice_sizes),
      /*indices_are_sorted=*/true);

  const auto left_gathered_calc =
      mlir::stablehlo::ConvertElementType(left_gathered, calc_type);
  const auto right_gathered_calc =
      mlir::stablehlo::ConvertElementType(right_gathered, calc_type);

  const Dimensions bcast_shape = {input_dims[0].size, input_dims[1].size,
                                  out_width};
  const auto weight_broadcast_type =
      mlir::RankedTensorType::get(AsArrayRef<int64_t>(bcast_shape), calc_type);
  const auto left_weight_bcast =
      BroadcastInDim(weight_broadcast_type, dim_info.left_weight, {2});
  const auto right_weight_bcast =
      BroadcastInDim(weight_broadcast_type, dim_info.right_weight, {2});

  TT_ASSIGN_OR_RETURN(const auto left_term,
                      BuildMulShlo(left_gathered_calc, left_weight_bcast));
  TT_ASSIGN_OR_RETURN(const auto right_term,
                      BuildMulShlo(right_gathered_calc, right_weight_bcast));

  TT_ASSIGN_OR_RETURN(const auto interpolated,
                      BuildAddShlo(left_term, right_term));

  auto result = mlir::stablehlo::ConvertElementType(interpolated, element_type);
  for (int i = 0; i < 2; ++i) {
    if (input_dims[i].boundOp.has_value()) {
      mlir::MlirOp bound_op_mlir(builder, *input_dims[i].boundOp);
      mlir::MlirOp dim_size = mlir::stablehlo::GetDimensionSize(
          bound_op_mlir, input_dims[i].boundOpDim);
      result = mlir::stablehlo::SetDimensionSize(result, dim_size, i);
    }
  }
  return result;
}

void CheckUpsampleLinear1dDtypes(const at::Tensor& tensor) {
  TT_CHECK_THROW(at::isFloatingType(tensor.scalar_type()),
                 error::kPythonNotImplementedError)
      << "not implemented for " << ToString(tensor.scalar_type());
}

void ValidateUpsampleLinear1dInputs(const at::Tensor& self,
                                    at::IntArrayRef output_size,
                                    const at::Tensor& out) {
  TT_CHECK_THROW(output_size.size() == 1, error::kInvalidArgument)
      << "expected output_size to have 1 element, got "
      << ToString(output_size.size());

  TT_CHECK_THROW(self.dim() == 3, error::kInvalidArgument)
      << "expected 3D input tensor, got " << ToString(self.dim()) << "D tensor";

  const int64_t input_width = self.size(2);
  const int64_t output_width = output_size[0];

  TT_CHECK_THROW(input_width > 0 && output_width > 0, error::kInvalidArgument)
      << "expected input and output sizes to be greater than 0, got input (W: "
      << ToString(input_width) << ") and output (W: " << ToString(output_width)
      << ")";

  CheckUpsampleLinear1dDtypes(self);

  TT_CHECK_THROW(self.scalar_type() == out.scalar_type(),
                 error::kInvalidArgument)
      << "expected out dtype " << ToString(self.scalar_type()) << ", got "
      << ToString(out.scalar_type());
}

}  // namespace

at::Tensor& AtenUpsampleLinear1dOut(const at::Tensor& self,
                                    at::IntArrayRef output_size,
                                    bool align_corners,
                                    std::optional<double> scales,
                                    at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleLinear1dOut, param_keys,
      (self, output_size, align_corners, scales, out), {
        ValidateUpsampleLinear1dInputs(self, output_size, out);

        const Dimensions expected_output_shape = {self.size(0), self.size(1),
                                                  output_size[0]};
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, expected_output_shape));

        if (self.size(2) == output_size[0]) {
          out.copy_(self);
          return out;
        }

        if (self.numel() == 0) {
          return out;
        }

        TT_ASSIGN_OR_THROW(const auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const at::ScalarType real_dtype =
            c10::toRealValueType(self.scalar_type());

        const auto op_builder =
            [out_shape = CopyIntVector(out.sizes()),
             align_corners =
                 align_corners ? AlignCorners::kYes : AlignCorners::kNo,
             scales,
             real_dtype](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          const mlir::Type calc_type =
              (real_dtype == at::kDouble)
                  ? input.getBuilder().getOpBuilder().getF64Type()
                  : input.getBuilder().getOpBuilder().getF32Type();
          return BuildUpsampleLinear1dShlo(input, out_shape, align_corners,
                                           scales, calc_type);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<1>(std::move(op_builder), {self}, out,
                              {.out_dtype = element_type,
                               .out_dims = CopyIntVector(out.sizes()),
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

}  // namespace torch_tpu
