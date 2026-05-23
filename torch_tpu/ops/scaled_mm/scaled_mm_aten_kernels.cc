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

#include "torch_tpu/ops/scaled_mm/scaled_mm_aten_kernels.h"

#include <array>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildStablehloDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = lhs.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  auto dot_dimension_numbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{1}, /*rhs_contracting_dimensions=*/{0});

  auto precision_config_attr =
      mlir::stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});

  auto lhs_type = GetTensorTypeOrDie(lhs);
  auto rhs_type = GetTensorTypeOrDie(rhs);
  Dimensions result_shape = {lhs_type.getShape()[0], rhs_type.getShape()[1]};

  TT_ASSIGN_OR_RETURN(
      mlir::ElementType lhs_elem_type,
      torch_tpu::ConvertTo<mlir::ElementType>(lhs_type.getElementType()));
  TT_ASSIGN_OR_RETURN(auto comp_dtype,
                      torch_tpu::InferComputationDtype(lhs_elem_type));

  auto result_type = mlir::RankedTensorType::get(
      result_shape, mlir::getElementType(ctx, comp_dtype));

  return mlir::stablehlo::DotGeneral(
      result_type, lhs, rhs, dot_dimension_numbers, precision_config_attr);
}

// Builds the StableHLO operations for scaled matrix multiplication.
// It performs matrix multiplication, applies scales, adds bias, and applies
// result scale.
//
// The high-level mathematical formula is:
//   result = (lhs @ rhs) * (scale_a * scale_b) + bias
// If scale_result is present and not folded:
//   result = ((lhs @ rhs) * (scale_a * scale_b) + bias) * scale_result
//
absl::StatusOr<mlir::MlirOp> BuildScaledMmShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::MlirOp scale_a,
    mlir::MlirOp scale_b, std::optional<mlir::MlirOp> bias,
    std::optional<mlir::MlirOp> scale_result,
    std::optional<mlir::Type> out_dtype, mlir::stablehlo::Precision precision) {
  // 1. Perform matrix multiplication with inferred computation dtype.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mm_result,
                      BuildStablehloDotGeneral(lhs, rhs, precision));

  // Cast mm_result to F32 to avoid type mismatch with scales (which are F32).
  TT_ASSIGN_OR_RETURN(mm_result,
                      CastIfNeeded(mm_result, mlir::ElementType::F32));

  // 2. Apply scales efficiently.
  TT_ASSIGN_OR_RETURN(auto broadcasted_scales,
                      ApplyBroadcastIfNeeded<2>({scale_a, scale_b}));

  mlir::MlirOp combined_scale =
      mlir::stablehlo::Mul(broadcasted_scales[0], broadcasted_scales[1]);

  // Optionally fold scale_result if there is no bias
  bool folded_scale_result = false;
  if (!bias.has_value() && scale_result.has_value()) {
    TT_ASSIGN_OR_RETURN(
        auto bcast_scale_result,
        ApplyBroadcastIfNeeded<2>({combined_scale, *scale_result}));

    combined_scale =
        mlir::stablehlo::Mul(bcast_scale_result[0], bcast_scale_result[1]);
    folded_scale_result = true;
  }

  // Multiply the combined scale with mm_result once
  TT_ASSIGN_OR_RETURN(auto broadcasted_mm_scale,
                      ApplyBroadcastIfNeeded<2>({mm_result, combined_scale}));

  mlir::MlirOp scaled_result =
      mlir::stablehlo::Mul(broadcasted_mm_scale[0], broadcasted_mm_scale[1]);

  // 3. Add bias if present.
  if (bias.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp bias_f32,
                        CastIfNeeded(*bias, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(auto broadcasted_bias,
                        ApplyBroadcastIfNeeded<2>({scaled_result, bias_f32}));

    scaled_result =
        mlir::stablehlo::Add(broadcasted_bias[0], broadcasted_bias[1]);
  }

  // 4. Apply result scale if present (and not folded).
  if (scale_result.has_value() && !folded_scale_result) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_result_f32,
                        CastIfNeeded(*scale_result, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(
        auto broadcasted_scale_result,
        ApplyBroadcastIfNeeded<2>({scaled_result, scale_result_f32}));

    scaled_result = mlir::stablehlo::Mul(broadcasted_scale_result[0],
                                         broadcasted_scale_result[1]);
  }

  // 5. Cast to out_dtype if present.
  if (out_dtype.has_value()) {
    scaled_result =
        mlir::stablehlo::ConvertElementType(scaled_result, *out_dtype);
  }

  return scaled_result;
}

absl::Status CheckScaledMmInputs(const at::Tensor& self, const at::Tensor& mat2,
                                 const at::Tensor& scale_a,
                                 const at::Tensor& scale_b,
                                 const std::optional<at::Tensor>& bias,
                                 const std::optional<at::Tensor>& scale_result,
                                 bool use_fast_accum) {
  TT_RET_CHECK(!use_fast_accum, error::kUnimplemented)
      << "use_fast_accum=true is not supported yet on TPU";

  TT_RETURN_IF_ERROR(CheckIsMatrix(self, "self"));
  TT_RETURN_IF_ERROR(CheckIsMatrix(mat2, "mat2"));
  TT_RET_CHECK(self.size(1) == mat2.size(0), error::kInvalidArgument)
      << "expected column size of first matrix to match row size of second "
         "matrix, got shapes "
      << ToString(self.sizes()) << " and " << ToString(mat2.sizes());

  // For now, we require dimensions to be divisible by 16. This is likely due to
  // TPU hardware alignment requirements or StableHLO lowering constraints for
  // efficient execution.
  TT_RET_CHECK(self.size(0) % 16 == 0 && self.size(1) % 16 == 0 &&
                   mat2.size(1) % 16 == 0,
               error::kInvalidArgument)
      << "expected matrix sizes to be divisible by 16, got shapes "
      << ToString(self.sizes()) << " and " << ToString(mat2.sizes());

  // For now, we only support tensorwise scaling (scales are scalars).
  TT_RET_CHECK(scale_a.numel() == 1, error::kUnimplemented)
      << "expected scale_a to have numel 1, got numel " << scale_a.numel();
  TT_RET_CHECK(scale_b.numel() == 1, error::kUnimplemented)
      << "expected scale_b to have numel 1, got numel " << scale_b.numel();

  if (scale_result.has_value() && scale_result->defined()) {
    TT_RET_CHECK(scale_result->numel() == 1, error::kUnimplemented)
        << "expected scale_result to have numel 1, got numel "
        << scale_result->numel();
  }

  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> ScaledMm(
    const at::Tensor& self, const at::Tensor& mat2, const at::Tensor& scale_a,
    const at::Tensor& scale_b, std::optional<at::Tensor> bias,
    std::optional<at::Tensor> scale_result,
    std::optional<at::ScalarType> out_dtype, bool use_fast_accum,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(CheckScaledMmInputs(self, mat2, scale_a, scale_b, bias,
                                         scale_result, use_fast_accum));

  Dimensions output_dims = {self.size(0), mat2.size(1)};
  at::ScalarType target_scalar_type =
      out_dtype.has_value() ? *out_dtype : self.scalar_type();
  TT_ASSIGN_OR_RETURN(mlir::ElementType target_elem_dtype,
                      ConvertTo<mlir::ElementType>(target_scalar_type));

  TT_ASSIGN_OR_RETURN(mlir::ElementType self_elem_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(mlir::ElementType comp_dtype,
                      InferComputationDtype(self_elem_dtype));

  const mlir::stablehlo::Precision current_precision = GetPrecision();
  TT_RETURN_IF_ERROR(param_keys.SetParam("precision", current_precision));

  bool has_bias = bias.has_value() && bias->defined();
  bool has_scale_result = scale_result.has_value() && scale_result->defined();

  std::vector<at::Tensor> inputs = {self, mat2, scale_a, scale_b};
  if (has_bias) {
    inputs.push_back(*bias);
  }
  if (has_scale_result) {
    inputs.push_back(*scale_result);
  }

  auto op_builder =
      [has_bias, has_scale_result, target_elem_dtype, current_precision](
          absl::Span<mlir::MlirOp> builder_inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp lhs_op = builder_inputs[0];
    mlir::MlirOp rhs_op = builder_inputs[1];
    mlir::MlirOp scale_a_op = builder_inputs[2];
    mlir::MlirOp scale_b_op = builder_inputs[3];

    std::optional<mlir::MlirOp> bias_op;
    std::optional<mlir::MlirOp> scale_result_op;

    if (has_bias) {
      bias_op = builder_inputs[4];
    }
    if (has_scale_result) {
      scale_result_op = builder_inputs[has_bias ? 5 : 4];
    }

    TT_ASSIGN_OR_RETURN(std::optional<mlir::Type> out_dtype_mlir,
                        GetMlirType(builder.getContext(), target_elem_dtype));

    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp result,
        BuildScaledMmShlo(lhs_op, rhs_op, scale_a_op, scale_b_op, bias_op,
                          scale_result_op, out_dtype_mlir, current_precision));

    return result;
  };

  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef result_buf,
      DispatchOp<kDynamicSize>(std::move(op_builder), inputs,
                               {.out_dtype = target_elem_dtype,
                                .out_dims = output_dims,
                                .computation_dtype = comp_dtype,
                                .op_param_cache_keys = std::move(param_keys)}));

  return result_buf;
}

}  // namespace

at::Tensor AtenScaledMm(const at::Tensor& self, const at::Tensor& mat2,
                        const at::Tensor& scale_a, const at::Tensor& scale_b,
                        const std::optional<at::Tensor>& bias,
                        const std::optional<at::Tensor>& scale_result,
                        std::optional<at::ScalarType> out_dtype,
                        bool use_fast_accum) {
  TT_KERNEL(
      OpName::kScaledMm, param_keys,
      (self, mat2, scale_a, scale_b, bias, scale_result, out_dtype,
       use_fast_accum),
      {
        at::ScalarType target_scalar_type =
            out_dtype.has_value() ? *out_dtype : self.scalar_type();
        TT_ASSIGN_OR_THROW(at::Tensor out,
                           MakeEmptyTensor({self.size(0), mat2.size(1)},
                                           target_scalar_type, self.device()));
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buf,
            ScaledMm(self, mat2, scale_a, scale_b, bias, scale_result,
                     out_dtype, use_fast_accum, std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor& AtenScaledMmOut(const at::Tensor& self, const at::Tensor& mat2,
                            const at::Tensor& scale_a,
                            const at::Tensor& scale_b,
                            const std::optional<at::Tensor>& bias,
                            const std::optional<at::Tensor>& scale_result,
                            std::optional<at::ScalarType> out_dtype,
                            bool use_fast_accum, at::Tensor& out) {
  TT_KERNEL(
      OpName::kScaledMmOut, param_keys,
      (self, mat2, scale_a, scale_b, bias, scale_result, out_dtype,
       use_fast_accum, out),
      {
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buf,
            ScaledMm(self, mat2, scale_a, scale_b, bias, scale_result,
                     out_dtype, use_fast_accum, std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
