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

#include "csrc/ops/dropout/dropout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

constexpr unsigned kDropoutRngBits = 32;
constexpr uint64_t kU32Cardinality = uint64_t{1} << kDropoutRngBits;

}  // namespace

absl::StatusOr<MlirOpResults<2>> BuildDropoutTrainShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp input, double p,
    DropoutProbabilityType prob_type, DropoutMaskType mask_type) {
  ABSL_VLOG(1) << "[BuildDropoutTrainShlo] input: "
               << mlir::debugString(input.getValue()) << ", p: " << p;
  ABSL_CHECK(p >= 0.0 && p <= 1.0)  // CRASH_OK=Caller validates p.
      << "expected p to be in the range [0, 1], got " << p;

  const mlir::ElementType mask_elem_type =
      (mask_type == DropoutMaskType::kUInt8) ? mlir::ElementType::UI8
                                             : mlir::ElementType::PRED;

  const bool keep_all = (prob_type == DropoutProbabilityType::kDropProbability)
                            ? (p <= 0.0)
                            : (p >= 1.0);
  const bool drop_all = (prob_type == DropoutProbabilityType::kDropProbability)
                            ? (p >= 1.0)
                            : (p <= 0.0);

  if (keep_all) {
    mlir::MlirOp mask_op = MakeConstantLike(input, true, mask_elem_type);
    return {{input, mask_op}};
  }
  // Return all zeros on drop_all (including _fused_dropout at p == 0.0),
  // matching native_dropout(p=1.0) and avoiding CUDA's 1.0 / 0.0 -> nan.
  if (drop_all) {
    mlir::MlirOp zero_const = MakeConstantLike(input, 0.0);
    mlir::MlirOp mask_op = MakeConstantLike(input, false, mask_elem_type);
    return {{zero_const, mask_op}};
  }

  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  const int64_t rounded_threshold =
      std::llround(p * static_cast<double>(kU32Cardinality));
  const uint64_t threshold = static_cast<uint64_t>(std::clamp<int64_t>(
      rounded_threshold, 0, std::numeric_limits<uint32_t>::max()));
  const mlir::Type uint32_type =
      op_builder.getIntegerType(kDropoutRngBits, /*isSigned=*/false);
  const mlir::RankedTensorType rng_bits_type = input_type.clone(uint32_type);
  const auto rng_alg = mlir::stablehlo::RngAlgorithmAttr::get(
      op_builder.getContext(), mlir::stablehlo::RngAlgorithm::DEFAULT);
  mlir::MlirOp rng_bits =
      RngBitGeneratorLike(rng_input_state, rng_bits_type, rng_alg, input)
          .output;
  mlir::MlirOp thresh_const = MakeConstantLike(rng_bits, threshold);
  const mlir::stablehlo::ComparisonDirection comparison_dir =
      (prob_type == DropoutProbabilityType::kKeepProbability)
          ? mlir::stablehlo::ComparisonDirection::LT
          : mlir::stablehlo::ComparisonDirection::GE;
  const mlir::Value mask_val = mlir::stablehlo::CompareOp::create(
      op_builder, input.getValue().getLoc(), rng_bits.getValue(),
      thresh_const.getValue(), comparison_dir,
      mlir::stablehlo::ComparisonType::UNSIGNED);
  mlir::MlirOp mask_op(builder, mask_val);

  const mlir::ElementType elem_type = GetElementTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_elem_type,
                      InferComputationDtype(elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp cast_input,
                      CastIfNeeded(input, compute_elem_type));

  const double scale = (prob_type == DropoutProbabilityType::kKeepProbability)
                           ? (1.0 / p)
                           : (1.0 / (1.0 - p));
  mlir::MlirOp scaled;
  if (IsComplexType(GetTensorTypeOrDie(cast_input))) {
    mlir::MlirOp real = mlir::stablehlo::Real(cast_input);
    mlir::MlirOp imag = mlir::stablehlo::Imag(cast_input);
    mlir::MlirOp real_scale = MakeConstantLike(real, scale);
    mlir::MlirOp scaled_real = mlir::stablehlo::Mul(real, real_scale);
    mlir::MlirOp scaled_imag = mlir::stablehlo::Mul(imag, real_scale);
    scaled = mlir::stablehlo::Complex(scaled_real, scaled_imag);
  } else {
    mlir::MlirOp scale_const = MakeConstantLike(cast_input, scale);
    scaled = mlir::stablehlo::Mul(cast_input, scale_const);
  }
  mlir::MlirOp zero_const = MakeConstantLike(scaled, 0.0);
  mlir::MlirOp output = mlir::stablehlo::Select(mask_op, scaled, zero_const);

  if (compute_elem_type != elem_type) {
    TT_ASSIGN_OR_RETURN(output, CastIfNeeded(output, elem_type));
  }
  if (mask_type == DropoutMaskType::kUInt8) {
    mask_op =
        mlir::stablehlo::ConvertElementType(mask_op, mlir::ElementType::UI8);
  }

  return {{output, mask_op}};
}

absl::StatusOr<MlirOpResults<1>> BuildDropoutBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp mask, double scale) {
  ABSL_VLOG(1) << "[BuildDropoutBackwardShlo] grad_output: "
               << mlir::debugString(grad_output.getValue())
               << ", mask: " << mlir::debugString(mask.getValue())
               << ", scale: " << scale;

  mlir::MlirBuilder& builder = grad_output.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  // Convert grad_output tensor to float type if it is not already a float or
  // complex type.
  const mlir::RankedTensorType grad_output_type =
      GetTensorTypeOrDie(grad_output);
  if (!IsFloatType(grad_output_type) && !IsComplexType(grad_output_type)) {
    grad_output = mlir::stablehlo::ConvertElementType(grad_output,
                                                      op_builder.getF32Type());
  }

  // Treat non-boolean (uint8) masks as binary (mask != 0) rather than integer
  // multipliers (intentional divergence from CUDA masked_scale_kernel;
  // fused_dropout masks are 0/1).
  if (!IsBooleanType(GetTensorTypeOrDie(mask))) {
    mlir::MlirOp zero_mask = MakeConstantLike(mask, 0);
    mask = mlir::stablehlo::Compare(mask, zero_mask,
                                    mlir::stablehlo::ComparisonDirection::NE);
  }

  TT_ASSIGN_OR_RETURN((auto [grad_output_bcst, mask_bcst]),
                      ApplyBroadcastIfNeeded(grad_output, mask));

  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType elem_type,
      ConvertTo<mlir::ElementType>(
          GetTensorTypeOrDie(grad_output_bcst).getElementType()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_elem_type,
                      InferComputationDtype(elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp cast_grad,
                      CastIfNeeded(grad_output_bcst, compute_elem_type));

  // Scale real and imaginary parts separately for complex inputs so inf * 0.0
  // in a full complex multiply with (scale, 0.0) does not produce NaN.
  mlir::MlirOp scaled;
  if (IsComplexType(GetTensorTypeOrDie(cast_grad))) {
    mlir::MlirOp real = mlir::stablehlo::Real(cast_grad);
    mlir::MlirOp imag = mlir::stablehlo::Imag(cast_grad);
    mlir::MlirOp real_scale = MakeConstantLike(real, scale);
    mlir::MlirOp scaled_real = mlir::stablehlo::Mul(real, real_scale);
    mlir::MlirOp scaled_imag = mlir::stablehlo::Mul(imag, real_scale);
    scaled = mlir::stablehlo::Complex(scaled_real, scaled_imag);
  } else {
    mlir::MlirOp scale_const = MakeConstantLike(cast_grad, scale);
    scaled = mlir::stablehlo::Mul(cast_grad, scale_const);
  }
  // Select 0.0 for masked-out elements even when scale is inf (e.g.
  // fused_dropout backward at p=0.0) or grad_output is inf/nan, intentionally
  // avoiding CUDA's 0 * inf -> nan.
  mlir::MlirOp zero_const = MakeConstantLike(scaled, 0.0);
  mlir::MlirOp output = mlir::stablehlo::Select(mask_bcst, scaled, zero_const);

  if (compute_elem_type != elem_type) {
    TT_ASSIGN_OR_RETURN(output, CastIfNeeded(output, elem_type));
  }
  return output;
}

}  // namespace torch_tpu
