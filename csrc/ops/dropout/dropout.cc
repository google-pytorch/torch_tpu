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

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Bit width of ui16 uniform PRNG words.
constexpr unsigned kU16Bits = 16;
// Exact 2^16 sample space cardinality of ui16 words.
constexpr uint64_t kU16Cardinality = uint64_t{1} << kU16Bits;
// Bit width of ui32 uniform PRNG words.
constexpr unsigned kU32Bits = 32;
// Exact 2^32 sample space cardinality of ui32 words.
constexpr uint64_t kU32Cardinality = uint64_t{1} << kU32Bits;
// Binary16 mantissa bits bounding relative probability quantization error.
constexpr unsigned kBinary16MantissaBits = 10;
// IEEE 754 binary16 machine epsilon (2^-10) for probability quantization.
constexpr double kMaxRelativeProbError =
    1.0 / static_cast<double>(uint64_t{1} << kBinary16MantissaBits);

bool IsAccurateQuantizedDropProb(int64_t rounded_thresh, uint64_t cardinality,
                                 double p) {
  if (rounded_thresh <= 0 ||
      static_cast<uint64_t>(rounded_thresh) >= cardinality) {
    return false;
  }
  const double quantized_p =
      static_cast<double>(rounded_thresh) / static_cast<double>(cardinality);
  const double min_tail = std::min(p, 1.0 - p);
  return std::abs(quantized_p - p) <= min_tail * kMaxRelativeProbError;
}

}  // namespace

absl::StatusOr<MlirOpResults<2>> BuildDropoutTrainShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp input, double p) {
  ABSL_VLOG(1) << "[BuildDropoutTrainShlo] input: "
               << mlir::debugString(input.getValue()) << ", p: " << p;
  ABSL_CHECK(p >= 0.0 && p <= 1.0)  // CRASH_OK=Caller validates p.
      << "expected p to be in the range [0, 1], got " << p;

  auto& builder = input.getBuilder();
  auto& op_builder = builder.getOpBuilder();

  if (p <= 0.0) {
    auto mask_op = MakeConstantLike(input, true, op_builder.getI1Type());
    return {{input, mask_op}};
  }
  if (p >= 1.0) {
    auto zero_const = MakeConstantLike(input, 0.0);
    auto mask_op = MakeConstantLike(input, false, op_builder.getI1Type());
    return {{zero_const, mask_op}};
  }

  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::RankedTensorType rand_type = input_type;
  if (auto complex_type =
          mlir::dyn_cast<mlir::ComplexType>(input_type.getElementType())) {
    rand_type = input_type.clone(complex_type.getElementType());
  }

  const int64_t thresh_u16 =
      std::llround(p * static_cast<double>(kU16Cardinality));
  const int64_t thresh_u32 =
      std::llround(p * static_cast<double>(kU32Cardinality));

  mlir::MlirOp mask_op;
  unsigned bits_width = 0;
  uint64_t threshold = 0;
  if (IsAccurateQuantizedDropProb(thresh_u16, kU16Cardinality, p)) {
    bits_width = kU16Bits;
    threshold = static_cast<uint64_t>(thresh_u16);
  } else if (IsAccurateQuantizedDropProb(thresh_u32, kU32Cardinality, p)) {
    bits_width = kU32Bits;
    threshold = static_cast<uint64_t>(thresh_u32);
  }

  if (bits_width != 0) {
    mlir::Type uint_elem_type =
        op_builder.getIntegerType(bits_width, /*isSigned=*/false);
    mlir::RankedTensorType rng_bits_type = rand_type.clone(uint_elem_type);
    const auto rng_alg = mlir::stablehlo::RngAlgorithmAttr::get(
        op_builder.getContext(), mlir::stablehlo::RngAlgorithm::DEFAULT);
    mlir::MlirOp rng_bits =
        RngBitGeneratorLike(rng_input_state, rng_bits_type, rng_alg, input)
            .output;
    mlir::MlirOp thresh_const = MakeConstantLike(rng_bits, threshold);
    mlir::Value mask_val = mlir::stablehlo::CompareOp::create(
        op_builder, input.getValue().getLoc(), rng_bits.getValue(),
        thresh_const.getValue(), mlir::stablehlo::ComparisonDirection::GE,
        mlir::stablehlo::ComparisonType::UNSIGNED);
    mask_op = mlir::MlirOp(builder, mask_val);
  } else {
    TT_ASSIGN_OR_RETURN(auto rand_op,
                        BuildUniformShlo(rng_input_state, /*from=*/0.0,
                                         /*to=*/1.0, rand_type, input));
    auto p_const = MakeConstantLike(rand_op, p);
    mask_op = mlir::stablehlo::Compare(
        rand_op, p_const, mlir::stablehlo::ComparisonDirection::GE);
  }

  auto zero_const = MakeConstantLike(input, 0.0);
  auto masked_input_op = mlir::stablehlo::Select(mask_op, input, zero_const);

  double scale = 1.0 / (1.0 - p);
  auto scale_const = MakeConstantLike(input, scale);
  auto output = mlir::stablehlo::Mul(masked_input_op, scale_const);

  // An optimization barrier on the returned mask prevents downstream
  // consumers (such as backward gradient lowerings) from fusing backwards
  // into the mask generation compare operation. Without this barrier, XLA's
  // instruction fusion duplicates the cheap compare(rng_bits, thresh) into
  // the backward computation, forcing the wider raw RNG bit buffer (u16/u32)
  // to remain live throughout forward execution into backward instead of
  // saving only the narrower 1-bit predicate mask tensor.
  auto barrier_ops = mlir::stablehlo::OptimizationBarrier(builder, {mask_op});
  return {{output, barrier_ops[0]}};
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

  TT_ASSIGN_OR_RETURN((auto [grad_output_bcst, mask_bcst]),
                      ApplyBroadcastIfNeeded(grad_output, mask));

  auto zero_const = MakeConstantLike(grad_output_bcst, 0.0);
  auto masked_grad_op =
      mlir::stablehlo::Select(mask_bcst, grad_output_bcst, zero_const);
  auto scale_const = MakeConstantLike(masked_grad_op, scale);
  return mlir::stablehlo::Mul(masked_grad_op, scale_const);
}

}  // namespace torch_tpu
