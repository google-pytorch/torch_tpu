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

#include "torch_tpu/ops/uniform/uniform.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

// Converts random bits to a uniform distribution in [from, to). The idea is
// to reinterpret as float types, and set the exponent bits to logically
// represent 1.
absl::StatusOr<mlir::MlirOp> BitsToUniform(mlir::MlirOp random_bits,
                                           mlir::MlirOp from, mlir::MlirOp to) {
  auto& builder = random_bits.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  // We will create a f64, f32 or f16 depending on the bit width of random bits,
  // and then cast it to the output type.
  mlir::MlirOp clear_exponent_mask;
  mlir::MlirOp set_exp_to_bias_mask;
  mlir::RankedTensorType random_bits_as_float_type;
  mlir::RankedTensorType from_type = GetTensorTypeOrDie(from);
  // Dispatch on element type rather than bit width (bf16 and f16 are both
  // 16-bit but have different layouts).
  auto from_elem_type = from_type.getElementType();
  if (from_elem_type.isF64()) {
    clear_exponent_mask =
        MakeConstantLike(random_bits, 0x000F'FFFF'FFFF'FFFFUL);
    set_exp_to_bias_mask =
        MakeConstantLike(random_bits, 0x3FF0'0000'0000'0000UL);
    random_bits_as_float_type = from_type.clone(op_builder.getF64Type());
  } else if (from_elem_type.isF32()) {
    // f32: 8 exp, 23 mantissa, bias=127
    clear_exponent_mask = MakeConstantLike(random_bits, 0x007F'FFFFUL);
    set_exp_to_bias_mask = MakeConstantLike(random_bits, 0x3F80'0000UL);
    random_bits_as_float_type = from_type.clone(op_builder.getF32Type());
  } else if (from_elem_type.isBF16()) {
    // bf16: 8 exp, 7 mantissa, bias=127
    clear_exponent_mask = MakeConstantLike(random_bits, 0x007FU);
    set_exp_to_bias_mask = MakeConstantLike(random_bits, 0x3F80U);
    random_bits_as_float_type = from_type.clone(op_builder.getBF16Type());
  } else if (from_elem_type.isF16()) {
    // f16: 5 exp, 10 mantissa, bias=15
    clear_exponent_mask = MakeConstantLike(random_bits, 0x03FFU);
    set_exp_to_bias_mask = MakeConstantLike(random_bits, 0x3C00U);
    random_bits_as_float_type = from_type.clone(op_builder.getF16Type());
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "unsupported float type for uniform";
  }

  // Float is 1 sign bit, k exponent bits, and n mantissa bits. Exponent is
  // an unsigned integer minus a bias; setting it to bias gives 1.0.
  //   f64:  k=11, n=52, bias=1023 →
  //   mask=0x000F'FFFF'FFFF'FFFF, 1.0=0x3FF0'0000'0000'0000 f32:  k=8,  n=23,
  //   bias=127  → mask=0x007F'FFFF,           1.0=0x3F80'0000 bf16: k=8,  n=7,
  //   bias=127  → mask=0x007F,                1.0=0x3F80 f16:  k=5,  n=10,
  //   bias=15   → mask=0x03FF,                1.0=0x3C00
  mlir::MlirOp random_mantissa =
      stablehlo::And(random_bits, clear_exponent_mask);

  mlir::MlirOp random_between_one_and_two =
      stablehlo::Or(random_mantissa, set_exp_to_bias_mask);

  // Interpret as floats
  random_between_one_and_two = stablehlo::BitcastConvert(
      random_bits_as_float_type, random_between_one_and_two);
  random_between_one_and_two =
      stablehlo::Convert(from_type, random_between_one_and_two);
  // Subtract 1.0
  mlir::MlirOp one_const = MakeConstantLike(random_between_one_and_two, 1.0);
  mlir::MlirOp random_between_zero_and_one =
      stablehlo::Subtract(random_between_one_and_two, one_const);
  // Scale to [from, to)
  auto diff_op = stablehlo::Subtract(to, from);
  auto scaled_op = stablehlo::Mul(diff_op, random_between_zero_and_one);
  return stablehlo::Add(from, scaled_op);
}

absl::StatusOr<MlirOpResults<2>> BuildUniformShlo(
    mlir::MlirOp rng_input_state, const double from, const double to,
    const llvm::ArrayRef<int64_t> sizes, const mlir::ElementType mlir_type) {
  const mlir::RankedTensorType rng_input_state_type =
      GetTensorTypeOrDie(rng_input_state);
  auto& builder = rng_input_state.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType output_tensor_type =
      makeTensorType(builder.getContext(), sizes, mlir_type);
  mlir::MlirOp from_op = MakeConstant(builder, from, output_tensor_type);
  mlir::MlirOp to_op = MakeConstant(builder, to, output_tensor_type);
  int64_t bit_width =
      output_tensor_type.getElementType().getIntOrFloatBitWidth();
  mlir::RankedTensorType rng_bits_type =
      output_tensor_type.clone(op_builder.getIntegerType(bit_width,
                                                         /*isSigned=*/false));
  const mlir::stablehlo::RngAlgorithmAttr rng_alg =
      stablehlo::RngAlgorithmAttr::get(op_builder.getContext(),
                                       stablehlo::RngAlgorithm::DEFAULT);
  auto rng_bits_op = stablehlo::RngBitGeneratorOp::create(
      op_builder, rng_input_state.getValue().getLoc(), rng_input_state_type,
      rng_bits_type, rng_alg, rng_input_state.getValue());

  mlir::MlirOp rng_output_state =
      mlir::MlirOp(builder, rng_bits_op.getOutputState());
  mlir::MlirOp rng_output_op = mlir::MlirOp(builder, rng_bits_op.getOutput());
  TT_ASSIGN_OR_RETURN(auto result,
                      BitsToUniform(rng_output_op, from_op, to_op));
  return {{rng_output_state, result}};
}

}  // namespace torch_tpu
