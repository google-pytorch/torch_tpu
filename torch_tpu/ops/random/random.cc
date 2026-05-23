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

#include "torch_tpu/ops/random/random.h"

#include <cstdint>
#include <limits>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// Returns a random integer in [from, to) as follows:
// 0. If to - from fits in a u32, use 32 bits, otherwise use 64 bits.
// 1. Compute the difference between to and from.
// 2. Broadcast this difference and use it to mod the random integers (either
// u32 or u64).
// 3. Add the from value to the result.
// 4. Convert to the output type.
//
// TODO(unda): We are currently applying an optimization when the range is a
// power of 2. It would be better if this was done in the XLA backend
// implementation instead, but this currently won't happen as the x64 expansion
// happens first.
//
// Note that this results in a slightly non-uniform distribution; we are loosely
// following what jax does here:
// https://github.com/jax-ml/jax/blob/b6996cbda6df4f9aac8379e570bb26531cfff530/jax/_src/random.py#L511

absl::StatusOr<mlir::MlirOp> BitsToRandomIntegersInRange(
    mlir::MlirOp random_bits, mlir::MlirOp diff, bool range_is_single_bit) {
  const mlir::RankedTensorType random_bits_type =
      GetTensorTypeOrDie(random_bits);
  const mlir::RankedTensorType diff_type = GetTensorTypeOrDie(diff);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Caller makes sure `random_bits` is
                 // of the same type as `diff`.
      random_bits_type.getElementType() == diff_type.getElementType(),
      error::kInvalidArgument)
      << "random bits and diff must be the same type, got "
      << mlir::debugString(random_bits_type) << " vs "
      << mlir::debugString(diff_type);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Caller makes sure `diff` is a constant.
      diff_type.getRank() == 0, error::kInvalidArgument)
      << "diff must be a scalar, got " << mlir::debugString(diff_type);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Python API function signature
                 // mismatches (it explicitly allows only integers) before.
      diff_type.getElementType().isInteger(), error::kInvalidArgument)
      << "diff must be an integer type, got " << mlir::debugString(diff_type);
  // Broadcast the difference and use it to mod the random u64s.
  diff = mlir::stablehlo::BroadcastInDim(random_bits_type, diff, {});
  // modulo_op entries in [0, diff_op)

  if (range_is_single_bit) {
    mlir::MlirOp one =
        MakeScalarConstant(diff.getBuilder(), 1, diff_type.getElementType());
    mlir::MlirOp broadcasted_one =
        mlir::stablehlo::BroadcastInDim(random_bits_type, one, {});
    mlir::MlirOp mask = mlir::stablehlo::Subtract(diff, broadcasted_one);
    return mlir::stablehlo::And(random_bits, mask);
  }
  return mlir::stablehlo::Rem(random_bits, diff);
}

absl::StatusOr<mlir::MlirOp> AddLowerBound(
    mlir::MlirOp random_in_range_integers, mlir::RankedTensorType output_type,
    mlir::MlirOp lower_bound) {
  const mlir::RankedTensorType lower_bound_type =
      GetTensorTypeOrDie(lower_bound);

  // broadcast the from value
  auto broadcasted_from_type = lower_bound_type.clone(output_type.getShape());
  auto broadcasted_from =
      mlir::stablehlo::BroadcastInDim(broadcasted_from_type, lower_bound, {});
  // Convert modulo op to from type
  auto signed_modulo_op =
      mlir::stablehlo::Convert(broadcasted_from_type, random_in_range_integers);
  auto in_range_op = mlir::stablehlo::Add(broadcasted_from, signed_modulo_op);
  return mlir::stablehlo::Convert(output_type, in_range_op);
}

absl::StatusOr<MlirOpResults<2>> BuildRandomShlo(mlir::MlirOp state,
                                                 Dimensions dims,
                                                 mlir::ElementType output_dtype,
                                                 int64_t from, int64_t to) {
  ABSL_VLOG(3) << "BuildRandomShlo: from=" << from << ", to=" << to;
  const auto rng_input_state_type = GetTensorTypeOrDie(state);
  auto& ctx = state.getContext();
  auto& builder = state.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error already caught in caller.
      to > from, error::kInvalidArgument)
      << "to must be greater than from, got to=" << to << ", from=" << from;
  // Cast _before_ computing the difference to avoid overflow.
  // Note that computations in uint64 are done (mod 2^64), so
  // after casting we get (x - y) % 2^64, but the unrestricted value is already
  // in [0, 2^64).
  const uint64_t range =
      static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
  // We will have to mod by `range`, which is slow in the 64-bit case, so if we
  // can, we do this in 32 bits.
  bool range_fits_in_u32 = range <= std::numeric_limits<uint32_t>::max();
  bool range_is_power_of_2 = false;
  // We know range > 0.
  if ((range & (range - 1)) == 0) {
    range_is_power_of_2 = true;
  }

  int target_bit_width = range_fits_in_u32 ? 32 : 64;
  mlir::IntegerType diff_type =
      op_builder.getIntegerType(target_bit_width, /*isSigned=*/false);
  auto diff_op = MakeScalarConstant(builder, to - from, diff_type);

  // Call rng bit generator
  auto input_shape_with_diff_type = mlir::makeTensorType(ctx, dims, diff_type);
  auto rng_op = mlir::stablehlo::RngBitGeneratorOp::create(
      op_builder, state.getValue().getLoc(), rng_input_state_type,
      input_shape_with_diff_type,
      mlir::stablehlo::RngAlgorithmAttr::get(
          op_builder.getContext(), mlir::stablehlo::RngAlgorithm::DEFAULT),
      state.getValue());

  mlir::MlirOp rng_output_state =
      mlir::MlirOp(builder, rng_op.getOutputState());
  mlir::MlirOp rng_output_op = mlir::MlirOp(builder, rng_op.getOutput());
  auto output_tensor_type = mlir::makeTensorType(ctx, dims, output_dtype);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch `check_from_to_in_range()`
                 // function already catches this error.
      output_tensor_type.getElementType().isInteger() ||
          output_tensor_type.getElementType().isFloat(),
      error::kInvalidArgument)
      << "output type must be an integer or float, got "
      << mlir::debugString(output_tensor_type);
  mlir::MlirOp from_op =
      MakeScalarConstant(builder, from, mlir::ElementType::I64);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp numbers_in_range,
      BitsToRandomIntegersInRange(rng_output_op, diff_op, range_is_power_of_2));
  TT_ASSIGN_OR_RETURN(auto result, AddLowerBound(numbers_in_range,
                                                 output_tensor_type, from_op));
  return {{rng_output_state, result}};
}

}  // namespace torch_tpu
