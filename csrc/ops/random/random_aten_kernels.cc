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

#include "csrc/ops/random/random_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/Context.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/native/DistributionTemplates.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/util/Optional.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/rng_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/util/BFloat16.h"
#include "torch/headeronly/util/Half.h"

namespace torch_tpu {

namespace {

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

absl::StatusOr<MlirOpResults<1>> BuildRandomShlo(mlir::MlirOp state,
                                                 Dimensions dims,
                                                 mlir::ElementType output_dtype,
                                                 int64_t from, uint64_t range) {
  ABSL_VLOG(3) << "BuildRandomShlo: from=" << from << ", range=" << range;
  const auto rng_input_state_type = GetTensorTypeOrDie(state);
  auto& ctx = state.getContext();
  auto& builder = state.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error already caught in caller.
      range > 0 || from == std::numeric_limits<int64_t>::lowest(),
      error::kInvalidArgument)
      << "expected range to be positive (or 0 mod 2^64 when from is "
         "INT64_MIN), got range="
      << range << ", from=" << from;
  // We will have to mod by `range`, which is slow in the 64-bit case, so if we
  // can, we do this in 32 bits.
  bool range_fits_in_u32 =
      range > 0 && range <= std::numeric_limits<uint32_t>::max();
  bool range_is_power_of_2 = false;
  // We know range > 0.
  if ((range & (range - 1)) == 0) {
    range_is_power_of_2 = true;
  }

  int target_bit_width = range_fits_in_u32 ? 32 : 64;
  mlir::IntegerType diff_type =
      op_builder.getIntegerType(target_bit_width, /*isSigned=*/false);
  auto diff_op =
      MakeScalarConstant(builder, static_cast<int64_t>(range), diff_type);

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
  return result;
}

struct RandomRange {
  int64_t from;
  uint64_t range;
};

NAryMlirOpBuilder<1, 1> GetRandomFunctional(Dimensions dims,
                                            mlir::ElementType output_dtype,
                                            int64_t from, uint64_t range) {
  return [dims, output_dtype, from,
          range](mlir::MlirOp rng_state) -> absl::StatusOr<MlirOpResults<1>> {
    return BuildRandomShlo(rng_state, dims, output_dtype, from, range);
  };
}

absl::Status UnsupportedFloatDtypeError(const at::ScalarType dtype) {
  return TT_ERROR(error::kInvalidArgument)
         << "expected the float type to be one of float64, float32, "
            "bfloat16, or float16, got "
         << ToString(dtype);
}

absl::Status ValidateFloatDtype(const at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Double:
    case at::ScalarType::Float:
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
      return absl::OkStatus();
    default:
      return UnsupportedFloatDtypeError(dtype);
  }
}

template <typename T>
void ApplyFloatBoundUpdates(int64_t& from, c10::optional<int64_t>& to) {
  if constexpr (std::numeric_limits<T>::max_exponent > 63) {
    // Largest value of type T strictly below 2^63 (and thus representable in
    // signed int64_t). Above this value, the next representable T is 2^63,
    // which overflows int64_t inside update_from / update_to.
    constexpr double kTwoPow63 = static_cast<double>(uint64_t{1} << 63);
    constexpr int64_t kMaxBelowTwoPow63 = static_cast<int64_t>(
        (uint64_t{1} << 63) -
        (uint64_t{1} << (63 - std::numeric_limits<T>::digits)));
    if (from > kMaxBelowTwoPow63) {
      from = std::numeric_limits<int64_t>::max();
    } else {
      from = at::native::templates::update_from<T>(from);
    }
    if (to.has_value()) {
      if (static_cast<double>(static_cast<T>(*to - 1)) >= kTwoPow63) {
        *to = kMaxBelowTwoPow63;
      } else {
        *to = at::native::templates::update_to<T>(*to);
      }
    }
    return;
  }
  from = at::native::templates::update_from<T>(from);
  if (to.has_value()) {
    *to = at::native::templates::update_to<T>(*to);
  }
}

absl::Status UpdateFloatBounds(const at::ScalarType dtype, int64_t& from,
                               c10::optional<int64_t>& to) {
  switch (dtype) {
    case at::ScalarType::Double:
      ApplyFloatBoundUpdates<double>(from, to);
      return absl::OkStatus();
    case at::ScalarType::Float:
      ApplyFloatBoundUpdates<float>(from, to);
      return absl::OkStatus();
    case at::ScalarType::Half:
      ApplyFloatBoundUpdates<c10::Half>(from, to);
      return absl::OkStatus();
    case at::ScalarType::BFloat16:
      ApplyFloatBoundUpdates<c10::BFloat16>(from, to);
      return absl::OkStatus();
    default:
      return UnsupportedFloatDtypeError(dtype);
  }
}

// See https://docs.pytorch.org/docs/stable/generated/torch.Tensor.random_.html
// for the logic behind this.
// Reusing logic from
// torch/aten/src/ATen/native/DistributionTemplates.h
absl::StatusOr<int64_t> ComputeToIncValue(const at::ScalarType dtype) {
  ABSL_VLOG(3) << "ComputeToIncValue: " << dtype;
  switch (dtype) {
    case at::ScalarType::Double:
      return int64_t{1} << std::numeric_limits<double>::digits;
    case at::ScalarType::Float:
      return int64_t{1} << std::numeric_limits<float>::digits;
    case at::ScalarType::Half:
      return int64_t{1} << std::numeric_limits<c10::Half>::digits;
    case at::ScalarType::BFloat16:
      return int64_t{1} << std::numeric_limits<c10::BFloat16>::digits;
    case at::ScalarType::Bool:
      return static_cast<int64_t>(true);
    case at::ScalarType::Byte:
      return static_cast<int64_t>(std::numeric_limits<uint8_t>::max());
    case at::ScalarType::Char:
      return static_cast<int64_t>(std::numeric_limits<int8_t>::max());
    case at::ScalarType::Short:
      return static_cast<int64_t>(std::numeric_limits<int16_t>::max());
    case at::ScalarType::Int:
      return static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    case at::ScalarType::Long:
      return std::numeric_limits<int64_t>::max();
    case at::ScalarType::UInt16:
      return static_cast<int64_t>(std::numeric_limits<uint16_t>::max());
    case at::ScalarType::UInt32:
      return static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    case at::ScalarType::UInt64:
      return static_cast<int64_t>(std::numeric_limits<uint64_t>::max());
    default:
      if (at::isFloatingType(dtype)) {
        return UnsupportedFloatDtypeError(dtype);
      }
      return TT_ERROR(error::kInvalidArgument)
             << "expected the input dtype to be integral, floating point, or "
                "bool, got "
             << ToString(dtype);
  }
}

absl::StatusOr<RandomRange> ComputeRandomRange(int64_t from,
                                               c10::optional<int64_t> to_opt,
                                               at::ScalarType dtype) {
  if (to_opt.has_value()) {
    TT_RET_CHECK(from < *to_opt, error::kInvalidArgument)
        << "expected 'from' to be < 'to', got " << from << " vs " << *to_opt;
    if (at::isFloatingType(dtype)) {
      TT_RETURN_IF_ERROR(ValidateFloatDtype(dtype));
    }
    at::native::templates::check_from_to_in_range(
        from, *to_opt - 1, c10::scalarTypeToTypeMeta(dtype));
    if (at::isFloatingType(dtype)) {
      TT_RETURN_IF_ERROR(UpdateFloatBounds(dtype, from, to_opt));
      TT_RET_CHECK(from < *to_opt, error::kInvalidArgument)
          << "expected 'from' casted to dtype to be less than 'to' casted to "
             "dtype, got from="
          << from << " >= to=" << *to_opt;
    }
    const int64_t to = *to_opt;
    const uint64_t range =
        static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    return RandomRange{from, range};
  }
  TT_ASSIGN_OR_RETURN(const int64_t to_inc, ComputeToIncValue(dtype));
  at::native::templates::check_from_to_in_range(
      from, to_inc, c10::scalarTypeToTypeMeta(dtype));
  if (from != std::numeric_limits<int64_t>::lowest() &&
      at::isFloatingType(dtype)) {
    if (from < to_inc) {
      TT_RETURN_IF_ERROR(UpdateFloatBounds(dtype, from, to_opt));
    }
    TT_RET_CHECK(from < to_inc, error::kInvalidArgument)
        << "expected 'from' casted to dtype to be less than or equal to "
           "'to_inc' casted to dtype, got from="
        << from << " > to_inc=" << to_inc;
  }
  if (from == std::numeric_limits<int64_t>::lowest()) {
    return RandomRange{from, 0};
  }
  const uint64_t range =
      static_cast<uint64_t>(to_inc) - static_cast<uint64_t>(from) + 1;
  return RandomRange{from, range};
}

absl::Status Random(at::Tensor& self, c10::optional<at::Generator> generator,
                    int64_t from, c10::optional<int64_t> to_opt,
                    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const RandomRange random_range,
                      ComputeRandomRange(from, to_opt, self.scalar_type()));
  if (self.numel() == 0) {
    return absl::OkStatus();
  }
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  auto dims = CopyIntVector(self.sizes());

  return DispatchRngOp(
      self, generator,
      [&](at::Tensor rng_input_state)  // REF_CAPTURE_OK=called synchronously
      -> absl::StatusOr<std::vector<DeviceBufferRef>> {
        TT_ASSIGN_OR_RETURN(
            auto buf,
            (DispatchOp<1, 1>(
                GetRandomFunctional(dims, output_dtype, random_range.from,
                                    random_range.range),
                {rng_input_state},
                {.out_dtype = output_dtype,
                 .out_dims = self.sizes(),
                 .op_param_cache_keys = std::move(param_keys)})));
        return std::vector<DeviceBufferRef>{std::move(buf)};
      });
}

}  // namespace

at::Tensor& AtenRandom_(at::Tensor& self,
                        c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kRandom_, param_keys, (self, generator), {
    TT_THROW_IF_ERROR(
        Random(self, generator, 0, std::nullopt, std::move(param_keys)));
    return self;
  });
}

at::Tensor& AtenRandom_From(at::Tensor& self, int64_t from,
                            c10::optional<int64_t> to,
                            c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kRandom_From, param_keys, (self, from, to, generator), {
    TT_THROW_IF_ERROR(Random(self, generator, from, to, std::move(param_keys)));
    return self;
  });
}

at::Tensor& AtenRandom_To(at::Tensor& self, int64_t to,
                          c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kRandom_To, param_keys, (self, to, generator), {
    TT_THROW_IF_ERROR(Random(self, generator, 0, to, std::move(param_keys)));
    return self;
  });
}

}  // namespace torch_tpu
