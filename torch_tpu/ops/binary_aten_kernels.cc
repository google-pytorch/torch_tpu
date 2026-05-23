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

#include "torch_tpu/ops/binary_aten_kernels.h"

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/ScalarOps.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/native/Resize.h"
#include "ATen/ops/div.h"
#include "ATen/ops/result_type.h"
#include "ATen/ops/sub.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;
namespace chlo = mlir::chlo;

namespace internal {

// When PyTorch executes mixed-device operations (e.g., `tpu_tensor +
// cpu_scalar`), the dispatcher bypasses the standard backend device guards. We
// must manually move 1-element CPU tensors to the TPU to avoid crashes when we
// later try to extract their TPU buffers.
absl::StatusOr<std::vector<at::Tensor>> MoveCpuScalarsToTpu(
    absl::Span<const at::Tensor> tensors) {
  std::vector<at::Tensor> tpu_tensors;
  tpu_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    // MakeTensor uses a cache to deduplicate scalar tensors.
    if (tensor.device().type() == c10::DeviceType::CPU && tensor.numel() == 1) {
      TT_ASSIGN_OR_RETURN(at::Tensor tpu_tensor, MakeTensor(tensor.item()));
      tpu_tensors.push_back(std::move(tpu_tensor));
    } else {
      tpu_tensors.push_back(tensor);
    }
  }
  return tpu_tensors;
}

absl::StatusOr<DeviceBufferRef> DispatchBinaryOp(const at::Tensor& self,
                                                 const at::Scalar& other,
                                                 MlirBinaryOpBuilder op_builder,
                                                 BinaryOpOptions opts) {
  // Lift the "other" scalar to a tensor, in either constant or variable mode.

  at::ScalarType result_type = at::result_type(self, other);
  TT_ASSIGN_OR_RETURN(at::Tensor other_tensor, MakeTensor(other, result_type));
  return DispatchBinaryOp(self, other_tensor, std::move(op_builder),
                          std::move(opts));
}

absl::StatusOr<DeviceBufferRef> DispatchBinaryOp(
    const at::Tensor& self, const at::Tensor& other,
    MlirBinaryOpBuilder bin_op_builder, BinaryOpOptions opts) {
  TT_ASSIGN_OR_RETURN(auto tpu_tensors, MoveCpuScalarsToTpu({self, other}));
  const at::Tensor& self_tpu = tpu_tensors[0];
  const at::Tensor& other_tpu = tpu_tensors[1];

  TT_ASSIGN_OR_RETURN(const Dimensions output_dims,
                      InferSize(self_tpu.sizes(), other_tpu.sizes()));

  at::ScalarType promoted_scalar_type = at::result_type(self_tpu, other_tpu);
  if (opts.force_float_inputs) {
    if (c10::isIntegralType(promoted_scalar_type, /*includeBool=*/true)) {
      promoted_scalar_type = c10::get_default_dtype_as_scalartype();
    }
  }
  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_dtype,
                      ToElementType(promoted_scalar_type));
  mlir::ElementType output_dtype = opts.output_dtype_override
                                       ? *opts.output_dtype_override
                                       : computation_dtype;

  auto op_builder = [bin_op_builder = std::move(bin_op_builder)](
                        FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    auto& [self_op, other_op] = inputs;
    return bin_op_builder(self_op, other_op);
  };
  return DispatchOp<2>(
      std::move(op_builder), {self_tpu, other_tpu},
      {.op_name = opts.op_name,
       .out_dtype = output_dtype,
       .out_dims = output_dims,
       .computation_dtype = computation_dtype,
       .op_param_cache_keys = std::move(opts.op_param_cache_keys),
       .split_mode = opts.split_mode});
}

absl::StatusOr<DeviceBufferRef> DispatchTernaryOp(
    const at::Tensor& self, const at::Tensor& other, const at::Tensor& third,
    MlirTernaryOpBuilder ternary_op_builder, TernaryOpOptions opts) {
  TT_ASSIGN_OR_RETURN(auto tpu_tensors,
                      MoveCpuScalarsToTpu({self, other, third}));
  const at::Tensor& self_tpu = tpu_tensors[0];
  const at::Tensor& other_tpu = tpu_tensors[1];
  const at::Tensor& third_tpu = tpu_tensors[2];

  TT_ASSIGN_OR_RETURN(
      const Dimensions output_dims,
      InferSize(self_tpu.sizes(), other_tpu.sizes(), third_tpu.sizes()));

  at::ScalarType promoted_scalar_type = at::result_type(self_tpu, other_tpu);

  if (opts.force_float_inputs) {
    if (c10::isIntegralType(promoted_scalar_type, /*includeBool=*/true)) {
      promoted_scalar_type = c10::get_default_dtype_as_scalartype();
    }
  }
  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_dtype,
                      ToElementType(promoted_scalar_type));
  mlir::ElementType output_dtype = opts.output_dtype_override
                                       ? *opts.output_dtype_override
                                       : computation_dtype;

  auto op_builder = [ternary_op_builder = std::move(ternary_op_builder)](
                        FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [self, other, third] = inputs;
    return ternary_op_builder(self, other, third);
  };
  return DispatchOp<3>(
      std::move(op_builder), {self_tpu, other_tpu, third_tpu},
      {.op_name = opts.op_name,
       .out_dtype = output_dtype,
       .out_dims = output_dims,
       .computation_dtype = computation_dtype,
       .op_param_cache_keys = std::move(opts.op_param_cache_keys),
       .split_mode = opts.split_mode});
}

}  // namespace internal

absl::Status FinalizeOpOutput(at::Tensor& out, DeviceBufferRef result_buf,
                              OutputOpMode mode) {
  if (mode == OutputOpMode::kInPlace) {
    TT_RET_CHECK(absl::MakeConstSpan(out.sizes()) == result_buf.dimensions(),
                 error::kInvalidArgument)
        << "output with shape " << ToString(result_buf.dimensions())
        << " doesn't match the broadcast shape of the tensor being operated on "
           "in-place, which has shape "
        << ToString(out.sizes());
  } else {
    at::native::resize_output(out, result_buf.dimensions());
  }

  return AssignBufferToAtTensor(std::move(result_buf), out);
}

absl::Status TernaryOpOut(const at::Tensor& self, const at::Tensor& other,
                          const at::Tensor& third, at::Tensor& out,
                          MlirTernaryOpBuilder op_builder,
                          TernaryOpOptions opts) {
  TT_RET_CHECK(out.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "the out tensor is expected to be on tpu, got " << out.device().str();
  if (!opts.output_dtype_override) {
    TT_ASSIGN_OR_RETURN(auto output_dtype,
                        ConvertTo<mlir::ElementType>(out.scalar_type()));
    opts.output_dtype_override = output_dtype;
  }
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      internal::DispatchTernaryOp(self, other, third, std::move(op_builder),
                                  std::move(opts)));

  bool is_inplace =
      out.is_alias_of(self) || out.is_alias_of(other) || out.is_alias_of(third);
  OutputOpMode mode =
      is_inplace ? OutputOpMode::kInPlace : OutputOpMode::kOutOfPlace;

  return FinalizeOpOutput(out, std::move(result_buf), mode);
}

namespace {

// PyTorch operations that take an `alpha` scalar only support real
// datatypes (Integer, Floating Point, or Boolean). We reject complex scalars
// here to match PyTorch constraints as complex alphas are not yet supported on
// TPU.
absl::Status CheckAlphaTypeSupported(const at::Scalar& alpha) {
  if (!alpha.isIntegral(/*include_bool=*/false) && !alpha.isFloatingPoint() &&
      !alpha.isBoolean()) {
    // TODO: add support to complex alpha on TPU.
    TT_ASSIGN_OR_RETURN(const auto alpha_element_type,
                        internal::ToElementType(alpha.type()));
    return TT_ERROR(error::kUnimplemented)
           << ToString(alpha_element_type)
           << " alpha value is not yet supported";
  }
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildAlphaMulShlo(mlir::MlirOp other_op,
                                               mlir::MlirOp alpha_op) {
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
  auto other_element_type = other_type.getElementType();
  auto casted_alpha_op =
      stablehlo::ConvertElementType(alpha_op, other_element_type);
  TT_ASSIGN_OR_RETURN((auto [broadcasted_other, broadcasted_alpha]),
                      ApplyBroadcastIfNeeded(other_op, casted_alpha_op));
  return BuildMulShlo(broadcasted_other, broadcasted_alpha);
}

// Builds the StableHLO sequence for Add + Relu operation: Relu(self + other).
absl::StatusOr<mlir::MlirOp> BuildAddReluShlo(mlir::MlirOp self_op,
                                              mlir::MlirOp other_op) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp add_op, BuildAddShlo(self_op, other_op));
  return BuildReluShlo(add_op);
}

// Builds the StableHLO sequence for Alpha-scaled Add + Relu operation:
// Relu(self + alpha * other).
absl::StatusOr<mlir::MlirOp> BuildAlphaAddReluShlo(mlir::MlirOp self_op,
                                                   mlir::MlirOp other_op,
                                                   mlir::MlirOp alpha_op) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mul_op,
                      BuildAlphaMulShlo(other_op, alpha_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp add_op, BuildAddShlo(self_op, mul_op));
  return BuildReluShlo(add_op);
}

struct DivOpOptions {
  MlirBinaryOpBuilder op_builder;
  OpParamCacheKeys op_param_cache_keys;
  std::optional<mlir::ElementType> output_dtype_override;
};

template <typename OtherType>
absl::StatusOr<DivOpOptions> GetDivOpOptionsNoMode(const at::Tensor& self,
                                                   const OtherType& other) {
  ABSL_VLOG(1) << "[GetDivOpOptionsNoMode]";
  // Capture the default dtype at **dispatch** time, not **compile** time,
  // and convert to XLA and MLIR types.
  at::ScalarType default_aten_type = c10::get_default_dtype_as_scalartype();
  TT_ASSIGN_OR_RETURN(auto default_dtype,
                      ConvertTo<mlir::ElementType>(default_aten_type));

  auto op_builder = [default_dtype](
                        mlir::MlirOp self_op,
                        mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
    // Convert integer and booleans to torch's default scalar type.
    TT_ASSIGN_OR_RETURN((auto [promoted_self_op, promoted_other_op]),
                        ConvertIfIntegers(self_op, other_op, default_dtype));
    return BuildDivShlo(promoted_self_op, promoted_other_op);
  };

  // Integer and booleans will be converted to torch's default scalar type.
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true) &&
      c10::isIntegralType(GetScalarType(other), /*includeBool=*/true)) {
    return DivOpOptions{.op_builder = std::move(op_builder),
                        .op_param_cache_keys = OpParamCacheKeys::Empty(),
                        .output_dtype_override = default_dtype};
  }

  return DivOpOptions{.op_builder = std::move(op_builder),
                      .op_param_cache_keys = OpParamCacheKeys::Empty()};
}

absl::StatusOr<DivOpOptions> GetDivOpOptionsTruncMode() {
  ABSL_VLOG(1) << "[GetDivOpOptionsTruncMode]";
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      *OpParamCacheKeysBuilder().SetParam("mode", "trunc"));
  auto op_builder = [](mlir::MlirOp self_op,
                       mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp div_op, BuildDivShlo(self_op, other_op));

    // Rounds the results of the division towards zero. Equivalent to
    // C-style integer division.
    const mlir::RankedTensorType div_type = GetTensorTypeOrDie(div_op);
    // For integer division, the result is already an integer.
    if (div_type.getElementType().isInteger()) {
      return div_op;
    }

    // For floating-point division, we floor the absolute value of the result
    // and multiply by the sign of the original result to achieve truncation
    // towards zero.
    auto abs = stablehlo::Abs(div_op);
    auto floor = stablehlo::Floor(abs);
    auto sign = stablehlo::Sign(div_op);
    return stablehlo::Mul(floor, sign);
  };
  return DivOpOptions{std::move(op_builder), std::move(param_keys),
                      std::nullopt};
}

absl::StatusOr<DivOpOptions> GetDivOpOptionsFloorMode() {
  ABSL_VLOG(1) << "[GetDivOpOptionsFloorMode]";
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      *OpParamCacheKeysBuilder().SetParam("mode", "floor"));
  auto op_builder = [](mlir::MlirOp self_op,
                       mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp div_op, BuildDivShlo(self_op, other_op));

    // Rounds the results of the division down. Equivalent to floor division
    // in Python (the // operator) and NumPy’s np.floor_divide.
    const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
    const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
    if (!self_type.getElementType().isInteger() &&
        !other_type.getElementType().isInteger()) {
      TT_ASSIGN_OR_RETURN(auto floor_div_op, BuildFloorShlo(div_op));
      return floor_div_op;
    }

    // Performing negative numerator / denominator is tricky.
    // Inspired by
    // [ref](https://github.com/tensorflow/tensorflow/blob/a0d9adf50eb4dbc3b8de346b0c95acce288959c0/tensorflow/compiler/mlir/tf2xla/transforms/legalize_tf_patterns.td#L151-L186)
    // Here's an equivalent pseudo-code:
    // T z = x / y
    // return (z * y != x && (x < 0) != (y < 0)) ? z - 1 : z
    auto div = chlo::BroadcastDiv(self_op, other_op);
    auto mul = chlo::BroadcastMul(div, other_op);
    auto l_and =
        chlo::BroadcastCompare(mul, self_op, chlo::ComparisonDirection::NE);

    auto l_zeros =
        MakeScalarConstant(self_op.getBuilder(), 0, self_type.getElementType());

    auto l_cmp =
        chlo::BroadcastCompare(self_op, l_zeros, chlo::ComparisonDirection::LT);

    auto r_cmp = chlo::BroadcastCompare(other_op, l_zeros,
                                        chlo::ComparisonDirection::LT);

    auto r_and =
        chlo::BroadcastCompare(l_cmp, r_cmp, chlo::ComparisonDirection::NE);

    auto pred_select = chlo::BroadcastAnd(l_and, r_and);

    const mlir::RankedTensorType div_type = GetTensorTypeOrDie(div_op);
    auto ones =
        MakeScalarConstant(self_op.getBuilder(), 1, div_type.getElementType());
    auto true_select = chlo::BroadcastSub(div, ones);

    return stablehlo::Select(pred_select, true_select, div);
  };
  return DivOpOptions{std::move(op_builder), std::move(param_keys),
                      std::nullopt};
}

// Returns the appropriate division operator builder based on the provided mode.
//
// See torch.div documentation for the mode options:
// https://docs.pytorch.org/docs/stable/generated/torch.div.html
//
// Args:
//  self: The dividend tensor.
//  other: The divisor tensor or scalar.
//  mode: The division mode. Can be "trunc", "floor", or std::nullopt.
//
// Returns:
//  A DivOpOptions struct containing the operator builder, cache keys, and
//  output dtype override.
template <typename OtherType>
absl::StatusOr<DivOpOptions> GetDivOpBuilder(
    const at::Tensor& self, const OtherType& other,
    std::optional<std::string_view> mode = {}) {
  if (!mode.has_value()) {
    return GetDivOpOptionsNoMode(self, other);
  }
  if (mode.value() == "trunc") {
    return GetDivOpOptionsTruncMode();
  }
  if (mode.value() == "floor") {
    return GetDivOpOptionsFloorMode();
  }
  return TT_ERROR(error::kInvalidArgument)
         << "Unsupported mode for div: " << mode.value();
}

absl::Status CheckFloorDivideInputs(const at::Tensor& self,
                                    const at::Tensor& other) {
  TT_RET_CHECK(!IsComplex(self) && !IsBool(self), error::kInvalidArgument)
      << "expected dtype of the first argument to be neither complex nor "
         "bool, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(!IsComplex(other) && !IsBool(other), error::kInvalidArgument)
      << "expected dtype of the second argument to be neither complex nor "
         "bool, got "
      << ToString(other.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckAtan2Inputs(const at::Tensor& x, const at::Tensor& y) {
  TT_RET_CHECK(!IsComplex(x), error::kInvalidArgument)
      << "expected the dtype of the first argument not to be complex, got "
      << ToString(x.scalar_type());

  TT_RET_CHECK(!IsComplex(y), error::kInvalidArgument)
      << "expected the dtype of the second argument not to be complex, got "
      << ToString(y.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckBitwiseOpsInputs(const at::Tensor& self,
                                   const at::Tensor& other) {
  TT_RET_CHECK(!IsFloatingPoint(self) && !IsComplex(self),
               error::kInvalidArgument)
      << "expected the dtype of the first argument to be neither "
         "floating-point nor complex, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(!IsFloatingPoint(other) && !IsComplex(other),
               error::kInvalidArgument)
      << "expected the dtype of the second argument to be neither "
         "floating-point nor complex, got "
      << ToString(other.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckBitwiseShiftInputs(const at::Tensor& self,
                                     const at::Tensor& other) {
  TT_RET_CHECK(IsInteger(self), error::kInvalidArgument)
      << "expected the dtype of the first argument to be integer, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(IsInteger(other), error::kInvalidArgument)
      << "expected the dtype of the second argument to be integer, got "
      << ToString(other.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckComplexOutInputs(const at::Tensor& real,
                                   const at::Tensor& imag,
                                   const at::Tensor& out) {
  TT_RET_CHECK(IsFloatOrDouble(real), error::kInvalidArgument)
      << "expected the dtype of the first argument to be float32 or float64, "
         "got "
      << ToString(real.scalar_type());

  TT_RET_CHECK(IsFloatOrDouble(imag), error::kInvalidArgument)
      << "expected the dtype of the second argument to be float32 or float64, "
         "got "
      << ToString(imag.scalar_type());

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Errors when creating the complex32
                 // output tensor before.
      IsXlaSupportedComplex(out), error::kInvalidArgument)
      << "expected the dtype of the output to be one of the OpenXLA supported "
         "complex dtypes (either complex64 or complex128), got "
      << ToString(out.scalar_type());

  return absl::OkStatus();
}

// Checks that the input dtypes are not complex.
//
// In this context, the template parameter `T` should be one of:
//
//   - at::Tensor
//   - at::Scalar
template <typename T>
absl::Status CheckInputsNotComplex(const at::Tensor& self, const T& other) {
  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the dtype of the first argument not to be complex, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(!IsComplex(other), error::kInvalidArgument)
      << "expected the dtype of the second argument not to be complex, got "
      << ToString(GetScalarType(other));

  return absl::OkStatus();
}

// The template parameters `T` and `U` should be one of:
//
//   - at::Tensor
//   - at::Scalar
template <typename T, typename U>
absl::Status CheckPowInputs(const T& self, const U& exponent) {
  // TODO: b/481396743 remove these checks once we start supporting bool dtype.

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch native devices supports boolean
                 // dtype.
      !IsBool(self), error::kInvalidArgument)
      << "expected the dtype of the first argument not to be boolean, got "
      << ToString(GetScalarType(self));

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch native devices supports boolean
                 // dtype.
      !IsBool(exponent), error::kInvalidArgument)
      << "expected the dtype of the second argument not to be boolean, got "
      << ToString(GetScalarType(exponent));

  return absl::OkStatus();
}

// Checks the dtypes of the inputs of the `remainder()` op.
//
// In this context, the template parameter `T` should be one of:
//
//   - at::Tensor
//   - at::Scalar
template <typename T>
absl::Status CheckRemainderInputs(const T& self, const at::Tensor& other) {
  at::ScalarType result_type = at::result_type(self, other);

  TT_RET_CHECK(!IsBool(result_type) && !IsComplex(result_type),
               error::kInvalidArgument)
      << "expected the dtype of the output (promoted inputs dtype) to be "
         "neither bool nor complex, got "
      << ToString(result_type);

  return absl::OkStatus();
}

absl::Status CheckPolarInputs(const at::Tensor& abs, const at::Tensor& angle) {
  TT_RET_CHECK(IsFloatOrDouble(abs), error::kInvalidArgument)
      << "expected the dtype of the first argument to be float32 or float64, "
         "got "
      << ToString(abs.scalar_type());

  TT_RET_CHECK(IsFloatOrDouble(angle), error::kInvalidArgument)
      << "expected the dtype of the second argument to be float32 or float64, "
         "got "
      << ToString(angle.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckSubInputs(const at::Tensor& self, const at::Tensor& other) {
  TT_RET_CHECK(!IsBool(self), error::kInvalidArgument)
      << "the dtype of the first argument cannot be bool";

  TT_RET_CHECK(!IsBool(other), error::kInvalidArgument)
      << "the dtype of the second argument cannot be bool";

  return absl::OkStatus();
}

absl::Status AtenComparisonScalarOutHelper(
    const at::Tensor& self, PromotedScalar& promoted_other, at::Tensor& out,
    stablehlo::ComparisonDirection direction) {
  // We can compare complex numbers for equality but not for ordering.
  if (direction != stablehlo::ComparisonDirection::EQ &&
      direction != stablehlo::ComparisonDirection::NE) {
    TT_RETURN_IF_ERROR(CheckInputsNotComplex(self, promoted_other.scalar()));
  }
  const at::ScalarType promoted_scalar_type =
      at::result_type(self, promoted_other.scalar());
  TT_ASSIGN_OR_RETURN(const at::Tensor other_tensor,
                      promoted_other.GetTensor(promoted_scalar_type));

  MlirBinaryOpBuilder builder;
  switch (direction) {
    case stablehlo::ComparisonDirection::EQ:
      builder = BuildEqShlo;
      break;
    case stablehlo::ComparisonDirection::GE:
      builder = BuildGeShlo;
      break;
    case stablehlo::ComparisonDirection::GT:
      builder = BuildGtShlo;
      break;
    case stablehlo::ComparisonDirection::LE:
      builder = BuildLeShlo;
      break;
    case stablehlo::ComparisonDirection::LT:
      builder = BuildLtShlo;
      break;
    case stablehlo::ComparisonDirection::NE:
      builder = BuildNeShlo;
      break;
  }

  return BinaryOpOut(self, other_tensor, out, std::move(builder),
                     {.op_param_cache_keys = OpParamCacheKeys::Empty()});
}

enum class BitwiseShiftDirection {
  kLeft,
  kRight,
};

absl::Status BitwiseShiftScalarHelper(const at::Tensor& self,
                                      PromotedScalar& promoted_other,
                                      at::Tensor& out,
                                      BitwiseShiftDirection direction) {
  TT_RET_CHECK(promoted_other.scalar().isIntegral(/*include_bool=*/false),
               error::kInvalidArgument)
      << "expected the dtype of the second argument to be integer, got "
      << ToString(GetScalarType(promoted_other.scalar()));

  TT_ASSIGN_OR_RETURN(const at::Tensor other_tensor,
                      promoted_other.GetTensor(out.scalar_type()));
  TT_RETURN_IF_ERROR(CheckBitwiseShiftInputs(self, other_tensor));

  const at::Tensor new_self = self.to(out.scalar_type());

  MlirBinaryOpBuilder builder = direction == BitwiseShiftDirection::kLeft
                                    ? BuildBitwiseLeftShiftShlo
                                    : BuildBitwiseRightShiftShlo;
  return BinaryOpOut(new_self, other_tensor, out, std::move(builder),
                     {.op_param_cache_keys = OpParamCacheKeys::Empty()});
}

}  // namespace

// NOLINTBEGIN
// clang-format off
// go/keep-sorted start ignore_prefixes=at::Tensor,at::Tensor& newline_separated=yes
// clang-format on
// NOLINTEND
at::Tensor& AtenAddOut(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha, at::Tensor& out) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kAddOut, param_keys, (self, other, promoted_alpha, out), {
    TT_THROW_IF_ERROR(CheckAlphaTypeSupported(alpha));

    // As an optimization, skip the scaling if alpha is 1.
    if (promoted_alpha.IsOne()) {
      TT_THROW_IF_ERROR(
          BinaryOpOut(self, other, out, BuildAddShlo,
                      {.op_param_cache_keys = std::move(param_keys)}));
      return out;
    }

    TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                       promoted_alpha.GetTensor(out.scalar_type()));

    auto op_builder =
        [](mlir::MlirOp self_op, mlir::MlirOp other_op,
           mlir::MlirOp alpha_op) -> absl::StatusOr<mlir::MlirOp> {
      TT_ASSIGN_OR_RETURN(auto mul_op, BuildAlphaMulShlo(other_op, alpha_op));
      return BuildAddShlo(self_op, mul_op);
    };

    TT_THROW_IF_ERROR(
        TernaryOpOut(self, other, alpha_tensor, out, std::move(op_builder),
                     {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

at::Tensor& AtenAddReluOut(const at::Tensor& self, const at::Tensor& other,
                           const at::Scalar& alpha, at::Tensor& out) {
  MaybePromotedScalar promoted_alpha =
      PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(
      OpName::kAddReluOut, param_keys, (self, other, promoted_alpha, out), {
        TT_THROW_IF_ERROR(CheckAlphaTypeSupported(alpha));

        // As an optimization, skip the scaling if alpha is 1.
        if (promoted_alpha.IsOne()) {
          TT_THROW_IF_ERROR(
              BinaryOpOut(self, other, out, BuildAddReluShlo,
                          {.op_param_cache_keys = std::move(param_keys)}));
          return out;
        }

        TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                           promoted_alpha.GetTensor(out.scalar_type()));

        TT_THROW_IF_ERROR(
            TernaryOpOut(self, other, alpha_tensor, out, BuildAlphaAddReluShlo,
                         {.op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor AtenAddReluScalar(const at::Tensor& self, const at::Scalar& other,
                             const at::Scalar& alpha) {
  PromotedScalar promoted_other = PromoteScalar(other);
  MaybePromotedScalar promoted_alpha =
      PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(
      OpName::kAddReluScalar, param_keys,
      (self, promoted_other, promoted_alpha), {
        TT_THROW_IF_ERROR(CheckAlphaTypeSupported(alpha));

        at::ScalarType promoted_scalar_type = at::result_type(self, other);
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(self.sizes(), promoted_scalar_type, self.device()));

        TT_ASSIGN_OR_THROW(const at::Tensor other_tensor,
                           promoted_other.GetTensor(promoted_scalar_type));

        if (promoted_alpha.IsOne()) {
          TT_THROW_IF_ERROR(
              BinaryOpOut(self, other_tensor, out, BuildAddReluShlo,
                          {.op_param_cache_keys = std::move(param_keys)}));
          return out;
        }

        TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                           promoted_alpha.GetTensor(promoted_scalar_type));

        TT_THROW_IF_ERROR(TernaryOpOut(
            self, other_tensor, alpha_tensor, out, BuildAlphaAddReluShlo,
            {.op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor AtenAddReluTensor(const at::Tensor& self, const at::Tensor& other,
                             const at::Scalar& alpha) {
  MaybePromotedScalar promoted_alpha =
      PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kAddReluTensor, param_keys, (self, other, promoted_alpha), {
    TT_THROW_IF_ERROR(CheckAlphaTypeSupported(alpha));

    const at::ScalarType promoted_scalar_type = at::result_type(self, other);
    TT_ASSIGN_OR_THROW(const Dimensions output_dims,
                       InferSize(self.sizes(), other.sizes()));
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(output_dims, promoted_scalar_type, self.device()));

    if (promoted_alpha.IsOne()) {
      TT_THROW_IF_ERROR(
          BinaryOpOut(self, other, out, BuildAddReluShlo,
                      {.op_param_cache_keys = std::move(param_keys)}));
      return out;
    }

    TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                       promoted_alpha.GetTensor(promoted_scalar_type));

    TT_THROW_IF_ERROR(
        TernaryOpOut(self, other, alpha_tensor, out, BuildAlphaAddReluShlo,
                     {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

at::Tensor& AtenAddRelu_Scalar(at::Tensor& self, const at::Scalar& other,
                               const at::Scalar& alpha) {
  TT_KERNEL(OpName::kAddRelu_Scalar, _,
            (self, IgnoreInCacheKey(other, "Legacy usage"),
             IgnoreInCacheKey(alpha, "Legacy usage")),
            {
              TT_ASSIGN_OR_THROW(auto wrapped_other, MakeTensor(other));
              AtenAddReluOut(self, wrapped_other, alpha, self);
              return self;
            });
}

at::Tensor& AtenAddRelu_Tensor(at::Tensor& self, const at::Tensor& other,
                               const at::Scalar& alpha) {
  MaybePromotedScalar promoted_alpha =
      PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(
      OpName::kAddRelu_Tensor, param_keys, (self, other, promoted_alpha), {
        TT_THROW_IF_ERROR(CheckAlphaTypeSupported(alpha));

        const at::ScalarType promoted_scalar_type =
            at::result_type(self, other);

        if (promoted_alpha.IsOne()) {
          TT_THROW_IF_ERROR(
              BinaryOpOut(self, other, self, BuildAddReluShlo,
                          {.op_param_cache_keys = std::move(param_keys)}));
          return self;
        }

        TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                           promoted_alpha.GetTensor(promoted_scalar_type));
        TT_THROW_IF_ERROR(
            TernaryOpOut(self, other, alpha_tensor, self, BuildAlphaAddReluShlo,
                         {.op_param_cache_keys = std::move(param_keys)}));
        return self;
      });
}

at::Tensor& AtenAtan2Out(const at::Tensor& x, const at::Tensor& y,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kAtan2Out, _, (x, y, out), {
    TT_THROW_IF_ERROR(CheckAtan2Inputs(x, y));
    TT_THROW_IF_ERROR(
        BinaryOpOut(x, y, out, BuildAtan2Shlo,
                    {.force_float_inputs = true,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseAndTensorOut(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseAndOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildBitwiseAndShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseLeftShiftTensorOut(const at::Tensor& self,
                                          const at::Tensor& other,
                                          at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseLeftShiftTensorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseShiftInputs(self, other));
    at::Tensor new_self = self.to(out.scalar_type());
    at::Tensor new_other = other.to(out.scalar_type());
    TT_THROW_IF_ERROR(
        BinaryOpOut(new_self, new_other, out, BuildBitwiseLeftShiftShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseOrTensorOut(const at::Tensor& self,
                                   const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseOrOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildBitwiseOrShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseRightShiftTensorOut(const at::Tensor& self,
                                           const at::Tensor& other,
                                           at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseRightShiftTensorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseShiftInputs(self, other));
    at::Tensor new_self = self.to(out.scalar_type());
    at::Tensor new_other = other.to(out.scalar_type());
    TT_THROW_IF_ERROR(
        BinaryOpOut(new_self, new_other, out, BuildBitwiseRightShiftShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseXorTensorOut(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseXorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_ASSIGN_OR_THROW(auto output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildBitwiseXorShlo,
                    {.output_dtype_override = output_dtype,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenComplexOut(const at::Tensor& real, const at::Tensor& imag,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kComplexOut, _, (real, imag, out), {
    TT_THROW_IF_ERROR(CheckComplexOutInputs(real, imag, out));
    TT_THROW_IF_ERROR(
        BinaryOpOut(real, imag, out, BuildComplexShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenDivOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kDivOut, _, (self, other, out), {
    TT_ASSIGN_OR_THROW(auto div_op_options, GetDivOpBuilder(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, std::move(div_op_options.op_builder),
                    {.op_param_cache_keys =
                         std::move(div_op_options.op_param_cache_keys)}));
    return out;
  });
}

at::Tensor& AtenDivOutMode(const at::Tensor& self, const at::Tensor& other,
                           std::optional<std::string_view> mode,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kDivOut, _,
            (self, other, IgnoreInCacheKey(mode, "Legacy usage"), out), {
              TT_ASSIGN_OR_THROW((auto [op_builder, op_param_cache_keys, _]),
                                 GetDivOpBuilder(self, other, mode));
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self, other, out, std::move(op_builder),
                  {.op_param_cache_keys = std::move(op_param_cache_keys)}));
              return out;
            });
}

at::Tensor& AtenEqScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kEqScalarOut, _, (self, promoted_other, out), {
    TT_THROW_IF_ERROR(AtenComparisonScalarOutHelper(
        self, promoted_other, out, stablehlo::ComparisonDirection::EQ));
    return out;
  });
}

at::Tensor& AtenEqTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kEqOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildEqShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenFloorDivide(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    return at::div(self, other, "floor");
  });
}

at::Tensor& AtenFloorDivideOut(const at::Tensor& self, const at::Tensor& other,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    return AtenDivOutMode(self, other, "floor", out);
  });
}

at::Tensor& AtenFloorDivide_Tensor(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    return self.div_(other, "floor");
  });
}

at::Tensor& AtenFmodTensorOut(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kFmodTensorOut, _, (self, other, out), {
    TT_CHECK_THROW(!self.is_complex() && !other.is_complex(),
                   error::kInvalidArgument)
        << "complex dtypes are not supported";
    TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool &&
                       other.scalar_type() != at::ScalarType::Bool,
                   error::kInvalidArgument)
        << "boolean dtypes are not supported";

    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildFmodTensorShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenGeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kGeScalarOut, _, (self, promoted_other, out), {
    TT_THROW_IF_ERROR(AtenComparisonScalarOutHelper(
        self, promoted_other, out, stablehlo::ComparisonDirection::GE));
    return out;
  });
}

at::Tensor& AtenGeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGeOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckInputsNotComplex(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildGeShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenGtScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kGtScalarOut, _, (self, promoted_other, out), {
    TT_THROW_IF_ERROR(AtenComparisonScalarOutHelper(
        self, promoted_other, out, stablehlo::ComparisonDirection::GT));
    return out;
  });
}

at::Tensor& AtenGtTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGtOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckInputsNotComplex(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildGtShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenIlshiftScalar(at::Tensor& self, const at::Scalar& other) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kIlshiftScalar, _, (self, promoted_other), {
    TT_THROW_IF_ERROR(BitwiseShiftScalarHelper(self, promoted_other, self,
                                               BitwiseShiftDirection::kLeft));
    return self;
  });
}

at::Tensor& AtenIlshiftTensor(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kIlshiftTensor, _, (self, other), {
    AtenBitwiseLeftShiftTensorOut(self, other, self);
    return self;
  });
}

at::Tensor& AtenIrshiftScalar(at::Tensor& self, const at::Scalar& other) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kIrshiftScalar, _, (self, promoted_other), {
    TT_THROW_IF_ERROR(BitwiseShiftScalarHelper(self, promoted_other, self,
                                               BitwiseShiftDirection::kRight));
    return self;
  });
}

at::Tensor& AtenIrshiftTensor(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kIrshiftTensor, _, (self, other), {
    AtenBitwiseRightShiftTensorOut(self, other, self);
    return self;
  });
}

at::Tensor& AtenLeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kLeScalarOut, _, (self, promoted_other, out), {
    TT_THROW_IF_ERROR(AtenComparisonScalarOutHelper(
        self, promoted_other, out, stablehlo::ComparisonDirection::LE));
    return out;
  });
}

at::Tensor& AtenLeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kLeOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckInputsNotComplex(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildLeShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenLshiftScalar(const at::Tensor& self, const at::Scalar& other) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kLshiftScalar, _, (self, promoted_other), {
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(BitwiseShiftScalarHelper(self, promoted_other, out,
                                               BitwiseShiftDirection::kLeft));
    return out;
  });
}

at::Tensor AtenLshiftTensor(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kLshiftTensor, _, (self, other), {
    at::ScalarType promoted_scalar_type = at::result_type(self, other);
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(self.sizes(), promoted_scalar_type, self.device()));
    AtenBitwiseLeftShiftTensorOut(self, other, out);
    return out;
  });
}

at::Tensor& AtenLtScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kLtScalarOut, _, (self, promoted_other, out), {
    TT_THROW_IF_ERROR(AtenComparisonScalarOutHelper(
        self, promoted_other, out, stablehlo::ComparisonDirection::LT));
    return out;
  });
}

at::Tensor& AtenLtTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kLtOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckInputsNotComplex(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildLtShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMaximumOut(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kMaximumOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildMaximumShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMinimumOut(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kMinimumOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildMinimumShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMulOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kMulOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildMulShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenNeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kNeScalarOut, _, (self, promoted_other, out), {
    TT_THROW_IF_ERROR(AtenComparisonScalarOutHelper(
        self, promoted_other, out, stablehlo::ComparisonDirection::NE));
    return out;
  });
}

at::Tensor& AtenNeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kNeOut, _, (self, other, out), {
    TT_ASSIGN_OR_THROW(auto output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildNeShlo,
                    {.output_dtype_override = output_dtype,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPolarOut(const at::Tensor& abs, const at::Tensor& angle,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kPolarOut, _, (abs, angle, out), {
    TT_THROW_IF_ERROR(CheckPolarInputs(abs, angle));
    TT_THROW_IF_ERROR(
        BinaryOpOut(abs, angle, out, BuildPolarShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPowScalarOut(const at::Scalar& self, const at::Tensor& exponent,
                             at::Tensor& out) {
  auto promoted_self = PromoteScalar(self);
  TT_KERNEL(OpName::kPowScalarOut, _, (promoted_self, exponent, out), {
    TT_THROW_IF_ERROR(CheckPowInputs(self, exponent));
    // Can't use reverse_operands here because a^b != b^a.
    // Cast to out tensor dtype to be consistent with PyTorch.
    TT_ASSIGN_OR_THROW(at::Tensor self_tensor,
                       promoted_self.GetTensor(out.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self_tensor, exponent, out, BuildPowShlo,
                    // Use kPowOut for cache key as the builder logic is the
                    // same as for AtenPowTensorTensorOut.
                    {.op_name = OpName::kPowOut,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPowTensorScalarOut(const at::Tensor& self,
                                   const at::Scalar& exponent,
                                   at::Tensor& out) {
  auto promoted_exponent = PromoteScalar(exponent);
  TT_KERNEL(OpName::kPowTensorScalarOut, _, (self, promoted_exponent, out), {
    TT_THROW_IF_ERROR(CheckPowInputs(self, exponent));
    // Cast to self dtype to be consistent with PyTorch.
    TT_ASSIGN_OR_THROW(at::Tensor exponent_tensor,
                       promoted_exponent.GetTensor(self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, exponent, out, BuildPowShlo,
                    // Use kPowOut for cache key as the builder logic is the
                    // same as for AtenPowTensorTensorOut.
                    {.op_name = OpName::kPowOut,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPowTensorTensorOut(const at::Tensor& self,
                                   const at::Tensor& exponent,
                                   at::Tensor& out) {
  TT_KERNEL(OpName::kPowOut, _, (self, exponent, out), {
    TT_THROW_IF_ERROR(CheckPowInputs(self, exponent));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, exponent, out, BuildPowShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenRemainderScalarTensor(const at::Scalar& self,
                                     const at::Tensor& other) {
  TT_KERNEL(
      OpName::kRemainder, _, (IgnoreInCacheKey(self, "Legacy usage"), other), {
        TT_THROW_IF_ERROR(CheckRemainderInputs(self, other));
        TT_ASSIGN_OR_THROW(auto div_opts, GetDivOpOptionsFloorMode());

        auto remainder_builder =
            [div_opts = std::move(div_opts)](
                mlir::MlirOp self_op,
                mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
          TT_ASSIGN_OR_RETURN((auto [cast_self_op, cast_other_op]),
                              ApplyBroadcastIfNeeded(self_op, other_op));
          TT_ASSIGN_OR_RETURN(auto div_op,
                              div_opts.op_builder(cast_self_op, cast_other_op));
          auto mul_op = stablehlo::Mul(div_op, cast_other_op);
          return stablehlo::Subtract(cast_self_op, mul_op);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            BinaryOp(other, self, std::move(remainder_builder),
                     {.reverse_operands = true,
                      .op_param_cache_keys = OpParamCacheKeys::Empty()}));
        return result;
      });
}

at::Tensor& AtenRemainderTensorOut(const at::Tensor& self,
                                   const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kRemainderOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckRemainderInputs(self, other));
    TT_ASSIGN_OR_THROW(auto div_opts, GetDivOpOptionsFloorMode());

    auto remainder_builder =
        [div_opts = std::move(div_opts)](
            mlir::MlirOp self_op,
            mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
      TT_ASSIGN_OR_RETURN((auto [cast_self_op, cast_other_op]),
                          ApplyBroadcastIfNeeded(self_op, other_op));
      TT_ASSIGN_OR_RETURN(auto div_op,
                          div_opts.op_builder(cast_self_op, cast_other_op));
      auto mul_op = stablehlo::Mul(div_op, cast_other_op);
      return stablehlo::Subtract(cast_self_op, mul_op);
    };
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, std::move(remainder_builder),
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenRshiftScalar(const at::Tensor& self, const at::Scalar& other) {
  PromotedScalar promoted_other = PromoteScalar(other);
  TT_KERNEL(OpName::kRshiftScalar, _, (self, promoted_other), {
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(BitwiseShiftScalarHelper(self, promoted_other, out,
                                               BitwiseShiftDirection::kRight));
    return out;
  });
}

at::Tensor AtenRshiftTensor(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kRshiftTensor, _, (self, other), {
    at::ScalarType promoted_scalar_type = at::result_type(self, other);
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(self.sizes(), promoted_scalar_type, self.device()));
    AtenBitwiseRightShiftTensorOut(self, other, out);
    return out;
  });
}

at::Tensor AtenRsubTensor(const at::Tensor& self, const at::Tensor& other,
                          const at::Scalar& alpha) {
  TT_KERNEL(OpName::kRsub, _,
            (self, other, IgnoreInCacheKey(alpha, "Legacy usage")),
            { return at::sub(other, self, alpha); });
}

at::Tensor& AtenSubOut(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha, at::Tensor& out) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kSubOut, param_keys, (self, other, promoted_alpha, out), {
    TT_THROW_IF_ERROR(CheckAlphaTypeSupported(alpha));
    TT_THROW_IF_ERROR(CheckSubInputs(self, other));

    // As an optimization, skip the scaling if alpha is 1.
    if (promoted_alpha.IsOne()) {
      TT_THROW_IF_ERROR(
          BinaryOpOut(self, other, out, BuildSubShlo,
                      {.op_param_cache_keys = std::move(param_keys)}));
      return out;
    }

    TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                       promoted_alpha.GetTensor(out.scalar_type()));

    auto op_builder =
        [](mlir::MlirOp self_op, mlir::MlirOp other_op,
           mlir::MlirOp alpha_op) -> absl::StatusOr<mlir::MlirOp> {
      TT_ASSIGN_OR_RETURN(auto mul_op, BuildAlphaMulShlo(other_op, alpha_op));
      return BuildSubShlo(self_op, mul_op);
    };

    TT_THROW_IF_ERROR(
        TernaryOpOut(self, other, alpha_tensor, out, std::move(op_builder),
                     {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

// go/keep-sorted end

}  // namespace torch_tpu
