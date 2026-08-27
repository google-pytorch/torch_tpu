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

#include "torch_tpu/csrc/ops/binary_aten_kernels.h"

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/result_type.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/binary.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/csrc/ops/unary.h"

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

  auto op_builder =
      [bin_op_builder = std::move(bin_op_builder),
       reverse = opts.reverse_operands](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [self_op, other_op] = inputs;
        if (reverse) {
          return bin_op_builder(other_op, self_op);
        }
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
    TT_RETURN_IF_ERROR(
        ResizeTensorIfShapeDiffers(out, result_buf.dimensions()));
  }

  return AssignBufferToAtTensor(std::move(result_buf), out);
}

absl::Status TernaryOpOut(const at::Tensor& self, const at::Tensor& other,
                          const at::Tensor& third, at::Tensor& out,
                          MlirTernaryOpBuilder op_builder,
                          TernaryOpOptions opts) {
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected output tensor to be on 'tpu' device, got '" << out.device()
      << "'";
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
    return TT_ERROR(error::kPythonNotImplementedError)
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

  MlirBinaryOpBuilder builder = direction == BitwiseShiftDirection::kLeft
                                    ? BuildBitwiseLeftShiftShlo
                                    : BuildBitwiseRightShiftShlo;
  return BinaryOpOut(self, other_tensor, out, std::move(builder),
                     {.op_param_cache_keys = OpParamCacheKeys::Empty()});
}

absl::Status SubHelperOut(const at::Tensor& a, const at::Tensor& b,
                          MaybePromotedScalar promoted_alpha,
                          at::ScalarType out_dtype, at::Tensor& out,
                          OpParamCacheKeys& param_keys) {
  // As an optimization, skip the scaling if alpha is 1.
  if (promoted_alpha.IsOne()) {
    return BinaryOpOut(a, b, out, BuildSubShlo,
                       {.op_param_cache_keys = std::move(param_keys)});
  }

  TT_ASSIGN_OR_RETURN(const at::Tensor alpha_tensor,
                      promoted_alpha.GetTensor(out_dtype));

  auto op_builder = [](mlir::MlirOp a_op, mlir::MlirOp b_op,
                       mlir::MlirOp alpha_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(auto mul_op, BuildAlphaMulShlo(b_op, alpha_op));
    return BuildSubShlo(a_op, mul_op);
  };

  return TernaryOpOut(a, b, alpha_tensor, out, std::move(op_builder),
                      {.op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<at::Tensor> AddReluScalarHelper(
    const at::Tensor& self, PromotedScalar& promoted_other,
    MaybePromotedScalar& promoted_alpha, at::Tensor& out,
    OpParamCacheKeys& param_keys) {
  TT_RETURN_IF_ERROR(CheckAlphaTypeSupported(promoted_alpha.scalar()));

  const at::ScalarType promoted_scalar_type =
      at::result_type(self, promoted_other.scalar());

  TT_ASSIGN_OR_RETURN(const at::Tensor other_tensor,
                      promoted_other.GetTensor(promoted_scalar_type));

  if (promoted_alpha.IsOne()) {
    TT_RETURN_IF_ERROR(
        BinaryOpOut(self, other_tensor, out, BuildAddReluShlo,
                    {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  }

  TT_ASSIGN_OR_RETURN(const at::Tensor alpha_tensor,
                      promoted_alpha.GetTensor(promoted_scalar_type));

  TT_RETURN_IF_ERROR(
      TernaryOpOut(self, other_tensor, alpha_tensor, out, BuildAlphaAddReluShlo,
                   {.op_param_cache_keys = std::move(param_keys)}));
  return out;
}

absl::Status DivOutMode(const at::Tensor& self, const at::Tensor& other,
                        std::optional<std::string_view> mode, at::Tensor& out) {
  TT_ASSIGN_OR_RETURN(auto div_opts, GetDivOpBuilder(self, other, mode));
  return BinaryOpOut(
      self, other, out, std::move(div_opts.op_builder),
      {.op_param_cache_keys = std::move(div_opts.op_param_cache_keys)});
}

absl::Status BitwiseLeftShiftTensor(
    const at::Tensor& self, const at::Tensor& other, at::Tensor& out,
    std::optional<OpName> op_name = OpName::kBitwiseLeftShiftTensorOut) {
  TT_RETURN_IF_ERROR(CheckBitwiseShiftInputs(self, other));
  // Input/output type conversions are automatically handled in BinaryOpOut.
  // We override op_name to allow all callers of this helper to share the same
  // compilation cache entry.
  return BinaryOpOut(
      self, other, out, BuildBitwiseLeftShiftShlo,
      {.op_name = op_name, .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

absl::Status BitwiseRightShiftTensor(
    const at::Tensor& self, const at::Tensor& other, at::Tensor& out,
    std::optional<OpName> op_name = OpName::kBitwiseRightShiftTensorOut) {
  TT_RETURN_IF_ERROR(CheckBitwiseShiftInputs(self, other));
  // Input/output type conversions are automatically handled in BinaryOpOut.
  // We override op_name to allow all callers of this helper to share the same
  // compilation cache entry.
  return BinaryOpOut(
      self, other, out, BuildBitwiseRightShiftShlo,
      {.op_name = op_name, .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

// Result helper for each scaling step in the static split method.
struct ScaleStepResult {
  mlir::MlirOp result;
  mlir::MlirOp remaining_exponent;
};

// Perform a single step of scaling.
// To prevent intermediate overflow or underflow (e.g. if we computed 2^exponent
// directly for a large exponent, which would overflow float/half ranges even
// if the final multiplied value is representable), we clamp the exponent to
// [min_exp_const, max_exp_const], calculate the scaling factor 2^clamped_exp,
// multiply 'self' by this factor, and calculate the remaining exponent that
// still needs to be scaled in subsequent steps.
absl::StatusOr<ScaleStepResult> ScaleStep(mlir::MlirOp self,
                                          mlir::MlirOp exponent,
                                          mlir::MlirOp min_exp_const,
                                          mlir::MlirOp max_exp_const,
                                          mlir::MlirOp two) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  auto element_type = self_type.getElementType();

  // Convert exponent to match the element type of 'self' (float/half/double).
  auto casted_exponent = stablehlo::ConvertElementType(exponent, element_type);

  // Clamp exponent to the safe limits [min_exp_const, max_exp_const] to prevent
  // overflow/underflow during the Pow (2^exponent) operation.
  // We use Min and Max instead of Clamp because stablehlo::Clamp drops dynamic
  // bounds when the first operand (min) is a scalar constant.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp min_exp_clamped,
                      BuildMinimumShlo(casted_exponent, max_exp_const));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp clamp_exp,
                      BuildMaximumShlo(min_exp_clamped, min_exp_const));

  // Compute 2^clamped_exp
  auto two_tensor = MakeConstantLike(clamp_exp, 2.0);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp factor, BuildPowShlo(two_tensor, clamp_exp));

  // Multiply self by 2^clamped_exp
  TT_ASSIGN_OR_RETURN(mlir::MlirOp result, BuildMulShlo(self, factor));

  // Compute remaining exponent that was not applied in this step
  TT_ASSIGN_OR_RETURN(mlir::MlirOp remaining_exponent,
                      BuildSubShlo(casted_exponent, clamp_exp));

  return ScaleStepResult{result, remaining_exponent};
}

absl::Status CheckLdexpOutput(const at::Tensor& out) {
  TT_RET_CHECK(IsFloatingPoint(out) || IsComplex(out), error::kInvalidArgument)
      << "ldexp can't be cast to the desired output type "
      << ToString(out.scalar_type());
  return absl::OkStatus();
}

// Performs 3 successive steps of scaling using ScaleStep to avoid intermediate
// overflow/underflow.
absl::StatusOr<mlir::MlirOp> Scale3Steps(mlir::MlirOp val, mlir::MlirOp exp,
                                         mlir::MlirOp min_exp,
                                         mlir::MlirOp max_exp,
                                         mlir::MlirOp two) {
  TT_ASSIGN_OR_RETURN(auto step1, ScaleStep(val, exp, min_exp, max_exp, two));
  TT_ASSIGN_OR_RETURN(
      auto step2,
      ScaleStep(step1.result, step1.remaining_exponent, min_exp, max_exp, two));
  TT_ASSIGN_OR_RETURN(
      auto step3,
      ScaleStep(step2.result, step2.remaining_exponent, min_exp, max_exp, two));
  return step3.result;
}

// Implements ldexp for complex base and/or exponent inputs.
absl::StatusOr<mlir::MlirOp> BuildLdexpComplexShlo(
    mlir::MlirOp self_op, mlir::MlirOp other_op,
    const mlir::RankedTensorType& self_type,
    const mlir::RankedTensorType& other_type, bool self_is_complex,
    bool other_is_complex) {
  auto& builder = self_op.getBuilder();
  TT_ASSIGN_OR_RETURN((auto [self_bcst, other_bcst]),
                      ApplyBroadcastIfNeeded(self_op, other_op));

  const mlir::Type float_type =
      self_is_complex
          ? mlir::cast<mlir::ComplexType>(self_type.getElementType())
                .getElementType()
          : self_type.getElementType();

  // Decompose base into real (a) and imaginary (b) components.
  mlir::MlirOp a =
      self_is_complex ? mlir::stablehlo::Real(self_bcst) : self_bcst;
  mlir::MlirOp b = self_is_complex
                       ? mlir::stablehlo::Imag(self_bcst)
                       : MakeConstantLike(self_bcst, 0.0, float_type);

  // Decompose exponent into real (c) and imaginary (d) components.
  const mlir::MlirOp c =
      other_is_complex ? mlir::stablehlo::Real(other_bcst) : other_bcst;
  const mlir::MlirOp d = other_is_complex
                             ? mlir::stablehlo::Imag(other_bcst)
                             : MakeConstantLike(other_bcst, 0.0, float_type);

  // Determine the maximum safe exponent range depending on precision limits.
  auto get_max_exp = [](mlir::Type type) -> absl::StatusOr<double> {
    if (type.isF64()) {
      return 1000.0;
    }
    if (type.isF32() || type.isBF16()) {
      return 120.0;
    }
    if (type.isF16()) {
      return 14.0;
    }
    return TT_ERROR(error::kInvalidArgument)
           << "unsupported float type for complex ldexp: " << ToString(type);
  };
  TT_ASSIGN_OR_RETURN(const double max_exp, get_max_exp(float_type));

  const auto min_exp_const = MakeScalarConstant(builder, -max_exp, float_type);
  const auto max_exp_const = MakeScalarConstant(builder, max_exp, float_type);
  const auto two = MakeScalarConstant(builder, 2.0, float_type);

  // 1. Compute the complex rotation factor (rotation = cos(theta) +
  // i*sin(theta)) where theta = d * ln(2).
  const auto ln2 =
      MakeScalarConstant(builder, 0.693147180559945309417, float_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp angle, BuildMulShlo(d, ln2));
  const auto one = MakeScalarConstant(builder, 1.0, float_type);
  TT_ASSIGN_OR_RETURN(auto rotation, BuildPolarShlo(one, angle));

  // 2. Perform rotation: base_rot = base * rotation
  const auto base = mlir::stablehlo::Complex(a, b);
  TT_ASSIGN_OR_RETURN(auto base_rot, BuildMulShlo(base, rotation));

  // 3. Decompose the rotated complex tensor back into components for scaling.
  const auto real_rot = mlir::stablehlo::Real(base_rot);
  const auto imag_rot = mlir::stablehlo::Imag(base_rot);

  // 4. Scale the components by 2^c using our Scale3Steps helper.
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp real_scaled,
      Scale3Steps(real_rot, c, min_exp_const, max_exp_const, two));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp imag_scaled,
      Scale3Steps(imag_rot, c, min_exp_const, max_exp_const, two));

  return mlir::MlirOp(mlir::stablehlo::Complex(real_scaled, imag_scaled));
}

// Build StableHLO for the ldexp operator.
// ldexp(x, exp) computes x * 2^exp.
// Since TPU (StableHLO) does not have a native ldexp operation, we implement it
// by computing x * 2^exp.
// To avoid intermediate overflow/underflow when computing 2^exp for large
// exponents, we use a 3-step Static Split Method. In each step, we scale by at
// most the maximum safe exponent of the datatype (max_exp).
//
// Maximum safe exponents:
// - Float64 (F64): max_exp = 1000.0 (Fits within double exponent range +/-
// 1023)
// - Float32 (F32), Bfloat16 (BF16): max_exp = 120.0 (Fits within exponent range
// +/- 127)
// - Float16 (F16): max_exp = 14.0 (Fits within exponent range +/- 15)
//
// By applying 3 successive steps of scaling, we cover maximum total exponents
// of:
// - F64: 3 * 1000 = 3000 (well beyond double limits)
// - F32, BF16: 3 * 120 = 360 (well beyond float limits)
// - F16: 3 * 14 = 42 (well beyond half limits)
// This ensures that any exponent value within standard range can be scaled
// without causing premature overflow/underflow in intermediate math.
absl::StatusOr<mlir::MlirOp> BuildLdexpShlo(mlir::MlirOp self_op,
                                            mlir::MlirOp other_op) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
  const bool self_is_complex = IsComplexType(self_type);
  const bool other_is_complex = IsComplexType(other_type);

  if (self_is_complex || other_is_complex) {
    return BuildLdexpComplexShlo(self_op, other_op, self_type, other_type,
                                 self_is_complex, other_is_complex);
  }

  // Real-only path:
  auto element_type = self_type.getElementType();
  auto& builder = self_op.getBuilder();

  TT_ASSIGN_OR_RETURN((auto [self_bcst, other_bcst]),
                      ApplyBroadcastIfNeeded(self_op, other_op));

  double max_exp = 120.0;
  if (element_type.isF64()) {
    max_exp = 1000.0;
  } else if (element_type.isF32() || element_type.isBF16()) {
    max_exp = 120.0;
  } else if (element_type.isF16()) {
    max_exp = 14.0;
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "unsupported element type for ldexp: " << ToString(element_type);
  }

  auto min_exp_const = MakeScalarConstant(builder, -max_exp, element_type);
  auto max_exp_const = MakeScalarConstant(builder, max_exp, element_type);
  auto two = MakeScalarConstant(builder, 2.0, element_type);

  return Scale3Steps(self_bcst, other_bcst, min_exp_const, max_exp_const, two);
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
        at::ScalarType promoted_scalar_type = at::result_type(self, other);
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(self.sizes(), promoted_scalar_type, self.device()));

        TT_ASSIGN_OR_THROW(
            at::Tensor out_tensor,
            AddReluScalarHelper(self, promoted_other, promoted_alpha, out,
                                param_keys));
        return out_tensor;
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
  PromotedScalar promoted_other = PromoteScalar(other);
  MaybePromotedScalar promoted_alpha =
      PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(
      OpName::kAddRelu_Scalar, param_keys,
      (self, promoted_other, promoted_alpha), {
        TT_THROW_IF_ERROR(AddReluScalarHelper(self, promoted_other,
                                              promoted_alpha, self, param_keys)
                              .status());
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
    // Explicitly pass std::nullopt to avoid redundant override check, since
    // the active context name is already kBitwiseLeftShiftTensorOut.
    TT_THROW_IF_ERROR(
        BitwiseLeftShiftTensor(self, other, out, /*op_name=*/std::nullopt));
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
    // Explicitly pass std::nullopt to avoid redundant override check, since
    // the active context name is already kBitwiseRightShiftTensorOut.
    TT_THROW_IF_ERROR(
        BitwiseRightShiftTensor(self, other, out, /*op_name=*/std::nullopt));
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
    TT_THROW_IF_ERROR(DivOutMode(self, other, std::nullopt, out));
    return out;
  });
}

at::Tensor& AtenDivOutMode(const at::Tensor& self, const at::Tensor& other,
                           std::optional<std::string_view> mode,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kDivOutMode, _,
            (self, other,
             IgnoreInCacheKey(
                 mode, "DivOutMode includes `mode` in op param cache keys"),
             out),
            {
              TT_THROW_IF_ERROR(DivOutMode(self, other, mode, out));
              return out;
            });
}

at::Tensor& AtenEqScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch dispatcher catches scalar
                   // comparison before kernel.
      self.scalar_type() != at::kFloat4_e2m1fn_x2, error::kInvalidArgument)
      << "Scalar comparison is not supported for float4_e2m1fn_x2";
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
    TT_ASSIGN_OR_THROW(auto output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    if (self.scalar_type() == at::kFloat4_e2m1fn_x2 ||
        other.scalar_type() == at::kFloat4_e2m1fn_x2) {
      TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch dispatcher catches dtype
                       // mismatch in promoteTypes before kernel.
          self.scalar_type() == other.scalar_type(), error::kInvalidArgument)
          << "Expected both self and other to be float4_e2m1fn_x2";
      at::Tensor self_u8 = self.view(at::kByte);
      at::Tensor other_u8 = other.view(at::kByte);
      TT_THROW_IF_ERROR(
          BinaryOpOut(self_u8, other_u8, out, BuildEqShlo,
                      {.output_dtype_override = output_dtype,
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));
      return out;
    }
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildEqShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenFloorDivide(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    TT_ASSIGN_OR_THROW(auto div_opts, GetDivOpOptionsFloorMode());
    TT_ASSIGN_OR_THROW(auto result,
                       BinaryOp(self, other, std::move(div_opts.op_builder),
                                {.op_param_cache_keys =
                                     std::move(div_opts.op_param_cache_keys)}));
    return result;
  });
}

at::Tensor& AtenFloorDivideOut(const at::Tensor& self, const at::Tensor& other,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kFloorDivideOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    TT_THROW_IF_ERROR(DivOutMode(self, other, "floor", out));
    return out;
  });
}

at::Tensor& AtenFloorDivide_Tensor(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kFloorDivide_Tensor, _, (self, other), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    TT_THROW_IF_ERROR(DivOutMode(self, other, "floor", self));
    return self;
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
    TT_THROW_IF_ERROR(BitwiseLeftShiftTensor(self, other, self));
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
    TT_THROW_IF_ERROR(BitwiseRightShiftTensor(self, other, self));
    return self;
  });
}

at::Tensor& AtenLdexpOut(const at::Tensor& self, const at::Tensor& other,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kLdexpOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckLdexpOutput(out));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildLdexpShlo,
                    {.force_float_inputs = true,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenLdexpTensor(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kLdexpTensor, _, (self, other), {
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        BinaryOp(self, other, BuildLdexpShlo,
                 {.force_float_inputs = true,
                  .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenLdexp_(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kLdexp_, _, (self, other), {
    TT_THROW_IF_ERROR(CheckLdexpOutput(self));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, self, BuildLdexpShlo,
                    {.force_float_inputs = true,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
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
    TT_THROW_IF_ERROR(BitwiseLeftShiftTensor(self, other, out));
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
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch dispatcher catches scalar
                   // comparison before kernel.
      self.scalar_type() != at::kFloat4_e2m1fn_x2, error::kInvalidArgument)
      << "Scalar comparison is not supported for float4_e2m1fn_x2";
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
    if (self.scalar_type() == at::kFloat4_e2m1fn_x2 ||
        other.scalar_type() == at::kFloat4_e2m1fn_x2) {
      TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch dispatcher catches dtype
                       // mismatch in promoteTypes before kernel.
          self.scalar_type() == other.scalar_type(), error::kInvalidArgument)
          << "Expected both self and other to be float4_e2m1fn_x2";
      at::Tensor self_u8 = self.view(at::kByte);
      at::Tensor other_u8 = other.view(at::kByte);
      TT_THROW_IF_ERROR(
          BinaryOpOut(self_u8, other_u8, out, BuildNeShlo,
                      {.output_dtype_override = output_dtype,
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));
      return out;
    }
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
    // Cast to self dtype to be consistent with PyTorch.
    TT_ASSIGN_OR_THROW(const at::Tensor exponent_tensor,
                       promoted_exponent.GetTensor(self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, exponent_tensor, out, BuildPowShlo,
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
  auto promoted_self = PromoteScalar(self);
  TT_KERNEL(OpName::kRemainderScalarTensor, _, (promoted_self, other), {
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

    const at::ScalarType promoted_scalar_type = at::result_type(self, other);
    TT_ASSIGN_OR_THROW(const at::Tensor self_tensor,
                       promoted_self.GetTensor(promoted_scalar_type));

    TT_ASSIGN_OR_THROW(
        auto result,
        BinaryOp(self_tensor, other, std::move(remainder_builder),
                 {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
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
    TT_THROW_IF_ERROR(BitwiseRightShiftTensor(self, other, out));
    return out;
  });
}

at::Tensor AtenRsubTensor(const at::Tensor& self, const at::Tensor& other,
                          const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kRsub, param_keys, (self, other, promoted_alpha), {
    TT_THROW_IF_ERROR(CheckAlphaTypeSupported(alpha));
    TT_THROW_IF_ERROR(CheckSubInputs(other, self));

    const at::ScalarType promoted_scalar_type = at::result_type(other, self);
    TT_ASSIGN_OR_THROW(const Dimensions output_dims,
                       InferSize(other.sizes(), self.sizes()));
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(output_dims, promoted_scalar_type, self.device()));

    TT_THROW_IF_ERROR(SubHelperOut(other, self, std::move(promoted_alpha),
                                   promoted_scalar_type, out, param_keys));
    return out;
  });
}

at::Tensor& AtenSubOut(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha, at::Tensor& out) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kSubOut, param_keys, (self, other, promoted_alpha, out), {
    TT_THROW_IF_ERROR(CheckAlphaTypeSupported(alpha));
    TT_THROW_IF_ERROR(CheckSubInputs(self, other));

    TT_THROW_IF_ERROR(SubHelperOut(self, other, std::move(promoted_alpha),
                                   out.scalar_type(), out, param_keys));
    return out;
  });
}

// go/keep-sorted end

}  // namespace torch_tpu
