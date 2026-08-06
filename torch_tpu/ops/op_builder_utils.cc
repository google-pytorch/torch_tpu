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

#include "torch_tpu/ops/op_builder_utils.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/Scalar.h"  // IWYU pragma: keep for c10::Scalar
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/util/complex.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/python_context.h"
#include "tsl/platform/path.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

template <typename T>
std::complex<T> ToStdComplex(const c10::complex<T>& complex_value) {
  return std::complex<T>(complex_value.real(), complex_value.imag());
}

// Given a potential filename, extract the filename without extension and path.
// For example, given "/path/to/file.py", returns "file".
//
// Parsing is optional, if no `/` is present will skip, if no `.` is present
// will skip: i.e. `file.py -> file` and /path/file -> file`.
std::string GetBasename(std::string_view filename) {
  // Parse fullpath
  filename = tsl::io::Basename({filename.data(), filename.size()});

  // Parse extension - TSL tools don't support optional extension parsing.
  const auto ext_dot = filename.rfind('.');
  if (ext_dot != std::string_view::npos) {
    filename = filename.substr(0, ext_dot);
  }
  return std::string(filename);
}

}  // namespace

mlir::stablehlo::Dimensions GetDimensions(mlir::Value value) {
  mlir::FailureOr<mlir::stablehlo::Dimensions> output_dims_or_fail =
      mlir::stablehlo::getDimensions(value);
  ABSL_CHECK(  // CRASH_OK: Internal invariant.
      mlir::succeeded(output_dims_or_fail));
  return std::move(*output_dims_or_fail);
}

mlir::stablehlo::Dimensions GetDimensions(mlir::MlirOp input) {
  return GetDimensions(input.getValue());
}

std::string DebugString(mlir::Operation* absl_nonnull op,
                        DebugStringOptions opts) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  bool enable_debug_info =
      opts == DebugStringOptions::kEnableDebugInfo ? true : false;
  op->print(stream, mlir::OpPrintingFlags().enableDebugInfo(enable_debug_info));
  return result;
}

absl::StatusOr<std::string> SerializeBytecode(mlir::ModuleOp module) {
  std::string bytecode_str;
  llvm::raw_string_ostream os(bytecode_str);
  TT_RET_CHECK(mlir::succeeded(mlir::writeBytecodeToFile(module, os)),
               error::kInternal)
      << "Failed to serialize MLIR module to bytecode.";
  return bytecode_str;
}

absl::StatusOr<std::string> SerializePortableArtifact(mlir::ModuleOp module) {
  std::string portable_artifact_str;
  llvm::raw_string_ostream os(portable_artifact_str);
  // TODO(hyeontaek): Migrate this call to supply a Shardy target version once
  // the new XLA API that accepts `sdy_version` is available for use by
  // torch_tpu.
  TT_ASSIGN_OR_RETURN(
      const std::string serialized,
      xla::SerializeUsingVersionedStablehlo(
          module,
          mlir::vhlo::Version::fromCompatibilityRequirement(
              mlir::vhlo::Version::CompatibilityRequirement::WEEK_4)
              .toString(),
          /*inplace=*/false,
          /*allow_mixed_serialization=*/true),
      _.SetPrepend() << "Failed to serialize MLIR module to StableHLO: ");
  return serialized;
}

mlir::RankedTensorType GetTensorTypeOrDie(const mlir::MlirOp& input) {
  auto ranked_tensor = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  ABSL_CHECK(ranked_tensor)  // CRASH_OK
      << ", where the input is " << debugString(input.getValue())
      << ". This is a torch_tpu bug as it should only call " << __func__
      << " with an input that is a tensor.";
  return ranked_tensor;
}

mlir::Type GetElementTypeOrSelf(mlir::Type type) {
  if (auto ranked = mlir::dyn_cast_or_null<mlir::RankedTensorType>(type)) {
    return ranked.getElementType();
  }
  return type;
}

mlir::Type GetElementTypeOrSelf(mlir::MlirOp type) {
  return GetElementTypeOrSelf(type.getType());
}

absl::StatusOr<mlir::ElementType> GetElementType(mlir::MlirOp op) {
  const mlir::RankedTensorType tensor_type = GetTensorTypeOrDie(op);
  return ConvertTo<mlir::ElementType>(tensor_type.getElementType());
}

mlir::ElementType GetElementTypeOrDie(mlir::MlirOp op) {
  auto elem_type = GetElementType(op);
  ABSL_CHECK(elem_type.ok())  // CRASH_OK
      << ", where the input is " << debugString(op.getValue())
      << ". This is a torch_tpu bug as it should only call " << __func__
      << " with an input that is a tensor.";
  return *elem_type;
}

absl::StatusOr<mlir::Type> GetMlirType(mlir::MLIRContext& ctx,
                                       const mlir::ElementType element_type) {
  ABSL_VLOG(1) << "[GetMlirType] element_type: " << ToString(element_type);
  return mlir::getElementType(ctx, element_type);
}

// REMOVE STATUS RESULT ONCE TESTED
// Using mlir::APFloat libraries to determine min finite value for a given type.
// Adding a comparison to the corresponding XLA type to ensure that the value
// matches. Once we are confident, we don't need this XLA redundancy check.
absl::StatusOr<mlir::DenseElementsAttr> GetVerifiedMinFiniteValue(
    mlir::MlirBuilder& builder, mlir::RankedTensorType constant_tensor_type) {
  mlir::FloatType dtype_float =
      mlir::dyn_cast<mlir::FloatType>(constant_tensor_type.getElementType());
  TT_RET_CHECK(dtype_float, error::kInvalidArgument)
      << "Input type is not float type.";

  mlir::APFloat min_finite_value = mlir::APFloat::getLargest(
      dtype_float.getFloatSemantics(), /*negative=*/true);

  // Return dense element attr for constants
  return mlir::makeConstant(min_finite_value, constant_tensor_type);
}

mlir::Attribute GetMinFiniteValueAttr(mlir::Type element_type,
                                      mlir::OpBuilder& builder) {
  // Check for mlir::FloatType
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(element_type)) {
    llvm::APFloat min_value = llvm::APFloat::getLargest(
        float_type.getFloatSemantics(), /*Negative=*/true);
    return builder.getFloatAttr(float_type, min_value);
  }

  // Check for IntegerType
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
    // Boolean type is signless 1-bit integer which is different from unsigned
    // 1-bit integer and needs to be handled separately.
    if (int_type.isUnsigned() || int_type.getWidth() == 1) {
      return builder.getIntegerAttr(
          int_type, llvm::APInt::getMinValue(int_type.getWidth()));
    }
    llvm::APInt signed_min_value =
        llvm::APInt::getSignedMinValue(int_type.getWidth());
    return builder.getIntegerAttr(int_type, signed_min_value);
  }
  // Return null if the type is not supported
  return nullptr;
}

mlir::Attribute GetMaxFiniteValueAttr(mlir::Type element_type,
                                      mlir::OpBuilder& builder) {
  // Check for mlir::FloatType
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(element_type)) {
    llvm::APFloat max_value = llvm::APFloat::getLargest(
        float_type.getFloatSemantics(), /*Negative=*/false);
    return builder.getFloatAttr(float_type, max_value);
  }

  // Check for IntegerType
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
    // Boolean type is signless 1-bit integer which is different from unsigned
    // 1-bit integer and needs to be handled separately.
    if (int_type.isUnsigned() || int_type.getWidth() == 1) {
      return builder.getIntegerAttr(
          int_type, llvm::APInt::getMaxValue(int_type.getWidth()));
    }
    llvm::APInt signed_max_value =
        llvm::APInt::getSignedMaxValue(int_type.getWidth());
    return builder.getIntegerAttr(int_type, signed_max_value);
  }
  // Return null if the type is not supported
  return nullptr;
}

double GetMaxFiniteDouble(mlir::Type element_type) {
  auto float_type = mlir::dyn_cast<mlir::FloatType>(element_type);
  ABSL_CHECK(float_type)  // CRASH_OK
      << "expected a floating-point type, got " << ToString(element_type);
  return llvm::APFloat::getLargest(float_type.getFloatSemantics(),
                                   /*Negative=*/false)
      .convertToDouble();
}

double GetMinFiniteDouble(mlir::Type element_type) {
  return -GetMaxFiniteDouble(element_type);
}

absl::StatusOr<mlir::MlirOp> MakeZeroSizedTensor(
    mlir::MlirBuilder& builder, mlir::Type element_type,
    mlir::ArrayRef<int64_t> shape) {
  TT_RET_CHECK(absl::c_linear_search(shape, 0), error::kInvalidArgument)
      << "Shape must contain at least one dimension of size 0 to be "
         "zero-sized.";
  auto& ctx = builder.getContext();
  auto tensor_type = mlir::makeTensorType(ctx, shape, element_type);
  return mlir::stablehlo::Constant(
      builder, mlir::makeConstant(mlir::ArrayRef<int64_t>{}, tensor_type));
}

absl::StatusOr<mlir::MlirOp> MakeZeroSizedTensor(
    mlir::MlirBuilder& builder, mlir::ElementType element_type,
    mlir::ArrayRef<int64_t> shape) {
  TT_RET_CHECK(absl::c_linear_search(shape, 0), error::kInvalidArgument)
      << "Shape must contain at least one dimension of size 0 to be "
         "zero-sized.";
  auto& ctx = builder.getContext();
  auto tensor_type = mlir::makeTensorType(ctx, shape, element_type);
  return mlir::stablehlo::Constant(
      builder, mlir::makeConstant(mlir::ArrayRef<int64_t>{}, tensor_type));
}

namespace {

template <typename T>
mlir::DenseElementsAttr MakeConstantAttr(mlir::MLIRContext& ctx, const T value,
                                         mlir::ArrayRef<int64_t> shape,
                                         const mlir::ElementType element_type) {
  return mlir::makeConstant(value,
                            mlir::makeTensorType(ctx, shape, element_type));
}

absl::StatusOr<mlir::DenseElementsAttr> MakeConstantAttr(
    mlir::MLIRContext& ctx, const at::Scalar& scalar_value,
    mlir::ArrayRef<int64_t> shape,
    c10::optional<mlir::ElementType> element_type_opt = std::nullopt) {
  const c10::ScalarType scalar_value_type = scalar_value.type();
  mlir::ElementType element_type;
  if (element_type_opt.has_value()) {
    element_type = *element_type_opt;
  } else {
    // We have to be careful using the result of scalar_value.type() here, since
    // it for example casts all complex types to ComplexDouble
    // (i.e. complex<double>).
    TT_ASSIGN_OR_RETURN(element_type,
                        ConvertTo<mlir::ElementType>(scalar_value_type));
  }
  switch (element_type) {
    // Unsigned ints.
    case mlir::ElementType::UI8:
      return MakeConstantAttr(ctx, scalar_value.toByte(), shape, element_type);
    case mlir::ElementType::UI16:
      return MakeConstantAttr(ctx, scalar_value.toUInt16(), shape,
                              element_type);
    case mlir::ElementType::UI32:
      return MakeConstantAttr(ctx, scalar_value.toUInt32(), shape,
                              element_type);
    case mlir::ElementType::UI64:
      return MakeConstantAttr(ctx, scalar_value.toUInt64(), shape,
                              element_type);
    // Signed ints.
    case mlir::ElementType::I8:
      return MakeConstantAttr(ctx, scalar_value.toChar(), shape, element_type);
    case mlir::ElementType::I16:
      return MakeConstantAttr(ctx, scalar_value.toShort(), shape, element_type);
    case mlir::ElementType::I32:
      return MakeConstantAttr(ctx, scalar_value.toInt(), shape, element_type);
    case mlir::ElementType::I64:
      return MakeConstantAttr(ctx, scalar_value.toLong(), shape, element_type);
    // Floats.
    case mlir::ElementType::BF16:
      return MakeConstantAttr(ctx, scalar_value.toBFloat16(), shape,
                              element_type);
    case mlir::ElementType::F16:
      return MakeConstantAttr(ctx, scalar_value.toHalf(), shape, element_type);
    case mlir::ElementType::F32:
      return MakeConstantAttr(ctx, scalar_value.toFloat(), shape, element_type);
    case mlir::ElementType::F64:
      return MakeConstantAttr(ctx, scalar_value.toDouble(), shape,
                              element_type);
    // Complex types.
    case mlir::ElementType::COMPLEXF32:
      return MakeConstantAttr(ctx, ToStdComplex(scalar_value.toComplexFloat()),
                              shape, element_type);
    case mlir::ElementType::COMPLEXF64:
      return MakeConstantAttr(ctx, ToStdComplex(scalar_value.toComplexDouble()),
                              shape, element_type);
      // Boolean.
    case mlir::ElementType::PRED:
      return MakeConstantAttr(ctx, scalar_value.toBool(), shape, element_type);
      // Deliberately no default case to catch new cases at compile time.
    // The following cases are not supported.
    // go/keep-sorted start
    case mlir::ElementType::F4E2M1FN:
    case mlir::ElementType::F6E2M3FN:
    case mlir::ElementType::F6E3M2FN:
    case mlir::ElementType::F8E3M4:
    case mlir::ElementType::F8E4M3:
    case mlir::ElementType::F8E4M3B11FNUZ:
    case mlir::ElementType::F8E4M3FN:
    case mlir::ElementType::F8E4M3FNUZ:
    case mlir::ElementType::F8E5M2:
    case mlir::ElementType::F8E5M2FNUZ:
    case mlir::ElementType::F8E8M0FNU:
    case mlir::ElementType::I2:
    case mlir::ElementType::I4:
    case mlir::ElementType::UI2:
    case mlir::ElementType::UI4:
      // go/keep-sorted end
      break;
  }
  return TT_ERROR(error::kInvalidArgument)
         << "unsupported at::Scalar type "
         << static_cast<int>(scalar_value_type);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> MakeConstant(
    mlir::MlirBuilder& builder, const at::Scalar& value,
    c10::optional<mlir::ElementType> element_type_opt,
    mlir::ArrayRef<int64_t> shape) {
  TT_ASSIGN_OR_RETURN(
      mlir::DenseElementsAttr attr,
      MakeConstantAttr(builder.getContext(), value, shape, element_type_opt));
  return mlir::stablehlo::Constant(builder, std::move(attr));
}

absl::StatusOr<mlir::MlirOp> MakeConstantLike(mlir::MlirOp input,
                                              const at::Scalar& value) {
  mlir::MlirBuilder& builder = input.getBuilder();
  auto input_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(
      const auto element_type,
      ConvertTo<mlir::ElementType>(input_type.getElementType()));

  if (input_type.hasStaticShape()) {
    return MakeConstant(builder, value, element_type, input_type.getShape());
  }

  // Handle bounded dynamism - make scalar constant and use broadcast APIs
  TT_ASSIGN_OR_RETURN(
      mlir::DenseElementsAttr attr,
      MakeConstantAttr(builder.getContext(), value, mlir::ArrayRef<int64_t>{},
                       c10::make_optional(element_type)));
  return mlir::chlo::ConstantLike(input, attr);
}

absl::StatusOr<mlir::MlirOp> BroadcastIfNeeded(
    mlir::MlirOp input, absl::Span<const int64_t> shape) {
  ABSL_VLOG(1) << "[BroadcastIfNeeded] input: " << input.ToString()
               << " shape: " << ToString(shape);
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  // TODO(b/460206467): Update all broadcast APIs to support dynamic shapes.
  ABSL_CHECK(  // CRASH_OK: Pending feature work, API needs refactor.
      input_type.getNumDynamicDims() == 0)
      << "Input shape must be static to use static `BroadcastIfNeeded` API.";
  auto input_shape = input_type.getShape();
  if (input_shape == AsArrayRef(shape)) {
    return input;
  }
  TT_RET_CHECK(input_shape.size() <= shape.size(), error::kInvalidArgument)
      << "Input shape cannot be larger than output shape.";
  size_t rank_diff = shape.size() - input_shape.size();
  Dimensions bcast_dims;
  bcast_dims.reserve(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); ++i) {
    TT_RET_CHECK(input_shape[i] == 1 || input_shape[i] == shape[i + rank_diff],
                 error::kInvalidArgument)
        << "Cannot broadcast input shape: "
        << ToString(absl::MakeSpan(input_shape))
        << " to target shape: " << ToString(shape);
    bcast_dims.push_back(i + rank_diff);
  }
  mlir::RankedTensorType output_type = input_type.clone(AsArrayRef(shape));
  return stablehlo::BroadcastInDim(output_type, input, bcast_dims);
}

absl::StatusOr<mlir::MlirOp> BroadcastIfNeeded(
    mlir::MlirOp input, mlir::stablehlo::Dimensions shape,
    absl::Span<const int64_t> broadcast_dimensions) {
  mlir::BaseScopedDiagnosticHandler diag_handler(input.getValue().getContext());
  mlir::FailureOr<mlir::Value> broadcasted_value_or_fail;
  if (!broadcast_dimensions.empty()) {
    broadcasted_value_or_fail = mlir::stablehlo::broadcastIfNeeded(
        input.getBuilder().getOpBuilder(), input.getValue(), shape,
        AsArrayRef(broadcast_dimensions));
  } else {
    broadcasted_value_or_fail = mlir::stablehlo::numpyBroadcastIfNeeded(
        input.getBuilder().getOpBuilder(), input.getValue(), shape);
  }
  TT_RET_CHECK(mlir::succeeded(broadcasted_value_or_fail),
               error::kInvalidArgument)
      << "failed to broadcast tensor: "
      << diag_handler.ConsumeStatus().message();
  return mlir::MlirOp(input.getBuilder(), *broadcasted_value_or_fail);
}

absl::StatusOr<mlir::MlirOp> BroadcastIfNeeded(
    mlir::MlirOp input, mlir::MlirOp target,
    absl::Span<const int64_t> broadcast_dimensions) {
  mlir::BaseScopedDiagnosticHandler diag_handler(
      target.getValue().getContext());
  mlir::stablehlo::Dimensions target_dims = GetDimensions(target);

  mlir::FailureOr<mlir::Value> broadcasted_value_or_fail;
  if (!broadcast_dimensions.empty()) {
    broadcasted_value_or_fail = mlir::stablehlo::broadcastIfNeeded(
        input.getBuilder().getOpBuilder(), input.getValue(), target_dims,
        AsArrayRef(broadcast_dimensions));
  } else {
    broadcasted_value_or_fail = mlir::stablehlo::numpyBroadcastIfNeeded(
        input.getBuilder().getOpBuilder(), input.getValue(), target_dims);
  }

  TT_RET_CHECK(mlir::succeeded(broadcasted_value_or_fail),
               error::kInvalidArgument)
      << "failed to broadcast tensor: "
      << diag_handler.ConsumeStatus().message();
  return mlir::MlirOp(input.getBuilder(), *broadcasted_value_or_fail);
}

absl::StatusOr<Dimensions> GetBroadcastShape(
    absl::Span<const mlir::MlirOp> ops) {
  TT_RET_CHECK(!ops.empty(), error::kInvalidArgument)
      << "no tensor to broadcast";
  Dimensions bcast_shape;  // An empty shape can be broadcasted to any shape.
  for (int i = 0; i < ops.size(); ++i) {
    const mlir::RankedTensorType tensor_type = GetTensorTypeOrDie(ops[i]);
    // TODO(b/460206467): Update all broadcast APIs to support dynamic shapes.
    ABSL_CHECK(  // CRASH_OK: Pending feature work, API needs refactor.
        tensor_type.getNumDynamicDims() == 0)
        << "Input shape must be static to use static `GetBroadcastShape` "
           "API.";
    TT_ASSIGN_OR_RETURN(bcast_shape,
                        InferSize(bcast_shape, tensor_type.getShape()),
                        _.SetPrepend() << "tensor " << i << ": ");
  }
  return std::move(bcast_shape);
}

absl::StatusOr<mlir::MlirOp> Broadcast(mlir::MlirOp input,
                                       absl::Span<const int64_t> output_dims,
                                       absl::Span<const int64_t> bcast_dims) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  ABSL_VLOG(3) << "[Broadcast] input_type: " << mlir::debugString(input_type)
               << " output_dims: " << ToString(output_dims)
               << " bcast_dims: " << ToString(bcast_dims);

  const int64_t input_rank = input_type.getRank();
  const int64_t output_rank = output_dims.size();

  TT_RET_CHECK(input_rank <= output_rank, error::kInvalidArgument)
      << "input rank " << input_rank << " must not be more than output rank "
      << output_rank << " for broadcast of input "
      << mlir::debugString(input_type) << " to output shape "
      << ToString(output_dims);

  if (input_type.hasStaticShape()) {
    auto output_type = input_type.clone(AsArrayRef(output_dims));
    return mlir::stablehlo::BroadcastInDim(output_type, input, bcast_dims);
  }

  //
  // Handle bounded dynamism on input.
  //
  // Bound is only propagated to the corresponding broadcast dimension.
  // For example, if input is [8, 10 (<=100), 128] with dim=1 being bounded and
  // we broadcast to output shape [1, 8, 2, 10, 128] with
  // broadcast_dims=[1, 3, 4], then bound is propagated to the dim=3 of the
  // output shape making it [1, 8, 2, 10 (<=100), 128].
  //
  // CAVEAT: Note that, the bound may not be correctly set on the input.
  // For example:
  // x = torch.randn(8, 1, 10, 128) where x.shape[2] is bounded to 100.
  // y = x.expand(x.shape[0], x.shape[2], x.shape[2], x.shape[3])
  // Ideally, the bound should be set on both the dim=1 and dim=2 of the tensor
  // y. However, the bound is set only on dim=2 of the tensor y, as in eager
  // mode there is no way to determine if x.shape[2] is bounded or not.
  // The view decomposition logic determines the view sequence to go from
  // [8, 1, 10 (<=100), 128] to [8, 10, 10, 128] where dim=1 is the replicated
  // dimension and dim=2 is the broadcasted dimension of the output making
  // it [8, 10, 10 (<=100), 128].
  //

  stablehlo::Dimensions broadcast_shape(output_dims.size());
  for (int i = 0; i < output_dims.size(); ++i) {
    // Initialize broadcast shape to output shape.
    broadcast_shape[i] = {.size = output_dims[i]};
  }
  stablehlo::Dimensions input_dims = GetDimensions(input);
  for (int i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i].boundOp.has_value()) {
      // Broadcast to padded size for bounded dimension.
      const int64_t broadcast_dim = bcast_dims[i];
      broadcast_shape[broadcast_dim] = {.size = input_dims[i].size};
    }
  }

  mlir::RankedTensorType broadcast_shape_type =
      getRankedTensorType(broadcast_shape, input_type.getElementType());
  ABSL_VLOG(3) << "[Broadcast] bcastType: "
               << mlir::debugString(broadcast_shape_type);

  mlir::MlirOp broadcasted_op =
      mlir::stablehlo::BroadcastInDim(broadcast_shape_type, input, bcast_dims);
  const mlir::RankedTensorType broadcasted_op_type =
      GetTensorTypeOrDie(broadcasted_op);
  ABSL_VLOG(3) << "[Broadcast] broadcasted_op_type: "
               << mlir::debugString(broadcasted_op_type);

  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i].boundOp.has_value()) {
      mlir::MlirOp boundOp =
          mlir::MlirOp(input.getBuilder(), *input_dims[i].boundOp);
      auto dimSize =
          stablehlo::GetDimensionSize(boundOp, input_dims[i].boundOpDim);
      const int64_t broadcast_dim = bcast_dims[i];
      broadcasted_op =
          stablehlo::SetDimensionSize(broadcasted_op, dimSize, broadcast_dim);
    }
  }

  const mlir::RankedTensorType broadcasted_op_updated_type =
      GetTensorTypeOrDie(broadcasted_op);
  ABSL_VLOG(3) << "[Broadcast] broadcasted_op_updated_type: "
               << mlir::debugString(broadcasted_op_updated_type);

  return broadcasted_op;
}

absl::StatusOr<std::pair<mlir::MlirOp, mlir::MlirOp>> ConvertIfIntegers(
    mlir::MlirOp op1, mlir::MlirOp op2, mlir::ElementType target_dtype) {
  ABSL_VLOG(1) << "[ConvertIfIntegers] op1: " << op1.ToString()
               << "\nop2: " << op2.ToString();

  const mlir::RankedTensorType op1_type = GetTensorTypeOrDie(op1);
  const mlir::RankedTensorType op2_type = GetTensorTypeOrDie(op2);
  if (op1_type.getElementType().isInteger()) {
    op1 = stablehlo::ConvertElementType(op1, target_dtype);
  }
  if (op2_type.getElementType().isInteger()) {
    op2 = stablehlo::ConvertElementType(op2, target_dtype);
  }
  return std::make_pair(op1, op2);
}

absl::StatusOr<mlir::MlirOp> ConvertIfInteger(mlir::MlirOp op,
                                              mlir::ElementType target_dtype) {
  ABSL_VLOG(1) << "[ConvertIfInteger] op: " << op.ToString();
  const mlir::RankedTensorType op_type = GetTensorTypeOrDie(op);
  if (op_type.getElementType().isInteger()) {
    return stablehlo::ConvertElementType(op, target_dtype);
  }
  return op;
}

absl::StatusOr<DynamicMlirOpResults> ToResultVector(
    absl::StatusOr<mlir::MlirOp> results) {
  TT_ASSIGN_OR_RETURN(auto results_value, results);
  return DynamicMlirOpResults{results_value};
}

std::string BuildModuleNameFromPyContext(
    mlir::MLIRContext& mlir_context,
    const PythonContext* absl_nullable python_context) {
  std::string module_name;
  llvm::raw_string_ostream os(module_name);
  os << "tt_jit";
  if (!python_context) {
    return module_name;
  }

  os << "_";
  mlir::Location location =
      torch_tpu::MakeMlirLocation(mlir_context, *python_context);

  // Traverse the location from the current frame up through callee frames
  // until we hit a viable user frame of the form
  //   NameLoc(funcname, FileLineColLoc(filename, line, col));
  // See `torch_tpu::MakeMlirLocation` for more details on this format.
  auto result =
      location->walk([&](mlir::Location child_loc) -> mlir::WalkResult {
        ABSL_VLOG(1) << "[BuildModuleNameFromPyContext] childLoc: "
                     << mlir::debugString(child_loc);
        auto name_loc = mlir::dyn_cast<mlir::NameLoc>(child_loc);
        if (!name_loc) return mlir::WalkResult::advance();
        auto file_line_col_loc =
            mlir::dyn_cast<mlir::FileLineColRange>(name_loc.getChildLoc());
        if (!file_line_col_loc) return mlir::WalkResult::advance();

        const mlir::StringRef func_name = name_loc.getName();
        const mlir::StringRef filename = file_line_col_loc.getFilename();
        const std::string basename =
            GetBasename({filename.data(), filename.size()});

        // Build module name of form `file_L#C#_func`
        os << basename << "_";
        os << "L" << file_line_col_loc.getStartLine();
        os << "C" << file_line_col_loc.getStartColumn();
        os << "_" << func_name;
        return mlir::WalkResult::interrupt();
      });

  // Append the op call chain that caused the materialization
  if (result.wasInterrupted()) {
    os << "_";
  }
  llvm::interleave(python_context->op_call_chain(), os, "_");
  return module_name;
}

void AnnotateBufferDonations(mlir::ModuleOp module,
                             mlir::ArrayRef<int64_t> donated_inputs) {
  // Nothing donated, skip.
  if (donated_inputs.empty()) return;

  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  ABSL_CHECK(main)  // CRASH_OK: Should never call API on a malformed module.
      << "MLIR module does not contain a main function for buffer donation.";

  mlir::Builder builder(main.getContext());
  for (int64_t input_idx : donated_inputs) {
    ABSL_CHECK(  // CRASH_OK: Only an infra bug could cause a bad donation idx
        input_idx >= 0 && input_idx < main.getNumArguments())
        << "donated_input " << input_idx << " is out of range [0, "
        << main.getNumArguments() << ")";
    main.setArgAttr(input_idx, "jax.buffer_donor", builder.getBoolAttr(true));
  }
}

absl::StatusOr<mlir::MlirOp> CastIfNeeded(
    mlir::MlirOp op, mlir::ElementType expected_output_type) {
  const mlir::RankedTensorType current_type = GetTensorTypeOrDie(op);

  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType actual_element_type,
      ConvertTo<mlir::ElementType>(current_type.getElementType()));
  const auto expected_scalar_type =
      ConvertTo<at::ScalarType>(expected_output_type);
  const auto actual_scalar_type =
      ConvertTo<at::ScalarType>(actual_element_type);
  TT_RET_CHECK(at::canCast(actual_scalar_type, expected_scalar_type),
               error::kInvalidArgument)
      << "result type " << ToString(actual_scalar_type)
      << " can't be cast to the desired output type "
      << ToString(expected_scalar_type);
  if (actual_element_type != expected_output_type) {
    ABSL_VLOG(3) << "[CastIfNeeded]: Casting needed. "
                 << "current element type: " << ToString(actual_element_type)
                 << ", expected output type: "
                 << ToString(expected_output_type);
    return mlir::stablehlo::ConvertElementType(op, expected_output_type);
  }
  ABSL_VLOG(3) << "[CastIfNeeded]: No casting needed. "
               << "Current and expected types are the same: "
               << ToString(actual_element_type);
  return op;
}

absl::Status BuildReduceBody(mlir::RegionBuilder& rb, mlir::Type element_type,
                             c10d::ReduceOp::RedOpType reduce_op_type) {
  switch (reduce_op_type) {
    case c10d::ReduceOp::SUM:
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          element_type, rb.getRegion(), rb.getOpBuilder());
      break;
    case c10d::ReduceOp::PRODUCT:
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MulOp>(
          element_type, rb.getRegion(), rb.getOpBuilder());
      break;
    case c10d::ReduceOp::MIN:
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MinOp>(
          element_type, rb.getRegion(), rb.getOpBuilder());
      break;
    case c10d::ReduceOp::MAX:
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
          element_type, rb.getRegion(), rb.getOpBuilder());
      break;
    case c10d::ReduceOp::BAND:
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AndOp>(
          element_type, rb.getRegion(), rb.getOpBuilder());
      break;
    case c10d::ReduceOp::BOR:
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::OrOp>(
          element_type, rb.getRegion(), rb.getOpBuilder());
      break;
    case c10d::ReduceOp::BXOR:
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::XorOp>(
          element_type, rb.getRegion(), rb.getOpBuilder());
      break;
    case c10d::ReduceOp::AVG:
      return TT_ERROR(error::kInvalidArgument)
             << "BuildReduceBody: 'avg' reduction should be handled using "
                "'sum'";
    case c10d::ReduceOp::PREMUL_SUM:
      return TT_ERROR(error::kInvalidArgument)
             << "BuildReduceBody: 'premul_sum' reduction is not supported";
    case c10d::ReduceOp::UNUSED:
      return TT_ERROR(error::kInvalidArgument)
             << "BuildReduceBody: 'unused' is not a valid reduction type";
      // default case not needed, since the switch is exhaustive.
  }
  return absl::OkStatus();
}

mlir::MlirOp BuildFillUninitialized(mlir::MlirBuilder& builder,
                                    mlir::ElementType tensor_element_type,
                                    absl::Span<const int64_t> shape) {
  switch (tensor_element_type) {
    case mlir::ElementType::PRED:
      return MakeConstant(builder, std::numeric_limits<bool>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::I2:
      return MakeConstant(builder, 1, tensor_element_type, shape);
    case mlir::ElementType::I4:
      return MakeConstant(builder, 7, tensor_element_type, shape);
    case mlir::ElementType::I8:
      return MakeConstant(builder, std::numeric_limits<int8_t>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::I16:
      return MakeConstant(builder, std::numeric_limits<int16_t>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::I32:
      return MakeConstant(builder, std::numeric_limits<int32_t>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::I64:
      return MakeConstant(builder, std::numeric_limits<int64_t>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::UI2:
      return MakeConstant(builder, 3, tensor_element_type, shape);
    case mlir::ElementType::UI4:
      return MakeConstant(builder, 15, tensor_element_type, shape);
    case mlir::ElementType::UI8:
      return MakeConstant(builder, std::numeric_limits<uint8_t>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::UI16:
      return MakeConstant(builder, std::numeric_limits<uint16_t>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::UI32:
      return MakeConstant(builder, std::numeric_limits<uint32_t>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::UI64:
      return MakeConstant(builder, std::numeric_limits<uint64_t>::max(),
                          tensor_element_type, shape);
    case mlir::ElementType::BF16:
    case mlir::ElementType::F16:
    case mlir::ElementType::F32:
    case mlir::ElementType::F4E2M1FN:
    case mlir::ElementType::F6E2M3FN:
    case mlir::ElementType::F6E3M2FN:
    case mlir::ElementType::F8E3M4:
    case mlir::ElementType::F8E4M3:
    case mlir::ElementType::F8E4M3FN:
    case mlir::ElementType::F8E4M3FNUZ:
    case mlir::ElementType::F8E4M3B11FNUZ:
    case mlir::ElementType::F8E5M2:
    case mlir::ElementType::F8E5M2FNUZ:
    case mlir::ElementType::F8E8M0FNU:
    case mlir::ElementType::COMPLEXF32:
      return MakeConstant(builder, std::numeric_limits<float>::quiet_NaN(),
                          tensor_element_type, shape);
    case mlir::ElementType::F64:
    case mlir::ElementType::COMPLEXF64:
      return MakeConstant(builder, std::numeric_limits<double>::quiet_NaN(),
                          tensor_element_type, shape);
      // Default case deliberately omitted to ensure compile-time errors if new
      // element types are not handled.
  }
}

mlir::MlirOp GetNumElements(mlir::MlirOp input, mlir::Type element_type,
                            mlir::ArrayRef<int64_t> dims) {
  // Set `dims` to all dimensions if not specified.
  mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  mlir::SmallVector<int64_t> all_dims(type.getRank());
  if (dims.empty()) {
    std::iota(all_dims.begin(), all_dims.end(), 0);
    dims = all_dims;
  }

  // Count all static dimensions, keep track if dynamic dim is queried.
  bool found_dynamic = false;
  auto shape = type.getShape();
  int64_t num_elements = 1;
  for (int64_t dim : dims) {
    bool is_dynamic = mlir::ShapedType::isDynamic(shape[dim]);
    found_dynamic |= is_dynamic;
    if (!is_dynamic) {
      num_elements *= shape[dim];
    }
  }

  // Static case - emit constant.
  mlir::MlirOp num_elements_op =
      MakeScalarConstant(input.getBuilder(), num_elements, element_type);
  if (!found_dynamic) {
    return num_elements_op;
  }

  // Bounded dynamic case - multiply static dim count by dynamic dims.
  for (int64_t dim : dims) {
    if (mlir::ShapedType::isDynamic(shape[dim])) {
      mlir::MlirOp dim_size = mlir::stablehlo::GetDimensionSize(input, dim);
      mlir::MlirOp dim_size_cast =
          mlir::stablehlo::ConvertElementType(dim_size, element_type);
      num_elements_op = mlir::stablehlo::Mul(num_elements_op, dim_size_cast);
    }
  }
  return num_elements_op;
}

mlir::ModuleOp GetModuleOp(mlir::MlirBuilder& builder) {
  return builder.getOpBuilder()
      .getBlock()
      ->getParentOp()
      ->getParentOfType<mlir::ModuleOp>();
}

absl::StatusOr<mlir::MlirOp> Unsqueeze(mlir::MlirOp input, int64_t dim) {
  // Get input dimensions
  auto type = GetTensorTypeOrDie(input);
  mlir::stablehlo::Dimensions output_dims = GetDimensions(input);

  // Insert a new dimension at the given position.
  int64_t rank = type.getRank();
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, rank + 1));
  output_dims.insert(output_dims.begin() + dim, {1});

  // Build reshape op
  // Dim expanding reshapes support dynamic shapes.
  auto result_type =
      mlir::stablehlo::getRankedTensorType(output_dims, type.getElementType());
  return mlir::stablehlo::Reshape(result_type, input);
}

absl::StatusOr<mlir::MlirOp> Squeeze(mlir::MlirOp input,
                                     absl::Span<const int64_t> dims) {
  auto type = GetTensorTypeOrDie(input);
  auto input_shape = type.getShape();
  int64_t rank = type.getRank();
  std::vector<bool> dim_squeezed(rank, false);
  bool will_squeeze = false;
  for (int64_t dim : dims) {
    int64_t d = dim < 0 ? dim + rank : dim;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Current usages are guaranteed to be
                   // within range.
        d >= 0 && d < rank, error::kInvalidArgument)
        << "invalid dimension, expected to be within [-" << rank << ", " << rank
        << "), got " << dim << " for tensor with " << rank << " dimensions";
    if (input_shape[d] == 1) {
      dim_squeezed[d] = true;
      will_squeeze = true;
    }
  }

  if (!will_squeeze) {
    return input;
  }

  Dimensions output_dims;
  for (int64_t i = 0; i < rank; ++i) {
    if (!dim_squeezed[i]) {
      output_dims.push_back(input_shape[i]);
    }
  }

  auto result_type =
      mlir::RankedTensorType::get(output_dims, type.getElementType());
  return mlir::stablehlo::Reshape(result_type, input);
}

mlir::MlirOp Flatten(mlir::MlirOp input) {
  auto type = GetTensorTypeOrDie(input);
  if (type.hasStaticShape()) {
    return mlir::stablehlo::Reshape(input, {type.getNumElements()});
  }
  // Calculate the new upper bound by multiplying static extents
  // i.e. [1,2,<=10] -> [<=20]
  mlir::stablehlo::Dimensions dims = GetDimensions(input);
  int64_t new_bound = llvm::accumulate(
      dims, 1, [](int64_t a, const mlir::stablehlo::DimensionInfo& b) {
        return a * b.size;
      });
  mlir::stablehlo::DimensionInfo new_dim{.size = new_bound,
                                         .boundOp = input.getValue()};
  mlir::RankedTensorType new_type =
      stablehlo::getRankedTensorType({new_dim}, type.getElementType());
  // Reshape seems to work for dynamic flatten, if any bugs pop up we can flip
  // this to use dynamic_reshape instead.
  return stablehlo::Reshape(new_type, input);
}

// Slice if left_pad or right_pad is negative to truncate the vector,
// otherwise return op.
absl::StatusOr<mlir::MlirOp> BuildMaybeSlice(mlir::MlirOp op, int64_t dimension,
                                             int64_t left_pad,
                                             int64_t right_pad) {
  if (left_pad >= 0 && right_pad >= 0) {
    return op;
  }

  auto input_tensor_type = GetTensorTypeOrDie(op);
  Dimensions input_shape(input_tensor_type.getShape().begin(),
                         input_tensor_type.getShape().end());

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Current usages are guaranteed to be
                 // within range.
      dimension < input_shape.size(), error::kInvalidArgument)
      << "dimension " << dimension << " is out of range [0, "
      << input_shape.size() << ")";

  // Left and right dimensions for slicing (a vector of 0s and of max vals)
  // and a vector of stride 1.
  auto left_dim = Dimensions(input_shape.size(), 0);
  auto right_dim = input_shape;
  if (left_pad < 0) {
    left_dim[dimension] -= left_pad;
  }
  if (right_pad < 0) {
    right_dim[dimension] = input_shape[dimension] + right_pad;
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Current usages are guaranteed to be
                 // within range.
      left_dim[dimension] < right_dim[dimension], error::kInvalidArgument)
      << "padding values for dimension " << dimension
      << " exceeds tensor size.";
  auto strides = Dimensions(input_shape.size(), 1);

  return mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
}

Dimensions GetAllDimensions(mlir::MlirOp op) {
  Dimensions dims(GetTensorTypeOrDie(op).getRank());
  absl::c_iota(dims, 0);
  return dims;
}

absl::StatusOr<mlir::MlirOp> PromoteFloatDtype(
    mlir::MlirOp op, mlir::ElementType min_precision) {
  mlir::RankedTensorType op_type = GetTensorTypeOrDie(op);
  mlir::Type min_type = mlir::getElementType(op.getContext(), min_precision);
  if (!IsFloatType(op_type)) {
    return op;
  }
  if (!min_type.isFloat()) {
    return TT_ERROR(error::kInvalidArgument)
           << "min_precision must be a float type, got "
           << mlir::debugString(min_type);
  }
  if (op_type.getElementTypeBitWidth() >= min_type.getIntOrFloatBitWidth()) {
    return op;
  }
  return CastIfNeeded(op, min_precision);
}

absl::StatusOr<mlir::MlirOp> BuildRngStateUpdateShlo(mlir::MlirOp state,
                                                     int64_t num_elements,
                                                     int64_t bit_width) {
  ABSL_CHECK_GE(num_elements, 0)  // CRASH_OK
      << "num_elements must be non-negative.";
  ABSL_CHECK_GE(bit_width, 0) << "bit_width must be non-negative.";  // CRASH_OK

  // Do the computation as unsigned numbers to match CUDA behavior.
  const uint64_t num_elements_u64 = static_cast<uint64_t>(num_elements);
  const uint64_t bit_width_u64 = static_cast<uint64_t>(bit_width);

  auto& builder = state.getBuilder();

  // Note that num_bits = num_elements * bit_width.
  // Each Philox step consumes 128 bits of randomness and increments the offset
  // by 1.
  TT_ASSIGN_OR_RETURN(const uint64_t num_bits_u64,
                      SafeMultiply(num_elements_u64, bit_width_u64));
  // Round up to the nearest multiple of 128. We deliberately want overflows to
  // wrap around, matching CUDA behavior.
  const uint64_t increment_val_u64 = (num_bits_u64 + 127u) / 128u;

  // Create a constant tensor [0, increment_val]
  mlir::RankedTensorType tensor_ui64_type =
      mlir::makeTensorType(builder.getContext(), {2}, mlir::ElementType::UI64);
  mlir::DenseElementsAttr value_attr = mlir::DenseElementsAttr::get(
      tensor_ui64_type, llvm::ArrayRef<uint64_t>({0, increment_val_u64}));
  mlir::MlirOp increment_tensor =
      mlir::stablehlo::Constant(builder, value_attr);

  // output_state[0] = initial_state[0] + 0
  // output_state[1] = initial_state[1] + increment_val_u64
  return {{mlir::stablehlo::Add(state, increment_tensor)}};
}

//////
// Reshape Utilities

namespace {
std::string ReshapeTypeToString(ReshapeType type) {
  switch (type) {
    case ReshapeType::kCollapse:
      return "collapse";
    case ReshapeType::kFlatten:
      return "flatten";
    case ReshapeType::kExpand:
      return "expand";
    case ReshapeType::kTransposeLike:
      return "transpose_like";
    case ReshapeType::kUnknown:
      return "unknown";
  }
}

// Returns a debug string for shape transitions of form: `from [1,2] to [3,4]`.
std::string ShapeTransitionToString(absl::Span<const int64_t> shape_before,
                                    absl::Span<const int64_t> shape_after) {
  return absl::StrCat("from ", ToString(shape_before), " to ",
                      ToString(shape_after));
}

std::optional<llvm::SmallVector<mlir::ReassociationIndices>>
GetReassociationIndicesForReshape(const Dimensions& static_shape_before,
                                  const Dimensions& static_shape_after) {
  if (static_shape_before.size() > static_shape_after.size())
    return mlir::getReassociationIndicesForCollapse(static_shape_before,
                                                    static_shape_after);
  if (static_shape_before.size() < static_shape_after.size())
    return mlir::getReassociationIndicesForCollapse(static_shape_after,
                                                    static_shape_before);
  return std::nullopt;
}

absl::Status HandleCollapseReshape(
    mlir::MlirOp op, const ReshapeReassociation& reassociation,
    const mlir::stablehlo::Dimensions& op_dims,
    const Dimensions& static_shape_after,
    mlir::SmallVector<mlir::stablehlo::DimensionInfo>& output_dims) {
  // Collapse: reassociation[output_idx] gives {input_idx, ...}
  // Ex: [1, 2, 3, 4] -> [1, 2, 12]
  // reassociation[outputIdx = 0] = inputIdx 0
  // reassociation[outputIdx = 1] = inputIdx 1
  // reassociation[outputIdx = 2] = inputIdx 2, 3
  for (int64_t outputIdx = 0; outputIdx < reassociation.reassociation.size();
       ++outputIdx) {
    const auto& group = reassociation.reassociation[outputIdx];
    bool contains_dynamic_input = false;
    for (int64_t inputIdx : group) {
      if (op_dims[inputIdx].boundOp.has_value()) {
        contains_dynamic_input = true;
        break;
      }
    }
    if (!contains_dynamic_input) continue;

    const int64_t output_group_bounded_size = std::accumulate(
        group.begin(), group.end(), 1L, [&](int64_t acc, int64_t inputIdx) {
          return acc * op_dims[inputIdx].size;
        });
    int64_t output_bound_dim = outputIdx;
    int64_t output_dyn_bound = output_group_bounded_size;

    ABSL_CHECK(  // CRASH_OK=should not happen, would imply a bug in the code
        static_shape_after[output_bound_dim] <= output_dyn_bound)
        << "output static shape at bound dim " << output_bound_dim
        << " for output shape [" << absl::StrJoin(static_shape_after, ",")
        << "] is greater than the inferred bound of " << output_dyn_bound;

    output_dims[output_bound_dim] = {.size = output_dyn_bound,
                                     .boundOp = op.getValue()};

    ABSL_VLOG(3) << "[HandleCollapseReshape] outputIdx: " << outputIdx
                 << " output_dyn_bound: " << output_dyn_bound;
  }
  return absl::OkStatus();
}

absl::Status HandleExpandReshape(
    mlir::MlirOp op, const ReshapeReassociation& reassociation,
    const mlir::stablehlo::Dimensions& op_dims,
    const Dimensions& static_shape_before, const Dimensions& static_shape_after,
    mlir::SmallVector<mlir::stablehlo::DimensionInfo>& output_dims) {
  // Expansion: reassociation[input_idx] gives {output_idx, ...}
  // Ex: [1, 2, 12] -> [1, 2, 3, 4]
  // reassociation[inputIdx = 0] = outputIdx 0
  // reassociation[inputIdx = 1] = outputIdx 1
  // reassociation[inputIdx = 2] = outputIdx 2, 3
  for (int64_t inputIdx = 0; inputIdx < op_dims.size(); ++inputIdx) {
    if (!op_dims[inputIdx].boundOp.has_value()) continue;

    ABSL_CHECK(  // CRASH_OK=would imply a bug in
                 // getReassociationIndicesForReshape
        inputIdx < reassociation.reassociation.size())
        << "invalid reassociation map, input bound dim " << inputIdx
        << " is out of bounds for reassociation "
        << ShapeTransitionToString(static_shape_before, static_shape_after);
    const auto& outputGroup = (reassociation.reassociation)[inputIdx];
    int64_t output_bound_dim = -1;
    int64_t output_dyn_bound = -1;

    if (outputGroup.size() == 1) {
      output_bound_dim = outputGroup[0];
      output_dyn_bound =
          op_dims[inputIdx].size;  // Bound is directly transferred
    } else {
      Dimensions non_one_output_dims;
      absl::c_copy_if(
          outputGroup, std::back_inserter(non_one_output_dims),
          [&](int64_t dim) { return static_shape_after[dim] != 1; });
      if (non_one_output_dims.size() == 1) {
        output_bound_dim = non_one_output_dims[0];
        output_dyn_bound =
            op_dims[inputIdx].size;  // Bound is directly transferred
      } else {
        return TT_ERROR(error::kInvalidArgument)
               << "unflatten ambiguous as input bound dim " << inputIdx
               << " expands to multiple non one output dims "
               << absl::StrJoin(non_one_output_dims, ",")
               << " for reassociation "
               << ShapeTransitionToString(static_shape_before,
                                          static_shape_after);
      }
    }
    ABSL_CHECK(  // CRASH_OK=should not happen, would imply a bug in the code
        static_shape_after[output_bound_dim] <= output_dyn_bound)
        << "output static shape at bound dim " << output_bound_dim
        << " for output shape [" << absl::StrJoin(static_shape_after, ",")
        << "] is greater than the inferred bound of " << output_dyn_bound;
    output_dims[output_bound_dim] = {.size = output_dyn_bound,
                                     .boundOp = op.getValue()};
    ABSL_VLOG(3) << "[HandleExpandReshape] output_bound_dim: "
                 << output_bound_dim
                 << " output_dyn_bound: " << op_dims[inputIdx].size;
  }
  return absl::OkStatus();
}

Dimensions GetCoreShape(const Dimensions& shape) {
  Dimensions core_shape;
  for (int64_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != 1) {
      core_shape.push_back(shape[i]);
    }
  }
  return core_shape;
}

absl::Status HandleTransposeLikeReshape(
    mlir::MlirOp op, const mlir::stablehlo::Dimensions& op_dims,
    const Dimensions& shape_before, const Dimensions& shape_after,
    mlir::SmallVector<mlir::stablehlo::DimensionInfo>& output_dims) {
  // Transpose-like: ranks are same, non-1 dimensions don't move relative to
  // each other. We extract core shapes and mappings to transfer bounds.
  ABSL_CHECK(  // CRASH_OK=a valid transpose-like reshape requires this.
      shape_before.size() == shape_after.size())
      << "mismatched ranks for transpose-like reshape "
      << ShapeTransitionToString(shape_before, shape_after);
  int rank = shape_before.size();

  for (int i = 0, j = 0; i < rank || j < rank; ++i, ++j) {
    // Find the next non-one dimension in both input and output shapes.
    while (i < rank && shape_before[i] == 1) {
      ++i;
    }
    while (j < rank && shape_after[j] == 1) {
      ++j;
    }

    // If we've reached the end of either shape, we must have also reached
    // the end of the other shape for a valid transpose-like reshape.
    if (i == rank || j == rank) {
      ABSL_CHECK(  // CRASH_OK=invalid transpose-like reshape
          i == rank && j == rank)
          << "mismatched non-one dimensions for transpose-like reshape "
          << ShapeTransitionToString(shape_before, shape_after);
      return absl::OkStatus();
    }

    // The non-one dimensions must have the same size for a valid transpose-like
    // reshape.
    ABSL_CHECK(  // CRASH_OK=invalid transpose-like reshape
        shape_before[i] == shape_after[j])
        << "mismatched non-one dimensions for transpose-like reshape "
        << ShapeTransitionToString(shape_before, shape_after);

    // Transfer bounded dynamism from input to output.
    if (op_dims[i].boundOp.has_value()) {
      output_dims[j] = {.size = op_dims[i].size, .boundOp = op.getValue()};
    }
  }

  return absl::OkStatus();
}

}  // namespace

// Returns a string representation of the reassociation indices for debugging.
std::string ReassociationToString(const ReshapeReassociation& reassociation) {
  ReshapeType type = reassociation.type;
  std::string result = ReshapeTypeToString(type);
  if (reassociation.reassociation.empty()) {
    return result;
  }
  // Collapse{[2,2,4]->[4,4]}: [0,1,2] -> {0,1}, {2}
  // Expand{[4,4]->[2,2,4]}: [0,1] -> {0}, {0}, {1}
  return absl::StrCat(
      result,
      absl::StrJoin(
          reassociation.reassociation, ",",
          [](std::string* out, const mlir::ReassociationIndices& group) {
            absl::StrAppend(out, "{", absl::StrJoin(group, ","), "}");
          }));
}

// Determines the reshape type for the given input and output shapes.
// Basic function for now, but room to grow into other reshape types.
ReshapeType GetReshapeType(const Dimensions& static_shape_before,
                           const Dimensions& static_shape_after) {
  if (static_shape_before.size() == static_shape_after.size() &&
      GetCoreShape(static_shape_before) == GetCoreShape(static_shape_after)) {
    return ReshapeType::kTransposeLike;
  }
  if (static_shape_after.size() > static_shape_before.size()) {
    return ReshapeType::kExpand;
  }
  if (static_shape_after.size() == 1) {
    return ReshapeType::kFlatten;
  }
  return ReshapeType::kCollapse;
}

// Returns the reassociation indices for the given input and output shapes.
// See ReshapeFromStaticDimensions for more details on reassociation storage.
// See ReshapeType for more details on supported reshape types.
absl::StatusOr<ReshapeReassociation> GetReshapeReassociation(
    const Dimensions& static_shape_before,
    const Dimensions& static_shape_after) {
  ReshapeType reshape_type =
      GetReshapeType(static_shape_before, static_shape_after);
  if (reshape_type == ReshapeType::kFlatten ||
      reshape_type == ReshapeType::kTransposeLike) {
    // No need to reassociation for flattening or transpose-like reshapes.
    return ReshapeReassociation{reshape_type, {}};
  }

  // Below is a restriction of the MLIR getReassociationIndicesForReshape
  TT_RET_CHECK(static_shape_before.size() != static_shape_after.size(),
               error::kInvalidArgument)
      << "reshape reassociation not supported for same sized reshapes: "
      << ShapeTransitionToString(static_shape_before, static_shape_after);

  // Determine the reassociation indices from the static shapes.
  // This is used to determine the output dimension that needs to be bounded
  // and the bound value.
  auto reassociation = GetReassociationIndicesForReshape(static_shape_before,
                                                         static_shape_after);

  TT_RET_CHECK(reassociation.has_value(), error::kInvalidArgument)
      << "unable to determine reassociation indices "
      << ShapeTransitionToString(static_shape_before, static_shape_after);
  return ReshapeReassociation{reshape_type, std::move(reassociation.value())};
}

absl::StatusOr<mlir::MlirOp> ReshapeFromStaticDimensions(
    mlir::MlirOp op, const Dimensions& static_shape_before,
    const Dimensions& static_shape_after) {
  auto type = GetTensorTypeOrDie(op);

  ABSL_VLOG(3) << "[ReshapeFromStaticDimensions] op: "
               << mlir::debugString(type) << " "
               << ShapeTransitionToString(static_shape_before,
                                          static_shape_after);

  // If the input shape is static, we can directly call the stablehlo::Reshape
  // function.
  if (type.hasStaticShape()) {
    return mlir::stablehlo::Reshape(op, static_shape_after);
  }

  ABSL_CHECK(  // CRASH_OK=input op rank must match static shape
      static_shape_before.size() == type.getRank())
      << "input op and static shape before should have the same rank, got "
      << type.getRank() << " for op " << mlir::debugString(type) << " and "
      << static_shape_before.size()
      << " for static_shape_before: " << ToString(static_shape_before);

  // Determine the reassociation indices from the static shapes.
  // This is used to determine the output dimension that needs to be bounded
  // and the bound value.
  TT_ASSIGN_OR_RETURN(
      ReshapeReassociation reassociation,
      GetReshapeReassociation(static_shape_before, static_shape_after));

  // Special case for when reshape fully flattens the input
  if (reassociation.type == ReshapeType::kFlatten) {
    ABSL_VLOG(3) << "[ReshapeFromStaticDimensions] Flattening op: "
                 << mlir::debugString(type);
    return Flatten(op);
  }
  ABSL_VLOG(3) << "[ReshapeFromStaticDimensions] "
               << ReassociationToString(reassociation);

  mlir::stablehlo::Dimensions op_dims = GetDimensions(op);

  mlir::SmallVector<mlir::stablehlo::DimensionInfo> output_dims(
      static_shape_after.size());
  for (int64_t i = 0; i < static_shape_after.size(); ++i) {
    output_dims[i] = {.size = static_shape_after[i]};
  }
  if (reassociation.type == ReshapeType::kCollapse) {
    TT_RETURN_IF_ERROR(HandleCollapseReshape(op, reassociation, op_dims,
                                             static_shape_after, output_dims));
  } else if (reassociation.type == ReshapeType::kExpand) {
    TT_RETURN_IF_ERROR(HandleExpandReshape(op, reassociation, op_dims,
                                           static_shape_before,
                                           static_shape_after, output_dims));
  } else if (reassociation.type == ReshapeType::kTransposeLike) {
    TT_RETURN_IF_ERROR(HandleTransposeLikeReshape(
        op, op_dims, static_shape_before, static_shape_after, output_dims));
  }

  mlir::RankedTensorType dynamic_shape_after =
      mlir::stablehlo::getRankedTensorType(output_dims, type.getElementType());
  return mlir::stablehlo::Reshape(dynamic_shape_after, op);
}

}  // namespace torch_tpu
