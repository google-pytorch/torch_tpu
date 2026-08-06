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

#ifndef TORCH_TPU_OPS_OP_BUILDER_UTILS_H_
#define TORCH_TPU_OPS_OP_BUILDER_UTILS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for at::IntArrayRef
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/ops/python_context.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace internal {

// Extracts the dimensions from a Tensor or IntArrayRef.
[[nodiscard]] inline at::IntArrayRef GetDims(const at::Tensor& tensor) {
  return tensor.sizes();
}
[[nodiscard]] inline at::IntArrayRef GetDims(const at::Scalar& scalar) {
  return at::IntArrayRef{};
}
[[nodiscard]] inline at::IntArrayRef GetDims(const at::IntArrayRef dims) {
  return dims;
}

// Implements InferSize() for a variable number (>= 2) of arguments.

// The base case:
template <typename Dims1, typename Dims2>
[[nodiscard]] auto InferSizeImpl(const Dims1& a, const Dims2& b) {
  return at::infer_size(GetDims(a), GetDims(b));  // AT_INFER_SIZE_OK
}

// The recursive case:
template <typename Dims1, typename Dims2, typename Dims3, typename... Dims>
[[nodiscard]] auto InferSizeImpl(const Dims1& a, const Dims2& b, const Dims3& c,
                                 const Dims&... rest) {
  return InferSizeImpl(InferSizeImpl(a, b), GetDims(c), rest...);
}

}  // namespace internal

// Computes a mutually broadcastable shape for by applying Numpy's broadcasting
// rules to the input dimensions. See
// https://numpy.org/devdocs//user/basics.broadcasting.html#broadcastable-arrays.
//
// InferSize() takes a variable number (>= 2) of arguments. Each argument must
// be a container of int64_t values (it can be a view type like ArrayRef or
// absl::Span) or an at::Tensor.
template <typename... Dims>
absl::StatusOr<Dimensions> InferSize(const Dims&... dims) {
  try {
    return CopyIntVector(internal::InferSizeImpl(dims...));
  } catch (const ::c10::Error& e) {
    // at::infer_size() just threw e, whose what() always includes a C++
    // backtrace. We don't want that, as we are already adding the C++ backtrace
    // (if requested by the user) when we eventually turn this Status error into
    // a Python exception. Therefore we just use what_without_backtrace() from
    // e.
    return TT_ERROR(error::kInvalidArgument)
           << AdaptExternalErrorMessage(e.what_without_backtrace());
  }
}

// Sentinel value for the number of inputs/outputs when it is not known at
// compile time.
inline constexpr int kDynamicSize = -1;

namespace internal {

// Traits for defining MlirOpResults.

template <int kNumOutputs>
struct MlirOpResultsTraits {
  static_assert(kNumOutputs >= 1, "An op must produce at least one output.");
  using type = std::array<mlir::MlirOp, kNumOutputs>;
};

template <>
struct MlirOpResultsTraits<1> {
  // No need for an std::array if there's only one output.
  using type = mlir::MlirOp;
};

template <>
struct MlirOpResultsTraits<kDynamicSize> {
  using type = mlir::SmallVector<mlir::MlirOp>;
};

}  // namespace internal

// Result type of an MLIR op builder for the given number of outputs. If
// kNumOutputs is kDynamicSize, the type is DynamicMlirOpResults,
// which can hold an arbitrary number of mlir::MlirOps. If it's 1, the type is
// MlirOp. Otherwise, the result type is an std::array<mlir::MlirOp,
// kNumOutputs>.
template <int kNumOutputs>
using MlirOpResults = internal::MlirOpResultsTraits<kNumOutputs>::type;

// Result type of an MLIR op builder for an arbitrary number of outputs.
// Used for ops whose number of outputs is not known at C++ compile time.
using DynamicMlirOpResults = MlirOpResults<kDynamicSize>;

// Gets the tensor type of the input op. Crashes if the input is not a tensor.
mlir::RankedTensorType GetTensorTypeOrDie(const mlir::MlirOp& input);

// Gets the element type if ranked type, otherwise returns self.
mlir::Type GetElementTypeOrSelf(mlir::Type type);
mlir::Type GetElementTypeOrSelf(mlir::MlirOp type);

// Convert from absl::Span to mlir::ArrayRef for interop with MLIR builder
// methods.
template <typename T>
[[nodiscard]] mlir::ArrayRef<T> AsArrayRef(absl::Span<const T> span) {
  return {span.begin(), span.end()};
}

// DebugStringOptions for enabling/disabling debug info in the debug string.
// With DebugInfo enabled the MLIR will print with all available file-line-col
// location information, this may be verbose.
enum class DebugStringOptions {
  kEnableDebugInfo,
  kDisableDebugInfo,
};

// Returns the dimensions of the value with bounded information if present.
mlir::stablehlo::Dimensions GetDimensions(mlir::Value value);

// Returns the dimensions of the input op with bounded information if present.
mlir::stablehlo::Dimensions GetDimensions(mlir::MlirOp input);

// Returns the bytecode of the module.
absl::StatusOr<std::string> SerializeBytecode(mlir::ModuleOp module);

// Returns the portable artifact (serialized Versioned StableHLO) of the module.
absl::StatusOr<std::string> SerializePortableArtifact(mlir::ModuleOp module);

// Returns a debugstring of the current operation or module
[[nodiscard]] std::string DebugString(
    mlir::Operation* absl_nonnull op,
    DebugStringOptions opts = DebugStringOptions::kEnableDebugInfo);

[[nodiscard]] inline bool IsBooleanType(mlir::RankedTensorType tensor_type) {
  // Booleans are 1-bit integers.
  return tensor_type.getElementType().isInteger(1);
}

[[nodiscard]] inline bool IsUnsignedType(mlir::RankedTensorType tensor_type) {
  return tensor_type.getElementType().isUnsignedInteger();
}

[[nodiscard]] inline bool IsFloatType(mlir::RankedTensorType tensor_type) {
  return tensor_type.getElementType().isFloat();
}

[[nodiscard]] inline bool IsComplexType(mlir::RankedTensorType tensor_type) {
  auto complex_type =
      mlir::dyn_cast<mlir::ComplexType>(tensor_type.getElementType());
  return complex_type != nullptr;
}

// Returns the element type of the input op. Requires that the input op is a
// tensor.
absl::StatusOr<mlir::ElementType> GetElementType(mlir::MlirOp op);
mlir::ElementType GetElementTypeOrDie(mlir::MlirOp op);

// Shorthand for `ToElementType -> getElementType`.
absl::StatusOr<mlir::Type> GetMlirType(mlir::MLIRContext& ctx,
                                       mlir::ElementType element_type);

// Temporary helper to ensure that the min finite value matches the
// corresponding XLA type.
// TODO(bbahl): Replace this with GetMinFiniteValueAttr
absl::StatusOr<mlir::DenseElementsAttr> GetVerifiedMinFiniteValue(
    mlir::MlirBuilder& builder, mlir::RankedTensorType constant_tensor_type);

// Returns an attribute for the minimum finite value for a given type. This is
// used for initializing the minimum finite value for operations like argmax.
[[nodiscard]] mlir::Attribute GetMinFiniteValueAttr(mlir::Type element_type,
                                                    mlir::OpBuilder& builder);

// Returns an attribute for the maximum finite value for a given type. This is
// used for initializing the maximum finite value for operations like argmin.
[[nodiscard]] mlir::Attribute GetMaxFiniteValueAttr(mlir::Type element_type,
                                                    mlir::OpBuilder& builder);

// Returns the maximum representable finite value for a given floating-point
// type as a double. Asserts (ABSL_CHECK) that the type is a floating-point
// type.
[[nodiscard]] double GetMaxFiniteDouble(mlir::Type element_type);

// Returns the minimum representable finite value for a given floating-point
// type as a double. Asserts (ABSL_CHECK) that the type is a floating-point
// type.
[[nodiscard]] double GetMinFiniteDouble(mlir::Type element_type);

// Returns a tensor of shape `shape` with zero elements and the given element
// type. The `shape` must contain at least one dimension of size 0 to produce
// a tensor with 0 elements, otherwise an error is returned. If no shape is
// provided, a 1-dimensional tensor with 0 elements is returned. All dimensions
// in shape must be non-negative.
absl::StatusOr<mlir::MlirOp> MakeZeroSizedTensor(
    mlir::MlirBuilder& builder, mlir::Type element_type,
    mlir::ArrayRef<int64_t> shape = {0});

// Returns a tensor of shape `shape` with zero elements and the given element
// type. The `shape` must contain at least one dimension of size 0 to produce
// a tensor with 0 elements, otherwise an error is returned. If no shape is
// provided, a 1-dimensional tensor with 0 elements is returned. All dimensions
// in shape must be non-negative.
absl::StatusOr<mlir::MlirOp> MakeZeroSizedTensor(
    mlir::MlirBuilder& builder, mlir::ElementType element_type,
    mlir::ArrayRef<int64_t> shape = {0});

// PyTorch Specific constant builders.
// Consider making a separate `literal_utils.h` file if these get more complex.
// Be careful when using this function without specifying an mlir::ElementType,
// as at::Scalar reports its type in the highest precision necessary to hold
// their value category, which might not be what you want.
absl::StatusOr<mlir::MlirOp> MakeConstant(
    mlir::MlirBuilder& builder, const at::Scalar& value,
    c10::optional<mlir::ElementType> element_type_opt = std::nullopt,
    mlir::ArrayRef<int64_t> shape = {});
inline absl::StatusOr<mlir::MlirOp> MakeConstant(
    mlir::MlirBuilder& builder, const at::Scalar& value,
    mlir::ElementType element_type, mlir::ArrayRef<int64_t> shape = {}) {
  return MakeConstant(builder, value, c10::make_optional(element_type), shape);
}

// Creates a constant with the tensor shape and element type.
//
// Prefer `MakeScalarConstant` and `MakeConstantLike` over this method which
// fully support static and bounded programs.
template <typename T>
[[nodiscard]] mlir::MlirOp MakeConstant(mlir::MlirBuilder& builder, T value,
                                        mlir::RankedTensorType tensor_type) {
  return mlir::stablehlo::Constant(builder, makeConstant(value, tensor_type));
}

// Creates a constant with the given shape and element type.
//
// Prefer `MakeScalarConstant` and `MakeConstantLike` over this method which
// fully support static and bounded programs.
template <typename T>
[[nodiscard]] mlir::MlirOp MakeConstant(mlir::MlirBuilder& builder, T value,
                                        mlir::Type element_type,
                                        mlir::ArrayRef<int64_t> shape) {
  auto& ctx = builder.getContext();
  auto const_type = mlir::makeTensorType(ctx, shape, element_type);
  return mlir::stablehlo::Constant(builder,
                                   mlir::makeConstant(value, const_type));
}

// Creates a constant with the given shape and element type.
template <typename T>
[[deprecated(
    "Prefer MakeScalarConstant if shape is empty, or MakeConstantLike "
    "otherwise. These functions fully support static and bounded programs.")]]
[[nodiscard]] mlir::MlirOp
MakeConstant(mlir::MlirBuilder& builder, T value,
             mlir::ElementType element_type, mlir::ArrayRef<int64_t> shape) {
  auto& ctx = builder.getContext();
  auto const_type = mlir::makeTensorType(ctx, shape, element_type);
  return mlir::stablehlo::Constant(builder,
                                   mlir::makeConstant(value, const_type));
}

// Creates a constant with a scalar shape of the given element type.
//
// Prefer this method over `MakeConstant` since the intent more clear and
// scalars are supported in programs with bounded dynamism.
template <typename T>
[[nodiscard]] mlir::MlirOp MakeScalarConstant(mlir::MlirBuilder& builder,
                                              T value,
                                              mlir::Type element_type) {
  auto& ctx = builder.getContext();
  auto const_type = mlir::makeTensorType(ctx, {}, element_type);
  return mlir::stablehlo::Constant(builder,
                                   mlir::makeConstant(value, const_type));
}

template <typename T>
[[nodiscard]] mlir::MlirOp MakeScalarConstant(mlir::MlirBuilder& builder,
                                              T value,
                                              mlir::ElementType element_type) {
  auto& ctx = builder.getContext();
  auto const_type = mlir::makeTensorType(ctx, {}, element_type);
  return mlir::stablehlo::Constant(builder,
                                   mlir::makeConstant(value, const_type));
}

// Creates a constant with the same shape as the input tensor.
//
// If the input tensor has a static shape, the constant is created with the
// same shape.
//
// If the input is bounded dynamic, the input value is created with the padded
// shape of the input tensor then sliced to match the bounded dynamic runtime
// shape of the input tensor.
template <typename T>
[[nodiscard]] mlir::MlirOp MakeConstantLike(
    mlir::MlirOp input, T val, std::optional<mlir::Type> element_type) {
  mlir::MlirBuilder& builder = input.getBuilder();
  auto input_type = GetTensorTypeOrDie(input);
  if (element_type.has_value()) {
    input_type = input_type.clone(GetElementTypeOrSelf(element_type.value()));
  }
  if (input_type.hasStaticShape()) {
    return MakeConstant(builder, val, input_type);
  }

  // Handle bounded dynamism - make scalar constant and use broadcast APIs
  // Make scalar type with no bounds, avoid `clone` since that preserves the
  // bounds, which are invalid since this uses a scalar shape.
  auto scalar_type = mlir::makeTensorType(builder.getContext(), {},
                                          input_type.getElementType());
  auto scalar_elements_attr = mlir::makeConstant(val, scalar_type);
  return mlir::chlo::ConstantLike(input, scalar_elements_attr);
}

template <typename T>
[[nodiscard]] mlir::MlirOp MakeConstantLike(
    mlir::MlirOp input, T val,
    std::optional<mlir::ElementType> element_type = std::nullopt) {
  if (element_type.has_value()) {
    mlir::Type element_type_mlir =
        mlir::getElementType(input.getContext(), element_type.value());
    return MakeConstantLike(input, val, element_type_mlir);
  } else {
    return MakeConstantLike(input, val, std::optional<mlir::Type>());
  }
}

// Returns a constant with the same shape as the input tensor.
//
// If the input tensor has a static shape, the constant is created with the
// same shape.
//
// If the input is bounded dynamic, creates a chlo.constant_like op which
// handles proper padding and slicing to match the bounded dynamic runtime
// shape of the input tensor.
absl::StatusOr<mlir::MlirOp> MakeConstantLike(mlir::MlirOp input,
                                              const at::Scalar& value);

// Returns an mlir::MlirOp representing the number of elements in the input
// tensor with the given element type, only counting the dimensions specified.
// If dims is empty, all dimensions are counted.
//
// If static returns a constant.
// If bounded dynamic, creates a computation to calculate the num elements:
//   i.e. `stablehlo.mul(get_dim_size, cst_dim_size))`
//
mlir::MlirOp GetNumElements(mlir::MlirOp input, mlir::Type element_type,
                            mlir::ArrayRef<int64_t> dims = {});

// Returns the current ModuleOp a builder is operating in.
mlir::ModuleOp GetModuleOp(mlir::MlirBuilder& builder);

// Returns the common shape these ops would broadcast to, or an error if the
// ops are not broadcastable.
absl::StatusOr<Dimensions> GetBroadcastShape(
    absl::Span<const mlir::MlirOp> ops);

absl::StatusOr<mlir::MlirOp> BroadcastIfNeeded(mlir::MlirOp input,
                                               absl::Span<const int64_t> shape);

// Returns a broadcasted version of the input tensor to the target shape.
//
// If the target shape is bounded, input is broadcasted to the padded shape and
// also made bounded.
// If the target shape is unbounded, input is broadcasted to the target shape
// without any bounds.
//
// If broadcast_dimensions is non-empty, broadcasting maps input dimensions
// accordingly instead of NumPy-style broadcasting.
absl::StatusOr<mlir::MlirOp> BroadcastIfNeeded(
    mlir::MlirOp input, mlir::stablehlo::Dimensions shape,
    absl::Span<const int64_t> broadcast_dimensions = {});

// Broadcasts input to target shape, returning an error if the shapes are
// incompatible.
//
// If broadcast_dimensions is non-empty, broadcasting maps input dimensions
// accordingly instead of NumPy-style broadcasting.
absl::StatusOr<mlir::MlirOp> BroadcastIfNeeded(
    mlir::MlirOp input, mlir::MlirOp target,
    absl::Span<const int64_t> broadcast_dimensions = {});

// Unsqueezes the input tensor by inserting a dimension of size 1 at the
// given dimension.
absl::StatusOr<mlir::MlirOp> Unsqueeze(mlir::MlirOp input, int64_t dim);
// Squeezes the input tensor by removing dimensions of size 1.
absl::StatusOr<mlir::MlirOp> Squeeze(mlir::MlirOp input,
                                     absl::Span<const int64_t> dims);

// Flattens the input tensor to a single dimension.
// Supports bounded dynamic flatten.
mlir::MlirOp Flatten(mlir::MlirOp input);

// Applies Numpy's broadcasting rules, if needed. See
// https://numpy.org/devdocs//user/basics.broadcasting.html#broadcastable-arrays

// This overload takes a fixed number of inputs as an std::array, and returns
// the same number of outputs.
template <int kNumInputs>
absl::StatusOr<std::array<mlir::MlirOp, kNumInputs>> ApplyBroadcastIfNeeded(
    std::array<mlir::MlirOp, kNumInputs> ops) {
  static_assert(kNumInputs > 0, "ApplyBroadcastIfNeeded requires >= 1 input");
  if (ABSL_VLOG_IS_ON(3)) {
    ABSL_VLOG(3) << "[ApplyBroadcastIfNeeded] Broadcasting ops:" << '\n';
    for (const mlir::MlirOp& op : ops) {
      ABSL_VLOG(3) << "  " << mlir::debugString(op.getType()) << " from op "
                   << op.ToString() << '\n';
    }
  }

  mlir::SmallVector<mlir::Value, kNumInputs> values = llvm::map_to_vector(
      ops, [](const mlir::MlirOp& op) { return op.getValue(); });
  mlir::BaseScopedDiagnosticHandler diag_handler(values[0].getContext());
  auto broadcasted_values_or_fail = mlir::stablehlo::numpyBroadcastIfNeeded(
      ops[0].getBuilder().getOpBuilder(), values);
  TT_RET_CHECK(mlir::succeeded(broadcasted_values_or_fail),
               error::kInvalidArgument)
      << "failed to broadcast tensors: "
      << diag_handler.ConsumeStatus().message();
  mlir::SmallVector<mlir::Value, kNumInputs> broadcasted_values =
      std::move(*broadcasted_values_or_fail);
  for (int i = 0; i < kNumInputs; ++i) {
    ops[i] = mlir::MlirOp(ops[i].getBuilder(), broadcasted_values[i]);
  }
  return std::move(ops);
}

// This overload takes a variable number of inputs, where each input is a
// mlir::MlirOp.
template <typename... Ops>
absl::StatusOr<std::array<mlir::MlirOp, sizeof...(Ops)>> ApplyBroadcastIfNeeded(
    Ops... ops) {
  return ApplyBroadcastIfNeeded<sizeof...(Ops)>(
      std::array<mlir::MlirOp, sizeof...(Ops)>{ops...});
}

absl::StatusOr<mlir::MlirOp> Broadcast(mlir::MlirOp input,
                                       absl::Span<const int64_t> output_dims,
                                       absl::Span<const int64_t> bcast_dims);

// Conditionally converts the element types of two input tensors to the
// provided element type.
//
// Checks for each input is an integer or boolean type. If so, converts it to
// the default PyTorch dtype. This function is typically used to handle
// operations like division, where integer inputs are promoted to floating-point
// in PyTorch to produce floating-point results.
absl::StatusOr<std::pair<mlir::MlirOp, mlir::MlirOp>> ConvertIfIntegers(
    mlir::MlirOp op1, mlir::MlirOp op2, mlir::ElementType target_dtype);

// Conditionally converts the element type of an input tensor to the
// provided element type.
//
// Conversion only happens if the input tensor has an integer or boolean
// element type. If the input tensor is already a floating-point type,
// no conversion is performed.
absl::StatusOr<mlir::MlirOp> ConvertIfInteger(mlir::MlirOp op,
                                              mlir::ElementType target_dtype);

// Aliases for op builders with varying numbers of inputs.

// The most general op builder, with zero or more inputs and one or more
// results.
using MlirOpBuilder = absl::AnyInvocable<absl::StatusOr<DynamicMlirOpResults>(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) const>;

namespace internal {

// Traits for defining NAryMlirOpBuilder.

// The primary template is for the case of kArity >= 2.
template <int kArity, int kNumOutputs>
struct NAryMlirOpBuilderTraits {
  using type = absl::AnyInvocable<absl::StatusOr<MlirOpResults<kNumOutputs>>(
      // No need for an MlirBuilder parameter, as any input MlirOp can provide
      // one.
      FixedSizeSpan<mlir::MlirOp, kArity> inputs) const>;
};

template <int kNumOutputs>
struct NAryMlirOpBuilderTraits<0, kNumOutputs> {
  using type = absl::AnyInvocable<absl::StatusOr<MlirOpResults<kNumOutputs>>(
      // Need a builder parameter as there's no input MlirOp to get one from.
      mlir::MlirBuilder& builder) const>;
};

template <int kNumOutputs>
struct NAryMlirOpBuilderTraits<1, kNumOutputs> {
  using type = absl::AnyInvocable<absl::StatusOr<MlirOpResults<kNumOutputs>>(
      // No need for an MlirBuilder parameter, as the input MlirOp can provide
      // one.
      //
      // Since there's only one input, it's more convenient to pass it as a
      // plain MlirOp instead of a FixedSizeSpan<const MlirOp, 1>.
      mlir::MlirOp input) const>;
};

template <int kNumOutputs>
struct NAryMlirOpBuilderTraits<kDynamicSize, kNumOutputs> {
  using type = absl::AnyInvocable<absl::StatusOr<MlirOpResults<kNumOutputs>>(
      absl::Span<mlir::MlirOp> inputs,
      // Need a builder parameter in case the inputs are empty.
      mlir::MlirBuilder& builder) const>;
};

}  // namespace internal

// An op builder whose number of inputs (the arity) is known at compile time.
// kNumOutputs can be specified if known at compile time, which will enable
// compile-time checking of the number of outputs. If the number of outputs is
// not known at compile time, use kDynamicSize for the kNumOutputs
// template parameter.
template <int kArity, int kNumOutputs = 1>
using NAryMlirOpBuilder =
    internal::NAryMlirOpBuilderTraits<kArity, kNumOutputs>::type;

// Converts an NAryMlirOpBuilder to an MlirOpBuilder, checking that the number
// of inputs is correct at runtime.
template <int kArity, int kNumOutputs>
[[nodiscard]] MlirOpBuilder ToMlirOpBuilder(
    NAryMlirOpBuilder<kArity, kNumOutputs> op_builder) {
  return [op_builder = std::move(op_builder)](mlir::MlirBuilder& builder,
                                              absl::Span<mlir::MlirOp> inputs)
             -> absl::StatusOr<DynamicMlirOpResults> {
    std::optional<MlirOpResults<kNumOutputs>> outputs;
    if constexpr (kArity == kDynamicSize) {
      TT_ASSIGN_OR_RETURN(outputs, op_builder(inputs, builder));
    } else {
      TT_RET_CHECK(inputs.size() == kArity, error::kInternal)
          << "expected " << kArity << " inputs, got " << inputs.size();
      if constexpr (kArity == 0) {
        TT_ASSIGN_OR_RETURN(outputs, op_builder(builder));
      } else if constexpr (kArity == 1) {
        TT_ASSIGN_OR_RETURN(outputs, op_builder(inputs[0]));
      } else {
        TT_ASSIGN_OR_RETURN(
            outputs, op_builder(FixedSizeSpan<mlir::MlirOp, kArity>(inputs)));
      }
    }
    // The type of op_builder guarantees that outputs has the correct number of
    // elements. The following check is just for catching bugs in the torch_tpu
    // implementation.
    if constexpr (kNumOutputs == 1) {
      return {{*outputs}};  // `outputs` is a single MlirOp.
    } else {
      if constexpr (kNumOutputs != kDynamicSize) {
        ABSL_CHECK_EQ(outputs->size(), kNumOutputs)  // CRASH_OK
            << "expected " << kNumOutputs << " outputs, got "
            << outputs->size();
      }
      return DynamicMlirOpResults(outputs->begin(), outputs->end());
    }
  };
}

// An op builder with zero inputs and one result, for example torch.ones().
using MlirNullaryOpBuilder = absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(
    mlir::MlirBuilder& builder) const>;

// An op builder with one input and one result, for example torch.abs().
using MlirUnaryOpBuilder =
    absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(mlir::MlirOp input) const>;

// An op builder with two inputs and one result, for example torch.add().
using MlirBinaryOpBuilder = absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(
    mlir::MlirOp input1, mlir::MlirOp input2) const>;

// An op builder with three inputs and one result.
using MlirTernaryOpBuilder = absl::AnyInvocable<absl::StatusOr<mlir::MlirOp>(
    mlir::MlirOp, mlir::MlirOp, mlir::MlirOp) const>;

// Builder function for an MLIR-based computation. This is an MlirOpBuilder
// which has been wrapped into a module with a @main function.
using MlirComputationBuilder =
    absl::AnyInvocable<absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>(
        mlir::MLIRContext& mlir_context) const>;

// Converts an op that returns a tuple of outputs into a vector of outputs.
template <size_t kNumOps>
absl::StatusOr<DynamicMlirOpResults> ToResultVector(
    absl::StatusOr<std::array<mlir::MlirOp, kNumOps>> results) {
  TT_ASSIGN_OR_RETURN(auto results_value, results);
  DynamicMlirOpResults results_vec{results_value.begin(), results_value.end()};
  return results_vec;
}

// Converts a single-output op return value into a vector of outputs,
// or passes an input error unmodified.
absl::StatusOr<DynamicMlirOpResults> ToResultVector(
    absl::StatusOr<mlir::MlirOp> results);

// Builds a module name from the current Python stack frame.
// In most cases this frame is the one that triggered a materialization.
//
// The module name is of the form "tt_jit_file_L#C#_func_opname" where `file` is
// the filename without extension or path, `L#C#` is the line and column number
// of the current stack frame (the one that triggered the materialization in
// most cases) i.e. L2C4, `func` is the function name of the current stack
// frame, and `opname` is the name of the torch_tpu operation that triggered
// compiled.
//
// Module names are preserved throughout compilation which will help with
// identifying the source of a compiled module in debugging and dumping
// workflows. Note that if `torch_tpu_internal_mlir_tracebacks` is disabled,
// the file-line-column-function info will be elided and the module name will
// just be the op name that triggered the materialization, resulting with a name
// of `tt_jit` with an op_call_chain if available.
std::string BuildModuleNameFromPyContext(
    mlir::MLIRContext& mlir_context,
    const PythonContext* absl_nullable python_context);

// Given a module name with a `main` function, annotates the donated inputs,
// adds buffer donation annotations to the main function.
void AnnotateBufferDonations(mlir::ModuleOp module,
                             mlir::ArrayRef<int64_t> donated_inputs);

// Casts a given op to the expected type if necessary, preserving the values.
// Returns an error if PyTorch doesn't support casting the PyTorch type
// corresponding to the op's MLIR type to the PyTorch type corresponding to
// `expected_output_type`.
absl::StatusOr<mlir::MlirOp> CastIfNeeded(
    mlir::MlirOp op, mlir::ElementType expected_output_type);

// Generates the body of a reduce operation for a given reduce type.
absl::Status BuildReduceBody(mlir::RegionBuilder& rb, mlir::Type element_type,
                             c10d::ReduceOp::RedOpType reduce_op_type);

// Creates a fill value used to represent "uninitialized memory".
// The operations for torch.empty() and torch.resize_() are expected to create
// memory that has uninitialized values. There is no way to represent
// uninitialized value in MLIR; instead, we follow the convention of
// torch.utils.deterministic.fill_uninitialized_memory, which is to use
// std::numeric_limits<scalar_t>::quiet_NaN() for float types and
// std::numeric_limits<scalar_t>::max() for integers and boolean types.
mlir::MlirOp BuildFillUninitialized(mlir::MlirBuilder& builder,
                                    mlir::ElementType tensor_element_type,
                                    absl::Span<const int64_t> shape = {});

// Slice if left_pad or right_pad is negative to truncate the tensor in the
// dimension specified, otherwise return op.
absl::StatusOr<mlir::MlirOp> BuildMaybeSlice(mlir::MlirOp op, int64_t dimension,
                                             int64_t left_pad,
                                             int64_t right_pad);

// Returns a vector containing all dimensions of the given MlirOp.
Dimensions GetAllDimensions(mlir::MlirOp op);

// Converts the data type of given float op to the given type if it has a lower
// precision. If given op is not a float op, it is returned unmodified.
absl::StatusOr<mlir::MlirOp> PromoteFloatDtype(
    mlir::MlirOp op, mlir::ElementType min_precision = mlir::ElementType::F32);

// Builds an MLIR op that updates the RNG state (Philox offset) based on the
// number of elements generated and their bit width.
// This is used for decoupling the state update from the actual random number
// generation, making the state update resilient to failures in the generation.
absl::StatusOr<mlir::MlirOp> BuildRngStateUpdateShlo(mlir::MlirOp state,
                                                     int64_t num_elements,
                                                     int64_t bit_width);

////
// Reshape Utilities

// The type of reshape operation, currently only handling reshapes that can be
// represented abstractly, either as a collapse, expand or transpose-like
// reshape.
//
// Currently only affine reshapes are supported, meaning that the input and
// output shapes must be contiguous:
//  - Collapse: [AxBxC] -> [A*BxC]
//  - Flatten: [AxBxC] -> [A*B*C] (special case of collapse, full collapse)
//  - Expand:   [A*B] -> [AxB]
//  - TransposeLike: [Ax1xBx1] -> [1xAx1xB] (only 1s can move around)
//
// This list may expand as support is added for other reshape types, such as:
//   - [AxB] -> [AxB] no-op
//   - [AxB] -> [AxBx1] unsqueeze
//   - [AxBx1] -> [AxB] squeeze
//   - [Ax1] -> [1xA] transpose
//   - [AxB] -> [BxA] dim-flipping
//   - [AxBxC] -> [BxA*C] non-affine collapses
enum class ReshapeType {
  kCollapse,
  kFlatten,
  kExpand,
  kTransposeLike,
  kUnknown,
};

// Helper struct to hold reshape reassociation information.
// See ReshapeFromStaticDimensions for more details on reassociation storage.
// See ReshapeType for more details on supported reshape types.
//
// Examples:
// - Squeeze{[4,1]->[4]}: [0,1]->{0}
// - Unsqueeze{[4,2]->[4,1,2]}: [0,1]->{0}{-1}{1}
// - Flatten{[4,4]->[16]}: []->{} // empty map, no reassociation needed.
// - Collapse{[2,2,4]->[4,4]}: [0,1,2]->{0,1},{2}
// - Expand{[4,4]->[2,2,4]}: [0,1]->{0},{0},{1}

struct ReshapeReassociation {
  ReshapeType type;
  llvm::SmallVector<mlir::ReassociationIndices> reassociation;
};

// String representation of the reshape reassociation.
std::string ReassociationToString(const ReshapeReassociation& reassociation);

// Returns the reassociation indices for the given input and output shapes.
// See ReshapeFromStaticDimensions for more details on reassociation storage.
// See ReshapeType for more details on supported reshape types.
absl::StatusOr<ReshapeReassociation> GetReshapeReassociation(
    const Dimensions& static_shape_before,
    const Dimensions& static_shape_after);

// Reshapes the input op to the output shape.
//
// For static input op:
//  - Reshape falls back to stablehlo::Reshape and `static_shape_before` is
//    ignored.
//
// For dynamic input op:
//  - Reshape first determines association between static input shape and output
//    shape and based on that propagates the bounds of dynamic dimension to the
//    output op using stablehlo::Reshape.
// - A valid reshape is performed iff:
//   - There must be an affine reassociation (contiguous collapse_shape or
//     expand_shape) between the input and output static shapes. For example:
//       - [a, b, c] -> [a*b, c] is valid (collapse_shape)
//       - [a, b*c] -> [a, b, c] is valid (expand_shape)
//       - [a, b, c] -> [c, a*b] is not.
//   - The input tensor must have a single dynamic dimension.
//   - The dynamic dimension must not expand to multiple dimensions after the
//     reshape. For example, when dimension 0 is dynamic:
//       - [a*b, c] -> [a, b, c] is invalid
//   - Arbitrary squeezes and unsqueezes are allowed.
//       - [1, a, b, 1] -> [a, 1, 1, b] is valid.
//
absl::StatusOr<mlir::MlirOp> ReshapeFromStaticDimensions(
    mlir::MlirOp op, const Dimensions& static_shape_before,
    const Dimensions& static_shape_after);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_OP_BUILDER_UTILS_H_
