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

#ifndef TORCH_TPU_CSRC_EAGER_OP_DISPATCHER_H_
#define TORCH_TPU_CSRC_EAGER_OP_DISPATCHER_H_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/op_name_stack.h"
#include "csrc/common/shape.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// The following functions can be used to dispatch ATEN ops for
// execution. Whether an op's execution happens right away (immediate execution)
// or it is deferred to later (deferred execution) is an implementation detail
// and is not controlled through this API.
//
// When dispatching an op the caller needs to provide (among other things) a
// unique `op_name` and an optional `op_param_cache_keys`. These parameters are
// used internally to compute a unique cache key for the op so that
// recompilation can be avoided for future op executions. Consider that a simple
// ATEN op that doesn't have parameters such as abs or relu doesn't need
// op_param_cache_keys; `op_name` is sufficient. However, a parametrized ATEN op
// such as gelu, needs an appropriately populated op_param_cache_keys map so as
// to distinguish gelu ops compiled for different values of their parameter.
//
// Example of non-parameterized ATEN op:
//
//   auto op_builder = [](mlir::MlirOp input_op) {
//     // return a SHLO graph corresponding to relu.
//     return ...
//   };
//   TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
//                      ConvertTo<mlir::ElementType>(
//                          input_tensor.scalar_type()));
//   return DispatchOp<1>(OpName::kRelu, std::move(op_builder), input_tensor,
//                        {.out_dtype = output_dtype,
//                         .out_dims = input_tensor.sizes());
//
// Example of a parameterized ATEN op:
//
//   TT_ASSIGN_OR_THROW(auto param_keys,
//                      *OpParamCacheKeysBuilder().SetParam("aprx",
//                      approximate));
//   auto op_builder =
//             [approximate = std::string(approximate)](mlir::MlirOp input_op) {
//     // return a SHLO graph corresponding to gelu with the provided
//     // `approximate` parameter.
//     return ...
//   };
//   TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
//                      ConvertTo<mlir::ElementType>(
//                          input_tensor.scalar_type()));
//   return DispatchOp<1>(OpName::kGelu, std::move(op_builder), input_tensor,
//                        {.out_dtype = output_dtype,
//                         .out_dims = input_tensor.sizes(),
//                         .op_param_cache_keys =
//                             std::move(param_keys)});
//

// A transient struct for passing options to the DispatchOp function. It has
// fields referencing potentially temporary objects, and thus should not be
// stored for later use.
//
// Fields typed `const T&` simulate required arguments - one cannot create an
// instance of this struct without specifying these fields.

// The primary template is for the case where the number of outputs is known at
// compile time and not equal to 1.
template <int kNumOutputs>
struct DispatchOpOptions {
  static_assert(kNumOutputs >= 2);

  // The op name for dispatching. If omitted, use the op name from the active
  // TT_KERNEL() context.
  std::optional<OpName> op_name = std::nullopt;
  const FixedSizeSpan<const mlir::ElementType, kNumOutputs>& out_dtypes;
  const FixedSizeSpan<const absl::Span<const int64_t>, kNumOutputs>&
      out_dims_list;
  // If specified, cast all op inputs to the given dtype before the provided
  // `op_builder` is applied. This is useful to simplify certain ops that need
  // to perform a computation on a dtype that is uplifted from the dtypes of all
  // the inputs (e.g., clamp and softmax).
  std::optional<mlir::ElementType> computation_dtype;
  OpParamCacheKeys op_param_cache_keys;
  OpSplitMode split_mode = OpSplitMode::kNone;
  Indices donated_indices = {};
};

// Specialization for the case where the number of outputs is unknown at compile
// time.
template <>
struct DispatchOpOptions<kDynamicSize> {
  // The op name for dispatching. If omitted, use the op name from the active
  // TT_KERNEL() context.
  std::optional<OpName> op_name = std::nullopt;
  const absl::Span<const mlir::ElementType>& out_dtypes;
  const absl::Span<const absl::Span<const int64_t>>& out_dims_list;
  std::optional<mlir::ElementType> computation_dtype;
  OpParamCacheKeys op_param_cache_keys;
  OpSplitMode split_mode = OpSplitMode::kNone;
  Indices donated_indices = {};
};

// Specialization for the case where the number of outputs is 1.
template <>
struct DispatchOpOptions<1> {
  // The op name for dispatching. If omitted, use the op name from the active
  // TT_KERNEL() context.
  std::optional<OpName> op_name = std::nullopt;
  const mlir::ElementType& out_dtype;
  const absl::Span<const int64_t>& out_dims;
  std::optional<mlir::ElementType> computation_dtype;
  OpParamCacheKeys op_param_cache_keys;
  OpSplitMode split_mode = OpSplitMode::kNone;
  Indices donated_indices = {};
};

namespace internal {

// Traits for defining DeviceBufferRefArray.

template <int kSize>
struct DeviceBufferRefArrayTraits {
  static_assert(kSize >= 2, "An op must produce at least one output.");
  using type = std::array<DeviceBufferRef, kSize>;
};

template <>
struct DeviceBufferRefArrayTraits<1> {
  // No need for an std::array or std::unique_ptr if there's only one output.
  using type = DeviceBufferRef;
};

template <>
struct DeviceBufferRefArrayTraits<kDynamicSize> {
  using type = std::vector<DeviceBufferRef>;
};

}  // namespace internal

// Holds an at::Tensor. This class is deliberately not default constructible.
// This allows us to detect at compile time if we pass fewer than necessary
// tensors to a DispatchOp<k> function. For example,
//
//   // Doesn't compile: expected 3 tensors, got 2.
//   std::array<TensorHolder, 3> tensors = {tensor1, tensor2};
//   // Doesn't compile: expected 3 tensors, got 4.
//   std::array<TensorHolder, 3> tensors = {tensor1, tensor2, tensor3, tensor4};
//
//   // Compiles: expected 3 tensors, got 2 from the list and 1 from
//   // default construction.
//   std::array<at::Tensor, 3> tensors = {tensor1, tensor2};
//   // Doesn't compile: expected 3 tensors, got 4.
//   std::array<at::Tensor, 3> tensors = {tensor1, tensor2, tensor3, tensor4};
class TensorHolder : public at::Tensor {
 public:
  // No default constructor. Must be initialized with a Tensor.
  TensorHolder() = delete;

  // Constructs a TensorHolder that holds the given tensor.
  TensorHolder(  // NOLINT - implicit conversion intended
      const at::Tensor& tensor)
      : at::Tensor(tensor) {}
};

static_assert(sizeof(TensorHolder) == sizeof(at::Tensor),
              "TensorHolder shouldn't have its own data members, as we need "
              "to convert a TensorHolder span to a Tensor span.");

// If kSize == kDynamicSize, the type is std::vector<DeviceBufferRef>;
// otherwise, it's std::array<DeviceBufferRef, kSize>. The latter type offers
// more type safety.
template <int kSize>
using DeviceBufferRefArray = internal::DeviceBufferRefArrayTraits<kSize>::type;

namespace internal {

// Information about an injected op dispatch failure. Used for internal testing
// only.
struct OpDispatchFailure {
  // The base name of the op that is forced to fail. If empty, no op is forced
  // to fail.
  std::string op_base_name;
  // The failure message of the injected op failure.
  std::string failure_message;
};

// Returns the injected op dispatch failure.
[[nodiscard]] OpDispatchFailure& GetOpDispatchFailure();

// Forces the op with the given base name to fail with the given message. This
// is NOT accumulative. If you call this multiple times, only the last call will
// take effect. Used for testing only.
void SetOpDispatchFailure(std::string op_base_name,
                          std::string failure_message);

template <int kSize>
struct TensorGroupTraits {
  using type = std::array<TensorHolder, kSize>;
};

template <>
struct TensorGroupTraits<1> {
  using type = at::Tensor;
};

template <>
struct TensorGroupTraits<kDynamicSize> {
  using type = std::vector<at::Tensor>;
};

template <int kArity>
using OpInputsTraits = TensorGroupTraits<kArity>;

template <int kNumOutputs>
using OpOutputsTraits = TensorGroupTraits<kNumOutputs>;

}  // namespace internal

template <int kArity>
using OpInputs = internal::OpInputsTraits<kArity>::type;

template <int kNumOutputs>
using OpOutputs = internal::OpOutputsTraits<kNumOutputs>::type;

namespace internal {

template <int kSize>
absl::Span<const at::Tensor> GetTensorSpan(const OpInputs<kSize>& tensors) {
  if constexpr (kSize == kDynamicSize) {
    return tensors;
  } else if constexpr (kSize == 1) {
    return absl::MakeSpan(&tensors, 1);
  } else {
    ABSL_CHECK_EQ(tensors.size(), kSize)  // CRASH_OK
        << "expected " << kSize << " tensors, got " << tensors.size();
    return absl::Span<const at::Tensor>(tensors.data(), tensors.size());
  }
}

// Resizes `output` (if shape differs) and assigns `result_buffer` to it.
absl::Status AssignBufferToOutput(const at::Tensor& output,
                                  DeviceBufferRef result_buffer);

// Resizes each defined tensor in `outputs` (if shape differs) and assigns the
// corresponding result buffer to it.
absl::Status AssignBuffersToOutputs(absl::Span<const at::Tensor> outputs,
                                    absl::Span<DeviceBufferRef> result_buffers);

// Automatically detects and populates buffer donations from `inputs` into
// `outputs` in eager mode if `donated_indices` is empty.
//
// IMPORTANT: A device buffer CANNOT be donated multiple times. In XLA, each
// input buffer can be donated to at most ONE output buffer.
void AutoDonateInPlaceBuffers(
    absl::Span<const at::Tensor> outputs,
    absl::Span<const mlir::ElementType> out_dtypes,
    absl::Span<const absl::Span<const int64_t>> out_dims_list,
    absl::Span<const at::Tensor> inputs, Indices& donated_indices);

}  // namespace internal

// Dispatches an op with the given number of inputs (kArity) and given number of
// outputs (kNumOutputs). This function enforces the number inputs or outputs at
// compile time.
//
// If the number of inputs is unknown at compile time, use kDynamicSize for
// kArity. If the number of outputs is unknown at compile time, use kDynamicSize
// for kNumOutputs.
template <int kArity, int kNumOutputs = 1>
absl::StatusOr<DeviceBufferRefArray<kNumOutputs>> DispatchOp(
    NAryMlirOpBuilder<kArity, kNumOutputs> op_builder,
    const OpInputs<kArity>& inputs, DispatchOpOptions<kNumOutputs> options) {
  // Check for sticky errors.
  if (ABSL_PREDICT_FALSE(HasStickyError())) {
    return GetStickyError();
  }

  // Get the op name from the active TT_KERNEL() context.
  const std::optional<OpName> maybe_ctx_op_name =
      internal::OpNameStack::MaybeTop();
  ABSL_CHECK(maybe_ctx_op_name.has_value())  // CRASH_OK
      << "DispatchOp() called without an active TT_KERNEL() context. "
         "Move the call inside a TT_KERNEL().";
  const OpName ctx_op_name = maybe_ctx_op_name.value();

  // If the op name is provided in the options, check that the override is
  // necessary (i.e. it doesn't match the op name from the active TT_KERNEL()
  // context).
  if (options.op_name.has_value()) {
    ABSL_CHECK_NE(*options.op_name, ctx_op_name)  // CRASH_OK
        << "The op name provided in the DispatchOp() options matches "
           "the one from the active TT_KERNEL() context. Please remove the "
           ".op_name override from the options as it's redundant.";
  }
  const auto op_name = options.op_name.value_or(ctx_op_name);

  const absl::Span<const at::Tensor> inputs_span =
      internal::GetTensorSpan<kArity>(inputs);

  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> inputs_vec,
                      GetBuffers(inputs_span));

  // Fail if the op is forced to fail for testing.
  const std::string_view op_base_name = ToBaseName(op_name);
  const auto& op_failure = internal::GetOpDispatchFailure();
  if (op_failure.op_base_name == op_base_name) {
    return TT_ERROR(error::kInternal) << op_failure.failure_message;
  }

  std::vector<DeviceBufferRef> results;
  if constexpr (kNumOutputs == 1) {
    internal::DeferredOpParams params{
        .op_name = op_name,
        .op_builder =
            ToMlirOpBuilder<kArity, kNumOutputs>(std::move(op_builder)),
        .op_param_cache_keys = std::move(options.op_param_cache_keys),
        .inputs = std::move(inputs_vec),
        .output_shapes = {Shape(CopyIntVector(options.out_dims),
                                options.out_dtype)},
        .computation_dtype = options.computation_dtype,
        .split_mode = options.split_mode,
        .donated_indices = std::move(options.donated_indices),
    };
    TT_ASSIGN_OR_RETURN(
        results, internal::CreateDeferredDeviceBufferList(std::move(params)));
  } else if constexpr (kNumOutputs == kDynamicSize) {
    const auto& out_dtypes = options.out_dtypes;
    const auto& out_dims_list = options.out_dims_list;
    const int num_outputs = out_dtypes.size();
    ABSL_CHECK_EQ(num_outputs,  // CRASH_OK=indicates invalid kernel
                  out_dims_list.size())
        << "Mismatching output sizes, num_outputs = " << num_outputs
        << ", out_dims_list.size() = " << out_dims_list.size();
    std::vector<Shape> output_shapes;
    output_shapes.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      output_shapes.emplace_back(CopyIntVector(out_dims_list[i]),
                                 out_dtypes[i]);
    }

    internal::DeferredOpParams params{
        .op_name = op_name,
        .op_builder =
            ToMlirOpBuilder<kArity, kNumOutputs>(std::move(op_builder)),
        .op_param_cache_keys = std::move(options.op_param_cache_keys),
        .inputs = std::move(inputs_vec),
        .output_shapes = std::move(output_shapes),
        .computation_dtype = options.computation_dtype,
        .split_mode = options.split_mode,
        .donated_indices = std::move(options.donated_indices),
    };
    TT_ASSIGN_OR_RETURN(
        results, internal::CreateDeferredDeviceBufferList(std::move(params)));
  } else {
    const auto& out_dtypes = options.out_dtypes;
    const auto& out_dims_list = options.out_dims_list;
    const int num_outputs = out_dtypes.size();
    ABSL_CHECK_EQ(num_outputs,  // CRASH_OK=indicates invalid kernel
                  out_dims_list.size())
        << "Mismatching output sizes, num_outputs = " << num_outputs
        << ", out_dims_list.size() = " << out_dims_list.size();
    std::vector<Shape> output_shapes;
    output_shapes.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      output_shapes.emplace_back(CopyIntVector(out_dims_list[i]),
                                 out_dtypes[i]);
    }

    internal::DeferredOpParams params{
        .op_name = op_name,
        .op_builder =
            ToMlirOpBuilder<kArity, kNumOutputs>(std::move(op_builder)),
        .op_param_cache_keys = std::move(options.op_param_cache_keys),
        .inputs = std::move(inputs_vec),
        .output_shapes = std::move(output_shapes),
        .computation_dtype = options.computation_dtype,
        .split_mode = options.split_mode,
        .donated_indices = std::move(options.donated_indices),
    };
    TT_ASSIGN_OR_RETURN(
        results, internal::CreateDeferredDeviceBufferList(std::move(params)));
  }
  if constexpr (kNumOutputs == kDynamicSize) {
    return std::move(results);
  } else {
    // The type of op_builder guarantees that results has the correct number of
    // elements. The following check is just for catching bugs in the torch_tpu
    // implementation.
    ABSL_CHECK_EQ(results.size(), kNumOutputs)  // CRASH_OK
        << "expected " << kNumOutputs << " outputs, got " << results.size();
    if constexpr (kNumOutputs == 1) {
      // `results` is a singleton vector.
      return std::move(results[0]);
    } else {
      // Convert results (an std::vector) to std::array.
      return MoveToStdArray(
          FixedSizeSpan<DeviceBufferRef, kNumOutputs>(absl::MakeSpan(results)));
    }
  }
}

// Dispatches an op with out-tensor(s) (or in-place destination tensor(s)).
//
// Automatically detects if any output is eligible to donate an input tensor's
// device buffer in eager mode (via `AutoDonateInPlaceBuffers`) when
// `donated_indices` is not explicitly specified in `options`. After
// dispatching, resizes each output (if its shape differs) and assigns the
// resulting device buffer to that output tensor.
template <int kArity, int kNumOutputs = 1>
absl::Status DispatchOpOut(NAryMlirOpBuilder<kArity, kNumOutputs> op_builder,
                           const OpInputs<kArity>& inputs,
                           const OpOutputs<kNumOutputs>& outputs,
                           DispatchOpOptions<kNumOutputs> options) {
  const absl::Span<const at::Tensor> inputs_span =
      internal::GetTensorSpan<kArity>(inputs);
  const absl::Span<const at::Tensor> outputs_span =
      internal::GetTensorSpan<kNumOutputs>(outputs);

  if constexpr (kNumOutputs == 1) {
    internal::AutoDonateInPlaceBuffers(outputs_span,
                                       absl::MakeSpan(&options.out_dtype, 1),
                                       absl::MakeSpan(&options.out_dims, 1),
                                       inputs_span, options.donated_indices);
  } else {
    internal::AutoDonateInPlaceBuffers(outputs_span, options.out_dtypes,
                                       options.out_dims_list, inputs_span,
                                       options.donated_indices);
  }

  TT_ASSIGN_OR_RETURN(auto result_bufs,
                      (DispatchOp<kArity, kNumOutputs>(
                          std::move(op_builder), inputs, std::move(options))));

  if constexpr (kNumOutputs == 1) {
    return internal::AssignBufferToOutput(outputs_span[0],
                                          std::move(result_bufs));
  } else {
    return internal::AssignBuffersToOutputs(outputs_span,
                                            absl::MakeSpan(result_bufs));
  }
}

// Creates an at::Tensor from an at::Scalar.
//
// There are two modes of operation for this:
//  - In torch.compile mode (i.e. when the defer mode is kAll), the scalar is
//    treated as a *constant*, and the DeviceBufferRef in the tensor will be an
//    mlir::Constant op. This enables more aggressive compiler optimizations
//    when we know the scalar value is constant, achieving better warmed-up
//    performance, but risks recompilation on future runs where the scalar value
//    changes.
//    If scalar_type_opt is specified, it will be used to determine the MLIR
//    type of the constant. Otherwise, the type will be determined
//    automatically.
//  - In eager mode the scalar is treated as a *variable*, and the
//    DeviceBufferRef in the tensor will be materialized as a single- element
//    buffer. This prevents compiler specialization on the value, meaning that
//    the executable can be reused in the future even if the scalar value
//    changes. To prevent redundant host-to-device transfers, created
//    DeviceBufferRefs are cached and reused.
[[deprecated("Use PromoteScalar() instead.")]]
absl::StatusOr<at::Tensor> MakeTensor(
    const at::Scalar& scalar,
    c10::optional<at::ScalarType> scalar_type_opt = std::nullopt);

// Promotes the given scalar to a tensor lazily.
[[nodiscard]] PromotedScalar PromoteScalar(at::Scalar scalar);

// Promotes the given optional scalar to an optional tensor lazily.
[[nodiscard]] std::optional<PromotedScalar> PromoteScalar(
    std::optional<at::Scalar> scalar);

// Promotes the given scalars to tensors lazily.
[[nodiscard]] std::vector<PromotedScalar> PromoteScalar(
    at::ArrayRef<at::Scalar> scalars);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_OP_DISPATCHER_H_
