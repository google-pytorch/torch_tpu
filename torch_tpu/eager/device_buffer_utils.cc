/*
 * Copyright 2026 Google LLC
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

#include "torch_tpu/eager/device_buffer_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/accumulate.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/layout_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/events_queue.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/repeated_ops_heuristic.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/ops/stride/stride_helper.h"
#include "torch_tpu/ops/view_decomposition/decomposition.h"
#include "torch_tpu/ops/view_decomposition/inversion.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {
namespace {

// Upcasts inputs to computation_dtype if specified.
std::vector<mlir::MlirOp> CastInputs(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs,
    std::optional<mlir::ElementType> computation_dtype) {
  std::vector<mlir::MlirOp> casted_inputs;
  casted_inputs.reserve(inputs.size());
  if (computation_dtype) {
    mlir::Type computation_type =
        mlir::getElementType(builder.getContext(), *computation_dtype);
    for (auto input : inputs) {
      const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
      if (input_type.getElementType() != computation_type) {
        input = mlir::stablehlo::ConvertElementType(input, *computation_dtype);
      }
      casted_inputs.push_back(input);
    }
  } else {
    casted_inputs.assign(inputs.begin(), inputs.end());
  }
  return casted_inputs;
}

// Downcasts outputs to their expected dtypes.
DynamicMlirOpResults CastOutputs(
    mlir::MlirBuilder& builder, const DynamicMlirOpResults& results,
    absl::Span<const mlir::ElementType> expected_out_dtypes) {
  DynamicMlirOpResults casted_results;
  casted_results.reserve(results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    mlir::MlirOp res = results[i];
    const mlir::RankedTensorType res_type = GetTensorTypeOrDie(res);
    const mlir::Type expected_type =
        mlir::getElementType(builder.getContext(), expected_out_dtypes[i]);
    if (res_type.getElementType() != expected_type) {
      res = mlir::stablehlo::ConvertElementType(res, expected_out_dtypes[i]);
    }
    casted_results.push_back(res);
  }
  return casted_results;
}

// Modifies the op_builder to reflect the computation dtype of the op, if
// specified.
void ApplyComputationDtype(internal::DeferredOpParams& params) {
  std::vector<mlir::ElementType> expected_out_dtypes;
  expected_out_dtypes.reserve(params.output_shapes.size());
  for (const auto& shape : params.output_shapes) {
    expected_out_dtypes.push_back(shape.dtype());
  }

  params.op_builder = [computation_dtype = params.computation_dtype,
                       expected_out_dtypes = std::move(expected_out_dtypes),
                       op_builder = std::move(params.op_builder)](
                          mlir::MlirBuilder& builder,
                          absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    std::vector<mlir::MlirOp> casted_inputs =
        CastInputs(builder, inputs, computation_dtype);

    // 2. Run the original op builder.
    TT_ASSIGN_OR_RETURN(DynamicMlirOpResults results,
                        op_builder(builder, absl::MakeSpan(casted_inputs)));

    // 3. Verify that the number of outputs matches expected.
    ABSL_CHECK_EQ(results.size(),  // CRASH_OK=bug in torch_tpu
                  expected_out_dtypes.size())
        << "expected " << expected_out_dtypes.size()
        << " outputs from op, but got " << results.size();

    // 4. Downcast outputs to their expected dtypes.
    return CastOutputs(builder, results, expected_out_dtypes);
  };
}

// Handles post-op creation events for the kDeferNever and
// kDeferNeverAndLaunchBlocking eager modes.
absl::Status DeferNeverDispatch(absl::Span<const DeviceBufferRef> results,
                                const OpName op_name, const bool block) {
  const auto& device_buffer_list = results[0].device_buffer_list();
  if (IsMetadataOnly(op_name)) {
    // Don't eagerly materialize metadata-only ops.
    return absl::OkStatus();
  }
  // materialize.cc and events_queue.h will respect OpSplitMode, even with
  // MaterializationMode::kFullGraph.
  TT_RETURN_IF_ERROR(Materialize(device_buffer_list,
                                 MaterializationReason::kDebugMode,
                                 MaterializationMode::kFullGraph));
  for (const auto& result : results) {
    MarkStreamActive(result.GetReadyFuture());
  }
  if (block) {
    return device_buffer_list->Synchronize();
  }
  return absl::OkStatus();
}

// Handles post-op creation events for the kDeferAndFuse eager mode.
absl::Status DeferAndFuseDispatch(absl::Span<const DeviceBufferRef> results,
                                  const OpName op_name) {
  const auto& device_buffer_list = results[0].device_buffer_list();

  // Don't consider metadata-only ops for the repeated ops heuristic.
  if (MustApplyRepeatedOpsHeuristic() && !IsMetadataOnly(op_name)) {
    return ApplyRepeatedOpsHeuristic(device_buffer_list);
  }
  return absl::OkStatus();
}

}  // namespace

namespace internal {

absl::StatusOr<std::vector<DeviceBufferRef>> CreateDeferredDeviceBufferList(
    DeferredOpParams&& params) {
  ApplyComputationDtype(params);

  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> results,
      DeviceBufferList::CreateDeferred(
          params.op_name, std::move(params.op_builder),
          std::move(params.inputs), std::move(params.op_param_cache_keys),
          std::move(params.output_shapes), params.split_mode,
          std::move(params.donated_indices)));

  RecordDeferredOpCreated(results[0].device_buffer_list());

  switch (GetEagerMode()) {
    case EagerMode::kDeferNever:
      TT_RETURN_IF_ERROR(DeferNeverDispatch(results, params.op_name,
                                            /*block=*/false));
      break;
    case EagerMode::kDeferNeverAndLaunchBlocking:
      TT_RETURN_IF_ERROR(DeferNeverDispatch(results, params.op_name,
                                            /*block=*/true));
      break;
    case EagerMode::kDeferAndFuse:
      TT_RETURN_IF_ERROR(DeferAndFuseDispatch(results, params.op_name));
      break;
    case EagerMode::kInternalDeferAll:
      break;
  }
  return results;
}

}  // namespace internal

namespace {

// Common pathway for CreateConstantDeviceBufferRef and
// CreateZeroSizeDeviceBufferRef, allowing for different op names.
absl::StatusOr<DeviceBufferRef> CreateConstantDeviceBufferRefImpl(
    std::vector<char> cpu_tensor_data, Dimensions dimensions,
    mlir::ElementType element_type, OpName op_name) {
  // Create the components of the DeferredOp.
  ScopedPythonContextCapturer capturer(op_name);

  // Create the cache keys for the op parameters.
  // Since a change in data causes recompilation, we include the hash of the
  // tensor data as part of the cache key.
  auto op_param_cache_keys = OpParamCacheKeys::Empty();
  TT_RETURN_IF_ERROR(
      op_param_cache_keys.SetParam("data", absl::HashOf(cpu_tensor_data)));
  TT_RETURN_IF_ERROR(op_param_cache_keys.SetParam("dimensions", dimensions));
  TT_RETURN_IF_ERROR(
      op_param_cache_keys.SetParam("element_type", element_type));

  // The op returns a single tensor with the given shape and dtype.
  std::vector<Shape> output_shapes;
  output_shapes.push_back(Shape(dimensions, element_type));  // intentional copy

  auto op_builder = [cpu_tensor_data = std::move(cpu_tensor_data), element_type,
                     dimensions = std::move(dimensions)](
                        mlir::MlirBuilder& builder,
                        absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.empty(), error::kInvalidArgument)
        << "unexpected input to constant op";

    auto ranked_tensor_type =
        mlir::makeTensorType(builder.getContext(), dimensions, element_type);

    if (element_type == mlir::ElementType::PRED) {
      // Special case for boolean tensors.
      //
      // PyTorch stores booleans as one-per-byte on CPU, but XLA uses packed
      // 1-bit booleans. So we just make a constant byte tensor and use
      // stablehlo.convert, which will do the packing.
      //
      // DenseIntElementsAttr::get has an assertion that checks if the
      // signedness of the data, as indicated by
      // std::numeric_limits<char>::is_signed, matches the signedness of the
      // element type. However, whether or not char is signed is implementation
      // defined, which can cause assertion failures in some environments.
      // So we match the signedness of the byte tensor (using I8 or UI8) to the
      // signedness of char according to the current implementation.
      const auto byte_element_type = std::numeric_limits<char>::is_signed
                                         ? mlir::ElementType::I8
                                         : mlir::ElementType::UI8;

      auto shaped_byte_tensor_type = mlir::makeTensorType(
          builder.getContext(), dimensions, byte_element_type);
      auto shaped_byte_constant = mlir::stablehlo::Constant(
          builder, mlir::DenseIntElementsAttr::get(shaped_byte_tensor_type,
                                                   cpu_tensor_data));
      return DynamicMlirOpResults{
          mlir::stablehlo::Convert(ranked_tensor_type, shaped_byte_constant)};
    }

    auto dense_elements_attr = mlir::DenseElementsAttr::getFromRawBuffer(
        ranked_tensor_type, cpu_tensor_data);
    return DynamicMlirOpResults{
        mlir::stablehlo::Constant(builder, dense_elements_attr)};
  };

  internal::DeferredOpParams params{
      .op_name = op_name,
      .op_builder = std::move(op_builder),
      .op_param_cache_keys = std::move(op_param_cache_keys),
      .inputs = {},
      .output_shapes = std::move(output_shapes),
  };

  TT_ASSIGN_OR_RETURN(auto results, internal::CreateDeferredDeviceBufferList(
                                        std::move(params)));
  TT_RET_CHECK(results.size() == 1, error::kInternal)
      << "CreateConstantDeviceBufferRef should return exactly one output";
  return std::move(results[0]);
}

}  // namespace

absl::StatusOr<DeviceBufferRef> CreateConstantDeviceBufferRef(
    std::vector<char> cpu_tensor_data, Dimensions dimensions,
    mlir::ElementType element_type) {
  return CreateConstantDeviceBufferRefImpl(std::move(cpu_tensor_data),
                                           std::move(dimensions), element_type,
                                           OpName::kTorchTpuInternalConstant);
}

absl::StatusOr<DeviceBufferRef> CreateEmptyDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  auto op_builder = [dimensions =
                         CopyIntVector(absl::MakeConstSpan(dimensions)),
                     element_type](mlir::MlirBuilder& builder,
                                   absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.empty(), error::kInvalidArgument)
        << "CreateEmptyDeviceBufferRef should not be called with any inputs";
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, element_type, dimensions)};
  };
  Shape output_shape(std::move(dimensions), element_type);

  internal::DeferredOpParams params{
      .op_name = OpName::kEmpty,
      .op_builder = std::move(op_builder),
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
      .inputs = {},
      .output_shapes = {std::move(output_shape)},
  };
  TT_ASSIGN_OR_RETURN(auto results, internal::CreateDeferredDeviceBufferList(
                                        std::move(params)));
  ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> CreateZeroSizeDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type) {
  bool is_zero_sized = false;
  for (int64_t dim : dimensions) {
    if (dim == 0) {
      is_zero_sized = true;
      break;
    }
  }
  TT_RET_CHECK(is_zero_sized, error::kInvalidArgument)
      << "CreateZeroSizeDeviceBufferRef requires a zero-sized tensor, but got: "
      << ToString(dimensions);
  return CreateConstantDeviceBufferRefImpl({}, std::move(dimensions),
                                           element_type,
                                           OpName::kTorchTpuInternalZeroSize);
}

namespace {

absl::StatusOr<int64_t> RequiredStorageBytes(
    absl::Span<const int64_t> view_dimensions,
    absl::Span<const int64_t> view_strides, mlir::ElementType view_element_type,
    int64_t view_storage_offset) {
  TT_RET_CHECK(view_dimensions.size() == view_strides.size(),
               error::kInvalidArgument)
      << "view dimensions and strides must have the same size.";
  // The number of storage elements required is
  //  storage_offset + 1 + maximum element index.
  // This accounts for overlapping or non-dense views, which can span more or
  // fewer storage elements than there are logical elements in the tensor.
  int64_t required_numel = view_storage_offset + 1;
  for (auto i = 0; i < view_dimensions.size(); ++i) {
    if (view_dimensions[i] == 0) return 0;
    required_numel += (view_dimensions[i] - 1) * view_strides[i];
  }
  // Multiply by the size of the scalar type to get the number of bits.
  // Round up to the nearest byte, in the case of sub-byte-size types
  const int64_t element_bitwidth = TorchEquivalentBitwidth(view_element_type);
  return (required_numel * element_bitwidth + 7) / 8;
}

// A builder and its required inputs for a view inversion (simulated write).
struct AsStridedInverseBuilderPair {
  // The MlirOpBuilder to put in the constructed DeferredOp.
  MlirOpBuilder builder;
  // Either the write buffer only (if the view covers the entire base), or both
  // the write buffer and the base buffer (if the write is only partial).
  std::vector<DeviceBufferRef> inputs;
};

// Helper function for AssignBufferToAtTensor to switch between two cases:
//   * If the view is not a slice (so that the inversion fully overwrites the
//     original tensor), then we don't care what the original tensor's values
//     were, and the DeferredOp is unary.
//   * If the view is a slice, then some amount of its original data will be
//     retained, so we need to carry a dependency on the original base_buffer,
//     and the DeferredOp is binary.
// Returns the appropriate builder and inputs for the DeferredOp.
AsStridedInverseBuilderPair MakeAsStridedInverseBuilder(
    InverseViewOperation inverse_view_operation,
    DeviceBufferRef base_buffer_ref, DeviceBufferRef write_buf) {
  if (inverse_view_operation.stages.size() == 1) {
    ABSL_VLOG(3)
        << "[MakeAsStridedInverseBuilder] View elements are 1:1 with base, "
           "creating an inverse DeferredOp that overwrites the base.";
    // Merge the view bitcast with the inverse sequence.
    ViewSequence merged_sequence;
    merged_sequence.reserve(inverse_view_operation.bitcast_view.size() +
                            inverse_view_operation.stages[0].inverse.size());
    for (auto& primitive : inverse_view_operation.bitcast_view) {
      merged_sequence.push_back(std::move(primitive));
    }
    for (auto& inverse_primitive : inverse_view_operation.stages[0].inverse) {
      std::visit(
          [&merged_sequence](auto& prim) {
            merged_sequence.push_back(std::move(prim));
          },
          inverse_primitive);
    }

    auto op_builder = [view_sequence = std::move(merged_sequence)](
                          mlir::MlirBuilder& builder,
                          absl::Span<mlir::MlirOp> inputs)
        -> absl::StatusOr<DynamicMlirOpResults> {
      TT_RET_CHECK(inputs.size() == 1, error::kInternal)
          << "expected 1 input, got " << inputs.size();
      TT_ASSIGN_OR_RETURN(auto output,
                          ViewSequenceShlo(inputs[0], view_sequence));
      return DynamicMlirOpResults{std::move(output)};
    };
    auto inputs = {std::move(write_buf)};
    return {.builder = std::move(op_builder), .inputs = std::move(inputs)};
  }

  // Otherwise, apply the full operation.
  ABSL_VLOG(3) << "[MakeAsStridedInverseBuilder] View is a slice, creating an "
                  "inverse DeferredOp that partially overwrites the base.";
  auto op_builder =
      [inverse_view_operation = std::move(inverse_view_operation)](
          mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.size() == 2, error::kInternal)
        << "expected 2 inputs, got " << inputs.size();
    TT_ASSIGN_OR_RETURN(
        auto output,
        InverseViewOperationShlo(inputs[0], inputs[1], inverse_view_operation));
    return DynamicMlirOpResults{std::move(output)};
  };
  auto inputs = {std::move(base_buffer_ref), std::move(write_buf)};
  return {std::move(op_builder), std::move(inputs)};
}

}  // namespace

absl::StatusOr<DeviceBufferRef> CreateViewDeviceBufferRef(
    DeviceBufferRef base_buffer_ref, const TpuLayout& tpu_layout,
    bool view_is_conj) {
  absl::Span<const int64_t> view_dimensions = tpu_layout.sizes;
  absl::Span<const int64_t> view_strides = tpu_layout.strides;
  int64_t view_storage_offset = tpu_layout.storage_offset;
  mlir::ElementType view_element_type = tpu_layout.element_type;
  TT_RET_CHECK(view_dimensions.size() == view_strides.size(),
               error::kInvalidArgument)
      << "view dimensions and strides must have the same size.";
  // If the view is zero-sized, then we don't need a view decomposition; a view
  // of nothing is valid.
  if (std::any_of(view_dimensions.begin(), view_dimensions.end(),
                  [](int64_t dim) { return dim == 0; })) {
    ABSL_VLOG(1) << "[CreateViewDeviceBufferRef] View is zero-sized, "
                    "returning zero-sized constant.";
    return CreateZeroSizeDeviceBufferRef(CopyIntVector(view_dimensions),
                                         view_element_type);
  }

  // We need to apply a view decomposition.
  // Check to make sure we have enough data to read the tensor.
  TT_ASSIGN_OR_RETURN(
      const int64_t required_bytes,
      RequiredStorageBytes(view_dimensions, view_strides, view_element_type,
                           view_storage_offset));
  const int64_t actual_bytes =
      (base_buffer_ref.num_elements() *
           TorchEquivalentBitwidth(base_buffer_ref.element_type()) +
       7) /
      8;

  TT_RET_CHECK(required_bytes <= actual_bytes, error::kPythonIndexError)
      << "cannot read " << required_bytes << " bytes ("
      << c10::multiply_integers(view_dimensions) << " elements of type "
      << ToString(view_element_type) << " with an offset of "
      << view_storage_offset << " elements) from a storage buffer with "
      << actual_bytes << " bytes";

  // Compute the steps in the decomposition.
  StridedLayout view_layout = {.storage_offset = view_storage_offset};
  view_layout.strided_dims.reserve(view_dimensions.size());
  for (auto i = 0; i < view_dimensions.size(); ++i) {
    view_layout.strided_dims.push_back(
        {.size = view_dimensions[i], .stride = view_strides[i]});
  }
  TT_ASSIGN_OR_RETURN(
      ViewSequence view_sequence,
      DecomposeIntoViewSequence(base_buffer_ref.dimensions(),
                                base_buffer_ref.element_type(), view_layout,
                                view_element_type, view_is_conj));
  Simplify(view_sequence, base_buffer_ref.dimensions());
  ABSL_VLOG(1) << "[GetBuffer] Decomposed base buffer dtype and shape "
               << ToString(base_buffer_ref.element_type())
               << ToString(base_buffer_ref.dimensions())
               << " into view sequence " << ToString(view_sequence)
               << " to achieve target view layout " << view_layout
               << " (is_conj=" << view_is_conj << ")";

  // Create cache key for the view sequence before moving into builder
  TT_ASSIGN_OR_RETURN(
      OpParamCacheKeys param_keys,
      ViewSequenceCacheKey(view_sequence, view_strides, view_storage_offset));

  // Build the values to construct an ephemeral DeferredOp.
  const auto op_name = OpName::kAsStrided;
  ScopedPythonContextCapturer capturer(op_name);

  auto op_builder = [view_sequence = std::move(view_sequence)](
                        mlir::MlirBuilder& builder,
                        absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.size() == 1, error::kInternal)
        << "expected 1 input, got " << inputs.size();
    TT_ASSIGN_OR_RETURN(auto output,
                        ViewSequenceShlo(inputs[0], view_sequence));
    return DynamicMlirOpResults{std::move(output)};
  };

  std::vector<DeviceBufferRef> inputs = {std::move(base_buffer_ref)};

  std::vector<Shape> output_shapes;
  output_shapes.emplace_back(CopyIntVector(view_dimensions), view_element_type);

  // Create the deferred op.
  internal::DeferredOpParams params{
      .op_name = op_name,
      .op_builder = std::move(op_builder),
      .op_param_cache_keys = std::move(param_keys),
      .inputs = std::move(inputs),
      .output_shapes = std::move(output_shapes),
  };
  TT_ASSIGN_OR_RETURN(auto results, internal::CreateDeferredDeviceBufferList(
                                        std::move(params)));
  ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> CreateInverseViewDeviceBufferRef(
    DeviceBufferRef base_buffer_ref, DeviceBufferRef write_buf,
    const TpuLayout& tpu_layout, bool view_is_conj) {
  absl::Span<const int64_t> view_dimensions = tpu_layout.sizes;
  absl::Span<const int64_t> view_strides = tpu_layout.strides;
  int64_t view_storage_offset = tpu_layout.storage_offset;
  mlir::ElementType view_element_type = tpu_layout.element_type;
  TT_RET_CHECK(view_dimensions.size() == view_strides.size(),
               error::kInvalidArgument)
      << "view dimensions and strides must have the same size.";
  TT_RET_CHECK(!IsOverlapping(view_dimensions, view_strides),
               error::kFailedPrecondition)
      << "inplace writes to overlapping views are undefined behavior and are "
         "not supported.\n"
      << "Because multiple logical tensor indices point to the same buffer "
         "elements, writes from multiple indices may overwrite each other.\n"
      << "Please use clone() or contiguous() to copy the tensor before "
         "writing";

  ABSL_VLOG(2) << "[AssignBufferToAtTensor] Assigning to non-contiguous base.";

  // Compute the operations needed to invert the view sequence.
  StridedLayout view_layout{.storage_offset = view_storage_offset};
  view_layout.strided_dims.reserve(view_dimensions.size());
  for (auto i = 0; i < view_dimensions.size(); ++i) {
    view_layout.strided_dims.push_back(
        {.size = view_dimensions[i], .stride = view_strides[i]});
  }
  TT_ASSIGN_OR_RETURN(
      InverseViewOperation inverse_view_operation,
      ComputeInverseViewOperation(base_buffer_ref.dimensions(),
                                  base_buffer_ref.element_type(), view_layout,
                                  view_element_type, view_is_conj));

  // Create a new DeferredOp representing the updated base value
  // Build the values to construct an ephemeral DeferredOp.
  const auto op_name = OpName::kAsStridedInverse;
  ScopedPythonContextCapturer capturer(op_name);

  std::vector<Shape> output_shapes = {inverse_view_operation.final_shape};

  auto [op_builder, inputs] = MakeAsStridedInverseBuilder(
      std::move(inverse_view_operation), std::move(base_buffer_ref),
      std::move(write_buf));

  TT_ASSIGN_OR_RETURN(OpParamCacheKeys param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(
                          view_strides, view_storage_offset, view_is_conj));

  // Create the deferred DeviceBufferRef.
  internal::DeferredOpParams params{
      .op_name = op_name,
      .op_builder = std::move(op_builder),
      .op_param_cache_keys = std::move(param_keys),
      .inputs = std::move(inputs),
      .output_shapes = std::move(output_shapes),
  };
  TT_ASSIGN_OR_RETURN(auto results, internal::CreateDeferredDeviceBufferList(
                                        std::move(params)));
  ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
  return std::move(results[0]);
}

}  // namespace torch_tpu
