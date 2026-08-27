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

#include "torch_tpu/csrc/ops/jagged/jagged_aten_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/structured_log_buffer.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/pjrt/pjrt_utils.h"

namespace torch_tpu {

namespace {

// Reads offsets tensor values into host Indices.
//
// In TorchTPU eager execution, calling ATen methods like .to(at::kCPU) triggers
// nested ATen dispatch (_to_copy), which is banned under composite op checks.
// To circumvent composite dispatch, we directly transfer device memory to host
// using MaterializeAndReturn + TpuMemcpyDtoHDirect when offsets reside on TPU.
absl::StatusOr<Indices> GetCpuOffsets(const at::Tensor& offsets_tensor) {
  const int64_t num_elements = offsets_tensor.numel();
  Indices offsets(num_elements);
  if (num_elements == 0) {
    return offsets;
  }
  if (offsets_tensor.is_cpu()) {
    at::Tensor contig = offsets_tensor.contiguous();
    std::memcpy(offsets.data(), contig.data_ptr<int64_t>(),
                num_elements * sizeof(int64_t));
  } else {
    TT_ASSIGN_OR_RETURN(
        auto materialized_buf,
        MaterializeAndReturn(offsets_tensor,
                             MaterializationReason::kScalarConversion));
    TT_RETURN_IF_ERROR(TpuMemcpyDtoHDirect(materialized_buf, offsets.data()));
  }
  return offsets;
}

// Validates the offsets TensorList, retrieves host-side offset coordinates, and
// registers them in OpParamCacheKeys.
//
// Registering offsets_vec in the cache key ensures that different sequence
// length distributions compile to isolated, dedicated executables rather than
// causing compilation cache collisions.
absl::StatusOr<Indices> ValidateAndGetOffsets(at::TensorList offsets,
                                              OpParamCacheKeys& param_keys) {
  TT_RET_CHECK(offsets.size() == 1, error::kPythonNotImplementedError)
      << "expected only 1 offset tensor (only 1 jagged dim is supported for "
         "now), got "
      << offsets.size() << " offset tensors";

  const at::Tensor& offsets_tensor = offsets[0];
  TT_RET_CHECK(offsets_tensor.dim() == 1, error::kInvalidArgument)
      << "expected offsets to have only 1 dimension, got "
      << offsets_tensor.dim();
  TT_RET_CHECK(offsets_tensor.size(0) >= 1, error::kInvalidArgument)
      << "expected offsets tensors to have size >= 1, got "
      << offsets_tensor.size(0);
  TT_RET_CHECK(offsets_tensor.scalar_type() == at::kLong,
               error::kInvalidArgument)
      << "expected offsets to be of dtype int64, got "
      << ToString(offsets_tensor.scalar_type());

  TT_ASSIGN_OR_RETURN(Indices offsets_vec, GetCpuOffsets(offsets_tensor));

  TT_RET_CHECK(offsets_vec[0] == 0, error::kInvalidArgument)
      << "expected the first offset to be 0, got " << offsets_vec[0];
  for (size_t i = 0; i + 1 < offsets_vec.size(); ++i) {
    TT_RET_CHECK(offsets_vec[i] <= offsets_vec[i + 1], error::kInvalidArgument)
        << "expected offsets to be non-decreasing, got offsets[" << i << "] ("
        << offsets_vec[i] << ") > offsets[" << i + 1 << "] ("
        << offsets_vec[i + 1] << ")";
  }

  TT_RETURN_IF_ERROR(
      param_keys.SetParam("offsets_vec", absl::MakeConstSpan(offsets_vec)));
  return offsets_vec;
}

// Helper to construct an int64 StableHLO constant tensor from an ArrayRef.
mlir::MlirOp CreateI64Constant(mlir::MlirBuilder& builder,
                               llvm::ArrayRef<int64_t> data,
                               llvm::ArrayRef<int64_t> shape) {
  auto type =
      mlir::RankedTensorType::get(shape, builder.getOpBuilder().getI64Type());
  return mlir::stablehlo::Constant(builder, mlir::makeConstant(data, type));
}

// Helper returning the extreme value scalar for infinite padding on
// integral/bool types.
at::Scalar GetInfinitePaddingScalar(at::ScalarType dtype, bool is_positive) {
  switch (dtype) {
    case at::kByte:
      return is_positive ? std::numeric_limits<uint8_t>::max()
                         : std::numeric_limits<uint8_t>::lowest();
    case at::kChar:
      return is_positive ? std::numeric_limits<int8_t>::max()
                         : std::numeric_limits<int8_t>::lowest();
    case at::kShort:
      return is_positive ? std::numeric_limits<int16_t>::max()
                         : std::numeric_limits<int16_t>::lowest();
    case at::kInt:
      return is_positive ? std::numeric_limits<int32_t>::max()
                         : std::numeric_limits<int32_t>::lowest();
    case at::kLong:
      return is_positive ? std::numeric_limits<int64_t>::max()
                         : std::numeric_limits<int64_t>::lowest();
    case at::kBool:
      return is_positive;
    default:
      return 0;
  }
}

// Normalizes the padding value. For integral and boolean types, non-finite
// floating-point padding values (e.g., +inf, -inf, NaN) are clamped to the
// type's extreme values to avoid undefined behavior during integer conversion,
// matching PyTorch's CUDA implementation in
// `ATen/native/nested/NestedTensorUtils.h: _get_padding_value` and
// `ATen/native/nested/cuda/NestedTensorTransformerFunctions.cu`.
at::Scalar NormalizePaddingValue(const at::Tensor& values,
                                 double padding_value) {
  if (values.is_floating_point() || values.is_complex()) {
    return at::Scalar(padding_value);
  }
  if (std::isinf(padding_value)) {
    return GetInfinitePaddingScalar(values.scalar_type(), padding_value > 0);
  }
  if (std::isnan(padding_value)) {
    return at::Scalar(0);
  }
  return at::Scalar(padding_value);
}

// Helper struct holding calculated index mappings for JaggedToPaddedDense.
struct JaggedToPaddedIndices {
  int64_t num_valid = 0;
  Indices scatter_indices;
  Indices gather_indices;
  bool needs_truncation = false;
};

// Computes 1D linear scatter/gather index arrays for JaggedToPaddedDense.
JaggedToPaddedIndices ComputeJaggedToPaddedIndices(
    absl::Span<const int64_t> offsets_vec, int64_t batch_size,
    int64_t max_length, int64_t total_values) {
  JaggedToPaddedIndices result;
  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t start = offsets_vec[i];
    const int64_t end = offsets_vec[i + 1];
    const int64_t len = std::min(end - start, max_length);
    if (len > 0) {
      result.num_valid += len;
    }
  }

  result.scatter_indices.reserve(result.num_valid);
  result.needs_truncation = result.num_valid < total_values;
  if (result.needs_truncation) {
    result.gather_indices.reserve(result.num_valid);
  }

  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t start = offsets_vec[i];
    const int64_t end = offsets_vec[i + 1];
    const int64_t len = std::min(end - start, max_length);
    for (int64_t j = 0; j < len; ++j) {
      result.scatter_indices.push_back(i * max_length + j);
      if (result.needs_truncation) {
        result.gather_indices.push_back(start + j);
      }
    }
  }
  return result;
}

// Lowers JaggedToPaddedDense StableHLO graph.
absl::StatusOr<mlir::MlirOp> BuildJaggedToPaddedDenseHlo(
    mlir::MlirOp values_op, const Dimensions& values_shape,
    const Dimensions& padded_shape, const Dimensions& flat_padded_shape,
    mlir::ElementType output_dtype, at::Scalar fill_scalar,
    const JaggedToPaddedIndices& indices) {
  auto& builder = values_op.getBuilder();
  const int64_t values_rank = values_shape.size();

  // Step 1: If sequence lengths exceed max_length, gather valid subset.
  mlir::MlirOp updates = values_op;
  if (indices.needs_truncation) {
    llvm::SmallVector<int64_t, 4> offset_dims;
    for (int64_t i = 1; i < values_rank; ++i) {
      offset_dims.push_back(i);
    }
    auto gather_dims = mlir::stablehlo::GatherDimensionNumbersAttr::get(
        &builder.getContext(),
        /*offset_dims=*/offset_dims,
        /*collapsed_slice_dims=*/{0},
        /*operand_batching_dims=*/{},
        /*start_index_batching_dims=*/{},
        /*start_index_map=*/{0},
        /*index_vector_dim=*/1);
    Dimensions slice_sizes(values_shape.begin(), values_shape.end());
    slice_sizes[0] = 1;

    auto gather_indices_op =
        CreateI64Constant(builder, indices.gather_indices, {indices.num_valid});
    updates = mlir::stablehlo::Gather(values_op, gather_indices_op, gather_dims,
                                      slice_sizes);
  }

  // Step 2: Initialize flattened padded buffer with padding constant.
  TT_ASSIGN_OR_RETURN(
      auto padded_init,
      MakeConstant(builder, fill_scalar, output_dtype, flat_padded_shape));

  // Step 3: Perform 1D scatter onto flat buffer.
  auto scatter_indices_op = CreateI64Constant(builder, indices.scatter_indices,
                                              {indices.num_valid, 1});

  llvm::SmallVector<int64_t, 4> update_window_dims;
  for (int64_t i = 1; i < values_rank; ++i) {
    update_window_dims.push_back(i);
  }
  auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      &builder.getContext(),
      /*update_window_dims=*/update_window_dims,
      /*inserted_window_dims=*/{0},
      /*input_batching_dims=*/{},
      /*scatter_indices_batching_dims=*/{},
      /*scatter_dims_to_operand_dims=*/{0},
      /*index_vector_dim=*/1);

  auto body = [elem_type = GetTensorTypeOrDie(values_op).getElementType()](
                  mlir::RegionBuilder& rb) {
    auto block_type = mlir::RankedTensorType::get({}, elem_type);
    mlir::MlirOp _ = mlir::Argument(rb, block_type);
    mlir::MlirOp new_val = mlir::Argument(rb, block_type);
    mlir::stablehlo::Return(rb, {new_val});
  };

  auto scatter_op = mlir::stablehlo::Scatter({padded_init}, scatter_indices_op,
                                             {updates}, body, scatter_dims)[0];

  // Step 4: Reshape back to target (batch_size, max_length, *features).
  return ReshapeFromStaticDimensions(scatter_op, flat_padded_shape,
                                     padded_shape);
}

// Lowers aten._jagged_to_padded_dense_forward to efficient StableHLO:
// 1. Emits 1D flattened linear scatter coordinates (i * max_length + j),
//    halving the index constant size and eliminating 2D address math on TPU
//    VPU.
// 2. Scatters directly onto flat constant-initialized buffer of shape
//    (batch_size * max_length, *features) with inserted_window_dims={0}.
// 3. Bitcast-reshapes flat buffer to (batch_size, max_length, *features).
absl::StatusOr<DeviceBufferRef> JaggedToPaddedDense(
    const at::Tensor& values, at::TensorList offsets,
    c10::SymIntArrayRef max_lengths, double padding_value,
    OpParamCacheKeys param_keys) {
  TT_RET_CHECK(values.dim() >= 1, error::kInvalidArgument)
      << "expected values to have >= 1 dimensions, got " << values.dim();

  TT_RET_CHECK(max_lengths.size() == 1, error::kInvalidArgument)
      << "expected max_lengths to have only 1 element, got "
      << max_lengths.size();

  const int64_t max_length = max_lengths[0].expect_int();
  TT_RET_CHECK(max_length >= 0, error::kInvalidArgument)
      << "expected max_lengths[0] to be >= 0, got " << max_length;

  TT_ASSIGN_OR_RETURN(const Indices offsets_vec,
                      ValidateAndGetOffsets(offsets, param_keys));

  TT_RET_CHECK(offsets_vec.back() <= values.size(0), error::kInvalidArgument)
      << "expected the last offset to be <= the number of values ("
      << values.size(0) << "), got " << offsets_vec.back();

  const int64_t batch_size = offsets_vec.size() - 1;
  const auto values_shape_ref = values.sizes();
  const Dimensions values_shape(values_shape_ref.begin(),
                                values_shape_ref.end());
  Dimensions padded_shape;
  padded_shape.reserve(values.dim() + 1);
  padded_shape.push_back(batch_size);
  padded_shape.push_back(max_length);
  padded_shape.insert(padded_shape.end(), values_shape.begin() + 1,
                      values_shape.end());

  Dimensions flat_padded_shape;
  flat_padded_shape.reserve(values.dim());
  flat_padded_shape.push_back(batch_size * max_length);
  flat_padded_shape.insert(flat_padded_shape.end(), values_shape.begin() + 1,
                           values_shape.end());

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(values.scalar_type()));
  const at::Scalar fill_scalar = NormalizePaddingValue(values, padding_value);

  JaggedToPaddedIndices indices = ComputeJaggedToPaddedIndices(
      offsets_vec, batch_size, max_length, values.size(0));

  // Fast path for empty/zero-size cases: emit constant buffer initialized
  // with fill_scalar.
  if (batch_size == 0 || max_length == 0 || values.numel() == 0 ||
      indices.num_valid == 0) {
    auto nullary_builder = [fill_scalar, output_dtype,
                            padded_shape](mlir::MlirBuilder& builder) {
      return MakeConstant(builder, fill_scalar, output_dtype, padded_shape);
    };
    return DispatchOp<0>(std::move(nullary_builder), /*inputs=*/{},
                         {.out_dtype = output_dtype,
                          .out_dims = padded_shape,
                          .op_param_cache_keys = std::move(param_keys)});
  }

  auto op_builder =
      [fill_scalar, output_dtype, flat_padded_shape, padded_shape, values_shape,
       indices = std::move(indices)](
          mlir::MlirOp values_op) -> absl::StatusOr<mlir::MlirOp> {
    return BuildJaggedToPaddedDenseHlo(values_op, values_shape, padded_shape,
                                       flat_padded_shape, output_dtype,
                                       fill_scalar, indices);
  };

  return DispatchOp<1>(std::move(op_builder), values,
                       {.out_dtype = output_dtype,
                        .out_dims = padded_shape,
                        .op_param_cache_keys = std::move(param_keys)});
}

// Computes 1D linear gather indices for PaddedDenseToJagged.
absl::StatusOr<Indices> ComputePaddedToJaggedIndices(
    absl::Span<const int64_t> offsets_vec, int64_t batch_size,
    int64_t max_length, int64_t total_L_computed) {
  Indices gather_indices;
  gather_indices.reserve(total_L_computed);

  // Length of each batch item.
  // Used for showing the length of each batch item in the error message.
  SmallInt64Vector lengths;
  lengths.reserve(batch_size);

  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t start = offsets_vec[i];
    const int64_t end = offsets_vec[i + 1];
    const int64_t len = end - start;

    TT_RET_CHECK(len <= max_length, error::kInvalidArgument)
        << "expected all batch items to have length <= " << max_length
        << " (max length specified by padded input), got " << len;
    for (int64_t j = 0; j < len; ++j) {
      gather_indices.push_back(i * max_length + j);
    }

    lengths.push_back(len);
  }

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Invariant: sum of segments matches
                 // total_L
      gather_indices.size() == total_L_computed, error::kInvalidArgument)
      << "expected the sum of the computed batch item lengths to match total_L "
         "("
      << total_L_computed << "), got " << batch_size << " batches of sizes "
      << SequenceToReadableStr(absl::MakeConstSpan(lengths));
  return gather_indices;
}

// Lowers PaddedDenseToJagged StableHLO graph.
absl::StatusOr<mlir::MlirOp> BuildPaddedToJaggedHlo(
    mlir::MlirOp dense_op, int64_t dense_rank, const Dimensions& dense_shape,
    const Dimensions& flat_dense_shape, int64_t total_L_computed,
    absl::Span<const int64_t> gather_indices) {
  auto& builder = dense_op.getBuilder();

  TT_ASSIGN_OR_RETURN(
      auto dense_flat,
      ReshapeFromStaticDimensions(dense_op, dense_shape, flat_dense_shape));

  auto gather_indices_op =
      CreateI64Constant(builder, gather_indices, {total_L_computed});

  llvm::SmallVector<int64_t, 4> offset_dims;
  for (int64_t i = 2; i < dense_rank; ++i) {
    offset_dims.push_back(i - 1);
  }
  auto dimension_numbers = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      &builder.getContext(),
      /*offset_dims=*/offset_dims,
      /*collapsed_slice_dims=*/{0},
      /*operand_batching_dims=*/{},
      /*start_index_batching_dims=*/{},
      /*start_index_map=*/{0},
      /*index_vector_dim=*/1);

  Dimensions slice_sizes(flat_dense_shape.begin(), flat_dense_shape.end());
  slice_sizes[0] = 1;

  return mlir::stablehlo::Gather(dense_flat, gather_indices_op,
                                 dimension_numbers, slice_sizes);
}

// Lowers aten._padded_dense_to_jagged_forward to efficient StableHLO:
// 1. Reshapes input dense to flat (batch_size * max_length, *features).
// 2. Gathers jagged elements with 1D pre-flattened linear indices
//    (i * max_length + j), avoiding multi-dimensional index stride computations
//    inside TPU VPU execution.
absl::StatusOr<DeviceBufferRef> PaddedDenseToJagged(
    const at::Tensor& dense, at::TensorList offsets, int64_t total_L_val,
    OpParamCacheKeys param_keys) {
  TT_RET_CHECK(dense.dim() >= 2, error::kInvalidArgument)
      << "expected dense to have >= 2 dimensions, got " << dense.dim();

  TT_ASSIGN_OR_RETURN(const Indices offsets_vec,
                      ValidateAndGetOffsets(offsets, param_keys));

  const int64_t num_offsets = offsets_vec.size();
  const int64_t final_offset =
      num_offsets > 0 ? offsets_vec[num_offsets - 1] : 0;
  const int64_t total_L_computed =
      total_L_val >= 0 ? total_L_val : final_offset;
  if (total_L_val >= 0) {
    TT_RET_CHECK(final_offset == total_L_val, error::kInvalidArgument)
        << "expected the last offset to match total_L (" << total_L_val
        << "), got " << final_offset;
  }

  const auto dense_shape_ref = dense.sizes();
  const Dimensions dense_shape(dense_shape_ref.begin(), dense_shape_ref.end());

  const int64_t batch_size = num_offsets - 1;
  TT_RET_CHECK(batch_size == dense_shape[0], error::kInvalidArgument)
      << "expected dense first dimension size to match the batch size inferred "
         "from the offsets ("
      << batch_size << "), got " << dense_shape[0];

  Dimensions flat_dense_shape;
  flat_dense_shape.reserve(dense.dim() - 1);
  flat_dense_shape.push_back(dense_shape[0] * dense_shape[1]);
  flat_dense_shape.insert(flat_dense_shape.end(), dense_shape.begin() + 2,
                          dense_shape.end());

  Dimensions values_shape;
  values_shape.reserve(dense.dim() - 1);
  values_shape.push_back(total_L_computed);
  values_shape.insert(values_shape.end(), dense_shape.begin() + 2,
                      dense_shape.end());

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(dense.scalar_type()));

  // Fast path for empty output: emit constant buffer of size 0.
  if (total_L_computed == 0) {
    auto nullary_builder = [output_dtype,
                            values_shape](mlir::MlirBuilder& builder) {
      return MakeConstant(builder, at::Scalar(0), output_dtype, values_shape);
    };
    return DispatchOp<0>(std::move(nullary_builder), /*inputs=*/{},
                         {.out_dtype = output_dtype,
                          .out_dims = values_shape,
                          .op_param_cache_keys = std::move(param_keys)});
  }

  // Construct 1D linear gather indices (i * max_length + j).
  const int64_t max_length = dense_shape[1];
  TT_ASSIGN_OR_RETURN(
      Indices gather_indices_data,
      ComputePaddedToJaggedIndices(offsets_vec, batch_size, max_length,
                                   total_L_computed));

  const int64_t dense_rank = dense.dim();

  auto op_builder = [dense_rank, dense_shape, flat_dense_shape,
                     total_L_computed,
                     gather_indices_data = std::move(gather_indices_data)](
                        mlir::MlirOp dense_op) -> absl::StatusOr<mlir::MlirOp> {
    return BuildPaddedToJaggedHlo(dense_op, dense_rank, dense_shape,
                                  flat_dense_shape, total_L_computed,
                                  gather_indices_data);
  };

  return DispatchOp<1>(std::move(op_builder), dense,
                       {.out_dtype = output_dtype,
                        .out_dims = values_shape,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenJaggedToPaddedDenseForward(const at::Tensor& values,
                                          at::TensorList offsets,
                                          c10::SymIntArrayRef max_lengths,
                                          double padding_value) {
  TT_KERNEL(OpName::kJaggedToPaddedDenseForward, param_keys,
            (values, offsets, max_lengths, padding_value), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buf,
                  JaggedToPaddedDense(values, offsets, max_lengths,
                                      padding_value, std::move(param_keys)));
              return MakeTensor(std::move(result_buf));
            });
}

at::Tensor AtenPaddedDenseToJaggedForward(const at::Tensor& dense,
                                          at::TensorList offsets,
                                          std::optional<c10::SymInt> total_L) {
  // Normalize optional<SymInt> to int64_t before TT_KERNEL argument logging
  // check.
  const int64_t total_L_val =
      total_L.has_value() ? total_L.value().expect_int() : -1;
  TT_KERNEL(OpName::kPaddedDenseToJaggedForward, param_keys,
            (dense, offsets, total_L_val), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buf,
                  PaddedDenseToJagged(dense, offsets, total_L_val,
                                      std::move(param_keys)));
              return MakeTensor(std::move(result_buf));
            });
}

// =============================================================================
// Nested Tensor (torch.jagged) View Construction and Metadata Accessors
// =============================================================================
// In PyTorch's jagged nested tensor architecture (torch.jagged layout), a
// jagged tensor is represented by a flat values buffer (`self`) and associated
// sequence metadata (offsets, lengths, ragged dimension index, and min/max
// sequence length bounds).
//
// At the backend device level (PrivateUse1):
// 1. Zero-Copy Buffer Sharing: View constructors (`_nested_view_from_jagged`,
//    `_nested_from_padded_tensor`) and buffer accessors (`_nested_get_values`,
//    `_nested_get_jagged_dummy`) reuse the underlying device buffer via
//    `self.alias()`, ensuring zero-copy performance and transparent backward
//    autograd gradient propagation.
// 2. Metadata Decoupling: Arguments such as `offsets`, `lengths`, `ragged_idx`,
//    and sequence bounds are purely metadata encapsulated on the frontend
//    NestedTensor wrapper. They do not alter physical TPU memory layout or
//    require device execution graphs, and are ignored in StableHLO cache keys.
// =============================================================================

at::Tensor AtenNestedViewFromJagged(
    const at::Tensor& self, const at::Tensor& offsets, const at::Tensor& dummy,
    const std::optional<at::Tensor>& lengths, int64_t ragged_idx,
    const std::optional<at::Tensor>& min_seqlen,
    const std::optional<at::Tensor>& max_seqlen) {
  TT_KERNEL(
      OpName::kNestedViewFromJagged, _,
      (self, offsets, dummy, IgnoreInCacheKey(lengths, "Doesn't affect SHLO"),
       IgnoreInCacheKey(ragged_idx, "Doesn't affect SHLO"),
       IgnoreInCacheKey(min_seqlen, "Doesn't affect SHLO"),
       IgnoreInCacheKey(max_seqlen, "Doesn't affect SHLO")),
      { return self.alias(); });
}

at::Tensor AtenNestedFromPaddedTensor(
    const at::Tensor& padded, const at::Tensor& offsets,
    const at::Tensor& dummy, int64_t ragged_idx,
    const std::optional<at::Tensor>& min_seqlen,
    const std::optional<at::Tensor>& max_seqlen,
    std::optional<c10::SymInt> sum_S) {
  at::Tensor values = AtenPaddedDenseToJaggedForward(padded, {offsets}, sum_S);
  return AtenNestedViewFromJagged(values, offsets, dummy,
                                  /*lengths=*/std::nullopt, ragged_idx,
                                  min_seqlen, max_seqlen);
}

at::Tensor AtenNestedGetValues(const at::Tensor& self) {
  TT_KERNEL(OpName::kNestedGetValues, _, (self), { return self.alias(); });
}

at::Tensor AtenNestedGetOffsets(const at::Tensor& self) {
  TT_KERNEL(OpName::kNestedGetOffsets, _, (self), { return at::Tensor(); });
}

at::Tensor AtenNestedGetLengths(const at::Tensor& self) {
  TT_KERNEL(OpName::kNestedGetLengths, _, (self), { return at::Tensor(); });
}

int64_t AtenNestedGetRaggedIdx(const at::Tensor& self) {
  TT_KERNEL(OpName::kNestedGetRaggedIdx, _, (self), { return 1; });
}

at::Tensor AtenNestedGetMinSeqlen(const at::Tensor& self) {
  TT_KERNEL(OpName::kNestedGetMinSeqlen, _, (self), { return at::Tensor(); });
}

at::Tensor AtenNestedGetMaxSeqlen(const at::Tensor& self) {
  TT_KERNEL(OpName::kNestedGetMaxSeqlen, _, (self), { return at::Tensor(); });
}

at::Tensor AtenNestedGetJaggedDummy(const at::Tensor& any) {
  TT_KERNEL(OpName::kNestedGetJaggedDummy, _, (any), { return any.alias(); });
}

}  // namespace torch_tpu
