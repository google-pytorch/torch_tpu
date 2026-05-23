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

#include "torch_tpu/ops/pooling/adaptive_avg_pool_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "c10/core/SymIntArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/pooling/avg_pool_aten_kernels.h"
#include "torch_tpu/ops/pooling/pooling.h"

namespace torch_tpu {
namespace {

// Computes the start and length of each interval along a given dimension based
// on the input and output sizes. `max_k` is the maximum length of any interval.
//
// The interval is defined as [start, end):
// start = floor(i * InputSize / OutputSize)
// end   = ceil((i + 1) * InputSize / OutputSize)
// length = end - start
void ComputeAdaptiveIndices(const int64_t in_size, const int64_t out_size,
                            Dimensions& starts, Dimensions& lengths,
                            int64_t& max_k) {
  for (int64_t i = 0; i < out_size; ++i) {
    const int64_t start = std::floor((double)i * in_size / out_size);
    const int64_t end = std::ceil((double)(i + 1) * in_size / out_size);

    starts[i] = start;
    lengths[i] = end - start;
    if (lengths[i] > max_k) {
      max_k = lengths[i];
    }
  }
}

// Helper: Create a 1D [N] int64 constant tensor.
absl::StatusOr<mlir::MlirOp> CreateI64Const(mlir::MlirBuilder& builder,
                                            const Dimensions& data) {
  auto type = mlir::RankedTensorType::get({static_cast<int64_t>(data.size())},
                                          builder.getOpBuilder().getI64Type());

  return mlir::stablehlo::Constant(
      builder, mlir::makeConstant(llvm::ArrayRef<int64_t>(data), type));
}

// Builds a 1D gather pooling operation for a given dimension using the
// Gather + Mask strategy. It includes padding the input tensor, preparing
// gather indices, masking, and summing along the reduced dimension.
//
// 1. Compute the start and length of each interval along the dimension.

// 2. Padding: Pad the input tensor to the maximum length along the dimension,
//    preventing out-of-bound access during Gather.
//
// 3. Gather: We extract `out_size` windows, where each window has size `max_k`.
//    By configuring `offset_dims`, it expands the result shape into
//    [..., out_size, max_k, ...]:
//    - `out_size` takes the position of the original `Dim_In`.
//    - `max_k` is inserted immediately after `out_size`.
//
// 4. Mask: Use a boolean mask (Iota < Lengths) to zero out padding elements.
//
// 5. Reduce: Sum along the `max_k` dimension. This collapses the 5D tensor back
//    to 4D, resulting in the shape [..., out_size, ...].
//
// 6. Div: divide the sum by the length of each window to get the average.
absl::StatusOr<mlir::MlirOp> BuildGatherPool1D(mlir::MlirBuilder& builder,
                                               mlir::MlirOp input,
                                               const int64_t dim,
                                               const int64_t in_size,
                                               const int64_t out_size) {
  auto input_type = GetTensorTypeOrDie(input);
  Dimensions input_shape = CopyIntVector(input_type.getShape());
  auto element_type = input_type.getElementType();
  const int64_t dim_size = input_type.getRank();

  // 1. Compute the start and length of each interval along the dimension.
  Dimensions starts(out_size);
  Dimensions lengths(out_size);
  int64_t max_k = 0;
  ComputeAdaptiveIndices(in_size, out_size, starts, lengths, max_k);

  // 2. Pad the input tensor to the maximum length along the dimension,
  // preventing out-of-bound access during Gather
  Dimensions pad_sizes(dim_size, 0);
  pad_sizes[dim] = max_k;
  auto zero = MakeScalarConstant(builder, 0.0, element_type);
  auto padded_input =
      mlir::stablehlo::Pad(input, zero,
                           /*edge_padding_low=*/Dimensions(dim_size, 0),
                           /*edge_padding_high=*/pad_sizes,
                           /*interior_padding=*/Dimensions(dim_size, 0));

  // 3. Prepare indices and execute Gather operation
  // start_indices [out_size, 1]: gather 'out_size' elements, each element is a
  // vector of size 1 indicating the start index
  TT_ASSIGN_OR_RETURN(auto start_const, CreateI64Const(builder, starts));
  auto start_indices = mlir::stablehlo::Reshape(start_const, {out_size, 1});

  // slice_sizes [..., max_k, ...]: extract window of size 'max_k'
  auto slice_sizes = input_shape;
  slice_sizes[dim] = max_k;

  // offset_dims: dimensions in the output that come from the input slice data.
  // We want the result layout to be [..., out_size, max_k, ...].
  // 'out_size' takes the place of original dim, and 'max_k' is inserted behind
  Dimensions offset_dims;
  for (int i = 0; i < dim_size; ++i) {
    if (i < dim) {
      offset_dims.push_back(i);
    } else {
      offset_dims.push_back(i + 1);
    }
  }

  auto dim_numbers = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      &input.getContext(),
      /*offset_dims=*/offset_dims,
      /*collapsed_slice_dims=*/{},  // keep the window dim (max_k)
      /*operand_batching_dims=*/{},
      /*start_indices_batching_dims=*/{},
      /*start_index_map=*/{dim},  // indices map to input's dim
      /*index_vector_dim=*/1      // last dim of start_indices
  );

  // Gather result shape: [..., out_size, max_k, ...]
  auto gather_result_shape = input_shape;
  gather_result_shape[dim] = out_size;
  gather_result_shape.insert(gather_result_shape.begin() + dim + 1, max_k);
  auto gather_result_type =
      mlir::RankedTensorType::get(gather_result_shape, element_type);
  auto gathered = mlir::stablehlo::Gather(padded_input, start_indices,
                                          dim_numbers, slice_sizes);

  // 4. Masking & Select: zero out the padded elements
  const int64_t reduced_dim_index = dim + 1;
  auto mask_shape_type = mlir::RankedTensorType::get(
      gather_result_shape, builder.getOpBuilder().getI64Type());

  // Iota: [max_k] -> Broadcast to [..., max_k, ...] along `reduced_dim_index`
  auto iota_type = makeTensorType(builder.getContext(), {max_k},
                                  builder.getOpBuilder().getI64Type());
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  auto iota_bcast = mlir::stablehlo::BroadcastInDim(mask_shape_type, iota,
                                                    {reduced_dim_index});

  // Lengths: [out_size] -> Broadcast to [..., out_size, ...] along `dim`
  TT_ASSIGN_OR_RETURN(auto lengths_const, CreateI64Const(builder, lengths));
  auto lengths_bcast =
      mlir::stablehlo::BroadcastInDim(mask_shape_type, lengths_const, {dim});

  // if (iota < lengths) then keep the gathered value, else use zero
  auto mask = mlir::stablehlo::Compare(
      iota_bcast, lengths_bcast, mlir::stablehlo::ComparisonDirection::LT);
  auto zero_bcast =
      mlir::stablehlo::BroadcastInDim(gather_result_type, zero, {});
  auto masked_gathered = mlir::stablehlo::Select(mask, gathered, zero_bcast);

  // 5. Reduce: sum along the reduced dimension
  // This collapses the tensor rank back (removes `reduced_dim_index`).
  auto sum_reduce_builder = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };
  mlir::MlirOp reduce_op =
      mlir::stablehlo::Reduce(builder, masked_gathered, zero,
                              sum_reduce_builder, {reduced_dim_index})[0];

  // 6. Div: it requires operands to have the same element type, so we
  // must convert the lengths to element_type before division.
  auto lengths_converted =
      mlir::stablehlo::ConvertElementType(lengths_const, element_type);
  auto reduce_op_type = GetTensorTypeOrDie(reduce_op);
  auto lengths_float_bcast =
      mlir::stablehlo::BroadcastInDim(reduce_op_type, lengths_converted, {dim});

  return mlir::stablehlo::Div(reduce_op, lengths_float_bcast);
}

// Implements the general case of 2D adaptive average pooling using separate 1D
// gather pooling operations for each spatial dimension.
//
// By applying `BuildGatherPool1D` sequentially, it avoids O(H*W) complexity and
// keeps the graph size constant.
absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool2dShlo(
    mlir::MlirOp input_op, const int64_t in_h, const int64_t in_w,
    const int64_t out_h, const int64_t out_w, const int64_t spatial_dim_count) {
  // Create a batch input
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input_op, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirBuilder& builder = batch_input.getBuilder();

  auto input_type = GetTensorTypeOrDie(batch_input);
  const int64_t h_dim = input_type.getRank() - 2;
  const int64_t w_dim = input_type.getRank() - 1;

  // Build the height gather pool operation
  TT_ASSIGN_OR_RETURN(
      auto h_op, BuildGatherPool1D(builder, batch_input, h_dim, in_h, out_h));

  // Build the width gather pool operation
  auto h_type = GetTensorTypeOrDie(h_op);
  auto h_shape = h_type.getShape();
  TT_ASSIGN_OR_RETURN(auto w_op, BuildGatherPool1D(builder, h_op, w_dim,
                                                   h_shape[w_dim], out_w));

  return RemoveTrivialBatch(w_op, batch_input_info.original_dim_size,
                            spatial_dim_count);
}

// Implements the general case of 3D adaptive average pooling using separate 1D
// gather pooling operations for each spatial dimension.
absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool3dShlo(
    mlir::MlirOp input_op, const int64_t in_d, const int64_t in_h,
    const int64_t in_w, const int64_t out_d, const int64_t out_h,
    const int64_t out_w, const int64_t spatial_dim_count) {
  // Create a batch input
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input_op, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirBuilder& builder = batch_input.getBuilder();

  auto input_type = GetTensorTypeOrDie(batch_input);
  const int64_t num_dims = input_type.getRank();
  const int64_t d_dim = num_dims - 3;
  const int64_t h_dim = num_dims - 2;
  const int64_t w_dim = num_dims - 1;

  // Build the depth gather pool operation
  TT_ASSIGN_OR_RETURN(
      auto d_op, BuildGatherPool1D(builder, batch_input, d_dim, in_d, out_d));

  // Build the height gather pool operation
  auto d_type = GetTensorTypeOrDie(d_op);
  auto d_shape = d_type.getShape();
  TT_ASSIGN_OR_RETURN(auto h_op, BuildGatherPool1D(builder, d_op, h_dim,
                                                   d_shape[h_dim], out_h));

  // Build the width gather pool operation
  auto h_type = GetTensorTypeOrDie(h_op);
  auto h_shape = h_type.getShape();
  TT_ASSIGN_OR_RETURN(auto w_op, BuildGatherPool1D(builder, h_op, w_dim,
                                                   h_shape[w_dim], out_w));

  return RemoveTrivialBatch(w_op, batch_input_info.original_dim_size,
                            spatial_dim_count);
}

// Builds a 1D scatter-add operation (Backward of GatherPool).
//
// Logic:
// 1. Div: grad_Output / lengths (distribute gradients)
// 2. Expand: Broadcast gradients to window size [..., out_size, max_k, ...]
// 3. Mask: Zero out gradients corresponding to padding
// 4. Indexing: Compute target indices [starts + iota]
// 5. Flatten: Merge out_size and max_k for simpler scatter config
// 6. Scatter: Accumulate gradients into padded grad_input
// 7. Slice: Remove extra padding
absl::StatusOr<mlir::MlirOp> BuildScatterAdd1D(mlir::MlirBuilder& builder,
                                               mlir::MlirOp grad_output,
                                               const int64_t dim,
                                               const int64_t in_size,
                                               const int64_t out_size) {
  auto grad_type = GetTensorTypeOrDie(grad_output);
  Dimensions grad_output_shape = CopyIntVector(grad_type.getShape());
  auto element_type = grad_type.getElementType();
  const int64_t dim_size = grad_type.getRank();
  auto i64_type = builder.getOpBuilder().getI64Type();

  // 1. Compute the start and length of each interval along the dimension.
  Dimensions starts(out_size), lengths(out_size);
  int64_t max_k = 0;
  ComputeAdaptiveIndices(in_size, out_size, starts, lengths, max_k);

  // 2. Distribute gradients: grad_output / lengths
  // Broadcast lengths [out_size] -> [..., out_size, ...] along `dim`
  TT_ASSIGN_OR_RETURN(auto lengths_const, CreateI64Const(builder, lengths));
  // div requires operands to have the same element type
  auto lengths_converted =
      mlir::stablehlo::ConvertElementType(lengths_const, element_type);
  auto lengths_bcast =
      mlir::stablehlo::BroadcastInDim(grad_type, lengths_converted, {dim});
  auto grad_output_scaled = mlir::stablehlo::Div(grad_output, lengths_bcast);

  // 3. Broadcast scaled grad_output to window size [..., out_size, max_k, ...]
  // 'out_size' takes the place of original dim, and 'max_k' is inserted behind
  auto expanded_shape = grad_output_shape;
  expanded_shape.insert(expanded_shape.begin() + dim + 1, max_k);
  auto expanded_type =
      mlir::RankedTensorType::get(expanded_shape, element_type);
  Dimensions broadcast_dims;
  for (int i = 0; i < dim_size; ++i) {
    if (i <= dim) {
      broadcast_dims.push_back(i);
    } else {
      broadcast_dims.push_back(i + 1);
    }
  }
  auto grad_expanded = mlir::stablehlo::BroadcastInDim(
      expanded_type, grad_output_scaled, broadcast_dims);

  // 4. Mask: zero out gradients corresponding to padding
  const int64_t reduced_dim_index = dim + 1;
  auto mask_type = mlir::RankedTensorType::get(expanded_shape, i64_type);

  // iota: [max_k] -> broadcast to [..., max_k, ...] along `reduced_dim_index`
  auto iota_type = makeTensorType(builder.getContext(), {max_k}, i64_type);
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  auto iota_bcast =
      mlir::stablehlo::BroadcastInDim(mask_type, iota, {reduced_dim_index});

  auto lengths_bcast_mask =
      mlir::stablehlo::BroadcastInDim(mask_type, lengths_const, {dim});

  // if (iota < lengths) then keep the gathered value, else use zero
  auto mask = mlir::stablehlo::Compare(
      iota_bcast, lengths_bcast_mask, mlir::stablehlo::ComparisonDirection::LT);
  auto zero = MakeScalarConstant(builder, 0.0, element_type);
  auto zero_bcast = mlir::stablehlo::BroadcastInDim(expanded_type, zero, {});
  auto grad_masked = mlir::stablehlo::Select(mask, grad_expanded, zero_bcast);

  // 5. Flatten grad_updates to [..., out_size * max_k, ...]
  // This converts the update tensor into a list of individual gradient
  // contributions.
  auto flattened_shape = CopyIntVector(grad_type.getShape());
  flattened_shape[dim] = out_size * max_k;
  auto grad_updates = mlir::stablehlo::Reshape(grad_masked, flattened_shape);

  // 6. Compute scatter indices by broadcasting starts and iota via Add.
  // Absolute_Index = Window_Start_Index + Offset_In_Window

  // broadcast both `starts` and `iota` to shape [out_size, max_k]
  TT_ASSIGN_OR_RETURN(auto starts_const, CreateI64Const(builder, starts));
  auto starts_2d = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({out_size, max_k}, i64_type), starts_const,
      {0});
  auto iota_2d = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({out_size, max_k}, i64_type), iota, {1});
  auto full_indices = mlir::stablehlo::Add(starts_2d, iota_2d);

  // flatten indices to [out_size * max_k, 1] to match `grad_updates`
  auto scatter_indices =
      mlir::stablehlo::Reshape(full_indices, {out_size * max_k, 1});

  // 7. Configure Scatter
  // Target shape: [..., in_size + max_k, ...] for out-of-bounds handling
  auto target_shape = CopyIntVector(grad_type.getShape());
  target_shape[dim] = in_size + max_k;
  auto target_type = mlir::RankedTensorType::get(target_shape, element_type);
  auto target_init = mlir::stablehlo::BroadcastInDim(target_type, zero, {});

  Dimensions update_window_dims;
  for (int i = 0; i < dim_size; ++i) {
    if (i != dim) {
      update_window_dims.push_back(i);
    }
  }

  auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      &builder.getContext(),
      /*update_window_dims=*/update_window_dims,
      /*inserted_window_dims=*/{dim},
      /*input_batching_dims=*/{},
      /*scatter_indices_batching_dims=*/{},
      /*scatter_dims_to_operand_dims=*/{dim},
      /*index_vector_dim=*/1);

  auto scatter_body = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };

  auto scatter_op = mlir::stablehlo::Scatter(
      /*operand=*/target_init,
      /*scatter_indices=*/scatter_indices,
      /*updates=*/grad_updates,
      /*update_computation=*/scatter_body,
      /*dimension_numbers=*/scatter_dims);

  // 8. Slice back to original in_size
  Dimensions start_indices_slice(dim_size, 0);
  auto limit_indices_slice = target_shape;
  limit_indices_slice[dim] = in_size;
  Dimensions strides(dim_size, 1);

  return mlir::stablehlo::Slice(scatter_op[0], start_indices_slice,
                                limit_indices_slice, strides);
}

absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool2dBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input) {
  const int64_t spatial_dim_count = 2;
  mlir::MlirBuilder& builder = grad_output.getBuilder();

  // Create batch inputs for both input and grad_output
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_info,
                      CreateBatchInput(grad_output, spatial_dim_count));

  auto input_type = GetTensorTypeOrDie(batch_input_info.batch_input);
  auto grad_output_type = GetTensorTypeOrDie(batch_grad_info.batch_input);

  const int64_t h_dim = input_type.getRank() - 2;
  const int64_t w_dim = input_type.getRank() - 1;
  const int64_t in_h = input_type.getShape()[h_dim];
  const int64_t in_w = input_type.getShape()[w_dim];
  const int64_t out_h = grad_output_type.getShape()[h_dim];
  const int64_t out_w = grad_output_type.getShape()[w_dim];

  // input: [..., out_h, out_w] -> output: [..., out_h, in_w]
  TT_ASSIGN_OR_RETURN(
      auto w_op, BuildScatterAdd1D(builder, batch_grad_info.batch_input, w_dim,
                                   in_w, out_w));

  // input: [..., out_h, in_w] -> output: [..., in_h, in_w]
  TT_ASSIGN_OR_RETURN(auto h_op,
                      BuildScatterAdd1D(builder, w_op, h_dim, in_h, out_h));

  return RemoveTrivialBatch(h_op, batch_input_info.original_dim_size,
                            spatial_dim_count);
}

absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool3dBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input) {
  const int64_t spatial_dim_count = 3;
  mlir::MlirBuilder& builder = grad_output.getBuilder();

  // Create batch inputs for both input and grad_output
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_info,
                      CreateBatchInput(grad_output, spatial_dim_count));

  auto input_type = GetTensorTypeOrDie(batch_input_info.batch_input);
  auto grad_output_type = GetTensorTypeOrDie(batch_grad_info.batch_input);

  const int64_t d_dim = input_type.getRank() - 3;
  const int64_t h_dim = input_type.getRank() - 2;
  const int64_t w_dim = input_type.getRank() - 1;
  const int64_t in_d = input_type.getShape()[d_dim];
  const int64_t in_h = input_type.getShape()[h_dim];
  const int64_t in_w = input_type.getShape()[w_dim];
  const int64_t out_d = grad_output_type.getShape()[d_dim];
  const int64_t out_h = grad_output_type.getShape()[h_dim];
  const int64_t out_w = grad_output_type.getShape()[w_dim];

  // input: [..., out_d, out_h, out_w] -> output: [..., out_d, out_h, in_w]
  TT_ASSIGN_OR_RETURN(
      auto w_op, BuildScatterAdd1D(builder, batch_grad_info.batch_input, w_dim,
                                   in_w, out_w));

  // input: [..., out_d, out_h, in_w] -> output: [..., out_d, in_h, in_w]
  TT_ASSIGN_OR_RETURN(auto h_op,
                      BuildScatterAdd1D(builder, w_op, h_dim, in_h, out_h));

  // input: [..., out_d, in_h, in_w] -> output: [..., in_d, in_h, in_w]
  TT_ASSIGN_OR_RETURN(auto d_op,
                      BuildScatterAdd1D(builder, h_op, d_dim, in_d, out_d));

  return RemoveTrivialBatch(d_op, batch_input_info.original_dim_size,
                            spatial_dim_count);
}
}  // namespace

at::Tensor& AtenAdaptiveAvgPool2dOut(const at::Tensor& self,
                                     c10::SymIntArrayRef output_size,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kAdaptiveAvgPool2dOut, param_keys, (self, output_size, out), {
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Byte &&
                           self.scalar_type() != at::ScalarType::Char &&
                           self.scalar_type() != at::ScalarType::Short &&
                           self.scalar_type() != at::ScalarType::Int &&
                           self.scalar_type() != at::ScalarType::Long &&
                           self.scalar_type() != at::ScalarType::ComplexFloat,
                       error::kInvalidArgument)
            << "not yet implemented for uint8, int8, int16, int32, int64,"
            << " and complex64 dtypes, got "
            << torch_tpu::ToString(self.scalar_type());

        const int64_t spatial_dim_count = 2;
        auto num_dims = self.dim();
        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        TT_CHECK_THROW(num_dims == spatial_dim_count + 1 ||
                           num_dims == spatial_dim_count + 2,
                       error::kInvalidArgument)
            << "input must be a " << spatial_dim_count + 1 << "-D or "
            << spatial_dim_count + 2 << "-D tensor"
            << ", got " << num_dims << "-D tensor";

        const int64_t in_h = self.size(num_dims - 2);
        const int64_t in_w = self.size(num_dims - 1);

        // The output size is either a single integer H or a tuple (H, W)
        const int64_t out_h = output_size[0].expect_int();
        int64_t out_w = out_h;
        if (output_size.size() > 1) {
          out_w = output_size[1].expect_int();
        }

        // If the input size is divisible by the output size, the adaptive pool
        // is mathematically equivalent to a standard average pool with:
        // stride = input / output
        // kernel = input - (output - 1) * stride
        if (in_h >= out_h && in_w >= out_w && in_h % out_h == 0 &&
            in_w % out_w == 0) {
          auto stride_h = in_h / out_h;
          auto stride_w = in_w / out_w;
          auto kernel_h = in_h - (out_h - 1) * stride_h;
          auto kernel_w = in_w - (out_w - 1) * stride_w;

          Dimensions kernel_size = {kernel_h, kernel_w};
          Dimensions stride = {stride_h, stride_w};
          Dimensions padding = {0, 0};

          TT_ASSIGN_OR_THROW(
              auto result,
              BuildAvgPoolOutNd(self, kernel_size, stride, padding,
                                /*ceil_mode=*/false,
                                /*count_include_pad=*/true,
                                /*divisor_override=*/std::nullopt, out,
                                spatial_dim_count, std::move(param_keys)));
          return out;
        } else {
          // Otherwise, we need to use the general adaptive pooling
          auto op_builder =
              [in_h, in_w, out_h,
               out_w](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
            return BuildAdaptiveAvgPool2dShlo(input, in_h, in_w, out_h, out_w,
                                              spatial_dim_count);
          };

          TT_ASSIGN_OR_THROW(
              auto result,
              DispatchOp<1>(std::move(op_builder), self,
                            // Override the OpName passed to TT_KERNEL() as we
                            // need a different computation than the one in the
                            // true branch above.
                            {.op_name = OpName::kAdaptiveAvgPool2d,
                             .out_dtype = output_dtype,
                             .out_dims = CopyIntVector(out.sizes()),
                             .op_param_cache_keys = std::move(param_keys)}));

          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
          return out;
        }
      });
}

at::Tensor AtenAdaptiveAvgPool2d(const at::Tensor& self,
                                 c10::SymIntArrayRef output_size) {
  TT_KERNEL(OpName::kAdaptiveAvgPool2d, _,
            (self, IgnoreInCacheKey(output_size, "Legacy usage")), {
              auto num_dims = self.dim();

              // The output size is either a single integer H or a tuple (H, W)
              const int64_t out_h = output_size[0].expect_int();
              int64_t out_w = out_h;
              if (output_size.size() > 1) {
                out_w = output_size[1].expect_int();
              }

              auto out_shape = CopyIntVector(self.sizes());
              out_shape[num_dims - 2] = out_h;
              out_shape[num_dims - 1] = out_w;

              at::Tensor out = at::empty(out_shape, self.options());

              AtenAdaptiveAvgPool2dOut(self, output_size, out);
              return out;
            });
}

at::Tensor& AtenAdaptiveAvgPool3dOut(const at::Tensor& self,
                                     c10::SymIntArrayRef output_size,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kAdaptiveAvgPool3dOut, param_keys, (self, output_size, out), {
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool &&
                           self.scalar_type() != at::ScalarType::Byte &&
                           self.scalar_type() != at::ScalarType::Char &&
                           self.scalar_type() != at::ScalarType::Short &&
                           self.scalar_type() != at::ScalarType::Int &&
                           self.scalar_type() != at::ScalarType::Long &&
                           self.scalar_type() != at::ScalarType::ComplexFloat,
                       error::kInvalidArgument)
            << "not yet implemented for bool, uint8, int8, int16, int32, int64,"
            << " and complex64 dtypes, got "
            << torch_tpu::ToString(self.scalar_type());

        const int64_t spatial_dim_count = 3;
        auto num_dims = self.dim();
        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        TT_CHECK_THROW(num_dims == spatial_dim_count + 1 ||
                           num_dims == spatial_dim_count + 2,
                       error::kInvalidArgument)
            << "input must be a " << spatial_dim_count + 1 << "-D or "
            << spatial_dim_count + 2 << "-D tensor"
            << ", got " << num_dims << "-D tensor";

        const int64_t in_d = self.size(num_dims - 3);
        const int64_t in_h = self.size(num_dims - 2);
        const int64_t in_w = self.size(num_dims - 1);

        // The output size is either a single integer or a triple-integer tuple
        const int64_t out_d = output_size[0].expect_int();
        int64_t out_h = out_d;
        int64_t out_w = out_d;
        if (output_size.size() > 1) {
          out_h = output_size[1].expect_int();
          out_w = output_size[2].expect_int();
        }

        // If the input size is divisible by the output size, the adaptive pool
        // is mathematically equivalent to a standard average pool with:
        // Stride = Input / Output
        // Kernel = Input - (Output - 1) * Stride
        if (in_d >= out_d && in_h >= out_h && in_w >= out_w &&
            in_d % out_d == 0 && in_h % out_h == 0 && in_w % out_w == 0) {
          auto stride_d = in_d / out_d;
          auto stride_h = in_h / out_h;
          auto stride_w = in_w / out_w;
          auto kernel_d = in_d - (out_d - 1) * stride_d;
          auto kernel_h = in_h - (out_h - 1) * stride_h;
          auto kernel_w = in_w - (out_w - 1) * stride_w;

          Dimensions kernel_size = {kernel_d, kernel_h, kernel_w};
          Dimensions stride = {stride_d, stride_h, stride_w};
          Dimensions padding = {0, 0, 0};

          TT_ASSIGN_OR_THROW(
              auto result,
              BuildAvgPoolOutNd(self, kernel_size, stride, padding,
                                /*ceil_mode=*/false,
                                /*count_include_pad=*/true,
                                /*divisor_override=*/std::nullopt, out,
                                spatial_dim_count, std::move(param_keys)));
          return out;
        } else {
          // Otherwise, we need to use the general adaptive pooling
          auto op_builder =
              [in_d, in_h, in_w, out_d, out_h,
               out_w](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
            return BuildAdaptiveAvgPool3dShlo(input, in_d, in_h, in_w, out_d,
                                              out_h, out_w, spatial_dim_count);
          };

          TT_ASSIGN_OR_THROW(
              auto result,
              DispatchOp<1>(std::move(op_builder), self,
                            // Override the OpName passed to TT_KERNEL() as we
                            // need a different computation than the one in the
                            // true branch above.
                            {.op_name = OpName::kAdaptiveAvgPool3dOut,
                             .out_dtype = output_dtype,
                             .out_dims = CopyIntVector(out.sizes()),
                             .op_param_cache_keys = std::move(param_keys)}));

          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
          return out;
        }
      });
}

at::Tensor AtenAdaptiveAvgPool3d(const at::Tensor& self,
                                 c10::SymIntArrayRef output_size) {
  TT_KERNEL(OpName::kAdaptiveAvgPool3d, _,
            (self, IgnoreInCacheKey(output_size, "Legacy usage")), {
              auto num_dims = self.dim();

              // The output size is either a single integer or a triple-integer
              // tuple
              const int64_t out_d = output_size[0].expect_int();
              int64_t out_h = out_d;
              int64_t out_w = out_d;
              if (output_size.size() > 1) {
                out_h = output_size[1].expect_int();
                out_w = output_size[2].expect_int();
              }

              auto out_shape = CopyIntVector(self.sizes());
              out_shape[num_dims - 3] = out_d;
              out_shape[num_dims - 2] = out_h;
              out_shape[num_dims - 1] = out_w;

              at::Tensor out = at::empty(out_shape, self.options());

              AtenAdaptiveAvgPool3dOut(self, output_size, out);
              return out;
            });
}

at::Tensor AtenAdaptiveAvgPool2dBackward(const at::Tensor& grad_output,
                                         const at::Tensor& self) {
  TT_KERNEL(
      OpName::kAdaptiveAvgPool2dBackward, param_keys, (grad_output, self), {
        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildAdaptiveAvgPool2dBackwardShlo(inputs[0], inputs[1]);
        };

        at::Tensor grad_input = at::empty(self.sizes(), self.options());
        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<2>(std::move(op_builder), {grad_output, self},
                           {.out_dtype = output_dtype,
                            .out_dims = CopyIntVector(self.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenAdaptiveAvgPool3dBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kAdaptiveAvgPool3dBackwardGradInput, param_keys,
      (grad_output, self, grad_input), {
        TT_ASSIGN_OR_THROW(
            const auto output_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildAdaptiveAvgPool3dBackwardShlo(inputs[0], inputs[1]);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<2>(std::move(op_builder), {grad_output, self},
                           {.out_dtype = output_dtype,
                            .out_dims = CopyIntVector(self.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}

at::Tensor AtenAdaptiveAvgPool3dBackward(const at::Tensor& grad_output,
                                         const at::Tensor& self) {
  TT_KERNEL(OpName::kAdaptiveAvgPool3dBackward, _, (grad_output, self), {
    auto grad_input = at::empty(self.sizes(), self.options());
    AtenAdaptiveAvgPool3dBackwardGradInput(grad_output, self, grad_input);
    return grad_input;
  });
}
}  // namespace torch_tpu
