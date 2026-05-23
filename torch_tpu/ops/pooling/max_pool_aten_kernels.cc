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

#include "torch_tpu/ops/pooling/max_pool_aten_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/max_pool2d_with_indices.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/csrc/autograd/function.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/pooling/pooling.h"

namespace torch_tpu {

namespace {

// Creates an index tensor containing the local flattened index for each
// spatial element.
mlir::MlirOp CreatePoolIndicesIota(const mlir::RankedTensorType& input_shape,
                                   mlir::MlirBuilder& builder,
                                   mlir::Type element_type) {
  // Compute the total number of spatial elements
  int64_t spatial_input_elements = std::accumulate(
      input_shape.getShape().begin() + 2, input_shape.getShape().end(),
      int64_t{1}, std::multiplies<int64_t>());

  // Create an flattened iota tensor of shape {spatial_input_elements}
  auto iota_type =
      mlir::RankedTensorType::get({spatial_input_elements}, element_type);
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type, 0);

  // Reshape the iota tensor to spatial dimensions
  llvm::SmallVector<int64_t> spatial_dims;
  for (int64_t i = 2; i < input_shape.getRank(); ++i) {
    spatial_dims.push_back(input_shape.getDimSize(i));
  }

  mlir::MlirOp reshaped_iota = mlir::stablehlo::Reshape(iota, spatial_dims);

  // Broadcast spatial dimensions to the same rank as the input tensor
  llvm::SmallVector<int64_t> broadcast_dims;
  for (size_t i = 0; i < spatial_dims.size(); ++i) {
    broadcast_dims.push_back(2 + i);
  }
  llvm::SmallVector<int64_t> target_shape(input_shape.getShape().begin(),
                                          input_shape.getShape().end());
  mlir::RankedTensorType target_tensor_type =
      mlir::RankedTensorType::get(target_shape, element_type);
  mlir::MlirOp iota_broadcast = mlir::stablehlo::BroadcastInDim(
      target_tensor_type, reshaped_iota, broadcast_dims);

  return iota_broadcast;
}

absl::StatusOr<std::vector<mlir::MlirOp>> ComputeMaxPoolWithIndices(
    const BatchInput& batch_input_info,
    const ReduceWindowAttributes& reduce_window_attributes,
    const int64_t spatial_dim_count) {
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  const int64_t original_dim_size = batch_input_info.original_dim_size;

  mlir::MlirBuilder& builder = batch_input.getBuilder();
  const mlir::RankedTensorType input_shape = GetTensorTypeOrDie(batch_input);
  const mlir::Type element_type = input_shape.getElementType();

  // TODO: Support int64 indices to match CPU behavior.
  // Currently TPU ReduceWindow does not support int64 reduction yet.
  // When the input tensor has > 2^31-1 spatial elements, we cannot silently
  // return incorrect data. Here we fail with a clear message.
  //
  // Drop the first 2 dimensions (Batch and Channel) because max_pool indices
  // are relative to the spatial dimensions only.
  auto shape = input_shape.getShape();
  TT_ASSIGN_OR_RETURN(int64_t spatial_input_elements,
                      NumElements(shape.drop_front(2)));

  TT_RET_CHECK(spatial_input_elements <= std::numeric_limits<int32_t>::max(),
               error::kUnimplemented)
      << "tpu doesn't support max_pool2d_with_indices on inputs with more than "
      << std::numeric_limits<int32_t>::max()
      << " spatial elements due to int32 indices limitation for now,"
      << " got " << spatial_input_elements;

  const mlir::Type index_type = builder.getOpBuilder().getI32Type();

  // Create global flattened indices
  auto iota = CreatePoolIndicesIota(input_shape, builder, index_type);

  // Create initial values for values and indices
  mlir::Attribute val_attr =
      GetMinFiniteValueAttr(element_type, builder.getOpBuilder());
  mlir::DenseElementsAttr init_value_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, element_type), val_attr);
  mlir::MlirOp init_value = mlir::stablehlo::Constant(builder, init_value_attr);

  const int64_t max_idx_val = std::numeric_limits<int32_t>::max();
  mlir::MlirOp init_invalid_iota =
      MakeScalarConstant(builder, max_idx_val, index_type);

  const mlir::RankedTensorType batch_input_shape =
      GetTensorTypeOrDie(batch_input);
  const mlir::RankedTensorType iota_shape = GetTensorTypeOrDie(iota);
  ABSL_CHECK_EQ(  // CRASH_OK
      batch_input_shape.getShape(), iota_shape.getShape())
      << "inputs should have the same shape for varidic ReduceWindow op"
      << ", got batch_input_shape " << batch_input_shape.getShape().size()
      << " and iota_shape " << iota_shape.getShape().size();

  // Perform variadic stablehlo::ReduceWindow on {batch_input, iota}
  auto reduce_results = mlir::stablehlo::ReduceWindow(
      builder,
      /*inputs=*/{batch_input, iota},
      /*init_values=*/{init_value, init_invalid_iota},
      /*body=*/
      [&](mlir::RegionBuilder& body) {
        mlir::stablehlo::buildMaxAndArgmaxBody(
            element_type, index_type, body.getRegion(), body.getOpBuilder());
      },
      reduce_window_attributes.window_dimensions,
      reduce_window_attributes.window_strides,
      reduce_window_attributes.base_dilations,
      reduce_window_attributes.window_dilations,
      reduce_window_attributes.padding);

  mlir::MlirOp batch_result = reduce_results[0];
  mlir::MlirOp batch_indices = reduce_results[1];

  mlir::MlirOp final_output =
      RemoveTrivialBatch(batch_result, original_dim_size, spatial_dim_count);
  mlir::MlirOp final_indices =
      RemoveTrivialBatch(batch_indices, original_dim_size, spatial_dim_count);

  return std::vector<mlir::MlirOp>{final_output, final_indices};
}

absl::StatusOr<mlir::MlirOp> ComputeMaxPool(
    const BatchInput& batch_input_info,
    const ReduceWindowAttributes& reduce_window_attributes,
    const int64_t spatial_dim_count) {
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  const int64_t original_dim_size = batch_input_info.original_dim_size;

  mlir::MlirBuilder& builder = batch_input.getBuilder();
  const mlir::RankedTensorType input_shape = GetTensorTypeOrDie(batch_input);
  const mlir::Type element_type = input_shape.getElementType();

  mlir::Attribute val_attr =
      GetMinFiniteValueAttr(element_type, builder.getOpBuilder());
  mlir::DenseElementsAttr init_value_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, element_type), val_attr);
  mlir::MlirOp init_value = mlir::stablehlo::Constant(builder, init_value_attr);

  auto reduce_results = mlir::stablehlo::ReduceWindow(
      builder,
      /*inputs=*/{batch_input},
      /*init_values=*/{init_value},
      /*body=*/
      [&](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
            element_type, rb.getRegion(), rb.getOpBuilder());
      },
      reduce_window_attributes.window_dimensions,
      reduce_window_attributes.window_strides,
      reduce_window_attributes.base_dilations,
      reduce_window_attributes.window_dilations,
      reduce_window_attributes.padding);

  mlir::MlirOp batch_result = reduce_results[0];

  mlir::MlirOp final_output =
      RemoveTrivialBatch(batch_result, original_dim_size, spatial_dim_count);

  return final_output;
}

// Calculates the size of a single output dimension (height, width, or depth)
// for max_pool3d, including support for floor and ceil modes.
//
// Inputs:
//   in: the size of the input dimension
//   i: the index of the dimension
//   kernel_size, stride, padding, dilation: the attributes of max_pool3d
//   ceil_mode: indicates whether the pooling uses the floor or ceil mode
//
// Returns:
//   int64_t: The calculated size of the single output dimension.
//
// Calculation Formula (Standard):
//   Output Size = floor/ceil((Input Size + 2 * Padding - Dilation * (Kernel
//   Size - 1) - 1) / Stride) + 1
int64_t GetSingleDim(const int64_t in, const int64_t i,
                     at::IntArrayRef kernel_size, at::IntArrayRef stride,
                     at::IntArrayRef padding, at::IntArrayRef dilation,
                     const bool ceil_mode) {
  const int64_t k = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[i];
  const int64_t s =
      stride.empty() ? k : (stride.size() == 1 ? stride[0] : stride[i]);
  const int64_t p = padding.size() == 1 ? padding[0] : padding[i];
  const int64_t d = dilation.size() == 1 ? dilation[0] : dilation[i];
  const int64_t numerator = in + 2 * p - d * (k - 1) - 1;

  if (!ceil_mode) {
    return numerator / s + 1;
  }

  const int64_t out = (numerator + s - 1) / s + 1;
  if ((out - 1) * s >= in + p) {
    return out - 1;
  }
  return out;
}

// Builds the max_pool operations with indices and out tensors using
// StableHLO's ReduceWindowOp.
// ReduceWindowOp can take variadic number of tensors. It utilizes the tuple
// reduction pattern to compute both the maximum value and its corresponding
// global linear index in a single pass.
//
// The final result vector contains:
// {Max Values(tensor<N, C, (D_out,) H_out, W_out>),
//  Indices(tensor<N, C, (D_out,) H_out, W_out>)}
//
// How it works:
// 1. Batch Input Normalization (STEP 1): The input tensor is unified into one
// shape via StableHLO's ReshapeOp if it doesn't have a batch dimension.
// For 2d, the unified shape is (N, C, H, W), for 3d, it's (N, C, D, H, W).
//
// 2. Attribute Expansion (STEP 2): Attribute inputs can be empty / single
// int / a tuple of two ints, expand them to match the spatial_dim_count.
//
// 3. Padding Attribute (STEP 3): Compute the padding pairs given the
// ceil mode, and generate the padding attributes.
//
// 4. ReduceWindow Attributes (STEP 4): Compute the final reduce window
// attributes.
//
// 5. Compute Max Pool with Indices (STEP 5): Compute both max values and
// global flattened indices using a single ReduceWindowOp.
absl::StatusOr<std::vector<mlir::MlirOp>> BuildMaxPoolWithIndicesShlo(
    mlir::MlirOp input, const int64_t spatial_dim_count, Dimensions kernel_size,
    Dimensions stride, Dimensions padding, Dimensions dilation,
    const bool ceil_mode) {
  auto& builder = input.getBuilder();

  // STEP 1. Normalize batch input
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  const mlir::RankedTensorType input_shape =
      GetTensorTypeOrDie(batch_input_info.batch_input);
  auto total_num_dims = input_shape.getRank();
  const int64_t expected_dims = spatial_dim_count + 2;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=`CreateBatchInput()` function above
                 // catches this error first.
      total_num_dims == expected_dims, error::kInvalidArgument)
      << "expected input shape dim size == " << expected_dims << " , got "
      << total_num_dims;

  // STEP 2. Expand attributes to match the spatial_dim_count.
  // The parameters kernel_size, stride, padding, dilation can either be a
  // single int or a tuple of two ints.
  TT_ASSIGN_OR_RETURN(Dimensions kernel_size_attr,
                      ExpandAttribute(kernel_size, spatial_dim_count));
  Dimensions stride_attr;
  if (stride.empty()) {
    stride_attr = kernel_size_attr;
  } else {
    TT_ASSIGN_OR_RETURN(stride_attr,
                        ExpandAttribute(stride, spatial_dim_count));
  }
  Dimensions dilation_attr;
  if (dilation.empty()) {
    dilation_attr = Dimensions(spatial_dim_count, 1);
  } else {
    TT_ASSIGN_OR_RETURN(dilation_attr,
                        ExpandAttribute(dilation, spatial_dim_count));
  }
  Dimensions padding_attr;
  if (padding.empty()) {
    padding_attr = Dimensions(spatial_dim_count, 0);
  } else {
    TT_ASSIGN_OR_RETURN(padding_attr,
                        ExpandAttribute(padding, spatial_dim_count));
  }

  // STEP 3. Compute the padding pairs given the padding and ceil mode.
  auto ceil_padding_pairs =
      CeilModePadding(input_shape, kernel_size_attr, stride_attr, padding_attr,
                      dilation_attr, ceil_mode);

  // STEP 4. Compute the final reduce window attributes.
  ReduceWindowAttributes reduce_window_attributes = GetReduceWindowAttributes(
      builder, kernel_size_attr, stride_attr, dilation_attr, ceil_padding_pairs,
      spatial_dim_count, total_num_dims);

  // STEP 5. Compute both max values and global flattened indices
  // using a single ReduceWindow.
  return ComputeMaxPoolWithIndices(batch_input_info, reduce_window_attributes,
                                   spatial_dim_count);
}

absl::StatusOr<mlir::MlirOp> BuildMaxPoolShlo(
    mlir::MlirOp input, const int64_t spatial_dim_count, Dimensions kernel_size,
    Dimensions stride, Dimensions padding, Dimensions dilation,
    const bool ceil_mode) {
  auto& builder = input.getBuilder();

  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  const mlir::RankedTensorType input_shape =
      GetTensorTypeOrDie(batch_input_info.batch_input);
  auto total_num_dims = input_shape.getRank();
  const int64_t expected_dims = spatial_dim_count + 2;
  TT_RET_CHECK(total_num_dims == expected_dims, error::kInvalidArgument)
      << "expected input shape dim size == " << expected_dims << " , got "
      << total_num_dims;

  TT_ASSIGN_OR_RETURN(Dimensions kernel_size_attr,
                      ExpandAttribute(kernel_size, spatial_dim_count));
  Dimensions stride_attr;
  if (stride.empty()) {
    stride_attr = kernel_size_attr;
  } else {
    TT_ASSIGN_OR_RETURN(stride_attr,
                        ExpandAttribute(stride, spatial_dim_count));
  }
  Dimensions dilation_attr;
  if (dilation.empty()) {
    dilation_attr = Dimensions(spatial_dim_count, 1);
  } else {
    TT_ASSIGN_OR_RETURN(dilation_attr,
                        ExpandAttribute(dilation, spatial_dim_count));
  }
  Dimensions padding_attr;
  if (padding.empty()) {
    padding_attr = Dimensions(spatial_dim_count, 0);
  } else {
    TT_ASSIGN_OR_RETURN(padding_attr,
                        ExpandAttribute(padding, spatial_dim_count));
  }

  auto ceil_padding_pairs =
      CeilModePadding(input_shape, kernel_size_attr, stride_attr, padding_attr,
                      dilation_attr, ceil_mode);

  ReduceWindowAttributes reduce_window_attributes = GetReduceWindowAttributes(
      builder, kernel_size_attr, stride_attr, dilation_attr, ceil_padding_pairs,
      spatial_dim_count, total_num_dims);

  return ComputeMaxPool(batch_input_info, reduce_window_attributes,
                        spatial_dim_count);
}

absl::StatusOr<mlir::MlirOp> BuildMaxPoolBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input,
    const int64_t spatial_dim_count, Dimensions kernel_size, Dimensions stride,
    Dimensions padding, Dimensions dilation, const bool ceil_mode) {
  TT_RET_CHECK(std::all_of(dilation.begin(), dilation.end(),
                           std::bind_front(std::equal_to(), 1)),
               error::kInvalidArgument)
      << "MaxPool2dBackwards only supports trivial dilations.";

  auto& builder = grad_output.getBuilder();

  // STEP 1. Create batch inputs for both input and grad_output
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_output_info,
                      CreateBatchInput(grad_output, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirOp batch_grad_output = batch_grad_output_info.batch_input;

  const mlir::RankedTensorType input_shape = GetTensorTypeOrDie(batch_input);
  auto total_num_dims = input_shape.getRank();

  // STEP 2. Expand attributes
  TT_ASSIGN_OR_RETURN(Dimensions kernel_size_attr,
                      ExpandAttribute(kernel_size, spatial_dim_count));
  Dimensions stride_attr;
  if (stride.empty()) {
    stride_attr = kernel_size_attr;
  } else {
    TT_ASSIGN_OR_RETURN(stride_attr,
                        ExpandAttribute(stride, spatial_dim_count));
  }
  Dimensions dilation_attr;
  if (dilation.empty()) {
    dilation_attr = Dimensions(spatial_dim_count, 1);
  } else {
    TT_ASSIGN_OR_RETURN(dilation_attr,
                        ExpandAttribute(dilation, spatial_dim_count));
  }
  Dimensions padding_attr;
  if (padding.empty()) {
    padding_attr = Dimensions(spatial_dim_count, 0);
  } else {
    TT_ASSIGN_OR_RETURN(padding_attr,
                        ExpandAttribute(padding, spatial_dim_count));
  }

  // STEP 3. Padding pairs
  auto ceil_padding_pairs =
      CeilModePadding(input_shape, kernel_size_attr, stride_attr, padding_attr,
                      dilation_attr, ceil_mode);

  // STEP 4. ReduceWindow Attributes
  ReduceWindowAttributes reduce_window_attributes = GetReduceWindowAttributes(
      builder, kernel_size_attr, stride_attr, dilation_attr, ceil_padding_pairs,
      spatial_dim_count, total_num_dims);

  const mlir::Type element_type = input_shape.getElementType();

  // STEP 5. Initialize init_value (0.0)
  mlir::Attribute val_attr =
      builder.getOpBuilder().getFloatAttr(element_type, 0.0);
  mlir::DenseElementsAttr init_value_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, element_type), val_attr);
  mlir::MlirOp init_value = mlir::stablehlo::Constant(builder, init_value_attr);

  // STEP 6. Region Builders for Select and Scatter
  auto select_body = [element_type](mlir::RegionBuilder& body) {
    mlir::OpBuilder op_builder = body.getOpBuilder();
    mlir::Region& region = body.getRegion();
    if (region.getBlocks().empty()) op_builder.createBlock(&region);
    mlir::Block* block = &region.getBlocks().front();

    mlir::Type value_type = mlir::RankedTensorType::get({}, element_type);
    mlir::Location loc = body.getLoc();
    block->addArguments({value_type, value_type}, {loc, loc});

    auto lhs_value = block->getArgument(0);
    auto rhs_value = block->getArgument(1);

    auto ge_pred = mlir::stablehlo::CompareOp::create(
                       op_builder, loc, lhs_value, rhs_value,
                       mlir::stablehlo::ComparisonDirection::GE)
                       .getResult();

    mlir::stablehlo::ReturnOp::create(op_builder, loc,
                                      mlir::ValueRange{ge_pred});
  };

  auto scatter_body = [element_type](mlir::RegionBuilder& body) {
    mlir::OpBuilder op_builder = body.getOpBuilder();
    mlir::Region& region = body.getRegion();
    if (region.getBlocks().empty()) op_builder.createBlock(&region);
    mlir::Block* block = &region.getBlocks().front();

    mlir::Type value_type = mlir::RankedTensorType::get({}, element_type);
    mlir::Location loc = body.getLoc();
    block->addArguments({value_type, value_type}, {loc, loc});

    auto lhs_value = block->getArgument(0);
    auto rhs_value = block->getArgument(1);

    auto add_val =
        mlir::stablehlo::AddOp::create(op_builder, loc, lhs_value, rhs_value)
            .getResult();

    mlir::stablehlo::ReturnOp::create(op_builder, loc,
                                      mlir::ValueRange{add_val});
  };

  // STEP 7. Perform SelectAndScatter
  auto select_results = mlir::stablehlo::SelectAndScatter(
      /*operand=*/batch_input,
      /*source=*/batch_grad_output,
      /*init_value=*/init_value, select_body, scatter_body,
      reduce_window_attributes.window_dimensions,
      reduce_window_attributes.window_strides,
      reduce_window_attributes.padding);

  mlir::MlirOp batch_result = select_results;

  mlir::MlirOp final_output = RemoveTrivialBatch(
      batch_result, batch_input_info.original_dim_size, spatial_dim_count);

  return final_output;
}

absl::Status BuildMaxPoolBackwardGradInputNd(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& grad_input, int64_t spatial_dim_count,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

  auto op_builder = [kernel_size_vec = CopyIntVector(kernel_size),
                     stride_vec = CopyIntVector(stride),
                     padding_vec = CopyIntVector(padding),
                     dilation_vec = CopyIntVector(dilation), ceil_mode,
                     spatial_dim_count](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildMaxPoolBackwardShlo(inputs[0], inputs[1], spatial_dim_count,
                                    kernel_size_vec, stride_vec, padding_vec,
                                    dilation_vec, ceil_mode);
  };

  TT_ASSIGN_OR_RETURN(
      auto result,
      (DispatchOp<2>(std::move(op_builder), {grad_output, self},
                     {.out_dtype = output_dtype,
                      .out_dims = CopyIntVector(grad_input.sizes()),
                      .op_param_cache_keys = std::move(param_keys)})));

  return AssignBufferToAtTensor(std::move(result), grad_input);
}

// Builds the backward pass for max_pool with indices using StableHLO's
// ScatterOp.
//
// PyTorch's max_pool indices are relative linear index within the spatial
// dimensions, but StableHLO Scatter requires global indices within the
// flattened input tensor.
//
// How it works:
// 1. Normalize batch inputs: The input and grad_output tensors are normalized
// to have a batch dimension if it's missing.
//
// 2. Calculates global indices:
// Firstly, the offset tensor is calculated as:
//   offset = (batch_idx * num_channels + channel_idx) * spatial_size
// Then, The global indices are the relative indices plus the offset tensor.
// Finally, the global indices are reshaped to [num_updates, 1].
//
// 3. Initializes grad_input with zeros.
//
// 4. Flattens grad_output to a 1D tensor.
//
// 5. Configures ScatterDimensionNumbers.
//
// 6. Performs stablehlo.scatter to propagate gradients.
absl::StatusOr<mlir::MlirOp> BuildMaxPoolWithIndicesBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input, mlir::MlirOp indices,
    const int64_t spatial_dim_count, Dimensions kernel_size, Dimensions stride,
    Dimensions padding, Dimensions dilation, const bool ceil_mode) {
  auto& builder = grad_output.getBuilder();

  // STEP 1. Create batch inputs for both input and grad_output
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_output_info,
                      CreateBatchInput(grad_output, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_indices_info,
                      CreateBatchInput(indices, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirOp batch_grad_output = batch_grad_output_info.batch_input;
  mlir::MlirOp batch_indices = batch_indices_info.batch_input;

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(batch_input);
  auto input_shape = input_type.getShape();
  auto num_dims = input_type.getRank();

  // STEP 2. Compute the global indices of the input tensor
  auto indices_type = GetTensorTypeOrDie(batch_indices);
  auto idx_element_type = indices_type.getElementType();
  const int64_t num_updates = indices_type.getNumElements();

  // Calculate NC dimension size and spatial dimension size
  int64_t nc_dim_size = input_shape[0] * input_shape[1];
  int64_t spatial_size =
      std::accumulate(input_shape.begin() + 2, input_shape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  // Generate iota tensor: [0, 1, 2, ..., NC-1]
  auto nc_type = mlir::RankedTensorType::get({nc_dim_size}, idx_element_type);
  mlir::MlirOp iota_nc = mlir::stablehlo::Iota(builder, nc_type, 0);

  // Generate offsets: [0, S, 2S, ..., (NC-1)*S]
  auto spatial_size_scalar =
      MakeScalarConstant(builder, spatial_size, idx_element_type);
  auto spatial_size_vec = mlir::stablehlo::BroadcastInDim(
      nc_type, spatial_size_scalar, /*broadcast_dims=*/{});
  mlir::MlirOp offsets_flat = mlir::stablehlo::Mul(iota_nc, spatial_size_vec);

  // Reshape offsets tensor to shape [N, C, 1, 1, ...]
  Dimensions offset_shape = {input_shape[0], input_shape[1]};
  offset_shape.resize(num_dims, 1);  // Fill the rest of the dimensions with 1
  mlir::MlirOp offsets = mlir::stablehlo::Reshape(offsets_flat, offset_shape);

  // Broadcast offsets tensor to match the indices shape
  Dimensions broadcast_dims;
  for (int64_t i = 0; i < indices_type.getRank(); ++i) {
    broadcast_dims.push_back(i);
  }
  mlir::MlirOp offsets_broadcasted =
      mlir::stablehlo::BroadcastInDim(indices_type, offsets, broadcast_dims);

  // Global indices = relative indices + offsets.
  mlir::MlirOp global_indices =
      mlir::stablehlo::Add(batch_indices, offsets_broadcasted);

  // Flatten global_indices to [num_updates, 1]
  mlir::MlirOp scatter_indices =
      mlir::stablehlo::Reshape(global_indices, {num_updates, 1});

  // STEP 3. Initialize grad_input with zeros
  auto grad_input_zero = MakeConstant(builder, 0.0, input_type);
  auto grad_input_flat =
      mlir::stablehlo::Reshape(grad_input_zero, {input_type.getNumElements()});

  // STEP 4. Flatten grad_output to a 1D tensor
  mlir::MlirOp grad_output_flat =
      mlir::stablehlo::Reshape(batch_grad_output, {num_updates});

  // STEP 5. Configure ScatterDimensionNumbers
  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dnums =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &builder.getContext(),
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/{0},
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/1);

  auto scatter_body = [input_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        input_type.getElementType(), rb.getRegion(), rb.getOpBuilder());
  };

  // STEP 6. Perform the Scatter operation.
  // This operation scatters the grad_output values into the
  // grad_input_zero tensor at the locations specified by 'scatter_indices'.
  auto batch_result = mlir::stablehlo::Scatter(
      /*operand=*/grad_input_flat,
      /*scatter_indices=*/scatter_indices,
      /*updates=*/grad_output_flat,
      /*update_computation=*/scatter_body,
      /*dimension_numbers=*/scatter_dnums,
      /*indices_are_sorted=*/false,
      /*unique_indices=*/true);

  auto result = mlir::stablehlo::Reshape(batch_result[0], input_shape);

  // Remove the trivial batch dimension if necessary.
  return RemoveTrivialBatch(result, batch_input_info.original_dim_size,
                            spatial_dim_count);
}

// Helper function to build and dispatch N-dimensional max_pool operations.
absl::Status BuildMaxPoolWithIndicesOutNd(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices, int64_t spatial_dim_count,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(auto indices_type,
                      ConvertTo<mlir::ElementType>(indices.scalar_type()));

  // Explicitly copy the data to ensure it's valid for the
  // asynchronous execution of DispatchOp and prevent memory corruption.
  auto op_builder =
      [kernel_size_vec = CopyIntVector(kernel_size),
       stride_vec = CopyIntVector(stride), padding_vec = CopyIntVector(padding),
       dilation_vec = CopyIntVector(dilation), ceil_mode, spatial_dim_count](
          mlir::MlirOp input_op) -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(auto results,
                        BuildMaxPoolWithIndicesShlo(
                            input_op, spatial_dim_count, kernel_size_vec,
                            stride_vec, padding_vec, dilation_vec, ceil_mode));
    return MlirOpResults<2>({results[0], results[1]});
  };

  TT_ASSIGN_OR_RETURN(
      (auto [result_buf, indices_buf]),
      (DispatchOp<1, 2>(std::move(op_builder), self,
                        {.out_dtypes = {element_type, indices_type},
                         .out_dims_list = {CopyIntVector(out.sizes()),
                                           CopyIntVector(indices.sizes())},
                         .op_param_cache_keys = std::move(param_keys)})));

  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
  return AssignBufferToAtTensor(std::move(indices_buf), indices);
}

absl::Status BuildMaxPoolOutNd(const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, bool ceil_mode,
                               at::Tensor& out, int64_t spatial_dim_count,
                               OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  auto op_builder =
      [kernel_size_vec = CopyIntVector(kernel_size),
       stride_vec = CopyIntVector(stride), padding_vec = CopyIntVector(padding),
       dilation_vec = CopyIntVector(dilation), ceil_mode, spatial_dim_count](
          mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    return BuildMaxPoolShlo(input_op, spatial_dim_count, kernel_size_vec,
                            stride_vec, padding_vec, dilation_vec, ceil_mode);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      (DispatchOp<1>(std::move(op_builder), self,
                     {.out_dtype = element_type,
                      .out_dims = CopyIntVector(out.sizes()),
                      .op_param_cache_keys = std::move(param_keys)})));

  return AssignBufferToAtTensor(std::move(result_buf), out);
}

// Helper function to build and dispatch N-dimensional max_pool backward ops.
absl::Status BuildMaxPoolWithIndicesBackwardGradInputNd(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices, at::Tensor& grad_input,
    int64_t spatial_dim_count, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

  auto op_builder = [kernel_size_vec = CopyIntVector(kernel_size),
                     stride_vec = CopyIntVector(stride),
                     padding_vec = CopyIntVector(padding),
                     dilation_vec = CopyIntVector(dilation), ceil_mode,
                     spatial_dim_count](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildMaxPoolWithIndicesBackwardShlo(
        inputs[0], inputs[1], inputs[2], spatial_dim_count, kernel_size_vec,
        stride_vec, padding_vec, dilation_vec, ceil_mode);
  };

  TT_ASSIGN_OR_RETURN(
      auto result,
      (DispatchOp<3>(std::move(op_builder), {grad_output, self, indices},
                     {.out_dtype = output_dtype,
                      .out_dims = CopyIntVector(grad_input.sizes()),
                      .op_param_cache_keys = std::move(param_keys)})));

  return AssignBufferToAtTensor(std::move(result), grad_input);
}

}  // namespace

std::tuple<at::Tensor&, at::Tensor&> AtenMaxPool2dWithIndicesOut(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices) {
  TT_KERNEL(
      OpName::kMaxPool2dWithIndicesOut, param_keys,
      (self, kernel_size, stride, padding, dilation, ceil_mode, out, indices), {
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool,
                       error::kInvalidArgument)
            << "bool dtype is not supported";

        const int64_t spatial_dim_count = 2;
        TT_THROW_IF_ERROR(BuildMaxPoolWithIndicesOutNd(
            self, kernel_size, stride, padding, dilation, ceil_mode, out,
            indices, spatial_dim_count, std::move(param_keys)));
        return {out, indices};
      });
}

std::tuple<at::Tensor&, at::Tensor&> AtenMaxPool3dWithIndicesOut(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices) {
  TT_KERNEL(
      OpName::kMaxPool3dWithIndicesOut, param_keys,
      (self, kernel_size, stride, padding, dilation, ceil_mode, out, indices), {
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool,
                       error::kInvalidArgument)
            << "bool dtype is not supported";

        const int64_t spatial_dim_count = 3;
        TT_THROW_IF_ERROR(BuildMaxPoolWithIndicesOutNd(
            self, kernel_size, stride, padding, dilation, ceil_mode, out,
            indices, spatial_dim_count, std::move(param_keys)));
        return {out, indices};
      });
}

std::tuple<at::Tensor, at::Tensor> AtenMaxPool3dWithIndices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  TT_KERNEL(OpName::kMaxPool3dWithIndices, _,
            (self, IgnoreInCacheKey(kernel_size, "Legacy usage"),
             IgnoreInCacheKey(stride, "Legacy usage"),
             IgnoreInCacheKey(padding, "Legacy usage"),
             IgnoreInCacheKey(dilation, "Legacy usage"),
             IgnoreInCacheKey(ceil_mode, "Legacy usage")),
            {
              auto output_size =
                  GetMaxPoolOutputSize(self.sizes(), kernel_size, stride,
                                       padding, dilation, ceil_mode, 3);

              at::Tensor out = at::empty(output_size, self.options());
              at::Tensor indices =
                  at::empty(output_size, self.options().dtype(at::kLong));

              AtenMaxPool3dWithIndicesOut(self, kernel_size, stride, padding,
                                          dilation, ceil_mode, out, indices);

              return std::make_tuple(out, indices);
            });
}

at::Tensor& AtenMaxPool2dWithIndicesBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices, at::Tensor& grad_input) {
  TT_KERNEL(OpName::kMaxPool2dWithIndicesBackwardGradInput, param_keys,
            (grad_output, self, kernel_size, stride, padding, dilation,
             ceil_mode, indices, grad_input),
            {
              TT_THROW_IF_ERROR(BuildMaxPoolWithIndicesBackwardGradInputNd(
                  grad_output, self, kernel_size, stride, padding, dilation,
                  ceil_mode, indices, grad_input, /*spatial_dim_count=*/2,
                  std::move(param_keys)));
              return grad_input;
            });
}

at::Tensor AtenMaxPool3dWithIndicesBackward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  TT_KERNEL(OpName::kMaxPool3dWithIndicesBackward, _,
            (grad_output, self, IgnoreInCacheKey(kernel_size, "Legacy usage"),
             IgnoreInCacheKey(stride, "Legacy usage"),
             IgnoreInCacheKey(padding, "Legacy usage"),
             IgnoreInCacheKey(dilation, "Legacy usage"),
             IgnoreInCacheKey(ceil_mode, "Legacy usage"), indices),
            {
              at::Tensor grad_input = at::empty(self.sizes(), self.options());
              AtenMaxPool3dWithIndicesBackwardGradInput(
                  grad_output, self, kernel_size, stride, padding, dilation,
                  ceil_mode, indices, grad_input);

              return grad_input;
            });
}

at::Tensor& AtenMaxPool3dWithIndicesBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices, at::Tensor& grad_input) {
  TT_KERNEL(OpName::kMaxPool3dWithIndicesBackwardGradInput, param_keys,
            (grad_output, self, kernel_size, stride, padding, dilation,
             ceil_mode, indices, grad_input),
            {
              TT_THROW_IF_ERROR(BuildMaxPoolWithIndicesBackwardGradInputNd(
                  grad_output, self, kernel_size, stride, padding, dilation,
                  ceil_mode, indices, grad_input, /*spatial_dim_count=*/3,
                  std::move(param_keys)));
              return grad_input;
            });
}

Dimensions GetMaxPoolOutputSize(at::IntArrayRef input_size,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                at::IntArrayRef dilation, const bool ceil_mode,
                                const int64_t spatial_dim_count) {
  const int64_t dims_num = input_size.size();
  // If (N, C, ...), spatial starts at 2. If (C, ...), starts at 1.
  const int64_t spatial_start_index =
      (dims_num == spatial_dim_count + 2) ? 2 : 1;

  Dimensions output_size(dims_num);
  std::copy(input_size.begin(), input_size.begin() + spatial_start_index,
            output_size.begin());

  // Calculates Spatial Dims (D, H, W).
  for (int i = 0; i < spatial_dim_count; ++i) {
    const int64_t in = input_size[spatial_start_index + i];
    const int64_t out =
        GetSingleDim(in, i, kernel_size, stride, padding, dilation, ceil_mode);
    output_size[spatial_start_index + i] = std::max<int64_t>(out, 1);
  }
  return output_size;
}

at::Tensor AtenMaxPool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                         at::IntArrayRef stride, at::IntArrayRef padding,
                         at::IntArrayRef dilation, bool ceil_mode) {
  TT_KERNEL(OpName::kMaxPool2d, _,
            (self, IgnoreInCacheKey(kernel_size, "Handled by downstream ops"),
             IgnoreInCacheKey(stride, "Handled by downstream ops"),
             IgnoreInCacheKey(padding, "Handled by downstream ops"),
             IgnoreInCacheKey(dilation, "Handled by downstream ops"),
             IgnoreInCacheKey(ceil_mode, "Handled by downstream ops")),
            {
              const bool is_dilation_trivial = dilation.allMatch(
                  std::bind_front(std::equal_to<int64_t>(), 1));

              if (is_dilation_trivial) {
                return TpuMaxPool2dAutograd::apply(
                    self, kernel_size, stride, padding, dilation, ceil_mode);
              }

              auto [tensor, indices] = at::max_pool2d_with_indices(
                  self, kernel_size, stride, padding, dilation, ceil_mode);
              return tensor;
            });
}

at::Tensor TpuMaxPool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                        at::IntArrayRef stride, at::IntArrayRef padding,
                        at::IntArrayRef dilation, bool ceil_mode) {
  TT_KERNEL(OpName::kMaxPool2d, param_keys,
            (self, kernel_size, stride, padding, dilation, ceil_mode), {
              TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool,
                             error::kInvalidArgument)
                  << "bool dtype is not supported";

              const int64_t spatial_dim_count = 2;
              auto output_size = GetMaxPoolOutputSize(
                  self.sizes(), kernel_size, stride, padding, dilation,
                  ceil_mode, spatial_dim_count);
              at::Tensor out = at::empty(output_size, self.options());

              TT_THROW_IF_ERROR(BuildMaxPoolOutNd(
                  self, kernel_size, stride, padding, dilation, ceil_mode, out,
                  spatial_dim_count, std::move(param_keys)));
              return out;
            });
}

at::Tensor TpuMaxPool2dBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                at::IntArrayRef dilation, bool ceil_mode) {
  TT_KERNEL(
      OpName::kMaxPool2dBackward, param_keys,
      (grad_output, self, kernel_size, stride, padding, dilation, ceil_mode), {
        at::Tensor grad_input = at::empty(self.sizes(), self.options());
        TT_THROW_IF_ERROR(BuildMaxPoolBackwardGradInputNd(
            grad_output, self, kernel_size, stride, padding, dilation,
            ceil_mode, grad_input, /*spatial_dim_count=*/2,
            std::move(param_keys)));
        return grad_input;
      });
}

at::Tensor TpuMaxPool2dAutograd::forward(
    torch::autograd::AutogradContext* ctx, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  ctx->save_for_backward({self});
  ctx->saved_data["kernel_size"] = kernel_size.vec();  // VEC_OK
  ctx->saved_data["stride"] = stride.vec();            // VEC_OK
  ctx->saved_data["padding"] = padding.vec();          // VEC_OK
  ctx->saved_data["dilation"] = dilation.vec();        // VEC_OK
  ctx->saved_data["ceil_mode"] = ceil_mode;

  at::AutoDispatchBelowADInplaceOrView guard;
  return at::Dispatcher::singleton()
      .findSchemaOrThrow("torch_tpu::max_pool2d", "")
      .typed<at::Tensor(const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
                        at::IntArrayRef, at::IntArrayRef, bool)>()
      .call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

torch::autograd::variable_list TpuMaxPool2dAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto self = saved[0];
  auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
  auto stride = ctx->saved_data["stride"].toIntVector();
  auto padding = ctx->saved_data["padding"].toIntVector();
  auto dilation = ctx->saved_data["dilation"].toIntVector();
  bool ceil_mode = ctx->saved_data["ceil_mode"].toBool();

  auto grad_output = grad_outputs[0];

  at::AutoDispatchBelowADInplaceOrView guard;
  auto grad_input =
      at::Dispatcher::singleton()
          .findSchemaOrThrow("torch_tpu::max_pool2d_backward", "")
          .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                            at::IntArrayRef, at::IntArrayRef, at::IntArrayRef,
                            at::IntArrayRef, bool)>()
          .call(grad_output, self, kernel_size, stride, padding, dilation,
                ceil_mode);
  return {grad_input,   at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor()};
}

}  // namespace torch_tpu
