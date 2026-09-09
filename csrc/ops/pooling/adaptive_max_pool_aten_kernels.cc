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

#include "csrc/ops/pooling/adaptive_max_pool_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/pooling/pooling.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

struct GatherMaxPool1DResult {
  mlir::MlirOp values;
  mlir::MlirOp indices;
};

struct AdaptiveMaxPoolBuffers {
  DeviceBufferRef values;
  DeviceBufferRef indices;
};

struct AdaptiveIntervals {
  Dimensions starts;
  Dimensions lengths;
  int64_t max_k = 0;
};

// Computes the start and length of each interval along a given dimension based
// on the input and output sizes. `max_k` is the maximum length of any interval.
AdaptiveIntervals ComputeAdaptiveIndices(const int64_t in_size,
                                         const int64_t out_size) {
  Dimensions starts(out_size);
  Dimensions lengths(out_size);
  int64_t max_k = 0;
  for (int64_t i = 0; i < out_size; ++i) {
    const int64_t start = (i * in_size) / out_size;
    const int64_t end = 1 + ((i + 1) * in_size - 1) / out_size;

    starts[i] = start;
    lengths[i] = end - start;
    if (lengths[i] > max_k) {
      max_k = lengths[i];
    }
  }
  return AdaptiveIntervals{
      .starts = std::move(starts),
      .lengths = std::move(lengths),
      .max_k = max_k,
  };
}

// Helper: Create a 1D [N] int64 constant tensor.
mlir::MlirOp CreateI64Const(mlir::MlirBuilder& builder,
                            const Dimensions& data) {
  const auto type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(data.size())}, builder.getOpBuilder().getI64Type());

  return mlir::stablehlo::Constant(
      builder, mlir::makeConstant(llvm::ArrayRef<int64_t>(data), type));
}

// Builds a 1D gather pooling operation for a given dimension using the
// Gather + Mask + Variadic Reduce strategy.
absl::StatusOr<GatherMaxPool1DResult> BuildGatherMaxPool1D(
    mlir::MlirBuilder& builder, mlir::MlirOp input_val, mlir::MlirOp input_idx,
    const int64_t dim, const int64_t in_size, const int64_t out_size) {
  const auto val_type = GetTensorTypeOrDie(input_val);
  const auto idx_type = GetTensorTypeOrDie(input_idx);
  const Dimensions input_shape = CopyIntVector(val_type.getShape());
  const auto element_type = val_type.getElementType();
  const auto idx_element_type = idx_type.getElementType();
  const int64_t dim_size = val_type.getRank();

  // 1. Compute intervals
  const AdaptiveIntervals intervals = ComputeAdaptiveIndices(in_size, out_size);

  // 2. Padding along `dim` by `max_k`
  Dimensions pad_sizes(dim_size, 0);
  pad_sizes[dim] = intervals.max_k;
  const Dimensions zero_pad(dim_size, 0);

  mlir::Attribute val_min_attr;
  if (const auto float_type = llvm::dyn_cast<mlir::FloatType>(element_type)) {
    val_min_attr = mlir::FloatAttr::get(
        element_type, llvm::APFloat::getInf(float_type.getFloatSemantics(),
                                            /*Negative=*/true));
  } else {
    val_min_attr = GetMinFiniteValueAttr(element_type, builder.getOpBuilder());
  }
  const mlir::DenseElementsAttr val_min_dense = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, element_type), val_min_attr);
  mlir::MlirOp val_min_scalar =
      mlir::stablehlo::Constant(builder, val_min_dense);

  const int64_t max_idx_val = std::numeric_limits<int32_t>::max();
  mlir::MlirOp idx_max_scalar =
      MakeScalarConstant(builder, max_idx_val, idx_element_type);

  auto padded_val = mlir::stablehlo::Pad(input_val, val_min_scalar,
                                         /*edge_padding_low=*/zero_pad,
                                         /*edge_padding_high=*/pad_sizes,
                                         /*interior_padding=*/zero_pad);
  auto padded_idx = mlir::stablehlo::Pad(input_idx, idx_max_scalar,
                                         /*edge_padding_low=*/zero_pad,
                                         /*edge_padding_high=*/pad_sizes,
                                         /*interior_padding=*/zero_pad);

  // 3. Gather
  const auto start_const = CreateI64Const(builder, intervals.starts);
  auto start_indices = mlir::stablehlo::Reshape(start_const, {out_size, 1});

  auto slice_sizes = input_shape;
  slice_sizes[dim] = intervals.max_k;

  Dimensions offset_dims;
  offset_dims.reserve(dim_size);
  for (int64_t i = 0; i < dim_size; ++i) {
    if (i < dim) {
      offset_dims.push_back(i);
    } else {
      offset_dims.push_back(i + 1);
    }
  }

  const auto dim_numbers = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      &input_val.getContext(),
      /*offset_dims=*/offset_dims,
      /*collapsed_slice_dims=*/{},
      /*operand_batching_dims=*/{},
      /*start_indices_batching_dims=*/{},
      /*start_index_map=*/{dim},
      /*index_vector_dim=*/1);

  auto gather_result_shape = input_shape;
  gather_result_shape[dim] = out_size;
  gather_result_shape.insert(gather_result_shape.begin() + dim + 1,
                             intervals.max_k);

  auto gathered_val = mlir::stablehlo::Gather(padded_val, start_indices,
                                              dim_numbers, slice_sizes);
  auto gathered_idx = mlir::stablehlo::Gather(padded_idx, start_indices,
                                              dim_numbers, slice_sizes);

  // 4. Masking & Select
  const int64_t reduced_dim_index = dim + 1;
  const auto mask_shape_type = mlir::RankedTensorType::get(
      gather_result_shape, builder.getOpBuilder().getI64Type());
  const auto gather_val_type =
      mlir::RankedTensorType::get(gather_result_shape, element_type);
  const auto gather_idx_type =
      mlir::RankedTensorType::get(gather_result_shape, idx_element_type);

  const auto iota_type = mlir::RankedTensorType::get(
      {intervals.max_k}, builder.getOpBuilder().getI64Type());
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  auto iota_bcast = mlir::stablehlo::BroadcastInDim(mask_shape_type, iota,
                                                    {reduced_dim_index});

  mlir::MlirOp lengths_const = CreateI64Const(builder, intervals.lengths);
  auto lengths_bcast =
      mlir::stablehlo::BroadcastInDim(mask_shape_type, lengths_const, {dim});

  auto mask = mlir::stablehlo::Compare(
      iota_bcast, lengths_bcast, mlir::stablehlo::ComparisonDirection::LT);

  auto val_min_bcast =
      mlir::stablehlo::BroadcastInDim(gather_val_type, val_min_scalar, {});
  auto idx_max_bcast =
      mlir::stablehlo::BroadcastInDim(gather_idx_type, idx_max_scalar, {});

  auto masked_val = mlir::stablehlo::Select(mask, gathered_val, val_min_bcast);
  auto masked_idx = mlir::stablehlo::Select(mask, gathered_idx, idx_max_bcast);

  // 5. Variadic Reduce along reduced_dim_index
  const auto max_argmax_builder = [element_type, idx_element_type](
                                      mlir::RegionBuilder& body) {
    mlir::stablehlo::buildMaxAndArgmaxBody(
        element_type, idx_element_type, body.getRegion(), body.getOpBuilder());
  };

  const auto reduce_results =
      mlir::stablehlo::Reduce(builder, /*inputs=*/{masked_val, masked_idx},
                              /*init_values=*/{val_min_scalar, idx_max_scalar},
                              max_argmax_builder, {reduced_dim_index});

  return GatherMaxPool1DResult{.values = reduce_results[0],
                               .indices = reduce_results[1]};
}

absl::StatusOr<MlirOpResults<2>> BuildAdaptiveMaxPool2dShlo(
    mlir::MlirOp input_op, const int64_t in_h, const int64_t in_w,
    const int64_t out_h, const int64_t out_w, const int64_t spatial_dim_count) {
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(input_op, CastIfNeeded(input_op, compute_dtype));

  TT_ASSIGN_OR_RETURN(const auto batch_input_info,
                      CreateBatchInput(input_op, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirBuilder& builder = batch_input.getBuilder();

  const mlir::RankedTensorType input_shape = GetTensorTypeOrDie(batch_input);
  const mlir::Type element_type = input_shape.getElementType();
  const mlir::Type index_type = builder.getOpBuilder().getI32Type();

  mlir::MlirOp batch_val_result;
  mlir::MlirOp batch_idx_result;

  if (in_h >= out_h && in_w >= out_w && in_h % out_h == 0 &&
      in_w % out_w == 0) {
    // Fast path: standard MaxPool
    const int64_t stride_h = in_h / out_h;
    const int64_t stride_w = in_w / out_w;
    const int64_t kernel_h = stride_h;
    const int64_t kernel_w = stride_w;

    const Dimensions kernel_size = {kernel_h, kernel_w};
    const Dimensions stride = {stride_h, stride_w};
    const Dimensions padding = {0, 0};
    const Dimensions dilation = {1, 1};

    const auto ceil_padding_pairs =
        CeilModePadding(input_shape, kernel_size, stride, padding, dilation,
                        /*ceil_mode=*/false);

    const ReduceWindowAttributes reduce_window_attributes =
        GetReduceWindowAttributes(builder, kernel_size, stride, dilation,
                                  ceil_padding_pairs, spatial_dim_count,
                                  /*total_num_dims=*/4);

    // Create global flattened indices
    const int64_t spatial_input_elements = in_h * in_w;
    const auto flat_iota_type =
        mlir::RankedTensorType::get({spatial_input_elements}, index_type);
    const mlir::MlirOp iota_flat =
        mlir::stablehlo::Iota(builder, flat_iota_type, 0);
    mlir::MlirOp iota_2d = mlir::stablehlo::Reshape(iota_flat, {in_h, in_w});

    const Dimensions target_shape = CopyIntVector(input_shape.getShape());
    const mlir::RankedTensorType target_tensor_type =
        mlir::RankedTensorType::get(target_shape, index_type);
    const mlir::MlirOp iota_broadcast =
        mlir::stablehlo::BroadcastInDim(target_tensor_type, iota_2d, {2, 3});

    mlir::Attribute val_attr;
    if (const auto float_type = llvm::dyn_cast<mlir::FloatType>(element_type)) {
      val_attr = mlir::FloatAttr::get(
          element_type, llvm::APFloat::getInf(float_type.getFloatSemantics(),
                                              /*Negative=*/true));
    } else {
      val_attr = GetMinFiniteValueAttr(element_type, builder.getOpBuilder());
    }
    const mlir::DenseElementsAttr init_value_attr =
        mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get({}, element_type), val_attr);
    const mlir::MlirOp init_value =
        mlir::stablehlo::Constant(builder, init_value_attr);

    const int64_t max_idx_val = std::numeric_limits<int32_t>::max();
    const mlir::MlirOp init_invalid_iota =
        MakeScalarConstant(builder, max_idx_val, index_type);

    const auto reduce_results = mlir::stablehlo::ReduceWindow(
        builder,
        /*inputs=*/{batch_input, iota_broadcast},
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

    batch_val_result = reduce_results[0];
    batch_idx_result = reduce_results[1];
  } else {
    // General path: 2-stage Gather + Mask + Variadic Reduce
    const int64_t spatial_input_elements = in_h * in_w;
    const auto flat_iota_type =
        mlir::RankedTensorType::get({spatial_input_elements}, index_type);
    const mlir::MlirOp iota_flat =
        mlir::stablehlo::Iota(builder, flat_iota_type, 0);
    mlir::MlirOp iota_2d = mlir::stablehlo::Reshape(iota_flat, {in_h, in_w});

    const Dimensions target_shape = CopyIntVector(input_shape.getShape());
    const mlir::RankedTensorType target_tensor_type =
        mlir::RankedTensorType::get(target_shape, index_type);
    const mlir::MlirOp iota_broadcast =
        mlir::stablehlo::BroadcastInDim(target_tensor_type, iota_2d, {2, 3});

    // Stage 1: Width-dim pooling (dim = 3)
    TT_ASSIGN_OR_RETURN(
        (auto [val_w, idx_w]),
        BuildGatherMaxPool1D(builder, batch_input, iota_broadcast, /*dim=*/3,
                             in_w, out_w));

    // Stage 2: Height-dim pooling (dim = 2)
    TT_ASSIGN_OR_RETURN(
        (auto [val_h, idx_h]),
        BuildGatherMaxPool1D(builder, val_w, idx_w, /*dim=*/2, in_h, out_h));

    batch_val_result = val_h;
    batch_idx_result = idx_h;
  }

  const mlir::MlirOp final_output = RemoveTrivialBatch(
      batch_val_result, batch_input_info.original_dim_size, spatial_dim_count);
  const mlir::MlirOp final_indices = RemoveTrivialBatch(
      batch_idx_result, batch_input_info.original_dim_size, spatial_dim_count);

  const mlir::MlirOp final_indices_i64 = mlir::stablehlo::ConvertElementType(
      final_indices, builder.getOpBuilder().getI64Type());

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp final_output_cast,
                      CastIfNeeded(final_output, orig_dtype));

  return MlirOpResults<2>({final_output_cast, final_indices_i64});
}

void CheckAdaptiveMaxPoolDtypes(const at::Tensor& self) {
  TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool,
                 error::kInvalidArgument)
      << "bool dtype is not supported";

  TT_CHECK_THROW(self.numel() == 0 ||
                     (self.scalar_type() != at::ScalarType::Byte &&
                      self.scalar_type() != at::ScalarType::Char &&
                      self.scalar_type() != at::ScalarType::Short &&
                      self.scalar_type() != at::ScalarType::Int &&
                      self.scalar_type() != at::ScalarType::Long &&
                      self.scalar_type() != at::ScalarType::ComplexFloat &&
                      self.scalar_type() != at::ScalarType::ComplexDouble),
                 error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());
}

absl::StatusOr<AdaptiveMaxPoolBuffers> AdaptiveMaxPool2dHelper(
    const at::Tensor& self, at::IntArrayRef output_size,
    OpParamCacheKeys param_keys) {
  const int64_t ndim = self.dim();
  TT_RET_CHECK(ndim == 3 || ndim == 4, error::kInvalidArgument)
      << "expected 3D or 4D tensor, got " << self.sizes();

  for (int64_t i = 1; i < ndim; ++i) {
    TT_RET_CHECK(self.size(i) > 0, error::kInvalidArgument)
        << "expected input to have non-zero size for non-batch dimensions, got "
        << self.sizes() << " with dimension " << i << " being empty";
  }

  TT_RET_CHECK(output_size.size() == 2, error::kInvalidArgument)
      << "expected output_size to have 2 elements, got " << output_size.size();

  const int64_t out_h = output_size[0];
  const int64_t out_w = output_size[1];

  TT_RET_CHECK(out_h >= 0 && out_w >= 0, error::kInvalidArgument)
      << "expected output size to be non-negative, got [" << out_h << ", "
      << out_w << "]";

  const int64_t in_h = self.size(ndim - 2);
  const int64_t in_w = self.size(ndim - 1);

  Dimensions out_dims = CopyIntVector(self.sizes());
  out_dims[ndim - 2] = out_h;
  out_dims[ndim - 1] = out_w;

  TT_ASSIGN_OR_RETURN(const auto element_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  const auto indices_type = mlir::ElementType::I64;

  if (out_h == 0 || out_w == 0) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef val_buf,
                        CreateZeroSizeDeviceBufferRef(out_dims, element_type));
    TT_ASSIGN_OR_RETURN(DeviceBufferRef idx_buf,
                        CreateZeroSizeDeviceBufferRef(out_dims, indices_type));
    return AdaptiveMaxPoolBuffers{.values = std::move(val_buf),
                                  .indices = std::move(idx_buf)};
  }

  const auto op_builder =
      [in_h, in_w, out_h,
       out_w](mlir::MlirOp input_op) -> absl::StatusOr<MlirOpResults<2>> {
    return BuildAdaptiveMaxPool2dShlo(input_op, in_h, in_w, out_h, out_w,
                                      /*spatial_dim_count=*/2);
  };

  TT_ASSIGN_OR_RETURN(
      (auto [val_buf, idx_buf]),
      (DispatchOp<1, 2>(std::move(op_builder), self,
                        {.out_dtypes = {element_type, indices_type},
                         .out_dims_list = {out_dims, out_dims},
                         .op_param_cache_keys = std::move(param_keys)})));

  return AdaptiveMaxPoolBuffers{.values = std::move(val_buf),
                                .indices = std::move(idx_buf)};
}

}  // namespace

std::tuple<at::Tensor&, at::Tensor&> AtenAdaptiveMaxPool2dOut(
    const at::Tensor& self, at::IntArrayRef output_size, at::Tensor& out,
    at::Tensor& indices) {
  TT_KERNEL(
      OpName::kAdaptiveMaxPool2dOut, param_keys,
      (self, output_size, out, indices), {
        CheckAdaptiveMaxPoolDtypes(self);

        TT_CHECK_THROW(out.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "expected out tensor to have dtype "
            << ToString(self.scalar_type()) << ", got "
            << ToString(out.scalar_type());

        TT_CHECK_THROW(indices.scalar_type() == at::ScalarType::Long,
                       error::kInvalidArgument)
            << "expected indices tensor to have dtype "
            << ToString(at::ScalarType::Long) << ", got "
            << ToString(indices.scalar_type());

        TT_ASSIGN_OR_THROW(
            (auto [result_buf, indices_buf]),
            AdaptiveMaxPool2dHelper(self, output_size, std::move(param_keys)));
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, result_buf.dimensions()));
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(indices, indices_buf.dimensions()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(indices_buf), indices));
        return {out, indices};
      });
}

}  // namespace torch_tpu
