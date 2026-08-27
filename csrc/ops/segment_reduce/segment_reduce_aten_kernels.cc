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

#include "torch_tpu/csrc/ops/segment_reduce/segment_reduce_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/string_view.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/cumsum/cumsum.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"

namespace torch_tpu {

namespace {

enum class SegmentReduceOp { kSum, kMean, kMax, kMin, kProd };

absl::StatusOr<SegmentReduceOp> ParseSegmentReduceOp(
    std::string_view reduce_str) {
  if (reduce_str == "sum") {
    return SegmentReduceOp::kSum;
  }
  if (reduce_str == "mean") {
    return SegmentReduceOp::kMean;
  }
  if (reduce_str == "max") {
    return SegmentReduceOp::kMax;
  }
  if (reduce_str == "min") {
    return SegmentReduceOp::kMin;
  }
  if (reduce_str == "prod") {
    return SegmentReduceOp::kProd;
  }
  return TT_ERROR(error::kInvalidArgument)
         << "expected reduce mode to be sum, mean, max, min, or prod, got "
         << reduce_str;
}

at::Scalar GetNeutralScalar(SegmentReduceOp reduce_op,
                            mlir::Type computation_type) {
  switch (reduce_op) {
    case SegmentReduceOp::kSum:
    case SegmentReduceOp::kMean:
      return 0;
    case SegmentReduceOp::kProd:
      return 1;
    case SegmentReduceOp::kMax:
      if (mlir::isa<mlir::FloatType>(computation_type)) {
        return -std::numeric_limits<double>::infinity();
      }
      if (computation_type.getIntOrFloatBitWidth() <= 32) {
        return std::numeric_limits<int32_t>::min();
      }
      return std::numeric_limits<int64_t>::min();
    case SegmentReduceOp::kMin:
      if (mlir::isa<mlir::FloatType>(computation_type)) {
        return std::numeric_limits<double>::infinity();
      }
      if (computation_type.getIntOrFloatBitWidth() <= 32) {
        return std::numeric_limits<int32_t>::max();
      }
      return std::numeric_limits<int64_t>::max();
  }
}

absl::StatusOr<mlir::MlirOp> ApplyInitialValue(mlir::MlirOp scatter_res,
                                               SegmentReduceOp reduce_op,
                                               mlir::MlirOp initial_op,
                                               const Dimensions& init_shape) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp init_const,
                      Broadcast(initial_op, init_shape, {}));
  switch (reduce_op) {
    case SegmentReduceOp::kSum:
      return mlir::stablehlo::Add(scatter_res, init_const);
    case SegmentReduceOp::kProd:
      return mlir::stablehlo::Mul(scatter_res, init_const);
    case SegmentReduceOp::kMax:
      return mlir::stablehlo::Max(scatter_res, init_const);
    case SegmentReduceOp::kMin:
      return mlir::stablehlo::Min(scatter_res, init_const);
    case SegmentReduceOp::kMean:
      return scatter_res;
  }
}

struct SegmentBoundariesAndLengths {
  mlir::MlirOp boundary_indices;
  mlir::MlirOp lengths;
  int64_t num_segments = 0;
};

struct SegmentIdsAndMaskedData {
  mlir::MlirOp segment_ids;
  mlir::MlirOp data_masked;
};

absl::StatusOr<SegmentBoundariesAndLengths> ComputeSegmentBoundariesAndLengths(
    std::optional<mlir::MlirOp> lengths_opt,
    std::optional<mlir::MlirOp> offsets_opt, int64_t axis) {
  if (offsets_opt.has_value()) {
    mlir::MlirOp offsets = *offsets_opt;
    const mlir::RankedTensorType offsets_type = GetTensorTypeOrDie(offsets);
    const Dimensions offsets_shape = CopyIntVector(offsets_type.getShape());
    const int64_t offsets_rank = offsets_type.getRank();
    const int64_t num_segments = offsets_shape.back() - 1;

    Dimensions start_indices_off(offsets_rank, 0);
    Dimensions limit_indices_off = offsets_shape;
    limit_indices_off.back() = num_segments;
    Dimensions strides(offsets_rank, 1);

    mlir::MlirOp off_start = mlir::stablehlo::Slice(offsets, start_indices_off,
                                                    limit_indices_off, strides);

    Dimensions start_indices_boundary(offsets_rank, 0);
    start_indices_boundary.back() = 1;
    Dimensions limit_indices_boundary = offsets_shape;
    limit_indices_boundary.back() = num_segments + 1;

    mlir::MlirOp boundary_indices = mlir::stablehlo::Slice(
        offsets, start_indices_boundary, limit_indices_boundary, strides);
    mlir::MlirOp lengths =
        mlir::stablehlo::Subtract(boundary_indices, off_start);
    return SegmentBoundariesAndLengths{boundary_indices, lengths, num_segments};
  }

  mlir::MlirOp lengths = *lengths_opt;
  const mlir::RankedTensorType lengths_type = GetTensorTypeOrDie(lengths);
  const int64_t num_segments = lengths_type.getShape().back();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp boundary_indices,
                      BuildCumsumShlo(/*normalized_dim=*/axis,
                                      /*out_dtype=*/std::nullopt, lengths));
  return SegmentBoundariesAndLengths{boundary_indices, lengths, num_segments};
}

absl::StatusOr<SegmentIdsAndMaskedData> ComputeSegmentIdsAndMaskedData(
    mlir::MlirOp data, mlir::MlirOp boundary_indices, mlir::MlirOp lengths,
    int64_t num_segments, int64_t axis, SegmentReduceOp reduce_op,
    mlir::Type computation_type, mlir::ElementType comp_elem_type,
    const Dimensions& data_shape, int64_t n_data_axis) {
  mlir::MlirBuilder& builder = data.getBuilder();
  const mlir::Type index_type = GetTensorTypeOrDie(lengths).getElementType();
  mlir::MlirOp cumsum_exclusive =
      mlir::stablehlo::Subtract(boundary_indices, lengths);

  Dimensions bound_shape;
  bound_shape.reserve(axis + 2);
  for (int64_t d = 0; d < axis; ++d) {
    bound_shape.push_back(data_shape[d]);
  }
  bound_shape.push_back(1);
  bound_shape.push_back(num_segments);
  mlir::MlirOp c_exc_reshaped =
      mlir::stablehlo::Reshape(cumsum_exclusive, bound_shape);
  mlir::MlirOp c_inc_reshaped =
      mlir::stablehlo::Reshape(boundary_indices, bound_shape);

  Dimensions comp_shape;
  comp_shape.reserve(axis + 2);
  for (int64_t d = 0; d < axis; ++d) {
    comp_shape.push_back(data_shape[d]);
  }
  comp_shape.push_back(n_data_axis);
  comp_shape.push_back(num_segments);

  auto iota_1d_type = mlir::RankedTensorType::get({n_data_axis}, index_type);
  mlir::MlirOp iota_1d = mlir::stablehlo::Iota(builder, iota_1d_type, 0);
  Dimensions iota_reshaped_dims(axis + 2, 1);
  iota_reshaped_dims[axis] = n_data_axis;
  mlir::MlirOp iota_n = mlir::stablehlo::Reshape(iota_1d, iota_reshaped_dims);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp iota_bcst,
                      BroadcastIfNeeded(iota_n, comp_shape));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp c_exc_bcst,
                      BroadcastIfNeeded(c_exc_reshaped, comp_shape));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp c_inc_bcst,
                      BroadcastIfNeeded(c_inc_reshaped, comp_shape));

  mlir::MlirOp is_ge = mlir::stablehlo::Compare(
      iota_bcst, c_exc_bcst, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp is_lt = mlir::stablehlo::Compare(
      iota_bcst, c_inc_bcst, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp in_segment = mlir::stablehlo::And(is_ge, is_lt);
  mlir::MlirOp in_segment_idx =
      mlir::stablehlo::ConvertElementType(in_segment, index_type);

  auto k_iota_type = mlir::RankedTensorType::get({num_segments}, index_type);
  mlir::MlirOp k_iota = mlir::stablehlo::Iota(builder, k_iota_type, 0);
  Dimensions k_reshaped_dims(axis + 2, 1);
  k_reshaped_dims[axis + 1] = num_segments;
  mlir::MlirOp k_reshaped = mlir::stablehlo::Reshape(k_iota, k_reshaped_dims);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp k_bcst,
                      BroadcastIfNeeded(k_reshaped, comp_shape));

  mlir::MlirOp k_weighted = mlir::stablehlo::Mul(in_segment_idx, k_bcst);

  mlir::MlirOp zero_init = MakeScalarConstant(builder, 0, index_type);
  mlir::MlirOp segment_ids = mlir::stablehlo::Reduce(
      builder, {k_weighted}, {zero_init},
      [index_type](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            index_type, rb.getRegion(), body_builder);
      },
      {axis + 1})[0];

  mlir::MlirOp is_valid_count = mlir::stablehlo::Reduce(
      builder, {in_segment_idx}, {zero_init},
      [index_type](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            index_type, rb.getRegion(), body_builder);
      },
      {axis + 1})[0];

  mlir::MlirOp zero_count = MakeConstantLike(is_valid_count, 0);
  mlir::MlirOp is_valid = mlir::stablehlo::Compare(
      is_valid_count, zero_count, mlir::stablehlo::ComparisonDirection::GT);

  Dimensions valid_bcast_dims(axis + 1);
  std::iota(valid_bcast_dims.begin(), valid_bcast_dims.end(), 0);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp is_valid_bcast,
                      Broadcast(is_valid, data_shape, valid_bcast_dims));

  at::Scalar neutral_scalar = GetNeutralScalar(reduce_op, computation_type);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp neutral_constant,
      MakeConstant(builder, neutral_scalar, comp_elem_type, data_shape));
  mlir::MlirOp data_masked =
      mlir::stablehlo::Select(is_valid_bcast, data, neutral_constant);

  return SegmentIdsAndMaskedData{segment_ids, data_masked};
}

mlir::MlirOp ComputeScatterIndices(mlir::MlirBuilder& builder,
                                   mlir::MlirOp segment_ids, int64_t axis,
                                   const Dimensions& data_shape,
                                   int64_t n_data_axis, mlir::Type index_type) {
  if (axis == 0) {
    return mlir::stablehlo::Reshape(segment_ids, {n_data_axis, 1});
  }

  Dimensions outer_shape;
  outer_shape.reserve(axis + 1);
  for (int64_t d = 0; d < axis; ++d) {
    outer_shape.push_back(data_shape[d]);
  }
  outer_shape.push_back(n_data_axis);

  Dimensions d_coord_shape = outer_shape;
  d_coord_shape.push_back(1);

  std::vector<mlir::MlirOp> coords;
  coords.reserve(axis + 1);
  for (int64_t d = 0; d < axis; ++d) {
    auto d_iota_type = mlir::RankedTensorType::get(outer_shape, index_type);
    mlir::MlirOp d_iota =
        mlir::stablehlo::Iota(builder, d_iota_type, /*iota_dimension=*/d);
    coords.push_back(mlir::stablehlo::Reshape(d_iota, d_coord_shape));
  }
  coords.push_back(mlir::stablehlo::Reshape(segment_ids, d_coord_shape));
  return mlir::stablehlo::Concatenate(builder, coords, /*dimension=*/axis + 1);
}

absl::StatusOr<mlir::MlirOp> ExecuteScatterReduction(
    mlir::MlirBuilder& builder, mlir::MlirOp data_masked,
    mlir::MlirOp scatter_indices, SegmentReduceOp reduce_op,
    mlir::Type computation_type, mlir::ElementType comp_elem_type,
    const Dimensions& init_shape, int64_t axis, int64_t rank,
    std::optional<mlir::MlirOp> initial_op) {
  at::Scalar neutral_scalar = GetNeutralScalar(reduce_op, computation_type);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp init_output,
      MakeConstant(builder, neutral_scalar, comp_elem_type, init_shape));

  Dimensions update_window_dims;
  update_window_dims.reserve(rank - 1 - axis);
  for (int64_t i = axis + 1; i < rank; ++i) {
    update_window_dims.push_back(i);
  }

  Dimensions inserted_window_dims(axis + 1);
  std::iota(inserted_window_dims.begin(), inserted_window_dims.end(), 0);

  Dimensions scatter_dims_to_operand_dims(axis + 1);
  std::iota(scatter_dims_to_operand_dims.begin(),
            scatter_dims_to_operand_dims.end(), 0);

  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dnums =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &builder.getContext(),
          /*update_window_dims=*/update_window_dims,
          /*inserted_window_dims=*/inserted_window_dims,
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/scatter_dims_to_operand_dims,
          /*index_vector_dim=*/axis + 1);

  auto scatter_region_builder = [reduce_op,
                                 computation_type](mlir::RegionBuilder& rb) {
    switch (reduce_op) {
      case SegmentReduceOp::kSum:
      case SegmentReduceOp::kMean:
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            computation_type, rb.getRegion(), rb.getOpBuilder());
        break;
      case SegmentReduceOp::kProd:
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::MulOp>(
            computation_type, rb.getRegion(), rb.getOpBuilder());
        break;
      case SegmentReduceOp::kMax:
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
            computation_type, rb.getRegion(), rb.getOpBuilder());
        break;
      case SegmentReduceOp::kMin:
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::MinOp>(
            computation_type, rb.getRegion(), rb.getOpBuilder());
        break;
    }
  };

  mlir::MlirOp scatter_res =
      mlir::stablehlo::Scatter(init_output, scatter_indices, data_masked,
                               scatter_region_builder, scatter_dnums)[0];

  if (initial_op.has_value()) {
    TT_ASSIGN_OR_RETURN(
        scatter_res,
        ApplyInitialValue(scatter_res, reduce_op, *initial_op, init_shape));
  }

  return scatter_res;
}

absl::StatusOr<mlir::MlirOp> PostProcessMeanReduction(
    mlir::MlirOp scatter_res, mlir::MlirOp lengths, int64_t axis,
    mlir::Type computation_type, const Dimensions& init_shape,
    std::optional<mlir::MlirOp> initial_op) {
  Dimensions bcast_dims_lengths(axis + 1);
  std::iota(bcast_dims_lengths.begin(), bcast_dims_lengths.end(), 0);

  mlir::MlirOp lengths_conv =
      mlir::stablehlo::ConvertElementType(lengths, computation_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp bcast_lengths,
                      Broadcast(lengths_conv, init_shape, bcast_dims_lengths));

  if (initial_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp initial_or_nan,
                        Broadcast(*initial_op, init_shape, {}));
    mlir::MlirOp sum = mlir::stablehlo::Add(scatter_res, initial_or_nan);
    mlir::MlirOp zero = MakeConstantLike(lengths, 0);
    mlir::MlirOp is_empty_mask = mlir::stablehlo::Compare(
        lengths, zero, mlir::stablehlo::ComparisonDirection::EQ);
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp is_empty,
        Broadcast(is_empty_mask, init_shape, bcast_dims_lengths));
    mlir::MlirOp div = mlir::stablehlo::Div(sum, bcast_lengths);
    return mlir::stablehlo::Select(is_empty, initial_or_nan, div);
  }
  return mlir::stablehlo::Div(scatter_res, bcast_lengths);
}

absl::StatusOr<mlir::MlirOp> BuildSegmentReduceShlo(
    mlir::MlirOp data, SegmentReduceOp reduce_op,
    std::optional<mlir::MlirOp> lengths_opt,
    std::optional<mlir::MlirOp> offsets_opt, int64_t axis,
    std::optional<mlir::MlirOp> initial_op) {
  mlir::MlirBuilder& builder = data.getBuilder();
  const mlir::RankedTensorType data_type = GetTensorTypeOrDie(data);
  const int64_t rank = data_type.getRank();
  const Dimensions data_shape = CopyIntVector(data_type.getShape());
  const int64_t n_data_axis = data_shape[axis];
  const mlir::Type computation_type = data_type.getElementType();
  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_elem_type,
                      ConvertTo<mlir::ElementType>(computation_type));

  TT_ASSIGN_OR_RETURN(
      const auto boundaries_and_lengths,
      ComputeSegmentBoundariesAndLengths(lengths_opt, offsets_opt, axis));
  const auto& [boundary_indices, lengths, num_segments] =
      boundaries_and_lengths;

  TT_ASSIGN_OR_RETURN(
      const auto ids_and_masked,
      ComputeSegmentIdsAndMaskedData(
          data, boundary_indices, lengths, num_segments, axis, reduce_op,
          computation_type, comp_elem_type, data_shape, n_data_axis));
  const auto& [segment_ids, data_masked] = ids_and_masked;

  const mlir::Type index_type = GetTensorTypeOrDie(lengths).getElementType();
  mlir::MlirOp scatter_indices = ComputeScatterIndices(
      builder, segment_ids, axis, data_shape, n_data_axis, index_type);

  Dimensions init_shape = data_shape;
  init_shape[axis] = num_segments;

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp scatter_res,
      ExecuteScatterReduction(builder, data_masked, scatter_indices, reduce_op,
                              computation_type, comp_elem_type, init_shape,
                              axis, rank, initial_op));

  if (reduce_op == SegmentReduceOp::kMean) {
    return PostProcessMeanReduction(scatter_res, lengths, axis,
                                    computation_type, init_shape, initial_op);
  }

  return scatter_res;
}

}  // namespace

at::Tensor AtenSegmentReduce(const at::Tensor& data, c10::string_view reduce,
                             const std::optional<at::Tensor>& lengths,
                             const std::optional<at::Tensor>& indices,
                             const std::optional<at::Tensor>& offsets,
                             int64_t axis, bool unsafe,
                             const std::optional<at::Scalar>& initial) {
  std::optional<PromotedScalar> promoted_initial = PromoteScalar(initial);

  TT_KERNEL(
      OpName::kSegmentReduce, param_keys,
      (data, reduce, lengths, indices, offsets, axis,
       IgnoreInCacheKey(unsafe, "Doesn't affect SHLO"), promoted_initial),
      {
        TT_CHECK_THROW(data.dim() >= 1, error::kInvalidArgument)
            << "expected input with at least 1 dim, got input with "
            << data.dim() << " dims";

        TT_CHECK_THROW(!indices.has_value(), error::kInvalidArgument)
            << "expected lengths or offsets for reduction, got indices";

        TT_CHECK_THROW(lengths.has_value() || offsets.has_value(),
                       error::kInvalidArgument)
            << "expected lengths or offsets to be defined, got neither";

        TT_ASSIGN_OR_THROW(const SegmentReduceOp reduce_op,
                           ParseSegmentReduceOp(reduce));

        TT_ASSIGN_OR_THROW(const int64_t wrapped_axis,
                           SafeWrapDim(axis, data.dim()));

        const at::Tensor& segment_info =
            lengths.has_value() ? *lengths : *offsets;
        TT_CHECK_THROW(data.dim() >= segment_info.dim(),
                       error::kInvalidArgument)
            << "expected data dim >= lengths/offsets dim, got " << data.dim();
        TT_CHECK_THROW(wrapped_axis == segment_info.dim() - 1,
                       error::kInvalidArgument)
            << "expected axis to be the last dimension of lengths/offsets, got "
            << axis;
        for (int64_t d = 0; d < wrapped_axis; ++d) {
          TT_CHECK_THROW(segment_info.size(d) == data.size(d),
                         error::kInvalidArgument)
              << "expected outer dimension " << d
              << " to match between data and lengths/offsets, got "
              << data.size(d) << " and " << segment_info.size(d);
        }

        int64_t num_segments = 0;
        if (lengths.has_value()) {
          num_segments = lengths->size(-1);
        } else {
          num_segments = offsets->size(-1) - 1;
        }

        Dimensions output_dims = CopyIntVector(data.sizes());
        output_dims[wrapped_axis] = num_segments;

        TT_ASSIGN_OR_THROW(const auto element_type,
                           ConvertTo<mlir::ElementType>(data.scalar_type()));

        std::optional<at::Tensor> initial_tensor;
        if (promoted_initial.has_value()) {
          TT_ASSIGN_OR_THROW(auto t,
                             promoted_initial->GetTensor(data.scalar_type()));
          initial_tensor = std::move(t);
        }

        const bool has_lengths = lengths.has_value();
        const bool has_initial = initial_tensor.has_value();

        auto op_builder = [reduce_op, wrapped_axis, has_lengths,
                           has_initial](absl::Span<mlir::MlirOp> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildSegmentReduceShlo(
              inputs[0], reduce_op,
              has_lengths ? std::make_optional(inputs[1]) : std::nullopt,
              has_lengths ? std::nullopt : std::make_optional(inputs[1]),
              wrapped_axis,
              has_initial ? std::make_optional(inputs[2]) : std::nullopt);
        };

        if (has_initial) {
          TT_ASSIGN_OR_THROW(
              DeviceBufferRef out_buf,
              (DispatchOp<3>(std::move(op_builder),
                             {data, segment_info, *initial_tensor},
                             {.out_dtype = element_type,
                              .out_dims = output_dims,
                              .op_param_cache_keys = std::move(param_keys)})));
          return MakeTensor(out_buf);
        }
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef out_buf,
            (DispatchOp<2>(std::move(op_builder), {data, segment_info},
                           {.out_dtype = element_type,
                            .out_dims = output_dims,
                            .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(out_buf);
      });
}

}  // namespace torch_tpu
