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

#include "csrc/ops/segment_reduce/segment_reduce_aten_kernels.h"

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
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/cumsum/cumsum.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

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
                                               mlir::MlirOp initial_op) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp init_const,
                      BroadcastIfNeeded(initial_op, scatter_res));
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
    TT_ASSIGN_OR_RETURN(mlir::MlirOp off_start,
                        BuildMaybeSlice(offsets, /*dimension=*/axis,
                                        /*left_pad=*/0, /*right_pad=*/-1));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp boundary_indices,
                        BuildMaybeSlice(offsets, /*dimension=*/axis,
                                        /*left_pad=*/-1, /*right_pad=*/0));
    mlir::MlirOp lengths =
        mlir::stablehlo::Subtract(boundary_indices, off_start);
    return SegmentBoundariesAndLengths{boundary_indices, lengths};
  }

  mlir::MlirOp lengths = *lengths_opt;
  TT_ASSIGN_OR_RETURN(mlir::MlirOp boundary_indices,
                      BuildCumsumShlo(/*normalized_dim=*/axis,
                                      /*out_dtype=*/std::nullopt, lengths));
  return SegmentBoundariesAndLengths{boundary_indices, lengths};
}

absl::StatusOr<SegmentIdsAndMaskedData> ComputeSegmentIdsAndMaskedData(
    mlir::MlirOp data, mlir::MlirOp boundary_indices, mlir::MlirOp lengths,
    int64_t axis, SegmentReduceOp reduce_op, mlir::Type computation_type) {
  mlir::MlirBuilder& builder = data.getBuilder();
  const mlir::Type index_type = GetTensorTypeOrDie(lengths).getElementType();
  mlir::MlirOp cumsum_exclusive =
      mlir::stablehlo::Subtract(boundary_indices, lengths);

  // Compares each index of `data` along `axis` with the boundaries of each
  // segment, in a tensor of the dimensions of `data` up to `axis` followed by
  // the segment dimension of `lengths`.
  const mlir::stablehlo::Dimensions data_dims = GetDimensions(data);
  mlir::stablehlo::Dimensions comp_dims(data_dims.begin(),
                                        data_dims.begin() + axis + 1);
  comp_dims.push_back(GetDimensions(lengths)[axis]);

  Dimensions segment_bcast_dims(axis + 1);
  std::iota(segment_bcast_dims.begin(), segment_bcast_dims.end(), 0);
  segment_bcast_dims[axis] = axis + 1;
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp c_exc_bcst,
      BroadcastIfNeeded(cumsum_exclusive, comp_dims, segment_bcast_dims));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp c_inc_bcst,
      BroadcastIfNeeded(boundary_indices, comp_dims, segment_bcast_dims));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp iota_bcst,
                      Iota(c_exc_bcst, /*iota_dimension=*/axis, index_type));

  mlir::MlirOp is_ge = mlir::stablehlo::Compare(
      iota_bcst, c_exc_bcst, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp is_lt = mlir::stablehlo::Compare(
      iota_bcst, c_inc_bcst, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp in_segment = mlir::stablehlo::And(is_ge, is_lt);
  mlir::MlirOp in_segment_idx =
      mlir::stablehlo::ConvertElementType(in_segment, index_type);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp k_bcst,
      Iota(c_exc_bcst, /*iota_dimension=*/axis + 1, index_type));

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
                      BroadcastIfNeeded(is_valid, data_dims, valid_bcast_dims));

  at::Scalar neutral_scalar = GetNeutralScalar(reduce_op, computation_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp neutral_constant,
                      MakeConstantLike(data, neutral_scalar));
  mlir::MlirOp data_masked =
      mlir::stablehlo::Select(is_valid_bcast, data, neutral_constant);

  return SegmentIdsAndMaskedData{segment_ids, data_masked};
}

absl::StatusOr<mlir::MlirOp> ComputeScatterIndices(mlir::MlirOp segment_ids,
                                                   int64_t axis) {
  const mlir::Type index_type =
      GetTensorTypeOrDie(segment_ids).getElementType();
  // Each coordinate has the dimensions of `segment_ids` followed by an index
  // vector dimension of size 1.
  mlir::stablehlo::Dimensions coord_dims = GetDimensions(segment_ids);
  coord_dims.push_back(mlir::stablehlo::DimensionInfo{1});
  Dimensions coord_bcast_dims(axis + 1);
  std::iota(coord_bcast_dims.begin(), coord_bcast_dims.end(), 0);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp segment_coord,
      BroadcastIfNeeded(segment_ids, coord_dims, coord_bcast_dims));
  if (axis == 0) {
    return segment_coord;
  }

  std::vector<mlir::MlirOp> coords;
  coords.reserve(axis + 1);
  for (int64_t d = 0; d < axis; ++d) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp d_iota,
                        Iota(segment_ids, /*iota_dimension=*/d, index_type));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp d_coord,
        BroadcastIfNeeded(d_iota, coord_dims, coord_bcast_dims));
    coords.push_back(d_coord);
  }
  coords.push_back(segment_coord);
  return mlir::stablehlo::Concatenate(segment_ids.getBuilder(), coords,
                                      /*dimension=*/axis + 1);
}

absl::StatusOr<mlir::MlirOp> ExecuteScatterReduction(
    mlir::MlirBuilder& builder, mlir::MlirOp data_masked,
    mlir::MlirOp scatter_indices, SegmentReduceOp reduce_op,
    mlir::Type computation_type, mlir::ElementType comp_elem_type,
    const mlir::stablehlo::Dimensions& init_dims, int64_t axis, int64_t rank,
    std::optional<mlir::MlirOp> initial_op) {
  at::Scalar neutral_scalar = GetNeutralScalar(reduce_op, computation_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp neutral,
                      MakeConstant(builder, neutral_scalar, comp_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp init_output,
                      BroadcastIfNeeded(neutral, init_dims));

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
    TT_ASSIGN_OR_RETURN(scatter_res,
                        ApplyInitialValue(scatter_res, reduce_op, *initial_op));
  }

  return scatter_res;
}

absl::StatusOr<mlir::MlirOp> PostProcessMeanReduction(
    mlir::MlirOp scatter_res, mlir::MlirOp lengths, int64_t axis,
    mlir::Type computation_type, std::optional<mlir::MlirOp> initial_op) {
  Dimensions bcast_dims_lengths(axis + 1);
  std::iota(bcast_dims_lengths.begin(), bcast_dims_lengths.end(), 0);

  mlir::MlirOp lengths_conv =
      mlir::stablehlo::ConvertElementType(lengths, computation_type);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp bcast_lengths,
      BroadcastIfNeeded(lengths_conv, scatter_res, bcast_dims_lengths));

  if (initial_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp initial_or_nan,
                        BroadcastIfNeeded(*initial_op, scatter_res));
    mlir::MlirOp sum = mlir::stablehlo::Add(scatter_res, initial_or_nan);
    mlir::MlirOp zero = MakeConstantLike(lengths, 0);
    mlir::MlirOp is_empty_mask = mlir::stablehlo::Compare(
        lengths, zero, mlir::stablehlo::ComparisonDirection::EQ);
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp is_empty,
        BroadcastIfNeeded(is_empty_mask, scatter_res, bcast_dims_lengths));
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
  const mlir::Type computation_type = data_type.getElementType();
  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_elem_type,
                      ConvertTo<mlir::ElementType>(computation_type));

  TT_ASSIGN_OR_RETURN(
      const auto boundaries_and_lengths,
      ComputeSegmentBoundariesAndLengths(lengths_opt, offsets_opt, axis));
  const auto& [boundary_indices, lengths] = boundaries_and_lengths;

  TT_ASSIGN_OR_RETURN(
      const auto ids_and_masked,
      ComputeSegmentIdsAndMaskedData(data, boundary_indices, lengths, axis,
                                     reduce_op, computation_type));
  const auto& [segment_ids, data_masked] = ids_and_masked;

  TT_ASSIGN_OR_RETURN(mlir::MlirOp scatter_indices,
                      ComputeScatterIndices(segment_ids, axis));

  // The output has the dimensions of `data`, except for the segment dimension
  // of `lengths` along `axis`.
  mlir::stablehlo::Dimensions init_dims = GetDimensions(data);
  init_dims[axis] = GetDimensions(lengths)[axis];

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp scatter_res,
      ExecuteScatterReduction(builder, data_masked, scatter_indices, reduce_op,
                              computation_type, comp_elem_type, init_dims, axis,
                              rank, initial_op));

  if (reduce_op == SegmentReduceOp::kMean) {
    return PostProcessMeanReduction(scatter_res, lengths, axis,
                                    computation_type, initial_op);
  }

  return scatter_res;
}

}  // namespace

at::Tensor AtenSegmentReduce(const at::Tensor& data, std::string_view reduce,
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

        TT_CHECK_THROW(!IsIntegral(data), error::kPythonNotImplementedError)
            << "not implemented for " << ToString(data.scalar_type());

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

        const int64_t num_segments =
            lengths.has_value() ? lengths->size(-1) : offsets->size(-1) - 1;

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

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef out_buf,
            has_initial ? (DispatchOp<3>(
                              std::move(op_builder),
                              {data, segment_info, *initial_tensor},
                              {.out_dtype = element_type,
                               .out_dims = output_dims,
                               .op_param_cache_keys = std::move(param_keys)}))
                        : (DispatchOp<2>(
                              std::move(op_builder), {data, segment_info},
                              {.out_dtype = element_type,
                               .out_dims = output_dims,
                               .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(out_buf);
      });
}

}  // namespace torch_tpu
