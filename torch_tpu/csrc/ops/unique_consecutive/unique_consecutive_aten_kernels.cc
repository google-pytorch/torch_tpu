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

#include "torch_tpu/csrc/ops/unique_consecutive/unique_consecutive_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/copy_from/tpu_to_cpu.h"
#include "torch_tpu/csrc/ops/index_select/index_select.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/unary_aten_kernels.h"

namespace torch_tpu {

namespace {

struct BuildUniqueConsecutiveShloOutputs {
  mlir::MlirOp unique_values;
  mlir::MlirOp inverse_indices;
  mlir::MlirOp counts;
};

mlir::MlirOp FlattenIfNeeded(mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  if (input_type.getRank() == 1) {
    return input;
  }
  return mlir::stablehlo::Reshape(input, {input_type.getNumElements()});
}

absl::StatusOr<mlir::MlirOp> BuildUniqueConsecutiveMask1D(
    mlir::MlirBuilder& builder, mlir::MlirOp flat_input) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(flat_input);
  const int64_t n = type.getDimSize(0);
  if (n == 0) {
    return MakeZeroSizedTensor(builder, mlir::ElementType::PRED);
  }
  if (n == 1) {
    return MakeConstant(builder, 1, mlir::ElementType::PRED, {1});
  }

  mlir::MlirOp first_elem_mask =
      MakeConstant(builder, 1, mlir::ElementType::PRED, {1});
  mlir::MlirOp slice_tail = mlir::stablehlo::Slice(flat_input, {1}, {n}, {1});
  mlir::MlirOp slice_head =
      mlir::stablehlo::Slice(flat_input, {0}, {n - 1}, {1});
  mlir::MlirOp diff_mask = mlir::stablehlo::Compare(
      slice_tail, slice_head, mlir::stablehlo::ComparisonDirection::NE);
  return mlir::stablehlo::Concatenate(builder, {first_elem_mask, diff_mask}, 0);
}

absl::StatusOr<mlir::MlirOp> BuildUniqueConsecutiveMaskDim(
    mlir::MlirBuilder& builder, mlir::MlirOp input, const int64_t dim) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  const int64_t rank = type.getRank();
  const int64_t s = type.getDimSize(dim);
  if (s == 0) {
    return MakeZeroSizedTensor(builder, mlir::ElementType::PRED);
  }
  if (s == 1) {
    return MakeConstant(builder, 1, mlir::ElementType::PRED, {1});
  }

  mlir::MlirOp input_dim0 = input;
  if (dim != 0) {
    Dimensions permutation(rank);
    permutation[0] = dim;
    for (int64_t i = 0, j = 1; i < rank; ++i) {
      if (i != dim) {
        permutation[j++] = i;
      }
    }
    input_dim0 = mlir::stablehlo::Transpose(input, permutation);
  }

  int64_t slice_elements = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      slice_elements *= type.getDimSize(i);
    }
  }

  mlir::MlirOp input_2d =
      mlir::stablehlo::Reshape(input_dim0, {s, slice_elements});
  mlir::MlirOp slice_tail =
      mlir::stablehlo::Slice(input_2d, {1, 0}, {s, slice_elements}, {1, 1});
  mlir::MlirOp slice_head =
      mlir::stablehlo::Slice(input_2d, {0, 0}, {s - 1, slice_elements}, {1, 1});

  mlir::MlirOp diff_2d = mlir::stablehlo::Compare(
      slice_tail, slice_head, mlir::stablehlo::ComparisonDirection::NE);

  mlir::MlirOp init_or =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI1Type());
  mlir::MlirOp diff_mask = mlir::stablehlo::Reduce(
      builder, {diff_2d}, {init_or},
      [](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::OrOp>(
            body_builder.getI1Type(), rb.getRegion(), body_builder);
      },
      {1})[0];

  mlir::MlirOp first_elem_mask =
      MakeConstant(builder, 1, mlir::ElementType::PRED, {1});
  return mlir::stablehlo::Concatenate(builder, {first_elem_mask, diff_mask}, 0);
}

absl::StatusOr<mlir::MlirOp> BuildCumsum1D(mlir::MlirBuilder& builder,
                                           mlir::MlirOp input) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  const int64_t n = type.getDimSize(0);
  if (n == 0) {
    return input;
  }

  const auto window_dimensions =
      mlir::DenseI64ArrayAttr::get(&builder.getContext(), {n});
  const auto window_strides =
      mlir::DenseI64ArrayAttr::get(&builder.getContext(), {1});

  const mlir::SmallVector<int64_t, 2> padding_values = {n - 1, 0};
  const auto padding = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({1, 2}, builder.getOpBuilder().getI64Type()),
      llvm::ArrayRef<int64_t>(padding_values));

  const auto body_builder = [](mlir::RegionBuilder& rb) {
    auto& b = rb.getOpBuilder();
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(b.getI64Type(),
                                                             rb.getRegion(), b);
  };

  mlir::MlirOp init_value =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI64Type());
  return mlir::stablehlo::ReduceWindow(builder, {input}, {init_value},
                                       body_builder, window_dimensions,
                                       window_strides,
                                       /*base_dilations=*/{},
                                       /*window_dilations=*/{}, padding)[0];
}

mlir::MlirOp ExtractSortedUniqueIndices(mlir::MlirBuilder& builder,
                                        mlir::MlirOp mask,
                                        const int64_t total_elements,
                                        const int64_t output_size) {
  const auto i64_type = builder.getOpBuilder().getI64Type();
  const auto mask_sort_comparator = [i64_type](mlir::RegionBuilder& rb) {
    auto& body_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {body_builder.getI1Type(), i64_type},
        mlir::stablehlo::ComparisonDirection::GE, std::nullopt, &rb.getRegion(),
        &body_builder);
  };

  const auto s_iota_type =
      mlir::makeTensorType(builder.getContext(), {total_elements}, i64_type);
  mlir::MlirOp s_iota = mlir::stablehlo::Iota(builder, s_iota_type, 0);
  // Sorts by mask to push unique indices to the front (is_stable=false);
  // a second sort below restores original chronological index order.
  auto unique_indices_sort = mlir::stablehlo::Sort(
      builder, {mask, s_iota}, mask_sort_comparator, 0, /*is_stable=*/false);
  mlir::MlirOp s_unique_indices =
      mlir::stablehlo::Slice(unique_indices_sort[1], {0}, {output_size}, {1});

  const auto i64_lt_comparator = [i64_type](mlir::RegionBuilder& rb) {
    auto& body_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {i64_type}, mlir::stablehlo::ComparisonDirection::LT, std::nullopt,
        &rb.getRegion(), &body_builder);
  };
  return mlir::stablehlo::Sort(builder, {s_unique_indices}, i64_lt_comparator,
                               0, /*is_stable=*/false)[0];
}

absl::StatusOr<mlir::MlirOp> BuildInverseIndices(
    mlir::MlirBuilder& builder, mlir::MlirOp mask,
    const mlir::RankedTensorType& original_type, std::optional<int64_t> dim) {
  const auto i64_type = builder.getOpBuilder().getI64Type();
  mlir::MlirOp mask_i64 = mlir::stablehlo::ConvertElementType(mask, i64_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp cumsum_mask_full,
                      BuildCumsum1D(builder, mask_i64));
  mlir::MlirOp one_cst = MakeConstantLike(cumsum_mask_full, 1LL);
  mlir::MlirOp inverse_indices =
      mlir::stablehlo::Subtract(cumsum_mask_full, one_cst);

  if (!dim.has_value() && original_type.getRank() > 1) {
    inverse_indices =
        mlir::stablehlo::Reshape(inverse_indices, original_type.getShape());
  }
  return inverse_indices;
}

mlir::MlirOp BuildCounts(mlir::MlirBuilder& builder,
                         mlir::MlirOp unique_indices,
                         const int64_t total_elements,
                         const int64_t output_size) {
  mlir::MlirOp n_cst =
      MakeConstant(builder, total_elements, mlir::ElementType::I64, {1});
  mlir::MlirOp unique_indices_plus_n =
      mlir::stablehlo::Concatenate(builder, {unique_indices, n_cst}, 0);
  mlir::MlirOp c_start = mlir::stablehlo::Slice(unique_indices_plus_n, {1},
                                                {output_size + 1}, {1});
  mlir::MlirOp c_end =
      mlir::stablehlo::Slice(unique_indices_plus_n, {0}, {output_size}, {1});
  return mlir::stablehlo::Subtract(c_start, c_end);
}

absl::StatusOr<mlir::MlirOp> BuildUniqueConsecutiveGetOutputSizeShlo(
    mlir::MlirOp input, std::optional<int64_t> dim) {
  auto& builder = input.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  if (input_type.getRank() == 0) {
    return MakeConstant(builder, 1, mlir::ElementType::I64, {});
  }

  mlir::MlirOp mask;
  if (dim.has_value()) {
    TT_ASSIGN_OR_RETURN(mask,
                        BuildUniqueConsecutiveMaskDim(builder, input, *dim));
  } else {
    mlir::MlirOp flattened_input = FlattenIfNeeded(input);
    TT_ASSIGN_OR_RETURN(mask,
                        BuildUniqueConsecutiveMask1D(builder, flattened_input));
  }

  mlir::MlirOp mask_i64 = mlir::stablehlo::ConvertElementType(
      mask, builder.getOpBuilder().getI64Type());
  mlir::MlirOp init_value =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI64Type());
  return mlir::stablehlo::Reduce(
      builder, {mask_i64}, {init_value},
      [](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            body_builder.getI64Type(), rb.getRegion(), body_builder);
      },
      {0})[0];
}

absl::StatusOr<BuildUniqueConsecutiveShloOutputs>
BuildUniqueConsecutiveScalarShlo(mlir::MlirBuilder& builder, mlir::MlirOp input,
                                 const bool return_inverse,
                                 const bool return_counts) {
  BuildUniqueConsecutiveShloOutputs outputs;
  outputs.unique_values = mlir::stablehlo::Reshape(input, {1});
  if (return_inverse) {
    outputs.inverse_indices =
        MakeConstant(builder, 0, mlir::ElementType::I64, {});
  } else {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }
  if (return_counts) {
    outputs.counts = MakeConstant(builder, 1, mlir::ElementType::I64, {1});
  } else {
    TT_ASSIGN_OR_RETURN(outputs.counts,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }
  return outputs;
}

absl::StatusOr<BuildUniqueConsecutiveShloOutputs>
BuildUniqueConsecutiveEmptyShlo(mlir::MlirBuilder& builder,
                                const mlir::RankedTensorType& input_type,
                                const mlir::ElementType val_type,
                                const bool return_inverse,
                                std::optional<int64_t> dim) {
  BuildUniqueConsecutiveShloOutputs outputs;
  TT_ASSIGN_OR_RETURN(outputs.counts,
                      MakeZeroSizedTensor(builder, mlir::ElementType::I64));

  if (!dim.has_value()) {
    TT_ASSIGN_OR_RETURN(outputs.unique_values,
                        MakeZeroSizedTensor(builder, val_type));
    if (return_inverse) {
      TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                          MakeZeroSizedTensor(builder, mlir::ElementType::I64,
                                              input_type.getShape()));
    } else {
      TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                          MakeZeroSizedTensor(builder, mlir::ElementType::I64));
    }
    return outputs;
  }

  Dimensions out_shape(input_type.getShape().begin(),
                       input_type.getShape().end());
  out_shape[*dim] = 0;
  TT_ASSIGN_OR_RETURN(outputs.unique_values,
                      MakeZeroSizedTensor(builder, val_type, out_shape));
  TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                      MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  return outputs;
}

absl::StatusOr<BuildUniqueConsecutiveShloOutputs> BuildUniqueConsecutiveDimShlo(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    const mlir::RankedTensorType& input_type, const int64_t dim,
    const int64_t output_size, const bool return_inverse,
    const bool return_counts) {
  BuildUniqueConsecutiveShloOutputs outputs;
  const int64_t s = input_type.getDimSize(dim);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildUniqueConsecutiveMaskDim(builder, input, dim));

  mlir::MlirOp unique_indices =
      ExtractSortedUniqueIndices(builder, mask, s, output_size);
  outputs.unique_values = BuildIndexSelectShlo(input, dim, unique_indices);

  if (return_inverse) {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        BuildInverseIndices(builder, mask, input_type, dim));
  } else {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }

  if (return_counts) {
    outputs.counts = BuildCounts(builder, unique_indices, s, output_size);
  } else {
    TT_ASSIGN_OR_RETURN(outputs.counts,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }
  return outputs;
}

absl::StatusOr<BuildUniqueConsecutiveShloOutputs>
BuildUniqueConsecutiveFlattenedShlo(mlir::MlirBuilder& builder,
                                    mlir::MlirOp input,
                                    const mlir::RankedTensorType& input_type,
                                    const int64_t output_size,
                                    const bool return_inverse,
                                    const bool return_counts) {
  BuildUniqueConsecutiveShloOutputs outputs;
  mlir::MlirOp flat_input = FlattenIfNeeded(input);
  const int64_t n = input_type.getNumElements();

  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildUniqueConsecutiveMask1D(builder, flat_input));
  mlir::MlirOp unique_indices =
      ExtractSortedUniqueIndices(builder, mask, n, output_size);

  mlir::MlirOp gather_indices =
      mlir::stablehlo::Reshape(unique_indices, {output_size, 1});
  auto dnums = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      &builder.getContext(),
      /*offsetDims=*/{},
      /*collapsedSliceDims=*/{0},
      /*operandBatchingDims=*/{},
      /*startIndicesBatchingDims=*/{},
      /*startIndexMap=*/{0},
      /*indexVectorDim=*/1);

  outputs.unique_values =
      mlir::stablehlo::Gather(flat_input, gather_indices, dnums, {1});

  if (return_inverse) {
    TT_ASSIGN_OR_RETURN(
        outputs.inverse_indices,
        BuildInverseIndices(builder, mask, input_type, std::nullopt));
  } else {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }

  if (return_counts) {
    outputs.counts = BuildCounts(builder, unique_indices, n, output_size);
  } else {
    TT_ASSIGN_OR_RETURN(outputs.counts,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }
  return outputs;
}

absl::StatusOr<BuildUniqueConsecutiveShloOutputs> BuildUniqueConsecutiveShlo(
    const int64_t output_size, mlir::MlirOp input, const bool return_inverse,
    const bool return_counts, std::optional<int64_t> dim) {
  auto& builder = input.getBuilder();
  const mlir::RankedTensorType original_input_type = GetTensorTypeOrDie(input);
  const mlir::ElementType val_type = GetElementTypeOrDie(input);

  if (original_input_type.getRank() == 0) {
    return BuildUniqueConsecutiveScalarShlo(builder, input, return_inverse,
                                            return_counts);
  }
  if (original_input_type.getNumElements() == 0) {
    return BuildUniqueConsecutiveEmptyShlo(builder, original_input_type,
                                           val_type, return_inverse, dim);
  }
  if (dim.has_value()) {
    return BuildUniqueConsecutiveDimShlo(builder, input, original_input_type,
                                         *dim, output_size, return_inverse,
                                         return_counts);
  }
  return BuildUniqueConsecutiveFlattenedShlo(builder, input,
                                             original_input_type, output_size,
                                             return_inverse, return_counts);
}

absl::StatusOr<int64_t> ExtractOutputSize(const at::Tensor& count_tensor) {
  at::Tensor count_cpu = at::empty(
      {}, at::TensorOptions().dtype(c10::ScalarType::Long).device(at::kCPU));
  TT_RETURN_IF_ERROR(
      CopyTpuToCpu(count_tensor, count_cpu, /*non_blocking=*/false));
  return count_cpu.item<int64_t>();
}

void ValidateUniqueConsecutiveDimInputs(const at::Tensor& self,
                                        const int64_t wrapped_dim) {
  int64_t num_zero_dims = 0;
  for (const int64_t s : self.sizes()) {
    if (s == 0) {
      ++num_zero_dims;
    }
  }

  if (self.size(wrapped_dim) == 0) {
    TT_CHECK_THROW(num_zero_dims == 1, error::kInvalidArgument)
        << "expected at most 1 zero sized dimension when applying unique on a "
           "zero sized dimension, got "
        << num_zero_dims;
  } else {
    TT_CHECK_THROW(num_zero_dims == 0, error::kInvalidArgument)
        << "expected 0 unselected zero sized dimensions, got " << num_zero_dims;
  }
}

absl::StatusOr<at::Tensor> GetUniqueConsecutiveCount(
    const at::Tensor& self, std::optional<int64_t> dim) {
  TT_ASSIGN_OR_RETURN(auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(dim));
  TT_ASSIGN_OR_RETURN(
      const at::Tensor count_tensor,
      UnaryOp(self,
              [dim](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
                return BuildUniqueConsecutiveGetOutputSizeShlo(input, dim);
              },
              {.op_param_cache_keys = std::move(param_keys),
               .out_dtype = mlir::ElementType::I64,
               .out_dims = at::IntArrayRef()}));
  return count_tensor;
}

Dimensions ComputeValuesDimensions(const at::Tensor& self,
                                   const int64_t output_size,
                                   std::optional<int64_t> dim) {
  if (!dim.has_value()) {
    return Dimensions({output_size});
  }
  Dimensions values_dims = CopyIntVector(self.sizes());
  values_dims[*dim] = output_size;
  return values_dims;
}

Dimensions ComputeInverseDimensions(const at::Tensor& self,
                                    const bool return_inverse,
                                    std::optional<int64_t> dim) {
  if (!return_inverse) {
    return Dimensions({0});
  }
  if (!dim.has_value()) {
    return CopyIntVector(self.sizes());
  }
  return Dimensions({self.size(*dim)});
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
UniqueConsecutiveDispatch(const at::Tensor& self, const int64_t output_size,
                          const bool return_inverse, const bool return_counts,
                          std::optional<int64_t> dim) {
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(output_size, return_inverse,
                                                  return_counts, dim));

  TT_ASSIGN_OR_RETURN(const auto val_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto i64_type,
                      ConvertTo<mlir::ElementType>(c10::ScalarType::Long));

  const std::array<mlir::ElementType, 3> dtypes = {val_type, i64_type,
                                                   i64_type};
  const FixedSizeSpan<const mlir::ElementType, 3> out_dtypes(dtypes);

  const Dimensions values_dims =
      ComputeValuesDimensions(self, output_size, dim);
  const Dimensions inverse_dims =
      ComputeInverseDimensions(self, return_inverse, dim);
  const Dimensions counts_dims =
      return_counts ? Dimensions({output_size}) : Dimensions({0});

  const std::array<absl::Span<const int64_t>, 3> dims = {
      absl::Span<const int64_t>(values_dims),
      absl::Span<const int64_t>(inverse_dims),
      absl::Span<const int64_t>(counts_dims)};
  const FixedSizeSpan<const absl::Span<const int64_t>, 3> out_dims_list(dims);

  const auto shlo_builder =
      [output_size, return_inverse, return_counts,
       dim](mlir::MlirOp input) -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
    TT_ASSIGN_OR_RETURN(
        const auto outputs,
        BuildUniqueConsecutiveShlo(output_size, input, return_inverse,
                                   return_counts, dim));
    return std::array<mlir::MlirOp, 3>{outputs.unique_values,
                                       outputs.inverse_indices, outputs.counts};
  };

  TT_ASSIGN_OR_RETURN(
      const auto results,
      (DispatchOp<1, 3>(std::move(shlo_builder), self,
                        {.out_dtypes = out_dtypes,
                         .out_dims_list = out_dims_list,
                         .op_param_cache_keys = std::move(param_keys)})));

  return std::make_tuple(MakeTensor(results[0]), MakeTensor(results[1]),
                         MakeTensor(results[2]));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> UniqueConsecutiveDimHelper(
    const at::Tensor& self, const int64_t dim, const bool return_inverse,
    const bool return_counts) {
  TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, self.dim()));
  ValidateUniqueConsecutiveDimInputs(self, wrapped_dim);

  if (self.size(wrapped_dim) == 0) {
    TT_ASSIGN_OR_THROW(
        at::Tensor empty_values,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_ASSIGN_OR_THROW(
        at::Tensor empty_inverse,
        MakeEmptyTensor({0}, c10::ScalarType::Long, self.device()));
    TT_ASSIGN_OR_THROW(
        at::Tensor empty_counts,
        MakeEmptyTensor({0}, c10::ScalarType::Long, self.device()));
    return std::make_tuple(empty_values, empty_inverse, empty_counts);
  }

  TT_ASSIGN_OR_THROW(const at::Tensor count_tensor,
                     GetUniqueConsecutiveCount(self, wrapped_dim));
  TT_ASSIGN_OR_THROW(const int64_t output_size,
                     ExtractOutputSize(count_tensor));

  TT_ASSIGN_OR_THROW(const auto result, UniqueConsecutiveDispatch(
                                            self, output_size, return_inverse,
                                            return_counts, wrapped_dim));
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> UniqueConsecutiveFlattenedHelper(
    const at::Tensor& self, const bool return_inverse,
    const bool return_counts) {
  if (self.numel() == 0) {
    TT_ASSIGN_OR_THROW(at::Tensor empty_values,
                       MakeEmptyTensor({0}, self.scalar_type(), self.device()));
    TT_ASSIGN_OR_THROW(
        at::Tensor empty_inverse,
        MakeEmptyTensor(return_inverse ? self.sizes() : at::IntArrayRef({0}),
                        c10::ScalarType::Long, self.device()));
    TT_ASSIGN_OR_THROW(
        at::Tensor empty_counts,
        MakeEmptyTensor({0}, c10::ScalarType::Long, self.device()));
    return std::make_tuple(empty_values, empty_inverse, empty_counts);
  }

  TT_ASSIGN_OR_THROW(const at::Tensor count_tensor,
                     GetUniqueConsecutiveCount(self, std::nullopt));
  TT_ASSIGN_OR_THROW(const int64_t output_size,
                     ExtractOutputSize(count_tensor));

  TT_ASSIGN_OR_THROW(const auto result, UniqueConsecutiveDispatch(
                                            self, output_size, return_inverse,
                                            return_counts, std::nullopt));
  return result;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenUniqueConsecutive(
    const at::Tensor& self, const bool return_inverse, const bool return_counts,
    std::optional<int64_t> dim) {
  TT_KERNEL(OpName::kUniqueConsecutive, _,
            (self, IgnoreInCacheKey(return_inverse, "delegates to helper"),
             IgnoreInCacheKey(return_counts, "delegates to helper"),
             IgnoreInCacheKey(dim, "delegates to helper")),
            {
              if (dim.has_value()) {
                return UniqueConsecutiveDimHelper(self, *dim, return_inverse,
                                                  return_counts);
              }
              return UniqueConsecutiveFlattenedHelper(self, return_inverse,
                                                      return_counts);
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenUniqueDimConsecutive(
    const at::Tensor& self, const int64_t dim, const bool return_inverse,
    const bool return_counts) {
  TT_KERNEL(OpName::kUniqueDimConsecutive, _,
            (self, IgnoreInCacheKey(dim, "delegates to helper"),
             IgnoreInCacheKey(return_inverse, "delegates to helper"),
             IgnoreInCacheKey(return_counts, "delegates to helper")),
            {
              return UniqueConsecutiveDimHelper(self, dim, return_inverse,
                                                return_counts);
            });
}

}  // namespace torch_tpu
