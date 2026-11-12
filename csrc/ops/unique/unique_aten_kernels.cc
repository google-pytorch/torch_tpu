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

#include "csrc/ops/unique/unique_aten_kernels.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "csrc/ops/unique/unique.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::StatusOr<int64_t> ExtractOutputSize(const at::Tensor& count_tensor) {
  at::Tensor count_cpu = at::empty(
      {}, at::TensorOptions().dtype(c10::ScalarType::Long).device(at::kCPU));
  TT_RETURN_IF_ERROR(CopyTpuToCpu(count_tensor, count_cpu,
                                  GetTransferSyncMode(/*non_blocking=*/false)));
  return count_cpu.item<int64_t>();
}

absl::StatusOr<at::Tensor> GetUniqueCount(const at::Tensor& self, bool sorted) {
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      auto unique_count,
      UnaryOp(self,
              [element_type,
               sorted](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
                return BuildUniqueGetOutputSizeShlo(element_type, input,
                                                    sorted);
              },
              {.op_param_cache_keys = OpParamCacheKeys::Empty(),
               .out_dtype = mlir::ElementType::I64,
               .out_dims = at::IntArrayRef()}));
  return unique_count;
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>> Unique2(
    const at::Tensor& self, int64_t output_size, bool sorted,
    bool return_inverse, bool return_counts) {
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(
                          output_size, sorted, return_inverse, return_counts));

  TT_ASSIGN_OR_RETURN(auto val_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(auto i64_type,
                      ConvertTo<mlir::ElementType>(c10::ScalarType::Long));

  std::array<mlir::ElementType, 3> dtypes = {val_type, i64_type, i64_type};
  FixedSizeSpan<const mlir::ElementType, 3> out_dtypes(dtypes);

  Dimensions values_dims = {output_size};
  Dimensions inverse_dims =
      return_inverse ? CopyIntVector(self.sizes()) : Dimensions({0});
  Dimensions counts_dims =
      return_counts ? Dimensions({output_size}) : Dimensions({0});

  std::array<absl::Span<const int64_t>, 3> dims = {
      absl::Span<const int64_t>(values_dims),
      absl::Span<const int64_t>(inverse_dims),
      absl::Span<const int64_t>(counts_dims)};
  FixedSizeSpan<const absl::Span<const int64_t>, 3> out_dims_list(dims);

  auto shlo_builder =
      [output_size, sorted, return_inverse, return_counts](
          mlir::MlirOp input) -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
    TT_ASSIGN_OR_RETURN(auto outputs,
                        BuildUnique2Shlo(output_size, input, sorted,
                                         return_inverse, return_counts));
    return std::array<mlir::MlirOp, 3>{outputs.unique_values,
                                       outputs.inverse_indices, outputs.counts};
  };

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<1, 3>(std::move(shlo_builder), self,
                        {.out_dtypes = out_dtypes,
                         .out_dims_list = out_dims_list,
                         .op_param_cache_keys = std::move(param_keys)})));

  return std::make_tuple(MakeTensor(results[0]), MakeTensor(results[1]),
                         MakeTensor(results[2]));
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>> UniqueDimPrepare(
    const at::Tensor& self, const int64_t dim) {
  TT_ASSIGN_OR_RETURN(auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(dim));

  const std::array<mlir::ElementType, 3> dtypes = {
      mlir::ElementType::I64, mlir::ElementType::I64, mlir::ElementType::PRED};
  const FixedSizeSpan<const mlir::ElementType, 3> out_dtypes(dtypes);

  const int64_t s = self.size(dim);
  const Dimensions count_dims = {};
  const Dimensions s_dims = {s};
  const std::array<absl::Span<const int64_t>, 3> dims = {
      absl::Span<const int64_t>(count_dims), absl::Span<const int64_t>(s_dims),
      absl::Span<const int64_t>(s_dims)};
  const FixedSizeSpan<const absl::Span<const int64_t>, 3> out_dims_list(dims);

  const auto shlo_builder =
      [dim](mlir::MlirOp input) -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
    TT_ASSIGN_OR_RETURN(const auto outputs,
                        BuildUniqueDimPrepareShlo(input, dim));
    return std::array<mlir::MlirOp, 3>{outputs.count, outputs.p_indices,
                                       outputs.mask};
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

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
UniqueDimFinalize(const at::Tensor& self, const at::Tensor& p_indices,
                  const at::Tensor& mask, const int64_t dim,
                  const int64_t output_size, const ReturnInverse return_inverse,
                  const ReturnCounts return_counts) {
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(output_size, return_inverse,
                                                  return_counts, dim));

  TT_ASSIGN_OR_RETURN(const auto val_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  const std::array<mlir::ElementType, 3> dtypes = {
      val_type, mlir::ElementType::I64, mlir::ElementType::I64};
  const FixedSizeSpan<const mlir::ElementType, 3> out_dtypes(dtypes);

  Dimensions values_dims = CopyIntVector(self.sizes());
  values_dims[dim] = output_size;
  const Dimensions inverse_dims = return_inverse == ReturnInverse::kTrue
                                      ? Dimensions({self.size(dim)})
                                      : Dimensions({0});
  const Dimensions counts_dims = return_counts == ReturnCounts::kTrue
                                     ? Dimensions({output_size})
                                     : Dimensions({0});

  const std::array<absl::Span<const int64_t>, 3> dims = {
      absl::Span<const int64_t>(values_dims),
      absl::Span<const int64_t>(inverse_dims),
      absl::Span<const int64_t>(counts_dims)};
  const FixedSizeSpan<const absl::Span<const int64_t>, 3> out_dims_list(dims);

  auto shlo_builder = [output_size, dim, return_inverse,
                       return_counts](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
    TT_ASSIGN_OR_RETURN(
        const auto outputs,
        BuildUniqueDimFinalizeShlo(output_size, inputs[0], inputs[1], inputs[2],
                                   dim, return_inverse, return_counts));
    return std::array<mlir::MlirOp, 3>{outputs.unique_values,
                                       outputs.inverse_indices, outputs.counts};
  };

  TT_ASSIGN_OR_RETURN(
      const auto results,
      (DispatchOp<3, 3>(std::move(shlo_builder), {self, p_indices, mask},
                        {.out_dtypes = out_dtypes,
                         .out_dims_list = out_dims_list,
                         .op_param_cache_keys = std::move(param_keys)})));

  return std::make_tuple(MakeTensor(results[0]), MakeTensor(results[1]),
                         MakeTensor(results[2]));
}

void ValidateUniqueDimInputs(const at::Tensor& self,
                             const int64_t wrapped_dim) {
  TT_CHECK_THROW(!IsComplex(self), error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());
  TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(self));

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

std::tuple<at::Tensor, at::Tensor, at::Tensor> UniqueDimHelper(
    const at::Tensor& self, const int64_t dim,
    const ReturnInverse return_inverse, const ReturnCounts return_counts) {
  TT_CHECK_THROW(self.dim() > 0, error::kIndexError)
      << "expected tensor with at least 1 dimension, got " << self.dim();
  TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, self.dim()));
  ValidateUniqueDimInputs(self, wrapped_dim);

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

  TT_ASSIGN_OR_THROW(const auto prep, UniqueDimPrepare(self, wrapped_dim));
  const auto& [count, p_indices, mask] = prep;
  TT_ASSIGN_OR_THROW(const int64_t output_size, ExtractOutputSize(count));

  TT_ASSIGN_OR_THROW(
      const auto result,
      UniqueDimFinalize(self, p_indices, mask, wrapped_dim, output_size,
                        return_inverse, return_counts));
  return result;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenUnique2(
    const at::Tensor& self, bool sorted, bool return_inverse,
    bool return_counts) {
  TT_KERNEL(OpName::kUnique2, _,
            (self, IgnoreInCacheKey(sorted, "delegates to Unique2"),
             IgnoreInCacheKey(return_inverse, "delegates to Unique2"),
             IgnoreInCacheKey(return_counts, "delegates to Unique2")),
            {
              TT_CHECK_THROW(sorted, error::kInvalidArgument)
                  << "sorted=False is not yet supported";

              TT_ASSIGN_OR_THROW(at::Tensor unique_count_tensor,
                                 GetUniqueCount(self, sorted));
              // Unlike cpu(), item() invalidates a graph capture, as unique
              // does on CUDA.
              const int64_t output_size = unique_count_tensor.item<int64_t>();

              TT_ASSIGN_OR_THROW(auto result,
                                 Unique2(self, output_size, sorted,
                                         return_inverse, return_counts));
              return result;
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenUniqueDim(
    const at::Tensor& self, const int64_t dim, const bool sorted,
    const bool return_inverse, const bool return_counts) {
  TT_KERNEL(OpName::kUniqueDim, _,
            (self, IgnoreInCacheKey(dim, "delegates to UniqueDimHelper"),
             IgnoreInCacheKey(sorted, "does not affect SHLO"),
             IgnoreInCacheKey(return_inverse, "delegates to UniqueDimHelper"),
             IgnoreInCacheKey(return_counts, "delegates to UniqueDimHelper")),
            {
              return UniqueDimHelper(
                  self, dim,
                  return_inverse ? ReturnInverse::kTrue : ReturnInverse::kFalse,
                  return_counts ? ReturnCounts::kTrue : ReturnCounts::kFalse);
            });
}

}  // namespace torch_tpu
