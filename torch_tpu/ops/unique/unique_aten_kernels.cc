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

#include "torch_tpu/ops/unique/unique_aten_kernels.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"
#include "torch_tpu/ops/unique/unique.h"

namespace torch_tpu {

namespace {

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

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenUnique2(
    const at::Tensor& self, bool sorted, bool return_inverse,
    bool return_counts) {
  TT_KERNEL(
      OpName::kUnique2, _,
      (self, IgnoreInCacheKey(sorted, "Legacy usage"),
       IgnoreInCacheKey(return_inverse, "Legacy usage"),
       IgnoreInCacheKey(return_counts, "Legacy usage")),
      {
        TT_CHECK_THROW(sorted, error::kInvalidArgument)
            << "sorted=False is not yet supported";

        TT_ASSIGN_OR_THROW(at::Tensor unique_count_tensor,
                           GetUniqueCount(self, sorted));
        const int64_t output_size = unique_count_tensor.cpu().item<int64_t>();

        TT_ASSIGN_OR_THROW(auto result, Unique2(self, output_size, sorted,
                                                return_inverse, return_counts));
        return result;
      });
}

}  // namespace torch_tpu
