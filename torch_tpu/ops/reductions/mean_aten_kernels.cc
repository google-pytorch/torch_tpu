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

#include "torch_tpu/ops/reductions/mean_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/full.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

namespace {

// Get the output scalar type and throw if it is float or complex.
// Types comes from dtype if it has a value, otherwise it is tensor's type.
absl::StatusOr<c10::ScalarType> GetOutputScalarType(
    const at::Tensor& tensor, std::optional<c10::ScalarType> dtype) {
  c10::ScalarType scalar_dtype = dtype.value_or(tensor.scalar_type());
  TT_RETURN_IF_ERROR(CheckFloatOrComplex(scalar_dtype));
  return scalar_dtype;
}

absl::StatusOr<mlir::MlirOp> BuildMeanShlo(
    mlir::MlirOp input, absl::Span<const int64_t> canonical_dims,
    ReductionMode reduction_mode, mlir::ElementType mlir_type) {
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp sum_op,
      BuildSumShlo(input, canonical_dims, reduction_mode, mlir_type));
  mlir::MlirOp num_elements =
      GetNumElements(input, mlir::getElementType(input.getContext(), mlir_type),
                     canonical_dims);
  return BuildDivShlo(sum_op, num_elements);
}

}  // namespace

at::Tensor& AtenMeanOut(const at::Tensor& self,
                        c10::OptionalArrayRef<int64_t> dim, bool keep_dim,
                        std::optional<c10::ScalarType> dtype, at::Tensor& out) {
  TT_KERNEL(
      OpName::kMeanOut, _,
      (self, IgnoreInCacheKey(dim, "Legacy usage"),
       IgnoreInCacheKey(keep_dim, "Legacy usage"),
       IgnoreInCacheKey(dtype, "Legacy usage"), out),
      {
        TT_ASSIGN_OR_THROW(c10::ScalarType scalar_dtype,
                           GetOutputScalarType(out, dtype));
        if (self.numel() == 0) {
          at::full_out(out, {}, std::numeric_limits<double>::quiet_NaN());
          return out;
        }

        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        TT_ASSIGN_OR_THROW(Dimensions canonical_dims,
                           CanonicalizeDims(self, dim));
        Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonical_dims);

        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_type,
                           ConvertTo<mlir::ElementType>(scalar_dtype));

        TT_ASSIGN_OR_THROW(
            auto param_keys,
            TT_MAKE_OP_PARAM_CACHE_KEYS(canonical_dims, keep_dim, dtype));

        auto op_builder =
            [canonical_dims = std::move(canonical_dims), reduction_mode,
             mlir_type](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
          return BuildMeanShlo(input_op, canonical_dims, reduction_mode,
                               mlir_type);
        };

        TT_THROW_IF_ERROR(
            UnaryOpOut(self, out, std::move(op_builder),
                       {.op_name = OpName::kMeanOut,
                        .op_param_cache_keys = std::move(param_keys),
                        .out_dtype = mlir_type,
                        .out_dims = std::move(output_dims)}));
        return out;
      });
}

}  // namespace torch_tpu
