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

#include "csrc/ops/reductions/mean_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

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

at::Tensor& AtenMeanOut(const at::Tensor& self,
                        c10::OptionalArrayRef<int64_t> dim, bool keep_dim,
                        std::optional<c10::ScalarType> dtype, at::Tensor& out) {
  TT_ASSIGN_OR_THROW(Dimensions canonical_dims, CanonicalizeDims(self, dim));

  TT_KERNEL(
      OpName::kMeanOut, param_keys,
      (self, canonical_dims, keep_dim, dtype, out), {
        if (self.scalar_type() == at::ScalarType::Float4_e2m1fn_x2) {
          TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(self));
        }
        if (!dtype.has_value()) {
          const c10::ScalarType self_dtype = self.scalar_type();
          // Allow float8_* inputs here so `mean(fp8, out=f32)` works when the
          // output dtype is not low-precision float; low-precision outputs are
          // still rejected below and by UnaryOpOut.
          TT_CHECK_THROW(
              c10::isFloatingType(self_dtype) || c10::isComplexType(self_dtype),
              error::kInvalidArgument)
              << "expected a floating point or complex dtype, got "
              << ToString(self_dtype);
        }
        const c10::ScalarType scalar_dtype = dtype.value_or(out.scalar_type());
        TT_THROW_IF_ERROR(ValidateFloatOrComplex(scalar_dtype));

        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonical_dims);

        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_type,
                           ConvertTo<mlir::ElementType>(scalar_dtype));

        const auto comp_type = ToAccumulateType(scalar_dtype);
        TT_ASSIGN_OR_THROW(const auto computation_dtype,
                           ConvertTo<mlir::ElementType>(comp_type));

        auto op_builder =
            [canonical_dims = std::move(canonical_dims), reduction_mode,
             computation_dtype](
                mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
          return BuildMeanShlo(input_op, canonical_dims, reduction_mode,
                               computation_dtype);
        };

        TT_THROW_IF_ERROR(UnaryOpOut(
            self, out, std::move(op_builder),
            {.op_param_cache_keys = std::move(param_keys),
             .out_dtype = mlir_type,
             .out_dims = std::move(output_dims),
             .computation_dtype = computation_dtype,
             .allow_out_dtype_cast = false,
             .allow_low_precision_float = !IsLowPrecisionFloat(scalar_dtype) &&
                                          !IsLowPrecisionFloat(out)}));
        return out;
      });
}

}  // namespace torch_tpu
