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

#include "csrc/ops/count_nonzero/count_nonzero_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

// count_nonzero == sum(self != 0), with the count accumulated in int64.
absl::StatusOr<mlir::MlirOp> BuildCountNonzeroShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> reduce_dims) {
  mlir::MlirOp zero = MakeConstantLike(input_op, 0);
  mlir::MlirOp mask = mlir::stablehlo::Compare(
      input_op, zero, mlir::stablehlo::ComparisonDirection::NE);
  mlir::MlirOp mask_i64 =
      mlir::stablehlo::ConvertElementType(mask, mlir::ElementType::I64);
  return BuildSumShlo(mask_i64, reduce_dims, ReductionMode::kDropDims,
                      mlir::ElementType::I64);
}

}  // namespace

at::Tensor AtenCountNonzeroDimIntList(const at::Tensor& self,
                                      at::IntArrayRef dim) {
  TT_KERNEL(OpName::kCountNonzeroDimIntList, param_keys, (self, dim), {
    TT_ASSIGN_OR_THROW(Dimensions reduce_dims, CanonicalizeDims(self, dim));
    Dimensions out_dims = GetSizesAfterReduction(
        self.sizes(), ReductionMode::kDropDims, reduce_dims);

    MlirUnaryOpBuilder op_builder =
        [reduce_dims](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildCountNonzeroShlo(input, reduce_dims);
    };
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = mlir::ElementType::I64,
                       .out_dims = out_dims,
                       .op_param_cache_keys = std::move(param_keys)}));
    return MakeTensor(std::move(result_buf));
  });
}

}  // namespace torch_tpu
