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

#include "csrc/ops/addcdiv/addcdiv_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildAddcdivShlo(mlir::MlirOp self,
                                              mlir::MlirOp tensor1,
                                              mlir::MlirOp tensor2,
                                              mlir::MlirOp value) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype, GetElementType(self));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(out_dtype));

  TT_ASSIGN_OR_RETURN(self, CastIfNeeded(self, compute_dtype));
  TT_ASSIGN_OR_RETURN(tensor1, CastIfNeeded(tensor1, compute_dtype));
  TT_ASSIGN_OR_RETURN(tensor2, CastIfNeeded(tensor2, compute_dtype));
  TT_ASSIGN_OR_RETURN(value, CastIfNeeded(value, compute_dtype));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp div, BuildDivShlo(tensor1, tensor2));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp value_div, BuildMulShlo(value, div));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp result, BuildAddShlo(self, value_div));
  return CastIfNeeded(result, out_dtype);
}

}  // namespace

at::Tensor& AtenAddcdivOut(const at::Tensor& self, const at::Tensor& tensor1,
                           const at::Tensor& tensor2, const at::Scalar& value,
                           at::Tensor& out) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kAddcdivOut, _, (self, tensor1, tensor2, promoted_value, out), {
        TT_ASSIGN_OR_THROW(auto t1_t2_size,
                           InferSize(tensor1.sizes(), tensor2.sizes()));
        TT_ASSIGN_OR_THROW(auto expected_size,
                           InferSize(self.sizes(), t1_t2_size));

        TT_ASSIGN_OR_THROW(at::Tensor value_tensor, promoted_value.GetTensor());

        // Build the op.
        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, tensor1_op, tensor2_op, value_op] = inputs;
          return BuildAddcdivShlo(self_op, tensor1_op, tensor2_op, value_op);
        };

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        TT_THROW_IF_ERROR(DispatchOpOut<4>(
            std::move(op_builder), {self, tensor1, tensor2, value_tensor}, out,
            {.out_dtype = out_dtype,
             .out_dims = expected_size,
             .op_param_cache_keys = OpParamCacheKeys::Empty()}));
        return out;
      });
}

}  // namespace torch_tpu
