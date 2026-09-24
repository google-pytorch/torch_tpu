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
namespace internal {
template <>
inline void CheckScalarInput<OpName::kAddcdivOut, UsesScalarInput::kYes>() {}
}  // namespace internal

namespace {

absl::StatusOr<mlir::MlirOp> BuildAddcdivScalarShlo(
    mlir::MlirOp self, mlir::MlirOp tensor1, mlir::MlirOp tensor2,
    const at::Scalar& value, mlir::ElementType out_dtype) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(out_dtype));
  TT_ASSIGN_OR_RETURN(self, CastIfNeeded(self, compute_dtype));
  TT_ASSIGN_OR_RETURN(tensor1, CastIfNeeded(tensor1, compute_dtype));
  TT_ASSIGN_OR_RETURN(tensor2, CastIfNeeded(tensor2, compute_dtype));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp div, BuildDivShlo(tensor1, tensor2));

  double val = value.toDouble();
  if (val == 1.0) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result, BuildAddShlo(self, div));
    return CastIfNeeded(result, out_dtype);
  }
  if (val == -1.0) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result, BuildSubShlo(self, div));
    return CastIfNeeded(result, out_dtype);
  }

  auto& builder = self.getBuilder();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp val_op,
                      MakeConstant(builder, value, compute_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp value_div, BuildMulShlo(val_op, div));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp result, BuildAddShlo(self, value_div));
  return CastIfNeeded(result, out_dtype);
}

}  // namespace

at::Tensor& AtenAddcdivOut(const at::Tensor& self, const at::Tensor& tensor1,
                           const at::Tensor& tensor2, const at::Scalar& value,
                           at::Tensor& out) {
  TT_KERNEL(
      OpName::kAddcdivOut, param_keys, (self, tensor1, tensor2, value, out), {
        TT_ASSIGN_OR_THROW(auto t1_t2_size,
                           InferSize(tensor1.sizes(), tensor2.sizes()));
        TT_ASSIGN_OR_THROW(auto expected_size,
                           InferSize(self.sizes(), t1_t2_size));

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        // Build the op.
        auto op_builder = [value,
                           out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, tensor1_op, tensor2_op] = inputs;
          return BuildAddcdivScalarShlo(self_op, tensor1_op, tensor2_op, value,
                                        out_dtype);
        };

        TT_THROW_IF_ERROR(DispatchOpOut<3>(
            std::move(op_builder), {self, tensor1, tensor2}, out,
            {.out_dtype = out_dtype,
             .out_dims = expected_size,
             .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

}  // namespace torch_tpu
