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

#include "torch_tpu/ops/addcdiv/addcdiv_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildAddcdivShlo(mlir::MlirOp self,
                                              mlir::MlirOp tensor1,
                                              mlir::MlirOp tensor2,
                                              mlir::MlirOp value) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp div, BuildDivShlo(tensor1, tensor2));
  const mlir::RankedTensorType div_type = GetTensorTypeOrDie(div);
  value = mlir::stablehlo::ConvertElementType(value, div_type.getElementType());
  TT_ASSIGN_OR_RETURN(mlir::MlirOp value_div, BuildMulShlo(value, div));
  return BuildAddShlo(self, value_div);
}

}  // namespace

at::Tensor& AtenAddcdivOut(const at::Tensor& self, const at::Tensor& tensor1,
                           const at::Tensor& tensor2, const at::Scalar& value,
                           at::Tensor& out) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kAddcdivOut, _, (self, tensor1, tensor2, promoted_value, out), {
        TT_ASSIGN_OR_THROW(at::Tensor value_tensor, promoted_value.GetTensor());

        // Build the op.
        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, tensor1_op, tensor2_op, value_op] = inputs;
          return BuildAddcdivShlo(self_op, tensor1_op, tensor2_op, value_op);
        };

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<4>(std::move(op_builder),
                          {self, tensor1, tensor2, value_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(out.sizes()),
                           .op_param_cache_keys = OpParamCacheKeys::Empty()}));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

}  // namespace torch_tpu
