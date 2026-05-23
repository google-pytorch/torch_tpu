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

#include "torch_tpu/ops/flip/flip_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildFlipShlo(mlir::MlirOp input_op,
                                           Dimensions flip_dims) {
  return mlir::stablehlo::Reverse(input_op, flip_dims);
}

}  // namespace

at::Tensor AtenFlip(const at::Tensor& self, at::IntArrayRef flip_dims) {
  Dimensions flip_dims_vec(flip_dims.begin(), flip_dims.end());
  // Normalize to positive values
  for (auto& dim : flip_dims_vec) {
    if (dim < 0) {
      dim = self.dim() + dim;
    }
  }
  TT_KERNEL(OpName::kFlip, param_keys, (self, flip_dims_vec), {
    TT_ASSIGN_OR_THROW(auto element_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    auto op_builder =
        [flip_dims_vec](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      TT_ASSIGN_OR_RETURN(auto output, BuildFlipShlo(input, flip_dims_vec));
      return output;
    };

    TT_ASSIGN_OR_THROW(
        auto out_buf,
        (DispatchOp<1>(std::move(op_builder), self,
                       {.out_dtype = element_type,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)})));
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
