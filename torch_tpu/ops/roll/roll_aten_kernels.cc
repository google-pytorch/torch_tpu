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

#include "torch_tpu/ops/roll/roll_aten_kernels.h"

#include <cstdint>
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

absl::StatusOr<mlir::MlirOp> BuildRollShlo(mlir::MlirOp input,
                                           at::IntArrayRef shifts,
                                           at::IntArrayRef dims) {
  auto input_type = GetTensorTypeOrDie(input);
  Dimensions input_dims = CopyIntVector(input_type.getShape());
  Strides strides(input_dims.size(), 1);
  mlir::MlirBuilder& builder = input.getBuilder();

  mlir::MlirOp result = input;

  // Apply a 1-D roll to each dimension.
  for (int i = 0; i < dims.size(); ++i) {
    int64_t dim = dims[i];
    int64_t dim_size = input_dims[dim];
    if (dim_size == 0) continue;
    int64_t shift = shifts[i];
    int64_t normalized_shift = (shift % dim_size + dim_size) % dim_size;
    if (normalized_shift == 0) continue;

    // For this particular dimension, we want:
    //   [ a_0 ... a_n | b_0 ... b_n ] -> [ b_0 ... b_n | a_0 ... a_n ]
    //                 ^
    //             split_point
    int64_t split_point = dim_size - normalized_shift;
    Indices a_start(input_dims.size(), 0);
    Indices a_end = input_dims;
    a_end[dim] = split_point;
    Indices b_start(input_dims.size(), 0);
    b_start[dim] = split_point;
    Indices b_end = input_dims;

    auto a = mlir::stablehlo::Slice(result, a_start, a_end, strides);
    auto b = mlir::stablehlo::Slice(result, b_start, b_end, strides);
    result = mlir::stablehlo::Concatenate(builder, {b, a}, dim);
  }

  return result;
}

}  // namespace

at::Tensor AtenRoll(const at::Tensor& self, at::IntArrayRef shifts,
                    at::IntArrayRef dims) {
  if (dims.empty()) {
    if (shifts.empty()) {
      return self.clone();
    }
    if (shifts.size() == 1) {
      at::Tensor flattened = self.flatten();
      at::Tensor rolled = AtenRoll(flattened, shifts, {0});
      return rolled.view(self.sizes());
    }
  }

  Dimensions normalized_dims = CopyIntVector(dims);
  for (auto& dim : normalized_dims) {
    if (dim < 0) {
      dim += self.dim();
    }
  }

  Indices shifts_vec = CopyIntVector(shifts);

  TT_KERNEL(OpName::kRoll, param_keys, (self, shifts_vec, normalized_dims), {
    TT_CHECK_THROW(
        shifts.size() == dims.size() || (dims.empty() && shifts.size() <= 1),
        error::kInvalidArgument)
        << "shifts and dims must align, got shifts: " << shifts.size()
        << ", dims: " << dims.size();

    TT_ASSIGN_OR_THROW(auto element_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    auto op_builder = [shifts_vec, normalized_dims](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildRollShlo(input, shifts_vec, normalized_dims);
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
