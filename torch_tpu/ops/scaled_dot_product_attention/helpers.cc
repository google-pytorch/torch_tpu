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

#include "torch_tpu/ops/scaled_dot_product_attention/helpers.h"

#include <cmath>
#include <cstdint>
#include <optional>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

mlir::MlirOp flatten_batch_dims(mlir::MlirOp mlir_op, int batch_size,
                                int num_batch_dims) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(mlir_op);
  int rank = type.getRank();
  int num_non_batch_dims = rank - num_batch_dims;
  Dimensions new_dims(num_non_batch_dims + 1);
  new_dims[0] = batch_size;
  for (int i = 0; i < num_non_batch_dims; i++)
    new_dims[i + 1] = type.getDimSize(i + num_batch_dims);

  return mlir::stablehlo::Reshape(mlir_op, new_dims);
}

mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  absl::Span<const int64_t> shape) {
  return mlir::stablehlo::Reshape(mlir_op, shape);
}

mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  mlir::MlirOp mlir_op_with_target_dims) {
  const mlir::RankedTensorType type =
      GetTensorTypeOrDie(mlir_op_with_target_dims);
  return mlir::stablehlo::Reshape(mlir_op, type.getShape());
}

int get_batch_size(absl::Span<const int64_t> shape) {
  ABSL_CHECK(shape.size() >= 3)  // CRASH_OK
      << "Shape must have at least 3 dimensions";
  int batch_size = 1;
  for (int i = 0; i < shape.size() - 3; i++) {
    batch_size *= shape[i];
  }
  return batch_size;
}

mlir::MlirOp GetScaleDefaulted(mlir::MlirBuilder& builder,
                               std::optional<double> maybe_scale,
                               int64_t head_dim, mlir::Type element_type) {
  double scale = maybe_scale.value_or(1.0 / std::sqrt(head_dim));

  if (element_type == nullptr) {
    element_type = builder.getOpBuilder().getF32Type();
  }
  return MakeScalarConstant(builder, scale, element_type);
}

}  // namespace torch_tpu
