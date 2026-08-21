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

#ifndef TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_HELPERS_H_
#define TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_HELPERS_H_

#include <cstdint>
#include <optional>

#include "absl/types/span.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

mlir::MlirOp flatten_batch_dims(mlir::MlirOp mlir_op, int batch_size,
                                int num_batch_dims);
mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  absl::Span<const int64_t> shape);
mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  mlir::MlirOp mlir_op_with_target_dims);
int get_batch_size(absl::Span<const int64_t> shape);

mlir::MlirOp GetScaleDefaulted(mlir::MlirBuilder& builder,
                               std::optional<double> maybe_scale,
                               int64_t head_dim,
                               mlir::Type element_type = nullptr);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_HELPERS_H_
