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

#ifndef TORCH_TPU_OPS_REDUCTION_SUM_H_
#define TORCH_TPU_OPS_REDUCTION_SUM_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildSumShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode = ReductionMode::kDropDims,
    c10::optional<mlir::ElementType> element_type_opt = std::nullopt);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_REDUCTION_SUM_H_
