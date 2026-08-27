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

#ifndef TORCH_TPU_OPS_REDUCTION_MEAN_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_REDUCTION_MEAN_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/OptionalArrayRef.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/ops/reductions/reductions.h"

namespace torch_tpu {

// Implements aten::mean.out.
at::Tensor& AtenMeanOut(const at::Tensor& self,
                        c10::OptionalArrayRef<int64_t> dim, bool keep_dim,
                        std::optional<c10::ScalarType> dtype, at::Tensor& out);

absl::StatusOr<mlir::MlirOp> BuildMeanShlo(
    mlir::MlirOp input, absl::Span<const int64_t> canonical_dims,
    ReductionMode reduction_mode, mlir::ElementType mlir_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_REDUCTION_MEAN_ATEN_KERNELS_H_
