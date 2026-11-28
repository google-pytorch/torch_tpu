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

// ANY_OP_IMPL_FILENAME_OK=shared validation helpers used across multiple
// upsample *_aten_kernels files

#ifndef TORCH_TPU_CSRC_OPS_UPSAMPLE_UPSAMPLE_UTILS_H_
#define TORCH_TPU_CSRC_OPS_UPSAMPLE_UPSAMPLE_UTILS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Controls which dtypes an upsample kernel accepts before dispatching.
enum class UpsampleDtypePolicy {
  // Accepts floating-point dtypes only (matches CUDA linear1d / *_aa kernels).
  kFloatingOnly,
  // Rejects bool, int32, and int64 (matches existing nearest / bilinear /
  // bicubic kernels).
  kExcludeBoolIntLong,
};

// Validates inputs to a forward `upsample_*.out` kernel and returns the
// expected output shape `{N, C, output_size...}`.
//
// Must be called before resizing `out` so that invalid calls do not mutate
// `out`.
absl::StatusOr<Dimensions> ValidateUpsampleForwardInputs(
    const at::Tensor& self, at::IntArrayRef output_size, const at::Tensor& out,
    int64_t spatial_dims, UpsampleDtypePolicy dtype_policy);

// Validates inputs to a backward `upsample_*_backward.grad_input` kernel and
// returns the expected `grad_input` shape (`input_size`).
//
// Must be called before resizing `grad_input` so that invalid calls do not
// mutate `grad_input`.
absl::StatusOr<Dimensions> ValidateUpsampleBackwardInputs(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, const at::Tensor& grad_input,
    int64_t spatial_dims, UpsampleDtypePolicy dtype_policy);

// Verifies that all spatial dimensions (`dim >= 2`) of `operand` are static.
// Batch (dim 0) and channel (dim 1) dimensions may be bounded dynamic.
absl::Status ValidateUpsampleSpatialDimsStatic(mlir::MlirOp operand);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_UPSAMPLE_UPSAMPLE_UTILS_H_
