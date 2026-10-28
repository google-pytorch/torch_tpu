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

#ifndef TORCH_TPU_CSRC_OPS_GCD_GCD_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_GCD_GCD_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Computes the elementwise greatest common divisor of `a` and `b`, which must
// share the same shape and an int8, uint8, int16, int32 or int64 element type.
// The result has the element type of the inputs.
//
// Absolute values are taken in the input element type and remainders retain
// their sign, matching PyTorch's behavior when abs(INT_MIN) overflows. For
// example, gcd(INT32_MIN, 6) is -2 and gcd(INT32_MIN, 0) is INT32_MIN.
absl::StatusOr<mlir::MlirOp> BuildGcdShlo(mlir::MlirOp a, mlir::MlirOp b);

at::Tensor& AtenGcdOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_GCD_GCD_ATEN_KERNELS_H_
