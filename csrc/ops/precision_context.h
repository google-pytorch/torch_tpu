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

#ifndef TORCH_TPU_CSRC_OPS_PRECISION_CONTEXT_H_
#define TORCH_TPU_CSRC_OPS_PRECISION_CONTEXT_H_

#include <string_view>

#include "csrc/common/cache_key.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {

// These functions manage the precision configuration for StableHLO operations.
// This allows different python threads to operate with different precision
// settings for matrix multiplications or convolutions, such as DEFAULT, HIGH,
// or HIGHEST.

// Sets the global float32 matmul precision.
void PySetGlobalPrecision(mlir::stablehlo::Precision precision);

// Returns the current global float32 matmul precision.
[[nodiscard]] mlir::stablehlo::Precision PyGetGlobalPrecision();

// Pushes the given precision context state onto the stack.
void PyPushPrecision(mlir::stablehlo::Precision precision);

// Pops the current precision context state from the stack.
void PyPopPrecision();

// Returns the current python thread's precision and adds it to the param_keys
// with the given parameter name. Precedence order:
// 1. Thread-local `PrecisionContextState` (from `torch.tpu.precision` context
//    manager)
// 2. ATen `at::NoTF32Guard` (`HIGHEST` when active, e.g., in linalg ops and
//    backward formulas)
// 3. Global precision setting (`torch.set_float32_matmul_precision`)
[[nodiscard]] mlir::stablehlo::Precision GetAndAddPrecisionTo(
    OpParamCacheKeys& param_keys, std::string_view param_name = "precision");

// Returns the current thread-local precision if explicitly set via a
// `torch_tpu.precision` context manager; otherwise returns
// `default_precision`. Records the resulting precision in `param_keys`.
[[nodiscard]] mlir::stablehlo::Precision GetAndAddPrecisionWithDefaultTo(
    OpParamCacheKeys& param_keys, mlir::stablehlo::Precision default_precision,
    std::string_view param_name = "precision");

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_PRECISION_CONTEXT_H_
