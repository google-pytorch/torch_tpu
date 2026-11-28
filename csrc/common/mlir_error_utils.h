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

#ifndef TORCH_TPU_CSRC_COMMON_MLIR_ERROR_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_MLIR_ERROR_UTILS_H_

#include <string>
#include <string_view>

#include "xla/mlir/utils/error_util.h"

namespace torch_tpu {

// Consumes the diagnostics of `diag_handler` and returns them formatted for an
// error message with `context` (wrapped via `WrapXlaError` if any diagnostic
// was reported). Never returns an empty string.
[[nodiscard]] std::string ConsumeMlirDiagnostics(
    mlir::BaseScopedDiagnosticHandler& diag_handler, std::string_view context);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_MLIR_ERROR_UTILS_H_
