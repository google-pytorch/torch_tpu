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

#include "csrc/common/mlir_error_utils.h"

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/error_utils.h"
#include "xla/mlir/utils/error_util.h"

namespace torch_tpu {

std::string ConsumeMlirDiagnostics(
    mlir::BaseScopedDiagnosticHandler& diag_handler,
    const std::string_view context) {
  // A non-OK status is always UNKNOWN with the diagnostics as its message, so
  // the status code carries no information.
  const absl::Status status = diag_handler.ConsumeStatus();
  return status.ok()
             ? absl::StrCat(context, ": no MLIR diagnostic was reported")
             : std::string(WrapXlaError(status, context).message());
}

}  // namespace torch_tpu
