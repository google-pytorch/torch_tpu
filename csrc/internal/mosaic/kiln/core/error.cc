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

#include "csrc/internal/mosaic/kiln/core/error.h"

#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "csrc/internal/mosaic/kiln/core/builder_scope.h"

namespace torch_tpu::kiln::internal {

void ReportError(absl::Status error) {
  ABSL_CHECK(!error.ok())  // CRASH_OK
      << "ReportError() requires a non-OK status";
  CurrentBuilderScope().RecordError(std::move(error));
}

}  // namespace torch_tpu::kiln::internal
