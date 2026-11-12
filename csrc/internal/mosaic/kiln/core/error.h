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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ERROR_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ERROR_H_

#include "absl/status/status.h"

namespace torch_tpu::kiln::internal {

// Reports `error` to the active scope of the current thread, which holds the
// first error reported to it; later errors are ignored, and `error` must not
// be OK. `BuildKernel()` returns the held error. Kiln handle types signal an
// error with a null `ValueHandle`; the error itself is reported here.
// Must be called within an active BuildKernel() or IndexMap execution.
void ReportError(absl::Status error);

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ERROR_H_
