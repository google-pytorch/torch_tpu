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

#include "csrc/internal/mosaic/kiln/core/value_handle.h"

#include "absl/log/absl_check.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {

ValueHandle::ValueHandle(mlir::Value value)
    : impl_(value.getAsOpaquePointer()) {
  ABSL_CHECK(value)  // CRASH_OK
      << "ValueHandle requires a non-null value";
}

mlir::Value ValueHandle::value() const {
  return mlir::Value::getFromOpaquePointer(impl_);
}

}  // namespace torch_tpu::kiln::internal
