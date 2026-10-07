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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_VALUE_BASE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_VALUE_BASE_H_

#include "csrc/internal/mosaic/kiln/core/dtype.h"

namespace torch_tpu::kiln {

namespace internal {
struct ValueAccess;
}  // namespace internal

// Common base for KILN SSA handles (`Scalar`, `Cond`, `Tile`, `Ref`). Holds
// an opaque pointer to the underlying IR value and its KILN element dtype.
class ValueBase {
 protected:
  ValueBase(const void* impl, DType dtype) : impl_(impl), dtype_(dtype) {}
  const void* impl() const { return impl_; }
  DType dtype() const { return dtype_; }

 private:
  friend struct internal::ValueAccess;
  const void* impl_ = nullptr;
  DType dtype_;
};

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_VALUE_BASE_H_
