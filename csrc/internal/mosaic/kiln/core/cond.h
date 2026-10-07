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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_COND_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_COND_H_

#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"

namespace torch_tpu::kiln {

// An i1 scalar condition handle.
class Cond : public ValueBase {
 public:
  friend Cond operator&&(Cond lhs, Cond rhs);
  friend Cond operator||(Cond lhs, Cond rhs);
  friend Cond operator!(Cond cond);

 private:
  friend struct internal::ValueAccess;
  Cond(const void* impl, DType dtype);
};

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_COND_H_
