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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_CONJ_PRIMITIVE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_CONJ_PRIMITIVE_H_

#include "absl/status/statusor.h"
#include "torch_tpu/csrc/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {

struct ConjPrimitive {
  bool is_set = true;
};

inline bool UpdateLayout(StridedLayout& layout,
                         const ConjPrimitive& primitive) {
  return primitive.is_set;
}
// Size should be 1 byte, but alignment/padding might make it more in variant.
// The static assertions in view_sequence.h will verify overall variant size.

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_CONJ_PRIMITIVE_H_
