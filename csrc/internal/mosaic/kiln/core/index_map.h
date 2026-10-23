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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_INDEX_MAP_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_INDEX_MAP_H_

#include <functional>
#include <utility>

#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {

using IndexIndices = llvm::SmallVector<IndexArg, 4>;

// Maps grid indices to block indices (Pallas `index_map`). Inside `IndexFn`,
// `ProgramId(axis)` and `ScalarPrefetch(i)` are bound to the function's args.
//
// Unlike `CreateKernel`, `If`, `For`, and `RunScoped`, `IndexMap` stores `fn`
// and invokes it later when `BuildKernel` / `CreateKernel` runs. Any variables
// captured by reference in `fn` must outlive the `BuildKernel` / `CreateKernel`
// call; in particular, an `IndexMap` returned from a helper function must not
// capture local variables of that helper by reference.
class IndexMap {
 public:
  using IndexFn = std::function<IndexIndices()>;  // STD_FUNCTION_OK

  IndexMap() = default;
  explicit IndexMap(IndexFn fn) : mapping_(std::move(fn)) {}

  bool has_mapping() const { return mapping_ != nullptr; }
  IndexIndices Invoke() const { return mapping_(); }

 private:
  IndexFn mapping_;
};

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_INDEX_MAP_H_
