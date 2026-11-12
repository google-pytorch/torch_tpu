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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_SORT_BITONIC_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_SORT_BITONIC_H_

#include "csrc/internal/mosaic/kiln/kiln.h"

// Bitonic sort for Kiln kernels. Which elements get compared depends only on
// the size, not on the data, so each step runs on the whole tile at once.

namespace torch_tpu {

// Sorts an (8, 128) i32 tile ascending, in row-major order, moving `payload`
// along with it. Equal keys are ordered by payload, so with each element's
// position as its payload this is a stable argsort.
void BitonicSortTile(kiln::Tile& keys, kiln::Tile& payload);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_SORT_BITONIC_H_
