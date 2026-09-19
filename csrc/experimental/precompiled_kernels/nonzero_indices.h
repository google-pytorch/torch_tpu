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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace torch_tpu {

// Builds indices for F32 input[M,N], returning logical S64[K,2] coordinates
// ordered by row, then column. The caller must know the exact K.
// The predicate excludes only +0.0 and -0.0, matching torch.nonzero.
//
// Buffer contract (all payloads have no dynamic-metadata prefix):
// - Input: F32[M,N]{1,0:T(8,128)}.
// - Destination: S32[R,128]{1,0:T(1,128)}, where R = 4 * ceil(K/128)
//   (rows [0, R/2) hold interleaved row/col low words; [R/2, R) hold zeroed
//   high words).
//   Note: Because Mosaic does not support S64 memrefs, the caller bitcasts the
//   S64[K,2]{0,1:T(2,128)} PJRT buffer to/from this S32 view before and after
//   execution. NB: requires K > 2 because XLA default layout for S64[K,2]
//   switches to {1,0:T(2,128)} when K <= 2.
// - Metadata: two S32[2]{0:T(128)} operands supplying [M,N] and [R,128]. The
//   HLO result must alias the destination; input and metadata are not donated.
//
// Caller-checked arithmetic bounds: 1 <= M <= INT32_MAX,
// 1 <= N <= INT32_MAX - 127, and 3 <= K <= min(M*N, INT32_MAX).

// Create the context with mlir::torch_tpu::CreateMlirContextWithDialects().
mlir::OwningOpRef<mlir::ModuleOp> BuildNonzeroIndices(
    mlir::MLIRContext& context);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_H_
