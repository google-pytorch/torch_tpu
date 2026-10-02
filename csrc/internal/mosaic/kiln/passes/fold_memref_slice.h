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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_PASSES_FOLD_MEMREF_SLICE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_PASSES_FOLD_MEMREF_SLICE_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace torch_tpu::kiln {

// Creates a pass that folds chains of `tpu.memref_slice` and
// `tpu.memref_squeeze` operations into scalar `memref.load` and `memref.store`
// operations.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFoldMemRefSlicePass();

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_PASSES_FOLD_MEMREF_SLICE_H_
