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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_AOT_ELEMENTWISE_KERNELS_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_AOT_ELEMENTWISE_KERNELS_H_

#include "absl/types/span.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// Describes a single-output elementwise operation.
// BuildKernel supplies the loops, loads and stores; build_tile emits
// arithmetic on the loaded input tiles and returns the result tile.
struct ElementwiseOp {
  using TileBuilder = mlir::Value (*)(mlir::OpBuilder&, mlir::Location,
                                      mlir::ValueRange);

  OpName name;
  int arity;  // Number of input tensors.
  TileBuilder build_tile;
};

// Returns the table of supported elementwise operations.
absl::Span<const ElementwiseOp> Operations();

// Builds a Mosaic module implementing an elementwise operation for the given
// data type and tensor rank.
mlir::OwningOpRef<mlir::ModuleOp> BuildKernel(mlir::MLIRContext& context,
                                              const ElementwiseOp& op,
                                              xla::PrimitiveType dtype,
                                              int rank);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_AOT_ELEMENTWISE_KERNELS_H_
