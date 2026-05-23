/*
 * Copyright 2025 Google LLC
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

#ifndef TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_OPS_H_
#define TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_OPS_H_

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"

namespace torch_tpu {

// Universal pad module. Given a list of shapes with dynamic dimensions, returns
// a module with as many inputs as the number of non-zero-sized shapes (Note
// rank 0 shapes are not zero-sized). For each shape, we create an input that
// has the same static dimensions as shape, mapped to an output where the
// dynamic dimensions have been padded to the upper bound, and one extra output
// for each dynamic dimension, specifying the original dimension size.
// Zero-sized tensors are not padded. E.g., for input shapes
//   {[3, 5 ;dim=0,<=10], [], [8, 2, 2 ; dim1,<=5, dim2,<=7], [6, 0 ; dim0,<=10]
// we get a module with the following signature:
//   ([3, 5], [], [8, 2, 2])
//   -> ([10, 5], i32, [], [8, 5, 7], i32, i32)
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetPadModule(
    mlir::MLIRContext& mlir_context, absl::Span<const Shape> shapes);

// Universal slice module. Given a list of padded and unpadded shapes, returns a
// module that converts padded dynamic dimensions into static dimensions and
// then slices the tensors to the desired unpadded shapes.
//
// For example, given input padded_dimensions_vec = [[10, 5], [8, 5, 7]]
// and dimensions_vec = [[3, 5], [8, 2, 2]], the module signature will be:
//   ([<=10, 5], [8, <=5, <=7]) -> ([3, 5], [8, 2, 2]).
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetSliceModule(
    mlir::MLIRContext& mlir_context,
    absl::Span<const Dimensions> dimensions_vec,
    absl::Span<const Dimensions> padded_dimensions_vec,
    absl::Span<const mlir::ElementType> input_dtypes);

}  // namespace torch_tpu

#endif  // TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_OPS_H_
