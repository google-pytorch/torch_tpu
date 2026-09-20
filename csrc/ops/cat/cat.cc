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

#include "csrc/ops/cat/cat.h"

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

// Attempts to fuse adjacent slice-concat pairs (e.g. RoPE rotate_half or
// identity slice-concat) along the minor dimension.
// In LLM training (Gemma, LLaMA), RoPE decomposes into:
//   x1 = x[..., :half]
//   x2 = x[..., half:]
//   cat((-x2, x1), dim=cat_dim)
// Slicing across the minor dimension forces un-aligned sublane tiles and causes
// XLA to insert expensive formatting copy operations (copy.1746, copy.1767,
// copy.1726) accessing over 2.4 GB HBM.
// By recognizing this pattern on static shapes and rewriting to:
//   Reshape(base, [..., 2, half]) -> Slice along outer dim -> (optional Negate)
//   -> Concatenate along outer dim -> Reshape(orig_shape),
// the minor dimension (size `half`) is never sliced or strided in memory,
// eliminating layout copy transpositions entirely.
std::optional<mlir::MlirOp> TryFuseSliceConcat(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> input_ops,
    int64_t dimension) {
  if (input_ops.size() != 2) {
    return std::nullopt;
  }

  mlir::Value val0 = input_ops[0].getValue();
  mlir::Value val1 = input_ops[1].getValue();

  bool neg0 = false;
  if (auto neg_op =
          llvm::dyn_cast_or_null<stablehlo::NegOp>(val0.getDefiningOp())) {
    val0 = neg_op.getOperand();
    neg0 = true;
  }

  bool neg1 = false;
  if (auto neg_op =
          llvm::dyn_cast_or_null<stablehlo::NegOp>(val1.getDefiningOp())) {
    val1 = neg_op.getOperand();
    neg1 = true;
  }

  auto slice0 =
      llvm::dyn_cast_or_null<stablehlo::SliceOp>(val0.getDefiningOp());
  auto slice1 =
      llvm::dyn_cast_or_null<stablehlo::SliceOp>(val1.getDefiningOp());
  if (!slice0 || !slice1) {
    return std::nullopt;
  }

  if (slice0.getOperand() != slice1.getOperand()) {
    return std::nullopt;
  }

  mlir::Value base = slice0.getOperand();
  auto ranked_type = mlir::dyn_cast<mlir::RankedTensorType>(base.getType());
  if (!ranked_type || !ranked_type.hasStaticShape()) {
    return std::nullopt;
  }

  int64_t rank = ranked_type.getRank();
  if (rank < 1) {
    return std::nullopt;
  }

  int64_t cat_dim = dimension < 0 ? dimension + rank : dimension;
  if (cat_dim < 0 || cat_dim >= rank) {
    return std::nullopt;
  }

  // Only fuse if slicing along the minor dimension (cat_dim == rank - 1).
  if (cat_dim != rank - 1) {
    return std::nullopt;
  }

  llvm::ArrayRef<int64_t> shape = ranked_type.getShape();
  int64_t d_size = shape[cat_dim];
  if (d_size % 2 != 0 || d_size <= 0) {
    return std::nullopt;
  }
  int64_t half = d_size / 2;

  llvm::ArrayRef<int64_t> start0 = slice0.getStartIndices();
  llvm::ArrayRef<int64_t> limit0 = slice0.getLimitIndices();
  llvm::ArrayRef<int64_t> strides0 = slice0.getStrides();

  llvm::ArrayRef<int64_t> start1 = slice1.getStartIndices();
  llvm::ArrayRef<int64_t> limit1 = slice1.getLimitIndices();
  llvm::ArrayRef<int64_t> strides1 = slice1.getStrides();

  // Verify all other dimensions are un-sliced
  for (int64_t d = 0; d < rank; ++d) {
    if (d == cat_dim) {
      if (strides0[d] != 1 || strides1[d] != 1) return std::nullopt;
    } else {
      if (start0[d] != 0 || limit0[d] != shape[d] || strides0[d] != 1) {
        return std::nullopt;
      }
      if (start1[d] != 0 || limit1[d] != shape[d] || strides1[d] != 1) {
        return std::nullopt;
      }
    }
  }

  int64_t s0 = start0[cat_dim];
  int64_t l0 = limit0[cat_dim];
  int64_t s1 = start1[cat_dim];
  int64_t l1 = limit1[cat_dim];

  // Case 1: Identity concatenation [0:half] followed by [half:d_size] without
  // negation.
  if (s0 == 0 && l0 == half && s1 == half && l1 == d_size && !neg0 && !neg1) {
    return mlir::MlirOp(builder, base);
  }

  // Case 2: RoPE rotate_half: [half:d_size] (negated) followed by [0:half]
  // (positive). Or general adjacent half-slice concatenation.
  bool is_rope = (s0 == half && l0 == d_size && s1 == 0 && l1 == half);
  bool is_identity_split =
      (s0 == 0 && l0 == half && s1 == half && l1 == d_size);
  if (!is_rope && !is_identity_split) {
    return std::nullopt;
  }

  // Rewrite using Reshape -> outer slice -> Concat -> Reshape
  llvm::SmallVector<int64_t> reshaped_shape;
  reshaped_shape.reserve(rank + 1);
  for (int64_t d = 0; d < rank - 1; ++d) {
    reshaped_shape.push_back(shape[d]);
  }
  reshaped_shape.push_back(2);
  reshaped_shape.push_back(half);

  mlir::MlirOp base_op(builder, base);
  mlir::MlirOp reshaped_base = stablehlo::Reshape(base_op, reshaped_shape);

  // We slice along dimension rank - 1 (the size 2 dimension)
  llvm::SmallVector<int64_t> outer_start0(rank + 1, 0);
  llvm::SmallVector<int64_t> outer_limit0(reshaped_shape.begin(),
                                          reshaped_shape.end());
  llvm::SmallVector<int64_t> outer_strides(rank + 1, 1);

  llvm::SmallVector<int64_t> outer_start1(rank + 1, 0);
  llvm::SmallVector<int64_t> outer_limit1(reshaped_shape.begin(),
                                          reshaped_shape.end());

  int64_t idx0 = (s0 == 0) ? 0 : 1;
  int64_t idx1 = (s1 == 0) ? 0 : 1;

  outer_start0[rank - 1] = idx0;
  outer_limit0[rank - 1] = idx0 + 1;

  outer_start1[rank - 1] = idx1;
  outer_limit1[rank - 1] = idx1 + 1;

  mlir::MlirOp out_slice0 = stablehlo::Slice(reshaped_base, outer_start0,
                                             outer_limit0, outer_strides);
  if (neg0) {
    out_slice0 = stablehlo::Neg(out_slice0);
  }

  mlir::MlirOp out_slice1 = stablehlo::Slice(reshaped_base, outer_start1,
                                             outer_limit1, outer_strides);
  if (neg1) {
    out_slice1 = stablehlo::Neg(out_slice1);
  }

  mlir::MlirOp cat_res =
      stablehlo::Concatenate(builder, {out_slice0, out_slice1}, rank - 1);
  mlir::MlirOp result = stablehlo::Reshape(cat_res, shape);
  return result;
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildCatShlo(absl::Span<mlir::MlirOp> input_ops,
                                          int64_t dimension) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
      !input_ops.empty(), error::kInvalidArgument)
      << "expected a non-empty list of Tensors";

  if (auto fused =
          TryFuseSliceConcat(input_ops[0].getBuilder(), input_ops, dimension)) {
    return *fused;
  }

  return stablehlo::Concatenate(input_ops[0].getBuilder(), input_ops,
                                dimension);
}

}  // namespace torch_tpu
