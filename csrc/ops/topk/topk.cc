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

#include "csrc/ops/topk/topk.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/sort/sort.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace chlo = mlir::chlo;
namespace stablehlo = mlir::stablehlo;

namespace {

// Returns `op` with the order of its elements reversed. Applying it twice gives
// back `op`. Floats are negated. Integers and bools use `stablehlo::Not`
// instead of negation, because negation maps the lowest signed integer and
// unsigned 0 to themselves, whereas bitwise NOT strictly reverses order for
// both signed two's-complement (`~x = -x - 1`) and unsigned (`~x = max - x`)
// integers (as well as bools).
mlir::MlirOp ReverseOrder(mlir::MlirOp op) {
  return IsFloatType(GetTensorTypeOrDie(op)) ? stablehlo::Neg(op)
                                             : stablehlo::Not(op);
}

}  // namespace

absl::StatusOr<TopKOutputs> BuildTopKShlo(
    mlir::MlirOp input_op, int64_t k, int64_t dim, TopKMode topk_mode,
    std::optional<TopKStableMode> topk_stable_mode) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const int64_t rank = input_type.getRank();
  TT_ASSIGN_OR_RETURN(const int64_t canonical_dim, SafeWrapDim(dim, rank));
  const bool is_stable = topk_stable_mode == TopKStableMode::kStable;

  if (!input_type.hasStaticShape()) {
    // chlo::TopK legalizes dynamic shapes via DynamicIotaOp (dropping
    // #stablehlo.bounds) and RealDynamicSliceOp, which XLA DynamicPadder does
    // not support. Lower bounded dynamic inputs via BuildSortShlo + Slice.
    const SortShloOutputs sorted =
        BuildSortShlo(input_op, is_stable, canonical_dim,
                      /*descending=*/topk_mode == TopKMode::kLargest);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp topk_values,
                        SlicePrefixAlongDim(sorted.values, canonical_dim, k));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp topk_indices,
                        SlicePrefixAlongDim(sorted.indices, canonical_dim, k));
    return TopKOutputs{.values = topk_values, .indices = topk_indices};
  }

  const bool need_transpose = (canonical_dim != rank - 1);

  // Construct permutation vector to swap canonical_dim with rank - 1.
  mlir::MlirOp sort_input = input_op;
  Dimensions permutation;
  if (need_transpose) {
    permutation = GetAllDimensions(input_op);
    std::swap(permutation[canonical_dim], permutation[rank - 1]);

    sort_input = stablehlo::Transpose(sort_input, permutation);
  }

  // chlo::TopK only returns the k largest elements, so for kSmallest we reverse
  // element order before calling it and reverse the returned values back.
  if (topk_mode == TopKMode::kSmallest) {
    sort_input = ReverseOrder(sort_input);
  }

  // Call chlo::TopK along the innermost dimension (rank - 1). The chlo::TopK
  // builder defaults is_stable to true; derive it from topk_stable_mode instead
  // so the default preserves the historical unstable-sort tie-selection
  // semantics. See github.com/google-pytorch/torch_tpu/issues/3392.
  llvm::SmallVector<mlir::MlirOp, 2> outputs =
      chlo::TopK(sort_input, static_cast<uint64_t>(k), is_stable);
  mlir::MlirOp topk_values = outputs[0];
  mlir::MlirOp topk_indices = outputs[1];

  if (topk_mode == TopKMode::kSmallest) {
    topk_values = ReverseOrder(topk_values);
  }

  if (need_transpose) {
    topk_values = stablehlo::Transpose(topk_values, permutation);
    topk_indices = stablehlo::Transpose(topk_indices, permutation);
  }

  // Convert index element type from default I32 to I64 to satisfy
  // PyTorch/ATen kernel conventions.
  topk_indices =
      stablehlo::ConvertElementType(topk_indices, mlir::ElementType::I64);

  return TopKOutputs{.values = topk_values, .indices = topk_indices};
}

}  // namespace torch_tpu
