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

#include "torch_tpu/csrc/ops/topk/topk.h"

#include <cstdint>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"

namespace torch_tpu {

namespace chlo = mlir::chlo;
namespace stablehlo = mlir::stablehlo;

absl::StatusOr<TopKOutputs> BuildTopKShlo(mlir::MlirOp input_op, int64_t k,
                                          int64_t dim, TopKMode topk_mode) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const int64_t rank = input_type.getRank();
  TT_ASSIGN_OR_RETURN(const int64_t canonical_dim, SafeWrapDim(dim, rank));
  const bool need_transpose = (canonical_dim != rank - 1);

  // Construct permutation vector to swap canonical_dim with rank - 1.
  mlir::MlirOp sort_input = input_op;
  Dimensions permutation;
  if (need_transpose) {
    permutation.resize(rank);
    absl::c_iota(permutation, 0);
    std::swap(permutation[canonical_dim], permutation[rank - 1]);

    sort_input = stablehlo::Transpose(sort_input, permutation);
  }

  if (topk_mode == TopKMode::kSmallest) {
    sort_input = stablehlo::Neg(sort_input);
  }

  // Call chlo::TopK along the innermost dimension (rank - 1).
  llvm::SmallVector<mlir::MlirOp, 2> outputs =
      chlo::TopK(sort_input, static_cast<uint64_t>(k));
  mlir::MlirOp topk_values = outputs[0];
  mlir::MlirOp topk_indices = outputs[1];

  if (topk_mode == TopKMode::kSmallest) {
    topk_values = stablehlo::Neg(topk_values);
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
