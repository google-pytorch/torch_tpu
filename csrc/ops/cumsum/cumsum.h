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

#ifndef TORCH_TPU_CSRC_OPS_CUMSUM_CUMSUM_H_
#define TORCH_TPU_CSRC_OPS_CUMSUM_CUMSUM_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Lowering backend chosen for `aten::cumsum` based on input shape and scan
// axis (b/588733851).
enum class CumsumLoweringStrategy {
  // Associative `chlo.ScanOp` (native TPU ScanEmitter). Preferred for dynamic
  // shapes, long scan axes exceeding TPU lane width (`L > 128`), and small
  // tensors (`numel < 262144`) that hit the row-parallel VMEM fast path.
  kChloScan,
  // `stablehlo.reduce_window` with window `[L]` and left pad `L - 1`. At
  // `L <= 128` (`target.LaneCount()`), XLA's `TpuReduceWindowRewriter` lowers
  // it as a single-pass parallel prefix sum without recursive chunking or a
  // `call %scan` barrier.
  kReduceWindow,
};

// Selects the StableHLO lowering strategy for a cumulative sum over
// `normalized_dim` of a tensor with type `input_type`.
[[nodiscard]] CumsumLoweringStrategy SelectCumsumLoweringStrategy(
    mlir::RankedTensorType input_type, int64_t normalized_dim);

absl::StatusOr<mlir::MlirOp> BuildCumsumShlo(
    int64_t normalized_dim, std::optional<mlir::ElementType> out_dtype,
    mlir::MlirOp input);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_CUMSUM_CUMSUM_H_
