// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/ops/clamp/clamp.h"

#include <optional>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildClampShlo(mlir::MlirOp input,
                                            std::optional<mlir::MlirOp> min,
                                            std::optional<mlir::MlirOp> max) {
  ABSL_VLOG(1) << "BuildClampShlo input=" << input.ToString()
               << " min=" << (min ? min->ToString() : "-")
               << " max=" << (max ? max->ToString() : "-");

  // Apply implicit broadcast, if necessary.
  if (min && max) {
    TT_ASSIGN_OR_RETURN((auto [input_, min_, max_]),
                        ApplyBroadcastIfNeeded(input, *min, *max));
    input = input_;
    min = min_;
    max = max_;
  } else if (min) {
    TT_ASSIGN_OR_RETURN((auto [input_, min_]),
                        ApplyBroadcastIfNeeded(input, *min));
    input = input_;
    min = min_;
  } else if (max) {
    TT_ASSIGN_OR_RETURN((auto [input_, max_]),
                        ApplyBroadcastIfNeeded(input, *max));
    input = input_;
    max = max_;
  }

  // Lower two-sided clamp as Min(Max(input, min), max) rather than
  // stablehlo::Clamp:
  // 1. XLA TPU's VclampFloat folds constant bounds with max <= min directly to
  //    max, dropping NaN elements in `input`, whereas Min(Max(x, min), max)
  //    preserves IEEE NaN propagation (while still fusing to hardware vclamp
  //    when min < max).
  // 2. stablehlo::Max / Min preserve bounded dynamic dimensions, unlike
  //    stablehlo::Clamp.
  mlir::MlirOp res = input;
  if (min) {
    res = mlir::stablehlo::Max(res, *min);
  }
  if (max) {
    res = mlir::stablehlo::Min(res, *max);
  }

  return res;
}

}  // namespace torch_tpu
