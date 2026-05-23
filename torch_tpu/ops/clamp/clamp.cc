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

#include "torch_tpu/ops/clamp/clamp.h"

#include <optional>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

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

  mlir::MlirOp res;
  if (min && max) {
    res = mlir::stablehlo::Clamp(*min, input, *max);
  } else if (min) {
    res = mlir::stablehlo::Max(input, *min);
  } else if (max) {
    res = mlir::stablehlo::Min(input, *max);
  } else {
    res = input;
  }

  return res;
}

}  // namespace torch_tpu
