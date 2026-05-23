// Copyright 2026 Google LLC
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

#ifndef TORCH_TPU_OPS_COL2IM_COL2IM_H_
#define TORCH_TPU_OPS_COL2IM_COL2IM_H_

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildCol2ImShlo(
    mlir::MlirOp input, SmallInt64Vector output_size, SmallInt64Vector col_size,
    SmallInt64Vector kernel_size, SmallInt64Vector dilation,
    SmallInt64Vector padding, SmallInt64Vector stride);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_COL2IM_COL2IM_H_
