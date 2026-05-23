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

#include "torch_tpu/ops/fmax/fmax_aten_kernels.h"

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/binary_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildFmaxShlo(mlir::MlirOp self,
                                           mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN((auto [self_, other_]),
                      ApplyBroadcastIfNeeded(self, other));

  mlir::MlirOp self_greater = mlir::stablehlo::Compare(
      self_, other_, mlir::stablehlo::ComparisonDirection::GT);
  mlir::MlirOp other_nan = mlir::stablehlo::Compare(
      other_, other_, mlir::stablehlo::ComparisonDirection::NE);

  mlir::MlirOp self_greater_or_other_nan =
      mlir::stablehlo::Or(self_greater, other_nan);

  return mlir::stablehlo::Select(self_greater_or_other_nan, self_, other_);
}

}  // namespace

at::Tensor& AtenFmaxOut(const at::Tensor& self, const at::Tensor& other,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kFmaxOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildFmaxShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

}  // namespace torch_tpu
