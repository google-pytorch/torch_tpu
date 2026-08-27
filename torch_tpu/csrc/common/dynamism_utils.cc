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

#include "torch_tpu/csrc/common/dynamism_utils.h"

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/eager/traversal.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<std::vector<DeviceRefDimensions>> GetTraversalOutputDimensions(
    mlir::MLIRContext& mlir_context, const Traversal& traversal) {
  TT_ASSIGN_OR_RETURN(auto mlir_module,
                      traversal.BuildMlirModule(mlir_context));
  auto func = mlir_module->lookupSymbol<mlir::func::FuncOp>("main");
  auto return_op =
      llvm::cast<mlir::func::ReturnOp>(func.getBody().front().getTerminator());
  auto output_values = return_op.getOperands();

  TT_RET_CHECK(output_values.size() == traversal.outputs().size(),
               error::kFailedPrecondition)
      << "output values size: " << output_values.size()
      << " does not match traversal outputs size: "
      << traversal.outputs().size();

  std::vector<DeviceRefDimensions> results;
  results.reserve(traversal.outputs().size());
  for (int i = 0; i < traversal.outputs().size(); ++i) {
    results.push_back(
        {traversal.outputs()[i], GetDimensions(output_values[i])});
  }
  return results;
}

}  // namespace torch_tpu
