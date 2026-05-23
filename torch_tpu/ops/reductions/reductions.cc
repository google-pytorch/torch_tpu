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

#include "torch_tpu/ops/reductions/reductions.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

mlir::MlirOp BuildReductionShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> dims, mlir::Type mlir_type,
    mlir::MlirOp init_val,
    absl::AnyInvocable<void(mlir::RegionBuilder&) const> reduce_fn,
    ReductionMode reduction_mode) {
  mlir::MlirBuilder& builder = input_op.getBuilder();

  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(input_op);

  mlir::MlirOp converted_input = input_op;
  if (input_tensor_type.getElementType() != mlir_type) {
    converted_input = mlir::stablehlo::ConvertElementType(input_op, mlir_type);
  }

  // mlir::stablehlo::Reduce() takes an std::function, which requires all
  // objects captured by the function to be copyable. However, reduce_fn is not
  // copyable.
  //
  // To get around this, we create a shared_ptr to reduce_fn and pass
  // the shared_ptr to the lambda that is passed to Reduce().
  auto shared_reduce_fn =
      std::make_shared<decltype(reduce_fn)>(std::move(reduce_fn));
  mlir::SmallVector<mlir::MlirOp> reduced_ops = mlir::stablehlo::Reduce(
      builder, {converted_input}, {init_val},
      [shared_reduce_fn](mlir::RegionBuilder& builder) {
        (*shared_reduce_fn)(builder);
      },
      dims);
  mlir::MlirOp result_op = reduced_ops[0];

  if (reduction_mode == ReductionMode::kKeepDims) {
    result_op = BuildKeepDimsShlo(input_op, result_op, dims);
  }
  return result_op;
}

mlir::MlirOp BuildKeepDimsShlo(mlir::MlirOp reduction_input,
                               mlir::MlirOp reduction_output,
                               absl::Span<const int64_t> dims) {
  // Keep dims on a 0-D tensor is a no-op.
  if (GetTensorTypeOrDie(reduction_input).getRank() == 0) {
    return reduction_output;
  }
  mlir::FailureOr<mlir::stablehlo::Dimensions> input_dims =
      mlir::stablehlo::getDimensions(reduction_input.getValue());
  ABSL_CHECK(mlir::succeeded(input_dims));  // CRASH_OK: Internal invariant.
  mlir::stablehlo::Dimensions output_dims = std::move(*input_dims);
  for (int64_t dim : dims) {
    output_dims[dim] = {1};
  }
  mlir::Type output_element_type =
      GetTensorTypeOrDie(reduction_output).getElementType();
  return mlir::stablehlo::Reshape(
      mlir::stablehlo::getRankedTensorType(output_dims, output_element_type),
      reduction_output);
}

}  // namespace torch_tpu
