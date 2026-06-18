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

#include "torch_tpu/ops/cummax/cummax.h"

#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/scan_builder.h"

namespace torch_tpu {

absl::StatusOr<CummaxOutputs> BuildCummaxShlo(const int64_t dim,
                                              mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  TT_ASSIGN_OR_RETURN(const int64_t normalized_dim, SafeWrapDim(dim, rank));
  const llvm::ArrayRef<int64_t> shape = input_type.getShape();
  // chlo.ScanOp carries are rank-reduced (the scan dimension is erased).
  llvm::SmallVector<int64_t> carry_shape(shape.begin(), shape.end());
  carry_shape.erase(carry_shape.begin() + normalized_dim);

  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::Type element_type = input_type.getElementType();
  // Run the argmax in s32: the scan emitter's combiner only supports 4-byte
  // element types, and the kernel already caps the scan length at int32 max.
  // torch.cummax wants s64 indices, so the s32 index output is widened back to
  // s64 after the scan.
  const mlir::IntegerType i32 = builder.getOpBuilder().getI32Type();

  // The running argmax needs each element's position. An associative scan body
  // has no loop counter, so feed the positions in as an extra scan input.
  const mlir::RankedTensorType iota_type =
      mlir::RankedTensorType::get(shape, i32);
  const mlir::MlirOp iota =
      mlir::stablehlo::Iota(builder, iota_type, normalized_dim);

  const mlir::MlirOp value_init = mlir::stablehlo::Constant(
      builder,
      mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({}, element_type),
          GetMinFiniteValueAttr(element_type, builder.getOpBuilder())));
  const mlir::MlirOp index_init = MakeScalarConstant(builder, 0, i32);

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp value_carry_init,
                      BroadcastIfNeeded(value_init, carry_shape));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp index_carry_init,
                      BroadcastIfNeeded(index_init, carry_shape));

  // Cumulative max. On equal values keep the highest index, matching
  // torch.cummax. max with max-index-on-tie is associative (it selects the
  // (value, index) pair that is largest in lexicographic order), so the result
  // is independent of the scan's tree reordering (is_associative).
  MultiInputScanBodyBuilder body_builder =
      [](mlir::OpBuilder& op_builder, mlir::Location loc,
         mlir::ValueRange input_slices, mlir::Value /*index*/,
         mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    const mlir::Value cur_val = input_slices[0];
    const mlir::Value cur_idx = input_slices[1];
    const mlir::Value run_val = carries[0];
    const mlir::Value run_idx = carries[1];

    const mlir::Value new_val =
        mlir::stablehlo::MaxOp::create(op_builder, loc, run_val, cur_val)
            .getResult();
    const mlir::Value gt = mlir::stablehlo::CompareOp::create(
                               op_builder, loc, run_val, cur_val,
                               mlir::stablehlo::ComparisonDirection::GT)
                               .getResult();
    const mlir::Value eq = mlir::stablehlo::CompareOp::create(
                               op_builder, loc, run_val, cur_val,
                               mlir::stablehlo::ComparisonDirection::EQ)
                               .getResult();
    const mlir::Value max_idx =
        mlir::stablehlo::MaxOp::create(op_builder, loc, run_idx, cur_idx)
            .getResult();
    // run > cur -> run_idx; run == cur -> highest index; run < cur -> cur_idx.
    const mlir::Value tie_or_less =
        mlir::stablehlo::SelectOp::create(op_builder, loc, eq, max_idx, cur_idx)
            .getResult();
    const mlir::Value new_idx = mlir::stablehlo::SelectOp::create(
                                    op_builder, loc, gt, run_idx, tie_or_less)
                                    .getResult();
    llvm::SmallVector<mlir::Value> out = {new_val, new_idx};
    // Per-position outputs equal the running (value, index) carries.
    return ScanBodyResults{out, out};
  };

  // Associative scan -> chlo.ScanOp (native scan emitter). Results are
  // [carries..., outputs...]; the per-position outputs are the cummax results.
  TT_ASSIGN_OR_RETURN(
      const DynamicMlirOpResults results,
      BuildScanShlo(
          builder, {input, iota}, normalized_dim,
          /*num_scan_inputs=*/2,
          /*carry_inits=*/{value_carry_init, index_carry_init},
          /*output_inits=*/{input, iota}, std::move(body_builder),
          ScanOptions{.should_squeeze = true, .is_associative = true}));

  // torch.cummax returns s64 indices; widen the s32 scan result.
  const mlir::MlirOp indices = mlir::stablehlo::ConvertElementType(
      results[/*index output=*/3], builder.getOpBuilder().getI64Type());
  return CummaxOutputs{.values = results[/*value output=*/2],
                       .indices = indices};
}

}  // namespace torch_tpu
