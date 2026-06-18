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

#include "torch_tpu/ops/scan_builder.h"

#include <cstdint>
#include <iterator>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace {

mlir::Value GetScanDimSize(mlir::MlirBuilder& builder, mlir::MlirOp input,
                           const int64_t scan_dim) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  const mlir::IntegerType i64 = builder.getOpBuilder().getI64Type();
  if (!type.isDynamicDim(scan_dim)) {
    return MakeScalarConstant(builder, type.getShape()[scan_dim], i64)
        .getValue();
  }
  mlir::MlirOp size = mlir::stablehlo::GetDimensionSize(input, scan_dim);
  if (!size.getType().isInteger(64)) {
    size = mlir::stablehlo::ConvertElementType(size, i64);
  }
  return size.getValue();
}

llvm::SmallVector<mlir::MlirOp, 4> CreateStartIndices(
    mlir::MlirBuilder& builder, const mlir::Value index, const int64_t rank,
    const int64_t scan_dim) {
  const mlir::IntegerType i64 = builder.getOpBuilder().getI64Type();
  llvm::SmallVector<mlir::MlirOp, 4> indices(
      rank, MakeScalarConstant(builder, 0, i64));
  indices[scan_dim] = mlir::MlirOp(builder, index);
  return indices;
}

mlir::Value IncrementLoopIndex(mlir::MlirBuilder& builder,
                               const mlir::Location loc,
                               const mlir::Value index) {
  const mlir::IntegerType i64 = builder.getOpBuilder().getI64Type();
  mlir::MlirOp index_op(builder, index);
  mlir::MlirOp one = MakeScalarConstant(builder, 1, i64);
  return mlir::stablehlo::Add(index_op, one).getValue();
}

struct ScanLoopState {
  int64_t scan_dim;
  llvm::SmallVector<mlir::Type> loop_types;
  llvm::SmallVector<mlir::Value> loop_inits;
  mlir::Location loc;
};

absl::StatusOr<ScanLoopState> PrepareScanLoop(
    mlir::MlirBuilder& builder, mlir::MlirOp input, const int64_t dim,
    const llvm::ArrayRef<mlir::MlirOp> carry_inits,
    const llvm::ArrayRef<mlir::MlirOp> output_inits) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  TT_ASSIGN_OR_RETURN(const int64_t scan_dim,
                      SafeWrapDim(dim, GetTensorTypeOrDie(input).getRank()));

  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType index_type =
      mlir::RankedTensorType::get({}, i64);

  llvm::SmallVector<mlir::Type> loop_types = {index_type};
  llvm::SmallVector<mlir::Value> loop_inits = {
      MakeScalarConstant(builder, 0, i64).getValue()};

  for (mlir::MlirOp c : carry_inits) {
    loop_types.push_back(c.getType());
    loop_inits.push_back(c.getValue());
  }
  for (mlir::MlirOp o : output_inits) {
    loop_types.push_back(o.getType());
    loop_inits.push_back(o.getValue());
  }

  return ScanLoopState{scan_dim, std::move(loop_types), std::move(loop_inits),
                       input.getValue().getLoc()};
}

mlir::Value GetSequenceIndex(mlir::MlirBuilder& builder,
                             const mlir::Location loc, const mlir::Value index,
                             const mlir::Value seq_len, bool reverse) {
  if (!reverse) {
    return index;
  }

  mlir::MlirOp seq_len_op(builder, seq_len);
  mlir::MlirOp one =
      MakeScalarConstant(builder, 1, builder.getOpBuilder().getI64Type());
  mlir::MlirOp seq_len_minus_1 = mlir::stablehlo::Subtract(seq_len_op, one);

  mlir::MlirOp index_op(builder, index);
  return mlir::stablehlo::Subtract(seq_len_minus_1, index_op).getValue();
}

absl::StatusOr<mlir::MlirOp> SliceInput(
    mlir::MlirBuilder& builder, const mlir::Location loc, mlir::MlirOp input,
    const mlir::Value seq_idx, const int64_t scan_dim, bool should_squeeze) {
  const mlir::stablehlo::Dimensions input_dims = GetDimensions(input);
  llvm::SmallVector<int64_t> slice_shape = llvm::to_vector<4>(
      llvm::map_range(input_dims, [](const auto& d) { return d.size; }));
  slice_shape[scan_dim] = 1;

  const llvm::SmallVector<mlir::MlirOp, 4> start_indices =
      CreateStartIndices(builder, seq_idx, input_dims.size(), scan_dim);
  mlir::MlirOp slice =
      mlir::stablehlo::DynamicSlice(input, start_indices, slice_shape);

  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (i != scan_dim && type.isDynamicDim(i)) {
      mlir::MlirOp dim_size = mlir::stablehlo::GetDimensionSize(input, i);
      slice = mlir::stablehlo::SetDimensionSize(slice, dim_size, i);
    }
  }

  return should_squeeze ? Squeeze(slice, {scan_dim}) : slice;
}

absl::StatusOr<mlir::Value> UpdateAccumulatorSlice(
    mlir::MlirBuilder& builder, const mlir::Location loc,
    const mlir::Value accumulator, const mlir::Value slice,
    const mlir::Value seq_idx, const int64_t scan_dim, bool was_squeezed) {
  mlir::MlirOp slice_op(builder, slice);
  if (was_squeezed) {
    TT_ASSIGN_OR_RETURN(slice_op, Unsqueeze(slice_op, scan_dim));
  }

  mlir::MlirOp accumulator_op(builder, accumulator);
  const int64_t rank = GetTensorTypeOrDie(accumulator_op).getRank();
  return mlir::stablehlo::DynamicUpdateSlice(
             accumulator_op, slice_op,
             CreateStartIndices(builder, seq_idx, rank, scan_dim))
      .getValue();
}

void PopulateScanCondition(mlir::stablehlo::WhileOp& while_op,
                           mlir::MlirBuilder& builder, mlir::MlirOp input,
                           const int64_t scan_dim,
                           const llvm::SmallVector<mlir::Type>& loop_types,
                           const mlir::Location loc) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::Block* const block = op_builder.createBlock(&while_op.getCond());
  block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(block);

  const mlir::Value cond = mlir::stablehlo::CompareOp::create(
                               op_builder, loc, block->getArgument(0),
                               GetScanDimSize(builder, input, scan_dim),
                               mlir::stablehlo::ComparisonDirection::LT)
                               .getResult();
  mlir::stablehlo::ReturnOp::create(op_builder, loc, cond);
}

absl::Status PopulateScanBody(
    mlir::stablehlo::WhileOp& while_op, mlir::MlirBuilder& builder,
    llvm::ArrayRef<mlir::MlirOp> scan_inputs, const int64_t scan_dim,
    const llvm::SmallVector<mlir::Type>& loop_types, const mlir::Location loc,
    const int num_scan_inputs, const int num_carries,
    const MultiInputScanBodyBuilder& body_builder, ScanDirection scan_direction,
    bool should_squeeze) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::Block* const block = op_builder.createBlock(&while_op.getBody());
  block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(block);

  const mlir::Value index = block->getArgument(0);
  const mlir::ValueRange carries = block->getArguments().slice(1, num_carries);
  const mlir::ValueRange outputs = block->getArguments().slice(1 + num_carries);

  const mlir::Value seq_len = GetScanDimSize(builder, scan_inputs[0], scan_dim);
  const mlir::Value seq_idx = GetSequenceIndex(
      builder, loc, index, seq_len, scan_direction == ScanDirection::kReverse);

  llvm::SmallVector<mlir::Value> sliced_inputs;
  sliced_inputs.reserve(scan_inputs.size());
  for (const mlir::MlirOp input : scan_inputs.take_front(num_scan_inputs)) {
    TT_ASSIGN_OR_RETURN(
        const mlir::MlirOp sliced_input,
        SliceInput(builder, loc, input, seq_idx, scan_dim, should_squeeze));
    sliced_inputs.push_back(sliced_input.getValue());
  }
  for (mlir::MlirOp input : scan_inputs.drop_front(num_scan_inputs)) {
    sliced_inputs.push_back(input.getValue());
  }

  TT_ASSIGN_OR_RETURN(
      ScanBodyResults new_results,
      body_builder(op_builder, loc, sliced_inputs, index, carries));

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      new_results.new_carries.size() == num_carries, error::kInvalidArgument)
      << "expected " << num_carries << " new carries, got "
      << new_results.new_carries.size();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      new_results.new_outputs.size() == outputs.size(), error::kInvalidArgument)
      << "expected " << outputs.size() << " new outputs, got "
      << new_results.new_outputs.size();

  llvm::SmallVector<mlir::Value> next_state = {
      IncrementLoopIndex(builder, loc, index)};
  next_state.append(std::move(new_results.new_carries));

  for (auto [output, new_output] :
       llvm::zip(outputs, new_results.new_outputs)) {
    TT_ASSIGN_OR_RETURN(
        mlir::Value updated,
        UpdateAccumulatorSlice(builder, loc, output, new_output, seq_idx,
                               scan_dim, should_squeeze));
    next_state.push_back(updated);
  }

  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);
  return absl::OkStatus();
}

absl::Status VerifyScanDimSizes(llvm::ArrayRef<mlir::MlirOp> scan_inputs,
                                int64_t dim, int64_t num_scan_inputs) {
  if (num_scan_inputs <= 1) {
    return absl::OkStatus();
  }

  const mlir::RankedTensorType first_type =
      GetTensorTypeOrDie(scan_inputs.front());
  TT_ASSIGN_OR_RETURN(const int64_t first_scan_dim,
                      SafeWrapDim(dim, first_type.getRank()));
  const int64_t first_size = first_type.getShape()[first_scan_dim];

  for (int i = 1; i < num_scan_inputs; ++i) {
    const mlir::RankedTensorType current_type =
        GetTensorTypeOrDie(scan_inputs[i]);
    TT_ASSIGN_OR_RETURN(const int64_t current_scan_dim,
                        SafeWrapDim(dim, current_type.getRank()));
    const int64_t current_size = current_type.getShape()[current_scan_dim];
    if (first_size != mlir::ShapedType::kDynamic &&
        current_size != mlir::ShapedType::kDynamic) {
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
          first_size == current_size, error::kInvalidArgument)
          << "expected all scannable inputs to have matching sizes along the "
             "scan dimension, but input 0 has size "
          << first_size << " and input " << i << " has size " << current_size;
    }
  }

  return absl::OkStatus();
}

// The type of one position's slice along the scan dimension: the input type
// with the scan dimension erased (chlo.ScanOp passes rank-reduced slices to the
// body).
mlir::RankedTensorType RankReduceType(const mlir::RankedTensorType type,
                                      const int64_t scan_dim) {
  llvm::SmallVector<int64_t> shape(type.getShape().begin(),
                                   type.getShape().end());
  shape.erase(std::next(shape.begin(), scan_dim));
  return mlir::RankedTensorType::get(shape, type.getElementType());
}

// Assembles the body's inputs: the scanned slices (unsqueezed back to size-1
// when !should_squeeze, since chlo.ScanOp hands rank-reduced slices) followed
// by the static inputs passed through as-is.
absl::StatusOr<llvm::SmallVector<mlir::Value>> BuildScanBodyInputs(
    mlir::MlirBuilder& builder, mlir::Block* body,
    const llvm::ArrayRef<mlir::MlirOp> scan_inputs,
    const int64_t num_scan_inputs, const int64_t scan_dim,
    const bool should_squeeze) {
  llvm::SmallVector<mlir::Value> body_inputs;
  body_inputs.reserve(scan_inputs.size());
  for (const mlir::Value slice :
       body->getArguments().take_front(num_scan_inputs)) {
    if (should_squeeze) {
      body_inputs.push_back(slice);
      continue;
    }
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp unsqueezed,
                        Unsqueeze(mlir::MlirOp(builder, slice), scan_dim));
    body_inputs.push_back(unsqueezed.getValue());
  }
  for (const mlir::MlirOp static_input :
       scan_inputs.drop_front(num_scan_inputs)) {
    body_inputs.push_back(static_input.getValue());
  }
  return body_inputs;
}

// Squeezes the body's per-position outputs back to rank-reduced form (when the
// body produced size-1 outputs) so chlo.ScanOp can stack them along the scan
// dimension.
absl::StatusOr<llvm::SmallVector<mlir::Value>> SqueezeScanOutputs(
    mlir::MlirBuilder& builder, const llvm::ArrayRef<mlir::Value> new_outputs,
    const int64_t scan_dim, const bool should_squeeze) {
  llvm::SmallVector<mlir::Value> outputs;
  outputs.reserve(new_outputs.size());
  for (const mlir::Value out : new_outputs) {
    if (should_squeeze) {
      outputs.push_back(out);
      continue;
    }
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp squeezed,
                        Squeeze(mlir::MlirOp(builder, out), {scan_dim}));
    outputs.push_back(squeezed.getValue());
  }
  return outputs;
}

// Lowers an associative prefix scan to chlo.ScanOp, which the TPU compiler's
// native scan emitter compiles directly. Matches the contract of the
// generalized while-loop BuildScanShlo: returns [carries..., outputs...].
//
// chlo.ScanOp passes rank-reduced (scan dim erased) slices to its body and
// stacks rank-reduced body outputs back along the scan dimension. We bridge the
// while-loop body convention here: when !should_squeeze the body
// expects/returns size-1 (non-reduced) slices, so we unsqueeze the inputs
// before the body and squeeze the outputs after. There is no loop counter in an
// associative scan, so the body's index argument is null (associative ops
// needing a position pass an iota as an extra scan input).
absl::StatusOr<DynamicMlirOpResults> BuildAssociativeScanChlo(
    mlir::MlirBuilder& builder, const llvm::ArrayRef<mlir::MlirOp> scan_inputs,
    const int64_t scan_dim, const int64_t num_scan_inputs,
    const llvm::ArrayRef<mlir::MlirOp> carry_inits,
    const llvm::ArrayRef<mlir::MlirOp> output_inits,
    const MultiInputScanBodyBuilder& body_builder, const ScanOptions& options) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = scan_inputs.front().getValue().getLoc();
  const mlir::RankedTensorType first_type =
      GetTensorTypeOrDie(scan_inputs.front());
  const int64_t scan_dim_size = first_type.getShape()[scan_dim];

  // chlo.ScanOp inputs are only the scanned tensors; static inputs (beyond
  // num_scan_inputs) are captured and handed to the body as-is. A reverse scan
  // is expressed as a forward scan over reversed inputs with reversed outputs,
  // since the decomposition and emitter handle forward scans.
  const bool reverse = options.direction == ScanDirection::kReverse;
  const llvm::SmallVector<int64_t, 1> reverse_dims = {scan_dim};
  llvm::SmallVector<mlir::Value> scanned_inputs;
  llvm::SmallVector<mlir::Type> body_arg_types;
  llvm::SmallVector<mlir::Location> body_arg_locs;
  scanned_inputs.reserve(num_scan_inputs);
  for (int64_t i = 0; i < num_scan_inputs; ++i) {
    mlir::MlirOp scanned = scan_inputs[i];
    if (reverse) {
      scanned = mlir::stablehlo::Reverse(scanned, reverse_dims);
    }
    scanned_inputs.push_back(scanned.getValue());
    body_arg_types.push_back(
        RankReduceType(GetTensorTypeOrDie(scan_inputs[i]), scan_dim));
    body_arg_locs.push_back(loc);
  }

  llvm::SmallVector<mlir::Value> init_values;
  llvm::SmallVector<mlir::Type> carry_types;
  init_values.reserve(carry_inits.size());
  carry_types.reserve(carry_inits.size());
  for (const mlir::MlirOp c : carry_inits) {
    init_values.push_back(c.getValue());
    carry_types.push_back(c.getType());
    body_arg_types.push_back(c.getType());
    body_arg_locs.push_back(loc);
  }

  llvm::SmallVector<mlir::Type> output_types;
  output_types.reserve(output_inits.size());
  for (const mlir::MlirOp o : output_inits) {
    output_types.push_back(o.getType());
  }

  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::IntegerAttr scan_dim_size_attr =
      mlir::ShapedType::isDynamic(scan_dim_size)
          ? nullptr
          : op_builder.getIntegerAttr(i64, scan_dim_size);

  auto scan_op = mlir::chlo::ScanOp::create(
      op_builder, loc, /*outputs=*/output_types, /*carries=*/carry_types,
      /*inputs=*/scanned_inputs, /*inits=*/init_values,
      /*dimension=*/op_builder.getIntegerAttr(i64, scan_dim),
      /*scan_dim_size=*/scan_dim_size_attr,
      // Reverse is handled by reversing inputs/outputs above and below, so the
      // chlo.ScanOp itself is always forward.
      /*is_reverse=*/op_builder.getBoolAttr(false),
      /*is_associative=*/op_builder.getBoolAttr(true));

  mlir::Block* const body = op_builder.createBlock(
      &scan_op.getBody(), {}, body_arg_types, body_arg_locs);
  op_builder.setInsertionPointToStart(body);

  TT_ASSIGN_OR_RETURN(
      const llvm::SmallVector<mlir::Value> body_inputs,
      BuildScanBodyInputs(builder, body, scan_inputs, num_scan_inputs, scan_dim,
                          options.should_squeeze));
  const mlir::ValueRange carry_args =
      body->getArguments().drop_front(num_scan_inputs);

  TT_ASSIGN_OR_RETURN(ScanBodyResults results,
                      body_builder(op_builder, loc, body_inputs,
                                   /*index=*/mlir::Value(), carry_args));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      results.new_carries.size() == carry_inits.size(), error::kInvalidArgument)
      << "expected " << carry_inits.size() << " new carries, got "
      << results.new_carries.size();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      results.new_outputs.size() == output_inits.size(),
      error::kInvalidArgument)
      << "expected " << output_inits.size() << " new outputs, got "
      << results.new_outputs.size();

  // chlo.ScanOp stacks rank-reduced body outputs along the scan dim; the body
  // return order is [outputs..., carries...].
  TT_ASSIGN_OR_RETURN(llvm::SmallVector<mlir::Value> returned,
                      SqueezeScanOutputs(builder, results.new_outputs, scan_dim,
                                         options.should_squeeze));
  returned.append(results.new_carries.begin(), results.new_carries.end());
  mlir::stablehlo::ReturnOp::create(op_builder, loc, returned);

  op_builder.setInsertionPointAfter(scan_op);
  // Match the while-loop contract: [carries..., outputs...]. For a reverse scan
  // the per-position outputs are produced in reversed order, so flip them back
  // along the scan dim. The carries are the full reduction and are unaffected.
  DynamicMlirOpResults op_results;
  op_results.reserve(carry_inits.size() + output_inits.size());
  for (const mlir::Value carry : scan_op.getCarries()) {
    op_results.push_back(mlir::MlirOp(builder, carry));
  }
  for (const mlir::Value out : scan_op.getOutputs()) {
    mlir::MlirOp out_op(builder, out);
    if (reverse) {
      out_op = mlir::stablehlo::Reverse(out_op, reverse_dims);
    }
    op_results.push_back(out_op);
  }
  return op_results;
}

}  // namespace

absl::StatusOr<DynamicMlirOpResults> BuildScanShlo(
    mlir::MlirBuilder& builder, mlir::MlirOp input, const int64_t dim,
    const llvm::ArrayRef<mlir::MlirOp> carry_inits,
    const llvm::ArrayRef<mlir::MlirOp> output_inits,
    const ScanBodyBuilder body_builder, const ScanOptions& options) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(const int64_t scan_dim, SafeWrapDim(dim, type.getRank()));

  if (type.getShape()[scan_dim] == 0) {
    return DynamicMlirOpResults(output_inits.begin(), output_inits.end());
  }

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      carry_inits.size() == output_inits.size(), error::kInvalidArgument)
      << "expected the number of carry inits (" << carry_inits.size()
      << ") and the number of output inits (" << output_inits.size()
      << ") to match";

  // Wraps the single-slice body: for this overload the per-position output is
  // the running carry.
  MultiInputScanBodyBuilder multi_body_builder =
      [&body_builder](
          mlir::OpBuilder& op_builder, mlir::Location loc,
          mlir::ValueRange slices, mlir::Value index,
          mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    TT_ASSIGN_OR_RETURN(
        llvm::SmallVector<mlir::Value> new_carries,
        body_builder(op_builder, loc, slices[0], index, carries));
    llvm::SmallVector<mlir::Value> new_outputs = new_carries;
    return ScanBodyResults{std::move(new_carries), std::move(new_outputs)};
  };

  // Associative scan -> chlo.ScanOp (native scan emitter). This overload's body
  // takes size-1 (non-squeezed) slices and returns the outputs only.
  if (options.is_associative) {
    const ScanOptions emit_options = {.direction = options.direction,
                                      .should_squeeze = false,
                                      .is_associative = true};
    TT_ASSIGN_OR_RETURN(DynamicMlirOpResults results,
                        BuildAssociativeScanChlo(
                            builder, {input}, scan_dim,
                            /*num_scan_inputs=*/1, carry_inits, output_inits,
                            multi_body_builder, emit_options));
    return DynamicMlirOpResults(results.begin() + carry_inits.size(),
                                results.end());
  }

  TT_ASSIGN_OR_RETURN(
      ScanLoopState state,
      PrepareScanLoop(builder, input, dim, carry_inits, output_inits));

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto while_op = mlir::stablehlo::WhileOp::create(
      op_builder, state.loc, state.loop_types, state.loop_inits);

  PopulateScanCondition(while_op, builder, input, state.scan_dim,
                        state.loop_types, state.loc);

  TT_RETURN_IF_ERROR(PopulateScanBody(
      while_op, builder, {input}, state.scan_dim, state.loop_types, state.loc,
      /*num_scan_inputs=*/1, carry_inits.size(), multi_body_builder,
      ScanDirection::kForward, /*should_squeeze=*/false));

  op_builder.setInsertionPointAfter(while_op);
  DynamicMlirOpResults results;
  results.reserve(output_inits.size());
  for (int i = 0; i < output_inits.size(); ++i) {
    results.push_back(
        mlir::MlirOp(builder, while_op.getResult(1 + carry_inits.size() + i)));
  }

  return results;
}

absl::StatusOr<DynamicMlirOpResults> BuildScanShlo(
    mlir::MlirBuilder& builder, llvm::ArrayRef<mlir::MlirOp> scan_inputs,
    int64_t dim, int64_t num_scan_inputs,
    llvm::ArrayRef<mlir::MlirOp> carry_inits,
    llvm::ArrayRef<mlir::MlirOp> output_inits,
    MultiInputScanBodyBuilder body_builder, const ScanOptions& options) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      !scan_inputs.empty(), error::kInvalidArgument)
      << "expected at least 1 scan input, got none";
  const mlir::RankedTensorType type = GetTensorTypeOrDie(scan_inputs.front());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_scan_inputs <= scan_inputs.size(), error::kInvalidArgument)
      << "expected num_scan_inputs (" << num_scan_inputs
      << ") to be less than or equal to the number of inputs ("
      << scan_inputs.size() << ")";
  TT_ASSIGN_OR_RETURN(const int64_t scan_dim, SafeWrapDim(dim, type.getRank()));
  TT_RETURN_IF_ERROR(VerifyScanDimSizes(scan_inputs, dim, num_scan_inputs));

  if (type.getShape()[scan_dim] == 0) {
    DynamicMlirOpResults results(carry_inits.begin(), carry_inits.end());
    results.append(output_inits.begin(), output_inits.end());
    return results;
  }

  // Associative scans go to the native scan emitter via chlo.ScanOp;
  // non-associative scans use the StableHLO while loop below.
  if (options.is_associative) {
    return BuildAssociativeScanChlo(builder, scan_inputs, scan_dim,
                                    num_scan_inputs, carry_inits, output_inits,
                                    body_builder, options);
  }

  TT_ASSIGN_OR_RETURN(ScanLoopState state,
                      PrepareScanLoop(builder, scan_inputs.front(), dim,
                                      carry_inits, output_inits));

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto while_op = mlir::stablehlo::WhileOp::create(
      op_builder, state.loc, state.loop_types, state.loop_inits);

  PopulateScanCondition(while_op, builder, scan_inputs.front(), state.scan_dim,
                        state.loop_types, state.loc);

  TT_RETURN_IF_ERROR(PopulateScanBody(
      while_op, builder, scan_inputs, state.scan_dim, state.loop_types,
      state.loc, num_scan_inputs, carry_inits.size(), body_builder,
      options.direction, options.should_squeeze));

  op_builder.setInsertionPointAfter(while_op);
  const int64_t num_results = carry_inits.size() + output_inits.size();
  DynamicMlirOpResults results;
  results.reserve(num_results);
  // Start at index 1, skipping the loop counter.
  for (mlir::Value result : while_op.getResults().slice(1, num_results)) {
    results.push_back(mlir::MlirOp(builder, result));
  }

  return results;
}

}  // namespace torch_tpu
