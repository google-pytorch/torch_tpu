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

#include "csrc/mlir/passes/flatten_spatial_layernorm.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Returns true if type is a static ranked tensor type, optionally setting
// out_type.
[[nodiscard]] bool IsStaticRankedTensor(mlir::Type type,
                                        mlir::RankedTensorType* out_type) {
  auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked || !ranked.hasStaticShape()) {
    return false;
  }
  if (out_type != nullptr) {
    *out_type = ranked;
  }
  return true;
}

// Returns true if bcast_op broadcasts along all leading rank - 1 axes [0, ...,
// rank - 2].
[[nodiscard]] bool IsLeadingAxesBroadcast(
    mlir::stablehlo::BroadcastInDimOp bcast_op, int64_t rank) {
  llvm::ArrayRef<int64_t> dims = bcast_op.getBroadcastDimensions();
  if (static_cast<int64_t>(dims.size()) !=
      rank - 1) {  // GENERIC: semantic rank - 1 leading unreduced axes
    return false;
  }
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

// Returns true if val is a 1D parameter broadcast along trailing axis rank - 1.
[[nodiscard]] bool IsTrailingAxis1DBroadcast(mlir::Value val, int64_t rank,
                                             int64_t channels,
                                             mlir::Value* out_1d_param) {
  auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast_op) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = bcast_op.getBroadcastDimensions();
  if (dims.size() != 1 ||
      dims[0] != rank - 1) {  // GENERIC: semantic trailing normalized axis
    return false;
  }
  mlir::RankedTensorType param_type;
  if (!IsStaticRankedTensor(bcast_op.getOperand().getType(), &param_type) ||
      param_type.getRank() != 1 ||
      param_type.getDimSize(0) != channels) {  // GENERIC: semantic 1D param
    return false;
  }
  *out_1d_param = bcast_op.getOperand();
  return true;
}

// Matches an upstream channel-to-last transpose on a unit-batch tensor:
//   Pattern 1 (view_decomposition):
//     %r = stablehlo.reshape %src : (tensor<1xCxd_1x..xd_{R-2}>)
//                                -> tensor<Cxd_1x..xd_{R-2}x1>
//     %t = stablehlo.transpose %r, dims = [R-1, 1, ..., R-2, 0]
//                                -> tensor<1xd_1x..xd_{R-2}xC>
//   Pattern 2 (direct transpose):
//     %t = stablehlo.transpose %src, dims = [0, 2, ..., R-1, 1]
//                                -> tensor<1xd_1x..xd_{R-2}xC>
// Returns the channel-first source tensor if matched.
[[nodiscard]] mlir::Value MatchChannelToLastTranspose(
    mlir::Value ln_input, llvm::ArrayRef<int64_t> ln_shape,
    mlir::Block* block) {
  const int64_t rank = ln_shape.size();
  // GENERIC: semantic unit-batch >= 3D spatial tensor
  if (rank < 3 || ln_shape[0] != 1 || !ln_input.hasOneUse()) {
    return nullptr;
  }
  auto t_op = ln_input.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!t_op || (block != nullptr && t_op->getBlock() != block)) {
    return nullptr;
  }
  const int64_t channels = ln_shape[rank - 1];
  llvm::ArrayRef<int64_t> perm = t_op.getPermutation();
  if (static_cast<int64_t>(perm.size()) != rank) {
    return nullptr;
  }

  // Check Pattern 1: perm == [R - 1, 1, 2, ..., R - 2, 0] on ReshapeOp.
  bool is_pattern1 = (perm[0] == rank - 1 && perm[rank - 1] == 0);
  for (int64_t i = 1; i < rank - 1 && is_pattern1; ++i) {
    if (perm[i] != i) {
      is_pattern1 = false;
    }
  }
  if (is_pattern1 && t_op.getOperand().hasOneUse()) {
    if (auto r_op =
            t_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (block != nullptr && r_op->getBlock() != block) {
        return nullptr;
      }
      mlir::RankedTensorType src_type;
      if (IsStaticRankedTensor(r_op.getOperand().getType(), &src_type) &&
          src_type.getRank() == rank && src_type.getDimSize(0) == 1 &&
          src_type.getDimSize(1) == channels) {
        bool spatial_match = true;
        for (int64_t i = 1; i < rank - 1; ++i) {
          if (src_type.getDimSize(i + 1) != ln_shape[i]) {
            spatial_match = false;
            break;
          }
        }
        if (spatial_match) {
          return r_op.getOperand();
        }
      }
    }
  }

  // Check Pattern 2: perm == [0, 2, ..., R - 1, 1].
  bool is_pattern2 = (perm[0] == 0 && perm[rank - 1] == 1);
  for (int64_t i = 1; i < rank - 1 && is_pattern2; ++i) {
    if (perm[i] != i + 1) {
      is_pattern2 = false;
    }
  }
  if (is_pattern2) {
    mlir::RankedTensorType src_type;
    if (IsStaticRankedTensor(t_op.getOperand().getType(), &src_type) &&
        src_type.getRank() == rank && src_type.getDimSize(0) == 1 &&
        src_type.getDimSize(1) == channels) {
      bool spatial_match = true;
      for (int64_t i = 1; i < rank - 1; ++i) {
        if (src_type.getDimSize(i + 1) != ln_shape[i]) {
          spatial_match = false;
          break;
        }
      }
      if (spatial_match) {
        return t_op.getOperand();
      }
    }
  }

  return nullptr;
}

// Returns true if val is a splat constant or a 0D scalar broadcast matching
// stats_shape.
[[nodiscard]] bool IsMomentsLeafConstant(mlir::Value val,
                                         llvm::ArrayRef<int64_t> stats_shape) {
  mlir::RankedTensorType type;
  if (!IsStaticRankedTensor(val.getType(), &type) ||
      type.getShape() != stats_shape) {
    return false;
  }
  if (auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return dense && dense.isSplat();
  }
  if (auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    mlir::RankedTensorType op_type;
    return bcast_op.getBroadcastDimensions().empty() &&
           IsStaticRankedTensor(bcast_op.getOperand().getType(), &op_type) &&
           op_type.getRank() == 0;  // GENERIC: semantic 0D scalar broadcast
  }
  return false;
}

// Returns true if the moments slice computing val from sum_0 and sum_1 is pure
// and supported.
[[nodiscard]] bool CollectMomentsSlice(
    mlir::Value val, mlir::Value sum_0, mlir::Value sum_1,
    llvm::ArrayRef<int64_t> stats_shape,
    llvm::DenseSet<mlir::Operation*>& visited,
    llvm::SmallVectorImpl<mlir::Operation*>& order) {
  if (val == sum_0 || val == sum_1 || IsMomentsLeafConstant(val, stats_shape)) {
    return true;
  }
  mlir::RankedTensorType type;
  if (!IsStaticRankedTensor(val.getType(), &type) ||
      type.getShape() != stats_shape) {
    return false;
  }
  mlir::Operation* op = val.getDefiningOp();
  if (op == nullptr || op->getNumResults() != 1) {
    return false;
  }
  if (visited.contains(op)) {
    return true;
  }
  if (!llvm::isa<mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                 mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                 mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp,
                 mlir::stablehlo::RsqrtOp, mlir::stablehlo::SqrtOp,
                 mlir::stablehlo::ConvertOp, mlir::stablehlo::AbsOp,
                 mlir::stablehlo::NegOp>(op)) {
    return false;
  }
  for (mlir::Value operand : op->getOperands()) {
    if (!CollectMomentsSlice(operand, sum_0, sum_1, stats_shape, visited,
                             order)) {
      return false;
    }
  }
  visited.insert(op);
  order.push_back(op);
  return true;
}

bool TryFlattenSpatialLayerNorm(mlir::stablehlo::ReshapeOp flatten_op) {
  if (!flatten_op || flatten_op->getBlock() == nullptr) {
    return false;
  }
  mlir::Block* block = flatten_op->getBlock();
  mlir::RankedTensorType in_type;
  mlir::RankedTensorType out_type;
  if (!IsStaticRankedTensor(flatten_op.getOperand().getType(), &in_type) ||
      !IsStaticRankedTensor(flatten_op.getType(), &out_type)) {
    return false;
  }
  const int64_t rank = in_type.getRank();
  // GENERIC: semantic >= 3D tensor flattened to 2D [N, C]
  if (rank < 3 || out_type.getRank() != 2) {
    return false;
  }
  const int64_t channels = in_type.getDimSize(rank - 1);
  const int64_t num_rows = out_type.getDimSize(0);
  if (channels <= 0 || num_rows <= 0 || out_type.getDimSize(1) != channels) {
    return false;
  }
  int64_t unreduced_prod = 1;
  for (int64_t i = 0; i < rank - 1; ++i) {
    unreduced_prod *= in_type.getDimSize(i);
  }
  if (unreduced_prod != num_rows) {
    return false;
  }

  mlir::Value cur = flatten_op.getOperand();
  if (!cur.hasOneUse()) {
    return false;
  }

  // Peel optional bias AddOp.
  mlir::Value bias_1d;
  if (auto add_op = cur.getDefiningOp<mlir::stablehlo::AddOp>()) {
    if (add_op->getBlock() != block) {
      return false;
    }
    if (IsTrailingAxis1DBroadcast(add_op.getRhs(), rank, channels, &bias_1d) &&
        add_op.getLhs().hasOneUse()) {
      cur = add_op.getLhs();
    } else if (IsTrailingAxis1DBroadcast(add_op.getLhs(), rank, channels,
                                         &bias_1d) &&
               add_op.getRhs().hasOneUse()) {
      cur = add_op.getRhs();
    }
  }

  // Peel optional weight MulOp.
  mlir::Value weight_1d;
  if (auto mul_op = cur.getDefiningOp<mlir::stablehlo::MulOp>()) {
    if (mul_op->getBlock() != block) {
      return false;
    }
    if (IsTrailingAxis1DBroadcast(mul_op.getRhs(), rank, channels,
                                  &weight_1d) &&
        mul_op.getLhs().hasOneUse()) {
      cur = mul_op.getLhs();
    } else if (IsTrailingAxis1DBroadcast(mul_op.getLhs(), rank, channels,
                                         &weight_1d) &&
               mul_op.getRhs().hasOneUse()) {
      cur = mul_op.getRhs();
    }
  }

  // Peel optional downcast ConvertOp (f32 -> bf16/f16).
  bool has_downcast = false;
  if (auto down_conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    if (down_conv->getBlock() != block || !down_conv.getOperand().hasOneUse()) {
      return false;
    }
    has_downcast = true;
    cur = down_conv.getOperand();
  }

  // Match (input_casted - mean_bcast) * rstd_bcast.
  auto norm_mul = cur.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!norm_mul || norm_mul->getBlock() != block) {
    return false;
  }
  mlir::stablehlo::SubtractOp sub_op;
  mlir::stablehlo::BroadcastInDimOp rstd_bcast;
  if (auto s = norm_mul.getLhs().getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    sub_op = s;
    rstd_bcast =
        norm_mul.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  } else if (auto s = norm_mul.getRhs()
                          .getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    sub_op = s;
    rstd_bcast =
        norm_mul.getLhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  }
  if (!sub_op || sub_op->getBlock() != block || !sub_op->hasOneUse() ||
      !rstd_bcast || rstd_bcast->getBlock() != block ||
      !rstd_bcast->hasOneUse() || !IsLeadingAxesBroadcast(rstd_bcast, rank)) {
    return false;
  }
  auto mean_bcast =
      sub_op.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!mean_bcast || mean_bcast->getBlock() != block ||
      !mean_bcast->hasOneUse() || !IsLeadingAxesBroadcast(mean_bcast, rank)) {
    return false;
  }

  mlir::Value input_casted = sub_op.getLhs();
  mlir::RankedTensorType stats_tensor_type;
  if (!IsStaticRankedTensor(input_casted.getType(), &stats_tensor_type) ||
      stats_tensor_type.getShape() != in_type.getShape()) {
    return false;
  }

  // Match the 2-input, 2-result moments ReduceOp across the trailing axis.
  mlir::stablehlo::ReduceOp reduce_op;
  mlir::stablehlo::MulOp x_sq_op;
  for (mlir::Operation* user : input_casted.getUsers()) {
    if (auto r = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(user)) {
      // GENERIC: semantic 2-moment trailing reduction
      if (r.getInputs().size() == 2 && r.getNumResults() == 2 &&
          r.getDimensions() == llvm::ArrayRef<int64_t>{rank - 1}) {
        reduce_op = r;
      }
    } else if (auto m = llvm::dyn_cast<mlir::stablehlo::MulOp>(user)) {
      if (m.getLhs() == input_casted && m.getRhs() == input_casted) {
        x_sq_op = m;
      }
    }
  }
  if (!reduce_op || reduce_op->getBlock() != block || !x_sq_op ||
      x_sq_op->getBlock() != block || !x_sq_op->hasOneUse()) {
    return false;
  }
  if (!((reduce_op.getInputs()[0] == input_casted &&
         reduce_op.getInputs()[1] == x_sq_op.getResult()) ||
        (reduce_op.getInputs()[1] == input_casted &&
         reduce_op.getInputs()[0] == x_sq_op.getResult()))) {
    return false;
  }
  for (mlir::Operation* user : input_casted.getUsers()) {
    if (user != reduce_op.getOperation() && user != x_sq_op.getOperation() &&
        user != sub_op.getOperation()) {
      return false;
    }
  }

  mlir::Value ln_input = input_casted;
  if (auto up_conv = input_casted.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    mlir::RankedTensorType raw_type;
    if (up_conv->getBlock() == block &&
        IsStaticRankedTensor(up_conv.getOperand().getType(), &raw_type) &&
        raw_type.getShape() == in_type.getShape()) {
      ln_input = up_conv.getOperand();
    }
  }

  mlir::Value channel_first_src =
      MatchChannelToLastTranspose(ln_input, in_type.getShape(), block);

  // Collect and verify the moments DAG between reduce_op and {mean, rstd}.
  llvm::ArrayRef<int64_t> stats_shape = in_type.getShape().drop_back(1);
  mlir::Value sum_0 = reduce_op.getResult(0);
  mlir::Value sum_1 = reduce_op.getResult(1);
  mlir::Value mean_val = mean_bcast.getOperand();
  mlir::Value rstd_val = rstd_bcast.getOperand();

  llvm::DenseSet<mlir::Operation*> moments_set;
  llvm::SmallVector<mlir::Operation*, 8> moments_order;
  if (!CollectMomentsSlice(mean_val, sum_0, sum_1, stats_shape, moments_set,
                           moments_order) ||
      !CollectMomentsSlice(rstd_val, sum_0, sum_1, stats_shape, moments_set,
                           moments_order)) {
    return false;
  }

  for (mlir::Operation* op : moments_order) {
    if (op->getBlock() != block) {
      return false;
    }
  }

  // Ensure neither sum_0/sum_1 nor any op in moments_set escapes outside
  // moments_set + {mean_bcast, rstd_bcast}.
  auto all_uses_internal = [&](mlir::Value v) {
    for (mlir::Operation* u : v.getUsers()) {
      if (!moments_set.contains(u) && u != mean_bcast.getOperation() &&
          u != rstd_bcast.getOperation()) {
        return false;
      }
    }
    return true;
  };
  if (!all_uses_internal(sum_0) || !all_uses_internal(sum_1)) {
    return false;
  }
  for (mlir::Operation* op : moments_order) {
    if (!all_uses_internal(op->getResult(0))) {
      return false;
    }
  }

  // Rewrite the entire LayerNorm subgraph in 2D [N, C] and 1D [N].
  mlir::OpBuilder builder(flatten_op);
  mlir::Location loc = flatten_op.getLoc();
  mlir::Type in_elem_type =
      llvm::cast<mlir::RankedTensorType>(ln_input.getType()).getElementType();
  auto in_2d_type =
      mlir::RankedTensorType::get({num_rows, channels}, in_elem_type);
  mlir::Type stats_elem_type = stats_tensor_type.getElementType();
  auto stats_2d_type =
      mlir::RankedTensorType::get({num_rows, channels}, stats_elem_type);

  mlir::Value in_2d;
  if (channel_first_src) {
    auto cf_2d_type =
        mlir::RankedTensorType::get({channels, num_rows}, in_elem_type);
    mlir::Value r_2d = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, cf_2d_type, channel_first_src);
    in_2d = builder.create<mlir::stablehlo::TransposeOp>(
        loc, in_2d_type, r_2d, builder.getDenseI64ArrayAttr({1, 0}));
  } else {
    in_2d =
        builder.create<mlir::stablehlo::ReshapeOp>(loc, in_2d_type, ln_input);
  }

  mlir::Value input_casted_2d = in_2d;
  if (stats_elem_type != in_elem_type) {
    input_casted_2d =
        builder.create<mlir::stablehlo::ConvertOp>(loc, stats_2d_type, in_2d);
  }
  mlir::Value x_sq_2d = builder.create<mlir::stablehlo::MulOp>(
      loc, stats_2d_type, input_casted_2d, input_casted_2d);

  llvm::SmallVector<mlir::Value, 2> red_inputs(2);
  if (reduce_op.getInputs()[0] == input_casted) {
    red_inputs[0] = input_casted_2d;
    red_inputs[1] = x_sq_2d;
  } else {
    red_inputs[0] = x_sq_2d;
    red_inputs[1] = input_casted_2d;
  }

  auto sum0_type = llvm::cast<mlir::RankedTensorType>(sum_0.getType());
  auto sum1_type = llvm::cast<mlir::RankedTensorType>(sum_1.getType());
  auto sum0_1d_type =
      mlir::RankedTensorType::get({num_rows}, sum0_type.getElementType());
  auto sum1_1d_type =
      mlir::RankedTensorType::get({num_rows}, sum1_type.getElementType());

  mlir::OperationState red_state(reduce_op.getLoc(),
                                 mlir::stablehlo::ReduceOp::getOperationName());
  red_state.addOperands(red_inputs);
  red_state.addOperands(reduce_op.getInitValues());
  red_state.addTypes({sum0_1d_type, sum1_1d_type});
  red_state.addAttribute(
      "dimensions",
      builder.getDenseI64ArrayAttr({1}));  // GENERIC: semantic 2D trailing axis
  mlir::Region* new_region = red_state.addRegion();
  mlir::IRMapping region_map;
  reduce_op.getBody().cloneInto(new_region, region_map);
  mlir::Operation* new_reduce = builder.create(red_state);

  llvm::DenseMap<mlir::Value, mlir::Value> val_map;
  val_map[sum_0] = new_reduce->getResult(0);
  val_map[sum_1] = new_reduce->getResult(1);

  auto map_moments_val = [&](mlir::Value v) -> mlir::Value {
    if (auto it = val_map.find(v); it != val_map.end()) {
      return it->second;
    }
    auto orig_type = llvm::cast<mlir::RankedTensorType>(v.getType());
    auto new_1d_type =
        mlir::RankedTensorType::get({num_rows}, orig_type.getElementType());
    if (auto const_op = v.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
      auto dense = llvm::cast<mlir::DenseElementsAttr>(const_op.getValue());
      mlir::Value c1d = builder.create<mlir::stablehlo::ConstantOp>(
          const_op.getLoc(), new_1d_type, dense.resizeSplat(new_1d_type));
      val_map[v] = c1d;
      return c1d;
    }
    auto bcast_op = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    mlir::Value b1d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        bcast_op.getLoc(), new_1d_type, bcast_op.getOperand(),
        builder.getDenseI64ArrayAttr({}));
    val_map[v] = b1d;
    return b1d;
  };

  for (mlir::Operation* op : moments_order) {
    llvm::SmallVector<mlir::Value, 2> new_operands;
    for (mlir::Value operand : op->getOperands()) {
      new_operands.push_back(map_moments_val(operand));
    }
    mlir::OperationState st(op->getLoc(), op->getName());
    st.addOperands(new_operands);
    auto orig_res_type =
        llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto new_1d_res_type =
        mlir::RankedTensorType::get({num_rows}, orig_res_type.getElementType());
    st.addTypes(new_1d_res_type);
    st.addAttributes(op->getAttrs());
    mlir::Operation* cloned = builder.create(st);
    val_map[op->getResult(0)] = cloned->getResult(0);
  }

  mlir::Value mean_1d = map_moments_val(mean_val);
  mlir::Value rstd_1d = map_moments_val(rstd_val);

  auto mean_1d_type = llvm::cast<mlir::RankedTensorType>(mean_1d.getType());
  auto rstd_1d_type = llvm::cast<mlir::RankedTensorType>(rstd_1d.getType());
  auto mean_2d_type = mlir::RankedTensorType::get(
      {num_rows, channels}, mean_1d_type.getElementType());
  auto rstd_2d_type = mlir::RankedTensorType::get(
      {num_rows, channels}, rstd_1d_type.getElementType());

  mlir::Value mean_bcast_2d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      mean_bcast.getLoc(), mean_2d_type, mean_1d,
      builder.getDenseI64ArrayAttr({0}));  // GENERIC: semantic 2D row axis
  mlir::Value rstd_bcast_2d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      rstd_bcast.getLoc(), rstd_2d_type, rstd_1d,
      builder.getDenseI64ArrayAttr({0}));  // GENERIC: semantic 2D row axis

  mlir::Value sub_2d = builder.create<mlir::stablehlo::SubtractOp>(
      sub_op.getLoc(), stats_2d_type, input_casted_2d, mean_bcast_2d);
  mlir::Value norm_2d = builder.create<mlir::stablehlo::MulOp>(
      norm_mul.getLoc(), stats_2d_type, sub_2d, rstd_bcast_2d);

  mlir::Value result_2d = norm_2d;
  if (has_downcast) {
    result_2d =
        builder.create<mlir::stablehlo::ConvertOp>(loc, out_type, result_2d);
  }
  if (weight_1d) {
    auto w_type = llvm::cast<mlir::RankedTensorType>(weight_1d.getType());
    auto w_2d_type = mlir::RankedTensorType::get({num_rows, channels},
                                                 w_type.getElementType());
    mlir::Value w_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, w_2d_type, weight_1d,
        builder.getDenseI64ArrayAttr({1}));  // GENERIC: semantic 2D col axis
    auto res_type = llvm::cast<mlir::RankedTensorType>(result_2d.getType());
    if (w_2d_type.getElementType() != res_type.getElementType()) {
      w_bcast = builder.create<mlir::stablehlo::ConvertOp>(
          loc, result_2d.getType(), w_bcast);
    }
    result_2d = builder.create<mlir::stablehlo::MulOp>(loc, result_2d.getType(),
                                                       result_2d, w_bcast);
  }
  if (bias_1d) {
    auto b_type = llvm::cast<mlir::RankedTensorType>(bias_1d.getType());
    auto b_2d_type = mlir::RankedTensorType::get({num_rows, channels},
                                                 b_type.getElementType());
    mlir::Value b_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, b_2d_type, bias_1d,
        builder.getDenseI64ArrayAttr({1}));  // GENERIC: semantic 2D col axis
    auto res_type = llvm::cast<mlir::RankedTensorType>(result_2d.getType());
    if (b_2d_type.getElementType() != res_type.getElementType()) {
      b_bcast = builder.create<mlir::stablehlo::ConvertOp>(
          loc, result_2d.getType(), b_bcast);
    }
    result_2d = builder.create<mlir::stablehlo::AddOp>(loc, result_2d.getType(),
                                                       result_2d, b_bcast);
  }

  if (result_2d.getType() != out_type) {
    result_2d =
        builder.create<mlir::stablehlo::ConvertOp>(loc, out_type, result_2d);
  }

  flatten_op.replaceAllUsesWith(result_2d);
  flatten_op.erase();
  return true;
}

void EraseDeadPureOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (op != func.getOperation() && op->use_empty() &&
          (mlir::isOpTriviallyDead(op) ||
           llvm::isa<mlir::stablehlo::ReduceOp>(op))) {
        dead_ops.push_back(op);
      }
    });
    for (mlir::Operation* op : dead_ops) {
      op->erase();
      changed = true;
    }
  }
}

struct FlattenSpatialLayerNormPass
    : public mlir::PassWrapper<FlattenSpatialLayerNormPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenSpatialLayerNormPass)

  llvm::StringRef getArgument() const override {
    return "flatten-spatial-layernorm";
  }
  llvm::StringRef getDescription() const override {
    return "Hoists 2D spatial-flattening ReshapeOp upstream across "
           "channel-last LayerNorm and folds upstream channel-to-last "
           "transposes.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshapes;
    func.walk([&](mlir::stablehlo::ReshapeOp op) { reshapes.push_back(op); });
    bool any_changed = false;
    for (mlir::stablehlo::ReshapeOp r : reshapes) {
      if (TryFlattenSpatialLayerNorm(r)) {
        any_changed = true;
      }
    }
    if (any_changed) {
      EraseDeadPureOps(func);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFlattenSpatialLayerNormPass() {
  return std::make_unique<FlattenSpatialLayerNormPass>();
}

}  // namespace torch_tpu
