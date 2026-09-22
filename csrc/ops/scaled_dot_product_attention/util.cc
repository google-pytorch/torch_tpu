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

#include "csrc/ops/scaled_dot_product_attention/util.h"

#include <cstdint>
#include <numbers>
#include <string>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "csrc/common/libtpu_version.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir::torch_tpu {

namespace {

// Exact value doesn't really matter as long as exp(min_val) is 0.
constexpr float kLogitsMin = -10000.0f;

Value SplatConstant(ImplicitLocOpBuilder& b, ShapedType type,
                    Attribute value_attr) {
  return arith::ConstantOp::create(b, DenseElementsAttr::get(type, value_attr));
}

Value SplatConstant(ImplicitLocOpBuilder& b, Type type, float value) {
  ShapedType shaped_type = cast<ShapedType>(type);
  Attribute value_attr = b.getFloatAttr(shaped_type.getElementType(), value);
  return SplatConstant(b, shaped_type, value_attr);
}

Value CreateReductionInitValue(ImplicitLocOpBuilder& b, FloatType element_type,
                               ArrayRef<int64_t> shape,
                               vector::CombiningKind kind) {
  VectorType type = VectorType::get(shape, element_type);

  if (kind == vector::CombiningKind::ADD) {
    return SplatConstant(b, type, 0.0f);
  }

  if (kind == vector::CombiningKind::MAXIMUMF) {
    auto ninf = b.getFloatAttr(element_type,
                               APFloat::getInf(element_type.getFloatSemantics(),
                                               /*Negative=*/true));
    return SplatConstant(b, type, ninf);
  }

  ABSL_CHECK(false)  // CRASH_OK
      << "Unsupported combining kind: "
      << mlir::vector::stringifyCombiningKind(kind).str();
}

}  // namespace

std::string GetOpString(Operation* op) {
  ABSL_CHECK(op != nullptr) << "Input op is null.";  // CRASH_OK
  std::string op_string;
  llvm::raw_string_ostream os(op_string);
  op->print(os, OpPrintingFlags().useLocalScope());
  return op_string;
}

TypedValue<VectorType> LoadTile(ImplicitLocOpBuilder& b, Value arg) {
  MemRefType arg_type = cast<MemRefType>(arg.getType());
  // The batch and head dimensions are simply 1 in the current implementation.
  ABSL_CHECK(  // CRASH_OK
      llvm::all_of(arg_type.getShape().drop_back(2),
                   [](int64_t dim) { return dim == 1; }));
  ArrayRef<int64_t> tile_shape = arg_type.getShape().take_back(2);
  VectorType tile_type = VectorType::get(tile_shape, arg_type.getElementType());

  Value i0 = arith::ConstantOp::create(b, b.getIndexAttr(0));
  SmallVector<Value> indices(arg_type.getRank(), i0);

  // Newer versions of LibTPU support sub-rank loads, but until we update in
  // OSS we must do it manually.
  VectorType full_tile_type =
      VectorType::get(arg_type.getShape(), arg_type.getElementType());
  Value full_tile = vector::LoadOp::create(b, full_tile_type, arg, indices);

  return vector::ShapeCastOp::create(b, tile_type, full_tile);
}

void StoreTile(ImplicitLocOpBuilder& b, Value value, Value arg) {
  MemRefType arg_type = cast<MemRefType>(arg.getType());

  // The batch and head dimensions are simply 1 in the current implementation.
  ABSL_CHECK(  // CRASH_OK
      llvm::all_of(arg_type.getShape().drop_back(2),
                   [](int64_t dim) { return dim == 1; }));
  Value i0 = arith::ConstantOp::create(b, b.getIndexAttr(0));
  SmallVector<Value> indices(arg_type.getRank(), i0);

  Value converted_value =
      ConvertElementType(b, arg_type.getElementType(), value);

  // Newer versions of LibTPU support sub-rank stores, but until we update in
  // OSS we must do it manually.
  VectorType full_tile_type =
      VectorType::get(arg_type.getShape(), arg_type.getElementType());
  Value full_tile =
      vector::ShapeCastOp::create(b, full_tile_type, converted_value);

  vector::StoreOp::create(b, full_tile, arg, indices);
}

void ZeroTile(ImplicitLocOpBuilder& b, Value arg) {
  MemRefType arg_type = cast<MemRefType>(arg.getType());
  VectorType tile_type = VectorType::get(arg_type.getShape().take_back(2),
                                         arg_type.getElementType());
  Value zero_vector = arith::ConstantOp::create(b, b.getZeroAttr(tile_type));
  StoreTile(b, zero_vector, arg);
}

void NInfTile(ImplicitLocOpBuilder& b, Value arg) {
  MemRefType arg_type = cast<MemRefType>(arg.getType());
  FloatType element_type = cast<FloatType>(arg_type.getElementType());
  VectorType tile_type =
      VectorType::get(arg_type.getShape().take_back(2), element_type);
  auto ninf = b.getFloatAttr(element_type,
                             APFloat::getInf(element_type.getFloatSemantics(),
                                             /*Negative=*/true));
  Value ninf_vector = SplatConstant(b, tile_type, ninf);
  StoreTile(b, ninf_vector, arg);
}

Value GetStructuredBias(ImplicitLocOpBuilder& b, Value row_block_idx,
                        Value col_block_idx, int64_t qt, int64_t kt,
                        int64_t q_seq_len, int64_t kv_seq_len, bool is_causal) {
  bool mask_q_seq_len = (q_seq_len % qt) != 0;
  bool mask_kv_seq_len = (kv_seq_len % kt) != 0;

  if (!is_causal && !mask_q_seq_len && !mask_kv_seq_len) {
    return nullptr;
  }

  Type i32 = b.getI32Type();
  VectorType mask_shape_i32 = VectorType::get({qt, kt}, i32);

  Value row_iota = CreateIotaOp(b, mask_shape_i32, 0);
  Value col_iota = CreateIotaOp(b, mask_shape_i32, 1);

  auto qt_const = arith::ConstantOp::create(b, b.getI32IntegerAttr(qt));
  Value row_offset = arith::MulIOp::create(b, row_block_idx, qt_const);
  Value row_offset_vec =
      vector::BroadcastOp::create(b, mask_shape_i32, row_offset);

  auto kt_const = arith::ConstantOp::create(b, b.getI32IntegerAttr(kt));
  Value col_offset = arith::MulIOp::create(b, col_block_idx, kt_const);
  Value col_offset_vec =
      vector::BroadcastOp::create(b, mask_shape_i32, col_offset);

  Value row_indices = arith::AddIOp::create(b, row_iota, row_offset_vec);
  Value col_indices = arith::AddIOp::create(b, col_iota, col_offset_vec);

  Value valid_mask = nullptr;
  if (is_causal) {
    valid_mask = arith::CmpIOp::create(b, arith::CmpIPredicate::sge,
                                       row_indices, col_indices);
  }

  if (mask_q_seq_len) {
    auto q_seq_len_const =
        arith::ConstantOp::create(b, b.getI32IntegerAttr(q_seq_len));
    Value q_seq_len_vec =
        vector::BroadcastOp::create(b, mask_shape_i32, q_seq_len_const);
    Value row_valid = arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                            row_indices, q_seq_len_vec);
    valid_mask = valid_mask ? arith::AndIOp::create(b, valid_mask, row_valid)
                            : row_valid;
  }

  if (mask_kv_seq_len) {
    auto kv_seq_len_const =
        arith::ConstantOp::create(b, b.getI32IntegerAttr(kv_seq_len));
    Value kv_seq_len_vec =
        vector::BroadcastOp::create(b, mask_shape_i32, kv_seq_len_const);
    Value col_valid = arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                            col_indices, kv_seq_len_vec);
    valid_mask = valid_mask ? arith::AndIOp::create(b, valid_mask, col_valid)
                            : col_valid;
  }

  VectorType bias_shape = VectorType::get({qt, kt}, b.getF32Type());
  Value zero_vector = SplatConstant(b, bias_shape, 0.0f);
  Value masking_vector = SplatConstant(b, bias_shape, kLogitsMin);

  return arith::SelectOp::create(b, valid_mask, zero_vector, masking_vector);
}

Value ReduceBroadcastLane(ImplicitLocOpBuilder& b, Value input,
                          vector::CombiningKind kind,
                          int64_t broadcast_lane_size) {
  VectorType input_type = cast<VectorType>(input.getType());
  FloatType element_type = cast<FloatType>(input_type.getElementType());
  ArrayRef<int64_t> input_shape = input_type.getShape();
  int64_t sublane_dim = input_shape[0];
  SmallVector<bool> reduction_mask = {false, true};

  Value init_value =
      CreateReductionInitValue(b, element_type, {sublane_dim}, kind);

  auto reduction_op = vector::MultiDimReductionOp::create(b, input, init_value,
                                                          reduction_mask, kind);

  auto cast_type = VectorType::get({sublane_dim, 1}, element_type);
  auto cast_op = vector::ShapeCastOp::create(b, cast_type, reduction_op);

  auto broadcast_type =
      VectorType::get({sublane_dim, broadcast_lane_size}, element_type);
  return vector::BroadcastOp::create(b, broadcast_type, cast_op);
}

Value ClampLogits(ImplicitLocOpBuilder& b, Value input) {
  VectorType input_type = cast<VectorType>(input.getType());
  Type element_type = input_type.getElementType();
  auto min_value_attr = b.getFloatAttr(element_type, kLogitsMin);
  auto min_finite_op = arith::ConstantOp::create(b, min_value_attr);
  auto min_value_broadcast =
      vector::BroadcastOp::create(b, input_type, min_finite_op);
  return arith::MaximumFOp::create(b, input, min_value_broadcast);
}

Value CreateBelowOrOnDiagCondition(ImplicitLocOpBuilder& b, Value row_idx,
                                   Value col_idx, int64_t row_block_size,
                                   int64_t col_block_size) {
  auto row_size_value =
      arith::ConstantOp::create(b, b.getI32IntegerAttr(row_block_size));
  auto col_size_value =
      arith::ConstantOp::create(b, b.getI32IntegerAttr(col_block_size));
  auto c1 = arith::ConstantOp::create(b, b.getI32IntegerAttr(1));

  Value row_idx_plus_1 = arith::AddIOp::create(b, row_idx, c1);
  Value lhs = arith::MulIOp::create(b, row_idx_plus_1, row_size_value);
  Value lhs_minus_1 = arith::SubIOp::create(b, lhs, c1);

  Value rhs = arith::MulIOp::create(b, col_idx, col_size_value);

  return arith::CmpIOp::create(b, arith::CmpIPredicate::sge, lhs_minus_1, rhs);
}

scf::IfOp CreateCausalIfOp(ImplicitLocOpBuilder& b, Value row_idx,
                           Value col_idx, int64_t row_block_size,
                           int64_t col_block_size) {
  Value cond = CreateBelowOrOnDiagCondition(b, row_idx, col_idx, row_block_size,
                                            col_block_size);
  return scf::IfOp::create(b, cond, /*withElse=*/false);
}

Value ScaleValue(ImplicitLocOpBuilder& b, Value input, float scale) {
  Value scale_value = SplatConstant(b, input.getType(), scale);
  return arith::MulFOp::create(b, input, scale_value);
}

Value ConvertElementType(ImplicitLocOpBuilder& b, Type target_element_type,
                         Value input) {
  VectorType input_type = cast<VectorType>(input.getType());
  Type input_element_type = input_type.getElementType();
  VectorType target_type = input_type.clone(target_element_type);

  int64_t input_bit_width = input_element_type.getIntOrFloatBitWidth();
  int64_t target_bit_width = target_element_type.getIntOrFloatBitWidth();

  if (input_bit_width < target_bit_width) {
    return arith::ExtFOp::create(b, target_type, input);
  }

  if (input_bit_width > target_bit_width) {
    return arith::TruncFOp::create(b, target_type, input);
  }

  ABSL_CHECK_EQ(input_element_type, target_element_type)  // CRASH_OK
      << "Input element type is not the same as target element type.";
  return input;
}

Value NormalizeLaneDim(ImplicitLocOpBuilder& builder, Value input,
                       int64_t target_lane_size) {
  auto input_vector = mlir::dyn_cast<mlir::TypedValue<mlir::VectorType>>(input);
  ABSL_CHECK(input_vector) << "Input must be a vector type.";  // CRASH_OK
  VectorType type = input_vector.getType();
  int64_t input_lane_size = type.getShape().back();
  if (input_lane_size == target_lane_size) {
    return input;
  }

  int64_t rank = type.getRank();
  SmallVector<int64_t> target_shape(type.getShape());
  target_shape.back() = target_lane_size;

  if (target_lane_size < input_lane_size) {
    return vector::ExtractStridedSliceOp::create(
        builder, input, SmallVector<int64_t>(rank, 0), target_shape,
        SmallVector<int64_t>(rank, 1));
  }

  return CreateRepeatOp(builder, input, rank - 1, target_lane_size);
}

DictionaryAttr CreateSymbolTransformIndicesAttr(
    Builder& builder, llvm::StringRef function_name,
    llvm::ArrayRef<int64_t> window_bounds) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr(
          "transform_indices",
          mlir::FlatSymbolRefAttr::get(builder.getContext(), function_name)),
      builder.getNamedAttr("window_bounds",
                           builder.getDenseI64ArrayAttr(window_bounds)),
  });
}

KVWindowMaps CreateKVWindowMaps(OpBuilder& builder, func::FuncOp fn,
                                const FlashAttnConfig& config,
                                const Tiling& tiling) {
  ABSL_DCHECK_GT(config.num_heads, 0);
  ABSL_DCHECK_GT(config.kv_num_heads, 0);
  ABSL_DCHECK_EQ(config.num_heads % config.kv_num_heads, 0);

  // GQA: Mosaic's AffineMap doesn't support FloorDiv, so we use a transform
  // function.
  ModuleOp module = fn->getParentOfType<ModuleOp>();
  auto kv_transform_fn =
      module ? module.lookupSymbol<func::FuncOp>("kv_transform_indices")
             : nullptr;
  if (!kv_transform_fn) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(fn);
    Location loc = fn.getLoc();
    Type i32 = builder.getI32Type();
    SmallVector<Type> input_types(4, i32);
    SmallVector<Type> output_types(4, i32);
    FunctionType fn_type = builder.getFunctionType(input_types, output_types);
    kv_transform_fn =
        func::FuncOp::create(builder, loc, "kv_transform_indices", fn_type);
    kv_transform_fn.setPrivate();
    Block* entry = kv_transform_fn.addEntryBlock();
    ImplicitLocOpBuilder fn_b = ImplicitLocOpBuilder::atBlockBegin(loc, entry);

    Value batch = entry->getArgument(0);
    Value head = entry->getArgument(1);
    Value q_tile = entry->getArgument(2);
    Value kv_tile = entry->getArgument(3);
    int64_t head_group_size = config.num_heads / config.kv_num_heads;
    Value group_size = arith::ConstantOp::create(
        fn_b, fn_b.getI32IntegerAttr(head_group_size));
    Value kv_head = arith::DivSIOp::create(fn_b, head, group_size);
    Value c0 = arith::ConstantOp::create(fn_b, fn_b.getI32IntegerAttr(0));

    Value next_kv_tile = kv_tile;
    if (config.is_causal) {
      // If the kv block is skipped, prefetch the next valid kv block, i.e. the
      // 0th one to be used for the next block_q rows.
      Value cond = CreateBelowOrOnDiagCondition(fn_b, q_tile, kv_tile,
                                                tiling.qt, tiling.kt);
      next_kv_tile = arith::SelectOp::create(fn_b, cond, kv_tile, c0);
    }

    func::ReturnOp::create(fn_b, ValueRange{batch, kv_head, next_kv_tile, c0});
  }

  DictionaryAttr k_map = CreateSymbolTransformIndicesAttr(
      builder, "kv_transform_indices", {1, 1, tiling.kt, config.qk_head_dim});

  DictionaryAttr v_map = CreateSymbolTransformIndicesAttr(
      builder, "kv_transform_indices", {1, 1, tiling.kt, config.vo_head_dim});

  return {k_map, v_map};
}

Value CreateLog2(ImplicitLocOpBuilder& b, Value input) {
  static bool is_natively_supported =
      ::torch_tpu::IsLibtpuVersionAtLeast("0.0.48");
  if (is_natively_supported) {
    return math::Log2Op::create(b, input);
  }
  float log2e = std::numbers::log2e_v<float>;
  Value natural_log = math::LogOp::create(b, input);
  return ScaleValue(b, natural_log, log2e);
}

}  // namespace mlir::torch_tpu
