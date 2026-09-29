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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>

#include "absl/base/casts.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "csrc/common/libtpu_version.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

namespace {

// Returns the indices for a load / store of the trailing 2D tile of `arg`
// starting at (row_offset, col_offset).
SmallVector<Value> TileIndices(ImplicitLocOpBuilder& b, MemRefType arg_type,
                               int64_t row_offset, int64_t col_offset) {
  // The batch and head dimensions are simply 1 in the current implementation.
  ABSL_CHECK(  // CRASH_OK
      llvm::all_of(arg_type.getShape().drop_back(2),
                   [](int64_t dim) { return dim == 1; }));
  const int64_t rank = arg_type.getRank();
  Value i0 = arith::ConstantOp::create(b, b.getIndexAttr(0));
  SmallVector<Value> indices(rank, i0);
  if (row_offset != 0) {
    indices[rank - 2] =
        arith::ConstantOp::create(b, b.getIndexAttr(row_offset));
  }
  if (col_offset != 0) {
    indices[rank - 1] =
        arith::ConstantOp::create(b, b.getIndexAttr(col_offset));
  }
  return indices;
}

}  // namespace

TypedValue<VectorType> LoadTileSlice(ImplicitLocOpBuilder& b, Value arg,
                                     int64_t row_offset, int64_t num_rows,
                                     int64_t col_offset, int64_t num_cols) {
  MemRefType arg_type = cast<MemRefType>(arg.getType());
  SmallVector<Value> indices = TileIndices(b, arg_type, row_offset, col_offset);

  // Newer versions of LibTPU support sub-rank loads, but until we update in
  // OSS we must do it manually.
  SmallVector<int64_t> full_shape(arg_type.getRank(), 1);
  full_shape[full_shape.size() - 2] = num_rows;
  full_shape[full_shape.size() - 1] = num_cols;
  VectorType full_tile_type =
      VectorType::get(full_shape, arg_type.getElementType());
  Value full_tile = vector::LoadOp::create(b, full_tile_type, arg, indices);

  VectorType tile_type =
      VectorType::get({num_rows, num_cols}, arg_type.getElementType());
  return vector::ShapeCastOp::create(b, tile_type, full_tile);
}

TypedValue<VectorType> LoadTile(ImplicitLocOpBuilder& b, Value arg) {
  ArrayRef<int64_t> shape = cast<MemRefType>(arg.getType()).getShape();
  return LoadTileSlice(b, arg, /*row_offset=*/0, shape[shape.size() - 2],
                       /*col_offset=*/0, shape[shape.size() - 1]);
}

void StoreTileSlice(ImplicitLocOpBuilder& b, Value value, Value arg,
                    int64_t row_offset, int64_t col_offset) {
  MemRefType arg_type = cast<MemRefType>(arg.getType());
  SmallVector<Value> indices = TileIndices(b, arg_type, row_offset, col_offset);

  Value converted_value =
      ConvertElementType(b, arg_type.getElementType(), value);
  ArrayRef<int64_t> value_shape =
      cast<VectorType>(converted_value.getType()).getShape();

  // Newer versions of LibTPU support sub-rank stores, but until we update in
  // OSS we must do it manually.
  SmallVector<int64_t> full_shape(arg_type.getRank(), 1);
  full_shape[full_shape.size() - 2] = value_shape[0];
  full_shape[full_shape.size() - 1] = value_shape[1];
  VectorType full_tile_type =
      VectorType::get(full_shape, arg_type.getElementType());
  Value full_tile =
      vector::ShapeCastOp::create(b, full_tile_type, converted_value);

  vector::StoreOp::create(b, full_tile, arg, indices);
}

void StoreTile(ImplicitLocOpBuilder& b, Value value, Value arg) {
  StoreTileSlice(b, value, arg, /*row_offset=*/0, /*col_offset=*/0);
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

bool NeedsPadMask(int64_t qt, int64_t kt, int64_t q_seq_len,
                  int64_t kv_seq_len) {
  return (q_seq_len % qt) != 0 || (kv_seq_len % kt) != 0;
}

Value ApplyStructuredMask(ImplicitLocOpBuilder& b, Value sij,
                          Value row_block_idx, Value col_block_idx, int64_t qt,
                          int64_t kt, int64_t q_seq_len, int64_t kv_seq_len,
                          bool is_causal, int64_t row_offset,
                          int64_t col_offset) {
  bool mask_q_seq_len = (q_seq_len % qt) != 0;
  bool mask_kv_seq_len = (kv_seq_len % kt) != 0;

  if (!is_causal && !mask_q_seq_len && !mask_kv_seq_len) {
    return sij;
  }

  Type i32 = b.getI32Type();
  VectorType mask_shape_i32 = cast<VectorType>(sij.getType()).clone(i32);

  Value row_iota = CreateIotaOp(b, mask_shape_i32, 0);
  Value col_iota = CreateIotaOp(b, mask_shape_i32, 1);

  // Global index of the first row / column of this sub-block:
  //   block_idx * block_size + static offset within the block.
  auto qt_const = arith::ConstantOp::create(b, b.getI32IntegerAttr(qt));
  Value row_start = arith::MulIOp::create(b, row_block_idx, qt_const);
  if (row_offset != 0) {
    row_start = arith::AddIOp::create(
        b, row_start,
        arith::ConstantOp::create(b, b.getI32IntegerAttr(row_offset)));
  }
  Value row_start_vec =
      vector::BroadcastOp::create(b, mask_shape_i32, row_start);

  auto kt_const = arith::ConstantOp::create(b, b.getI32IntegerAttr(kt));
  Value col_start = arith::MulIOp::create(b, col_block_idx, kt_const);
  if (col_offset != 0) {
    col_start = arith::AddIOp::create(
        b, col_start,
        arith::ConstantOp::create(b, b.getI32IntegerAttr(col_offset)));
  }
  Value col_start_vec =
      vector::BroadcastOp::create(b, mask_shape_i32, col_start);

  Value row_indices = arith::AddIOp::create(b, row_iota, row_start_vec);
  Value col_indices = arith::AddIOp::create(b, col_iota, col_start_vec);

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

  Value masking_vector = SplatConstant(b, sij.getType(), kLogitsMin);
  return arith::SelectOp::create(b, valid_mask, sij, masking_vector);
}

Value CreateIntersectsDiagCondition(ImplicitLocOpBuilder& b, Value row_idx,
                                    Value col_idx, int64_t row_block_size,
                                    int64_t col_block_size) {
  // The tile contains a masked element iff its last column index is greater
  // than its first row index:
  //   (col_idx + 1) * col_block_size - 1 > row_idx * row_block_size.
  auto row_size_value =
      arith::ConstantOp::create(b, b.getI32IntegerAttr(row_block_size));
  auto col_size_value =
      arith::ConstantOp::create(b, b.getI32IntegerAttr(col_block_size));
  auto c1 = arith::ConstantOp::create(b, b.getI32IntegerAttr(1));

  Value col_idx_plus_1 = arith::AddIOp::create(b, col_idx, c1);
  Value last_col = arith::SubIOp::create(
      b, arith::MulIOp::create(b, col_idx_plus_1, col_size_value), c1);
  Value first_row = arith::MulIOp::create(b, row_idx, row_size_value);
  return arith::CmpIOp::create(b, arith::CmpIPredicate::sgt, last_col,
                               first_row);
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

void AppendScalarPrefetchTypes(MLIRContext* context,
                               const FlashAttnConfig& config,
                               SmallVectorImpl<Type>& types) {
  if (config.dropout_rate <= 0.0f) {
    return;
  }
  types.push_back(
      GetSmemMemRefType(context, {4}, IntegerType::get(context, 32)));
}

void SetScalarPrefetchAttr(const FlashAttnConfig& config, func::FuncOp fn,
                           OpBuilder& builder) {
  if (config.dropout_rate <= 0.0f) {
    return;
  }
  fn->setAttr("scalar_prefetch", builder.getI64IntegerAttr(1));
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
    AppendScalarPrefetchTypes(builder.getContext(), config, input_types);
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

Value BuildDropoutKeepMask(ImplicitLocOpBuilder& b,
                           const FlashAttnConfig& config, int64_t qt,
                           int64_t kt, Value batch_idx, Value head_idx,
                           Value row_block_idx, Value col_block_idx,
                           Value seed_smem, int64_t row_offset,
                           int64_t col_offset, int64_t rows, int64_t cols) {
  const int64_t mask_rows = rows > 0 ? rows : qt;
  const int64_t mask_cols = cols > 0 ? cols : kt;
  Type i32 = b.getI32Type();
  VectorType vec_i32 = VectorType::get({mask_rows, mask_cols}, i32);

  // Builds an i32 constant shaped like `type`. `fmix32` below runs on both a
  // scalar and a tile, and `arith` requires operands to match, so the constants
  // have to follow whichever it was handed. `SplatConstant` only accepts shaped
  // types, hence the scalar fallback.
  auto constant_like = [&](Type type, int64_t value) -> Value {
    IntegerAttr attr = b.getI32IntegerAttr(
        absl::bit_cast<int32_t>(static_cast<uint32_t>(value)));
    if (auto shaped_type = llvm::dyn_cast<ShapedType>(type)) {
      return SplatConstant(b, shaped_type, attr);
    }
    return arith::ConstantOp::create(b, type, attr);
  };

  // MurmurHash3's fmix32 avalanche; see
  // https://en.wikipedia.org/wiki/MurmurHash. It is a bijection on uint32, so
  // the packed key below cannot collide. The `arith` ops are type-polymorphic,
  // so this runs unchanged on the scalar and vector paths.
  auto fmix32 = [&](Value h) -> Value {
    Type type = h.getType();
    auto k = [&](uint32_t c) { return constant_like(type, c); };
    h = arith::XOrIOp::create(b, h, arith::ShRUIOp::create(b, h, k(16)));
    h = arith::MulIOp::create(b, h, k(0x85EBCA6Bu));
    h = arith::XOrIOp::create(b, h, arith::ShRUIOp::create(b, h, k(13)));
    h = arith::MulIOp::create(b, h, k(0xC2B2AE35u));
    h = arith::XOrIOp::create(b, h, arith::ShRUIOp::create(b, h, k(16)));
    return h;
  };

  auto c_idx = [&](int64_t idx) -> Value {
    return arith::ConstantOp::create(b, b.getIndexType(), b.getIndexAttr(idx));
  };

  // 1. Load the Philox {seed_lo, seed_hi, offset_lo, offset_hi} from the
  // scalar_prefetch SMEM buffer.
  Value seed_lo = memref::LoadOp::create(b, seed_smem, ValueRange{c_idx(0)});
  Value seed_hi = memref::LoadOp::create(b, seed_smem, ValueRange{c_idx(1)});
  Value offset_lo = memref::LoadOp::create(b, seed_smem, ValueRange{c_idx(2)});
  Value offset_hi = memref::LoadOp::create(b, seed_smem, ValueRange{c_idx(3)});

  // 2. Fold the stream identity into one word. This is scalar work, so it costs
  // once per tile rather than once per element, and we can afford to avalanche
  // after every term.
  Value stream = seed_lo;
  for (Value part : {seed_hi, offset_lo, offset_hi, batch_idx, head_idx}) {
    stream = fmix32(arith::XOrIOp::create(b, stream, part));
  }
  Value stream_vec = vector::BroadcastOp::create(b, vec_i32, stream);

  // 3. Absolute (q, kv) coordinates: grid block index * tile size + static
  // sub-block offset, plus the within-tile iota.
  Value q_start =
      arith::MulIOp::create(b, row_block_idx, constant_like(i32, qt));
  if (row_offset != 0) {
    q_start = arith::AddIOp::create(b, q_start, constant_like(i32, row_offset));
  }
  Value q_off = vector::BroadcastOp::create(b, vec_i32, q_start);
  Value kv_start =
      arith::MulIOp::create(b, col_block_idx, constant_like(i32, kt));
  if (col_offset != 0) {
    kv_start =
        arith::AddIOp::create(b, kv_start, constant_like(i32, col_offset));
  }
  Value kv_off = vector::BroadcastOp::create(b, vec_i32, kv_start);
  Value q_abs = arith::AddIOp::create(b, CreateIotaOp(b, vec_i32, 0), q_off);
  Value kv_abs = arith::AddIOp::create(b, CreateIotaOp(b, vec_i32, 1), kv_off);

  // 4. Pack the coordinates into a single word, then avalanche. The pack is
  // exactly injective while padded_q * padded_kv <= 2^32, so within a stream no
  // two positions can share a hash.
  Value key = arith::AddIOp::create(
      b,
      arith::MulIOp::create(
          b, q_abs, constant_like(vec_i32, config.padded_kv_sequence_length)),
      kv_abs);
  Value hash = fmix32(arith::XOrIOp::create(b, key, stream_vec));

  // 5. Keep iff uint32(hash) >= round(dropout_rate * 2^32). Only a signed
  // compare is available, so flip the sign bit on both sides.
  const uint32_t threshold = static_cast<uint32_t>(std::min<double>(
      std::floor(static_cast<double>(config.dropout_rate) * 4294967296.0),
      4294967295.0));
  Value hash_signed =
      arith::XOrIOp::create(b, hash, constant_like(vec_i32, 0x80000000u));
  return arith::CmpIOp::create(b, arith::CmpIPredicate::sge, hash_signed,
                               constant_like(vec_i32, threshold ^ 0x80000000u));
}

}  // namespace mlir::torch_tpu
