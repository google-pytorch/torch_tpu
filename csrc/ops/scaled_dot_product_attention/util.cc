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
#include "csrc/common/version.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
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

namespace torch_tpu {

namespace {

mlir::Value CreateReductionInitValue(mlir::ImplicitLocOpBuilder& b,
                                     mlir::FloatType element_type,
                                     mlir::ArrayRef<int64_t> shape,
                                     mlir::vector::CombiningKind kind) {
  mlir::VectorType type = mlir::VectorType::get(shape, element_type);

  if (kind == mlir::vector::CombiningKind::ADD) {
    return SplatConstant(b, type, 0.0f);
  }

  if (kind == mlir::vector::CombiningKind::MAXIMUMF) {
    auto ninf = b.getFloatAttr(
        element_type, mlir::APFloat::getInf(element_type.getFloatSemantics(),
                                            /*Negative=*/true));
    return SplatConstant(b, type, ninf);
  }

  ABSL_CHECK(false)  // CRASH_OK
      << "Unsupported combining kind: "
      << mlir::vector::stringifyCombiningKind(kind).str();
}

// Returns the indices for a load / store of the trailing 2D tile of `arg`
// starting at (row_offset, col_offset).
mlir::SmallVector<mlir::Value> TileIndices(mlir::ImplicitLocOpBuilder& b,
                                           mlir::MemRefType arg_type,
                                           int64_t row_offset,
                                           int64_t col_offset) {
  // The batch and head dimensions are simply 1 in the current implementation.
  ABSL_CHECK(  // CRASH_OK
      llvm::all_of(arg_type.getShape().drop_back(2),
                   [](int64_t dim) { return dim == 1; }));
  const int64_t rank = arg_type.getRank();
  mlir::Value i0 = mlir::arith::ConstantOp::create(b, b.getIndexAttr(0));
  mlir::SmallVector<mlir::Value> indices(rank, i0);
  if (row_offset != 0) {
    indices[rank - 2] =
        mlir::arith::ConstantOp::create(b, b.getIndexAttr(row_offset));
  }
  if (col_offset != 0) {
    indices[rank - 1] =
        mlir::arith::ConstantOp::create(b, b.getIndexAttr(col_offset));
  }
  return indices;
}

}  // namespace

std::string GetOpString(mlir::Operation* op) {
  ABSL_CHECK(op != nullptr) << "Input op is null.";  // CRASH_OK
  std::string op_string;
  llvm::raw_string_ostream os(op_string);
  op->print(os, mlir::OpPrintingFlags().useLocalScope());
  return op_string;
}

mlir::Value SplatConstant(mlir::ImplicitLocOpBuilder& b, mlir::ShapedType type,
                          mlir::Attribute value_attr) {
  return mlir::arith::ConstantOp::create(
      b, mlir::DenseElementsAttr::get(type, value_attr));
}

mlir::Value SplatConstant(mlir::ImplicitLocOpBuilder& b, mlir::Type type,
                          float value) {
  mlir::ShapedType shaped_type = mlir::cast<mlir::ShapedType>(type);
  mlir::Attribute value_attr =
      b.getFloatAttr(shaped_type.getElementType(), value);
  return SplatConstant(b, shaped_type, value_attr);
}

mlir::Value SplatNegInf(mlir::ImplicitLocOpBuilder& b, mlir::Type type) {
  mlir::ShapedType shaped_type = mlir::cast<mlir::ShapedType>(type);
  mlir::FloatType element_type =
      mlir::cast<mlir::FloatType>(shaped_type.getElementType());
  mlir::Attribute value_attr = b.getFloatAttr(
      element_type, mlir::APFloat::getInf(element_type.getFloatSemantics(),
                                          /*Negative=*/true));
  return SplatConstant(b, shaped_type, value_attr);
}

mlir::TypedValue<mlir::VectorType> LoadTileSlice(
    mlir::ImplicitLocOpBuilder& b, mlir::Value arg, int64_t row_offset,
    int64_t num_rows, int64_t col_offset, int64_t num_cols) {
  mlir::MemRefType arg_type = mlir::cast<mlir::MemRefType>(arg.getType());
  mlir::SmallVector<mlir::Value> indices =
      TileIndices(b, arg_type, row_offset, col_offset);

  // Newer versions of LibTPU support sub-rank loads, but until we update in
  // OSS we must do it manually.
  mlir::SmallVector<int64_t> full_shape(arg_type.getRank(), 1);
  full_shape[full_shape.size() - 2] = num_rows;
  full_shape[full_shape.size() - 1] = num_cols;
  mlir::VectorType full_tile_type =
      mlir::VectorType::get(full_shape, arg_type.getElementType());
  mlir::Value full_tile =
      mlir::vector::LoadOp::create(b, full_tile_type, arg, indices);

  mlir::VectorType tile_type =
      mlir::VectorType::get({num_rows, num_cols}, arg_type.getElementType());
  return mlir::vector::ShapeCastOp::create(b, tile_type, full_tile);
}

mlir::TypedValue<mlir::VectorType> LoadTile(mlir::ImplicitLocOpBuilder& b,
                                            mlir::Value arg) {
  mlir::ArrayRef<int64_t> shape =
      mlir::cast<mlir::MemRefType>(arg.getType()).getShape();
  return LoadTileSlice(b, arg, /*row_offset=*/0, shape[shape.size() - 2],
                       /*col_offset=*/0, shape[shape.size() - 1]);
}

void StoreTileSlice(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
                    mlir::Value arg, int64_t row_offset, int64_t col_offset) {
  mlir::MemRefType arg_type = mlir::cast<mlir::MemRefType>(arg.getType());
  mlir::SmallVector<mlir::Value> indices =
      TileIndices(b, arg_type, row_offset, col_offset);

  mlir::Value converted_value =
      ConvertElementType(b, arg_type.getElementType(), value);
  mlir::ArrayRef<int64_t> value_shape =
      mlir::cast<mlir::VectorType>(converted_value.getType()).getShape();

  // Newer versions of LibTPU support sub-rank stores, but until we update in
  // OSS we must do it manually.
  mlir::SmallVector<int64_t> full_shape(arg_type.getRank(), 1);
  full_shape[full_shape.size() - 2] = value_shape[0];
  full_shape[full_shape.size() - 1] = value_shape[1];
  mlir::VectorType full_tile_type =
      mlir::VectorType::get(full_shape, arg_type.getElementType());
  mlir::Value full_tile =
      mlir::vector::ShapeCastOp::create(b, full_tile_type, converted_value);

  mlir::vector::StoreOp::create(b, full_tile, arg, indices);
}

void StoreTile(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
               mlir::Value arg) {
  StoreTileSlice(b, value, arg, /*row_offset=*/0, /*col_offset=*/0);
}

void ZeroTile(mlir::ImplicitLocOpBuilder& b, mlir::Value arg) {
  mlir::MemRefType arg_type = mlir::cast<mlir::MemRefType>(arg.getType());
  mlir::VectorType tile_type = mlir::VectorType::get(
      arg_type.getShape().take_back(2), arg_type.getElementType());
  mlir::Value zero_vector =
      mlir::arith::ConstantOp::create(b, b.getZeroAttr(tile_type));
  StoreTile(b, zero_vector, arg);
}

void NInfTile(mlir::ImplicitLocOpBuilder& b, mlir::Value arg) {
  mlir::MemRefType arg_type = mlir::cast<mlir::MemRefType>(arg.getType());
  mlir::FloatType element_type =
      mlir::cast<mlir::FloatType>(arg_type.getElementType());
  mlir::VectorType tile_type =
      mlir::VectorType::get(arg_type.getShape().take_back(2), element_type);
  auto ninf = b.getFloatAttr(
      element_type, mlir::APFloat::getInf(element_type.getFloatSemantics(),
                                          /*Negative=*/true));
  mlir::Value ninf_vector = SplatConstant(b, tile_type, ninf);
  StoreTile(b, ninf_vector, arg);
}

bool NeedsPadMask(int64_t qt, int64_t kt, int64_t q_seq_len,
                  int64_t kv_seq_len) {
  return (q_seq_len % qt) != 0 || (kv_seq_len % kt) != 0;
}

mlir::Value ApplyStructuredMask(mlir::ImplicitLocOpBuilder& b, mlir::Value sij,
                                mlir::Value row_block_idx,
                                mlir::Value col_block_idx, int64_t qt,
                                int64_t kt, int64_t q_seq_len,
                                int64_t kv_seq_len, bool is_causal,
                                int64_t row_offset, int64_t col_offset) {
  bool mask_q_seq_len = (q_seq_len % qt) != 0;
  bool mask_kv_seq_len = (kv_seq_len % kt) != 0;

  if (!is_causal && !mask_q_seq_len && !mask_kv_seq_len) {
    return sij;
  }

  mlir::Type i32 = b.getI32Type();
  mlir::VectorType mask_shape_i32 =
      mlir::cast<mlir::VectorType>(sij.getType()).clone(i32);

  mlir::Value row_iota = CreateIotaOp(b, mask_shape_i32, 0);
  mlir::Value col_iota = CreateIotaOp(b, mask_shape_i32, 1);

  // Global index of the first row / column of this sub-block:
  //   block_idx * block_size + static offset within the block.
  auto qt_const = mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(qt));
  mlir::Value row_start =
      mlir::arith::MulIOp::create(b, row_block_idx, qt_const);
  if (row_offset != 0) {
    row_start = mlir::arith::AddIOp::create(
        b, row_start,
        mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(row_offset)));
  }
  mlir::Value row_start_vec =
      mlir::vector::BroadcastOp::create(b, mask_shape_i32, row_start);

  auto kt_const = mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(kt));
  mlir::Value col_start =
      mlir::arith::MulIOp::create(b, col_block_idx, kt_const);
  if (col_offset != 0) {
    col_start = mlir::arith::AddIOp::create(
        b, col_start,
        mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(col_offset)));
  }
  mlir::Value col_start_vec =
      mlir::vector::BroadcastOp::create(b, mask_shape_i32, col_start);

  mlir::Value row_indices =
      mlir::arith::AddIOp::create(b, row_iota, row_start_vec);
  mlir::Value col_indices =
      mlir::arith::AddIOp::create(b, col_iota, col_start_vec);

  mlir::Value valid_mask = nullptr;
  if (is_causal) {
    valid_mask = mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::sge,
                                             row_indices, col_indices);
  }

  if (mask_q_seq_len) {
    auto q_seq_len_const =
        mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(q_seq_len));
    mlir::Value q_seq_len_vec =
        mlir::vector::BroadcastOp::create(b, mask_shape_i32, q_seq_len_const);
    mlir::Value row_valid = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::slt, row_indices, q_seq_len_vec);
    valid_mask = valid_mask
                     ? mlir::arith::AndIOp::create(b, valid_mask, row_valid)
                     : row_valid;
  }

  if (mask_kv_seq_len) {
    auto kv_seq_len_const =
        mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(kv_seq_len));
    mlir::Value kv_seq_len_vec =
        mlir::vector::BroadcastOp::create(b, mask_shape_i32, kv_seq_len_const);
    mlir::Value col_valid = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::slt, col_indices, kv_seq_len_vec);
    valid_mask = valid_mask
                     ? mlir::arith::AndIOp::create(b, valid_mask, col_valid)
                     : col_valid;
  }

  mlir::Value masking_vector = SplatNegInf(b, sij.getType());
  return mlir::arith::SelectOp::create(b, valid_mask, sij, masking_vector);
}

mlir::Value CreateIntersectsDiagCondition(mlir::ImplicitLocOpBuilder& b,
                                          mlir::Value row_idx,
                                          mlir::Value col_idx,
                                          int64_t row_block_size,
                                          int64_t col_block_size) {
  // The tile contains a masked element iff its last column index is greater
  // than its first row index:
  //   (col_idx + 1) * col_block_size - 1 > row_idx * row_block_size.
  auto row_size_value =
      mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(row_block_size));
  auto col_size_value =
      mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(col_block_size));
  auto c1 = mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(1));

  mlir::Value col_idx_plus_1 = mlir::arith::AddIOp::create(b, col_idx, c1);
  mlir::Value last_col = mlir::arith::SubIOp::create(
      b, mlir::arith::MulIOp::create(b, col_idx_plus_1, col_size_value), c1);
  mlir::Value first_row =
      mlir::arith::MulIOp::create(b, row_idx, row_size_value);
  return mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::sgt,
                                     last_col, first_row);
}

mlir::Value ReduceBroadcastLane(mlir::ImplicitLocOpBuilder& b,
                                mlir::Value input,
                                mlir::vector::CombiningKind kind,
                                int64_t broadcast_lane_size) {
  mlir::VectorType input_type = mlir::cast<mlir::VectorType>(input.getType());
  mlir::FloatType element_type =
      mlir::cast<mlir::FloatType>(input_type.getElementType());
  mlir::ArrayRef<int64_t> input_shape = input_type.getShape();
  int64_t sublane_dim = input_shape[0];
  mlir::SmallVector<bool> reduction_mask = {false, true};

  mlir::Value init_value =
      CreateReductionInitValue(b, element_type, {sublane_dim}, kind);

  auto reduction_op = mlir::vector::MultiDimReductionOp::create(
      b, input, init_value, reduction_mask, kind);

  auto cast_type = mlir::VectorType::get({sublane_dim, 1}, element_type);
  auto cast_op = mlir::vector::ShapeCastOp::create(b, cast_type, reduction_op);

  auto broadcast_type =
      mlir::VectorType::get({sublane_dim, broadcast_lane_size}, element_type);
  return mlir::vector::BroadcastOp::create(b, broadcast_type, cast_op);
}

mlir::Value ClampMinFinite(mlir::ImplicitLocOpBuilder& b, mlir::Value input) {
  mlir::VectorType input_type = mlir::cast<mlir::VectorType>(input.getType());
  mlir::FloatType element_type =
      mlir::cast<mlir::FloatType>(input_type.getElementType());
  auto min_value_attr = b.getFloatAttr(
      element_type, mlir::APFloat::getLargest(element_type.getFloatSemantics(),
                                              /*Negative=*/true));
  auto min_finite_op = mlir::arith::ConstantOp::create(b, min_value_attr);
  auto min_value_broadcast =
      mlir::vector::BroadcastOp::create(b, input_type, min_finite_op);
  return mlir::arith::MaximumFOp::create(b, input, min_value_broadcast);
}

mlir::Value CreateBelowOrOnDiagCondition(mlir::ImplicitLocOpBuilder& b,
                                         mlir::Value row_idx,
                                         mlir::Value col_idx,
                                         int64_t row_block_size,
                                         int64_t col_block_size) {
  auto row_size_value =
      mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(row_block_size));
  auto col_size_value =
      mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(col_block_size));
  auto c1 = mlir::arith::ConstantOp::create(b, b.getI32IntegerAttr(1));

  mlir::Value row_idx_plus_1 = mlir::arith::AddIOp::create(b, row_idx, c1);
  mlir::Value lhs =
      mlir::arith::MulIOp::create(b, row_idx_plus_1, row_size_value);
  mlir::Value lhs_minus_1 = mlir::arith::SubIOp::create(b, lhs, c1);

  mlir::Value rhs = mlir::arith::MulIOp::create(b, col_idx, col_size_value);

  return mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::sge,
                                     lhs_minus_1, rhs);
}

mlir::scf::IfOp CreateCausalIfOp(mlir::ImplicitLocOpBuilder& b,
                                 mlir::Value row_idx, mlir::Value col_idx,
                                 int64_t row_block_size,
                                 int64_t col_block_size) {
  mlir::Value cond = CreateBelowOrOnDiagCondition(
      b, row_idx, col_idx, row_block_size, col_block_size);
  return mlir::scf::IfOp::create(b, cond, /*withElse=*/false);
}

mlir::Value ScaleValue(mlir::ImplicitLocOpBuilder& b, mlir::Value input,
                       float scale) {
  mlir::Value scale_value = SplatConstant(b, input.getType(), scale);
  return mlir::arith::MulFOp::create(b, input, scale_value);
}

mlir::Value ConvertElementType(mlir::ImplicitLocOpBuilder& b,
                               mlir::Type target_element_type,
                               mlir::Value input) {
  mlir::VectorType input_type = mlir::cast<mlir::VectorType>(input.getType());
  mlir::Type input_element_type = input_type.getElementType();
  mlir::VectorType target_type = input_type.clone(target_element_type);

  int64_t input_bit_width = input_element_type.getIntOrFloatBitWidth();
  int64_t target_bit_width = target_element_type.getIntOrFloatBitWidth();

  if (input_bit_width < target_bit_width) {
    return mlir::arith::ExtFOp::create(b, target_type, input);
  }

  if (input_bit_width > target_bit_width) {
    return mlir::arith::TruncFOp::create(b, target_type, input);
  }

  ABSL_CHECK_EQ(input_element_type, target_element_type)  // CRASH_OK
      << "Input element type is not the same as target element type.";
  return input;
}

mlir::Value NormalizeLaneDim(mlir::ImplicitLocOpBuilder& builder,
                             mlir::Value input, int64_t target_lane_size) {
  auto input_vector = mlir::dyn_cast<mlir::TypedValue<mlir::VectorType>>(input);
  ABSL_CHECK(input_vector) << "Input must be a vector type.";  // CRASH_OK
  mlir::VectorType type = input_vector.getType();
  int64_t input_lane_size = type.getShape().back();
  if (input_lane_size == target_lane_size) {
    return input;
  }

  int64_t rank = type.getRank();
  mlir::SmallVector<int64_t> target_shape(type.getShape());
  target_shape.back() = target_lane_size;

  if (target_lane_size < input_lane_size) {
    return mlir::vector::ExtractStridedSliceOp::create(
        builder, input, mlir::SmallVector<int64_t>(rank, 0), target_shape,
        mlir::SmallVector<int64_t>(rank, 1));
  }

  return CreateRepeatOp(builder, input, rank - 1, target_lane_size);
}

mlir::DictionaryAttr CreateSymbolTransformIndicesAttr(
    mlir::Builder& builder, llvm::StringRef function_name,
    llvm::ArrayRef<int64_t> window_bounds) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr(
          "transform_indices",
          mlir::FlatSymbolRefAttr::get(builder.getContext(), function_name)),
      builder.getNamedAttr("window_bounds",
                           builder.getDenseI64ArrayAttr(window_bounds)),
  });
}

void AppendScalarPrefetchTypes(mlir::MLIRContext* context,
                               const FlashAttnConfig& config,
                               mlir::SmallVectorImpl<mlir::Type>& types) {
  if (config.dropout_rate <= 0.0f) {
    return;
  }
  types.push_back(
      GetSmemMemRefType(context, {4}, mlir::IntegerType::get(context, 32)));
}

void SetScalarPrefetchAttr(const FlashAttnConfig& config, mlir::func::FuncOp fn,
                           mlir::OpBuilder& builder) {
  if (config.dropout_rate <= 0.0f) {
    return;
  }
  fn->setAttr("scalar_prefetch", builder.getI64IntegerAttr(1));
}

KVWindowMaps CreateKVWindowMaps(mlir::OpBuilder& builder, mlir::func::FuncOp fn,
                                const FlashAttnConfig& config,
                                const Tiling& tiling) {
  ABSL_DCHECK_GT(config.num_heads, 0);
  ABSL_DCHECK_GT(config.kv_num_heads, 0);
  ABSL_DCHECK_EQ(config.num_heads % config.kv_num_heads, 0);

  // GQA: Mosaic's AffineMap doesn't support FloorDiv, so we use a transform
  // function.
  mlir::ModuleOp module = fn->getParentOfType<mlir::ModuleOp>();
  auto kv_transform_fn =
      module ? module.lookupSymbol<mlir::func::FuncOp>("kv_transform_indices")
             : nullptr;
  if (!kv_transform_fn) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(fn);
    mlir::Location loc = fn.getLoc();
    mlir::Type i32 = builder.getI32Type();
    mlir::SmallVector<mlir::Type> input_types(4, i32);
    AppendScalarPrefetchTypes(builder.getContext(), config, input_types);
    mlir::SmallVector<mlir::Type> output_types(4, i32);
    mlir::FunctionType fn_type =
        builder.getFunctionType(input_types, output_types);
    kv_transform_fn = mlir::func::FuncOp::create(
        builder, loc, "kv_transform_indices", fn_type);
    kv_transform_fn.setPrivate();
    mlir::Block* entry = kv_transform_fn.addEntryBlock();
    mlir::ImplicitLocOpBuilder fn_b =
        mlir::ImplicitLocOpBuilder::atBlockBegin(loc, entry);

    mlir::Value batch = entry->getArgument(0);
    mlir::Value head = entry->getArgument(1);
    mlir::Value q_tile = entry->getArgument(2);
    mlir::Value kv_tile = entry->getArgument(3);
    int64_t head_group_size = config.num_heads / config.kv_num_heads;
    mlir::Value group_size = mlir::arith::ConstantOp::create(
        fn_b, fn_b.getI32IntegerAttr(head_group_size));
    mlir::Value kv_head = mlir::arith::DivSIOp::create(fn_b, head, group_size);
    mlir::Value c0 =
        mlir::arith::ConstantOp::create(fn_b, fn_b.getI32IntegerAttr(0));

    mlir::Value next_kv_tile = kv_tile;
    if (config.is_causal) {
      // If the kv block is skipped, prefetch the next valid kv block, i.e. the
      // 0th one to be used for the next block_q rows.
      mlir::Value cond = CreateBelowOrOnDiagCondition(fn_b, q_tile, kv_tile,
                                                      tiling.qt, tiling.kt);
      next_kv_tile = mlir::arith::SelectOp::create(fn_b, cond, kv_tile, c0);
    }

    mlir::func::ReturnOp::create(
        fn_b, mlir::ValueRange{batch, kv_head, next_kv_tile, c0});
  }

  mlir::DictionaryAttr k_map = CreateSymbolTransformIndicesAttr(
      builder, "kv_transform_indices", {1, 1, tiling.kt, config.qk_head_dim});

  mlir::DictionaryAttr v_map = CreateSymbolTransformIndicesAttr(
      builder, "kv_transform_indices", {1, 1, tiling.kt, config.vo_head_dim});

  return {k_map, v_map};
}

mlir::Value CreateLog2(mlir::ImplicitLocOpBuilder& b, mlir::Value input) {
  static bool is_natively_supported =
      ::torch_tpu::GetLibtpuVersion() >= ::torch_tpu::Version(0, 0, 48);
  if (is_natively_supported) {
    return mlir::math::Log2Op::create(b, input);
  }
  float log2e = std::numbers::log2e_v<float>;
  mlir::Value natural_log = mlir::math::LogOp::create(b, input);
  return ScaleValue(b, natural_log, log2e);
}

mlir::Value BuildDropoutKeepMask(
    mlir::ImplicitLocOpBuilder& b, const FlashAttnConfig& config, int64_t qt,
    int64_t kt, mlir::Value batch_idx, mlir::Value head_idx,
    mlir::Value row_block_idx, mlir::Value col_block_idx, mlir::Value seed_smem,
    int64_t row_offset, int64_t col_offset, int64_t rows, int64_t cols) {
  const int64_t mask_rows = rows > 0 ? rows : qt;
  const int64_t mask_cols = cols > 0 ? cols : kt;
  mlir::Type i32 = b.getI32Type();
  mlir::VectorType vec_i32 = mlir::VectorType::get({mask_rows, mask_cols}, i32);

  // Builds an i32 constant shaped like `type`. `fmix32` below runs on both a
  // scalar and a tile, and `arith` requires operands to match, so the constants
  // have to follow whichever it was handed. `SplatConstant` only accepts shaped
  // types, hence the scalar fallback.
  auto constant_like = [&](mlir::Type type, int64_t value) -> mlir::Value {
    mlir::IntegerAttr attr = b.getI32IntegerAttr(
        absl::bit_cast<int32_t>(static_cast<uint32_t>(value)));
    if (auto shaped_type = llvm::dyn_cast<mlir::ShapedType>(type)) {
      return SplatConstant(b, shaped_type, attr);
    }
    return mlir::arith::ConstantOp::create(b, type, attr);
  };

  // MurmurHash3's fmix32 avalanche; see
  // https://en.wikipedia.org/wiki/MurmurHash. It is a bijection on uint32, so
  // the packed key below cannot collide. The `arith` ops are type-polymorphic,
  // so this runs unchanged on the scalar and vector paths.
  auto fmix32 = [&](mlir::Value h) -> mlir::Value {
    mlir::Type type = h.getType();
    auto k = [&](uint32_t c) { return constant_like(type, c); };
    h = mlir::arith::XOrIOp::create(b, h,
                                    mlir::arith::ShRUIOp::create(b, h, k(16)));
    h = mlir::arith::MulIOp::create(b, h, k(0x85EBCA6Bu));
    h = mlir::arith::XOrIOp::create(b, h,
                                    mlir::arith::ShRUIOp::create(b, h, k(13)));
    h = mlir::arith::MulIOp::create(b, h, k(0xC2B2AE35u));
    h = mlir::arith::XOrIOp::create(b, h,
                                    mlir::arith::ShRUIOp::create(b, h, k(16)));
    return h;
  };

  auto c_idx = [&](int64_t idx) -> mlir::Value {
    return mlir::arith::ConstantOp::create(b, b.getIndexType(),
                                           b.getIndexAttr(idx));
  };

  // 1. Load the Philox {seed_lo, seed_hi, offset_lo, offset_hi} from the
  // scalar_prefetch SMEM buffer.
  mlir::Value seed_lo =
      mlir::memref::LoadOp::create(b, seed_smem, mlir::ValueRange{c_idx(0)});
  mlir::Value seed_hi =
      mlir::memref::LoadOp::create(b, seed_smem, mlir::ValueRange{c_idx(1)});
  mlir::Value offset_lo =
      mlir::memref::LoadOp::create(b, seed_smem, mlir::ValueRange{c_idx(2)});
  mlir::Value offset_hi =
      mlir::memref::LoadOp::create(b, seed_smem, mlir::ValueRange{c_idx(3)});

  // 2. Fold the stream identity into one word. This is scalar work, so it costs
  // once per tile rather than once per element, and we can afford to avalanche
  // after every term.
  mlir::Value stream = seed_lo;
  for (mlir::Value part :
       {seed_hi, offset_lo, offset_hi, batch_idx, head_idx}) {
    stream = fmix32(mlir::arith::XOrIOp::create(b, stream, part));
  }
  mlir::Value stream_vec =
      mlir::vector::BroadcastOp::create(b, vec_i32, stream);

  // 3. Absolute (q, kv) coordinates: grid block index * tile size + static
  // sub-block offset, plus the within-tile iota.
  mlir::Value q_start =
      mlir::arith::MulIOp::create(b, row_block_idx, constant_like(i32, qt));
  if (row_offset != 0) {
    q_start =
        mlir::arith::AddIOp::create(b, q_start, constant_like(i32, row_offset));
  }
  mlir::Value q_off = mlir::vector::BroadcastOp::create(b, vec_i32, q_start);
  mlir::Value kv_start =
      mlir::arith::MulIOp::create(b, col_block_idx, constant_like(i32, kt));
  if (col_offset != 0) {
    kv_start = mlir::arith::AddIOp::create(b, kv_start,
                                           constant_like(i32, col_offset));
  }
  mlir::Value kv_off = mlir::vector::BroadcastOp::create(b, vec_i32, kv_start);
  mlir::Value q_abs =
      mlir::arith::AddIOp::create(b, CreateIotaOp(b, vec_i32, 0), q_off);
  mlir::Value kv_abs =
      mlir::arith::AddIOp::create(b, CreateIotaOp(b, vec_i32, 1), kv_off);

  // 4. Pack the coordinates into a single word, then avalanche. The pack is
  // exactly injective while padded_q * padded_kv <= 2^32, so within a stream no
  // two positions can share a hash.
  mlir::Value key = mlir::arith::AddIOp::create(
      b,
      mlir::arith::MulIOp::create(
          b, q_abs, constant_like(vec_i32, config.padded_kv_sequence_length)),
      kv_abs);
  mlir::Value hash = fmix32(mlir::arith::XOrIOp::create(b, key, stream_vec));

  // 5. Keep iff uint32(hash) >= round(dropout_rate * 2^32). Only a signed
  // compare is available, so flip the sign bit on both sides.
  const uint32_t threshold = static_cast<uint32_t>(std::min<double>(
      std::floor(static_cast<double>(config.dropout_rate) * 4294967296.0),
      4294967295.0));
  mlir::Value hash_signed =
      mlir::arith::XOrIOp::create(b, hash, constant_like(vec_i32, 0x80000000u));
  return mlir::arith::CmpIOp::create(
      b, mlir::arith::CmpIPredicate::sge, hash_signed,
      constant_like(vec_i32, threshold ^ 0x80000000u));
}

}  // namespace torch_tpu
