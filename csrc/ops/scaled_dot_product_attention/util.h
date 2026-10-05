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

#ifndef TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_
#define TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_

#include <cstdint>
#include <optional>
#include <string>

#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace torch_tpu {

struct KVWindowMaps {
  mlir::DictionaryAttr k_map;
  mlir::DictionaryAttr v_map;
};

mlir::DictionaryAttr CreateSymbolTransformIndicesAttr(
    mlir::Builder& builder, llvm::StringRef function_name,
    llvm::ArrayRef<int64_t> window_bounds);

// Creates the KV window maps (k_map and v_map) for MHA, GQA, or MQA.
KVWindowMaps CreateKVWindowMaps(mlir::OpBuilder& builder, mlir::func::FuncOp fn,
                                const FlashAttnConfig& config,
                                const Tiling& tiling);

// Load a 2D tile from the given argument.
mlir::TypedValue<mlir::VectorType> LoadTile(mlir::ImplicitLocOpBuilder& b,
                                            mlir::Value arg);
// Load the [num_rows, num_cols] sub-tile of the given argument starting at
// (row_offset, col_offset) of its trailing 2D tile.
mlir::TypedValue<mlir::VectorType> LoadTileSlice(
    mlir::ImplicitLocOpBuilder& b, mlir::Value arg, int64_t row_offset,
    int64_t num_rows, int64_t col_offset, int64_t num_cols);
// Store a 2D tile to the given argument.
// If the value is not the same type as the argument, it will be converted.
void StoreTile(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
               mlir::Value arg);
// Store a 2D value at (row_offset, col_offset) of the trailing 2D tile of the
// given argument.
void StoreTileSlice(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
                    mlir::Value arg, int64_t row_offset, int64_t col_offset);

// Initialize a 2D tile with zeros.
void ZeroTile(mlir::ImplicitLocOpBuilder& b, mlir::Value arg);

// Initialize a 2D tile with -inf.
void NInfTile(mlir::ImplicitLocOpBuilder& b, mlir::Value arg);

// Whether any sequence boundary masking is required for the given
// configuration.
bool NeedsPadMask(int64_t qt, int64_t kt, int64_t q_seq_len,
                  int64_t kv_seq_len);

// Apply structured masking (causal and/or sequence boundary masking) to `sij`.
// `sij` is the sub-block of the (row_block_idx, col_block_idx) grid tile that
// starts at the static (row_offset, col_offset) within the tile; the tile
// itself is [qt, kt]. Callers should skip calling this on sub-blocks known not
// to intersect a mask boundary (see CreateIntersectsDiagCondition).
mlir::Value ApplyStructuredMask(mlir::ImplicitLocOpBuilder& b, mlir::Value sij,
                                mlir::Value row_block_idx,
                                mlir::Value col_block_idx, int64_t qt,
                                int64_t kt, int64_t q_seq_len,
                                int64_t kv_seq_len, bool is_causal,
                                int64_t row_offset = 0, int64_t col_offset = 0);

// Reduce the 2D input to a 1D across the lane dimension and then broadcast the
// results lane dimension to the given size.
mlir::Value ReduceBroadcastLane(mlir::ImplicitLocOpBuilder& b,
                                mlir::Value input,
                                mlir::vector::CombiningKind kind,
                                int64_t broadcast_lane_size = 128);

// Clamp the given value to the minimum finite value.
mlir::Value ClampMinFinite(mlir::ImplicitLocOpBuilder& b, mlir::Value input);

// Creates a constant vector with all elements set to `value`.
mlir::Value SplatConstant(mlir::ImplicitLocOpBuilder& b, mlir::Type type,
                          float value);
mlir::Value SplatConstant(mlir::ImplicitLocOpBuilder& b, mlir::ShapedType type,
                          mlir::Attribute value_attr);

// Creates a constant vector with all elements set to -infinity.
mlir::Value SplatNegInf(mlir::ImplicitLocOpBuilder& b, mlir::Type type);

// Returns an i1 condition indicating whether the tile at (row_idx, col_idx)
// is below or on the diagonal (i.e. intersects the causal mask).
mlir::Value CreateBelowOrOnDiagCondition(mlir::ImplicitLocOpBuilder& b,
                                         mlir::Value row_idx,
                                         mlir::Value col_idx,
                                         int64_t row_block_size,
                                         int64_t col_block_size);

// Returns an i1 condition indicating whether the tile at (row_idx, col_idx)
// contains at least one element strictly above the diagonal, i.e. whether a
// causal mask needs to be applied to it. Tiles strictly below the diagonal
// need no masking.
mlir::Value CreateIntersectsDiagCondition(mlir::ImplicitLocOpBuilder& b,
                                          mlir::Value row_idx,
                                          mlir::Value col_idx,
                                          int64_t row_block_size,
                                          int64_t col_block_size);

// Create an scf.IfOp where the then block will be executed when
// row_idx >= col_idx.
mlir::scf::IfOp CreateCausalIfOp(mlir::ImplicitLocOpBuilder& b,
                                 mlir::Value row_idx, mlir::Value col_idx,
                                 int64_t row_block_size,
                                 int64_t col_block_size);

// Scale the input vector by the given scale factor.
mlir::Value ScaleValue(mlir::ImplicitLocOpBuilder& b, mlir::Value input,
                       float scale);

// Cast the input value element type to the given target type.
// Assumes that the input is a vector type.
// If the input is already the correct type, the input is returned unchanged.
mlir::Value ConvertElementType(mlir::ImplicitLocOpBuilder& b,
                               mlir::Type target_element_type,
                               mlir::Value input);

// Normalizes the lane dimension of the input vector.
// This is intended to be used with the statistic tiles (e.g. lse, di, etc.)
// It will slice the lane dimension if the target lane size is smaller than the
// input lane size, and will repeat the lane dimension if the target lane
// size is larger than the input lane size.
// **NOTE** This will fail if the target lane size is larger and not a
// multiple of the input lane size.
mlir::Value NormalizeLaneDim(mlir::ImplicitLocOpBuilder& builder,
                             mlir::Value input, int64_t target_lane_size);

// Create a log2 operation.
// For older versions of libtpu that do not support the log2 op we will use a
// scaled ln operation.
mlir::Value CreateLog2(mlir::ImplicitLocOpBuilder& b, mlir::Value input);

// Dropout needs a single Mosaic `scalar_prefetch` operand: a 4-word SMEM buffer
// holding the Philox {seed_lo, seed_hi, offset_lo, offset_hi}.
//
// The two functions below must stay in sync. Mosaic binds scalar-prefetch
// operands immediately after the grid iteration indices, and requires the
// `scalar_prefetch` attribute to match the number of operands appended to the
// kernel signature *and* to every symbol-based `transform_indices` function,
// which must take exactly `iteration_bounds.size() + scalar_prefetch` args.

// Appends the `scalar_prefetch` operand types for `config` to `types`.
void AppendScalarPrefetchTypes(mlir::MLIRContext* context,
                               const FlashAttnConfig& config,
                               mlir::SmallVectorImpl<mlir::Type>& types);

// Sets the `scalar_prefetch` attribute on `fn`, if `config` needs one.
void SetScalarPrefetchAttr(const FlashAttnConfig& config, mlir::func::FuncOp fn,
                           mlir::OpBuilder& builder);

// Builds a stateless, tile-size-invariant dropout mask of type
// vector<rows x cols x i1> (true = keep, false = drop) for the sub-block of
// the (row_block_idx, col_block_idx) grid tile that starts at the static
// (row_offset, col_offset) within the [qt, kt] tile. When `rows` or `cols` are
// <= 0 they default to `qt` and `kt` respectively.
//
// The mask is a pure function of (batch_idx, head_idx, absolute q, absolute
// kv) and the Philox {seed, offset} in `seed_smem` (memref<4xi32, smem>), so
// the backward kernel regenerates it rather than storing it. It is keyed on
// absolute rather than within-tile coordinates so that the forward and
// backward kernels agree despite their different tilings.
//
// `head_idx` must be the *query* head index.
mlir::Value BuildDropoutKeepMask(
    mlir::ImplicitLocOpBuilder& b, const FlashAttnConfig& config, int64_t qt,
    int64_t kt, mlir::Value batch_idx, mlir::Value head_idx,
    mlir::Value row_block_idx, mlir::Value col_block_idx, mlir::Value seed_smem,
    int64_t row_offset = 0, int64_t col_offset = 0, int64_t rows = 0,
    int64_t cols = 0);
// Get the string representation of an operation.
std::string GetOpString(mlir::Operation* op);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_
