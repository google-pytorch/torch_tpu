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

#ifndef TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_
#define TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_

#include <cstdint>
#include <string_view>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::torch_tpu {

// Load a 2D tile from the given argument.
TypedValue<VectorType> LoadTile(ImplicitLocOpBuilder& b, Value arg);
// Store a 2D tile to the given argument.
// If the value is not the same type as the argument, it will be converted.
void StoreTile(ImplicitLocOpBuilder& b, Value value, Value arg);

// Initialize a 2D tile with zeros.
void ZeroTile(ImplicitLocOpBuilder& b, Value arg);

// Initialize a 2D tile with -inf.
void NInfTile(ImplicitLocOpBuilder& b, Value arg);

// Get the causal bias for the given block indices.
Value GetCausalBias(ImplicitLocOpBuilder& b, Value row_block_idx,
                    Value col_block_idx, int64_t qt, int64_t kt);

// Reduce the 2D input to a 1D across the lane dimension and then broadcast the
// results lane dimension to the given size.
Value ReduceBroadcastLane(ImplicitLocOpBuilder& b, Value input,
                          vector::CombiningKind kind,
                          int64_t broadcast_lane_size = 128);

// Clamp the input logits to a minimum value.
Value ClampLogits(ImplicitLocOpBuilder& b, Value input);

// Create an scf.IfOp where the then block will be executed when
// row_idx >= col_idx.
scf::IfOp CreateCausalIfOp(ImplicitLocOpBuilder& b, Value row_idx,
                           Value col_idx, int64_t row_block_size,
                           int64_t col_block_size);

// Scale the input vector by the given scale factor.
Value ScaleValue(ImplicitLocOpBuilder& b, Value input, float scale);

// Cast the input value element type to the given target type.
// Assumes that the input is a vector type.
// If the input is already the correct type, the input is returned unchanged.
Value ConvertElementType(ImplicitLocOpBuilder& b, Type target_element_type,
                         Value input);

// Normalizes the lane dimension of the input vector.
// This is intended to be used with the statistic tiles (e.g. lse, di, etc.)
// It will slice the lane dimension if the target lane size is smaller than the
// input lane size, and will repeat the lane dimension if the target lane
// size is larger than the input lane size.
// **NOTE** This will fail if the target lane size is larger and not a
// multiple of the input lane size.
Value NormalizeLaneDim(ImplicitLocOpBuilder& builder, Value input,
                       int64_t target_lane_size);

// Helper to create a stablehlo::CustomCallOp with mosaic kernel.
stablehlo::CustomCallOp CreateCustomCallOp(OpBuilder& builder, Location loc,
                                           std::string_view kernel_mlir,
                                           ValueRange inputs,
                                           TypeRange output_types);

}  // namespace mlir::torch_tpu

#endif  // TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_
