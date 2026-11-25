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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_TILE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_TILE_H_

#include <cstdint>
#include <optional>

#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/elementwise.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/value_handle.h"
#include "llvm/ADT/ArrayRef.h"

namespace torch_tpu::kiln {

// A vector-typed value: a tile living in vregs.
//
// Arithmetic, bitwise, comparison (yielding an i1 mask tile), min / max /
// select and unary math operators come from `WithElementwiseOps`. Their
// operands broadcast numpy-style (size-1 dims and lower ranks are broadcast,
// e.g. `(qt, kt) - (qt, 1)`), and accept a `Literal` (splatted in the tile's
// dtype) or `Scalar` (broadcast to the tile's dimensions) on either side.
// Element types are never promoted implicitly; use `AsType`.
class Tile final : public WithElementwiseOps<Tile>,
                   public WithOperandOps<Tile, Literal>,
                   public WithOperandOps<Tile, Scalar> {
 public:
  // CHECK-fails unless `handle` is a vector value whose element type is the IR
  // type of `dtype`.
  Tile(internal::ValueHandle handle, DType dtype);

  [[nodiscard]] internal::ValueHandle handle() const { return handle_; }
  [[nodiscard]] DType dtype() const { return dtype_; }

  llvm::ArrayRef<int64_t> dims() const;
  [[nodiscard]] Shape shape() const { return Shape(dims(), dtype()); }
  [[nodiscard]] int64_t rank() const;
  [[nodiscard]] int64_t num_elements() const;

  // Same number of elements, different shape.
  [[nodiscard]] Tile Reshape(llvm::ArrayRef<int64_t> shape) const;
  // Drops leading unit dimensions, but never below rank 2.
  [[nodiscard]] Tile Squeeze() const;
  // Reshapes the tile to the given shape, broadcasting if necessary.
  [[nodiscard]] Tile ToShape(llvm::ArrayRef<int64_t> shape) const;
  [[nodiscard]] Tile BroadcastTo(llvm::ArrayRef<int64_t> shape) const;
  // Resizes the lane (last) dimension to `n` by slicing or repeating. This is
  // the idiom for lane-replicated row statistics stored as (rows, 128).
  [[nodiscard]] Tile Lanes(int64_t n) const;
  // Static window of this tile.
  [[nodiscard]] Tile Slice(llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes) const;
  // Cyclically rotates elements along `axis` by `amount` (`tpu.dynamic_rotate`
  // / Pallas `pltpu.roll`).
  [[nodiscard]] Tile Rotate(IndexArg amount, int32_t axis) const;
  // Gathers elements along `dimensions` using integer `indices`
  // (`tpu.dynamic_gather`).
  [[nodiscard]] Tile Gather(Tile indices,
                            llvm::ArrayRef<int32_t> dimensions) const;
  [[nodiscard]] Tile Gather(Tile indices, int32_t axis) const {
    return Gather(indices, llvm::ArrayRef<int32_t>{axis});
  }
  // Extracts a scalar element at static `indices` (`vector.extract`).
  [[nodiscard]] Scalar Extract(llvm::ArrayRef<int64_t> indices) const;
  // A splat constant with the same shape and dtype as this tile.
  [[nodiscard]] Tile Full(Literal value) const;
  // Broadcasts `scalar` to the dimensions of this tile, preserving
  // `scalar.dtype()`.
  [[nodiscard]] Tile Splat(Scalar scalar) const;

  // Elementwise selection using a boolean `mask` tile when neither branch is a
  // `Tile`. These cannot live on `WithOperandOps` because `mask` provides the
  // shape while the `Scalar` branch(es) provide the result element type (for
  // the same reason, `Select(Tile, Literal, Literal)` is omitted because the
  // result dtype would be ambiguous).
  [[nodiscard]] friend Tile Select(Tile mask, Scalar if_true, Scalar if_false) {
    return Select(mask, mask.Splat(if_true), mask.Splat(if_false));
  }
  [[nodiscard]] friend Tile Select(Tile mask, Scalar if_true,
                                   Literal if_false) {
    return Select(mask, mask.Splat(if_true), if_false);
  }
  [[nodiscard]] friend Tile Select(Tile mask, Literal if_true,
                                   Scalar if_false) {
    return Select(mask, if_true, mask.Splat(if_false));
  }

 private:
  friend class WithOperandOps<Tile, Literal>;
  friend class WithOperandOps<Tile, Scalar>;

  [[nodiscard]] Tile Coerce(Literal value) const { return Full(value); }
  [[nodiscard]] Tile Coerce(Scalar scalar) const { return Splat(scalar); }

  internal::ValueHandle handle_;
  DType dtype_;
};

// Tile constructors.
[[nodiscard]] Tile Zeros(const Shape& shape);
[[nodiscard]] Tile Ones(const Shape& shape);
[[nodiscard]] Tile Full(const Shape& shape, Literal value);
[[nodiscard]] Tile NInf(const Shape& shape);
[[nodiscard]] Tile Iota(const Shape& shape, int axis);
[[nodiscard]] Tile Splat(Scalar scalar, llvm::ArrayRef<int64_t> shape);

[[nodiscard]] inline Tile Rotate(Tile t, IndexArg amount, int32_t axis) {
  return t.Rotate(amount, axis);
}
[[nodiscard]] inline Tile Gather(Tile source, Tile indices,
                                 llvm::ArrayRef<int32_t> dimensions) {
  return source.Gather(indices, dimensions);
}
[[nodiscard]] inline Tile Gather(Tile source, Tile indices, int32_t axis) {
  return source.Gather(indices, axis);
}
[[nodiscard]] inline Scalar Extract(Tile t, llvm::ArrayRef<int64_t> indices) {
  return t.Extract(indices);
}

enum class ReciprocalMode {
  kExact,
  kApprox,
};

// Exact or approximate reciprocal (`tpu.reciprocal`).
[[nodiscard]] Tile Reciprocal(Tile t,
                              ReciprocalMode mode = ReciprocalMode::kExact);

// Reductions along `axis`, keeping the reduced dimension as size 1.
[[nodiscard]] Tile MaxReduce(Tile t, int axis);
[[nodiscard]] Tile SumReduce(Tile t, int axis);

// Concatenates `tiles` along `axis` (`tpu.concatenate` / `jnp.concatenate`).
// All tiles must share the same rank, dtype, and non-`axis` dimensions.
[[nodiscard]] Tile Concat(llvm::ArrayRef<Tile> tiles, int32_t axis);

// 2-D transpose.
[[nodiscard]] Tile Transpose(Tile t);

enum class ContractPrecision {
  kBF16,
  kFP32,
  kBF16x3,
};

struct DotOptions {
  bool transpose_lhs = false;
  bool transpose_rhs = false;
  std::optional<ContractPrecision> precision = std::nullopt;
};

// 2-D matmul on the MXU with f32 accumulation.
[[nodiscard]] Tile Dot(Tile lhs, Tile rhs, const DotOptions& options = {});

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_TILE_H_
