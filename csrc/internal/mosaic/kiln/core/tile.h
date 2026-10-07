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
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "llvm/ADT/ArrayRef.h"

namespace torch_tpu::kiln {

// A vector-typed value: a tile living in vregs.
//
// Arithmetic, bitwise, comparison (yielding an i1 mask tile), min / max /
// select and unary math operators come from `WithElementwiseOps`. Their
// operands broadcast numpy-style (size-1 dims and lower ranks are broadcast,
// e.g. `(qt, kt) - (qt, 1)`), and accept a `Literal` or `Scalar` on either
// side which is splatted in the tile's dtype. Element types are never promoted
// implicitly; use `AsType`.
class Tile : public ValueBase,
             public WithElementwiseOps<Tile, Tile>,
             public WithLiteralOps<Tile, Tile> {
 public:
  // Construct a 0-D tile from a scalar.
  /* implicit */ Tile(Scalar scalar);  // NOLINT(google-explicit-constructor)

  Tile MakeConstant(Literal value) const { return Full(value); }

  llvm::ArrayRef<int64_t> shape() const;
  DType dtype() const;
  int64_t rank() const;
  int64_t num_elements() const;

  // Same number of elements, different shape.
  Tile Reshape(llvm::ArrayRef<int64_t> shape) const;
  // Drops leading unit dimensions, but never below rank 2.
  Tile Squeeze() const;
  // Reshapes the tile to the given shape, broadcasting if necessary.
  Tile ToShape(llvm::ArrayRef<int64_t> shape) const;
  Tile BroadcastTo(llvm::ArrayRef<int64_t> shape) const;
  // Resizes the lane (last) dimension to `n` by slicing or repeating. This is
  // the idiom for lane-replicated row statistics stored as (rows, 128).
  Tile Lanes(int64_t n) const;
  // Static window of this tile.
  Tile Slice(llvm::ArrayRef<int64_t> offsets,
             llvm::ArrayRef<int64_t> sizes) const;
  // Cyclically rotates elements along `axis` by `amount` (`tpu.dynamic_rotate`
  // / Pallas `pltpu.roll`).
  Tile Rotate(IndexArg amount, int32_t axis) const;
  // Gathers elements along `dimensions` using integer `indices`
  // (`tpu.dynamic_gather`).
  Tile Gather(Tile indices, llvm::ArrayRef<int32_t> dimensions) const;
  Tile Gather(Tile indices, int32_t axis) const {
    return Gather(indices, llvm::ArrayRef<int32_t>{axis});
  }
  // Extracts a scalar element at static `indices` (`vector.extract`).
  Scalar Extract(llvm::ArrayRef<int64_t> indices) const;
  // A splat constant with the same shape and dtype as this tile.
  Tile Full(Literal value) const;

 private:
  friend struct internal::ValueAccess;
  explicit Tile(const void* impl);
};

// Tile constructors.
Tile Zeros(llvm::ArrayRef<int64_t> shape, DType dtype);
Tile Full(llvm::ArrayRef<int64_t> shape, DType dtype, Literal value);
Tile NInf(llvm::ArrayRef<int64_t> shape, DType dtype);
Tile Iota(llvm::ArrayRef<int64_t> shape, DType dtype, int axis);
Tile Splat(Scalar scalar, llvm::ArrayRef<int64_t> shape);

inline Tile Rotate(Tile t, IndexArg amount, int32_t axis) {
  return t.Rotate(amount, axis);
}
inline Tile Gather(Tile source, Tile indices,
                   llvm::ArrayRef<int32_t> dimensions) {
  return source.Gather(indices, dimensions);
}
inline Tile Gather(Tile source, Tile indices, int32_t axis) {
  return source.Gather(indices, axis);
}
inline Scalar Extract(Tile t, llvm::ArrayRef<int64_t> indices) {
  return t.Extract(indices);
}

// Approximate or exact reciprocal (`tpu.reciprocal`).
Tile Reciprocal(Tile t, bool approx = false);

// Reductions along `axis`, keeping the reduced dimension as size 1.
Tile MaxReduce(Tile t, int axis);
Tile SumReduce(Tile t, int axis);

// 2-D transpose.
Tile Transpose(Tile t);

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
Tile Dot(Tile lhs, Tile rhs, const DotOptions& options = {});

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_TILE_H_
