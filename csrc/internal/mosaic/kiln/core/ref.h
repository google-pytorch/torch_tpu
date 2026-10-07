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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_REF_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_REF_H_

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/tile.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "llvm/ADT/ArrayRef.h"

namespace torch_tpu::kiln {

// Sentinel representing full slice along a dimension (`:` in JAX / Pallas).
struct All {};
inline constexpr All all{};

struct SliceDim {
  IndexArg offset;
  int64_t size;
};

inline SliceDim Slice(IndexArg offset, int64_t size) {
  return SliceDim{offset, size};
}

class Ref;

namespace internal {

struct DimDesc {
  IndexArg offset = 0;
  int64_t size = 0;
  bool is_all = false;     // true for full dimension (All)
  bool collapsed = false;  // true for scalar index (collapses dimension)

  static DimDesc From(All) {
    return DimDesc{.offset = 0, .size = 0, .is_all = true, .collapsed = false};
  }
  static DimDesc From(const SliceDim& s) {
    return DimDesc{.offset = s.offset,
                   .size = s.size,
                   .is_all = false,
                   .collapsed = false};
  }
  static DimDesc From(IndexArg s) {
    return DimDesc{.offset = s, .size = 1, .is_all = false, .collapsed = true};
  }
};

Ref SliceRefImpl(Ref base, llvm::ArrayRef<DimDesc> dims);

}  // namespace internal

// A memref kernel argument (VMEM block, SMEM scalar-prefetch buffer or
// scratch). Slicing emits `tpu::MemRefSliceOp` / `tpu::MemRefSqueezeOp` and
// returns another `Ref`, so sliced and non-sliced references share the exact
// same interface.
class Ref : public ValueBase {
 public:
  Ref(const Ref&) = default;
  Ref& operator=(const Ref&) = delete;

  Ref& operator=(Tile rhs);
  Ref& operator=(Scalar rhs);

  explicit operator Scalar() const;
  explicit operator Tile() const;

  // Stores `value` into this VMEM ref, optionally masked by an i1 `mask` tile
  // (`tpu.vector_store` / Pallas `pltpu.store`).
  void Store(Tile value, std::optional<Tile> mask = std::nullopt);

  llvm::ArrayRef<int64_t> shape() const;
  DType dtype() const;
  int64_t rank() const;

  // Slices a view of the ref:
  //   ref(0, 0)                        (collapses leading dims)
  //   ref(0, Slice(c, kChunk), all)
  // Trailing omitted dimensions defaults to `all`.
  template <typename... Dims>
  Ref operator()(Dims&&... dims) const {
    std::array<internal::DimDesc, sizeof...(Dims)> descs = {
        internal::DimDesc::From(std::forward<Dims>(dims))...};
    return internal::SliceRefImpl(*this, descs);
  }

 private:
  friend struct internal::ValueAccess;
  explicit Ref(const void* impl);
};

// Stores `value` into `dst` (a VMEM ref), optionally masked by an i1 `mask`
// tile (`tpu.vector_store` / Pallas `pltpu.store`).
inline void Store(Ref dst, Tile value,
                  std::optional<Tile> mask = std::nullopt) {
  dst.Store(value, mask);
}

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_REF_H_
