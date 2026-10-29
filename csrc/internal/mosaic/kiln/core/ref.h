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
#include <type_traits>
#include <utility>
#include <variant>

#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "csrc/internal/mosaic/kiln/core/tile.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "llvm/ADT/ArrayRef.h"

namespace torch_tpu::kiln {

// Sentinel representing full slice along a dimension (`:` in JAX / Pallas).
// Note: on a dynamic dimension in a non-identity slice, `all` emits a bare
// `memref.dim` without tile-alignment metadata, so it is only valid for
// untiled leading dynamic dimensions. Along trailing tiled dynamic dimensions,
// pass `Slice(0, MultipleOf(ref.Dim(d), tile_dim))` instead.
struct All {};
inline constexpr All all{};

struct SliceDim {
  IndexArg offset;
  IndexArg size;
};

// Specifies a slice of `[offset, offset + size)` along a single dimension.
// `size` must be a positive integer or a runtime integer `Scalar` (do not pass
// `kDynamic`; use `all` for a full-dimension slice). Along trailing tiled
// dimensions, any dynamic `offset` or `size` must be provably divisible by the
// tile size of that dimension (e.g., via `MultipleOf` or a stepped `For`
// induction variable).
inline SliceDim Slice(IndexArg offset, IndexArg size) {
  return SliceDim{offset, size};
}

// One dimension of `Ref::At`: an index (which collapses the dimension), a
// `Slice(offset, size)`, or `all`.
using Indexer = std::variant<IndexArg, SliceDim, All>;

namespace internal {

struct DimDesc {
  enum class Kind {
    kAll,        // Full dimension (All)
    kSlice,      // Range slice (dimension is kept even if the size is 1)
    kCollapsed,  // Dimension is squeezed
  };

  IndexArg offset = 0;
  IndexArg size = 0;
  Kind kind = Kind::kAll;

  static DimDesc From(All) {
    return DimDesc{.offset = 0, .size = 0, .kind = Kind::kAll};
  }
  static DimDesc From(const SliceDim& s) {
    return DimDesc{.offset = s.offset, .size = s.size, .kind = Kind::kSlice};
  }
  static DimDesc From(IndexArg s) {
    return DimDesc{.offset = s, .size = 1, .kind = Kind::kCollapsed};
  }
};

const void* SliceMemRefImpl(const void* base_impl,
                            llvm::ArrayRef<DimDesc> dims);

// Common implementation of `Ref` and `TypedRef<kSpace>`: a handle to a memref
// value with shape queries and slicing helpers. Copy assignment is protected so
// that a typed ref can never be rebound through a base reference to a ref in
// another memory space.
class RefBase : public ValueBase {
 public:
  llvm::ArrayRef<int64_t> dims() const;
  Shape shape() const { return Shape(dims(), dtype()); }
  MemorySpace memory_space() const;
  int64_t rank() const;
  bool HasStaticShape() const;
  bool IsDynamicDim(int64_t axis) const;

  // Queries the size of dimension `axis` as an i32 Scalar (`memref.dim` for
  // dynamic dimensions, or a constant `I32(dims()[axis])` for static
  // dimensions).
  Scalar Dim(int64_t axis) const;

 protected:
  RefBase(const void* impl, DType dtype);
  // CHECK-fails unless the memref is in `expected_space`.
  RefBase(const void* impl, DType dtype, MemorySpace expected_space);
  RefBase(const RefBase&) = default;
  RefBase& operator=(const RefBase&) = default;
  ~RefBase() = default;

  // Returns the opaque memref value of `ref(dims...)` (see
  // `TypedRef::operator()`).
  template <typename... Dims>
  const void* SliceImpl(Dims&&... dims) const {
    const std::array<DimDesc, sizeof...(Dims)> descs = {
        DimDesc::From(std::forward<Dims>(dims))...};
    return SliceMemRefImpl(impl(), descs);
  }

  // Returns the opaque memref value of `ref.At(indexers)`.
  const void* AtImpl(llvm::ArrayRef<Indexer> indexers) const;
};

}  // namespace internal

class Ref;

// A memref kernel argument or allocation whose memory space is part of its
// type: a VMEM block (`VmemRef`), an SMEM buffer (`SmemRef`), or an HBM buffer
// (`HbmRef`). Slicing emits `tpu::MemRefSliceOp` / `tpu::MemRefSqueezeOp` and
// returns a ref in the same memory space. Memory is only accessed through the
// explicit free functions `Load` and `Store`, whose overloads depend on the
// memory space, and through DMA.
template <MemorySpace kSpace>
class TypedRef final : public internal::RefBase {
 public:
  static constexpr MemorySpace kMemorySpace = kSpace;

  TypedRef(const TypedRef&) = default;
  TypedRef& operator=(const TypedRef&) = default;

  // Slices a view of the ref, in the same memory space:
  //   ref(0, 0)                        (collapses leading dims)
  //   ref(0, Slice(c, kChunk), all)
  // Trailing omitted dimensions default to `all`. Along trailing tiled dynamic
  // dimensions, non-identity slices must use `Slice(offset, size)` with
  // tile-aligned `Scalar` bounds rather than `all`.
  template <typename... Dims>
  TypedRef operator()(Dims&&... dims) const {
    return TypedRef(SliceImpl(std::forward<Dims>(dims)...), dtype());
  }

  // Same as `operator()`, with the dimensions given as a list, e.g. when their
  // number is only known when the kernel is built (Pallas `ref.at[...]`).
  TypedRef At(llvm::ArrayRef<Indexer> indexers) const {
    return TypedRef(AtImpl(indexers), dtype());
  }

 private:
  friend struct internal::ValueAccess;
  friend class Ref;

  TypedRef(const void* impl, DType dtype) : RefBase(impl, dtype, kSpace) {}
};

using VmemRef = TypedRef<MemorySpace::kVmem>;
using SmemRef = TypedRef<MemorySpace::kSmem>;
using HbmRef = TypedRef<MemorySpace::kHbm>;

// A memref whose memory space is only known when the kernel is built, for code
// that is generic over memory spaces (e.g. DMA). Any `TypedRef` converts to a
// `Ref` implicitly; converting back requires the checked `As<TypedRefT>()`.
class Ref final : public internal::RefBase {
 public:
  Ref(const Ref&) = default;
  Ref& operator=(const Ref&) = default;

  // Forgets the static memory space of `ref`.
  template <MemorySpace kSpace>
  Ref(TypedRef<kSpace> ref)  // NOLINT(google-explicit-constructor)
      : RefBase(ref) {}

  // Returns this ref as `TypedRefT` (`VmemRef`, `SmemRef`, or `HbmRef`), e.g.
  // `ref.As<VmemRef>()`. CHECK-fails if this ref is in another memory space.
  template <typename TypedRefT>
  TypedRefT As() const {
    static_assert(
        std::is_same_v<TypedRefT, TypedRef<TypedRefT::kMemorySpace>>,
        "Ref::As<T>() requires T to be a TypedRef (VmemRef, SmemRef, HbmRef)");
    return TypedRefT(impl(), dtype());
  }

  // Same as `TypedRef::operator()` and `TypedRef::At`, returning a `Ref`.
  template <typename... Dims>
  Ref operator()(Dims&&... dims) const {
    return Ref(SliceImpl(std::forward<Dims>(dims)...), dtype());
  }
  Ref At(llvm::ArrayRef<Indexer> indexers) const {
    return Ref(AtImpl(indexers), dtype());
  }

 private:
  friend struct internal::ValueAccess;

  Ref(const void* impl, DType dtype) : RefBase(impl, dtype) {}
};

// Loads the whole of `src`, a VMEM ref with a static shape, as a tile. Slice
// first to load a window: `Load(ref(0, Slice(c, kChunk)))`.
Tile Load(VmemRef src);

// Loads `src`, a 0-D SMEM ref, as a scalar (`memref.load`). Index first to load
// an element: `Load(ref(i, j))`.
Scalar Load(SmemRef src);

// TODO(willfroom): Remove the untyped `Load`, `LoadScalar`, and `Store`
// overloads below once all callers use typed refs.

// Same as `Load(VmemRef)`, checking that `src` is in VMEM when the kernel is
// built.
Tile Load(Ref src);

// Same as `Load(SmemRef)`, checking that `src` is in SMEM when the kernel is
// built.
Scalar LoadScalar(Ref src);

// TODO(willfroom): Add the "add" option.
struct StoreOptions {
  // If set, an i1 tile; only elements where `mask` is true are written.
  // Reshaped or broadcast to the dimensions of the destination.
  std::optional<Tile> mask = std::nullopt;
};

// Stores `value` into `dst`, a VMEM ref with a static shape. `value` is
// converted to the dtype of `dst` and reshaped or broadcast to its dimensions.
void Store(VmemRef dst, Tile value, const StoreOptions& options = {});

// Stores `value` into `dst`, a 0-D SMEM ref (`memref.store`). `value` is
// converted to the dtype of `dst`.
void Store(SmemRef dst, Scalar value);

// Same as the typed `Store` overloads, checking the memory space of `dst` when
// the kernel is built.
void Store(Ref dst, Tile value, const StoreOptions& options = {});
void Store(Ref dst, Scalar value);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_REF_H_
