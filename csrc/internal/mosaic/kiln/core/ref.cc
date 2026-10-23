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

#include "csrc/internal/mosaic/kiln/core/ref.h"

#include <cstdint>
#include <optional>
#include <variant>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/mlir_utils.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "csrc/internal/mosaic/kiln/core/tile.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace torch_tpu::kiln {
namespace {

namespace arith = mlir::arith;
namespace memref = mlir::memref;
namespace tpu = mlir::tpu;
namespace vector = mlir::vector;

llvm::SmallVector<mlir::Value> ZeroIndices(mlir::ImplicitLocOpBuilder& b,
                                           int64_t rank) {
  mlir::Value zero = arith::ConstantOp::create(b, b.getIndexAttr(0));
  return llvm::SmallVector<mlir::Value>(rank, zero);
}

bool HasMemorySpace(const Ref& r, tpu::MemorySpace expected_space) {
  auto space = llvm::dyn_cast_or_null<tpu::MemorySpaceAttr>(
      internal::MemRefTypeOf(r).getMemorySpace());
  return space && space.getValue() == expected_space;
}

bool IsSmemRef(const Ref& r) {
  return HasMemorySpace(r, tpu::MemorySpace::kSmem);
}

bool IsVmemRef(const Ref& r) {
  return HasMemorySpace(r, tpu::MemorySpace::kVmem);
}

int64_t ResolveSliceDimSize(const internal::DimDesc& spec, int64_t source_dim) {
  if (spec.is_all) {
    return source_dim;
  }
  if (spec.size.IsInt()) {
    ABSL_CHECK_NE(spec.size.AsInt(), kDynamic)  // CRASH_OK
        << "Slice size cannot be kDynamic; pass a runtime Scalar size or use "
           "all";
  }
  internal::ValidateI32IndexArg(spec.size, /*require_positive=*/true,
                                "Slice size");
  if (!spec.size.IsInt()) {
    return kDynamic;
  }
  const int64_t sz = spec.size.AsInt();
  if (source_dim != kDynamic) {
    ABSL_CHECK_LE(sz, source_dim)  // CRASH_OK
        << "Slice size must be less than or equal to the dimension size";
  }
  return sz;
}

void ValidateSliceDimOffset(const internal::DimDesc& spec, int64_t source_dim,
                            int64_t slice_size) {
  internal::ValidateI32IndexArg(spec.offset, /*require_positive=*/false,
                                "Slice offset");
  if (!spec.offset.IsInt()) {
    return;
  }
  if (source_dim != kDynamic && slice_size != kDynamic) {
    ABSL_CHECK_LE(spec.offset.AsInt(), source_dim - slice_size)  // CRASH_OK
        << "Slice offset + size must not exceed dimension size";
  }
}

bool IsIdentitySliceDim(const internal::DimDesc& spec, int64_t source_dim,
                        int64_t slice_size) {
  if (!spec.offset.IsInt() || spec.offset.AsInt() != 0) {
    return false;
  }
  return spec.is_all || (slice_size != kDynamic && slice_size == source_dim);
}

mlir::Value EmitMemRefSlice(mlir::ImplicitLocOpBuilder& b, mlir::Value source,
                            llvm::ArrayRef<internal::DimDesc> dims,
                            llvm::ArrayRef<int64_t> sliced_shape) {
  auto base_type = llvm::cast<mlir::MemRefType>(source.getType());
  const int64_t rank = sliced_shape.size();
  llvm::SmallVector<mlir::Value, 4> base_indices;
  base_indices.reserve(rank);
  llvm::SmallVector<mlir::Value> dynamic_sizes;
  for (int64_t d = 0; d < rank; ++d) {
    const internal::DimDesc spec =
        d < dims.size() ? dims[d] : internal::DimDesc::From(all);
    base_indices.push_back(internal::unwrap(spec.offset.ToScalar().ToI32()));
    if (sliced_shape[d] != kDynamic) {
      continue;
    }
    if (!spec.is_all) {
      dynamic_sizes.push_back(internal::unwrap(spec.size.ToScalar().ToI32()));
    } else {
      mlir::Value dim_idx = memref::DimOp::create(b, source, d);
      dynamic_sizes.push_back(
          arith::IndexCastOp::create(b, b.getI32Type(), dim_idx));
    }
  }
  auto sliced_type =
      mlir::MemRefType::get(sliced_shape, base_type.getElementType(), nullptr,
                            base_type.getMemorySpace());
  return tpu::MemRefSliceOp::create(b, sliced_type, source, base_indices,
                                    dynamic_sizes);
}

}  // namespace

Ref::Ref(const void* impl, DType dtype) : ValueBase(impl, dtype) {
  ABSL_CHECK(  // CRASH_OK
      llvm::isa<mlir::MemRefType>(internal::unwrap(*this).getType()))
      << "Ref must be a memref value";
}

Ref& Ref::operator=(Tile rhs) {
  Store(rhs, std::nullopt);
  return *this;
}

Ref& Ref::operator=(Scalar rhs) {
  ABSL_CHECK_EQ(rank(), 0)  // CRASH_OK
      << "Ref::operator=(Scalar) expects a 0-D ref";
  ABSL_CHECK(IsSmemRef(*this))  // CRASH_OK
      << "Ref::operator=(Scalar) expects an SMEM ref";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  memref::StoreOp::create(b, internal::unwrap(rhs.AsType(dtype())),
                          internal::unwrap(*this), mlir::ValueRange{});
  return *this;
}

Ref::operator Scalar() const {
  ABSL_CHECK_EQ(rank(), 0)  // CRASH_OK
      << "Ref::operator Scalar() expects a 0-D ref";
  ABSL_CHECK(IsSmemRef(*this))  // CRASH_OK
      << "Ref::operator Scalar() expects an SMEM ref";
  return internal::wrap<Scalar>(
      memref::LoadOp::create(internal::CurrentBuilder(),
                             internal::unwrap(*this), mlir::ValueRange{}),
      dtype());
}

Ref::operator Tile() const {
  ABSL_CHECK(HasStaticShape())  // CRASH_OK
      << "Tile load requires a Ref with a static shape";
  ABSL_CHECK(IsVmemRef(*this))  // CRASH_OK
      << "Tile load requires a VMEM ref (use AsyncCopy for HBM)";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(
      vector::LoadOp::create(b, internal::ToMlirVectorType(shape()),
                             internal::unwrap(*this), ZeroIndices(b, rank())),
      dtype());
}

void Ref::Store(Tile value, std::optional<Tile> mask) {
  ABSL_CHECK(HasStaticShape())  // CRASH_OK
      << "Tile store requires a Ref with a static shape";
  ABSL_CHECK(IsVmemRef(*this))  // CRASH_OK
      << "Tile store requires a VMEM ref (use AsyncCopy for HBM)";
  Tile converted = value.AsType(dtype());
  Tile shaped = converted.ToShape(dims());
  mlir::Value mask_value;
  if (mask.has_value()) {
    ABSL_CHECK(mask->dtype().IsBool())  // CRASH_OK
        << "Masked tile store requires an i1 mask tile";
    Tile shaped_mask = mask->ToShape(dims());
    mask_value = internal::unwrap(shaped_mask);
  }

  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  tpu::VectorStoreOp::create(b, internal::unwrap(shaped),
                             internal::unwrap(*this), ZeroIndices(b, rank()),
                             mask_value, /*add=*/false);
}

llvm::ArrayRef<int64_t> Ref::dims() const {
  return internal::MemRefTypeOf(*this).getShape();
}

MemorySpace Ref::memory_space() const {
  auto space = llvm::dyn_cast_or_null<tpu::MemorySpaceAttr>(
      internal::MemRefTypeOf(*this).getMemorySpace());
  ABSL_CHECK(space != nullptr)  // CRASH_OK
      << "Ref is missing a TPU memory space attribute";
  switch (space.getValue()) {
    case tpu::MemorySpace::kVmem:
      return MemorySpace::kVmem;
    case tpu::MemorySpace::kSmem:
      return MemorySpace::kSmem;
    case tpu::MemorySpace::kHbm:
      return MemorySpace::kHbm;
    default:
      ABSL_CHECK(false)  // CRASH_OK
          << "Unsupported TPU memory space in Kiln Ref";
      return MemorySpace::kVmem;
  }
}

int64_t Ref::rank() const { return internal::MemRefTypeOf(*this).getRank(); }

bool Ref::HasStaticShape() const {
  return internal::MemRefTypeOf(*this).hasStaticShape();
}

bool Ref::IsDynamicDim(int64_t axis) const {
  ABSL_CHECK_GE(axis, 0) << "Dimension axis must be non-negative";  // CRASH_OK
  ABSL_CHECK_LT(axis, rank())                                       // CRASH_OK
      << "Dimension axis " << axis << " out of bounds for rank " << rank();
  return internal::MemRefTypeOf(*this).isDynamicDim(axis);
}

Scalar Ref::Dim(int64_t axis) const {
  if (!IsDynamicDim(axis)) {
    return I32(dims()[axis]);
  }
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::Value dim_idx = memref::DimOp::create(b, internal::unwrap(*this), axis);
  return internal::wrap<Scalar>(
      arith::IndexCastOp::create(b, b.getI32Type(), dim_idx), i32);
}

Ref Ref::At(llvm::ArrayRef<Indexer> indexers) const {
  llvm::SmallVector<internal::DimDesc, 4> dims;
  for (const Indexer& indexer : indexers) {
    dims.push_back(std::visit(
        [](const auto& v) { return internal::DimDesc::From(v); }, indexer));
  }
  return internal::SliceRefImpl(*this, dims);
}

namespace internal {

const void* SliceMemRefImpl(const void* base_impl,
                            llvm::ArrayRef<DimDesc> dims) {
  mlir::ImplicitLocOpBuilder& b = CurrentBuilder();
  mlir::Value current = mlir::Value::getFromOpaquePointer(base_impl);
  auto base_type = llvm::cast<mlir::MemRefType>(current.getType());
  llvm::ArrayRef<int64_t> source_shape = base_type.getShape();
  const int64_t rank = source_shape.size();
  ABSL_CHECK_LE(static_cast<int64_t>(dims.size()), rank)  // CRASH_OK
      << "Too many slice indices for rank " << rank;

  llvm::SmallVector<int64_t> sliced_shape;
  llvm::SmallVector<int64_t> squeezed_shape;
  bool is_identity_slice = true;
  for (int64_t d = 0; d < rank; ++d) {
    const DimDesc spec = d < dims.size() ? dims[d] : DimDesc::From(all);
    const int64_t sz = ResolveSliceDimSize(spec, source_shape[d]);
    ValidateSliceDimOffset(spec, source_shape[d], sz);
    is_identity_slice &= IsIdentitySliceDim(spec, source_shape[d], sz);
    sliced_shape.push_back(sz);
    if (!spec.collapsed) {
      squeezed_shape.push_back(sz);
    }
  }

  if (!is_identity_slice) {
    current = EmitMemRefSlice(b, current, dims, sliced_shape);
  }

  if (squeezed_shape.size() < sliced_shape.size()) {
    auto squeezed_type =
        mlir::MemRefType::get(squeezed_shape, base_type.getElementType(),
                              nullptr, base_type.getMemorySpace());
    current = tpu::MemRefSqueezeOp::create(b, squeezed_type, current);
  }

  return current.getAsOpaquePointer();
}

Ref SliceRefImpl(Ref base, llvm::ArrayRef<DimDesc> dims) {
  const void* sliced = SliceMemRefImpl(unwrap(base).getAsOpaquePointer(), dims);
  return wrap<Ref>(mlir::Value::getFromOpaquePointer(sliced), base.dtype());
}

}  // namespace internal

}  // namespace torch_tpu::kiln
