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

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
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

mlir::MemRefType MemRefTypeOf(const Ref& r) {
  return llvm::cast<mlir::MemRefType>(internal::unwrap(r).getType());
}

bool HasMemorySpace(const Ref& r, tpu::MemorySpace expected_space) {
  auto space = llvm::dyn_cast_or_null<tpu::MemorySpaceAttr>(
      MemRefTypeOf(r).getMemorySpace());
  return space && space.getValue() == expected_space;
}

bool IsSmemRef(const Ref& r) {
  return HasMemorySpace(r, tpu::MemorySpace::kSmem);
}

bool IsVmemRef(const Ref& r) {
  return HasMemorySpace(r, tpu::MemorySpace::kVmem);
}

}  // namespace

Ref::Ref(const void* impl) : ValueBase(impl) {
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
  return internal::wrap<Scalar>(memref::LoadOp::create(
      internal::CurrentBuilder(), internal::unwrap(*this), mlir::ValueRange{}));
}

Ref::operator Tile() const {
  ABSL_CHECK(IsVmemRef(*this))  // CRASH_OK
      << "Ref::operator Tile() expects a VMEM ref";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(vector::LoadOp::create(
      b, mlir::VectorType::get(shape(), internal::ToMlirType(dtype())),
      internal::unwrap(*this), ZeroIndices(b, rank())));
}

void Ref::Store(Tile value, std::optional<Tile> mask) {
  ABSL_CHECK(MemRefTypeOf(*this).hasStaticShape())  // CRASH_OK
      << "Tile store requires a Ref with a static shape";
  ABSL_CHECK(IsVmemRef(*this))  // CRASH_OK
      << "Tile store requires a VMEM ref (use AsyncCopy for HBM)";
  Tile converted = value.AsType(dtype());
  Tile shaped = converted.ToShape(shape());
  mlir::Value mask_value;
  if (mask.has_value()) {
    ABSL_CHECK(mask->dtype().IsBool())  // CRASH_OK
        << "Masked tile store requires an i1 mask tile";
    Tile shaped_mask = mask->ToShape(shape());
    mask_value = internal::unwrap(shaped_mask);
  }

  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  tpu::VectorStoreOp::create(b, internal::unwrap(shaped),
                             internal::unwrap(*this), ZeroIndices(b, rank()),
                             mask_value, /*add=*/false);
}

llvm::ArrayRef<int64_t> Ref::shape() const {
  return MemRefTypeOf(*this).getShape();
}

DType Ref::dtype() const {
  return internal::FromMlirType(MemRefTypeOf(*this).getElementType());
}

int64_t Ref::rank() const { return MemRefTypeOf(*this).getRank(); }

namespace internal {

Ref SliceRefImpl(Ref base, llvm::ArrayRef<DimDesc> dims) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  llvm::ArrayRef<int64_t> source_shape = base.shape();
  const int64_t rank = source_shape.size();
  ABSL_CHECK_LE(static_cast<int64_t>(dims.size()), rank)  // CRASH_OK
      << "Too many slice indices for Ref of rank " << rank;

  llvm::SmallVector<mlir::Value> base_indices;
  llvm::SmallVector<int64_t> sliced_shape;
  llvm::SmallVector<int64_t> squeezed_shape;

  bool has_non_zero_offset = false;
  for (int64_t d = 0; d < rank; ++d) {
    const DimDesc spec = d < dims.size() ? dims[d] : DimDesc::From(all);
    int64_t sz = spec.is_all ? source_shape[d] : spec.size;
    if (!spec.is_all) {
      ABSL_CHECK_GT(sz, 0) << "Slice size must be positive";  // CRASH_OK
      ABSL_CHECK_LE(sz, source_shape[d])                      // CRASH_OK
          << "Slice size must be less than or equal to the dimension size";
    }
    if (spec.offset.IsInt()) {
      int64_t offset = spec.offset.AsInt();
      ABSL_CHECK_GE(offset, 0)  // CRASH_OK
          << "Slice offset must be non-negative";
      ABSL_CHECK_LE(offset + sz, source_shape[d])  // CRASH_OK
          << "Slice offset + size must not exceed dimension size";
      has_non_zero_offset |= (offset != 0);
    } else {
      has_non_zero_offset = true;
    }
    sliced_shape.push_back(sz);
    if (!spec.collapsed) {
      squeezed_shape.push_back(sz);
    }
  }

  mlir::Value current = unwrap(base);
  mlir::MemRefType base_type = MemRefTypeOf(base);
  if (has_non_zero_offset || sliced_shape != source_shape) {
    llvm::SmallVector<mlir::Value, 4> base_indices;
    base_indices.reserve(rank);
    for (int64_t d = 0; d < rank; ++d) {
      const DimDesc spec = d < dims.size() ? dims[d] : DimDesc::From(all);
      base_indices.push_back(unwrap(spec.offset.ToScalar().ToI32()));
    }
    auto sliced_type =
        mlir::MemRefType::get(sliced_shape, base_type.getElementType(), nullptr,
                              base_type.getMemorySpace());
    current =
        mlir::tpu::MemRefSliceOp::create(b, sliced_type, current, base_indices,
                                         /*dynamic_sizes=*/{});
  }

  if (squeezed_shape.size() < sliced_shape.size()) {
    auto squeezed_type =
        mlir::MemRefType::get(squeezed_shape, base_type.getElementType(),
                              nullptr, base_type.getMemorySpace());
    current = tpu::MemRefSqueezeOp::create(b, squeezed_type, current);
  }

  return wrap<Ref>(current);
}

}  // namespace internal

}  // namespace torch_tpu::kiln
