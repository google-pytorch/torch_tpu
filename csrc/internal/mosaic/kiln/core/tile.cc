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

#include "csrc/internal/mosaic/kiln/core/tile.h"

#include <cstdint>
#include <optional>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/elementwise.h"
#include "csrc/internal/mosaic/kiln/core/elementwise_impl.h"  // IWYU pragma: keep
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/mlir_utils.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace torch_tpu::kiln {
namespace {

namespace arith = mlir::arith;
namespace tpu = mlir::tpu;
namespace vector = mlir::vector;

mlir::VectorType VectorTypeOf(const Tile& t) {
  return llvm::cast<mlir::VectorType>(internal::unwrap(t).getType());
}

mlir::Value SplatConstant(mlir::ImplicitLocOpBuilder& b, mlir::VectorType type,
                          mlir::TypedAttr attr) {
  return arith::ConstantOp::create(b, mlir::DenseElementsAttr::get(type, attr));
}

// Get the min value for the given dtype.
// For floats, this is negative infinity.
// For unsigned integers, this is 0.
// For signed integers, this is the most negative integer.
mlir::TypedAttr MinValue(mlir::ImplicitLocOpBuilder& b, DType dtype) {
  ABSL_CHECK(dtype.IsIntOrFloat())  // CRASH_OK
      << "MinValue requires an integer or floating-point dtype";
  mlir::Type mlir_type = internal::ToMlirType(dtype);
  if (auto float_type = llvm::dyn_cast<mlir::FloatType>(mlir_type)) {
    return b.getFloatAttr(mlir_type,
                          llvm::APFloat::getInf(float_type.getFloatSemantics(),
                                                /*Negative=*/true));
  }
  if (dtype.IsUnsigned()) {
    return b.getIntegerAttr(
        mlir_type, llvm::APInt::getMinValue(dtype.GetIntOrFloatBitWidth()));
  }
  return b.getIntegerAttr(
      mlir_type, llvm::APInt::getSignedMinValue(dtype.GetIntOrFloatBitWidth()));
}

Tile Reduce(const Tile& t, int axis, vector::CombiningKind kind,
            mlir::TypedAttr init) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  llvm::ArrayRef<int64_t> dims = t.dims();
  ABSL_CHECK(axis >= 0 && axis < t.rank());  // CRASH_OK
  Dimensions reduced(dims);
  reduced.erase(reduced.begin() + axis);
  llvm::SmallVector<bool> mask(t.rank(), false);
  mask[axis] = true;
  mlir::Value acc = SplatConstant(
      b, mlir::VectorType::get(reduced, internal::ToMlirType(t.dtype())), init);
  mlir::Value r = vector::MultiDimReductionOp::create(b, internal::unwrap(t),
                                                      acc, mask, kind);
  Dimensions keepdims(dims);
  keepdims[axis] = 1;
  return internal::wrap<Tile>(r, t.dtype()).Reshape(keepdims);
}

}  // namespace

Tile::Tile(const void* impl, DType dtype) : ValueBase(impl, dtype) {
  ABSL_CHECK(  // CRASH_OK
      llvm::isa<mlir::VectorType>(internal::unwrap(*this).getType()))
      << "Tile must be a vector value";
}

llvm::ArrayRef<int64_t> Tile::dims() const {
  return VectorTypeOf(*this).getShape();
}

int64_t Tile::rank() const { return VectorTypeOf(*this).getRank(); }

int64_t Tile::num_elements() const {
  return VectorTypeOf(*this).getNumElements();
}

Tile Tile::Reshape(llvm::ArrayRef<int64_t> shape) const {
  if (shape == dims()) return *this;
  mlir::Type elem_type = internal::ToMlirType(dtype());
  ABSL_CHECK_EQ(num_elements(),  // CRASH_OK
                mlir::VectorType::get(shape, elem_type).getNumElements())
      << "Reshape must preserve the number of elements";
  return internal::wrap<Tile>(
      vector::ShapeCastOp::create(internal::CurrentBuilder(),
                                  mlir::VectorType::get(shape, elem_type),
                                  internal::unwrap(*this)),
      dtype());
}

Tile Tile::Squeeze() const {
  llvm::ArrayRef<int64_t> s = dims();
  while (s.size() > 2 && s.front() == 1) s = s.drop_front();
  return Reshape(s);
}

Tile Tile::ToShape(llvm::ArrayRef<int64_t> shape) const {
  if (num_elements() == llvm::product_of(shape)) {
    return Reshape(shape);
  }
  // TODO(willfroom): Add a check that the shape transition is legal.
  return BroadcastTo(shape);
}

Tile Tile::BroadcastTo(llvm::ArrayRef<int64_t> shape) const {
  return internal::wrap<Tile>(
      internal::EmitBroadcast(internal::unwrap(*this), shape), dtype());
}

Tile Tile::Lanes(int64_t n) const {
  ABSL_CHECK_GT(rank(), 0)  // CRASH_OK
      << "Lanes() requires a tile of rank >= 1";
  ABSL_CHECK_GT(n, 0) << "Lanes() target size must be positive";  // CRASH_OK
  const int64_t lane = dims().back();
  if (lane == n) return *this;
  Dimensions target(dims());
  target.back() = n;
  if (lane == 1) return BroadcastTo(target);
  if (n < lane) {
    return Slice(llvm::SmallVector<int64_t>(rank(), 0), target);
  }
  return internal::wrap<Tile>(
      CreateRepeatOp(internal::CurrentBuilder(), internal::unwrap(*this),
                     rank() - 1, n),
      dtype());
}

Tile Tile::Slice(llvm::ArrayRef<int64_t> offsets,
                 llvm::ArrayRef<int64_t> sizes) const {
  ABSL_CHECK_EQ(static_cast<int64_t>(offsets.size()), rank())  // CRASH_OK
      << "Slice() requires one offset per tile dimension";
  ABSL_CHECK_EQ(static_cast<int64_t>(sizes.size()), rank())  // CRASH_OK
      << "Slice() requires one size per tile dimension";
  for (int64_t i = 0; i < rank(); ++i) {
    ABSL_CHECK_GE(offsets[i], 0)  // CRASH_OK
        << "Slice() offset must be non-negative";
    ABSL_CHECK_GT(sizes[i], 0)  // CRASH_OK
        << "Slice() size must be positive";
    ABSL_CHECK_LE(offsets[i] + sizes[i], dims()[i])  // CRASH_OK
        << "Slice() out of bounds for dimension " << i;
  }
  return internal::wrap<Tile>(
      vector::ExtractStridedSliceOp::create(
          internal::CurrentBuilder(), internal::unwrap(*this), offsets, sizes,
          llvm::SmallVector<int64_t>(rank(), 1)),
      dtype());
}

Tile Tile::Rotate(IndexArg amount, int32_t axis) const {
  ABSL_CHECK(axis >= 0 && axis < rank())  // CRASH_OK
      << "Rotate() axis " << axis << " out of bounds for rank " << rank();
  Scalar shift = amount.ToScalar();
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::Value v = internal::unwrap(*this);
  return internal::wrap<Tile>(
      tpu::DynamicRotateOp::create(b, v.getType(), v,
                                   internal::unwrap(shift.ToI32()), axis,
                                   mlir::IntegerAttr(), mlir::IntegerAttr()),
      dtype());
}

Tile Tile::Gather(Tile indices, llvm::ArrayRef<int32_t> dimensions) const {
  ABSL_CHECK(  // CRASH_OK
      indices.dtype().IsInt())
      << "Gather() indices must have an integer dtype";
  ABSL_CHECK_EQ(rank(), indices.rank())  // CRASH_OK
      << "Gather() source and indices must have the same rank";
  for (int32_t d : dimensions) {
    ABSL_CHECK(d >= 0 && d < rank())  // CRASH_OK
        << "Gather() dimension " << d << " out of bounds for rank " << rank();
  }
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(
      tpu::DynamicGatherOp::create(b, internal::unwrap(*this),
                                   internal::unwrap(indices),
                                   b.getDenseI32ArrayAttr(dimensions)),
      dtype());
}

Scalar Tile::Extract(llvm::ArrayRef<int64_t> indices) const {
  ABSL_CHECK_EQ(static_cast<int64_t>(indices.size()), rank())  // CRASH_OK
      << "Extract() requires one index per tile dimension";
  for (int64_t i = 0; i < rank(); ++i) {
    llvm::ArrayRef<int64_t> dimensions = dims();
    ABSL_CHECK(indices[i] >= 0 && indices[i] < dimensions[i])  // CRASH_OK
        << "Extract() index " << indices[i] << " out of bounds for dimension "
        << i << " of size " << dimensions[i];
  }
  return internal::wrap<Scalar>(
      vector::ExtractOp::create(internal::CurrentBuilder(),
                                internal::unwrap(*this), indices),
      dtype());
}

Tile Tile::Full(Literal value) const { return kiln::Full(shape(), value); }

Tile Tile::Splat(Scalar scalar) const { return kiln::Splat(scalar, dims()); }

Tile Zeros(const Shape& shape) { return Full(shape, /*value=*/0.0); }

Tile Full(const Shape& shape, Literal value) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(
      SplatConstant(b, internal::ToMlirVectorType(shape),
                    internal::LiteralToAttr(b, shape.dtype(), value)),
      shape.dtype());
}

Tile NInf(const Shape& shape) {
  ABSL_CHECK(shape.dtype().IsFloat())  // CRASH_OK
      << "NInf() requires a floating-point dtype";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(
      SplatConstant(b, internal::ToMlirVectorType(shape),
                    MinValue(b, shape.dtype())),
      shape.dtype());
}

Tile Iota(const Shape& shape, int axis) {
  ABSL_CHECK(shape.dtype().IsInt())  // CRASH_OK
      << "Iota() requires an integer dtype";
  ABSL_CHECK(axis >= 0 &&  // CRASH_OK
             axis < static_cast<int>(shape.rank()))
      << "Iota() axis " << axis << " out of bounds for rank " << shape.rank();
  for (int64_t dim : shape.dims()) {
    ABSL_CHECK_GT(dim, 0)  // CRASH_OK
        << "Iota() shape dimensions must be positive";
  }
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(
      CreateIotaOp(b, internal::ToMlirVectorType(shape), axis), shape.dtype());
}

Tile Splat(Scalar scalar, llvm::ArrayRef<int64_t> shape) {
  return internal::wrap<Tile>(
      internal::EmitBroadcast(internal::unwrap(scalar), shape), scalar.dtype());
}

Tile Reciprocal(Tile t, bool approx) {
  ABSL_CHECK(t.dtype().IsFloat());  // CRASH_OK
  return internal::wrap<Tile>(
      tpu::ReciprocalOp::create(internal::CurrentBuilder(), internal::unwrap(t),
                                approx,
                                /*full_range=*/false),
      t.dtype());
}

Tile MaxReduce(Tile t, int axis) {
  ABSL_CHECK(t.dtype().IsIntOrFloat())  // CRASH_OK
      << "MaxReduce() requires an integer or floating-point tile";
  const bool is_float = t.dtype().IsFloat();
  vector::CombiningKind kind;
  if (is_float) {
    kind = vector::CombiningKind::MAXIMUMF;
  } else if (t.dtype().IsUnsigned()) {
    kind = vector::CombiningKind::MAXUI;
  } else {
    kind = vector::CombiningKind::MAXSI;
  }
  return Reduce(t, axis, kind, MinValue(internal::CurrentBuilder(), t.dtype()));
}

Tile SumReduce(Tile t, int axis) {
  ABSL_CHECK(t.dtype().IsIntOrFloat())  // CRASH_OK
      << "SumReduce() requires an integer or floating-point tile";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return Reduce(t, axis, vector::CombiningKind::ADD,
                internal::LiteralToAttr(b, t.dtype(), 0.0));
}

Tile Transpose(Tile t) {
  ABSL_CHECK_EQ(t.rank(), 2) << "Transpose() is 2-D only";  // CRASH_OK
  return internal::wrap<Tile>(
      vector::TransposeOp::create(internal::CurrentBuilder(),
                                  internal::unwrap(t),
                                  llvm::ArrayRef<int64_t>{1, 0}),
      t.dtype());
}

Tile Dot(Tile lhs, Tile rhs, const DotOptions& options) {
  ABSL_CHECK(lhs.rank() == 2 && rhs.rank() == 2)  // CRASH_OK
      << "Dot() operands must be 2-D";
  ABSL_CHECK(lhs.dtype().IsIntOrFloat() &&  // CRASH_OK
             rhs.dtype().IsIntOrFloat())
      << "Dot() operands must be integer or floating-point tiles";
  const int64_t lhs_k = lhs.dims()[options.transpose_lhs ? 0 : 1];
  const int64_t rhs_k = rhs.dims()[options.transpose_rhs ? 1 : 0];
  ABSL_CHECK_EQ(lhs_k, rhs_k)  // CRASH_OK
      << "Dot() contracting dimensions must match (" << lhs_k << " vs " << rhs_k
      << ")";
  std::optional<tpu::ContractPrecision> precision;
  if (options.precision.has_value()) {
    switch (*options.precision) {
      case ContractPrecision::kBF16:
        precision = tpu::ContractPrecision::kBF16;
        break;
      case ContractPrecision::kFP32:
        precision = tpu::ContractPrecision::kFP32;
        break;
      case ContractPrecision::kBF16x3:
        precision = tpu::ContractPrecision::kBF16x3;
        break;
    }
  }
  // Matmuls always produce a float32 result.
  DType result_dtype = f32;
  return internal::wrap<Tile>(
      CreateMatmul(internal::CurrentBuilder(), internal::unwrap(lhs),
                   internal::unwrap(rhs),
                   MatmulOptions{.transpose_lhs = options.transpose_lhs,
                                 .transpose_rhs = options.transpose_rhs,
                                 .precision = precision}),
      result_dtype);
}

template struct internal::ElementwiseOps<Tile>;

}  // namespace torch_tpu::kiln
