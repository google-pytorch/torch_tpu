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

#include <algorithm>
#include <cstdint>
#include <numbers>
#include <optional>

#include "absl/log/absl_check.h"
#include "csrc/common/libtpu_version.h"
#include "csrc/common/version.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
namespace math = mlir::math;
namespace tpu = mlir::tpu;
namespace vector = mlir::vector;

mlir::VectorType VectorTypeOf(const Tile& t) {
  return llvm::cast<mlir::VectorType>(internal::unwrap(t).getType());
}

mlir::Value SplatConstant(mlir::ImplicitLocOpBuilder& b, mlir::VectorType type,
                          mlir::TypedAttr attr) {
  return arith::ConstantOp::create(b, mlir::DenseElementsAttr::get(type, attr));
}

mlir::Value SplatConstant(mlir::ImplicitLocOpBuilder& b, mlir::VectorType type,
                          Literal literal) {
  return SplatConstant(
      b, type,
      internal::LiteralToAttr(b, internal::FromMlirType(type.getElementType()),
                              literal));
}

mlir::TypedAttr NegativeInfinity(mlir::ImplicitLocOpBuilder& b, DType dtype) {
  ABSL_CHECK(dtype.IsIntOrFloat())  // CRASH_OK
      << "NegativeInfinity requires an integer or floating-point dtype";
  mlir::Type mlir_type = internal::ToMlirType(dtype);
  if (auto float_type = llvm::dyn_cast<mlir::FloatType>(mlir_type)) {
    return b.getFloatAttr(mlir_type,
                          llvm::APFloat::getInf(float_type.getFloatSemantics(),
                                                /*Negative=*/true));
  }
  return b.getIntegerAttr(
      mlir_type, llvm::APInt::getSignedMinValue(dtype.GetIntOrFloatBitWidth()));
}

// numpy broadcasting rule over two static shapes.
Shape BroadcastShapes(llvm::ArrayRef<int64_t> lhs,
                      llvm::ArrayRef<int64_t> rhs) {
  const int64_t rank = std::max(lhs.size(), rhs.size());
  Shape result(rank, 1);
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t l = i < rank - static_cast<int64_t>(lhs.size())
                          ? 1
                          : lhs[i - (rank - lhs.size())];
    const int64_t r = i < rank - static_cast<int64_t>(rhs.size())
                          ? 1
                          : rhs[i - (rank - rhs.size())];
    ABSL_CHECK(l == r || l == 1 || r == 1)  // CRASH_OK
        << "shapes are not broadcastable";
    result[i] = std::max(l, r);
  }
  return result;
}

struct BroadcastedTiles {
  Tile lhs;
  Tile rhs;
};

// Brings `lhs` and `rhs` to a common shape. Dtypes must already match.
BroadcastedTiles Broadcast(Tile lhs, Tile rhs) {
  ABSL_CHECK_EQ(lhs.dtype(), rhs.dtype())  // CRASH_OK
      << "Tile binary operators require operands of the same dtype";
  const Shape shape = BroadcastShapes(lhs.shape(), rhs.shape());
  return {lhs.BroadcastTo(shape), rhs.BroadcastTo(shape)};
}

template <typename FloatOp, typename IntOp>
Tile Arith(Tile lhs, Tile rhs) {
  auto [l, r] = Broadcast(lhs, rhs);
  ABSL_CHECK(l.dtype().IsIntOrFloat())  // CRASH_OK
      << "Tile arithmetic requires integer or floating-point operands";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  if (l.dtype().IsFloat()) {
    return internal::wrap<Tile>(
        FloatOp::create(b, internal::unwrap(l), internal::unwrap(r)));
  }
  return internal::wrap<Tile>(
      IntOp::create(b, internal::unwrap(l), internal::unwrap(r)));
}

Tile Compare(Tile lhs, Tile rhs, arith::CmpFPredicate fpred,
             arith::CmpIPredicate ipred) {
  auto [l, r] = Broadcast(lhs, rhs);
  ABSL_CHECK(l.dtype().IsIntOrFloat())  // CRASH_OK
      << "Tile comparison requires integer or floating-point operands";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  if (l.dtype().IsFloat()) {
    return internal::wrap<Tile>(arith::CmpFOp::create(
        b, fpred, internal::unwrap(l), internal::unwrap(r)));
  }
  return internal::wrap<Tile>(arith::CmpIOp::create(
      b, ipred, internal::unwrap(l), internal::unwrap(r)));
}

template <typename Op>
Tile BitwiseTile(Tile lhs, Tile rhs) {
  DType lhs_dtype = lhs.dtype();
  DType rhs_dtype = rhs.dtype();
  ABSL_CHECK(  // CRASH_OK
      (lhs_dtype.IsInt() || lhs_dtype.IsBool()) &&
      (rhs_dtype.IsInt() || rhs_dtype.IsBool()))
      << "Bitwise tile operators require integer or mask operands";
  auto [l, r] = Broadcast(lhs, rhs);
  return internal::wrap<Tile>(Op::create(
      internal::CurrentBuilder(), internal::unwrap(l), internal::unwrap(r)));
}

template <typename Op>
Tile UnaryFloat(Tile t) {
  ABSL_CHECK(t.dtype().IsFloat())  // CRASH_OK
      << "Unary math operation requires a floating-point tile";
  return internal::wrap<Tile>(
      Op::create(internal::CurrentBuilder(), internal::unwrap(t)));
}

Tile Reduce(const Tile& t, int axis, vector::CombiningKind kind,
            mlir::TypedAttr init) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  llvm::ArrayRef<int64_t> shape = t.shape();
  ABSL_CHECK(axis >= 0 && axis < t.rank());  // CRASH_OK
  Shape reduced(shape);
  reduced.erase(reduced.begin() + axis);
  llvm::SmallVector<bool> mask(t.rank(), false);
  mask[axis] = true;
  mlir::Value acc = SplatConstant(
      b, mlir::VectorType::get(reduced, internal::ToMlirType(t.dtype())), init);
  mlir::Value r = vector::MultiDimReductionOp::create(b, internal::unwrap(t),
                                                      acc, mask, kind);
  Shape keepdims(shape);
  keepdims[axis] = 1;
  return internal::wrap<Tile>(r).Reshape(keepdims);
}

}  // namespace

Tile::Tile(const void* impl) : ValueBase(impl) {
  ABSL_CHECK(  // CRASH_OK
      llvm::isa<mlir::VectorType>(internal::unwrap(*this).getType()))
      << "Tile must be a vector value";
}

Tile::Tile(Scalar scalar) : Tile(Splat(scalar, {})) {}

llvm::ArrayRef<int64_t> Tile::shape() const {
  return VectorTypeOf(*this).getShape();
}

DType Tile::dtype() const {
  return internal::FromMlirType(VectorTypeOf(*this).getElementType());
}

int64_t Tile::rank() const { return VectorTypeOf(*this).getRank(); }

int64_t Tile::num_elements() const {
  return VectorTypeOf(*this).getNumElements();
}

Tile Tile::AsType(DType target) const {
  return internal::wrap<Tile>(
      internal::EmitCast(internal::unwrap(*this), dtype(), target));
}

Tile Tile::Bitcast(DType target) const {
  return internal::wrap<Tile>(
      internal::EmitBitcast(internal::unwrap(*this), dtype(), target));
}

Tile Tile::ToI32() const { return AsType(Int32Type()); }
Tile Tile::ToF32() const { return AsType(Float32Type()); }
Tile Tile::ToBF16() const { return AsType(BFloat16Type()); }

Tile Tile::Reshape(llvm::ArrayRef<int64_t> shape) const {
  if (shape == this->shape()) return *this;
  mlir::Type elem_type = internal::ToMlirType(dtype());
  ABSL_CHECK_EQ(num_elements(),  // CRASH_OK
                mlir::VectorType::get(shape, elem_type).getNumElements())
      << "Reshape must preserve the number of elements";
  return internal::wrap<Tile>(vector::ShapeCastOp::create(
      internal::CurrentBuilder(), mlir::VectorType::get(shape, elem_type),
      internal::unwrap(*this)));
}

Tile Tile::Squeeze() const {
  llvm::ArrayRef<int64_t> s = shape();
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
  if (shape == this->shape()) {
    return *this;
  }
  ABSL_CHECK_EQ(BroadcastShapes(this->shape(), shape), shape)  // CRASH_OK
      << "Cannot broadcast tile to target shape";
  return internal::wrap<Tile>(vector::BroadcastOp::create(
      internal::CurrentBuilder(),
      mlir::VectorType::get(shape, internal::ToMlirType(dtype())),
      internal::unwrap(*this)));
}

Tile Tile::Lanes(int64_t n) const {
  ABSL_CHECK_GT(rank(), 0)  // CRASH_OK
      << "Lanes() requires a tile of rank >= 1";
  ABSL_CHECK_GT(n, 0) << "Lanes() target size must be positive";  // CRASH_OK
  const int64_t lane = shape().back();
  if (lane == n) return *this;
  Shape target(shape());
  target.back() = n;
  if (lane == 1) return BroadcastTo(target);
  if (n < lane) {
    return Slice(llvm::SmallVector<int64_t>(rank(), 0), target);
  }
  return internal::wrap<Tile>(mlir::torch_tpu::CreateRepeatOp(
      internal::CurrentBuilder(), internal::unwrap(*this), rank() - 1, n));
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
    ABSL_CHECK_LE(offsets[i] + sizes[i], shape()[i])  // CRASH_OK
        << "Slice() out of bounds for dimension " << i;
  }
  return internal::wrap<Tile>(vector::ExtractStridedSliceOp::create(
      internal::CurrentBuilder(), internal::unwrap(*this), offsets, sizes,
      llvm::SmallVector<int64_t>(rank(), 1)));
}

Tile Tile::Rotate(IndexArg amount, int32_t axis) const {
  ABSL_CHECK(axis >= 0 && axis < rank())  // CRASH_OK
      << "Rotate() axis " << axis << " out of bounds for rank " << rank();
  Scalar shift = amount.ToScalar();
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::Value v = internal::unwrap(*this);
  return internal::wrap<Tile>(tpu::DynamicRotateOp::create(
      b, v.getType(), v, internal::unwrap(shift.ToI32()), axis,
      mlir::IntegerAttr(), mlir::IntegerAttr()));
}

Tile Tile::Gather(Tile indices, llvm::ArrayRef<int32_t> dimensions) const {
  ABSL_CHECK(  // CRASH_OK
      indices.dtype().IsInt())
      << "Gather() indices must have an integer dtype";
  if (indices.rank() == 0) {
    indices = indices.BroadcastTo(shape());
  }
  ABSL_CHECK_EQ(rank(), indices.rank())  // CRASH_OK
      << "Gather() source and indices must have the same rank";
  for (int32_t d : dimensions) {
    ABSL_CHECK(d >= 0 && d < rank())  // CRASH_OK
        << "Gather() dimension " << d << " out of bounds for rank " << rank();
  }
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(tpu::DynamicGatherOp::create(
      b, internal::unwrap(*this), internal::unwrap(indices),
      b.getDenseI32ArrayAttr(dimensions)));
}

Scalar Tile::Extract(llvm::ArrayRef<int64_t> indices) const {
  ABSL_CHECK_EQ(static_cast<int64_t>(indices.size()), rank())  // CRASH_OK
      << "Extract() requires one index per tile dimension";
  for (int64_t i = 0; i < rank(); ++i) {
    ABSL_CHECK(indices[i] >= 0 && indices[i] < shape()[i])  // CRASH_OK
        << "Extract() index " << indices[i] << " out of bounds for dimension "
        << i << " of size " << shape()[i];
  }
  return internal::wrap<Scalar>(vector::ExtractOp::create(
      internal::CurrentBuilder(), internal::unwrap(*this), indices));
}

Tile Tile::Full(Literal value) const {
  return internal::wrap<Tile>(
      SplatConstant(internal::CurrentBuilder(), VectorTypeOf(*this), value));
}

Tile Zeros(llvm::ArrayRef<int64_t> shape, DType dtype) {
  return Full(shape, dtype, 0.0);
}

Tile Full(llvm::ArrayRef<int64_t> shape, DType dtype, Literal value) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(SplatConstant(
      b, mlir::VectorType::get(shape, internal::ToMlirType(dtype)), value));
}

Tile NInf(llvm::ArrayRef<int64_t> shape, DType dtype) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(SplatConstant(
      b, mlir::VectorType::get(shape, internal::ToMlirType(dtype)),
      NegativeInfinity(b, dtype)));
}

Tile Iota(llvm::ArrayRef<int64_t> shape, DType dtype, int axis) {
  ABSL_CHECK(dtype.IsInt()) << "Iota() requires an integer dtype";  // CRASH_OK
  ABSL_CHECK(axis >= 0 && axis < static_cast<int>(shape.size()))    // CRASH_OK
      << "Iota() axis " << axis << " out of bounds for rank " << shape.size();
  for (int64_t dim : shape) {
    ABSL_CHECK_GT(dim, 0)  // CRASH_OK
        << "Iota() shape dimensions must be positive";
  }
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(mlir::torch_tpu::CreateIotaOp(
      b, mlir::VectorType::get(shape, internal::ToMlirType(dtype)), axis));
}

Tile Splat(Scalar scalar, llvm::ArrayRef<int64_t> shape) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Tile>(vector::BroadcastOp::create(
      b, mlir::VectorType::get(shape, internal::ToMlirType(scalar.dtype())),
      internal::unwrap(scalar)));
}

Tile operator-(Tile t) {
  ABSL_CHECK(t.dtype().IsIntOrFloat())  // CRASH_OK
      << "Unary - requires an integer or floating-point tile";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  if (t.dtype().IsFloat()) {
    return internal::wrap<Tile>(arith::NegFOp::create(b, internal::unwrap(t)));
  }
  return t.Full(0.0) - t;
}
Tile operator+(Tile lhs, Tile rhs) {
  return Arith<arith::AddFOp, arith::AddIOp>(lhs, rhs);
}
Tile operator-(Tile lhs, Tile rhs) {
  return Arith<arith::SubFOp, arith::SubIOp>(lhs, rhs);
}
Tile operator*(Tile lhs, Tile rhs) {
  return Arith<arith::MulFOp, arith::MulIOp>(lhs, rhs);
}
Tile operator/(Tile lhs, Tile rhs) {
  return Arith<arith::DivFOp, arith::DivSIOp>(lhs, rhs);
}
Tile operator%(Tile lhs, Tile rhs) {
  ABSL_CHECK(lhs.dtype().IsInt())  // CRASH_OK
      << "Tile % requires integer operands";
  auto [l, r] = Broadcast(lhs, rhs);
  return internal::wrap<Tile>(arith::RemSIOp::create(
      internal::CurrentBuilder(), internal::unwrap(l), internal::unwrap(r)));
}

Tile operator==(Tile lhs, Tile rhs) {
  return Compare(lhs, rhs, arith::CmpFPredicate::OEQ, arith::CmpIPredicate::eq);
}
Tile operator!=(Tile lhs, Tile rhs) {
  return Compare(lhs, rhs, arith::CmpFPredicate::ONE, arith::CmpIPredicate::ne);
}
Tile operator<(Tile lhs, Tile rhs) {
  return Compare(lhs, rhs, arith::CmpFPredicate::OLT,
                 arith::CmpIPredicate::slt);
}
Tile operator<=(Tile lhs, Tile rhs) {
  return Compare(lhs, rhs, arith::CmpFPredicate::OLE,
                 arith::CmpIPredicate::sle);
}
Tile operator>(Tile lhs, Tile rhs) {
  return Compare(lhs, rhs, arith::CmpFPredicate::OGT,
                 arith::CmpIPredicate::sgt);
}
Tile operator>=(Tile lhs, Tile rhs) {
  return Compare(lhs, rhs, arith::CmpFPredicate::OGE,
                 arith::CmpIPredicate::sge);
}

Tile operator&(Tile lhs, Tile rhs) {
  return BitwiseTile<arith::AndIOp>(lhs, rhs);
}
Tile operator|(Tile lhs, Tile rhs) {
  return BitwiseTile<arith::OrIOp>(lhs, rhs);
}
Tile operator^(Tile lhs, Tile rhs) {
  return BitwiseTile<arith::XOrIOp>(lhs, rhs);
}
Tile operator<<(Tile lhs, Tile rhs) {
  return BitwiseTile<arith::ShLIOp>(lhs, rhs);
}
Tile operator>>(Tile lhs, Tile rhs) {
  return BitwiseTile<arith::ShRSIOp>(lhs, rhs);
}

Tile LogicalRightShift(Tile lhs, Tile rhs) {
  return BitwiseTile<arith::ShRUIOp>(lhs, rhs);
}

Tile Exp(Tile t) { return UnaryFloat<math::ExpOp>(t); }
Tile Exp2(Tile t) { return UnaryFloat<math::Exp2Op>(t); }
Tile Log(Tile t) { return UnaryFloat<math::LogOp>(t); }
Tile Log2(Tile t) {
  // Older libtpu releases have no native log2 lowering.
  static const bool is_natively_supported =
      torch_tpu::GetLibtpuVersion() >= torch_tpu::Version(0, 0, 48);
  if (is_natively_supported) {
    return UnaryFloat<math::Log2Op>(t);
  }
  return Log(t) * std::numbers::log2e;
}
Tile Sqrt(Tile t) { return UnaryFloat<math::SqrtOp>(t); }
Tile Rsqrt(Tile t) { return UnaryFloat<math::RsqrtOp>(t); }
Tile Tanh(Tile t) { return UnaryFloat<math::TanhOp>(t); }
Tile Reciprocal(Tile t, bool approx) {
  ABSL_CHECK(t.dtype().IsFloat());  // CRASH_OK
  return internal::wrap<Tile>(tpu::ReciprocalOp::create(
      internal::CurrentBuilder(), internal::unwrap(t), approx,
      /*full_range=*/false));
}
Tile Abs(Tile t) {
  ABSL_CHECK(t.dtype().IsIntOrFloat())  // CRASH_OK
      << "Abs() requires an integer or floating-point tile";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  if (t.dtype().IsFloat()) {
    return internal::wrap<Tile>(math::AbsFOp::create(b, internal::unwrap(t)));
  }
  return internal::wrap<Tile>(math::AbsIOp::create(b, internal::unwrap(t)));
}

Tile MaxReduce(Tile t, int axis) {
  ABSL_CHECK(t.dtype().IsIntOrFloat())  // CRASH_OK
      << "MaxReduce() requires an integer or floating-point tile";
  const bool is_float = t.dtype().IsFloat();
  return Reduce(
      t, axis,
      is_float ? vector::CombiningKind::MAXIMUMF : vector::CombiningKind::MAXSI,
      NegativeInfinity(internal::CurrentBuilder(), t.dtype()));
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
  return internal::wrap<Tile>(vector::TransposeOp::create(
      internal::CurrentBuilder(), internal::unwrap(t),
      llvm::ArrayRef<int64_t>{1, 0}));
}

Tile Max(Tile lhs, Tile rhs) {
  // TODO(willfroom): maybe think about a new CondTile type?
  if (lhs.dtype().IsBool() && rhs.dtype().IsBool()) {
    return lhs | rhs;
  }
  return Arith<arith::MaximumFOp, arith::MaxSIOp>(lhs, rhs);
}
Tile Min(Tile lhs, Tile rhs) {
  if (lhs.dtype().IsBool() && rhs.dtype().IsBool()) {
    return lhs & rhs;
  }
  return Arith<arith::MinimumFOp, arith::MinSIOp>(lhs, rhs);
}

Tile Select(Tile mask, Tile if_true, Tile if_false) {
  ABSL_CHECK(mask.dtype().IsBool()) << "mask must be i1";  // CRASH_OK
  ABSL_CHECK_EQ(if_true.dtype(), if_false.dtype())         // CRASH_OK
      << "Select() branches must have the same dtype";
  const Shape shape = BroadcastShapes(
      mask.shape(), BroadcastShapes(if_true.shape(), if_false.shape()));
  return internal::wrap<Tile>(arith::SelectOp::create(
      internal::CurrentBuilder(), internal::unwrap(mask.BroadcastTo(shape)),
      internal::unwrap(if_true.BroadcastTo(shape)),
      internal::unwrap(if_false.BroadcastTo(shape))));
}

Tile Dot(Tile lhs, Tile rhs, const DotOptions& options) {
  ABSL_CHECK(lhs.rank() == 2 && rhs.rank() == 2)  // CRASH_OK
      << "Dot() operands must be 2-D";
  ABSL_CHECK(lhs.dtype().IsIntOrFloat() &&  // CRASH_OK
             rhs.dtype().IsIntOrFloat())
      << "Dot() operands must be integer or floating-point tiles";
  const int64_t lhs_k = lhs.shape()[options.transpose_lhs ? 0 : 1];
  const int64_t rhs_k = rhs.shape()[options.transpose_rhs ? 1 : 0];
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
  return internal::wrap<Tile>(mlir::torch_tpu::CreateMatmul(
      internal::CurrentBuilder(), internal::unwrap(lhs), internal::unwrap(rhs),
      mlir::torch_tpu::MatmulOptions{.transpose_lhs = options.transpose_lhs,
                                     .transpose_rhs = options.transpose_rhs,
                                     .precision = precision}));
}

}  // namespace torch_tpu::kiln
