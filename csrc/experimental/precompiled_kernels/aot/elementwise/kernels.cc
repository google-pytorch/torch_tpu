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

// Elementwise arithmetic and the shared HBM -> VMEM -> HBM tile loop.

#include "csrc/experimental/precompiled_kernels/aot/elementwise/kernels.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "csrc/ops/op_names.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {
namespace tpu = mlir::tpu;

// The kernel table. An op belongs here only if its name plus the signature
// (dtypes, ranks, arity) fix its semantics, because that is all the runtime
// matches on: add.out with alpha != 1 has a third operand and so never fits
// a two-input signature. Add arithmetic here and a row to kOps below.
mlir::Value Relu(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange v) {
  mlir::Value zero =
      b.create<mlir::arith::ConstantOp>(loc, b.getZeroAttr(v[0].getType()));
  return b.create<mlir::arith::MaximumFOp>(loc, v[0], zero);
}
mlir::Value Neg(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange v) {
  return b.create<mlir::arith::NegFOp>(loc, v[0]);
}
mlir::Value Abs(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange v) {
  return b.create<mlir::math::AbsFOp>(loc, v[0]);
}
mlir::Value Add(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange v) {
  return b.create<mlir::arith::AddFOp>(loc, v[0], v[1]);
}
mlir::Value Sub(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange v) {
  return b.create<mlir::arith::SubFOp>(loc, v[0], v[1]);
}
mlir::Value Mul(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange v) {
  return b.create<mlir::arith::MulFOp>(loc, v[0], v[1]);
}
mlir::Value Max(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange v) {
  return b.create<mlir::arith::MaximumFOp>(loc, v[0], v[1]);
}
mlir::Value Min(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange v) {
  return b.create<mlir::arith::MinimumFOp>(loc, v[0], v[1]);
}

const ElementwiseOp kOps[] = {
    {OpName::kRelu, 1, Relu},      {OpName::kNegOut, 1, Neg},
    {OpName::kAbsOut, 1, Abs},     {OpName::kAddOut, 2, Add},
    {OpName::kSubOut, 2, Sub},     {OpName::kMulOut, 2, Mul},
    {OpName::kMaximumOut, 2, Max}, {OpName::kMinimumOut, 2, Min},
};
constexpr int64_t kSublanes = 8;
constexpr int64_t kLanes = 128;

}  // namespace

absl::Span<const ElementwiseOp> Operations() { return kOps; }

// The Mosaic program for one kernel:
//   func @main(%in_0 .. %in_{arity-1}, %private, %out : memref<?x..x?xT>)
// The custom call aliases %private to %out, so the kernel writes %out and
// never touches %private. Leading axes step by one element, the trailing two
// by one 8x128 tile; tail tiles read and write the padding that the tiled
// layout guarantees.
mlir::OwningOpRef<mlir::ModuleOp> BuildKernel(mlir::MLIRContext& context,
                                              const ElementwiseOp& op,
                                              xla::PrimitiveType dtype,
                                              int rank) {
  context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                      mlir::math::MathDialect, mlir::memref::MemRefDialect,
                      mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                      tpu::TPUDialect>();
  const mlir::Type element = dtype == xla::BF16
                                 ? mlir::Type(mlir::BFloat16Type::get(&context))
                                 : mlir::Type(mlir::Float32Type::get(&context));
  mlir::OpBuilder b(&context);
  const mlir::Location loc = b.getUnknownLoc();
  auto space = [&](tpu::MemorySpace s) {
    return tpu::MemorySpaceAttr::get(&context, s);
  };
  llvm::SmallVector<int64_t, 4> tile(rank, 1);
  tile[rank - 2] = kSublanes;
  tile[rank - 1] = kLanes;
  const auto dynamic = mlir::MemRefType::get(
      llvm::SmallVector<int64_t, 4>(rank, mlir::ShapedType::kDynamic), element,
      nullptr, space(tpu::MemorySpace::kHbm));
  const auto hbm_tile = mlir::MemRefType::get(tile, element, nullptr,
                                              space(tpu::MemorySpace::kHbm));
  const auto vmem_tile = mlir::MemRefType::get(tile, element, nullptr,
                                               space(tpu::MemorySpace::kVmem));
  const auto semaphore =
      mlir::MemRefType::get({}, tpu::SemaphoreType::get(&context), nullptr,
                            space(tpu::MemorySpace::kSemaphoreMem));
  const auto vector = mlir::VectorType::get(tile, element);

  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  b.setInsertionPointToEnd(module->getBody());
  auto function = b.create<mlir::func::FuncOp>(
      loc, "main",
      b.getFunctionType(llvm::SmallVector<mlir::Type, 4>(op.arity + 2, dynamic),
                        {}));
  mlir::Block* entry = function.addEntryBlock();
  b.setInsertionPointToStart(entry);

  auto i32 = [&](int64_t value) -> mlir::Value {
    return b.create<mlir::arith::ConstantIntOp>(loc, value, 32);
  };
  const llvm::SmallVector<mlir::Value, 4> origin(
      rank, b.create<mlir::arith::ConstantIndexOp>(loc, 0));
  std::vector<mlir::Value> scratch;
  scratch.reserve(op.arity);
  for (int i = 0; i < op.arity; ++i) {
    scratch.push_back(b.create<mlir::memref::AllocaOp>(loc, vmem_tile));
  }
  mlir::Value read_done = b.create<tpu::AllocaSemaphoreOp>(loc, semaphore);
  mlir::Value write_done = b.create<tpu::AllocaSemaphoreOp>(loc, semaphore);

  std::vector<mlir::Value> offsets;
  offsets.reserve(rank);
  for (int axis = 0; axis < rank; ++axis) {
    mlir::Value extent = b.create<mlir::arith::IndexCastOp>(
        loc, b.getI32Type(),
        b.create<mlir::memref::DimOp>(loc, entry->getArgument(0), axis));
    // ceil(extent / tile) without overflowing at the largest S32 extent.
    mlir::Value last = b.create<mlir::arith::SubIOp>(loc, extent, i32(1));
    mlir::Value count = b.create<mlir::arith::ShRSIOp>(
        loc, last, i32(llvm::Log2_64(tile[axis])));
    count = b.create<mlir::arith::AddIOp>(loc, count, i32(1));
    auto loop = b.create<mlir::scf::ForOp>(loc, i32(0), count, i32(1));
    b.setInsertionPointToStart(loop.getBody());
    offsets.push_back(b.create<mlir::arith::MulIOp>(loc, loop.getInductionVar(),
                                                    i32(tile[axis])));
  }
  auto slice = [&](mlir::Value buffer) -> mlir::Value {
    return b.create<tpu::MemRefSliceOp>(loc, hbm_tile, buffer, offsets,
                                        mlir::ValueRange{});
  };
  auto copy = [&](mlir::Value source, mlir::Value target, mlir::Value done) {
    b.create<tpu::EnqueueDMAOp>(loc, source, mlir::Value(), target, done,
                                mlir::Value(), mlir::Value(), mlir::Value());
    b.create<tpu::SemaphoreWaitOp>(loc, done, i32(1));
  };
  std::vector<mlir::Value> tiles;
  tiles.reserve(op.arity);
  for (int i = 0; i < op.arity; ++i) {
    copy(slice(entry->getArgument(i)), scratch[i], read_done);
    tiles.push_back(
        b.create<mlir::vector::LoadOp>(loc, vector, scratch[i], origin));
  }
  b.create<mlir::vector::StoreOp>(loc, op.build_tile(b, loc, tiles), scratch[0],
                                  origin);
  copy(scratch[0], slice(entry->getArgument(op.arity + 1)), write_done);

  b.setInsertionPointToEnd(entry);
  b.create<mlir::func::ReturnOp>(loc);
  return module;
}

}  // namespace torch_tpu
