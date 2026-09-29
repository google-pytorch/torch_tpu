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

#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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

namespace torch_tpu {
namespace {

namespace tpu = mlir::tpu;
// Scan rows in order. For each 128-column block, load the containing 8x128
// input tile, extract the current row, and inspect its valid lanes in order.
// This baseline reloads the input tile for each row.
//
// Each selected element appends (row,column) to two 128-element staging
// vectors at slot count % 128. Increment count and flush both vectors whenever
// the stage fills. Staging persists across input tiles and rows. After the
// scan, flush a partial stage into the output's normal tile padding.
//
// Conceptually:
//   count = 0
//   for row in [0,M):
//     for col in [0,N):
//       if IsNonzero(input[row,col]): output[count++] = (row,col)
// This preserves lexicographic ordering without a sort or prefix sum.
class NonzeroBuilder {
 public:
  explicit NonzeroBuilder(mlir::MLIRContext& context)
      : b_(mlir::UnknownLoc::get(&context), &context) {}

  // Emit the complete scalar scan, then flush the remaining output coordinates.
  mlir::OwningOpRef<mlir::ModuleOp> Build() {
    auto module = CreateModule();
    Initialize();
    mlir::Value num_rows = Dim(input_, 0), num_cols = Dim(input_, 1);
    For(Const(0), num_rows, 1, [&](mlir::Value row) {
      For(Const(0), num_cols, 128, [&](mlir::Value col) {
        mlir::Value values = LoadRow(row, col);
        mlir::Value limit = Min(Const(128), Sub(num_cols, col));
        For(Const(0), limit, 1, [&](mlir::Value lane) {
          When(IsNonzero(values, lane), [&] { Append(row, Add(col, lane)); });
        });
      });
    });
    FlushTail();
    mlir::func::ReturnOp::create(b_);
    return module;
  }

 private:
  // Classify one logical lane using the predicate specified in the header.
  mlir::Value IsNonzero(mlir::Value values, mlir::Value lane) {
    // TPU scalar arithmetic is integer-only. Compare float magnitude bits:
    // any set bit is nonzero, which excludes only +0.0 and -0.0.
    mlir::Value bits = mlir::arith::BitcastOp::create(
        b_, b_.getI32Type(), ExtractLane(values, lane));
    return Compare(mlir::arith::CmpIPredicate::sgt, And(bits, 0x7fffffff),
                   Const(0));
  }

  // DMA requires whole physical tiles. Select the requested logical row after
  // loading its containing 8x128 tile; the scalar scan masks the column tail.
  mlir::Value LoadRow(mlir::Value row, mlir::Value col) {
    auto base = tpu::AssumeMultipleOp::create(b_, And(row, -8), 8);
    auto offset = tpu::AssumeMultipleOp::create(b_, col, 128);
    auto src = tpu::MemRefSliceOp::create(
        b_, Ref({8, 128}, b_.getF32Type(), tpu::MemorySpace::kHbm), input_,
        mlir::ValueRange{base, offset}, mlir::ValueRange{});
    Copy(src, tile_);
    auto type = mlir::VectorType::get({8, 128}, b_.getF32Type());
    mlir::Value values = mlir::vector::LoadOp::create(
        b_, type, tile_, mlir::ValueRange{Index(0), Index(0)});
    values = tpu::DynamicRotateOp::create(
        b_, type, values, Sub(Const(8), And(row, 7)), 0, mlir::IntegerAttr(),
        mlir::IntegerAttr());
    return mlir::vector::ExtractStridedSliceOp::create(
        b_, values, llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{1, 128}, llvm::ArrayRef<int64_t>{1, 1});
  }

  // Place one pair at count % 128 and advance the device counter.
  // Flush only when all 128 staged pairs have been filled.
  void Append(mlir::Value row, mlir::Value col) {
    mlir::Value count =
        mlir::memref::LoadOp::create(b_, count_, mlir::ValueRange{});
    mlir::Value mask = mlir::arith::CmpIOp::create(
        b_, mlir::arith::CmpIPredicate::eq, position_, Splat(And(count, 127)));
    Store(Splat(row), stage_[0], mask);
    Store(Splat(col), stage_[1], mask);
    mlir::Value next = Add(count, Const(1));
    mlir::memref::StoreOp::create(b_, next, count_, mlir::ValueRange{});
    When(Compare(mlir::arith::CmpIPredicate::eq, And(next, 127), Const(0)),
         [&] { Flush(And(count, -128)); });
  }

  // Each 128-pair block occupies two low-word tiles: rows, then columns.
  // first is a multiple of 128, so its tile index is 2 * first / 128.
  void Flush(mlir::Value first) {
    mlir::Value tile = mlir::arith::ShRSIOp::create(b_, first, Const(6));
    WriteTile(stage_[0], tile);
    WriteTile(stage_[1], Add(tile, Const(1)));
  }

  // Flush a partial final stage; an exactly full stage was already written.
  void FlushTail() {
    mlir::Value count =
        mlir::memref::LoadOp::create(b_, count_, mlir::ValueRange{});
    When(Compare(mlir::arith::CmpIPredicate::ne, And(count, 127), Const(0)),
         [&] { Flush(And(count, -128)); });
  }

  // Copy one 128-word VMEM tile to a storage-row index in the native output.
  void WriteTile(mlir::Value tile, mlir::Value tile_index) {
    auto dst = tpu::MemRefSliceOp::create(
        b_, Ref({1, 128}, b_.getI32Type(), tpu::MemorySpace::kHbm), output_,
        mlir::ValueRange{tile_index, Const(0)}, mlir::ValueRange{});
    Copy(tile, dst);
  }

  // Gather a runtime-selected lane, then extract its replicated value as a
  // scalar.
  mlir::Value ExtractLane(mlir::Value values, mlir::Value lane) {
    mlir::Value gathered =
        tpu::DynamicGatherOp::create(b_, values.getType(), values, Splat(lane),
                                     b_.getDenseI32ArrayAttr({1}));
    return mlir::vector::ExtractOp::create(b_, gathered,
                                           llvm::ArrayRef<int64_t>{0, 0});
  }

  // Write staging lanes; a mask preserves coordinates already in the stage.
  void Store(mlir::Value value, mlir::Value ref,
             mlir::Value mask = mlir::Value()) {
    tpu::VectorStoreOp::create(
        b_, value, ref, mlir::ValueRange{Index(0), Index(0)}, mask, false);
  }

  // Synchronous DMA: source and destination are safe to reuse after this
  // returns.
  void Copy(mlir::Value src, mlir::Value dst) {
    tpu::EnqueueDMAOp::create(b_, src, mlir::Value(), dst, sem_, mlir::Value(),
                              mlir::Value(), mlir::Value());
    tpu::WaitDMAOp::create(b_, src, mlir::Value(), dst, sem_, mlir::Value(),
                           mlir::Value(), mlir::Value(), /*priority=*/0,
                           /*strict_ordering=*/false, /*wait_target=*/true);
  }

  // Local IR arithmetic helpers; Const emits an S32 constant, Index an MLIR
  // index.
  mlir::Value Const(int64_t v) {
    return mlir::arith::ConstantIntOp::create(b_, v, 32);
  }

  mlir::Value Index(int64_t v) {
    return mlir::arith::ConstantIndexOp::create(b_, v);
  }

  mlir::Value Add(mlir::Value a, mlir::Value b) {
    return mlir::arith::AddIOp::create(b_, a, b);
  }

  mlir::Value Sub(mlir::Value a, mlir::Value b) {
    return mlir::arith::SubIOp::create(b_, a, b);
  }

  mlir::Value Min(mlir::Value a, mlir::Value b) {
    return mlir::arith::MinSIOp::create(b_, a, b);
  }

  mlir::Value Compare(mlir::arith::CmpIPredicate predicate, mlir::Value a,
                      mlir::Value b) {
    return mlir::arith::CmpIOp::create(b_, predicate, a, b);
  }

  mlir::Value And(mlir::Value a, int64_t mask) {
    return mlir::arith::AndIOp::create(b_, a, Const(mask));
  }

  // Replicate a scalar across the 128 lanes of a staging vector.
  mlir::Value Splat(mlir::Value v) {
    return mlir::vector::BroadcastOp::create(
        b_, mlir::VectorType::get({1, 128}, v.getType()), v);
  }

  // Read an actual runtime extent from the memref metadata as S32.
  mlir::Value Dim(mlir::Value ref, int axis) {
    return mlir::arith::IndexCastOp::create(
        b_, b_.getI32Type(), mlir::memref::DimOp::create(b_, ref, axis));
  }

  // Describe buffer shape and memory space without allocating storage.
  mlir::MemRefType Ref(llvm::ArrayRef<int64_t> dims, mlir::Type type,
                       tpu::MemorySpace space) {
    return mlir::MemRefType::get(
        dims, type, nullptr, tpu::MemorySpaceAttr::get(b_.getContext(), space));
  }

  // Emit a device loop over [begin,end); the C++ callback builds its body once.
  void For(mlir::Value begin, mlir::Value end, int step,
           llvm::function_ref<void(mlir::Value)> body) {
    auto loop = mlir::scf::ForOp::create(b_, begin, end, Const(step));
    mlir::OpBuilder::InsertionGuard guard(b_);
    b_.setInsertionPointToStart(loop.getBody());
    body(loop.getInductionVar());
  }

  // Emit a device conditional; this is not a host-side branch.
  void When(mlir::Value condition, llvm::function_ref<void()> body) {
    auto branch = mlir::scf::IfOp::create(b_, condition, false);
    mlir::OpBuilder::InsertionGuard guard(b_);
    b_.setInsertionPointToStart(&branch.getThenRegion().front());
    body();
  }

  // Allocate one input tile, two coordinate stages and a scalar count.
  // Zero the output high words once; scan and flush only write coordinate low
  // words.
  void Initialize() {
    tile_ = mlir::memref::AllocaOp::create(
        b_, Ref({8, 128}, b_.getF32Type(), tpu::MemorySpace::kVmem));
    for (auto& plane : stage_) {
      plane = mlir::memref::AllocaOp::create(
          b_, Ref({1, 128}, b_.getI32Type(), tpu::MemorySpace::kVmem));
      Store(Splat(Const(0)), plane);
    }
    count_ = mlir::memref::AllocaOp::create(
        b_, Ref({}, b_.getI32Type(), tpu::MemorySpace::kSmem));
    mlir::memref::StoreOp::create(b_, Const(0), count_, mlir::ValueRange{});
    sem_ = tpu::AllocaSemaphoreOp::create(
        b_, Ref({}, tpu::DMASemaphoreType::get(b_.getContext()),
                tpu::MemorySpace::kSemaphoreMem));
    auto type = mlir::VectorType::get({1, 128}, b_.getI32Type());
    position_ = tpu::IotaOp::create(b_, type, 1);
    // Coordinates fit S32: initialize the native int64 high words to zero.
    mlir::Value num_tiles = Dim(output_, 0);
    mlir::Value high = mlir::arith::ShRSIOp::create(b_, num_tiles, Const(1));
    For(high, num_tiles, 1,
        [&](mlir::Value tile) { WriteTile(stage_[0], tile); });
  }

  // Define the Mosaic memrefs: input, private destination and aliased result.
  // The HLO wrapper provides runtime dimensions and the result/destination
  // alias.
  mlir::OwningOpRef<mlir::ModuleOp> CreateModule() {
    auto module =
        mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(b_.getLoc()));
    b_.setInsertionPointToEnd(module->getBody());
    const int64_t dyn = mlir::ShapedType::kDynamic;
    auto input = Ref({dyn, dyn}, b_.getF32Type(), tpu::MemorySpace::kHbm);
    auto output = Ref({dyn, dyn}, b_.getI32Type(), tpu::MemorySpace::kHbm);
    auto fn = mlir::func::FuncOp::create(
        b_, "main", b_.getFunctionType({input, output, output}, {}));
    auto* entry = fn.addEntryBlock();
    b_.setInsertionPointToStart(entry);
    input_ = entry->getArgument(0);
    output_ = entry->getArgument(2);  // Result aliases the private destination.
    return module;
  }
  mlir::ImplicitLocOpBuilder b_;
  mlir::Value input_, output_, tile_, stage_[2], count_, sem_, position_;
};

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> BuildNonzeroIndices(
    mlir::MLIRContext& context) {
  return NonzeroBuilder(context).Build();
}

}  // namespace torch_tpu
