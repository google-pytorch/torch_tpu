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
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace torch_tpu {

namespace {

namespace tpu = mlir::tpu;
constexpr int64_t kLanes = 128;
constexpr int64_t kSub = 8;
constexpr int64_t kTile = kLanes * kSub;
constexpr int64_t kDShift = 10;      // Distance field offset in a packed word.
constexpr int64_t kAlive = 1 << 20;  // Alive bit of a packed word.
constexpr int64_t kWindow = 16384;   // Scratch capacity, not an input limit.
constexpr int64_t kZeroRows = 16;    // Storage rows per high-word zeroing DMA.
// Shifts for the power-of-two geometry above, used to divide device values.
constexpr int64_t kSubShift = 3;    // kSub == 1 << kSubShift
constexpr int64_t kLaneShift = 7;   // kLanes == 1 << kLaneShift
constexpr int64_t kTileShift = 10;  // kTile == 1 << kTileShift

// Index phase of nonzero for a rank-two F32 input: write the (row, column)
// coordinates of every nonzero element, in input row-major order, into the
// caller's native S64[K,2] allocation. The caller already knows K, so nothing
// is returned to the host. TPU float semantics apply: NaNs and infinities are
// nonzero, subnormals are zero.
//
// This is a stream compaction. The lowering it replaces obtains one by sorting
// all M*N elements; here each nonzero's rank is computed directly and the
// element is routed to it, at a fixed number of vector operations per input
// element.
//
// Work proceeds in tiles of kTile = 1024 elements, held as an (8, 128) vector
// of sublanes by lanes and read in flat row-major order. Per tile:
//
//   1. Classify. NonzeroFlags marks the elements that are both nonzero and
//      inside the logical input, so padding introduced by the physical tiling
//      is masked here rather than trimmed later.
//
//   2. Rank. Prefix counts the zeros strictly before each element, which is
//      exactly how far left a live element must travel. Counting within a
//      128-lane row is one MXU matmul against an upper-triangular matrix;
//      carrying the row totals down the eight sublanes is three
//      rotate-and-add steps.
//
//   3. Route. CompactPositions packs each live element into one i32 word,
//
//          bit 20 | bits 19..10 | bits 9..0
//          alive  | distance    | position
//
//      then runs a ten-round butterfly: round s moves every word whose
//      distance still has bit s set exactly 2^s slots to the left and clears
//      that bit. A distance is below 1024, so ten rounds drive all of them to
//      zero, leaving each live word in the slot its rank names. A round is one
//      lane rotate, plus a sublane rotate for the lanes that cross a row
//      boundary.
//
//      Take the tile [0, a, 0, b]: a sits at position 1 behind one zero and b
//      at position 3 behind two, so round 0 moves a to slot 0 and round 1
//      moves b to slot 1.
//
//   4. Emit. Append copies the compacted coordinates into a staging plane pair
//      at the current output offset and flushes to HBM once 1024 pairs have
//      accumulated. Only this step touches HBM.
//
// Input arrives by DMA eight physical rows at a time, but compaction always
// visits logical rows in order, so the output never needs reordering. A row
// too wide for the kWindow scratch buffer is processed one window at a time.
// Input and output DMA are both synchronous at this stage.
//
// The output operand is an S32 view of the int64 allocation, shaped as
// 128-word storage rows. Two properties of that layout drive the code:
//
//   * An int64 TPU buffer is two S32 planes, all low words followed by all
//     high words. Coordinates are nonnegative and fit in a low word, so
//     ZeroHighWords clears the high plane once and nothing writes it again.
//
//   * For K >= 3, XLA lays S64[K,2] out with K along lanes, so every 128 pairs
//     occupy two consecutive storage rows: 128 row indices, then their 128
//     column indices. K = 1, 2 instead get a row-major layout with each pair
//     in its own row, which FlushSmall handles separately.
//
// Every write lands inside the allocation's ordinary tile padding, so the
// caller need not over-allocate.
//
// This C++ builds IR rather than running the algorithm: a mlir::Value is a
// device value, For and When emit device control flow, and ordinary C++ loops
// unroll fixed instruction patterns at build time.
class NonzeroBuilder {
 public:
  explicit NonzeroBuilder(mlir::MLIRContext& context)
      : b_(mlir::UnknownLoc::get(&context), &context) {
    i32_ = b_.getI32Type();
    f32_ = b_.getF32Type();
    vi32_ = mlir::VectorType::get({kSub, kLanes}, i32_);
    vf32_ = mlir::VectorType::get({kSub, kLanes}, f32_);
    ri32_ = mlir::VectorType::get({1, kLanes}, i32_);
  }

  mlir::OwningOpRef<mlir::ModuleOp> Build();

 private:
  // Small IR conveniences used by the compaction and transfer stages.
  mlir::Value Const(int64_t v) {
    return mlir::arith::ConstantIntOp::create(b_, v, 32);
  }
  mlir::Value Index(int64_t v) {
    return mlir::arith::ConstantIndexOp::create(b_, v);
  }
  mlir::Value IndexCast(mlir::Value i32) {
    return mlir::arith::IndexCastOp::create(b_, b_.getIndexType(), i32);
  }
  mlir::Value Add(mlir::Value a, mlir::Value c) {
    return mlir::arith::AddIOp::create(b_, a, c);
  }
  mlir::Value Sub(mlir::Value a, mlir::Value c) {
    return mlir::arith::SubIOp::create(b_, a, c);
  }
  mlir::Value Mul(mlir::Value a, mlir::Value c) {
    return mlir::arith::MulIOp::create(b_, a, c);
  }
  mlir::Value And(mlir::Value a, mlir::Value c) {
    return mlir::arith::AndIOp::create(b_, a, c);
  }
  mlir::Value Or(mlir::Value a, mlir::Value c) {
    return mlir::arith::OrIOp::create(b_, a, c);
  }
  mlir::Value ShiftLeft(mlir::Value a, mlir::Value c) {
    return mlir::arith::ShLIOp::create(b_, a, c);
  }
  mlir::Value ShiftRight(mlir::Value a, mlir::Value c) {
    return mlir::arith::ShRSIOp::create(b_, a, c);
  }
  mlir::Value Compare(mlir::arith::CmpIPredicate p, mlir::Value a,
                      mlir::Value c) {
    return mlir::arith::CmpIOp::create(b_, p, a, c);
  }
  mlir::Value Select(mlir::Value c, mlir::Value t, mlir::Value f) {
    return mlir::arith::SelectOp::create(b_, c, t, f);
  }
  // Materialize a constant vector; Broadcast instead expands a runtime value.
  mlir::Value Splat(int64_t v, mlir::VectorType type = nullptr) {
    type = type ? type : vi32_;
    mlir::Type elt = type.getElementType();
    mlir::Attribute value = mlir::isa<mlir::FloatType>(elt)
                                ? mlir::Attribute(b_.getFloatAttr(elt, v))
                                : mlir::Attribute(b_.getIntegerAttr(elt, v));
    return mlir::arith::ConstantOp::create(
        b_, type, mlir::DenseElementsAttr::get(type, value));
  }
  // Expand a scalar or singleton dimension to the chosen vector shape.
  mlir::Value Broadcast(mlir::Value scalar, mlir::VectorType type = nullptr) {
    return mlir::vector::BroadcastOp::create(b_, type ? type : vi32_, scalar);
  }
  // Lane coordinates for dim=1, sublane coordinates for dim=0.
  mlir::Value Iota(int dim, mlir::VectorType type = nullptr) {
    return tpu::IotaOp::create(b_, type ? type : vi32_,
                               b_.getDenseI32ArrayAttr({dim}));
  }
  // Tells the compiler an offset is a multiple of `m` (it cannot prove it).
  mlir::Value AssumeMultipleOf(mlir::Value v, int64_t m) {
    return tpu::AssumeMultipleOp::create(b_, i32_, v, m);
  }
  // Cyclic permutation along sublanes (dim=0) or lanes (dim=1).
  mlir::Value Rotate(mlir::Value v, mlir::Value amount, int dim) {
    return tpu::DynamicRotateOp::create(
        b_, v.getType(), v, amount, b_.getSI32IntegerAttr(dim),
        mlir::IntegerAttr(), mlir::IntegerAttr());
  }
  mlir::Value Rotate(mlir::Value v, int64_t amount, int dim) {
    return Rotate(v, Const(amount), dim);
  }
  // Read a fixed vector element into a scalar device register.
  mlir::Value Extract(mlir::Value v, int64_t row, int64_t lane) {
    return mlir::vector::ExtractOp::create(b_, v,
                                           llvm::ArrayRef<int64_t>{row, lane});
  }
  mlir::Value RowOf(mlir::Value v, int64_t r) {  // (8,128) -> (1,128)
    mlir::Value flat =
        mlir::vector::ExtractOp::create(b_, v, llvm::ArrayRef<int64_t>{r});
    return mlir::vector::ShapeCastOp::create(b_, ri32_, flat);
  }
  // Masked VMEM store: inactive elements retain their previous staged values.
  void Store(mlir::Value buffer, std::vector<mlir::Value> idx, mlir::Value v,
             mlir::Value mask = nullptr) {
    tpu::VectorStoreOp::create(b_, v, buffer, idx, mask, false);
  }
  // Load a vector from VMEM at logical indices; this does not issue HBM DMA.
  mlir::Value Load(mlir::VectorType type, mlir::Value buffer,
                   std::vector<mlir::Value> idx) {
    return mlir::vector::LoadOp::create(b_, type, buffer, idx);
  }
  // Allocate device scratch in the memory space encoded by the memref type.
  mlir::Value Allocate(mlir::MemRefType type) {
    return mlir::memref::AllocaOp::create(b_, type);
  }
  // SMEM holds the running coordinate and flush counters as one-element
  // buffers.
  mlir::Value LoadScalar(mlir::Value ref) {
    return mlir::memref::LoadOp::create(b_, ref, Index(0));
  }
  void StoreScalar(mlir::Value ref, mlir::Value v) {
    mlir::memref::StoreOp::create(b_, v, ref, Index(0));
  }
  // Describe a buffer type and memory space without allocating storage.
  mlir::MemRefType Ref(llvm::ArrayRef<int64_t> shape, mlir::Type elt,
                       tpu::MemorySpace space) {
    return mlir::MemRefType::get(
        shape, elt, nullptr, tpu::MemorySpaceAttr::get(b_.getContext(), space));
  }
  // A scf.for over [lo, hi) with i32 bounds, body emitted by `body(iv)`.
  void For(mlir::Value lo, mlir::Value hi,
           llvm::function_ref<void(mlir::Value)> body, int64_t step = 1) {
    auto op = mlir::scf::ForOp::create(b_, lo, hi, Const(step));
    mlir::OpBuilder::InsertionGuard guard(b_);
    b_.setInsertionPointToStart(op.getBody());
    body(op.getInductionVar());
  }
  // Emit a device conditional; the C++ callback builds its body once.
  void When(mlir::Value cond, llvm::function_ref<void()> body) {
    auto op = mlir::scf::IfOp::create(b_, cond, /*withElseRegion=*/false);
    mlir::OpBuilder::InsertionGuard guard(b_);
    b_.setInsertionPointToStart(&op.getThenRegion().front());
    body();
  }
  // Enqueue one transfer. Its source must stay unchanged until Wait completes.
  void Dma(mlir::Value src, mlir::Value dst, mlir::Value sem) {
    tpu::EnqueueDMAOp::create(b_, src, mlir::Value(), dst, sem, mlir::Value(),
                              mlir::Value(), mlir::Value());
  }
  // Consume one completed DMA notification from this semaphore.
  void Wait(mlir::Value sem) {
    tpu::SemaphoreWaitOp::create(b_, sem, Const(1));
  }
  // Make an offset view into an existing buffer; no allocation or data
  // movement.
  mlir::Value Slice(mlir::MemRefType type, mlir::Value ref,
                    std::vector<mlir::Value> base,
                    std::vector<mlir::Value> sizes = {}) {
    return tpu::MemRefSliceOp::create(b_, type, ref, base, sizes);
  }
  // View rows of the S32 output storage; q counts 128-word storage rows.
  mlir::Value OutputTile(mlir::Value q, int64_t rows = 1) {
    return Slice(Ref({rows, kLanes}, i32_, tpu::MemorySpace::kHbm), output_,
                 {q, Const(0)});
  }
  // Synchronously write one storage row, used for zeroing and partial tails.
  void WriteRow(mlir::Value tile, mlir::Value q) {
    Dma(tile, OutputTile(q), out_sem_);
    Wait(out_sem_);
  }
  void ZeroHighWords();
  void FlushSmall(mlir::Value count);
  void ProcessWideRows();
  void Window(mlir::Value base, mlir::Value width);

  // One compacted input segment: packed positions w, live count total, and its
  // logical row and starting column. These are device values, not host data.
  struct Stream {
    mlir::Value positions, total, row, col0;
  };
  // Zeros strictly before each live element, and the tile's live count.
  struct Ranks {
    mlir::Value distance;
    mlir::Value total;
  };
  Ranks Prefix(mlir::Value z, mlir::Value m);
  mlir::Value NonzeroFlags(mlir::Value x, mlir::Value row, mlir::Value col,
                           mlir::Value n, mlir::Value rows);
  mlir::Value CompactPositions(mlir::Value flags, mlir::Value distance);
  Stream CompactTile(mlir::Value buffer, mlir::Value t,
                     mlir::Value row_in_block, mlir::Value row, mlir::Value n,
                     mlir::Value rows_total);
  void Append(const Stream& stream);
  void Flush(mlir::Value q);

  mlir::OwningOpRef<mlir::ModuleOp> CreateModule();
  void Initialize();
  mlir::Value Triangle();
  void Geometry();
  void Fetch(mlir::Value rb, mlir::Value slot);
  void WaitInput(mlir::Value slot);
  void ProcessBlock(mlir::Value slot, mlir::Value r0);
  void ProcessInput();
  void FinishOutput();
  mlir::Value x_;  // HBM input memref
  // Runtime input geometry, read once by Geometry().
  mlir::Value m_, n_;
  mlir::Value row_blocks_;  // Eight-row input blocks covering m_
  // Current scratch window, recomputed by Window().
  mlir::Value window_;       // Logical column origin
  mlir::Value n_pad_;        // Width rounded up to whole 128-lane pieces
  mlir::Value tiles_;        // 1024-column compaction steps
  mlir::Value wide_;         // 1024-column DMA pieces
  mlir::Value narrow_from_;  // First column covered by 128-column pieces
  mlir::Value narrow_;       // 128-column DMA pieces
  mlir::Value in_sem_;       // Input DMA completion semaphore

  mlir::ImplicitLocOpBuilder b_;
  mlir::Type i32_, f32_;
  mlir::VectorType vi32_, vf32_, ri32_;
  // Kernel state.
  mlir::Value output_;    // HBM view of native int64 storage
  mlir::Value packed_;    // VMEM interleaved coordinate tiles
  mlir::Value buf_;       // VMEM (1, 8, kWindow) f32
  mlir::Value stage_[2];  // VMEM (16, 128) i32 per plane
  mlir::Value out_sem_;
  mlir::Value fill_;  // Total appended, not just staged, coordinates
  mlir::Value lane_, subl_, pos_, tri_;  // constant vectors
};

// Given zero flags z and live flags m, return each live element's left-shift
// distance (zeros strictly before it) and the number of live elements in the
// tile.
NonzeroBuilder::Ranks NonzeroBuilder::Prefix(mlir::Value z, mlir::Value m) {
  // Zeros before each element in flat order: within the row from the MXU,
  // in the rows above from three sublane rolls.
  mlir::Value zb = mlir::arith::SIToFPOp::create(
      b_, mlir::VectorType::get({kSub, kLanes}, b_.getBF16Type()), z);
  mlir::Value accf = tpu::MatmulOp::create(
      b_, b_.getLoc(), vf32_, zb, tri_, Splat(0, vf32_), false, false,
      tpu::ContractPrecisionAttr(),
      tpu::defaultDimensionNumbers(b_, false, false), false);
  mlir::Value acc = mlir::arith::FPToSIOp::create(b_, vi32_, accf);
  mlir::Value last = mlir::vector::ExtractStridedSliceOp::create(
      b_, acc, llvm::ArrayRef<int64_t>{0, kLanes - 1},
      llvm::ArrayRef<int64_t>{kSub, 1}, llvm::ArrayRef<int64_t>{1, 1});
  mlir::Value row_zeros = Broadcast(last);
  mlir::Value above = row_zeros;
  for (int s = 0; s < kSubShift; ++s) {
    const int64_t st = 1 << s;
    mlir::Value rolled = Rotate(above, st, 0);
    above =
        Add(above,
            Select(Compare(mlir::arith::CmpIPredicate::sge, subl_, Splat(st)),
                   rolled, Splat(0)));
  }
  mlir::Value d = Mul(Add(Sub(acc, z), Sub(above, row_zeros)), m);
  return {d, Sub(Const(kTile), Extract(above, kSub - 1, 0))};
}

// Load one 1024-column segment from a fetched row and turn it into an ordered
// stream of live positions. col0 includes the current scratch-window offset.
NonzeroBuilder::Stream NonzeroBuilder::CompactTile(
    mlir::Value buffer, mlir::Value t, mlir::Value row_in_block,
    mlir::Value row, mlir::Value n, mlir::Value rows_total) {
  mlir::Value col0 = AssumeMultipleOf(ShiftLeft(t, Const(kTileShift)), kTile);
  // 1024 consecutive elements of the row, as an (8, 128) tile.
  mlir::Value slab =
      Load(mlir::VectorType::get({1, kTile}, f32_), buf_,
           {IndexCast(buffer), IndexCast(row_in_block), IndexCast(col0)});
  mlir::Value x = mlir::vector::ShapeCastOp::create(b_, vf32_, slab);
  col0 = Add(col0, window_);
  mlir::Value flags = NonzeroFlags(x, row, col0, n, rows_total);
  auto [distance, total] = Prefix(Sub(Splat(1), flags), flags);
  return Stream{CompactPositions(flags, distance), total, row, col0};
}

// Classify valid logical elements, independently of the physical input tiles.
mlir::Value NonzeroBuilder::NonzeroFlags(mlir::Value x, mlir::Value row,
                                         mlir::Value col, mlir::Value n,
                                         mlir::Value rows) {
  mlir::Value cols = Add(Broadcast(col), pos_);
  // Match TPU nonzero semantics: include NaNs, flush subnormals to zero.
  mlir::Value bits = mlir::arith::BitcastOp::create(b_, vi32_, x);
  mlir::Value nonzero = Compare(mlir::arith::CmpIPredicate::sgt,
                                And(bits, Splat(0x7fffffff)), Splat(0));
  mlir::Value in_row =
      Compare(mlir::arith::CmpIPredicate::slt, cols, Broadcast(n));
  mlir::Value in_rows =
      Compare(mlir::arith::CmpIPredicate::slt, Broadcast(row), Broadcast(rows));
  return Select(And(And(nonzero, in_row), in_rows), Splat(1), Splat(0));
}

// Pack each live element's position and required left shift, then apply the
// shift one bit at a time. Surviving positions retain their original order.
mlir::Value NonzeroBuilder::CompactPositions(mlir::Value flags,
                                             mlir::Value distance) {
  mlir::Value word =
      Select(Compare(mlir::arith::CmpIPredicate::eq, flags, Splat(1)),
             Or(Or(pos_, ShiftLeft(distance, Splat(kDShift))), Splat(kAlive)),
             Splat(0));
  for (int s = 0; s < kTileShift; ++s) {
    const int64_t bit = int64_t{1} << (kDShift + s);
    mlir::Value src;
    if (s < kLaneShift) {  // left by st lanes; lanes past the end come from the
                           // next row
      const int64_t st = 1 << s;
      mlir::Value same = Rotate(word, kLanes - st, 1);
      src = Select(
          Compare(mlir::arith::CmpIPredicate::slt, lane_, Splat(kLanes - st)),
          same, Rotate(same, kSub - 1, 0));
    } else {  // up by st rows
      const int64_t st = 1 << (s - kLaneShift);
      src = Select(
          Compare(mlir::arith::CmpIPredicate::slt, subl_, Splat(kSub - st)),
          Rotate(word, kSub - st, 0), Splat(0));
    }
    mlir::Value arriving =
        Compare(mlir::arith::CmpIPredicate::ne, And(src, Splat(bit)), Splat(0));
    mlir::Value leaving = Compare(mlir::arith::CmpIPredicate::ne,
                                  And(word, Splat(bit)), Splat(0));
    word = Select(arriving, And(src, Splat(~bit)),
                  Select(leaving, Splat(0), word));
  }
  return word;
}

// Merge up to 1024 coordinates after the pending tail. fill_ counts all pairs
// appended so far; fill_ % 1024 is the current stage occupancy. The second
// half of each staging plane holds overflow until the first half is flushed.
void NonzeroBuilder::Append(const Stream& stream) {
  // Element q of the stream goes to output row (fill+q)//128, lane
  // (fill+q)%128.
  mlir::Value fill = LoadScalar(fill_);
  mlir::Value lane_offset = And(fill, Const(kLanes - 1));
  mlir::Value output_row = ShiftRight(fill, Const(kLaneShift));
  mlir::Value stage_row = And(output_row, Const(kSub - 1));
  mlir::Value rotated = Rotate(stream.positions, lane_offset, 1);
  mlir::Value above = Rotate(rotated, 1, 0);
  // Rows output_row .. output_row+7: wrapped lanes come from the row above; in
  // row 0 they would be the last row's wrap, which belongs to the tail row.
  mlir::Value body = Select(
      Compare(mlir::arith::CmpIPredicate::sge, lane_, Broadcast(lane_offset)),
      rotated,
      Select(Compare(mlir::arith::CmpIPredicate::sgt, subl_, Splat(0)), above,
             Splat(0)));
  // Row output_row+8, holding the lanes that wrapped past the last row.
  mlir::Value tail =
      Select(Compare(mlir::arith::CmpIPredicate::slt, Iota(1, ri32_),
                     Broadcast(lane_offset, ri32_)),
             RowOf(rotated, kSub - 1), Splat(0, ri32_));
  mlir::Value valid = Compare(mlir::arith::CmpIPredicate::ne,
                              And(body, Splat(kAlive)), Splat(0));
  mlir::Value valid_tail =
      Compare(mlir::arith::CmpIPredicate::ne, And(tail, Splat(kAlive, ri32_)),
              Splat(0, ri32_));
  mlir::Value cols = Add(Broadcast(stream.col0), And(body, Splat(kTile - 1)));
  mlir::Value cols_tail =
      Add(Broadcast(stream.col0, ri32_), And(tail, Splat(kTile - 1, ri32_)));
  Store(stage_[0], {IndexCast(stage_row), Index(0)}, Broadcast(stream.row),
        valid);
  Store(stage_[1], {IndexCast(stage_row), Index(0)}, cols, valid);
  mlir::Value overflow_row = IndexCast(Add(stage_row, Const(kSub)));
  Store(stage_[0], {overflow_row, Index(0)}, Broadcast(stream.row, ri32_),
        valid_tail);
  Store(stage_[1], {overflow_row, Index(0)}, cols_tail, valid_tail);
  mlir::Value new_fill = Add(fill, stream.total);
  StoreScalar(fill_, new_fill);
  // A complete 8x128 tile of output: rows q..q+7, q = kSub * (fill / kTile).
  When(
      Compare(mlir::arith::CmpIPredicate::sgt,
              ShiftRight(new_fill, Const(kTileShift)),
              ShiftRight(fill, Const(kTileShift))),
      [&] {
        Flush(ShiftLeft(ShiftRight(fill, Const(kTileShift)), Const(kSubShift)));
      });
}

// Interleave row/column low-word tiles into the native allocation order.
// A full stage writes exactly 1024 pairs; keep its unfinished tail in VMEM.
void NonzeroBuilder::Flush(mlir::Value q) {
  for (int r = 0; r < kSub; ++r) {
    for (int p = 0; p < 2; ++p) {
      mlir::Value row = Load(ri32_, stage_[p], {Index(r), Index(0)});
      Store(packed_, {Index(2 * r + p), Index(0)}, row);
    }
  }
  Dma(packed_, OutputTile(Mul(q, Const(2)), 2 * kSub), out_sem_);
  Wait(out_sem_);
  for (int p = 0; p < 2; ++p) {
    mlir::Value next = Load(vi32_, stage_[p], {Index(kSub), Index(0)});
    Store(stage_[p], {Index(0), Index(0)}, next);
    Store(stage_[p], {Index(kSub), Index(0)}, Splat(0));
  }
}

// Coordinates are nonnegative S32 values. Zero the high-word half of native
// int64 storage, using 16-row DMAs plus single-row tails to stay in bounds.
void NonzeroBuilder::ZeroHighWords() {
  mlir::Value rows = mlir::arith::IndexCastOp::create(
      b_, i32_, mlir::memref::DimOp::create(b_, output_, 0));
  mlir::Value half = ShiftRight(rows, Const(1));
  mlir::Value end = Sub(rows, And(half, Const(kZeroRows - 1)));
  // Borrow the flush buffer as a zero source; it runs before any coordinate.
  Store(packed_, {Index(0), Index(0)}, Splat(0));
  Store(packed_, {Index(kSub), Index(0)}, Splat(0));
  For(
      half, end,
      [&](mlir::Value q) {
        Dma(packed_, OutputTile(q, kZeroRows), out_sem_);
        Wait(out_sem_);
      },
      kZeroRows);
  mlir::Value zero = Slice(Ref({1, kLanes}, i32_, tpu::MemorySpace::kVmem),
                           packed_, {Const(0), Const(0)});
  For(end, rows, [&](mlir::Value q) { WriteRow(zero, q); });
}

// Repack K=1,2 as native row-major pairs; K=0 naturally performs no writes.
void NonzeroBuilder::FlushSmall(mlir::Value count) {
  // Default S64[1,2] and S64[2,2] layouts store each coordinate pair in a row.
  // Flush() never ran for such small K, so packed_ is free as row scratch.
  For(Const(0), count, [&](mlir::Value k) {
    auto scalar = [&](mlir::Value stage) {
      mlir::Value values = Load(ri32_, stage, {Index(0), Index(0)});
      mlir::Value gathered = tpu::DynamicGatherOp::create(
          b_, ri32_, values, Broadcast(k, ri32_), b_.getDenseI32ArrayAttr({1}));
      return Extract(gathered, 0, 0);
    };
    mlir::Value row = Broadcast(scalar(stage_[0]), ri32_);
    mlir::Value col = Broadcast(scalar(stage_[1]), ri32_);
    Store(packed_, {Index(0), Index(0)},
          Select(Compare(mlir::arith::CmpIPredicate::eq, Iota(1, ri32_),
                         Splat(0, ri32_)),
                 row, col));
    mlir::Value tile = Slice(Ref({1, kLanes}, i32_, tpu::MemorySpace::kVmem),
                             packed_, {Const(0), Const(0)});
    WriteRow(tile, k);
  });
}

// Emit the complete kernel: initialize scratch, visit input, then drain output.
mlir::OwningOpRef<mlir::ModuleOp> NonzeroBuilder::Build() {
  auto module = CreateModule();
  Initialize();
  Geometry();
  ZeroHighWords();
  ProcessInput();
  FinishOutput();
  mlir::func::ReturnOp::create(b_);
  return module;
}

// Use row blocks when a row fits scratch; otherwise finish each row in windows.
void NonzeroBuilder::ProcessInput() {
  auto wide = mlir::scf::IfOp::create(
      b_, Compare(mlir::arith::CmpIPredicate::sgt, n_, Const(kWindow)), true);
  {
    mlir::OpBuilder::InsertionGuard guard(b_);
    b_.setInsertionPointToStart(&wide.getElseRegion().front());
    For(Const(0), row_blocks_, [&](mlir::Value rb) {
      Fetch(rb, Const(0));
      WaitInput(Const(0));
      ProcessBlock(Const(0), ShiftLeft(rb, Const(kSubShift)));
    });
  }
  {
    mlir::OpBuilder::InsertionGuard guard(b_);
    b_.setInsertionPointToStart(&wide.getThenRegion().front());
    ProcessWideRows();
  }
}

// Mosaic sees input, private destination, and aliased result memrefs. The HLO
// wrapper supplies runtime extents separately; only the result view is written.
mlir::OwningOpRef<mlir::ModuleOp> NonzeroBuilder::CreateModule() {
  auto space = [&](tpu::MemorySpace s) {
    return tpu::MemorySpaceAttr::get(b_.getContext(), s);
  };
  const int64_t dyn = mlir::ShapedType::kDynamic;
  const auto x_type = mlir::MemRefType::get({dyn, dyn}, f32_, nullptr,
                                            space(tpu::MemorySpace::kHbm));
  const auto output_type = mlir::MemRefType::get({dyn, dyn}, i32_, nullptr,
                                                 space(tpu::MemorySpace::kHbm));

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(b_.getLoc());
  b_.setInsertionPointToEnd(module->getBody());
  auto function = mlir::func::FuncOp::create(
      b_, "main", b_.getFunctionType({x_type, output_type, output_type}, {}));
  mlir::Block* entry = function.addEntryBlock();
  b_.setInsertionPointToStart(entry);
  x_ = entry->getArgument(0);
  output_ = entry->getArgument(2);

  return module;
}

// Build T[i,j] = (i <= j), so z*T gives inclusive zero counts along each
// 128-lane row. Inputs are 0/1 BF16 and accumulation is F32, so counts are
// exact.
mlir::Value NonzeroBuilder::Triangle() {
  const auto sq_i32 = mlir::VectorType::get({kLanes, kLanes}, i32_);
  const auto sq_f32 = mlir::VectorType::get({kLanes, kLanes}, f32_);
  const auto sq_bf16 =
      mlir::VectorType::get({kLanes, kLanes}, b_.getBF16Type());
  mlir::Value le = Compare(mlir::arith::CmpIPredicate::sle, Iota(0, sq_i32),
                           Iota(1, sq_i32));
  mlir::Value one = mlir::arith::ConstantOp::create(
      b_, sq_f32,
      mlir::DenseElementsAttr::get(sq_f32, b_.getF32FloatAttr(1.0f)));
  mlir::Value zero = mlir::arith::ConstantOp::create(
      b_, sq_f32,
      mlir::DenseElementsAttr::get(sq_f32, b_.getF32FloatAttr(0.0f)));
  return mlir::arith::TruncFOp::create(b_, sq_bf16, Select(le, one, zero));
}

// Read runtime M,N and derive the number of eight-row input blocks.
void NonzeroBuilder::Geometry() {
  auto dim = [&](mlir::Value ref, int64_t d) {
    return mlir::arith::IndexCastOp::create(
        b_, i32_, mlir::memref::DimOp::create(b_, ref, d));
  };
  m_ = dim(x_, 0);
  n_ = dim(x_, 1);
  row_blocks_ = ShiftRight(Add(m_, Const(kSub - 1)), Const(kSubShift));
  Window(Const(0), n_);
}

// Set the logical column origin and transfer geometry for one scratch window.
// wide_ counts 1024-column DMA pieces; narrow_ counts the remaining 128-column
// pieces. Round only physical transfers up; NonzeroFlags masks logical tails.
void NonzeroBuilder::Window(mlir::Value base, mlir::Value width) {
  window_ = base;
  n_pad_ = AssumeMultipleOf(
      And(Add(width, Const(kLanes - 1)), Const(~(kLanes - 1))), kLanes);
  tiles_ = ShiftRight(Add(n_pad_, Const(kTile - 1)), Const(kTileShift));
  wide_ = ShiftRight(n_pad_, Const(kTileShift));
  narrow_from_ = AssumeMultipleOf(ShiftLeft(wide_, Const(kTileShift)), kTile);
  narrow_ = ShiftRight(Sub(n_pad_, narrow_from_), Const(kLaneShift));
}

// Allocate VMEM input/output staging, SMEM counters and DMA semaphores. Zero
// staging so unused packed positions start inactive; prebuild vector indices.
void NonzeroBuilder::Initialize() {
  auto dma_sem = Ref({}, tpu::SemaphoreType::get(b_.getContext()),
                     tpu::MemorySpace::kSemaphoreMem);

  packed_ = Allocate(Ref({2 * kSub, kLanes}, i32_, tpu::MemorySpace::kVmem));
  buf_ = Allocate(Ref({1, kSub, kWindow}, f32_, tpu::MemorySpace::kVmem));
  for (int p = 0; p < 2; ++p) {
    stage_[p] =
        Allocate(Ref({2 * kSub, kLanes}, i32_, tpu::MemorySpace::kVmem));
  }
  out_sem_ = tpu::AllocaSemaphoreOp::create(b_, dma_sem);
  in_sem_ = tpu::AllocaSemaphoreOp::create(b_, dma_sem);
  fill_ = Allocate(Ref({1}, i32_, tpu::MemorySpace::kSmem));
  StoreScalar(fill_, Const(0));
  for (int p = 0; p < 2; ++p) {
    Store(stage_[p], {Index(0), Index(0)}, Splat(0));
    Store(stage_[p], {Index(kSub), Index(0)}, Splat(0));
  }
  lane_ = Iota(1);
  subl_ = Iota(0);
  pos_ = Add(Mul(subl_, Splat(kLanes)), lane_);
  tri_ = Triangle();
}

// Enqueue an eight-row block into the selected scratch slot using aligned
// 1024- and 128-column pieces. Call WaitInput before reading that slot.
void NonzeroBuilder::Fetch(mlir::Value rb, mlir::Value slot) {
  mlir::Value r0 = AssumeMultipleOf(ShiftLeft(rb, Const(kSubShift)), kSub);
  auto piece = [&](int64_t width, mlir::Value col) {
    mlir::Value src = Slice(Ref({kSub, width}, f32_, tpu::MemorySpace::kHbm),
                            x_, {r0, Add(window_, col)});
    mlir::Value dst = tpu::MemRefSqueezeOp::create(
        b_, Ref({kSub, width}, f32_, tpu::MemorySpace::kVmem),
        Slice(Ref({1, kSub, width}, f32_, tpu::MemorySpace::kVmem), buf_,
              {slot, Const(0), col}));
    Dma(src, dst, in_sem_);
  };
  For(Const(0), wide_, [&](mlir::Value t) {
    piece(kTile, AssumeMultipleOf(ShiftLeft(t, Const(kTileShift)), kTile));
  });
  For(
      narrow_from_, n_pad_,
      [&](mlir::Value col) { piece(kLanes, AssumeMultipleOf(col, kLanes)); },
      kLanes);
}

// Wait once per piece enqueued by Fetch; only then may compaction read the
// slot.
void NonzeroBuilder::WaitInput(mlir::Value slot) {
  For(Const(0), Add(wide_, narrow_), [&](mlir::Value) { Wait(in_sem_); });
}

// Consume rows in order, independent of physical input tiling.
void NonzeroBuilder::ProcessBlock(mlir::Value slot, mlir::Value r0) {
  For(Const(0), Const(kSub), [&](mlir::Value r) {
    For(Const(0), tiles_, [&](mlir::Value t) {
      Append(CompactTile(slot, t, r, Add(r0, r), n_, m_));
    });
  });
}

// Beyond the scratch width, finish each logical row window by window.
// This preserves row-major output order without an artificial width limit.
void NonzeroBuilder::ProcessWideRows() {
  For(Const(0), m_, [&](mlir::Value row) {
    For(
        Const(0), n_,
        [&](mlir::Value base) {
          Window(base, mlir::arith::MinSIOp::create(b_, Const(kWindow),
                                                    Sub(n_, base)));
          Fetch(ShiftRight(row, Const(kSubShift)), Const(0));
          WaitInput(Const(0));
          For(Const(0), tiles_, [&](mlir::Value t) {
            Append(CompactTile(Const(0), t, And(row, Const(kSub - 1)), row, n_,
                               m_));
          });
        },
        kWindow);
  });
}

// Finish any pending native output tiles, handling small K separately.
// A partial last tile may touch ordinary allocation padding, never extra slack.
void NonzeroBuilder::FinishOutput() {
  mlir::Value count = LoadScalar(fill_);
  When(Compare(mlir::arith::CmpIPredicate::slt, count, Const(3)),
       [&] { FlushSmall(count); });
  When(Compare(mlir::arith::CmpIPredicate::sge, count, Const(3)), [&] {
    mlir::Value q =
        ShiftLeft(ShiftRight(count, Const(kTileShift)), Const(kSubShift));
    mlir::Value remaining = And(count, Const(kTile - 1));
    mlir::Value rows =
        ShiftRight(Add(remaining, Const(kLanes - 1)), Const(kLaneShift));
    For(Const(0), rows, [&](mlir::Value r) {
      for (int p = 0; p < 2; ++p) {
        mlir::Value tile =
            Slice(Ref({1, kLanes}, i32_, tpu::MemorySpace::kVmem), stage_[p],
                  {r, Const(0)});
        WriteRow(tile, Add(Mul(Add(q, r), Const(2)), Const(p)));
      }
    });
  });
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> BuildNonzeroIndices(
    mlir::MLIRContext& context) {
  return NonzeroBuilder(context).Build();
}

}  // namespace torch_tpu
