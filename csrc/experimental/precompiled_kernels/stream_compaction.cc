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

#include "csrc/experimental/precompiled_kernels/stream_compaction.h"

#include <bit>
#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "csrc/experimental/precompiled_kernels/kernel_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"

namespace torch_tpu {
namespace {

constexpr int64_t kZeroRows = 16;  // Storage rows per high-word zeroing DMA.

// Shifts an (8, 128) tile left by `stride` lanes, pulling wrapped lanes from
// the next sublane.
kiln::Tile ShiftLanesLeft(const kiln::Tile tile, const int64_t stride,
                          const kiln::Tile keep) {
  const kiln::Tile same = tile.Rotate(kLanes - stride, 1);
  return Select(keep, same, same.Rotate(kSublanes - 1, 0));
}

}  // namespace

TileWindow MakeTileWindow(const kiln::Scalar base, const kiln::Scalar width,
                          const int64_t granule) {
  ABSL_CHECK(granule > 0 &&  // CRASH_OK=compile-time kernel construction bug
             std::has_single_bit(static_cast<uint64_t>(granule)))
      << "The window granule must be a positive power of two. This is a "
         "TorchTPU bug.";
  // TODO(b/562066667): Drop MultipleOf once the OSS jaxlib proves `& -granule`.
  const kiln::Scalar n_pad =
      kiln::MultipleOf((width + (granule - 1)) & -granule, granule);
  return {base, n_pad, (n_pad + (kTile - 1)) >> kTileShift};
}

kiln::Tile PrefixTriangle() {
  const kiln::Tile i = kiln::Iota(kiln::i32(kLanes, kLanes), 0);
  const kiln::Tile j = kiln::Iota(kiln::i32(kLanes, kLanes), 1);
  const kiln::Tile one = kiln::Full(kiln::bf16(kLanes, kLanes), 1.0);
  return Select(i <= j, one, 0.0);
}

CompactionRanks ComputePrefixRanks(const kiln::Tile live_flags,
                                   const kiln::Tile sublane,
                                   const kiln::Tile tri) {
  const kiln::Tile dead_flags = 1 - live_flags;
  // Inclusive in-row prefix sums of the dead flags.
  const kiln::Tile acc = kiln::Dot(dead_flags.ToBF16(), tri).ToI32();
  const kiln::Tile row_dead =
      acc.Slice({0, kLanes - 1}, {kSublanes, 1}).Lanes(kLanes);
  // Inclusive prefix sums of the per-row dead counts over sublanes.
  kiln::Tile above = row_dead;
  for (int s = 0; s < kSublaneShift; ++s) {
    const int64_t st = int64_t{1} << s;
    above += Select(sublane >= st, above.Rotate(st, 0), 0);
  }
  return {(acc - dead_flags + (above - row_dead)) * live_flags,
          kTile - above.Extract({kSublanes - 1, 0})};
}

kiln::Tile CompactTile(kiln::Tile control, const int distance_shift,
                       const kiln::Tile lane, const kiln::Tile sublane,
                       const absl::Span<kiln::Tile* const> payloads) {
  ABSL_CHECK(  // CRASH_OK=compile-time kernel construction bug
      distance_shift >= 0 && distance_shift + kTileShift <= 32)
      << "The 10-bit distance field must fit in 32 bits. This is a TorchTPU "
         "bug.";
  for (int s = 0; s < kTileShift; ++s) {
    const int64_t bit = int64_t{1} << (distance_shift + s);
    const bool lane_stage = s < kLaneShift;
    const int64_t stride = int64_t{1} << (lane_stage ? s : s - kLaneShift);
    const kiln::Tile keep =
        lane_stage ? (lane < kLanes - stride) : (sublane < kSublanes - stride);
    const kiln::Tile src_control =
        lane_stage ? ShiftLanesLeft(control, stride, keep)
                   : Select(keep, control.Rotate(kSublanes - stride, 0), 0);
    const kiln::Tile arriving = (src_control & bit) != 0;
    const kiln::Tile leaving = (control & bit) != 0;
    for (kiln::Tile* payload : payloads) {
      const kiln::Tile src = lane_stage
                                 ? ShiftLanesLeft(*payload, stride, keep)
                                 : payload->Rotate(kSublanes - stride, 0);
      *payload = Select(arriving, src, *payload);
    }
    control = Select(arriving, src_control & ~bit, Select(leaving, 0, control));
  }
  return control;
}

void ZeroHighWords(const kiln::HbmRef output) {
  const kiln::VmemRef zeros = kiln::AllocaVmem(kiln::i32(kZeroRows, kLanes));
  kiln::Store(zeros, kiln::Zeros(kiln::i32(kZeroRows, kLanes)));
  const kiln::Scalar rows = output.Dim(0);
  const kiln::Scalar half = rows >> 1;
  // 16-row DMAs over [half, end), then 1-row DMAs over [end, rows).
  const kiln::Scalar end = rows - (half & (kZeroRows - 1));
  kiln::For(half, end, kZeroRows, [&output, &zeros](kiln::Scalar q) {
    kiln::SyncCopy(zeros,
                   output(kiln::Slice(q, kZeroRows), kiln::Slice(0, kLanes)));
  });
  kiln::For(end, rows, [&output, &zeros](kiln::Scalar q) {
    kiln::SyncCopy(zeros(kiln::Slice(0, 1)),
                   output(kiln::Slice(q, 1), kiln::Slice(0, kLanes)));
  });
}

StagedOutput::StagedOutput(const kiln::HbmRef output, const int64_t planes,
                           const int64_t tile_rows, const kiln::Scalar max_rows)
    : output_(output),
      planes_(planes),
      tile_rows_(tile_rows),
      max_groups_(max_rows >>
                  std::countr_zero(static_cast<uint64_t>(tile_rows))),
      stage_(kiln::AllocaVmem(kiln::i32(planes, 2 * kSublanes, kLanes))),
      packed_(
          kiln::AllocaVmem(kiln::i32(kSlots, tile_rows * kSublanes, kLanes))),
      fill_(kiln::AllocaSmem(kiln::i32())),
      flush_count_(kiln::AllocaSmem(kiln::i32())),
      flush_sems_(kiln::AllocaDmaSemaphore({kSlots})),
      lane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 1)),
      pos_(kiln::Iota(kiln::i32(kSublanes, kLanes), 0) * kLanes + lane_),
      row_lane_(kiln::Iota(kiln::i32(1, kLanes), 1)) {
  ABSL_CHECK(tile_rows > 0 &&  // CRASH_OK=compile-time kernel construction bug
             std::has_single_bit(static_cast<uint64_t>(tile_rows)))
      << "The storage rows per 128-element group must be a positive power of "
         "two. This is a TorchTPU bug.";
  ABSL_CHECK(planes > 0 &&  // CRASH_OK=compile-time kernel construction bug
             planes <= tile_rows)
      << "Each element must emit between 1 and `tile_rows` planes. This is a "
         "TorchTPU bug.";
  kiln::Store(fill_, kiln::I32(0));
  kiln::Store(flush_count_, kiln::I32(0));
  kiln::Store(stage_, kiln::Zeros(kiln::i32(planes_, 2 * kSublanes, kLanes)));
}

kiln::HbmRef StagedOutput::OutputRows(const kiln::Scalar q,
                                      const int64_t rows) const {
  return output_(kiln::Slice(q, rows), kiln::Slice(0, kLanes));
}

void StagedOutput::Append(
    const absl::Span<const kiln::Tile> tile_planes, const kiln::Scalar total,
    const absl::Span<const kiln::Scalar> scalar_prefix_planes) {
  ABSL_CHECK_EQ(  // CRASH_OK=compile-time kernel construction bug
      static_cast<int64_t>(scalar_prefix_planes.size() + tile_planes.size()),
      planes_)
      << "Each appended element must provide every output plane. This is a "
         "TorchTPU bug.";
  const kiln::Scalar fill = kiln::Load(fill_);
  const kiln::Scalar rem = fill & (kTile - 1);
  const kiln::Scalar lane_offset = rem & (kLanes - 1);
  const kiln::Scalar stage_row = rem >> kLaneShift;
  const kiln::Scalar end = lane_offset + total;
  // Element i lands at flat position `lane_offset + i` from staging row
  // `stage_row`: rows stage_row..stage_row+7 (the body) take positions below
  // 1024, and row stage_row+8 (the tail) takes those that wrapped past it.
  const kiln::Tile body_live = (pos_ >= lane_offset) & (pos_ < end);
  const kiln::Tile tail_live = row_lane_ < (end - kTile);
  const kiln::Tile body_select = lane_ >= lane_offset;

  int64_t plane = 0;
  for (const kiln::Scalar& s : scalar_prefix_planes) {
    kiln::Store(stage_(plane, kiln::Slice(stage_row, kSublanes)), pos_.Splat(s),
                {.mask = body_live});
    kiln::Store(stage_(plane, kiln::Slice(stage_row + kSublanes, 1)),
                row_lane_.Splat(s), {.mask = tail_live});
    ++plane;
  }
  for (const kiln::Tile& values : tile_planes) {
    const kiln::Tile rot = values.Rotate(lane_offset, 1);
    // Lanes below `lane_offset` wrapped around, so they take the row above.
    const kiln::Tile body = Select(body_select, rot, rot.Rotate(1, 0));
    const kiln::Tile tail = rot.Slice({kSublanes - 1, 0}, {1, kLanes});
    kiln::Store(stage_(plane, kiln::Slice(stage_row, kSublanes)), body,
                {.mask = body_live});
    kiln::Store(stage_(plane, kiln::Slice(stage_row + kSublanes, 1)), tail,
                {.mask = tail_live});
    ++plane;
  }

  const kiln::Scalar new_fill = fill + total;
  kiln::Store(fill_, new_fill);
  kiln::If((new_fill >> kTileShift) > (fill >> kTileShift),
           [this] { Flush(); });
}

// DMA descriptor from flush `slot` to the `tile_rows_ * 8` output rows at
// `tile_rows_ * q`. `Wait()` only depends on the semaphore and transfer size,
// so any `q` suffices.
kiln::AsyncCopyDescriptor StagedOutput::FlushCopy(const kiln::IndexArg slot,
                                                  const kiln::Scalar q) const {
  const int64_t rows = tile_rows_ * kSublanes;
  return kiln::MakeAsyncCopy(packed_(slot, kiln::Slice(0, rows)),
                             OutputRows(tile_rows_ * q, rows),
                             flush_sems_(slot));
}

// Packs the 1024 staged elements into a flush slot (interleaving the planes of
// each group when `tile_rows_ > 1`), copies it to HBM asynchronously, and
// shifts the overflow rows down.
void StagedOutput::Flush() {
  const kiln::Scalar flushes = kiln::Load(flush_count_);
  const kiln::Scalar slot = flushes & (kSlots - 1);
  const kiln::Scalar q = flushes << kSublaneShift;
  // Never write past `max_groups_`.
  kiln::If(q + kSublanes <= max_groups_, [this, flushes, slot, q] {
    const kiln::AsyncCopyDescriptor copy = FlushCopy(slot, q);
    kiln::If(flushes >= kSlots, [&copy] { copy.Wait(); });
    kiln::VmemRef packed = packed_(slot);
    if (tile_rows_ == 1) {
      kiln::Store(packed(kiln::Slice(0, kSublanes)),
                  kiln::Load(stage_(0, kiln::Slice(0, kSublanes))));
    } else {
      for (int r = 0; r < kSublanes; ++r) {
        for (int c = 0; c < planes_; ++c) {
          kiln::Store(packed(kiln::Slice(tile_rows_ * r + c, 1)),
                      kiln::Load(stage_(c, kiln::Slice(r, 1))));
        }
      }
    }
    copy.Start();
  });
  kiln::Store(stage_(kiln::all, kiln::Slice(0, kSublanes)),
              kiln::Load(stage_(kiln::all, kiln::Slice(kSublanes, kSublanes))));
  kiln::Store(stage_(kiln::all, kiln::Slice(kSublanes, kSublanes)),
              kiln::Zeros(kiln::i32(planes_, kSublanes, kLanes)));
  kiln::Store(flush_count_, flushes + 1);
}

void StagedOutput::Finish() {
  const kiln::Scalar flushes = kiln::Load(flush_count_);
  const kiln::Scalar copies = Min(flushes, max_groups_ >> kSublaneShift);
  for (int s = 0; s < kSlots; ++s) {
    kiln::If(copies > s, [this, s] { FlushCopy(s, kiln::I32(0)).Wait(); });
  }

  const kiln::Scalar rem = kiln::Load(fill_) & (kTile - 1);
  const kiln::Scalar q = flushes << kSublaneShift;
  const kiln::Scalar rows =
      Min((rem + (kLanes - 1)) >> kLaneShift, Max(max_groups_ - q, 0));
  kiln::For(rows, [this, q](kiln::Scalar r) {
    const kiln::Scalar out_row = tile_rows_ * (q + r);
    for (int c = 0; c < planes_; ++c) {
      kiln::SyncCopy(stage_(c, kiln::Slice(r, 1)), OutputRows(out_row + c, 1));
    }
  });
}

}  // namespace torch_tpu
