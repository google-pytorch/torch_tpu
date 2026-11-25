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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_STREAM_COMPACTION_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_STREAM_COMPACTION_H_

// Kiln device-side building blocks for stream-compaction kernels, which keep
// the live elements of each 1024-element tile and append them to an output.

#include <cstdint>

#include "absl/types/span.h"
#include "csrc/experimental/precompiled_kernels/kernel_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"

namespace torch_tpu {

// Geometry of a scratch window padded to a transfer granule and stepped in
// 1024-element tiles.
struct TileWindow {
  kiln::Scalar base;   // First logical element of the window.
  kiln::Scalar n_pad;  // Width padded up to a multiple of `granule`.
  kiln::Scalar tiles;  // Number of 1024-element tiles.
};

// Pads `width` up to a multiple of `granule` (power of two, default 128 lanes)
// and returns the resulting window geometry.
[[nodiscard]] TileWindow MakeTileWindow(kiln::Scalar base, kiln::Scalar width,
                                        int64_t granule = kLanes);

// Per-element shift distances and live count of a compacted 1024-element tile.
struct CompactionRanks {
  kiln::Tile distance;  // Prior dead elements before each live element.
  kiln::Scalar total;   // Number of live elements in the tile.
};

// Upper-triangular 128x128 bf16 matrix T[i, j] = (i <= j) for exact in-row
// prefix sums of 0/1 flags on the MXU (sums are at most 128).
[[nodiscard]] kiln::Tile PrefixTriangle();

// From 0/1 live flags on an (8, 128) tile, computes each live element's
// left-shift distance (dead elements before it; 0 for dead elements) and the
// tile's live count, using `tri` from `PrefixTriangle()` and
// `sublane = Iota((8, 128), 0)`.
[[nodiscard]] CompactionRanks ComputePrefixRanks(kiln::Tile live_flags,
                                                 kiln::Tile sublane,
                                                 kiln::Tile tri);

// Routes each element of an (8, 128) tile left by the 10-bit shift distance
// stored at bits `[distance_shift, distance_shift + 10)` of `control`, clearing
// those distance bits as elements move and zeroing vacated slots. Returns the
// compacted tile, which may be ignored when only routing `payloads`.
//
// Each tile in `payloads` is routed in lockstep with `control`. Unlike
// `control`, vacated payload slots keep stale values, so callers must mask
// them by the live count.
kiln::Tile CompactTile(kiln::Tile control, int distance_shift, kiln::Tile lane,
                       kiln::Tile sublane,
                       absl::Span<kiln::Tile* const> payloads = {});

// Zeroes the high words of S64 storage viewed as `output`, an S32[R, 128] view
// whose low words fill rows [0, R/2) and high words rows [R/2, R), with R even.
// Kernels whose S64 results are nonnegative and fit in 32 bits call this, then
// write only the low words.
void ZeroHighWords(kiln::HbmRef output);

// Double-buffered VMEM staging and async HBM flushing for stream-compaction
// kernels that emit `planes` S32 values per live element into an S32[R, 128]
// output view where each 128-element group occupies `tile_rows` storage rows
// (a power of two >= `planes`).
//
// The appended elements must fit in the first `max_rows` storage rows of
// `output`. Writes past `max_rows` are skipped rather than written out of
// bounds, so if they don't fit, the output is incomplete.
//
// Must be constructed inside `kiln::CreateKernel`: the constructor allocates
// and zero-initializes the staging state.
class StagedOutput {
 public:
  StagedOutput(kiln::HbmRef output, int64_t planes, int64_t tile_rows,
               kiln::Scalar max_rows);

  // Owns VMEM, SMEM and semaphore allocations, which copies would alias.
  StagedOutput(const StagedOutput&) = delete;
  StagedOutput& operator=(const StagedOutput&) = delete;

  // Appends `total` live elements (occupying positions [0, total) in each tile
  // of `tile_planes`), prefixed by constant `scalar_prefix_planes` (which need
  // no lane rotation). Flushes asynchronously whenever a full 1024-element tile
  // completes.
  void Append(absl::Span<const kiln::Tile> tile_planes, kiln::Scalar total,
              absl::Span<const kiln::Scalar> scalar_prefix_planes = {});

  // Waits for in-flight async flushes and copies any remaining partial rows.
  void Finish();

 private:
  static constexpr int64_t kSlots = 2;

  // Views `rows` 128-element storage rows of the output starting at row `q`.
  kiln::HbmRef OutputRows(kiln::Scalar q, int64_t rows) const;
  kiln::AsyncCopyDescriptor FlushCopy(kiln::IndexArg slot,
                                      kiln::Scalar q) const;
  void Flush();

  kiln::HbmRef output_;
  int64_t planes_;
  int64_t tile_rows_;
  kiln::Scalar max_groups_;
  kiln::VmemRef stage_;
  kiln::VmemRef packed_;
  kiln::SmemRef fill_;
  kiln::SmemRef flush_count_;
  kiln::SemaphoreArray flush_sems_;
  kiln::Tile lane_;
  kiln::Tile pos_;
  kiln::Tile row_lane_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_STREAM_COMPACTION_H_
