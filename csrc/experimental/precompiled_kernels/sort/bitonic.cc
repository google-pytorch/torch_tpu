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

#include "csrc/experimental/precompiled_kernels/sort/bitonic.h"

#include <cstdint>

#include "csrc/internal/mosaic/kiln/kiln.h"

namespace torch_tpu {
namespace {

using kiln::Tile;

constexpr int64_t kRows = 8;
constexpr int64_t kLanes = 128;
constexpr int64_t kLaneBits = 7;   // kLanes == 1 << kLaneBits
constexpr int64_t kTileBits = 10;  // kRows * kLanes == 1 << kTileBits

// Row and column of every element in the tile.
struct TileIndices {
  Tile rows;
  Tile cols;
};

TileIndices MakeTileIndices() {
  return {.rows = kiln::Iota(kiln::i32(kRows, kLanes), /*axis=*/0),
          .cols = kiln::Iota(kiln::i32(kRows, kLanes), /*axis=*/1)};
}

// True where bit `bit` of the element's index is set. The low 7 bits of the
// index are the column, the top 3 the row.
Tile IndexBit(const TileIndices& idx, int64_t bit) {
  if (bit < kLaneBits) return (idx.cols & (int64_t{1} << bit)) != 0;
  return (idx.rows & (int64_t{1} << (bit - kLaneBits))) != 0;
}

// Gives each element the value of its partner, the element whose index differs
// only in bit `bit`.
Tile Partner(Tile x, const TileIndices& idx, int64_t bit) {
  if (bit < kLaneBits) {
    return kiln::Gather(x, idx.cols ^ (int64_t{1} << bit), /*axis=*/1);
  }
  return kiln::Gather(x, idx.rows ^ (int64_t{1} << (bit - kLaneBits)),
                      /*axis=*/0);
}

// Whether (a, a_payload) comes before (b, b_payload).
Tile PairLess(const Tile& a, const Tile& a_payload, const Tile& b,
              const Tile& b_payload) {
  return Select(a == b, a_payload < b_payload, a < b);
}

// Compares each element with its partner. The lower index of each pair keeps
// the smaller (key, payload), or the larger one where `desc` is set.
void CompareExchange(Tile& keys, Tile& payload, const TileIndices& idx,
                     int64_t bit, const Tile& desc) {
  const Tile partner = Partner(keys, idx, bit);
  const Tile partner_payload = Partner(payload, idx, bit);
  const Tile keep_min = IndexBit(idx, bit) == desc;
  const Tile take =
      Select(keep_min, PairLess(partner, partner_payload, keys, payload),
             PairLess(keys, payload, partner, partner_payload));
  keys = Select(take, partner, keys);
  payload = Select(take, partner_payload, payload);
}

}  // namespace

void BitonicSortTile(Tile& keys, Tile& payload) {
  const TileIndices idx = MakeTileIndices();
  const Tile ascending = kiln::Zeros(kiln::i32(kRows, kLanes)) != 0;
  // Stage s builds sorted runs of 2^(s+1), alternating up and down so that
  // neighbouring runs can be merged by the next stage. The last stage sorts
  // the whole tile ascending.
  for (int64_t stage = 0; stage < kTileBits; ++stage) {
    const Tile desc =
        stage + 1 < kTileBits ? IndexBit(idx, stage + 1) : ascending;
    for (int64_t bit = stage; bit >= 0; --bit) {
      CompareExchange(keys, payload, idx, bit, desc);
    }
  }
}

}  // namespace torch_tpu
