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

#include "csrc/internal/mosaic/kiln/kiln.h"

// This is a proxy for "are any MLIR headers being included?"
// The MLIR context will be included in any MLIR-based operation.
#ifdef MLIR_IR_MLIRCONTEXT_H
#error Kiln should not include MLIR headers in the public API.
#endif

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/testlib/filecheck.h"

namespace torch_tpu::kiln {
namespace {

using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;
using testing::ElementsAre;
using testing::IsEmpty;

class KilnTest : public testing::Test {
 protected:
  // Verifies that the kernel survives the same serialization step as the
  // production kernels and that its textual form matches `pattern`, a
  // FileCheck pattern (see https://llvm.org/docs/CommandGuide/FileCheck.html).
  // Patterns are written as they would be in an MLIR lit test; lines that are
  // not `CHECK` directives (e.g. `//` comments) are ignored by FileCheck.
  void ExpectKernelMatches(Kernel&& kernel, std::string_view pattern) {
    std::string ir = ToString(kernel);
    EXPECT_THAT(kernel.Verify(), IsOk());
    EXPECT_THAT(std::move(kernel).Serialize(), IsOk()) << ir;
    EXPECT_THAT(xla::RunFileCheck(ir, pattern), IsOkAndHolds(true));
  }
};

// out[i, j] += a[i, k] @ b[k, j], with K as the innermost ("arbitrary") grid
// axis and an f32 accumulator in scratch.
TEST_F(KilnTest, BlockwiseMatmul) {
  constexpr int64_t kM = 1024, kN = 512, kK = 2048;
  constexpr int64_t kTm = 256, kTn = 256, kTk = 512;

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{
          .name = "matmul",
          .grid = {kM / kTm, kN / kTn, kK / kTk},
          .dimension_semantics = {DimensionSemantics::kParallel,
                                  DimensionSemantics::kParallel,
                                  DimensionSemantics::kArbitrary},
      },
      InSpec{bf16(kTm, kTk),
             IndexMap([] { return IndexIndices{ProgramId(0), ProgramId(2)}; })},
      InSpec{bf16(kTk, kTn),
             IndexMap([] { return IndexIndices{ProgramId(2), ProgramId(1)}; })},
      OutSpec{bf16(kTm, kTn), IndexMap([] {
                return IndexIndices{ProgramId(0), ProgramId(1)};
              })},
      ScratchSpec{f32(kTm, kTn)}, [&](Ref a, Ref b, Ref out, Ref acc) {
        Scalar kk = ProgramId(2);
        If(kk == 0, [&] { acc = Zeros(acc.shape()); });
        acc = Tile(acc) + Dot(Tile(a), Tile(b));
        If(kk == NumPrograms(2) - 1, [&] { out = Tile(acc).ToBF16(); });
      });
  ASSERT_THAT(kernel, IsOk());

  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func private @matmul_index_map_0(
    CHECK-SAME:    %[[I:arg[0-9]+]]: i32, %[[J:arg[0-9]+]]: i32, %[[K:arg[0-9]+]]: i32) -> (i32, i32)
    CHECK-NEXT:    return %[[I]], %[[K]]
    CHECK-LABEL: func.func private @matmul_index_map_1(
    CHECK-SAME:    %[[I:arg[0-9]+]]: i32, %[[J:arg[0-9]+]]: i32, %[[K:arg[0-9]+]]: i32) -> (i32, i32)
    CHECK-NEXT:    return %[[K]], %[[J]]
    CHECK-LABEL: func.func private @matmul_index_map_2(
    CHECK-SAME:    %[[I:arg[0-9]+]]: i32, %[[J:arg[0-9]+]]: i32, %[[K:arg[0-9]+]]: i32) -> (i32, i32)
    CHECK-NEXT:    return %[[I]], %[[J]]

    CHECK-LABEL: func.func @matmul(
    CHECK-SAME:    %[[I:arg[0-9]+]]: i32, %[[J:arg[0-9]+]]: i32, %[[K:arg[0-9]+]]: i32,
    CHECK-SAME:    %[[A:arg[0-9]+]]: memref<256x512xbf16, #tpu.memory_space<vmem>>,
    CHECK-SAME:    %[[B:arg[0-9]+]]: memref<512x256xbf16, #tpu.memory_space<vmem>>,
    CHECK-SAME:    %[[OUT:arg[0-9]+]]: memref<256x256xbf16, #tpu.memory_space<vmem>>,
    CHECK-SAME:    %[[ACC:arg[0-9]+]]: memref<256x256xf32, #tpu.memory_space<vmem>>)
    CHECK-SAME:    dimension_semantics = [#tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<arbitrary>]
    CHECK-SAME:    iteration_bounds = array<i64: 4, 2, 4>
    CHECK-SAME:    scratch_operands = 1
    CHECK-SAME:    window_params = [{transform_indices = @matmul_index_map_0, window_bounds = array<i64: 256, 512>}, {transform_indices = @matmul_index_map_1, window_bounds = array<i64: 512, 256>}, {transform_indices = @matmul_index_map_2, window_bounds = array<i64: 256, 256>}]

    CHECK:         %[[C0:.*]] = arith.constant 0 : i32
    CHECK:         %[[IS_FIRST:.*]] = arith.cmpi eq, %[[K]], %[[C0]] : i32
    CHECK:         scf.if %[[IS_FIRST]] {
    CHECK:           %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<256x256xf32>
    CHECK:           tpu.vector_store %[[ACC]][%{{.*}}, %{{.*}}], %[[ZERO]]
    CHECK:         }
    CHECK:         %[[ACC_V:.*]] = vector.load %[[ACC]]
    CHECK:         %[[A_V:.*]] = vector.load %[[A]]
    CHECK:         %[[B_V:.*]] = vector.load %[[B]]
    CHECK:         %[[DOT:.*]] = tpu.matmul %[[A_V]], %[[B_V]], %{{.*}} : vector<256x512xbf16>, vector<512x256xbf16>, vector<256x256xf32> -> vector<256x256xf32>
    CHECK:         %[[SUM:.*]] = arith.addf %[[ACC_V]], %[[DOT]] : vector<256x256xf32>
    CHECK:         tpu.vector_store %[[ACC]][%{{.*}}, %{{.*}}], %[[SUM]]
    CHECK:         %[[C3:.*]] = arith.constant 3 : i32
    CHECK:         %[[IS_LAST:.*]] = arith.cmpi eq, %[[K]], %[[C3]] : i32
    CHECK:         scf.if %[[IS_LAST]] {
    CHECK:           %[[FINAL:.*]] = vector.load %[[ACC]]
    CHECK:           %[[BF16:.*]] = arith.truncf %[[FINAL]] : vector<256x256xf32> to vector<256x256xbf16>
    CHECK:           tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[BF16]]
    CHECK:         }
    CHECK:         return
  )");
}

// Scalar prefetch + function index map: gathers rows by an index table held in
// SMEM and sums fixed-size chunks of each row.
TEST_F(KilnTest, ScalarPrefetchAndLoop) {
  constexpr int64_t kRows = 64, kCols = 1024, kChunk = 128;

  IndexMap gathered = IndexMap(
      [] { return IndexIndices{Scalar(ScalarPrefetch(0)(ProgramId(0))), 0}; });

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "gather_rows", .grid = {kRows}},
      ScalarPrefetchSpec{i32(kRows)}, InSpec{f32(1, kCols), gathered},
      OutSpec{f32(1, kChunk),
              IndexMap([] { return IndexIndices{ProgramId(0), 0}; })},
      [&](Ref /*indices*/, Ref rows, Ref out) {
        Tile total = Zeros(f32(1, kChunk));
        for (int64_t c = 0; c < kCols; c += kChunk) {
          total = total + Tile(rows(0, Slice(c, kChunk)));
        }
        out(0, all) = total;
      });
  ASSERT_THAT(kernel, IsOk());

  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func private @gather_rows_index_map_0(
    CHECK-SAME:    %[[I:arg[0-9]+]]: i32, %[[IDX:arg[0-9]+]]: memref<64xi32, #tpu.memory_space<smem>>) -> (i32, i32)
    CHECK:         %[[SLICE:.*]] = tpu.memref_slice %[[IDX]][%[[I]]] : memref<64xi32, #tpu.memory_space<smem>> -> memref<1xi32, #tpu.memory_space<smem>>
    CHECK:         %[[SQ:.*]] = tpu.memref_squeeze %[[SLICE]] : memref<1xi32, #tpu.memory_space<smem>> -> memref<i32, #tpu.memory_space<smem>>
    CHECK:         %[[ROW:.*]] = memref.load %[[SQ]][] : memref<i32, #tpu.memory_space<smem>>
    CHECK:         %[[C0:.*]] = arith.constant 0 : i32
    CHECK:         return %[[ROW]], %[[C0]]

    CHECK-LABEL: func.func private @gather_rows_index_map_1(
    CHECK-SAME:    %[[I:arg[0-9]+]]: i32, %{{.*}}: memref<64xi32, #tpu.memory_space<smem>>) -> (i32, i32)
    CHECK:         return %[[I]], %{{.*}}

    CHECK-LABEL: func.func @gather_rows(
    CHECK-SAME:    %{{.*}}: i32, %{{.*}}: memref<64xi32, #tpu.memory_space<smem>>,
    CHECK-SAME:    %[[ROWS:arg[0-9]+]]: memref<1x1024xf32, #tpu.memory_space<vmem>>,
    CHECK-SAME:    %[[OUT:arg[0-9]+]]: memref<1x128xf32, #tpu.memory_space<vmem>>)
    CHECK-SAME:    iteration_bounds = array<i64: 64>
    CHECK-SAME:    scalar_prefetch = 1
    CHECK-SAME:    window_params = [{transform_indices = @gather_rows_index_map_0, window_bounds = array<i64: 1, 1024>}, {transform_indices = @gather_rows_index_map_1, window_bounds = array<i64: 1, 128>}]

    CHECK:         %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<1x128xf32>
    CHECK:         %[[S0:.*]] = tpu.memref_slice %[[ROWS]][%{{.*}}, %{{.*}}] : memref<1x1024xf32, #tpu.memory_space<vmem>> -> memref<1x128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[SQ0:.*]] = tpu.memref_squeeze %[[S0]] : memref<1x128xf32, #tpu.memory_space<vmem>> -> memref<128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[L0:.*]] = vector.load %[[SQ0]][%{{.*}}] : memref<128xf32, #tpu.memory_space<vmem>>, vector<128xf32>
    CHECK:         %[[B0:.*]] = vector.broadcast %[[L0]] : vector<128xf32> to vector<1x128xf32>
    CHECK:         arith.addf %[[ZERO]], %[[B0]] : vector<1x128xf32>
    CHECK-COUNT-7: tpu.memref_slice %[[ROWS]]
    CHECK:         %[[OUT_SQ:.*]] = tpu.memref_squeeze %[[OUT]] : memref<1x128xf32, #tpu.memory_space<vmem>> -> memref<128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[TOTAL:.*]] = vector.shape_cast %{{.*}} : vector<1x128xf32> to vector<128xf32>
    CHECK:         tpu.vector_store %[[OUT_SQ]][%{{.*}}], %[[TOTAL]]
  )");
}

TEST_F(KilnTest, RefSliceRankCollapsing) {
  // 4-D block: (1, 1, 32, 128). Slicing in4d(0, 0) yields a 2-D tile (32, 128).
  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "rank_collapse", .grid = {1}},
                   InSpec{.shape = f32(1, 1, 32, 128)},
                   OutSpec{.shape = f32(32, 128)}, [&](Ref in4d, Ref out2d) {
                     // in4d(0, 0) collapses the first 2 dimensions, resulting
                     // in rank 2.
                     Tile tile2d(in4d(0, 0));
                     EXPECT_EQ(tile2d.rank(), 2);
                     EXPECT_THAT(tile2d.dims(), ElementsAre(32, 128));
                     out2d = tile2d;
                   });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @rank_collapse(
    CHECK-SAME:    %{{.*}}: i32, %[[IN:arg[0-9]+]]: memref<1x1x32x128xf32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>)
    CHECK:         %[[SQ:.*]] = tpu.memref_squeeze %[[IN]] : memref<1x1x32x128xf32, #tpu.memory_space<vmem>> -> memref<32x128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[V:.*]] = vector.load %[[SQ]][%{{.*}}, %{{.*}}] : memref<32x128xf32, #tpu.memory_space<vmem>>, vector<32x128xf32>
    CHECK:         tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[V]]
  )");
}

TEST_F(KilnTest, LiteralOverloads) {
  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "literal_ops", .grid = {1}},
                   InSpec{f32(32, 128)}, InSpec{i32(32, 128)},
                   OutSpec{f32(32, 128)}, [&](Ref f_in, Ref i_in, Ref out) {
                     Tile f(f_in);
                     Tile i(i_in);

                     // Float tile with int literal and float literal
                     Tile f_add_int = f + 1;      // int literal
                     Tile f_sub_float = f - 2.5;  // float literal
                     Tile f_scale = 0.5 * f;      // literal on LHS
                     Tile f_clamp = Max(f, -10000.0);
                     Tile f_mask = Select(f > 0, f, 0.0);

                     // Int tile with int literal
                     Tile i_add_int = i + 10;
                     Tile i_cmp = i == 0;
                     Tile i_combined = Select(i_cmp, i_add_int, 0);

                     // Scalar with int and float literal + Cond <-> Scalar
                     Scalar s = ProgramId(0);
                     Scalar s_plus = s + 1;
                     Cond c = s_plus == 1;
                     Scalar s_bool = c;
                     EXPECT_TRUE(s_bool.dtype().IsBool());
                     EXPECT_FALSE(s_bool.dtype().IsIntOrFloat());
                     Cond c_roundtrip = static_cast<Cond>(s_bool);

                     If(
                         c_roundtrip,
                         [&] {
                           out = f_add_int + f_sub_float + f_scale + f_clamp +
                                 f_mask + i_combined.ToF32();
                         },
                         [&] { out = Zeros(out.shape()); });
                   });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @literal_ops(
    CHECK-SAME:    %[[PID:arg[0-9]+]]: i32, %[[F_IN:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>, %[[I_IN:arg[0-9]+]]: memref<32x128xi32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>)
    CHECK:         %[[F:.*]] = vector.load %[[F_IN]]
    CHECK:         %[[I:.*]] = vector.load %[[I_IN]]

    // Literals adopt the dtype of the tile they are combined with.
    CHECK:         %[[C1:.*]] = arith.constant dense<1.000000e+00> : vector<32x128xf32>
    CHECK:         %[[F_ADD:.*]] = arith.addf %[[F]], %[[C1]] : vector<32x128xf32>
    CHECK:         %[[C2_5:.*]] = arith.constant dense<2.500000e+00> : vector<32x128xf32>
    CHECK:         %[[F_SUB:.*]] = arith.subf %[[F]], %[[C2_5]] : vector<32x128xf32>
    CHECK:         %[[C0_5:.*]] = arith.constant dense<5.000000e-01> : vector<32x128xf32>
    CHECK:         %[[F_SCALE:.*]] = arith.mulf %[[F]], %[[C0_5]] : vector<32x128xf32>
    CHECK:         %[[C_NEG:.*]] = arith.constant dense<-1.000000e+04> : vector<32x128xf32>
    CHECK:         %[[F_CLAMP:.*]] = arith.maximumf %[[F]], %[[C_NEG]] : vector<32x128xf32>
    CHECK:         %[[GT:.*]] = arith.cmpf ogt, %[[F]], %{{.*}} : vector<32x128xf32>
    CHECK:         %[[F_MASK:.*]] = arith.select %[[GT]], %[[F]], %{{.*}} : vector<32x128xi1>, vector<32x128xf32>
    CHECK:         %[[C10:.*]] = arith.constant dense<10> : vector<32x128xi32>
    CHECK:         %[[I_ADD:.*]] = arith.addi %[[I]], %[[C10]] : vector<32x128xi32>
    CHECK:         %[[I_EQ:.*]] = arith.cmpi eq, %[[I]], %{{.*}} : vector<32x128xi32>
    CHECK:         %[[I_SEL:.*]] = arith.select %[[I_EQ]], %[[I_ADD]], %{{.*}} : vector<32x128xi1>, vector<32x128xi32>

    // Cond -> Scalar -> Cond round trip still yields a plain i1 for scf.if.
    CHECK:         %[[S_PLUS:.*]] = arith.addi %[[PID]], %{{.*}} : i32
    CHECK:         %[[COND:.*]] = arith.cmpi eq, %[[S_PLUS]], %{{.*}} : i32
    CHECK:         scf.if %[[COND]] {
    CHECK:           arith.addf %[[F_ADD]], %[[F_SUB]]
    CHECK:           arith.addf %{{.*}}, %[[F_SCALE]]
    CHECK:           arith.addf %{{.*}}, %[[F_CLAMP]]
    CHECK:           arith.addf %{{.*}}, %[[F_MASK]]
    CHECK:           %[[I_F:.*]] = arith.sitofp %[[I_SEL]] : vector<32x128xi32> to vector<32x128xf32>
    CHECK:           %[[RES:.*]] = arith.addf %{{.*}}, %[[I_F]]
    CHECK:           tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[RES]]
    CHECK:         } else {
    CHECK:           %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<32x128xf32>
    CHECK:           tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[ZERO]]
    CHECK:         }
  )");
}

TEST_F(KilnTest, ScalarFloatOperations) {
  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "scalar_float_ops", .grid = {1}},
                   InSpec{.shape = f32(32, 128), .space = MemorySpace::kSmem},
                   OutSpec{.shape = f32(32, 128), .space = MemorySpace::kSmem},
                   [&](Ref in, Ref out) {
                     Scalar sf = F32(2.5);
                     EXPECT_TRUE(sf.dtype().IsFloat());
                     EXPECT_FALSE(sf.dtype().IsInt());

                     Scalar sf2 = sf * 2.0;
                     Scalar sf3 = sf2 + 1.0;
                     Scalar sf_div = sf3 / 2.0;
                     Scalar sf_neg = -sf;

                     Scalar sf_exp = Exp(sf);
                     Scalar sf_log = Log(sf);
                     Scalar sf_sqrt = Sqrt(sf);
                     Scalar sf_rsqrt = Rsqrt(sf);
                     Scalar sf_abs = Abs(sf_neg);

                     Scalar sf_sel = Select(sf > 1.0, sf, 0.0);
                     Scalar sf_min = Min(sf, 10.0);
                     Scalar sf_max = Max(sf, 0.0);

                     Scalar loaded(in(0, 0));
                     out(0, 0) = loaded + sf_exp + sf_log + sf_sqrt + sf_rsqrt +
                                 sf_abs + sf_sel + sf_min + sf_max + sf_div;
                   });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @scalar_float_ops(
    CHECK-SAME:    %{{.*}}: i32, %[[IN:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<smem>>, %[[OUT:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<smem>>)
    CHECK:         %[[SF:.*]] = arith.constant 2.500000e+00 : f32
    CHECK:         %[[MUL:.*]] = arith.mulf %[[SF]], %{{.*}} : f32
    CHECK:         %[[ADD:.*]] = arith.addf %[[MUL]], %{{.*}} : f32
    CHECK:         %[[DIV:.*]] = arith.divf %[[ADD]], %{{.*}} : f32
    CHECK:         %[[NEG:.*]] = arith.negf %[[SF]] : f32
    CHECK:         %[[EXP:.*]] = math.exp %[[SF]] : f32
    CHECK:         %[[LOG:.*]] = math.log %[[SF]] : f32
    CHECK:         %[[SQRT:.*]] = math.sqrt %[[SF]] : f32
    CHECK:         %[[RSQRT:.*]] = math.rsqrt %[[SF]] : f32
    CHECK:         %[[ABS:.*]] = math.absf %[[NEG]] : f32
    CHECK:         %[[GT:.*]] = arith.cmpf ogt, %[[SF]], %{{.*}} : f32
    CHECK:         %[[SEL:.*]] = arith.select %[[GT]], %[[SF]], %{{.*}} : f32
    CHECK:         %[[MIN:.*]] = arith.minimumf %[[SF]], %{{.*}} : f32
    CHECK:         %[[MAX:.*]] = arith.maximumf %[[SF]], %{{.*}} : f32

    // Scalar(ref) on an SMEM ref is a scalar memref.load of the squeezed slice.
    CHECK:         %[[IN_SLICE:.*]] = tpu.memref_slice %[[IN]][%{{.*}}, %{{.*}}] : memref<32x128xf32, #tpu.memory_space<smem>> -> memref<1x1xf32, #tpu.memory_space<smem>>
    CHECK:         %[[IN_SQ:.*]] = tpu.memref_squeeze %[[IN_SLICE]] : memref<1x1xf32, #tpu.memory_space<smem>> -> memref<f32, #tpu.memory_space<smem>>
    CHECK:         %[[LOADED:.*]] = memref.load %[[IN_SQ]][] : memref<f32, #tpu.memory_space<smem>>
    CHECK:         arith.addf %[[LOADED]], %[[EXP]] : f32
    CHECK:         arith.addf %{{.*}}, %[[LOG]] : f32
    CHECK:         arith.addf %{{.*}}, %[[SQRT]] : f32
    CHECK:         arith.addf %{{.*}}, %[[RSQRT]] : f32
    CHECK:         arith.addf %{{.*}}, %[[ABS]] : f32
    CHECK:         arith.addf %{{.*}}, %[[SEL]] : f32
    CHECK:         arith.addf %{{.*}}, %[[MIN]] : f32
    CHECK:         arith.addf %{{.*}}, %[[MAX]] : f32
    CHECK:         %[[RES:.*]] = arith.addf %{{.*}}, %[[DIV]] : f32
    CHECK:         %[[OUT_SLICE:.*]] = tpu.memref_slice %[[OUT]]
    CHECK:         %[[OUT_SQ:.*]] = tpu.memref_squeeze %[[OUT_SLICE]]
    CHECK:         memref.store %[[RES]], %[[OUT_SQ]][] : memref<f32, #tpu.memory_space<smem>>
  )");
}

// The same integer hash written once and applied to both a Scalar and a Tile.
template <typename T>
T Fmix32(T h) {
  h = h ^ (h >> 16);
  h = h * 0x85EBCA6Bu;
  h = h ^ (h >> 13);
  h = h * 0xC2B2AE35u;
  h = h ^ (h >> 16);
  return h;
}

TEST_F(KilnTest, BitwiseOperations) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "bitwise_ops", .grid = {4}}, InSpec{u32(32, 128)},
      OutSpec{u32(32, 128)}, [&](Ref in, Ref out) {
        Scalar pid = ProgramId(0).AsType(u32);
        Scalar stream = Fmix32(pid ^ 0x9E3779B9u);
        Scalar shifted = (stream << 1) | (stream & 1);

        Tile t(in);
        Tile hashed = Fmix32(t ^ shifted);
        Tile masked = (hashed & 0xFFFF) | (hashed >> 16);
        Tile masked_signed = masked.AsType(i32);
        out = Select(masked_signed >= 0, masked_signed, 0);
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @bitwise_ops(
    CHECK-SAME:    %[[PID:arg[0-9]+]]: i32, %[[IN:arg[0-9]+]]: memref<32x128xi32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<32x128xi32, #tpu.memory_space<vmem>>)
    CHECK-SAME:    iteration_bounds = array<i64: 4>

    // Scalar hash: unsigned literals are emitted as their i32 bit pattern.
    CHECK:         %[[SEED:.*]] = arith.constant -1640531527 : i32
    CHECK:         %[[H0:.*]] = arith.xori %[[PID]], %[[SEED]] : i32
    CHECK:         %[[SHR0:.*]] = arith.shrui %[[H0]], %{{.*}} : i32
    CHECK:         %[[H1:.*]] = arith.xori %[[H0]], %[[SHR0]] : i32
    CHECK:         arith.muli %[[H1]], %{{.*}} : i32
    CHECK:         arith.muli %{{.*}}, %{{.*}} : i32
    CHECK:         %[[STREAM:.*]] = arith.xori %{{.*}}, %{{.*}} : i32
    CHECK:         %[[SHL:.*]] = arith.shli %[[STREAM]], %{{.*}} : i32
    CHECK:         %[[AND:.*]] = arith.andi %[[STREAM]], %{{.*}} : i32
    CHECK:         %[[SHIFTED:.*]] = arith.ori %[[SHL]], %[[AND]] : i32

    // Tile hash: the Scalar is splat once, then the same op sequence follows.
    CHECK:         %[[T:.*]] = vector.load %[[IN]]
    CHECK:         %[[S_0D:.*]] = vector.broadcast %[[SHIFTED]] : i32 to vector<i32>
    CHECK:         %[[S:.*]] = vector.broadcast %[[S_0D]] : vector<i32> to vector<32x128xi32>
    CHECK:         %[[TX:.*]] = arith.xori %[[T]], %[[S]] : vector<32x128xi32>
    CHECK:         arith.shrui %[[TX]], %{{.*}} : vector<32x128xi32>
    CHECK:         arith.muli %{{.*}}, %{{.*}} : vector<32x128xi32>
    CHECK:         arith.muli %{{.*}}, %{{.*}} : vector<32x128xi32>
    CHECK:         %[[HASHED:.*]] = arith.xori %{{.*}}, %{{.*}} : vector<32x128xi32>
    CHECK:         %[[MASK:.*]] = arith.constant dense<65535> : vector<32x128xi32>
    CHECK:         %[[LO:.*]] = arith.andi %[[HASHED]], %[[MASK]] : vector<32x128xi32>
    CHECK:         %[[HI:.*]] = arith.shrui %[[HASHED]], %{{.*}} : vector<32x128xi32>
    CHECK:         %[[MASKED:.*]] = arith.ori %[[LO]], %[[HI]] : vector<32x128xi32>
    CHECK:         %[[GE:.*]] = arith.cmpi sge, %[[MASKED]], %{{.*}} : vector<32x128xi32>
    CHECK:         %[[RES:.*]] = arith.select %[[GE]], %[[MASKED]], %{{.*}} : vector<32x128xi1>, vector<32x128xi32>
    CHECK:         tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[RES]]
  )");
}

TEST_F(KilnTest, TileScalarOperations) {
  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "tile_scalar_ops", .grid = {2}},
                   InSpec{f32(32, 128)}, InSpec{i32(32, 128)},
                   OutSpec{f32(32, 128)}, [&](Ref f_in, Ref i_in, Ref out) {
                     Tile f(f_in);
                     Tile i(i_in);

                     Scalar s_idx = ProgramId(0);
                     Scalar s_f = F32(1.5);
                     Scalar s_i = I32(42);

                     // Verify 0-D tile properties when constructed from a
                     // scalar.
                     Tile t_scalar(s_i);
                     EXPECT_EQ(t_scalar.rank(), 0);
                     EXPECT_THAT(t_scalar.dims(), IsEmpty());
                     EXPECT_EQ(t_scalar.dtype(), i32);
                     EXPECT_EQ(t_scalar.shape(), i32());

                     // Arithmetic: Tile & Scalar (both LHS and RHS)
                     Tile f_add = f + s_f;
                     Tile f_sub = s_f - f;
                     Tile f_mul = f * s_idx.ToF32();
                     Tile f_div = s_f / f;

                     // Bitwise: Tile & Scalar
                     Tile i_xor = i ^ s_i;
                     Tile i_or = s_i | i;
                     Tile i_and = i & s_i;
                     Tile i_shl = i << s_i;
                     Tile i_shr = i >> s_i;

                     // Comparisons: produce mask tile
                     Tile cmp1 = f > s_f;
                     Tile cmp2 = s_f <= f;
                     Tile cmp3 = i == s_i;
                     Tile cmp4 = s_i != i;

                     // Select with Scalar operands
                     Tile w1 = Select(cmp1, f, s_f);
                     Tile w2 = Select(cmp2, s_f, f);
                     Tile w3 = Select(cmp3, s_f, 0.0);
                     Tile w4 = Select(cmp4, s_f, s_f * 2.0);

                     // Min/Max with Scalar
                     Tile max_f = Max(f, s_f);
                     Tile min_f = Min(s_f, f);

                     out = f_add + f_sub + f_mul + f_div + i_xor.ToF32() +
                           i_or.ToF32() + i_and.ToF32() + i_shl.ToF32() +
                           i_shr.ToF32() + w1 + w2 + w3 + w4 + max_f + min_f;
                   });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @tile_scalar_ops(
    CHECK-SAME:    %[[PID:arg[0-9]+]]: i32, %[[F_IN:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>, %[[I_IN:arg[0-9]+]]: memref<32x128xi32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>)
    CHECK-SAME:    iteration_bounds = array<i64: 2>
    CHECK:         %[[F:.*]] = vector.load %[[F_IN]]
    CHECK:         %[[I:.*]] = vector.load %[[I_IN]]
    CHECK:         %[[S_F:.*]] = arith.constant 1.500000e+00 : f32
    CHECK:         %[[S_I:.*]] = arith.constant 42 : i32

    // Tile(scalar) is a 0-D vector.
    CHECK:         vector.broadcast %[[S_I]] : i32 to vector<i32>

    // Tile <op> Scalar splats the scalar through a 0-D vector and keeps operand
    // order.
    CHECK:         %[[S_F_0D:.*]] = vector.broadcast %[[S_F]] : f32 to vector<f32>
    CHECK:         %[[S_F_V:.*]] = vector.broadcast %[[S_F_0D]] : vector<f32> to vector<32x128xf32>
    CHECK:         arith.addf %[[F]], %[[S_F_V]] : vector<32x128xf32>
    CHECK:         arith.subf %{{.*}}, %[[F]] : vector<32x128xf32>
    CHECK:         %[[PID_F:.*]] = arith.sitofp %[[PID]] : i32 to f32
    CHECK:         vector.broadcast %[[PID_F]] : f32 to vector<f32>
    CHECK:         arith.mulf %[[F]], %{{.*}} : vector<32x128xf32>
    CHECK:         arith.divf %{{.*}}, %[[F]] : vector<32x128xf32>
    CHECK:         arith.xori %[[I]], %{{.*}} : vector<32x128xi32>
    CHECK:         arith.ori %{{.*}}, %[[I]] : vector<32x128xi32>
    CHECK:         arith.andi %[[I]], %{{.*}} : vector<32x128xi32>
    CHECK:         arith.shli %[[I]], %{{.*}} : vector<32x128xi32>
    CHECK:         arith.shrsi %[[I]], %{{.*}} : vector<32x128xi32>
    CHECK:         %[[CMP1:.*]] = arith.cmpf ogt, %[[F]], %{{.*}} : vector<32x128xf32>
    CHECK:         %[[CMP2:.*]] = arith.cmpf ole, %{{.*}}, %[[F]] : vector<32x128xf32>
    CHECK:         %[[CMP3:.*]] = arith.cmpi eq, %[[I]], %{{.*}} : vector<32x128xi32>
    CHECK:         %[[CMP4:.*]] = arith.cmpi ne, %{{.*}}, %[[I]] : vector<32x128xi32>
    CHECK:         arith.select %[[CMP1]], %[[F]], %{{.*}} : vector<32x128xi1>, vector<32x128xf32>
    CHECK:         arith.select %[[CMP2]], %{{.*}}, %[[F]] : vector<32x128xi1>, vector<32x128xf32>
    CHECK:         arith.select %[[CMP3]], %{{.*}}, %{{.*}} : vector<32x128xi1>, vector<32x128xf32>
    CHECK:         arith.select %[[CMP4]], %{{.*}}, %{{.*}} : vector<32x128xi1>, vector<32x128xf32>
    CHECK:         arith.maximumf %[[F]], %{{.*}} : vector<32x128xf32>
    CHECK:         arith.minimumf %{{.*}}, %[[F]] : vector<32x128xf32>
    CHECK:         tpu.vector_store %[[OUT]]
  )");
}

TEST_F(KilnTest, RotateGatherExtractBitcastAndMaskedStore) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "rotate_gather_extract_bitcast_store", .grid = {1}},
      InSpec{f32(8, 128)}, OutSpec{i32(1, 128)}, [&](Ref in, Ref out) {
        Scalar step = ProgramId(0);
        Tile rotated = Rotate(Tile(in), 8 - (step & 7), /*axis=*/0);
        Tile row0 = rotated.Slice({0, 0}, {1, 128});
        Tile gathered = Gather(row0, Splat(step, {1, 128}), /*axis=*/1);
        Scalar lane_f32 = Extract(gathered, {0, 0});
        Scalar lane_bits = Bitcast(lane_f32, i32);
        Tile row_bits = Bitcast(row0, i32);
        Tile mask = Iota(i32(1, 128), /*axis=*/1) == (step & 127);
        Store(out, row_bits + lane_bits, mask);
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @rotate_gather_extract_bitcast_store(
    CHECK-SAME:    %[[STEP:arg[0-9]+]]: i32, %[[IN:arg[0-9]+]]: memref<8x128xf32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<1x128xi32, #tpu.memory_space<vmem>>)
    CHECK:         %[[V:.*]] = vector.load %[[IN]]
    CHECK:         %[[C7:.*]] = arith.constant 7 : i32
    CHECK:         %[[LOW:.*]] = arith.andi %[[STEP]], %[[C7]] : i32
    CHECK:         %[[C8:.*]] = arith.constant 8 : i32
    CHECK:         %[[AMOUNT:.*]] = arith.subi %[[C8]], %[[LOW]] : i32
    CHECK:         %[[ROTATED:.*]] = tpu.dynamic_rotate %[[V]] by %[[AMOUNT]] dim 0 : vector<8x128xf32>, i32 -> vector<8x128xf32>
    CHECK:         %[[ROW0:.*]] = vector.extract_strided_slice %[[ROTATED]] offsets = [0, 0], sizes = [1, 128], strides = [1, 1] : vector<8x128xf32> to vector<1x128xf32>
    CHECK:         %[[INDICES:.*]] = vector.broadcast %[[STEP]] : i32 to vector<1x128xi32>
    CHECK:         %[[GATHERED:.*]] = tpu.dynamic_gather %[[ROW0]][%[[INDICES]]] in [1] : vector<1x128xf32>, vector<1x128xi32> -> vector<1x128xf32>
    CHECK:         %[[LANE:.*]] = vector.extract %[[GATHERED]][0, 0] : f32 from vector<1x128xf32>
    CHECK:         %[[LANE_BITS:.*]] = arith.bitcast %[[LANE]] : f32 to i32
    CHECK:         %[[ROW_BITS:.*]] = arith.bitcast %[[ROW0]] : vector<1x128xf32> to vector<1x128xi32>
    CHECK:         %[[IOTA:.*]] = tpu.iota {dimensions = array<i32: 1>} : vector<1x128xi32>
    CHECK:         %[[C127:.*]] = arith.constant 127 : i32
    CHECK:         %[[LANE_ID:.*]] = arith.andi %[[STEP]], %[[C127]] : i32
    CHECK:         %[[LANE_ID_0D:.*]] = vector.broadcast %[[LANE_ID]] : i32 to vector<i32>
    CHECK:         %[[LANE_ID_V:.*]] = vector.broadcast %[[LANE_ID_0D]] : vector<i32> to vector<1x128xi32>
    CHECK:         %[[MASK:.*]] = arith.cmpi eq, %[[IOTA]], %[[LANE_ID_V]] : vector<1x128xi32>
    CHECK:         %[[LANE_BITS_0D:.*]] = vector.broadcast %[[LANE_BITS]] : i32 to vector<i32>
    CHECK:         %[[LANE_BITS_V:.*]] = vector.broadcast %[[LANE_BITS_0D]] : vector<i32> to vector<1x128xi32>
    CHECK:         %[[SUM:.*]] = arith.addi %[[ROW_BITS]], %[[LANE_BITS_V]] : vector<1x128xi32>
    CHECK:         tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[SUM]] masked %[[MASK]]
  )");
}

TEST_F(KilnTest, ForLoopWithInductionVarAndBounds) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "for_loop", .grid = {1}}, InSpec{f32(32, 128)},
      OutSpec{f32(32, 128)}, [&](Ref in, Ref out) {
        // Loop with static bounds.
        For(0, 10, [&](Scalar i) {
          EXPECT_TRUE(i.dtype().IsInt());
          EXPECT_FALSE(i.dtype().IsFloat());
          EXPECT_EQ(i.dtype(), i32);
          Scalar i_next = i + 1;
          If(i_next < 5, [&] { out = Tile(in); });
        });

        // Loop with dynamic Scalar bounds.
        Scalar iters = I32(10);
        For(iters, [&](Scalar i) {
          EXPECT_EQ(i.dtype(), i32);
          If(i == 0, [&] { out = Tile(in); });
        });
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @for_loop(
    CHECK-SAME:    %{{.*}}: i32, %[[IN:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>)

    // Static bounds become index constants; the induction variable is exposed to
    // the body as an i32.
    CHECK:         %[[C0:.*]] = arith.constant 0 : index
    CHECK:         %[[C10:.*]] = arith.constant 10 : index
    CHECK:         %[[C1:.*]] = arith.constant 1 : index
    CHECK:         scf.for %[[IV:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
    CHECK:           %[[I:.*]] = arith.index_cast %[[IV]] : index to i32
    CHECK:           %[[I_NEXT:.*]] = arith.addi %[[I]], %{{.*}} : i32
    CHECK:           %[[LT:.*]] = arith.cmpi slt, %[[I_NEXT]], %{{.*}} : i32
    CHECK:           scf.if %[[LT]] {
    CHECK:             %[[V:.*]] = vector.load %[[IN]]
    CHECK:             tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[V]]
    CHECK:           }
    CHECK:         }

    // A Scalar bound is cast from i32 to index.
    CHECK:         %[[ITERS:.*]] = arith.constant 10 : i32
    CHECK:         %[[UB:.*]] = arith.index_cast %[[ITERS]] : i32 to index
    CHECK:         scf.for %[[IV2:.*]] = %{{.*}} to %[[UB]] step %{{.*}} {
    CHECK:           %[[I2:.*]] = arith.index_cast %[[IV2]] : index to i32
    CHECK:           %[[EQ:.*]] = arith.cmpi eq, %[[I2]], %{{.*}} : i32
    CHECK:           scf.if %[[EQ]] {
    CHECK:             vector.load %[[IN]]
    CHECK:             tpu.vector_store %[[OUT]]
    CHECK:           }
    CHECK:         }
  )");
}

TEST_F(KilnTest, ScalarMismatchedTypesFail) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "mismatched", .grid = {1}}, InSpec{f32(32, 128)},
      InSpec{i32(32, 128)}, InSpec{u32(32, 128)}, OutSpec{f32(32, 128)},
      [&](Ref f_in, Ref i_in, Ref u_in, Ref out) {
        Scalar s_i = I32(1);
        Scalar s_u = U32(1);
        Scalar s_f = F32(1.0);
        EXPECT_DEATH(s_i + s_f, "\\+ requires operands of the same dtype");
        EXPECT_DEATH(s_i + s_u, "\\+ requires operands of the same dtype");
        EXPECT_DEATH(For(s_f, [&](Scalar) {}),
                     "IndexArg requires an integer scalar");

        Tile t_f(f_in);
        Tile t_i(i_in);
        Tile t_u(u_in);
        EXPECT_DEATH(t_f + t_i, "\\+ requires operands of the same dtype");
        EXPECT_DEATH(t_i + t_u, "\\+ requires operands of the same dtype");
        EXPECT_DEATH(static_cast<void>(t_i == t_u),
                     "Comparison requires operands of the same dtype");
        EXPECT_DEATH(Min(t_i, t_u),
                     "Min\\(\\) requires operands of the same dtype");
        EXPECT_DEATH(t_f & t_f, "& requires integer or bool operands");

        EXPECT_DEATH(f_in(0, 0, 0), "Too many slice indices");
        EXPECT_DEATH(f_in(0, Slice(0, 0)), "Slice size must be positive");
        EXPECT_DEATH(static_cast<void>(Scalar(f_in)),
                     "Ref::operator Scalar\\(\\) expects a 0-D ref");
        EXPECT_DEATH(static_cast<void>(Scalar(f_in(0, 0))),
                     "Ref::operator Scalar\\(\\) expects an SMEM ref");

        out = t_f;
      });
  ASSERT_THAT(kernel, IsOk());
}

TEST_F(KilnTest, DTypeProperties) {
  EXPECT_TRUE(f32.IsFloat());
  EXPECT_FALSE(f32.IsInt());
  EXPECT_FALSE(f32.IsSigned());
  EXPECT_FALSE(f32.IsUnsigned());
  EXPECT_TRUE(f32.IsIntOrFloat());
  EXPECT_EQ(f32.GetIntOrFloatBitWidth(), 32);
  EXPECT_EQ(f32.ToString(), "f32");

  std::ostringstream ss;
  ss << f32;
  EXPECT_EQ(ss.str(), "f32");

  EXPECT_FALSE(i32.IsFloat());
  EXPECT_TRUE(i32.IsInt());
  EXPECT_TRUE(i32.IsSigned());
  EXPECT_FALSE(i32.IsUnsigned());
  EXPECT_TRUE(i32.IsIntOrFloat());
  EXPECT_TRUE(i32.IsInteger(32));
  EXPECT_TRUE(i32.IsSignedInteger(32));
  EXPECT_FALSE(i32.IsUnsignedInteger(32));
  EXPECT_FALSE(i32.IsInteger(64));
  EXPECT_EQ(i32.GetIntOrFloatBitWidth(), 32);
  EXPECT_EQ(i32.ToString(), "i32");

  EXPECT_FALSE(u8.IsFloat());
  EXPECT_TRUE(u8.IsInt());
  EXPECT_FALSE(u8.IsSigned());
  EXPECT_TRUE(u8.IsUnsigned());
  EXPECT_TRUE(u8.IsUnsignedInteger(8));
  EXPECT_FALSE(u8.IsSignedInteger(8));
  EXPECT_EQ(u8.GetIntOrFloatBitWidth(), 8);
  EXPECT_EQ(u8.ToString(), "u8");

  EXPECT_TRUE(u16.IsUnsigned());
  EXPECT_TRUE(u16.IsUnsignedInteger(16));
  EXPECT_EQ(u16.GetIntOrFloatBitWidth(), 16);
  EXPECT_EQ(u16.ToString(), "u16");

  EXPECT_FALSE(u32.IsFloat());
  EXPECT_TRUE(u32.IsInt());
  EXPECT_FALSE(u32.IsSigned());
  EXPECT_TRUE(u32.IsUnsigned());
  EXPECT_TRUE(u32.IsUnsignedInteger(32));
  EXPECT_FALSE(u32.IsSignedInteger(32));
  EXPECT_EQ(u32.GetIntOrFloatBitWidth(), 32);
  EXPECT_EQ(u32.ToString(), "u32");

  EXPECT_TRUE(u64.IsUnsigned());
  EXPECT_TRUE(u64.IsUnsignedInteger(64));
  EXPECT_EQ(u64.GetIntOrFloatBitWidth(), 64);
  EXPECT_EQ(u64.ToString(), "u64");

  EXPECT_EQ(f32, DType(DType::Kind::kF32));
  EXPECT_NE(f32, i32);
  EXPECT_NE(u32, i32);
}

TEST_F(KilnTest, TypeHelpersInsideKernel) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "type_helpers", .grid = {1}}, InSpec{f32(32, 128)},
      OutSpec{f32(32, 128)}, [&](Ref in, Ref out) {
        EXPECT_TRUE(bool_.IsBool());
        EXPECT_FALSE(bool_.IsIntOrFloat());
        EXPECT_TRUE(i8.IsSignedInteger(8));
        EXPECT_TRUE(i16.IsSignedInteger(16));
        EXPECT_TRUE(i32.IsSignedInteger(32));
        EXPECT_TRUE(i64.IsSignedInteger(64));
        EXPECT_TRUE(u8.IsUnsignedInteger(8));
        EXPECT_TRUE(u16.IsUnsignedInteger(16));
        EXPECT_TRUE(u32.IsUnsignedInteger(32));
        EXPECT_TRUE(u64.IsUnsignedInteger(64));
        EXPECT_TRUE(f16.IsFloat());
        EXPECT_EQ(f16.GetIntOrFloatBitWidth(), 16);
        EXPECT_TRUE(f32.IsFloat());
        EXPECT_EQ(f32.GetIntOrFloatBitWidth(), 32);
        EXPECT_TRUE(bf16.IsFloat());
        EXPECT_EQ(bf16.GetIntOrFloatBitWidth(), 16);
        out = Tile(in);
      });
  ASSERT_THAT(kernel, IsOk());
}

TEST_F(KilnTest, DTypeInTileAndScalarOps) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "dtype_ops", .grid = {1}}, InSpec{f32(32, 128)},
      OutSpec{f32(32, 128)}, [&](Ref in, Ref out) {
        EXPECT_EQ(in.dtype(), f32);
        EXPECT_EQ(out.dtype(), f32);
        EXPECT_EQ(in.shape(), f32(32, 128));
        EXPECT_EQ(out.shape(), Shape({32, 128}, f32));

        Tile t(in);
        EXPECT_EQ(t.dtype(), f32);
        EXPECT_EQ(t.shape(), f32(32, 128));
        EXPECT_EQ(t.shape().rank(), 2);

        Tile t_bf16 = t.ToBF16();
        EXPECT_EQ(t_bf16.dtype(), bf16);
        EXPECT_EQ(t_bf16.shape(), bf16(32, 128));
        EXPECT_NE(t_bf16.shape(), t.shape());

        Tile t_i32 = t.ToI32();
        EXPECT_EQ(t_i32.dtype(), i32);
        EXPECT_EQ(t_i32.shape(), i32(in.dims()));

        Scalar s = ProgramId(0);
        EXPECT_EQ(s.dtype(), i32);

        Scalar s_f32 = s.AsType(f32);
        EXPECT_EQ(s_f32.dtype(), f32);

        Tile z = Zeros(in.shape());
        EXPECT_EQ(z.dtype(), f32);
        EXPECT_EQ(z.shape(), in.shape());

        Tile f = Full(t.shape(), /*value=*/3.0);
        EXPECT_EQ(f.shape(), f32(32, 128));

        Tile ninf = NInf(f32({32, 128}));
        EXPECT_EQ(ninf.shape(), f32(32, 128));

        out = t_bf16.ToF32();
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @dtype_ops(
    CHECK-SAME:    %[[PID:arg[0-9]+]]: i32, %[[IN:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>)
    CHECK:         %[[T:.*]] = vector.load %[[IN]]
    CHECK:         %[[T_BF16:.*]] = arith.truncf %[[T]] : vector<32x128xf32> to vector<32x128xbf16>
    CHECK:         arith.fptosi %[[T]] : vector<32x128xf32> to vector<32x128xi32>
    CHECK:         arith.sitofp %[[PID]] : i32 to f32
    CHECK:         %[[T_F32:.*]] = arith.extf %[[T_BF16]] : vector<32x128xbf16> to vector<32x128xf32>
    CHECK:         tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[T_F32]]
  )");
}

TEST_F(KilnTest, SimplifiedOverloadsAndBounds) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "simplified_ops", .grid = {1}}, InSpec{f32(32, 128)},
      OutSpec{f32(32, 128)}, [&](Ref in, Ref out) {
        Scalar zero = I32(0);
        Scalar ten = I32(10);

        // For with (Scalar, int64_t) - previously unsupported combination.
        For(zero, 10, [&](Scalar i) { If(i == 0, [&] { out = Tile(in); }); });

        // For with (int64_t, Scalar).
        For(0, ten, [&](Scalar i) { If(i == 5, [&] { out = Tile(in); }); });

        // Bitwise shifts with Literal on LHS: 1 << i, 1024 >> i.
        Scalar shift_val = I32(2);
        Scalar shl = 1 << shift_val;
        Scalar shr = 1024 >> shift_val;
        EXPECT_EQ(shl.dtype(), i32);
        EXPECT_EQ(shr.dtype(), i32);

        // Min/Max with literals on both sides.
        Scalar min1 = Min(shift_val, 1);
        Scalar min2 = Min(1, shift_val);
        Scalar max1 = Max(shift_val, 10);
        Scalar max2 = Max(10, shift_val);
        EXPECT_EQ(min1.dtype(), i32);
        EXPECT_EQ(min2.dtype(), i32);
        EXPECT_EQ(max1.dtype(), i32);
        EXPECT_EQ(max2.dtype(), i32);

        // Select with literals.
        Cond cond = shift_val == 2;
        Scalar sel1 = Select(cond, shift_val, 0);
        Scalar sel2 = Select(cond, 0, shift_val);
        EXPECT_EQ(sel1.dtype(), i32);
        EXPECT_EQ(sel2.dtype(), i32);

        // Slicing with IndexArg and 0-arg operator().
        Ref in_full = in();
        EXPECT_EQ(in_full.rank(), 2);
        Ref in_slice = in(zero, all);
        Ref in_ds = in(0, Slice(zero, 64));
        EXPECT_EQ(in_slice.rank(), 1);
        EXPECT_EQ(in_ds.rank(), 1);
        Ref in_2d = in(all, Slice(zero, 64));
        EXPECT_EQ(in_2d.rank(), 2);

        // Tile modulo with Tile and Literal.
        Tile iota = Iota(i32(32, 128), /*axis=*/1);
        Tile mod1 = iota % 4;
        Tile mod2 = 100 % (iota + 1);
        Tile mod3 = mod1 % mod2;
        EXPECT_EQ(mod3.dtype(), i32);
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @simplified_ops(
    CHECK-SAME:    %{{.*}}: i32, %[[IN:arg[0-9]+]]: memref<32x128xf32, #tpu.memory_space<vmem>>, %{{.*}}: memref<32x128xf32, #tpu.memory_space<vmem>>)
    CHECK:         %[[ZERO:.*]] = arith.constant 0 : i32
    CHECK:         %[[TEN:.*]] = arith.constant 10 : i32

    // For(Scalar, int64_t): only the Scalar bound needs an index_cast.
    CHECK:         %[[LB:.*]] = arith.index_cast %[[ZERO]] : i32 to index
    CHECK:         %[[C10:.*]] = arith.constant 10 : index
    CHECK:         scf.for %{{.*}} = %[[LB]] to %[[C10]] step %{{.*}} {

    // For(int64_t, Scalar).
    CHECK:         %[[UB:.*]] = arith.index_cast %[[TEN]] : i32 to index
    CHECK:         scf.for %{{.*}} = %{{.*}} to %[[UB]] step %{{.*}} {

    // Literal on the LHS of a shift.
    CHECK:         %[[TWO:.*]] = arith.constant 2 : i32
    CHECK:         %[[ONE:.*]] = arith.constant 1 : i32
    CHECK:         arith.shli %[[ONE]], %[[TWO]] : i32
    CHECK:         %[[C1024:.*]] = arith.constant 1024 : i32
    CHECK:         arith.shrsi %[[C1024]], %[[TWO]] : i32

    // Min/Max/Select with a literal on either side.
    CHECK:         arith.minsi %[[TWO]], %{{.*}} : i32
    CHECK:         arith.minsi %{{.*}}, %[[TWO]] : i32
    CHECK:         arith.maxsi %[[TWO]], %{{.*}} : i32
    CHECK:         arith.maxsi %{{.*}}, %[[TWO]] : i32
    CHECK:         %[[COND:.*]] = arith.cmpi eq, %[[TWO]], %{{.*}} : i32
    CHECK:         arith.select %[[COND]], %[[TWO]], %{{.*}} : i32
    CHECK:         arith.select %[[COND]], %{{.*}}, %[[TWO]] : i32

    // Ref slicing with an IndexArg (dynamic index / dynamic slice start).
    CHECK:         %[[ROW:.*]] = tpu.memref_slice %[[IN]][%[[ZERO]], %{{.*}}] : memref<32x128xf32, #tpu.memory_space<vmem>> -> memref<1x128xf32, #tpu.memory_space<vmem>>
    CHECK:         tpu.memref_squeeze %[[ROW]] : memref<1x128xf32, #tpu.memory_space<vmem>> -> memref<128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[DS:.*]] = tpu.memref_slice %[[IN]][%{{.*}}, %[[ZERO]]] : memref<32x128xf32, #tpu.memory_space<vmem>> -> memref<1x64xf32, #tpu.memory_space<vmem>>
    CHECK:         tpu.memref_squeeze %[[DS]] : memref<1x64xf32, #tpu.memory_space<vmem>> -> memref<64xf32, #tpu.memory_space<vmem>>
    CHECK:         tpu.memref_slice %[[IN]][%{{.*}}, %[[ZERO]]] : memref<32x128xf32, #tpu.memory_space<vmem>> -> memref<32x64xf32, #tpu.memory_space<vmem>>

    // Tile modulo with a Tile or Literal on either side.
    CHECK:         %[[IOTA:.*]] = tpu.iota {dimensions = array<i32: 1>} : vector<32x128xi32>
    CHECK:         %[[C4:.*]] = arith.constant dense<4> : vector<32x128xi32>
    CHECK:         %[[MOD1:.*]] = arith.remsi %[[IOTA]], %[[C4]] : vector<32x128xi32>
    CHECK:         %[[PLUS1:.*]] = arith.addi %[[IOTA]], %{{.*}} : vector<32x128xi32>
    CHECK:         %[[C100:.*]] = arith.constant dense<100> : vector<32x128xi32>
    CHECK:         %[[MOD2:.*]] = arith.remsi %[[C100]], %[[PLUS1]] : vector<32x128xi32>
    CHECK:         arith.remsi %[[MOD1]], %[[MOD2]] : vector<32x128xi32>
  )");
}

TEST_F(KilnTest, UnsignedOperations) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "unsigned_ops", .grid = {1}}, InSpec{u32(32, 128)},
      InSpec{u32(32, 128)}, OutSpec{u32(32, 128)},
      [&](Ref in_a, Ref in_b, Ref out) {
        Tile a(in_a);
        Tile b(in_b);
        EXPECT_TRUE(a.dtype().IsUnsigned());
        EXPECT_FALSE(a.dtype().IsSigned());
        EXPECT_TRUE(b.dtype().IsUnsigned());

        // Arithmetic
        Tile div = a / b;
        Tile rem = a % b;
        Tile min_val = Min(a, b);
        Tile max_val = Max(a, b);

        // Shifts: >> on unsigned should be logical right shift (shrui).
        Tile shr = a >> b;

        // Comparisons
        Tile lt = a < b;
        Tile le = a <= b;
        Tile gt = a > b;
        Tile ge = a >= b;
        Tile eq = a == b;
        Tile ne = a != b;

        // Select
        Tile sel = Select(lt, a, b) + Select(le, a, b) + Select(gt, a, b) +
                   Select(ge, a, b) + Select(eq, a, b) + Select(ne, a, b);

        // MaxReduce on unsigned uses MAXUI and 0 init
        Tile max_red = MaxReduce(a, 1);
        Tile red_lane = max_red.Lanes(128);

        // Abs on unsigned is identity
        Tile abs_a = Abs(a);

        // Scalar unsigned
        Scalar s_u = U32(42);
        EXPECT_TRUE(s_u.dtype().IsUnsigned());
        Scalar s_div = s_u / U32(2);
        Scalar s_rem = s_u % U32(5);
        Scalar s_shr = s_u >> U32(1);
        Cond s_lt = s_u < U32(100);

        Tile s_tile = Splat(s_div + s_rem + s_shr, {32, 128});
        If(s_lt, [&] {
          out = sel + red_lane + div + rem + min_val + max_val + shr + abs_a +
                s_tile;
        });
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @unsigned_ops(
    CHECK-SAME:    %{{.*}}: i32, %[[A:arg[0-9]+]]: memref<32x128xi32, #tpu.memory_space<vmem>>, %[[B:arg[0-9]+]]: memref<32x128xi32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<32x128xi32, #tpu.memory_space<vmem>>)
    CHECK-DAG:     %[[VA:.*]] = vector.load %[[A]]
    CHECK-DAG:     %[[VB:.*]] = vector.load %[[B]]
    CHECK:         arith.divui %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.remui %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.minui %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.maxui %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.shrui %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.cmpi ult, %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.cmpi ule, %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.cmpi ugt, %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.cmpi uge, %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.cmpi eq, %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.cmpi ne, %[[VA]], %[[VB]] : vector<32x128xi32>
    CHECK:         arith.select %{{.*}}, %[[VA]], %[[VB]] : vector<32x128xi1>, vector<32x128xi32>
    CHECK:         %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
    CHECK:         vector.multi_reduction <maxui>, %[[VA]], %[[ZERO]] [1] : vector<32x128xi32> to vector<32xi32>
    CHECK:         %[[C42:.*]] = arith.constant 42 : i32
    CHECK:         %[[C2:.*]] = arith.constant 2 : i32
    CHECK:         arith.divui %[[C42]], %[[C2]] : i32
    CHECK:         %[[C5:.*]] = arith.constant 5 : i32
    CHECK:         arith.remui %[[C42]], %[[C5]] : i32
    CHECK:         %[[C1:.*]] = arith.constant 1 : i32
    CHECK:         arith.shrui %[[C42]], %[[C1]] : i32
    CHECK:         %[[C100:.*]] = arith.constant 100 : i32
    CHECK:         arith.cmpi ult, %[[C42]], %[[C100]] : i32
  )");
}

TEST_F(KilnTest, UnsignedAndSignedCasts) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "unsigned_casts", .grid = {1}}, InSpec{f32(32, 128)},
      InSpec{u8(32, 128)}, InSpec{i8(32, 128)}, OutSpec{f32(32, 128)},
      [&](Ref in_f, Ref in_u8, Ref in_i8, Ref out) {
        Tile f(in_f);
        Tile u(in_u8);
        Tile i(in_i8);

        // Float to Unsigned vs Signed
        Tile f_to_u32 = f.AsType(u32);
        Tile f_to_i32 = f.AsType(i32);

        // Unsigned vs Signed to Float
        Tile u_to_f = u.AsType(f32);
        Tile i_to_f = i.AsType(f32);

        // Extension: extui vs extsi
        Tile u8_to_u32 = u.AsType(u32);
        Tile i8_to_i32 = i.AsType(i32);

        // Bitcast between same-width int types (identity)
        Tile u_bitcast_i = f_to_u32.Bitcast(i32);
        EXPECT_EQ(u_bitcast_i.dtype(), i32);

        out = f_to_u32.AsType(f32) + f_to_i32.AsType(f32) + u_to_f + i_to_f +
              u8_to_u32.AsType(f32) + i8_to_i32.AsType(f32);
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @unsigned_casts(
    CHECK:         %[[F:.*]] = vector.load %{{.*}} : memref<32x128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[U8:.*]] = vector.load %{{.*}} : memref<32x128xi8, #tpu.memory_space<vmem>>
    CHECK:         %[[I8:.*]] = vector.load %{{.*}} : memref<32x128xi8, #tpu.memory_space<vmem>>
    CHECK:         arith.fptoui %[[F]] : vector<32x128xf32> to vector<32x128xi32>
    CHECK:         arith.fptosi %[[F]] : vector<32x128xf32> to vector<32x128xi32>
    CHECK:         arith.uitofp %[[U8]] : vector<32x128xi8> to vector<32x128xf32>
    CHECK:         arith.sitofp %[[I8]] : vector<32x128xi8> to vector<32x128xf32>
    CHECK:         arith.extui %[[U8]] : vector<32x128xi8> to vector<32x128xi32>
    CHECK:         arith.extsi %[[I8]] : vector<32x128xi8> to vector<32x128xi32>
  )");
}

TEST_F(KilnTest, DynamicMemrefSliceSizesAndAll) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "dynamic_slices"},
      InSpec::Vmem(f32(kDynamic, kDynamic, 128)), OutSpec::Vmem(f32(8, 128)),
      [&](Ref dyn_in, Ref /*vmem_out*/) {
        EXPECT_EQ(dyn_in.memory_space(), MemorySpace::kVmem);
        EXPECT_FALSE(dyn_in.HasStaticShape());
        EXPECT_TRUE(dyn_in.IsDynamicDim(0));
        EXPECT_TRUE(dyn_in.IsDynamicDim(1));
        EXPECT_FALSE(dyn_in.IsDynamicDim(2));

        Scalar batch = dyn_in.Dim(0);
        Scalar rows = dyn_in.Dim(1);
        Scalar static_cols = dyn_in.Dim(2);
        Scalar half_rows = MultipleOf((rows >> 1), 8);
        Scalar aligned_cols = MultipleOf(static_cols, 128);

        // Identity slice on dynamic ref does not emit memref_slice.
        Ref full = dyn_in(all, all, all);
        EXPECT_FALSE(full.HasStaticShape());

        // `all` on untiled leading dynamic dim 0 + dynamic slice size on dim 1
        // + `all` on static dim 2.
        Ref sub_in = dyn_in(all, Slice(0, half_rows), all);
        EXPECT_TRUE(sub_in.IsDynamicDim(0));
        EXPECT_TRUE(sub_in.IsDynamicDim(1));
        EXPECT_FALSE(sub_in.IsDynamicDim(2));
        EXPECT_EQ(sub_in.rank(), 3);

        // Explicit dynamic slice sizes on all 3 dimensions.
        Ref sub_in2 = dyn_in(Slice(0, batch), Slice(0, half_rows),
                             Slice(0, aligned_cols));
        EXPECT_TRUE(sub_in2.IsDynamicDim(0));
        EXPECT_TRUE(sub_in2.IsDynamicDim(1));
        EXPECT_TRUE(sub_in2.IsDynamicDim(2));
        EXPECT_EQ(sub_in2.rank(), 3);
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @dynamic_slices(
    CHECK-SAME:    %[[IN:arg[0-9]+]]: memref<?x?x128xf32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<8x128xf32, #tpu.memory_space<vmem>>)
    CHECK:         %[[C0_IDX:.*]] = arith.constant 0 : index
    CHECK:         %[[DIM0:.*]] = memref.dim %[[IN]], %[[C0_IDX]] : memref<?x?x128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[BATCH:.*]] = arith.index_cast %[[DIM0]] : index to i32
    CHECK:         %[[C1_IDX:.*]] = arith.constant 1 : index
    CHECK:         %[[DIM1:.*]] = memref.dim %[[IN]], %[[C1_IDX]] : memref<?x?x128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[ROWS:.*]] = arith.index_cast %[[DIM1]] : index to i32
    CHECK:         %[[STATIC_COLS:.*]] = arith.constant 128 : i32
    CHECK-NOT:     memref.dim
    CHECK:         %[[HALF:.*]] = arith.shrsi %[[ROWS]], %{{.*}} : i32
    CHECK:         %[[HALF_ROWS:.*]] = tpu.assume_multiple %[[HALF]], 8 : i32
    CHECK:         %[[ALIGNED_COLS:.*]] = tpu.assume_multiple %[[STATIC_COLS]], 128 : i32
    CHECK:         %[[SLICE0_DIM0:.*]] = memref.dim %[[IN]], %{{.*}} : memref<?x?x128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[SLICE0_DYN0:.*]] = arith.index_cast %[[SLICE0_DIM0]] : index to i32
    CHECK:         tpu.memref_slice %[[IN]][%{{.*}}, %{{.*}}, %{{.*}}] <%[[SLICE0_DYN0]], %[[HALF_ROWS]]> : memref<?x?x128xf32, #tpu.memory_space<vmem>> -> memref<?x?x128xf32, #tpu.memory_space<vmem>>
    CHECK:         tpu.memref_slice %[[IN]][%{{.*}}, %{{.*}}, %{{.*}}] <%[[BATCH]], %[[HALF_ROWS]], %[[ALIGNED_COLS]]> : memref<?x?x128xf32, #tpu.memory_space<vmem>> -> memref<?x?x?xf32, #tpu.memory_space<vmem>>
    CHECK-NOT:     tpu.memref_slice
  )");
}

TEST_F(KilnTest, SteppedForScopedAllocaAndUnsignedBounds) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "stepped_for_alloca"}, InSpec::Vmem(f32(kDynamic, 128)),
      OutSpec::Vmem(f32(8, 128)), [&](Ref dyn_in, Ref vmem_out) {
        Ref local_smem = AllocaSmem(i32(1));
        Ref scalar_smem = AllocaSmem(i32());
        EXPECT_EQ(local_smem.memory_space(), MemorySpace::kSmem);
        EXPECT_TRUE(local_smem.HasStaticShape());
        EXPECT_EQ(scalar_smem.rank(), 0);
        EXPECT_EQ(scalar_smem.shape(), i32());

        Scalar rows = dyn_in.Dim(0);
        For(0, rows, 8, [&](Scalar r) {
          Scalar aligned_r = MultipleOf(r, 8);
          RunScoped([&] {
            Ref scratch_tile = AllocaVmem(f32(8, 128));
            EXPECT_EQ(scratch_tile.memory_space(), MemorySpace::kVmem);
            Ref tile_in = dyn_in(Slice(aligned_r, 8), all);
            EXPECT_TRUE(tile_in.HasStaticShape());
            scratch_tile = Tile(tile_in);
            vmem_out = Tile(scratch_tile);
          });
        });

        For(U32(0), U32(8), U32(4), [&](Scalar idx) { local_smem(0) = idx; });
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @stepped_for_alloca(
    CHECK-SAME:    %[[IN:arg[0-9]+]]: memref<?x128xf32, #tpu.memory_space<vmem>>, %[[OUT:arg[0-9]+]]: memref<8x128xf32, #tpu.memory_space<vmem>>)
    CHECK:         %[[SMEM:.*]] = memref.alloca() : memref<1xi32, #tpu.memory_space<smem>>
    CHECK:         %[[DIM0:.*]] = memref.dim %[[IN]], %{{.*}} : memref<?x128xf32, #tpu.memory_space<vmem>>
    CHECK:         %[[ROWS:.*]] = arith.index_cast %[[DIM0]] : index to i32
    CHECK:         %[[LB:.*]] = arith.constant 0 : index
    CHECK:         %[[UB:.*]] = arith.index_cast %[[ROWS]] : i32 to index
    CHECK:         %[[STEP:.*]] = arith.constant 8 : index
    CHECK:         scf.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
    CHECK:           %[[R:.*]] = arith.index_cast %[[IV]] : index to i32
    CHECK:           %[[ALIGNED_R:.*]] = tpu.assume_multiple %[[R]], 8 : i32
    CHECK:           "tpu.region"() ({
    CHECK:             %[[SCRATCH:.*]] = memref.alloca() : memref<8x128xf32, #tpu.memory_space<vmem>>
    CHECK:             %[[TILE_IN:.*]] = tpu.memref_slice %[[IN]][%[[ALIGNED_R]], %{{.*}}] : memref<?x128xf32, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
    CHECK:             %[[V:.*]] = vector.load %[[TILE_IN]]
    CHECK:             tpu.vector_store %[[SCRATCH]][%{{.*}}, %{{.*}}], %[[V]]
    CHECK:             %[[V2:.*]] = vector.load %[[SCRATCH]]
    CHECK:             tpu.vector_store %[[OUT]][%{{.*}}, %{{.*}}], %[[V2]]
    CHECK:             tpu.yield
    CHECK:           }) : () -> ()
    CHECK:         }
    CHECK:         %[[ULB:.*]] = arith.index_castui %{{.*}} : i32 to index
    CHECK:         %[[UUB:.*]] = arith.index_castui %{{.*}} : i32 to index
    CHECK:         %[[USTEP:.*]] = arith.index_castui %{{.*}} : i32 to index
    CHECK:         scf.for %{{.*}} = %[[ULB]] to %[[UUB]] step %[[USTEP]] {
  )");
}

TEST_F(KilnTest, DynamicShapeSliceAndAllocaValidationErrors) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "validation_errors"}, InSpec::Vmem(f32(kDynamic, 128)),
      OutSpec::Vmem(f32(8, 128)), [&](Ref dyn_in, Ref vmem_out) {
        Scalar rows = dyn_in.Dim(0);
        EXPECT_DEATH(static_cast<void>(Tile(dyn_in)),
                     "Tile load requires a Ref with a static shape");
        EXPECT_DEATH(dyn_in.Store(Tile(vmem_out)),
                     "Tile store requires a Ref with a static shape");
        EXPECT_DEATH(static_cast<void>(dyn_in.IsDynamicDim(-1)),
                     "Dimension axis must be non-negative");
        EXPECT_DEATH(static_cast<void>(dyn_in.IsDynamicDim(2)),
                     "Dimension axis 2 out of bounds for rank 2");
        EXPECT_DEATH(static_cast<void>(dyn_in.Dim(2)),
                     "Dimension axis 2 out of bounds for rank 2");
        EXPECT_DEATH(static_cast<void>(MultipleOf(F32(1.0), 8)),
                     "MultipleOf\\(\\) requires an integer scalar");
        EXPECT_DEATH(static_cast<void>(MultipleOf(rows, 0)),
                     "MultipleOf\\(\\) multiple must be positive");
        EXPECT_DEATH(static_cast<void>(dyn_in(Slice(0, kDynamic), all)),
                     "Slice size cannot be kDynamic");
        EXPECT_DEATH(static_cast<void>(dyn_in(Slice(0, I32(0)), all)),
                     "Slice size must be positive");
        EXPECT_DEATH(static_cast<void>(dyn_in(Slice(I32(-1), 8), all)),
                     "Slice offset must be non-negative");
        EXPECT_DEATH(static_cast<void>(dyn_in(Slice(0, 1LL << 40), all)),
                     "Slice size exceeds int32_t max");
        EXPECT_DEATH(static_cast<void>(dyn_in(Slice(1LL << 40, 8), all)),
                     "Slice offset exceeds int32_t max");
        EXPECT_DEATH(static_cast<void>(dyn_in(Slice(0, U32(0x80000000u)), all)),
                     "Slice size exceeds int32_t max");
        EXPECT_DEATH(static_cast<void>(dyn_in(
                         all, Slice(std::numeric_limits<int32_t>::max(), 8))),
                     "Slice offset \\+ size must not exceed dimension size");
        EXPECT_DEATH(For(0, 16, 0, [&](Scalar) {}),
                     "Loop step must be positive");
        EXPECT_DEATH(For(0, 16, I32(0), [&](Scalar) {}),
                     "Loop step must be positive");
        EXPECT_DEATH(static_cast<void>(AllocaSmem(i32(0))),
                     "Alloca\\(\\) requires static positive dimensions");
        EXPECT_DEATH(static_cast<void>(AllocaSmem(bf16())),
                     "SMEM Alloca\\(\\) requires a 32-bit element type");
        EXPECT_DEATH(static_cast<void>(AllocaSmem(i16(1))),
                     "SMEM Alloca\\(\\) requires a 32-bit element type");
        EXPECT_DEATH(static_cast<void>(AllocaVmem(f32())),
                     "VMEM Alloca\\(\\) requires at least a 1D shape");
        EXPECT_DEATH(static_cast<void>(AllocaVmem(bool_(8, 128))),
                     "VMEM Alloca\\(\\) does not support bool");
        EXPECT_DEATH(static_cast<void>(AllocaVmem(i64(8, 128))),
                     "VMEM Alloca\\(\\) does not support bool \\(i1\\) or "
                     "64-bit element types");
      });
  ASSERT_THAT(kernel, IsOk());

  // Verify BuilderScope::~BuilderScope() restores tls_builder to nullptr after
  // BuildKernel() completes.
  EXPECT_DEATH(static_cast<void>(I32(1)),
               "Kiln operations may only be called inside BuildKernel");
}

TEST_F(KilnTest, SemaphoreAndBarrierOperations) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "semaphores", .grid = {1}}, InSpec::Hbm(f32(8, 128)),
      OutSpec::Vmem(f32(8, 128)), [&](Ref, Ref) {
        Scalar dst_dev = (ProgramId(0) + 1) % 2;

        Semaphore dma_sem0 = AllocaDmaSemaphore();
        SemaphoreArray dma_sems = AllocaDmaSemaphore({2});
        Semaphore dma_sem = dma_sems(0);
        Semaphore sync_sem = AllocaSemaphore();
        EXPECT_TRUE(dma_sem0.is_dma());
        EXPECT_EQ(dma_sems.rank(), 1);
        EXPECT_EQ(dma_sems.shape()[0], 2);
        EXPECT_TRUE(dma_sems.is_dma());
        EXPECT_TRUE(dma_sem.is_dma());
        EXPECT_FALSE(sync_sem.is_dma());

        Semaphore barrier_sem = GetBarrierSemaphore();
        EXPECT_FALSE(barrier_sem.is_dma());
        barrier_sem.Signal(1, {.device_id = dst_dev, .core_id = 0});
        barrier_sem.Wait(1);

        sync_sem.Signal(2);
        Scalar count = sync_sem.Read();
        sync_sem.Wait(count);
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @semaphores(
    CHECK-SAME:    %[[PID:arg0]]: i32, %[[HBM_IN:arg1]]: memref<8x128xf32, #tpu.memory_space<hbm>>, %[[VMEM_OUT:arg2]]: memref<8x128xf32, #tpu.memory_space<vmem>>)
    CHECK-SAME:    window_params = [{}, {transform_indices = affine_map<(d0) -> (0, 0)>}]
    CHECK:         %[[C1_PID:.*]] = arith.constant 1 : i32
    CHECK:         %[[NEXT_DEV:.*]] = arith.addi %[[PID]], %[[C1_PID]] : i32
    CHECK:         %[[C2_MOD:.*]] = arith.constant 2 : i32
    CHECK:         %[[DST_DEV:.*]] = arith.remsi %[[NEXT_DEV]], %[[C2_MOD]] : i32
    CHECK:         %[[DMA_SEM0:.*]] = tpu.sem_alloc : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[DMA_SEMS:.*]] = tpu.sem_alloc : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[C0_SLICE:.*]] = arith.constant 0 : i32
    CHECK:         %[[DMA_SLICE:.*]] = tpu.memref_slice %[[DMA_SEMS]][%[[C0_SLICE]]] : memref<2x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[DMA_SEM:.*]] = tpu.memref_squeeze %[[DMA_SLICE]] : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[SYNC_SEM:.*]] = tpu.sem_alloc : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[BARRIER_SEM:.*]] = tpu.sem_barrier : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[SIG1_AMT:.*]] = arith.constant 1 : i32
    CHECK:         %[[CORE0:.*]] = arith.constant 0 : i32
    CHECK:         tpu.sem_signal %[[BARRIER_SEM]], %[[SIG1_AMT]] device_id %[[DST_DEV]] core_id %[[CORE0]] : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[WAIT1_AMT:.*]] = arith.constant 1 : i32
    CHECK:         tpu.sem_wait %[[BARRIER_SEM]], %[[WAIT1_AMT]] : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[SIG2_AMT:.*]] = arith.constant 2 : i32
    CHECK:         tpu.sem_signal %[[SYNC_SEM]], %[[SIG2_AMT]] : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[COUNT:.*]] = tpu.sem_read %[[SYNC_SEM]] : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>> -> i32
    CHECK:         tpu.sem_wait %[[SYNC_SEM]], %[[COUNT]] : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
  )");
}

TEST_F(KilnTest, HbmAndSemaphoreValidationErrors) {
  // HBM input/output specs with index_map are rejected by BuildKernel.
  absl::StatusOr<Kernel> bad_hbm_in_map = CreateKernel(
      GridSpec{.name = "bad_hbm_in", .grid = {1}},
      InSpec{f32(8, 128), IndexMap([] { return IndexIndices{0, 0}; }),
             MemorySpace::kHbm},
      OutSpec::Vmem(f32(8, 128)), [&](Ref, Ref) {});
  EXPECT_FALSE(bad_hbm_in_map.ok());
  EXPECT_THAT(bad_hbm_in_map.status().message(),
              testing::HasSubstr("HBM operands do not support index_map"));

  absl::StatusOr<Kernel> bad_hbm_out_map = CreateKernel(
      GridSpec{.name = "bad_hbm_out", .grid = {1}}, InSpec::Vmem(f32(8, 128)),
      OutSpec{f32(8, 128), IndexMap([] { return IndexIndices{0, 0}; }),
              MemorySpace::kHbm},
      [&](Ref, Ref) {});
  EXPECT_FALSE(bad_hbm_out_map.ok());
  EXPECT_THAT(bad_hbm_out_map.status().message(),
              testing::HasSubstr("HBM operands do not support index_map"));

  absl::StatusOr<Kernel> bad_hbm_scratch = CreateKernel(
      GridSpec{.name = "bad_hbm_scratch", .grid = {1}},
      InSpec::Vmem(f32(8, 128)), OutSpec::Vmem(f32(8, 128)),
      ScratchSpec{f32(8, 128), MemorySpace::kHbm}, [&](Ref, Ref, Ref) {});
  EXPECT_FALSE(bad_hbm_scratch.ok());
  EXPECT_THAT(bad_hbm_scratch.status().message(),
              testing::HasSubstr("Scratch operands cannot use HBM"));

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "hbm_sem_errors"}, InSpec::Hbm(f32(kDynamic, 128)),
      OutSpec::Vmem(f32(8, 128)), [&](Ref hbm_in, Ref vmem_out) {
        SemaphoreArray dma_sems = AllocaDmaSemaphore({2});
        SemaphoreArray reg_sems = AllocaSemaphore({2, 3});
        Semaphore reg_sem = AllocaSemaphore();
        EXPECT_EQ(reg_sems.rank(), 2);
        EXPECT_FALSE(reg_sems.is_dma());
        Ref hbm_tile = hbm_in(Slice(0, 8), all);
        EXPECT_DEATH(static_cast<void>(
                         Alloca(ScratchSpec{f32(8, 128), MemorySpace::kHbm})),
                     "Cannot allocate HBM in a kernel");
        EXPECT_DEATH(
            static_cast<void>(Alloca(ScratchSpec{f32(0), MemorySpace::kHbm})),
            "Cannot allocate HBM in a kernel");
        EXPECT_DEATH(static_cast<void>(AllocaDmaSemaphore(Dimensions{})),
                     "AllocaDmaSemaphore\\(shape\\) requires a non-empty "
                     "shape");
        EXPECT_DEATH(static_cast<void>(AllocaSemaphore(Dimensions{})),
                     "AllocaSemaphore\\(shape\\) requires a non-empty shape");
        EXPECT_DEATH(static_cast<void>(AllocaDmaSemaphore({0})),
                     "AllocaDmaSemaphore\\(shape\\) requires static positive "
                     "dimensions");
        EXPECT_DEATH(static_cast<void>(AllocaSemaphore({-1})),
                     "AllocaSemaphore\\(shape\\) requires static positive "
                     "dimensions");
        EXPECT_DEATH(static_cast<void>(reg_sems(0)),
                     "SemaphoreArray indexing requires exactly 2 indices");
        EXPECT_DEATH(static_cast<void>(dma_sems(0, 0)),
                     "SemaphoreArray indexing requires exactly 1 indices");
        EXPECT_DEATH(static_cast<void>(dma_sems(2)),
                     "Slice offset \\+ size must not exceed dimension size");
        EXPECT_DEATH(static_cast<void>(Tile(hbm_tile)),
                     "Tile load requires a VMEM ref");
        EXPECT_DEATH((hbm_tile = Tile(vmem_out)),
                     "Tile store requires a VMEM ref");
        EXPECT_DEATH(static_cast<void>(Tile(hbm_in)),
                     "Tile load requires a Ref with a static shape");
        EXPECT_DEATH(dma_sems(0).Signal(1),
                     "Semaphore::Signal\\(\\) requires a regular semaphore");
        EXPECT_DEATH(dma_sems(0).Wait(1),
                     "Semaphore::Wait\\(\\) requires a regular semaphore");
        EXPECT_DEATH(reg_sem.Signal(0),
                     "Semaphore::Signal\\(\\) amount must be positive");
        EXPECT_DEATH(reg_sem.Wait(0),
                     "Semaphore::Wait\\(\\) amount must be positive");
        EXPECT_DEATH(reg_sem.Signal(1, {.device_id = -1}),
                     "SemaphoreSignalOptions::device_id must be non-negative");
        EXPECT_DEATH(reg_sem.Signal(I32(1).AsType(i64)),
                     "Semaphore::Signal\\(\\) amount scalar bitwidth must be "
                     "<= 32");
      });
  ASSERT_THAT(kernel, IsOk());
}

TEST_F(KilnTest, DmaOperations) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "dma_ops"}, InSpec::Hbm(f32(8, 128)),
      OutSpec::Vmem(f32(8, 128)), [&](Ref hbm_in, Ref vmem_out) {
        Scalar dev_id = DeviceId();
        Semaphore sem = AllocaDmaSemaphore();
        Semaphore send_sem = AllocaDmaSemaphore();
        Semaphore recv_sem = AllocaDmaSemaphore();

        // Local AsyncCopy (immediate start + Wait).
        AsyncCopy(hbm_in, vmem_out, sem).Wait();

        // Local MakeAsyncCopy with non-default priority.
        AsyncCopyDescriptor local =
            MakeAsyncCopy(vmem_out, hbm_in, sem, {.priority = 1});
        EXPECT_EQ(local.src_ref().memory_space(), MemorySpace::kVmem);
        EXPECT_EQ(local.dst_ref().memory_space(), MemorySpace::kHbm);
        EXPECT_TRUE(local.sem().is_dma());
        EXPECT_EQ(local.options().priority, 1);
        local.Start();
        local.Wait();

        // Remote AsyncRemoteCopy with explicit WaitSend + WaitRecv.
        AsyncRemoteCopyDescriptor remote = AsyncRemoteCopy(
            hbm_in, vmem_out, send_sem, recv_sem, {.device_id = dev_id});
        EXPECT_EQ(remote.src_ref().memory_space(), MemorySpace::kHbm);
        EXPECT_EQ(remote.dst_ref().memory_space(), MemorySpace::kVmem);
        EXPECT_TRUE(remote.send_sem().is_dma());
        EXPECT_TRUE(remote.recv_sem().is_dma());
        EXPECT_TRUE(remote.device_id().has_value());
        EXPECT_FALSE(remote.core_id().has_value());
        remote.WaitSend();
        remote.WaitRecv();

        // Remote MakeAsyncRemoteCopy with both device_id and core_id + Wait.
        AsyncRemoteCopyDescriptor remote2 =
            MakeAsyncRemoteCopy(hbm_in, vmem_out, send_sem, recv_sem,
                                {.device_id = dev_id, .core_id = 0});
        EXPECT_TRUE(remote2.device_id().has_value());
        EXPECT_TRUE(remote2.core_id().has_value());
        remote2.Start();
        remote2.Wait();

        // Local MakeAsyncCopy with a regular semaphore + Semaphore::Wait.
        Semaphore reg_sem = AllocaSemaphore();
        MakeAsyncCopy(hbm_in, vmem_out, reg_sem).Start();
        reg_sem.Wait(1);

        // Remote MakeAsyncRemoteCopy with only core_id and regular semaphores.
        Semaphore reg_send_sem = AllocaSemaphore();
        Semaphore reg_recv_sem = AllocaSemaphore();
        AsyncRemoteCopyDescriptor remote_core_only = MakeAsyncRemoteCopy(
            hbm_in, vmem_out, reg_send_sem, reg_recv_sem, {.core_id = 1});
        EXPECT_FALSE(remote_core_only.device_id().has_value());
        EXPECT_TRUE(remote_core_only.core_id().has_value());
        EXPECT_FALSE(remote_core_only.send_sem().is_dma());
        EXPECT_FALSE(remote_core_only.recv_sem().is_dma());
        remote_core_only.Start();
        reg_send_sem.Wait(1);
        reg_recv_sem.Wait(1);
      });
  ASSERT_THAT(kernel, IsOk());
  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @dma_ops(
    CHECK-SAME:    %[[HBM:arg0]]: memref<8x128xf32, #tpu.memory_space<hbm>>, %[[VMEM:arg1]]: memref<8x128xf32, #tpu.memory_space<vmem>>)
    CHECK:         %[[DEV:.*]] = tpu.device_id : i32
    CHECK:         %[[SEM:.*]] = tpu.sem_alloc : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[SEND_SEM:.*]] = tpu.sem_alloc : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[RECV_SEM:.*]] = tpu.sem_alloc : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         tpu.enqueue_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) target_semaphore(%[[SEM]] : {{.*}})
    CHECK:         tpu.wait_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) target_semaphore(%[[SEM]] : {{.*}}) {wait_target = true}
    CHECK:         tpu.enqueue_dma source(%[[VMEM]] : {{.*}}) target(%[[HBM]] : {{.*}}) target_semaphore(%[[SEM]] : {{.*}}) {priority = 1 : i32}
    CHECK:         tpu.wait_dma source(%[[VMEM]] : {{.*}}) target(%[[HBM]] : {{.*}}) target_semaphore(%[[SEM]] : {{.*}}) {priority = 1 : i32, wait_target = true}
    CHECK:         tpu.enqueue_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) source_semaphore(%[[SEND_SEM]] : {{.*}}) target_semaphore(%[[RECV_SEM]] : {{.*}}) device_id(%[[DEV]])
    CHECK:         tpu.wait_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) source_semaphore(%[[SEND_SEM]] : {{.*}}) target_semaphore(%[[RECV_SEM]] : {{.*}}) device_id(%[[DEV]]) {wait_target = false}
    CHECK:         tpu.wait_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) source_semaphore(%[[SEND_SEM]] : {{.*}}) target_semaphore(%[[RECV_SEM]] : {{.*}}) device_id(%[[DEV]]) {wait_target = true}
    CHECK:         %[[CORE0:.*]] = arith.constant 0 : i32
    CHECK:         tpu.enqueue_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) source_semaphore(%[[SEND_SEM]] : {{.*}}) target_semaphore(%[[RECV_SEM]] : {{.*}}) device_id(%[[DEV]]) core_id(%[[CORE0]])
    CHECK:         tpu.wait_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) source_semaphore(%[[SEND_SEM]] : {{.*}}) target_semaphore(%[[RECV_SEM]] : {{.*}}) device_id(%[[DEV]]) core_id(%[[CORE0]]) {wait_target = false}
    CHECK:         tpu.wait_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) source_semaphore(%[[SEND_SEM]] : {{.*}}) target_semaphore(%[[RECV_SEM]] : {{.*}}) device_id(%[[DEV]]) core_id(%[[CORE0]]) {wait_target = true}
    CHECK:         %[[REG_SEM:.*]] = tpu.sem_alloc : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         tpu.enqueue_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) target_semaphore(%[[REG_SEM]] : {{.*}})
    CHECK:         %[[WAIT_AMT:.*]] = arith.constant 1 : i32
    CHECK:         tpu.sem_wait %[[REG_SEM]], %[[WAIT_AMT]] : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[REG_SEND_SEM:.*]] = tpu.sem_alloc : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[REG_RECV_SEM:.*]] = tpu.sem_alloc : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[CORE1:.*]] = arith.constant 1 : i32
    CHECK:         tpu.enqueue_dma source(%[[HBM]] : {{.*}}) target(%[[VMEM]] : {{.*}}) source_semaphore(%[[REG_SEND_SEM]] : {{.*}}) target_semaphore(%[[REG_RECV_SEM]] : {{.*}}) core_id(%[[CORE1]])
    CHECK:         %[[SEND_WAIT_AMT:.*]] = arith.constant 1 : i32
    CHECK:         tpu.sem_wait %[[REG_SEND_SEM]], %[[SEND_WAIT_AMT]] : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[RECV_WAIT_AMT:.*]] = arith.constant 1 : i32
    CHECK:         tpu.sem_wait %[[REG_RECV_SEM]], %[[RECV_WAIT_AMT]] : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
  )");
}

TEST_F(KilnTest, UnboundedDynamicDma) {
  // Unbounded-dynamic HBM <-> VMEM tiled copy loop.
  absl::StatusOr<Kernel> dyn_kernel = CreateKernel(
      GridSpec{.name = "unbounded_dynamic_dma"},
      InSpec::Hbm(f32(kDynamic, 128)), OutSpec::Hbm(f32(kDynamic, 128)),
      ScratchSpec::Vmem(f32(8, 128)),
      [&](Ref hbm_in, Ref hbm_out, Ref vmem_buf) {
        Semaphore in_sem = AllocaDmaSemaphore();
        Semaphore out_sem = AllocaDmaSemaphore();
        Scalar rows = hbm_in.Dim(0);
        For(0, rows, 8, [&](Scalar r) {
          Scalar aligned_r = MultipleOf(r, 8);
          Ref in_tile = hbm_in(Slice(aligned_r, 8), all);
          Ref out_tile = hbm_out(Slice(aligned_r, 8), all);
          AsyncCopy(in_tile, vmem_buf, in_sem).Wait();
          vmem_buf = Tile(vmem_buf) + 1.0f;
          AsyncCopy(vmem_buf, out_tile, out_sem).Wait();
        });
      });
  ASSERT_THAT(dyn_kernel, IsOk());
  ExpectKernelMatches(std::move(*dyn_kernel), R"(
    CHECK-LABEL: func.func @unbounded_dynamic_dma(
    CHECK-SAME:    %[[HBM_IN:arg0]]: memref<?x128xf32, #tpu.memory_space<hbm>>, %[[HBM_OUT:arg1]]: memref<?x128xf32, #tpu.memory_space<hbm>>, %[[VMEM_BUF:arg2]]: memref<8x128xf32, #tpu.memory_space<vmem>>)
    CHECK:         %[[IN_SEM:.*]] = tpu.sem_alloc : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[OUT_SEM:.*]] = tpu.sem_alloc : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    CHECK:         %[[DIM0:.*]] = memref.dim %[[HBM_IN]], %{{.*}} : memref<?x128xf32, #tpu.memory_space<hbm>>
    CHECK:         %[[ROWS:.*]] = arith.index_cast %[[DIM0]] : index to i32
    CHECK:         scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
    CHECK:           %[[R:.*]] = arith.index_cast %[[IV]] : index to i32
    CHECK:           %[[ALIGNED_R:.*]] = tpu.assume_multiple %[[R]], 8 : i32
    CHECK:           %[[IN_TILE:.*]] = tpu.memref_slice %[[HBM_IN]][%[[ALIGNED_R]], %{{.*}}] : memref<?x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
    CHECK:           %[[OUT_TILE:.*]] = tpu.memref_slice %[[HBM_OUT]][%[[ALIGNED_R]], %{{.*}}] : memref<?x128xf32, #tpu.memory_space<hbm>> -> memref<8x128xf32, #tpu.memory_space<hbm>>
    CHECK:           tpu.enqueue_dma source(%[[IN_TILE]] : {{.*}}) target(%[[VMEM_BUF]] : {{.*}}) target_semaphore(%[[IN_SEM]] : {{.*}})
    CHECK:           tpu.wait_dma source(%[[IN_TILE]] : {{.*}}) target(%[[VMEM_BUF]] : {{.*}}) target_semaphore(%[[IN_SEM]] : {{.*}}) {wait_target = true}
    CHECK:           %[[V:.*]] = vector.load %[[VMEM_BUF]]
    CHECK:           %[[V_PLUS_1:.*]] = arith.addf %[[V]], %{{.*}} : vector<8x128xf32>
    CHECK:           tpu.vector_store %[[VMEM_BUF]][%{{.*}}, %{{.*}}], %[[V_PLUS_1]]
    CHECK:           tpu.enqueue_dma source(%[[VMEM_BUF]] : {{.*}}) target(%[[OUT_TILE]] : {{.*}}) target_semaphore(%[[OUT_SEM]] : {{.*}})
    CHECK:           tpu.wait_dma source(%[[VMEM_BUF]] : {{.*}}) target(%[[OUT_TILE]] : {{.*}}) target_semaphore(%[[OUT_SEM]] : {{.*}}) {wait_target = true}
    CHECK:         }
  )");
}

TEST_F(KilnTest, DmaValidationErrors) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "dma_errors"}, InSpec::Hbm(f32(kDynamic, 128)),
      OutSpec::Vmem(f32(8, 128)), [&](Ref hbm_in, Ref vmem_out) {
        SemaphoreArray dma_sems = AllocaDmaSemaphore({2});
        Semaphore reg_sem = AllocaSemaphore();
        Ref hbm_tile = hbm_in(Slice(0, 8), all);
        Ref vmem_i32 = AllocaVmem(i32(8, 128));
        Ref smem_f32 = AllocaSmem(f32(8, 128));

        EXPECT_DEATH(AsyncCopy(hbm_tile, vmem_i32, dma_sems(0)),
                     "DMA source and destination Refs must have matching "
                     "shapes");
        EXPECT_DEATH(AsyncCopy(hbm_in, vmem_out, dma_sems(0)),
                     "DMA source and destination Refs must have matching "
                     "shapes");
        EXPECT_DEATH(
            AsyncCopy(hbm_tile, vmem_out, dma_sems(0), {.priority = -1}),
            "DMA priority must be 0 or 1");
        EXPECT_DEATH(
            AsyncCopy(hbm_tile, vmem_out, dma_sems(0), {.priority = 2}),
            "DMA priority must be 0 or 1");
        EXPECT_DEATH(MakeAsyncCopy(hbm_tile, smem_f32, reg_sem),
                     "Non-DMA semaphores are not supported for DMAs involving "
                     "SMEM");
        EXPECT_DEATH(MakeAsyncCopy(smem_f32, vmem_out, reg_sem),
                     "Non-DMA semaphores are not supported for DMAs involving "
                     "SMEM");
        EXPECT_DEATH(MakeAsyncRemoteCopy(smem_f32, vmem_out, reg_sem, reg_sem,
                                         {.device_id = 0}),
                     "Non-DMA semaphores are not supported for DMAs involving "
                     "SMEM");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, smem_f32, reg_sem, reg_sem,
                                         {.device_id = 0}),
                     "Non-DMA semaphores are not supported for DMAs involving "
                     "SMEM");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, dma_sems(0),
                                         dma_sems(1), {}),
                     "Remote DMA requires device_id or core_id");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, dma_sems(0),
                                         dma_sems(1), {.device_id = -1}),
                     "AsyncRemoteCopyOptions::device_id must be non-negative");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, dma_sems(0),
                                         dma_sems(1), {.core_id = -1}),
                     "AsyncRemoteCopyOptions::core_id must be non-negative");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, dma_sems(0),
                                         dma_sems(1), {.device_id = I32(-1)}),
                     "AsyncRemoteCopyOptions::device_id must be non-negative");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, dma_sems(0),
                                         dma_sems(1), {.core_id = I32(-1)}),
                     "AsyncRemoteCopyOptions::core_id must be non-negative");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, dma_sems(0),
                                         dma_sems(1), {.device_id = 1LL << 40}),
                     "AsyncRemoteCopyOptions::device_id exceeds int32_t max");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, dma_sems(0),
                                         dma_sems(1), {.core_id = 1LL << 40}),
                     "AsyncRemoteCopyOptions::core_id exceeds int32_t max");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, dma_sems(0),
                                         reg_sem, {.device_id = 0}),
                     "DMA source and target semaphores must have the same "
                     "kind");
        EXPECT_DEATH(MakeAsyncCopy(hbm_tile, vmem_out, reg_sem).Wait(),
                     "Wait\\(\\) requires a DMA semaphore");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, reg_sem, reg_sem,
                                         {.device_id = 0})
                         .WaitSend(),
                     "WaitSend\\(\\) requires a DMA semaphore");
        EXPECT_DEATH(MakeAsyncRemoteCopy(vmem_out, vmem_out, reg_sem, reg_sem,
                                         {.device_id = 0})
                         .WaitRecv(),
                     "WaitRecv\\(\\) requires a DMA semaphore");
      });
  ASSERT_THAT(kernel, IsOk());
}

TEST_F(KilnTest, CanonicalizeFoldsMemRefSliceAndEliminatesDeadOps) {
  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "canonicalize_smem", .grid = {4}},
                   ScalarPrefetchSpec{i32(4)}, OutSpec::Smem(i32(4, 8)),
                   [&](Ref idx, Ref out) {
                     Scalar i = ProgramId(0);
                     Scalar v = Scalar(idx(i));
                     out(i, 0) = v;
                   });
  ASSERT_THAT(kernel, IsOk());
  ASSERT_THAT(kernel->Canonicalize(), IsOk());

  ExpectKernelMatches(std::move(*kernel), R"(
    CHECK-LABEL: func.func @canonicalize_smem(
    CHECK-SAME:    %[[I:arg[0-9]+]]: i32,
    CHECK-SAME:    %[[IDX:arg[0-9]+]]: memref<4xi32, #tpu.memory_space<smem>>,
    CHECK-SAME:    %[[OUT:arg[0-9]+]]: memref<4x8xi32, #tpu.memory_space<smem>>)
    CHECK-NOT:     tpu.memref_slice
    CHECK-NOT:     tpu.memref_squeeze
    CHECK:         %[[I_IDX:.*]] = arith.index_cast %[[I]] : i32 to index
    CHECK:         %[[V:.*]] = memref.load %[[IDX]][%[[I_IDX]]] : memref<4xi32, #tpu.memory_space<smem>>
    CHECK:         %[[C0:.*]] = arith.constant 0 : index
    CHECK:         memref.store %[[V]], %[[OUT]][%[[I_IDX]], %[[C0]]] : memref<4x8xi32, #tpu.memory_space<smem>>
    CHECK:         return
  )");
}

TEST_F(KilnTest, ConstructsShapesFromDTypeCallOperator) {
  Shape scalar_i32 = i32();
  EXPECT_EQ(scalar_i32.rank(), 0);
  EXPECT_THAT(scalar_i32.dims(), IsEmpty());
  EXPECT_EQ(scalar_i32.dtype(), i32);
  EXPECT_EQ(scalar_i32, i32(/*dims=*/{}));
  EXPECT_EQ(scalar_i32, (Shape{/*dims=*/{}, i32}));

  Shape vec_i32 = i32(128);
  EXPECT_EQ(vec_i32.rank(), 1);
  EXPECT_THAT(vec_i32.dims(), ElementsAre(128));
  EXPECT_EQ(vec_i32, i32({128}));

  Shape mat_f32 = f32(8, 128);
  EXPECT_EQ(mat_f32.rank(), 2);
  EXPECT_THAT(mat_f32.dims(), ElementsAre(8, 128));
  EXPECT_EQ(mat_f32.dtype(), f32);
  EXPECT_EQ(mat_f32, f32({8, 128}));
  EXPECT_EQ(mat_f32, (Shape{{8, 128}, f32}));
  EXPECT_NE(mat_f32, bf16(8, 128));
  EXPECT_NE(mat_f32, f32(16, 128));

  EXPECT_EQ(scalar_i32.ToString(), "i32[]");
  EXPECT_EQ(vec_i32.ToString(), "i32[128]");
  EXPECT_EQ(mat_f32.ToString(), "f32[8, 128]");
  EXPECT_EQ(f32(kDynamic, 128).ToString(), "f32[?, 128]");
  EXPECT_EQ(absl::StrCat(mat_f32), "f32[8, 128]");

  std::ostringstream ss;
  ss << mat_f32;
  EXPECT_EQ(ss.str(), "f32[8, 128]");
}

}  // namespace
}  // namespace torch_tpu::kiln
