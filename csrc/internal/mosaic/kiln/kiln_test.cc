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
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/testlib/filecheck.h"

namespace torch_tpu::kiln {
namespace {

using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;

class KilnTest : public testing::Test {
 protected:
  // Verifies that the kernel survives the same serialization step as the
  // production kernels and that its textual form matches `pattern`, a
  // FileCheck pattern (see https://llvm.org/docs/CommandGuide/FileCheck.html).
  // Patterns are written as they would be in an MLIR lit test; lines that are
  // not `CHECK` directives (e.g. `//` comments) are ignored by FileCheck.
  void ExpectKernelMatches(Kernel&& kernel, std::string_view pattern) {
    std::string ir = ToString(kernel);
    ABSL_LOG(INFO) << "\n" << ir;
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
  DType bf16 = BFloat16Type();
  DType f32 = Float32Type();

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{
          .name = "matmul",
          .grid = {kM / kTm, kN / kTn, kK / kTk},
          .dimension_semantics = {DimensionSemantics::kParallel,
                                  DimensionSemantics::kParallel,
                                  DimensionSemantics::kArbitrary},
      },
      InSpec{{kTm, kTk}, bf16, IndexMap([] {
               return IndexIndices{ProgramId(0), ProgramId(2)};
             })},
      InSpec{{kTk, kTn}, bf16, IndexMap([] {
               return IndexIndices{ProgramId(2), ProgramId(1)};
             })},
      OutSpec{{kTm, kTn}, bf16, IndexMap([] {
                return IndexIndices{ProgramId(0), ProgramId(1)};
              })},
      ScratchSpec{{kTm, kTn}, f32}, [&](Ref a, Ref b, Ref out, Ref acc) {
        Scalar kk = ProgramId(2);
        If(kk == 0, [&] { acc = Zeros({kTm, kTn}, f32); });
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
  DType f32 = Float32Type();
  DType i32 = Int32Type();

  IndexMap gathered = IndexMap(
      [] { return IndexIndices{Scalar(ScalarPrefetch(0)(ProgramId(0))), 0}; });

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "gather_rows", .grid = {kRows}},
      ScalarPrefetchSpec{{kRows}, i32}, InSpec{{1, kCols}, f32, gathered},
      OutSpec{{1, kChunk}, f32, IndexMap([] {
                return IndexIndices{ProgramId(0), 0};
              })},
      [&](Ref /*indices*/, Ref rows, Ref out) {
        Tile total = Zeros({1, kChunk}, f32);
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
  DType f32 = Float32Type();
  // 4-D block: (1, 1, 32, 128). Slicing in4d(0, 0) yields a 2-D tile (32, 128).
  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "rank_collapse", .grid = {1}},
                   InSpec{.block_shape = {1, 1, 32, 128}, .dtype = f32},
                   OutSpec{{32, 128}, f32}, [&](Ref in4d, Ref out2d) {
                     // in4d(0, 0) collapses the first 2 dimensions, resulting
                     // in rank 2.
                     Tile tile2d(in4d(0, 0));
                     EXPECT_EQ(tile2d.rank(), 2);
                     EXPECT_EQ(tile2d.shape()[0], 32);
                     EXPECT_EQ(tile2d.shape()[1], 128);
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
  DType f32 = Float32Type();
  DType i32 = Int32Type();

  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "literal_ops", .grid = {1}},
                   InSpec{{32, 128}, f32}, InSpec{{32, 128}, i32},
                   OutSpec{{32, 128}, f32}, [&](Ref f_in, Ref i_in, Ref out) {
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
                     EXPECT_TRUE(s_bool.IsBool());
                     EXPECT_FALSE(s_bool.IsIntOrFloat());
                     Cond c_roundtrip = static_cast<Cond>(s_bool);

                     If(
                         c_roundtrip,
                         [&] {
                           out = f_add_int + f_sub_float + f_scale + f_clamp +
                                 f_mask + i_combined.ToF32();
                         },
                         [&] { out = Zeros({32, 128}, f32); });
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
  DType f32 = Float32Type();
  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "scalar_float_ops", .grid = {1}},
                   InSpec{{32, 128}, f32, IndexMap(), MemorySpace::kSmem},
                   OutSpec{{32, 128}, f32, IndexMap(), MemorySpace::kSmem},
                   [&](Ref in, Ref out) {
                     Scalar sf = F32(2.5);
                     EXPECT_TRUE(sf.IsFloat());
                     EXPECT_FALSE(sf.IsInt());

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
  h = h ^ (LogicalRightShift(h, 16));
  h = h * 0x85EBCA6Bu;
  h = h ^ (LogicalRightShift(h, 13));
  h = h * 0xC2B2AE35u;
  h = h ^ (LogicalRightShift(h, 16));
  return h;
}

TEST_F(KilnTest, BitwiseOperations) {
  DType i32 = Int32Type();
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "bitwise_ops", .grid = {4}}, InSpec{{32, 128}, i32},
      OutSpec{{32, 128}, i32}, [&](Ref in, Ref out) {
        Scalar stream = Fmix32(ProgramId(0) ^ 0x9E3779B9u);
        Scalar shifted = (stream << 1) | (stream & 1);

        Tile t(in);
        Tile hashed = Fmix32(t ^ shifted);
        Tile masked = (hashed & 0xFFFF) | (LogicalRightShift(hashed, 16));
        out = Select(masked >= 0, masked, 0);
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
  DType f32 = Float32Type();
  DType i32 = Int32Type();

  absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "tile_scalar_ops", .grid = {2}},
                   InSpec{{32, 128}, f32}, InSpec{{32, 128}, i32},
                   OutSpec{{32, 128}, f32}, [&](Ref f_in, Ref i_in, Ref out) {
                     Tile f(f_in);
                     Tile i(i_in);

                     Scalar s_idx = ProgramId(0);
                     Scalar s_f = F32(1.5);
                     Scalar s_i = I32(42);

                     // Verify 0-D tile properties when constructed from a
                     // scalar.
                     Tile t_scalar(s_i);
                     EXPECT_EQ(t_scalar.rank(), 0);
                     EXPECT_TRUE(t_scalar.shape().empty());
                     EXPECT_EQ(t_scalar.dtype(), i32);

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
  DType f32 = Float32Type();
  DType i32 = Int32Type();

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "rotate_gather_extract_bitcast_store", .grid = {1}},
      InSpec{{8, 128}, f32}, OutSpec{{1, 128}, i32}, [&](Ref in, Ref out) {
        Scalar step = ProgramId(0);
        Tile rotated = Rotate(Tile(in), 8 - (step & 7), /*axis=*/0);
        Tile row0 = rotated.Slice({0, 0}, {1, 128});
        Tile gathered = Gather(row0, Splat(step, {1, 128}), /*axis=*/1);
        Scalar lane_f32 = Extract(gathered, {0, 0});
        Scalar lane_bits = Bitcast(lane_f32, i32);
        Tile row_bits = Bitcast(row0, i32);
        Tile mask = Iota({1, 128}, i32, /*axis=*/1) == (step & 127);
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
  DType f32 = Float32Type();
  DType i32 = Int32Type();

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "for_loop", .grid = {1}}, InSpec{{32, 128}, f32},
      OutSpec{{32, 128}, f32}, [&](Ref in, Ref out) {
        // Loop with static bounds.
        For(0, 10, [&](Scalar i) {
          EXPECT_TRUE(i.IsInt());
          EXPECT_FALSE(i.IsFloat());
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
    CHECK:         %[[C1:.*]] = arith.constant 1 : index
    CHECK:         %[[C0:.*]] = arith.constant 0 : index
    CHECK:         %[[C10:.*]] = arith.constant 10 : index
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
      GridSpec{.name = "mismatched", .grid = {1}},
      InSpec{{32, 128}, Float32Type()}, InSpec{{32, 128}, Int32Type()},
      OutSpec{{32, 128}, Float32Type()}, [&](Ref f_in, Ref i_in, Ref out) {
        Scalar s_i = I32(1);
        Scalar s_f = F32(1.0);
        EXPECT_DEATH(s_i + s_f, "\\+ requires operands of the same dtype");
        EXPECT_DEATH(For(s_f, [&](Scalar) {}),
                     "IndexArg requires an integer scalar");

        Tile t_f(f_in);
        Tile t_i(i_in);
        EXPECT_DEATH(t_f + t_i, "\\+ requires operands of the same dtype");
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
  DType f32 = Float32Type();
  EXPECT_TRUE(f32.IsFloat());
  EXPECT_FALSE(f32.IsInt());
  EXPECT_TRUE(f32.IsIntOrFloat());
  EXPECT_EQ(f32.GetIntOrFloatBitWidth(), 32);
  EXPECT_EQ(f32.ToString(), "f32");
  EXPECT_TRUE(static_cast<bool>(f32));

  std::ostringstream ss;
  ss << f32;
  EXPECT_EQ(ss.str(), "f32");

  DType i32 = Int32Type();
  EXPECT_FALSE(i32.IsFloat());
  EXPECT_TRUE(i32.IsInt());
  EXPECT_TRUE(i32.IsIntOrFloat());
  EXPECT_TRUE(i32.IsInteger(32));
  EXPECT_FALSE(i32.IsInteger(64));
  EXPECT_EQ(i32.GetIntOrFloatBitWidth(), 32);
  EXPECT_EQ(i32.ToString(), "i32");

  EXPECT_EQ(f32, Float32Type());
  EXPECT_NE(f32, i32);
}

TEST_F(KilnTest, TypeHelpersInsideKernel) {
  DType f32 = Float32Type();
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "type_helpers", .grid = {1}}, InSpec{{32, 128}, f32},
      OutSpec{{32, 128}, f32}, [&](Ref in, Ref out) {
        EXPECT_TRUE(BoolType().IsBool());
        EXPECT_FALSE(BoolType().IsIntOrFloat());
        EXPECT_TRUE(Int8Type().IsInteger(8));
        EXPECT_TRUE(Int16Type().IsInteger(16));
        EXPECT_TRUE(Int32Type().IsInteger(32));
        EXPECT_TRUE(Int64Type().IsInteger(64));
        EXPECT_TRUE(Float16Type().IsFloat());
        EXPECT_EQ(Float16Type().GetIntOrFloatBitWidth(), 16);
        EXPECT_TRUE(Float32Type().IsFloat());
        EXPECT_EQ(Float32Type().GetIntOrFloatBitWidth(), 32);
        EXPECT_TRUE(BFloat16Type().IsFloat());
        EXPECT_EQ(BFloat16Type().GetIntOrFloatBitWidth(), 16);
        out = Tile(in);
      });
  ASSERT_THAT(kernel, IsOk());
}

TEST_F(KilnTest, DTypeNullability) {
  DType null_dt;
  EXPECT_FALSE(static_cast<bool>(null_dt));
  EXPECT_EQ(null_dt, DType());
  EXPECT_NE(null_dt, Float32Type());
  EXPECT_EQ(null_dt.ToString(), "invalid");
}

TEST_F(KilnTest, DTypeInTileAndScalarOps) {
  DType f32 = Float32Type();
  DType bf16 = BFloat16Type();
  DType i32 = Int32Type();

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "dtype_ops", .grid = {1}}, InSpec{{32, 128}, f32},
      OutSpec{{32, 128}, f32}, [&](Ref in, Ref out) {
        EXPECT_EQ(in.dtype(), f32);
        EXPECT_EQ(out.dtype(), f32);

        Tile t(in);
        EXPECT_EQ(t.dtype(), f32);

        Tile t_bf16 = t.ToBF16();
        EXPECT_EQ(t_bf16.dtype(), bf16);

        Tile t_i32 = t.ToI32();
        EXPECT_EQ(t_i32.dtype(), i32);

        Scalar s = ProgramId(0);
        EXPECT_EQ(s.dtype(), i32);

        Scalar s_f32 = s.AsType(f32);
        EXPECT_EQ(s_f32.dtype(), f32);

        Tile z = Zeros({32, 128}, f32);
        EXPECT_EQ(z.dtype(), f32);

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
  DType f32 = Float32Type();
  DType i32 = Int32Type();

  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "simplified_ops", .grid = {1}}, InSpec{{32, 128}, f32},
      OutSpec{{32, 128}, f32}, [&](Ref in, Ref out) {
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
        Tile iota = Iota({32, 128}, i32, 1);
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

}  // namespace
}  // namespace torch_tpu::kiln
