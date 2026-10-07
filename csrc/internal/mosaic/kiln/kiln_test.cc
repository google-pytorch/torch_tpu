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
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu::kiln {
namespace {

using absl_testing::IsOk;
using testing::HasSubstr;

class KilnTest : public testing::Test {
 protected:
  // Verifies that the kernel survives the same serialization step as the
  // production kernels and returns its textual form.
  std::string CheckAndPrint(Kernel&& kernel) {
    std::string ir = ToString(kernel);
    ABSL_LOG(INFO) << "\n" << ir;
    EXPECT_THAT(kernel.Verify(), IsOk());
    EXPECT_THAT(std::move(kernel).Serialize(), IsOk()) << ir;
    return ir;
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

  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("iteration_bounds = array<i64: 4, 2, 4>"));
  EXPECT_THAT(ir, HasSubstr("scratch_operands = 1"));
  EXPECT_THAT(ir, HasSubstr("tpu.matmul"));
  EXPECT_THAT(ir, HasSubstr("scf.if"));
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

  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("scalar_prefetch = 1"));
  EXPECT_THAT(ir, HasSubstr("func.func private @gather_rows_index_map_0"));
  EXPECT_THAT(ir, HasSubstr("memref.load"));
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
  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("tpu.memref_squeeze"));
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
  CheckAndPrint(std::move(*kernel));
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
  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("math.exp"));
  EXPECT_THAT(ir, HasSubstr("math.log"));
  EXPECT_THAT(ir, HasSubstr("math.sqrt"));
  EXPECT_THAT(ir, HasSubstr("math.rsqrt"));
  EXPECT_THAT(ir, HasSubstr("math.absf"));
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
  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("arith.xori"));
  EXPECT_THAT(ir, HasSubstr("arith.shrui"));
  EXPECT_THAT(ir, HasSubstr("arith.shli"));
  EXPECT_THAT(ir, HasSubstr("arith.andi"));
  EXPECT_THAT(ir, HasSubstr("arith.ori"));
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
  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("vector.broadcast"));
  EXPECT_THAT(ir, HasSubstr("arith.addf"));
  EXPECT_THAT(ir, HasSubstr("arith.xori"));
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
  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("tpu.dynamic_rotate"));
  EXPECT_THAT(ir, HasSubstr("tpu.dynamic_gather"));
  EXPECT_THAT(ir, HasSubstr("vector.extract"));
  EXPECT_THAT(ir, HasSubstr("arith.bitcast"));
  EXPECT_THAT(ir, HasSubstr("tpu.vector_store"));
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
  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("scf.for"));
  EXPECT_THAT(ir, HasSubstr("arith.index_cast"));
}

TEST_F(KilnTest, ScalarMismatchedTypesFail) {
  absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "mismatched", .grid = {1}},
      InSpec{{32, 128}, Float32Type()}, InSpec{{32, 128}, Int32Type()},
      OutSpec{{32, 128}, Float32Type()}, [&](Ref f_in, Ref i_in, Ref out) {
        Scalar s_i = I32(1);
        Scalar s_f = F32(1.0);
        EXPECT_DEATH(s_i + s_f,
                     "Scalar arithmetic requires operands of the same type");
        EXPECT_DEATH(For(s_f, [&](Scalar) {}),
                     "IndexArg requires an integer scalar");

        Tile t_f(f_in);
        Tile t_i(i_in);
        EXPECT_DEATH(
            t_f + t_i,
            "Tile binary operators require operands of the same dtype");
        EXPECT_DEATH(t_f & t_f,
                     "Bitwise tile operators require integer or mask operands");

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
  const std::string ir = CheckAndPrint(std::move(*kernel));
  EXPECT_THAT(ir, HasSubstr("scf.for"));
  EXPECT_THAT(ir, HasSubstr("arith.remsi"));
}

}  // namespace
}  // namespace torch_tpu::kiln
