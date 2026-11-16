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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/experimental/precompiled_kernels/kernel_test_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/internal/mosaic/kiln/mlir/stablehlo_splice.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/types.h"

namespace torch_tpu::kiln {
namespace {

using absl_testing::IsOk;

// Test fixture that compiles and executes Kiln kernels end-to-end on TPU via
// XLA's `tpu_custom_call` and compares the output against expected literals.
class KilnE2eTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                            CreateKernelTestClient());
    runner_ = new xla::HloRunner(std::move(client));
  }

  static void TearDownTestSuite() {
    delete runner_;
    runner_ = nullptr;
  }

  void SetUp() override { ASSERT_NE(runner_, nullptr); }

  // Compiles and executes a serialized Kiln `kernel` end-to-end on TPU via a
  // single `tpu_custom_call`.
  absl::StatusOr<xla::Literal> RunKernel(
      std::string_view serialized, absl::Span<const xla::Literal* const> inputs,
      const xla::Shape& result_shape, bool needs_hlo_passes = false) {
    xla::HloComputation::Builder builder("main");
    std::vector<xla::HloInstruction*> operands;
    operands.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      operands.push_back(
          builder.AddInstruction(xla::HloInstruction::CreateParameter(
              i, inputs[i]->shape(), absl::StrCat("p", i))));
    }
    xla::HloInstruction* call =
        builder.AddInstruction(xla::HloInstruction::CreateCustomCall(
            result_shape, operands, "tpu_custom_call"));
    call->set_raw_backend_config_string(absl::StrCat(
        R"({"device_type":"DEVICE_TYPE_TENSORCORE","custom_call_config":{"body":")",
        absl::Base64Escape(serialized),
        R"(","serialization_format":1,"needs_layout_passes":true,"needs_hlo_passes":)",
        needs_hlo_passes ? "true" : "false", "}}"));

    auto hlo = std::make_unique<xla::HloModule>("kiln_e2e_kernel",
                                                xla::HloModuleConfig());
    hlo->AddEntryComputationWithLayouts(builder.Build());
    return AdaptXlaError(
        runner_->Execute(std::move(hlo), inputs, /*run_hlo_passes=*/true),
        "failed to execute Kiln kernel on TPU");
  }

  // Runs a serialized Kiln kernel on TPU with `inputs` and compares the result
  // against `expected`.
  void RunAndCompare(std::string_view serialized,
                     absl::Span<const xla::Literal* const> inputs,
                     const xla::Literal& expected,
                     std::optional<xla::ErrorSpec> error_spec = std::nullopt,
                     bool needs_hlo_passes = false) {
    TT_ASSERT_OK_AND_ASSIGN(
        xla::Literal actual,
        RunKernel(serialized, inputs, expected.shape(), needs_hlo_passes));
    EXPECT_TRUE(
        xla::LiteralTestUtil::NearOrEqual(expected, actual, error_spec));
  }

  // Verifies `kernel`, runs it on TPU with `inputs`, and compares the result
  // against `expected`.
  void RunAndCompare(Kernel&& kernel,
                     absl::Span<const xla::Literal* const> inputs,
                     const xla::Literal& expected,
                     std::optional<xla::ErrorSpec> error_spec = std::nullopt,
                     bool needs_hlo_passes = false) {
    EXPECT_THAT(kernel.Verify(), IsOk());
    TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                            std::move(kernel).Serialize());
    RunAndCompare(serialized, inputs, expected, error_spec, needs_hlo_passes);
  }

  template <typename Fn>
  static xla::Literal Make2dF32(int64_t rows, int64_t cols, Fn&& fn) {
    xla::Literal lit(xla::ShapeUtil::MakeShape(xla::F32, {rows, cols}));
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        lit.Set<float>({r, c}, fn(r, c));
      }
    }
    return lit;
  }

  template <typename Fn>
  static xla::Literal Make1dF32(int64_t n, Fn&& fn) {
    xla::Literal lit(xla::ShapeUtil::MakeShape(xla::F32, {n}));
    for (int64_t i = 0; i < n; ++i) {
      lit.Set<float>({i}, fn(i));
    }
    return lit;
  }

  // Verifies and serializes `kernel`, to run it on several inputs.
  static absl::StatusOr<std::string> VerifyAndSerialize(Kernel&& kernel) {
    TT_RETURN_IF_ERROR(kernel.Verify());
    return std::move(kernel).Serialize();
  }

  static inline xla::HloRunner* runner_ = nullptr;
};

TEST_F(KilnE2eTest, BlockwiseMatmulWithScratchAccumulator) {
  constexpr int64_t kM = 256, kN = 256, kK = 256;
  constexpr int64_t kTm = 128, kTn = 128, kTk = 128;

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{
              .name = "matmul_e2e",
              .grid = {kM / kTm, kN / kTn, kK / kTk},
              .dimension_semantics = {DimensionSemantics::kParallel,
                                      DimensionSemantics::kParallel,
                                      DimensionSemantics::kArbitrary},
          },
          InSpec{f32(kTm, kTk), IndexMap([] {
                   return IndexIndices{ProgramId(0), ProgramId(2)};
                 })},
          InSpec{f32(kTk, kTn), IndexMap([] {
                   return IndexIndices{ProgramId(2), ProgramId(1)};
                 })},
          OutSpec{f32(kTm, kTn), IndexMap([] {
                    return IndexIndices{ProgramId(0), ProgramId(1)};
                  })},
          ScratchSpec::Vmem(f32(kTm, kTn)),
          [](VmemRef a, VmemRef b, VmemRef out, VmemRef acc) {
            Scalar kk = ProgramId(2);
            If(kk == 0, [&] { Store(acc, Zeros(acc.shape())); });
            Store(acc,
                  Load(acc) + Dot(Load(a), Load(b),
                                  {.precision = ContractPrecision::kFP32}));
            If(kk == NumPrograms(2) - 1, [&] { Store(out, Load(acc)); });
          }));

  xla::Literal a_lit = Make2dF32(kM, kK, [](int64_t m, int64_t k) {
    return static_cast<float>((m + k) % 7 - 3) * 0.33f;
  });
  xla::Literal b_lit = Make2dF32(kK, kN, [](int64_t k, int64_t n) {
    return static_cast<float>((k * 2 + n) % 5 - 2) * 0.54321f;
  });
  xla::Literal expected = Make2dF32(kM, kN, [&](int64_t m, int64_t n) {
    float sum = 0.0f;
    for (int64_t k = 0; k < kK; ++k) {
      sum += a_lit.Get<float>({m, k}) * b_lit.Get<float>({k, n});
    }
    return sum;
  });

  RunAndCompare(std::move(kernel), {&a_lit, &b_lit}, expected,
                xla::ErrorSpec(1e-5, 1e-5));
}

TEST_F(KilnE2eTest, ScalarPrefetchGatherAndReduce) {
  constexpr int64_t kRows = 4, kTileRows = 8, kCols = 256, kChunk = 128;

  IndexMap gathered = IndexMap(
      [] { return IndexIndices{Load(ScalarPrefetch(0)(ProgramId(0))), 0}; });

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(GridSpec{.name = "gather_rows_e2e", .grid = {kRows}},
                   ScalarPrefetchSpec{i32(kRows)},
                   InSpec{f32(kTileRows, kCols), gathered},
                   OutSpec{f32(kTileRows, kChunk), IndexMap([] {
                             return IndexIndices{ProgramId(0), 0};
                           })},
                   [](SmemRef /*indices*/, VmemRef rows, VmemRef out) {
                     Tile total = Zeros(out.shape());
                     for (int64_t c = 0; c < kCols; c += kChunk) {
                       total = total + Load(rows(all, Slice(c, kChunk)));
                     }
                     Store(out, total);
                   }));

  std::array<int32_t, kRows> idx_values = {3, 1, 2, 0};
  xla::Literal indices = xla::LiteralUtil::CreateR1<int32_t>(idx_values);
  xla::Literal table = Make2dF32(
      kRows * kTileRows, kCols,
      [](int64_t r, int64_t c) { return static_cast<float>(r * 1000 + c); });
  xla::Literal expected =
      Make2dF32(kRows * kTileRows, kChunk, [&](int64_t out_r, int64_t c) {
        int64_t src_r =
            idx_values[out_r / kTileRows] * kTileRows + (out_r % kTileRows);
        float sum = 0.0f;
        for (int64_t base = 0; base < kCols; base += kChunk) {
          sum += table.Get<float>({src_r, base + c});
        }
        return sum;
      });

  RunAndCompare(std::move(kernel), {&indices, &table}, expected);
}

// Splices a StableHLO module computing (a + b) * a into the body. The add is
// done in f64, the way TorchTPU lowers Python-scalar arithmetic, to exercise
// the demotion to f32.
TEST_F(KilnE2eTest, SpliceStablehloIntoKernelBody) {
  constexpr int64_t kRows = 8, kCols = 128;
  constexpr char kModule[] = R"mlir(
    module {
      func.func @main(%a: tensor<8x128xf32>, %b: tensor<8x128xf32>) -> tensor<8x128xf32> {
        %a64 = stablehlo.convert %a : (tensor<8x128xf32>) -> tensor<8x128xf64>
        %b64 = stablehlo.convert %b : (tensor<8x128xf32>) -> tensor<8x128xf64>
        %sum64 = stablehlo.add %a64, %b64 : tensor<8x128xf64>
        %sum = stablehlo.convert %sum64 : (tensor<8x128xf64>) -> tensor<8x128xf32>
        %out = stablehlo.multiply %sum, %a : tensor<8x128xf32>
        return %out : tensor<8x128xf32>
      }
    })mlir";

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "splice_e2e", .grid = {1}},
          InSpec{f32(kRows, kCols)}, InSpec{f32(kRows, kCols)},
          OutSpec{f32(kRows, kCols)}, [&](VmemRef a, VmemRef b, VmemRef out) {
            TT_ASSERT_OK_AND_ASSIGN(
                Tile result, SpliceStablehlo(kModule, {Load(a), Load(b)}));
            Store(out, result);
          }));

  // Small exact values of both signs, so the comparison can be bit-exact.
  xla::Literal a = Make2dF32(kRows, kCols, [](int64_t r, int64_t c) {
    return static_cast<float>(r - 4) * 0.5f;
  });
  xla::Literal b = Make2dF32(kRows, kCols, [](int64_t r, int64_t c) {
    return static_cast<float>(c % 8) * 0.25f;
  });
  xla::Literal expected = Make2dF32(kRows, kCols, [&](int64_t r, int64_t c) {
    const float x = a.Get<float>({r, c});
    return (x + b.Get<float>({r, c})) * x;
  });

  RunAndCompare(std::move(kernel), {&a, &b}, expected,
                /*error_spec=*/std::nullopt, /*needs_hlo_passes=*/true);
}

// CHLO is legalized before splicing; the scalar broadcast it adds folds away.
TEST_F(KilnE2eTest, SpliceStablehloLegalizesChlo) {
  constexpr int64_t kRows = 8, kCols = 128;
  constexpr char kModule[] = R"mlir(
    module {
      func.func @main(%a: tensor<8x128xf32>) -> tensor<8x128xf32> {
        %two = stablehlo.constant dense<2.0> : tensor<f32>
        %out = chlo.broadcast_multiply %a, %two : (tensor<8x128xf32>, tensor<f32>) -> tensor<8x128xf32>
        return %out : tensor<8x128xf32>
      }
    })mlir";

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(GridSpec{.name = "splice_chlo_e2e", .grid = {1}},
                   InSpec{f32(kRows, kCols)}, OutSpec{f32(kRows, kCols)},
                   [&](VmemRef a, VmemRef out) {
                     TT_ASSERT_OK_AND_ASSIGN(
                         Tile result, SpliceStablehlo(kModule, {Load(a)}));
                     Store(out, result);
                   }));

  xla::Literal a = Make2dF32(kRows, kCols, [](int64_t r, int64_t c) {
    return static_cast<float>(r - 4) * 0.5f;
  });
  xla::Literal expected = Make2dF32(kRows, kCols, [&](int64_t r, int64_t c) {
    return 2.0f * a.Get<float>({r, c});
  });

  RunAndCompare(std::move(kernel), {&a}, expected,
                /*error_spec=*/std::nullopt, /*needs_hlo_passes=*/true);
}

TEST_F(KilnE2eTest, TileAndScalarControlFlowOperations) {
  constexpr int64_t kRows = 8, kCols = 128;

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(GridSpec{.name = "tile_scalar_e2e", .grid = {1}},
                   InSpec{f32(kRows, kCols)}, OutSpec{f32(kRows, kCols)},
                   [](VmemRef in, VmemRef out) {
                     Store(out, Load(in));
                     For(0, 4, [&](Scalar i) {
                       Store(out, Load(out) + i.AsType(f32));
                     });
                     Tile acc = Load(out);
                     Tile col_iota = Iota(i32(kRows, kCols), /*axis=*/1);
                     Tile even_mask = (col_iota % 2) == 0;
                     Store(out, Select(even_mask, acc * 2.0, acc - 1.0));
                   }));

  xla::Literal input = Make2dF32(kRows, kCols, [](int64_t r, int64_t c) {
    return static_cast<float>(r * kCols + c) * 0.5f;
  });
  xla::Literal expected = Make2dF32(kRows, kCols, [&](int64_t r, int64_t c) {
    float after_loop = input.Get<float>({r, c}) + (0.0f + 1.0f + 2.0f + 3.0f);
    return (c % 2 == 0) ? after_loop * 2.0f : after_loop - 1.0f;
  });

  RunAndCompare(std::move(kernel), {&input}, expected);
}

// Compares two bool masks elementwise. The masks come from runtime data so
// that the comparison cannot be constant-folded away before Mosaic sees it.
TEST_F(KilnE2eTest, EqualityOfBoolMasks) {
  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(GridSpec{.name = "bool_mask_eq_e2e", .grid = {1}},
                   InSpec{f32(8, 128)}, OutSpec{f32(8, 128)},
                   [](VmemRef in, VmemRef out) {
                     Tile x = Load(in);
                     Store(out, Select((x > 0.0) == (x > 1.0), x.Full(1.0),
                                       x.Full(0.0)));
                   }));

  xla::Literal input = Make2dF32(8, 128, [](int64_t r, int64_t c) {
    return static_cast<float>(c % 3);  // 0, 1, 2: masks disagree only on 1.
  });
  xla::Literal expected = Make2dF32(
      8, 128, [](int64_t r, int64_t c) { return c % 3 == 1 ? 0.0f : 1.0f; });

  RunAndCompare(std::move(kernel), {&input}, expected);
}

// Like EqualityOfBoolMasks, for `!=`.
TEST_F(KilnE2eTest, InequalityOfBoolMasks) {
  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(GridSpec{.name = "bool_mask_ne_e2e", .grid = {1}},
                   InSpec{f32(8, 128)}, OutSpec{f32(8, 128)},
                   [](VmemRef in, VmemRef out) {
                     Tile x = Load(in);
                     Store(out, Select((x > 0.0) != (x > 1.0), x.Full(1.0),
                                       x.Full(0.0)));
                   }));

  xla::Literal input = Make2dF32(8, 128, [](int64_t r, int64_t c) {
    return static_cast<float>(c % 3);  // 0, 1, 2: masks disagree only on 1.
  });
  xla::Literal expected = Make2dF32(
      8, 128, [](int64_t r, int64_t c) { return c % 3 == 1 ? 1.0f : 0.0f; });

  RunAndCompare(std::move(kernel), {&input}, expected);
}

// Without a grid, XLA copies VMEM operands in and out whole.
TEST_F(KilnE2eTest, GridFreeKernelWithVmemOperands) {
  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "grid_free_vmem_e2e"}, InSpec::Vmem(f32(8, 128)),
          OutSpec::Vmem(f32(8, 128)),
          [](VmemRef in, VmemRef out) { Store(out, Load(in) * 2.0); }));
  xla::Literal input = Make2dF32(8, 128, [](int64_t r, int64_t c) {
    return static_cast<float>(r * 128 + c);
  });
  xla::Literal expected = Make2dF32(8, 128, [](int64_t r, int64_t c) {
    return 2.0f * static_cast<float>(r * 128 + c);
  });
  RunAndCompare(std::move(kernel), {&input}, expected);
}

TEST_F(KilnE2eTest, GridKernelWithHbmAndDma) {
  constexpr int64_t kSteps = 4, kTileRows = 8, kCols = 128;

  // Exercises dynamic 2-slot VMEM scratch and semaphore array indexing
  // (`step % 2`) alongside explicit HBM-VMEM DMA priority attributes.
  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(GridSpec{.name = "grid_hbm_dma_e2e", .grid = {kSteps}},
                   InSpec::Hbm(f32(kSteps * kTileRows, kCols)),
                   InSpec::Vmem(f32(kTileRows, kCols)),
                   OutSpec::Hbm(f32(kSteps * kTileRows, kCols)),
                   ScratchSpec::Vmem(f32(2, kTileRows, kCols)),
                   [](HbmRef hbm_in, VmemRef vmem_bias, HbmRef hbm_out,
                      VmemRef vmem_bufs) {
                     SemaphoreArray sems = AllocaDmaSemaphore({2});
                     Scalar step = ProgramId(0);
                     Scalar slot = step % 2;
                     Scalar row = MultipleOf(step * kTileRows, kTileRows);

                     VmemRef buf = vmem_bufs(slot);
                     Semaphore sem = sems(slot);

                     AsyncCopyDescriptor in_copy =
                         MakeAsyncCopy(hbm_in(Slice(row, kTileRows), all), buf,
                                       sem, {.priority = 1});
                     in_copy.Start();
                     in_copy.Wait();

                     Store(buf, Load(buf) + Load(vmem_bias));

                     AsyncCopy(buf, hbm_out(Slice(row, kTileRows), all), sem,
                               {.priority = 0})
                         .Wait();
                   }));

  xla::Literal vmem_bias =
      Make2dF32(kTileRows, kCols, [](int64_t r, int64_t c) {
        return static_cast<float>(r * 10 + c) * 0.25f;
      });
  xla::Literal hbm_in = Make2dF32(
      kSteps * kTileRows, kCols,
      [](int64_t r, int64_t c) { return static_cast<float>(r * kCols + c); });
  xla::Literal expected =
      Make2dF32(kSteps * kTileRows, kCols, [&](int64_t r, int64_t c) {
        return hbm_in.Get<float>({r, c}) +
               vmem_bias.Get<float>({r % kTileRows, c});
      });

  RunAndCompare(std::move(kernel), {&hbm_in, &vmem_bias}, expected);
}

TEST_F(KilnE2eTest, DynamicHbmTiledLoopWithDma) {
  constexpr int64_t kTileRows = 8, kCols = 128;

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "dynamic_hbm_scale_e2e"},
          InSpec::Hbm(f32(kDynamic, kCols)), OutSpec::Hbm(f32(kDynamic, kCols)),
          [](HbmRef hbm_in, HbmRef hbm_out) {
            VmemRef vmem_tile = AllocaVmem(f32(kTileRows, kCols));
            Semaphore sem = AllocaDmaSemaphore();

            Scalar rows = hbm_in.Dim(0);
            For(0, rows, kTileRows, [&](Scalar row_offset) {
              Scalar aligned = MultipleOf(row_offset, kTileRows);
              HbmRef in_slice = hbm_in(Slice(aligned, kTileRows), all);
              HbmRef out_slice = hbm_out(Slice(aligned, kTileRows), all);

              AsyncCopy(in_slice, vmem_tile, sem).Wait();
              Store(vmem_tile, Load(vmem_tile) * 2.0);
              AsyncCopy(vmem_tile, out_slice, sem).Wait();
            });
          }));

  EXPECT_THAT(kernel.Verify(), IsOk());
  TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          std::move(kernel).Serialize());

  for (int64_t num_rows : {8, 16, 32}) {
    SCOPED_TRACE(absl::StrCat("num_rows=", num_rows));
    xla::Literal input = Make2dF32(num_rows, kCols, [](int64_t r, int64_t c) {
      return static_cast<float>((r * kCols + c) % 23 - 11);
    });
    xla::Literal expected = Make2dF32(
        num_rows, kCols,
        [&](int64_t r, int64_t c) { return input.Get<float>({r, c}) * 2.0f; });
    RunAndCompare(serialized, {&input}, expected);
  }
}

// One compiled kernel sums the 128-element chunks of an HBM vector: the chunks
// are streamed with EmitPipeline into a VMEM accumulator, which is copied to
// the HBM output at the end.
TEST_F(KilnE2eTest, PipelinedSum) {
  constexpr int64_t kChunk = 128;

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "pipelined_sum_e2e"}, InSpec::Hbm(f32(kDynamic)),
          OutSpec::Hbm(f32(kChunk)), ScratchSpec::Vmem(f32(kChunk)),
          [&](HbmRef in, HbmRef out, VmemRef acc) {
            PipelineOperand input{
                .hbm = in, .block_shape = {kChunk}, .region = [](Scalar step) {
                  return BlockRegion{.offsets = {step * kChunk},
                                     .sizes = {kChunk}};
                }};
            Store(acc, Zeros(acc.shape()));
            EmitPipeline(0, in.Dim(0) / kChunk, {std::move(input)},
                         /*outputs=*/{}, [&](const PipelineStep& s) {
                           Store(acc, Load(acc) + Load(s.inputs[0].ref));
                         });
            AsyncCopy(acc, out, AllocaDmaSemaphore()).Wait();
          }));

  TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          VerifyAndSerialize(std::move(kernel)));

  // A single chunk never prefetches.
  for (int64_t num_chunks : {1, 5}) {
    const int64_t n = num_chunks * kChunk;
    SCOPED_TRACE(absl::StrCat("n=", n));
    xla::Literal input =
        Make1dF32(n, [](int64_t i) { return static_cast<float>(i % 17 - 8); });
    xla::Literal expected = Make1dF32(kChunk, [&](int64_t i) {
      float sum = 0.0f;
      for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
        sum += input.Get<float>({chunk * kChunk + i});
      }
      return sum;
    });
    RunAndCompare(serialized, {&input}, expected);
  }
}

// One compiled kernel sums the elementwise products of the 8-row bands of two
// HBM matrices, streaming both inputs with one EmitPipeline.
TEST_F(KilnE2eTest, PipelinedSumOfProducts) {
  constexpr int64_t kRows = 8, kCols = 128;
  auto band = [](Scalar step) {
    return BlockRegion{.offsets = {step * kRows, 0}, .sizes = {kRows, kCols}};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "pipelined_sum_of_products_e2e"},
          InSpec::Hbm(f32(kDynamic, kCols)), InSpec::Hbm(f32(kDynamic, kCols)),
          OutSpec::Hbm(f32(kRows, kCols)), ScratchSpec::Vmem(f32(kRows, kCols)),
          [&](HbmRef a, HbmRef b, HbmRef out, VmemRef acc) {
            Store(acc, Zeros(acc.shape()));
            EmitPipeline(
                0, a.Dim(0) / kRows,
                {{.hbm = a, .block_shape = {kRows, kCols}, .region = band},
                 {.hbm = b, .block_shape = {kRows, kCols}, .region = band}},
                /*outputs=*/{}, [&](const PipelineStep& s) {
                  Store(acc, Load(acc) +
                                 Load(s.inputs[0].ref) * Load(s.inputs[1].ref));
                });
            AsyncCopy(acc, out, AllocaDmaSemaphore()).Wait();
          }));

  TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          VerifyAndSerialize(std::move(kernel)));

  for (int64_t num_bands : {1, 5}) {
    SCOPED_TRACE(absl::StrCat("num_bands=", num_bands));
    const int64_t rows = num_bands * kRows;
    xla::Literal a = Make2dF32(rows, kCols, [](int64_t r, int64_t c) {
      return static_cast<float>((r + c) % 7 - 3);
    });
    xla::Literal b = Make2dF32(rows, kCols, [](int64_t r, int64_t c) {
      return static_cast<float>((3 * r + c) % 5 - 2);
    });
    xla::Literal expected = Make2dF32(kRows, kCols, [&](int64_t r, int64_t c) {
      float sum = 0.0f;
      for (int64_t band = 0; band < num_bands; ++band) {
        sum += a.Get<float>({band * kRows + r, c}) *
               b.Get<float>({band * kRows + r, c});
      }
      return sum;
    });
    RunAndCompare(serialized, {&a, &b}, expected);
  }
}

// One compiled kernel doubles an HBM vector chunk by chunk, streaming both the
// input and the output chunks.
TEST_F(KilnE2eTest, PipelinedInputOutputStreaming) {
  constexpr int64_t kChunk = 128;
  auto chunk = [](Scalar step) {
    return BlockRegion{.offsets = {step * kChunk}, .sizes = {kChunk}};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "pipelined_io_streaming_e2e"},
          InSpec::Hbm(f32(kDynamic)), OutSpec::Hbm(f32(kDynamic)),
          [&](HbmRef in, HbmRef out) {
            EmitPipeline(
                0, in.Dim(0) / kChunk,
                {{.hbm = in, .block_shape = {kChunk}, .region = chunk}},
                {{.hbm = out, .block_shape = {kChunk}, .region = chunk}},
                [&](const PipelineStep& s) {
                  Store(s.outputs[0].ref, Load(s.inputs[0].ref) * 2.0);
                });
          }));

  TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          VerifyAndSerialize(std::move(kernel)));

  // A single chunk never waits for a slot to be copied out before reusing it.
  for (int64_t num_chunks : {1, 5}) {
    const int64_t n = num_chunks * kChunk;
    SCOPED_TRACE(absl::StrCat("n=", n));
    xla::Literal input =
        Make1dF32(n, [](int64_t i) { return static_cast<float>(i % 17 - 8); });
    xla::Literal expected =
        Make1dF32(n, [&](int64_t i) { return input.Get<float>({i}) * 2.0f; });
    RunAndCompare(serialized, {&input}, expected);
  }
}

// Streams two 2D outputs computed from one input, band by band.
TEST_F(KilnE2eTest, PipelinedTwo2dOutputs) {
  constexpr int64_t kRows = 8, kCols = 128;
  auto band = [](Scalar step) {
    return BlockRegion{.offsets = {step * kRows, 0}, .sizes = {kRows, kCols}};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "pipelined_two_2d_outputs_e2e"},
          InSpec::Hbm(f32(kDynamic, kCols)), OutSpec::Hbm(f32(kDynamic, kCols)),
          OutSpec::Hbm(f32(kDynamic, kCols)),
          [&](HbmRef in, HbmRef out0, HbmRef out1) {
            EmitPipeline(
                0, in.Dim(0) / kRows,
                {{.hbm = in, .block_shape = {kRows, kCols}, .region = band}},
                {{.hbm = out0, .block_shape = {kRows, kCols}, .region = band},
                 {.hbm = out1, .block_shape = {kRows, kCols}, .region = band}},
                [&](const PipelineStep& s) {
                  Store(s.outputs[0].ref, Load(s.inputs[0].ref) * 2.0);
                  Store(s.outputs[1].ref, Load(s.inputs[0].ref) + 1.0);
                });
          }));

  TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          VerifyAndSerialize(std::move(kernel)));

  for (int64_t num_bands : {1, 5}) {
    SCOPED_TRACE(absl::StrCat("num_bands=", num_bands));
    const int64_t rows = num_bands * kRows;
    auto x = [](int64_t r, int64_t c) {
      return static_cast<float>((r + c) % 7 - 3);
    };
    xla::Literal input = Make2dF32(rows, kCols, x);
    xla::Literal expected = xla::LiteralUtil::MakeTupleOwned(
        Make2dF32(rows, kCols,
                  [&](int64_t r, int64_t c) { return x(r, c) * 2.0f; }),
        Make2dF32(rows, kCols,
                  [&](int64_t r, int64_t c) { return x(r, c) + 1.0f; }));
    RunAndCompare(serialized, {&input}, expected);
  }
}

// One compiled kernel sums consecutive pairs of chunks of an HBM vector: each
// output block accumulates over two steps.
TEST_F(KilnE2eTest, PipelinedBlockPairSums) {
  constexpr int64_t kChunk = 128;
  auto chunk = [](Scalar i) {
    return BlockRegion{.offsets = {i * kChunk}, .sizes = {kChunk}};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "pipelined_block_pairs_e2e"},
          InSpec::Hbm(f32(kDynamic)), OutSpec::Hbm(f32(kDynamic)),
          [&](HbmRef in, HbmRef out) {
            EmitPipeline(
                0, in.Dim(0) / kChunk,
                {{.hbm = in, .block_shape = {kChunk}, .region = chunk}},
                {{.hbm = out,
                  .block_shape = {kChunk},
                  .block_index = [](Scalar step) { return step / 2; },
                  .region = chunk}},
                [&](const PipelineStep& s) {
                  VmemRef acc = s.outputs[0].ref;
                  If(s.outputs[0].is_first_step,
                     [&] { Store(acc, Zeros(acc.shape())); });
                  Store(acc, Load(acc) + Load(s.inputs[0].ref));
                });
          }));

  TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          VerifyAndSerialize(std::move(kernel)));

  for (int64_t num_pairs : {1, 3}) {
    SCOPED_TRACE(absl::StrCat("num_pairs=", num_pairs));
    xla::Literal input = Make1dF32(2 * num_pairs * kChunk, [](int64_t i) {
      return static_cast<float>(i % 17 - 8);
    });
    xla::Literal expected = Make1dF32(num_pairs * kChunk, [&](int64_t i) {
      const int64_t pair = i / kChunk, lane = i % kChunk;
      return input.Get<float>({2 * pair * kChunk + lane}) +
             input.Get<float>({(2 * pair + 1) * kChunk + lane});
    });
    RunAndCompare(serialized, {&input}, expected);
  }
}

// Output block indices need not increase by one: `reversed` visits pairs of
// chunks from last to first, and `uneven` has a one-step first and last block
// (`(step + 1) / 2`), so its blocks straddle the pairs.
TEST_F(KilnE2eTest, PipelinedBlocksInAnyOrder) {
  constexpr int64_t kChunk = 128;
  auto chunk = [](Scalar i) {
    return BlockRegion{.offsets = {i * kChunk}, .sizes = {kChunk}};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "pipelined_blocks_any_order_e2e"},
          InSpec::Hbm(f32(kDynamic)), OutSpec::Hbm(f32(kDynamic)),
          OutSpec::Hbm(f32(kDynamic)),
          [&](HbmRef in, HbmRef reversed, HbmRef uneven) {
            const Scalar num_steps = in.Dim(0) / kChunk;
            const Scalar last_pair = num_steps / 2 - 1;
            EmitPipeline(
                0, num_steps,
                {{.hbm = in, .block_shape = {kChunk}, .region = chunk}},
                {{.hbm = reversed,
                  .block_shape = {kChunk},
                  .block_index =
                      [last_pair](Scalar step) { return last_pair - step / 2; },
                  .region = chunk},
                 {.hbm = uneven,
                  .block_shape = {kChunk},
                  .block_index = [](Scalar step) { return (step + 1) / 2; },
                  .region = chunk}},
                [&](const PipelineStep& s) {
                  for (int o = 0; o < 2; ++o) {
                    VmemRef acc = s.outputs[o].ref;
                    If(s.outputs[o].is_first_step,
                       [&] { Store(acc, Zeros(acc.shape())); });
                    Store(acc, Load(acc) + Load(s.inputs[0].ref));
                  }
                });
          }));

  TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          VerifyAndSerialize(std::move(kernel)));

  for (int64_t num_pairs : {1, 3}) {
    SCOPED_TRACE(absl::StrCat("num_pairs=", num_pairs));
    const int64_t num_steps = 2 * num_pairs;
    xla::Literal input = Make1dF32(num_steps * kChunk, [](int64_t i) {
      return static_cast<float>(i % 17 - 8);
    });
    // Element `lane` of input chunk `c`, or 0 outside the input.
    auto in = [&](int64_t c, int64_t lane) {
      return c >= 0 && c < num_steps ? input.Get<float>({c * kChunk + lane})
                                     : 0.0f;
    };
    xla::Literal expected = xla::LiteralUtil::MakeTupleOwned(
        Make1dF32(num_pairs * kChunk,
                  [&](int64_t i) {
                    const int64_t pair = num_pairs - 1 - i / kChunk;
                    return in(2 * pair, i % kChunk) +
                           in(2 * pair + 1, i % kChunk);
                  }),
        Make1dF32((num_pairs + 1) * kChunk, [&](int64_t i) {
          const int64_t block = i / kChunk;
          return in(2 * block - 1, i % kChunk) + in(2 * block, i % kChunk);
        }));
    RunAndCompare(serialized, {&input}, expected);
  }
}

TEST_F(KilnE2eTest, DynamicMemrefSliceSizesAndScopedDma) {
  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(GridSpec{.name = "dynamic_slices_e2e"},
                   InSpec::Hbm(f32(kDynamic, kDynamic)),
                   OutSpec::Hbm(f32(kDynamic, kDynamic)),
                   [](HbmRef hbm_in, HbmRef hbm_out) {
                     Scalar r = hbm_in.Dim(0);
                     Scalar c = hbm_in.Dim(1);
                     Scalar half_rows = MultipleOf((r >> 1), 8);
                     Scalar aligned_cols = MultipleOf(c, 128);

                     HbmRef in_top =
                         hbm_in(Slice(0, half_rows), Slice(0, aligned_cols));
                     HbmRef in_bottom = hbm_in(Slice(half_rows, half_rows),
                                               Slice(0, aligned_cols));
                     HbmRef out_top =
                         hbm_out(Slice(0, half_rows), Slice(0, aligned_cols));
                     HbmRef out_bottom = hbm_out(Slice(half_rows, half_rows),
                                                 Slice(0, aligned_cols));

                     RunScoped([&] {
                       Semaphore sem = AllocaDmaSemaphore();
                       AsyncCopy(in_bottom, out_top, sem).Wait();
                       AsyncCopy(in_top, out_bottom, sem).Wait();
                     });
                   }));

  EXPECT_THAT(kernel.Verify(), IsOk());
  TT_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          std::move(kernel).Serialize());

  // Swaps the top and bottom halves of a dynamic 2D HBM tensor using dynamic
  // slice sizes and `RunScoped`.
  for (std::array<int64_t, 2> dims :
       {std::array<int64_t, 2>{16, 128}, {32, 256}}) {
    const int64_t rows = dims[0];
    const int64_t cols = dims[1];
    SCOPED_TRACE(absl::StrCat("rows=", rows, ", cols=", cols));

    xla::Literal input = Make2dF32(rows, cols, [](int64_t r, int64_t c) {
      return static_cast<float>(r * 1000 + c);
    });
    xla::Literal expected = Make2dF32(rows, cols, [&](int64_t r, int64_t c) {
      int64_t src_r = (r < rows / 2) ? (r + rows / 2) : (r - rows / 2);
      return input.Get<float>({src_r, c});
    });
    RunAndCompare(serialized, {&input}, expected);
  }
}

TEST_F(KilnE2eTest, WindowedLoadConcatAndValueIf) {
  // Tests partial non-tile-aligned `Load`, `Concat` along both sublane and lane
  // axes, and value-returning `If` (a tuple of a `Scalar` and a `Tile`) on BF16
  // and F32 tiles.
  constexpr int64_t kRows = 128;
  constexpr int64_t kCols = 256;
  constexpr int64_t kValidRows = 19;
  constexpr int64_t kValidCols = 144;

  IndexMap row_block_map([] { return IndexIndices{ProgramId(0), 0}; });
  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "clamped_load_e2e", .grid = {2}},
          InSpec{bf16(kRows, kCols), row_block_map},
          InSpec{bf16(1, kCols), IndexMap([] { return IndexIndices{0, 0}; })},
          OutSpec{f32(kRows, kCols), row_block_map},
          [&](VmemRef x_ref, VmemRef bias_ref, VmemRef out_ref) {
            Scalar pid = ProgramId(0);
            auto [scale, x] = If(
                pid == 1,
                [&] {
                  Tile head = Load(x_ref, {0, 0}, {kValidRows, kCols});
                  return std::tuple{
                      F32(2.0f),
                      Concat({head, Zeros(bf16(kRows - kValidRows, kCols))},
                             0)};
                },
                [&] { return std::tuple{F32(1.0f), Load(x_ref)}; });
            Tile b = Load(bias_ref, {0, 0}, {1, kValidCols})
                         .AsType(f32)
                         .BroadcastTo({kRows, kValidCols});
            Tile b_padded =
                Concat({b, Zeros(f32(kRows, kCols - kValidCols))}, 1);
            Store(out_ref, x.AsType(f32) * scale + b_padded);
          }));

  xla::Literal x_in(xla::ShapeUtil::MakeShape(xla::BF16, {2 * kRows, kCols}));
  for (int64_t r = 0; r < 2 * kRows; ++r) {
    for (int64_t c = 0; c < kCols; ++c) {
      x_in.Set<xla::bfloat16>(
          {r, c}, static_cast<xla::bfloat16>(static_cast<float>((r + c) % 13)));
    }
  }
  xla::Literal bias_in(xla::ShapeUtil::MakeShape(xla::BF16, {1, kCols}));
  for (int64_t c = 0; c < kCols; ++c) {
    // Put a poison-like sentinel in the tail [kValidCols, kCols) to verify
    // that the clamped load + zero concat masks it out.
    const float v = (c < kValidCols) ? static_cast<float>(c % 7 + 1) : 1000.0f;
    bias_in.Set<xla::bfloat16>({0, c}, static_cast<xla::bfloat16>(v));
  }

  xla::Literal expected =
      Make2dF32(2 * kRows, kCols, [&](int64_t r, int64_t c) {
        const bool in_ragged_tail = (r >= kRows + kValidRows);
        const float xv =
            in_ragged_tail
                ? 0.0f
                : static_cast<float>(x_in.Get<xla::bfloat16>({r, c}));
        const float scale = (r >= kRows) ? 2.0f : 1.0f;
        const float bv =
            (c < kValidCols)
                ? static_cast<float>(bias_in.Get<xla::bfloat16>({0, c}))
                : 0.0f;
        return xv * scale + bv;
      });

  RunAndCompare(std::move(kernel), {&x_in, &bias_in}, expected);
}

TEST_F(KilnE2eTest, MemRefReshapeDynamicStridedBandDma) {
  constexpr int64_t kHbmRows = 64;
  constexpr int64_t kCols = 128;

  TT_ASSERT_OK_AND_ASSIGN(
      Kernel kernel,
      CreateKernel(
          GridSpec{.name = "reshape_strided_band_dma_e2e"},
          ScalarPrefetchSpec{i32(4)}, InSpec::Hbm(u32(kHbmRows, 1, kCols)),
          OutSpec::Hbm(u32(kHbmRows, 1, kCols)),
          [](SmemRef meta, HbmRef in, HbmRef out) {
            const Scalar bands = Load(meta(0));
            const Scalar width = Load(meta(1));
            const Scalar in_stride = Load(meta(2));
            const Scalar out_stride = Load(meta(3));

            VmemRef stage = AllocaVmem(u32(kHbmRows, 1, kCols));
            Store(stage, Zeros(stage.shape()));
            SyncCopy(stage, out);

            HbmRef in_bands =
                in(Slice(0, bands * in_stride), all, all)
                    .Reshape({bands, in_stride, kCols})(
                        Slice(0, bands), Slice(0, width), Slice(0, kCols));
            VmemRef stage_bands = stage(Slice(0, bands * width), all, all)
                                      .Reshape({bands, width, kCols});
            SyncCopy(in_bands, stage_bands);

            HbmRef out_bands =
                out(Slice(0, bands * out_stride), all, all)
                    .Reshape({bands, out_stride, kCols})(
                        Slice(0, bands), Slice(0, width), Slice(0, kCols));
            SyncCopy(stage_bands, out_bands);
          }));

  const xla::Literal meta_lit =
      xla::LiteralUtil::CreateR1<int32_t>({3, 8, 16, 20});
  xla::Literal in_lit(
      xla::ShapeUtil::MakeShape(xla::U32, {kHbmRows, 1, kCols}));
  xla::Literal expected(
      xla::ShapeUtil::MakeShape(xla::U32, {kHbmRows, 1, kCols}));
  for (int64_t r = 0; r < kHbmRows; ++r) {
    for (int64_t c = 0; c < kCols; ++c) {
      in_lit.Set<uint32_t>({r, 0, c}, static_cast<uint32_t>(r * 1000 + c + 1));
      expected.Set<uint32_t>({r, 0, c}, 0u);
    }
  }
  for (int64_t b = 0; b < 3; ++b) {
    for (int64_t w = 0; w < 8; ++w) {
      for (int64_t c = 0; c < kCols; ++c) {
        expected.Set<uint32_t>({b * 20 + w, 0, c},
                               in_lit.Get<uint32_t>({b * 16 + w, 0, c}));
      }
    }
  }
  RunAndCompare(std::move(kernel), {&meta_lit, &in_lit}, expected);
}

}  // namespace
}  // namespace torch_tpu::kiln
