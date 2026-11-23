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

// ANY_OP_IMPL_FILENAME_OK=Unit test for cumsum lowering strategy selector.

#include "csrc/ops/cumsum/cumsum.h"

#include <cstdint>
#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

using testing::HasSubstr;
using testing::Not;

std::string PrintModule(mlir::ModuleOp module) {
  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  return out;
}

class CumsumStrategyTest : public testing::Test {
 protected:
  CumsumStrategyTest() {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    ctx_.appendDialectRegistry(registry);
    ctx_.loadAllAvailableDialects();
  }

  mlir::RankedTensorType F32(llvm::ArrayRef<int64_t> shape) {
    return mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx_));
  }

  mlir::MLIRContext ctx_;
};

TEST_F(CumsumStrategyTest, SelectsReduceWindowForMambaSegsumAndShortAxes) {
  // Mamba-2 / Nemotron-H 5D segsum (67.1M elems, L=128) -> reduce_window.
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({8, 64, 8, 128, 128}), 3),
            CumsumLoweringStrategy::kReduceWindow);
  // Standalone [8, 64, 8, 128] dim=2 (L=8) & dim=3 (L=128) -> reduce_window.
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({8, 64, 8, 128}), 2),
            CumsumLoweringStrategy::kReduceWindow);
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({8, 64, 8, 128}), 3),
            CumsumLoweringStrategy::kReduceWindow);
  // Exact threshold check: 2048 * 128 = 262144 elems, L=128 -> reduce_window.
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({2048, 128}), 1),
            CumsumLoweringStrategy::kReduceWindow);
}

TEST_F(CumsumStrategyTest, PreservesChloScanForLongSmallAndDynamicShapes) {
  // Long axis L=2048 (> 128 lane width) -> chlo.ScanOp (2.98x faster).
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({4, 64, 2048, 128}), 2),
            CumsumLoweringStrategy::kChloScan);
  // Boundary L=129 -> chlo.ScanOp.
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({4096, 129}), 1),
            CumsumLoweringStrategy::kChloScan);
  // Small tensor [2, 32, 32, 64] (131K elems < 262144) -> chlo.ScanOp.
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({2, 32, 32, 64}), 3),
            CumsumLoweringStrategy::kChloScan);
  // Trivial axis L=1 / empty -> chlo.ScanOp.
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({512, 1024, 1}), 2),
            CumsumLoweringStrategy::kChloScan);
  // Empty tensor with huge leading dim stays on kChloScan.
  EXPECT_EQ(SelectCumsumLoweringStrategy(F32({524288, 0, 128}), 2),
            CumsumLoweringStrategy::kChloScan);
  // Dynamic shape -> chlo.ScanOp (safe fallback for bounded dynamism).
  EXPECT_EQ(SelectCumsumLoweringStrategy(
                F32({8, mlir::ShapedType::kDynamic, 128}), 2),
            CumsumLoweringStrategy::kChloScan);
}

TEST_F(CumsumStrategyTest, BuildCumsumEmitsExpectedOp) {
  auto build_ir = [this](llvm::ArrayRef<int64_t> shape,
                         int64_t dim) -> std::string {
    mlir::ModuleBuilder mb(ctx_, mlir::unknownLoc(ctx_));
    {
      mlir::func::FunctionBuilder fb(mb, "main");
      mlir::MlirOp arg = mlir::func::Argument(fb, F32(shape));
      auto out_or = BuildCumsumShlo(dim, std::nullopt, arg);
      EXPECT_TRUE(out_or.ok()) << out_or.status();
      if (!out_or.ok()) {
        return "";
      }
      mlir::func::Return(fb, *out_or);
    }
    mlir::OwningOpRef<mlir::ModuleOp> mod = mb.build();
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*mod)));
    return PrintModule(*mod);
  };

  const std::string segsum_ir = build_ir({8, 64, 8, 128, 128}, 3);
  EXPECT_THAT(segsum_ir, HasSubstr("stablehlo.reduce_window"));
  EXPECT_THAT(segsum_ir, Not(HasSubstr("chlo.scan")));

  const std::string long_ir = build_ir({4, 64, 2048, 128}, 2);
  EXPECT_THAT(long_ir, HasSubstr("chlo.scan"));
  EXPECT_THAT(long_ir, Not(HasSubstr("stablehlo.reduce_window")));

  const std::string small_ir = build_ir({2, 32, 32, 64}, 3);
  EXPECT_THAT(small_ir, HasSubstr("chlo.scan"));
}

}  // namespace
}  // namespace torch_tpu
