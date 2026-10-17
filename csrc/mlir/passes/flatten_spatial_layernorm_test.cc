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

#include "csrc/mlir/passes/flatten_spatial_layernorm.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 19 + 7) % 29) * 0.125f - 1.0f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

class FlattenSpatialLayerNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  std::string RunPass(const char* ir) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(ir, &context_);
    if (!module) {
      ADD_FAILURE() << "Failed to parse MLIR module";
      return "";
    }
    mlir::PassManager pm(&context_);
    pm.addNestedPass<mlir::func::FuncOp>(createFlattenSpatialLayerNormPass());
    if (mlir::failed(pm.run(*module))) {
      ADD_FAILURE() << "Pass failed to run on module";
      return "";
    }
    std::string out;
    llvm::raw_string_ostream os(out);
    module->print(os);
    return out;
  }

  mlir::MLIRContext context_;
};

TEST_F(FlattenSpatialLayerNormTest,
       Flattens4DChannelLastLayerNormAndFoldsUpstreamTranspose) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @convnext_block_ln(
          %x: tensor<1x256x28x28xbf16>,
          %w: tensor<256xbf16>,
          %b: tensor<256xbf16>) -> tensor<784x256xbf16> {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %inv_n = stablehlo.constant dense<0.00390625> : tensor<1x28x28xf32>
        %zero_v = stablehlo.constant dense<0.0> : tensor<1x28x28xf32>
        %eps = stablehlo.constant dense<1.000000e-06> : tensor<1x28x28xf32>
        %r1 = stablehlo.reshape %x : (tensor<1x256x28x28xbf16>) -> tensor<256x28x28x1xbf16>
        %t1 = stablehlo.transpose %r1, dims = [3, 1, 2, 0] : (tensor<256x28x28x1xbf16>) -> tensor<1x28x28x256xbf16>
        %x_f32 = stablehlo.convert %t1 : (tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xf32>
        %x_sq = stablehlo.multiply %x_f32, %x_f32 : tensor<1x28x28x256xf32>
        %sums:2 = stablehlo.reduce(%x_f32 init: %zero), (%x_sq init: %zero)
             across dimensions = [3]
             : (tensor<1x28x28x256xf32>, tensor<1x28x28x256xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x28x28xf32>, tensor<1x28x28xf32>)
         reducer(%arg0: tensor<f32>, %arg2: tensor<f32>) (%arg1: tensor<f32>, %arg3: tensor<f32>) {
          %a0 = stablehlo.add %arg0, %arg2 : tensor<f32>
          %a1 = stablehlo.add %arg1, %arg3 : tensor<f32>
          stablehlo.return %a0, %a1 : tensor<f32>, tensor<f32>
        }
        %mean = stablehlo.multiply %sums#0, %inv_n : tensor<1x28x28xf32>
        %mean2 = stablehlo.multiply %sums#1, %inv_n : tensor<1x28x28xf32>
        %mean_sq = stablehlo.multiply %mean, %mean : tensor<1x28x28xf32>
        %var_raw = stablehlo.subtract %mean2, %mean_sq : tensor<1x28x28xf32>
        %var = stablehlo.maximum %var_raw, %zero_v : tensor<1x28x28xf32>
        %var_eps = stablehlo.add %var, %eps : tensor<1x28x28xf32>
        %rstd = stablehlo.rsqrt %var_eps : tensor<1x28x28xf32>
        %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<1x28x28xf32>) -> tensor<1x28x28x256xf32>
        %sub = stablehlo.subtract %x_f32, %mean_b : tensor<1x28x28x256xf32>
        %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<1x28x28xf32>) -> tensor<1x28x28x256xf32>
        %norm = stablehlo.multiply %sub, %rstd_b : tensor<1x28x28x256xf32>
        %norm_bf16 = stablehlo.convert %norm : (tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xbf16>
        %w_b = stablehlo.broadcast_in_dim %w, dims = [3] : (tensor<256xbf16>) -> tensor<1x28x28x256xbf16>
        %scaled = stablehlo.multiply %norm_bf16, %w_b : tensor<1x28x28x256xbf16>
        %b_b = stablehlo.broadcast_in_dim %b, dims = [3] : (tensor<256xbf16>) -> tensor<1x28x28x256xbf16>
        %aff = stablehlo.add %scaled, %b_b : tensor<1x28x28x256xbf16>
        %flat = stablehlo.reshape %aff : (tensor<1x28x28x256xbf16>) -> tensor<784x256xbf16>
        return %flat : tensor<784x256xbf16>
      }
    }
  )mlir";

  std::string out = RunPass(kIr);
  EXPECT_THAT(out, HasSubstr("tensor<256x784xbf16>"));
  EXPECT_THAT(out, HasSubstr("dims = [1, 0]"));
  EXPECT_THAT(out, HasSubstr("tensor<784x256xf32>"));
  EXPECT_THAT(out, HasSubstr("tensor<784xf32>"));
  EXPECT_THAT(out, Not(HasSubstr("tensor<1x28x28x256xf32>")));
  EXPECT_THAT(out, Not(HasSubstr("tensor<1x28x28xf32>")));
}

TEST_F(FlattenSpatialLayerNormTest,
       ExactNumericalEquivalenceOnReferenceInterpreter) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(
          %x: tensor<1x8x4x4xf32>,
          %w: tensor<8xf32>,
          %b: tensor<8xf32>) -> tensor<16x8xf32> {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %inv_n = stablehlo.constant dense<0.125> : tensor<1x4x4xf32>
        %zero_v = stablehlo.constant dense<0.0> : tensor<1x4x4xf32>
        %eps = stablehlo.constant dense<1.000000e-06> : tensor<1x4x4xf32>
        %r1 = stablehlo.reshape %x : (tensor<1x8x4x4xf32>) -> tensor<8x4x4x1xf32>
        %t1 = stablehlo.transpose %r1, dims = [3, 1, 2, 0] : (tensor<8x4x4x1xf32>) -> tensor<1x4x4x8xf32>
        %x_sq = stablehlo.multiply %t1, %t1 : tensor<1x4x4x8xf32>
        %sums:2 = stablehlo.reduce(%t1 init: %zero), (%x_sq init: %zero)
             across dimensions = [3]
             : (tensor<1x4x4x8xf32>, tensor<1x4x4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x4x4xf32>, tensor<1x4x4xf32>)
         reducer(%arg0: tensor<f32>, %arg2: tensor<f32>) (%arg1: tensor<f32>, %arg3: tensor<f32>) {
          %a0 = stablehlo.add %arg0, %arg2 : tensor<f32>
          %a1 = stablehlo.add %arg1, %arg3 : tensor<f32>
          stablehlo.return %a0, %a1 : tensor<f32>, tensor<f32>
        }
        %mean = stablehlo.multiply %sums#0, %inv_n : tensor<1x4x4xf32>
        %mean2 = stablehlo.multiply %sums#1, %inv_n : tensor<1x4x4xf32>
        %mean_sq = stablehlo.multiply %mean, %mean : tensor<1x4x4xf32>
        %var_raw = stablehlo.subtract %mean2, %mean_sq : tensor<1x4x4xf32>
        %var = stablehlo.maximum %var_raw, %zero_v : tensor<1x4x4xf32>
        %var_eps = stablehlo.add %var, %eps : tensor<1x4x4xf32>
        %rstd = stablehlo.rsqrt %var_eps : tensor<1x4x4xf32>
        %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<1x4x4xf32>) -> tensor<1x4x4x8xf32>
        %sub = stablehlo.subtract %t1, %mean_b : tensor<1x4x4x8xf32>
        %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<1x4x4xf32>) -> tensor<1x4x4x8xf32>
        %norm = stablehlo.multiply %sub, %rstd_b : tensor<1x4x4x8xf32>
        %w_b = stablehlo.broadcast_in_dim %w, dims = [3] : (tensor<8xf32>) -> tensor<1x4x4x8xf32>
        %scaled = stablehlo.multiply %norm, %w_b : tensor<1x4x4x8xf32>
        %b_b = stablehlo.broadcast_in_dim %b, dims = [3] : (tensor<8xf32>) -> tensor<1x4x4x8xf32>
        %aff = stablehlo.add %scaled, %b_b : tensor<1x4x4x8xf32>
        %flat = stablehlo.reshape %aff : (tensor<1x4x4x8xf32>) -> tensor<16x8xf32>
        return %flat : tensor<16x8xf32>
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context_);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context_);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(createFlattenSpatialLayerNormPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  mlir::DenseElementsAttr x = MakeF32Tensor(context_, {1, 8, 4, 4}, 0.3f);
  mlir::DenseElementsAttr w = MakeF32Tensor(context_, {8}, 1.0f);
  mlir::DenseElementsAttr b = MakeF32Tensor(context_, {8}, -0.2f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, w, b}, cfg);
  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, w, b}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));

  auto exp_it = (*expected)[0].value_begin<llvm::APFloat>();
  auto act_it = (*actual)[0].value_begin<llvm::APFloat>();
  for (; exp_it != (*expected)[0].value_end<llvm::APFloat>();
       ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ((*act_it).convertToFloat(), (*exp_it).convertToFloat());
  }
}

TEST_F(FlattenSpatialLayerNormTest,
       Flattens4DChannelLastLayerNormWithDirectTransposePattern2) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @convnext_pattern2(
          %x: tensor<1x256x28x28xbf16>,
          %w: tensor<256xbf16>,
          %b: tensor<256xbf16>) -> tensor<784x256xbf16> {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %inv_n = stablehlo.constant dense<0.00390625> : tensor<1x28x28xf32>
        %zero_v = stablehlo.constant dense<0.0> : tensor<1x28x28xf32>
        %eps = stablehlo.constant dense<1.000000e-06> : tensor<1x28x28xf32>
        %t1 = stablehlo.transpose %x, dims = [0, 2, 3, 1] : (tensor<1x256x28x28xbf16>) -> tensor<1x28x28x256xbf16>
        %x_f32 = stablehlo.convert %t1 : (tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xf32>
        %x_sq = stablehlo.multiply %x_f32, %x_f32 : tensor<1x28x28x256xf32>
        %sums:2 = stablehlo.reduce(%x_f32 init: %zero), (%x_sq init: %zero)
             across dimensions = [3]
             : (tensor<1x28x28x256xf32>, tensor<1x28x28x256xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x28x28xf32>, tensor<1x28x28xf32>)
         reducer(%arg0: tensor<f32>, %arg2: tensor<f32>) (%arg1: tensor<f32>, %arg3: tensor<f32>) {
          %a0 = stablehlo.add %arg0, %arg2 : tensor<f32>
          %a1 = stablehlo.add %arg1, %arg3 : tensor<f32>
          stablehlo.return %a0, %a1 : tensor<f32>, tensor<f32>
        }
        %mean = stablehlo.multiply %sums#0, %inv_n : tensor<1x28x28xf32>
        %mean2 = stablehlo.multiply %sums#1, %inv_n : tensor<1x28x28xf32>
        %mean_sq = stablehlo.multiply %mean, %mean : tensor<1x28x28xf32>
        %var_raw = stablehlo.subtract %mean2, %mean_sq : tensor<1x28x28xf32>
        %var = stablehlo.maximum %var_raw, %zero_v : tensor<1x28x28xf32>
        %var_eps = stablehlo.add %var, %eps : tensor<1x28x28xf32>
        %rstd = stablehlo.rsqrt %var_eps : tensor<1x28x28xf32>
        %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<1x28x28xf32>) -> tensor<1x28x28x256xf32>
        %sub = stablehlo.subtract %x_f32, %mean_b : tensor<1x28x28x256xf32>
        %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<1x28x28xf32>) -> tensor<1x28x28x256xf32>
        %norm = stablehlo.multiply %sub, %rstd_b : tensor<1x28x28x256xf32>
        %norm_bf16 = stablehlo.convert %norm : (tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xbf16>
        %w_b = stablehlo.broadcast_in_dim %w, dims = [3] : (tensor<256xbf16>) -> tensor<1x28x28x256xbf16>
        %scaled = stablehlo.multiply %norm_bf16, %w_b : tensor<1x28x28x256xbf16>
        %b_b = stablehlo.broadcast_in_dim %b, dims = [3] : (tensor<256xbf16>) -> tensor<1x28x28x256xbf16>
        %aff = stablehlo.add %scaled, %b_b : tensor<1x28x28x256xbf16>
        %flat = stablehlo.reshape %aff : (tensor<1x28x28x256xbf16>) -> tensor<784x256xbf16>
        return %flat : tensor<784x256xbf16>
      }
    }
  )mlir";

  std::string out = RunPass(kIr);
  EXPECT_THAT(out, HasSubstr("tensor<256x784xbf16>"));
  EXPECT_THAT(out, HasSubstr("dims = [1, 0]"));
  EXPECT_THAT(out, HasSubstr("tensor<784x256xf32>"));
  EXPECT_THAT(out, HasSubstr("tensor<784xf32>"));
  EXPECT_THAT(out, Not(HasSubstr("tensor<1x28x28x256xf32>")));
  EXPECT_THAT(out, Not(HasSubstr("tensor<1x28x28xf32>")));
}

TEST_F(FlattenSpatialLayerNormTest,
       Flattens4DLayerNormWithoutUpstreamTranspose) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @ln_no_transpose(
          %x: tensor<1x28x28x256xbf16>,
          %w: tensor<256xbf16>,
          %b: tensor<256xbf16>) -> tensor<784x256xbf16> {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %inv_n = stablehlo.constant dense<0.00390625> : tensor<1x28x28xf32>
        %zero_v = stablehlo.constant dense<0.0> : tensor<1x28x28xf32>
        %eps = stablehlo.constant dense<1.000000e-06> : tensor<1x28x28xf32>
        %x_f32 = stablehlo.convert %x : (tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xf32>
        %x_sq = stablehlo.multiply %x_f32, %x_f32 : tensor<1x28x28x256xf32>
        %sums:2 = stablehlo.reduce(%x_f32 init: %zero), (%x_sq init: %zero)
             across dimensions = [3]
             : (tensor<1x28x28x256xf32>, tensor<1x28x28x256xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x28x28xf32>, tensor<1x28x28xf32>)
         reducer(%arg0: tensor<f32>, %arg2: tensor<f32>) (%arg1: tensor<f32>, %arg3: tensor<f32>) {
          %a0 = stablehlo.add %arg0, %arg2 : tensor<f32>
          %a1 = stablehlo.add %arg1, %arg3 : tensor<f32>
          stablehlo.return %a0, %a1 : tensor<f32>, tensor<f32>
        }
        %mean = stablehlo.multiply %sums#0, %inv_n : tensor<1x28x28xf32>
        %mean2 = stablehlo.multiply %sums#1, %inv_n : tensor<1x28x28xf32>
        %mean_sq = stablehlo.multiply %mean, %mean : tensor<1x28x28xf32>
        %var_raw = stablehlo.subtract %mean2, %mean_sq : tensor<1x28x28xf32>
        %var = stablehlo.maximum %var_raw, %zero_v : tensor<1x28x28xf32>
        %var_eps = stablehlo.add %var, %eps : tensor<1x28x28xf32>
        %rstd = stablehlo.rsqrt %var_eps : tensor<1x28x28xf32>
        %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<1x28x28xf32>) -> tensor<1x28x28x256xf32>
        %sub = stablehlo.subtract %x_f32, %mean_b : tensor<1x28x28x256xf32>
        %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<1x28x28xf32>) -> tensor<1x28x28x256xf32>
        %norm = stablehlo.multiply %sub, %rstd_b : tensor<1x28x28x256xf32>
        %norm_bf16 = stablehlo.convert %norm : (tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xbf16>
        %w_b = stablehlo.broadcast_in_dim %w, dims = [3] : (tensor<256xbf16>) -> tensor<1x28x28x256xbf16>
        %scaled = stablehlo.multiply %norm_bf16, %w_b : tensor<1x28x28x256xbf16>
        %b_b = stablehlo.broadcast_in_dim %b, dims = [3] : (tensor<256xbf16>) -> tensor<1x28x28x256xbf16>
        %aff = stablehlo.add %scaled, %b_b : tensor<1x28x28x256xbf16>
        %flat = stablehlo.reshape %aff : (tensor<1x28x28x256xbf16>) -> tensor<784x256xbf16>
        return %flat : tensor<784x256xbf16>
      }
    }
  )mlir";

  std::string out = RunPass(kIr);
  EXPECT_THAT(out, HasSubstr("tensor<784x256xf32>"));
  EXPECT_THAT(out, HasSubstr("tensor<784xf32>"));
  EXPECT_THAT(out, Not(HasSubstr("dims = [1, 0]")));
  EXPECT_THAT(out, Not(HasSubstr("tensor<1x28x28x256xf32>")));
  EXPECT_THAT(out, Not(HasSubstr("tensor<1x28x28xf32>")));
}

TEST_F(FlattenSpatialLayerNormTest, LeavesModuleUnchangedWhenMomentsEscape) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @moments_escape(
          %x: tensor<1x256x28x28xbf16>,
          %w: tensor<256xbf16>,
          %b: tensor<256xbf16>) -> (tensor<784x256xbf16>, tensor<1x28x28xf32>) {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %inv_n = stablehlo.constant dense<0.00390625> : tensor<1x28x28xf32>
        %zero_v = stablehlo.constant dense<0.0> : tensor<1x28x28xf32>
        %eps = stablehlo.constant dense<1.000000e-06> : tensor<1x28x28xf32>
        %r1 = stablehlo.reshape %x : (tensor<1x256x28x28xbf16>) -> tensor<256x28x28x1xbf16>
        %t1 = stablehlo.transpose %r1, dims = [3, 1, 2, 0] : (tensor<256x28x28x1xbf16>) -> tensor<1x28x28x256xbf16>
        %x_f32 = stablehlo.convert %t1 : (tensor<1x28x28x256xbf16>) -> tensor<1x28x28x256xf32>
        %x_sq = stablehlo.multiply %x_f32, %x_f32 : tensor<1x28x28x256xf32>
        %sums:2 = stablehlo.reduce(%x_f32 init: %zero), (%x_sq init: %zero)
             across dimensions = [3]
             : (tensor<1x28x28x256xf32>, tensor<1x28x28x256xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x28x28xf32>, tensor<1x28x28xf32>)
         reducer(%arg0: tensor<f32>, %arg2: tensor<f32>) (%arg1: tensor<f32>, %arg3: tensor<f32>) {
          %a0 = stablehlo.add %arg0, %arg2 : tensor<f32>
          %a1 = stablehlo.add %arg1, %arg3 : tensor<f32>
          stablehlo.return %a0, %a1 : tensor<f32>, tensor<f32>
        }
        %mean = stablehlo.multiply %sums#0, %inv_n : tensor<1x28x28xf32>
        %mean2 = stablehlo.multiply %sums#1, %inv_n : tensor<1x28x28xf32>
        %mean_sq = stablehlo.multiply %mean, %mean : tensor<1x28x28xf32>
        %var_raw = stablehlo.subtract %mean2, %mean_sq : tensor<1x28x28xf32>
        %var = stablehlo.maximum %var_raw, %zero_v : tensor<1x28x28xf32>
        %var_eps = stablehlo.add %var, %eps : tensor<1x28x28xf32>
        %rstd = stablehlo.rsqrt %var_eps : tensor<1x28x28xf32>
        %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<1x28x28xf32>) -> tensor<1x28x28x256xf32>
        %sub = stablehlo.subtract %x_f32, %mean_b : tensor<1x28x28x256xf32>
        %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<1x28x28xf32>) -> tensor<1x28x28x256xf32>
        %norm = stablehlo.multiply %sub, %rstd_b : tensor<1x28x28x256xf32>
        %norm_bf16 = stablehlo.convert %norm : (tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xbf16>
        %w_b = stablehlo.broadcast_in_dim %w, dims = [3] : (tensor<256xbf16>) -> tensor<1x28x28x256xbf16>
        %scaled = stablehlo.multiply %norm_bf16, %w_b : tensor<1x28x28x256xbf16>
        %b_b = stablehlo.broadcast_in_dim %b, dims = [3] : (tensor<256xbf16>) -> tensor<1x28x28x256xbf16>
        %aff = stablehlo.add %scaled, %b_b : tensor<1x28x28x256xbf16>
        %flat = stablehlo.reshape %aff : (tensor<1x28x28x256xbf16>) -> tensor<784x256xbf16>
        return %flat, %mean : tensor<784x256xbf16>, tensor<1x28x28xf32>
      }
    }
  )mlir";

  std::string out = RunPass(kIr);
  EXPECT_THAT(out, HasSubstr("tensor<1x28x28x256xf32>"));
  EXPECT_THAT(out, HasSubstr("tensor<1x28x28xf32>"));
  EXPECT_THAT(out, Not(HasSubstr("tensor<784xf32>")));
}

}  // namespace
}  // namespace torch_tpu
