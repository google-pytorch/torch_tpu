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

// ANY_OP_IMPL_FILENAME_OK=unit test file
#include "csrc/ops/linalg/solve_triangular/linalg_solve_triangular_kernels.h"

#include <array>
#include <cstdint>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class TestModuleBuilder {
 public:
  TestModuleBuilder()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::ModuleBuilder& get() { return module_builder_; }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

// Returns an f32 function argument of `shape` with `bound_dims` made bounded
// dynamic by `set_dimension_size` with runtime size arguments.
mlir::MlirOp MakeBoundedArg(mlir::func::FunctionBuilder& fb,
                            const Dimensions& shape,
                            absl::Span<const int64_t> bound_dims) {
  mlir::OpBuilder& op_builder = fb.getOpBuilder();
  mlir::MlirOp op = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(shape, op_builder.getF32Type()));
  for (const int64_t dim : bound_dims) {
    mlir::MlirOp size = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
    op = mlir::stablehlo::SetDimensionSize(op, size, dim);
  }
  return op;
}

absl::StatusOr<mlir::MlirOp> BuildSolveTriangular(
    mlir::MlirOp a, mlir::MlirOp b, SolveTriangularOptions options) {
  std::array<mlir::MlirOp, 2> inputs = {a, b};
  return LinalgSolveTriangularBuilder(options)(
      FixedSizeSpan<mlir::MlirOp, 2>(inputs));
}

TEST(LinalgSolveTriangularBuilderTest,
     EmitsStaticTriangularSolveForBoundedOperands) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp a = MakeBoundedArg(fb, {6, 8, 8}, {0, 1, 2});
  mlir::MlirOp b = MakeBoundedArg(fb, {6, 8, 5}, {0, 1, 2});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildSolveTriangular(a, b, {.upper = true, .left = true}));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  int num_solves = 0;
  module->walk([&](mlir::stablehlo::TriangularSolveOp solve) {
    ++num_solves;
    EXPECT_TRUE(solve.getA().getType().hasStaticShape());
    EXPECT_TRUE(solve.getB().getType().hasStaticShape());
    EXPECT_TRUE(solve.getType().hasStaticShape());
  });
  EXPECT_EQ(num_solves, 1);

  const mlir::stablehlo::Dimensions dims = GetDimensions(result);
  ASSERT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0].size, 6);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 8);
  EXPECT_TRUE(dims[1].boundOp.has_value());
  EXPECT_EQ(dims[2].size, 5);
  EXPECT_TRUE(dims[2].boundOp.has_value());
}

TEST(LinalgSolveTriangularBuilderTest,
     UsesLastDimAsSharedMatrixDimForRightSide) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  // Solving X * A = B makes the last dimension of `b` the one shared with `a`.
  // It is static on `b`, so only `a` carries its runtime size.
  mlir::MlirOp a = MakeBoundedArg(fb, {6, 8, 8}, {0, 1, 2});
  mlir::MlirOp b = MakeBoundedArg(fb, {6, 5, 8}, {0, 1});

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result,
                          BuildSolveTriangular(a, b, {.left = false}));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  int num_solves = 0;
  module->walk([&](mlir::stablehlo::TriangularSolveOp solve) {
    ++num_solves;
    EXPECT_FALSE(solve.getLeftSide());
  });
  EXPECT_EQ(num_solves, 1);

  const mlir::stablehlo::Dimensions dims = GetDimensions(result);
  ASSERT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0].size, 6);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 5);
  EXPECT_TRUE(dims[1].boundOp.has_value());
  EXPECT_EQ(dims[2].size, 8);
  EXPECT_TRUE(dims[2].boundOp.has_value());
}

TEST(LinalgSolveTriangularBuilderTest,
     PassesStaticMatrixUnpaddedForBoundedRhs) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp a = MakeBoundedArg(fb, {8, 8}, {});
  mlir::MlirOp b = MakeBoundedArg(fb, {8, 5}, {1});

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result,
                          BuildSolveTriangular(a, b, {.left = true}));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  int num_solves = 0;
  module->walk([&](mlir::stablehlo::TriangularSolveOp solve) {
    ++num_solves;
    EXPECT_EQ(solve.getA(), a.getValue());
    EXPECT_TRUE(solve.getB().getType().hasStaticShape());
  });
  EXPECT_EQ(num_solves, 1);

  const mlir::stablehlo::Dimensions dims = GetDimensions(result);
  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 8);
  EXPECT_FALSE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 5);
  EXPECT_TRUE(dims[1].boundOp.has_value());
}

TEST(LinalgSolveTriangularBuilderTest,
     PropagatesMatrixRuntimeSizeToStaticSharedDim) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  // The dimension of `b` shared with `a` (dimension 1) is static, so only `a`
  // carries its runtime size.
  mlir::MlirOp a = MakeBoundedArg(fb, {6, 8, 8}, {0, 1, 2});
  mlir::MlirOp b = MakeBoundedArg(fb, {6, 8, 5}, {0, 2});

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result,
                          BuildSolveTriangular(a, b, {.left = true}));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::stablehlo::Dimensions dims = GetDimensions(result);
  ASSERT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0].size, 6);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 8);
  EXPECT_TRUE(dims[1].boundOp.has_value());
  EXPECT_EQ(dims[2].size, 5);
  EXPECT_TRUE(dims[2].boundOp.has_value());
}

TEST(LinalgSolveTriangularBuilderTest, RejectsNonSquareStaticMatrix) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp a = MakeBoundedArg(fb, {6, 8, 10}, {});
  mlir::MlirOp b = MakeBoundedArg(fb, {6, 8, 5}, {});

  EXPECT_THAT(
      BuildSolveTriangular(a, b, {.left = true}),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("expected square matrix in triangular solve, got 8 and "
                    "10")));
}

TEST(LinalgSolveTriangularBuilderTest, RejectsMismatchedStaticSharedMatrixDim) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp a = MakeBoundedArg(fb, {6, 8, 8}, {});
  mlir::MlirOp b = MakeBoundedArg(fb, {6, 10, 5}, {});

  EXPECT_THAT(
      BuildSolveTriangular(a, b, {.left = true}),
      StatusIs(error::kInvalidArgument,
               HasSubstr("expected the shared matrix dimensions in triangular "
                         "solve to match, got 8 and 10")));
}

TEST(LinalgSolveTriangularBuilderTest, RejectsNonSquareUpperBoundMatrix) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp a = MakeBoundedArg(fb, {6, 8, 10}, {0, 1, 2});
  mlir::MlirOp b = MakeBoundedArg(fb, {6, 8, 5}, {0, 1, 2});

  EXPECT_THAT(
      BuildSolveTriangular(a, b, {.left = true}),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr(
              "expected square upper-bound matrix in triangular solve, got 8 "
              "and 10")));
}

TEST(LinalgSolveTriangularBuilderTest,
     RejectsMismatchedSharedMatrixUpperBound) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp a = MakeBoundedArg(fb, {6, 8, 8}, {0, 1, 2});
  mlir::MlirOp b = MakeBoundedArg(fb, {6, 10, 5}, {0, 1});

  EXPECT_THAT(
      BuildSolveTriangular(a, b, {.left = true}),
      StatusIs(error::kInvalidArgument,
               HasSubstr("expected the upper bounds of the shared matrix "
                         "dimension in triangular solve to match, got 8 and "
                         "10")));
}

}  // namespace
}  // namespace torch_tpu
