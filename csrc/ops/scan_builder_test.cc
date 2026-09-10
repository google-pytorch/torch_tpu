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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

// We disable clang-format here to force the StableHLO Reference headers to
// be included before the TorchTPU StablehloBuilder headers. This prevents
// a namespace collision where both packages attempt to define 'Tuple'.
// TODO(b/505045588): Remove this workaround after cl/927205178.
// clang-format off
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"
// clang-format on

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/native_scan_support.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/scan_builder.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/Passes.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;

mlir::MlirOp MakeDenseConstant(mlir::MlirBuilder& builder,
                               llvm::ArrayRef<int64_t> shape,
                               llvm::ArrayRef<int32_t> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type =
      mlir::RankedTensorType::get(shape, op_builder.getI32Type());
  const mlir::DenseElementsAttr attr =
      mlir::DenseElementsAttr::get(type, values);
  return mlir::stablehlo::Constant(builder, attr);
}
using testing::HasSubstr;

// Parameterized over the scan lowering: false = StableHLO while loop, true =
// chlo.ScanOp (native scan emitter). Every (associative) scenario must produce
// identical results on both paths.
class ScanBuilderTest : public testing::TestWithParam<bool> {
 protected:
  ScanBuilderTest()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::MlirBuilder& builder() { return module_builder_; }
  mlir::ModuleBuilder& module_builder() { return module_builder_; }
  mlir::OpBuilder& op_builder() { return module_builder_.getOpBuilder(); }
  mlir::MLIRContext* context() { return &context_; }

  mlir::MlirOp CreateBoundedTensorArg(
      mlir::func::FunctionBuilder& function_builder,
      mlir::ArrayRef<int64_t> shape, mlir::ArrayRef<int64_t> bound_dims,
      mlir::Type element_type) {
    // A bounded dynamic tensor requires:
    // 1. `dim_sizes`: The base shape, where dynamic dims are marked with
    //    kDynamic.
    // 2. `bounds_vec`: The memory padding limits. Static dimensions have no
    //    bounds, so they are marked with kDynamic.
    mlir::SmallVector<int64_t> bounds_vec;
    mlir::SmallVector<int64_t> dim_sizes;
    for (int64_t i = 0; i < shape.size(); ++i) {
      if (absl::c_find(bound_dims, i) != bound_dims.end()) {
        bounds_vec.push_back(shape[i]);
        dim_sizes.push_back(mlir::ShapedType::kDynamic);
      } else {
        bounds_vec.push_back(mlir::ShapedType::kDynamic);
        dim_sizes.push_back(shape[i]);
      }
    }

    const auto bounds_attr =
        mlir::DenseI64ArrayAttr::get(context(), bounds_vec);
    const auto encoding =
        mlir::stablehlo::TypeExtensionsAttr::get(context(), bounds_attr);
    const auto type =
        mlir::RankedTensorType::get(dim_sizes, element_type, encoding);

    for (const int64_t bound_dim : bound_dims) {
      EXPECT_TRUE(type.isDynamicDim(bound_dim));
    }

    return mlir::func::Argument(function_builder, type);
  }

  // Body builder that adds the slice to the carry (cumulative sum).
  ScanBodyBuilder CreateAddBodyBuilder() {
    return
        [this](mlir::OpBuilder& op_builder, mlir::Location loc,
               mlir::Value slice, mlir::Value index, mlir::ValueRange carries)
            -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
          MultiInputScanBodyBuilder multi_builder = CreateMultiAddBodyBuilder();
          llvm::SmallVector<mlir::Value, 1> slices = {slice};
          TT_ASSIGN_OR_RETURN(
              ScanBodyResults results,
              multi_builder(op_builder, loc, slices, index, carries));
          return results.new_carries;
        };
  }

  // Body builder that adds each input slice to its corresponding carry
  // (cumulative sum).
  MultiInputScanBodyBuilder CreateMultiAddBodyBuilder() {
    return [](mlir::OpBuilder& op_builder, mlir::Location loc,
              mlir::ValueRange slices, mlir::Value index,
              mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
      llvm::SmallVector<mlir::Value> new_carries;
      for (size_t i = 0; i < slices.size(); ++i) {
        new_carries.push_back(mlir::stablehlo::AddOp::create(
                                  op_builder, loc, slices[i], carries[i])
                                  .getResult());
      }
      return ScanBodyResults{new_carries, new_carries};
    };
  }

  // Body builder that returns carries directly as new carries and outputs.
  MultiInputScanBodyBuilder CreateIdentityBodyBuilder() {
    return [](mlir::OpBuilder& op_builder, mlir::Location loc,
              mlir::ValueRange slices, mlir::Value index,
              mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
      llvm::SmallVector<mlir::Value> carries_vec(carries.begin(),
                                                 carries.end());
      return ScanBodyResults{carries_vec, carries_vec};
    };
  }

  bool IsEmitter() const { return GetParam(); }

  // The scan lowering op: chlo.ScanOp on the emitter path, stablehlo.while
  // otherwise.
  void ExpectScanLoweringOp(mlir::Operation* op) {
    if (IsEmitter()) {
      EXPECT_TRUE(mlir::isa<mlir::chlo::ScanOp>(op));
    } else {
      EXPECT_TRUE(mlir::isa<mlir::stablehlo::WhileOp>(op));
    }
  }

  // Route BuildScanShlo to the while loop or the chlo.ScanOp emitter based on
  // the test parameter, so each scenario is exercised on both paths.
  absl::StatusOr<DynamicMlirOpResults> Scan(
      mlir::MlirBuilder& scan_builder, mlir::MlirOp input, int64_t dim,
      llvm::ArrayRef<mlir::MlirOp> carry_inits,
      llvm::ArrayRef<mlir::MlirOp> output_inits, ScanBodyBuilder body,
      ScanOptions options = {}) {
    options.is_associative = IsEmitter();
    return BuildScanShlo(scan_builder, input, dim, carry_inits, output_inits,
                         std::move(body), options);
  }

  absl::StatusOr<DynamicMlirOpResults> Scan(
      mlir::MlirBuilder& scan_builder, llvm::ArrayRef<mlir::MlirOp> scan_inputs,
      int64_t dim, int64_t num_scan_inputs,
      llvm::ArrayRef<mlir::MlirOp> carry_inits,
      llvm::ArrayRef<mlir::MlirOp> output_inits, MultiInputScanBodyBuilder body,
      ScanOptions options = {}) {
    options.is_associative = IsEmitter();
    return BuildScanShlo(scan_builder, scan_inputs, dim, num_scan_inputs,
                         carry_inits, output_inits, std::move(body), options);
  }

  // chlo.ScanOp must be decomposed to StableHLO before the reference
  // interpreter can evaluate it. A no-op for the pure-StableHLO while loop.
  void LegalizeChlo(mlir::ModuleOp module) {
    // ChloLegalizeToStablehloPass is anchored on func::FuncOp.
    mlir::PassManager pm(context());
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::stablehlo::createChloLegalizeToStablehloPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(module)));
    // Older pinned StableHLO versions decompose chlo.scan using ops the
    // reference interpreter rejects (the deprecated stablehlo.broadcast and a
    // runtime-start real_dynamic_slice). Rewrite both to interpreter-supported
    // forms so the module is evaluatable on any pinned version. Newer versions
    // already emit the supported forms, so these are no-ops there.
    RewriteDeprecatedBroadcasts(module);
    RewriteStaticRealDynamicSlices(module);
  }

  // Rewrites every deprecated stablehlo.broadcast to broadcast_in_dim.
  // stablehlo.broadcast prepends its broadcast_sizes dimensions to the operand
  // shape, so the operand maps to the trailing operand-rank dimensions of the
  // result.
  static void RewriteDeprecatedBroadcasts(mlir::ModuleOp module) {
    llvm::SmallVector<mlir::stablehlo::BroadcastOp> ops;
    module.walk([&](mlir::stablehlo::BroadcastOp op) { ops.push_back(op); });
    for (mlir::stablehlo::BroadcastOp op : ops) {
      mlir::OpBuilder builder(op);
      auto resultType = mlir::cast<mlir::RankedTensorType>(op.getType());
      auto operandType =
          mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
      const int64_t numPrepended = resultType.getRank() - operandType.getRank();
      llvm::SmallVector<int64_t> broadcastDims;
      broadcastDims.reserve(operandType.getRank());
      for (int64_t i = 0; i < operandType.getRank(); ++i) {
        broadcastDims.push_back(numPrepended + i);
      }
      auto newOp = mlir::stablehlo::BroadcastInDimOp::create(
          builder, op.getLoc(), resultType, op.getOperand(),
          builder.getDenseI64ArrayAttr(broadcastDims));
      op.getResult().replaceAllUsesWith(newOp.getResult());
      op.erase();
    }
  }

  // Rewrites every static-result stablehlo.real_dynamic_slice to dynamic_slice,
  // which the reference interpreter supports. The scan input slice has a
  // runtime start but unit strides and a static result shape, so it maps
  // directly: the per-dimension start scalars are pulled out of the
  // start_indices tensor and the slice sizes are the (static) result shape.
  // Genuinely dynamic-shaped slices are left untouched.
  static void RewriteStaticRealDynamicSlices(mlir::ModuleOp module) {
    llvm::SmallVector<mlir::stablehlo::RealDynamicSliceOp> ops;
    module.walk(
        [&](mlir::stablehlo::RealDynamicSliceOp op) { ops.push_back(op); });
    for (mlir::stablehlo::RealDynamicSliceOp op : ops) {
      auto resultType = mlir::cast<mlir::RankedTensorType>(op.getType());
      if (!resultType.hasStaticShape()) continue;
      mlir::OpBuilder builder(op);
      mlir::Value startTensor = op.getStartIndices();
      auto scalarType = mlir::RankedTensorType::get(
          {}, mlir::cast<mlir::RankedTensorType>(startTensor.getType())
                  .getElementType());
      const int64_t rank = resultType.getRank();
      // dynamic_slice takes one scalar start index per dimension; extract each
      // element of the 1-D start_indices tensor via slice + reshape.
      llvm::SmallVector<mlir::Value> startScalars;
      startScalars.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        const int64_t lo = i;
        const int64_t hi = i + 1;
        const int64_t step = 1;
        auto element = mlir::stablehlo::SliceOp::create(
            builder, op.getLoc(), startTensor,
            builder.getDenseI64ArrayAttr({lo}),
            builder.getDenseI64ArrayAttr({hi}),
            builder.getDenseI64ArrayAttr({step}));
        startScalars.push_back(mlir::stablehlo::ReshapeOp::create(
                                   builder, op.getLoc(), scalarType, element)
                                   .getResult());
      }
      auto newOp = mlir::stablehlo::DynamicSliceOp::create(
          builder, op.getLoc(), resultType, op.getOperand(), startScalars,
          builder.getDenseI64ArrayAttr(resultType.getShape()));
      op.getResult().replaceAllUsesWith(newOp.getResult());
      op.erase();
    }
  }

  void EvaluateAndVerifyOutputs(mlir::ModuleOp module,
                                llvm::ArrayRef<int32_t> expected_values) {
    LegalizeChlo(module);
    const mlir::stablehlo::InterpreterConfiguration config;
    const llvm::SmallVector<mlir::DenseElementsAttr> empty_inputs;
    const mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>>
        eval_result = mlir::stablehlo::evalModule(module, empty_inputs, config);
    ASSERT_TRUE(mlir::succeeded(eval_result));
    ASSERT_EQ(eval_result->size(), 1);

    const mlir::DenseElementsAttr res_attr = (*eval_result)[0];
    const llvm::SmallVector<int32_t> res_vals(res_attr.getValues<int32_t>());
    EXPECT_THAT(res_vals, testing::ElementsAreArray(expected_values));
  }

  void EvaluateAndVerifyMultiOutputs(
      mlir::ModuleOp module,
      llvm::ArrayRef<llvm::ArrayRef<int32_t>> expected_outputs) {
    LegalizeChlo(module);
    const mlir::stablehlo::InterpreterConfiguration config;
    const llvm::SmallVector<mlir::DenseElementsAttr> empty_inputs;
    const mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>>
        eval_result = mlir::stablehlo::evalModule(module, empty_inputs, config);
    ASSERT_TRUE(mlir::succeeded(eval_result));
    ASSERT_EQ(eval_result->size(), expected_outputs.size());

    for (size_t i = 0; i < expected_outputs.size(); ++i) {
      const mlir::DenseElementsAttr res_attr = (*eval_result)[i];
      const llvm::SmallVector<int32_t> res_vals(res_attr.getValues<int32_t>());
      EXPECT_THAT(res_vals, testing::ElementsAreArray(expected_outputs[i]));
    }
  }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

using MultiScanBuilderTest = ScanBuilderTest;

INSTANTIATE_TEST_SUITE_P(ScanLowering, ScanBuilderTest, testing::Bool(),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "Emitter" : "WhileLoop";
                         });
INSTANTIATE_TEST_SUITE_P(ScanLowering, MultiScanBuilderTest, testing::Bool(),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "Emitter" : "WhileLoop";
                         });

TEST_P(ScanBuilderTest, InvalidDimension) {
  // Input tensor of shape [4, 5]. Scan along an invalid dimension 2.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator. For cumulative sum, the output shape matches the input.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry tensor. The shape does not matter in this case, since an error will
  // be returned.
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), input, /*dim=*/2, {carry_init}, {output_init},
           CreateAddBodyBuilder());
  EXPECT_THAT(result, StatusIs(error::kPythonIndexError,
                               HasSubstr("expected dimension to be in range")));
}

TEST_P(ScanBuilderTest, EmptyTensor) {
  // Input tensor of shape [0, 5], empty along the scan dimension.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {0, 5});
  // Output accumulator. For cumulative sum, the output shape matches the input.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {0, 5});
  // Carry tensor. The scan loop extracts a slice of the input along the
  // scan dimension (size 1) at each iteration. Therefore, the carry
  // initialization must match this sliced shape.
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  TT_ASSERT_OK_AND_ASSIGN(const DynamicMlirOpResults result,
                          Scan(builder(), input, /*dim=*/0, {carry_init},
                               {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);
  // For an empty scan dimension, the loop is skipped, so the returned
  // value matches the initial output accumulator.
  EXPECT_EQ(result[0].getValue(), output_init.getValue());
}

TEST_P(ScanBuilderTest, StaticShape1D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 1D tensor [1, 2, 3, 4] for input.
  const mlir::MlirOp input =
      MakeDenseConstant(function_builder, {4}, {1, 2, 3, 4});

  // Output accumulator of shape [4], initialized to zeros.
  const mlir::MlirOp output_init = MakeConstant(function_builder, 0, i32, {4});
  // Carry initialized with shape [1] (slice along dim 0).
  const mlir::MlirOp carry_init = MakeConstant(function_builder, 0, i32, {1});

  TT_ASSERT_OK_AND_ASSIGN(const DynamicMlirOpResults result,
                          Scan(function_builder, input, /*dim=*/0, {carry_init},
                               {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 3, 6, 10});
}

// When the compiler (libtpu) is too old to compile the native scan emitter, an
// associative scan must fall back to the StableHLO while loop (b/529376045)
// instead of emitting chlo.ScanOp, and still produce the correct result. Not
// parameterized on IsEmitter(): it drives the emitter path explicitly and
// toggles the global gate. Independent of the test parameter, so running it
// under both instantiations is harmless.
TEST_P(ScanBuilderTest, NativeScanUnsupportedFallsBackToWhileLoop) {
  // Restore the default (supported) even if an assertion below fails, so this
  // process-global gate never leaks into other tests.
  const absl::Cleanup restore = [] { SetNativeScanEmitterSupported(true); };
  SetNativeScanEmitterSupported(false);

  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();
  const mlir::MlirOp input =
      MakeDenseConstant(function_builder, {4}, {1, 2, 3, 4});
  const mlir::MlirOp output_init = MakeConstant(function_builder, 0, i32, {4});
  const mlir::MlirOp carry_init = MakeConstant(function_builder, 0, i32, {1});

  // Request the native scan emitter explicitly; the gate must downgrade it.
  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      BuildScanShlo(function_builder, input, /*dim=*/0, {carry_init},
                    {output_init}, CreateAddBodyBuilder(),
                    ScanOptions{.is_associative = true}));
  ASSERT_EQ(result.size(), 1);

  mlir::Operation* const op = result[0].getValue().getDefiningOp();
  EXPECT_TRUE(mlir::isa<mlir::stablehlo::WhileOp>(op));
  EXPECT_FALSE(mlir::isa<mlir::chlo::ScanOp>(op));

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 3, 6, 10});
}

TEST_P(ScanBuilderTest, StaticShape2D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 2D tensor [[1, 2], [3, 4]] for input. We will scan along dim 0.
  const mlir::RankedTensorType type = mlir::RankedTensorType::get({2, 2}, i32);
  const mlir::DenseElementsAttr input_attr =
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<int32_t>{1, 2, 3, 4});
  const mlir::MlirOp input =
      mlir::stablehlo::Constant(function_builder, input_attr);

  // Output accumulator of shape [2, 2], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {2, 2});
  // Carry initialized with shape [1, 2] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {1, 2});

  TT_ASSERT_OK_AND_ASSIGN(const DynamicMlirOpResults result,
                          Scan(function_builder, input, /*dim=*/0, {carry_init},
                               {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 2, 4, 6});
}

TEST_P(ScanBuilderTest, StaticShape2D_ScanDim1) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 2D tensor [[1, 2], [3, 4]] for input. We will scan along dim 1.
  const mlir::RankedTensorType type = mlir::RankedTensorType::get({2, 2}, i32);
  const mlir::DenseElementsAttr input_attr =
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<int32_t>{1, 2, 3, 4});
  const mlir::MlirOp input =
      mlir::stablehlo::Constant(function_builder, input_attr);

  // Output accumulator of shape [2, 2], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {2, 2});
  // Carry initialized with shape [2, 1] (slice along dim 1).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {2, 1});

  TT_ASSERT_OK_AND_ASSIGN(const DynamicMlirOpResults result,
                          Scan(function_builder, input, /*dim=*/1, {carry_init},
                               {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 3, 3, 7});
}

TEST_P(ScanBuilderTest, StaticShape3D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 3D tensor [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] for input. We scan
  // along dim 2.
  const mlir::RankedTensorType type =
      mlir::RankedTensorType::get({2, 2, 2}, i32);
  const mlir::DenseElementsAttr input_attr = mlir::DenseElementsAttr::get(
      type, llvm::ArrayRef<int32_t>{1, 2, 3, 4, 5, 6, 7, 8});
  const mlir::MlirOp input =
      mlir::stablehlo::Constant(function_builder, input_attr);

  // Output accumulator of shape [2, 2, 2], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {2, 2, 2});
  // Carry initialized with shape [2, 2, 1] (slice along dim 2).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {2, 2, 1});

  TT_ASSERT_OK_AND_ASSIGN(const DynamicMlirOpResults result,
                          Scan(function_builder, input, /*dim=*/2, {carry_init},
                               {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);

  const mlir::MlirOp out(builder(), result[0].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 3, 3, 7, 5, 11, 7, 15});
}

TEST_P(ScanBuilderTest, BodyBuilderError) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5].
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // A body builder that always returns an error to simulate a failure
  // during the construction of the loop body.
  auto error_body_builder = [](mlir::OpBuilder& op_builder, mlir::Location loc,
                               mlir::Value slice, mlir::Value index,
                               mlir::ValueRange carries)
      -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
    return TT_ERROR(error::kInternal) << "Simulated body builder error";
  };

  // Verify that the error from the body builder is correctly propagated
  // up by BuildScanShlo.
  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), input, /*dim=*/0, {carry_init}, {output_init},
           error_body_builder);
  EXPECT_THAT(result, StatusIs(error::kInternal,
                               HasSubstr("Simulated body builder error")));
}

TEST_P(ScanBuilderTest, MismatchedInits) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5].
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0). We provide
  // two carry inits, but only one output init, causing a mismatch.
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), input, /*dim=*/0, {carry_init, carry_init}, {output_init},
           CreateAddBodyBuilder());
  EXPECT_THAT(
      result,
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("expected the number of carry inits (2) and the number of "
                    "output inits (1) to match")));
}

TEST_P(ScanBuilderTest, MismatchedCarries) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5].
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // A body builder that returns two carries instead of the expected one.
  auto wrong_carries_builder = [](mlir::OpBuilder& op_builder,
                                  mlir::Location loc, mlir::Value slice,
                                  mlir::Value index, mlir::ValueRange carries)
      -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
    return llvm::SmallVector<mlir::Value>{carries[0], carries[0]};
  };

  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), input, /*dim=*/0, {carry_init}, {output_init},
           wrong_carries_builder);
  EXPECT_THAT(result, StatusIs(error::kInvalidArgument,
                               HasSubstr("expected 1 new carries, got 2")));
}

TEST_P(ScanBuilderTest, DynamicShapeScanDim) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Input tensor of bounded shape [<=4, 5], where dimension 0 is dynamic.
  mlir::MlirOp input_arg =
      CreateBoundedTensorArg(function_builder, {4, 5}, {0}, i32);
  // Set the runtime size of the dynamic dimension to 2 (less than the bound of
  // 4).
  mlir::MlirOp size = MakeConstant(function_builder, 2, i32, {});
  const mlir::MlirOp input =
      mlir::stablehlo::SetDimensionSize(input_arg, size, /*dim=*/0);

  // Output accumulator of static shape [4, 5] (the static pad boundaries).
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {1, 5});

  TT_ASSERT_OK_AND_ASSIGN(const DynamicMlirOpResults result,
                          Scan(function_builder, input, /*dim=*/0, {carry_init},
                               {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);
  ExpectScanLoweringOp(result[0].getValue().getDefiningOp());
}

TEST_P(ScanBuilderTest, DynamicShapeNonScanDim) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Input tensor of bounded shape [4, <=5], where dimension 1 is dynamic.
  mlir::MlirOp input_arg =
      CreateBoundedTensorArg(function_builder, {4, 5}, {1}, i32);
  // Set the runtime size of the dynamic dimension to 2 (less than the bound of
  // 5).
  mlir::MlirOp size = MakeConstant(function_builder, 2, i32, {});
  const mlir::MlirOp input =
      mlir::stablehlo::SetDimensionSize(input_arg, size, /*dim=*/1);

  // Output accumulator of static shape [4, 5] (the static pad boundaries).
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(function_builder, 0, i32, {1, 5});

  TT_ASSERT_OK_AND_ASSIGN(const DynamicMlirOpResults result,
                          Scan(function_builder, input, /*dim=*/0, {carry_init},
                               {output_init}, CreateAddBodyBuilder()));
  ASSERT_EQ(result.size(), 1);
  ExpectScanLoweringOp(result[0].getValue().getDefiningOp());
}

TEST_P(MultiScanBuilderTest, InvalidDimension) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), {input}, /*dim=*/2, /*num_scan_inputs=*/1, {carry_init},
           {output_init}, CreateIdentityBodyBuilder());
  EXPECT_THAT(result, StatusIs(error::kPythonIndexError,
                               HasSubstr("expected dimension to be in range")));
}

TEST_P(MultiScanBuilderTest, EmptyInputsListError) {
  // Output accumulator of shape [4, 5], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // Pass an empty inputs list, which is invalid.
  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), {}, /*dim=*/0, /*num_scan_inputs=*/0, {carry_init},
           {output_init}, CreateIdentityBodyBuilder());
  EXPECT_THAT(result,
              StatusIs(error::kInvalidArgument,
                       HasSubstr("expected at least 1 scan input, got none")));
}

TEST_P(MultiScanBuilderTest, TooManyScanInputsError) {
  // Input tensors of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input1 =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  const mlir::MlirOp input2 =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});

  // Output accumulator of shape [4, 5], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // We provide 2 inputs, but specify num_scan_inputs = 3, which is invalid.
  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), {input1, input2}, /*dim=*/0, /*num_scan_inputs=*/3,
           {carry_init}, {output_init}, CreateIdentityBodyBuilder());
  EXPECT_THAT(result,
              StatusIs(error::kInvalidArgument,
                       HasSubstr("expected num_scan_inputs (3) to be less "
                                 "than or equal to the number of inputs (2)")));
}

TEST_P(MultiScanBuilderTest, MismatchedCarriesError) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // A body builder that returns an incorrect number of carries (2 instead of
  // 1).
  auto wrong_carries_builder =
      [](mlir::OpBuilder& op_builder, mlir::Location loc,
         mlir::ValueRange slices, mlir::Value index,
         mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    return ScanBodyResults{{carries[0], carries[0]}, {carries[0]}};
  };

  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), {input}, /*dim=*/0, /*num_scan_inputs=*/1, {carry_init},
           {output_init}, wrong_carries_builder);
  EXPECT_THAT(result, StatusIs(error::kInvalidArgument,
                               HasSubstr("expected 1 new carries, got 2")));
}

TEST_P(MultiScanBuilderTest, MismatchedOutputsError) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // A body builder that returns an incorrect number of outputs (2 instead of
  // 1).
  auto wrong_outputs_builder =
      [](mlir::OpBuilder& op_builder, mlir::Location loc,
         mlir::ValueRange slices, mlir::Value index,
         mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    return ScanBodyResults{{carries[0]}, {carries[0], carries[0]}};
  };

  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), {input}, /*dim=*/0, /*num_scan_inputs=*/1, {carry_init},
           {output_init}, wrong_outputs_builder);
  EXPECT_THAT(result, StatusIs(error::kInvalidArgument,
                               HasSubstr("expected 1 new outputs, got 2")));
}

TEST_P(MultiScanBuilderTest, BodyBuilderError) {
  // Input tensor of shape [4, 5]. We scan along dim 0.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Output accumulator of shape [4, 5], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  // A body builder that returns an internal error.
  auto error_body_builder =
      [](mlir::OpBuilder& op_builder, mlir::Location loc,
         mlir::ValueRange slices, mlir::Value index,
         mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    return TT_ERROR(error::kInternal) << "Simulated multi body builder error";
  };

  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), {input}, /*dim=*/0, /*num_scan_inputs=*/1, {carry_init},
           {output_init}, error_body_builder);
  EXPECT_THAT(result,
              StatusIs(error::kInternal,
                       HasSubstr("Simulated multi body builder error")));
}

TEST_P(MultiScanBuilderTest, EmptyTensor) {
  // Input tensor of shape [0, 5], empty along the scan dimension.
  const mlir::MlirOp input =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {0, 5});
  // Output accumulator of shape [0, 5], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {0, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults result,
      Scan(builder(), {input}, /*dim=*/0, /*num_scan_inputs=*/1, {carry_init},
           {output_init}, CreateIdentityBodyBuilder()));
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].getValue(), carry_init.getValue());
  EXPECT_EQ(result[1].getValue(), output_init.getValue());
}

TEST_P(MultiScanBuilderTest, StaticShape1D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 1D tensors [1, 2, 3, 4] and [10, 20, 30, 40] for input.
  const mlir::MlirOp input1 =
      MakeDenseConstant(function_builder, {4}, {1, 2, 3, 4});
  const mlir::MlirOp input2 =
      MakeDenseConstant(function_builder, {4}, {10, 20, 30, 40});

  // Carry initializers with shape [1] (slice along dim 0).
  const mlir::MlirOp carry1_init = MakeConstant(function_builder, 0, i32, {1});
  const mlir::MlirOp carry2_init = MakeConstant(function_builder, 10, i32, {1});
  // Output accumulators of shape [4], initialized to zeros.
  const mlir::MlirOp output1_init = MakeConstant(function_builder, 0, i32, {4});
  const mlir::MlirOp output2_init = MakeConstant(function_builder, 0, i32, {4});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input1, input2}, /*dim=*/0,
           /*num_scan_inputs=*/2, {carry1_init, carry2_init},
           {output1_init, output2_init}, CreateMultiAddBodyBuilder()));
  ASSERT_EQ(results.size(), 4);

  mlir::func::Return(function_builder,
                     {mlir::MlirOp(builder(), results[2].getValue()),
                      mlir::MlirOp(builder(), results[3].getValue())});
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  EvaluateAndVerifyMultiOutputs(*module, {{1, 3, 6, 10}, {20, 40, 70, 110}});
}

TEST_P(MultiScanBuilderTest, StaticShape2D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 2D tensors [[1, 2], [3, 4]] and [[10, 20], [30, 40]] for input. We
  // scan along dim 0.
  const mlir::MlirOp input1 =
      MakeDenseConstant(function_builder, {2, 2}, {1, 2, 3, 4});
  const mlir::MlirOp input2 =
      MakeDenseConstant(function_builder, {2, 2}, {10, 20, 30, 40});

  // Carry initializers with shape [1, 2] (slice along dim 0).
  const mlir::MlirOp carry1_init =
      MakeConstant(function_builder, 0, i32, {1, 2});
  const mlir::MlirOp carry2_init =
      MakeConstant(function_builder, 10, i32, {1, 2});
  // Output accumulators of shape [2, 2], initialized to zeros.
  const mlir::MlirOp output1_init =
      MakeConstant(function_builder, 0, i32, {2, 2});
  const mlir::MlirOp output2_init =
      MakeConstant(function_builder, 0, i32, {2, 2});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input1, input2}, /*dim=*/0,
           /*num_scan_inputs=*/2, {carry1_init, carry2_init},
           {output1_init, output2_init}, CreateMultiAddBodyBuilder()));
  ASSERT_EQ(results.size(), 4);

  mlir::func::Return(function_builder,
                     {mlir::MlirOp(builder(), results[2].getValue()),
                      mlir::MlirOp(builder(), results[3].getValue())});
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  EvaluateAndVerifyMultiOutputs(*module, {{1, 2, 4, 6}, {20, 30, 50, 70}});
}

TEST_P(MultiScanBuilderTest, StaticShape2D_ScanDim1) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 2D tensors [[1, 2], [3, 4]] and [[10, 20], [30, 40]] for input. We
  // scan along dim 1.
  const mlir::MlirOp input1 =
      MakeDenseConstant(function_builder, {2, 2}, {1, 2, 3, 4});
  const mlir::MlirOp input2 =
      MakeDenseConstant(function_builder, {2, 2}, {10, 20, 30, 40});

  // Carry initializers with shape [2, 1] (slice along dim 1).
  const mlir::MlirOp carry1_init =
      MakeConstant(function_builder, 0, i32, {2, 1});
  const mlir::MlirOp carry2_init =
      MakeConstant(function_builder, 10, i32, {2, 1});
  // Output accumulators of shape [2, 2], initialized to zeros.
  const mlir::MlirOp output1_init =
      MakeConstant(function_builder, 0, i32, {2, 2});
  const mlir::MlirOp output2_init =
      MakeConstant(function_builder, 0, i32, {2, 2});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input1, input2}, /*dim=*/1,
           /*num_scan_inputs=*/2, {carry1_init, carry2_init},
           {output1_init, output2_init}, CreateMultiAddBodyBuilder()));
  ASSERT_EQ(results.size(), 4);

  mlir::func::Return(function_builder,
                     {mlir::MlirOp(builder(), results[2].getValue()),
                      mlir::MlirOp(builder(), results[3].getValue())});
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  EvaluateAndVerifyMultiOutputs(*module, {{1, 3, 3, 7}, {20, 40, 40, 80}});
}

TEST_P(MultiScanBuilderTest, StaticShape3D) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 3D tensors of shape [2, 2, 2] for input. We scan along dim 2.
  const mlir::MlirOp input1 =
      MakeDenseConstant(function_builder, {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  const mlir::MlirOp input2 = MakeDenseConstant(
      function_builder, {2, 2, 2}, {10, 20, 30, 40, 50, 60, 70, 80});

  // Carry initializers with shape [2, 2, 1] (slice along dim 2).
  const mlir::MlirOp carry1_init =
      MakeConstant(function_builder, 0, i32, {2, 2, 1});
  const mlir::MlirOp carry2_init =
      MakeConstant(function_builder, 10, i32, {2, 2, 1});
  // Output accumulators of shape [2, 2, 2], initialized to zeros.
  const mlir::MlirOp output1_init =
      MakeConstant(function_builder, 0, i32, {2, 2, 2});
  const mlir::MlirOp output2_init =
      MakeConstant(function_builder, 0, i32, {2, 2, 2});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input1, input2}, /*dim=*/2,
           /*num_scan_inputs=*/2, {carry1_init, carry2_init},
           {output1_init, output2_init}, CreateMultiAddBodyBuilder()));
  ASSERT_EQ(results.size(), 4);

  mlir::func::Return(function_builder,
                     {mlir::MlirOp(builder(), results[2].getValue()),
                      mlir::MlirOp(builder(), results[3].getValue())});
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  EvaluateAndVerifyMultiOutputs(*module, {{1, 3, 3, 7, 5, 11, 7, 15},
                                          {20, 40, 40, 80, 60, 120, 80, 160}});
}

TEST_P(MultiScanBuilderTest, ReverseScan) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 1D tensor [1, 2, 3, 4] for input.
  const mlir::MlirOp input =
      MakeDenseConstant(function_builder, {4}, {1, 2, 3, 4});

  // Carry initialized with shape [1] (slice along dim 0).
  const mlir::MlirOp carry_init = MakeConstant(function_builder, 0, i32, {1});
  // Output accumulator of shape [4].
  const mlir::MlirOp output_init = MakeConstant(function_builder, 0, i32, {4});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input}, /*dim=*/0, /*num_scan_inputs=*/1,
           {carry_init}, {output_init}, CreateMultiAddBodyBuilder(),
           ScanOptions{.direction = ScanDirection::kReverse}));
  ASSERT_EQ(results.size(), 2);

  const mlir::MlirOp out(builder(), results[1].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  if (IsEmitter()) {
    // scan_builder emits chlo.scan with is_reverse set (verified below). An
    // older OSS StableHLO pin may not yet honor is_reverse when it lowers
    // chlo.scan to a while loop, producing the forward result instead; in that
    // case skip the numeric check (reverse numerics are covered by StableHLO's
    // chlo/scan.mlir interpreter test and the TPU tests).
    (*module).walk([](mlir::chlo::ScanOp scan_op) {
      EXPECT_TRUE(scan_op.getIsReverse());
    });
    LegalizeChlo(*module);
    const mlir::stablehlo::InterpreterConfiguration config;
    const llvm::SmallVector<mlir::DenseElementsAttr> empty_inputs;
    const mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>>
        eval_result =
            mlir::stablehlo::evalModule(*module, empty_inputs, config);
    ASSERT_TRUE(mlir::succeeded(eval_result));
    ASSERT_EQ(eval_result->size(), 1);
    const llvm::SmallVector<int32_t> values(
        (*eval_result)[0].getValues<int32_t>());
    if (testing::Matches(testing::ElementsAreArray({1, 3, 6, 10}))(values)) {
      GTEST_SKIP() << "pinned StableHLO ignores chlo.scan is_reverse";
    }
    EXPECT_THAT(values, testing::ElementsAreArray({10, 9, 7, 4}));
    return;
  }
  EvaluateAndVerifyOutputs(*module, {10, 9, 7, 4});
}

TEST_P(MultiScanBuilderTest, SqueezeScan) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 2D tensor [[1, 2], [3, 4]] for input. We scan along dim 0.
  const mlir::MlirOp input =
      MakeDenseConstant(function_builder, {2, 2}, {1, 2, 3, 4});

  // Carry initialized with shape [2] (squeezed slice along dim 0).
  const mlir::MlirOp carry_init = MakeConstant(function_builder, 0, i32, {2});
  // Output accumulator of shape [2, 2].
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {2, 2});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input}, /*dim=*/0, /*num_scan_inputs=*/1,
           {carry_init}, {output_init}, CreateMultiAddBodyBuilder(),
           ScanOptions{.should_squeeze = true}));
  ASSERT_EQ(results.size(), 2);

  const mlir::MlirOp out(builder(), results[1].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutputs(*module, {1, 2, 4, 6});
}

TEST_P(MultiScanBuilderTest, WithStaticInput) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Scannable input tensor of shape [4].
  const mlir::MlirOp input =
      MakeDenseConstant(function_builder, {4}, {1, 2, 3, 4});

  // Loop-invariant static input tensor of shape [1].
  const mlir::MlirOp static_val =
      MakeDenseConstant(function_builder, {1}, {10});

  // Carry initialized with shape [1].
  const mlir::MlirOp carry_init = MakeConstant(function_builder, 0, i32, {1});
  // Output accumulator of shape [4], initialized to zeros.
  const mlir::MlirOp output_init = MakeConstant(function_builder, 0, i32, {4});

  // A body builder that adds the input slice, carry, and loop-invariant static
  // val.
  auto body_builder =
      [](mlir::OpBuilder& op_builder, mlir::Location loc,
         mlir::ValueRange slices, mlir::Value index,
         mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    const mlir::Value x = slices[0];
    const mlir::Value static_v = slices[1];
    const mlir::Value carry = carries[0];

    // x_and_static = x + static_v
    const mlir::Value x_and_static =
        mlir::stablehlo::AddOp::create(op_builder, loc, x, static_v)
            .getResult();
    // new_c = x_and_static + carry
    const mlir::Value new_c =
        mlir::stablehlo::AddOp::create(op_builder, loc, x_and_static, carry)
            .getResult();
    return ScanBodyResults{{new_c}, {new_c}};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input, static_val}, /*dim=*/0,
           /*num_scan_inputs=*/1, {carry_init}, {output_init}, body_builder));
  ASSERT_EQ(results.size(), 2);

  const mlir::MlirOp out(builder(), results[1].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  EvaluateAndVerifyOutputs(*module, {11, 23, 36, 50});
}

TEST_P(MultiScanBuilderTest, MismatchedInitsAllowed) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 2D tensor [[1, 2], [3, 4]] for input. We scan along dim 0.
  const mlir::RankedTensorType input_type =
      mlir::RankedTensorType::get({2, 2}, i32);
  const mlir::DenseElementsAttr input_attr = mlir::DenseElementsAttr::get(
      input_type, llvm::ArrayRef<int32_t>{1, 2, 3, 4});
  const mlir::MlirOp input =
      mlir::stablehlo::Constant(function_builder, input_attr);

  // Carry initialized with shape [1, 2] (slice along dim 0).
  const mlir::RankedTensorType carry_type =
      mlir::RankedTensorType::get({1, 2}, i32);
  const mlir::DenseElementsAttr carry_attr =
      mlir::DenseElementsAttr::get(carry_type, llvm::ArrayRef<int32_t>{10, 20});
  const mlir::MlirOp carry_init =
      mlir::stablehlo::Constant(function_builder, carry_attr);

  // Output accumulator of shape [2, 2], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(function_builder, 0, i32, {2, 2});

  // A body builder that computes new_c1 = x + c1 and new_c2 = x + c2, returning
  // both as carries, and new_c1 as output.
  auto mismatched_inits_body_builder =
      [](mlir::OpBuilder& op_builder, mlir::Location loc,
         mlir::ValueRange slices, mlir::Value index,
         mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    const mlir::Value c1 = carries[0];
    const mlir::Value c2 = carries[1];
    const mlir::Value x = slices[0];
    const mlir::Value new_c1 =
        mlir::stablehlo::AddOp::create(op_builder, loc, x, c1).getResult();
    const mlir::Value new_c2 =
        mlir::stablehlo::AddOp::create(op_builder, loc, x, c2).getResult();
    return ScanBodyResults{{new_c1, new_c2}, {new_c1}};
  };

  TT_ASSERT_OK_AND_ASSIGN(const DynamicMlirOpResults results,
                          Scan(function_builder, {input}, /*dim=*/0,
                               /*num_scan_inputs=*/1, {carry_init, carry_init},
                               {output_init}, mismatched_inits_body_builder));
  ASSERT_EQ(results.size(), 3);

  mlir::func::Return(function_builder,
                     {mlir::MlirOp(builder(), results[0].getValue()),
                      mlir::MlirOp(builder(), results[1].getValue()),
                      mlir::MlirOp(builder(), results[2].getValue())});
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  EvaluateAndVerifyMultiOutputs(*module,
                                {{14, 26}, {14, 26}, {11, 22, 14, 26}});
}

TEST_P(MultiScanBuilderTest, DynamicShapeScanDim) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Input tensors of bounded shape [<=4, 5], where dimension 0 is dynamic.
  mlir::MlirOp input_arg1 =
      CreateBoundedTensorArg(function_builder, {4, 5}, {0}, i32);
  mlir::MlirOp input_arg2 =
      CreateBoundedTensorArg(function_builder, {4, 5}, {0}, i32);
  // Set the runtime size of the dynamic dimension to 2 (less than the bound of
  // 4).
  mlir::MlirOp size = MakeConstant(function_builder, 2, i32, {});
  const mlir::MlirOp input1 =
      mlir::stablehlo::SetDimensionSize(input_arg1, size, /*dim=*/0);
  const mlir::MlirOp input2 =
      mlir::stablehlo::SetDimensionSize(input_arg2, size, /*dim=*/0);

  // Output accumulators of shape [4, 5] (the static pad boundaries).
  const mlir::MlirOp output1_init =
      MakeConstant(function_builder, 0, i32, {4, 5});
  const mlir::MlirOp output2_init =
      MakeConstant(function_builder, 0, i32, {4, 5});
  // Carry initializers with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry1_init =
      MakeConstant(function_builder, 0, i32, {1, 5});
  const mlir::MlirOp carry2_init =
      MakeConstant(function_builder, 0, i32, {1, 5});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input1, input2}, /*dim=*/0,
           /*num_scan_inputs=*/2, {carry1_init, carry2_init},
           {output1_init, output2_init}, CreateIdentityBodyBuilder()));
  ASSERT_EQ(results.size(), 4);
  ExpectScanLoweringOp(results[2].getValue().getDefiningOp());
}

TEST_P(MultiScanBuilderTest, DynamicShapeNonScanDim) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Input tensors of bounded shape [4, <=5], where dimension 1 is dynamic.
  mlir::MlirOp input_arg1 =
      CreateBoundedTensorArg(function_builder, {4, 5}, {1}, i32);
  mlir::MlirOp input_arg2 =
      CreateBoundedTensorArg(function_builder, {4, 5}, {1}, i32);
  // Set the runtime size of the dynamic dimension to 2 (less than the bound of
  // 5).
  mlir::MlirOp size = MakeConstant(function_builder, 2, i32, {});
  const mlir::MlirOp input1 =
      mlir::stablehlo::SetDimensionSize(input_arg1, size, /*dim=*/1);
  const mlir::MlirOp input2 =
      mlir::stablehlo::SetDimensionSize(input_arg2, size, /*dim=*/1);

  // Output accumulators of shape [4, 5] (the static pad boundaries).
  const mlir::MlirOp output1_init =
      MakeConstant(function_builder, 0, i32, {4, 5});
  const mlir::MlirOp output2_init =
      MakeConstant(function_builder, 0, i32, {4, 5});
  // Carry initializers with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry1_init =
      MakeConstant(function_builder, 0, i32, {1, 5});
  const mlir::MlirOp carry2_init =
      MakeConstant(function_builder, 0, i32, {1, 5});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input1, input2}, /*dim=*/0,
           /*num_scan_inputs=*/2, {carry1_init, carry2_init},
           {output1_init, output2_init}, CreateIdentityBodyBuilder()));
  ASSERT_EQ(results.size(), 4);
  ExpectScanLoweringOp(results[2].getValue().getDefiningOp());
}

TEST_P(MultiScanBuilderTest, MismatchedInputScanDimSizeError) {
  // Input 1 of shape [4, 5].
  const mlir::MlirOp input1 =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Input 2 of shape [3, 5] (mismatched size along scan dimension 0).
  const mlir::MlirOp input2 =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {3, 5});

  // Output accumulator of shape [4, 5], initialized to zeros.
  const mlir::MlirOp output_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {4, 5});
  // Carry initialized with shape [1, 5] (slice along dim 0).
  const mlir::MlirOp carry_init =
      MakeConstant(builder(), 0, op_builder().getI32Type(), {1, 5});

  const absl::StatusOr<DynamicMlirOpResults> result =
      Scan(builder(), {input1, input2}, /*dim=*/0,
           /*num_scan_inputs=*/2, {carry_init, carry_init},
           {output_init, output_init}, CreateIdentityBodyBuilder());
  EXPECT_THAT(
      result,
      StatusIs(error::kInvalidArgument,
               HasSubstr("expected all scannable inputs to have matching sizes "
                         "along the scan dimension, but input 0 has size 4 and "
                         "input 1 has size 3")));
}

TEST_P(MultiScanBuilderTest, Squeeze1DTo0DTensor) {
  mlir::func::FunctionBuilder function_builder(module_builder(), "main");
  const mlir::Type i32 = op_builder().getI32Type();

  // Create 1D tensor [1, 2, 3, 4] for input. We scan along dim 0.
  const mlir::MlirOp input =
      MakeDenseConstant(function_builder, {4}, {1, 2, 3, 4});

  // Carry initialized with shape [] (0D tensor).
  const mlir::RankedTensorType carry_type =
      mlir::RankedTensorType::get({}, i32);
  const mlir::DenseElementsAttr carry_attr =
      mlir::DenseElementsAttr::get(carry_type, llvm::ArrayRef<int32_t>{10});
  const mlir::MlirOp carry_init =
      mlir::stablehlo::Constant(function_builder, carry_attr);

  // Output accumulator of shape [4] (since we don't squeeze the accumulator).
  const mlir::MlirOp output_init = MakeConstant(function_builder, 0, i32, {4});

  TT_ASSERT_OK_AND_ASSIGN(
      const DynamicMlirOpResults results,
      Scan(function_builder, {input}, /*dim=*/0, /*num_scan_inputs=*/1,
           {carry_init}, {output_init}, CreateMultiAddBodyBuilder(),
           ScanOptions{.should_squeeze = true}));
  ASSERT_EQ(results.size(), 2);

  const mlir::MlirOp out(builder(), results[1].getValue());
  mlir::func::Return(function_builder, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  EvaluateAndVerifyOutputs(*module, {11, 13, 16, 20});
}

}  // namespace
}  // namespace torch_tpu
