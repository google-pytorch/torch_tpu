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
#include "csrc/ops/sparse_coo/sparse_coo.h"

#include <array>
#include <cstdint>
#include <optional>

#include "csrc/common/dtype.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

using CoalesceOutputs = std::array<mlir::MlirOp, 3>;

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

testing::AssertionResult HasSingleScatterAddElementType(
    mlir::ModuleOp module, mlir::Type expected_elem_type) {
  int num_scatters = 0;
  testing::AssertionResult result = testing::AssertionSuccess();
  module.walk([&](mlir::stablehlo::ScatterOp scatter) {
    ++num_scatters;
    if (scatter.getInputs().size() != 1 || scatter.getUpdates().size() != 1 ||
        scatter.getResults().size() != 1) {
      result = testing::AssertionFailure()
               << "expected single-operand ScatterOp, got inputs="
               << scatter.getInputs().size()
               << ", updates=" << scatter.getUpdates().size()
               << ", results=" << scatter.getResults().size();
      return mlir::WalkResult::interrupt();
    }
    const auto input_type =
        mlir::cast<mlir::RankedTensorType>(scatter.getInputs()[0].getType());
    const auto update_type =
        mlir::cast<mlir::RankedTensorType>(scatter.getUpdates()[0].getType());
    const auto result_type =
        mlir::cast<mlir::RankedTensorType>(scatter.getResults()[0].getType());
    if (input_type.getElementType() != expected_elem_type ||
        update_type.getElementType() != expected_elem_type ||
        result_type.getElementType() != expected_elem_type) {
      result =
          testing::AssertionFailure()
          << "expected ScatterOp element type "
          << mlir::debugString(expected_elem_type)
          << ", got input=" << mlir::debugString(input_type.getElementType())
          << ", update=" << mlir::debugString(update_type.getElementType())
          << ", result=" << mlir::debugString(result_type.getElementType());
      return mlir::WalkResult::interrupt();
    }

    int num_adds = 0;
    scatter.getUpdateComputation().walk([&](mlir::stablehlo::AddOp add_op) {
      ++num_adds;
      const auto add_type =
          mlir::cast<mlir::RankedTensorType>(add_op.getResult().getType());
      if (add_type.getElementType() != expected_elem_type) {
        result = testing::AssertionFailure()
                 << "expected ScatterOp AddOp element type "
                 << mlir::debugString(expected_elem_type) << ", got "
                 << mlir::debugString(add_type.getElementType());
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (!result) {
      return mlir::WalkResult::interrupt();
    }
    if (num_adds != 1) {
      result = testing::AssertionFailure()
               << "expected 1 AddOp in ScatterOp update computation, got "
               << num_adds;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!result) {
    return result;
  }
  if (num_scatters != 1) {
    return testing::AssertionFailure()
           << "expected 1 ScatterOp, got " << num_scatters;
  }
  return testing::AssertionSuccess();
}

TEST(SparseCooTest, BuildSparseToDensePromotesLowPrecisionToF32) {
  for (const mlir::ElementType out_dtype :
       {mlir::ElementType::BF16, mlir::ElementType::F16}) {
    SCOPED_TRACE(ToString(out_dtype));
    TestModuleBuilder test_builder;
    mlir::ModuleBuilder& mb = test_builder.get();
    mlir::func::FunctionBuilder fb(mb, "main");
    mlir::OpBuilder& ob = fb.getOpBuilder();

    TT_ASSERT_OK_AND_ASSIGN(const mlir::Type out_mlir_type,
                            GetMlirType(mb.getContext(), out_dtype));
    mlir::MlirOp indices = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({2, 4}, ob.getI64Type()));
    mlir::MlirOp values = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({4}, out_mlir_type));

    const std::array<int64_t, 2> out_sizes = {2, 3};
    TT_ASSERT_OK_AND_ASSIGN(
        mlir::MlirOp dense,
        BuildSparseToDenseShlo(indices, values, out_sizes, out_dtype));
    mlir::func::Return(fb, dense);

    mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
        << DebugString(module.get());

    const mlir::RankedTensorType dense_type = GetTensorTypeOrDie(dense);
    EXPECT_EQ(dense_type.getElementType(), out_mlir_type);
    EXPECT_EQ(dense_type.getShape(), mlir::ArrayRef<int64_t>({2, 3}));
    EXPECT_TRUE(HasSingleScatterAddElementType(module.get(), ob.getF32Type()));
  }
}

TEST(SparseCooTest, BuildSparseToDenseF32AccumulatesInNativeDtype) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& ob = fb.getOpBuilder();

  mlir::MlirOp indices = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({2, 4}, ob.getI64Type()));
  mlir::MlirOp values = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({4}, ob.getF32Type()));

  const std::array<int64_t, 2> out_sizes = {2, 3};
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp dense,
                          BuildSparseToDenseShlo(indices, values, out_sizes,
                                                 mlir::ElementType::F32));
  mlir::func::Return(fb, dense);

  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  EXPECT_EQ(GetTensorTypeOrDie(dense).getElementType(), ob.getF32Type());
  EXPECT_TRUE(HasSingleScatterAddElementType(module.get(), ob.getF32Type()));
}

TEST(SparseCooTest, BuildSparseToDenseEmptyValuesSkipsScatter) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& ob = fb.getOpBuilder();

  mlir::MlirOp indices = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({2, 0}, ob.getI64Type()));
  mlir::MlirOp values = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({0}, ob.getBF16Type()));

  const std::array<int64_t, 2> out_sizes = {2, 3};
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp dense,
                          BuildSparseToDenseShlo(indices, values, out_sizes,
                                                 mlir::ElementType::BF16));
  mlir::func::Return(fb, dense);

  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::RankedTensorType dense_type = GetTensorTypeOrDie(dense);
  EXPECT_EQ(dense_type.getElementType(), ob.getBF16Type());
  EXPECT_EQ(dense_type.getShape(), mlir::ArrayRef<int64_t>({2, 3}));

  int num_scatters = 0;
  module->walk([&](mlir::stablehlo::ScatterOp) { ++num_scatters; });
  EXPECT_EQ(num_scatters, 0);
}

TEST(SparseCooTest, BuildAddDenseSparsePromotesLowPrecisionToF32) {
  for (const mlir::ElementType out_dtype :
       {mlir::ElementType::BF16, mlir::ElementType::F16}) {
    for (const bool with_alpha : {false, true}) {
      SCOPED_TRACE(testing::Message() << "out_dtype=" << ToString(out_dtype)
                                      << ", with_alpha=" << with_alpha);
      TestModuleBuilder test_builder;
      mlir::ModuleBuilder& mb = test_builder.get();
      mlir::func::FunctionBuilder fb(mb, "main");
      mlir::OpBuilder& ob = fb.getOpBuilder();

      TT_ASSERT_OK_AND_ASSIGN(const mlir::Type out_mlir_type,
                              GetMlirType(mb.getContext(), out_dtype));
      mlir::MlirOp dense_self = mlir::func::Argument(
          fb, mlir::RankedTensorType::get({2, 3}, out_mlir_type));
      mlir::MlirOp indices = mlir::func::Argument(
          fb, mlir::RankedTensorType::get({2, 4}, ob.getI64Type()));
      mlir::MlirOp values = mlir::func::Argument(
          fb, mlir::RankedTensorType::get({4}, out_mlir_type));
      std::optional<mlir::MlirOp> alpha;
      if (with_alpha) {
        alpha = mlir::func::Argument(
            fb, mlir::RankedTensorType::get({}, ob.getF32Type()));
      }

      TT_ASSERT_OK_AND_ASSIGN(
          mlir::MlirOp result,
          BuildAddDenseSparseShlo(dense_self, indices, values, alpha,
                                  out_dtype));
      mlir::func::Return(fb, result);

      mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
      ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
          << DebugString(module.get());

      const mlir::RankedTensorType result_type = GetTensorTypeOrDie(result);
      EXPECT_EQ(result_type.getElementType(), out_mlir_type);
      EXPECT_EQ(result_type.getShape(), mlir::ArrayRef<int64_t>({2, 3}));
      EXPECT_TRUE(
          HasSingleScatterAddElementType(module.get(), ob.getF32Type()));

      int num_muls = 0;
      module->walk([&](mlir::stablehlo::MulOp mul_op) {
        ++num_muls;
        const auto mul_type =
            mlir::cast<mlir::RankedTensorType>(mul_op.getResult().getType());
        EXPECT_EQ(mul_type.getElementType(), ob.getF32Type());
      });
      EXPECT_EQ(num_muls, with_alpha ? 1 : 0);
    }
  }
}

TEST(SparseCooTest, BuildCoalesceAndCountPromotesLowPrecisionToF32) {
  for (const mlir::ElementType elem_dtype :
       {mlir::ElementType::BF16, mlir::ElementType::F16}) {
    SCOPED_TRACE(ToString(elem_dtype));
    TestModuleBuilder test_builder;
    mlir::ModuleBuilder& mb = test_builder.get();
    mlir::func::FunctionBuilder fb(mb, "main");
    mlir::OpBuilder& ob = fb.getOpBuilder();

    TT_ASSERT_OK_AND_ASSIGN(const mlir::Type elem_mlir_type,
                            GetMlirType(mb.getContext(), elem_dtype));
    mlir::MlirOp indices = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({2, 4}, ob.getI64Type()));
    mlir::MlirOp values = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({4}, elem_mlir_type));

    TT_ASSERT_OK_AND_ASSIGN(CoalesceOutputs outputs,
                            BuildCoalesceAndCountShlo(indices, values));
    mlir::func::Return(fb, {outputs[0], outputs[1], outputs[2]});

    mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
        << DebugString(module.get());

    const mlir::RankedTensorType compacted_indices_type =
        GetTensorTypeOrDie(outputs[0]);
    const mlir::RankedTensorType compacted_values_type =
        GetTensorTypeOrDie(outputs[1]);
    const mlir::RankedTensorType count_type = GetTensorTypeOrDie(outputs[2]);
    EXPECT_EQ(compacted_indices_type.getElementType(), ob.getI64Type());
    EXPECT_EQ(compacted_values_type.getElementType(), elem_mlir_type);
    EXPECT_EQ(compacted_values_type.getShape(), mlir::ArrayRef<int64_t>({4}));
    EXPECT_EQ(count_type.getElementType(), ob.getI64Type());
    EXPECT_EQ(count_type.getRank(), 0);
    EXPECT_TRUE(HasSingleScatterAddElementType(module.get(), ob.getF32Type()));
  }
}

TEST(SparseCooTest, BuildCoalesceAndCountF32AccumulatesInNativeDtype) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& ob = fb.getOpBuilder();

  mlir::MlirOp indices = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({2, 4}, ob.getI64Type()));
  mlir::MlirOp values = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({4}, ob.getF32Type()));

  TT_ASSERT_OK_AND_ASSIGN(CoalesceOutputs outputs,
                          BuildCoalesceAndCountShlo(indices, values));
  mlir::func::Return(fb, {outputs[0], outputs[1], outputs[2]});

  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  EXPECT_EQ(GetTensorTypeOrDie(outputs[1]).getElementType(), ob.getF32Type());
  EXPECT_TRUE(HasSingleScatterAddElementType(module.get(), ob.getF32Type()));
}

}  // namespace
}  // namespace torch_tpu
