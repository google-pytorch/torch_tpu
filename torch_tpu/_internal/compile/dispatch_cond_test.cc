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

#include "torch_tpu/_internal/compile/dispatch_cond.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::Property;
using testing::Throws;

static constexpr std::string_view kTrueAddMlir = R"mlir(
  module {
    func.func @main(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
      %0 = stablehlo.add %x, %y : tensor<2xi32>
      return %0 : tensor<2xi32>
    }
  }
)mlir";

static constexpr std::string_view kFalseSubMlir = R"mlir(
  module {
    func.func @main(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
      %0 = stablehlo.subtract %x, %y : tensor<2xi32>
      return %0 : tensor<2xi32>
    }
  }
)mlir";

static constexpr std::string_view kTrueMultiResultMlir = R"mlir(
  module {
    func.func @main(%x: tensor<2xi32>, %y: tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
      %0 = stablehlo.add %x, %y : tensor<2xi32>
      %1 = stablehlo.multiply %x, %y : tensor<2xi32>
      return %0, %1 : tensor<2xi32>, tensor<2xi32>
    }
  }
)mlir";

static constexpr std::string_view kFalseMultiResultMlir = R"mlir(
  module {
    func.func @main(%x: tensor<2xi32>, %y: tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
      %0 = stablehlo.subtract %x, %y : tensor<2xi32>
      %1 = stablehlo.maximum %x, %y : tensor<2xi32>
      return %0, %1 : tensor<2xi32>, tensor<2xi32>
    }
  }
)mlir";

static constexpr std::string_view kNoMainMlir = R"mlir(
  module {
    func.func @branch(%x: tensor<2xi32>) -> tensor<2xi32> {
      return %x : tensor<2xi32>
    }
  }
)mlir";

static constexpr std::string_view kMismatchedShapeMlir = R"mlir(
  module {
    func.func @main(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<3xi32> {
      %c = stablehlo.constant dense<0> : tensor<3xi32>
      return %c : tensor<3xi32>
    }
  }
)mlir";

static constexpr std::string_view kMismatchedDtypeMlir = R"mlir(
  module {
    func.func @main(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xf32> {
      %c = stablehlo.constant dense<0.0> : tensor<2xf32>
      return %c : tensor<2xf32>
    }
  }
)mlir";

class DispatchCondTest : public testing::Test {
 protected:
  DispatchCondTest() {
    prev_mode_ = GetEagerMode();
    SetEagerMode(EagerMode::kInternalDeferAll);
  }

  ~DispatchCondTest() override { SetEagerMode(prev_mode_); }

  std::shared_ptr<ContextedModule> CreateBranchModule(
      std::string_view mlir_src) {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::stablehlo::StablehloDialect>();
    context->loadDialect<mlir::func::FuncDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module_op =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_src, context.get());
    EXPECT_TRUE(module_op);
    return std::make_shared<ContextedModule>(std::move(context),
                                             std::move(module_op));
  }

  at::Tensor MakePlaceholder(Dimensions dimensions,
                             mlir::ElementType element_type) {
    const auto placeholder_or =
        DeviceBufferList::CreatePlaceholder(dimensions, element_type);
    ABSL_CHECK(placeholder_or.ok()) << placeholder_or.status();
    return MakeTensor(*placeholder_or);
  }

  at::Tensor MakePlaceholderI32(Dimensions dimensions) {
    return MakePlaceholder(dimensions, mlir::ElementType::I32);
  }

  at::Tensor MakePlaceholderPred(Dimensions dimensions) {
    return MakePlaceholder(dimensions, mlir::ElementType::PRED);
  }

  mlir::RankedTensorType GetTensorType(const at::Tensor& t,
                                       mlir::MLIRContext& context) {
    const auto buffer_or = GetBuffer(t);
    ABSL_CHECK(buffer_or.ok()) << buffer_or.status();
    const DeviceBufferRef buffer = *buffer_or;
    const mlir::Type element_type =
        mlir::getElementType(context, buffer.shape().dtype());
    return mlir::RankedTensorType::get(buffer.shape().dimensions(),
                                       element_type);
  }

  struct LoweringResults {
    absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> results_or;
    std::unique_ptr<mlir::MLIRContext> context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
  };

  LoweringResults LowerToMlir(DeferredOp* deferred_op,
                              absl::Span<const at::Tensor> inputs) {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::stablehlo::StablehloDialect>();
    context->loadDialect<mlir::func::FuncDialect>();
    mlir::ModuleBuilder builder(*context);

    llvm::SmallVector<mlir::MlirOp> mlir_inputs;
    mlir_inputs.reserve(inputs.size());
    for (const at::Tensor& t : inputs) {
      const mlir::RankedTensorType ranked_tensor_type =
          GetTensorType(t, builder.getContext());
      const mlir::DenseElementsAttr attr =
          mlir::makeConstant(0, ranked_tensor_type);
      mlir_inputs.push_back(builder.create<mlir::stablehlo::ConstantOp>(attr));
    }

    auto results_or =
        deferred_op->op_builder()(builder, absl::MakeSpan(mlir_inputs));

    return LoweringResults{
        .results_or = std::move(results_or),
        .context = std::move(context),
        .module = builder.build(),
    };
  }

  void LowerAndVerify(const std::vector<at::Tensor>& results,
                      absl::Span<const at::Tensor> input_tensors,
                      int64_t expected_results_size,
                      std::string* lowered_mlir_out = nullptr) {
    ASSERT_EQ(results.size(), expected_results_size);
    TF_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef output_buf,
                            GetBuffer(results[0]));
    const std::shared_ptr<DeferredOp> deferred_op = output_buf.deferred_op();
    ASSERT_TRUE(deferred_op != nullptr);

    const LoweringResults lowered =
        LowerToMlir(deferred_op.get(), input_tensors);
    ASSERT_TRUE(lowered.results_or.ok());
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));

    if (lowered_mlir_out != nullptr) {
      llvm::raw_string_ostream os(*lowered_mlir_out);
      lowered.module.get().print(os);
    }
  }

 private:
  EagerMode prev_mode_;
};

TEST_F(DispatchCondTest, WithOneResult) {
  const at::Tensor pred = MakePlaceholderPred(Dimensions{});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  const std::vector<at::Tensor> results =
      PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueAddMlir),
                     CreateBranchModule(kFalseSubMlir), {dummy_output});
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef output_buf,
                          GetBuffer(results[0]));
  EXPECT_THAT(output_buf.shape().dimensions(), ElementsAre(2));

  std::string lowered_mlir;
  LowerAndVerify(results, {pred, x, y}, /*expected_results_size=*/1,
                 &lowered_mlir);
  EXPECT_THAT(lowered_mlir, HasSubstr("stablehlo.if"));
  EXPECT_THAT(lowered_mlir, HasSubstr("stablehlo.add"));
  EXPECT_THAT(lowered_mlir, HasSubstr("stablehlo.subtract"));
}

TEST_F(DispatchCondTest, WithVectorPredicate) {
  const at::Tensor pred = MakePlaceholderPred({1});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  const std::vector<at::Tensor> results =
      PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueAddMlir),
                     CreateBranchModule(kFalseSubMlir), {dummy_output});

  std::string lowered_mlir;
  LowerAndVerify(results, {pred, x, y}, /*expected_results_size=*/1,
                 &lowered_mlir);
  EXPECT_THAT(lowered_mlir, HasSubstr("stablehlo.reshape"));
  EXPECT_THAT(lowered_mlir, HasSubstr("stablehlo.if"));
}

TEST_F(DispatchCondTest, WithMultipleResults) {
  const at::Tensor pred = MakePlaceholderPred(Dimensions{});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output0 = MakePlaceholderI32({2});
  const at::Tensor dummy_output1 = MakePlaceholderI32({2});

  const std::vector<at::Tensor> results =
      PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueMultiResultMlir),
                     CreateBranchModule(kFalseMultiResultMlir),
                     {dummy_output0, dummy_output1});
  ASSERT_EQ(results.size(), 2);

  TF_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef output0_buf,
                          GetBuffer(results[0]));
  TF_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef output1_buf,
                          GetBuffer(results[1]));
  EXPECT_THAT(output0_buf.shape().dimensions(), ElementsAre(2));
  EXPECT_THAT(output1_buf.shape().dimensions(), ElementsAre(2));

  std::string lowered_mlir;
  LowerAndVerify(results, {pred, x, y}, /*expected_results_size=*/2,
                 &lowered_mlir);
  EXPECT_THAT(lowered_mlir, HasSubstr("stablehlo.if"));
  EXPECT_THAT(lowered_mlir, HasSubstr("stablehlo.multiply"));
  EXPECT_THAT(lowered_mlir, HasSubstr("stablehlo.maximum"));
}

TEST_F(DispatchCondTest, VerifyCacheKeys) {
  const at::Tensor pred = MakePlaceholderPred(Dimensions{});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  const std::vector<at::Tensor> results =
      PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueAddMlir),
                     CreateBranchModule(kFalseSubMlir), {dummy_output});
  TF_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef output_buf,
                          GetBuffer(results[0]));
  const std::shared_ptr<DeferredOp> deferred_op = output_buf.deferred_op();
  ASSERT_TRUE(deferred_op != nullptr);

  const OpParamCacheKeys& cache_keys = deferred_op->op_param_cache_keys();
  EXPECT_THAT(cache_keys, testing::UnorderedElementsAre(
                              testing::Pair("true_branch_mlir", testing::_),
                              testing::Pair("false_branch_mlir", testing::_),
                              testing::Pair("num_operands", Fingerprint("2")),
                              testing::Pair("num_outputs", Fingerprint("1"))));
}

TEST_F(DispatchCondTest, InvalidPredicateDtypeError) {
  const at::Tensor pred = MakePlaceholderI32(Dimensions{});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  auto create_cond_op = [&] {
    PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueAddMlir),
                   CreateBranchModule(kFalseSubMlir), {dummy_output});
  };
  EXPECT_THAT(create_cond_op,
              Throws<TtError>(Property(
                  &TtError::what,
                  HasSubstr("torch.cond predicate must be a bool tensor"))));
}

TEST_F(DispatchCondTest, InvalidPredicateNumElementsError) {
  const at::Tensor pred = MakePlaceholderPred({2});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  auto create_cond_op = [&] {
    PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueAddMlir),
                   CreateBranchModule(kFalseSubMlir), {dummy_output});
  };
  EXPECT_THAT(create_cond_op,
              Throws<TtError>(Property(
                  &TtError::what,
                  HasSubstr("predicate must contain exactly one element"))));
}

TEST_F(DispatchCondTest, MissingMainError) {
  const at::Tensor pred = MakePlaceholderPred(Dimensions{});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  auto create_cond_op = [&] {
    PyCreateCondOp(pred, {x}, CreateBranchModule(kNoMainMlir),
                   CreateBranchModule(kNoMainMlir), {dummy_output});
  };
  EXPECT_THAT(
      create_cond_op,
      Throws<TtError>(Property(
          &TtError::what,
          HasSubstr("expected 'main' function in true branch MLIR module"))));
}

TEST_F(DispatchCondTest, MismatchedShapeError) {
  const at::Tensor pred = MakePlaceholderPred(Dimensions{});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  auto create_cond_op = [&] {
    PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueAddMlir),
                   CreateBranchModule(kMismatchedShapeMlir), {dummy_output});
  };
  EXPECT_THAT(create_cond_op,
              Throws<TtError>(Property(
                  &TtError::what, HasSubstr("branch output 0 type mismatch"))));
}

TEST_F(DispatchCondTest, MismatchedDtypeError) {
  const at::Tensor pred = MakePlaceholderPred(Dimensions{});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  auto create_cond_op = [&] {
    PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueAddMlir),
                   CreateBranchModule(kMismatchedDtypeMlir), {dummy_output});
  };
  EXPECT_THAT(create_cond_op,
              Throws<TtError>(Property(
                  &TtError::what, HasSubstr("branch output 0 type mismatch"))));
}

TEST_F(DispatchCondTest, MismatchedInputsDuringLowering) {
  const at::Tensor pred = MakePlaceholderPred(Dimensions{});
  const at::Tensor x = MakePlaceholderI32({2});
  const at::Tensor y = MakePlaceholderI32({2});
  const at::Tensor dummy_output = MakePlaceholderI32({2});

  const std::vector<at::Tensor> results =
      PyCreateCondOp(pred, {x, y}, CreateBranchModule(kTrueAddMlir),
                     CreateBranchModule(kFalseSubMlir), {dummy_output});
  TF_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef output_buf,
                          GetBuffer(results[0]));
  const std::shared_ptr<DeferredOp> deferred_op = output_buf.deferred_op();

  const LoweringResults lowered = LowerToMlir(deferred_op.get(), {pred, x});
  EXPECT_THAT(
      lowered.results_or,
      StatusIs(error::kInvalidArgument, HasSubstr("expected 3 inputs, got 2")));
}

}  // namespace
}  // namespace torch_tpu
