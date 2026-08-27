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

#include "torch_tpu/_internal/compile/dispatch_invoke_subgraph.h"

#include <memory>
#include <string_view>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/hlo_module_splitter.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::HasSubstr;
using testing::Property;
using testing::Throws;

static constexpr std::string_view kBodyMlir = R"mlir(
  module {
    func.func @main(%lhs: tensor<4x4xbf16>, %rhs: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
      %0 = stablehlo.add %lhs, %rhs : tensor<4x4xbf16>
      return %0 : tensor<4x4xbf16>
    }
  }
)mlir";

class DispatchInvokeSubgraphTest : public testing::Test {
 protected:
  DispatchInvokeSubgraphTest() {
    previous_mode_ = GetEagerMode();
    SetEagerMode(EagerMode::kInternalCompileFxGraph);
  }

  ~DispatchInvokeSubgraphTest() override { SetEagerMode(previous_mode_); }

  std::shared_ptr<ContextedModule> CreateBodyModule(
      std::string_view mlir_source = kBodyMlir) {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::stablehlo::StablehloDialect>();
    context->loadDialect<mlir::func::FuncDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_source, context.get());
    EXPECT_TRUE(module);
    return std::make_shared<ContextedModule>(std::move(context),
                                             std::move(module));
  }

  at::Tensor MakePlaceholder(Dimensions dimensions) {
    const auto placeholder = DeviceBufferList::CreatePlaceholder(
        dimensions, mlir::ElementType::BF16);
    ABSL_CHECK(placeholder.ok()) << placeholder.status();
    return MakeTensor(*placeholder);
  }

 private:
  EagerMode previous_mode_;
};

TEST_F(DispatchInvokeSubgraphTest, EmitsLateCompiledFunctionCall) {
  const at::Tensor lhs = MakePlaceholder({4, 4});
  const at::Tensor rhs = MakePlaceholder({4, 4});
  const at::Tensor dummy_output = MakePlaceholder({4, 4});
  const std::vector<at::Tensor> results = PyCreateInvokeSubgraphOp(
      {lhs, rhs}, CreateBodyModule(), {dummy_output}, "qwen_layer");

  ASSERT_EQ(results.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef output, GetBuffer(results[0]));
  EXPECT_THAT(output.shape().dimensions(), ElementsAre(4, 4));
  const std::shared_ptr<DeferredOp> deferred_op = output.deferred_op();
  ASSERT_TRUE(deferred_op != nullptr);

  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::stablehlo::StablehloDialect>();
  context->loadDialect<mlir::func::FuncDialect>();
  mlir::ModuleBuilder builder(*context);
  mlir::func::FunctionBuilder function_builder(builder, "main");

  std::vector<mlir::MlirOp> mlir_inputs;
  const mlir::Type input_type =
      mlir::makeTensorType(*context, {4, 4}, mlir::ElementType::BF16);
  for (int i = 0; i < 2; ++i) {
    mlir_inputs.push_back(mlir::func::Argument(function_builder, input_type));
  }

  const auto first_call =
      deferred_op->op_builder()(function_builder, absl::MakeSpan(mlir_inputs));
  ASSERT_TRUE(first_call.ok()) << first_call.status();
  const auto second_call =
      deferred_op->op_builder()(function_builder, absl::MakeSpan(mlir_inputs));
  ASSERT_TRUE(second_call.ok()) << second_call.status();
  mlir::func::Return(function_builder, second_call.value());
  mlir::OwningOpRef<mlir::ModuleOp> module = builder.build();
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

  int private_function_count = 0;
  for (mlir::func::FuncOp function : module->getOps<mlir::func::FuncOp>()) {
    if (function.isPrivate()) ++private_function_count;
  }
  EXPECT_EQ(private_function_count, 1);

  int call_count = 0;
  module->walk([&](mlir::func::CallOp call) {
    ++call_count;
    const auto attrs =
        call->getAttrOfType<mlir::DictionaryAttr>("mhlo.frontend_attributes");
    ASSERT_TRUE(attrs);
    EXPECT_EQ(mlir::cast<mlir::StringAttr>(attrs.get("inlineable")).getValue(),
              "xla_late");
    EXPECT_EQ(
        mlir::cast<mlir::StringAttr>(attrs.get("compilation_unit")).getValue(),
        "qwen_layer");
  });
  EXPECT_EQ(call_count, 2);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                          xla::ConvertStablehloToHlo(*module));
  int splittable_call_count = 0;
  for (xla::HloInstruction* instruction :
       hlo_module->entry_computation()->instructions()) {
    if (instruction->opcode() == xla::HloOpcode::kCall &&
        xla::HloModuleSplitter::ShouldSplitCall(instruction)) {
      ++splittable_call_count;
    }
  }
  EXPECT_EQ(splittable_call_count, 2);

  xla::HloModuleSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(const bool changed, splitter.Run(hlo_module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(splitter.submodules().size(), 1);

  int multi_module_call_count = 0;
  for (xla::HloInstruction* instruction :
       hlo_module->entry_computation()->instructions()) {
    if (instruction->opcode() == xla::HloOpcode::kCustomCall &&
        instruction->custom_call_target() == "_xla_multi_module_call") {
      ++multi_module_call_count;
    }
  }
  EXPECT_EQ(multi_module_call_count, 2);
}

TEST_F(DispatchInvokeSubgraphTest, RejectsMismatchedInputs) {
  const at::Tensor input = MakePlaceholder({4, 4});
  const at::Tensor output = MakePlaceholder({4, 4});
  auto create_op = [&] {
    PyCreateInvokeSubgraphOp({input}, CreateBodyModule(), {output},
                             "qwen_layer");
  };
  EXPECT_THAT(create_op,
              Throws<TtError>(Property(&TtError::what,
                                       HasSubstr("expected 2 inputs, got 1"))));
}

TEST_F(DispatchInvokeSubgraphTest, RejectsMismatchedOutputs) {
  const at::Tensor lhs = MakePlaceholder({4, 4});
  const at::Tensor rhs = MakePlaceholder({4, 4});
  auto create_op = [&] {
    PyCreateInvokeSubgraphOp({lhs, rhs}, CreateBodyModule(), {}, "qwen_layer");
  };
  EXPECT_THAT(create_op,
              Throws<TtError>(Property(
                  &TtError::what, HasSubstr("expected at least 1 output"))));
}

}  // namespace
}  // namespace torch_tpu
