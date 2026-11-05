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

#include "csrc/internal/compile/dispatch_invoke_subgraph.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_assignment.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Property;
using ::testing::Throws;

constexpr std::string_view kAddSubgraphMlir = R"mlir(
  module {
    func.func @main(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
      %0 = stablehlo.add %x, %y : tensor<2x3xf32>
      return %0 : tensor<2x3xf32>
    }
  }
)mlir";

constexpr std::string_view kAddSubgraphWithLocationsMlir = R"mlir(
  module {
    func.func @main(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
      %0 = stablehlo.add %x, %y : tensor<2x3xf32> loc("subgraph_add.py":42:5)
      return %0 : tensor<2x3xf32>
    }
  }
)mlir";

constexpr std::string_view kBoundedDynamicSubgraphMlir = R"mlir(
  module {
    func.func @main(%x: tensor<?xi32, #stablehlo.type_extensions<bounds = [4]>>) -> tensor<i32> {
      %0 = stablehlo.get_dimension_size %x, dim = 0
          : (tensor<?xi32, #stablehlo.type_extensions<bounds = [4]>>) -> tensor<i32>
      return %0 : tensor<i32>
    }
  }
)mlir";

constexpr std::string_view kPartialBoundedDynamicSubgraphMlir = R"mlir(
  module {
    func.func @main(%x: tensor<?x3xi32, #stablehlo.type_extensions<bounds = [4, ?]>>) -> tensor<i32> {
      %0 = stablehlo.get_dimension_size %x, dim = 0
          : (tensor<?x3xi32, #stablehlo.type_extensions<bounds = [4, ?]>>) -> tensor<i32>
      return %0 : tensor<i32>
    }
  }
)mlir";

constexpr std::string_view kMultiAxisDynamicMlir = R"mlir(
  module {
    func.func @main(%x: tensor<?x?xi32, #stablehlo.type_extensions<bounds = [4, 5]>>) -> tensor<i32> {
      %0 = stablehlo.get_dimension_size %x, dim = 0
          : (tensor<?x?xi32, #stablehlo.type_extensions<bounds = [4, 5]>>) -> tensor<i32>
      %1 = stablehlo.get_dimension_size %x, dim = 1
          : (tensor<?x?xi32, #stablehlo.type_extensions<bounds = [4, 5]>>) -> tensor<i32>
      %2 = stablehlo.add %0, %1 : tensor<i32>
      return %2 : tensor<i32>
    }
  }
)mlir";

int CountOccurrences(std::string_view haystack, std::string_view needle) {
  int count = 0;
  size_t pos = 0;
  while ((pos = haystack.find(needle, pos)) != std::string_view::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

std::string FindSymbolWithPrefix(mlir::ModuleOp module,
                                 std::string_view prefix) {
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (func.getName().starts_with(prefix)) {
      return func.getName().str();
    }
  }
  return "";
}

class DispatchInvokeSubgraphTest : public testing::Test {
 protected:
  DispatchInvokeSubgraphTest() {
    prev_mode_ = GetEagerMode();
    SetEagerMode(EagerMode::kInternalCompileFxGraph);
  }

  ~DispatchInvokeSubgraphTest() override { SetEagerMode(prev_mode_); }

  void LoadDialects(mlir::MLIRContext& context) {
    context.loadDialect<mlir::stablehlo::StablehloDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
  }

  std::shared_ptr<ContextedModule> CreateSubgraphModule(
      std::string_view mlir_src) {
    auto context = std::make_unique<mlir::MLIRContext>();
    LoadDialects(*context);
    mlir::OwningOpRef<mlir::ModuleOp> module_op =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_src, context.get());
    ABSL_CHECK(module_op);
    return std::make_shared<ContextedModule>(std::move(context),
                                             std::move(module_op));
  }

  at::Tensor MakePlaceholder(Dimensions dimensions, mlir::ElementType dtype) {
    absl::StatusOr<DeviceBufferRef> placeholder =
        DeviceBufferList::CreatePlaceholder(std::move(dimensions), dtype);
    ABSL_CHECK_OK(placeholder);
    return MakeTensor(std::move(*placeholder));
  }

  at::Tensor MakeBoundedDynamicPlaceholder(
      Dimensions dimensions, mlir::ElementType dtype,
      absl::InlinedVector<BoundedDynamicDimension, 1> dynamic_dims) {
    absl::StatusOr<DeviceBufferRef> buffer =
        DeviceBufferList::CreatePendingExecution(
            Shape(std::move(dimensions), dtype, std::move(dynamic_dims)),
            /*device_index=*/0, /*stream_id=*/0);
    ABSL_CHECK_OK(buffer);
    return MakeTensor(std::move(*buffer));
  }

  mlir::MlirOp BuildMockInput(mlir::MlirBuilder& builder, const at::Tensor& t) {
    absl::StatusOr<DeviceBufferRef> buffer = GetBuffer(t);
    ABSL_CHECK_OK(buffer);
    const Shape& shape = buffer->shape();
    mlir::MLIRContext& context = builder.getContext();
    const mlir::Type element_type =
        mlir::getElementType(context, shape.dtype());

    Dimensions padded_dims = shape.dimensions();
    for (const BoundedDynamicDimension& dyn_dim : shape.dynamic_dimensions()) {
      padded_dims[dyn_dim.dimension] = dyn_dim.upper_bound;
    }
    const mlir::RankedTensorType static_type =
        mlir::RankedTensorType::get(padded_dims, element_type);
    const mlir::DenseElementsAttr attr = mlir::makeConstant(0, static_type);
    mlir::MlirOp op = builder.create<mlir::stablehlo::ConstantOp>(attr);

    const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
    for (const BoundedDynamicDimension& dyn_dim : shape.dynamic_dimensions()) {
      mlir::MlirOp size_const = MakeScalarConstant(
          builder, shape.dimensions()[dyn_dim.dimension], i32_type);
      auto barrier_ops =
          mlir::stablehlo::OptimizationBarrier(builder, {size_const});
      op = mlir::stablehlo::SetDimensionSize(op, barrier_ops[0],
                                             dyn_dim.dimension);
    }
    return op;
  }

  struct LoweringResults {
    absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> results;
    std::unique_ptr<mlir::MLIRContext> context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
  };

  LoweringResults LowerOpsToSingleModule(
      absl::Span<const at::Tensor> first_inputs, DeferredOp* first_op,
      absl::Span<const at::Tensor> second_inputs = {},
      DeferredOp* second_op = nullptr,
      std::string_view pre_existing_helper_src = "") {
    auto context = std::make_unique<mlir::MLIRContext>();
    LoadDialects(*context);
    mlir::ModuleBuilder builder(*context);

    if (!pre_existing_helper_src.empty()) {
      mlir::OwningOpRef<mlir::ModuleOp> helper_mod =
          mlir::parseSourceString<mlir::ModuleOp>(pre_existing_helper_src,
                                                  &builder.getContext());
      ABSL_CHECK(helper_mod);
      mlir::ModuleOp dst_module = GetModuleOp(builder);
      for (mlir::SymbolOpInterface sym :
           helper_mod->getOps<mlir::SymbolOpInterface>()) {
        dst_module.push_back(sym->clone());
      }
    }

    auto build_mock_inputs = [&](absl::Span<const at::Tensor> tensors) {
      llvm::SmallVector<mlir::MlirOp> mlir_inputs;
      mlir_inputs.reserve(tensors.size());
      for (const at::Tensor& t : tensors) {
        mlir_inputs.push_back(BuildMockInput(builder, t));
      }
      return mlir_inputs;
    };

    llvm::SmallVector<mlir::MlirOp> first_mlir_inputs =
        build_mock_inputs(first_inputs);
    absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> results =
        first_op->op_builder()(builder, absl::MakeSpan(first_mlir_inputs));
    if (results.ok() && second_op != nullptr) {
      llvm::SmallVector<mlir::MlirOp> second_mlir_inputs =
          build_mock_inputs(second_inputs);
      results =
          second_op->op_builder()(builder, absl::MakeSpan(second_mlir_inputs));
    }

    return LoweringResults{
        .results = std::move(results),
        .context = std::move(context),
        .module = builder.build(),
    };
  }

  absl::StatusOr<OpParamCacheKeys> GetOpParamCacheKeys(const at::Tensor& t) {
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef buf, GetBuffer(t));
    TT_RET_CHECK(buf.deferred_op() != nullptr, error::kInternal)
        << "expected deferred op on tensor";
    return buf.deferred_op()->op_param_cache_keys().Clone();
  }

 private:
  EagerMode prev_mode_;
};

TEST_F(DispatchInvokeSubgraphTest, SimpleSubgraphAndLocationPreservation) {
  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kAddSubgraphWithLocationsMlir);

  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x, y}, subgraph);
  ASSERT_EQ(results.size(), 1);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf, GetBuffer(results[0]));
  EXPECT_THAT(out_buf.shape().dimensions(), ElementsAre(2, 3));
  EXPECT_EQ(out_buf.shape().dtype(), mlir::ElementType::F32);

  const LoweringResults lowered =
      LowerOpsToSingleModule({x, y}, out_buf.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));

  const std::string mlir_with_locs =
      DebugString(*lowered.module, DebugStringOptions::kEnableDebugInfo);
  EXPECT_THAT(mlir_with_locs, HasSubstr("func.call @invoke_subgraph_"));
  EXPECT_THAT(mlir_with_locs, HasSubstr("func.func private @invoke_subgraph_"));
  EXPECT_THAT(mlir_with_locs, HasSubstr("subgraph_add.py"));
}

TEST_F(DispatchInvokeSubgraphTest, SanitizesIdentifiers) {
  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kAddSubgraphMlir);

  const std::vector<at::Tensor> r1 =
      PyCreateInvokeSubgraphOp({x, y}, subgraph, "layer-0/attn");
  ASSERT_EQ(r1.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b1, GetBuffer(r1[0]));
  const LoweringResults l1 =
      LowerOpsToSingleModule({x, y}, b1.deferred_op().get());
  ASSERT_THAT(l1.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*l1.module)));
  EXPECT_THAT(DebugString(*l1.module), HasSubstr("func.call @layer_0_attn_"));

  const std::vector<at::Tensor> r2 =
      PyCreateInvokeSubgraphOp({x, y}, subgraph, "0_block");
  ASSERT_EQ(r2.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b2, GetBuffer(r2[0]));
  const LoweringResults l2 =
      LowerOpsToSingleModule({x, y}, b2.deferred_op().get());
  ASSERT_THAT(l2.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*l2.module)));
  EXPECT_THAT(DebugString(*l2.module), HasSubstr("func.call @_0_block_"));
}

TEST_F(DispatchInvokeSubgraphTest, SupportsMultipleOutputs) {
  constexpr std::string_view kMultiOutputSubgraphMlir = R"mlir(
    module {
      func.func @main(%x: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2xi32>) {
        %0 = stablehlo.add %x, %x : tensor<2x3xf32>
        %c = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
        return %0, %c : tensor<2x3xf32>, tensor<2xi32>
      }
    }
  )mlir";
  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> multi =
      CreateSubgraphModule(kMultiOutputSubgraphMlir);
  const std::vector<at::Tensor> results = PyCreateInvokeSubgraphOp({x}, multi);
  ASSERT_EQ(results.size(), 2);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b0, GetBuffer(results[0]));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b1, GetBuffer(results[1]));
  EXPECT_THAT(b0.shape().dimensions(), ElementsAre(2, 3));
  EXPECT_EQ(b0.shape().dtype(), mlir::ElementType::F32);
  EXPECT_THAT(b1.shape().dimensions(), ElementsAre(2));
  EXPECT_EQ(b1.shape().dtype(), mlir::ElementType::I32);

  const LoweringResults lowered =
      LowerOpsToSingleModule({x}, b0.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));
}

TEST_F(DispatchInvokeSubgraphTest, SupportsZeroInputs) {
  constexpr std::string_view kZeroInputSubgraphMlir = R"mlir(
    module {
      func.func @main() -> tensor<2xi32> {
        %c = stablehlo.constant dense<[3, 4]> : tensor<2xi32>
        return %c : tensor<2xi32>
      }
    }
  )mlir";
  const std::shared_ptr<ContextedModule> zero_in =
      CreateSubgraphModule(kZeroInputSubgraphMlir);
  const std::vector<at::Tensor> results = PyCreateInvokeSubgraphOp({}, zero_in);
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b, GetBuffer(results[0]));
  const LoweringResults lowered =
      LowerOpsToSingleModule({}, b.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));
}

TEST_F(DispatchInvokeSubgraphTest, ZeroOutputsReturnsEmpty) {
  constexpr std::string_view kZeroOutputSubgraphMlir = R"mlir(
    module {
      func.func @main(%x: tensor<2xi32>) -> () {
        return
      }
    }
  )mlir";
  const at::Tensor x = MakePlaceholder({2}, mlir::ElementType::I32);
  const std::shared_ptr<ContextedModule> zero_out =
      CreateSubgraphModule(kZeroOutputSubgraphMlir);
  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x}, zero_out);
  EXPECT_THAT(results, IsEmpty());

  const at::Tensor bad_x = MakePlaceholder({3}, mlir::ElementType::I32);
  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({bad_x}, zero_out); },
              Throws<TtError>(Property(&TtError::what,
                                       HasSubstr("shape to be [2], got [3]"))));
}

TEST_F(DispatchInvokeSubgraphTest, SupportsZeroDimScalarTensor) {
  constexpr std::string_view kScalarSubgraphMlir = R"mlir(
    module {
      func.func @main(%x: tensor<f32>) -> tensor<f32> {
        %0 = stablehlo.negate %x : tensor<f32>
        return %0 : tensor<f32>
      }
    }
  )mlir";
  const at::Tensor scalar = MakePlaceholder({}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> scalar_sub =
      CreateSubgraphModule(kScalarSubgraphMlir);
  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({scalar}, scalar_sub);
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b, GetBuffer(results[0]));
  EXPECT_THAT(b.shape().dimensions(), IsEmpty());
  const LoweringResults lowered =
      LowerOpsToSingleModule({scalar}, b.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));
}

TEST_F(DispatchInvokeSubgraphTest, SupportsZeroElementTensor) {
  constexpr std::string_view kEmptySubgraphMlir = R"mlir(
    module {
      func.func @main(%x: tensor<0x3xf32>) -> tensor<0x3xf32> {
        %0 = stablehlo.negate %x : tensor<0x3xf32>
        return %0 : tensor<0x3xf32>
      }
    }
  )mlir";
  const at::Tensor empty = MakePlaceholder({0, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> empty_sub =
      CreateSubgraphModule(kEmptySubgraphMlir);
  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({empty}, empty_sub);
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b, GetBuffer(results[0]));
  EXPECT_THAT(b.shape().dimensions(), ElementsAre(0, 3));
  const LoweringResults lowered =
      LowerOpsToSingleModule({empty}, b.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));
}

TEST_F(DispatchInvokeSubgraphTest,
       HelperCalleesImportedAndCollidingHelpersRenamed) {
  constexpr std::string_view kNestedHelperSubgraphMlir = R"mlir(
    module {
      func.func private @leaf_helper(%a: tensor<2xi32>, %b: tensor<2xi32>) -> tensor<2xi32> {
        %0 = stablehlo.multiply %a, %b : tensor<2xi32>
        return %0 : tensor<2xi32>
      }
      func.func private @mid_helper(%a: tensor<2xi32>, %b: tensor<2xi32>) -> tensor<2xi32> {
        %0 = func.call @leaf_helper(%a, %b) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
        return %0 : tensor<2xi32>
      }
      func.func @main(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
        %0 = func.call @mid_helper(%x, %y) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
        return %0 : tensor<2xi32>
      }
    }
  )mlir";
  // Pre-seed dst_module with a conflicting private `@leaf_helper` so symbol
  // deduplication must rename the imported helper cleanly and rewrite callers.
  constexpr std::string_view kPreExistingHelper = R"mlir(
    module {
      func.func private @leaf_helper(%a: tensor<2xi32>, %b: tensor<2xi32>) -> tensor<2xi32> {
        %0 = stablehlo.add %a, %b : tensor<2xi32>
        return %0 : tensor<2xi32>
      }
    }
  )mlir";

  const at::Tensor x = MakePlaceholder({2}, mlir::ElementType::I32);
  const at::Tensor y = MakePlaceholder({2}, mlir::ElementType::I32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kNestedHelperSubgraphMlir);

  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x, y}, subgraph);
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf, GetBuffer(results[0]));

  const LoweringResults lowered = LowerOpsToSingleModule(
      {x, y}, out_buf.deferred_op().get(), /*second_inputs=*/{},
      /*second_op=*/nullptr, kPreExistingHelper);
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));

  const std::string printed = DebugString(*lowered.module);
  EXPECT_THAT(printed, HasSubstr("func.func private @mid_helper"));
  EXPECT_THAT(printed, HasSubstr("func.func private @leaf_helper_0"));
  EXPECT_THAT(printed, HasSubstr("call @leaf_helper_0("));
  EXPECT_THAT(printed, HasSubstr("stablehlo.multiply"));
  EXPECT_THAT(printed, HasSubstr("stablehlo.add"));
}

TEST_F(DispatchInvokeSubgraphTest, CollidingEntrySymbolUsesFallbackRename) {
  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kAddSubgraphMlir);

  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x, y}, subgraph, "block");
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf, GetBuffer(results[0]));

  // First lower into an empty module to discover the deterministic symbol name.
  LoweringResults probe =
      LowerOpsToSingleModule({x, y}, out_buf.deferred_op().get());
  ASSERT_THAT(probe.results, IsOk());
  const std::string entry_sym = FindSymbolWithPrefix(*probe.module, "block_");
  ASSERT_FALSE(entry_sym.empty());

  // 1. Pre-seed a colliding symbol with incompatible result types and check
  // that lowering falls back to a uniquely renamed symbol instead of reusing
  // it.
  const std::string pre_existing_result_mismatch = absl::StrCat(
      R"mlir(
    module {
      func.func private @)mlir",
      entry_sym,
      R"mlir((%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
        %0 = stablehlo.add %a, %b : tensor<2x3xf32>
        return %0, %0 : tensor<2x3xf32>, tensor<2x3xf32>
      }
    }
  )mlir");

  const LoweringResults lowered_res_mismatch = LowerOpsToSingleModule(
      {x, y}, out_buf.deferred_op().get(), /*second_inputs=*/{},
      /*second_op=*/nullptr, pre_existing_result_mismatch);
  ASSERT_THAT(lowered_res_mismatch.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_res_mismatch.module)));

  const std::string printed_res_mismatch =
      DebugString(*lowered_res_mismatch.module);
  EXPECT_THAT(printed_res_mismatch,
              HasSubstr(absl::StrCat("func.func private @", entry_sym, "_0(")));
  EXPECT_THAT(printed_res_mismatch,
              HasSubstr(absl::StrCat("func.call @", entry_sym, "_0(")));

  // 2. Pre-seed a colliding symbol whose result type matches @main, but whose
  // argument types are incompatible (`tensor<4x3xf32>`), and verify fallback
  // renaming.
  const std::string pre_existing_arg_mismatch = absl::StrCat(
      R"mlir(
    module {
      func.func private @)mlir",
      entry_sym,
      R"mlir((%a: tensor<4x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
        return %b : tensor<2x3xf32>
      }
    }
  )mlir");

  const LoweringResults lowered_arg_mismatch = LowerOpsToSingleModule(
      {x, y}, out_buf.deferred_op().get(), /*second_inputs=*/{},
      /*second_op=*/nullptr, pre_existing_arg_mismatch);
  ASSERT_THAT(lowered_arg_mismatch.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_arg_mismatch.module)));

  const std::string printed_arg_mismatch =
      DebugString(*lowered_arg_mismatch.module);
  EXPECT_THAT(printed_arg_mismatch,
              HasSubstr(absl::StrCat("func.func private @", entry_sym, "_0(")));
  EXPECT_THAT(printed_arg_mismatch,
              HasSubstr(absl::StrCat("func.call @", entry_sym, "_0(")));
}

TEST_F(DispatchInvokeSubgraphTest, PreservesNonFuncSymbolsAndDeduplicates) {
  constexpr std::string_view kNonFuncSymSubgraphMlir = R"mlir(
    module {
      module @meta attributes {test.tag = "imported"} {}
      func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
        %0 = stablehlo.negate %arg0 {test.ref = @meta} : tensor<8xf32>
        return %0 : tensor<8xf32>
      }
    }
  )mlir";
  constexpr std::string_view kPreExistingNonFuncSym = R"mlir(
    module {
      module @meta attributes {test.tag = "existing"} {}
    }
  )mlir";

  const at::Tensor x = MakePlaceholder({8}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kNonFuncSymSubgraphMlir);
  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x}, subgraph);
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf, GetBuffer(results[0]));

  const LoweringResults lowered = LowerOpsToSingleModule(
      {x}, out_buf.deferred_op().get(), /*second_inputs=*/{},
      /*second_op=*/nullptr, kPreExistingNonFuncSym);
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));

  const std::string printed = DebugString(*lowered.module);
  EXPECT_THAT(printed,
              HasSubstr("module @meta attributes {test.tag = \"existing\"}"));
  EXPECT_THAT(printed,
              HasSubstr("module @meta_0 attributes {test.tag = \"imported\"}"));
  EXPECT_THAT(printed, HasSubstr("test.ref = @meta_0"));
}

TEST_F(DispatchInvokeSubgraphTest, DeduplicatesRepeatedCallsAndSharesCacheKey) {
  constexpr std::string_view kSubtractSubgraphMlir = R"mlir(
    module {
      func.func @main(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %0 = stablehlo.subtract %x, %y : tensor<2x3xf32>
        return %0 : tensor<2x3xf32>
      }
    }
  )mlir";

  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> sub_a =
      CreateSubgraphModule(kAddSubgraphMlir);
  const std::shared_ptr<ContextedModule> sub_with_loc =
      CreateSubgraphModule(kAddSubgraphWithLocationsMlir);
  const std::shared_ptr<ContextedModule> sub_diff =
      CreateSubgraphModule(kSubtractSubgraphMlir);

  const std::vector<at::Tensor> r1 =
      PyCreateInvokeSubgraphOp({x, y}, sub_a, "block");
  const std::vector<at::Tensor> r2 =
      PyCreateInvokeSubgraphOp({x, y}, sub_with_loc, "block");
  const std::vector<at::Tensor> r_diff_id =
      PyCreateInvokeSubgraphOp({x, y}, sub_a, "other_block");
  const std::vector<at::Tensor> r_diff_mlir =
      PyCreateInvokeSubgraphOp({x, y}, sub_diff, "block");
  ASSERT_EQ(r1.size(), 1);
  ASSERT_EQ(r2.size(), 1);
  ASSERT_EQ(r_diff_id.size(), 1);
  ASSERT_EQ(r_diff_mlir.size(), 1);

  // Debug locations and human-readable identifiers do not alter the cache key.
  TT_ASSERT_OK_AND_ASSIGN(const OpParamCacheKeys k1,
                          GetOpParamCacheKeys(r1[0]));
  TT_ASSERT_OK_AND_ASSIGN(const OpParamCacheKeys k2,
                          GetOpParamCacheKeys(r2[0]));
  TT_ASSERT_OK_AND_ASSIGN(const OpParamCacheKeys k_diff_id,
                          GetOpParamCacheKeys(r_diff_id[0]));
  TT_ASSERT_OK_AND_ASSIGN(const OpParamCacheKeys k_diff_mlir,
                          GetOpParamCacheKeys(r_diff_mlir[0]));
  EXPECT_THAT(k2, ElementsAreArray(k1.begin(), k1.end()));
  EXPECT_THAT(k_diff_id, ElementsAreArray(k1.begin(), k1.end()));
  EXPECT_THAT(k_diff_mlir, Not(ElementsAreArray(k1.begin(), k1.end())));

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b1, GetBuffer(r1[0]));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b2, GetBuffer(r2[0]));
  const LoweringResults lowered = LowerOpsToSingleModule(
      {x, y}, b1.deferred_op().get(), {x, y}, b2.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));

  const std::string printed = DebugString(*lowered.module);
  EXPECT_EQ(CountOccurrences(printed, "func.call @block_"), 2);
  EXPECT_EQ(CountOccurrences(printed, "func.func private @block_"), 1);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b_diff_id,
                          GetBuffer(r_diff_id[0]));
  const LoweringResults lowered_diff_id = LowerOpsToSingleModule(
      {x, y}, b1.deferred_op().get(), {x, y}, b_diff_id.deferred_op().get());
  ASSERT_THAT(lowered_diff_id.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_diff_id.module)));
  const std::string diff_id_printed = DebugString(*lowered_diff_id.module);
  EXPECT_THAT(diff_id_printed, HasSubstr("func.call @block_"));
  EXPECT_THAT(diff_id_printed, HasSubstr("func.call @other_block_"));
}

TEST_F(DispatchInvokeSubgraphTest,
       DistinctDeviceAssignmentsProduceDistinctFunctions) {
  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> sub_a =
      CreateSubgraphModule(kAddSubgraphMlir);
  const std::shared_ptr<ContextedModule> sub_b =
      CreateSubgraphModule(kAddSubgraphMlir);
  sub_a->set_device_assignment(DeviceAssignment(/*local=*/{0, 1}));
  sub_b->set_device_assignment(DeviceAssignment(/*local=*/{1, 0}));

  const std::vector<at::Tensor> r1 =
      PyCreateInvokeSubgraphOp({x, y}, sub_a, "tp");
  const std::vector<at::Tensor> r2 =
      PyCreateInvokeSubgraphOp({x, y}, sub_b, "tp");
  ASSERT_EQ(r1.size(), 1);
  ASSERT_EQ(r2.size(), 1);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b1, GetBuffer(r1[0]));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b2, GetBuffer(r2[0]));
  const LoweringResults lowered = LowerOpsToSingleModule(
      {x, y}, b1.deferred_op().get(), {x, y}, b2.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));

  const std::string printed = DebugString(*lowered.module);
  EXPECT_EQ(CountOccurrences(printed, "func.func private @tp_"), 2);
  EXPECT_EQ(CountOccurrences(printed, "func.call @tp_"), 2);
}

TEST_F(DispatchInvokeSubgraphTest,
       BoundedDynamicAdaptsStaticAndReusesExactDynamicInput) {
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kBoundedDynamicSubgraphMlir);

  // Static input shorter than upper bound is padded and bounded to match @main.
  const at::Tensor x_static = MakePlaceholder({2}, mlir::ElementType::I32);
  const std::vector<at::Tensor> res_static =
      PyCreateInvokeSubgraphOp({x_static}, subgraph);
  ASSERT_EQ(res_static.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf,
                          GetBuffer(res_static[0]));
  EXPECT_THAT(out_buf.shape().dimensions(), IsEmpty());
  EXPECT_EQ(out_buf.shape().dtype(), mlir::ElementType::I32);

  // Bounded-dynamic input matching @main's upper bound reuses the same callee.
  const at::Tensor x_dyn_exact = MakeBoundedDynamicPlaceholder(
      {2}, mlir::ElementType::I32,
      {{.dimension = 0, .lower_bound = 2, .upper_bound = 4}});
  const std::vector<at::Tensor> res_dyn_exact =
      PyCreateInvokeSubgraphOp({x_dyn_exact}, subgraph);
  ASSERT_EQ(res_dyn_exact.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef dyn_exact_buf,
                          GetBuffer(res_dyn_exact[0]));

  const LoweringResults lowered_reuse =
      LowerOpsToSingleModule({x_static}, out_buf.deferred_op().get(),
                             {x_dyn_exact}, dyn_exact_buf.deferred_op().get());
  ASSERT_THAT(lowered_reuse.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_reuse.module)));
  const std::string printed_reuse = DebugString(*lowered_reuse.module);
  EXPECT_EQ(CountOccurrences(printed_reuse, "func.func private @"), 1);
  EXPECT_EQ(CountOccurrences(printed_reuse, "func.call @"), 2);
  EXPECT_THAT(printed_reuse, HasSubstr("stablehlo.pad"));
  EXPECT_THAT(printed_reuse, HasSubstr("stablehlo.optimization_barrier"));
  EXPECT_THAT(printed_reuse, HasSubstr("stablehlo.set_dimension_size"));
}

TEST_F(DispatchInvokeSubgraphTest,
       BoundedDynamicStaticInputAtUpperBoundSkipsPad) {
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kBoundedDynamicSubgraphMlir);
  const at::Tensor x_full = MakePlaceholder({4}, mlir::ElementType::I32);
  const std::vector<at::Tensor> res_full =
      PyCreateInvokeSubgraphOp({x_full}, subgraph);
  ASSERT_EQ(res_full.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef full_buf,
                          GetBuffer(res_full[0]));
  const LoweringResults lowered_full =
      LowerOpsToSingleModule({x_full}, full_buf.deferred_op().get());
  ASSERT_THAT(lowered_full.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_full.module)));
  EXPECT_THAT(DebugString(*lowered_full.module),
              Not(HasSubstr("stablehlo.pad")));
}

TEST_F(DispatchInvokeSubgraphTest, BoundedDynamicAdaptsSmallerBoundedInput) {
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kBoundedDynamicSubgraphMlir);
  const at::Tensor x_smaller_dyn = MakeBoundedDynamicPlaceholder(
      {2}, mlir::ElementType::I32,
      {{.dimension = 0, .lower_bound = 2, .upper_bound = 3}});
  const std::vector<at::Tensor> res_smaller_dyn =
      PyCreateInvokeSubgraphOp({x_smaller_dyn}, subgraph);
  ASSERT_EQ(res_smaller_dyn.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef smaller_dyn_buf,
                          GetBuffer(res_smaller_dyn[0]));
  const LoweringResults lowered_smaller = LowerOpsToSingleModule(
      {x_smaller_dyn}, smaller_dyn_buf.deferred_op().get());
  ASSERT_THAT(lowered_smaller.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_smaller.module)));
  const std::string printed_smaller = DebugString(*lowered_smaller.module);
  EXPECT_THAT(printed_smaller, HasSubstr("stablehlo.get_dimension_size"));
  EXPECT_THAT(printed_smaller, HasSubstr("stablehlo.pad"));
}

TEST_F(DispatchInvokeSubgraphTest, BoundedDynamicPartialStaticAndDynamicAxes) {
  const std::shared_ptr<ContextedModule> partial_sub =
      CreateSubgraphModule(kPartialBoundedDynamicSubgraphMlir);
  const at::Tensor x_partial = MakePlaceholder({2, 3}, mlir::ElementType::I32);
  const std::vector<at::Tensor> res_partial =
      PyCreateInvokeSubgraphOp({x_partial}, partial_sub);
  ASSERT_EQ(res_partial.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef partial_buf,
                          GetBuffer(res_partial[0]));
  EXPECT_THAT(partial_buf.shape().dimensions(), IsEmpty());
  EXPECT_EQ(partial_buf.shape().dtype(), mlir::ElementType::I32);
  const LoweringResults lowered_partial =
      LowerOpsToSingleModule({x_partial}, partial_buf.deferred_op().get());
  ASSERT_THAT(lowered_partial.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_partial.module)));
}

TEST_F(DispatchInvokeSubgraphTest,
       DynamicIntermediateInputToStaticSubgraphFailsAtLowering) {
  const at::Tensor x_static = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y_static = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kAddSubgraphMlir);

  // Dispatch succeeds when the tensor's Shape has no dynamic_dimensions (as
  // happens for intermediate tensors computed from dynamic inputs).
  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x_static, y_static}, subgraph);
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf, GetBuffer(results[0]));

  // At MLIR lowering time, the intermediate value's MLIR type is bounded
  // dynamic (`tensor<?x3xf32, #stablehlo.type_extensions<bounds = [2, ?]>>`),
  // which cannot be adapted to static `tensor<2x3xf32>`.
  const at::Tensor x_dyn_at_lowering = MakeBoundedDynamicPlaceholder(
      {2, 3}, mlir::ElementType::F32,
      {{.dimension = 0, .lower_bound = 2, .upper_bound = 2}});
  const LoweringResults lowered = LowerOpsToSingleModule(
      {x_dyn_at_lowering, y_static}, out_buf.deferred_op().get());
  EXPECT_THAT(
      lowered.results,
      StatusIs(error::kInvalidArgument,
               HasSubstr("the subgraph was exported with a static or "
                         "smaller-bounded shape, but this input is dynamic")));
}

TEST_F(DispatchInvokeSubgraphTest, InvalidSubgraphModuleArgumentsThrow) {
  constexpr std::string_view kNoMainSubgraphMlir = R"mlir(
    module {
      func.func private @helper(%a: tensor<2x3xf32>) -> tensor<2x3xf32> {
        return %a : tensor<2x3xf32>
      }
    }
  )mlir";

  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kAddSubgraphMlir);

  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({x, y}, nullptr); },
              Throws<TtError>(Property(&TtError::what, HasSubstr("null"))));

  const std::shared_ptr<ContextedModule> no_main =
      CreateSubgraphModule(kNoMainSubgraphMlir);
  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({x}, no_main); },
              Throws<TtError>(Property(&TtError::what, HasSubstr("@main"))));

  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({x}, subgraph); },
      Throws<TtError>(Property(&TtError::what, HasSubstr("expected 2"))));

  const at::Tensor cpu_tensor = at::zeros({2, 3}, at::kFloat);
  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({cpu_tensor, y}, subgraph); },
              Throws<TtError>());
}

TEST_F(DispatchInvokeSubgraphTest, InvalidStaticInputTypesThrow) {
  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kAddSubgraphMlir);

  const at::Tensor bad_dtype = MakePlaceholder({2, 3}, mlir::ElementType::I32);
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({x, bad_dtype}, subgraph); },
      Throws<TtError>(Property(&TtError::what, HasSubstr("dtype to be"))));

  const at::Tensor bad_rank = MakePlaceholder({6}, mlir::ElementType::F32);
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({x, bad_rank}, subgraph); },
      Throws<TtError>(Property(&TtError::what, HasSubstr("rank to be"))));

  const at::Tensor bad_shape = MakePlaceholder({3, 2}, mlir::ElementType::F32);
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({x, bad_shape}, subgraph); },
      Throws<TtError>(Property(&TtError::what, HasSubstr("shape to be"))));

  const at::Tensor dyn_for_static = MakeBoundedDynamicPlaceholder(
      {2, 3}, mlir::ElementType::F32,
      {{.dimension = 0, .lower_bound = 2, .upper_bound = 2}});
  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({dyn_for_static, y}, subgraph); },
              Throws<TtError>(
                  Property(&TtError::what, HasSubstr("shape to be static"))));
}

TEST_F(DispatchInvokeSubgraphTest, InvalidBoundedDynamicInputsThrow) {
  const std::shared_ptr<ContextedModule> partial_sub =
      CreateSubgraphModule(kPartialBoundedDynamicSubgraphMlir);

  const at::Tensor dyn_on_static_axis = MakeBoundedDynamicPlaceholder(
      {2, 3}, mlir::ElementType::I32,
      {{.dimension = 0, .lower_bound = 2, .upper_bound = 4},
       {.dimension = 1, .lower_bound = 2, .upper_bound = 3}});
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({dyn_on_static_axis}, partial_sub); },
      Throws<TtError>(
          Property(&TtError::what, HasSubstr("dimension 1 to be static"))));

  const at::Tensor bad_static_axis =
      MakePlaceholder({2, 2}, mlir::ElementType::I32);
  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({bad_static_axis}, partial_sub); },
              Throws<TtError>(Property(
                  &TtError::what, HasSubstr("dimension 1 to be 3, got 2"))));

  const at::Tensor too_big_static =
      MakePlaceholder({5, 3}, mlir::ElementType::I32);
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({too_big_static}, partial_sub); },
      Throws<TtError>(Property(
          &TtError::what, HasSubstr("dimension 0 to be at most 4, got 5"))));

  const at::Tensor too_big_dyn_bound = MakeBoundedDynamicPlaceholder(
      {2, 3}, mlir::ElementType::I32,
      {{.dimension = 0, .lower_bound = 2, .upper_bound = 5}});
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({too_big_dyn_bound}, partial_sub); },
      Throws<TtError>(Property(
          &TtError::what, HasSubstr("dimension 0 to be at most 4, got 5"))));
}

TEST_F(DispatchInvokeSubgraphTest, UnboundedDynamicArgThrows) {
  constexpr std::string_view kUnboundedArgMlir = R"mlir(
    module {
      func.func @main(%a: tensor<?xf32>) -> tensor<i32> {
        %0 = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?xf32>) -> tensor<i32>
        return %0 : tensor<i32>
      }
    }
  )mlir";
  const at::Tensor vec = MakePlaceholder({2}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> unbounded =
      CreateSubgraphModule(kUnboundedArgMlir);
  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({vec}, unbounded); },
              Throws<TtError>(
                  Property(&TtError::what, HasSubstr("unbounded dynamic"))));
}

TEST_F(DispatchInvokeSubgraphTest, PartiallyBoundedDynamicArgThrows) {
  constexpr std::string_view kPartiallyBoundedArgMlir = R"mlir(
    module {
      func.func @main(%a: tensor<?x?xf32, #stablehlo.type_extensions<bounds = [4, ?]>>)
          -> tensor<i32> {
        %0 = stablehlo.get_dimension_size %a, dim = 0
            : (tensor<?x?xf32, #stablehlo.type_extensions<bounds = [4, ?]>>) -> tensor<i32>
        return %0 : tensor<i32>
      }
    }
  )mlir";
  const std::shared_ptr<ContextedModule> partially_bounded =
      CreateSubgraphModule(kPartiallyBoundedArgMlir);
  const at::Tensor mat = MakePlaceholder({2, 2}, mlir::ElementType::F32);
  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({mat}, partially_bounded); },
              Throws<TtError>(
                  Property(&TtError::what, HasSubstr("unbounded dynamic"))));
}

TEST_F(DispatchInvokeSubgraphTest, DynamicResultThrows) {
  constexpr std::string_view kUnboundedResultMlir = R"mlir(
    module {
      func.func @main(%a: tensor<2xf32>, %size: tensor<1xi32>) -> tensor<?xf32> {
        %0 = stablehlo.dynamic_broadcast_in_dim %a, %size, dims = [0]
            : (tensor<2xf32>, tensor<1xi32>) -> tensor<?xf32>
        return %0 : tensor<?xf32>
      }
    }
  )mlir";
  constexpr std::string_view kBoundedResultMlir = R"mlir(
    module {
      func.func @main(%a: tensor<2xf32>) -> tensor<?xf32, #stablehlo.type_extensions<bounds = [2]>> {
        %c = stablehlo.constant dense<2> : tensor<i32>
        %0 = stablehlo.set_dimension_size %a, %c, dim = 0
            : (tensor<2xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.type_extensions<bounds = [2]>>
        return %0 : tensor<?xf32, #stablehlo.type_extensions<bounds = [2]>>
      }
    }
  )mlir";
  const at::Tensor vec = MakePlaceholder({2}, mlir::ElementType::F32);
  const at::Tensor shape_arg = MakePlaceholder({1}, mlir::ElementType::I32);
  const std::shared_ptr<ContextedModule> unbounded_result =
      CreateSubgraphModule(kUnboundedResultMlir);
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({vec, shape_arg}, unbounded_result); },
      Throws<TtError>(Property(
          &TtError::what,
          HasSubstr("has dynamic dimensions, which are not supported"))));

  const std::shared_ptr<ContextedModule> bounded_result =
      CreateSubgraphModule(kBoundedResultMlir);
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({vec}, bounded_result); },
      Throws<TtError>(Property(
          &TtError::what,
          HasSubstr("has dynamic dimensions, which are not supported"))));
}

TEST_F(DispatchInvokeSubgraphTest, UnrankedArgThrows) {
  constexpr std::string_view kUnrankedArgMlir = R"mlir(
    module {
      func.func @main(%a: tensor<*xf32>) -> tensor<*xf32> {
        return %a : tensor<*xf32>
      }
    }
  )mlir";
  const at::Tensor vec = MakePlaceholder({2}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> unranked =
      CreateSubgraphModule(kUnrankedArgMlir);
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({vec}, unranked); },
      Throws<TtError>(Property(&TtError::what, HasSubstr("ranked tensor"))));
}

TEST_F(DispatchInvokeSubgraphTest, UnrankedResultThrows) {
  constexpr std::string_view kUnrankedResultMlir = R"mlir(
    module {
      func.func @main(%a: tensor<2xf32>) -> tensor<*xf32> {
        %0 = stablehlo.custom_call @dummy(%a) : (tensor<2xf32>) -> tensor<*xf32>
        return %0 : tensor<*xf32>
      }
    }
  )mlir";
  const at::Tensor vec = MakePlaceholder({2}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> unranked_result =
      CreateSubgraphModule(kUnrankedResultMlir);
  EXPECT_THAT(
      [&] { PyCreateInvokeSubgraphOp({vec}, unranked_result); },
      Throws<TtError>(Property(&TtError::what, HasSubstr("ranked tensor"))));
}

TEST_F(DispatchInvokeSubgraphTest, BoundedDynamicUpperBoundLessThanTwoThrows) {
  constexpr std::string_view kSmallBoundArgMlir = R"mlir(
    module {
      func.func @main(%a: tensor<?xf32, #stablehlo.type_extensions<bounds = [1]>>)
          -> tensor<i32> {
        %0 = stablehlo.get_dimension_size %a, dim = 0
            : (tensor<?xf32, #stablehlo.type_extensions<bounds = [1]>>) -> tensor<i32>
        return %0 : tensor<i32>
      }
    }
  )mlir";
  const at::Tensor vec = MakePlaceholder({1}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> small_bound =
      CreateSubgraphModule(kSmallBoundArgMlir);
  EXPECT_THAT([&] { PyCreateInvokeSubgraphOp({vec}, small_bound); },
              Throws<TtError>(
                  Property(&TtError::what, HasSubstr("to be >= 2, got 1"))));
}

TEST_F(DispatchInvokeSubgraphTest,
       DeviceAssignmentAndPallasFlagAffectCacheKeys) {
  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> sub_base =
      CreateSubgraphModule(kAddSubgraphMlir);
  const std::shared_ptr<ContextedModule> sub_pallas =
      CreateSubgraphModule(kAddSubgraphMlir);
  const std::shared_ptr<ContextedModule> sub_global_only =
      CreateSubgraphModule(kAddSubgraphMlir);
  const std::shared_ptr<ContextedModule> sub_local_and_global =
      CreateSubgraphModule(kAddSubgraphMlir);
  sub_base->set_device_assignment(
      DeviceAssignment(/*local=*/{0, 1}, /*global=*/{0, 1},
                       /*contains_pallas_kernel=*/false));
  sub_pallas->set_device_assignment(
      DeviceAssignment(/*local=*/{0, 1}, /*global=*/{0, 1},
                       /*contains_pallas_kernel=*/true));
  sub_global_only->set_device_assignment(
      DeviceAssignment(/*local=*/{}, /*global=*/{0, 1}));
  sub_local_and_global->set_device_assignment(
      DeviceAssignment(/*local=*/{0, 1}, /*global=*/{0, 1, 2, 3}));

  const std::vector<at::Tensor> r_base =
      PyCreateInvokeSubgraphOp({x, y}, sub_base, "da");
  const std::vector<at::Tensor> r_pallas =
      PyCreateInvokeSubgraphOp({x, y}, sub_pallas, "da");
  const std::vector<at::Tensor> r_global_only =
      PyCreateInvokeSubgraphOp({x, y}, sub_global_only, "da");
  const std::vector<at::Tensor> r_local_and_global =
      PyCreateInvokeSubgraphOp({x, y}, sub_local_and_global, "da");
  ASSERT_EQ(r_base.size(), 1);
  ASSERT_EQ(r_pallas.size(), 1);
  ASSERT_EQ(r_global_only.size(), 1);
  ASSERT_EQ(r_local_and_global.size(), 1);

  TT_ASSERT_OK_AND_ASSIGN(const OpParamCacheKeys k_base,
                          GetOpParamCacheKeys(r_base[0]));
  TT_ASSERT_OK_AND_ASSIGN(const OpParamCacheKeys k_pallas,
                          GetOpParamCacheKeys(r_pallas[0]));
  EXPECT_THAT(k_pallas, Not(ElementsAreArray(k_base.begin(), k_base.end())));

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b_base, GetBuffer(r_base[0]));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b_global_only,
                          GetBuffer(r_global_only[0]));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef b_local_and_global,
                          GetBuffer(r_local_and_global[0]));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<Traversal> t_base,
      Traversal::Create(std::vector<DeviceBufferRef>{b_base}));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<Traversal> t_global_only,
      Traversal::Create(std::vector<DeviceBufferRef>{b_global_only}));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<Traversal> t_local_and_global,
      Traversal::Create(std::vector<DeviceBufferRef>{b_local_and_global}));
  EXPECT_NE(
      t_base->GetCacheKey(CompileOptionsKey(0), {}, {}).graph_key(),
      t_global_only->GetCacheKey(CompileOptionsKey(0), {}, {}).graph_key());
  EXPECT_NE(t_base->GetCacheKey(CompileOptionsKey(0), {}, {}).graph_key(),
            t_local_and_global->GetCacheKey(CompileOptionsKey(0), {}, {})
                .graph_key());
}

TEST_F(DispatchInvokeSubgraphTest,
       BoundedDynamicAdaptsMultiAxisMixedDynamicInputs) {
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kMultiAxisDynamicMlir);
  // Axis 0 already matches @main's upper bound 4 (needs runtime size saved
  // across Pad); axis 1 has upper bound 3 and must be padded up to 5.
  const at::Tensor x_mixed = MakeBoundedDynamicPlaceholder(
      {2, 2}, mlir::ElementType::I32,
      {{.dimension = 0, .lower_bound = 2, .upper_bound = 4},
       {.dimension = 1, .lower_bound = 2, .upper_bound = 3}});
  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x_mixed}, subgraph);
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf, GetBuffer(results[0]));
  EXPECT_THAT(out_buf.shape().dimensions(), IsEmpty());
  EXPECT_EQ(out_buf.shape().dtype(), mlir::ElementType::I32);

  const LoweringResults lowered =
      LowerOpsToSingleModule({x_mixed}, out_buf.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));
  const std::string printed = DebugString(*lowered.module);
  EXPECT_THAT(printed, HasSubstr("stablehlo.get_dimension_size"));
  EXPECT_THAT(printed, HasSubstr("stablehlo.pad"));
  EXPECT_THAT(printed, HasSubstr("stablehlo.set_dimension_size"));
}

TEST_F(DispatchInvokeSubgraphTest,
       BoundedDynamicAdaptsStaticAxisWithoutPadding) {
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kMultiAxisDynamicMlir);
  // Axis 0 already matches @main's bound 4, while axis 1 is static 5:
  // needs_pad is false, so axis 0 stays dynamic and axis 1 gets a barriered
  // runtime size of 5.
  const at::Tensor x_no_pad = MakeBoundedDynamicPlaceholder(
      {2, 5}, mlir::ElementType::I32,
      {{.dimension = 0, .lower_bound = 2, .upper_bound = 4}});
  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x_no_pad}, subgraph);
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf, GetBuffer(results[0]));

  const LoweringResults lowered =
      LowerOpsToSingleModule({x_no_pad}, out_buf.deferred_op().get());
  ASSERT_THAT(lowered.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered.module)));
  const std::string printed = DebugString(*lowered.module);
  EXPECT_THAT(printed, Not(HasSubstr("stablehlo.pad")));
  EXPECT_THAT(printed, HasSubstr("stablehlo.optimization_barrier"));
  EXPECT_THAT(printed, HasSubstr("stablehlo.set_dimension_size"));
}

TEST_F(DispatchInvokeSubgraphTest, PublicOrDeclarationEntrySymbolIsNotReused) {
  const at::Tensor x = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const at::Tensor y = MakePlaceholder({2, 3}, mlir::ElementType::F32);
  const std::shared_ptr<ContextedModule> subgraph =
      CreateSubgraphModule(kAddSubgraphMlir);
  const std::vector<at::Tensor> results =
      PyCreateInvokeSubgraphOp({x, y}, subgraph, "block");
  ASSERT_EQ(results.size(), 1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef out_buf, GetBuffer(results[0]));

  LoweringResults probe =
      LowerOpsToSingleModule({x, y}, out_buf.deferred_op().get());
  ASSERT_THAT(probe.results, IsOk());
  const std::string entry_sym = FindSymbolWithPrefix(*probe.module, "block_");
  ASSERT_FALSE(entry_sym.empty());

  const std::string pre_existing_public = absl::StrCat(
      R"mlir(
    module {
      func.func @)mlir",
      entry_sym,
      R"mlir((%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %0 = stablehlo.add %a, %b : tensor<2x3xf32>
        return %0 : tensor<2x3xf32>
      }
    }
  )mlir");
  const LoweringResults lowered_public = LowerOpsToSingleModule(
      {x, y}, out_buf.deferred_op().get(), /*second_inputs=*/{},
      /*second_op=*/nullptr, pre_existing_public);
  ASSERT_THAT(lowered_public.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_public.module)));
  EXPECT_THAT(DebugString(*lowered_public.module),
              HasSubstr(absl::StrCat("func.call @", entry_sym, "_0(")));

  const std::string pre_existing_decl = absl::StrCat(
      R"mlir(
    module {
      func.func private @)mlir",
      entry_sym,
      R"mlir((tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    }
  )mlir");
  const LoweringResults lowered_decl = LowerOpsToSingleModule(
      {x, y}, out_buf.deferred_op().get(), /*second_inputs=*/{},
      /*second_op=*/nullptr, pre_existing_decl);
  ASSERT_THAT(lowered_decl.results, IsOk());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered_decl.module)));
  EXPECT_THAT(DebugString(*lowered_decl.module),
              HasSubstr(absl::StrCat("func.call @", entry_sym, "_0(")));
}

}  // namespace
}  // namespace torch_tpu
