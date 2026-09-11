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

#include "csrc/internal/compile/dispatch_scan.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "absl/log/absl_check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/scan_builder.h"
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
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/mlir/utils/error_util.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::Property;
using testing::Throws;

// Basic MLIR step-function: next_carry = carry + x, output = carry + x.
//
// mlir signature:
//   func.func @main(%carry: tensor<1xi32>, %x: tensor<1xi32>)
//       -> (tensor<1xi32>, tensor<1xi32>)
static constexpr std::string_view kAddStepMlir = R"mlir(
  module {
    func.func @main(%carry: tensor<1xi32>, %x: tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>) {
      %0 = stablehlo.add %x, %carry : tensor<1xi32>
      return %0, %0 : tensor<1xi32>, tensor<1xi32>
    }
  }
)mlir";

// MLIR step-function that calls a private helper instead of doing the work
// inline. Cloning @main copies the call but not the helper, so this is the
// fixture that fails without ImportStepFunctionCallees.
//
// mlir signature:
//   func.func @main(%carry: tensor<1xi32>, %x: tensor<1xi32>)
//       -> (tensor<1xi32>, tensor<1xi32>)
static constexpr std::string_view kCalleeStepMlir = R"mlir(
  module {
    func.func private @scan_body_add(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
      %0 = stablehlo.add %a, %b : tensor<1xi32>
      return %0 : tensor<1xi32>
    }
    func.func @main(%carry: tensor<1xi32>, %x: tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>) {
      %0 = func.call @scan_body_add(%x, %carry) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
      return %0, %0 : tensor<1xi32>, tensor<1xi32>
    }
  }
)mlir";

// MLIR step-function with a two-deep helper chain: @main calls
// @scan_body_outer, which calls @scan_body_inner. Both must be renamed before
// either is cloned, or the copy of @scan_body_outer calls the old name.
//
// mlir signature:
//   func.func @main(%carry: tensor<1xi32>, %x: tensor<1xi32>)
//       -> (tensor<1xi32>, tensor<1xi32>)
static constexpr std::string_view kNestedCalleeStepMlir = R"mlir(
  module {
    func.func private @scan_body_inner(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
      %0 = stablehlo.add %a, %b : tensor<1xi32>
      return %0 : tensor<1xi32>
    }
    func.func private @scan_body_outer(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
      %0 = func.call @scan_body_inner(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
      return %0 : tensor<1xi32>
    }
    func.func @main(%carry: tensor<1xi32>, %x: tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>) {
      %0 = func.call @scan_body_outer(%x, %carry) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
      return %0, %0 : tensor<1xi32>, tensor<1xi32>
    }
  }
)mlir";

// MLIR step-function with multiple carries (%c1, %c2) and a single input slice
// (%x): next_c1 = c1 + x, next_c2 = c2 + x, output = c1 + x.
//
// mlir signature:
//   func.func @main(%c1: tensor<1xi32>, %c2: tensor<1xi32>, %x: tensor<1xi32>)
//       -> (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
static constexpr std::string_view kMultiCarryAddStepMlir = R"mlir(
  module {
    func.func @main(%c1: tensor<1xi32>, %c2: tensor<1xi32>, %x: tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) {
      %0 = stablehlo.add %x, %c1 : tensor<1xi32>
      %1 = stablehlo.add %x, %c2 : tensor<1xi32>
      return %0, %1, %0 : tensor<1xi32>, tensor<1xi32>, tensor<1xi32>
    }
  }
)mlir";

// MLIR step-function with a carry (%carry), a scannable input slice (%x), and a
// loop-invariant static input (%static_v):
// next_carry = carry + x + static_v, output = carry + x + static_v.
//
// mlir signature:
//   func.func @main(%carry: tensor<1xi32>, %x: tensor<1xi32>, %static_v:
//   tensor<1xi32>)
//       -> (tensor<1xi32>, tensor<1xi32>)
static constexpr std::string_view kStaticInputAddStepMlir = R"mlir(
  module {
    func.func @main(%carry: tensor<1xi32>, %x: tensor<1xi32>, %static_v: tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>) {
      %0 = stablehlo.add %x, %carry : tensor<1xi32>
      %1 = stablehlo.add %0, %static_v : tensor<1xi32>
      return %1, %1 : tensor<1xi32>, tensor<1xi32>
    }
  }
)mlir";

// MLIR step-function with a carry (%carry) and two scannable input slices (%x1,
// %x2): next_carry = carry + x1 + x2, output = carry + x1 + x2.
//
// mlir signature:
//   func.func @main(%carry: tensor<1xi32>, %x1: tensor<1xi32>, %x2:
//   tensor<1xi32>)
//       -> (tensor<1xi32>, tensor<1xi32>)
static constexpr std::string_view kMultiScannedInputsStepMlir = R"mlir(
  module {
    func.func @main(%carry: tensor<1xi32>, %x1: tensor<1xi32>, %x2: tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>) {
      %0 = stablehlo.add %carry, %x1 : tensor<1xi32>
      %1 = stablehlo.add %0, %x2 : tensor<1xi32>
      return %1, %1 : tensor<1xi32>, tensor<1xi32>
    }
  }
)mlir";

// Invalid MLIR step-function module that lacks a '@main' function (has
// '@step_fn' instead).
// computation: next_carry = carry + x, output = carry + x.
//
// mlir signature:
//   func.func @step_fn(%carry: tensor<1xi32>, %x: tensor<1xi32>)
//       -> (tensor<1xi32>, tensor<1xi32>)
static constexpr std::string_view kNoMainStepMlir = R"mlir(
  module {
    func.func @step_fn(%carry: tensor<1xi32>, %x: tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>) {
      %0 = stablehlo.add %x, %carry : tensor<1xi32>
      return %0, %0 : tensor<1xi32>, tensor<1xi32>
    }
  }
)mlir";

// Step-function module that takes only a carry input and no scannable inputs:
// next_carry = carry.
//
// mlir signature:
//   func.func @main(%carry: tensor<1xi32>) -> (tensor<1xi32>)
static constexpr std::string_view kEmptyInputsStepMlir = R"mlir(
  module {
    func.func @main(%carry: tensor<1xi32>) -> (tensor<1xi32>) {
      return %carry : tensor<1xi32>
    }
  }
)mlir";

// MLIR step-function module that takes a 2D carry and an input slice (both
// shape [3]): next_carry = carry + x, output = carry + x.
//
// mlir signature:
//   func.func @main(%carry: tensor<3xi32>, %x: tensor<3xi32>)
//       -> (tensor<3xi32>, tensor<3xi32>)
static constexpr std::string_view kAddStep2DMlir = R"mlir(
  module {
    func.func @main(%carry: tensor<3xi32>, %x: tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>) {
      %0 = stablehlo.add %x, %carry : tensor<3xi32>
      return %0, %0 : tensor<3xi32>, tensor<3xi32>
    }
  }
)mlir";

class DispatchScanTest : public testing::Test {
 protected:
  DispatchScanTest() {
    prev_mode_ = GetEagerMode();
    // Use `kInternalCompileFxGraph` to allow us to lower to MLIR and verify the
    // correctness of the lowering phase.
    SetEagerMode(EagerMode::kInternalCompileFxGraph);
  }

  ~DispatchScanTest() override { SetEagerMode(prev_mode_); }

  std::shared_ptr<ContextedModule> CreateBodyModule(std::string_view mlir_src) {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::stablehlo::StablehloDialect>();
    context->loadDialect<mlir::func::FuncDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module_op =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_src, context.get());
    EXPECT_TRUE(module_op);
    return std::make_shared<ContextedModule>(std::move(context),
                                             std::move(module_op));
  }

  struct Standard1DInputs {
    at::Tensor input;         // UNINITIALIZED_TENSOR_OK
    at::Tensor carry;         // UNINITIALIZED_TENSOR_OK
    at::Tensor dummy_output;  // UNINITIALIZED_TENSOR_OK
    std::shared_ptr<ContextedModule> body_module;
  };

  Standard1DInputs GetStandard1DInputs(
      std::string_view mlir_src = kAddStepMlir) {
    return {.input = MakePlaceholderI32({4, 1}),
            .carry = MakePlaceholderI32({1}),
            .dummy_output = MakePlaceholderI32({4, 1}),
            .body_module = CreateBodyModule(mlir_src)};
  }

  at::Tensor MakePlaceholderI32(Dimensions dimensions) {
    const auto placeholder_or =
        DeviceBufferList::CreatePlaceholder(dimensions, mlir::ElementType::I32);
    ABSL_CHECK(placeholder_or.ok()) << placeholder_or.status();
    return MakeTensor(*placeholder_or);
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
    // Lowering only requires shape and type metadata to build the StableHLO
    // loop structure and verify type safety. Therefore, we translate each
    // eager PyTorch tensor into a mock StableHLO ConstantOp initialized with
    // dummy zero values of the exact same shape and type.
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
                      int64_t output_buffer_index = 1,
                      std::string* lowered_mlir_out = nullptr) {
    ASSERT_EQ(results.size(), expected_results_size);
    TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef outputs_buf,
                            GetBuffer(results[output_buffer_index]));
    const std::shared_ptr<DeferredOp> deferred_op = outputs_buf.deferred_op();
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

TEST_F(DispatchScanTest, With1D) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs();

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init}, {input}, body_module, ScanDirection::kForward,
      {dummy_output}, /*num_scan_inputs=*/1);
  ASSERT_EQ(results.size(), 2);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef carry_buf,
                          GetBuffer(results[0]));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef outputs_buf,
                          GetBuffer(results[1]));

  EXPECT_THAT(carry_buf.shape().dimensions(), ElementsAre(1));
  EXPECT_THAT(outputs_buf.shape().dimensions(), ElementsAre(4, 1));

  SCOPED_TRACE("With1D");
  LowerAndVerify(results, {carry_init, input}, /*expected_results_size=*/2);
}

TEST_F(DispatchScanTest, StepFunctionCalleeIsImported) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs(kCalleeStepMlir);

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init}, {input}, body_module, ScanDirection::kForward,
      {dummy_output}, /*num_scan_inputs=*/1);
  ASSERT_EQ(results.size(), 2);

  SCOPED_TRACE("StepFunctionCalleeIsImported");
  std::string lowered_mlir;
  LowerAndVerify(results, {carry_init, input}, /*expected_results_size=*/2,
                 /*output_buffer_index=*/1, &lowered_mlir);

  // Check the definition came with the call, so the call is not a dangling
  // reference.
  EXPECT_THAT(lowered_mlir, HasSubstr("func.call @scan_body_add"));
  EXPECT_THAT(lowered_mlir, HasSubstr("func.func private @scan_body_add"));
}

TEST_F(DispatchScanTest, NestedStepFunctionCalleesAreImported) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs(kNestedCalleeStepMlir);

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init}, {input}, body_module, ScanDirection::kForward,
      {dummy_output}, /*num_scan_inputs=*/1);
  ASSERT_EQ(results.size(), 2);

  SCOPED_TRACE("NestedStepFunctionCalleesAreImported");
  std::string lowered_mlir;
  LowerAndVerify(results, {carry_init, input}, /*expected_results_size=*/2,
                 /*output_buffer_index=*/1, &lowered_mlir);

  EXPECT_THAT(lowered_mlir, HasSubstr("func.func private @scan_body_outer"));
  EXPECT_THAT(lowered_mlir, HasSubstr("func.func private @scan_body_inner"));
  // No `func.` prefix: MLIR elides it inside a `func.func` body, where `func`
  // is the default dialect. The `_0` is the rename the import does before
  // cloning, so matching it pins the nested call to the imported callee.
  EXPECT_THAT(lowered_mlir, HasSubstr("call @scan_body_inner_0"));
}

TEST_F(DispatchScanTest, With2D) {
  const at::Tensor input = MakePlaceholderI32({4, 3});
  const at::Tensor carry_init = MakePlaceholderI32({3});
  const at::Tensor dummy_output = MakePlaceholderI32({4, 3});

  const std::shared_ptr<ContextedModule> body_module =
      CreateBodyModule(kAddStep2DMlir);

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init}, {input}, body_module, ScanDirection::kForward,
      {dummy_output}, /*num_scan_inputs=*/1);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef carry_buf,
                          GetBuffer(results[0]));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef outputs_buf,
                          GetBuffer(results[1]));

  EXPECT_THAT(carry_buf.shape().dimensions(), ElementsAre(3));
  EXPECT_THAT(outputs_buf.shape().dimensions(), ElementsAre(4, 3));

  std::string lowered_mlir;
  SCOPED_TRACE("With2D");
  LowerAndVerify(results, {carry_init, input}, /*expected_results_size=*/2,
                 /*output_buffer_index=*/1, &lowered_mlir);

  // This verifies that slicing occurred unconditionally along dimension 0
  // (seq_len 4) producing slice shapes [3] and output accumulator shape [4, 3].
  EXPECT_THAT(lowered_mlir, HasSubstr("tensor<3xi32>"));
  EXPECT_THAT(lowered_mlir, HasSubstr("tensor<4x3xi32>"));
  // Should not slice features of size 4.
  EXPECT_THAT(lowered_mlir, testing::Not(HasSubstr("tensor<4xi32>")));
}

TEST_F(DispatchScanTest, WithMultiCarries) {
  const auto [input, unused_carry, dummy_output, body_module] =
      GetStandard1DInputs(kMultiCarryAddStepMlir);
  (void)unused_carry;
  const at::Tensor carry1 = MakePlaceholderI32({1});
  const at::Tensor carry2 = MakePlaceholderI32({1});

  const std::vector<at::Tensor> results =
      PyCreateScanOp({carry1, carry2}, {input}, body_module,
                     ScanDirection::kForward, {dummy_output},
                     /*num_scan_inputs=*/1);

  SCOPED_TRACE("WithMultiCarries");
  LowerAndVerify(results, {carry1, carry2, input},
                 /*expected_results_size=*/3,
                 /*output_buffer_index=*/2);
}

TEST_F(DispatchScanTest, WithStaticInputs) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs(kStaticInputAddStepMlir);
  const at::Tensor static_val = MakePlaceholderI32({1});

  const std::vector<at::Tensor> results =
      PyCreateScanOp({carry_init}, {input, static_val}, body_module,
                     ScanDirection::kForward, {dummy_output},
                     /*num_scan_inputs=*/1);

  SCOPED_TRACE("WithStaticInputs");
  LowerAndVerify(results, {carry_init, input, static_val},
                 /*expected_results_size=*/2);
}

TEST_F(DispatchScanTest, WithMultiScannedInputs) {
  const auto [input1, carry_init, dummy_output, body_module] =
      GetStandard1DInputs(kMultiScannedInputsStepMlir);
  const at::Tensor input2 = MakePlaceholderI32({4, 1});

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init}, {input1, input2}, body_module, ScanDirection::kForward,
      {dummy_output}, /*num_scan_inputs=*/2);

  SCOPED_TRACE("WithMultiScannedInputs");
  LowerAndVerify(results, {carry_init, input1, input2},
                 /*expected_results_size=*/2);
}

TEST_F(DispatchScanTest, WithReverse) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs();

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init}, {input}, body_module, ScanDirection::kReverse,
      {dummy_output}, /*num_scan_inputs=*/1);

  SCOPED_TRACE("WithReverse");
  LowerAndVerify(results, {carry_init, input}, /*expected_results_size=*/2);
}

TEST_F(DispatchScanTest, VerifyCacheKeys) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs();

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init}, {input}, body_module, ScanDirection::kForward,
      {dummy_output}, /*num_scan_inputs=*/1);
  ASSERT_EQ(results.size(), 2);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef outputs_buf,
                          GetBuffer(results[1]));
  const std::shared_ptr<DeferredOp> deferred_op = outputs_buf.deferred_op();
  ASSERT_TRUE(deferred_op != nullptr);

  const OpParamCacheKeys& cache_keys = deferred_op->op_param_cache_keys();
  EXPECT_THAT(
      cache_keys,
      testing::UnorderedElementsAre(
          testing::Pair("scan_direction", Fingerprint("forward")),
          testing::Pair("num_scan_inputs", 1), testing::Pair("num_carries", 1),
          testing::Pair("body_mlir", testing::_)));
}

TEST_F(DispatchScanTest, CpuTensorInputsError) {
  auto params = GetStandard1DInputs();
  params.input = at::zeros({4, 1}, at::kInt);  // CPU tensor.
  const auto [input, carry_init, dummy_output, body_module] = params;

  auto create_scan_op = [carry_init, input, body_module, dummy_output] {
    PyCreateScanOp({carry_init}, {input}, body_module, ScanDirection::kForward,
                   {dummy_output}, /*num_scan_inputs=*/1);
  };
  EXPECT_THAT(create_scan_op,
              Throws<TtError>(Property(
                  &TtError::what, HasSubstr("tensor is expected to be on"))));
}

TEST_F(DispatchScanTest, InvalidMlirModuleError) {
  auto params = GetStandard1DInputs();
  auto context = std::make_unique<mlir::MLIRContext>();
  mlir::OwningOpRef<mlir::ModuleOp> null_module;
  params.body_module = std::make_shared<ContextedModule>(
      std::move(context), std::move(null_module));
  const auto [input, carry_init, dummy_output, body_module] = params;

  auto create_scan_op = [carry_init, input, body_module, dummy_output] {
    PyCreateScanOp({carry_init}, {input}, body_module, ScanDirection::kForward,
                   {dummy_output}, /*num_scan_inputs=*/1);
  };
  EXPECT_THAT(create_scan_op,
              Throws<TtError>(
                  Property(&TtError::what, HasSubstr("invalid body module"))));
}

TEST_F(DispatchScanTest, MissingMainError) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs(kNoMainStepMlir);

  auto create_scan_op = [carry_init, input, body_module, dummy_output] {
    PyCreateScanOp({carry_init}, {input}, body_module, ScanDirection::kForward,
                   {dummy_output}, /*num_scan_inputs=*/1);
  };
  EXPECT_THAT(create_scan_op,
              Throws<TtError>(Property(
                  &TtError::what,
                  HasSubstr("expected 'main' function in body MLIR module"))));
}

TEST_F(DispatchScanTest, EmptyInputsError) {
  const auto [unused_input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs(kEmptyInputsStepMlir);
  (void)unused_input;

  auto create_scan_op = [carry_init, body_module, dummy_output] {
    PyCreateScanOp({carry_init}, {}, body_module, ScanDirection::kForward,
                   {dummy_output}, /*num_scan_inputs=*/0);
  };
  EXPECT_THAT(
      create_scan_op,
      Throws<TtError>(Property(&TtError::what,
                               HasSubstr("expected at least 1 input tensor"))));
}

TEST_F(DispatchScanTest, MismatchedDummyOutputsError) {
  const auto [input, carry_init, unused_output, body_module] =
      GetStandard1DInputs();
  (void)unused_output;
  const at::Tensor dummy_output1 = MakePlaceholderI32({4, 1});
  const at::Tensor dummy_output2 = MakePlaceholderI32({4, 1});

  auto create_scan_op = [carry_init, input, body_module, dummy_output1,
                         dummy_output2] {
    PyCreateScanOp({carry_init}, {input}, body_module, ScanDirection::kForward,
                   {dummy_output1, dummy_output2}, /*num_scan_inputs=*/1);
  };
  EXPECT_THAT(
      create_scan_op,
      Throws<TtError>(Property(&TtError::what,
                               HasSubstr("expected 1 dummy outputs, got 2"))));
}

TEST_F(DispatchScanTest, TooFewResultsError) {
  const auto [input, unused_carry, dummy_output, body_module] =
      GetStandard1DInputs();
  (void)unused_carry;
  const at::Tensor carry1 = MakePlaceholderI32({1});
  const at::Tensor carry2 = MakePlaceholderI32({1});
  const at::Tensor carry3 = MakePlaceholderI32({1});

  auto create_scan_op = [carry1, carry2, carry3, input, body_module,
                         dummy_output] {
    PyCreateScanOp({carry1, carry2, carry3}, {input}, body_module,
                   ScanDirection::kForward, {dummy_output},
                   /*num_scan_inputs=*/1);
  };
  EXPECT_THAT(
      create_scan_op,
      Throws<TtError>(Property(
          &TtError::what,
          HasSubstr("expected step function to have at least 3 result types "
                    "(corresponding to carries), but it only has 2"))));
}

TEST_F(DispatchScanTest, MismatchedInputsDuringLowering) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs();

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init}, {input}, body_module, ScanDirection::kForward,
      {dummy_output}, /*num_scan_inputs=*/1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef outputs_buf,
                          GetBuffer(results[1]));
  const std::shared_ptr<DeferredOp> deferred_op = outputs_buf.deferred_op();

  const LoweringResults lowered = LowerToMlir(deferred_op.get(), {carry_init});
  EXPECT_THAT(
      lowered.results_or,
      StatusIs(error::kInvalidArgument, HasSubstr("expected 2 inputs, got 1")));
}

TEST_F(DispatchScanTest, DtypeMismatchError) {
  const auto [input, unused_carry, dummy_output, body_module] =
      GetStandard1DInputs();
  (void)unused_carry;

  // Create an F32 carry tensor locally to trigger dtype mismatch.
  const auto placeholder_or =
      DeviceBufferList::CreatePlaceholder({1}, mlir::ElementType::F32);
  ABSL_CHECK(placeholder_or.ok()) << placeholder_or.status();
  const at::Tensor carry_init_f32 = MakeTensor(*placeholder_or);

  const std::vector<at::Tensor> results = PyCreateScanOp(
      {carry_init_f32}, {input}, body_module, ScanDirection::kForward,
      {dummy_output}, /*num_scan_inputs=*/1);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef outputs_buf,
                          GetBuffer(results[1]));
  const std::shared_ptr<DeferredOp> deferred_op = outputs_buf.deferred_op();

  const LoweringResults lowered =
      LowerToMlir(deferred_op.get(), {carry_init_f32, input});
  ASSERT_TRUE(lowered.results_or.ok());

  mlir::BaseScopedDiagnosticHandler handler(lowered.context.get());
  EXPECT_TRUE(mlir::failed(mlir::verify(*lowered.module)));
  EXPECT_THAT(
      handler.ConsumeStatus(),
      StatusIs(
          error::kPythonRuntimeError,
          HasSubstr("expect operands to be compatible with body block return "
                    "types but got 'tensor<i64>', 'tensor<1xf32>', "
                    "'tensor<4x1xi32>' vs 'tensor<i64>', 'tensor<1xi32>', "
                    "'tensor<4x1xi32>'")));
}

struct NumScanInputsParam {
  int64_t num_scan_inputs;
  std::string_view expected_error;
};

class DispatchScanNumScanInputsTest
    : public DispatchScanTest,
      public testing::WithParamInterface<NumScanInputsParam> {};

TEST_P(DispatchScanNumScanInputsTest, InvalidNumScanInputs) {
  const auto [input, carry_init, dummy_output, body_module] =
      GetStandard1DInputs();
  const NumScanInputsParam& param = GetParam();

  const int64_t num_scan_inputs = param.num_scan_inputs;
  auto create_scan_op = [carry_init, input, body_module, dummy_output,
                         num_scan_inputs] {
    PyCreateScanOp({carry_init}, {input}, body_module, ScanDirection::kForward,
                   {dummy_output}, num_scan_inputs);
  };
  EXPECT_THAT(create_scan_op,
              Throws<TtError>(
                  Property(&TtError::what, HasSubstr(param.expected_error))));
}

INSTANTIATE_TEST_SUITE_P(
    DispatchScanTest, DispatchScanNumScanInputsTest,
    testing::Values(
        NumScanInputsParam{
            .num_scan_inputs = 0,
            .expected_error = "expected num_scan_inputs to be greater than 0"},
        NumScanInputsParam{
            .num_scan_inputs = -1,
            .expected_error = "expected num_scan_inputs to be greater than 0"},
        NumScanInputsParam{
            .num_scan_inputs = 2,
            .expected_error =
                "to be less than or equal to the number of inputs"}));

}  // namespace
}  // namespace torch_tpu
