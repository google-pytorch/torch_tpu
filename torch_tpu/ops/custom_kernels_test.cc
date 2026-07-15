// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/custom_kernels.h"

#include <string_view>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "re2/re2.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::HasSubstr;

constexpr std::string_view kUnrefinedRank1AddMlirKernel = R"(
module @kernel_add {
  func.func public @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
    %1 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf32>) -> tensor<i32>
    %2 = stablehlo.compare  GE, %0, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%2, %0) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable 'a'. Using the following polymorphic shapes specifications: args[0].shape = (a,),args[1].shape = (a,). Obtained dimension variables: 'a' = {0} from specification 'a' for dimension args[0].shape[0] (= {0}),", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
    %3 = stablehlo.compare  EQ, %1, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%3, %1, %0) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Found inconsistency between dimension size args[1].shape[0] (= {0}) and the specification 'a' (= {1}). Using the following polymorphic shapes specifications: args[0].shape = (a,),args[1].shape = (a,). Obtained dimension variables: 'a' = {1} from specification 'a' for dimension args[0].shape[0] (= {1}), .", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
    %4 = call @_wrapped_export_main(%0, %arg0, %arg1) : (tensor<i32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %4 : tensor<?xf32>
  }
  func.func private @_wrapped_export_main(%arg0: tensor<i32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> (tensor<?xf32>) {
    %0 = stablehlo.add %arg1, %arg2 : tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}
)";

// A kernel whose scan is encoded as a stablehlo.custom_call with the body in
// a helper function referenced via called_computations. This is the form
// hlo-legalize-to-stablehlo produces for mhlo.scan (for example after the
// partitioning round-trip through HLO), so the kernel module holds more than
// one function.
constexpr std::string_view kCumsumScanMlirKernel = R"(
module @kernel_cumsum {
  func.func public @main(%arg0: tensor<4x2xi32>) -> tensor<4x2xi32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %init = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %0:2 = stablehlo.custom_call @mhlo.scan(%arg0, %init) {
      called_computations = [@scan_body],
      mhlo.attributes = {
        dimension = 1 : i64,
        is_associative = true,
        is_reverse = false,
        operandSegmentSizes = array<i32: 1, 1>,
        resultSegmentSizes = array<i32: 1, 1>,
        scan_dim_size = 2 : i64
      },
      mhlo.version = 1 : i64
    } : (tensor<4x2xi32>, tensor<4xi32>) -> (tensor<4x2xi32>, tensor<4xi32>)
    return %0#0 : tensor<4x2xi32>
  }
  func.func private @scan_body(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xi32>
    stablehlo.return %0, %0 : tensor<4xi32>, tensor<4xi32>
  }
}
)";

class CustomKernelRegistryTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This must be done before CustomKernelRegistry::GetInstance() is called,
    // as the latter depends on the PjRt client.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    absl::Status status = PjrtBackend::GetInstance().EnsureInitialized();
    ASSERT_TRUE(status.ok()) << status;
  }
};

TEST_F(CustomKernelRegistryTest, RegisterAndCallMlirKernel) {
  // Register the custom kernel
  RegisterCustomKernel("kernel_add", "", kUnrefinedRank1AddMlirKernel);

  // Initialize the MLIR builder
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  mlir::ModuleBuilder mb(context, mlir::unknownLoc(context));
  mlir::func::FunctionBuilder fb(mb, "main");

  // Load and call the loaded kernel and return
  mlir::Type arg_type =
      mlir::makeTensorType(fb.getContext(), {10}, mlir::ElementType::F32);
  mlir::MlirOp op1 = mlir::func::Argument(fb, arg_type);
  mlir::MlirOp op2 = mlir::func::Argument(fb, arg_type);
  auto results_status = CallCustomKernel(fb, {op1, op2}, "kernel_add", "");
  ASSERT_EQ(results_status.status(), absl::OkStatus());
  auto results = results_status.value();
  mlir::func::Return(fb, results);

  // Finish building the module and stringify it
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  auto debug_string = DebugString(module.get());

  // Check that we have both the loaded kernel and the main function.
  EXPECT_THAT(debug_string, HasSubstr("func.func private @kernel_add_0x"));
  EXPECT_THAT(debug_string, HasSubstr("func.func @main"));

  // Check that the loaded kernel is called.
  EXPECT_THAT(debug_string, HasSubstr("call @kernel_add_0x"));

  // Check that module is refined to the shapes of the concrete arguments.
  EXPECT_THAT(debug_string,
              HasSubstr("stablehlo.add %arg0, %arg1 : tensor<10xf32>"));
}

TEST_F(CustomKernelRegistryTest, RegisterAndCallKernelWithHelperFunction) {
  // Register a kernel whose module has a main function plus a helper function
  // referenced through called_computations.
  RegisterCustomKernel("kernel_cumsum", "", kCumsumScanMlirKernel);

  // Initialize the MLIR builder
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  mlir::ModuleBuilder mb(context, mlir::unknownLoc(context));
  mlir::func::FunctionBuilder fb(mb, "main");

  // Load and call the loaded kernel and return
  mlir::Type arg_type =
      mlir::makeTensorType(fb.getContext(), {4, 2}, mlir::ElementType::I32);
  mlir::MlirOp op1 = mlir::func::Argument(fb, arg_type);
  auto results_status = CallCustomKernel(fb, {op1}, "kernel_cumsum", "");
  ASSERT_TRUE(results_status.ok()) << results_status.status();
  mlir::func::Return(fb, results_status.value());

  // Finish building the module and stringify it
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  auto debug_string = DebugString(module.get());

  // The kernel entry function and the main function are both present, and the
  // kernel is called. The entry name embeds a hash of the kwargs and input
  // shapes/dtypes, so capture the actual name and assert the same one is
  // called rather than hard-coding the (unstable) suffix.
  std::string kernel_fn_name;
  ASSERT_TRUE(RE2::PartialMatch(debug_string,
                                R"(func\.func private (@kernel_cumsum_\w+))",
                                &kernel_fn_name))
      << debug_string;
  EXPECT_THAT(debug_string, HasSubstr("func.func @main"));
  EXPECT_THAT(debug_string, HasSubstr(absl::StrCat("call ", kernel_fn_name)));

  // The partitioning round-trip through HLO turns the custom_call encoding
  // back into an mhlo.scan op with its body inline (the helper function is
  // consumed); PJRT serialization re-encodes it when the graph is compiled.
  EXPECT_THAT(debug_string, HasSubstr("mhlo.scan"));
  EXPECT_THAT(debug_string, HasSubstr("dimension=1"));
}

TEST_F(CustomKernelRegistryTest, RegistrationIsIdempotent) {
  // Register the custom kernel
  RegisterCustomKernel("kernel_add", "", kUnrefinedRank1AddMlirKernel);

  // Try to register the custom kernel again
  bool inserted_again =
      RegisterCustomKernel("kernel_add", "", kUnrefinedRank1AddMlirKernel);
  // Check that it was not inserted a second time
  ASSERT_FALSE(inserted_again);
}

TEST_F(CustomKernelRegistryTest, RegisterWithDifferentKwargs) {
  // Register the custom kernel once
  RegisterCustomKernel("kernel_add", "", kUnrefinedRank1AddMlirKernel);

  // Register the custom kernel again with different kwargs
  bool inserted_again = RegisterCustomKernel("kernel_add", "different_kwargs",
                                             kUnrefinedRank1AddMlirKernel);
  // Check that we did insert a second copy of the kernel
  ASSERT_TRUE(inserted_again);
}

TEST_F(CustomKernelRegistryTest, CannotLoadNonexistentKernel) {
  // Initialize an MLIR builder
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  mlir::ModuleBuilder mb(context, mlir::unknownLoc(context));
  mlir::func::FunctionBuilder fb(mb, "main");

  // Try to call a kernel that has not been registered
  mlir::Type arg_type =
      mlir::makeTensorType(fb.getContext(), {10}, mlir::ElementType::F32);
  mlir::MlirOp op1 = mlir::func::Argument(fb, arg_type);
  mlir::MlirOp op2 = mlir::func::Argument(fb, arg_type);
  Shape arg_shape(Dimensions{10}, mlir::ElementType::F32);
  auto kernel_add_func_status =
      CallCustomKernel(fb, {op1, op2}, "does_not_exist", "");

  // Check that we get a not found error
  EXPECT_THAT(kernel_add_func_status,
              StatusIs(absl::StatusCode::kNotFound));  // STATUS_CODE_OK
}

TEST_F(CustomKernelRegistryTest, RegisterAndCallMlirKernel_InvalidShape) {
  // Register the custom kernel
  RegisterCustomKernel("kernel_add", "", kUnrefinedRank1AddMlirKernel);

  // Initialize the MLIR builder
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  mlir::ModuleBuilder mb(context, mlir::unknownLoc(context));
  mlir::func::FunctionBuilder fb(mb, "main");

  // Load the custom kernel
  // The kernel above has a shape assertion that the first argument has a
  // rank of >=1, so this refinement will fail.
  mlir::Type arg_type =
      mlir::makeTensorType(fb.getContext(), {0}, mlir::ElementType::F32);
  mlir::MlirOp op1 = mlir::func::Argument(fb, arg_type);
  mlir::MlirOp op2 = mlir::func::Argument(fb, arg_type);
  auto status = CallCustomKernel(fb, {op1, op2}, "kernel_add", "");
  EXPECT_THAT(status, StatusIs(error::kInternal));

  // Check for TorchTPU error wrapper
  EXPECT_THAT(status.status().message(),
              HasSubstr("failed to validate shape assertions"));
  EXPECT_THAT(
      status.status().message(),
      HasSubstr(
          "Input shapes do not match the polymorphic shapes specification"));
}

TEST_F(CustomKernelRegistryTest,
       RegisterAndCallMlirKernel_InvalidArgumentCount) {
  // Register the custom kernel
  RegisterCustomKernel("kernel_add", "", kUnrefinedRank1AddMlirKernel);

  // Initialize the MLIR builder
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  mlir::ModuleBuilder mb(context, mlir::unknownLoc(context));
  mlir::func::FunctionBuilder fb(mb, "main");

  // Load the custom kernel
  // The kernel above has a two arguments, so passing one argument will fail.
  mlir::Type arg_type =
      mlir::makeTensorType(fb.getContext(), {10}, mlir::ElementType::F32);
  mlir::MlirOp op = mlir::func::Argument(fb, arg_type);
  auto status = CallCustomKernel(fb, {op}, "kernel_add", "");
  EXPECT_THAT(status, StatusIs(error::kInternal));

  // Check for TorchTPU error wrapper
  EXPECT_THAT(status.status().message(),
              HasSubstr("failed to specialize custom kernel"));

  // Check for MLIR error message (controlled by the string in IR above)
  EXPECT_THAT(
      status.status().message(),
      HasSubstr(
          "number of refinements must match number of op operands 1 vs 2"));
}

}  // namespace
}  // namespace torch_tpu
