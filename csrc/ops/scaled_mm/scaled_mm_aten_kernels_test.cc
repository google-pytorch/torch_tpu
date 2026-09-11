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

#include "csrc/ops/scaled_mm/scaled_mm_aten_kernels.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "ATen/BlasBackend.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Exception.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::HasSubstr;
using Precision = mlir::stablehlo::Precision;

class ScaledMmBuilderTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    (void)PjrtBackend::GetInstance().EnsureInitialized();
  }

  ScaledMmBuilderTest()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    mlir::func::registerAllExtensions(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
    func_builder_ =
        std::make_unique<mlir::func::FunctionBuilder>(module_builder_, "main");
  }

  mlir::MLIRContext* context() { return &context_; }
  mlir::MlirBuilder& builder() { return *func_builder_; }
  mlir::OpBuilder& op_builder() { return func_builder_->getOpBuilder(); }

  std::string DumpModule(mlir::MlirOp result) {
    mlir::func::Return(*func_builder_, result);
    mlir::OwningOpRef<mlir::ModuleOp> module = module_builder_.build();
    std::string ir;
    llvm::raw_string_ostream os(ir);
    module->print(os);
    return ir;
  }

  mlir::MlirOp MakeFp8Constant(int64_t rows, int64_t cols,
                               mlir::Type fp8_type) {
    mlir::MlirOp f32_const =
        MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {rows, cols});
    return mlir::stablehlo::ConvertElementType(f32_const, fp8_type);
  }

  void VerifyNativeKernelResult(
      const absl::StatusOr<mlir::MlirOp>& status_or_result,
      llvm::ArrayRef<int64_t> expected_shape,
      std::string_view expected_kernel_name) {
    if (PjrtBackend::GetInstance().IsInitialized()) {
      TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result, status_or_result);
      auto result_type =
          mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
      ASSERT_TRUE(result_type);
      EXPECT_EQ(result_type.getShape(), expected_shape);
      EXPECT_THAT(DumpModule(result), HasSubstr(expected_kernel_name));
    } else {
      // On CPU host unit tests without a TPU PjRt client initialized,
      // CallCustomKernel specializes and validates the embedded Pallas kernel
      // MLIR (verifying shape compatibility of padded operands/scales) before
      // failing at PjRt client lookup in InjectCustomKernel.
      EXPECT_THAT(status_or_result,
                  StatusIs(error::kInternal,
                           HasSubstr("PjRt client is not initialized.")));
    }
  }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
  std::unique_ptr<mlir::func::FunctionBuilder> func_builder_;
};

TEST_F(ScaledMmBuilderTest, NativeBlockWiseE4M3Aligned) {
  mlir::Type e4m3_type = mlir::Float8E4M3FNType::get(context());
  mlir::MlirOp lhs = MakeFp8Constant(128, 256, e4m3_type);
  mlir::MlirOp rhs = MakeFp8Constant(256, 128, e4m3_type);
  mlir::MlirOp scale_a =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {128, 8});
  mlir::MlirOp scale_b =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {8, 128});

  auto status_or_result = BuildScaledMmV2Shlo(
      lhs, rhs, scale_a, scale_b, /*bias=*/std::nullopt,
      /*out_dtype=*/std::nullopt,
      Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=Unit test fixture input
      at::blas::ScalingType::BlockWise1x32,
      at::blas::ScalingType::BlockWise1x32, at::blas::SwizzleType::NO_SWIZZLE,
      at::blas::SwizzleType::NO_SWIZZLE);

  VerifyNativeKernelResult(status_or_result, {128, 128},
                           "blockwise_scaled_mm_e4m3");
}

TEST_F(ScaledMmBuilderTest, NativeBlockWiseE5M2Aligned) {
  mlir::Type e5m2_type = mlir::Float8E5M2Type::get(context());
  mlir::MlirOp lhs = MakeFp8Constant(128, 256, e5m2_type);
  mlir::MlirOp rhs = MakeFp8Constant(256, 128, e5m2_type);
  mlir::MlirOp scale_a =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {128, 8});
  mlir::MlirOp scale_b =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {8, 128});

  auto status_or_result = BuildScaledMmV2Shlo(
      lhs, rhs, scale_a, scale_b, /*bias=*/std::nullopt,
      /*out_dtype=*/std::nullopt,
      Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=Unit test fixture input
      at::blas::ScalingType::BlockWise1x32,
      at::blas::ScalingType::BlockWise1x32, at::blas::SwizzleType::NO_SWIZZLE,
      at::blas::SwizzleType::NO_SWIZZLE);

  VerifyNativeKernelResult(status_or_result, {128, 128},
                           "blockwise_scaled_mm_e5m2");
}

TEST_F(ScaledMmBuilderTest, NativeBlockWiseUnalignedPaddingAndBias) {
  mlir::Type e4m3_type = mlir::Float8E4M3FNType::get(context());
  // M=64, K=128, N=64 exercises both M/K padding and K/N padding branches in
  // BuildBlockWiseScaledMmNativeShlo (padding to 128x256x128).
  mlir::MlirOp lhs = MakeFp8Constant(64, 128, e4m3_type);
  mlir::MlirOp rhs = MakeFp8Constant(128, 64, e4m3_type);
  mlir::MlirOp scale_a =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {64, 4});
  mlir::MlirOp scale_b =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {4, 64});
  mlir::MlirOp bias =
      MakeConstant(builder(), 0.5f, op_builder().getF32Type(), {64});

  auto status_or_result = BuildScaledMmV2Shlo(
      lhs, rhs, scale_a, scale_b, bias, op_builder().getBF16Type(),
      Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=Unit test fixture input
      at::blas::ScalingType::BlockWise1x32,
      at::blas::ScalingType::BlockWise1x32, at::blas::SwizzleType::NO_SWIZZLE,
      at::blas::SwizzleType::NO_SWIZZLE);

  VerifyNativeKernelResult(status_or_result, {64, 64},
                           "blockwise_scaled_mm_e4m3");
}

TEST_F(ScaledMmBuilderTest, NativeBlockWiseWithSwizzle32_4_4) {
  mlir::Type e4m3_type = mlir::Float8E4M3FNType::get(context());
  mlir::MlirOp lhs = MakeFp8Constant(128, 128, e4m3_type);
  mlir::MlirOp rhs = MakeFp8Constant(128, 128, e4m3_type);
  // For M=128, N=128, K=128 (K_logical = 4):
  // Padded SWIZZLE_32_4_4 scale buffers have 128 * 4 = 512 elements.
  mlir::MlirOp scale_a =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {512});
  mlir::MlirOp scale_b =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {512});

  auto status_or_result = BuildScaledMmV2Shlo(
      lhs, rhs, scale_a, scale_b, /*bias=*/std::nullopt,
      /*out_dtype=*/std::nullopt,
      Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=Unit test fixture input
      at::blas::ScalingType::BlockWise1x32,
      at::blas::ScalingType::BlockWise1x32,
      at::blas::SwizzleType::SWIZZLE_32_4_4,
      at::blas::SwizzleType::SWIZZLE_32_4_4);

  VerifyNativeKernelResult(status_or_result, {128, 128},
                           "blockwise_scaled_mm_e4m3");
}

TEST_F(ScaledMmBuilderTest, NativeBlockWiseUnalignedWithSwizzle32_4_4) {
  mlir::Type e4m3_type = mlir::Float8E4M3FNType::get(context());
  // M=32, K=64 (K_logical=2), N=64 exercises both BuildUnswizzle's ceil_div +
  // Slice (unswizzling 512-element padded tile to [32, 2] and [2, 64]) and
  // BuildBlockWiseScaledMmNativeShlo's Pad + CallCustomKernel + Slice.
  mlir::MlirOp lhs = MakeFp8Constant(32, 64, e4m3_type);
  mlir::MlirOp rhs = MakeFp8Constant(64, 64, e4m3_type);
  mlir::MlirOp scale_a =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {512});
  mlir::MlirOp scale_b =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {512});

  auto status_or_result = BuildScaledMmV2Shlo(
      lhs, rhs, scale_a, scale_b, /*bias=*/std::nullopt,
      op_builder().getBF16Type(),
      Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=Unit test fixture input
      at::blas::ScalingType::BlockWise1x32,
      at::blas::ScalingType::BlockWise1x32,
      at::blas::SwizzleType::SWIZZLE_32_4_4,
      at::blas::SwizzleType::SWIZZLE_32_4_4);

  VerifyNativeKernelResult(status_or_result, {32, 64},
                           "blockwise_scaled_mm_e4m3");
}

TEST_F(ScaledMmBuilderTest, AsymmetricLhsBlockWiseRhsTensorWiseFallback) {
  mlir::Type e4m3_type = mlir::Float8E4M3FNType::get(context());
  mlir::MlirOp lhs = MakeFp8Constant(128, 128, e4m3_type);
  mlir::MlirOp rhs = MakeFp8Constant(128, 128, e4m3_type);
  mlir::MlirOp scale_a =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {128, 4});
  mlir::MlirOp scale_b =
      MakeConstant(builder(), 2.0f, op_builder().getF32Type(), {1, 1});
  mlir::MlirOp bias =
      MakeConstant(builder(), 0.5f, op_builder().getF32Type(), {128});

  auto status_or_result = BuildScaledMmV2Shlo(
      lhs, rhs, scale_a, scale_b, bias, op_builder().getBF16Type(),
      Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=Unit test fixture input
      at::blas::ScalingType::BlockWise1x32, at::blas::ScalingType::TensorWise,
      at::blas::SwizzleType::NO_SWIZZLE, at::blas::SwizzleType::NO_SWIZZLE);
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result, status_or_result);

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_EQ(result_type.getShape(), mlir::ArrayRef<int64_t>({128, 128}));
  EXPECT_TRUE(result_type.getElementType().isBF16());

  std::string ir = DumpModule(result);
  EXPECT_THAT(ir, HasSubstr("stablehlo.dot_general"));
}

// Verifies asymmetric fallback when LHS uses TensorWise and RHS uses BlockWise.
TEST_F(ScaledMmBuilderTest, AsymmetricLhsTensorWiseRhsBlockWiseFallback) {
  mlir::Type e4m3_type = mlir::Float8E4M3FNType::get(context());
  mlir::MlirOp lhs = MakeFp8Constant(128, 128, e4m3_type);
  mlir::MlirOp rhs = MakeFp8Constant(128, 128, e4m3_type);
  mlir::MlirOp scale_a =
      MakeConstant(builder(), 2.0f, op_builder().getF32Type(), {1, 1});
  mlir::MlirOp scale_b =
      MakeConstant(builder(), 1.0f, op_builder().getF32Type(), {4, 128});

  auto status_or_result = BuildScaledMmV2Shlo(
      lhs, rhs, scale_a, scale_b, /*bias=*/std::nullopt,
      /*out_dtype=*/std::nullopt,
      Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=Unit test fixture input
      at::blas::ScalingType::TensorWise, at::blas::ScalingType::BlockWise1x32,
      at::blas::SwizzleType::NO_SWIZZLE, at::blas::SwizzleType::NO_SWIZZLE);
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result, status_or_result);

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_EQ(result_type.getShape(), mlir::ArrayRef<int64_t>({128, 128}));

  std::string ir = DumpModule(result);
  EXPECT_THAT(ir, HasSubstr("stablehlo.dot_general"));
}

TEST_F(ScaledMmBuilderTest, TensorWiseBothOperandsFallback) {
  mlir::Type e4m3_type = mlir::Float8E4M3FNType::get(context());
  mlir::MlirOp lhs = MakeFp8Constant(128, 128, e4m3_type);
  mlir::MlirOp rhs = MakeFp8Constant(128, 128, e4m3_type);
  mlir::MlirOp scale_a =
      MakeConstant(builder(), 2.0f, op_builder().getF32Type(), {1, 1});
  mlir::MlirOp scale_b =
      MakeConstant(builder(), 2.0f, op_builder().getF32Type(), {1, 1});

  auto status_or_result = BuildScaledMmV2Shlo(
      lhs, rhs, scale_a, scale_b, /*bias=*/std::nullopt,
      /*out_dtype=*/std::nullopt,
      Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=Unit test fixture input
      at::blas::ScalingType::TensorWise, at::blas::ScalingType::TensorWise,
      at::blas::SwizzleType::NO_SWIZZLE, at::blas::SwizzleType::NO_SWIZZLE);
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result, status_or_result);

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_EQ(result_type.getShape(), mlir::ArrayRef<int64_t>({128, 128}));
}

TEST(ScaledMmV2ValidationTest, PreconditionsAndOutDtypeChecks) {
  const int64_t kBlockWise =
      static_cast<int64_t>(at::blas::ScalingType::BlockWise1x32);
  const int64_t kTensorWise =
      static_cast<int64_t>(at::blas::ScalingType::TensorWise);
  const int64_t kNoSwizzle =
      static_cast<int64_t>(at::blas::SwizzleType::NO_SWIZZLE);
  const int64_t kSwizzle32 =
      static_cast<int64_t>(at::blas::SwizzleType::SWIZZLE_32_4_4);

  at::Tensor lhs = at::empty({128, 128}, at::kFloat8_e4m3fn);
  at::Tensor rhs = at::empty({128, 128}, at::kFloat8_e4m3fn);
  at::Tensor scale_a = at::empty({128, 4}, at::kFloat8_e8m0fnu);
  at::Tensor scale_b = at::empty({4, 128}, at::kFloat8_e8m0fnu);
  at::Tensor swizzled_scale_a = at::empty({512}, at::kFloat8_e8m0fnu);
  at::Tensor swizzled_scale_b = at::empty({512}, at::kFloat8_e8m0fnu);
  at::Tensor out_f32 = at::empty({128, 128}, at::kFloat);

  auto call_v2_out =
      [&](const at::Tensor& a, const at::Tensor& b, at::ArrayRef<at::Tensor> sa,
          at::IntArrayRef ra, at::IntArrayRef swa, at::ArrayRef<at::Tensor> sb,
          at::IntArrayRef rb, at::IntArrayRef swb,
          std::optional<at::Tensor> bias = std::nullopt,
          std::optional<at::ScalarType> out_dtype = at::kBFloat16,
          at::IntArrayRef contraction_dim = {1, 0}) {
        AtenScaledMmV2Out(a, b, sa, ra, swa, sb, rb, swb, bias, out_dtype,
                          contraction_dim, /*use_fast_accum=*/false, out_f32);
      };

  // 1. contraction_dim checks.
  EXPECT_THROW(
      call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle}, {scale_b},
                  {kBlockWise}, {kNoSwizzle}, std::nullopt, at::kBFloat16, {1}),
      c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle},
                           {scale_b}, {kBlockWise}, {kNoSwizzle}, std::nullopt,
                           at::kBFloat16, {0, 1}),
               c10::Error);

  // 2. Matrix rank and dimension checks.
  EXPECT_THROW(call_v2_out(at::empty({128}, at::kFloat8_e4m3fn), rhs, {scale_a},
                           {kBlockWise}, {kNoSwizzle}, {scale_b}, {kBlockWise},
                           {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, at::empty({64, 128}, at::kFloat8_e4m3fn),
                           {scale_a}, {kBlockWise}, {kNoSwizzle}, {scale_b},
                           {kBlockWise}, {kNoSwizzle}),
               c10::Error);

  // 3. List length checks.
  EXPECT_THROW(call_v2_out(lhs, rhs, {}, {kBlockWise}, {kNoSwizzle}, {scale_b},
                           {kBlockWise}, {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle}, {},
                           {kBlockWise}, {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {}, {kNoSwizzle}, {scale_b},
                           {kBlockWise}, {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle},
                           {scale_b}, {}, {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {}, {scale_b},
                           {kBlockWise}, {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle},
                           {scale_b}, {kBlockWise}, {}),
               c10::Error);

  // 4. Unsupported recipe and swizzle enums.
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {99}, {kNoSwizzle}, {scale_b},
                           {kBlockWise}, {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {99}, {scale_b},
                           {kBlockWise}, {kNoSwizzle}),
               c10::Error);

  // 5. ValidateMxFp8Inputs checks: swizzle compatibility, K alignment, shapes.
  EXPECT_THROW(call_v2_out(lhs, rhs, {swizzled_scale_a}, {kTensorWise},
                           {kSwizzle32}, {scale_b}, {kBlockWise}, {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle},
                           {swizzled_scale_b}, {kTensorWise}, {kSwizzle32}),
               c10::Error);
  EXPECT_THROW(call_v2_out(at::empty({128, 48}, at::kFloat8_e4m3fn),
                           at::empty({48, 128}, at::kFloat8_e4m3fn), {scale_a},
                           {kBlockWise}, {kNoSwizzle}, {scale_b}, {kBlockWise},
                           {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(
      call_v2_out(lhs, rhs, {at::empty({4}, at::kFloat8_e8m0fnu)}, {kBlockWise},
                  {kNoSwizzle}, {scale_b}, {kBlockWise}, {kNoSwizzle}),
      c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {at::empty({64, 4}, at::kFloat8_e8m0fnu)},
                           {kBlockWise}, {kNoSwizzle}, {scale_b}, {kBlockWise},
                           {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle},
                           {at::empty({4}, at::kFloat8_e8m0fnu)}, {kBlockWise},
                           {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle},
                           {at::empty({4, 64}, at::kFloat8_e8m0fnu)},
                           {kBlockWise}, {kNoSwizzle}),
               c10::Error);
  EXPECT_THROW(call_v2_out(lhs, rhs, {at::empty({256}, at::kFloat8_e8m0fnu)},
                           {kBlockWise}, {kSwizzle32}, {swizzled_scale_b},
                           {kBlockWise}, {kSwizzle32}),
               c10::Error);
  EXPECT_THROW(
      call_v2_out(lhs, rhs, {swizzled_scale_a}, {kBlockWise}, {kSwizzle32},
                  {at::empty({256}, at::kFloat8_e8m0fnu)}, {kBlockWise},
                  {kSwizzle32}),
      c10::Error);

  // 6. bias and out_dtype checks.
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle},
                           {scale_b}, {kBlockWise}, {kNoSwizzle},
                           at::empty({1, 1, 128}, at::kBFloat16)),
               c10::Error);
  EXPECT_THROW(
      call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle}, {scale_b},
                  {kBlockWise}, {kNoSwizzle}, at::empty({128}, at::kInt)),
      c10::Error);
  EXPECT_THROW(
      call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle}, {scale_b},
                  {kBlockWise}, {kNoSwizzle}, std::nullopt, at::kInt),
      c10::Error);

  // 7. Full happy path of ValidateMxFp8Inputs & ValidateScaledMmV2Inputs
  // (both NO_SWIZZLE and SWIZZLE_32_4_4), terminating at ValidateOutDtype
  // when out_f32 (kFloat) mismatches out_dtype (kBFloat16).
  EXPECT_THROW(call_v2_out(lhs, rhs, {scale_a}, {kBlockWise}, {kNoSwizzle},
                           {scale_b}, {kBlockWise}, {kNoSwizzle},
                           at::empty({128}, at::kBFloat16), at::kBFloat16),
               c10::Error);
  EXPECT_THROW(
      call_v2_out(lhs, rhs, {swizzled_scale_a}, {kBlockWise}, {kSwizzle32},
                  {swizzled_scale_b}, {kBlockWise}, {kSwizzle32},
                  at::empty({128}, at::kFloat), at::kBFloat16),
      c10::Error);
}

}  // namespace
}  // namespace torch_tpu
