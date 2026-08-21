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

#include "torch_tpu/csrc/eager/op_dispatcher.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/context_states.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/shape.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/device_buffer_utils.h"
#include "torch_tpu/csrc/eager/eager_mode.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/python_context.h"

namespace torch_tpu {
namespace {

TEST(PromoteScalar, Single) {
  at::Scalar s(1.0);
  auto ps = PromoteScalar(s);
  EXPECT_EQ(ps.scalar().toDouble(), 1.0);
}

TEST(PromoteScalar, Optional) {
  std::optional<at::Scalar> os(2.0);
  auto ops = PromoteScalar(os);
  ASSERT_TRUE(ops.has_value());
  EXPECT_EQ(ops->scalar().toDouble(), 2.0);

  std::optional<at::Scalar> empty_os;
  auto empty_ops = PromoteScalar(empty_os);
  EXPECT_FALSE(empty_ops.has_value());
}

TEST(PromoteScalar, Array) {
  std::vector<at::Scalar> vs = {at::Scalar(3.0), at::Scalar(4.0)};
  auto vps = PromoteScalar(vs);
  ASSERT_EQ(vps.size(), 2);

  EXPECT_EQ(vps[0].scalar().toDouble(), 3.0);
  EXPECT_EQ(vps[1].scalar().toDouble(), 4.0);
}

TEST(EncodeParamCacheKey, OptionalPromotedScalar) {
  at::Scalar s(1.0);
  auto ps = PromoteScalar(s);
  std::optional<PromotedScalar> ops = std::move(ps);
  EXPECT_EQ(internal::EncodeParamCacheKey(ops), "s");
  std::optional<PromotedScalar> empty;
  EXPECT_EQ(internal::EncodeParamCacheKey(empty), "");
}

TEST(OpDispatcher, OutputCastingWithoutComputationDtype) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });

  mlir::MLIRContext context;
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir::ModuleBuilder builder(context);

  auto bf16_mlir_type = mlir::getElementType(context, mlir::ElementType::BF16);
  auto tensor_type = mlir::RankedTensorType::get({2, 2}, bf16_mlir_type);
  auto attr = mlir::DenseElementsAttr::get(
      tensor_type, builder.getOpBuilder().getFloatAttr(bf16_mlir_type, 1.0));
  mlir::MlirOp input_op = builder.create<mlir::stablehlo::ConstantOp>(attr);

  // Mock original_builder to return F32 output even though input is BF16
  // and we don't specify computation_dtype.
  auto original_builder = [&](mlir::MlirBuilder& b,
                              absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    auto f32_mlir_type = mlir::getElementType(context, mlir::ElementType::F32);
    auto f32_tensor_type = mlir::RankedTensorType::get({2, 2}, f32_mlir_type);
    auto f32_attr = mlir::DenseElementsAttr::get(
        f32_tensor_type, b.getOpBuilder().getFloatAttr(f32_mlir_type, 2.0));
    mlir::MlirOp res = b.create<mlir::stablehlo::ConstantOp>(f32_attr);
    return DynamicMlirOpResults{res};
  };

  auto input_ref_or = DeviceBufferList::CreatePlaceholder(
      Dimensions{2, 2}, mlir::ElementType::BF16);
  ASSERT_TRUE(input_ref_or.ok());
  auto input_ref = input_ref_or.value();

  internal::DeferredOpParams params{
      .op_name = OpName::kAdd,
      .op_builder = original_builder,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
      .inputs = {input_ref},
      .output_shapes = {Shape(Dimensions{2, 2}, mlir::ElementType::BF16)},
  };

  auto results_or = internal::CreateDeferredDeviceBufferList(std::move(params));
  ASSERT_TRUE(results_or.ok());
  auto results = results_or.value();
  ASSERT_EQ(results.size(), 1);

  auto deferred_op = results[0].deferred_op();
  ASSERT_TRUE(deferred_op != nullptr);
  const auto& wrapped_builder = deferred_op->op_builder();

  auto wrapped_results_or =
      wrapped_builder(builder, absl::MakeSpan(&input_op, 1));
  ASSERT_TRUE(wrapped_results_or.ok());
  auto wrapped_results = wrapped_results_or.value();
  ASSERT_EQ(wrapped_results.size(), 1);

  // The output should have been casted to BF16
  auto res_type = GetTensorTypeOrDie(wrapped_results[0]);
  EXPECT_EQ(res_type.getElementType(), bf16_mlir_type);
}

}  // namespace
}  // namespace torch_tpu
