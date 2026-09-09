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

#include "csrc/eager/op_dispatcher.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/ones.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

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

  TT_ASSERT_OK_AND_ASSIGN(auto input_ref,
                          DeviceBufferList::CreatePlaceholder(
                              Dimensions{2, 2}, mlir::ElementType::BF16));

  internal::DeferredOpParams params{
      .op_name = OpName::kAdd,
      .op_builder = original_builder,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
      .inputs = {input_ref},
      .output_shapes = {Shape(Dimensions{2, 2}, mlir::ElementType::BF16)},
  };

  TT_ASSERT_OK_AND_ASSIGN(
      auto results,
      internal::CreateDeferredDeviceBufferList(std::move(params)));
  ASSERT_EQ(results.size(), 1);

  auto deferred_op = results[0].deferred_op();
  ASSERT_TRUE(deferred_op != nullptr);
  const auto& wrapped_builder = deferred_op->op_builder();

  TT_ASSERT_OK_AND_ASSIGN(
      auto wrapped_results,
      wrapped_builder(builder, absl::MakeSpan(&input_op, 1)));
  ASSERT_EQ(wrapped_results.size(), 1);

  // The output should have been casted to BF16
  auto res_type = GetTensorTypeOrDie(wrapped_results[0]);
  EXPECT_EQ(res_type.getElementType(), bf16_mlir_type);
}

void AutoDonateInPlaceBuffer(const at::Tensor& out,
                             absl::Span<const at::Tensor> inputs,
                             mlir::ElementType out_dtype,
                             std::optional<at::IntArrayRef> destination_dims,
                             Indices& donated_indices) {
  if (!donated_indices.empty() || !out.defined()) {
    return;
  }
  const at::IntArrayRef dims = destination_dims.value_or(out.sizes());
  const absl::Span<const int64_t> dims_span(dims.data(), dims.size());
  internal::AutoDonateInPlaceBuffers(
      absl::MakeSpan(&out, 1), absl::MakeSpan(&out_dtype, 1),
      absl::MakeSpan(&dims_span, 1), inputs, donated_indices);
}

TEST(AutoDonateInPlaceBuffer, EligibilityAndAliasingTests) {
  constexpr auto f32_type = mlir::ElementType::F32;
  constexpr auto f64_type = mlir::ElementType::F64;

  auto should_donate = [](const at::Tensor& dest,
                          absl::Span<const at::Tensor> inputs,
                          mlir::ElementType out_dtype,
                          std::optional<at::IntArrayRef> destination_dims =
                              std::nullopt) -> bool {
    Indices donated;
    AutoDonateInPlaceBuffer(dest, inputs, out_dtype, destination_dims, donated);
    return !donated.empty();
  };

  at::Tensor t = at::ones({2, 3}, at::kFloat);
  at::Tensor other = at::ones({2, 3}, at::kFloat);

  // In kDeferAndFuse, donation is disabled.
  SetEagerMode(EagerMode::kDeferAndFuse);
  EXPECT_FALSE(should_donate(t, {t}, f32_type));

  // In kDeferNever, eligible tensor donates.
  SetEagerMode(EagerMode::kDeferNever);
  EXPECT_TRUE(should_donate(t, {t}, f32_type));

  // In kDeferNeverAndLaunchBlocking, eligible tensor donates.
  SetEagerMode(EagerMode::kDeferNeverAndLaunchBlocking);
  EXPECT_TRUE(should_donate(t, {t}, f32_type));

  // Reset to kDeferNever for the remaining tests.
  SetEagerMode(EagerMode::kDeferNever);

  // Non-aliasing tensors do not donate.
  EXPECT_FALSE(should_donate(other, {t}, f32_type));

  // Dtype mismatch does not donate.
  EXPECT_FALSE(should_donate(t, {t}, f64_type));

  // Shape mismatch does not donate.
  EXPECT_FALSE(should_donate(t, {t}, f32_type, at::IntArrayRef({3, 2})));
  EXPECT_FALSE(should_donate(t, {t}, f32_type, at::IntArrayRef({6})));

  // Explicit matching output_dims donates.
  EXPECT_TRUE(should_donate(t, {t}, f32_type, at::IntArrayRef({2, 3})));

  // Non-contiguous tensor does not donate.
  at::Tensor t_transposed = t.t();
  EXPECT_FALSE(should_donate(t_transposed, {t_transposed}, f32_type,
                             at::IntArrayRef({3, 2})));

  // Non-zero storage offset does not donate.
  at::Tensor t_sliced = t.slice(/*dim=*/0, /*start=*/1);
  EXPECT_FALSE(
      should_donate(t_sliced, {t_sliced}, f32_type, at::IntArrayRef({1, 3})));

  // Empty tensor does not donate.
  at::Tensor t_empty = at::ones({0}, at::kFloat);
  EXPECT_FALSE(
      should_donate(t_empty, {t_empty}, f32_type, at::IntArrayRef({0})));

  // Restore default mode.
  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AutoDonateInPlaceBuffer, DetectsDonation) {
  constexpr auto f32_type = mlir::ElementType::F32;
  SetEagerMode(EagerMode::kDeferNever);

  at::Tensor a = at::ones({2, 3}, at::kFloat);
  at::Tensor b = at::ones({2, 3}, at::kFloat);

  Indices donated;
  AutoDonateInPlaceBuffer(a, {b, a}, f32_type, std::nullopt, donated);
  EXPECT_EQ(donated, Indices{1});

  // Does not overwrite pre-existing donated indices.
  Indices pre_existing = {0};
  AutoDonateInPlaceBuffer(a, {b, a}, f32_type, std::nullopt, pre_existing);
  EXPECT_EQ(pre_existing, Indices{0});

  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AutoDonateInPlaceBuffers, MultiOutputDonation) {
  constexpr auto f32_type = mlir::ElementType::F32;
  SetEagerMode(EagerMode::kDeferNever);

  at::Tensor in0 = at::ones({4, 5}, at::kFloat);
  at::Tensor in1 = at::ones({2, 3}, at::kFloat);
  at::Tensor out0 = in1;  // Aliases in1.
  at::Tensor out1 = in0;  // Aliases in0.

  const std::array<int64_t, 2> dims0 = {2, 3};
  const std::array<int64_t, 2> dims1 = {4, 5};
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {dims0, dims1};
  const std::array<mlir::ElementType, 2> out_dtypes = {f32_type, f32_type};

  Indices donated;
  internal::AutoDonateInPlaceBuffers({out0, out1}, out_dtypes, out_dims_list,
                                     {in0, in1}, donated);
  // out0 matches in1 (index 1), out1 matches in0 (index 0).
  EXPECT_EQ(donated, (Indices{1, 0}));

  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AutoDonateInPlaceBuffers, CannotDonateBufferMultipleTimes) {
  constexpr auto f32_type = mlir::ElementType::F32;
  SetEagerMode(EagerMode::kDeferNever);

  at::Tensor in = at::ones({2, 3}, at::kFloat);
  at::Tensor out0 = in;  // Aliases in.
  at::Tensor out1 = in;  // Also aliases in.

  const std::array<int64_t, 2> dims = {2, 3};
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {dims, dims};
  const std::array<mlir::ElementType, 2> out_dtypes = {f32_type, f32_type};

  Indices donated;
  internal::AutoDonateInPlaceBuffers({out0, out1}, out_dtypes, out_dims_list,
                                     {in}, donated);
  // 'in' can only be donated to at most ONE output (out0). It must NOT be
  // donated a second time to out1.
  EXPECT_EQ(donated, Indices{0});

  // Similarly, if two inputs alias each other, the underlying buffer cannot be
  // donated multiple times.
  at::Tensor in_alias = in;  // Shares storage with 'in'.
  Indices donated_aliased_inputs;
  internal::AutoDonateInPlaceBuffers({out0, out1}, out_dtypes, out_dims_list,
                                     {in, in_alias}, donated_aliased_inputs);
  EXPECT_EQ(donated_aliased_inputs, Indices{0});

  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AutoDonateInPlaceBuffers,
     DoesNotDonateSharedBufferWithMultipleLiveDataPtrs) {
  constexpr auto f32_type = mlir::ElementType::F32;
  SetEagerMode(EagerMode::kDeferNever);
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));

  const std::array<int64_t, 2> dims = {2, 3};
  const std::array<absl::Span<const int64_t>, 1> out_dims_list = {dims};
  const std::array<mlir::ElementType, 1> out_dtypes = {f32_type};

  // Case 1: Single tensor created from dummy_buffer (live_data_ptrs == 1).
  at::Tensor a = MakeTensor(dummy_buffer);
  EXPECT_EQ(dummy_buffer.device_buffer_list()->live_data_ptrs(), 1);

  Indices donated;
  internal::AutoDonateInPlaceBuffers({a}, out_dtypes, out_dims_list, {a},
                                     donated);
  EXPECT_EQ(donated, Indices{0});

  // Case 2: Second tensor shares the same DeviceBufferRef (live_data_ptrs ==
  // 2), simulating shallow-copy clone semantics. Buffer donation must be
  // blocked.
  {
    at::Tensor b = MakeTensor(dummy_buffer);
    EXPECT_EQ(dummy_buffer.device_buffer_list()->live_data_ptrs(), 2);
    // b and a share the buffer, but have separate storages (not is_alias_of).
    EXPECT_FALSE(b.is_alias_of(a));

    Indices donated_shared;
    internal::AutoDonateInPlaceBuffers({b}, out_dtypes, out_dims_list, {b},
                                       donated_shared);
    EXPECT_TRUE(donated_shared.empty());

    // In-place on 'a' must also not donate while 'b' is alive.
    Indices donated_a_shared;
    internal::AutoDonateInPlaceBuffers({a}, out_dtypes, out_dims_list, {a},
                                       donated_a_shared);
    EXPECT_TRUE(donated_a_shared.empty());

    // In-place on 'b' with 'a' also passed as an input operand.
    Indices donated_both_inputs;
    internal::AutoDonateInPlaceBuffers({b}, out_dtypes, out_dims_list, {b, a},
                                       donated_both_inputs);
    EXPECT_TRUE(donated_both_inputs.empty());
  }

  // Case 3: Once 'b' is destroyed, live_data_ptrs returns to 1, and donation
  // is eligible again.
  EXPECT_EQ(dummy_buffer.device_buffer_list()->live_data_ptrs(), 1);
  Indices donated_after_b_destroyed;
  internal::AutoDonateInPlaceBuffers({a}, out_dtypes, out_dims_list, {a},
                                     donated_after_b_destroyed);
  EXPECT_EQ(donated_after_b_destroyed, Indices{0});

  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AssignBufferToOutput, ErrorOnUndefinedOutput) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  at::Tensor undefined_output;  // UNINITIALIZED_TENSOR_OK
  EXPECT_FALSE(undefined_output.defined());

  TT_ASSERT_OK_AND_ASSIGN(
      auto dummy_buffer,
      CreateEmptyDeviceBufferRef({2, 3}, mlir::ElementType::F32));

  const absl::Status status =
      internal::AssignBufferToOutput(undefined_output, std::move(dummy_buffer));
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
}

}  // namespace
}  // namespace torch_tpu
