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
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Optional;
using ::testing::Property;

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

TEST(MakeDeferredOpParams, SetsOutputLayout) {
  const CustomLayout layout = {
      .minor_to_major = {1, 0},
      .tiles = {Indices({8})},
  };
  constexpr mlir::ElementType f32_type = mlir::ElementType::F32;
  const std::array<int64_t, 2> dims = {16, 8};
  DispatchOpOptions<1> options = {
      .out_dtype = f32_type,
      .out_dims = dims,
      .out_layout = layout,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  const internal::DeferredOpParams params =
      internal::MakeDeferredOpParams<1, 1>(
          OpName::kAdd, [](mlir::MlirOp input) { return input; },
          /*inputs_vec=*/{}, options);

  EXPECT_THAT(params.output_shapes,
              ElementsAre(Property(&Shape::layout, Optional(layout))));
}

TEST(MakeDeferredOpParams, SetsOutputLayouts) {
  const CustomLayout layout = {
      .minor_to_major = {1, 0},
      .tiles = {Indices({8})},
  };
  constexpr mlir::ElementType f32_type = mlir::ElementType::F32;
  const std::array<int64_t, 2> dims = {16, 8};
  const std::array<mlir::ElementType, 2> out_dtypes = {f32_type, f32_type};
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {dims, dims};
  DispatchOpOptions<2> options = {
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .out_layouts = {layout, std::nullopt},
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  const internal::DeferredOpParams params =
      internal::MakeDeferredOpParams<1, 2>(
          OpName::kFusedAdamw,
          [](mlir::MlirOp input) { return MlirOpResults<2>{input, input}; },
          /*inputs_vec=*/{}, options);

  EXPECT_THAT(params.output_shapes,
              ElementsAre(Property(&Shape::layout, Optional(layout)),
                          Property(&Shape::layout, Eq(std::nullopt))));
}

void AutoDonateInPlaceBuffer(const at::Tensor& out,
                             absl::Span<const at::Tensor> inputs,
                             mlir::ElementType out_dtype,
                             std::optional<at::IntArrayRef> destination_dims,
                             Indices& donated_indices,
                             std::vector<uint8_t>& output_donated) {
  if (!donated_indices.empty() || !out.defined()) {
    return;
  }
  const at::IntArrayRef dims = destination_dims.value_or(out.sizes());
  const absl::Span<const int64_t> dims_span(dims.data(), dims.size());
  internal::AutoDonateInPlaceBuffers(
      absl::MakeSpan(&out, 1), absl::MakeSpan(&out_dtype, 1),
      absl::MakeSpan(&dims_span, 1), inputs, OpSplitMode::kNone,
      donated_indices, output_donated);
}

TEST(AutoDonateInPlaceBuffer, EligibilityAndAliasingTests) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  constexpr auto f32_type = mlir::ElementType::F32;
  constexpr auto f64_type = mlir::ElementType::F64;

  auto should_donate = [](const at::Tensor& dest,
                          absl::Span<const at::Tensor> inputs,
                          mlir::ElementType out_dtype,
                          std::optional<at::IntArrayRef> destination_dims =
                              std::nullopt) -> bool {
    Indices donated;
    std::vector<uint8_t> output_donated = {0};
    AutoDonateInPlaceBuffer(dest, inputs, out_dtype, destination_dims, donated,
                            output_donated);
    return !donated.empty();
  };

  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor t = MakeTensor(std::move(dummy_buffer));
  TT_ASSERT_OK_AND_ASSIGN(dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor other = MakeTensor(std::move(dummy_buffer));
  ASSERT_FALSE(t.is_alias_of(other));

  // In kDeferAndFuse, eligible tensor donates.
  SetEagerMode(EagerMode::kDeferAndFuse);
  EXPECT_TRUE(should_donate(t, {t}, f32_type));

  // In kDeferNever, eligible tensor donates.
  SetEagerMode(EagerMode::kDeferNever);
  EXPECT_TRUE(should_donate(t, {t}, f32_type));

  // In kDeferNeverAndLaunchBlocking, eligible tensor donates.
  SetEagerMode(EagerMode::kDeferNeverAndLaunchBlocking);
  EXPECT_TRUE(should_donate(t, {t}, f32_type));

  // In kInternalDeferAll, eligible tensor donates.
  SetEagerMode(EagerMode::kInternalDeferAll);
  EXPECT_TRUE(should_donate(t, {t}, f32_type));

  // In kInternalCompileFxGraph, automated in-place donation is disabled.
  SetEagerMode(EagerMode::kInternalCompileFxGraph);
  EXPECT_FALSE(should_donate(t, {t}, f32_type));

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

  // Non-contiguous tensor can donate, as long as it is both in and out.
  at::Tensor t_transposed = t;
  t_transposed.t_();  // Work around "missing aten::as_strided" error
  EXPECT_TRUE(should_donate(t_transposed, {t_transposed}, f32_type));

  // Partial slices donate, as long as they are both in and out.
  // Equivalent to at::Tensor t_sliced = t.slice(/*dim=*/0, /*start=*/1);
  at::Tensor t_sliced = t;
  t_sliced.as_strided_(/*size=*/{1, 3}, /*stride=*/{3, 1},
                       /*storage_offset=*/3);
  EXPECT_TRUE(should_donate(t_sliced, {t_sliced}, f32_type));

  // Zero-sized tensor does not donate.
  TT_ASSERT_OK_AND_ASSIGN(dummy_buffer,
                          CreateEmptyDeviceBufferRef({0}, f32_type));
  at::Tensor t_zero_sized = MakeTensor(std::move(dummy_buffer));
  EXPECT_FALSE(should_donate(t_zero_sized, {t_zero_sized}, f32_type));

  // Contiguous FP4 tensor donates, while transposed or sliced FP4 views do not.
  constexpr auto fp4_type = mlir::ElementType::F4E2M1FN;
  TT_ASSERT_OK_AND_ASSIGN(dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 8}, fp4_type));
  at::Tensor t_fp4 = MakeTensor(std::move(dummy_buffer));
  // Output dims are on-device dims (as in DispatchOpOptions), whose packed
  // dimension holds 2 FP4 values per PyTorch element: [2, 4] -> [2, 8].
  EXPECT_TRUE(should_donate(t_fp4, {t_fp4}, fp4_type, at::IntArrayRef({2, 8})));
  // PyTorch sizes as output dims describe a result of another shape.
  EXPECT_FALSE(
      should_donate(t_fp4, {t_fp4}, fp4_type, at::IntArrayRef({2, 4})));

  at::Tensor t_fp4_transposed = t_fp4;
  t_fp4_transposed.t_();
  EXPECT_FALSE(should_donate(t_fp4_transposed, {t_fp4_transposed}, fp4_type,
                             at::IntArrayRef({4, 4})));

  at::Tensor t_fp4_sliced = t_fp4;
  t_fp4_sliced.as_strided_(/*size=*/{1, 4}, /*stride=*/{4, 1},
                           /*storage_offset=*/4);
  EXPECT_FALSE(should_donate(t_fp4_sliced, {t_fp4_sliced}, fp4_type,
                             at::IntArrayRef({1, 8})));

  // Restore default mode.
  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AutoDonateInPlaceBuffer, DetectsDonation) {
  constexpr auto f32_type = mlir::ElementType::F32;
  SetEagerMode(EagerMode::kDeferNever);
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor a = MakeTensor(std::move(dummy_buffer));
  TT_ASSERT_OK_AND_ASSIGN(dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor b = MakeTensor(std::move(dummy_buffer));
  ASSERT_FALSE(a.is_alias_of(b));

  Indices donated;
  std::vector<uint8_t> output_donated = {0};
  AutoDonateInPlaceBuffer(a, {b, a}, f32_type, std::nullopt, donated,
                          output_donated);
  EXPECT_EQ(donated, Indices{1});
  EXPECT_EQ(output_donated, std::vector<uint8_t>({1}));

  // Does not overwrite pre-existing donated indices.
  Indices pre_existing = {0};
  output_donated = {0};
  AutoDonateInPlaceBuffer(a, {b, a}, f32_type, std::nullopt, pre_existing,
                          output_donated);
  EXPECT_EQ(pre_existing, Indices{0});
  EXPECT_EQ(output_donated, std::vector<uint8_t>({0}));

  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AutoDonateInPlaceBuffers, MultiOutputDonation) {
  constexpr auto f32_type = mlir::ElementType::F32;
  SetEagerMode(EagerMode::kDeferNever);
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({4, 5}, f32_type));
  at::Tensor in0 = MakeTensor(std::move(dummy_buffer));
  TT_ASSERT_OK_AND_ASSIGN(dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor in1 = MakeTensor(std::move(dummy_buffer));
  ASSERT_FALSE(in0.is_alias_of(in1));
  at::Tensor out0 = in1;  // Aliases in1.
  at::Tensor out1 = in0;  // Aliases in0.

  const std::array<int64_t, 2> dims0 = {2, 3};
  const std::array<int64_t, 2> dims1 = {4, 5};
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {dims0, dims1};
  const std::array<mlir::ElementType, 2> out_dtypes = {f32_type, f32_type};

  Indices donated;
  std::vector<uint8_t> output_donated = {0, 0};
  internal::AutoDonateInPlaceBuffers({out0, out1}, out_dtypes, out_dims_list,
                                     {in0, in1}, OpSplitMode::kNone, donated,
                                     output_donated);
  // out0 matches in1 (index 1), out1 matches in0 (index 0).
  EXPECT_EQ(donated, (Indices{1, 0}));
  EXPECT_EQ(output_donated, std::vector<uint8_t>({1, 1}));

  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AutoDonateInPlaceBuffers, CannotDonateBufferMultipleTimes) {
  constexpr auto f32_type = mlir::ElementType::F32;
  SetEagerMode(EagerMode::kDeferNever);
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor in = MakeTensor(std::move(dummy_buffer));
  at::Tensor out0 = in;  // Aliases in.
  at::Tensor out1 = in;  // Also aliases in.

  const std::array<int64_t, 2> dims = {2, 3};
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {dims, dims};
  const std::array<mlir::ElementType, 2> out_dtypes = {f32_type, f32_type};

  Indices donated;
  std::vector<uint8_t> output_donated = {0, 0};
  internal::AutoDonateInPlaceBuffers({out0, out1}, out_dtypes, out_dims_list,
                                     {in}, OpSplitMode::kNone, donated,
                                     output_donated);
  // 'in' can only be donated to at most ONE output (out0). It must NOT be
  // donated a second time to out1.
  EXPECT_EQ(donated, Indices{0});
  EXPECT_EQ(output_donated, std::vector<uint8_t>({1, 0}));

  // Similarly, if two inputs alias each other, the underlying buffer cannot be
  // donated multiple times.
  at::Tensor in_alias = in;  // Shares storage with 'in'.
  Indices donated_aliased_inputs;
  output_donated = {0, 0};
  internal::AutoDonateInPlaceBuffers({out0, out1}, out_dtypes, out_dims_list,
                                     {in, in_alias}, OpSplitMode::kNone,
                                     donated_aliased_inputs, output_donated);
  EXPECT_EQ(donated_aliased_inputs, Indices{0});
  EXPECT_EQ(output_donated, std::vector<uint8_t>({1, 0}));

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
  std::vector<uint8_t> output_donated = {0};
  internal::AutoDonateInPlaceBuffers({a}, out_dtypes, out_dims_list, {a},
                                     OpSplitMode::kNone, donated,
                                     output_donated);
  EXPECT_EQ(donated, Indices{0});
  EXPECT_EQ(output_donated, std::vector<uint8_t>({1}));

  // Case 2: Second tensor shares the same DeviceBufferRef (live_data_ptrs ==
  // 2), simulating shallow-copy clone semantics. Buffer donation must be
  // blocked.
  {
    at::Tensor b = MakeTensor(dummy_buffer);
    EXPECT_EQ(dummy_buffer.device_buffer_list()->live_data_ptrs(), 2);
    // b and a share the buffer, but have separate storages (not is_alias_of).
    EXPECT_FALSE(b.is_alias_of(a));

    Indices donated_shared;
    output_donated = {0};
    internal::AutoDonateInPlaceBuffers({b}, out_dtypes, out_dims_list, {b},
                                       OpSplitMode::kNone, donated_shared,
                                       output_donated);
    EXPECT_TRUE(donated_shared.empty());
    EXPECT_EQ(output_donated, std::vector<uint8_t>({0}));

    // In-place on 'a' must also not donate while 'b' is alive.
    Indices donated_a_shared;
    output_donated = {0};
    internal::AutoDonateInPlaceBuffers({a}, out_dtypes, out_dims_list, {a},
                                       OpSplitMode::kNone, donated_a_shared,
                                       output_donated);
    EXPECT_TRUE(donated_a_shared.empty());
    EXPECT_EQ(output_donated, std::vector<uint8_t>({0}));

    // In-place on 'b' with 'a' also passed as an input operand.
    Indices donated_both_inputs;
    output_donated = {0};
    internal::AutoDonateInPlaceBuffers({b}, out_dtypes, out_dims_list, {b, a},
                                       OpSplitMode::kNone, donated_both_inputs,
                                       output_donated);
    EXPECT_TRUE(donated_both_inputs.empty());
    EXPECT_EQ(output_donated, std::vector<uint8_t>({0}));
  }

  // Case 3: Once 'b' is destroyed, live_data_ptrs returns to 1, and donation
  // is eligible again.
  EXPECT_EQ(dummy_buffer.device_buffer_list()->live_data_ptrs(), 1);
  Indices donated_after_b_destroyed;
  output_donated = {0};
  internal::AutoDonateInPlaceBuffers({a}, out_dtypes, out_dims_list, {a},
                                     OpSplitMode::kNone,
                                     donated_after_b_destroyed, output_donated);
  EXPECT_EQ(donated_after_b_destroyed, Indices{0});
  EXPECT_EQ(output_donated, std::vector<uint8_t>({1}));

  SetEagerMode(EagerMode::kDeferAndFuse);
}

TEST(AutoDonateInPlaceBuffers,
     MultiOutputDeviceBufferListTracksPerSlotLiveDataPtrs) {
  constexpr auto f32_type = mlir::ElementType::F32;
  SetEagerMode(EagerMode::kDeferNever);
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  auto dummy_builder = [](mlir::MlirBuilder& b, absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return DynamicMlirOpResults(inputs.begin(), inputs.end());
  };

  TT_ASSERT_OK_AND_ASSIGN(auto in0,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(auto in1,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(auto in2,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));

  std::vector<Shape> out_shapes = {
      Shape(Dimensions{2, 3}, f32_type),
      Shape(Dimensions{2, 3}, f32_type),
      Shape(Dimensions{2, 3}, f32_type),
  };
  TT_ASSERT_OK_AND_ASSIGN(
      auto multi_refs, DeviceBufferList::CreateDeferred(
                           OpName::kFusedAdamw, dummy_builder, {in0, in1, in2},
                           OpParamCacheKeys::Empty(), out_shapes));
  ASSERT_EQ(multi_refs.size(), 3);

  at::Tensor t0 = MakeTensor(multi_refs[0]);
  at::Tensor t1 = MakeTensor(multi_refs[1]);
  at::Tensor t2 = MakeTensor(multi_refs[2]);

  // List-wide live_data_ptrs is 3, while each output slot has live_data_ptrs 1.
  EXPECT_EQ(multi_refs[0].device_buffer_list()->live_data_ptrs(), 3);
  EXPECT_EQ(multi_refs[0].live_data_ptrs(), 1);
  EXPECT_EQ(multi_refs[1].live_data_ptrs(), 1);
  EXPECT_EQ(multi_refs[2].live_data_ptrs(), 1);

  const std::array<int64_t, 2> dims = {2, 3};
  const std::array<absl::Span<const int64_t>, 3> out_dims_list = {dims, dims,
                                                                  dims};
  const std::array<mlir::ElementType, 3> out_dtypes = {f32_type, f32_type,
                                                       f32_type};

  // All 3 slots are eligible for donation on subsequent steps.
  Indices donated_all;
  std::vector<uint8_t> output_donated = {0, 0, 0};
  internal::AutoDonateInPlaceBuffers({t0, t1, t2}, out_dtypes, out_dims_list,
                                     {t0, t1, t2}, OpSplitMode::kNone,
                                     donated_all, output_donated);
  EXPECT_EQ(donated_all, (Indices{0, 1, 2}));
  EXPECT_THAT(output_donated, ElementsAre(1, 1, 1));

  TT_ASSERT_OK_AND_ASSIGN(
      auto step2_refs,
      DeviceBufferList::CreateDeferred(OpName::kFusedAdamw, dummy_builder,
                                       multi_refs, OpParamCacheKeys::Empty(),
                                       out_shapes, OpSplitMode::kNone,
                                       /*donated_indices=*/Indices{0, 1, 2}));
  ASSERT_TRUE(step2_refs[0].deferred_op() != nullptr);
  EXPECT_EQ(step2_refs[0].deferred_op()->donated_indices(),
            (absl::Span<const int64_t>({0, 1, 2})));

  // Shallow clone on slot 1 increments only slot 1's count to 2.
  {
    at::Tensor t1_clone = MakeTensor(multi_refs[1]);
    EXPECT_EQ(multi_refs[0].device_buffer_list()->live_data_ptrs(), 4);
    EXPECT_EQ(multi_refs[0].live_data_ptrs(), 1);
    EXPECT_EQ(multi_refs[1].live_data_ptrs(), 2);
    EXPECT_EQ(multi_refs[2].live_data_ptrs(), 1);

    // Slots 0 and 2 still donate; slot 1 is blocked because it has a live
    // clone.
    Indices donated_with_clone;
    output_donated = {0, 0, 0};
    internal::AutoDonateInPlaceBuffers({t0, t1, t2}, out_dtypes, out_dims_list,
                                       {t0, t1, t2}, OpSplitMode::kNone,
                                       donated_with_clone, output_donated);
    EXPECT_EQ(donated_with_clone, (Indices{0, 2}));
    EXPECT_EQ(output_donated, std::vector<uint8_t>({1, 0, 1}));

    TT_ASSERT_OK_AND_ASSIGN(
        auto step3_refs,
        DeviceBufferList::CreateDeferred(OpName::kFusedAdamw, dummy_builder,
                                         multi_refs, OpParamCacheKeys::Empty(),
                                         out_shapes, OpSplitMode::kNone,
                                         /*donated_indices=*/Indices{0, 1, 2}));
    ASSERT_TRUE(step3_refs[0].deferred_op() != nullptr);
    EXPECT_EQ(step3_refs[0].deferred_op()->donated_indices(),
              (absl::Span<const int64_t>({0, 2})));
  }

  // Once t1_clone is destroyed, slot 1 returns to live_data_ptrs == 1.
  EXPECT_EQ(multi_refs[1].live_data_ptrs(), 1);

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
      internal::AssignBufferToOutput(undefined_output, std::move(dummy_buffer),
                                     /*enable_autodonation=*/false);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::kInvalidArgument);
}

TEST(AssignBufferToOutput, PackedDimOfPackingFactorKeepsTrailingSizeOne) {
  constexpr auto fp4_type = mlir::ElementType::F4E2M1FN;
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  // A `float4_e2m1fn_x2` tensor of shape [2, 1] holds a device buffer of shape
  // [2, 2] (MakeTensor), and its TPU layout doubles the trailing dim back to 2
  // (ResolveTpuLayout).
  TT_ASSERT_OK_AND_ASSIGN(auto output_buf,
                          CreateEmptyDeviceBufferRef({2, 2}, fp4_type));
  at::Tensor output = MakeTensor(std::move(output_buf));
  EXPECT_THAT(output.sizes(), ElementsAre(2, 1));
  TT_ASSERT_OK_AND_ASSIGN(const TpuLayout layout, ResolveTpuLayout(output));
  EXPECT_THAT(layout.sizes, ElementsAre(2, 2));

  // Assigning a [2, 2] result keeps the output's [2, 1] shape, so the output
  // is not resized, and the output now holds the [2, 2] result buffer.
  TT_ASSERT_OK_AND_ASSIGN(auto result_buf,
                          CreateEmptyDeviceBufferRef({2, 2}, fp4_type));
  ASSERT_TRUE(internal::AssignBufferToOutput(output, result_buf,
                                             /*enable_autodonation=*/false)
                  .ok());
  EXPECT_THAT(output.sizes(), ElementsAre(2, 1));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef assigned, GetBaseBuffer(output));
  EXPECT_EQ(assigned, result_buf);
}

// Returns a deferred buffer computed by a (non-constant) op. Unlike the
// `torch.empty()` buffers of CreateEmptyDeviceBufferRef, which are deferred
// constants, it may be donated to a view-inversion write-back.
absl::StatusOr<DeviceBufferRef> CreateComputedDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type) {
  Shape shape(dimensions, element_type);
  internal::DeferredOpParams params{
      .op_name = OpName::kAdd,
      .op_builder = [dimensions, element_type](mlir::MlirBuilder& builder,
                                               absl::Span<mlir::MlirOp>)
          -> absl::StatusOr<DynamicMlirOpResults> {
        return DynamicMlirOpResults{
            BuildFillUninitialized(builder, element_type, dimensions)};
      },
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
      .output_shapes = {std::move(shape)},
      .allow_materialization = false,
  };
  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> refs,
      internal::CreateDeferredDeviceBufferList(std::move(params)));
  TT_RET_CHECK(refs.size() == 1, error::kInternal)
      << "expected 1 output, got " << refs.size();
  return std::move(refs[0]);
}

// Returns a tensor of `dtype` sharing the storage of `base`, with the given
// sizes, strides, and storage offset (in elements of `dtype`).
at::Tensor MakeStorageAlias(const at::Tensor& base, at::ScalarType dtype,
                            at::IntArrayRef sizes, at::IntArrayRef strides,
                            int64_t storage_offset) {
  at::Tensor alias(c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(base.storage()), base.key_set(),
      c10::scalarTypeToTypeMeta(dtype)));
  alias.as_strided_(sizes, strides, storage_offset);
  return alias;
}

TEST(AutoDonateInPlaceBuffers,
     PartialViewOutputDonatesBaseBufferWithoutAliasingInput) {
  constexpr auto f32_type = mlir::ElementType::F32;
  constexpr auto fp4_type = mlir::ElementType::F4E2M1FN;
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferNever);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  TT_ASSERT_OK_AND_ASSIGN(auto base_buf,
                          CreateComputedDeviceBufferRef({2, 3}, f32_type));
  at::Tensor base = MakeTensor(base_buf);
  TT_ASSERT_OK_AND_ASSIGN(auto other_buf,
                          CreateEmptyDeviceBufferRef({1, 3}, f32_type));
  at::Tensor other = MakeTensor(std::move(other_buf));

  at::Tensor out_slice = base;
  out_slice.as_strided_(/*size=*/{1, 3}, /*stride=*/{3, 1},
                        /*storage_offset=*/3);

  const std::array<int64_t, 2> slice_dims = {1, 3};
  const std::array<absl::Span<const int64_t>, 1> out_dims_list = {slice_dims};
  const std::array<mlir::ElementType, 1> out_dtypes = {f32_type};

  // Non-aliasing input into a partial output slice donates the output's base
  // buffer without marking the non-aliasing input as donated.
  Indices donated;
  std::vector<uint8_t> output_donated = {0};
  internal::AutoDonateInPlaceBuffers({out_slice}, out_dtypes, out_dims_list,
                                     {other}, OpSplitMode::kNone, donated,
                                     output_donated);
  EXPECT_TRUE(donated.empty());
  EXPECT_THAT(output_donated, ElementsAre(1));

  // Different-shape aliasing input view with a partial output slice marks both
  // output_donated and the aliasing input index so GetBuffers fuses the read.
  at::Tensor in_col_slice = MakeStorageAlias(base, at::kFloat, /*sizes=*/{2, 1},
                                             /*strides=*/{3, 1},
                                             /*storage_offset=*/0);
  ASSERT_TRUE(in_col_slice.is_alias_of(out_slice));
  EXPECT_EQ(base_buf.live_data_ptrs(), 1);
  donated.clear();
  output_donated = {0};
  internal::AutoDonateInPlaceBuffers({out_slice}, out_dtypes, out_dims_list,
                                     {in_col_slice}, OpSplitMode::kNone,
                                     donated, output_donated);
  EXPECT_THAT(donated, ElementsAre(0));
  EXPECT_THAT(output_donated, ElementsAre(1));

  // Shallow clone on the base buffer blocks output-base donation.
  {
    at::Tensor base_clone = MakeTensor(base_buf);
    EXPECT_EQ(base_buf.live_data_ptrs(), 2);
    donated.clear();
    output_donated = {0};
    internal::AutoDonateInPlaceBuffers({out_slice}, out_dtypes, out_dims_list,
                                       {other}, OpSplitMode::kNone, donated,
                                       output_donated);
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }

  // Non-default split_mode blocks partial-output base donation across splits.
  donated.clear();
  output_donated = {0};
  internal::AutoDonateInPlaceBuffers({out_slice}, out_dtypes, out_dims_list,
                                     {other}, OpSplitMode::kSplitAfter, donated,
                                     output_donated);
  EXPECT_TRUE(donated.empty());
  EXPECT_THAT(output_donated, ElementsAre(0));

  // FP4 partial slices do not donate their base buffer.
  TT_ASSERT_OK_AND_ASSIGN(auto fp4_base_buf,
                          CreateEmptyDeviceBufferRef({2, 8}, fp4_type));
  at::Tensor fp4_base = MakeTensor(std::move(fp4_base_buf));
  TT_ASSERT_OK_AND_ASSIGN(auto fp4_other_buf,
                          CreateEmptyDeviceBufferRef({1, 8}, fp4_type));
  at::Tensor fp4_other = MakeTensor(std::move(fp4_other_buf));
  at::Tensor fp4_slice = fp4_base;
  fp4_slice.as_strided_(/*size=*/{1, 4}, /*stride=*/{4, 1},
                        /*storage_offset=*/4);
  const std::array<int64_t, 2> fp4_out_dims = {1, 8};
  const std::array<absl::Span<const int64_t>, 1> fp4_dims_list = {fp4_out_dims};
  const std::array<mlir::ElementType, 1> fp4_dtypes = {fp4_type};
  donated.clear();
  output_donated = {0};
  internal::AutoDonateInPlaceBuffers({fp4_slice}, fp4_dtypes, fp4_dims_list,
                                     {fp4_other}, OpSplitMode::kNone, donated,
                                     output_donated);
  EXPECT_TRUE(donated.empty());
  EXPECT_THAT(output_donated, ElementsAre(0));
}

// Runs AutoDonateInPlaceBuffers for `outputs` written with their own sizes and
// dtypes, and returns {donated input indices, per-output donated flags}.
std::pair<Indices, std::vector<uint8_t>> AutoDonate(
    absl::Span<const at::Tensor> outputs, absl::Span<const at::Tensor> inputs,
    OpSplitMode split_mode = OpSplitMode::kNone) {
  std::vector<mlir::ElementType> out_dtypes;
  // On-device output dims (as in DispatchOpOptions), whose packed dimension
  // holds `TorchPackingFactor` values per PyTorch element.
  std::vector<Dimensions> out_dims;
  for (const at::Tensor& out : outputs) {
    // Undefined outputs are skipped, so their dtype and dims don't matter.
    const mlir::ElementType dtype =
        out.defined() ? ConvertTo<mlir::ElementType>(out.scalar_type())
                            .value_or(mlir::ElementType::F32)
                      : mlir::ElementType::F32;
    out_dtypes.push_back(dtype);
    Dimensions dims = out.defined() ? CopyIntVector(out.sizes()) : Dimensions();
    if (!dims.empty()) {
      dims.back() *= TorchPackingFactor(dtype);
    }
    out_dims.push_back(std::move(dims));
  }
  std::vector<absl::Span<const int64_t>> out_dims_list(out_dims.begin(),
                                                       out_dims.end());
  Indices donated;
  std::vector<uint8_t> output_donated(outputs.size(), 0);
  const bool any_donated = internal::AutoDonateInPlaceBuffers(
      outputs, out_dtypes, out_dims_list, inputs, split_mode, donated,
      output_donated);
  // The result reports whether any buffer was donated, including an output's
  // base donated to its write-back without any input being donated.
  EXPECT_EQ(any_donated, !donated.empty() ||
                             absl::c_any_of(output_donated,
                                            [](uint8_t d) { return d != 0; }));
  return {std::move(donated), std::move(output_donated)};
}

TEST(AutoDonateInPlaceBuffers, ReturnsWhetherAnyBufferWasDonated) {
  constexpr auto f32_type = mlir::ElementType::F32;
  constexpr auto fp4_type = mlir::ElementType::F4E2M1FN;
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferNever);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  TT_ASSERT_OK_AND_ASSIGN(auto base_buf,
                          CreateComputedDeviceBufferRef({2, 3}, f32_type));
  at::Tensor base = MakeTensor(std::move(base_buf));
  TT_ASSERT_OK_AND_ASSIGN(auto other_buf,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor other = MakeTensor(std::move(other_buf));
  at::Tensor row1 = MakeStorageAlias(base, at::kFloat, /*sizes=*/{1, 3},
                                     /*strides=*/{3, 1}, /*storage_offset=*/3);
  const std::array<mlir::ElementType, 1> f32_dtypes = {f32_type};

  {
    SCOPED_TRACE("Only the output's base is donated, to its write-back");
    const std::array<int64_t, 2> dims = {1, 3};
    const std::array<absl::Span<const int64_t>, 1> dims_list = {dims};
    Indices donated;
    std::vector<uint8_t> output_donated = {0};
    EXPECT_TRUE(internal::AutoDonateInPlaceBuffers(
        {row1}, f32_dtypes, dims_list, {other}, OpSplitMode::kNone, donated,
        output_donated));
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(1));
  }
  {
    SCOPED_TRACE("Nothing is donated");
    const std::array<int64_t, 2> dims = {2, 3};
    const std::array<absl::Span<const int64_t>, 1> dims_list = {dims};
    Indices donated;
    std::vector<uint8_t> output_donated = {0};
    EXPECT_FALSE(internal::AutoDonateInPlaceBuffers(
        {base}, f32_dtypes, dims_list, {other}, OpSplitMode::kNone, donated,
        output_donated));
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }

  // FP4 output dims are on-device dims, whose packed dimension holds 2 values
  // per PyTorch element; an odd packed dimension can't be converted.
  TT_ASSERT_OK_AND_ASSIGN(auto fp4_buf,
                          CreateEmptyDeviceBufferRef({2, 8}, fp4_type));
  at::Tensor fp4 = MakeTensor(std::move(fp4_buf));
  const std::array<mlir::ElementType, 1> fp4_dtypes = {fp4_type};
  {
    SCOPED_TRACE("FP4 output with convertible on-device dims");
    const std::array<int64_t, 2> dims = {2, 8};
    const std::array<absl::Span<const int64_t>, 1> dims_list = {dims};
    Indices donated;
    std::vector<uint8_t> output_donated = {0};
    EXPECT_TRUE(internal::AutoDonateInPlaceBuffers({fp4}, fp4_dtypes, dims_list,
                                                   {fp4}, OpSplitMode::kNone,
                                                   donated, output_donated));
    EXPECT_THAT(donated, ElementsAre(0));
    EXPECT_THAT(output_donated, ElementsAre(1));
  }
  {
    SCOPED_TRACE("FP4 output with an odd packed on-device dim");
    const std::array<int64_t, 2> dims = {2, 7};
    const std::array<absl::Span<const int64_t>, 1> dims_list = {dims};
    Indices donated;
    std::vector<uint8_t> output_donated = {0};
    EXPECT_FALSE(internal::AutoDonateInPlaceBuffers(
        {fp4}, fp4_dtypes, dims_list, {fp4}, OpSplitMode::kNone, donated,
        output_donated));
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }
}

// Covers each decision of GetBaseDonation, IsMatchingInputDonor, and the
// donation routes of AutoDonateInPlaceBuffers that other tests don't.
TEST(AutoDonateInPlaceBuffers, BaseDonationDecisions) {
  constexpr auto f32_type = mlir::ElementType::F32;
  constexpr auto fp4_type = mlir::ElementType::F4E2M1FN;
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferNever);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  ScopedPythonContextCapturer capturer(OpName::kAdd);

  TT_ASSERT_OK_AND_ASSIGN(auto base_buf,
                          CreateComputedDeviceBufferRef({2, 3}, f32_type));
  at::Tensor base = MakeTensor(base_buf);
  TT_ASSERT_OK_AND_ASSIGN(auto other_buf,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor other = MakeTensor(std::move(other_buf));
  at::Tensor row0 = MakeStorageAlias(base, at::kFloat, {1, 3}, {3, 1}, 0);
  at::Tensor row1 = MakeStorageAlias(base, at::kFloat, {1, 3}, {3, 1}, 3);
  at::Tensor col0 = MakeStorageAlias(base, at::kFloat, {2, 1}, {3, 1}, 0);

  {
    SCOPED_TRACE("Case 4: a full overwrite with no aliasing input");
    auto [donated, output_donated] = AutoDonate({base}, {other});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }
  {
    SCOPED_TRACE("Undefined outputs are skipped and undefined inputs ignored");
    at::Tensor undefined;  // UNINITIALIZED_TENSOR_OK
    auto [donated, output_donated] =
        AutoDonate({undefined, base}, {undefined, base});
    EXPECT_THAT(donated, ElementsAre(1));
    EXPECT_THAT(output_donated, ElementsAre(0, 1));
  }
  {
    SCOPED_TRACE("Matching input of another dtype: full overwrite");
    at::Tensor base_i32 = MakeStorageAlias(base, at::kInt, {2, 3}, {3, 1}, 0);
    auto [donated, output_donated] = AutoDonate({base}, {base_i32});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }
  {
    SCOPED_TRACE("Matching input of another dtype: partial overwrite");
    // The input doesn't match, so the base is donated to the write-back, and
    // the aliasing input is marked so its read is fused with it.
    at::Tensor row1_i32 = MakeStorageAlias(base, at::kInt, {1, 3}, {3, 1}, 3);
    auto [donated, output_donated] = AutoDonate({row1}, {row1_i32});
    EXPECT_THAT(donated, ElementsAre(0));
    EXPECT_THAT(output_donated, ElementsAre(1));
  }
  {
    SCOPED_TRACE("Two partial outputs of one base share an aliasing input");
    // Both write-backs read the base, but the input is donated only once.
    auto [donated, output_donated] = AutoDonate({row0, row1}, {col0});
    EXPECT_THAT(donated, ElementsAre(0));
    EXPECT_THAT(output_donated, ElementsAre(1, 1));
  }
  {
    SCOPED_TRACE("Overlapping partial views are not donated to a write-back");
    at::Tensor broadcast = MakeStorageAlias(base, at::kFloat, {4}, {0}, 0);
    TT_ASSERT_OK_AND_ASSIGN(auto src_buf,
                            CreateEmptyDeviceBufferRef({4}, f32_type));
    at::Tensor src = MakeTensor(std::move(src_buf));
    auto [donated, output_donated] = AutoDonate({broadcast}, {src});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }

  {
    SCOPED_TRACE("Partial view of an unmaterialized torch.empty() base");
    // The base is a deferred constant, which DeferNever keeps unmaterialized;
    // it's only donated through a matching input, never to a write-back.
    TT_ASSERT_OK_AND_ASSIGN(auto empty_buf,
                            CreateEmptyDeviceBufferRef({2, 3}, f32_type));
    at::Tensor empty = MakeTensor(std::move(empty_buf));
    at::Tensor empty_row1 =
        MakeStorageAlias(empty, at::kFloat, {1, 3}, {3, 1}, 3);
    TT_ASSERT_OK_AND_ASSIGN(auto src_buf,
                            CreateEmptyDeviceBufferRef({1, 3}, f32_type));
    at::Tensor src = MakeTensor(std::move(src_buf));
    auto [donated, output_donated] = AutoDonate({empty_row1}, {src});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));

    std::tie(donated, output_donated) = AutoDonate({empty_row1}, {empty_row1});
    EXPECT_THAT(donated, ElementsAre(0));
    EXPECT_THAT(output_donated, ElementsAre(1));
  }

  // A PRED base must be upcast out-of-place for a non-boolean write.
  TT_ASSERT_OK_AND_ASSIGN(auto pred_buf, CreateComputedDeviceBufferRef(
                                             {2, 4}, mlir::ElementType::PRED));
  at::Tensor pred_base = MakeTensor(std::move(pred_buf));
  {
    SCOPED_TRACE("Non-boolean write into a PRED base: full overwrite");
    at::Tensor pred_as_u8 =
        MakeStorageAlias(pred_base, at::kByte, {2, 4}, {4, 1}, 0);
    auto [donated, output_donated] = AutoDonate({pred_as_u8}, {pred_as_u8});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }
  {
    SCOPED_TRACE("Non-boolean write into a PRED base: partial overwrite");
    at::Tensor pred_row_as_u8 =
        MakeStorageAlias(pred_base, at::kByte, {1, 4}, {4, 1}, 4);
    TT_ASSERT_OK_AND_ASSIGN(auto u8_buf, CreateEmptyDeviceBufferRef(
                                             {1, 4}, mlir::ElementType::UI8));
    at::Tensor u8_src = MakeTensor(std::move(u8_buf));
    auto [donated, output_donated] = AutoDonate({pred_row_as_u8}, {u8_src});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }

  // FP4 (`float4_e2m1fn_x2`) buffers hold two values per PyTorch element.
  TT_ASSERT_OK_AND_ASSIGN(auto fp4_buf,
                          CreateComputedDeviceBufferRef({4, 8}, fp4_type));
  at::Tensor fp4_base = MakeTensor(std::move(fp4_buf));
  {
    SCOPED_TRACE("Contiguous FP4 output not covering its whole buffer");
    at::Tensor fp4_row0 =
        MakeStorageAlias(fp4_base, at::kFloat4_e2m1fn_x2, {1, 4}, {4, 1}, 0);
    auto [donated, output_donated] = AutoDonate({fp4_row0}, {fp4_row0});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }
  {
    SCOPED_TRACE("Non-contiguous FP4 input of the output's shape");
    at::Tensor fp4_transposed =
        MakeStorageAlias(fp4_base, at::kFloat4_e2m1fn_x2, {4, 4}, {1, 4}, 0);
    auto [donated, output_donated] = AutoDonate({fp4_base}, {fp4_transposed});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));
  }
  {
    SCOPED_TRACE("Partial non-FP4 view of an FP4 base");
    // Only donated through a matching input, never to a write-back.
    at::Tensor fp4_row_as_u8 =
        MakeStorageAlias(fp4_base, at::kByte, {1, 4}, {4, 1}, 4);
    TT_ASSERT_OK_AND_ASSIGN(auto u8_buf, CreateEmptyDeviceBufferRef(
                                             {1, 4}, mlir::ElementType::UI8));
    at::Tensor u8_src = MakeTensor(std::move(u8_buf));
    auto [donated, output_donated] = AutoDonate({fp4_row_as_u8}, {u8_src});
    EXPECT_TRUE(donated.empty());
    EXPECT_THAT(output_donated, ElementsAre(0));

    std::tie(donated, output_donated) =
        AutoDonate({fp4_row_as_u8}, {fp4_row_as_u8});
    EXPECT_THAT(donated, ElementsAre(0));
    EXPECT_THAT(output_donated, ElementsAre(1));
  }
}

}  // namespace
}  // namespace torch_tpu
