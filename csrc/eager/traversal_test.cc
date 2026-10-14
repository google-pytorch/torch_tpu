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

#include "csrc/eager/traversal.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {
namespace {

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& /*builder*/, absl::Span<mlir::MlirOp> /*inputs*/) {
  return DynamicMlirOpResults{};
}

// An MLIR op builder that negates its single input.
absl::StatusOr<DynamicMlirOpResults> NegBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return DynamicMlirOpResults{
      builder.create<mlir::stablehlo::NegOp>(inputs[0].getValue())};
}

class TraversalTest : public testing::Test {
 protected:
  void SetUp() override {
    CompilationCache::GetInstance().SetAllowCacheMode(false);
    // Ensure each test invocation receives a globally unique CompileOptionsKey
    // to avoid inter-test collisions in the process-global CompilationCache.
    static std::atomic<int> key_counter = 10000;
    compile_options_key_ = CompileOptionsKey(++key_counter);
  }

  void TearDown() override {
    CompilationCache::GetInstance().EvictAll();
    CompilationCache::GetInstance().SetAllowCacheMode(true);
  }

  // Automatically handles Python context for every test.
  ScopedPythonContextCapturer capturer_{OpName::kEmpty};
  // Common shape used for creating deferred buffers.
  Shape shape_{Dimensions{8}, mlir::ElementType::F32};
  // Unique CompileOptionsKey per test invocation.
  CompileOptionsKey compile_options_key_{0};
};

TEST_F(TraversalTest, ReadableString) {
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_a = refs_a[0];

  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_b = refs_b[0];

  TT_ASSERT_OK_AND_ASSIGN(
      auto traversal,
      Traversal::Create({ref_b}, {ref_a.device_buffer_list().get()}));
  traversal->SortByCreationOrder();

  std::string readable =
      traversal->ReadableString(MaterializationReason::kUnknown);

  EXPECT_EQ(readable,
            "# Graph: 1 ops, 1 inputs, reason: unknown\n"
            "%0: f32[8] = input\n"
            "%1: f32[8] = add(%0)\n"
            "return %1\n");
}

TEST_F(TraversalTest, ReadableStringMultiOutput) {
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs, DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder,
                                                  {}, OpParamCacheKeys::Empty(),
                                                  {shape_, shape_}));

  TT_ASSERT_OK_AND_ASSIGN(auto traversal,
                          Traversal::Create({refs[0], refs[1]}));
  traversal->SortByCreationOrder();

  std::string readable =
      traversal->ReadableString(MaterializationReason::kUnknown);

  EXPECT_EQ(readable,
            "# Graph: 1 ops, 0 inputs, reason: unknown\n"
            "%0, %1: f32[8], f32[8] = add()\n"
            "return %0, %1\n");
}

TEST_F(TraversalTest, ReadableStringReasons) {
  TT_ASSERT_OK_AND_ASSIGN(auto refs, DeviceBufferList::CreateDeferred(
                                         OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape_}));
  auto ref = refs[0];

  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create({ref}));

  std::string readable =
      traversal->ReadableString(MaterializationReason::kCpuTransfer);

  EXPECT_EQ(readable,
            "# Graph: 1 ops, 0 inputs, reason: .cpu()\n"
            "%0: f32[8] = add()\n"
            "return %0\n");
}

TEST_F(TraversalTest, ReadableStringComplexGraph) {
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_a = refs_a[0];

  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_b = refs_b[0];

  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_c,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_c = refs_c[0];

  TT_ASSERT_OK_AND_ASSIGN(auto refs_d,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, DummyBuilder, {ref_b, ref_c},
                              OpParamCacheKeys::Empty(), {shape_}));
  auto ref_d = refs_d[0];

  TT_ASSERT_OK_AND_ASSIGN(
      auto traversal,
      Traversal::Create({ref_d}, {ref_a.device_buffer_list().get()}));
  traversal->SortByCreationOrder();

  std::string readable =
      traversal->ReadableString(MaterializationReason::kUnknown);

  EXPECT_EQ(readable,
            "# Graph: 3 ops, 1 inputs, reason: unknown\n"
            "%0: f32[8] = input\n"
            "%1: f32[8] = add(%0)\n"
            "%2: f32[8] = add(%0)\n"
            "%3: f32[8] = add(%1, %2)\n"
            "return %3\n");
}

TEST_F(TraversalTest, TopLevelDonationChangesCacheKey) {
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  DeviceBufferRef ref_a = refs_a[0];
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  DeviceBufferRef ref_b = refs_b[0];
  TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> refs_c,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, DummyBuilder, {ref_a, ref_b},
                              OpParamCacheKeys::Empty(), {shape_}));
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Traversal> traversal,
      Traversal::Create({refs_c[0]}, {ref_a.device_buffer_list().get(),
                                      ref_b.device_buffer_list().get()}));

  const CompileOptionsKey kCompileOptionsKey(0);
  const CompilationCacheKey no_donation = traversal->GetCacheKey(
      kCompileOptionsKey, /*argument_layouts=*/{}, /*donated_inputs=*/{});
  const CompilationCacheKey donate_0 = traversal->GetCacheKey(
      kCompileOptionsKey, /*argument_layouts=*/{}, /*donated_inputs=*/{0});
  const CompilationCacheKey donate_1 = traversal->GetCacheKey(
      kCompileOptionsKey, /*argument_layouts=*/{}, /*donated_inputs=*/{1});
  const CompilationCacheKey donate_0_1 = traversal->GetCacheKey(
      kCompileOptionsKey, /*argument_layouts=*/{}, /*donated_inputs=*/{0, 1});

  EXPECT_NE(no_donation, donate_0);
  EXPECT_NE(donate_0, donate_1);
  EXPECT_EQ(donate_0_1, traversal->GetCacheKey(kCompileOptionsKey,
                                               /*argument_layouts=*/{},
                                               /*donated_inputs=*/{1, 0, 1}));

  EXPECT_EQ(no_donation.CompactFormat(),
            "c111837714da5d00_0d51927151d176ea_0000000000000000");
  EXPECT_EQ(donate_0.CompactFormat(),
            "f696a73c2abe5a40_0d51927151d176ea_0000000000000000");
  EXPECT_EQ(donate_1.CompactFormat(),
            "1d516e396e48000c_0d51927151d176ea_0000000000000000");
  EXPECT_EQ(donate_0_1.CompactFormat(),
            "6edccdc229483441_0d51927151d176ea_0000000000000000");
}

TEST_F(TraversalTest, GetCacheKeyDifferentiatesLayoutOverridePolicy) {
  TT_ASSERT_OK_AND_ASSIGN(auto refs, DeviceBufferList::CreateDeferred(
                                         OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape_}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create({refs[0]}));
  const CompileOptionsKey kCompileOptionsKey(0);
  const CompilationCacheKey key_disallow = traversal->GetCacheKey(
      kCompileOptionsKey, {}, {}, ArgumentLayoutOverridePolicy::kDisallow);
  const CompilationCacheKey key_allow = traversal->GetCacheKey(
      kCompileOptionsKey, {}, {}, ArgumentLayoutOverridePolicy::kAllow);
  EXPECT_NE(key_disallow, key_allow);
}

TEST_F(TraversalTest, ReadableStringWithTraceback) {
  auto traceback = std::make_shared<PythonTraceback>();
  traceback->frames.push_back({"/path/to/user_code.py", "my_function", 42});

  ScopedPythonContextCapturer::SetTracebackForTesting(traceback);

  TT_ASSERT_OK_AND_ASSIGN(auto refs, DeviceBufferList::CreateDeferred(
                                         OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape_}));

  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create({refs[0]}));

  std::string readable =
      traversal->ReadableString(MaterializationReason::kUnknown);
  EXPECT_THAT(readable,
              testing::HasSubstr("# /path/to/user_code.py:42 in my_function"));
}

TEST_F(TraversalTest, CompileAnnotatesArgumentLayouts) {
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_a = refs_a[0];  // NOLINT

  auto identity_builder = [](mlir::MlirBuilder& /*builder*/,
                             absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return DynamicMlirOpResults{inputs[0]};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, identity_builder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_b = refs_b[0];  // NOLINT

  TT_ASSERT_OK_AND_ASSIGN(
      auto traversal,
      Traversal::Create({ref_b}, {ref_a.device_buffer_list().get()}));

  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       compile_options_key_);
  std::string mlir_text;
  TT_ASSERT_OK_AND_ASSIGN(
      const CompiledKernel kernel,
      traversal->Compile(
          std::move(spec), UnboundedKernelPolicy::kDisallow, &mlir_text,
          /*use_stablehlo_bounds=*/false,
          /*argument_layouts=*/{CustomLayout{.minor_to_major = {1, 0}}}));
  kernel.fixed_shape_kernel.wait();
  EXPECT_THAT(mlir_text, testing::HasSubstr("mhlo.layout_mode = \"{1,0}\""));
}

TEST_F(TraversalTest, CompileAnnotatesArgumentLayoutsWithTilingAndCaching) {
  Shape shape_2d(Dimensions{128, 64}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_2d}));
  DeviceBufferRef ref_a = refs_a[0];

  MlirOpBuilder identity_builder = [](mlir::MlirBuilder& /*builder*/,
                                      absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return DynamicMlirOpResults{inputs[0]};
  };

  TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> refs_b,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, std::move(identity_builder),
                              {ref_a}, OpParamCacheKeys::Empty(), {shape_2d}));
  DeviceBufferRef ref_b = refs_b[0];

  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Traversal> tr,
      Traversal::Create({ref_b}, {ref_a.device_buffer_list().get()}));

  CustomLayout tiled_layout{.minor_to_major = {1, 0}, .tiles = {{8}}};
  ref_b.set_layout(tiled_layout);
  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       compile_options_key_);
  std::string mlir_text;
  TT_ASSERT_OK_AND_ASSIGN(
      const CompiledKernel kernel,
      tr->Compile(std::move(spec), UnboundedKernelPolicy::kDisallow, &mlir_text,
                  /*use_stablehlo_bounds=*/false,
                  /*argument_layouts=*/{tiled_layout}));
  kernel.fixed_shape_kernel.wait();
  EXPECT_THAT(
      mlir_text,
      testing::HasSubstr(
          "func.func @main(%arg0: tensor<128x64xf32> {mhlo.layout_mode = "
          "\"{1,0:T(8)}\"}) -> (tensor<128x64xf32> {mhlo.layout_mode = "
          "\"{1,0:T(8)}\"})"));

  CustomLayout different_tile_layout{.minor_to_major = {1, 0}, .tiles = {{16}}};
  CompilationCacheKey key_untiled = tr->GetCacheKey(
      compile_options_key_, {CustomLayout{.minor_to_major = {1, 0}}},
      /*donated_inputs=*/{});
  CompilationCacheKey key_tiled8 = tr->GetCacheKey(
      compile_options_key_, {tiled_layout}, /*donated_inputs=*/{});
  CompilationCacheKey key_tiled16 =
      tr->GetCacheKey(compile_options_key_, {different_tile_layout},
                      /*donated_inputs=*/{});
  EXPECT_NE(key_untiled, key_tiled8);
  EXPECT_NE(key_tiled8, key_tiled16);
}

TEST_F(TraversalTest, ValidateAndReorderArgumentsErrorCases) {
  TT_ASSERT_OK_AND_ASSIGN(
      const DeviceBufferRef ref_a,
      DeviceBufferList::CreatePlaceholder(shape_.dimensions(), shape_.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(
      const DeviceBufferRef ref_b,
      DeviceBufferList::CreatePlaceholder(shape_.dimensions(), shape_.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(const std::vector<DeviceBufferRef> refs_out,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, DummyBuilder, {ref_a, ref_b},
                              OpParamCacheKeys::Empty(), {shape_}));
  const DeviceBufferRef ref_out = refs_out[0];

  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> tr,
                          Traversal::Create({ref_out}));

  // Case 1: Duplicate input buffer should return kInvalidArgument with shape
  // string.
  const absl::Status duplicate_status =
      tr->ValidateAndReorderArguments({ref_a, ref_a});
  EXPECT_EQ(duplicate_status.code(), error::kInvalidArgument);
  EXPECT_THAT(duplicate_status.message(),
              testing::HasSubstr("identified a duplicate input: 8xfloat32"));

  // Case 2: Missing input buffer should return kInvalidArgument with shape
  // string.
  const absl::Status missing_status = tr->ValidateAndReorderArguments({ref_a});
  EXPECT_EQ(missing_status.code(), error::kInvalidArgument);
  EXPECT_THAT(missing_status.message(),
              testing::HasSubstr(
                  "identified an argument that was not provided: 8xfloat32"));

  // Case 3: Deferred input buffer should return kInvalidArgument with shape
  // string.
  const absl::Status deferred_status =
      tr->ValidateAndReorderArguments({ref_a, ref_out});
  EXPECT_EQ(deferred_status.code(), error::kInvalidArgument);
  EXPECT_THAT(
      deferred_status.message(),
      testing::HasSubstr(
          "found a deferred argument, which is not allowed: 8xfloat32"));

  // Case 4: Valid reordered input buffers should succeed.
  EXPECT_TRUE(tr->ValidateAndReorderArguments({ref_b, ref_a}).ok());
}

TEST_F(TraversalTest, CompilePopulatesAndResetsOutCacheKey) {
  Shape shape_2d(Dimensions{128, 64}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_2d}));
  DeviceBufferRef ref_a = refs_a[0];

  MlirOpBuilder identity_builder = [](mlir::MlirBuilder& /*builder*/,
                                      absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return DynamicMlirOpResults{inputs[0]};
  };

  TT_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> refs_b,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, std::move(identity_builder),
                              {ref_a}, OpParamCacheKeys::Empty(), {shape_2d}));
  DeviceBufferRef ref_b = refs_b[0];

  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Traversal> tr,
      Traversal::Create({ref_b}, {ref_a.device_buffer_list().get()}));

  CustomLayout tiled_layout{.minor_to_major = {1, 0}, .tiles = {{8}}};
  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       compile_options_key_);
  std::optional<CompilationCacheKey> out_cache_key;
  TT_ASSERT_OK_AND_ASSIGN(
      const CompiledKernel kernel,
      tr->Compile(spec, UnboundedKernelPolicy::kDisallow,
                  /*out_mlir_text=*/nullptr,
                  /*use_stablehlo_bounds=*/false,
                  /*argument_layouts=*/{tiled_layout},
                  /*donated_inputs=*/{},
                  /*module_name=*/std::nullopt, &out_cache_key));
  kernel.fixed_shape_kernel.wait();
  ASSERT_TRUE(out_cache_key.has_value());
  EXPECT_EQ(*out_cache_key,
            tr->GetCacheKey(compile_options_key_, {tiled_layout},
                            /*donated_inputs=*/{},
                            ArgumentLayoutOverridePolicy::kAllow));

  // Verify that an early validation failure resets out_cache_key to nullopt.
  EXPECT_FALSE(tr->Compile(spec, UnboundedKernelPolicy::kDisallow,
                           /*out_mlir_text=*/nullptr,
                           /*use_stablehlo_bounds=*/false,
                           /*argument_layouts=*/{tiled_layout, tiled_layout},
                           /*donated_inputs=*/{},
                           /*module_name=*/std::nullopt, &out_cache_key)
                   .ok());
  EXPECT_FALSE(out_cache_key.has_value());
}

TEST_F(TraversalTest, DifferentDeviceAssignmentChangesCacheKey) {
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape_},
          OpSplitMode::kNone, /*donated_indices=*/{},
          /*device_assignment=*/{0, 1}));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs_b,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape_},
          OpSplitMode::kNone, /*donated_indices=*/{},
          /*device_assignment=*/{2, 3}));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs_c,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape_},
          OpSplitMode::kNone, /*donated_indices=*/{},
          /*device_assignment=*/{0, 1, 2, 3}));

  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> traversal_a,
                          Traversal::Create({refs_a[0]}));
  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> traversal_b,
                          Traversal::Create({refs_b[0]}));
  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> traversal_c,
                          Traversal::Create({refs_c[0]}));

  const CompileOptionsKey compile_options_key(12345);
  const CompilationCacheKey key_a =
      traversal_a->GetCacheKey(compile_options_key, {}, {});
  const CompilationCacheKey key_b =
      traversal_b->GetCacheKey(compile_options_key, {}, {});
  const CompilationCacheKey key_c =
      traversal_c->GetCacheKey(compile_options_key, {}, {});

  EXPECT_NE(key_a, key_b);
  EXPECT_NE(key_a, key_c);
  EXPECT_NE(key_b, key_c);
}

TEST_F(TraversalTest, ApplyDeviceAssignmentToCompileSpecUpdatesSpec) {
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape_},
          OpSplitMode::kNone, /*donated_indices=*/{},
          /*device_assignment=*/{2, 3}));

  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> traversal,
                          Traversal::Create({refs[0]}));

  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       CompileOptionsKey(12345));
  spec.xla_compile_options->executable_build_options.mutable_debug_options();
  const CompileOptionsKey initial_key = spec.compile_options_key;

  EXPECT_FALSE(spec.xla_compile_options->executable_build_options
                   .has_device_assignment());

  EXPECT_TRUE(traversal->ApplyDeviceAssignmentToCompileSpec(spec).ok());

  const auto& exec_options = spec.xla_compile_options->executable_build_options;
  EXPECT_TRUE(exec_options.has_device_assignment());
  EXPECT_EQ(exec_options.num_replicas(), 1);
  EXPECT_EQ(exec_options.num_partitions(), 2);
  const auto& da = exec_options.device_assignment();
  EXPECT_EQ(da(0, 0), 2);
  EXPECT_EQ(da(0, 1), 3);
  EXPECT_NE(spec.compile_options_key, initial_key);
}

TEST_F(TraversalTest, PreCompileAndCompileCacheKeyMatch) {
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape_},
          OpSplitMode::kNone, /*donated_indices=*/{},
          /*device_assignment=*/{2, 3}));

  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> traversal,
                          Traversal::Create({refs[0]}));

  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       CompileOptionsKey(12345));
  spec.xla_compile_options->executable_build_options.mutable_debug_options();
  const CompilationCacheKey key_before_da =
      traversal->GetCacheKey(spec.compile_options_key, {}, {});

  EXPECT_TRUE(traversal->ApplyDeviceAssignmentToCompileSpec(spec).ok());

  const CompilationCacheKey key_after_da =
      traversal->GetCacheKey(spec.compile_options_key, {}, {});

  EXPECT_NE(key_before_da, key_after_da);
}

TEST_F(TraversalTest, OpWithPreExistingDeviceAssignmentParamDoesNotCrash) {
  OpParamCacheKeys op_param_keys = OpParamCacheKeys::Empty();
  ASSERT_TRUE(op_param_keys
                  .SetParam("device_assignment",
                            std::vector<int64_t>{2, 3})  // INT_VEC_OK
                  .ok());

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs_pre_existing,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd, DummyBuilder, {}, std::move(op_param_keys), {shape_},
          OpSplitMode::kNone, /*donated_indices=*/{},
          /*device_assignment=*/{2, 3}));

  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<Traversal> traversal_pre_existing,
      Traversal::Create({refs_pre_existing[0]}));

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs_empty,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape_},
          OpSplitMode::kNone, /*donated_indices=*/{},
          /*device_assignment=*/{2, 3}));

  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> traversal_empty,
                          Traversal::Create({refs_empty[0]}));

  const CompileOptionsKey kCompileOptionsKey(12345);
  const CompilationCacheKey key_pre_existing =
      traversal_pre_existing->GetCacheKey(kCompileOptionsKey, {}, {});
  const CompilationCacheKey key_empty =
      traversal_empty->GetCacheKey(kCompileOptionsKey, {}, {});

  EXPECT_EQ(key_pre_existing, key_empty);
}

TEST_F(TraversalTest,
       BuildMlirModuleRegistersFuncInlinerExtensionAndInlinesHelpers) {
  TT_ASSERT_OK_AND_ASSIGN(
      const DeviceBufferRef ref_a,
      DeviceBufferList::CreatePlaceholder(shape_.dimensions(), shape_.dtype()));

  auto call_helper_builder = [](mlir::MlirBuilder& builder,
                                absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    mlir::ModuleOp module_op = GetModuleOp(builder);
    mlir::OpBuilder module_builder(module_op.getBodyRegion());
    const mlir::Type input_type = inputs[0].getValue().getType();
    auto helper_func = mlir::func::FuncOp::create(
        module_builder, builder.getLoc(), "private_helper",
        mlir::FunctionType::get(&builder.getContext(), {input_type},
                                {input_type}));
    helper_func.setPrivate();
    mlir::Block* entry_block = helper_func.addEntryBlock();
    mlir::OpBuilder helper_builder = mlir::OpBuilder::atBlockBegin(entry_block);
    auto negated = mlir::stablehlo::NegOp::create(
        helper_builder, builder.getLoc(), entry_block->getArgument(0));
    mlir::func::ReturnOp::create(helper_builder, builder.getLoc(),
                                 mlir::ValueRange{negated.getResult()});

    llvm::SmallVector<mlir::MlirOp> call_results =
        mlir::func::Call(builder, helper_func, inputs);
    return DynamicMlirOpResults(call_results.begin(), call_results.end());
  };

  TT_ASSERT_OK_AND_ASSIGN(const std::vector<DeviceBufferRef> refs_b,
                          DeviceBufferList::CreateDeferred(
                              OpName::kNeg, std::move(call_helper_builder),
                              {ref_a}, OpParamCacheKeys::Empty(), {shape_}));
  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> traversal,
                          Traversal::Create({refs_b[0]}));

  // Construct an MLIRContext with only StableHLO dialects registered (matching
  // callers like GetTraversalOutputDimensions in dynamism_utils_test that do
  // not call MakeMlirContext). BuildMlirModule must register the func inliner
  // extension itself rather than aborting inside mlir::createInlinerPass().
  mlir::MLIRContext mlir_context;
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir_context.appendDialectRegistry(registry);
  mlir_context.loadAllAvailableDialects();

  TT_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          traversal->BuildMlirModule(mlir_context));
  EXPECT_EQ(module->lookupSymbol<mlir::func::FuncOp>("private_helper"),
            nullptr);
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main_func, nullptr);
  EXPECT_TRUE(main_func.getOps<mlir::func::CallOp>().empty());
  EXPECT_FALSE(main_func.getOps<mlir::stablehlo::NegOp>().empty());
}

TEST_F(TraversalTest, NameScopesAttachToMlirLocationsWithoutChangingCacheKey) {
  TT_ASSERT_OK_AND_ASSIGN(
      const DeviceBufferRef input_ref,
      DeviceBufferList::CreatePlaceholder(shape_.dimensions(), shape_.dtype()));

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> unscoped_refs,
      DeviceBufferList::CreateDeferred(OpName::kNeg, NegBuilder, {input_ref},
                                       OpParamCacheKeys::Empty(), {shape_}));
  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> unscoped_traversal,
                          Traversal::Create({unscoped_refs[0]}));

  std::vector<DeviceBufferRef> layer0_refs;
  {
    const NameScopeHierarchyGuard layer0_scope({"layers.0[Block]"});
    TT_ASSERT_OK_AND_ASSIGN(
        layer0_refs,
        DeviceBufferList::CreateDeferred(OpName::kNeg, NegBuilder, {input_ref},
                                         OpParamCacheKeys::Empty(), {shape_}));
  }
  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> layer0_traversal,
                          Traversal::Create({layer0_refs[0]}));

  std::vector<DeviceBufferRef> layer1_refs;
  {
    const NameScopeHierarchyGuard layer1_scope({"layers.1[Block]"});
    TT_ASSERT_OK_AND_ASSIGN(
        layer1_refs,
        DeviceBufferList::CreateDeferred(OpName::kNeg, NegBuilder, {input_ref},
                                         OpParamCacheKeys::Empty(), {shape_}));
  }
  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> layer1_traversal,
                          Traversal::Create({layer1_refs[0]}));

  const CompileOptionsKey kCompileOptionsKey(0);
  const CompilationCacheKey unscoped_key =
      unscoped_traversal->GetCacheKey(kCompileOptionsKey, {}, {});
  const CompilationCacheKey layer0_key =
      layer0_traversal->GetCacheKey(kCompileOptionsKey, {}, {});
  const CompilationCacheKey layer1_key =
      layer1_traversal->GetCacheKey(kCompileOptionsKey, {}, {});

  EXPECT_EQ(unscoped_key, layer0_key);
  EXPECT_EQ(layer0_key, layer1_key);

  mlir::MLIRContext mlir_context;
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir_context.appendDialectRegistry(registry);
  mlir_context.loadAllAvailableDialects();

  TT_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> layer0_module,
                          layer0_traversal->BuildMlirModule(mlir_context));
  const std::string mlir_text =
      DebugString(layer0_module.get(), DebugStringOptions::kEnableDebugInfo);
  EXPECT_THAT(mlir_text,
              testing::HasSubstr("loc(\"layers.0[Block]/empty/empty\""));
}

TEST_F(TraversalTest, CompileAnnotatesRaggedDotConnected3DLayouts) {
  Shape shape_rhs(Dimensions{128, 1440, 2880}, mlir::ElementType::BF16);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef ref_rhs,
      DeviceBufferList::CreatePlaceholder(Dimensions{128, 1440, 2880},
                                          mlir::ElementType::BF16));
  ref_rhs.set_layout(CustomLayout{.minor_to_major = {2, 1, 0}});
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef ref_exp_avg,
      DeviceBufferList::CreatePlaceholder(Dimensions{128, 1440, 2880},
                                          mlir::ElementType::BF16));

  MlirOpBuilder update_builder = [](mlir::MlirBuilder& builder,
                                    absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    mlir::OpBuilder& op_builder = builder.getOpBuilder();
    mlir::Location loc = builder.getLoc();
    mlir::Value updated_rhs =
        mlir::stablehlo::AddOp::create(op_builder, loc, inputs[0].getValue(),
                                       inputs[1].getValue())
            .getResult();
    return DynamicMlirOpResults{mlir::MlirOp(builder, updated_rhs)};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_out,
      DeviceBufferList::CreateDeferred(
          OpName::kGroupedMm, std::move(update_builder), {ref_rhs, ref_exp_avg},
          OpParamCacheKeys::Empty(), {shape_rhs}));

  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Traversal> tr,
                          Traversal::Create(refs_out));
  const auto reorder_status =
      tr->ValidateAndReorderArguments({ref_rhs, ref_exp_avg});
  ASSERT_TRUE(reorder_status.ok()) << reorder_status;

  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       compile_options_key_);
  std::string mlir_text;
  const std::vector<CustomLayout> default_arg_layouts = {
      CustomLayout{.minor_to_major = {2, 1, 0}},
      CustomLayout{.minor_to_major = {0, 2, 1}, .tiles = {{8, 128}, {2, 1}}},
  };
  TT_ASSERT_OK_AND_ASSIGN(
      const CompiledKernel kernel,
      tr->Compile(spec, UnboundedKernelPolicy::kDisallow, &mlir_text,
                  /*use_stablehlo_bounds=*/false, default_arg_layouts));
  kernel.fixed_shape_kernel.wait();
  EXPECT_THAT(
      mlir_text,
      testing::AllOf(testing::HasSubstr("%arg0: tensor<128x1440x2880xbf16> "
                                        "{mhlo.layout_mode = \"{2,1,0}\"}"),
                     testing::HasSubstr("%arg1: tensor<128x1440x2880xbf16> "
                                        "{mhlo.layout_mode = "
                                        "\"{0,2,1:T(8,128)(2,1)}\"}")));
}

TEST_F(TraversalTest,
       CompileAnnotatesRaggedDot3DOperandsAndOutputsWithoutPresetLayouts) {
  Shape shape_lhs(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_rhs_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);
  Shape shape_group_sizes(Dimensions{8}, mlir::ElementType::I32);
  Shape shape_bias_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);
  Shape shape_dot_out(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_out_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);

  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref_lhs,
                          DeviceBufferList::CreatePlaceholder(
                              shape_lhs.dimensions(), shape_lhs.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref_rhs_3d,
                          DeviceBufferList::CreatePlaceholder(
                              shape_rhs_3d.dimensions(), shape_rhs_3d.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef ref_group_sizes,
      DeviceBufferList::CreatePlaceholder(shape_group_sizes.dimensions(),
                                          shape_group_sizes.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef ref_bias_3d,
      DeviceBufferList::CreatePlaceholder(shape_bias_3d.dimensions(),
                                          shape_bias_3d.dtype()));

  MlirOpBuilder ragged_dot_builder = [](mlir::MlirBuilder& builder,
                                        absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    mlir::MlirOp rhs_t = mlir::stablehlo::Transpose(inputs[1], {0, 2, 1});

    const auto dimension_numbers =
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(
            &builder.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{1},
            /*rhsContractingDimensions=*/{1},
            /*lhsRaggedDimensions=*/{0},
            /*rhsGroupDimensions=*/{0});
    auto precision_attr = mlir::chlo::PrecisionAttr::get(
        &builder.getContext(),
        mlir::chlo::Precision::DEFAULT);  // EXPLICIT_PRECISION_OK=test IR
                                          // construction
    auto precision_config = mlir::ArrayAttr::get(
        &builder.getContext(), {precision_attr, precision_attr});
    const mlir::Type out_type = mlir::makeTensorType(
        builder.getContext(), {128, 128}, mlir::ElementType::BF16);
    mlir::MlirOp ragged_dot =
        mlir::chlo::RaggedDot(out_type, inputs[0], rhs_t, inputs[2],
                              dimension_numbers, precision_config);

    mlir::MlirOp add_3d = mlir::stablehlo::Add(inputs[1], inputs[3]);

    return DynamicMlirOpResults{ragged_dot, add_3d};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_out,
      DeviceBufferList::CreateDeferred(
          OpName::kRaggedDot, std::move(ragged_dot_builder),
          {ref_lhs, ref_rhs_3d, ref_group_sizes, ref_bias_3d},
          OpParamCacheKeys::Empty(), {shape_dot_out, shape_out_3d}));

  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Traversal> tr,
                          Traversal::Create(refs_out));
  const auto ragged_reorder_status = tr->ValidateAndReorderArguments(
      {ref_lhs, ref_rhs_3d, ref_group_sizes, ref_bias_3d});
  ASSERT_TRUE(ragged_reorder_status.ok()) << ragged_reorder_status;

  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       compile_options_key_);
  std::string mlir_text;
  TT_ASSERT_OK_AND_ASSIGN(
      const CompiledKernel kernel,
      tr->Compile(spec, UnboundedKernelPolicy::kDisallow, &mlir_text,
                  /*use_stablehlo_bounds=*/false, /*argument_layouts=*/{}));
  kernel.fixed_shape_kernel.wait();
  // %arg0 (2D lhs) must NOT be seeded with any layout annotation.
  EXPECT_THAT(mlir_text, testing::Not(testing::HasSubstr(
                             "%arg0: tensor<128x128xbf16> {mhlo.layout_mode")));
  // %arg1 (pre-transpose [G,N,K]) must have {1,2,0} pulled back through
  // {0,2,1}.
  EXPECT_THAT(
      mlir_text,
      testing::AllOf(
          testing::HasSubstr(
              "%arg1: tensor<8x128x128xbf16> {mhlo.layout_mode = \"{1,2,0}\"}"),
          testing::HasSubstr(
              "%arg3: tensor<8x128x128xbf16> {mhlo.layout_mode = \"{1,2,0}\"}"),
          testing::HasSubstr(
              "tensor<8x128x128xbf16> {mhlo.layout_mode = \"{1,2,0}\"}")));
  if (kernel.fixed_shape_kernel.get().ok()) {
    const auto& exec = *kernel.fixed_shape_kernel.get();
    EXPECT_FALSE(exec->output_shapes()[0].layout().has_value());
    ASSERT_TRUE(exec->output_shapes()[1].layout().has_value());
    EXPECT_EQ(exec->output_shapes()[1].layout()->minor_to_major,
              Indices({1, 2, 0}));
  }
}

TEST_F(TraversalTest, RaggedDotLayoutsPreservedOnCacheHit) {
  CompilationCache::GetInstance().SetAllowCacheMode(true);

  Shape shape_lhs(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_rhs_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);
  Shape shape_group_sizes(Dimensions{8}, mlir::ElementType::I32);
  Shape shape_bias_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);
  Shape shape_dot_out(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_out_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);

  auto make_traversal = [&]() -> absl::StatusOr<std::unique_ptr<Traversal>> {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef ref_lhs,
                        DeviceBufferList::CreatePlaceholder(
                            shape_lhs.dimensions(), shape_lhs.dtype()));
    TT_ASSIGN_OR_RETURN(DeviceBufferRef ref_rhs_3d,
                        DeviceBufferList::CreatePlaceholder(
                            shape_rhs_3d.dimensions(), shape_rhs_3d.dtype()));
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef ref_group_sizes,
        DeviceBufferList::CreatePlaceholder(shape_group_sizes.dimensions(),
                                            shape_group_sizes.dtype()));
    TT_ASSIGN_OR_RETURN(DeviceBufferRef ref_bias_3d,
                        DeviceBufferList::CreatePlaceholder(
                            shape_bias_3d.dimensions(), shape_bias_3d.dtype()));

    MlirOpBuilder ragged_dot_builder = [](mlir::MlirBuilder& builder,
                                          absl::Span<mlir::MlirOp> inputs)
        -> absl::StatusOr<DynamicMlirOpResults> {
      mlir::MlirOp rhs_t = mlir::stablehlo::Transpose(inputs[1], {0, 2, 1});

      const auto dimension_numbers =
          mlir::chlo::RaggedDotDimensionNumbersAttr::get(
              &builder.getContext(), /*lhsBatchingDimensions=*/{},
              /*rhsBatchingDimensions=*/{},
              /*lhsContractingDimensions=*/{1},
              /*rhsContractingDimensions=*/{1},
              /*lhsRaggedDimensions=*/{0},
              /*rhsGroupDimensions=*/{0});
      auto precision_attr = mlir::chlo::PrecisionAttr::get(
          &builder.getContext(),
          mlir::chlo::Precision::DEFAULT);  // EXPLICIT_PRECISION_OK=test IR
                                            // construction
      auto precision_config = mlir::ArrayAttr::get(
          &builder.getContext(), {precision_attr, precision_attr});
      const mlir::Type out_type = mlir::makeTensorType(
          builder.getContext(), {128, 128}, mlir::ElementType::BF16);
      mlir::MlirOp ragged_dot =
          mlir::chlo::RaggedDot(out_type, inputs[0], rhs_t, inputs[2],
                                dimension_numbers, precision_config);

      mlir::MlirOp add_3d = mlir::stablehlo::Add(inputs[1], inputs[3]);

      return DynamicMlirOpResults{ragged_dot, add_3d};
    };

    TT_ASSIGN_OR_RETURN(
        std::vector<DeviceBufferRef> refs_out,
        DeviceBufferList::CreateDeferred(
            OpName::kRaggedDot, std::move(ragged_dot_builder),
            {ref_lhs, ref_rhs_3d, ref_group_sizes, ref_bias_3d},
            OpParamCacheKeys::Empty(), {shape_dot_out, shape_out_3d}));

    TT_ASSIGN_OR_RETURN(std::unique_ptr<Traversal> tr,
                        Traversal::Create(refs_out));
    TT_RETURN_IF_ERROR(tr->ValidateAndReorderArguments(
        {ref_lhs, ref_rhs_3d, ref_group_sizes, ref_bias_3d}));
    return tr;
  };

  CompilationSpec spec1(std::make_unique<xla::CompileOptions>(),
                        compile_options_key_);

  // 1. First compile (cache miss).
  TT_ASSERT_OK_AND_ASSIGN(auto tr1, make_traversal());
  TT_ASSERT_OK_AND_ASSIGN(
      const CompiledKernel kernel1,
      tr1->Compile(spec1, UnboundedKernelPolicy::kDisallow,
                   /*out_mlir_text=*/nullptr,
                   /*use_stablehlo_bounds=*/false, /*argument_layouts=*/{}));
  kernel1.fixed_shape_kernel.wait();

  // 2. Second compile (cache hit).
  CompilationSpec spec2(std::make_unique<xla::CompileOptions>(),
                        compile_options_key_);
  TT_ASSERT_OK_AND_ASSIGN(auto tr2, make_traversal());
  TT_ASSERT_OK_AND_ASSIGN(
      const CompiledKernel kernel2,
      tr2->Compile(spec2, UnboundedKernelPolicy::kDisallow,
                   /*out_mlir_text=*/nullptr,
                   /*use_stablehlo_bounds=*/false, /*argument_layouts=*/{}));
  kernel2.fixed_shape_kernel.wait();

  // Verify same cached executable is returned and output layouts match.
  EXPECT_EQ(kernel1.fixed_shape_kernel.get(), kernel2.fixed_shape_kernel.get());
  if (kernel2.fixed_shape_kernel.get().ok()) {
    const auto& exec = *kernel2.fixed_shape_kernel.get();
    EXPECT_FALSE(exec->output_shapes()[0].layout().has_value());
    ASSERT_TRUE(exec->output_shapes()[1].layout().has_value());
    EXPECT_EQ(exec->output_shapes()[1].layout()->minor_to_major,
              Indices({1, 2, 0}));
  }
}

TEST_F(TraversalTest,
       RaggedDotLayoutAnnotatedCorrectlyWithBoundedDynamicArguments) {
  Shape shape_dyn(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_lhs(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_rhs_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);
  Shape shape_group_sizes(Dimensions{8}, mlir::ElementType::I32);
  Shape shape_bias_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);
  Shape shape_dot_out(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_out_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);

  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref_dyn,
                          DeviceBufferList::CreatePlaceholder(
                              shape_dyn.dimensions(), shape_dyn.dtype()));
  ASSERT_EQ(ref_dyn.MarkDynamic(/*dimension=*/0, /*lower_bound=*/2,
                                /*upper_bound=*/128),
            absl::OkStatus());

  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref_lhs,
                          DeviceBufferList::CreatePlaceholder(
                              shape_lhs.dimensions(), shape_lhs.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref_rhs_3d,
                          DeviceBufferList::CreatePlaceholder(
                              shape_rhs_3d.dimensions(), shape_rhs_3d.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef ref_group_sizes,
      DeviceBufferList::CreatePlaceholder(shape_group_sizes.dimensions(),
                                          shape_group_sizes.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef ref_bias_3d,
      DeviceBufferList::CreatePlaceholder(shape_bias_3d.dimensions(),
                                          shape_bias_3d.dtype()));

  MlirOpBuilder ragged_dot_builder = [](mlir::MlirBuilder& builder,
                                        absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    mlir::MlirOp rhs_t = mlir::stablehlo::Transpose(inputs[2], {0, 2, 1});

    const auto dimension_numbers =
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(
            &builder.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{1},
            /*rhsContractingDimensions=*/{1}, /*lhsRaggedDimensions=*/{0},
            /*rhsGroupDimensions=*/{0});
    auto precision_attr = mlir::chlo::PrecisionAttr::get(
        &builder.getContext(),
        mlir::chlo::Precision::DEFAULT);  // EXPLICIT_PRECISION_OK=test IR
                                          // construction
    auto precision_config = mlir::ArrayAttr::get(
        &builder.getContext(), {precision_attr, precision_attr});
    const mlir::Type out_type = mlir::makeTensorType(
        builder.getContext(), {128, 128}, mlir::ElementType::BF16);
    mlir::MlirOp ragged_dot =
        mlir::chlo::RaggedDot(out_type, inputs[1], rhs_t, inputs[3],
                              dimension_numbers, precision_config);
    mlir::MlirOp add_3d = mlir::stablehlo::Add(inputs[2], inputs[4]);
    return DynamicMlirOpResults{ragged_dot, add_3d};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_out,
      DeviceBufferList::CreateDeferred(
          OpName::kRaggedDot, std::move(ragged_dot_builder),
          {ref_dyn, ref_lhs, ref_rhs_3d, ref_group_sizes, ref_bias_3d},
          OpParamCacheKeys::Empty(), {shape_dot_out, shape_out_3d}));

  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Traversal> tr,
                          Traversal::Create(refs_out));
  const auto ragged_reorder_status = tr->ValidateAndReorderArguments(
      {ref_dyn, ref_lhs, ref_rhs_3d, ref_group_sizes, ref_bias_3d});
  ASSERT_TRUE(ragged_reorder_status.ok()) << ragged_reorder_status;

  std::vector<CustomLayout> argument_layouts(5);
  argument_layouts[2] = CustomLayout{.minor_to_major = {1, 2, 0}};

  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       compile_options_key_);
  std::string mlir_text;
  TT_ASSERT_OK_AND_ASSIGN(
      const CompiledKernel kernel,
      tr->Compile(spec, UnboundedKernelPolicy::kDisallow, &mlir_text,
                  /*use_stablehlo_bounds=*/false, argument_layouts));
  kernel.fixed_shape_kernel.wait();

  // The dynamic dimension scalar argument (%arg0: tensor<i32>) must NOT have
  // any mhlo.layout_mode.
  EXPECT_THAT(
      mlir_text,
      testing::Not(testing::HasSubstr("%arg0: tensor<i32> {mhlo.layout_mode")));
  // The 3D tensor argument in @main (arg index shifted to %arg3 by %arg0
  // tensor<i32>) must have {1,2,0} layout attached.
  EXPECT_THAT(
      mlir_text,
      testing::HasSubstr(
          "%arg3: tensor<8x128x128xbf16> {mhlo.layout_mode = \"{1,2,0}\"}"));
}

}  // namespace
}  // namespace torch_tpu
