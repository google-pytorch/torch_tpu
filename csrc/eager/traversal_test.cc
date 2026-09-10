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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {
namespace {

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& /*builder*/, absl::Span<mlir::MlirOp> /*inputs*/) {
  return DynamicMlirOpResults{};
}

class TraversalTest : public testing::Test {
 protected:
  // Automatically handles Python context for every test.
  ScopedPythonContextCapturer capturer_{OpName::kEmpty};
  // Common shape used for creating deferred buffers.
  Shape shape_{Dimensions{8}, mlir::ElementType::F32};
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
                       CompileOptionsKey(12345));
  std::string mlir_text;
  ASSERT_TRUE(
      traversal
          ->Compile(
              std::move(spec), &mlir_text,
              /*use_stablehlo_bounds=*/false,
              /*argument_layouts=*/{CustomLayout{.minor_to_major = {1, 0}}})
          .ok());
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
  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       CompileOptionsKey(12345));
  std::string mlir_text;
  ASSERT_TRUE(tr->Compile(std::move(spec), &mlir_text,
                          /*use_stablehlo_bounds=*/false,
                          /*argument_layouts=*/{tiled_layout})
                  .ok());
  EXPECT_THAT(mlir_text,
              testing::HasSubstr("mhlo.layout_mode = \"{1,0:T(8,128)}\""));

  CustomLayout different_tile_layout{.minor_to_major = {1, 0}, .tiles = {{16}}};
  CompilationCacheKey key_untiled = tr->GetCacheKey(
      CompileOptionsKey(12345), {CustomLayout{.minor_to_major = {1, 0}}});
  CompilationCacheKey key_tiled8 =
      tr->GetCacheKey(CompileOptionsKey(12345), {tiled_layout});
  CompilationCacheKey key_tiled16 =
      tr->GetCacheKey(CompileOptionsKey(12345), {different_tile_layout});
  EXPECT_NE(key_untiled, key_tiled8);
  EXPECT_NE(key_tiled8, key_tiled16);
}

}  // namespace
}  // namespace torch_tpu
