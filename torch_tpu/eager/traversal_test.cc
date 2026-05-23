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

#include "torch_tpu/eager/traversal.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "xla/tsl/platform/statusor.h"

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_a = refs_a[0];

  TF_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_b = refs_b[0];

  TF_ASSERT_OK_AND_ASSIGN(
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto refs, DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder,
                                                  {}, OpParamCacheKeys::Empty(),
                                                  {shape_, shape_}));

  TF_ASSERT_OK_AND_ASSIGN(auto traversal,
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
  TF_ASSERT_OK_AND_ASSIGN(auto refs, DeviceBufferList::CreateDeferred(
                                         OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape_}));
  auto ref = refs[0];

  TF_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create({ref}));

  std::string readable =
      traversal->ReadableString(MaterializationReason::kCpuTransfer);

  EXPECT_EQ(readable,
            "# Graph: 1 ops, 0 inputs, reason: .cpu()\n"
            "%0: f32[8] = add()\n"
            "return %0\n");
}

TEST_F(TraversalTest, ReadableStringComplexGraph) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_a = refs_a[0];

  TF_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_b = refs_b[0];

  TF_ASSERT_OK_AND_ASSIGN(
      auto refs_c,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape_}));
  auto ref_c = refs_c[0];

  TF_ASSERT_OK_AND_ASSIGN(auto refs_d,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, DummyBuilder, {ref_b, ref_c},
                              OpParamCacheKeys::Empty(), {shape_}));
  auto ref_d = refs_d[0];

  TF_ASSERT_OK_AND_ASSIGN(
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

  TF_ASSERT_OK_AND_ASSIGN(auto refs, DeviceBufferList::CreateDeferred(
                                         OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape_}));

  TF_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create({refs[0]}));

  std::string readable =
      traversal->ReadableString(MaterializationReason::kUnknown);
  EXPECT_THAT(readable,
              testing::HasSubstr("# /path/to/user_code.py:42 in my_function"));
}

}  // namespace
}  // namespace torch_tpu
