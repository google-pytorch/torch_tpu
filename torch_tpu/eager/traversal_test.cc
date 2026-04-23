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

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return DynamicMlirOpResults{};
}

TEST(TraversalTest, ReadableString) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  auto traversal_or =
      Traversal::Create({ref_b}, {ref_a.device_buffer_list().get()});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal.SortByCreationOrder();

  std::string readable =
      traversal.ReadableString(MaterializationReason::kUnknown);

  EXPECT_THAT(readable, testing::HasSubstr("# Graph: 1 ops"));
  EXPECT_THAT(readable, testing::HasSubstr("%0: f32[8] = input"));
  EXPECT_THAT(readable, testing::HasSubstr("= add"));
}

}  // namespace
}  // namespace torch_tpu
