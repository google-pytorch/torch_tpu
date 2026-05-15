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
#include <utility>
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
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
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

TEST_F(TraversalTest, DeadCodeElimination) {
  auto traceback = std::make_shared<PythonTraceback>();
  traceback->frames.push_back({"/path/to/user_code.py", "my_function", 42});

  ScopedPythonContextCapturer::SetTracebackForTesting(traceback);
  std::vector<DeviceBufferRef> refs;

  // Create four parallel graphs:
  // a [placeholder] -> b [has tensor]              -> c [dead code]
  // d [placeholder] -> e [distributed collective]  -> f [dead code]
  // g [placeholder] -> h [dead code, but required] -> i [dead code]
  // j [placeholder] -> k [dead code]               -> l [dead code]

  TF_ASSERT_OK_AND_ASSIGN(auto ref_a, DeviceBufferList::CreateEmpty(
                                          shape_.dimensions(), shape_.dtype()));
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_a},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_b = std::move(refs[0]);
  auto ref_b_tensor = MakeTensor(ref_b);
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_b},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_c = std::move(refs[0]);

  TF_ASSERT_OK_AND_ASSIGN(auto ref_d, DeviceBufferList::CreateEmpty(
                                          shape_.dimensions(), shape_.dtype()));
  TF_ASSERT_OK_AND_ASSIGN(
      refs, DeviceBufferList::CreateDeferred(
                OpName::kDistributedAllReduce, DummyBuilder, {ref_d},
                OpParamCacheKeys::Empty(), {shape_}));
  auto ref_e = std::move(refs[0]);
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_e},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_f = std::move(refs[0]);

  TF_ASSERT_OK_AND_ASSIGN(auto ref_g, DeviceBufferList::CreateEmpty(
                                          shape_.dimensions(), shape_.dtype()));
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_g},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_h = std::move(refs[0]);
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_h},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_i = std::move(refs[0]);

  TF_ASSERT_OK_AND_ASSIGN(auto ref_j, DeviceBufferList::CreateEmpty(
                                          shape_.dimensions(), shape_.dtype()));
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_j},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_k = std::move(refs[0]);
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_k},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_l = std::move(refs[0]);

  // Collect these all into a single traversal.
  TF_ASSERT_OK_AND_ASSIGN(auto traversal,
                          Traversal::Create({ref_c, ref_f, ref_i, ref_l}));
  traversal->SortByCreationOrder();

  // Run the dead code elimination, keeping h as a required node.
  auto new_traversal = Traversal::EliminateDeadCode(
      std::move(traversal), {ref_h.device_buffer_list().get()});
  // The new traversal should not be null.
  ASSERT_NE(new_traversal, nullptr);

  // The new traversal should be:
  // Arguments: a, d, g. Buffer j was dead and eliminated.
  // Execution nodes: b, e, h. Nodes c, f, i, and l were dead and eliminated.
  // Outputs: b, e, h.
  ASSERT_EQ(new_traversal->arguments().size(), 3);
  EXPECT_THAT(new_traversal->arguments(),
              testing::ElementsAre(ref_a, ref_d, ref_g));
  ASSERT_EQ(new_traversal->execution_order().size(), 3);
  EXPECT_THAT(new_traversal->execution_order(),
              testing::ElementsAre(ref_b.device_buffer_list(),
                                   ref_e.device_buffer_list(),
                                   ref_h.device_buffer_list()));
  ASSERT_EQ(new_traversal->outputs().size(), 3);
  EXPECT_THAT(new_traversal->outputs(),
              testing::ElementsAre(ref_b, ref_e, ref_h));
}

TEST_F(TraversalTest, DeadCodeEliminationReturnsNullptr) {
  auto traceback = std::make_shared<PythonTraceback>();
  traceback->frames.push_back({"/path/to/user_code.py", "my_function", 42});

  ScopedPythonContextCapturer::SetTracebackForTesting(traceback);
  std::vector<DeviceBufferRef> refs;

  // Create a graph of dead code:
  // a [placeholder] -> b [dead code]               -> c [dead code]

  TF_ASSERT_OK_AND_ASSIGN(auto ref_a, DeviceBufferList::CreateEmpty(
                                          shape_.dimensions(), shape_.dtype()));
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_a},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_b = std::move(refs[0]);
  TF_ASSERT_OK_AND_ASSIGN(refs, DeviceBufferList::CreateDeferred(
                                    OpName::kAdd, DummyBuilder, {ref_b},
                                    OpParamCacheKeys::Empty(), {shape_}));
  auto ref_c = std::move(refs[0]);
  TF_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create({ref_c}));

  // Run the dead code elimination, keeping c as a required node.
  auto new_traversal = Traversal::EliminateDeadCode(
      std::move(traversal), {ref_c.device_buffer_list().get()});
  // This should return nullptr because the entire graph is eliminated.
  EXPECT_EQ(new_traversal, nullptr);
}

}  // namespace
}  // namespace torch_tpu
