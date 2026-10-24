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

#include "csrc/eager/device_buffer_utils.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::Pair;

// A dummy MLIR op builder for testing purposes that will fail to compile.
absl::StatusOr<DynamicMlirOpResults> FailsToCompileBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return TT_ERROR(error::kInternal) << "Intentional compilation failure";
}

TEST(DeviceBufferUtilsTest,
     ComputeConstantDeviceBufferRefOpParamCacheKeysMatchesGoldenMap) {
  const std::vector<char> data = {'a', 'b', 'c', 'd'};
  const Dimensions dims = {2, 3};
  const auto dtype = mlir::ElementType::F32;

  TT_ASSERT_OK_AND_ASSIGN(
      const OpParamCacheKeys keys,
      internal::ComputeConstantDeviceBufferRefOpParamCacheKeys(data, dims,
                                                               dtype));

  const std::map<std::string, FingerprintType> actual_map(keys.begin(),
                                                          keys.end());

  EXPECT_THAT(actual_map, ElementsAre(
                              // go/keep-sorted start
                              Pair("data", 1897425971756105985ULL),
                              Pair("dimensions", FingerprintCat("", 2, 3)),
                              Pair("element_type",
                                   Fingerprint("f32"))  //
                                                        // go/keep-sorted end
                              ));
}

TEST(DeviceBufferUtilsTest, DisallowMaterializationInDeferNever) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Make sure we're in kDeferNever mode.
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferNever);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });

  // Dispatch an op that explicitly cannot compile or execute. If this tries
  // to execute, the test will fail.
  internal::DeferredOpParams params{
      .op_name = OpName::kAdd,
      .op_builder = FailsToCompileBuilder,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
      .output_shapes = {shape},
      .allow_materialization = false,
  };

  // Create the deferred device buffer list. This should be successful because
  // it will not attempt to materialize.
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs, internal::CreateDeferredDeviceBufferList(std::move(params)));
  ASSERT_EQ(refs.size(), 1);
  EXPECT_TRUE(refs[0].is_deferred());

  // Sleep for 100ms. This is enough that a background materialization should
  // have started if there was a materialization attempt.
  absl::SleepFor(absl::Milliseconds(100));

  // Make sure the ref is still deferred.
  EXPECT_TRUE(refs[0].is_deferred());
  // Drop the invalid ref from the stream.
  ClearAllStreams();
}

TEST(DeviceBufferUtilsTest, DisallowMaterializationInDeferAndFuse) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Make sure we're in kDeferAndFuse mode.
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferAndFuse);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });

  // Create 10 copies of an op that explicitly cannot compile or execute.
  // This would be enough to start a repeated op materialization.
  std::vector<DeviceBufferRef> ten_refs;
  ten_refs.reserve(10);
  for (int i = 0; i < 10; ++i) {
    internal::DeferredOpParams params{
        .op_name = OpName::kAdd,
        .op_builder = FailsToCompileBuilder,
        .op_param_cache_keys = OpParamCacheKeys::Empty(),
        .output_shapes = {shape},
        .allow_materialization = false,
    };
    TT_ASSERT_OK_AND_ASSIGN(
        auto refs, internal::CreateDeferredDeviceBufferList(std::move(params)));
    ASSERT_EQ(refs.size(), 1);
    EXPECT_TRUE(refs[0].is_deferred());
    ten_refs.push_back(std::move(refs[0]));
  }

  // Sleep for 100ms. This is enough that a background materialization should
  // have started if there was a materialization attempt.
  absl::SleepFor(absl::Milliseconds(100));

  // Make sure all refs are still deferred.
  EXPECT_TRUE(std::all_of(
      ten_refs.begin(), ten_refs.end(),
      [](const DeviceBufferRef& ref) { return ref.is_deferred(); }));
  // Drop the invalid refs from the stream.
  ClearAllStreams();
}

}  // namespace
}  // namespace torch_tpu
