// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/runtime/cerebro/cerebro_cluster_pipeline.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "c10/core/ScalarType.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/cerebro/cerebro_backend_impl.h"
#include "torch_tpu/runtime/device_buffer.h"

namespace torch_tpu {
namespace {

TEST(CerebroClusterPipelineTest, TopologyAndCoordinateResolution) {
  BackendOptions options;
  CerebroBackendImpl backend(options);
  CerebroClusterPipeline pipeline(&backend);

  EXPECT_GT(pipeline.GetClusterDeviceCount(), 0);

  ASSERT_OK_AND_ASSIGN(auto coords, pipeline.GetDeviceCoordinate(0));
  EXPECT_EQ(coords.size(), 3);

  EXPECT_FALSE(pipeline.GetDeviceCoordinate(-1).ok());
  EXPECT_FALSE(pipeline.GetDeviceCoordinate(9999).ok());
}

TEST(CerebroClusterPipelineTest, CollectiveExecutionAndBarrier) {
  BackendOptions options;
  CerebroBackendImpl backend(options);
  CerebroClusterPipeline pipeline(&backend);

  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};  // INT_VEC_OK

  ASSERT_OK_AND_ASSIGN(auto buf1, backend.MakeBufferFromHost(
                                      host_data.data(), at::ScalarType::Float,
                                      shape, backend.GetDefaultDevice()));
  ASSERT_NE(buf1, nullptr);

  ASSERT_OK_AND_ASSIGN(auto buf2, backend.MakeBufferFromHost(
                                      host_data.data(), at::ScalarType::Float,
                                      shape, backend.GetDefaultDevice()));
  ASSERT_NE(buf2, nullptr);

  std::vector<DeviceBuffer*> input_bufs = {buf1.get(), buf2.get()};
  ASSERT_OK_AND_ASSIGN(
      auto collective_results,
      pipeline.ExecuteCollective(CollectiveOp::kAllReduce, input_bufs));
  EXPECT_EQ(collective_results.size(), 2);

  EXPECT_TRUE(pipeline.Barrier().ok());
}

}  // namespace
}  // namespace torch_tpu
