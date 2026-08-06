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

#include "torch_tpu/ifrt/ifrt_state.h"

#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"

namespace torch_tpu {
namespace {

class IfrtStateTest : public testing::Test {
 protected:
  void SetUp() override {
    IfrtBackend::GetInstance().Shutdown();
    PjrtBackend::GetInstance().Shutdown();
  }

  void TearDown() override {
    IfrtBackend::GetInstance().Shutdown();
    PjrtBackend::GetInstance().Shutdown();
  }
};

TEST_F(IfrtStateTest, InitializeIfrtWrapsPjRt) {
  ASSERT_FALSE(IfrtBackend::GetInstance().IsInitialized());

  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});

  auto* client = IfrtBackend::GetInstance().GetClient();
  ASSERT_NE(client, nullptr);
  EXPECT_TRUE(IfrtBackend::GetInstance().IsInitialized());
  EXPECT_EQ(client->runtime_type(), "pjrt_ifrt");

  auto device_count_or = IfrtBackend::GetInstance().GetGlobalDeviceCount();
  ASSERT_TRUE(device_count_or.ok());
  EXPECT_GT(*device_count_or, 0);

  auto device_id_or = IfrtBackend::GetInstance().GetGlobalDeviceId();
  ASSERT_TRUE(device_id_or.ok());
  EXPECT_GE(*device_id_or, 0);

  auto* device = IfrtBackend::GetInstance().GetDevice();
  ASSERT_NE(device, nullptr);
  EXPECT_EQ(device->Id().value(), *device_id_or);
}

TEST_F(IfrtStateTest, ConcurrentInitializationThreadSafe) {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});

  constexpr int kNumThreads = 16;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  std::vector<absl::Status> init_statuses(kNumThreads);
  std::vector<xla::ifrt::Client*> clients(kNumThreads, nullptr);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([i, &init_statuses, &clients]() {
      init_statuses[i] = IfrtBackend::GetInstance().EnsureInitialized();
      clients[i] = IfrtBackend::GetInstance().GetClient();
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < kNumThreads; ++i) {
    EXPECT_TRUE(init_statuses[i].ok());
    EXPECT_NE(clients[i], nullptr);
    EXPECT_EQ(clients[i], clients[0]);
  }

  EXPECT_TRUE(IfrtBackend::GetInstance().IsInitialized());
}

TEST_F(IfrtStateTest, MultipleShutdownAndReinitialization) {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});

  for (int cycle = 0; cycle < 3; ++cycle) {
    EXPECT_FALSE(IfrtBackend::GetInstance().IsInitialized());
    auto* client = IfrtBackend::GetInstance().GetClient();
    ASSERT_NE(client, nullptr);
    EXPECT_TRUE(IfrtBackend::GetInstance().IsInitialized());

    IfrtBackend::GetInstance().Shutdown();
    EXPECT_FALSE(IfrtBackend::GetInstance().IsInitialized());
  }
}

TEST_F(IfrtStateTest, SharedClientLifecycleAndOwnership) {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});

  auto shared_client_or = IfrtBackend::GetInstance().GetSharedClient();
  ASSERT_TRUE(shared_client_or.ok());
  auto shared_client = *shared_client_or;
  ASSERT_NE(shared_client, nullptr);
  EXPECT_GT(shared_client.use_count(), 1);

  // Shut down backend; caller's shared_ptr should retain valid client memory.
  IfrtBackend::GetInstance().Shutdown();
  EXPECT_FALSE(IfrtBackend::GetInstance().IsInitialized());
  EXPECT_EQ(shared_client->runtime_type(), "pjrt_ifrt");
}

TEST_F(IfrtStateTest, HostBufferToIfrtArrayAllocation) {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});

  auto* client = IfrtBackend::GetInstance().GetClient();
  ASSERT_NE(client, nullptr);

  auto* device = IfrtBackend::GetInstance().GetDevice();
  ASSERT_NE(device, nullptr);

  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  xla::ifrt::Shape shape({2, 2});
  xla::ifrt::DType dtype(xla::ifrt::DType::kF32);
  auto sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  auto array_or = client->MakeArrayFromHostBuffer(
      host_data.data(), dtype, shape, /*byte_strides=*/std::nullopt,
      std::move(sharding), /*layout=*/nullptr,
      xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/nullptr);

  ASSERT_TRUE(array_or.ok()) << array_or.status();
  auto array = *array_or;
  ASSERT_NE(array, nullptr);
  EXPECT_EQ(array->shape(), shape);
  EXPECT_EQ(array->dtype(), dtype);
}

}  // namespace
}  // namespace torch_tpu
