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

#include "torch_tpu/runtime/backend.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "torch_tpu/runtime/mock_backend.h"

namespace torch_tpu {
namespace {

TEST(BackendTest, FactoryCreatesMockBackend) {
  BackendOptions options;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Backend> backend,
                       Backend::Create(BackendKind::kMock, options));
  EXPECT_NE(backend, nullptr);

  auto devices = backend->GetDevices();
  EXPECT_EQ(devices.size(), 1);
  EXPECT_EQ(devices[0]->kind(), "MockTPU");
}

TEST(BackendTest, GlobalSingletonManagement) {
  auto cleanup = absl::MakeCleanup([]() { Backend::ResetInstance(); });

  auto mock_backend = std::make_unique<MockBackend>();
  Backend* mock_ptr = mock_backend.get();
  Backend::SetInstance(std::move(mock_backend));

  EXPECT_EQ(Backend::GetInstance(), mock_ptr);
}

TEST(BackendTest, MakeBufferFromHostAndCopyToHost) {
  MockBackend backend;
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DeviceBuffer> buffer,
      backend.MakeBufferFromHost(host_data.data(), at::ScalarType::Float,
                                 {2, 2}, backend.GetDefaultDevice()));

  EXPECT_EQ(buffer->dtype(), at::ScalarType::Float);
  EXPECT_EQ(buffer->shape().size(), 2);
  EXPECT_EQ(buffer->shape()[0], 2);
  EXPECT_EQ(buffer->shape()[1], 2);
  EXPECT_EQ(buffer->GetOnDeviceSizeInBytes(), 4 * sizeof(float));

  std::vector<float> readback(4, 0.0f);
  auto future =
      buffer->CopyToHostAsync(readback.data(), readback.size() * sizeof(float));
  EXPECT_TRUE(buffer->BlockHostUntilReady().ok());

  for (size_t i = 0; i < host_data.size(); ++i) {
    EXPECT_FLOAT_EQ(readback[i], host_data[i]);
  }
}

TEST(BackendTest, MockCompilerAndExecutable) {
  MockBackend backend;
  ASSERT_OK_AND_ASSIGN(auto compiler, backend.GetCompiler());

  CompileOptions comp_opts;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoadedExecutable> exe,
                       compiler->Compile("dummy_stablehlo_program", comp_opts));

  ASSERT_OK_AND_ASSIGN(std::string fp, exe->Fingerprint());
  EXPECT_EQ(fp, "mock_executable_fingerprint_12345");

  std::vector<float> data = {5.0f};
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DeviceBuffer> buf,
      backend.MakeBufferFromHost(data.data(), at::ScalarType::Float, {1},
                                 backend.GetDefaultDevice()));

  std::vector<DeviceBuffer*> args = {buf.get()};
  ExecuteOptions exec_opts;
  ASSERT_OK_AND_ASSIGN(auto results, exe->Execute(args, exec_opts));
  EXPECT_EQ(results.size(), 1);
}

TEST(BackendTest, SynchronizationAndDonationTracking) {
  MockBackend backend;
  EXPECT_OK(backend.SynchronizeDevice(0));
  EXPECT_OK(backend.SynchronizeAllDevices());

  std::vector<float> data = {1.0f, 2.0f};
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DeviceBuffer> buf,
      backend.MakeBufferFromHost(data.data(), at::ScalarType::Float, {2},
                                 backend.GetDefaultDevice()));

  EXPECT_FALSE(buf->IsDonated());
  buf->MarkDonated();
  EXPECT_TRUE(buf->IsDonated());
  EXPECT_EQ(buf->mesh_coordinates(), std::nullopt);
}

}  // namespace
}  // namespace torch_tpu
