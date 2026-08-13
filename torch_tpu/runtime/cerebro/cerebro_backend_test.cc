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

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/cerebro/cerebro_backend_impl.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"

namespace torch_tpu {
namespace {

TEST(CerebroBackendTest, ClusterTopologyAndSingleController) {
  BackendOptions options;
  auto backend = std::make_unique<CerebroBackendImpl>(options);

  EXPECT_TRUE(backend->IsSingleController());
  auto devices = backend->GetDevices();
  EXPECT_EQ(devices.size(), 32);  // 4 nodes * 4 chips * 2 cores = 32

  Device* dev0 = devices[0];
  EXPECT_EQ(dev0->id(), 0);
  EXPECT_EQ(dev0->kind(), "TPU v5e");
  EXPECT_TRUE(dev0->is_addressable());

  auto coords = dev0->coords();
  ASSERT_TRUE(coords.has_value());
  EXPECT_EQ((*coords)[0], 0);
  EXPECT_EQ((*coords)[1], 0);
  EXPECT_EQ((*coords)[2], 0);
}

TEST(CerebroBackendTest, BufferAllocationAndView) {
  BackendOptions options;
  auto backend = std::make_unique<CerebroBackendImpl>(options);

  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DeviceBuffer> buffer,
      backend->MakeBufferFromHost(host_data.data(), at::ScalarType::Float,
                                  {2, 2}, backend->GetDefaultDevice()));

  EXPECT_EQ(buffer->dtype(), at::ScalarType::Float);
  EXPECT_EQ(buffer->shape().size(), 2);
  EXPECT_EQ(buffer->shape()[0], 2);
  EXPECT_EQ(buffer->shape()[1], 2);
  EXPECT_EQ(buffer->GetOnDeviceSizeInBytes(), 4 * sizeof(float));

  ASSERT_OK_AND_ASSIGN(auto view, buffer->CreateView({4}));
  EXPECT_EQ(view->shape().size(), 1);
  EXPECT_EQ(view->shape()[0], 4);
}

TEST(CerebroBackendTest, CompileAndExecute) {
  BackendOptions options;
  auto backend = std::make_unique<CerebroBackendImpl>(options);

  ASSERT_OK_AND_ASSIGN(auto compiler, backend->GetCompiler());

  CompileOptions comp_opts;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoadedExecutable> exe,
                       compiler->Compile("mock_stablehlo_module", comp_opts));

  ASSERT_OK_AND_ASSIGN(std::string fp, exe->Fingerprint());
  EXPECT_FALSE(fp.empty());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DeviceBuffer> buf,
      backend->MakeBufferFromHost(nullptr, at::ScalarType::Float, {2, 2},
                                  backend->GetDefaultDevice()));

  std::vector<DeviceBuffer*> args = {buf.get()};
  ExecuteOptions exec_opts;
  ASSERT_OK_AND_ASSIGN(auto results, exe->Execute(args, exec_opts));
  EXPECT_EQ(results.size(), 1);

  EXPECT_OK(backend->SynchronizeDevice(0));
  EXPECT_OK(backend->SynchronizeAllDevices());
  EXPECT_TRUE(buf->mesh_coordinates().has_value());
  EXPECT_FALSE(buf->IsDonated());
  buf->MarkDonated();
  EXPECT_TRUE(buf->IsDonated());
}

}  // namespace
}  // namespace torch_tpu
