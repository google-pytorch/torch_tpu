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

#include "torch_tpu/_internal/compile/compiled_mode.h"

#include "ATen/core/TensorBody.h"
#include "ATen/ops/arange.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/tpu_hooks.h"
#include "torch_tpu/ops/copy_from/cpu_to_tpu.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

TEST(CompiledModeTest, HandlesMinorToMajorLayoutCorrectly) {
  const std::string device_type = "xla_cpu";
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = device_type});

  auto status = AddTpuHooks();
  EXPECT_TRUE(status.ok());
  RegisterTpuAllocator();

  // Simulating a col-major buffer dynamically returned by PjRt / XLA.
  // We manually provide a perfectly identical buffer directly to MakeTensor.
  at::Tensor cpu_tensor = at::arange(16, at::kFloat).view({4, 4}).t();
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef buffer,
      CopyCpuToTpuBuffer(cpu_tensor, /*non_blocking=*/false));

  // TorchTPU wraps the returned XLA buffer into an ATen Tensor using
  // MakeTensor.
  at::Tensor output = MakeTensor(buffer);

  EXPECT_EQ(output.sizes()[0], 4);
  EXPECT_EQ(output.sizes()[1], 4);

  // The tensor should not incorrectly identify itself as row-major contiguous.
  EXPECT_FALSE(output.unsafeGetTensorImpl()->is_contiguous());
}

}  // namespace
}  // namespace torch_tpu
