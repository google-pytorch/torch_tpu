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

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/shape.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/materialize.h"
#include "torch_tpu/csrc/eager/structured_log_buffer.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/eager/tpu_hooks.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

TEST(ShutdownTest, E2EShutdownCompletes) {
  // 1. Set up PjrtBackend initialization to run using TPU.
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});

  // 2. Call AddTpuHooks() and RegisterTpuAllocator().
  ASSERT_EQ(AddTpuHooks(), absl::OkStatus());
  RegisterTpuAllocator();

  // 3. Construct a dummy deferred DeviceBufferList and call Materialize() on a
  // list containing it to trigger lazy worker initialization and threads
  // startup.
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  const Shape shape(Dimensions{8}, mlir::ElementType::F32);
  auto builder = [shape](mlir::MlirBuilder& builder,
                         absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    if (!inputs.empty()) return DynamicMlirOpResults{inputs[0]};
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, shape.dtype(), shape.dimensions())};
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(OpName::kEmpty, builder,
                                       /*inputs=*/{}, OpParamCacheKeys::Empty(),
                                       {shape}));

  // Trigger materialization. We don't await the result because we just want to
  // start the process and then shutdown.
  ASSERT_EQ(Materialize(refs, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  // 4. Call ShutDownMaterializationState().
  ShutDownMaterializationState();

  // Clean up backend to avoid impacting other tests in the same process.
  PjrtBackend::GetInstance().Shutdown();
}

}  // namespace
}  // namespace torch_tpu
