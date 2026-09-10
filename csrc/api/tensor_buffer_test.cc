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

#include "csrc/api/tensor_buffer.h"

#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/tpu_hooks.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {
namespace {

class TensorBufferTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string device_type = "xla_cpu";
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = device_type});
    ASSERT_EQ(AddTpuHooks(), absl::OkStatus());
    RegisterTpuAllocator();
    CompilationCache::GetInstance().SetOptions({});
  }

  static void TearDownTestSuite() {
    ShutDownMaterializationState();
    CompilationCache::ShutDown();
    PjrtBackend::GetInstance().Shutdown();
  }

  static at::Tensor CreateDeferredTpuTensor(const Shape& shape) {
    const ScopedPythonContextCapturer capturer(OpName::kEmpty);
    auto builder = [shape](mlir::MlirBuilder& builder,
                           absl::Span<mlir::MlirOp>) {
      return DynamicMlirOpResults{
          BuildFillUninitialized(builder, shape.dtype(), shape.dimensions())};
    };
    auto refs_or =
        DeviceBufferList::CreateDeferred(OpName::kEmpty, builder, /*inputs=*/{},
                                         OpParamCacheKeys::Empty(), {shape});
    if (!refs_or.ok()) {
      return at::Tensor();  // UNINITIALIZED_TENSOR_OK
    }
    return MakeTensor((*refs_or)[0]);
  }
};

TEST_F(TensorBufferTest, UndefinedTensorRejection) {
  at::Tensor undefined_tensor;  // UNINITIALIZED_TENSOR_OK
  auto handle_or = GetBaseTensorBuffer(undefined_tensor);
  EXPECT_FALSE(handle_or.ok());
}

TEST_F(TensorBufferTest, CpuTensorRejection) {
  at::Tensor cpu_tensor = at::zeros({2, 3}, at::kFloat);
  auto handle_or = GetBaseTensorBuffer(cpu_tensor);
  EXPECT_FALSE(handle_or.ok());
}

TEST_F(TensorBufferTest, TpuTensorBufferExtraction) {
  const Shape shape(Dimensions{4, 2}, mlir::ElementType::F32);
  at::Tensor tensor = CreateDeferredTpuTensor(shape);

  TT_ASSERT_OK_AND_ASSIGN(TensorBufferHandle handle,
                          GetBaseTensorBuffer(tensor));
  EXPECT_EQ(handle.size_bytes(), 8 * sizeof(float));
  EXPECT_EQ(handle.num_elements(), 8);
  EXPECT_THAT(handle.dimensions(), testing::ElementsAre(4, 2));
  EXPECT_EQ(handle.state(), DeviceBufferState::kDeferred);
}

TEST_F(TensorBufferTest, MoveSemantics) {
  const Shape shape(Dimensions{8}, mlir::ElementType::F32);
  at::Tensor tensor = CreateDeferredTpuTensor(shape);

  TT_ASSERT_OK_AND_ASSIGN(TensorBufferHandle handle,
                          GetBaseTensorBuffer(tensor));
  EXPECT_EQ(handle.size_bytes(), 8 * sizeof(float));

  // Move construction
  TensorBufferHandle move_constructed(std::move(handle));
  EXPECT_EQ(move_constructed.size_bytes(), 8 * sizeof(float));
  EXPECT_EQ(move_constructed.num_elements(), 8);
  EXPECT_THAT(move_constructed.dimensions(), testing::ElementsAre(8));
  EXPECT_EQ(move_constructed.state(), DeviceBufferState::kDeferred);

  // Move assignment
  TensorBufferHandle move_assigned = std::move(move_constructed);
  EXPECT_EQ(move_assigned.size_bytes(), 8 * sizeof(float));
  EXPECT_EQ(move_assigned.num_elements(), 8);
  EXPECT_THAT(move_assigned.dimensions(), testing::ElementsAre(8));
  EXPECT_EQ(move_assigned.state(), DeviceBufferState::kDeferred);
}

TEST_F(TensorBufferTest, MaterializeAndAwaitBuffer) {
  const Shape shape(Dimensions{16}, mlir::ElementType::F32);
  at::Tensor tensor = CreateDeferredTpuTensor(shape);

  TT_ASSERT_OK_AND_ASSIGN(TensorBufferHandle handle,
                          GetBaseTensorBuffer(tensor));
  EXPECT_EQ(handle.state(), DeviceBufferState::kDeferred);

  EXPECT_TRUE(handle.Materialize().ok());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer * buffer, handle.AwaitBuffer());
  ASSERT_NE(buffer, nullptr);

  EXPECT_TRUE(handle.Synchronize().ok());
  EXPECT_EQ(handle.state(), DeviceBufferState::kMaterialized);
}

TEST_F(TensorBufferTest, ViewTensorBaseBufferSharing) {
  const Shape base_shape(Dimensions{8}, mlir::ElementType::F32);
  at::Tensor base_tensor = CreateDeferredTpuTensor(base_shape);
  c10::Storage storage_copy = base_tensor.storage();
  at::Tensor view_tensor(c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage_copy), base_tensor.key_set(),
      base_tensor.dtype()));
  view_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      Dimensions{2, 4}, Strides{4, 1}, /*storage_offset=*/0);

  TT_ASSERT_OK_AND_ASSIGN(TensorBufferHandle base_handle,
                          GetBaseTensorBuffer(base_tensor));

  TT_ASSERT_OK_AND_ASSIGN(TensorBufferHandle view_handle,
                          GetBaseTensorBuffer(view_tensor));

  EXPECT_THAT(base_handle.dimensions(), testing::ElementsAre(8));
  EXPECT_THAT(view_handle.dimensions(), testing::ElementsAre(8));
  EXPECT_EQ(base_handle.size_bytes(), view_handle.size_bytes());

  EXPECT_TRUE(base_handle.Materialize().ok());

  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer * base_buf,
                          base_handle.AwaitBuffer());

  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer * view_buf,
                          view_handle.AwaitBuffer());
  EXPECT_EQ(base_buf, view_buf);
}

}  // namespace
}  // namespace torch_tpu
