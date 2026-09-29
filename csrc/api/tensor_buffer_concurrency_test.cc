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

#include <atomic>
#include <cstdint>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/api/tensor_buffer.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

constexpr int64_t kNumElements = 16;
constexpr int kNumReaders = 4;
constexpr int kNumWrites = 2000;

// GetBaseTensorBuffer on other threads, called on the tensor or on another
// tensor sharing its storage, must not race in-place writes that rebind the
// storage to a new buffer.
TEST(TensorBufferConcurrencyTest, GetBaseTensorBufferDuringStorageRebind) {
  std::vector<DeviceBufferRef> buffers;
  buffers.reserve(kNumWrites + 1);
  for (int i = 0; i <= kNumWrites; ++i) {
    TT_ASSERT_OK_AND_ASSIGN(
        DeviceBufferRef buffer,
        DeviceBufferList::CreatePlaceholder(Dimensions{kNumElements},
                                            mlir::ElementType::F32));
    buffers.push_back(std::move(buffer));
  }
  const at::Tensor tensor = MakeTensor(std::move(buffers[0]));
  c10::Storage storage_copy = tensor.storage();
  at::Tensor shared_storage_tensor(c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage_copy), tensor.key_set(),
      tensor.dtype()));
  shared_storage_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      Dimensions{kNumElements}, Strides{1}, /*storage_offset=*/0);

  std::atomic<int> num_started = 0;
  std::atomic<bool> done = false;
  std::vector<std::thread> readers;
  readers.reserve(kNumReaders);
  for (int i = 0; i < kNumReaders; ++i) {
    const at::Tensor* const target =
        i % 2 == 0 ? &tensor : &shared_storage_tensor;
    readers.emplace_back([target, &num_started, &done] {
      num_started.fetch_add(1);
      while (!done.load()) {
        absl::StatusOr<TensorBufferHandle> handle =
            GetBaseTensorBuffer(*target);
        ASSERT_TRUE(handle.ok()) << handle.status();
        ASSERT_EQ(handle->num_elements(), kNumElements);
      }
    });
  }
  while (num_started.load() < kNumReaders) {
    std::this_thread::yield();
  }
  for (int i = 1; i <= kNumWrites; ++i) {
    EXPECT_EQ(AssignBufferToAtTensor(std::move(buffers[i]), tensor),
              absl::OkStatus());
  }
  done.store(true);
  for (std::thread& reader : readers) {
    reader.join();
  }
}

}  // namespace
}  // namespace torch_tpu
