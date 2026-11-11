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

#include "csrc/eager/device_buffer.h"

#include <cstdint>
#include <optional>

#include "csrc/common/dimension_types.h"
#include "csrc/common/status_test_utils.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

TEST(DeviceBufferRefTest, WeakTest) {
  // Creates a DeviceBufferRef.
  std::optional<DeviceBufferRef> keepalive;
  TT_ASSERT_OK_AND_ASSIGN(
      keepalive, DeviceBufferList::CreatePlaceholder(Dimensions({2, 3}),
                                                     mlir::ElementType::F32));
  ASSERT_TRUE(keepalive.has_value());

  const DeviceBufferList* const list_addr =
      keepalive->device_buffer_list().get();
  const int64_t index = keepalive->index();

  // Make a weak reference to it.
  DeviceBufferRef::Weak weak = keepalive->weak();

  {
    // Check that we can recover the original DeviceBufferRef while it exists.
    EXPECT_FALSE(weak.expired());
    std::optional<DeviceBufferRef> recovered = weak.lock();
    ASSERT_TRUE(recovered.has_value());
    EXPECT_EQ(recovered->device_buffer_list().get(), list_addr);
    EXPECT_EQ(recovered->index(), index);
  }

  // The original DeviceBufferRef is dropped.
  keepalive.reset();

  // Check that the weak reference is now empty.
  EXPECT_TRUE(weak.expired());
  std::optional<DeviceBufferRef> recovered = weak.lock();
  EXPECT_FALSE(recovered.has_value());
}

TEST(DeviceBufferRefTest, PlaceholderGetSharedBufferAndAwaitBufferFail) {
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef placeholder,
                          DeviceBufferList::CreatePlaceholder(
                              Dimensions({2, 3}), mlir::ElementType::F32));
  EXPECT_FALSE(placeholder.GetSharedBuffer().ok());
  EXPECT_FALSE(placeholder.AwaitBuffer().ok());
  EXPECT_FALSE(placeholder.pjrt_buffer_size().ok());
  EXPECT_FALSE(placeholder.device_buffer_list()->Synchronize().ok());
}

}  // namespace
}  // namespace torch_tpu
