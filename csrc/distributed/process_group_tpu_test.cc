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

#include "csrc/distributed/process_group_tpu.h"

#include "gtest/gtest.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {
namespace {

TEST(ProcessGroupTpuTest, GetCrossHostTransferKeyDeterministic) {
  auto key1 = GetCrossHostTransferKey(0, 1, 10, 0);
  auto key2 = GetCrossHostTransferKey(0, 1, 10, 0);
  EXPECT_EQ(key1.value(), 6634354603530518008LL);
  EXPECT_EQ(key2.value(), 6634354603530518008LL);
}

TEST(ProcessGroupTpuTest, GetCrossHostTransferKeyOrderDependent) {
  auto key_src_dst = GetCrossHostTransferKey(0, 1, 10, 0);
  auto key_dst_src = GetCrossHostTransferKey(1, 0, 10, 0);
  EXPECT_EQ(key_src_dst.value(), 6634354603530518008LL);
  EXPECT_EQ(key_dst_src.value(), -2505766347492712790LL);
}

TEST(ProcessGroupTpuTest, GetCrossHostTransferKeyDistinctInputs) {
  auto base_key = GetCrossHostTransferKey(0, 1, 10, 0);
  EXPECT_EQ(base_key.value(), 6634354603530518008LL);

  // Different src
  EXPECT_EQ(GetCrossHostTransferKey(2, 1, 10, 0).value(),
            4435930954392163242LL);
  // Different dst
  EXPECT_EQ(GetCrossHostTransferKey(0, 2, 10, 0).value(),
            6648459279852910998LL);
  // Different tag
  EXPECT_EQ(GetCrossHostTransferKey(0, 1, 20, 0).value(),
            3626232797306906734LL);
  // Different tensor_index
  EXPECT_EQ(GetCrossHostTransferKey(0, 1, 10, 1).value(),
            -2239758845552025766LL);
}

}  // namespace
}  // namespace torch_tpu
