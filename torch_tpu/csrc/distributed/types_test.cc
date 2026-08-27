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

#include "torch_tpu/csrc/distributed/types.h"

#include "gtest/gtest.h"
#include "torch_tpu/csrc/common/fingerprint_utils.h"

namespace torch_tpu {
namespace {

TEST(DeviceGroupListTest, SubgroupsFingerprintGoldenValues) {
  DeviceGroupList dgl0;
  EXPECT_EQ(Fingerprint(dgl0), 0ULL);

  DeviceGroupList dgl1({{0, 1, 2, 3}});
  EXPECT_EQ(Fingerprint(dgl1), 5550725823952181179ULL);

  DeviceGroupList dgl2({{0, 1}, {2, 3}});
  EXPECT_EQ(Fingerprint(dgl2), 14421527645074699692ULL);
}

TEST(DeviceGroupListTest, Normalization) {
  DeviceGroupList dgl1({{0, 1}, {2, 3}});

  // Test normalization of subgroup order.
  DeviceGroupList dgl2({{2, 3}, {0, 1}});
  EXPECT_EQ(dgl2, dgl1);
  EXPECT_EQ(Fingerprint(dgl2), Fingerprint(dgl1));

  // Test deduplication of identical subgroups.
  DeviceGroupList dgl3({{2, 3}, {0, 1}, {2, 3}});
  EXPECT_EQ(dgl3, dgl1);
  EXPECT_EQ(Fingerprint(dgl3), Fingerprint(dgl1));

  // Test that inner device order is preserved.
  DeviceGroupList dgl4({{1, 0}, {2, 3}});
  EXPECT_NE(dgl4, dgl1);
}

}  // namespace
}  // namespace torch_tpu
