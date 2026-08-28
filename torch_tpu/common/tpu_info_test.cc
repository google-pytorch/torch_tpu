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

#include "torch_tpu/common/tpu_info.h"

#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::HasSubstr;

TEST(TpuInfoTest, ChipVersionFromDeviceKind) {
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU v2"), ChipVersion::kTPUv2);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU v3"), ChipVersion::kTPUv3);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU v4"), ChipVersion::kTPUv4);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU v4 lite"), ChipVersion::kTPUv4i);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU v5e"), ChipVersion::kTPUv5e);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU v5 lite"), ChipVersion::kTPUv5e);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU v5p"), ChipVersion::kTPUv5p);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU v6e"), ChipVersion::kTPUv6e);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU7"), ChipVersion::kTPU7);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU7x"), ChipVersion::kTPU7x);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU8i"), ChipVersion::kTPU8i);
  EXPECT_EQ(ChipVersionFromDeviceKind("TPU8t"), ChipVersion::kTPU8t);
  EXPECT_EQ(ChipVersionFromDeviceKind("unknown"), std::nullopt);
}

TEST(TpuInfoTest, GetTpuInfoForV5e) {
  const std::optional<TpuInfo> info = GetTpuInfoForChip(ChipVersion::kTPUv5e);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->chip_version, ChipVersion::kTPUv5e);
  EXPECT_EQ(info->generation, 5);
  EXPECT_EQ(info->num_cores, 1);
  EXPECT_EQ(info->num_lanes, 128);
  EXPECT_EQ(info->num_sublanes, 8);
  EXPECT_EQ(info->mxu_column_size, 128);
  EXPECT_EQ(info->num_mxus, 4);
  EXPECT_EQ(info->num_accumulators, 0);
  EXPECT_GT(info->vmem_capacity_bytes, 0);
  EXPECT_GT(info->hbm_capacity_bytes, 0);
  EXPECT_GT(info->mem_bw_bytes_per_second, 0);
  EXPECT_GT(info->bf16_ops_per_second, 0);
  EXPECT_TRUE(info->is_lite);
  EXPECT_FALSE(info->is_split_chip);
  EXPECT_FALSE(info->is_megacore);
}

TEST(TpuInfoTest, GetTpuInfoForV4) {
  const std::optional<TpuInfo> info = GetTpuInfoForChip(ChipVersion::kTPUv4);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->generation, 4);
  EXPECT_EQ(info->num_cores, 1);
  EXPECT_TRUE(info->is_split_chip);
  EXPECT_FALSE(info->is_megacore);
}

TEST(TpuInfoTest, GetTpuInfoByDeviceKind) {
  const std::optional<TpuInfo> info_v6e = GetTpuInfo("TPU v6e");
  ASSERT_TRUE(info_v6e.has_value());
  EXPECT_EQ(info_v6e->chip_version, ChipVersion::kTPUv6e);
  EXPECT_EQ(info_v6e->generation, 6);
  EXPECT_EQ(info_v6e->mxu_column_size, 256);
  EXPECT_EQ(info_v6e->num_mxus, 2);

  const std::optional<TpuInfo> info_v5p = GetTpuInfo("TPU v5p");
  ASSERT_TRUE(info_v5p.has_value());
  EXPECT_EQ(info_v5p->chip_version, ChipVersion::kTPUv5p);
  EXPECT_EQ(info_v5p->generation, 5);

  EXPECT_EQ(GetTpuInfo("unknown"), std::nullopt);
}

TEST(TpuInfoTest, OStreamOverload) {
  const std::optional<TpuInfo> info = GetTpuInfoForChip(ChipVersion::kTPUv5e);
  ASSERT_TRUE(info.has_value());

  std::ostringstream ss;
  ss << *info;
  const std::string str = ss.str();

  // Verify multi-line indented format with one entry per line
  EXPECT_THAT(str, HasSubstr("TpuInfo {\n"));
  EXPECT_THAT(str, HasSubstr("  chip_version: v5e,\n"));
  EXPECT_THAT(str, HasSubstr("  generation: 5,\n"));
  EXPECT_THAT(str, HasSubstr("  num_cores: 1,\n"));
  EXPECT_THAT(str, HasSubstr("  vmem_capacity_bytes: "));
  EXPECT_THAT(str, HasSubstr("  sparse_core: N/A,\n"));
  EXPECT_THAT(str, HasSubstr("  is_lite: true,\n"));
  EXPECT_THAT(str, HasSubstr("\n}"));

  std::ostringstream chip_ss;
  chip_ss << ChipVersion::kTPUv6e;
  EXPECT_EQ(chip_ss.str(), "v6e");

  SparseCoreInfo sc{
      .num_cores = 4,
      .num_subcores = 2,
      .num_lanes = 16,
      .vmem_capacity_bytes = 1024,
      .dma_granule_size_bytes = 64,
  };
  std::ostringstream sc_ss;
  sc_ss << sc;
  EXPECT_THAT(sc_ss.str(), HasSubstr("SparseCoreInfo {\n"));
  EXPECT_THAT(sc_ss.str(), HasSubstr("  num_cores: 4,\n"));
}

}  // namespace
}  // namespace torch_tpu
