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
#include "csrc/common/premapped_buffer_defaults.h"

#include <ftw.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/env_vars.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

constexpr int64_t kMiB = int64_t{1} << 20;
constexpr int64_t kGiB = int64_t{1} << 30;

// Creates a temporary directory tree for fake cgroup and PCI files, and removes
// it on destruction.
class ScopedFakeCgroup {
 public:
  ScopedFakeCgroup() {
    std::string buf = absl::StrCat(testing::TempDir(), "/fake_cgroup_XXXXXX");
    char* const dir = mkdtemp(buf.data());
    ABSL_CHECK_NE(dir, nullptr);
    root_ = dir;
    cgroup_path_ = absl::StrCat(root_, "/cgroup");
    v2_root_ = absl::StrCat(root_, "/sys/fs/cgroup");
    v1_borg_root_ = absl::StrCat(root_, "/dev/cgroup/memory");
    v1_sysfs_root_ = absl::StrCat(root_, "/sys/fs/cgroup/memory");
    pci_devices_dir_ = absl::StrCat(root_, "/pci_devices");
    vfio_dir_ = absl::StrCat(root_, "/dev/vfio");
  }

  ~ScopedFakeCgroup() {
    // FTW_DEPTH visits children first; FTW_PHYS avoids following symlinks.
    nftw(root_.c_str(), &RemoveEntry, /*nopenfd=*/16, FTW_DEPTH | FTW_PHYS);
  }

  ScopedFakeCgroup(const ScopedFakeCgroup&) = delete;
  ScopedFakeCgroup& operator=(const ScopedFakeCgroup&) = delete;

  [[nodiscard]] const std::string& pci_devices_dir() const {
    return pci_devices_dir_;
  }
  [[nodiscard]] const std::string& vfio_dir() const { return vfio_dir_; }

  // Writes `content` to the file at `rel` under the fake root, creating any
  // missing parent directories.
  void WriteFile(std::string_view rel, std::string_view content) const {
    const std::string path = absl::StrCat(root_, "/", rel);
    MakeParentDirs(path);
    std::ofstream ofs(path);
    ABSL_CHECK(ofs.is_open()) << "Failed to open " << path;
    ofs << content;
    ABSL_CHECK(ofs.good());
  }

  // Adds a fake TPU v7 PCI device at `address` whose IOMMU group
  // `iommu_group` is exposed through VFIO as /dev/null, a character device
  // that any process can read and write.
  void AddVfioExposedTpu(std::string_view address,
                         std::string_view iommu_group) const {
    const std::string device_rel = absl::StrCat("pci_devices/", address);
    WriteFile(absl::StrCat(device_rel, "/vendor"), "0x1ae0\n");
    WriteFile(absl::StrCat(device_rel, "/device"), "0x0076\n");
    AddLink(absl::StrCat(root_, "/", device_rel, "/iommu_group"),
            absl::StrCat(root_, "/iommu_groups/", iommu_group));
    const std::string vfio_node = absl::StrCat(vfio_dir_, "/", iommu_group);
    MakeParentDirs(vfio_node);
    AddLink(vfio_node, "/dev/null");
  }

  MemoryLimitProbePaths ProbePaths(
      std::optional<int64_t> physical_memory_bytes_override = int64_t{512} *
                                                              kGiB) const {
    return MemoryLimitProbePaths{
        .proc_self_cgroup = cgroup_path_,
        .cgroup_v2_root = v2_root_,
        .cgroup_v1_borg_root = v1_borg_root_,
        .cgroup_v1_sysfs_root = v1_sysfs_root_,
        .physical_memory_bytes_override = physical_memory_bytes_override,
    };
  }

 private:
  // Creates every missing directory between `root_` and `path`'s parent.
  void MakeParentDirs(const std::string& path) const {
    for (size_t pos = path.find('/', root_.size() + 1);
         pos != std::string::npos; pos = path.find('/', pos + 1)) {
      const std::string dir = path.substr(0, pos);
      ABSL_CHECK(mkdir(dir.c_str(), 0700) == 0 || errno == EEXIST)
          << "Failed to create " << dir;
    }
  }

  static void AddLink(const std::string& path, std::string_view target) {
    ABSL_CHECK_EQ(symlink(std::string(target).c_str(), path.c_str()), 0);
  }

  static int RemoveEntry(const char* path, const struct stat* /*sb*/,
                         int /*typeflag*/, struct FTW* /*ftwbuf*/) {
    return remove(path);
  }

  std::string root_;
  std::string cgroup_path_;
  std::string v2_root_;
  std::string v1_borg_root_;
  std::string v1_sysfs_root_;
  std::string pci_devices_dir_;
  std::string vfio_dir_;
};

TEST(ComputeDefaultPremappedBufferSizeTest, MatchesExpectedMemoryBudgets) {
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(std::nullopt, 1), 0);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(0, 1), 0);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(48 * kGiB, 0), 0);

  // Tight or shared multi-process containers (< 16 GiB/worker) stay at 0.
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(12 * kGiB, 1), 0);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(16 * kGiB - kMiB, 1), 0);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(24 * kGiB, 2), 0);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(24 * kGiB, 8), 0);

  // 16-31 GiB/worker (including 24 GiB 1-chip Forge) defaults to 1 GiB.
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(16 * kGiB, 1), 1 * kGiB);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(24 * kGiB, 1), 1 * kGiB);

  // 32-63 GiB/worker defaults to 2 GiB (e.g. TPU v5e at 48 GiB/worker).
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(32 * kGiB, 1), 2 * kGiB);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(48 * kGiB, 1), 2 * kGiB);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(384 * kGiB, 8), 2 * kGiB);

  // >= 64 GiB/worker saturates at the 4 GiB cap (e.g. TPU v6e / v7x).
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(64 * kGiB, 1), 4 * kGiB);
  EXPECT_EQ(ComputeDefaultPremappedBufferSizeBytes(960 * kGiB, 8), 4 * kGiB);
}

TEST(GetEffectiveMemoryLimitBytesTest, ReadsCgroupV2ParentLimitWhenLeafIsMax) {
  ScopedFakeCgroup fake;
  fake.WriteFile("cgroup", "0::/job/worker0\n");
  fake.WriteFile("sys/fs/cgroup/memory.max", "max\n");
  fake.WriteFile("sys/fs/cgroup/job/memory.max", absl::StrCat(24 * kGiB, "\n"));
  fake.WriteFile("sys/fs/cgroup/job/worker0/memory.max", "max\n");

  EXPECT_EQ(GetEffectiveMemoryLimitBytes(fake.ProbePaths()), 24 * kGiB);
}

TEST(GetEffectiveMemoryLimitBytesTest, ReadsCgroupV2ParentLimitBelowLeafLimit) {
  ScopedFakeCgroup fake;
  fake.WriteFile("cgroup", "0::/job/worker0\n");
  fake.WriteFile("sys/fs/cgroup/job/memory.max", absl::StrCat(16 * kGiB, "\n"));
  fake.WriteFile("sys/fs/cgroup/job/worker0/memory.max",
                 absl::StrCat(32 * kGiB, "\n"));

  EXPECT_EQ(GetEffectiveMemoryLimitBytes(fake.ProbePaths()), 16 * kGiB);
}

TEST(GetEffectiveMemoryLimitBytesTest, IgnoresCgroupPathWithParentTraversal) {
  ScopedFakeCgroup fake;
  fake.WriteFile("cgroup", "0::/../escape\n");
  fake.WriteFile("sys/fs/cgroup/memory.max", absl::StrCat(16 * kGiB, "\n"));
  // Resolving the `..` segment would read this smaller limit outside the
  // cgroup root.
  fake.WriteFile("sys/fs/escape/memory.max", absl::StrCat(4 * kGiB, "\n"));

  EXPECT_EQ(GetEffectiveMemoryLimitBytes(fake.ProbePaths()), 16 * kGiB);
}

TEST(GetEffectiveMemoryLimitBytesTest, HandlesNamespacedCgroupV2RootFallback) {
  ScopedFakeCgroup fake;
  fake.WriteFile("cgroup", "0::/host/outside/namespace\n");
  fake.WriteFile("sys/fs/cgroup/memory.max", absl::StrCat(16 * kGiB, "\n"));

  EXPECT_EQ(GetEffectiveMemoryLimitBytes(fake.ProbePaths()), 16 * kGiB);
}

TEST(GetEffectiveMemoryLimitBytesTest,
     ReadsBorgStyleCgroupV1AndIgnoresSentinels) {
  ScopedFakeCgroup fake;
  fake.WriteFile("cgroup", "4:cpuacct,memory:/actions/act_1\n");
  fake.WriteFile("dev/cgroup/memory/memory.limit_in_bytes",
                 "9223372036854771712\n");
  fake.WriteFile("dev/cgroup/memory/actions/memory.limit_in_bytes",
                 absl::StrCat(64 * kGiB, "\n"));
  fake.WriteFile("dev/cgroup/memory/actions/act_1/memory.limit_in_bytes",
                 absl::StrCat(24 * kGiB, "\n"));

  EXPECT_EQ(GetEffectiveMemoryLimitBytes(fake.ProbePaths()), 24 * kGiB);
}

TEST(GetEffectiveMemoryLimitBytesTest, CapsByPhysicalMemoryWhenSmaller) {
  ScopedFakeCgroup fake;
  fake.WriteFile("cgroup", "0::/\n");
  fake.WriteFile("sys/fs/cgroup/memory.max", absl::StrCat(128 * kGiB, "\n"));

  EXPECT_EQ(GetEffectiveMemoryLimitBytes(fake.ProbePaths(48 * kGiB)),
            48 * kGiB);
}

// EstimateLocalWorkerCount() reads memoized env vars, so each case runs in a
// subprocess via EXPECT_EXIT().
TEST(EstimateLocalWorkerCountDeathTest, SingleAndDistributedModes) {
  EXPECT_EXIT(
      {
        unsetenv(kWorldSizeEnvVar);
        ABSL_CHECK_EQ(EstimateLocalWorkerCount("/nonexistent", "/nonexistent"),
                      1);
        exit(0);
      },
      ExitedWithCode(0), "");

  EXPECT_EXIT(
      {
        setenv(kWorldSizeEnvVar, "16", 1);
        setenv(kNnodesEnvVar, "2", 1);
        unsetenv(kTorchTpuDevicesPerHostEnvVar);
        ABSL_CHECK_EQ(EstimateLocalWorkerCount("/nonexistent", "/nonexistent"),
                      8);
        exit(0);
      },
      ExitedWithCode(0), "");

  EXPECT_EXIT(
      {
        setenv(kWorldSizeEnvVar, "16", 1);
        setenv(kNnodesEnvVar, "4", 1);
        setenv(kTorchTpuDevicesPerHostEnvVar, "2", 1);
        ABSL_CHECK_EQ(EstimateLocalWorkerCount("/nonexistent", "/nonexistent"),
                      2);
        exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EstimateLocalWorkerCountDeathTest, RoundsUpNonDivisibleWorldSize) {
  EXPECT_EXIT(
      {
        setenv(kWorldSizeEnvVar, "10", 1);
        setenv(kNnodesEnvVar, "4", 1);
        unsetenv(kTorchTpuDevicesPerHostEnvVar);
        ABSL_CHECK_EQ(EstimateLocalWorkerCount("/nonexistent", "/nonexistent"),
                      3);
        exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EstimateLocalWorkerCountDeathTest, FallsBackToLocalTpuDeviceCount) {
  ScopedFakeCgroup fake;
  fake.AddVfioExposedTpu("0000:00:04.0", "0");
  fake.AddVfioExposedTpu("0000:00:05.0", "1");
  fake.AddVfioExposedTpu("0000:00:06.0", "2");
  fake.AddVfioExposedTpu("0000:00:07.0", "3");

  // Without TORCH_TPU_DEVICES_PER_HOST or NNODES, the worker count is the
  // number of local TPU devices, capped by WORLD_SIZE.
  EXPECT_EXIT(
      {
        setenv(kWorldSizeEnvVar, "16", 1);
        unsetenv(kNnodesEnvVar);
        unsetenv(kTorchTpuDevicesPerHostEnvVar);
        ABSL_CHECK_EQ(
            EstimateLocalWorkerCount(fake.pci_devices_dir(), fake.vfio_dir()),
            4);
        exit(0);
      },
      ExitedWithCode(0), "");

  EXPECT_EXIT(
      {
        setenv(kWorldSizeEnvVar, "2", 1);
        unsetenv(kNnodesEnvVar);
        unsetenv(kTorchTpuDevicesPerHostEnvVar);
        ABSL_CHECK_EQ(
            EstimateLocalWorkerCount(fake.pci_devices_dir(), fake.vfio_dir()),
            2);
        exit(0);
      },
      ExitedWithCode(0), "");

  // Without any local TPU devices, all WORLD_SIZE workers are assumed local.
  EXPECT_EXIT(
      {
        setenv(kWorldSizeEnvVar, "16", 1);
        unsetenv(kNnodesEnvVar);
        unsetenv(kTorchTpuDevicesPerHostEnvVar);
        ABSL_CHECK_EQ(EstimateLocalWorkerCount("/nonexistent", "/nonexistent"),
                      16);
        exit(0);
      },
      ExitedWithCode(0), "");
}

// The default is computed during PjRt initialization, so it must not memoize
// env vars that process group creation reads later.
TEST(EstimateLocalWorkerCountDeathTest, DoesNotMemoizeEnvVars) {
  EXPECT_EXIT(
      {
        setenv(kWorldSizeEnvVar, "8", 1);
        setenv(kTorchTpuDevicesPerHostEnvVar, "8", 1);
        ABSL_CHECK_EQ(EstimateLocalWorkerCount("/nonexistent", "/nonexistent"),
                      8);
        setenv(kTorchTpuDevicesPerHostEnvVar, "4", 1);
        const std::optional<int> devices_per_host =
            GetIntegerEnvOnce<kTorchTpuDevicesPerHostEnvVar>();
        ABSL_CHECK(devices_per_host.has_value() && *devices_per_host == 4);
        exit(0);
      },
      ExitedWithCode(0), "");
}

}  // namespace
}  // namespace torch_tpu
