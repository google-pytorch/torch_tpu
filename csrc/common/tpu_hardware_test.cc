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
#include "csrc/common/tpu_hardware.h"

#include <ftw.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <fstream>
#include <ios>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"  // IWYU pragma: keep
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

// Scoped temporary directory mimicking /dev (including /dev/vfio,
// /dev/vfio/devices and /dev/accel), /dev_borg, /proc/devices and
// /sys/bus/pci/devices.
class ScopedFakeHardwareDir {
 public:
  ScopedFakeHardwareDir() {
    std::string tmpl =
        absl::StrCat(testing::TempDir(), "/fake_tpu_hardware_XXXXXX");
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* dir = mkdtemp(buf.data());
    ABSL_CHECK_NE(dir, nullptr);
    root_ = dir;
    dev_dir_ = absl::StrCat(root_, "/dev");
    vfio_dir_ = absl::StrCat(dev_dir_, "/vfio");
    vfio_devices_dir_ = absl::StrCat(vfio_dir_, "/devices");
    dev_borg_dir_ = absl::StrCat(root_, "/dev_borg");
    proc_devices_ = absl::StrCat(root_, "/proc_devices");
    pci_devices_dir_ = absl::StrCat(root_, "/pci_devices");
  }

  ~ScopedFakeHardwareDir() { RemoveTree(root_); }

  ScopedFakeHardwareDir(const ScopedFakeHardwareDir&) = delete;
  ScopedFakeHardwareDir& operator=(const ScopedFakeHardwareDir&) = delete;

  [[nodiscard]] const std::string& dev_dir() const { return dev_dir_; }
  [[nodiscard]] const std::string& vfio_dir() const { return vfio_dir_; }
  [[nodiscard]] const std::string& dev_borg_dir() const {
    return dev_borg_dir_;
  }
  [[nodiscard]] const std::string& proc_devices() const {
    return proc_devices_;
  }
  [[nodiscard]] const std::string& pci_devices_dir() const {
    return pci_devices_dir_;
  }

  // Adds a file named `name` to the fake /dev directory.
  void AddDevNode(std::string_view name) {
    MakeDir(dev_dir_);
    WriteFile(absl::StrCat(dev_dir_, "/", name), "");
  }

  // Adds a file named `name` to the fake /dev_borg directory.
  void AddDevBorgNode(std::string_view name) {
    MakeDir(dev_borg_dir_);
    WriteFile(absl::StrCat(dev_borg_dir_, "/", name), "");
  }

  // Adds an entry named `name` to the fake /dev directory that resolves to the
  // real device node `target`. Tests cannot create device nodes themselves.
  void AddDevNodeLinkTo(std::string_view name, std::string_view target) {
    MakeDir(dev_dir_);
    AddLink(absl::StrCat(dev_dir_, "/", name), target);
  }

  // Like AddDevNodeLinkTo(), for the fake /dev_borg directory.
  void AddDevBorgNodeLinkTo(std::string_view name, std::string_view target) {
    MakeDir(dev_borg_dir_);
    AddLink(absl::StrCat(dev_borg_dir_, "/", name), target);
  }

  // Writes the fake /proc/devices, listing `char_majors` as the registered
  // character device majors.
  void WriteProcDevices(absl::Span<const int> char_majors) {
    std::string contents = "Character devices:\n";
    for (const int major : char_majors) {
      absl::StrAppend(&contents, absl::StrFormat("%3d", major), " drv\n");
    }
    // A block device major must not count as a character device major.
    absl::StrAppend(&contents, "\nBlock devices:\n  7 loop\n");
    WriteFile(proc_devices_, contents);
  }

  // Adds a VFIO group node for `iommu_group` that resolves to the real device
  // node `target`, which defaults to /dev/null: a character device that any
  // process can read and write.
  void AddVfioGroup(std::string_view iommu_group,
                    std::string_view target = "/dev/null") {
    MakeDir(dev_dir_);
    MakeDir(vfio_dir_);
    AddLink(absl::StrCat(vfio_dir_, "/", iommu_group), target);
  }

  // Adds an empty regular file without any permissions at the VFIO group node
  // path for `iommu_group`, like the placeholders that some sandboxes put at
  // the paths of TPUs that they don't allocate to the process.
  void AddVfioGroupPlaceholder(std::string_view iommu_group) {
    MakeDir(dev_dir_);
    MakeDir(vfio_dir_);
    const std::string path = absl::StrCat(vfio_dir_, "/", iommu_group);
    WriteFile(path, "");
    ABSL_CHECK_EQ(chmod(path.c_str(), 0), 0);
  }

  // Creates the fake /dev/vfio/devices directory, which holds the VFIO
  // per-device (iommufd) nodes, and returns its path.
  const std::string& AddVfioDevicesDir() {
    MakeDir(dev_dir_);
    MakeDir(vfio_dir_);
    MakeDir(vfio_devices_dir_);
    return vfio_devices_dir_;
  }

  // Adds a VFIO per-device (iommufd) node named `name`, e.g. "vfio0".
  void AddVfioDeviceNode(std::string_view name) {
    WriteFile(absl::StrCat(AddVfioDevicesDir(), "/", name), "");
  }

  // Adds an entry named `name` to the fake /dev/accel directory that resolves
  // to the real device node `target`.
  void AddDevAccelDirNodeLinkTo(std::string_view name,
                                std::string_view target) {
    MakeDir(dev_dir_);
    const std::string accel_dir = absl::StrCat(dev_dir_, "/accel");
    MakeDir(accel_dir);
    AddLink(absl::StrCat(accel_dir, "/", name), target);
  }

  // Like AddDevAccelDirNodeLinkTo(), for the fake /dev_borg/accel directory.
  void AddDevBorgAccelDirNodeLinkTo(std::string_view name,
                                    std::string_view target) {
    MakeDir(dev_borg_dir_);
    const std::string accel_dir = absl::StrCat(dev_borg_dir_, "/accel");
    MakeDir(accel_dir);
    AddLink(absl::StrCat(accel_dir, "/", name), target);
  }

  // Adds a fake PCI device under `pci_devices_dir_`.
  void AddPciDevice(std::string_view address,
                    std::optional<std::string_view> vendor_id,
                    std::optional<std::string_view> device_id,
                    std::optional<std::string_view> iommu_group = std::nullopt,
                    bool vfio_exposed = true) {
    MakeDir(pci_devices_dir_);
    const std::string device_dir = absl::StrCat(pci_devices_dir_, "/", address);
    MakeDir(device_dir);
    if (vendor_id.has_value()) {
      WriteFile(absl::StrCat(device_dir, "/vendor"),
                absl::StrCat(*vendor_id, "\n"));
    }
    if (device_id.has_value()) {
      WriteFile(absl::StrCat(device_dir, "/device"),
                absl::StrCat(*device_id, "\n"));
    }
    if (iommu_group.has_value()) {
      AddLink(absl::StrCat(device_dir, "/iommu_group"),
              absl::StrCat(root_, "/iommu_groups/", *iommu_group));
      if (vfio_exposed) {
        AddVfioGroup(*iommu_group);
      }
    }
  }

 private:
  static void MakeDir(const std::string& path) {
    ABSL_CHECK(mkdir(path.c_str(), 0700) == 0 || errno == EEXIST);
  }

  static void AddLink(const std::string& path, std::string_view target) {
    ABSL_CHECK_EQ(symlink(std::string(target).c_str(), path.c_str()), 0);
  }

  static void WriteFile(const std::string& path, std::string_view contents) {
    std::ofstream file(path, std::ios::binary);
    file.write(contents.data(), contents.size());
    ABSL_CHECK(file.good());
  }

  static int RemoveEntry(const char* path, const struct stat* /*sb*/,
                         int /*typeflag*/, struct FTW* /*ftwbuf*/) {
    return remove(path);
  }

  static void RemoveTree(const std::string& root) {
    // FTW_DEPTH visits children first; FTW_PHYS avoids following symlinks.
    nftw(root.c_str(), &RemoveEntry, /*nopenfd=*/16, FTW_DEPTH | FTW_PHYS);
  }

  std::string root_;
  std::string dev_dir_;
  std::string vfio_dir_;
  std::string vfio_devices_dir_;
  std::string dev_borg_dir_;
  std::string proc_devices_;
  std::string pci_devices_dir_;
};

// Returns the major number of the character device `path`.
[[nodiscard]] int CharDeviceMajor(const char* path) {
  struct stat st;
  ABSL_CHECK_EQ(stat(path, &st), 0);
  ABSL_CHECK(S_ISCHR(st.st_mode));
  return static_cast<int>(major(st.st_rdev));
}

TEST(HasTpuDeviceNodesTest, ReturnsFalseWithoutDevDirs) {
  ScopedFakeHardwareDir fake;
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresUnrelatedNodes) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevNodeLinkTo("null", "/dev/null");
  // The VFIO container node exists whenever the VFIO module is loaded, with or
  // without any device bound to VFIO.
  fake.AddVfioGroup("vfio");
  fake.AddDevBorgNodeLinkTo("null", "/dev/null");
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsVfioGroupNode) {
  ScopedFakeHardwareDir fake;
  fake.AddVfioGroup("vfio");
  fake.AddVfioGroup("12");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsAccelNodeWithRegisteredMajor) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevNodeLinkTo("accel0", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsDevBorgAccelNodeWithRegisteredMajor) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevBorgNodeLinkTo("accel3", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresAccelNodeWithUnregisteredMajor) {
  ScopedFakeHardwareDir fake;
  // Some static /dev trees hold many accel nodes without a driver behind them.
  fake.WriteProcDevices({CharDeviceMajor("/dev/null") + 1});
  fake.AddDevNodeLinkTo("accel0", "/dev/null");
  fake.AddDevBorgNodeLinkTo("accel1", "/dev/null");
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresAccelEntryThatIsNotCharDevice) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevNode("accel0");
  fake.AddDevBorgNode("accel1");
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresAccelBlockDevice) {
  const char* block_device = nullptr;
  for (const char* const path : {"/dev/loop0", "/dev/sda", "/dev/vda",
                                 "/dev/nvme0n1", "/dev/ram0", "/dev/xda"}) {
    struct stat st;
    if (stat(path, &st) == 0 && S_ISBLK(st.st_mode)) {
      block_device = path;
      break;
    }
  }
  if (block_device == nullptr) {
    GTEST_SKIP() << "no block device found to link to";
  }
  struct stat st;
  ASSERT_EQ(stat(block_device, &st), 0);
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({static_cast<int>(major(st.st_rdev))});
  fake.AddDevNodeLinkTo("accel0", block_device);
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, ReturnsTrueForAccelNodeIfProcDevicesUnreadable) {
  ScopedFakeHardwareDir fake;
  // No fake /proc/devices file.
  fake.AddDevNodeLinkTo("accel0", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsVfioPerDeviceNode) {
  ScopedFakeHardwareDir fake;
  // The VFIO cdev (iommufd) interface exposes /dev/vfio/devices/vfioN instead
  // of, or in addition to, the group nodes.
  fake.AddVfioGroup("vfio");
  fake.AddVfioDeviceNode("vfio0");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresEmptyVfioDevicesDir) {
  ScopedFakeHardwareDir fake;
  fake.AddVfioDevicesDir();
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, ReturnsTrueWhenVfioDevicesDirCannotBeListed) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "root can list directories without read permission";
  }
  ScopedFakeHardwareDir fake;
  const std::string& vfio_devices_dir = fake.AddVfioDevicesDir();
  ASSERT_EQ(chmod(vfio_devices_dir.c_str(), 0), 0);
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
  ASSERT_EQ(chmod(vfio_devices_dir.c_str(), 0700), 0);
}

TEST(HasTpuDeviceNodesTest, DetectsAccelDirNodeWithRegisteredMajor) {
  ScopedFakeHardwareDir fake;
  // The kernel's compute accelerator subsystem creates /dev/accel/accelN.
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevAccelDirNodeLinkTo("accel0", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsDevBorgAccelDirNodeWithRegisteredMajor) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevBorgAccelDirNodeLinkTo("accel2", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresAccelDirNodeWithUnregisteredMajor) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null") + 1});
  fake.AddDevAccelDirNodeLinkTo("accel0", "/dev/null");
  fake.AddDevBorgAccelDirNodeLinkTo("accel1", "/dev/null");
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, ReturnsTrueForAccelDirNodeIfProcDevicesUnreadable) {
  ScopedFakeHardwareDir fake;
  // No fake /proc/devices file.
  fake.AddDevAccelDirNodeLinkTo("accel0", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, ReturnsTrueWhenDevDirCannotBeListed) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "root can list directories without read permission";
  }
  ScopedFakeHardwareDir fake;
  fake.AddDevNode("null");
  ASSERT_EQ(chmod(fake.dev_dir().c_str(), 0), 0);
  // Without listing the directory, the absence of TPU device nodes is unknown.
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
  ASSERT_EQ(chmod(fake.dev_dir().c_str(), 0700), 0);
}

TEST(TpuSpecTest, PredefinedSpecsMatchExpected) {
  constexpr int64_t kGiB = int64_t{1} << 30;
  constexpr int64_t kTFlops = 1'000'000'000'000LL;

  const TpuSpec& v5p = TpuSpec::V5p();
  EXPECT_EQ(v5p.name, "TPU v5p");
  EXPECT_EQ(v5p.devices_per_chip, 1);
  EXPECT_EQ(v5p.hbm_bytes_per_chip, 95 * kGiB);
  EXPECT_EQ(v5p.hbm_bytes_per_device(), 95 * kGiB);
  EXPECT_EQ(v5p.bf16_flops_per_chip, 459 * kTFlops);
  EXPECT_EQ(v5p.bf16_flops_per_device(), 459 * kTFlops);
  EXPECT_EQ(v5p.sparse_core_spec, (SparseCoreSpec{4, 8}));
  EXPECT_EQ(v5p.topologies.at(4), (TpuSpec::Topology{2, 2, 1}));

  const TpuSpec& v5e = TpuSpec::V5e();
  EXPECT_EQ(v5e.name, "TPU v5e");
  EXPECT_EQ(v5e.devices_per_chip, 1);
  EXPECT_EQ(v5e.hbm_bytes_per_chip, 16 * kGiB);
  EXPECT_EQ(v5e.bf16_flops_per_chip, 197 * kTFlops);
  EXPECT_EQ(v5e.sparse_core_spec, std::nullopt);
  EXPECT_EQ(v5e.topologies.at(8),
            (TT_IS_INTERNAL_TORCH_TPU ? TpuSpec::Topology{2, 2, 2}
                                      : TpuSpec::Topology{2, 4, 1}));

  const TpuSpec& v6e = TpuSpec::V6e();
  EXPECT_EQ(v6e.name, "TPU v6e");
  EXPECT_EQ(v6e.devices_per_chip, 1);
  EXPECT_EQ(v6e.hbm_bytes_per_chip, 32 * kGiB);
  EXPECT_EQ(v6e.bf16_flops_per_chip, 918 * kTFlops);
  EXPECT_EQ(v6e.sparse_core_spec, (SparseCoreSpec{2, 8}));
  EXPECT_EQ(v6e.topologies.at(8), (TpuSpec::Topology{2, 4, 1}));

  const TpuSpec& v7 = TpuSpec::V7();
  EXPECT_EQ(v7.name, "TPU v7");
  EXPECT_EQ(v7.devices_per_chip, 2);
  EXPECT_EQ(v7.hbm_bytes_per_chip, 192 * kGiB);
  EXPECT_EQ(v7.hbm_bytes_per_device(), 96 * kGiB);
  EXPECT_EQ(v7.bf16_flops_per_chip, 2307 * kTFlops);
  EXPECT_EQ(v7.bf16_flops_per_device(), (2307 * kTFlops) / 2);
  EXPECT_EQ(v7.sparse_core_spec, (SparseCoreSpec{2, 16}));
  EXPECT_EQ(v7.topologies.at(2), (TpuSpec::Topology{1, 1, 1, 2}));

  const TpuSpec& v8i = TpuSpec::V8i();
  EXPECT_EQ(v8i.name, "TPU v8i");
  EXPECT_EQ(v8i.devices_per_chip, 2);
  EXPECT_EQ(v8i.hbm_bytes_per_chip, 309'000'000'000LL);
  EXPECT_EQ(v8i.hbm_bytes_per_device(), 309'000'000'000LL / 2);
  EXPECT_EQ(v8i.bf16_flops_per_chip, 1101 * kTFlops);
  EXPECT_EQ(v8i.bf16_flops_per_device(), (1101 * kTFlops) / 2);
  EXPECT_EQ(v8i.sparse_core_spec, (SparseCoreSpec{1, 16}));
  EXPECT_EQ(v8i.topologies.at(2), (TpuSpec::Topology{1, 1, 1, 2}));

  const TpuSpec& v8t = TpuSpec::V8t();
  EXPECT_EQ(v8t.name, "TPU v8t");
  EXPECT_EQ(v8t.devices_per_chip, 1);
  EXPECT_EQ(v8t.hbm_bytes_per_chip, 231'000'000'000LL);
  EXPECT_EQ(v8t.bf16_flops_per_chip, 996'100'000'000'000LL);
  EXPECT_EQ(v8t.sparse_core_spec, (SparseCoreSpec{2, 16}));
  EXPECT_EQ(v8t.topologies.at(4), (TpuSpec::Topology{2, 2, 1}));
}

TEST(TpuSpecTest, SpecsByPciDeviceIdMatchesPredefinedSpecs) {
  const auto& map = TpuSpec::SpecsByPciDeviceId();
  EXPECT_EQ(map.size(), 6);
  EXPECT_EQ(map.at("0x0062"), &TpuSpec::V5p());
  EXPECT_EQ(map.at("0x0063"), &TpuSpec::V5e());
  EXPECT_EQ(map.at("0x006f"), &TpuSpec::V6e());
  EXPECT_EQ(map.at("0x0076"), &TpuSpec::V7());
  EXPECT_EQ(map.at("0x007c"), &TpuSpec::V8t());
  EXPECT_EQ(map.at("0x0083"), &TpuSpec::V8i());
}

TEST(TpuSpecTest, GetTopologyReturnsExpectedOrError) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:10:01.0", "0x1ae0", "0x0076", "0");
  fake.AddPciDevice("0000:10:01.1", "0x1ae0", "0x0076", "1");

  const absl::StatusOr<TpuSpec::Topology> auto_topology =
      TpuSpec::V7().GetTopology(std::nullopt, fake.pci_devices_dir(),
                                fake.vfio_dir());
  ASSERT_TRUE(auto_topology.ok());
  EXPECT_EQ(*auto_topology, (TpuSpec::Topology{1, 1, 1, 2}));

  const absl::StatusOr<TpuSpec::Topology> explicit_topology =
      TpuSpec::V7().GetTopology(4, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_TRUE(explicit_topology.ok());
  EXPECT_EQ(*explicit_topology, (TpuSpec::Topology{1, 2, 1, 2}));

  const absl::StatusOr<TpuSpec::Topology> bad_topology =
      TpuSpec::V7().GetTopology(3, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_FALSE(bad_topology.ok());
  EXPECT_EQ(bad_topology.status().code(), error::kInvalidArgument);
  EXPECT_EQ(bad_topology.status().message(),
            "No TPU topology found for count: 3");
}

TEST(GetTpuSpecTest, ReturnsSpecForAllSupportedGenerations) {
  struct TestCase {
    std::string_view device_id;
    const TpuSpec* expected;
  };
  const TestCase cases[] = {
      {"0x0062", &TpuSpec::V5p()}, {"0x0063", &TpuSpec::V5e()},
      {"0x006f", &TpuSpec::V6e()}, {"0x0076", &TpuSpec::V7()},
      {"0x007c", &TpuSpec::V8t()}, {"0x0083", &TpuSpec::V8i()},
  };
  for (const TestCase& tc : cases) {
    ScopedFakeHardwareDir fake;
    fake.AddPciDevice("0000:00:04.0", "0x1ae0", tc.device_id);
    EXPECT_EQ(TpuSpec::Get(fake.pci_devices_dir()), tc.expected)
        << tc.device_id;
  }
}

TEST(GetTpuSpecTest, DoesNotRequireVfioDeviceNode) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x0076",
                    /*iommu_group=*/std::nullopt, /*vfio_exposed=*/false);
  EXPECT_EQ(TpuSpec::Get(fake.pci_devices_dir()), &TpuSpec::V7());
}

TEST(GetTpuSpecTest, ReturnsFirstRecognizedSpecInSortedPciAddressOrder) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:01.0", "0x8086", "0x1234");
  fake.AddPciDevice("0000:00:05.0", "0x1ae0", "0x0062");
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x006f");
  EXPECT_EQ(TpuSpec::Get(fake.pci_devices_dir()), &TpuSpec::V6e());
}

TEST(GetTpuSpecTest, IgnoresUnsupportedOrMalformedPciDevices) {
  ScopedFakeHardwareDir fake;
  // Unsupported TPU v4 device ID.
  fake.AddPciDevice("0000:00:01.0", "0x1ae0", "0x005e");
  // Non-Google vendor with a TPU device ID.
  fake.AddPciDevice("0000:00:02.0", "0x8086", "0x006f");
  // Missing vendor or device files.
  fake.AddPciDevice("0000:00:03.0", std::nullopt, "0x006f");
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", std::nullopt);
  EXPECT_EQ(TpuSpec::Get(fake.pci_devices_dir()), nullptr);
}

TEST(GetTpuSpecTest, ReturnsNullptrWhenPciDirMissing) {
  ScopedFakeHardwareDir fake;
  EXPECT_EQ(TpuSpec::Get(fake.pci_devices_dir()), nullptr);
}

TEST(GetTpuDeviceCountAndTopologyTest, ReturnsZeroAndNulloptWhenPciDirMissing) {
  ScopedFakeHardwareDir fake;
  EXPECT_EQ(GetTpuDeviceCount(fake.pci_devices_dir(), fake.vfio_dir()), 0);
  const absl::StatusOr<std::optional<std::string>> topology =
      GetTpuTopology(std::nullopt, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_TRUE(topology.ok());
  EXPECT_EQ(*topology, std::nullopt);
}

TEST(GetTpuDeviceCountAndTopologyTest, CountsVfioExposedPciFunctions) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x006f", "0");
  fake.AddPciDevice("0000:00:05.0", "0x1ae0", "0x006f", "1");
  fake.AddPciDevice("0000:00:06.0", "0x1ae0", "0x006f", "2");
  fake.AddPciDevice("0000:00:07.0", "0x1ae0", "0x006f", "3");
  EXPECT_EQ(GetTpuDeviceCount(fake.pci_devices_dir(), fake.vfio_dir()), 4);
  const absl::StatusOr<std::optional<std::string>> topology =
      GetTpuTopology(std::nullopt, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_TRUE(topology.ok());
  EXPECT_EQ(*topology, "2,2,1");
}

TEST(GetTpuDeviceCountAndTopologyTest, CountsDualFunctionChipDevices) {
  ScopedFakeHardwareDir fake;
  // Each dual-core chip exposes 2 PCI functions with their own IOMMU groups.
  fake.AddPciDevice("0000:10:01.0", "0x1ae0", "0x0076", "0");
  fake.AddPciDevice("0000:10:01.1", "0x1ae0", "0x0076", "1");
  fake.AddPciDevice("0000:11:01.0", "0x1ae0", "0x0076", "2");
  fake.AddPciDevice("0000:11:01.1", "0x1ae0", "0x0076", "3");
  EXPECT_EQ(GetTpuDeviceCount(fake.pci_devices_dir(), fake.vfio_dir()), 4);
  const absl::StatusOr<std::optional<std::string>> topology =
      GetTpuTopology(std::nullopt, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_TRUE(topology.ok());
  EXPECT_EQ(*topology, "1,2,1,2");
}

TEST(GetTpuDeviceCountAndTopologyTest,
     IgnoresDevicesWithoutIommuGroupOrVfioNode) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x006f", "0",
                    /*vfio_exposed=*/true);
  fake.AddPciDevice("0000:00:05.0", "0x1ae0", "0x006f", "1",
                    /*vfio_exposed=*/false);
  fake.AddPciDevice("0000:00:06.0", "0x1ae0", "0x006f",
                    /*iommu_group=*/std::nullopt, /*vfio_exposed=*/false);
  fake.AddPciDevice("0000:00:07.0", "0x1ae0", "0x006f", "3",
                    /*vfio_exposed=*/true);
  EXPECT_EQ(GetTpuDeviceCount(fake.pci_devices_dir(), fake.vfio_dir()), 2);
  const absl::StatusOr<std::optional<std::string>> topology =
      GetTpuTopology(/*world_size=*/1, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_TRUE(topology.ok());
  EXPECT_EQ(*topology, "1,1,1");
}

TEST(GetTpuDeviceCountAndTopologyTest, IgnoresVfioGroupPlaceholders) {
  ScopedFakeHardwareDir fake;
  // A 1-chip sandbox on an 8-chip host has a real VFIO group node only
  // for its own chip, and placeholders for the other chips.
  for (int i = 0; i < 8; ++i) {
    const std::string group = absl::StrCat(100 + i);
    fake.AddPciDevice(absl::StrFormat("0000:00:%02x.0", i + 4), "0x1ae0",
                      "0x0063", group, /*vfio_exposed=*/i == 5);
    if (i != 5) {
      fake.AddVfioGroupPlaceholder(group);
    }
  }
  EXPECT_EQ(GetTpuDeviceCount(fake.pci_devices_dir(), fake.vfio_dir()), 1);
  const absl::StatusOr<std::optional<std::string>> topology =
      GetTpuTopology(std::nullopt, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_TRUE(topology.ok());
  EXPECT_EQ(*topology, "1,1,1");
}

TEST(GetTpuDeviceCountAndTopologyTest,
     ReturnsZeroAndNulloptWhenAllTpusLackVfioNode) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x006f", "0",
                    /*vfio_exposed=*/false);
  EXPECT_EQ(GetTpuDeviceCount(fake.pci_devices_dir(), fake.vfio_dir()), 0);
  const absl::StatusOr<std::optional<std::string>> topology =
      GetTpuTopology(std::nullopt, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_TRUE(topology.ok());
  EXPECT_EQ(*topology, std::nullopt);
}

TEST(GetTpuDeviceCountAndTopologyTest,
     ReturnsErrorForUnsupportedDeviceCountOrWorldSize) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x0062", "0");
  fake.AddPciDevice("0000:00:05.0", "0x1ae0", "0x0062", "1");
  fake.AddPciDevice("0000:00:06.0", "0x1ae0", "0x0062", "2");
  const absl::StatusOr<std::optional<std::string>> topology =
      GetTpuTopology(std::nullopt, fake.pci_devices_dir(), fake.vfio_dir());
  ASSERT_FALSE(topology.ok());
  EXPECT_EQ(topology.status().code(), error::kInvalidArgument);
  EXPECT_EQ(topology.status().message(), "No TPU topology found for count: 3");
}

TEST(GetSingleDeviceTpuBoundsTest, Returns3DBoundsWhenNoSupportedTpuIsPresent) {
  ScopedFakeHardwareDir fake;
  EXPECT_EQ(GetSingleDeviceTpuBounds(fake.pci_devices_dir()),
            (TpuSpec::Topology{1, 1, 1}));
  EXPECT_EQ(GetSingleDeviceTpuBoundsString(fake.pci_devices_dir()), "1,1,1");
  const std::string nonexistent_dir =
      absl::StrCat(fake.pci_devices_dir(), "/nonexistent");
  EXPECT_EQ(GetSingleDeviceTpuBounds(nonexistent_dir),
            (TpuSpec::Topology{1, 1, 1}));
  EXPECT_EQ(GetSingleDeviceTpuBoundsString(nonexistent_dir), "1,1,1");
}

TEST(GetSingleDeviceTpuBoundsTest,
     Returns3DBoundsForSingleDevicePerChipGenerations) {
  for (std::string_view device_id : {"0x0062", "0x0063", "0x006f", "0x007c"}) {
    ScopedFakeHardwareDir fake;
    fake.AddPciDevice("0000:00:04.0", "0x1ae0", device_id, "0",
                      /*vfio_exposed=*/false);
    EXPECT_EQ(GetSingleDeviceTpuBounds(fake.pci_devices_dir()),
              (TpuSpec::Topology{1, 1, 1}))
        << device_id;
    EXPECT_EQ(GetSingleDeviceTpuBoundsString(fake.pci_devices_dir()), "1,1,1")
        << device_id;
  }
}

TEST(GetSingleDeviceTpuBoundsTest,
     Returns4DBoundsForMultiDevicePerChipGenerations) {
  for (std::string_view device_id : {"0x0076", "0x0083"}) {
    ScopedFakeHardwareDir fake;
    fake.AddPciDevice("0000:00:04.0", "0x1ae0", device_id, "0",
                      /*vfio_exposed=*/false);
    EXPECT_EQ(GetSingleDeviceTpuBounds(fake.pci_devices_dir()),
              (TpuSpec::Topology{1, 1, 1, 1}))
        << device_id;
    EXPECT_EQ(GetSingleDeviceTpuBoundsString(fake.pci_devices_dir()), "1,1,1,1")
        << device_id;
  }
}

}  // namespace
}  // namespace torch_tpu
