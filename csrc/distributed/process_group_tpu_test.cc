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

#include <atomic>
#include <chrono>  // NOLINT - needed for c10d::Store timeouts
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>  // NOLINT - needed for publish/read concurrency tests
#include <memory>
#include <mutex>  // NOLINT - models a store connection lock
#include <string>
#include <string_view>
#include <thread>  // NOLINT - needed for async send test thread
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/error_utils.h"
#include "csrc/distributed/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch/csrc/distributed/c10d/FileStore.hpp"
#include "torch/csrc/distributed/c10d/HashStore.hpp"
#include "torch/csrc/distributed/c10d/PrefixStore.hpp"
#include "torch/csrc/distributed/c10d/Store.hpp"
#include "torch/csrc/distributed/c10d/TCPStore.hpp"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {
namespace {

// Counts calls to the world store. HashStore never calls these virtual methods
// internally (multiGet does not call get), so each counter sees one increment
// per caller request.
class CountingHashStore : public c10d::HashStore {
 public:
  void set(const std::string& key, const std::vector<uint8_t>& data) override {
    set_calls_.fetch_add(1, std::memory_order_relaxed);
    c10d::HashStore::set(key, data);
  }

  std::vector<uint8_t> get(const std::string& key) override {
    get_calls_.fetch_add(1, std::memory_order_relaxed);
    return c10d::HashStore::get(key);
  }

  int64_t add(const std::string& key, int64_t value) override {
    if (value > 0) {
      add_increment_calls_.fetch_add(1, std::memory_order_relaxed);
    }
    return c10d::HashStore::add(key, value);
  }

  bool check(const std::vector<std::string>& keys) override {
    check_calls_.fetch_add(1, std::memory_order_relaxed);
    return c10d::HashStore::check(keys);
  }

  void wait(const std::vector<std::string>& keys) override {
    wait_calls_.fetch_add(1, std::memory_order_relaxed);
    c10d::HashStore::wait(keys);
  }

  void wait(const std::vector<std::string>& keys,
            const std::chrono::milliseconds& timeout) override {
    wait_calls_.fetch_add(1, std::memory_order_relaxed);
    c10d::HashStore::wait(keys, timeout);
  }

  void multiSet(const std::vector<std::string>& keys,
                const std::vector<std::vector<uint8_t>>& values) override {
    multi_set_calls_.fetch_add(1, std::memory_order_relaxed);
    c10d::HashStore::multiSet(keys, values);
  }

  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override {
    multi_get_calls_.fetch_add(1, std::memory_order_relaxed);
    return c10d::HashStore::multiGet(keys);
  }

  int64_t set_calls() const {
    return set_calls_.load(std::memory_order_relaxed);
  }
  int64_t get_calls() const {
    return get_calls_.load(std::memory_order_relaxed);
  }
  int64_t add_increment_calls() const {
    return add_increment_calls_.load(std::memory_order_relaxed);
  }
  int64_t check_calls() const {
    return check_calls_.load(std::memory_order_relaxed);
  }
  int64_t wait_calls() const {
    return wait_calls_.load(std::memory_order_relaxed);
  }
  int64_t multi_set_calls() const {
    return multi_set_calls_.load(std::memory_order_relaxed);
  }
  int64_t multi_get_calls() const {
    return multi_get_calls_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<int64_t> set_calls_{0};
  std::atomic<int64_t> get_calls_{0};
  std::atomic<int64_t> add_increment_calls_{0};
  std::atomic<int64_t> check_calls_{0};
  std::atomic<int64_t> wait_calls_{0};
  std::atomic<int64_t> multi_set_calls_{0};
  std::atomic<int64_t> multi_get_calls_{0};
};

struct AsyncSendWaiter {
  std::atomic<bool> done{false};
  std::exception_ptr error;

  void MarkDone() { done.store(true, std::memory_order_release); }
  void SetError(std::exception_ptr eptr) {
    error = std::move(eptr);
    done.store(true, std::memory_order_release);
  }
  void WaitAndThrow() {
    while (!done.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (error) {
      std::rethrow_exception(error);
    }
  }
};

// Arguments for one rank's AllGatherViaNodeLocalStore call.
struct AllGatherRankSpec {
  c10d::Store* local_store = nullptr;  // nullptr: the rank is its own node.
  int64_t local_rank = 0;
  int64_t local_world_size = 1;
  std::string my_key;
  std::string my_value;
};

// Outcome of one rank's AllGatherViaNodeLocalStore call.
struct AllGatherRankResult {
  absl::Status status;
  std::vector<std::string> values;
};

// Runs AllGatherViaNodeLocalStore for every spec on its own thread.
std::vector<AllGatherRankResult> RunAllGather(
    c10d::Store* world_store, const std::vector<AllGatherRankSpec>& specs,
    const std::vector<std::string>& peer_keys) {
  std::vector<AllGatherRankResult> results(specs.size());
  std::vector<std::thread> threads;
  threads.reserve(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    threads.emplace_back([world_store, &specs, &peer_keys, &results, i]() {
      const AllGatherRankSpec& spec = specs[i];
      absl::StatusOr<std::vector<std::vector<uint8_t>>> raw_values =
          AllGatherViaNodeLocalStore(world_store, spec.local_store,
                                     spec.local_rank, spec.local_world_size,
                                     spec.my_key, spec.my_value, peer_keys);
      results[i].status = raw_values.status();
      if (raw_values.ok()) {
        for (const auto& bytes : *raw_values) {
          results[i].values.emplace_back(bytes.begin(), bytes.end());
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  return results;
}

std::string TestKey(int64_t r) { return absl::StrFormat("k:%d", r); }
std::string TestValue(int64_t r) { return absl::StrFormat("value_for_%d", r); }

std::vector<std::string> TestKeys(int64_t n) {
  std::vector<std::string> keys(n);
  for (int64_t r = 0; r < n; ++r) {
    keys[r] = TestKey(r);
  }
  return keys;
}

std::vector<std::string> TestValues(int64_t n) {
  std::vector<std::string> values(n);
  for (int64_t r = 0; r < n; ++r) {
    values[r] = TestValue(r);
  }
  return values;
}

constexpr std::chrono::seconds kTestStoreTimeout{10};

// Returns a FileStore for one rank of a node with `num_workers` ranks.
c10::intrusive_ptr<c10d::FileStore> MakeNodeLocalStore(std::string_view name,
                                                       int64_t node,
                                                       int64_t num_workers) {
  auto store = c10::make_intrusive<c10d::FileStore>(
      absl::StrFormat("%s/%s_node_%d.store", testing::TempDir(), name, node),
      static_cast<int>(num_workers));
  store->setTimeout(kTestStoreTimeout);
  return store;
}

// Node layout for MakeLayoutSpecs.
struct NodeLayout {
  int64_t size = 1;
  // When false, each of the node's ranks has no local store and acts as a
  // 1-rank node.
  bool use_local_store = true;
};

// Builds one spec per rank for ranks laid out node by node. `world_ranks[i]` is
// the world rank of the i-th rank in node order; it selects the rank's key and
// value.
std::vector<AllGatherRankSpec> MakeLayoutSpecs(
    std::string_view name, const std::vector<NodeLayout>& nodes,
    const std::vector<int64_t>& world_ranks,  // INT_VEC_OK
    std::vector<c10::intrusive_ptr<c10d::FileStore>>& local_stores) {
  std::vector<AllGatherRankSpec> specs;
  size_t idx = 0;
  for (size_t node = 0; node < nodes.size(); ++node) {
    for (int64_t lr = 0; lr < nodes[node].size; ++lr, ++idx) {
      const int64_t world_rank = world_ranks[idx];
      AllGatherRankSpec spec{.my_key = TestKey(world_rank),
                             .my_value = TestValue(world_rank)};
      if (nodes[node].use_local_store) {
        local_stores.push_back(MakeNodeLocalStore(
            name, static_cast<int64_t>(node), nodes[node].size));
        spec.local_store = local_stores.back().get();
        spec.local_rank = lr;
        spec.local_world_size = nodes[node].size;
      }
      specs.push_back(std::move(spec));
    }
  }
  return specs;
}

void ExpectAllSucceeded(const std::vector<AllGatherRankResult>& results,
                        const std::vector<std::string>& expected) {
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_TRUE(results[i].status.ok())
        << "rank index " << i << " failed: " << results[i].status;
    EXPECT_EQ(results[i].values, expected) << "mismatch on rank index " << i;
  }
}

void ExpectAllFailedWith(const std::vector<AllGatherRankResult>& results,
                         std::string_view substr) {
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_FALSE(results[i].status.ok()) << "rank index " << i << " succeeded";
    EXPECT_THAT(results[i].status.message(), testing::HasSubstr(substr))
        << "rank index " << i;
  }
}

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

TEST(ProcessGroupTpuTest,
     PermuteDeviceAssignmentForSubgroupsMultiSlicePermutedLauncherRanks) {
  // Simulate a 2-slice (16-device) world where launcher ranks are permuted
  // differently across hosts on Slice 0 vs Slice 1:
  //   Slice 0 (world ranks 0..7):
  //     ranks 0..3 -> physical devices 4, 5, 6, 7 (Host 1)
  //     ranks 4..7 -> physical devices 0, 1, 2, 3 (Host 0)
  //   Slice 1 (world ranks 8..15):
  //     ranks 8..11  -> physical devices 9, 8, 11, 10 (Host 2)
  //     ranks 12..15 -> physical devices 13, 12, 15, 14 (Host 3)
  const std::vector<int64_t>  // INT_VEC_OK
      rank_to_device_id = {4, 5, 6,  7,  0,  1,  2,  3,
                           9, 8, 11, 10, 13, 12, 15, 14};

  // topology_aware_mesh orders each 8-rank slice subgroup by physical TPU
  // coordinate traversal order (devices 0..7 on Slice 0, devices 8..15 on
  // Slice 1):
  //   Slice 0 subgroup world ranks: {4, 5, 6, 7, 0, 1, 2, 3}
  //   Slice 1 subgroup world ranks: {9, 8, 11, 10, 13, 12, 15, 14}
  const DeviceGroupList subgroup_member_world_ranks({
      {4, 5, 6, 7, 0, 1, 2, 3},
      {9, 8, 11, 10, 13, 12, 15, 14},
  });

  std::vector<int64_t> device_assignment;  // INT_VEC_OK
  DeviceGroupList replica_groups;
  PermuteDeviceAssignmentForSubgroups(subgroup_member_world_ranks,
                                      rank_to_device_id, device_assignment,
                                      replica_groups);

  const std::vector<int64_t>  // INT_VEC_OK
      expected_da = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  EXPECT_EQ(device_assignment, expected_da);

  const DeviceGroupList expected_replica_groups({
      {0, 1, 2, 3, 4, 5, 6, 7},
      {8, 9, 10, 11, 12, 13, 14, 15},
  });
  EXPECT_EQ(replica_groups, expected_replica_groups);
}

TEST(ProcessGroupTpuTest, AllGatherViaNodeLocalStorePublishesOneBlobPerNode) {
  constexpr int64_t kNumNodes = 2;
  constexpr int64_t kLocalWorldSize = 4;
  constexpr int64_t kWorldSize = kNumNodes * kLocalWorldSize;
  CountingHashStore world_store;
  world_store.setTimeout(kTestStoreTimeout);
  std::vector<c10::intrusive_ptr<c10d::FileStore>> local_stores;
  // World ranks are permuted across nodes and local ranks to verify that keys
  // are gathered regardless of node assignment.
  const std::vector<AllGatherRankSpec> specs = MakeLayoutSpecs(
      "nlag_one_blob",
      std::vector<NodeLayout>(kNumNodes, NodeLayout{.size = kLocalWorldSize}),
      {5, 1, 7, 3, 4, 0, 6, 2}, local_stores);

  const std::vector<AllGatherRankResult> results =
      RunAllGather(&world_store, specs, TestKeys(kWorldSize));

  ExpectAllSucceeded(results, TestValues(kWorldSize));
  // Each node leader publishes one blob; the collector publishes the table and
  // the other leader reads it. Individual peer keys are never written.
  EXPECT_TRUE(world_store.check(
      {absl::StrFormat("nlag:%s:%d:table", TestKey(0), kWorldSize)}));
  EXPECT_FALSE(world_store.check({TestKey(0)}));
  EXPECT_EQ(world_store.multi_set_calls(), 0);
  EXPECT_EQ(world_store.set_calls(), kNumNodes + 1);
  EXPECT_EQ(world_store.get_calls(), kNumNodes - 1);
  EXPECT_EQ(world_store.add_increment_calls(), kNumNodes);
  EXPECT_GE(world_store.multi_get_calls(), 1);
  EXPECT_LE(world_store.multi_get_calls(), kNumNodes);
}

TEST(ProcessGroupTpuTest, GatherSubgroupsViaNodeLocalStoreCrossNodeSubgroups) {
  constexpr int64_t kNumNodes = 2;
  constexpr int64_t kLocalWorldSize = 4;
  constexpr int64_t kWorldSize = kNumNodes * kLocalWorldSize;
  constexpr int64_t kPgSize = 2;

  auto world_store = c10::make_intrusive<c10d::HashStore>();
  const std::string base_dir = testing::TempDir();
  std::vector<c10::intrusive_ptr<c10d::FileStore>> rank_local_stores;
  rank_local_stores.reserve(kWorldSize);
  for (int64_t idx = 0; idx < kWorldSize; ++idx) {
    const int64_t node_rank = idx / kLocalWorldSize;
    rank_local_stores.push_back(c10::make_intrusive<c10d::FileStore>(
        absl::StrFormat("%s/cross_node_%d.store", base_dir, node_rank),
        kLocalWorldSize));
  }

  // Node 0 has world ranks {5, 1, 7, 3}; Node 1 has world ranks {4, 0, 6, 2}.
  const std::vector<int64_t>  // INT_VEC_OK
      world_ranks = {5, 1, 7, 3, 4, 0, 6, 2};
  const std::vector<int64_t>  // INT_VEC_OK
      global_rank_to_device_id = {4, 5, 6, 7, 0, 1, 2, 3};
  // Physical-topology position of each world rank. Deliberately disagrees
  // with device-id order so the test proves subgroups are sorted by topology
  // order, not by device id.
  const std::vector<int64_t>  // INT_VEC_OK
      global_rank_to_topo_order = {6, 1, 4, 0, 7, 2, 3, 5};

  // 4 cross-node subgroups of size 2 (spanning Node 0 and Node 1, with
  // non-sorted subgroup rank order to also test sort_ranks=False preservation):
  //   "group_a": [0, 5] -> leader 0: device_id=4, topo_order=6
  //   "group_b": [1, 4] -> leader 1: device_id=5, topo_order=1
  //   "group_c": [6, 7] -> leader 6: device_id=2, topo_order=3
  //   "group_d": [3, 2] -> leader 3: device_id=7, topo_order=0
  struct RankSubgroupSpec {
    std::string group_name;
    int64_t pg_rank;
    DeviceGroup expected_local_subgroup;
  };
  const std::vector<RankSubgroupSpec> specs_by_world_rank = {
      {"group_a", 0, {0, 5}},  // world_rank 0
      {"group_b", 0, {1, 4}},  // world_rank 1
      {"group_d", 1, {3, 2}},  // world_rank 2
      {"group_d", 0, {3, 2}},  // world_rank 3
      {"group_b", 1, {1, 4}},  // world_rank 4
      {"group_a", 1, {0, 5}},  // world_rank 5
      {"group_c", 0, {6, 7}},  // world_rank 6
      {"group_c", 1, {6, 7}},  // world_rank 7
  };

  std::vector<DeviceGroup> out_member_world_ranks(kWorldSize);
  std::vector<DeviceGroupList> out_subgroup_member_world_ranks(kWorldSize);
  std::vector<std::thread> threads;
  threads.reserve(kWorldSize);

  for (int64_t idx = 0; idx < kWorldSize; ++idx) {
    const int64_t local_rank = idx % kLocalWorldSize;
    const int64_t world_rank = world_ranks[idx];
    const RankSubgroupSpec& spec = specs_by_world_rank[world_rank];

    auto level1 =
        c10::make_intrusive<c10d::PrefixStore>("default_pg", world_store);
    auto level2 =
        c10::make_intrusive<c10d::PrefixStore>(spec.group_name, level1);
    auto pg_store = c10::make_intrusive<c10d::PrefixStore>("tpu", level2);

    threads.emplace_back([pg_store, &rank_local_stores,
                          &global_rank_to_device_id, &global_rank_to_topo_order,
                          &out_member_world_ranks,
                          &out_subgroup_member_world_ranks, idx, local_rank,
                          world_rank, pg_rank = spec.pg_rank]() {
      const absl::Status status = GatherSubgroupsViaNodeLocalStore(
          pg_store.get(), rank_local_stores[idx].get(), local_rank,
          kLocalWorldSize, pg_rank, kPgSize, world_rank,
          global_rank_to_device_id, global_rank_to_topo_order,
          /*gather_id=*/42, out_member_world_ranks[world_rank],
          out_subgroup_member_world_ranks[world_rank]);
      EXPECT_TRUE(status.ok()) << status;
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // Subgroups sorted by topo order of group[0]:
  //   {3, 2} (topo 0), {1, 4} (topo 1), {6, 7} (topo 3), {0, 5} (topo 6).
  // Sorting by device id instead would give
  //   {6, 7} (dev 2), {0, 5} (dev 4), {1, 4} (dev 5), {3, 2} (dev 7).
  const std::vector<DeviceGroup> expected_groups = {
      {3, 2},
      {1, 4},
      {6, 7},
      {0, 5},
  };
  // device_assignment = rank_to_device_id[r] for r in flattened groups
  // {3, 2, 1, 4, 6, 7, 0, 5} -> {7, 6, 5, 0, 2, 3, 4, 1}.
  const std::vector<int64_t>  // INT_VEC_OK
      expected_da = {7, 6, 5, 0, 2, 3, 4, 1};
  const std::vector<DeviceGroup> expected_replica_groups = {
      {0, 1},
      {2, 3},
      {4, 5},
      {6, 7},
  };
  for (int64_t r = 0; r < kWorldSize; ++r) {
    EXPECT_EQ(out_member_world_ranks[r],
              specs_by_world_rank[r].expected_local_subgroup)
        << "member_world_ranks mismatch on world_rank " << r;
    EXPECT_EQ(out_subgroup_member_world_ranks[r].groups(), expected_groups)
        << "subgroup_member_world_ranks mismatch on world_rank " << r;

    std::vector<int64_t> device_assignment;  // INT_VEC_OK
    DeviceGroupList replica_groups;
    PermuteDeviceAssignmentForSubgroups(out_subgroup_member_world_ranks[r],
                                        global_rank_to_device_id,
                                        device_assignment, replica_groups);
    EXPECT_EQ(device_assignment, expected_da)
        << "device_assignment mismatch on world_rank " << r;
    EXPECT_EQ(replica_groups.groups(), expected_replica_groups)
        << "replica_groups mismatch on world_rank " << r;
  }
}

TEST(ProcessGroupTpuTest,
     GatherSubgroupsViaNodeLocalStoreWorldSizedPreservesOrder) {
  constexpr int64_t kNumNodes = 2;
  constexpr int64_t kLocalWorldSize = 4;
  constexpr int64_t kWorldSize = kNumNodes * kLocalWorldSize;
  constexpr int64_t kPgSize = kWorldSize;

  auto world_store = c10::make_intrusive<c10d::HashStore>();
  const std::string base_dir = testing::TempDir();
  std::vector<c10::intrusive_ptr<c10d::FileStore>> rank_local_stores;
  rank_local_stores.reserve(kWorldSize);
  for (int64_t idx = 0; idx < kWorldSize; ++idx) {
    const int64_t node_rank = idx / kLocalWorldSize;
    rank_local_stores.push_back(c10::make_intrusive<c10d::FileStore>(
        absl::StrFormat("%s/world_order_node_%d.store", base_dir, node_rank),
        kLocalWorldSize));
  }

  const std::vector<int64_t>  // INT_VEC_OK
      global_rank_to_device_id = {4, 5, 6, 7, 0, 1, 2, 3};
  // A single world-sized subgroup is never re-sorted, so the topology order
  // has no effect here.
  const std::vector<int64_t>  // INT_VEC_OK
      global_rank_to_topo_order = {4, 5, 6, 7, 0, 1, 2, 3};
  const DeviceGroup expected_subgroup = {4, 5, 6, 7, 0, 1, 2, 3};
  // Map world_rank -> pg_rank in the permuted world-sized group.
  const std::vector<int64_t>  // INT_VEC_OK
      pg_rank_by_world_rank = {4, 5, 6, 7, 0, 1, 2, 3};

  std::vector<DeviceGroup> out_member_world_ranks(kWorldSize);
  std::vector<DeviceGroupList> out_subgroup_member_world_ranks(kWorldSize);
  std::vector<std::thread> threads;
  threads.reserve(kWorldSize);

  for (int64_t idx = 0; idx < kWorldSize; ++idx) {
    const int64_t local_rank = idx % kLocalWorldSize;
    const int64_t world_rank = idx;
    const int64_t pg_rank = pg_rank_by_world_rank[world_rank];

    auto level1 =
        c10::make_intrusive<c10d::PrefixStore>("default_pg", world_store);
    auto level2 = c10::make_intrusive<c10d::PrefixStore>("world_mesh", level1);
    auto pg_store = c10::make_intrusive<c10d::PrefixStore>("tpu", level2);

    threads.emplace_back([pg_store, &rank_local_stores,
                          &global_rank_to_device_id, &global_rank_to_topo_order,
                          &out_member_world_ranks,
                          &out_subgroup_member_world_ranks, idx, local_rank,
                          world_rank, pg_rank]() {
      const absl::Status status = GatherSubgroupsViaNodeLocalStore(
          pg_store.get(), rank_local_stores[idx].get(), local_rank,
          kLocalWorldSize, pg_rank, kPgSize, world_rank,
          global_rank_to_device_id, global_rank_to_topo_order,
          /*gather_id=*/99, out_member_world_ranks[world_rank],
          out_subgroup_member_world_ranks[world_rank]);
      EXPECT_TRUE(status.ok()) << status;
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  const DeviceGroupList expected_subgroup_member_world_ranks(
      {expected_subgroup});
  for (int64_t r = 0; r < kWorldSize; ++r) {
    EXPECT_EQ(out_member_world_ranks[r], expected_subgroup)
        << "member_world_ranks mismatch on world_rank " << r;
    EXPECT_EQ(out_subgroup_member_world_ranks[r],
              expected_subgroup_member_world_ranks)
        << "subgroup_member_world_ranks mismatch on world_rank " << r;
  }
}

TEST(ProcessGroupTpuTest, AllGatherViaNodeLocalStoreAllDegraded) {
  constexpr int64_t kWorldSize = 8;
  CountingHashStore world_store;
  world_store.setTimeout(kTestStoreTimeout);
  std::vector<AllGatherRankSpec> specs;
  for (int64_t r = 0; r < kWorldSize; ++r) {
    specs.push_back({.my_key = TestKey(r), .my_value = TestValue(r)});
  }

  const std::vector<AllGatherRankResult> results =
      RunAllGather(&world_store, specs, TestKeys(kWorldSize));

  ExpectAllSucceeded(results, TestValues(kWorldSize));
  // Every rank is its own node: 8 blobs + 1 table, 7 non-collector table reads.
  EXPECT_EQ(world_store.multi_set_calls(), 0);
  EXPECT_EQ(world_store.set_calls(), kWorldSize + 1);
  EXPECT_EQ(world_store.get_calls(), kWorldSize - 1);
  EXPECT_EQ(world_store.add_increment_calls(), kWorldSize);
  EXPECT_GE(world_store.multi_get_calls(), 1);
  EXPECT_LE(world_store.multi_get_calls(), kWorldSize);
}

TEST(ProcessGroupTpuTest, AllGatherViaNodeLocalStoreMixedLocalAndDegraded) {
  constexpr int64_t kWorldSize = 8;
  c10d::HashStore world_store;
  world_store.setTimeout(kTestStoreTimeout);
  std::vector<c10::intrusive_ptr<c10d::FileStore>> local_stores;
  // Node 0 shares a local store; node 1's ranks each act as a 1-rank node.
  // World rank 0 (the collector's key) is on a degraded rank.
  const std::vector<AllGatherRankSpec> specs = MakeLayoutSpecs(
      "nlag_mixed", {{.size = 4}, {.size = 4, .use_local_store = false}},
      {5, 1, 7, 3, 4, 0, 6, 2}, local_stores);

  const std::vector<AllGatherRankResult> results =
      RunAllGather(&world_store, specs, TestKeys(kWorldSize));

  ExpectAllSucceeded(results, TestValues(kWorldSize));
}

TEST(ProcessGroupTpuTest, AllGatherViaNodeLocalStoreFourNodesByFour) {
  constexpr int64_t kNumNodes = 4;
  constexpr int64_t kLocalWorldSize = 4;
  constexpr int64_t kWorldSize = kNumNodes * kLocalWorldSize;
  c10d::HashStore world_store;
  world_store.setTimeout(kTestStoreTimeout);
  // 5 is coprime with 16, so this is a permutation spreading consecutive world
  // ranks across nodes.
  std::vector<int64_t> world_ranks(kWorldSize);  // INT_VEC_OK
  for (int64_t idx = 0; idx < kWorldSize; ++idx) {
    world_ranks[idx] = (idx * 5 + 3) % kWorldSize;
  }
  std::vector<c10::intrusive_ptr<c10d::FileStore>> local_stores;
  const std::vector<AllGatherRankSpec> specs = MakeLayoutSpecs(
      "nlag_4x4",
      std::vector<NodeLayout>(kNumNodes, NodeLayout{.size = kLocalWorldSize}),
      world_ranks, local_stores);

  const std::vector<AllGatherRankResult> results =
      RunAllGather(&world_store, specs, TestKeys(kWorldSize));

  ExpectAllSucceeded(results, TestValues(kWorldSize));
}

TEST(ProcessGroupTpuTest,
     AllGatherViaNodeLocalStoreDuplicateKeyFailsEveryRank) {
  c10d::HashStore world_store;
  world_store.setTimeout(kTestStoreTimeout);
  std::vector<c10::intrusive_ptr<c10d::FileStore>> local_stores;
  // Two ranks on node 1 both claim k:2, so k:3 is never provided.
  const std::vector<AllGatherRankSpec> specs = MakeLayoutSpecs(
      "nlag_duplicate", {{.size = 2}, {.size = 2}}, {0, 1, 2, 2}, local_stores);

  const std::vector<AllGatherRankResult> results =
      RunAllGather(&world_store, specs, TestKeys(4));

  ExpectAllFailedWith(results, "duplicate key k:2");
  // specs[0] holds k:0, so it is the collector.
  EXPECT_EQ(results[0].status.code(), error::kFailedPrecondition);
}

TEST(ProcessGroupTpuTest, AllGatherViaNodeLocalStoreUnknownKeyFailsEveryRank) {
  constexpr int64_t kWorldSize = 4;
  c10d::HashStore world_store;
  world_store.setTimeout(kTestStoreTimeout);
  std::vector<AllGatherRankSpec> specs;
  for (int64_t r = 0; r < kWorldSize - 1; ++r) {
    specs.push_back({.my_key = TestKey(r), .my_value = TestValue(r)});
  }
  specs.push_back({.my_key = "bogus", .my_value = "bogus_value"});

  const std::vector<AllGatherRankResult> results =
      RunAllGather(&world_store, specs, TestKeys(kWorldSize));

  ExpectAllFailedWith(results, "unknown key bogus");
  EXPECT_EQ(results[0].status.code(), error::kFailedPrecondition);
}

TEST(ProcessGroupTpuTest,
     AllGatherViaNodeLocalStoreMissingRankTimesOutWithDiagnostics) {
  constexpr int64_t kWorldSize = 4;
  c10d::HashStore world_store;
  world_store.setTimeout(std::chrono::milliseconds(500));
  // The rank owning k:3 never runs.
  std::vector<AllGatherRankSpec> specs;
  for (int64_t r = 0; r < kWorldSize - 1; ++r) {
    specs.push_back({.my_key = TestKey(r), .my_value = TestValue(r)});
  }

  const std::vector<AllGatherRankResult> results =
      RunAllGather(&world_store, specs, TestKeys(kWorldSize));

  // specs[0] holds k:0, so it is the collector.
  ASSERT_FALSE(results[0].status.ok());
  EXPECT_THAT(results[0].status.message(),
              testing::HasSubstr("missing keys [k:3]"));
  EXPECT_THAT(results[0].status.message(), testing::HasSubstr("3/4"));
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_FALSE(results[i].status.ok()) << "rank index " << i << " succeeded";
  }
}

TEST(ProcessGroupTpuTest, ComputeRankToTopologyOrderTwoSliceTpu7x) {
  // Two slices, each a 2x1x2 (x, y, z) cuboid of chips with 2 cores per chip:
  // 16 devices. Device ids are numbered z-major within a slice,
  //   id = 8 * slice + 4 * z + 2 * x + core,
  // which differs from the x-major physical order that Python
  // `_get_physical_tpu_mesh` produces (out[x, y, z, core], slices outermost).
  constexpr int64_t kNumSlices = 2;
  constexpr int64_t kDimX = 2;
  constexpr int64_t kDimZ = 2;
  constexpr int64_t kCores = 2;
  constexpr int64_t kWorldSize = kNumSlices * kDimX * kDimZ * kCores;
  const auto device_id_of = [](int64_t slice, int64_t x, int64_t z,
                               int64_t core) {
    return 8 * slice + 4 * z + 2 * x + core;
  };

  // Scrambled launcher ranks: rank r drives device rank_to_device_id[r].
  const std::vector<int64_t>  // INT_VEC_OK
      rank_to_device_id = {13, 2,  7, 8,  0, 15, 5, 10,
                           3,  12, 1, 14, 6, 9,  4, 11};
  std::vector<int64_t> device_id_to_rank(kWorldSize, -1);  // INT_VEC_OK
  for (int64_t r = 0; r < kWorldSize; ++r) {
    device_id_to_rank[rank_to_device_id[r]] = r;
  }

  std::vector<PjRtDeviceAttributes> device_attrs(kWorldSize);
  for (int64_t slice = 0; slice < kNumSlices; ++slice) {
    for (int64_t x = 0; x < kDimX; ++x) {
      for (int64_t z = 0; z < kDimZ; ++z) {
        for (int64_t core = 0; core < kCores; ++core) {
          const int64_t id = device_id_of(slice, x, z, core);
          PjRtDeviceAttributes& attrs = device_attrs[device_id_to_rank[id]];
          attrs.id = id;
          attrs.process_index = id / 4;
          attrs.device_kind = "TPU7x";
          attrs.coords = std::vector<int64_t>{x, 0, z};  // INT_VEC_OK
          attrs.core_on_chip = core;
          attrs.slice_index = slice;
        }
      }
    }
  }

  // Reference: flatten (slice, x, y, z, core) with the innermost index
  // varying fastest, exactly like flattening the Python physical mesh with
  // slices stacked outermost, and record each rank's position.
  std::vector<int64_t> flattened_rank_to_pos(kWorldSize, -1);  // INT_VEC_OK
  int64_t pos = 0;
  for (int64_t slice = 0; slice < kNumSlices; ++slice) {
    for (int64_t x = 0; x < kDimX; ++x) {
      for (int64_t z = 0; z < kDimZ; ++z) {
        for (int64_t core = 0; core < kCores; ++core) {
          flattened_rank_to_pos[device_id_to_rank[device_id_of(slice, x, z,
                                                               core)]] = pos++;
        }
      }
    }
  }

  // Within a slice the physical order visits local ids
  //   0, 1, 4, 5, 2, 3, 6, 7  (x=0: z=0 c0/c1, z=1 c0/c1; then x=1 ...),
  // so local id l has position {0:0, 1:1, 2:4, 3:5, 4:2, 5:3, 6:6, 7:7}[l],
  // plus 8 for slice 1. E.g. rank 0 -> id 13 (slice 1, l=5) -> 8 + 3 = 11.
  const std::vector<int64_t>  // INT_VEC_OK
      expected = {11, 4, 7, 8, 0, 15, 3, 12, 5, 10, 1, 14, 6, 9, 2, 13};
  EXPECT_EQ(flattened_rank_to_pos, expected);

  const std::vector<int64_t> order =  // INT_VEC_OK
      ComputeRankToTopologyOrder(device_attrs);
  EXPECT_EQ(order, expected);

  // Device ids are a permutation of 0..15, so numeric device-id order assigns
  // rank r position rank_to_device_id[r]; topology order must differ.
  EXPECT_NE(order, rank_to_device_id);
}

TEST(ProcessGroupTpuTest, ComputeRankToTopologyOrderMissingAttributes) {
  // Without slice_index / coords / core_on_chip every key ties on the
  // topology fields, so the order falls back to numeric device id.
  std::vector<PjRtDeviceAttributes> device_attrs(3);
  device_attrs[0].id = 7;
  device_attrs[1].id = 2;
  device_attrs[2].id = 5;
  const std::vector<int64_t>  // INT_VEC_OK
      expected = {2, 0, 1};
  EXPECT_EQ(ComputeRankToTopologyOrder(device_attrs), expected);

  // A missing slice_index sorts as -1, i.e. before any reported slice.
  device_attrs[1].slice_index = 0;
  const std::vector<int64_t>  // INT_VEC_OK
      expected_with_slice = {1, 2, 0};
  EXPECT_EQ(ComputeRankToTopologyOrder(device_attrs), expected_with_slice);
}

TEST(ProcessGroupTpuTest,
     P2pSendQueueDispatchesWhenDescriptorsArriveAfterEnqueue) {
  auto store = c10::make_intrusive<CountingHashStore>();
  P2pSendQueue queue(store);

  auto waiter = std::make_shared<AsyncSendWaiter>();
  std::atomic<bool> callback_invoked{false};
  std::vector<std::vector<uint8_t>> received_descriptors;

  auto status = queue.Enqueue(
      /*dst_rank=*/1, {"desc_0"},
      [&, waiter](std::vector<std::vector<uint8_t>> descriptors) {
        received_descriptors = std::move(descriptors);
        callback_invoked.store(true, std::memory_order_release);
        waiter->MarkDone();
      },
      [waiter](std::exception_ptr eptr) { waiter->SetError(std::move(eptr)); });
  ASSERT_TRUE(status.ok()) << status;

  // Send returns immediately while "desc_0" is absent.
  EXPECT_FALSE(waiter->done.load(std::memory_order_acquire));
  EXPECT_FALSE(callback_invoked.load(std::memory_order_acquire));

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(store->check_calls(), 0);

  // Posting "desc_0" triggers ReadyCallback, completes waiter, and deletes
  // "desc_0" from store.
  const std::vector<uint8_t> payload = {'d', 'a', 't', 'a'};
  store->set("desc_0", payload);

  waiter->WaitAndThrow();
  EXPECT_TRUE(callback_invoked.load(std::memory_order_acquire));
  ASSERT_EQ(received_descriptors.size(), 1);
  EXPECT_EQ(received_descriptors[0], payload);
  EXPECT_EQ(store->check_calls(), 0);
  EXPECT_EQ(store->wait_calls(), 0);
  EXPECT_EQ(store->multi_get_calls(), 1);

  queue.Shutdown();
  EXPECT_FALSE(store->check({"desc_0"}));
}

TEST(ProcessGroupTpuTest,
     P2pSendQueueDispatchesOutOfOrderWithoutHeadOfLineBlocking) {
  auto store = c10::make_intrusive<CountingHashStore>();
  P2pSendQueue queue(store);

  auto waiter_a = std::make_shared<AsyncSendWaiter>();
  auto waiter_b = std::make_shared<AsyncSendWaiter>();

  ASSERT_TRUE(queue
                  .Enqueue(
                      /*dst_rank=*/1, {"key_a"},
                      [waiter_a](std::vector<std::vector<uint8_t>>) {
                        waiter_a->MarkDone();
                      },
                      [waiter_a](std::exception_ptr eptr) {
                        waiter_a->SetError(std::move(eptr));
                      })
                  .ok());
  ASSERT_TRUE(queue
                  .Enqueue(
                      /*dst_rank=*/2, {"key_b"},
                      [waiter_b](std::vector<std::vector<uint8_t>>) {
                        waiter_b->MarkDone();
                      },
                      [waiter_b](std::exception_ptr eptr) {
                        waiter_b->SetError(std::move(eptr));
                      })
                  .ok());

  // Post "key_b" first -> task B completes while task A remains incomplete.
  store->set("key_b", {'b'});
  waiter_b->WaitAndThrow();
  EXPECT_FALSE(waiter_a->done.load(std::memory_order_acquire));

  // Post "key_a" -> task A completes.
  store->set("key_a", {'a'});
  waiter_a->WaitAndThrow();

  EXPECT_EQ(store->check_calls(), 0);

  queue.Shutdown();
  EXPECT_FALSE(store->check({"key_a"}));
  EXPECT_FALSE(store->check({"key_b"}));
}

TEST(ProcessGroupTpuTest, P2pSendQueueFailsFutureOnStoreTimeout) {
  auto store = c10::make_intrusive<CountingHashStore>();
  store->setTimeout(std::chrono::milliseconds(30));
  P2pSendQueue queue(store);

  auto waiter = std::make_shared<AsyncSendWaiter>();
  ASSERT_TRUE(queue
                  .Enqueue(
                      /*dst_rank=*/1, {"missing_key"},
                      [](std::vector<std::vector<uint8_t>>) {},
                      [waiter](std::exception_ptr eptr) {
                        waiter->SetError(std::move(eptr));
                      })
                  .ok());

  try {
    waiter->WaitAndThrow();
    FAIL() << "expected store timeout exception";
  } catch (const std::exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr(
                              "p2p send timed out waiting for descriptors"));
  }
  EXPECT_EQ(store->check_calls(), 0);
}

TEST(ProcessGroupTpuTest,
     P2pSendQueueShutdownCancelsPendingAndRejectsNewEnqueues) {
  auto store = c10::make_intrusive<CountingHashStore>();
  P2pSendQueue queue(store);

  auto waiter = std::make_shared<AsyncSendWaiter>();
  ASSERT_TRUE(queue
                  .Enqueue(
                      /*dst_rank=*/1, {"missing_key"},
                      [](std::vector<std::vector<uint8_t>>) {},
                      [waiter](std::exception_ptr eptr) {
                        waiter->SetError(std::move(eptr));
                      })
                  .ok());

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  queue.Shutdown();

  try {
    waiter->WaitAndThrow();
    FAIL() << "expected shutdown cancellation exception";
  } catch (const std::exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("process group is shut down"));
  }

  auto waiter_after = std::make_shared<AsyncSendWaiter>();
  const auto status = queue.Enqueue(
      /*dst_rank=*/1, {"another_key"}, [](std::vector<std::vector<uint8_t>>) {},
      [waiter_after](std::exception_ptr eptr) {
        waiter_after->SetError(std::move(eptr));
      });
  EXPECT_EQ(status.code(), error::kCancelled);
  EXPECT_THAT(status.message(),
              testing::HasSubstr("process group is shut down"));
  EXPECT_FALSE(store->check({"missing_key"}));
}

// Models TCPStore's per-connection serialization: every operation holds this
// object's connection lock, and multiGet holds it while it waits for a missing
// key. Clones share the keys but get their own connection lock, as TCPStore
// clones open their own connection. `reads_started` counts reads that hold
// their connection lock, so a test can wait until a read is blocked in it.
class SerializedStore : public c10d::HashStore {
 public:
  SerializedStore(c10::intrusive_ptr<c10d::HashStore> data,
                  std::shared_ptr<std::atomic<int>> reads_started)
      : data_(std::move(data)), reads_started_(std::move(reads_started)) {}

  c10::intrusive_ptr<c10d::Store> clone() override {
    return c10::make_intrusive<SerializedStore>(data_, reads_started_);
  }

  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override {
    std::lock_guard<std::mutex> lock(connection_);
    reads_started_->fetch_add(1, std::memory_order_release);
    return data_->multiGet(keys);
  }

  void multiSet(const std::vector<std::string>& keys,
                const std::vector<std::vector<uint8_t>>& values) override {
    std::lock_guard<std::mutex> lock(connection_);
    data_->multiSet(keys, values);
  }

  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    std::lock_guard<std::mutex> lock(connection_);
    data_->set(key, value);
  }

  bool deleteKey(const std::string& key) override {
    std::lock_guard<std::mutex> lock(connection_);
    return data_->deleteKey(key);
  }

 private:
  c10::intrusive_ptr<c10d::HashStore> data_;
  std::shared_ptr<std::atomic<int>> reads_started_;
  std::mutex connection_;
};

// Polls `predicate` until it holds or `timeout` passes; returns its last value.
template <typename Predicate>
bool WaitUntil(Predicate predicate, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return predicate();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

// Wraps `store` in `depth` PrefixStore layers named "pg" and returns the key
// prefix they add.
std::string WrapInPrefixStores(c10::intrusive_ptr<c10d::Store>& store,
                               int depth) {
  std::string prefix;
  for (int i = 0; i < depth; ++i) {
    store = c10::make_intrusive<c10d::PrefixStore>("pg", store);
    prefix += "pg/";
  }
  return prefix;
}

class P2pSendQueueSerializedStoreTest : public testing::TestWithParam<int> {};

// A send waits on the next stage while this rank publishes its own receive to
// the previous stage. The blocked descriptor read must not hold up the
// publication. Never fail fatally before the read is released: a broken
// isolation would leave the worker blocked in multiGet.
TEST_P(P2pSendQueueSerializedStoreTest, PublishesWhileSendWaitsForDescriptor) {
  auto data = c10::make_intrusive<c10d::HashStore>();
  data->setTimeout(kTestStoreTimeout);
  auto reads_started = std::make_shared<std::atomic<int>>(0);
  c10::intrusive_ptr<c10d::Store> store =
      c10::make_intrusive<SerializedStore>(data, reads_started);
  const std::string prefix = WrapInPrefixStores(store, GetParam());
  P2pSendQueue queue(store);

  auto waiter = std::make_shared<AsyncSendWaiter>();
  auto received = std::make_shared<std::vector<std::vector<uint8_t>>>();
  ASSERT_TRUE(queue
                  .Enqueue(
                      /*dst_rank=*/1, {"downstream"},
                      [waiter, received](std::vector<std::vector<uint8_t>> d) {
                        *received = std::move(d);
                        waiter->MarkDone();
                      },
                      [waiter](std::exception_ptr eptr) {
                        waiter->SetError(std::move(eptr));
                      })
                  .ok());
  const bool read_started = WaitUntil(
      [&] { return reads_started->load(std::memory_order_acquire) >= 1; },
      std::chrono::seconds(5));

  auto publish = std::async(
      std::launch::async, [&] { store->multiSet({"upstream"}, {{1, 2, 3}}); });
  const auto published = publish.wait_for(std::chrono::seconds(2));
  const bool send_waiting = !waiter->done.load(std::memory_order_acquire);

  // An independent peer releases the read even when the isolation is broken.
  data->multiSet({prefix + "downstream"}, {{4, 5, 6}});
  waiter->WaitAndThrow();
  publish.get();

  EXPECT_TRUE(read_started);
  EXPECT_EQ(published, std::future_status::ready)
      << "publishing must not wait for a pending send's descriptor read";
  EXPECT_TRUE(send_waiting);
  EXPECT_EQ(*received, (std::vector<std::vector<uint8_t>>{{4, 5, 6}}));
  EXPECT_EQ(data->multiGet({prefix + "upstream"}),
            (std::vector<std::vector<uint8_t>>{{1, 2, 3}}));
  queue.Shutdown();
  EXPECT_FALSE(data->check({prefix + "downstream"}));
}

// Sends to different peers read their descriptors on their own connections,
// so a send whose receiver has not posted yet does not block a send to a
// receiver that has.
TEST_P(P2pSendQueueSerializedStoreTest, KeepsPeerReadsOnSeparateConnections) {
  auto data = c10::make_intrusive<c10d::HashStore>();
  data->setTimeout(kTestStoreTimeout);
  auto reads_started = std::make_shared<std::atomic<int>>(0);
  c10::intrusive_ptr<c10d::Store> store =
      c10::make_intrusive<SerializedStore>(data, reads_started);
  const std::string prefix = WrapInPrefixStores(store, GetParam());
  P2pSendQueue queue(store);

  auto enqueue = [&queue](int dst_rank, const std::string& key,
                          std::shared_ptr<AsyncSendWaiter> waiter) {
    return queue.Enqueue(
        dst_rank, {key},
        [waiter](std::vector<std::vector<uint8_t>>) { waiter->MarkDone(); },
        [waiter](std::exception_ptr eptr) {
          waiter->SetError(std::move(eptr));
        });
  };
  auto waiter_a = std::make_shared<AsyncSendWaiter>();
  auto waiter_b = std::make_shared<AsyncSendWaiter>();
  ASSERT_TRUE(enqueue(/*dst_rank=*/1, "key_a", waiter_a).ok());
  ASSERT_TRUE(enqueue(/*dst_rank=*/2, "key_b", waiter_b).ok());
  const bool both_reading = WaitUntil(
      [&] { return reads_started->load(std::memory_order_acquire) >= 2; },
      std::chrono::seconds(5));

  // Rank 2's receiver posts first; rank 1's send keeps waiting.
  data->set(prefix + "key_b", {'b'});
  const bool b_done =
      WaitUntil([&] { return waiter_b->done.load(std::memory_order_acquire); },
                std::chrono::seconds(2));
  const bool a_waiting = !waiter_a->done.load(std::memory_order_acquire);

  data->set(prefix + "key_a", {'a'});
  waiter_a->WaitAndThrow();
  waiter_b->WaitAndThrow();

  EXPECT_TRUE(both_reading) << "each peer's read needs its own connection";
  EXPECT_TRUE(b_done) << "a send must not wait behind another peer's read";
  EXPECT_TRUE(a_waiting);
  queue.Shutdown();
}

INSTANTIATE_TEST_SUITE_P(PrefixDepth, P2pSendQueueSerializedStoreTest,
                         testing::Values(0, 2));

// Through a real TCPStore and nested PrefixStores, descriptors published on
// the original store reach the send intact, and the consumed keys are deleted
// under the same prefixed names.
TEST(ProcessGroupTpuTest, P2pSendQueueTCPStorePreservesDescriptorNamespace) {
  c10d::TCPStoreOptions options;
  options.port = 0;
  options.isServer = true;
  options.waitWorkers = false;
  options.timeout = kTestStoreTimeout;
  auto root = c10::make_intrusive<c10d::TCPStore>("127.0.0.1", options);
  c10::intrusive_ptr<c10d::Store> store = root;
  const std::string prefix = WrapInPrefixStores(store, /*depth=*/2);
  P2pSendQueue queue(store);

  auto waiter = std::make_shared<AsyncSendWaiter>();
  auto received = std::make_shared<std::vector<std::vector<uint8_t>>>();
  ASSERT_TRUE(queue
                  .Enqueue(
                      /*dst_rank=*/1, {"second", "first"},
                      [waiter, received](std::vector<std::vector<uint8_t>> d) {
                        *received = std::move(d);
                        waiter->MarkDone();
                      },
                      [waiter](std::exception_ptr eptr) {
                        waiter->SetError(std::move(eptr));
                      })
                  .ok());
  store->multiSet({"first", "second"}, {{0, 255}, {3, 4}});
  waiter->WaitAndThrow();
  queue.Shutdown();

  EXPECT_EQ(*received, (std::vector<std::vector<uint8_t>>{{3, 4}, {0, 255}}));
  EXPECT_FALSE(root->check({prefix + "first"}));
  EXPECT_FALSE(root->check({prefix + "second"}));
}

}  // namespace
}  // namespace torch_tpu
