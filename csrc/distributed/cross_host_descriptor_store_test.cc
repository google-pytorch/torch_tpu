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

#include "csrc/distributed/cross_host_descriptor_store.h"

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "c10/util/intrusive_ptr.h"
#include "gtest/gtest.h"
#include "torch/csrc/distributed/c10d/HashStore.hpp"
#include "torch/csrc/distributed/c10d/PrefixStore.hpp"
#include "torch/csrc/distributed/c10d/TCPStore.hpp"

namespace torch_tpu {
namespace {

using Bytes = std::vector<std::vector<uint8_t>>;
using namespace std::chrono_literals;

// Models TCPStore's per-client serialization, with a handshake after acquiring
// the connection lock. Unlike a sleep before multiSet, this guarantees that the
// blocking read owns the connection before the publishing thread starts.
// Clones share keys but have independent connection locks.
class SerializedStore : public c10d::HashStore {
 public:
  SerializedStore(c10::intrusive_ptr<c10d::HashStore> data,
                  std::shared_ptr<std::promise<void>> reading)
      : data_(std::move(data)), reading_(std::move(reading)) {}

  c10::intrusive_ptr<c10d::Store> clone() override {
    return c10::make_intrusive<SerializedStore>(data_, reading_);
  }

  Bytes multiGet(const std::vector<std::string>& keys) override {
    std::lock_guard<std::mutex> lock(connection_);
    reading_->set_value();
    return data_->multiGet(keys);
  }

  void multiSet(const std::vector<std::string>& keys,
                const Bytes& values) override {
    std::lock_guard<std::mutex> lock(connection_);
    data_->multiSet(keys, values);
  }

 private:
  c10::intrusive_ptr<c10d::HashStore> data_;
  std::shared_ptr<std::promise<void>> reading_;
  std::mutex connection_;
};

class CrossHostDescriptorStoreTest : public testing::TestWithParam<int> {};

TEST_P(CrossHostDescriptorStoreTest, PublishWhileWaitingForDownstream) {
  auto data = c10::make_intrusive<c10d::HashStore>();
  data->setTimeout(10s);
  auto reading = std::make_shared<std::promise<void>>();
  auto read_started = reading->get_future();
  c10::intrusive_ptr<c10d::Store> store =
      c10::make_intrusive<SerializedStore>(data, reading);
  std::string prefix;
  for (int i = 0; i < GetParam(); ++i) {
    store = c10::make_intrusive<c10d::PrefixStore>("pg", store);
    prefix += "pg/";
  }
  CrossHostDescriptorStore descriptors(store);
  auto read = std::async(std::launch::async, [&] {
    return descriptors.Get({"downstream"});
  });
  // Do not use a fatal assertion: always release the reader and join workers.
  const auto started = read_started.wait_for(5s);
  auto publish = std::async(std::launch::async, [&] {
    descriptors.Publish({"upstream"}, Bytes{{1, 2, 3}});
  });
  const auto published = publish.wait_for(2s);
  const auto read_pending = read.wait_for(0s);
  // Independent peer releases the read even when the isolation is broken.
  data->multiSet({prefix + "downstream"}, Bytes{{4, 5, 6}});
  EXPECT_EQ(started, std::future_status::ready);
  EXPECT_EQ(published, std::future_status::ready)
      << "publishing must not wait for a downstream receive descriptor";
  EXPECT_EQ(read_pending, std::future_status::timeout);
  EXPECT_EQ(read.get(), (Bytes{{4, 5, 6}}));
  publish.get();
  EXPECT_EQ(data->multiGet({prefix + "upstream"}), (Bytes{{1, 2, 3}}));
}

// Exercise real TCPStore cloning and nested PrefixStore key preservation too.
TEST_P(CrossHostDescriptorStoreTest, TCPStorePreservesDescriptorNamespace) {
  c10d::TCPStoreOptions options;
  options.port = 0;
  options.isServer = true;
  options.waitWorkers = false;
  options.timeout = 10s;
  auto root = c10::make_intrusive<c10d::TCPStore>("127.0.0.1", options);
  c10::intrusive_ptr<c10d::Store> store = root;
  std::string prefix;
  for (int i = 0; i < GetParam(); ++i) {
    store = c10::make_intrusive<c10d::PrefixStore>("pg", store);
    prefix += "pg/";
  }
  CrossHostDescriptorStore descriptors(store);
  descriptors.Publish({"first", "second"}, Bytes{{0, 255}, {3, 4}});
  EXPECT_EQ(descriptors.Get({"second", "first"}), (Bytes{{3, 4}, {0, 255}}));
  EXPECT_EQ(root->multiGet({prefix + "first"}), (Bytes{{0, 255}}));
}

INSTANTIATE_TEST_SUITE_P(PrefixDepth, CrossHostDescriptorStoreTest,
                        testing::Values(0, 2));

}  // namespace
}  // namespace torch_tpu
