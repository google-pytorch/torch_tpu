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

#ifndef TORCH_TPU_CSRC_DISTRIBUTED_CROSS_HOST_DESCRIPTOR_STORE_H_
#define TORCH_TPU_CSRC_DISTRIBUTED_CROSS_HOST_DESCRIPTOR_STORE_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Store.hpp"

namespace torch_tpu {

// A send may wait for the next pipeline stage while a receive publishes to the
// previous stage. A blocking read must not monopolize the publishing
// connection.
class CrossHostDescriptorStore {
 public:
  explicit CrossHostDescriptorStore(c10::intrusive_ptr<c10d::Store> store)
      : publish_store_(std::move(store)),
        read_store_(publish_store_->clone()) {}

  std::vector<std::vector<uint8_t>> Get(
      const std::vector<std::string>& keys) {
    return read_store_->multiGet(keys);
  }

  void Publish(const std::vector<std::string>& keys,
               const std::vector<std::vector<uint8_t>>& descriptors) {
    publish_store_->multiSet(keys, descriptors);
  }

 private:
  c10::intrusive_ptr<c10d::Store> publish_store_;
  c10::intrusive_ptr<c10d::Store> read_store_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_DISTRIBUTED_CROSS_HOST_DESCRIPTOR_STORE_H_
