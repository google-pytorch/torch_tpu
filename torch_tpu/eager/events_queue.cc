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

#include "torch_tpu/eager/events_queue.h"

#include <cstdint>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

namespace {

// A singleton class that records events related to the creation and destruction
// of c10::DataPtrs referencing DeviceBufferRefs.s
class EventsQueue {
 public:
  // Returns the singleton instance of the EventsQueue.
  static EventsQueue& GetInstance() {
    static absl::NoDestructor<EventsQueue> instance;
    return *instance;
  }

  // Records on the events queue that a new c10::DataPtr referencing the given
  // DeviceBufferRef has been created.
  void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
    // Placeholders and empty tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_empty()) {
      return;
    }
    absl::MutexLock lock(mu_);
    if (!device_buffer_ref.is_materialized()) {
      // Insert or increment the count for the DeviceBufferList.
      live_nodes_[device_buffer_ref.device_buffer_list()]++;
    } else {
      // Once the ref has a ready PjRtBuffer, we can stop tracking it.
      live_nodes_.erase(device_buffer_ref.device_buffer_list());
    }
  }

  // Records on the events queue that a c10::DataPtr referencing the given
  // DeviceBufferRef has been destroyed.
  void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref) {
    // Placeholders and empty tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_empty()) {
      return;
    }
    absl::MutexLock lock(mu_);
    // If the ref is already removed from the map, we do nothing. This can
    // happen if the ref became ready after insertion, or if the queue was
    // cleared.
    auto it = live_nodes_.find(device_buffer_ref.device_buffer_list());
    if (it == live_nodes_.end()) {
      return;
    }
    // Once the ref is materialized, or the count drops to zero, we can remove
    // it from the map.
    if (device_buffer_ref.is_materialized() || --it->second <= 0) {
      live_nodes_.erase(it);
    }
  }

  // Returns a vector of all the DeviceBufferLists that are currently referenced
  // by at least one c10::DataPtr, and are not in a final "ready" state.
  std::vector<SharedDeviceBufferList> GetAllLiveUnsyncedDataPtrs() {
    absl::MutexLock lock(mu_);
    std::vector<SharedDeviceBufferList> result;
    result.reserve(live_nodes_.size());
    // Can't clear the map while also iterating over it.
    std::vector<const DeviceBufferList*> to_remove;
    for (const auto& [node, _] : live_nodes_) {
      if (node->is_materialized()) {
        to_remove.push_back(node.get());
      } else {
        result.push_back(node);
      }
    }
    for (const auto* device_buffer_list : to_remove) {
      live_nodes_.erase(device_buffer_list);
    }
    return result;
  }

  // Clears all tracked DeviceBufferLists from the events queue.
  void Clear() {
    absl::MutexLock lock(mu_);
    live_nodes_.clear();
  }

 private:
  absl::Mutex mu_;
  // Hold a strong pointer to the DeviceBufferList as the key; as long as there
  // is a live DataPtr, the DeviceBufferList can't be dropped.
  absl::flat_hash_map<SharedDeviceBufferList, int64_t> live_nodes_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace

void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
  EventsQueue::GetInstance().RecordNewDataPtrCreated(device_buffer_ref);
}

void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref) {
  EventsQueue::GetInstance().RecordDataPtrDestroyed(device_buffer_ref);
}

std::vector<SharedDeviceBufferList> GetAllLiveUnsyncedDataPtrs() {
  return EventsQueue::GetInstance().GetAllLiveUnsyncedDataPtrs();
}

void ClearEventsQueue() { EventsQueue::GetInstance().Clear(); }

}  // namespace torch_tpu
