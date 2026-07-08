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

#include "torch_tpu/eager/current_stream.h"

#include <array>
#include <atomic>

#include "absl/log/absl_check.h"

namespace torch_tpu {

namespace {

constexpr int kMaxDevices = 8;

// TODO: make this per-python-thread.
// clang-format off
// NOLINTNEXTLINE(whitespace/line_length)
thread_local DeviceIndex current_device_index_ = 0;  // CPP_THREAD_LOCAL_OK=State is per-C++-thread for device guard
// NOLINTNEXTLINE(whitespace/line_length)
thread_local std::array<StreamId, kMaxDevices> current_stream_ids_ = {0};  // CPP_THREAD_LOCAL_OK=State is per-C++-thread for device guard
// clang-format on

// This is not a thread_local because all threads share the same next stream ID
// counter.
// Counter starts at 1 because 0 is reserved for the default stream.
static std::array<std::atomic<StreamId>, kMaxDevices> next_stream_ids_ = {
    1, 1, 1, 1, 1, 1, 1, 1};

}  // namespace

DeviceIndex GetCurrentDeviceIndex() { return current_device_index_; }

DeviceIndex ExchangeCurrentDeviceIndex(DeviceIndex new_device_index) {
  ABSL_CHECK_LT(new_device_index, kMaxDevices);  // CRASH_OK
  DeviceIndex old_device_index = current_device_index_;
  current_device_index_ = new_device_index;
  return old_device_index;
}

StreamId GetCurrentStreamId(DeviceIndex device_index) {
  ABSL_CHECK_LT(device_index, kMaxDevices);  // CRASH_OK
  return current_stream_ids_[device_index];
}

StreamId ExchangeCurrentStreamId(DeviceIndex device_index,
                                 StreamId new_stream_id) {
  StreamId old_stream_id = current_stream_ids_[device_index];
  current_stream_ids_[device_index] = new_stream_id;
  return old_stream_id;
}

DeviceStreamId GetCurrentDeviceStreamId() {
  return DeviceStreamId{current_device_index_,
                        current_stream_ids_[current_device_index_]};
}

StreamId NextStreamId(DeviceIndex device_index) {
  ABSL_CHECK_LT(device_index, kMaxDevices);  // CRASH_OK
  return next_stream_ids_[device_index].fetch_add(1, std::memory_order_relaxed);
}

}  // namespace torch_tpu
