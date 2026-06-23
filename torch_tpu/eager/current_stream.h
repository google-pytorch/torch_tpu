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

#ifndef TORCH_TPU_EAGER_CURRENT_STREAM_H_
#define TORCH_TPU_EAGER_CURRENT_STREAM_H_

#include <cstdint>

// Utility functions for retrieving and updating the current device and stream
// for the calling thread. Factored out to avoid circular build dependencies.

namespace torch_tpu {

// These are the same as c10::DeviceIndex and c10::StreamId, but importing
// Streah.h and Device.h causes build issues for some reason.
using DeviceIndex = int8_t;
using StreamId = int64_t;

// Returns the current device index for the calling thread.
DeviceIndex GetCurrentDeviceIndex();

// Sets the current device index for the calling thread and returns the previous
// device index.
DeviceIndex ExchangeCurrentDeviceIndex(DeviceIndex new_device_index);

// Returns the current stream ID for the given device index on the calling
// thread.
StreamId GetCurrentStreamId(DeviceIndex device_index);

// Sets the current stream ID for the given device index on the calling thread
// and returns the previous stream ID.
StreamId ExchangeCurrentStreamId(DeviceIndex device_index,
                                 StreamId new_stream_id);

// A full stream identifier, consisting of a device index and a stream ID.
// c10::StreamIds are only unique per device; "stream 0" means "default stream"
// for each device, so device 0 stream 0 and device 1 stream 0 are different.
struct DeviceStreamId {
  DeviceIndex device_index;
  StreamId stream_id;
};

// Returns the current device and stream IDs for the calling thread.
DeviceStreamId GetCurrentDeviceStreamId();

// Returns the next available stream ID for the given device index.
StreamId NextStreamId(DeviceIndex device_index);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_CURRENT_STREAM_H_
