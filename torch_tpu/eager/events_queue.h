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

#ifndef TORCH_TPU_EAGER_EVENTS_QUEUE_H_
#define TORCH_TPU_EAGER_EVENTS_QUEUE_H_

#include <vector>

#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

// Records on the events queue that a new c10::DataPtr referencing the given
// DeviceBufferRef has been created.
// As long as at least one DataPtr exists, the DeviceBufferList cannot be freed,
// and if deferred, must be materialized on any global sync
// (torch.tpu.synchronize()).
void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref);

// Records on the events queue that a c10::DataPtr referencing the given
// DeviceBufferRef has been destroyed.
// Once all references to a DeviceBufferList have been destroyed, the
// DeviceBufferList can be freed, and does not need to be materialized.
void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref);

// Returns a vector of all the DeviceBufferLists that are currently referenced
// by at least one c10::DataPtr, and are not in a final "ready" state.
// The order of the returned vector is **not** specified, but will not contain
// duplicates.
std::vector<SharedDeviceBufferList> GetAllLiveUnsyncedDataPtrs();

// Clears all references to DeviceBufferLists from the events queue.
void ClearEventsQueue();

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_EVENTS_QUEUE_H_
