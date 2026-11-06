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

#ifndef TORCH_TPU_CSRC_EAGER_HOST_ALLOCATOR_H_
#define TORCH_TPU_CSRC_EAGER_HOST_ALLOCATOR_H_

#include "ATen/core/CachingHostAllocator.h"
#include "xla/future.h"

namespace torch_tpu {

// Returns the Tpu pinned host memory allocator.
// Not to be confused with TpuAllocator (see tensor_to_buffer.h) which is the
// device-side allocator; this allocator is for host-side memory that is managed
// by the TPU backend.
at::HostAllocator* GetTpuPinnedAllocator();

// Returns true if the given pointer was allocated by the Tpu pinned memory
// allocator.
bool IsTpuPinnedPtr(const void* ptr);

// If the pointer is pinned using the TPU pinned host memory allocator, this
// instructs the allocator to keep the pointer alive until the future
// completes.
// Returns false if the pointer is not pinned using the TPU pinned host memory
// allocator, or true if the future was successfully registered.
bool HoldPinnedPointerUntil(const void* ptr, const xla::Future<void>& future);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_HOST_ALLOCATOR_H_
