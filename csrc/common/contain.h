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

#ifndef TORCH_TPU_COMMON_CONTAIN_H_
#define TORCH_TPU_COMMON_CONTAIN_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"

namespace torch_tpu {

// Dynamic container for tracking memory usage.
//
// This is a global singleton class that is initialized lazily on first use. It
// supports only one implicit container. Entering via
// ScopedMemMeasuringContainer multiple times simply moves into said container
// for the duration of the scope, it does not create or destroy containers per
// call. Attempting to enter the container from a thread which is already inside
// of it has no effect.
//
// Example:
//  auto threads = torch_tpu::ThreadPool();
//  threads.Schedule([]() {
//    auto guard = ScopedMemMeasuringContainer();
//    DoWork();
//    auto child_threads = torch_tpu::ThreadPool();
//    child_threads.Schedule([]() {
//      DoWork();
//    });
//  });
//
// ContainerPeakHostMemoryBytes() will reflect all calls to DoWork();

class ScopedMemMeasuringContainer {
 public:
  ScopedMemMeasuringContainer();
  ~ScopedMemMeasuringContainer();

  ScopedMemMeasuringContainer(const ScopedMemMeasuringContainer&) = delete;
  ScopedMemMeasuringContainer& operator=(const ScopedMemMeasuringContainer&) =
      delete;
  ScopedMemMeasuringContainer(ScopedMemMeasuringContainer&&) = delete;
  ScopedMemMeasuringContainer& operator=(ScopedMemMeasuringContainer&&) =
      delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Returns peak memory usage in bytes inside the container. Work done by any
// thread while inside the container, or any thread spawned inside the container
// will be accounted for.
absl::StatusOr<int64_t> ContainerPeakHostMemoryBytes();

// Destroys the container and resets the singleton state. Facilitates
// "resetting" of memory tracking, provided there are no background threads in
// the current container that would be lost.
void CleanUpContainer();

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_CONTAIN_H_
