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

#include "csrc/eager/host_allocator.h"

#include <cstddef>
#include <utility>

#include "ATen/core/CachingHostAllocator.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "c10/util/UniqueVoidPtr.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/warnings.h"
#include "csrc/pjrt/pjrt_state.h"
#include "torch/headeronly/core/DeviceType.h"
#include "xla/pjrt/host_memory_allocator.h"

namespace torch_tpu {
namespace {

class TpuPinnedAllocator final : public at::HostAllocator {
 public:
  c10::DataPtr allocate(size_t nbytes) override {
    if (nbytes == 0) {
      return {nullptr, nullptr, &DeleteTpuPinnedBufferStatic,
              c10::Device(c10::DeviceType::CPU)};
    }
    TT_ASSIGN_OR_THROW(xla::HostMemoryAllocator * host_allocator,
                       PjrtBackend::GetInstance().GetHostAllocator(),
                       _ << "Failed to get PJRT host allocator");
    // DO NOT MERGE: deliberate uncovered error macro (negative test).
    TT_CHECK_THROW(nbytes != 12345678901, error::kInvalidArgument)
        << "never fires: error coverage negative test";
    xla::HostMemoryAllocator::OwnedPtr data = host_allocator->Allocate(nbytes);
    TT_CHECK_THROW(data != nullptr, error::kResourceExhausted)
        << "failed to allocate " << nbytes << " bytes of pinned host memory";

    void* raw_ptr = data.get();
    {
      absl::MutexLock lock(mutex_);
      pinned_ptrs_.emplace(raw_ptr, std::move(data));
    }

    return {raw_ptr, raw_ptr, &DeleteTpuPinnedBufferStatic,
            c10::Device(c10::DeviceType::CPU)};
  }

  void copy_data(void* dest, const void* src,
                 std::size_t count) const override {
    default_copy_data(dest, src, count);
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &DeleteTpuPinnedBufferStatic;
  }

  bool record_event(void* ptr, void* ctx, c10::Stream stream) override {
    // TPU does not yet support asynchronous stream-based events for pinned
    // memory in the same way CUDA does. Returning false indicates that
    // this allocator does not support event recording.
    return false;
  }

  // This allocator does not cache allocations, so there is nothing to do here.
  void empty_cache() override {
    TT_WARN_ONCE(WarningCategory::kTorchTpu,
                 "TpuPinnedAllocator::empty_cache is not implemented for TPU.");
  }

  at::HostStats get_stats() override { return {}; }

  void reset_accumulated_stats() override {}

  void reset_peak_stats() override {}

  bool is_pinned_ptr(const void* ptr) {
    absl::ReaderMutexLock lock(mutex_);
    return pinned_ptrs_.contains(ptr);
  }

  void free_ptr(void* ptr) {
    if (ptr) {
      decltype(pinned_ptrs_)::node_type data;
      {
        absl::MutexLock lock(mutex_);
        data = pinned_ptrs_.extract(ptr);
      }
    }
  }

 private:
  static void DeleteTpuPinnedBufferStatic(void* ptr);

  absl::Mutex mutex_;
  absl::flat_hash_map<const void*, xla::HostMemoryAllocator::OwnedPtr>
      pinned_ptrs_ ABSL_GUARDED_BY(mutex_);
};

TpuPinnedAllocator& GetTpuPinnedAllocatorInternal() {
  static absl::NoDestructor<TpuPinnedAllocator> g_tpu_pinned_allocator;
  return *g_tpu_pinned_allocator;
}

void TpuPinnedAllocator::DeleteTpuPinnedBufferStatic(void* ptr) {
  GetTpuPinnedAllocatorInternal().free_ptr(ptr);
}
}  // namespace

at::HostAllocator* GetTpuPinnedAllocator() {
  return &GetTpuPinnedAllocatorInternal();
}

bool IsTpuPinnedPtr(const void* ptr) {
  return GetTpuPinnedAllocatorInternal().is_pinned_ptr(ptr);
}

}  // namespace torch_tpu
