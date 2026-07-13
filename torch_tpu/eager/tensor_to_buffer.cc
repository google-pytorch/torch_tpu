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

#include "torch_tpu/eager/tensor_to_buffer.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/CachingHostAllocator.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Allocator.h"
#include "c10/core/CachingDeviceAllocator.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/core/Storage.h"
#include "c10/core/StorageImpl.h"
#include "c10/core/Stream.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "c10/util/UniqueVoidPtr.h"
#include "c10/util/intrusive_ptr.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/layout_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_buffer_utils.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/events_queue.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/host_memory_allocator.h"

namespace torch_tpu {

absl::Status AssignBufferToAtTensor(DeviceBufferRef result_buf,
                                    const at::Tensor& tensor) {
  ABSL_VLOG(1)
      << "[AssignBufferToAtTensor] Assigning DeviceBufferRef to existing "
         "tensor\n"
      << result_buf.DebugString() << "\n"
      << "\nTensor:" << ToString(tensor)
      << "\n\tscalar_type: " << ToString(tensor.scalar_type())
      << "\n\tsizes: " << ToString(tensor.sizes())
      << "\n\tstrides: " << ToString(tensor.strides())
      << (tensor.is_contiguous() ? " (contiguous)" : " (non-contiguous)")
      << "\n\tstorage nbytes: " << tensor.storage().nbytes()
      << "\n\tstorage offset: " << tensor.storage_offset();

  // It's a critical failure if we are trying to write a buffer to a tensor
  // with the wrong amount or type of data; we need a 1:1 byte mapping from
  // the logical tensor elements to the buffer elements.
  // However, we do allow at::Tensors to be backed by DeviceBufferRefs of
  // view-equivalent shapes, for example, a (10, 10) tensor may be backed by
  // a DeviceBufferRef of shape (100,), or vice-versa.
  // Shape alignments will be handled by view decomposition logic on extraction
  // during GetBuffer later, if and only if required.
  TT_ASSIGN_OR_RETURN(const TpuLayout tpu_layout, ResolveTpuLayout(tensor));
  ABSL_CHECK_EQ(result_buf.element_type(),  // CRASH_OK
                tpu_layout.element_type);

  // Calculate expected number of elements from the resolved TPU layout.
  // This already accounts for FP4 size doubling.
  int64_t expected_elements = 1;
  for (const int64_t size : tpu_layout.sizes) {
    TT_ASSIGN_OR_RETURN(expected_elements,
                        SafeMultiply(expected_elements, size));
  }

  ABSL_CHECK_EQ(result_buf.num_elements(), expected_elements)  // CRASH_OK
      << "Cannot assign buffer of size " << result_buf.num_elements()
      << " to tensor of shape " << ToString(tensor.sizes())
      << " (resolved TPU layout shape: " << ToString(tpu_layout.sizes)
      << ", expected elements: " << expected_elements << ")."
      << (tpu_layout.element_type == mlir::ElementType::F4E2M1FN
              ? " For FP4 tensors, the expected element count is doubled "
                "because they are packed 2-per-byte in PyTorch."
              : "");

  if (result_buf.num_elements() == 0) {
    // Writing 0 elements is a no-op.
    return absl::OkStatus();
  }

  // We must skip this optimization for Conj tensors because their storage
  // contains the *original* values, not the conjugated values. Direct storage
  // assignment would lose the conjugation bit, resulting in incorrect values.
  // By falling through to ComputeInverseViewOperation, we ensure the proper
  // view inversion (which includes conjugation) is applied during the
  // write-back.
  if (tensor.is_contiguous() && !tensor.is_conj() &&
      tensor.storage_offset() == 0 &&
      tensor.storage().nbytes() == result_buf.size_bytes()) {
    // The result_buf can be used directly as the new storage buffer; even if
    // it is a different shape, any contiguous view can be the base.
    //
    // Note: this means it's possible for an operation like
    // ```
    //   a = torch.zeros(100)
    //   b = a.view(10, 10)
    //   b.add_(1.0)
    // ```
    // to change the shape of the base DeviceBufferRef from one contiguous shape
    // to another equivalent shape, in this case (100,) -> (10, 10). In some
    // sense, this changes from `b` being a view of `a`, to `a` being a view
    // of `b`, in a "last write wins" ordering.
    ABSL_VLOG(2) << "[AssignBufferToAtTensor] Directly assigning contiguous "
                    "base buffer of "
                 << result_buf.size_bytes() << " bytes.";

    // Change the c10::DataPtr in storage to use the new DeviceBufferRef.
    // This ensures that all active views on this same contiguous base tensor
    // will be updated by the write.
    tensor.storage().set_data_ptr(
        MakeDataPtr(std::move(result_buf), tensor.device().index()));
  } else {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref, GetBaseBuffer(tensor));
    TT_ASSIGN_OR_RETURN(DeviceBufferRef new_base_buffer_ref,
                        CreateInverseViewDeviceBufferRef(
                            std::move(base_buffer_ref), std::move(result_buf),
                            tpu_layout, tensor.is_conj()));

    // Assign the new deferred DeviceBufferRef to the c10::DataPtr
    // All reads that happened before the write will be unaffected, but reads
    // to all extant views will reflect the new value on later accesses.
    tensor.storage().set_data_ptr(
        MakeDataPtr(std::move(new_base_buffer_ref), tensor.device().index()));
  }

  // Bump the version to tell PyTorch how to handle functionalization
  tensor.unsafeGetTensorImpl()->bump_version();

  ABSL_VLOG(1) << "[AssignBufferToAtTensor] Final tensor after assignment:"
               << ToString(tensor)
               << "\nscalar_type: " << ToString(tensor.scalar_type())
               << "\nsizes: " << ToString(tensor.sizes())
               << "\nstrides: " << ToString(tensor.strides())
               << (tensor.is_contiguous() ? " (contiguous)"
                                          : " (non-contiguous)")
               << "\nstorage nbytes: " << tensor.storage().nbytes()
               << "\nstorage offset: " << tensor.storage_offset();
  return absl::OkStatus();
}
absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const at::Tensor& tensor) {
  TT_RET_CHECK(tensor.defined(), error::kInvalidArgument)
      << "tensor is undefined";

  TT_RET_CHECK(tensor.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "tensor is expected to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << tensor.device().str();

  TT_RET_CHECK(tensor.storage().data_ptr(), error::kPythonNotImplementedError)
      << "tensor is on the custom PrivateUse1 device. But its storage data_ptr "
         "is null. This is usually caused by FakeTensor being run on TPU ops. "
         "And it is not supported";

  ABSL_CHECK_EQ(tensor.storage().allocator(), GetTpuAllocator())  // CRASH_OK
      << "tensor is on PrivateUse1 device, but is not allocated by "
         "g_tpu_allocator";

  return GetBaseBuffer(tensor.storage());
}

absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const c10::Storage& storage) {
  // The DeviceBufferRef in the c10::StorageImpl represents the "contiguous
  // base" buffer--this is the shape of the original tensor that was used to
  // construct the tensor, and represents all of the data available to all
  // views that share the same Storage.
  const DeviceBufferRef* base_buffer_ref =
      static_cast<const DeviceBufferRef*>(storage.data_ptr().get_context());
  ABSL_CHECK(base_buffer_ref)  // CRASH_OK
      << "tensor storage has a null DeviceBufferRef via data_ptr context";
  return *base_buffer_ref;
}

absl::StatusOr<DeviceBufferRef> GetBuffer(const at::Tensor& tensor) {
  TT_RET_CHECK(tensor.defined(), error::kInvalidArgument)
      << "tensor is undefined";
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref, GetBaseBuffer(tensor));
  TT_ASSIGN_OR_RETURN(const TpuLayout tpu_layout, ResolveTpuLayout(tensor));

  if (!tensor.is_conj() && tensor.is_contiguous() &&
      tensor.storage_offset() == 0 &&
      tpu_layout.sizes == base_buffer_ref.dimensions() &&
      tpu_layout.element_type == base_buffer_ref.element_type()) {
    ABSL_VLOG(1) << "[GetBuffer] Tensor is a contiguous base, "
                    "returning base buffer.";
    return base_buffer_ref;
  }

  return CreateViewDeviceBufferRef(base_buffer_ref, tpu_layout,
                                   tensor.is_conj());
}

absl::StatusOr<std::vector<DeviceBufferRef>> GetBuffers(
    absl::Span<const at::Tensor> tensors) {
  std::vector<DeviceBufferRef> buffers;
  buffers.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer, GetBuffer(tensor));
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

c10::Storage MakeStorage(DeviceBufferRef buffer_ref, int device_idx) {
  ABSL_VLOG(1) << "[MakeStorage] Received DeviceBufferRef with dims: ["
               << absl::StrJoin(buffer_ref.dimensions(), ",") << "]"
               << " and dtype: " << ToString(buffer_ref.element_type())
               << " on device_idx: " << device_idx;
  const auto size = buffer_ref.size_bytes();
  return c10::Storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size,
      MakeDataPtr(std::move(buffer_ref), device_idx), GetTpuAllocator(),
      /*resizable=*/true));
}

at::Tensor MakeTensor(DeviceBufferRef buffer_ref, int device_idx) {
  ABSL_VLOG(1) << "[MakeTensor] Creating new ATen tensor for "
               << buffer_ref.DebugString() << " on device_idx: " << device_idx;

  const auto dtype = ConvertTo<at::ScalarType>(buffer_ref.element_type());
  auto caffe2_type_meta = c10::scalarTypeToTypeMeta(dtype);
  Dimensions sizes(buffer_ref.dimensions().begin(),
                   buffer_ref.dimensions().end());
  if (buffer_ref.element_type() == mlir::ElementType::F4E2M1FN) {
    ABSL_CHECK(!sizes.empty());         // CRASH_OK
    ABSL_CHECK_EQ(sizes.back() % 2, 0)  // CRASH_OK
        << "Last dimension of FP4 must be even, got " << sizes.back();
    sizes.back() /= 2;
  }

  c10::Storage storage = MakeStorage(std::move(buffer_ref), device_idx);
  at::Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      caffe2_type_meta));
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  tensor.unsafeGetTensorImpl()->set_storage_offset(0);
  ABSL_VLOG(1) << "[MakeTensor] Final ATen tensor created:" << ToString(tensor)
               << "\nscalar_type: " << ToString(tensor.scalar_type())
               << "\nsizes: " << ToString(tensor.sizes())
               << "\nstrides: " << ToString(tensor.strides())
               << (tensor.is_contiguous() ? " (contiguous)"
                                          : " (non-contiguous)")
               << "\nstorage nbytes: " << tensor.storage().nbytes()
               << "\nstorage offset: " << tensor.storage_offset();
  ABSL_CHECK_EQ(tensor.device().type(),  // CRASH_OK
                GetPrivateUse1DeviceType())
      << "Tensor created does NOT have PrivateUse1 device type.";
  return tensor;
}

class TpuAllocator final : public c10::DeviceAllocator {
 public:
  TpuAllocator() = default;

  // This class is move-only.
  TpuAllocator(TpuAllocator&& other) = default;
  TpuAllocator& operator=(TpuAllocator&& other) = default;
  TpuAllocator(const TpuAllocator&) = delete;
  TpuAllocator& operator=(const TpuAllocator&) = delete;

  c10::DataPtr allocate(size_t nbytes) override {
    ScopedPythonContextCapturer _(OpName::kTorchTpuStorageAllocate);
    c10::DeviceIndex device_idx = 0;
    if (const auto* device = PjrtBackend::GetInstance().GetDevice()) {
      device_idx =
          static_cast<c10::DeviceIndex>(device->local_hardware_id().value());
    }
    // Check that the size_t does not overflow an int64_t.
    // This function is only ever called from PyTorch so safe to throw an
    // exception on failure.
    TT_CHECK_THROW(
        nbytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        error::kResourceExhausted)
        << "allocation size " << nbytes
        << " overflows as a signed 64-bit integer";

    // Allocated DeviceBufferRef will be equivalent to
    // torch.empty(nbytes, dtype=torch.uint8) with
    // fill_uninitialized_memory=True, which fills the buffer with 0xFF bytes.
    TT_ASSIGN_OR_THROW(auto buffer_ref, CreateEmptyDeviceBufferRef(
                                            {static_cast<int64_t>(nbytes)},
                                            mlir::ElementType::UI8));
    return MakeDataPtr(std::move(buffer_ref), device_idx);
  }

  void copy_data(void* dest, const void* src,
                 std::size_t count) const override {
    TT_CHECK_THROW(false, error::kPythonNotImplementedError)
        << "TpuAllocator::copy_data is not implemented and should not be "
           "called directly for this conceptual allocator type.";
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return DeleteDeviceBufferRef;
  }

  bool initialized() override { return true; }

  void emptyCache(c10::MempoolId_t mempool_id) override {
    // No-op for now as PjRt handles memory management.
  }

  void recordStream(const c10::DataPtr& ptr, c10::Stream stream) override {
    // No-op for now.
  }

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override {
    c10::CachingDeviceAllocator::DeviceStats stats;
    auto pjrt_stats_or = PjrtBackend::GetInstance().GetAllocatorStats();
    if (!pjrt_stats_or.ok()) {
      TORCH_WARN("Failed to get allocator stats: ", pjrt_stats_or.status());
      return stats;
    }
    const auto& pjrt_stats = pjrt_stats_or.value();

    // Map available stats
    using StatType = c10::CachingAllocator::StatType;

    // Both allocated_bytes and active_bytes are fulfilled by bytes in use by
    // PjRt. There is no differentiation between the two in the underlying stats
    // implementation.
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        pjrt_stats.bytes_in_use;
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        pjrt_stats.peak_bytes_in_use;

    stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        pjrt_stats.bytes_in_use;
    stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        pjrt_stats.peak_bytes_in_use;

    // bytes_reserved -> reserved_bytes[all].current
    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        pjrt_stats.bytes_reserved;
    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        pjrt_stats.peak_bytes_reserved;

    // num_allocs -> allocation[all].current
    stats.allocation[static_cast<size_t>(StatType::AGGREGATE)].current =
        pjrt_stats.num_allocs;

    return stats;
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    TORCH_WARN_ONCE(
        "torch.accelerator.memory.reset_accumulated_memory_stats is not "
        "implemented for TPU.");
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    auto status = PjrtBackend::GetInstance().ClearAllocatorStats();
    if (!status.ok()) {
      TORCH_WARN("Failed to reset peak memory stats: ", status.ToString());
    }
  }

  std::pair<size_t, size_t>  // STD_PAIR_OK=dictated by PyTorch.
  getMemoryInfo(c10::DeviceIndex device) override {
    auto pjrt_stats_or = PjrtBackend::GetInstance().GetAllocatorStats();
    if (!pjrt_stats_or.ok()) {
      TORCH_WARN("Failed to get allocator stats: ", pjrt_stats_or.status());
      return {0, 0};
    }
    const auto& pjrt_stats = pjrt_stats_or.value();
    size_t limit = pjrt_stats.bytes_limit.value_or(0);
    size_t used = pjrt_stats.bytes_in_use;
    return {limit - used, limit};
  }
};

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
    xla::HostMemoryAllocator::OwnedPtr data = host_allocator->Allocate(nbytes);
    TT_CHECK_THROW(data != nullptr, error::kResourceExhausted)
        << "Failed to allocate " << nbytes << " bytes of pinned host memory";

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
    TORCH_WARN_ONCE(
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

at::HostAllocator* GetTpuPinnedAllocatorInternal() {
  static absl::NoDestructor<TpuPinnedAllocator> g_tpu_pinned_allocator;
  return g_tpu_pinned_allocator.get();
}

void TpuPinnedAllocator::DeleteTpuPinnedBufferStatic(void* ptr) {
  static_cast<TpuPinnedAllocator*>(GetTpuPinnedAllocatorInternal())
      ->free_ptr(ptr);
}

c10::Allocator* GetTpuAllocator() {
  static absl::NoDestructor<TpuAllocator> g_tpu_allocator;
  return g_tpu_allocator.get();
}

at::HostAllocator* GetTpuPinnedAllocator() {
  return GetTpuPinnedAllocatorInternal();
}

bool IsTpuPinnedPtr(const void* ptr) {
  return static_cast<TpuPinnedAllocator*>(GetTpuPinnedAllocatorInternal())
      ->is_pinned_ptr(ptr);
}

void RegisterTpuAllocator() {
  c10::SetAllocator(GetPrivateUse1DeviceType(), GetTpuAllocator());
  at::setHostAllocator(GetPrivateUse1DeviceType(), GetTpuPinnedAllocator());
}

void DeleteDeviceBufferRef(void* ctx_ptr) {
  DeviceBufferRef* const ref_ptr = static_cast<DeviceBufferRef*>(ctx_ptr);
  if (ref_ptr) {
    ABSL_VLOG(3) << "[c10::DataPtr deleter] deleting "
                 << ref_ptr->DebugString();
    ref_ptr->device_buffer_list()->DecrementLiveDataPtrs();
    // Don't check eager mode here.
    // Compiled mode trace tensors are never registered with
    // RecordNewDataPtrCreated, so recording their deletion is harmless.
    // Failing to record the deletion of an eager mode tensor (if it
    // happens during a torch.compile trace) would be a bug as the refcount
    // would never go to zero.
    RecordDataPtrDestroyed(*ref_ptr);
  }
  delete ref_ptr;
}

c10::DataPtr MakeDataPtr(DeviceBufferRef buffer_ref, const int device_idx) {
  auto* absl_nonnull const raw_ref_ptr =
      new DeviceBufferRef(std::move(buffer_ref));
  raw_ref_ptr->device_buffer_list()->IncrementLiveDataPtrs();
  // Tensors created during the torch.compile trace process don't need to be
  // tracked for the purposes of synchronization, as they can't be
  // materialized anyway.
  if (GetEagerMode() != EagerMode::kInternalDeferAll) {
    RecordNewDataPtrCreated(*raw_ref_ptr);
  }
  return c10::DataPtr(raw_ref_ptr, raw_ref_ptr, DeleteDeviceBufferRef,
                      c10::Device(GetPrivateUse1DeviceType(), device_idx));
}

absl::StatusOr<std::vector<DeviceBufferRef>> MaterializeAndReturn(
    absl::Span<const at::Tensor> tensors, MaterializationReason reason) {
  tsl::profiler::TraceMe trace("MaterializeAndReturn");
  if (tensors.empty()) {
    return std::vector<DeviceBufferRef>();
  }
  // We need to materialize both the base and view for each tensor.
  // Materialization is by node, not by buffer.
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  nodes_to_materialize.reserve(tensors.size() * 2);

  // We always return one buffer per input tensor.
  std::vector<DeviceBufferRef> buffers_to_return;
  buffers_to_return.reserve(tensors.size());

  for (const at::Tensor& tensor : tensors) {
    // If the tensor is a (non-trivial) view, these will be two different
    // buffers. If the tensor's base is the same shape and layout as the tensor,
    // these will be the same buffer.
    // Deduplication will happen inside the Materialize() call; this will cover
    // both the case where a tensor is a trivial view, and the case where two
    // tensors are different views of the same base buffer.
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef base_buffer_ref,
                        GetBaseBuffer(tensor));
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef view_buffer_ref,
                        GetBuffer(tensor));

    nodes_to_materialize.push_back(base_buffer_ref.device_buffer_list());
    nodes_to_materialize.push_back(view_buffer_ref.device_buffer_list());
    buffers_to_return.push_back(std::move(view_buffer_ref));
  }
  TT_RETURN_IF_ERROR(Materialize(nodes_to_materialize, reason,
                                 MaterializationMode::kSplitGraph));
  nodes_to_materialize.clear();
  return buffers_to_return;
}

namespace {

std::string LayoutToString(c10::optional<at::Layout> layout_opt) {
  if (layout_opt == at::Layout::Jagged) {
    return "torch.jagged";
  }
  if (layout_opt.has_value()) {
    std::stringstream ss;
    ss << layout_opt.value();
    return ss.str();
  }
  return "nullopt";  // Should not be reached.
}

}  // namespace

absl::StatusOr<at::Tensor> MakeEmptyMemoryFormat(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  TT_RET_CHECK(!dtype_opt.has_value() ||
                   dtype_opt.value() != at::ScalarType::ComplexHalf,
               error::kPythonNotImplementedError)
      << "TorchTPU does not yet support dtype complex32";
  // Check that we support all the provided options.
  CheckDeviceIsTpu(device_opt, "empty");
  TT_RET_CHECK(layout_opt.value_or(at::Layout::Strided) == at::Layout::Strided,
               error::kPythonNotImplementedError)
      << "only layout=torch.strided is supported by TorchTPU for now, "
         "got "
      << LayoutToString(layout_opt);

  // If device_opt is unspecified, we use the global default dtype.
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(dtype_opt.value_or(
                          c10::get_default_dtype_as_scalartype())));

  Dimensions xla_sizes = CopyIntVector(size);
  if (dtype == mlir::ElementType::F4E2M1FN) {
    TT_RET_CHECK(!xla_sizes.empty(), error::kInvalidArgument)
        << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional, "
           "got 0-dimensional";
    // A PyTorch tensor of shape [..., N] of dtype float4_e2m1fn_x2 represents
    // N container elements (each 1 byte), which contain 2N packed FP4 values.
    // On the TPU/XLA side, FP4 elements are represented as unpacked 1-byte
    // elements. To store all 2N values, the TPU buffer must have a physical
    // size of 2N bytes. We achieve this by doubling the size of the last
    // dimension in the XLA shape.
    xla_sizes.back() *= 2;
  }

  // Create an empty DeviceBufferRef.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer,
                      CreateEmptyDeviceBufferRef(std::move(xla_sizes), dtype));
  at::Tensor result = MakeTensor(std::move(buffer));

  // MakeTensor defaults to a contiguous tensor, and so
  // does aten::empty().
  if (memory_format_opt.value_or(at::MemoryFormat::Contiguous) !=
      at::MemoryFormat::Contiguous) {
    // This may throw an error if the memory format is not supported for
    // the given tensor shape.
    result.unsafeGetTensorImpl()->empty_tensor_restride(
        memory_format_opt.value());
  }
  return result;
}

absl::StatusOr<at::Tensor> MakeEmptyTensor(
    at::IntArrayRef size, c10::ScalarType dtype,
    c10::optional<at::Device> device_opt) {
  return MakeEmptyMemoryFormat(size, dtype, /*layout_opt=*/c10::nullopt,
                               device_opt, /*pin_memory_opt=*/c10::nullopt,
                               /*memory_format_opt=*/c10::nullopt);
}

}  // namespace torch_tpu
