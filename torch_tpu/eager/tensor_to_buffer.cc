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
#include <utility>
#include <vector>

#include "ATen/core/CachingHostAllocator.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/declare.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Allocator.h"
#include "c10/core/CachingDeviceAllocator.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/core/Storage.h"
#include "c10/core/StorageImpl.h"
#include "c10/core/Stream.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/Exception.h"
#include "c10/util/UniqueVoidPtr.h"
#include "c10/util/intrusive_ptr.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/common/status_builder.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_buffer_utils.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/experimental/eager/materialize_new.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/host_memory_allocator.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_new_materialization);

namespace torch_tpu {

namespace {

at::ScalarType GetScalarType(const c10::TensorImpl& tensor) {
  // For some reason, c10::TensorImpl does not have a scalar_type() method;
  // we have to use "dtype()" and convert instead.
  auto type_meta = tensor.dtype();
  return c10::typeMetaToScalarType(type_meta);
}

}  // namespace

absl::Status AssignBufferToAtTensor(DeviceBufferRef result_buf,
                                    const at::Tensor& tensor) {
  ABSL_VLOG(1)
      << "[AssignBufferToAtTensor] Assigning DeviceBufferRef to existing "
         "tensor\n"
      << result_buf.DebugString() << "\n"
      << "\nTensor:" << ToString(tensor)
      << "\n\tscalar_type: " << c10::toString(tensor.scalar_type())
      << "\n\tsizes: " << ToString(absl::MakeConstSpan(tensor.sizes()))
      << "\n\tstrides: " << ToString(absl::MakeConstSpan(tensor.strides()))
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
  ABSL_CHECK_EQ(result_buf.num_elements(), tensor.numel());  // CRASH_OK
  TT_ASSIGN_OR_RETURN(const auto tensor_element_type,
                      ConvertTo<mlir::ElementType>(tensor.scalar_type()));
  ABSL_CHECK_EQ(result_buf.element_type(), tensor_element_type);  // CRASH_OK

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
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef new_base_buffer_ref,
        CreateInverseViewDeviceBufferRef(
            std::move(base_buffer_ref), std::move(result_buf), tensor.sizes(),
            tensor.strides(), tensor.storage_offset(), tensor_element_type,
            tensor.is_conj()));

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
               << "\nscalar_type: " << c10::toString(tensor.scalar_type())
               << "\nsizes: " << ToString(absl::MakeConstSpan(tensor.sizes()))
               << "\nstrides: "
               << ToString(absl::MakeConstSpan(tensor.strides()))
               << (tensor.is_contiguous() ? " (contiguous)"
                                          : " (non-contiguous)")
               << "\nstorage nbytes: " << tensor.storage().nbytes()
               << "\nstorage offset: " << tensor.storage_offset();
  return absl::OkStatus();
}
absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const c10::TensorImpl& tensor) {
  // If the user tries to call a torch_tpu kernel with tensors that are
  // on different devices, then we return an error status, which will be
  // propagated to the user as a Python exception (with context).
  // We only need to uphold invariants for tensors torch_tpu constructs.
  TT_RET_CHECK(tensor.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "tensor is expected to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << tensor.device().str();

  // This will trigger a device mismatch in Pytorch, with a good call stack.
  TT_RET_CHECK(tensor.storage().data_ptr(), error::kUnimplemented)
      << "tensor is on the custom PrivateUse1 device. But its storage data_ptr "
         "is null. This is usually caused by FakeTensor being run on TPU ops. "
         "And it is not supported";

  // If the tensor is on the PrivateUse1 device, then we check invariants.
  // Any errors indicate serious bugs and we crash to prevent worsening the
  // application state.
  ABSL_CHECK_EQ(tensor.storage().allocator(), GetTpuAllocator())  // CRASH_OK
      << "tensor is on PrivateUse1 device, but is not allocated by "
         "g_tpu_allocator";

  return GetBaseBuffer(tensor.storage());
}

absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const at::Tensor& tensor) {
  const c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined.";
  return GetBaseBuffer(*tensor.unsafeGetTensorImpl());
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
absl::StatusOr<DeviceBufferRef> GetBuffer(const c10::TensorImpl& tensor) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref, GetBaseBuffer(tensor));

  // Get the element types; if they're different, we need to do a bitwise cast.
  auto scalar_type = GetScalarType(tensor);
  TT_ASSIGN_OR_RETURN(const auto tensor_element_type,
                      ConvertTo<mlir::ElementType>(scalar_type));

  if (!tensor.is_conj() && tensor.is_contiguous() &&
      tensor.storage_offset() == 0 &&
      tensor.sizes() == base_buffer_ref.dimensions() &&
      tensor_element_type == base_buffer_ref.element_type()) {
    ABSL_VLOG(1) << "[GetBuffer] Tensor is a contiguous base, "
                    "returning base buffer.";
    // This is the base tensor, no view to apply.
    return base_buffer_ref;
  }

  return CreateViewDeviceBufferRef(base_buffer_ref, tensor.sizes(),
                                   tensor.strides(), tensor.storage_offset(),
                                   tensor_element_type, tensor.is_conj());
}

absl::StatusOr<DeviceBufferRef> GetBuffer(const at::Tensor& tensor) {
  // Trying to get the buffer from an undefined tensor is a user error, not
  // a critical invariant violation.
  const c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined";
  return GetBuffer(*tensor.unsafeGetTensorImpl());
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

c10::Storage MakeStorage(DeviceBufferRef buffer_ref) {
  ABSL_VLOG(1) << "[MakeStorage] Received DeviceBufferRef with dims: ["
               << absl::StrJoin(buffer_ref.dimensions(), ",") << "]"
               << " and dtype: " << ToString(buffer_ref.element_type());
  const auto size = buffer_ref.size_bytes();
  return c10::Storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size,
      MakeDataPtr(std::move(buffer_ref), /*device_idx=*/0), GetTpuAllocator(),
      /*resizable=*/true));
}

at::Tensor MakeTensor(DeviceBufferRef buffer_ref) {
  ABSL_VLOG(1) << "[MakeTensor] Creating new ATen tensor for "
               << buffer_ref.DebugString();

  const auto dtype = ConvertTo<at::ScalarType>(buffer_ref.element_type());
  auto caffe2_type_meta = c10::scalarTypeToTypeMeta(dtype);
  absl::Span<const int64_t> sizes = buffer_ref.dimensions();

  c10::Storage storage = MakeStorage(std::move(buffer_ref));
  at::Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      caffe2_type_meta));
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  tensor.unsafeGetTensorImpl()->set_storage_offset(0);
  ABSL_VLOG(1) << "[MakeTensor] Final ATen tensor created:" << ToString(tensor)
               << "\nscalar_type: " << c10::toString(tensor.scalar_type())
               << "\nsizes: " << ToString(absl::MakeConstSpan(tensor.sizes()))
               << "\nstrides: "
               << ToString(absl::MakeConstSpan(tensor.strides()))
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
    TT_CHECK_THROW(false, error::kUnimplemented)
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
    TORCH_WARN_ONCE(
        "torch.accelerator.memory.reset_peak_memory_stats is not implemented "
        "for TPU.");
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

namespace {

// Translates a raw background execution Out-of-Memory (OOM) error to a
// high-level exception that is easy to understand by PyTorch users. Under
// experimental asynchronous materialization, compilation and execution are
// enqueued to the background thread. If an execution OOM happens in the
// background, it is caught early during BlockOnPendingMaterializations() inside
// GetMaterialized(), bypassing the standard copy-transfer wrapper's
// (TranslateXlaTensorOomError) execution blocks. We replicate the OOM mapping
// here to ensure user exception signature consistency.
absl::Status TranslateXlaOomError(const absl::Status& status,
                                  const at::Tensor& tensor) {
  if (!IsXlaOomError(status)) {
    return status;
  }
  auto dtype_or = ConvertTo<mlir::ElementType>(tensor.scalar_type());
  if (!dtype_or.ok()) {
    return status;
  }
  return TT_ERROR(error::kResourceExhausted)
         << "the TPU ran out of memory while awaiting the materialization of "
            "value "
         << ToString(*dtype_or) << "[" << absl::StrJoin(tensor.sizes(), ", ")
         << "]:\n"
         << status.message();
}

absl::Status MaybeBlockOnPendingMaterializationsNew(const at::Tensor& tensor) {
  if (GetFlagOnce<bool,
                  &FLAGS_torch_tpu_internal_enable_new_materialization>()) {
    // Under experimental asynchronous materialization, the main thread blocks
    // and awaits for the background worker compilation and execution thread to
    // catch up before returning the resolved device buffer reference.
    auto status = BlockOnPendingMaterializations();
    if (!status.ok()) {
      if (IsXlaOomError(status)) {
        // 1. If it is an execution OOM error, translate it to standard format
        //    while preserving the default copy caller context prefix
        //    ("to_copy").
        return TranslateXlaOomError(status, tensor);
      } else {
        // 2. If it is a compilation error, wrap the existing status natively
        //    via StatusBuilder to preserve the background thread's original
        //    operator name context payload (e.g., "gather", "scatter",
        //    "cumprod") captured during graph construction under
        //    ScopedPythonContextProvider. Do NOT reconstruct the status (e.g.
        //    using TT_ERROR), as that would overwrite the payload with the copy
        //    thread's "to_copy" wrapper.
        return ::torch_tpu::StatusBuilder(std::move(status)).SetPrepend()
               << "materialization failed with: ";
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<DeviceBufferRef> GetMaterialized(const at::Tensor& tensor,
                                                MaterializationReason reason) {
  tsl::profiler::TraceMe trace("GetMaterialized");
  // Make sure the base DeviceBufferRef is materialized
  const auto* tensor_impl = tensor.unsafeGetTensorImpl();
  TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined";
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef base_buffer_ref,
                      GetBaseBuffer(*tensor_impl));
  TT_RETURN_IF_ERROR(
      Materialize(base_buffer_ref, reason, MaterializationMode::kSplitGraph));

  // Get the view DeviceBufferRef (may be the same as the base)
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef view_buffer_ref, GetBuffer(tensor));
  // Materialize the view (no-op if the tensor is a continuous base tensor)
  TT_RETURN_IF_ERROR(
      Materialize(view_buffer_ref, reason, MaterializationMode::kSplitGraph));
  // The new materialization algorithm needs us to block here until all pending
  // jobs are done.
  TT_RETURN_IF_ERROR(MaybeBlockOnPendingMaterializationsNew(tensor));

  return view_buffer_ref;
}

absl::StatusOr<std::vector<DeviceBufferRef>> GetMaterialized(
    absl::Span<const at::Tensor> tensors, MaterializationReason reason) {
  tsl::profiler::TraceMe trace("GetMaterialized (batch)");
  if (tensors.empty()) {
    return std::vector<DeviceBufferRef>();
  }
  // Materialize all of the base DeviceBufferRefs (in a single execution)
  std::vector<DeviceBufferRef> base_buffer_refs;
  base_buffer_refs.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    const auto* tensor_impl = tensor.unsafeGetTensorImpl();
    TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined";
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef base_buffer_ref,
                        GetBaseBuffer(*tensor_impl));
    base_buffer_refs.push_back(base_buffer_ref);
  }
  TT_RETURN_IF_ERROR(
      Materialize(base_buffer_refs, reason, MaterializationMode::kSplitGraph));

  // Materialize all of the views (no-op if all tensors are contiguous bases)
  std::vector<DeviceBufferRef> view_buffer_refs;
  view_buffer_refs.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef view_buffer_ref,
                        GetBuffer(tensor));
    view_buffer_refs.push_back(view_buffer_ref);
  }
  TT_RETURN_IF_ERROR(
      Materialize(view_buffer_refs, reason, MaterializationMode::kSplitGraph));
  // The new materialization algorithm needs us to block here until all pending
  // jobs are done.
  TT_RETURN_IF_ERROR(MaybeBlockOnPendingMaterializationsNew(tensors[0]));

  return view_buffer_refs;
}

}  // namespace torch_tpu
