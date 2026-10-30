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

#include "csrc/eager/tensor_to_buffer.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/CachingHostAllocator.h"
#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
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
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/core/impl/VirtualGuardImpl.h"
#include "c10/util/Optional.h"
#include "c10/util/UniqueVoidPtr.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

// Records that a DeviceBufferRef has one more c10::DataPtr using it.
void AddDataPtrAlias(const DeviceBufferRef& buffer_ref) {
  buffer_ref.device_buffer_list()->IncrementLiveDataPtrs(buffer_ref.index());
}

// Records that a ref_ptr has one fewer DataPtr using it.
void RemoveDataPtrAlias(const DeviceBufferRef& ref) {
  ABSL_VLOG(3) << "[RemoveDataPtrAlias] Cleaning up DataPtr usage for "
               << ref.DebugString();
  ref.device_buffer_list()->DecrementLiveDataPtrs(ref.index());
}

// Delegate responsibility for deleting the DeviceBufferRef to the
// c10::DataPtr.
// The DeviceBufferRef* is used as both the "data" and "context" of the
// c10::DataPtr; this mirrors the semantics of a std::unique_ptr.
void DeleteDeviceBufferRef(void* ctx_ptr) {
  DeviceBufferRef* const ref_ptr = static_cast<DeviceBufferRef*>(ctx_ptr);
  if (ref_ptr) RemoveDataPtrAlias(*ref_ptr);
  delete ref_ptr;
}

// Overwrites the referent of the data_ptr to be the new buffer_ref.
//
// This does *not* modify the ctx_ or deleter of the data_ptr. This is because
// PyTorch will sometimes create "borrowing" DataPtrs, which share the data_
// of another "owning" DataPtr, but where the deleter is a no-op for the
// borrowing pointer and DeleteDeviceBufferRef for the owning pointer.
//
// An in-place write (AssignBufferToAtTensor) on a "borrowing" DataPtr should
// be reflected in the owning DataPtr and other borrowing DataPtrs as well;
// this means that the address of the data_ must be unchanged, but needs to be
// replaced with the SharedDeviceBufferList and list index of the new
// buffer_ref so that it references the updated data.
void OverwriteDataPtr(const c10::DataPtr& data_ptr,
                      DeviceBufferRef&& buffer_ref) {
  DeviceBufferRef* const old_ref_ptr =
      static_cast<DeviceBufferRef*>(data_ptr.get());
  ABSL_CHECK(old_ref_ptr)  // CRASH_OK
      << "tensor storage has a null DeviceBufferRef* via c10::DataPtr::get()";
  RemoveDataPtrAlias(*old_ref_ptr);
  AddDataPtrAlias(buffer_ref);

  *old_ref_ptr = std::move(buffer_ref);
}

}  // namespace

absl::Status AssignBufferToAtTensor(DeviceBufferRef result_buf,
                                    const at::Tensor& tensor,
                                    bool enable_autodonation,
                                    bool bump_version) {
  TT_RET_CHECK(tensor.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "expected tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << tensor.device().str();
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

  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref, GetBaseBuffer(tensor));

  // A full overwrite will replace every byte of the original base buffer, so
  // the base buffer can simply be dropped and doesn't need explicit donation.
  const bool full_overwrite =
      tensor.storage_offset() == 0 &&
      tensor.storage().nbytes() == result_buf.size_bytes();
  // A full overwrite that is also contiguous and non-conjugate doesn't require
  // a view inversion at all.
  const bool is_contiguous_overwrite =
      full_overwrite && tensor.is_contiguous() && !tensor.is_conj();
  // Writing non-boolean data to a boolean buffer requires changing the
  // data type of the storage to UINT8 to avoid losing data; this prevents
  // donation.
  const bool write_bytes_to_bool =
      base_buffer_ref.element_type() == mlir::ElementType::PRED &&
      result_buf.element_type() != mlir::ElementType::PRED;

  // We can donate the base buffer if:
  //  - Autodonation is enabled, and
  //  - The write is not a full overwrite of the base buffer, and
  //  - The base buffer doesn't need expansion to hold non-boolean data, and
  //  - We're overwriting the last live data pointer of the base buffer.
  const bool auto_donate = enable_autodonation && !full_overwrite &&
                           !write_bytes_to_bool &&
                           base_buffer_ref.live_data_ptrs() <= 1;

  if (is_contiguous_overwrite) {
    // The result_buf can be used directly as the new storage buffer.
    //
    // It's possible for an operation like
    // ```
    //   a = torch.zeros(100)
    //   b = a.view(10, 10)
    //   b.add_(1.0)
    // ```
    // to change the shape of the base DeviceBufferRef from one contiguous shape
    // to another equivalent shape, in this case (100,) -> (10, 10). In some
    // sense, this changes from `b` being a view of `a`, to `a` being a view
    // of `b`, in a "last write wins" ordering.
    //
    // In this case, the base can still be donated to the `add_` operation, but
    // not into an AsStridedInverse view operation (since none is created).
    ABSL_VLOG(2) << "[AssignBufferToAtTensor] Directly assigning contiguous "
                    "base buffer of "
                 << result_buf.size_bytes() << " bytes.";

    // Overwrite the c10::DataPtr in storage to use the new DeviceBufferRef.
    // This ensures that all active views on this same contiguous base tensor
    // will be updated by the write.
    OverwriteDataPtr(tensor.storage().data_ptr(), std::move(result_buf));
  } else {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef new_base_buffer_ref,
                        CreateInverseViewDeviceBufferRef(
                            std::move(base_buffer_ref), std::move(result_buf),
                            tpu_layout, tensor.is_conj(),
                            /*donate_base=*/auto_donate));

    // Overwrite the c10::DataPtr in storage to use the new DeviceBufferRef.
    // All reads that happened before the write will be unaffected, but reads
    // to all extant views will reflect the new value on later accesses.
    OverwriteDataPtr(tensor.storage().data_ptr(),
                     std::move(new_base_buffer_ref));
  }

  // Bump the version counter for autograd in-place tracking and
  // functionalization.
  if (bump_version) {
    tensor.unsafeGetTensorImpl()->bump_version();
  }

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

bool IsFullStorageReinterpretView(const at::Tensor& tensor,
                                  const DeviceBufferRef& base) {
  if (!tensor.defined() || tensor.is_conj() || !tensor.is_contiguous() ||
      tensor.storage_offset() != 0) {
    return false;
  }
  if (!base.dynamic_dimensions().empty() || base.on_device_shape_is_dynamic()) {
    return false;
  }
  const size_t base_bytes = base.size_bytes();
  if (tensor.nbytes() != base_bytes ||
      tensor.storage().nbytes() != base_bytes) {
    return false;
  }
  const absl::StatusOr<mlir::ElementType> tensor_type =
      ConvertTo<mlir::ElementType>(tensor.scalar_type());
  if (!tensor_type.ok()) {
    return false;
  }
  // Only types whose XLA representation is a whole number of bytes can be
  // reinterpreted byte-for-byte; this excludes PRED (1 bit) and the packed
  // 4-bit types.
  return XlaEquivalentBitwidth(*tensor_type) % 8 == 0 &&
         XlaEquivalentBitwidth(base.element_type()) % 8 == 0;
}

absl::Status AssignBufferAsStorageBase(DeviceBufferRef new_base,
                                       const at::Tensor& tensor) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked by caller.
      IsFullStorageReinterpretView(tensor, new_base), error::kInvalidArgument)
      << "tensor of dtype " << ToString(tensor.scalar_type()) << " and shape "
      << ToString(tensor.sizes()) << " with storage of "
      << tensor.storage().nbytes()
      << " bytes is not a full-storage reinterpretation of buffer "
      << new_base.DebugString();
  ABSL_VLOG(2) << "[AssignBufferAsStorageBase] Replacing storage base with "
               << new_base.DebugString() << " for tensor of dtype "
               << ToString(tensor.scalar_type()) << " and shape "
               << ToString(tensor.sizes());

  // Overwrite the c10::DataPtr in storage to use the new DeviceBufferRef, so
  // that all views sharing the storage observe the write.
  OverwriteDataPtr(tensor.storage().data_ptr(), std::move(new_base));

  // Bump the version to tell PyTorch how to handle functionalization
  tensor.unsafeGetTensorImpl()->bump_version();
  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const at::Tensor& tensor) {
  TT_RET_CHECK(tensor.defined(), error::kInvalidArgument)
      << "tensor is undefined";

  TT_RET_CHECK(tensor.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "expected tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << tensor.device();

  TT_RET_CHECK(tensor.storage().data_ptr(), error::kPythonNotImplementedError)
      << "tensor is on the custom PrivateUse1 device. But its storage data_ptr "
         "is null. This is usually caused by FakeTensor being run on TPU ops. "
         "And it is not supported";

  ABSL_CHECK(tensor.storage().allocator() == nullptr ||  // CRASH_OK
             tensor.storage().allocator() == GetTpuAllocator())
      << "If tensor storage is owning, it must be allocated by g_tpu_allocator";

  return GetBaseBuffer(tensor.storage());
}

absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const c10::Storage& storage) {
  // The DeviceBufferRef in the c10::StorageImpl represents the "contiguous
  // base" buffer--this is the shape of the original tensor that was used to
  // construct the tensor, and represents all of the data available to all
  // views that share the same Storage.
  const DeviceBufferRef* base_buffer_ref =
      static_cast<const DeviceBufferRef*>(storage.data_ptr().get());
  ABSL_CHECK(base_buffer_ref)  // CRASH_OK
      << "tensor storage has a null DeviceBufferRef* via c10::DataPtr::get()";
  return *base_buffer_ref;
}

absl::StatusOr<DeviceBufferRef> GetBuffer(const at::Tensor& tensor,
                                          DonateBase donate_base) {
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
                                   tensor.is_conj(), donate_base);
}

absl::StatusOr<DeviceBufferRef> GetBuffer(const at::Tensor& tensor) {
  return GetBuffer(tensor, DonateBase::kDoNotDonate);
}

absl::StatusOr<std::vector<DeviceBufferRef>> GetBuffers(
    absl::Span<const at::Tensor> tensors,
    absl::Span<const int64_t> donated_indices) {
  std::vector<DeviceBufferRef> buffers;
  buffers.reserve(tensors.size());
  if (donated_indices.empty()) {
    for (const auto& tensor : tensors) {
      TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer, GetBuffer(tensor));
      buffers.push_back(std::move(buffer));
    }
  } else {
    for (int64_t i = 0; i < tensors.size(); ++i) {
      const DonateBase donate_base =
          (absl::c_find(donated_indices, i) != donated_indices.end())
              ? DonateBase::kDonate
              : DonateBase::kDoNotDonate;
      TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer,
                          GetBuffer(tensors[i], donate_base));
      buffers.push_back(std::move(buffer));
    }
  }
  return buffers;
}

absl::StatusOr<std::vector<DeviceBufferRef>> GetBuffers(
    absl::Span<const at::Tensor> tensors) {
  return GetBuffers(tensors, {});
}

c10::Storage MakeStorage(DeviceBufferRef buffer_ref,
                         std::optional<c10::DeviceIndex> device_idx_opt) {
  const c10::DeviceIndex device_idx =
      device_idx_opt.value_or(buffer_ref.device_index());
  ABSL_VLOG(1) << "[MakeStorage] Received DeviceBufferRef with dims: ["
               << absl::StrJoin(buffer_ref.dimensions(), ",") << "]"
               << " and dtype: " << ToString(buffer_ref.element_type())
               << " on device_idx: " << static_cast<int>(device_idx);
  const auto size = buffer_ref.size_bytes();
  return c10::Storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size,
      MakeDataPtr(std::move(buffer_ref), device_idx), GetTpuAllocator(),
      /*resizable=*/true));
}

at::Tensor MakeTensor(DeviceBufferRef buffer_ref,
                      std::optional<c10::DeviceIndex> device_idx_opt) {
  const c10::DeviceIndex device_idx =
      device_idx_opt.value_or(buffer_ref.device_index());
  ABSL_VLOG(1) << "[MakeTensor] Creating new ATen tensor for "
               << buffer_ref.DebugString()
               << " on device_idx: " << static_cast<int>(device_idx);

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
    // Propagate the initialization error (e.g. VFIO collision) instead of a
    // generic one.
    TT_THROW_IF_ERROR(PjrtBackend::GetInstance().EnsureInitialized());
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
    // Tag the DataPtr with the device index the buffer was created on (the
    // current device of the calling thread), so that the tensor's device
    // metadata matches the events queue the buffer belongs to.
    const c10::DeviceIndex device_idx = buffer_ref.device_index();
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
    ValidateDevice(device);
    c10::CachingDeviceAllocator::DeviceStats stats;
    auto pjrt_stats_or = PjrtBackend::GetInstance().GetAllocatorStats();
    if (!pjrt_stats_or.ok()) {
      TT_WARN(WarningCategory::kPerformance,
              "Failed to get allocator stats: ", pjrt_stats_or.status());
      return stats;
    }
    const auto& pjrt_stats = pjrt_stats_or.value();

    // Map available stats
    using StatType = c10::CachingAllocator::StatType;

    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        pjrt_stats.bytes_in_use + pjrt_stats.bytes_reserved;
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        pjrt_stats.peak_allocated_bytes;

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
    ValidateDevice(device);
    TT_WARN_ONCE(
        WarningCategory::kTorchTpu,
        "torch.accelerator.memory.reset_accumulated_memory_stats is not "
        "implemented for TPU.");
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    ValidateDevice(device);
    auto status = PjrtBackend::GetInstance().ClearAllocatorStats();
    if (!status.ok()) {
      TT_WARN(WarningCategory::kPerformance,
              "Failed to reset peak memory stats: ", status.ToString());
    }
  }

  std::pair<size_t, size_t>  // STD_PAIR_OK=dictated by PyTorch.
  getMemoryInfo(c10::DeviceIndex device) override {
    ValidateDevice(device);
    auto pjrt_stats_or = PjrtBackend::GetInstance().GetAllocatorStats();
    if (!pjrt_stats_or.ok()) {
      TT_WARN(WarningCategory::kPerformance,
              "Failed to get allocator stats: ", pjrt_stats_or.status());
      return {0, 0};
    }
    const auto& pjrt_stats = pjrt_stats_or.value();
    size_t limit = pjrt_stats.bytes_limit.value_or(0);
    size_t used = pjrt_stats.bytes_in_use;
    return {limit - used, limit};
  }

 private:
  static void ValidateDevice(c10::DeviceIndex device) {
    if (c10::impl::hasDeviceGuardImpl(GetPrivateUse1DeviceType())) {
      c10::impl::VirtualGuardImpl guard_impl(GetPrivateUse1DeviceType());
      guard_impl.getDefaultStream(
          c10::Device(GetPrivateUse1DeviceType(), device));
    }
  }
};

c10::Allocator* GetTpuAllocator() {
  static absl::NoDestructor<TpuAllocator> g_tpu_allocator;
  return g_tpu_allocator.get();
}

void RegisterTpuAllocator() {
  c10::SetAllocator(GetPrivateUse1DeviceType(), GetTpuAllocator());
  at::setHostAllocator(GetPrivateUse1DeviceType(), GetTpuPinnedAllocator());
}

c10::DataPtr MakeDataPtr(DeviceBufferRef buffer_ref,
                         const c10::DeviceIndex device_idx) {
  AddDataPtrAlias(buffer_ref);
  auto* absl_nonnull const raw_ref_ptr =
      new DeviceBufferRef(std::move(buffer_ref));
  return c10::DataPtr(raw_ref_ptr, raw_ref_ptr, DeleteDeviceBufferRef,
                      c10::Device(GetPrivateUse1DeviceType(), device_idx));
}

absl::StatusOr<DeviceBufferRef> MaterializeAndReturn(
    const at::Tensor& tensor, MaterializationReason reason,
    MaterializationBlockMode block_mode) {
  tsl::profiler::TraceMe trace("MaterializeAndReturn");
  // If the tensor is a (non-trivial) view, these will be two different
  // buffers. If the tensor's base is the same shape and layout as the tensor,
  // these will be the same buffer.
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef base_buffer_ref,
                      GetBaseBuffer(tensor));
  TT_ASSIGN_OR_RETURN(DeviceBufferRef view_buffer_ref, GetBuffer(tensor));
  if (base_buffer_ref.device_index() != view_buffer_ref.device_index() ||
      base_buffer_ref.stream_id() != view_buffer_ref.stream_id()) {
    // The base and view were created on different streams, but the implied
    // dependency forces the base buffer to materialize first. Block until the
    // base buffer has been enqueued to avoid scheduling the view buffer before
    // its input.
    //
    // Note: in most cases, the materialization on exiting the prior stream
    // will have already forced the base buffer to materialize and this will
    // be a no-op. But exceptions are possible (such as in a
    // one-stream-per-thread setup).
    if (!base_buffer_ref.is_materializing()) {
      TT_RETURN_IF_ERROR(
          Materialize(base_buffer_ref, reason, GraphSplitMode::kSplitGraph,
                      MaterializationBlockMode::kBlockUntilStarted));
    }
    TT_RETURN_IF_ERROR(Materialize(view_buffer_ref, reason,
                                   GraphSplitMode::kSplitGraph, block_mode));
  } else {
    // Deduplication will happen inside the Materialize() call; this will cover
    // the case where a tensor is a trivial view.
    // If the view is not trivial but the base and view are on the same stream,
    // we can materialize them jointly.
    TT_RETURN_IF_ERROR(Materialize({base_buffer_ref, view_buffer_ref}, reason,
                                   GraphSplitMode::kSplitGraph, block_mode));
  }

  return view_buffer_ref;
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
  const auto layout = layout_opt.value_or(at::Layout::Strided);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches other layouts first.
      layout == at::Layout::Strided || layout == at::Layout::Jagged,
      error::kPythonNotImplementedError)
      << "expected layout to be either torch.strided or torch.jagged (other "
         "layouts are not yet supported), got "
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
