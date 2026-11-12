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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
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
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
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
#include "c10/core/impl/COW.h"
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
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/graph_capture_state.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/eager/materialize.h"
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

// A tensor storage that existed before graph capture and was read or written
// during capture.
struct PreexistingCapturedStorage {
  c10::Storage storage;
  // The DeviceBufferRef held by `storage` when capture first accessed it.
  DeviceBufferRef initial_buffer;
  // The placeholder installed on the first read of `storage`, if it was read
  // before being written.
  std::optional<DeviceBufferRef> placeholder;
};

// Tracks the tensor storages accessed during a graph capture on its owner
// thread.
//
// Every tracked c10::StorageImpl is held by a strong reference until capture
// ends, so that its address cannot be reused by a different storage.
struct GraphCaptureStorageState {
  // The stream that capture records operations from.
  DeviceStreamId capture_stream;
  // True when capture tracking began while stream capture was active; used to
  // ignore stale TLS state if ClearAllStreams() resets stream capture state.
  bool require_stream_capture = false;
  // Storages that existed before capture, in the order they were first
  // accessed.
  std::vector<PreexistingCapturedStorage> preexisting;
  absl::flat_hash_set<const c10::StorageImpl*> preexisting_impls;
  // Storages created during capture, in creation order. At the end of capture,
  // a storage that is referenced only from here was freed during capture.
  std::vector<c10::Storage> created;
  absl::flat_hash_set<const c10::StorageImpl*> created_impls;
};

namespace {

// The storage tracking state of the graph capture that the current thread
// owns, if any. Not owning: the graph capture keeps the state alive until
// EndGraphCaptureStorageTracking() clears this pointer.
thread_local  // CPP_THREAD_LOCAL_OK=ownership stays on the capturing thread
    GraphCaptureStorageState* current_thread_capture_storage = nullptr;

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

// Returns the storage tracking state of the graph capture that the current
// thread owns, or nullptr if the current thread does not capture a graph.
GraphCaptureStorageState* absl_nullable GetCurrentCaptureStorageState() {
  GraphCaptureStorageState* const state = current_thread_capture_storage;
  if (ABSL_PREDICT_TRUE(state == nullptr)) {
    return nullptr;
  }
  if (state->require_stream_capture && !IsCurrentThreadCapturingGraph()) {
    // Reconcile TLS if ClearAllStreams() ended stream capture while the
    // previous graph's storage state was still referenced on this thread.
    current_thread_capture_storage = nullptr;
    return nullptr;
  }
  // The tensors of an explicit FX trace during capture, such as the tracing of
  // a higher-order op like `scan`, belong to the trace, not to the graph.
  if (GetEagerMode() == EagerMode::kInternalCompileFxGraph) {
    return nullptr;
  }
  return state;
}

// Returns the DeviceBufferRef slot that `storage` holds via its c10::DataPtr.
DeviceBufferRef& GetStorageSlot(const c10::Storage& storage) {
  DeviceBufferRef* const slot =
      static_cast<DeviceBufferRef*>(storage.data_ptr().get());
  ABSL_CHECK(slot)  // CRASH_OK
      << "tensor storage has a null DeviceBufferRef* via c10::DataPtr::get()";
  return *slot;
}

// Replaces the DeviceBufferRef in `slot` with `new_ref`.
void ReplaceSlotBuffer(DeviceBufferRef& slot, DeviceBufferRef new_ref) {
  RemoveDataPtrAlias(slot);
  AddDataPtrAlias(new_ref);
  slot = std::move(new_ref);
}

// Returns an error if graph capture is about to write `buffer` to a tensor
// storage on a stream other than the capture stream, or if `buffer` is the
// result of an operation issued on a device or stream other than the capture
// stream: capture records only the operations issued on the capture stream, so
// the graph would miss the write. Besides, materializing such a result could
// hang on the capture placeholders it read.
absl::Status ValidateWrittenOnCaptureStream(
    const GraphCaptureStorageState& state, const DeviceBufferRef& buffer) {
  // A write that runs no operation, such as a copy between two tensors of the
  // same shape and element type, only moves a buffer into the storage.
  TT_RETURN_IF_ERROR(ValidateGraphCaptureStream(state.capture_stream,
                                                GetCurrentDeviceStreamId()));
  if (!buffer.is_deferred()) {
    return absl::OkStatus();
  }
  return ValidateGraphCaptureStream(
      state.capture_stream,
      DeviceStreamId{.device_index = buffer.device_index(),
                     .stream_id = buffer.stream_id()});
}

// Returns true if `storage` existed before capture and capture has not
// accessed it yet.
bool IsUntrackedPreexistingStorage(const GraphCaptureStorageState& state,
                                   const c10::Storage& storage) {
  const c10::StorageImpl* const impl = storage.unsafeGetStorageImpl();
  return !state.created_impls.contains(impl) &&
         !state.preexisting_impls.contains(impl);
}

// Starts tracking `storage`, which existed before capture and holds
// `initial_buffer`.
void TrackPreexistingStorage(GraphCaptureStorageState& state,
                             const c10::Storage& storage,
                             DeviceBufferRef initial_buffer,
                             std::optional<DeviceBufferRef> placeholder) {
  state.preexisting_impls.insert(storage.unsafeGetStorageImpl());
  state.preexisting.push_back(
      PreexistingCapturedStorage{.storage = storage,
                                 .initial_buffer = std::move(initial_buffer),
                                 .placeholder = std::move(placeholder)});
}

// Replaces `slot`, the buffer of `storage`, with a placeholder, so that the
// captured graph reads the contents of `storage` at replay time instead of
// baking in its current buffer. `storage` must be an untracked storage that
// existed before capture.
//
// The placeholder is in a failed state: only the replays of the graph provide
// its contents, so a thread that reads `storage` until capture ends fails
// instead of waiting forever.
absl::Status InstallCapturePlaceholder(GraphCaptureStorageState& state,
                                       const c10::Storage& storage,
                                       DeviceBufferRef& slot) {
  // Placeholders represent fixed-shape storage slots at replay time.
  Shape placeholder_shape(CopyIntVector(slot.dimensions()),
                          slot.element_type());
  if (slot.shape().layout().has_value()) {
    placeholder_shape.set_layout(*slot.shape().layout());
  }
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef placeholder,
      DeviceBufferList::CreatePendingExecution(
          std::move(placeholder_shape), slot.device_index(), slot.stream_id()));
  placeholder.device_buffer_list()->SetAsError(
      TensorAccessedByGraphCaptureError());
  if (slot.is_pinned_host()) {
    placeholder.set_is_pinned_host();
  }
  TrackPreexistingStorage(state, storage, slot, placeholder);
  ReplaceSlotBuffer(slot, std::move(placeholder));
  return absl::OkStatus();
}

// Overwrites the referent of the storage's data_ptr to be the new buffer_ref.
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
absl::Status OverwriteDataPtr(const c10::Storage& storage,
                              DeviceBufferRef&& buffer_ref) {
  TT_RET_CHECK(!c10::impl::cow::is_cow_data_ptr(storage.data_ptr()),
               error::kPythonNotImplementedError)
      << "cannot overwrite a copy-on-write tensor storage on TPU";
  DeviceBufferRef& slot = GetStorageSlot(storage);
  if (GraphCaptureStorageState* const state = GetCurrentCaptureStorageState();
      ABSL_PREDICT_FALSE(state != nullptr)) {
    TT_RETURN_IF_ERROR(ValidateWrittenOnCaptureStream(*state, buffer_ref));
    if (IsUntrackedPreexistingStorage(*state, storage)) {
      TrackPreexistingStorage(*state, storage, slot,
                              /*placeholder=*/std::nullopt);
    }
  }
  ReplaceSlotBuffer(slot, std::move(buffer_ref));
  return absl::OkStatus();
}

absl::Status ValidateTpuTensorStorage(const at::Tensor& tensor) {
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
  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> ReadStorageSlot(
    const c10::Storage& storage, const bool install_placeholder) {
  // The DeviceBufferRef in the c10::StorageImpl represents the "contiguous
  // base" buffer--this is the shape of the original tensor that was used to
  // construct the tensor, and represents all of the data available to all
  // views that share the same Storage.
  DeviceBufferRef& base_buffer_ref = GetStorageSlot(storage);
  if (!install_placeholder) {
    return base_buffer_ref;
  }
  GraphCaptureStorageState* const state = GetCurrentCaptureStorageState();
  if (ABSL_PREDICT_TRUE(state == nullptr)) {
    return base_buffer_ref;
  }
  if (c10::impl::cow::is_cow_data_ptr(storage.data_ptr())) {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "cannot read a copy-on-write tensor storage while capturing a "
              "graph";
  }
  if (!base_buffer_ref.is_placeholder() &&
      IsUntrackedPreexistingStorage(*state, storage)) {
    TT_RETURN_IF_ERROR(
        InstallCapturePlaceholder(*state, storage, base_buffer_ref));
  }
  return base_buffer_ref;
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
    TT_RETURN_IF_ERROR(
        OverwriteDataPtr(tensor.storage(), std::move(result_buf)));
  } else {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef new_base_buffer_ref,
                        CreateInverseViewDeviceBufferRef(
                            std::move(base_buffer_ref), std::move(result_buf),
                            tpu_layout, tensor.is_conj(),
                            /*donate_base=*/auto_donate));

    // Overwrite the c10::DataPtr in storage to use the new DeviceBufferRef.
    // All reads that happened before the write will be unaffected, but reads
    // to all extant views will reflect the new value on later accesses.
    TT_RETURN_IF_ERROR(
        OverwriteDataPtr(tensor.storage(), std::move(new_base_buffer_ref)));
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
  TT_RETURN_IF_ERROR(OverwriteDataPtr(tensor.storage(), std::move(new_base)));

  // Bump the version to tell PyTorch how to handle functionalization
  tensor.unsafeGetTensorImpl()->bump_version();
  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const at::Tensor& tensor) {
  TT_RETURN_IF_ERROR(ValidateTpuTensorStorage(tensor));
  return GetBaseBuffer(tensor.storage());
}

absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const c10::Storage& storage) {
  return ReadStorageSlot(storage, /*install_placeholder=*/true);
}

absl::StatusOr<DeviceBufferRef> PeekBaseBuffer(const at::Tensor& tensor) {
  TT_RETURN_IF_ERROR(ValidateTpuTensorStorage(tensor));
  return PeekBaseBuffer(tensor.storage());
}

absl::StatusOr<DeviceBufferRef> PeekBaseBuffer(const c10::Storage& storage) {
  return ReadStorageSlot(storage, /*install_placeholder=*/false);
}

size_t PhysicalByteSize(const DeviceBufferRef& buffer) {
  const size_t size_bytes = buffer.size_bytes();
  return buffer.element_type() == mlir::ElementType::F4E2M1FN ? size_bytes / 2
                                                              : size_bytes;
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
  c10::Storage storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size,
      MakeDataPtr(std::move(buffer_ref), device_idx), GetTpuAllocator(),
      /*resizable=*/true));
  if (GraphCaptureStorageState* const state = GetCurrentCaptureStorageState();
      ABSL_PREDICT_FALSE(state != nullptr)) {
    state->created_impls.insert(storage.unsafeGetStorageImpl());
    state->created.push_back(storage);
  }
  return storage;
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
  Dimensions sizes = CopyIntVector(buffer_ref.dimensions());
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

  TpuAllocator(TpuAllocator&&) = delete;
  TpuAllocator& operator=(TpuAllocator&&) = delete;
  TpuAllocator(const TpuAllocator&) = delete;
  TpuAllocator& operator=(const TpuAllocator&) = delete;

  c10::MempoolId_t AllocateGraphMempoolId() {
    return {next_graph_pool_id_.fetch_add(1, std::memory_order_relaxed), 0};
  }

  c10::DataPtr allocate(size_t nbytes) override {
    ScopedPythonContextCapturer _(OpName::kTorchTpuStorageAllocate);
    // Graph capture tracks the storages created by MakeStorage. A storage
    // allocated here outside an FX trace would be mistaken for one that existed
    // before capture.
    TT_CHECK_THROW(!(GetEagerMode() != EagerMode::kInternalCompileFxGraph &&
                     IsCurrentThreadCapturingGraph()),
                   error::kFailedPrecondition)
        << "cannot allocate an untyped storage while capturing a graph";
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

  std::atomic<uint64_t> next_graph_pool_id_{1};
};

c10::Allocator* GetTpuAllocator() {
  static absl::NoDestructor<TpuAllocator> g_tpu_allocator;
  return g_tpu_allocator.get();
}

c10::MempoolId_t AllocateGraphMempoolId() {
  return static_cast<TpuAllocator*>(GetTpuAllocator())
      ->AllocateGraphMempoolId();
}

void RegisterTpuAllocator() {
  c10::SetAllocator(GetPrivateUse1DeviceType(), GetTpuAllocator());
  at::setHostAllocator(GetPrivateUse1DeviceType(), GetTpuPinnedAllocator());
}

c10::DataPtr MakeDataPtr(DeviceBufferRef buffer_ref,
                         const c10::DeviceIndex device_idx) {
  if (const GraphCaptureStorageState* const state =
          GetCurrentCaptureStorageState();
      ABSL_PREDICT_FALSE(state != nullptr)) {
    TT_THROW_IF_ERROR(ValidateWrittenOnCaptureStream(*state, buffer_ref));
  }
  AddDataPtrAlias(buffer_ref);
  auto* absl_nonnull const raw_ref_ptr =
      new DeviceBufferRef(std::move(buffer_ref));
  return c10::DataPtr(raw_ref_ptr, raw_ref_ptr, DeleteDeviceBufferRef,
                      c10::Device(GetPrivateUse1DeviceType(), device_idx));
}

absl::StatusOr<DeviceBufferRef> MaterializeAndReturn(
    const at::Tensor& tensor, MaterializationReason reason) {
  tsl::profiler::TraceMe trace("MaterializeAndReturn");
  // There are three cases here, all handled the same way:
  //   1. `tensor` is not a view. In this case, materializing the view is the
  //      same as materializing the base, and will execute everything on the
  //      stream that `tensor` was created on.
  //   2. `tensor` is a view, and its base was created on the current stream. In
  //      this case, materializing the view will also force the base to
  //      materialize (because it has a live Tensor using it) as part of the
  //      same execution graph, along with everything else on `tensor'`s stream.
  //   3. `tensor` is a view, and its base was created on a different stream.
  //      This represents a cross-stream dependency; users are expected to
  //      coordinate cross-stream dependencies using Event.wait and syncs, so
  //      we assume that no further coordination is needed here. If this is not
  //      the case, we detect the error and fail (see execution_task.cc).
  //      Currently we also enforce this by blocking on Event creation (see
  //      tpu_hooks.cc) which handles the single-thread, multi-stream case.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef view_buffer_ref, GetBuffer(tensor));
  TT_RETURN_IF_ERROR(Materialize(view_buffer_ref, reason));
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
        << "expected float4_e2m1fn_x2 tensors to be at least 1D, "
           "got 0D";
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

namespace {

// Appends to `captured` the storages created during capture that are still
// alive, with the buffers they hold.
void CollectCreatedStorageWrites(const GraphCaptureStorageState& state,
                                 CapturedGraphStorages& captured) {
  for (const c10::Storage& storage : state.created) {
    // A storage referenced only by `state` was freed during capture, and a
    // storage whose c10::DataPtr was moved to another storage holds no buffer.
    if (storage.use_count() > 1 && storage.data_ptr().get() != nullptr) {
      captured.writes.push_back(
          CapturedStorageWrite{.storage = storage,
                               .final_buffer = GetStorageSlot(storage),
                               .created = true});
    }
  }
}

// Appends to `captured` the placeholders installed in storages that existed
// before capture, and the storages among them that capture wrote.
void CollectPreexistingStorageWrites(const GraphCaptureStorageState& state,
                                     CapturedGraphStorages& captured) {
  for (const PreexistingCapturedStorage& entry : state.preexisting) {
    const DeviceBufferRef& final_buffer = GetStorageSlot(entry.storage);
    if (entry.placeholder.has_value()) {
      captured.placeholder_to_storage.emplace(*entry.placeholder,
                                              entry.storage);
      if (final_buffer == *entry.placeholder) {
        // Capture only read the storage.
        continue;
      }
    }
    // Capture wrote the storage, either after reading it or before reading it
    // (without installing a placeholder).
    captured.writes.push_back(CapturedStorageWrite{.storage = entry.storage,
                                                   .final_buffer = final_buffer,
                                                   .created = false});
  }
}

// Restores the buffers that storages that existed before capture held when
// capture first accessed them. Restores them in reverse order of first access,
// so that storages sharing a DeviceBufferRef slot through borrowing
// c10::DataPtrs end up holding the buffer that the slot held before capture.
void RestorePreexistingStorages(const GraphCaptureStorageState& state) {
  for (auto it = state.preexisting.rbegin(); it != state.preexisting.rend();
       ++it) {
    ReplaceSlotBuffer(GetStorageSlot(it->storage), it->initial_buffer);
  }
}

}  // namespace

absl_nonnull std::shared_ptr<GraphCaptureStorageState>
BeginGraphCaptureStorageTracking(const DeviceStreamId capture_stream) {
  if (current_thread_capture_storage != nullptr &&
      current_thread_capture_storage->require_stream_capture &&
      !IsCurrentThreadCapturingGraph()) {
    current_thread_capture_storage = nullptr;
  }
  ABSL_DCHECK(current_thread_capture_storage == nullptr);
  auto state =
      std::make_shared<GraphCaptureStorageState>(GraphCaptureStorageState{
          .capture_stream = capture_stream,
          .require_stream_capture = IsCurrentThreadCapturingGraph()});
  current_thread_capture_storage = state.get();
  return state;
}

CapturedGraphStorages EndGraphCaptureStorageTracking(
    GraphCaptureStorageState& state) {
  if (current_thread_capture_storage == &state) {
    current_thread_capture_storage = nullptr;
  }
  CapturedGraphStorages captured;
  CollectCreatedStorageWrites(state, captured);
  CollectPreexistingStorageWrites(state, captured);
  RestorePreexistingStorages(state);
  state.preexisting.clear();
  state.preexisting_impls.clear();
  state.created.clear();
  state.created_impls.clear();
  return captured;
}

absl::Status ClearCapturedDeferredStorages(
    const CapturedGraphStorages& captured, const bool aborted) {
  ScopedPythonContextCapturer _(OpName::kTorchTpuStorageAllocate);
  absl::Status first_error;
  for (const CapturedStorageWrite& write : captured.writes) {
    const DeviceBufferRef& final_buffer = write.final_buffer;
    if (!write.created ||
        (!final_buffer.is_deferred() &&
         !captured.placeholder_to_storage.contains(final_buffer))) {
      continue;
    }
    absl::StatusOr<DeviceBufferRef> replacement;
    if (aborted) {
      replacement = DeviceBufferList::CreatePendingExecution(
          Shape(CopyIntVector(final_buffer.dimensions()),
                final_buffer.element_type()),
          final_buffer.device_index(), final_buffer.stream_id());
      if (replacement.ok()) {
        replacement->device_buffer_list()->SetAsError(
            TT_ERROR(error::kFailedPrecondition)
            << "cannot use a tensor created during an aborted graph capture");
      }
    } else {
      replacement =
          CreateEmptyDeviceBufferRef(CopyIntVector(final_buffer.dimensions()),
                                     final_buffer.element_type());
      if (!replacement.ok()) {
        const absl::Status empty_error = replacement.status();
        if (first_error.ok()) {
          first_error = empty_error;
        }
        replacement = DeviceBufferList::CreatePendingExecution(
            Shape(CopyIntVector(final_buffer.dimensions()),
                  final_buffer.element_type()),
            final_buffer.device_index(), final_buffer.stream_id());
        if (replacement.ok()) {
          replacement->device_buffer_list()->SetAsError(empty_error);
        }
      }
    }
    if (!replacement.ok()) {
      if (first_error.ok()) {
        first_error = replacement.status();
      }
      continue;
    }
    if (const absl::Status status =
            OverwriteDataPtr(write.storage, std::move(*replacement));
        !status.ok() && first_error.ok()) {
      first_error = status;
    }
  }
  return first_error;
}

absl::Status ValidateStorageResizableDuringGraphCapture(
    const c10::Storage& storage) {
  const GraphCaptureStorageState* const state = GetCurrentCaptureStorageState();
  if (state == nullptr ||
      state->created_impls.contains(storage.unsafeGetStorageImpl())) {
    return absl::OkStatus();
  }
  return TT_ERROR(error::kFailedPrecondition)
         << "cannot resize a tensor storage created before graph capture "
            "while capturing a graph";
}

absl::Status AssignBufferToStorage(DeviceBufferRef buffer,
                                   const c10::Storage& storage) {
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef current, PeekBaseBuffer(storage));
  const size_t current_bytes = PhysicalByteSize(current);
  const size_t buffer_bytes = PhysicalByteSize(buffer);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked by caller.
      buffer_bytes == current_bytes, error::kFailedPrecondition)
      << "cannot assign a buffer of " << buffer_bytes
      << " bytes to a tensor storage of " << current_bytes << " bytes";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked by caller.
      buffer.device_index() == storage.device().index(),
      error::kInvalidArgument)
      << "cannot assign a buffer on device "
      << static_cast<int>(buffer.device_index())
      << " to a tensor storage on device "
      << static_cast<int>(storage.device().index());
  return OverwriteDataPtr(storage, std::move(buffer));
}

}  // namespace torch_tpu
