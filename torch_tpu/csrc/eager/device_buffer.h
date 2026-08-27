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

#ifndef TORCH_TPU_EAGER_DEVICE_BUFFER_H_
#define TORCH_TPU_EAGER_DEVICE_BUFFER_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/shape.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/current_stream.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/python_context.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"

// This library holds the core internal representation of the "data" of a
// tensor in TorchTPU.
//
// In Torch/CUDA or Torch/CPU, a c10::DataPtr points to an output buffer, that
// contains the data backing the tensor. This data may be in-flight, pending an
// asynchronous computation, but every new c10::DataPtr represents a physical
// block of device memory.
//
// In TorchTPU, however, a c10::DataPtr may point to either a materialized
// output buffer, **or** a deferred operation that can produce it. This deferred
// operation allows us to build a graph incrementally, and merge them into a
// larger compilation unit. It also means that we do not need to persist every
// intermediate tensor; only the tensors which have their data moved off-device
// are strictly necessary to materialize to a physical buffer.
//
// However, there is some complexity introduced by the fact that one aten
// operation can produce multiple outputs; this is common for normalization
// functions (e.g. layer norm, batch norm, group norm). This means that deferred
// operations are not one-to-one with individual tensors.
//
// This requires a three-part recursive graph structure:
///  - A DeviceBufferRef represents the data of a single tensor. It is nothing
//     more than a shared pointer to a DeviceBufferList and an index into that
//     list.
//   - A DeviceBufferList represents the data that was, or could be, produced by
//     a single aten operation. Since an aten operation may produce multiple
//     tensors, a DeviceBufferList can contain multiple DeviceBufferRefs. If the
//     operation is deferred, it has a DeferredOp; otherwise it usually
//     contains one or more PjRtBuffers (see note below).
//   - A DeferredOp represents a single aten operation that has not yet been
//     executed. It contains a list of DeviceBufferRefs that represent the
//     inputs to the operation.
//
// While "materialized" and "deferred" are the most common states, one special
// "dataless" case also exists. To support full-graph compiled mode, we know the
// shapes and dtypes of inputs in advance, but we need to trace the FX graph
// without real data. For this, we define a "placeholder" state, which
// represents a tensor that will be provided in the future when the compiled
// function is executed.
//
// This is generally a low-level implementation detail that should be invisible
// to most users of PyTorch or libtorch; the at::Tensors at the surface level
// should "just work", matching the observable semantics of CUDA eager ops.
//
// Internally, most aten kernel registrations should not need to directly use
// DeviceBufferList or DeviceBufferRef, instead using the higher-level
// abstractions provided in op_dispatcher.h and other utility libraries.

namespace torch_tpu {

enum class OpSplitMode {
  kNone,

  // Split the execution graph before this op.
  kSplitBefore,

  // Split the execution graph after this op.
  kSplitAfter,

  // Split the execution graph before and after this op.
  kSplitBoth,
};

inline bool IsSplitBefore(OpSplitMode split_mode) {
  return split_mode == OpSplitMode::kSplitBefore ||
         split_mode == OpSplitMode::kSplitBoth;
}

inline bool IsSplitAfter(OpSplitMode split_mode) {
  return split_mode == OpSplitMode::kSplitAfter ||
         split_mode == OpSplitMode::kSplitBoth;
}

class DeviceBufferList;
class DeferredOp;

using SharedDeviceBufferList = absl_nonnull std::shared_ptr<DeviceBufferList>;

// A DeviceBufferRef is a reference to an element in a DeviceBufferList.
// It is nothing more than a container for a std::shared_ptr<DeviceBufferList>
// and an index into that list, with accessor methods for convenience.
//
// The c10::DataPtr at the root of the torch/aten/c10 data hierarchy holds a
// unique pointer (as a c10::UniqueVoidPointer) to a DeviceBufferRef.
// In most cases, this is an owning pointer that will drop the DeviceBufferRef
// when the c10::DataPtr is dropped. However, in rare cases (such as
// `at::from_blob`) it is possible that two c10::DataPtrs will share the same
// DeviceBufferRef, with all but one being "borrowers" without responsibility
// for deallocation; the borrowers will have the `c10::detail::deleteNothing`
// deleter for their DataPtr's context.
//
// Whenever an at::Tensor is dropped, it may trigger a deallocation of the
// entire shared pointer chain (c10::TensorImpl, c10::StorageImpl, c10::DataPtr,
// DeviceBufferRef, and DeviceBufferList) unless there are additional shared
// pointers (at::Tensors, c10:Storages, or DeviceBufferRefs) preventing
// deallocation, or the c10::DataPtr is a non-owning borrow.
//
// This means that while a DeviceBufferRef can always be safely dereferenced to
// a DeviceBufferList, it is **unsafe** to hold a DeviceBufferRef& or
// DeviceBufferRef* longer than a single aten kernel execution.
//
// The DeviceBufferRef itself is immutable; it is invalid (and impossible) to
// modify a DeviceBufferRef to point to a different buffer. However, the
// underlying DeviceBufferList may be modified in some ways, such as by
// materialization.
//
// Additionally, the aten/c10 hierarchy may overwrite the void* data_ in a
// c10::DataPtr (in the c10::StorageImpl), changing which DeviceBufferRef backs
// the at::Tensor (and all other shared uses of the buffer). This can be used to
// "mutate" an at::Tensor from the perspective of aten, while respecting
// immutability at the XLA level, and preserving the `void* data_` pointer's
// heap address.
class DeviceBufferRef {
 public:
  // DeviceBufferRefs require a non-null SharedDeviceBufferList and a valid
  // index, which cannot be default-constructed.
  DeviceBufferRef() = delete;

  // Creates a DeviceBufferRef directly from a SharedDeviceBufferList and index.
  // The index must be in bounds for the DeviceBufferList.
  static absl::StatusOr<DeviceBufferRef> Create(
      SharedDeviceBufferList device_buffer_list, int64_t index);

  // DeviceBufferRef is copyable and movable. Per "rule of five"
  // (https://en.cppreference.com/w/cpp/language/rule_of_three.html), we should
  // declare the copy and move constructors and assignment operators explicitly,
  // or the compiler will silently delete some of them, leading to suboptimal
  // performance.
  //
  // WARNING: moving out of a DeviceBufferRef will leave it in an invalid state
  // (containing a nullptr, which is otherwise not allowed). Use with caution.
  DeviceBufferRef(const DeviceBufferRef& other) = default;
  DeviceBufferRef(DeviceBufferRef&& other) = default;
  DeviceBufferRef& operator=(const DeviceBufferRef& other) = default;
  DeviceBufferRef& operator=(DeviceBufferRef&& other) = default;

  // The logical size of the referenced buffer or placeholder, as a function of
  // its shape and XLA element type.
  [[nodiscard]] size_t size_bytes() const;

  // The number of physical bytes backing this buffer, if it can be determined.
  absl::StatusOr<size_t> pjrt_buffer_size() const;

  // A detailed debug string describing the buffer.
  // This is verbose and should only be used for debugging, such as with
  // logging, crashing, or error::kInternal status messages.
  // It should never be used in user-facing error messages.
  [[nodiscard]] std::string DebugString() const;

  // Returns true if this DeviceBufferRef is a placeholder.
  // This will never change state.
  [[nodiscard]] bool is_placeholder() const;

  // Returns true if this DeviceBufferRef is a deferred op and has not
  // yet started materialization.
  [[nodiscard]] bool is_deferred() const;

  // Returns true if this DeviceBufferRef has started materialization.
  // It may or may not have completed materialization.
  [[nodiscard]] bool is_materializing() const;

  // Returns true if this DeviceBufferRef is executing.
  // The execution may or may not have completed.
  [[nodiscard]] bool is_executing() const;

  // Returns true if this DeviceBufferRef has finished materialization.
  // It may have succeeded or failed.
  [[nodiscard]] bool is_materialized() const;

  // Returns true if this DeviceBufferRef was created by a deferred op with a
  // statically-known constant value.
  // This includes "empty" operations (e.g. torch.empty) where the constant
  // value is a deterministic fill with NaN or max-value ints, matching the
  // semantics of `torch.utils.deterministic.fill_uninitialized_memory`.
  // We avoid materializing these buffers whenever possible, to preserve
  // constant-folding and avoiding materializing uninitialized memory.
  [[nodiscard]] bool is_constant() const;

  [[nodiscard]] const Shape& shape() const;

  // The logical dimensions of the referenced buffer.
  [[nodiscard]] absl::Span<const int64_t> dimensions() const;

  // The number of elements in the referenced buffer.
  [[nodiscard]] int64_t num_elements() const;

  // The element type of the referenced buffer.
  [[nodiscard]] mlir::ElementType element_type() const;

  // If the DeviceBufferRef has a DeferredOp, returns a non-null shared_pointer
  // to it. Otherwise, return a nullptr.
  [[nodiscard]] absl_nullable std::shared_ptr<DeferredOp> deferred_op() const;

  // Awaits for the device buffer to be ready to read. If the buffer is the
  // output of computation, then also waits for the computation to be done.
  absl::Status Synchronize() const;

  // Awaits the PjRtBuffer to be materialized, and then returns either the
  // error encountered during materialization, or a pointer to the
  // PjRtBuffer.
  absl::StatusOr<xla::PjRtBuffer* absl_nonnull> AwaitBuffer() const;

  // Returns a future that is ready when the device buffer has an available
  // PjRtBuffer (or an error if materialization fails).
  // For a deferred DeviceBufferRef, this is a future that is ready when
  // execution has started (but not necessarily completed).
  // Once this future is ready, subsequent calls to AwaitBuffer() will not
  // block.
  [[nodiscard]] xla::Future<void> GetMaterializationFuture() const;

  // Returns a future that is ready when the device has a PjRtBuffer with
  // ready data; this means that either the buffer has successfully finished
  // execution, or has finished a host-to-device transfer.
  [[nodiscard]] xla::Future<void> GetReadyFuture() const;

  // The DeviceBufferList that holds the referenced buffer.
  [[nodiscard]] const SharedDeviceBufferList& device_buffer_list() const {
    return device_buffer_list_;
  }

  // The index of the referenced buffer within the DeviceBufferList.
  [[nodiscard]] int64_t index() const { return index_; }

  // Marks a dimension of the given DeviceBufferRef as dynamic.
  // The dimension must be in bounds for the DeviceBufferRef.
  // The lower and upper bounds must be in bounds for the dimension as well.
  absl::Status MarkDynamic(int64_t dimension, int64_t lower_bound,
                           int64_t upper_bound) const;

  // Returns whether the given DeviceBufferRef has dynamic on device shape.
  [[nodiscard]] bool on_device_shape_is_dynamic() const;

  // The dynamic dimensions of the given DeviceBufferRef.
  // May be empty if the DeviceBufferRef is not dynamic.
  [[nodiscard]] absl::Span<const BoundedDynamicDimension> dynamic_dimensions()
      const;

  [[nodiscard]] bool operator==(const DeviceBufferRef& other) const {
    // std::shared_ptr equality is a pointer equality check, not a value
    // equality check, which is what we want here.
    // In other words:
    //   Two DeviceBufferRefs are equal if they reference the same index in the
    //   same DeviceBufferList.
    //   Two DeviceBufferRefs are unequal if they reference different
    //   DeviceBufferLists or different indices in the same DeviceBufferList,
    //   even if the two referenced buffers happen to be equivalent.
    return device_buffer_list_ == other.device_buffer_list_ &&
           index_ == other.index_;
  }

  [[nodiscard]] bool operator<(const DeviceBufferRef& other) const {
    return device_buffer_list_.get() < other.device_buffer_list_.get() ||
           (device_buffer_list_.get() == other.device_buffer_list_.get() &&
            index_ < other.index_);
  }

  // DeviceBufferRef is hashable, making it suitable for use as a key in
  // absl::flat_hash_map<DeviceBufferRef, ValueType>
  // or absl::flat_hash_set<DeviceBufferRef>.
  // Implementation is based on example from
  // https://abseil.io/docs/cpp/guides/hash.
  template <typename H>
  friend H AbslHashValue(H h, const DeviceBufferRef& ref) {
    return H::combine(std::move(h), ref.index_, ref.device_buffer_list_);
  }

  // Records that a child op has been created which depends on this
  // DeviceBufferList. This updates num_child_ops_ and last_child_index_.
  void RecordChildOp(uint64_t child_index) const;

  // Returns the device index of the referenced buffer.
  [[nodiscard]] c10::DeviceIndex device_index() const;

  // Returns the stream ID of the referenced buffer.
  [[nodiscard]] torch_tpu::StreamId stream_id() const;

 private:
  // DeviceBufferRefs can only be constructed by DeviceBufferList::Create*
  // functions, to ensure the reference and referent are created simultaneously.
  // The index must be in bounds for the DeviceBufferList.
  friend class DeviceBufferList;
  DeviceBufferRef(absl_nonnull std::shared_ptr<DeviceBufferList> device_buffer,
                  int64_t index)
      : device_buffer_list_(std::move(device_buffer)), index_(index) {}

  // The actual shared_ptr to the referenced DeviceBufferList
  absl_nonnull std::shared_ptr<DeviceBufferList> device_buffer_list_;
  // The index into the referenced DeviceBufferList.
  // Initialized to -1 to avoid uninitialized memory.
  int64_t index_ = -1;
};

// A deferred operation, used to back a DeviceBufferList.
//
// When aten ops are dispatched through op_dispatcher.h, they may return one or
// more DeviceBufferRefs to a single DeviceBufferList with a DeferredOp, rather
// than immediately compiling and materializing the op's results. At some point
// in the future, the DeferredOp will be compiled and executed, returning new
// DeviceBufferRefs to a materialized DeviceBufferList with PjRtBuffers. The
// existing DeviceBufferRefs are unmodified; instead, an at::Tensor can be
// updated to reference the new DeviceBufferRef using
// MakeStorage.
//
// In eager mode, this allows us to delay compilation until we can compile
// multiple ops at once, reducing total compilation time and achieving better
// performance through XLA optimizations (e.g. kernel fusion) than we could with
// individual op-by-op executions.
//
// In compiled mode, the FX graph is compiled by first creating a DeferredOp
// graph from the FX graph's aten ops, and then compiling this graph to a
// reusable executable using PjRt.
class DeferredOp {
 public:
  // Constructs a new DeferredOp.
  // op_builder must be a function that requires the same number of inputs as
  // `inputs`, and returns at least one output. This is **unchecked** here (as
  // there is no way to introspect an absl::AnyInvocable); an invalid op_builder
  // may fail when building the MLIR for XLA compilation.
  DeferredOp(OpName op_name, MlirOpBuilder op_builder,
             std::vector<DeviceBufferRef> inputs,
             OpParamCacheKeys op_param_cache_keys,
             std::vector<Shape> output_shapes,
             OpSplitMode split_mode = OpSplitMode::kNone,
             Indices donated_indices = {})
      : op_name_(op_name),
        op_builder_(std::move(op_builder)),
        inputs_(std::move(inputs)),
        donated_indices_(std::move(donated_indices)),
        output_shapes_(std::move(output_shapes)),
        op_param_cache_keys_(std::move(op_param_cache_keys)),
        op_context_(ScopedPythonContextCapturer::GetContext()),
        split_mode_(split_mode) {}

  // DeferredOps are copyable and movable. Per "rule of five"
  // (https://en.cppreference.com/w/cpp/language/rule_of_three.html), we should
  // declare the copy and move constructors and assignment operators explicitly,
  // or the compiler will silently delete some of them, leading to suboptimal
  // performance.
  DeferredOp(const DeferredOp& other) = default;
  DeferredOp(DeferredOp&& other) = default;
  DeferredOp& operator=(const DeferredOp& other) = default;
  DeferredOp& operator=(DeferredOp&& other) = default;

  // A quick hash, but prone to collisions.
  size_t Hash() const;

  // The name of the deferred op, like "add" or "matmul".
  OpName op_name() const { return op_name_; }

  // The MlirOpBuilder that will be used to compile the deferred op.
  const MlirOpBuilder& op_builder() const { return op_builder_; }

  // The inputs to the deferred op.
  absl::Span<const DeviceBufferRef> inputs() const { return inputs_; }

  // Context where the op is used in the user's PyTorch code.
  [[nodiscard]] const PythonContext& op_context() const { return op_context_; }

  // Returns the cache keys for the op parameters. These are used to ensure that
  // the compilation cache does not reuse a cached compiled op if there are
  // meaningful differences in the op_builder (such as "floor" vs "trunc" in
  // the div() op).
  [[nodiscard]] const OpParamCacheKeys& op_param_cache_keys() const {
    return op_param_cache_keys_;
  }

  // Returns the split mode of the DeferredOp, which determines how the op is
  // split into subgraphs for compilation.
  [[nodiscard]] OpSplitMode split_mode() const { return split_mode_; }

  [[nodiscard]] absl::Span<const int64_t> donated_indices() const {
    return donated_indices_;
  }

 private:
  // The name of the deferred op.
  OpName op_name_;
  // The MlirOpBuilder that will be used to compile the deferred op.
  MlirOpBuilder op_builder_;
  // The inputs to the deferred op. These are held as DeviceBufferRefs, which
  // ensures that the DeviceBuffers backing the at::Tensors are not freed as
  // long as this DeferredOp exists.
  std::vector<DeviceBufferRef> inputs_;

  // The indices of the inputs that should be donated. This is only
  // used for custom kernels, and only when the DeferredOp directly depends on
  // an input to the overall MLIR module.
  Indices donated_indices_;

  // The shapes of the outputs of the deferred op.
  std::vector<Shape> output_shapes_;

  // The cache keys for the op parameters. These are used to ensure that the
  // compilation cache does not reuse a cached compiled op if there are
  // meaningful differences in the op_builder (such as "floor" vs "trunc" in
  // the div() op).
  OpParamCacheKeys op_param_cache_keys_;

  // Context where the op is used in the user's PyTorch code.
  //
  // When computing the cache key, use op_name_ to ensure that the key is
  // unique.
  PythonContext op_context_;

  // The split mode of the DeferredOp, which determines how the op is split into
  // subgraphs for compilation.
  const OpSplitMode split_mode_ = OpSplitMode::kNone;

  friend std::ostream& operator<<(std::ostream& os,
                                  const DeferredOp& deferred_op);
};

// A DeviceBufferList contains the data backing one or more tensors.
//
// Unlike CUDA, where one c10::DataPtr is always a physical data buffer
// (possibly containing uninitialized memory), a torch_tpu DeviceBufferList
// can represent a set of bufferless tensors (compiled mode placeholders), a
// deferred operation (no data, but an operation to compute it), or a set of
// materialized tensors (with PjRtBuffers).
//
// While CUDA can allocate a buffer of uninitialized memory, XLA requires
// specifying buffer values on initialization; accordingly, a DeviceBufferList
// will never contain uninitialized memory, but it may have a DeferredOp to
// fill an "empty" buffer with dummy data.
//
// A CUDA c10::DataPtr represents a 1D untyped data buffer that may be shared by
// multiple view tensors; the stride pattern (on the c10::TensorImpl) describes
// a conversion from an index tuple to a 1D offset into the buffer. In
// contrast, an XLA PjRtBuffer is a concrete, typed buffer with some on-device
// layout, but that does not support random-access indexing. Equivalent "view"
// semantics are achieved by lazily constructing DeferredOps defining the
// base-to-view conversions on-demand (see GetBuffer and
// view_decomposition library).
//
// The conceptual buffers contained in a DeviceBufferList are immutable. It is
// invalid to change the shape or type of any buffer, as this could break the
// alignment with at::Tensors using that buffer. It is also invalid to change
// the DeferredOp, as this would change the behavior of already-dispatched ops.
// And PjRtBuffers are (mostly) immutable as a constraint of the XLA system.
//
// But, the DeviceBufferList itself can be modified through materialization
// (SetMaterialized), replacing deferred buffers with materialized buffers, or
// replacing previously-materialized PjRtBuffers with new ones. The
// c10::StorageImpl can also be modified by swapping out the c10::DataPtr,
// changing which DeviceBufferRef backs the tensors using that storage
// (see AssignBufferToAtTensor).
class DeviceBufferList {
 public:
  // DeviceBufferLists require a specified shapes and types, which does not
  // allow for default construction.
  DeviceBufferList() = delete;

  // Creates a single-output DeviceBufferList with a materialized PjRtBuffer,
  // and returns a DeviceBufferRef to it.
  // The shape and element type will be inferred from
  // PjRtBuffer::on_device_shape().
  static absl::StatusOr<DeviceBufferRef> CreateMaterialized(
      absl_nonnull std::unique_ptr<xla::PjRtBuffer> buffer);

  // Creates a deferred DeviceBufferList containing a DeferredOp, and returns as
  // many DeviceBufferRefs as there are outputs in the list.
  // The op_builder must take the same number of inputs as `inputs`, and must
  // return at least one output, which must match the specified Shapes.
  // This is **unchecked** here (as there is no way to introspect an
  // absl::AnyInvocable); an invalid op_builder may fail when building the MLIR
  // for XLA compilation.
  static absl::StatusOr<std::vector<DeviceBufferRef>> CreateDeferred(
      OpName op_name, MlirOpBuilder op_builder,
      std::vector<DeviceBufferRef> inputs, OpParamCacheKeys op_param_cache_keys,
      std::vector<Shape> output_shapes,
      OpSplitMode split_mode = OpSplitMode::kNone,
      Indices donated_indices = {});

  // Creates a DeviceBufferList that represents a compiled-mode placeholder.
  // This is a buffer that is not backed by any data, but is used to represent
  // an argument to a compiled executable.
  static absl::StatusOr<DeviceBufferRef> CreatePlaceholder(
      Dimensions dimensions, mlir::ElementType element_type);

  // Creates a DeviceBufferList that represents a pending materialized buffer.
  // This is used to hold the output of calling a precompiled executable, such
  // as for torch.compile mode.
  static absl::StatusOr<DeviceBufferRef> CreatePending(
      const Shape& shape, c10::DeviceIndex device_index,
      c10::StreamId stream_id);

  // Sets the DeviceBufferList to a pending-materialized state.
  // If the DeviceBufferList is already materialized, this is a no-op.
  // Returns an error if the DeviceBufferList is a placeholder, as placeholders
  // cannot be materialized.
  absl::Status SetAsPendingMaterialization();

  // Sets the DeviceBufferList to a materialized state with the given buffers.
  // Returns an error if the DeviceBufferList is already materialized.
  // The number of buffers, and their Shapes must match the existing
  // DeviceBufferList, otherwise the DeviceBufferList will set to an error state
  // and a copy of the error will be returned.
  absl::Status SetAsMaterialized(
      std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers);

  // Sets the DeviceBufferList to an error state with the given error.
  // Returns an error if the DeviceBufferList is already materialized.
  void SetAsError(absl::Status error);

  // Sets a dimension to be dynamic.
  absl::Status MarkDynamic(int64_t index, int64_t dimension,
                           int64_t lower_bound, int64_t upper_bound);
  // Returns whether the indexed buffer has dynamic on device shape.
  [[nodiscard]] bool on_device_shape_is_dynamic(int64_t index) const;

  // Returns the dynamic dimensions of the indexed buffer. This may be empty if
  // the indexed buffer has no dynamic dimensions.
  [[nodiscard]] absl::Span<const BoundedDynamicDimension> dynamic_dimensions(
      int64_t index) const;

  // DeviceBufferList is copyable and movable. Per "rule of five"
  // (https://en.cppreference.com/w/cpp/language/rule_of_three.html), we should
  // declare the copy and move constructors and assignment operators explicitly,
  // or the compiler will silently delete some of them, leading to suboptimal
  // performance.
  //
  // However, copies and moves of DeviceBufferList should generally be avoided;
  // prefer to copy and move DeviceBufferRefs instead.
  DeviceBufferList(const DeviceBufferList& other) = default;
  DeviceBufferList(DeviceBufferList&& other) = default;
  DeviceBufferList& operator=(const DeviceBufferList& other) = default;
  DeviceBufferList& operator=(DeviceBufferList&& other) = default;

  // Returns the number of buffers in the DeviceBufferList.
  [[nodiscard]] int64_t size() const { return shapes_.size(); }

  // Returns a span of all of the buffer shapes in the DeviceBufferList.
  [[nodiscard]] absl::Span<const Shape> shapes() const { return shapes_; }

  // Note: all methods which take an index will crash if the index is invalid.

  // The logical size of the indexed buffer or placeholder, as a function of
  // its shape and XLA element type.
  [[nodiscard]] size_t size_bytes(int64_t index) const;

  // The number of physical bytes backing the indexed buffer, if it can be
  // determined.
  absl::StatusOr<size_t> pjrt_buffer_size(int64_t index) const;

  // Returns true if this DeviceBufferList is a placeholder.
  // This will never change state.
  [[nodiscard]] bool is_placeholder() const { return data_.is_placeholder(); }

  // Returns true if this DeviceBufferList is a deferred op and has not
  // yet started materialization.
  [[nodiscard]] bool is_deferred() const { return data_.is_deferred(); }

  // Returns true if this DeviceBufferList has started materialization.
  // It may or may not have completed materialization.
  [[nodiscard]] bool is_materializing() const {
    return data_.is_materializing();
  }

  // Returns true if this DeviceBufferList was created by a deferred op with a
  // statically-known constant value.
  // This includes "empty" operations (e.g. torch.empty) where the constant
  // value is a deterministic fill with NaN or max-value ints, matching the
  // semantics of `torch.utils.deterministic.fill_uninitialized_memory`.
  // We avoid materializing these buffers whenever possible, to preserve
  // constant-folding and avoiding materializing uninitialized memory.
  [[nodiscard]] bool is_constant() const { return data_.is_constant(); }

  // Returns true if this DeviceBufferList is executing.
  // The execution may or may not have completed.
  [[nodiscard]] bool is_executing() const { return data_.is_executing(); }

  // Returns true if this DeviceBufferList has finished materialization.
  // It may have succeeded or failed.
  [[nodiscard]] bool is_materialized() const { return data_.is_materialized(); }

  // The logical dimensions of the indexed buffer.
  [[nodiscard]] absl::Span<const int64_t> dimensions(int64_t index) const;

  // The number of elements in the indexed buffer.
  [[nodiscard]] int64_t num_elements(int64_t index) const;

  // The element type of the referenced buffer.
  [[nodiscard]] mlir::ElementType element_type(int64_t index) const;

  // If the DeviceBufferList has a DeferredOp, returns a shared pointer to it.
  [[nodiscard]] std::shared_ptr<DeferredOp> absl_nullable deferred_op() const {
    return data_.deferred_op();
  }

  [[nodiscard]] xla::Future<void> GetMaterializationFuture() const {
    return data_.materialization_future();
  }

  // Awaits for the device buffer to be ready to read. If the buffer is the
  // output of computation, then also waits for the computation to be done.
  absl::Status Synchronize() const;

  // Awaits the PjRtBuffer to be materialized, and then returns either the
  // error encountered during materialization, or a pointer to the
  // PjRtBuffer.
  absl::StatusOr<xla::PjRtBuffer* absl_nonnull> AwaitBuffer(
      int64_t index) const;

  // If the DeviceBufferList has no live data pointers, it is "stale", meaning
  // that it will never be directly materialized and will never have any new
  // DeferredOps appended to it. This allows for more optimal materialization
  // patterns in some cases.
  [[nodiscard]] bool is_stale() const { return live_data_ptrs_ == 0; }

  // Returns the global index of the creation of this DeviceBufferList.  Lower
  // means earlier.
  [[nodiscard]] uint64_t creation_index() const { return creation_index_; }

  std::string DebugString() const;

  // Returns the number of other DeferredOps that have been created which depend
  // on this one. This is also called the "fanout" of the node.
  // This does *not* track the liveness of these ops; if a DeferredOp is
  // destroyed, the refcount will not be decremented.
  [[nodiscard]] int64_t num_child_ops() const { return num_child_ops_; }

  // Returns the creation_index of the last op which depends on this
  // DeviceBufferList.
  [[nodiscard]] uint64_t last_child_index() const { return last_child_index_; }

  // Records that a child op has been created which depends on this
  // DeviceBufferList. This updates num_child_ops_ and last_child_index_.
  void RecordChildOp(uint64_t child_index) const;

  std::ostream& DebugData(std::ostream& os) const;

  // Increments or decrements the live data pointer count.
  void IncrementLiveDataPtrs() { live_data_ptrs_++; }
  void DecrementLiveDataPtrs() { live_data_ptrs_--; }

  // The index of the device that this DeviceBufferList was created on.
  [[nodiscard]] c10::DeviceIndex device_index() const { return device_index_; }

  // The stream ID that this DeviceBufferList was created on.
  // Note: two streams on different devices can have the same ID, so this must
  // be combined with device_index() to get a unique identifier.
  // 0 indicates the default stream for the device.
  [[nodiscard]] c10::StreamId stream_id() const { return stream_id_; }

 private:
  // Private constructor for a DeviceBufferList wrapping a single materialized
  // PjRtBuffer.
  // DeviceBufferListss should only ever be accessed via DeviceBufferRef, which
  // is returned by the Create* functions.
  // The shape will be inferred from the PjRtBuffer; the element type will be
  // the type specified by the argument.
  DeviceBufferList(absl_nonnull std::unique_ptr<xla::PjRtBuffer> buffer,
                   const mlir::ElementType element_type)
      : data_(std::move(buffer)) {
    const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
    device_index_ = device_index;
    stream_id_ = stream_id;
    creation_index_ = g_creation_index_.fetch_add(1);

    auto buffer_or = data_[0];
    ABSL_CHECK_OK(buffer_or);  // CRASH_OK: we just created it
    const xla::PjRtBuffer* absl_nonnull buffer_ptr = buffer_or.value();

    const xla::Shape& on_device_shape = buffer_ptr->on_device_shape();
    Shape shape(CopyIntVector(on_device_shape.dimensions()), element_type);
    if (on_device_shape.has_layout()) {
      CustomLayout layout;
      layout.minor_to_major =
          CopyIntVector(on_device_shape.layout().minor_to_major());
      layout.element_size_in_bits = XlaEquivalentBitwidth(element_type);
      shape.set_layout(std::move(layout));
    }
    shapes_.push_back(std::move(shape));

    ABSL_VLOG(3) << "[DeviceBuffer CONSTRUCTOR (materialized)] Created. Dims: "
                 << ToString(shapes_[0].dimensions())
                 << ", Type: " << ToString(shapes_[0].dtype())
                 << ", PjRtBuffer: " << buffer_ptr
                 << ", creation_index: " << creation_index_;
  }

  // Private constructor for a deferred DeviceBufferList.
  // DeviceBufferListss should only ever be accessed via DeviceBufferRef, which
  // is returned by the Create* functions.
  // The dimensions and element types must match the return values of the
  // DeferredOp's op_builder; if they do not, then there may be compilation
  // failures.
  DeviceBufferList(std::unique_ptr<DeferredOp> absl_nonnull deferred_op,
                   std::vector<Shape> shapes)
      : shapes_(std::move(shapes)), data_(std::move(deferred_op)) {
    const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
    device_index_ = device_index;
    stream_id_ = stream_id;
    creation_index_ = g_creation_index_.fetch_add(1);

    {
      const auto shared_deferred_op = data_.deferred_op();
      ABSL_CHECK(shared_deferred_op);  // CRASH_OK=we just created it
      for (const auto& input : shared_deferred_op->inputs()) {
        input.RecordChildOp(creation_index_);
      }
    }

    if (ABSL_VLOG_IS_ON(3)) {
      const auto shared_deferred_op = data_.deferred_op();
      ABSL_CHECK(shared_deferred_op);  // CRASH_OK
      ABSL_VLOG(3) << "[DeviceBuffer CONSTRUCTOR (deferred)] Created. "
                      "DeferredOp: "
                   << shared_deferred_op->op_name()
                   << ", Number of outputs: " << shapes_.size();
    }
  }

  // Private constructor for a DeviceBufferList with a single unbacked buffer.
  // DeviceBufferListss should only ever be accessed via DeviceBufferRef, which
  // is returned by the Create* functions.
  //
  // If placeholder is true, the buffer is a compiled mode placeholder,
  // representing data that the compiled executable will expect to be provided
  // as an argument.
  //
  // If placeholder is false, the buffer is in the pending materialization
  // state and can later be materialized.
  DeviceBufferList(
      Dimensions dimensions, const mlir::ElementType element_type,
      bool placeholder,
      std::optional<c10::DeviceIndex> device_index_override = std::nullopt,
      std::optional<c10::StreamId> stream_id_override = std::nullopt)
      : data_(placeholder) {
    const auto [current_device_index, current_stream_id] =
        GetCurrentDeviceStreamId();
    device_index_ = device_index_override.value_or(current_device_index);
    stream_id_ = stream_id_override.value_or(current_stream_id);
    creation_index_ = g_creation_index_.fetch_add(1);

    shapes_.emplace_back(std::move(dimensions), element_type);
    ABSL_VLOG(3) << "[DeviceBuffer CONSTRUCTOR (bufferless)] Created. Dims: "
                 << ToString(shapes_[0].dimensions())
                 << ", Type: " << ToString(shapes_[0].dtype());
  }

  DeviceBufferList(
      const Shape& shape, bool placeholder,
      std::optional<c10::DeviceIndex> device_index_override = std::nullopt,
      std::optional<c10::StreamId> stream_id_override = std::nullopt)
      : data_(placeholder) {
    const auto [current_device_index, current_stream_id] =
        GetCurrentDeviceStreamId();
    device_index_ = device_index_override.value_or(current_device_index);
    stream_id_ = stream_id_override.value_or(current_stream_id);
    creation_index_ = g_creation_index_.fetch_add(1);

    shapes_.push_back(shape);
    ABSL_VLOG(3)
        << "[DeviceBuffer CONSTRUCTOR (bufferless)] Created with shape. Dims: "
        << ToString(shapes_[0].dimensions())
        << ", Type: " << ToString(shapes_[0].dtype());
  }

  // Helper function to check that the given buffers are valid for this
  // DeviceBufferList.
  //
  // This function will crash if the given buffers are invalid for this
  // DeviceBufferList, e.g. the buffers' shapes and dtypes match this
  // DeviceBufferList's shapes and dtypes.
  void CheckMaterialization(
      absl::Span<const absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers)
      const;

  static std::atomic_uint64_t g_creation_index_;

  // The shapes of all the buffers in the DeviceBufferList.
  std::vector<Shape> shapes_;
  // A global index of the creation of this DeviceBufferList (created atomically
  // for thread safety). Lower means earlier.
  uint64_t creation_index_ = 0;

  // The number of other DeferredOps that have been created which depend
  // on this DeviceBufferList. This does *not* track the liveness of these ops;
  // if a DeferredOp is destroyed, the refcount will not be updated.
  mutable std::atomic_int64_t num_child_ops_ = 0;

  // The creation_index of the last op which depends on this DeviceBufferList.
  // Will be 0 if no child ops have been created.
  mutable std::atomic_uint64_t last_child_index_ = 0;

  // The number of live c10::DataPtrs to this DeviceBufferList.
  // This is incremented by MakeDataPtr and decremented by
  // DeleteDeviceBufferRef, and nowhere else.
  std::atomic_int64_t live_data_ptrs_ = 0;

  // A DeviceBufferList::Data is the data backing a DeviceBufferList. It is a
  // thread-safe class that allows for deferred ops (or placeholders) to later
  // be replaced with materialized buffers, or with an error if materialization
  // fails.
  class Data {
   public:
    // Encapsulates the results of a successful materialization.
    // Holds a shared pointer to the `xla::PjRtClient` to guarantee that the
    // client remains alive as long as the materialized buffers are active.
    class MaterializedData {
     public:
      MaterializedData(
          std::shared_ptr<xla::PjRtClient> absl_nonnull client,
          std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers)
          : client_(std::move(client)), buffers_(std::move(buffers)) {
        ABSL_CHECK(client_ != nullptr);  // CRASH_OK
      }

      [[nodiscard]] const std::shared_ptr<xla::PjRtClient>& client() const {
        return client_;
      }

      [[nodiscard]] absl::Span<const
                               absl_nonnull std::unique_ptr<xla::PjRtBuffer>>
      buffers() const {
        return buffers_;
      }

     private:
      std::shared_ptr<xla::PjRtClient> absl_nonnull client_;
      std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers_;
    };

    // Creates a either a placeholder DeviceBufferList::Data, or a pending
    // materialization DeviceBufferList::Data.
    explicit Data(bool placeholder = true);

    // Creates a DeviceBufferList::Data with a DeferredOp, which can later be
    // materialized.
    explicit Data(absl_nonnull std::shared_ptr<DeferredOp> deferred_op);

    // Creates a DeviceBufferList::Data with a materialized PjRtBuffer.
    explicit Data(absl_nonnull std::unique_ptr<xla::PjRtBuffer> buffer);

    // Returns the DeferredOp backing this DeviceBufferList::Data, if it exists.
    // Otherwise, returns nullptr.
    // Returns nullptr for placeholders, or if the DeviceBufferList::Data is
    // pending materialization or fully materialized.
    absl_nullable std::shared_ptr<DeferredOp> deferred_op() const;

    // Returns the future that will be set when materialization is started or
    // failed.
    xla::Future<> materialization_future() const {
      return materialization_future_;
    }

    // Marks this DeviceBufferList::Data as pending materialization.
    // This will delete the DeferredOp if it exists.
    // This is idempotent and safe to call multiple times; the second and later
    // calls will be no-ops.
    // This has no effect if the DeviceBufferList::Data is already fully
    // materialized.
    // Returns an error if called on a placeholder; placeholders cannot be
    // materialized.
    absl::Status SetAsPendingMaterialization();

    // Marks this DeviceBufferList::Data as having failed to materialize.
    // If the DeviceBufferList::Data has already started materialization (or was
    // previously marked as having failed to materialize), this will not modify
    // the DeviceBufferList::Data and will return an error to the caller.
    absl::Status SetMaterializationError(absl::Status status);

    // Marks this DeviceBufferList::Data as having successfully started
    // materialization. If the DeviceBufferList::Data has already started
    // materialization (or was previously marked as having failed to
    // materialize), this will not modify the DeviceBufferList::Data, will drop
    // the PjRtBuffers, and will return an error to the caller.
    absl::Status SetMaterializationStarted(
        std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers);

    // Returns the PjRtBuffer at the given index.
    // If the DeviceBufferList::Data is not yet materialized, this will block
    // the caller until it has a PjRtBuffer to return. Returns an error if:
    //   - The materialization failed
    //   - The data is a placeholder or depends on a placeholder
    //   - The index is out of bounds.
    absl::StatusOr<xla::PjRtBuffer* absl_nonnull> operator[](
        int64_t index) const;

    // Returns true if this DeviceBufferList::Data is a placeholder.
    // This will never change state.
    [[nodiscard]] bool is_placeholder() const { return placeholder_; }

    // Returns true if this DeviceBufferList::Data is a deferred op and has not
    // yet started materialization.
    [[nodiscard]] bool is_deferred() const {
      return !placeholder_ && !materialization_pending_;
    }

    // Returns true if this DeviceBufferList::Data has started materialization.
    // It may or may not have completed materialization.
    [[nodiscard]] bool is_materializing() const {
      return materialization_pending_;
    }

    // Returns true if this DeviceBufferList::Data is executing.
    // The execution may or may not have completed.
    [[nodiscard]] bool is_executing() const { return materialization_started_; }

    // Returns true if this DeviceBufferList::Data has finished materialization.
    // It may have succeeded or failed.
    [[nodiscard]] bool is_materialized() const {
      return materialization_future_.IsKnownReady();
    }

    // Returns true if the DeviceBufferList::Data was created as a constant or
    // by a "torch.empty()" op.
    [[nodiscard]] bool is_constant() const { return constant_; }

    // Prints a debug string for the DeviceBufferList::Data into the ostream.
    std::ostream& PrintDebug(std::ostream& os) const;

   private:
    void ValidateMaterializedBuffers(int64_t index) const;

    // If the DeviceBufferList::Data was created by a constant or
    // "torch.empty()" op, then we will try to avoid materializing it, to
    // preserve constant folding capability in the compiler.
    const bool constant_ = false;

    // If the DeviceBufferList::Data has a DeferredOp, it is stored here,
    // protected by a mutex. This ensures that if one thread is trying to access
    // the DeferredOp, there won't be a data race with another thread that is
    // clearing it. Marked as mutable so that deferred_op() can be const while
    // still acquiring the mutex.
    mutable absl::Mutex deferred_op_mutex_;
    absl_nullable std::shared_ptr<DeferredOp> deferred_op_
        ABSL_GUARDED_BY(deferred_op_mutex_);

    // After the DeviceBufferList::Data is materialized, the optional
    // materialized_buffers_ will be populated with a nonempty list of
    // materialized_buffers_->buffers(), or the materialization_status_ field
    // will be set to a non-OK status. These are NOT mutex-guarded; once they
    // have been set, they are immutable, and can be safely read concurrently.
    // The materialization_started_ flag is used to assign responsibility for
    // setting these fields, and the materialization_promise_ and
    // materialization_future_ are used to notify that the critical section is
    // over and it is safe to read them.
    absl::Status materialization_status_ = absl::OkStatus();
    std::optional<MaterializedData> materialized_buffers_;

    // If the DeviceBufferList::Data was created as a placeholder, it will never
    // change state, and we don't need to ever acquire a mutex or check any
    // atomic flags.
    const bool placeholder_ = false;

    // The following flags are used to atomically enforce a one-way "ramp":
    //
    // deferred -> materialization pending. The first time this flag is set, the
    // DeviceBufferList::Data which sets the flag is responsible for clearing
    // the DeferredOp.
    //
    // materialization_pending -> (materialized or error). The first time that
    // the materialization_started_ flag is set, the DeviceBufferList::Data
    // which sets the flag is responsible for setting either the
    // materialized_buffers_ or materialization_status_ field (but not both),
    // and then fulfilling the materialization_promise_.
    //
    // Once the materialization_future_ is ready, materialization_status_ and
    // materialized_buffers_->buffers() can be read without locking.

    // This is set to true by the first caller to SetAsPendingMaterialization.
    // This first caller is responsible for clearing the DeferredOp (if it
    // exists).
    std::atomic_bool materialization_pending_ = false;

    // This is set to true by the first caller to either
    // SetMaterializationStarted or SetMaterializationError. This first caller
    // is responsible for setting the materialized_buffers_ or
    // materialization_status_ field, and then setting the
    // materialization_promise_.
    std::atomic_bool materialization_started_ = false;

    // When this future returns, the DeviceBufferList::Data has either
    // successfully started materialization or failed to materialize.
    // This is NOT mutex-guarded; is is safe to access as soon as the Data is
    // created, but may not be populated until later.
    // Once the future is ready and has an OK status, the
    // materialized_buffers_->buffers() can be safely accessed without locking.
    // WARNING: attempting to access a PjRtBuffer before this future is ready
    // will block the caller until the promise is set by another thread.
    xla::Future<> materialization_future_;

    // This promise is used to set the materialization_future_. It should only
    // be set by the first thread to set materialization_started_ to true,
    // and only after populating the materialized_buffers_ field.
    xla::Promise<> materialization_promise_;
  };

  // The data backing the DeviceBufferList.
  // This uses locks and atomics internally to enforce a thread-safe transition
  // from DeferredOp to a materialized  set of PjRtBuffers (or an error if
  // materialization fails).
  Data data_;

  // In PyTorch, every piece of data is associated with a device and stream.
  // This includes both data loaded from host-to-device, and data produced on
  // device by a computation.
  // In TorchTPU, our "data" object is a DeviceBufferList, and so every
  // DeviceBufferList is associated with a device and stream by their IDs.
  c10::DeviceIndex device_index_;
  c10::StreamId stream_id_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_DEVICE_BUFFER_H_
