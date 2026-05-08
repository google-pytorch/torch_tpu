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
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Allocator.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
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
//     contains a MaterializedBuffers (see note below).
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
// TODO(b/452027126): investigate thread-safety options

namespace torch_tpu {

// The current evaluation state of a referenced buffer in a DeviceBufferList.
// The only valid state transition is kDeferred -> kMaterialized, which occurs
// when a DeferredOp is executed.
// kMaterialized is an absorbing state. Once the data exists, it is immutable.
// kPlaceholder DeviceBufferRefs cannot change state; they can be used for
// compilation, but later executions will use different kMaterialized buffers.
enum class DeviceBufferRefState {
  // The reference is to a materialized, ready-to-use PjRtBuffer.
  kMaterialized,
  // The reference is to a deferred operation that has not been applied.
  kDeferred,
  // The reference represents a tensor for the purposes of compiled mode, but
  // doesn't actually have any data on-device. All compiled mode leaf inputs
  // must be placeholders, but eager mode tensors should never be placeholders.
  kPlaceholder,
};

enum class OpSplitMode {
  kNone,

  // Split the execution graph before this op.
  kSplitBefore,

  // Split the execution graph after this op.
  kSplitAfter,

  // Split the execution graph before and after this op.
  kSplitBoth,
};

class DeviceBufferList;
class DeferredOp;

using SharedDeviceBufferList = absl_nonnull std::shared_ptr<DeviceBufferList>;

// An OpQueue represents a set of deferred operations that are logically
// connected. OpQueues can be merged when an operation takes inputs from
// multiple queues.
// Each queue maintains its own list of deferred operations.
// TODO(bawilson): factor this into a separate file.
class OpQueue : public std::enable_shared_from_this<OpQueue> {
 public:
  // Constructs an op queue.
  OpQueue() = default;

  // Creates a new op queue or returns the global instance.
  static std::shared_ptr<OpQueue> Create();

  // Pushes a deferred node onto this queue.
  void push(std::weak_ptr<DeviceBufferList> device_buffer);

  // Returns the leaf nodes of the queue.
  std::vector<SharedDeviceBufferList> GetLeafNodes();

  // Acquires a snapshot of nodes for processing.
  // Locks processing_mu_ and returns the nodes.
  std::vector<SharedDeviceBufferList> AcquireProcessingSnapshot()
      ABSL_EXCLUSIVE_LOCK_FUNCTION(processing_mu_);

  // Releases the processing lock.
  void ReleaseProcessingSnapshot() ABSL_UNLOCK_FUNCTION(processing_mu_);

  // Returns all nodes in the processing queue.
  std::vector<SharedDeviceBufferList> GetAllNodes();

  class ABSL_SCOPED_LOCKABLE SnapshotLock {
   public:
    explicit SnapshotLock(std::shared_ptr<OpQueue> queue)
        ABSL_EXCLUSIVE_LOCK_FUNCTION(queue->processing_mu_)
        : queue_(std::move(queue)) {
      nodes_ = queue_->AcquireProcessingSnapshot();
    }
    ~SnapshotLock() ABSL_UNLOCK_FUNCTION() {
      queue_->ReleaseProcessingSnapshot();
    }
    const std::vector<SharedDeviceBufferList>& nodes() const { return nodes_; }

   private:
    std::shared_ptr<OpQueue> queue_;
    std::vector<SharedDeviceBufferList> nodes_;
  };

 private:
  // Prunes the queue to remove expired nodes.
  void Prune() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_staging_);

  absl::Mutex mu_staging_;
  std::vector<std::weak_ptr<DeviceBufferList>> staging_queue_
      ABSL_GUARDED_BY(mu_staging_);

  absl::Mutex processing_mu_;
  std::vector<std::weak_ptr<DeviceBufferList>> processing_queue_
      ABSL_GUARDED_BY(processing_mu_);
};

// A global singleton registry of all OpQueues that currently exist.
// TODO(bawilson): factor this into a separate file.
class OpQueueRegistry {
 public:
  // Returns the singleton registry instance.
  [[nodiscard]] static OpQueueRegistry& GetInstance();

  // Returns the single global op queue.
  absl_nonnull std::shared_ptr<OpQueue> MakeNewOpQueue();

  // Returns the single global op queue.
  absl_nonnull std::shared_ptr<OpQueue> MergeAll();

 private:
  OpQueueRegistry() = default;
  // OpQueueRegistry is neither copyable nor movable.
  OpQueueRegistry(const OpQueueRegistry&) = delete;
  OpQueueRegistry& operator=(const OpQueueRegistry&) = delete;
  OpQueueRegistry(OpQueueRegistry&&) = delete;
  OpQueueRegistry& operator=(OpQueueRegistry&&) = delete;

  absl::Mutex mu_;
  std::shared_ptr<OpQueue> global_queue_ ABSL_GUARDED_BY(mu_);
};

// A DeviceBufferRef is a reference to an element in a DeviceBufferList.
// It is nothing more than a container for a std::shared_ptr<DeviceBufferList>
// and an index into that list, with accessor methods for convenience.
//
// The c10::DataPtr at the root of the torch/aten/c10 data hierarchy holds an
// owning unique pointer (as a c10::UniqueVoidPointer) to a DeviceBufferRef.
// Whenever an at::Tensor is dropped, it may trigger a deallocation of the
// entire shared pointer chain (c10::TensorImpl, c10::StorageImpl, c10::DataPtr,
// DeviceBufferRef, and DeviceBufferList) unless there are additional shared
// pointers (at::Tensors, c10:Storages, or DeviceBufferRefs) preventing
// deallocation.
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
// Additionally, the aten/c10 hierarchy may swap out the
// c10::DataPtr (in the c10::StorageImpl), changing which DeviceBufferRef backs
// the at::Tensor. This can be used to "mutate" an at::Tensor from the
// perspective of aten, while respecting immutability at the XLA level.
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

  // The current state of the referenced buffer, as an enum.
  [[nodiscard]] DeviceBufferRefState state() const;

  [[nodiscard]] bool IsMaterialized() const;

  // The logical dimensions of the referenced buffer.
  [[nodiscard]] absl::Span<const int64_t> dimensions() const;

  // The number of elements in the referenced buffer.
  [[nodiscard]] int64_t num_elements() const;

  // The element type of the referenced buffer.
  [[nodiscard]] mlir::ElementType element_type() const;

  // If the DeviceBufferRef has a DeferredOp, returns a non-null pointer to
  // it. Otherwise, return a nullptr.
  [[nodiscard]] const DeferredOp* absl_nullable deferred_op() const;

  // Awaits for the device buffer to be ready to read. If the buffer is the
  // output of computation, then also waits for the computation to be done.
  absl::Status Synchronize() const;

  // Awaits the PjRtBuffer to be materialized, and then returns either the
  // error encountered during materialization, or a pointer to the
  // PjRtBuffer.
  absl::StatusOr<xla::PjRtBuffer* absl_nonnull> GetOrMaterializeBuffer() const;

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

  // Increments the number of child ops for the DeviceBufferList that this
  // DeviceBufferRef is attached to.
  void IncrementNumChildOps() const;

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

// Creates a c10::DataPtr to hold the given DeviceBufferRef on the given
// PrivateUse1 device index.
[[nodiscard]] c10::DataPtr MakeDataPtr(DeviceBufferRef buffer_ref,
                                       int device_idx);

// Delegate responsibility for deleting the DeviceBufferRef to the
// c10::DataPtr.
// The DeviceBufferRef* is used as both the "data" and "context" of the
// c10::DataPtr; this mirrors the semantics of a std::unique_ptr.
void DeleteDeviceBufferRef(void* ctx_ptr);

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
             std::shared_ptr<OpQueue> subgraph,
             OpSplitMode split_mode = OpSplitMode::kNone,
             Indices donated_indices = {})
      : op_name_(op_name),
        op_builder_(std::move(op_builder)),
        inputs_(std::move(inputs)),
        donated_indices_(std::move(donated_indices)),
        op_param_cache_keys_(std::move(op_param_cache_keys)),
        op_context_(ScopedPythonContextCapturer::GetContext()),
        split_mode_(split_mode),
        subgraph_(std::move(subgraph)) {
    absl::flat_hash_set<const DeviceBufferList*> unique_input_lists;
    for (const auto& input : inputs_) {
      if (unique_input_lists.insert(input.device_buffer_list().get()).second) {
        input.IncrementNumChildOps();
      }
    }
  }

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
  size_t Hash() const {
    auto h = static_cast<size_t>(op_name_);
    HashCombine(h, op_param_cache_keys_.size());
    return h;
  }

  // The name of the deferred op, like "add" or "matmul".
  OpName op_name() const { return op_name_; }

  // The MlirOpBuilder that will be used to compile the deferred op.
  const MlirOpBuilder& op_builder() const { return op_builder_; }

  // The inputs to the deferred op.
  absl::Span<const DeviceBufferRef> inputs() const { return inputs_; }

  // Context where the op is used in the user's PyTorch code.
  [[nodiscard]] const PythonContext& op_context() const { return op_context_; }

  // If true, then this DeferredOp has been executed at least once as a
  // non-output node; that is, it has been included as part of a materialized
  // DeferredOp graph without being a target materialization node. For example,
  // ```
  //   x = torch.zeros(1)
  //   y = x + 1
  //   print(y.cpu())
  // ```
  // will materialize `y` (replacing its DeferredOp with a PjRtBuffer), but
  // `x` will be left as a DeferredOp with has_been_executed() == true.
  // This can be used to detect nodes which are used persistently across loop
  // iterations; see ReexecutionHeuristic for its use.
  [[nodiscard]] bool has_been_executed() const { return has_been_executed_; }

  // Marks this DeferredOp as having been executed at least once.
  void mark_executed() const { has_been_executed_ = true; }

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

  // Returns the name of the custom kernel, if this DeferredOp is a custom
  // kernel. Otherwise, returns an empty string.
  std::string_view custom_kernel_name() const {
    if (op_name_ != OpName::kCustomKernel) {
      return "";
    }
    for (const auto& [key, value] : op_param_cache_keys_) {
      if (key == "custom_kernel_name") {
        return value;
      }
    }
    return "";
  }

  // Returns the kwargs string of the custom kernel, if this DeferredOp is a
  // custom kernel. Otherwise, returns an empty string.
  std::string_view custom_kernel_kwargs() const {
    if (op_name_ != OpName::kCustomKernel) {
      return "";
    }
    for (const auto& [key, value] : op_param_cache_keys_) {
      if (key == "custom_kernel_kwargs") {
        return value;
      }
    }
    return "";
  }

  // Returns the op queue this op belongs to.
  std::shared_ptr<OpQueue> subgraph() const { return subgraph_; }

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

  // If true, then this DeferredOp has been executed at least once as a
  // non-output node. This can be used to detect nodes which are used
  // persistently across loop iterations.
  mutable bool has_been_executed_ = false;

  // Records the number of other DeferredOps that have been created which depend
  // on this one. This is also called the "fanout" of the node.
  // This does *not* track the liveness of these ops; if a DeferredOp is
  // destroyed, the refcount will not be decremented.
  mutable int64_t num_child_ops_ = 0;

  // The split mode of the DeferredOp, which determines how the op is split into
  // subgraphs for compilation.
  const OpSplitMode split_mode_ = OpSplitMode::kNone;

  // The op queue this op belongs to.
  std::shared_ptr<OpQueue> subgraph_;

  friend std::ostream& operator<<(std::ostream& os,
                                  const DeferredOp& deferred_op);
};

// Lightweight wrapper around PjRtBuffers that are "materialized" but not yet
// available.
//
// This is used to allow callers to wait for a set of PjRtBuffers to be ready
// before accessing them.
class MaterializedBuffers {
 public:
  MaterializedBuffers() {
    auto promise_pair = xla::MakePromise();
    promise_ = std::move(promise_pair.first);
    future_ = std::move(promise_pair.second);
  }

  explicit MaterializedBuffers(
      std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers)
      : buffers_(std::move(buffers)) {
    auto promise_pair = xla::MakePromise();
    promise_ = std::move(promise_pair.first);
    future_ = std::move(promise_pair.second);

    // Fulfill the promise immediately, since the buffers are already available.
    std::move(promise_).Set(absl::OkStatus());
  }

  // Creates an unavailable MaterializedBuffers object. The Future object will
  // be used to determine when the buffers become available.
  MaterializedBuffers(
      std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers,
      xla::Future<> future)
      : future_(std::move(future)), buffers_(std::move(buffers)) {}

  bool IsAvailable() const { return future_.IsReady(); }

  // Wait for the buffers to be materialized.
  absl::Status Await() const { return future_.Await(); }

  int64_t size() const {
    TT_CHECK_THROW(IsAvailable(), error::kInternal)
        << "Attempted to access a PjRtBuffer that is not yet available.";
    return buffers_.size();
  }

  // Returns a reference to the PjRtBuffer at the given index, or nullptr if
  // the index is out of bounds or the buffers are not yet available.
  xla::PjRtBuffer* absl_nullable operator[](int64_t index) const {
    if (index < 0 || index >= buffers_.size()) {
      return nullptr;
    }
    if (!IsAvailable()) {
      return nullptr;
    }
    return buffers_[index].get();
  }

  void SetAsError(absl::Status status) {
    if (!future_.IsReady()) {
      promise_.Set(status);
    }
  }

  absl::Status SetAsAvailable(
      std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers) {
    buffers_ = std::move(buffers);
    promise_.Set(absl::OkStatus());
    return absl::OkStatus();
  }

 private:
  xla::Future<> future_;
  xla::Promise<> promise_;
  std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers_;
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
// base-to-view conversions on-demand (see GetBufferFromAtTensor and
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

  // Creates a deferred DeviceBufferList containing a constant tensor value,
  // based on a 1D byte array.
  // Under the hood, this uses `mlir::DenseElementsAttr::getFromRawBuffer`,
  // with special handling for booleans and complex numbers (which have
  // differing layout requirements between PyTorch and XLA).
  // This value will be "baked into" the MLIR for XLA compilation, allowing for
  // greater optimization potential.
  // However, this means that any future execution which have a different value
  // for the constant will require a re-compilation.
  static absl::StatusOr<DeviceBufferRef> CreateConstant(
      std::vector<char> cpu_tensor_data, Dimensions dimensions,
      mlir::ElementType element_type);

  // Creates a DeviceBufferList as if by using the torch.empty() operation
  // with fill_uninitialized_memory=True.
  // This will be a DeferredOp that fills the buffer with NaNs for floats
  // (including complex floats), and max for integers and booleans.
  static absl::StatusOr<DeviceBufferRef> CreateEmpty(
      Dimensions dimensions, mlir::ElementType element_type);

  // Creates a DeviceBufferList with a constant value of "no data".
  // Equivalent to CreateConstant with an empty cpu_tensor_data vector.
  static absl::StatusOr<DeviceBufferRef> CreateZeroSize(
      Dimensions dimensions, mlir::ElementType element_type);

  // Creates a DeviceBufferList that represents a compiled-mode placeholder.
  // This is a buffer that is not backed by any data, but is used to represent
  // an argument to a compiled executable.
  static absl::StatusOr<DeviceBufferRef> MakePlaceholder(
      Dimensions dimensions, mlir::ElementType element_type);

  // Sets the DeviceBufferList to a materialized state, initializing the future
  // to indicate when the buffers become available.
  // If the DeviceBufferList is already materialized, this is a no-op.
  absl::Status SetAsMaterialized();

  // Sets the DeviceBufferList to a materialized state with the given buffers.
  // If the DeviceBufferList is already materialized, the previous PjRtBuffers
  // will be replaced with the new ones.
  // The number of buffers, and their Shapes must match the existing
  // DeviceBufferList, otherwise an error is returned and no update is applied.
  absl::Status SetAsMaterialized(
      std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers);

  // Sets a dimension to be dynamic.
  absl::Status MarkDynamic(int64_t index, int64_t dimension,
                           int64_t lower_bound, int64_t upper_bound);
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

  // The current state of the indexed buffer, as an enum.
  [[nodiscard]] DeviceBufferRefState state(int64_t index) const;

  // The logical dimensions of the indexed buffer.
  [[nodiscard]] absl::Span<const int64_t> dimensions(int64_t index) const;

  // The number of elements in the indexed buffer.
  [[nodiscard]] int64_t num_elements(int64_t index) const;

  // The element type of the referenced buffer.
  [[nodiscard]] mlir::ElementType element_type(int64_t index) const;

  // If the DeviceBufferList has a DeferredOp, returns a non-null pointer to
  // it. Otherwise, return a nullptr.
  [[nodiscard]] const DeferredOp* absl_nullable deferred_op() const {
    return std::get_if<DeferredOp>(&data_);
  }

  // Awaits for the device buffer to be ready to read. If the buffer is the
  // output of computation, then also waits for the computation to be done.
  absl::Status Synchronize() const;

  // Awaits the PjRtBuffer to be materialized, and then returns either the
  // error encountered during materialization, or a pointer to the
  // PjRtBuffer.
  absl::StatusOr<xla::PjRtBuffer* absl_nonnull> GetOrMaterializeBuffer(
      int64_t index) const;

  // If the DeviceBufferList is materialized, returns a non-null pointer to
  // the MaterializedBuffers. Otherwise, returns a nullptr.
  [[nodiscard]] const MaterializedBuffers* absl_nullable materialized_buffers()
      const {
    return std::get_if<MaterializedBuffers>(&data_);
  }
  [[nodiscard]] MaterializedBuffers* absl_nullable materialized_buffers() {
    return std::get_if<MaterializedBuffers>(&data_);
  }

  // Returns the representative ID of the subgraph this node belongs to.
  [[nodiscard]] std::shared_ptr<OpQueue> subgraph() const { return subgraph_; }

  // If the DeviceBufferList has no live data pointers, it is "stale", meaning
  // that it will never be directly materialized and will never have any new
  // DeferredOps appended to it. This allows for more optimal materialization
  // patterns in some cases.
  [[nodiscard]] bool is_stale() const { return live_data_ptrs_ == 0; }

  // Returns the global index of the creation of this DeviceBufferList.  Lower
  // means earlier.
  [[nodiscard]] uint64_t creation_index() const { return creation_index_; }

  std::string DebugString() const;

  int64_t num_child_ops() const { return num_child_ops_; }
  void IncrementNumChildOps() { num_child_ops_++; }

 private:
  // Private constructor for a DeviceBufferList wrapping a single materialized
  // PjRtBuffer.
  // DeviceBufferListss should only ever be accessed via DeviceBufferRef, which
  // is returned by the Create* functions.
  // The shape will be inferred from the PjRtBuffer; the element type will be
  // the type specified by the argument.
  DeviceBufferList(absl_nonnull std::unique_ptr<xla::PjRtBuffer> buffer,
                   const mlir::ElementType element_type)
      : subgraph_(nullptr) {
    creation_index_ = g_creation_index.fetch_add(1);

    const xla::Shape& on_device_shape = buffer->on_device_shape();
    Shape shape(CopyIntVector(on_device_shape.dimensions()), element_type);
    shapes_.push_back(std::move(shape));

    auto buffer_address = buffer.get();
    std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
    buffers.push_back(std::move(buffer));
    data_ = MaterializedBuffers(std::move(buffers));
    ABSL_VLOG(3) << "[DeviceBuffer CONSTRUCTOR (materialized)] Created. Dims: "
                 << ToString(shapes_[0].dimensions())
                 << ", Type: " << ToString(shapes_[0].dtype())
                 << ", PjRtBuffer: " << buffer_address
                 << ", creation_index: " << creation_index_;
  }

  // Private constructor for a deferred DeviceBufferList.
  // DeviceBufferListss should only ever be accessed via DeviceBufferRef, which
  // is returned by the Create* functions.
  // The dimensions and element types must match the return values of the
  // DeferredOp's op_builder; if they do not, then there may be compilation
  // failures.
  DeviceBufferList(DeferredOp deferred_op, std::vector<Shape> shapes,
                   std::shared_ptr<OpQueue> subgraph)
      : data_(DeferredOp(std::move(deferred_op))),
        shapes_(std::move(shapes)),
        subgraph_(std::move(subgraph)) {
    creation_index_ = g_creation_index.fetch_add(1);
    ABSL_VLOG(3)
        << "[DeviceBuffer CONSTRUCTOR (deferred)] Created. DeferredOp: "
        << std::get<DeferredOp>(data_).op_name()
        << ", Number of outputs: " << shapes_.size()
        << ", Subgraph: " << subgraph_.get();
  }

  // Private constructor for a DeviceBufferList with a single unbacked buffer.
  // DeviceBufferListss should only ever be accessed via DeviceBufferRef, which
  // is returned by the Create* functions.
  //
  // The buffer is a compiled mode placeholder, representing data that the
  // compiled executable will expect to be provided as an argument.
  DeviceBufferList(Dimensions dimensions, const mlir::ElementType element_type)
      : subgraph_(nullptr) {
    creation_index_ = g_creation_index.fetch_add(1);
    shapes_.emplace_back(std::move(dimensions), element_type);
    ABSL_VLOG(3) << "[DeviceBuffer CONSTRUCTOR (bufferless)] Created. Dims: "
                 << ToString(shapes_[0].dimensions())
                 << ", Type: " << ToString(shapes_[0].dtype());
  }

  static std::atomic_uint64_t g_creation_index;

  // The data backing the DeviceBufferList.
  //   std::monostate: compiled mode placeholder buffers.
  //   DeferredOp: a deferred operation, with no actual data but an op_builder
  //     to materialize it when needed.
  //   MaterializedBuffers: materialized buffers, which may or may not yet be
  //   backed by PjRtBuffers - however, the materialization has been scheduled.
  std::variant<std::monostate, DeferredOp, MaterializedBuffers> data_;
  // The shapes of all the buffers in the DeviceBufferList.
  std::vector<Shape> shapes_;
  // The op queue this node belongs to. Only valid for deferred nodes.
  std::shared_ptr<OpQueue> subgraph_;
  // A global index of the creation of this DeviceBufferList (created atomically
  // for thread safety). Lower means earlier.
  uint64_t creation_index_ = 0;

  // The number of other DeferredOps that have been created which depend
  // on this DeviceBufferList. This does *not* track the liveness of these ops;
  // if a DeferredOp is destroyed, the refcount will not be updated.
  mutable std::atomic_int64_t num_child_ops_ = 0;

  // The number of live c10::DataPtrs to this DeviceBufferList.
  // This is incremented by MakeDataPtr and decremented by
  // DeleteDeviceBufferRef, and nowhere else.
  std::atomic_int64_t live_data_ptrs_ = 0;
  friend c10::DataPtr MakeDataPtr(DeviceBufferRef buffer_ref, int device_idx);
  friend void DeleteDeviceBufferRef(void* ctx_ptr);
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_DEVICE_BUFFER_H_
