/*
 * Copyright 2025 Google LLC
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

#include "torch_tpu/eager/device_buffer.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/util/accumulate.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

SubgraphRegistry& SubgraphRegistry::GetInstance() {
  static SubgraphRegistry* const registry = new SubgraphRegistry();
  return *registry;
}

size_t SubgraphRegistry::AllocateSubgraph() {
  absl::MutexLock lock(mu_);
  if (!free_list_.empty()) {
    const size_t index = free_list_.front();
    free_list_.pop_front();
    auto& subgraph_data = subgraph_data_[index];
    subgraph_data.ref_count = 1;
    return index;
  }

  // There are no previously-allocated subgraphs to reuse. Allocate a new one.
  size_t allocated_index = subgraph_data_.size();
  // The newly-created SubgraphData is a root (is its own parent).
  subgraph_data_.push_back(
      SubgraphData{.parent = allocated_index, .ref_count = 1});
  return allocated_index;
}

void SubgraphRegistry::DecrementRefCountLocked(size_t subgraph_index) {
  size_t curr = subgraph_index;
  auto* curr_data = &subgraph_data_[curr];
  curr_data->ref_count--;

  // Cascading free all SubgraphData with a ref_count of 0.
  while (curr_data->ref_count == 0) {
    // Record what the parent value is before we overwrite it.
    size_t next = curr_data->parent;

    // Mark curr as free.
    curr_data->queue.clear();
    curr_data->parent = curr;  // Mark as root.
    free_list_.push_back(curr);

    // If we have reached the root, we can stop.
    // Otherwise, loop and decrement the ref_count of the parent.
    if (next == curr) return;
    curr = next;
    curr_data = &subgraph_data_[curr];
    curr_data->ref_count--;
  }
}

Subgraph::Subgraph()
    : subgraph_index_(SubgraphRegistry::GetInstance().AllocateSubgraph()) {}

Subgraph::~Subgraph() {
  SubgraphRegistry& registry = SubgraphRegistry::GetInstance();
  absl::MutexLock lock(registry.mu_);
  registry.DecrementRefCountLocked(subgraph_index_);
}

size_t SubgraphRegistry::FindLocked(size_t subgraph_index) {
  // Walk up the tree to find the root subgraph, keeping track of the path in a
  // stack.
  // Once we find a subgraph that is its own parent, we have found the root.
  size_t root = subgraph_data_[subgraph_index].parent;
  while (subgraph_data_[root].parent != root) {
    root = subgraph_data_[root].parent;
  }

  // Path compression: update the parents of all subgraphs along the path to
  // the root to make future walks faster.
  size_t curr = subgraph_index;
  while (subgraph_data_[curr].parent != root) {
    size_t old_parent = subgraph_data_[curr].parent;

    // Move the reference from the old parent to the new root.
    // Increment FIRST, so that we don't accidentally free something we need to
    // retain.
    subgraph_data_[root].ref_count++;
    DecrementRefCountLocked(old_parent);

    subgraph_data_[curr].parent = root;
    curr = old_parent;
  }

  return root;
}

void SubgraphRegistry::SubgraphData::Prune() {
  queue.erase(std::remove_if(queue.begin(), queue.end(),
                             [](std::weak_ptr<DeviceBufferList>& weak_node) {
                               std::shared_ptr<DeviceBufferList> node =
                                   weak_node.lock();
                               return (!node || node->num_child_ops() > 0);
                             }),
              queue.end());
}

void SubgraphRegistry::SubgraphData::PruneAndReturnLeafNodes(
    std::vector<SharedDeviceBufferList>& leaf_nodes_out) {
  queue.erase(
      std::remove_if(
          queue.begin(), queue.end(),
          [&leaf_nodes_out](std::weak_ptr<DeviceBufferList>& weak_node) {
            std::shared_ptr<DeviceBufferList> node = weak_node.lock();
            if ((!node || node->num_child_ops() > 0)) {
              return true;
            }
            leaf_nodes_out.push_back(std::move(node));
            return false;
          }),
      queue.end());
}

void SubgraphRegistry::Push(size_t subgraph_index,
                            std::weak_ptr<DeviceBufferList> device_buffer) {
  absl::MutexLock lock(mu_);
  size_t root = FindLocked(subgraph_index);
  if (subgraph_data_[root].queue.size() >=
      subgraph_data_[root].queue.capacity()) {
    // Try to free up capacity by pruning down to just the live leaf nodes.
    subgraph_data_[root].Prune();
  }
  subgraph_data_[root].queue.push_back(std::move(device_buffer));
}

void Subgraph::push(std::weak_ptr<DeviceBufferList> device_buffer) const {
  SubgraphRegistry::GetInstance().Push(subgraph_index_,
                                       std::move(device_buffer));
}

void SubgraphRegistry::MergeIndexes(absl::Span<const size_t> subgraph_indexes) {
  // Merging nothing or merging a single subgraph has no effect.
  if (subgraph_indexes.size() < 2) return;

  const size_t new_root = FindLocked(subgraph_indexes[0]);
  auto& new_root_data = subgraph_data_[new_root];
  for (size_t i = 1; i < subgraph_indexes.size(); ++i) {
    const size_t old_root = FindLocked(subgraph_indexes[i]);
    if (new_root == old_root) continue;  // Already merged.

    auto& old_root_data = subgraph_data_[old_root];

    // If we can fit all of the merged nodes into the existing capacity, we
    // don't need to check any weak_ptrs for expiration and can just do a fast
    // insert.
    if (new_root_data.queue.size() + old_root_data.queue.size() >
        new_root_data.queue.capacity()) {
      new_root_data.Prune();
      old_root_data.Prune();
    }
    new_root_data.queue.insert(
        new_root_data.queue.end(),
        std::make_move_iterator(old_root_data.queue.begin()),
        std::make_move_iterator(old_root_data.queue.end()));

    // Set the old root to be an empty pointer to the new root.
    old_root_data.queue.clear();

    new_root_data.ref_count++;

    old_root_data.parent = new_root;
  }
}

void SubgraphRegistry::Merge(
    absl::Span<const std::shared_ptr<Subgraph>> subgraphs) {
  // Merging nothing or merging a single subgraph has no effect.
  if (subgraphs.size() < 2) return;

  std::vector<size_t> subgraph_indexes;
  subgraph_indexes.reserve(subgraphs.size());
  for (const auto& subgraph : subgraphs) {
    subgraph_indexes.push_back(subgraph->subgraph_index());
  }

  absl::MutexLock lock(mu_);
  MergeIndexes(subgraph_indexes);
}

void SubgraphRegistry::MergeAll() {
  absl::MutexLock lock(mu_);
  std::vector<size_t> root_subgraphs;
  for (size_t i = 0; i < subgraph_data_.size(); ++i) {
    if (subgraph_data_[i].ref_count > 0 && subgraph_data_[i].parent == i) {
      root_subgraphs.push_back(i);
    }
  }
  MergeIndexes(root_subgraphs);
}

std::vector<SharedDeviceBufferList> SubgraphRegistry::GetLeafNodes(
    size_t subgraph_index) {
  absl::MutexLock lock(mu_);
  const size_t root = FindLocked(subgraph_index);
  std::vector<SharedDeviceBufferList> leaf_nodes;
  subgraph_data_[root].PruneAndReturnLeafNodes(leaf_nodes);
  return leaf_nodes;
}

std::vector<SharedDeviceBufferList> Subgraph::GetLeafNodes() const {
  return SubgraphRegistry::GetInstance().GetLeafNodes(subgraph_index_);
}

[[nodiscard]] bool SubgraphRegistry::SubgraphsEqual(const Subgraph& subgraph1,
                                                    const Subgraph& subgraph2) {
  if (subgraph1.subgraph_index() == subgraph2.subgraph_index()) {
    return true;
  }
  absl::MutexLock lock(mu_);
  const size_t root1 = FindLocked(subgraph1.subgraph_index());
  const size_t root2 = FindLocked(subgraph2.subgraph_index());
  return root1 == root2;
}

size_t DeviceBufferList::size_bytes(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  const auto xla_type = ConvertTo<xla::PrimitiveType>(shapes_[index].dtype());
  absl::Span<const int64_t> dimensions = shapes_[index].dimensions();
  if (dimensions.empty()) {
    // Scalars are 1 element, size depends on the element type.
    return xla::ShapeUtil::ByteSizeOfPrimitiveType(xla_type);
  }
  for (int64_t dim : dimensions) {
    if (dim == 0) {
      // Non-scalar tensors with a 0 in any dimension have no data.
      return 0;
    }
  }
  return xla::ShapeUtil::ByteSizeOf(
      xla::ShapeUtil::MakeShape(xla_type, dimensions));
}

absl::StatusOr<size_t> DeviceBufferList::pjrt_buffer_size(int64_t index) const {
  TT_ASSIGN_OR_RETURN(xla::PjRtBuffer* const pjrt_buffer,
                      GetOrMaterializeBuffer(index));

  TT_RET_CHECK(!pjrt_buffer->IsDeleted(), error::kFailedPrecondition)
      << "DeviceBufferRef has a PjRtBuffer, but it is deleted";
  auto on_device_size_in_bytes = pjrt_buffer->GetOnDeviceSizeInBytes();
  if (on_device_size_in_bytes.ok()) {
    return on_device_size_in_bytes.value();
  }
  ABSL_VLOG(1)
      << "[DeviceBufferRef::pjrt_buffer_size] Failed to get "
         "PjRtBuffer OnDeviceSizeInBytes: "
      << on_device_size_in_bytes.status()
      << ". Inferring from PjRtBuffer shape and DeviceBufferRef elementtype.";
  xla::Shape physical_buffer_shape_estimate = xla::ShapeUtil::MakeShape(
      ConvertTo<xla::PrimitiveType>(shapes_[index].dtype()),
      pjrt_buffer->on_device_shape().dimensions());
  return xla::ShapeUtil::ByteSizeOf(physical_buffer_shape_estimate);
}

namespace {

void DebugMaterializedState(std::ostream& os,
                            const MaterializedBuffers& materialized_buffers) {
  os << "materialized";
  if (materialized_buffers.IsAvailable()) {
    os << ", ready";
  } else {
    os << ", pending";
    return;
  }
  if (!materialized_buffers.size()) {
    os << ", empty";
    return;
  }
  xla::PjRtBuffer* absl_nullable maybe_pjrt_buffer = materialized_buffers[0];
  if (maybe_pjrt_buffer == nullptr) {
    os << ", null";
    return;
  }
  const xla::PjRtBuffer* pjrt_buffer = maybe_pjrt_buffer;
  if (pjrt_buffer->IsDeleted()) {
    os << ", deleted";
    return;
  }
  os << ", on_device_shape: " << pjrt_buffer->on_device_shape().ToString(true);
}

void DebugDeferredOpState(std::ostream& os, const DeferredOp& deferred_op) {
  os << "deferred, op_name: " << deferred_op.op_name();
}

// Logs the data variant state with one of these formats:
//   materialized, pjrt_buffer addr 0x1234567890, on_device_shape: f32[8,16]
//   materialized, pjrt_buffer addr 0x1234567890, deleted
//   deferred, op_name: add
//   placeholder
void DebugDataState(std::ostream& os,
                    const DeferredOp* absl_nullable deferred_op,
                    const MaterializedBuffers* absl_nullable buffers,
                    absl::Span<const int64_t> dimensions) {
  if (deferred_op != nullptr) {
    DebugDeferredOpState(os, *deferred_op);
  } else if (buffers != nullptr) {
    DebugMaterializedState(os, *buffers);
  } else {
    os << "placeholder";
  }
}

}  // namespace

std::string DeviceBufferList::DebugString() const {
  std::ostringstream os;
  os << "DeviceBufferList:"
     << "\n\tAddress: " << this << "\n\tNum buffers: " << size();
  for (auto i = 0; i < size(); ++i) {
    os << "\n\t\t=== Buffer " << i << " ===\n\t\t";
    DebugDataState(os, deferred_op(), materialized_buffers(), dimensions(i));
  }
  return os.str();
}

std::string DeviceBufferRef::DebugString() const {
  std::ostringstream os;
  os << "DeviceBufferRef:"
     << "\n\tAddress of ref: " << this
     << "\n\tAddress of DeviceBufferList: " << device_buffer_list_.get()
     << "\n\tIndex: " << index_
     << "\n\tShape and type: " << ToString(element_type())
     << ToString(dimensions()) << "\n\tData state: ";
  DebugDataState(os, deferred_op(), device_buffer_list_->materialized_buffers(),
                 dimensions());
  return os.str();
}

DeviceBufferRefState DeviceBufferList::state(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  if (std::holds_alternative<DeferredOp>(data_)) {
    return DeviceBufferRefState::kDeferred;
  }
  if (std::holds_alternative<MaterializedBuffers>(data_)) {
    return DeviceBufferRefState::kMaterialized;
  }
  return DeviceBufferRefState::kPlaceholder;
}

absl::Span<const int64_t> DeviceBufferList::dimensions(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  return shapes_[index].dimensions();
}

mlir::ElementType DeviceBufferList::element_type(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  return shapes_[index].dtype();
}

int64_t DeviceBufferList::num_elements(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  // Validated at construction time to not overflow.
  return c10::multiply_integers(shapes_[index].dimensions());
}

absl::Status DeviceBufferList::Synchronize() const {
  TT_RET_CHECK(state(0) == DeviceBufferRefState::kMaterialized,
               error::kFailedPrecondition)
      << "cannot synchronize a DeviceBufferList that is not materialized";
  for (auto i = 0; i < size(); ++i) {
    TT_ASSIGN_OR_RETURN(auto* pjrt_buffer, GetOrMaterializeBuffer(i));
    auto future = pjrt_buffer->GetReadyFuture();
    TT_RETURN_IF_ERROR(future.Await());
  }
  return absl::OkStatus();
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull>
DeviceBufferList::GetOrMaterializeBuffer(int64_t index) const {
  if (std::holds_alternative<MaterializedBuffers>(data_)) {
    const auto& buffers = std::get<MaterializedBuffers>(data_);
    TT_RETURN_IF_ERROR(buffers.Await());
    TT_RET_CHECK(index >= 0 && index < buffers.size(), error::kInvalidArgument)
        << "Index " << index << " is out of bounds for buffers of size "
        << buffers.size();
    xla::PjRtBuffer* maybe_buffer = buffers[index];
    TT_RET_CHECK(maybe_buffer, error::kFailedPrecondition)
        << "MaterializedBuffers has no/null PjRtBuffer at index " << index;
    return maybe_buffer;
  }
  return TT_ERROR(error::kFailedPrecondition)
         << "DeviceBufferList does not have a PjRtBuffer at index " << index
         << " because it is not materialized";
}

// Delegate responsibility for deleting the DeviceBufferRef to the
// c10::DataPtr.
// The DeviceBufferRef* is used as both the "data" and "context" of the
// c10::DataPtr; this mirrors the semantics of a std::unique_ptr.
void DeleteDeviceBufferRef(void* ctx_ptr) {
  DeviceBufferRef* const ref_ptr = static_cast<DeviceBufferRef*>(ctx_ptr);
  if (ref_ptr) {
    ABSL_VLOG(3) << "[c10::DataPtr deleter] deleting "
                 << ref_ptr->DebugString();
    ref_ptr->device_buffer_list()->live_data_ptrs_--;
  }
  delete ref_ptr;
}

c10::DataPtr MakeDataPtr(DeviceBufferRef buffer_ref, const int device_idx) {
  auto* absl_nonnull const raw_ref_ptr =
      new DeviceBufferRef(std::move(buffer_ref));
  raw_ref_ptr->device_buffer_list()->live_data_ptrs_++;
  return c10::DataPtr(raw_ref_ptr, raw_ref_ptr, DeleteDeviceBufferRef,
                      c10::Device(GetPrivateUse1DeviceType(), device_idx));
}

std::atomic_uint64_t DeviceBufferList::g_creation_index = 0;

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateMaterialized(
    absl_nonnull std::unique_ptr<xla::PjRtBuffer> buffer) {
  Dimensions dimensions = CopyIntVector(buffer->on_device_shape().dimensions());
  TT_ASSIGN_OR_RETURN(
      const auto element_type,
      ConvertTo<mlir::ElementType>(buffer->on_device_shape().element_type()));
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  // Can't use make_shared because the constructor is private.
  auto device_buffer_list = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(buffer), element_type));
  return DeviceBufferRef(std::move(device_buffer_list), 0);
}

absl::StatusOr<DeviceBufferRef>
DeviceBufferList::CreateMaterializedNonAvailable(
    absl_nonnull std::unique_ptr<xla::PjRtBuffer> buffer,
    xla::Future<> future) {
  Dimensions dimensions = CopyIntVector(buffer->on_device_shape().dimensions());
  TT_ASSIGN_OR_RETURN(
      const auto element_type,
      ConvertTo<mlir::ElementType>(buffer->on_device_shape().element_type()));
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));

  // Can't use make_shared because the constructor is private.
  auto device_buffer_list = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(buffer), element_type, std::move(future)));
  return DeviceBufferRef(std::move(device_buffer_list), 0);
}

absl::StatusOr<std::vector<DeviceBufferRef>> DeviceBufferList::CreateDeferred(
    OpName op_name, MlirOpBuilder op_builder,
    std::vector<DeviceBufferRef> inputs, OpParamCacheKeys op_param_cache_keys,
    std::vector<Shape> output_shapes, OpSplitMode split_mode,
    Indices donated_indices) {
  // Validate that the output shapes are valid.
  for (const auto& output_shape : output_shapes) {
    TT_RETURN_IF_ERROR(ValidateTensorByteSize(output_shape.dimensions(),
                                              output_shape.dtype()));
  }
  int64_t num_outputs = output_shapes.size();

  std::shared_ptr<Subgraph> subgraph = nullptr;

  if (IsDistributedOp(op_name)) {
    // Each distributed op acts as a barrier; all prior operations (connected or
    // not) that were created before it must be considered part of the same
    // graph so that proper ordering is maintained.
    // Otherwise, two independent collective operations could be isolated in
    // disconnected subgraphs, and different rank processes could have different
    // orderings of these subgraphs, leading to a deadlock.
    //
    // Create a new subgraph for this distributed op, and then merge all
    // subgraphs; after merging, this subgraph's root will be the global root.
    subgraph = std::make_shared<Subgraph>();
    SubgraphRegistry::GetInstance().MergeAll();
  } else {
    // Dispatching a new op may cause 2 or more existing subgraphs to merge.
    std::vector<std::shared_ptr<Subgraph>> input_subgraphs;
    for (const auto& input : inputs) {
      if (const auto* deferred_op = input.deferred_op()) {
        input_subgraphs.push_back(deferred_op->subgraph());
      }
    }
    if (input_subgraphs.size() > 1) {
      // If all input subgraphs currently share the same root, this operation
      // will be a no-op.
      SubgraphRegistry::GetInstance().Merge(input_subgraphs);
    }
    if (!input_subgraphs.empty()) {
      subgraph = input_subgraphs[0];
    }
  }

  if (!subgraph) {
    // Nullary, non-distributed ops start a new subgraph.
    subgraph = std::make_shared<Subgraph>();
  }

  // Create the DeferredOp.
  auto op = DeferredOp(op_name, std::move(op_builder), std::move(inputs),
                       std::move(op_param_cache_keys), subgraph, split_mode,
                       std::move(donated_indices));

  // Wrap the DeferredOp in a DeviceBufferList.
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(op), std::move(output_shapes), subgraph));

  subgraph->push(std::weak_ptr<DeviceBufferList>(device_buffer));

  // Construct one DeviceBufferRef for each output.
  std::vector<DeviceBufferRef> device_buffer_refs;
  device_buffer_refs.reserve(num_outputs);
  for (int64_t i = 0; i < num_outputs; ++i) {
    device_buffer_refs.push_back(DeviceBufferRef(device_buffer, i));
  }
  return device_buffer_refs;
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateConstant(
    std::vector<char> cpu_tensor_data, Dimensions dimensions,
    mlir::ElementType element_type) {
  // Create the components of the DeferredOp.
  auto op_name = OpName::kTorchTpuInternalConstant;
  ScopedPythonContextCapturer capturer(op_name);

  // Create the cache keys for the op parameters.
  // Since a change in data causes recompilation, we include the hash of the
  // tensor data as part of the cache key.
  auto op_param_cache_keys = OpParamCacheKeys::Empty();
  TT_RETURN_IF_ERROR(
      op_param_cache_keys.SetParam("data", absl::HashOf(cpu_tensor_data)));
  TT_RETURN_IF_ERROR(op_param_cache_keys.SetParam("dimensions", dimensions));
  TT_RETURN_IF_ERROR(
      op_param_cache_keys.SetParam("element_type", element_type));

  // The op returns a single tensor with the given shape and dtype.
  std::vector<Shape> output_shapes;
  output_shapes.push_back(Shape(dimensions, element_type));  // intentional copy

  auto op_builder = [cpu_tensor_data = std::move(cpu_tensor_data), element_type,
                     dimensions = std::move(dimensions)](
                        mlir::MlirBuilder& builder,
                        absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.empty(), error::kInvalidArgument)
        << "unexpected input to constant op";

    auto ranked_tensor_type =
        mlir::makeTensorType(builder.getContext(), dimensions, element_type);

    if (element_type == mlir::ElementType::PRED) {
      // Special case for boolean tensors.
      //
      // PyTorch stores booleans as one-per-byte on CPU, but XLA uses packed
      // 1-bit booleans. So we just make a constant byte tensor and use
      // stablehlo.convert, which will do the packing.
      //
      // DenseIntElementsAttr::get has an assertion that checks if the
      // signedness of the data, as indicated by
      // std::numeric_limits<char>::is_signed, matches the signedness of the
      // element type. However, whether or not char is signed is implementation
      // defined, which can cause assertion failures in some environments.
      // So we match the signedness of the byte tensor (using I8 or UI8) to the
      // signedness of char according to the current implementation.
      const auto byte_element_type = std::numeric_limits<char>::is_signed
                                         ? mlir::ElementType::I8
                                         : mlir::ElementType::UI8;

      auto shaped_byte_tensor_type = mlir::makeTensorType(
          builder.getContext(), dimensions, byte_element_type);
      auto shaped_byte_constant = mlir::stablehlo::Constant(
          builder, mlir::DenseIntElementsAttr::get(shaped_byte_tensor_type,
                                                   cpu_tensor_data));
      return DynamicMlirOpResults{
          mlir::stablehlo::Convert(ranked_tensor_type, shaped_byte_constant)};
    }

    auto dense_elements_attr = mlir::DenseElementsAttr::getFromRawBuffer(
        ranked_tensor_type, cpu_tensor_data);
    return DynamicMlirOpResults{
        mlir::stablehlo::Constant(builder, dense_elements_attr)};
  };

  // OpSplitMode is kNone; we don't need to split around a constant.
  // No device inputs, so no aliased inputs.
  TT_ASSIGN_OR_RETURN(
      auto results,
      DeviceBufferList::CreateDeferred(
          op_name, std::move(op_builder), /*inputs=*/{},
          std::move(op_param_cache_keys), std::move(output_shapes)));
  TT_RET_CHECK(results.size() == 1, error::kInternal)
      << "CreateConstant should return exactly one output";
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateEmpty(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  auto op_builder = [dimensions =
                         CopyIntVector(absl::MakeConstSpan(dimensions)),
                     element_type](mlir::MlirBuilder& builder,
                                   absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.empty(), error::kInvalidArgument)
        << "CreateEmpty should not be called with any inputs";
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, element_type, dimensions)};
  };
  Shape output_shape(std::move(dimensions), element_type);
  TT_ASSIGN_OR_RETURN(
      auto results, DeviceBufferList::CreateDeferred(
                        OpName::kEmpty, std::move(op_builder), /*inputs=*/{},
                        OpParamCacheKeys::Empty(), {std::move(output_shape)}));
  ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateZeroSize(
    Dimensions dimensions, mlir::ElementType element_type) {
  bool is_zero_sized = false;
  for (int64_t dim : dimensions) {
    if (dim == 0) {
      is_zero_sized = true;
      break;
    }
  }
  TT_RET_CHECK(is_zero_sized, error::kInvalidArgument)
      << "CreateZeroSize requires a zero-sized tensor, but got: "
      << ToString(dimensions);
  return CreateConstant({}, std::move(dimensions), element_type);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::MakePlaceholder(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(dimensions), element_type));
  return DeviceBufferRef(std::move(device_buffer), 0);
}

absl::Status DeviceBufferList::SetAsMaterialized() {
  ABSL_VLOG(1) << "[SetAsMaterialized] Setting as empty materialized";
  if (!std::holds_alternative<MaterializedBuffers>(data_)) {
    data_ = MaterializedBuffers();
  }
  return absl::OkStatus();
}

namespace {

// Verify that the requested `at_shape` that a deferred op expects is valid for
// the given `buffer_shape` that is on device.
//
// For static dimensions the buffer and op shape must match exactly, for bounded
// dimensions, only check upper bounds. We could get the real on-device shape
// from `buffers[i]->logical_dimensions()`, but this is a device syncing
// operation (blocking), so we want to avoid it.
absl::Status ValidateBufferShape(const Shape& at_shape,
                                 const xla::Shape& buffer_shape) {
  // Ranks must match
  absl::Span<const int64_t> buffer_dims = buffer_shape.dimensions();
  TT_RET_CHECK(at_shape.dimensions().size() == buffer_dims.size(),
               error::kInvalidArgument)
      << "unexpected rank for buffer; expected: "
      << ToString(at_shape.dimensions().size())
      << " but got: " << ToString(buffer_dims.size());

  // Static dims must match and dynamic dims must be LTE the upper bound.
  for (int64_t d = 0; d < at_shape.dimensions().size(); ++d) {
    TT_RET_CHECK(at_shape.dimensions()[d] == buffer_dims[d] ||
                     (buffer_shape.is_dynamic_dimension(d) &&
                      at_shape.dimensions()[d] <= buffer_dims[d]),
                 error::kInvalidArgument)
        << "incompatible buffer shapes at dimension " << d << "; for op shape "
        << ToString(at_shape.dimensions()) << " and buffer shape "
        << buffer_shape.ToString();
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status DeviceBufferList::SetAsMaterialized(
    std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers) {
  // Call SetAsMaterialized() so as to initialize field data_ with
  // MaterializedBuffers. The body of this function expects that.
  TT_RETURN_IF_ERROR(SetAsMaterialized());

  TT_RET_CHECK(shapes_.size() == buffers.size(), error::kInvalidArgument)
      << "unexpected number of buffers; expected: " << shapes_.size()
      << " but got: " << buffers.size();
  auto* materialized_buffers = std::get_if<MaterializedBuffers>(&data_);
  TT_RET_CHECK(materialized_buffers, error::kInvalidArgument)
      << "DeviceBufferList is not in a materialized state";
  TT_RET_CHECK(!materialized_buffers->IsAvailable(), error::kInvalidArgument)
      << "DeviceBufferList was already made available";
  for (size_t i = 0; i < shapes_.size(); ++i) {
    TT_RET_CHECK(!buffers[i]->IsDeleted(), error::kInvalidArgument)
        << "buffer " << i << " is deleted";
    TT_RETURN_IF_ERROR(
        ValidateBufferShape(shapes_[i], buffers[i]->on_device_shape()))
            .SetPrepend()
        << "buffer " << i << ": ";
    TT_ASSIGN_OR_RETURN(
        mlir::ElementType actual_element_type,
        ConvertTo<mlir::ElementType>(buffers[i]->element_type()),
        _.SetOverride() << "buffer " << i
                        << " has an unsupported element type: "
                        << xla::primitive_util::LowercasePrimitiveTypeName(
                               buffers[i]->element_type()));
    TT_RET_CHECK(actual_element_type == shapes_[i].dtype(),
                 error::kInvalidArgument)
        << "unexpected element type for buffer " << i
        << "; expected: " << ToString(shapes_[i].dtype())
        << " but got: " << ToString(actual_element_type);
  }
  ABSL_VLOG(1) << "[SetAsMaterialized] Setting as materialized";
  return materialized_buffers->SetAsAvailable(std::move(buffers));
}

absl::Status DeviceBufferList::MarkDynamic(int64_t index, int64_t dimension,
                                           int64_t lower_bound,
                                           int64_t upper_bound) {
  TT_RET_CHECK(lower_bound >= 2 && lower_bound <= upper_bound,
               error::kInvalidArgument)
      << "trying to mark dimension " << dimension
      << " as dynamic with invalid bounds [" << lower_bound << ", "
      << upper_bound << "]";
  TT_RET_CHECK(index >= 0 && index < shapes_.size(), error::kOutOfRange)
      << "index " << index << " is out of bounds for DeviceBufferList of size "
      << shapes_.size();
  Shape& shape = shapes_[index];
  TT_RET_CHECK(dimension >= 0 && dimension < shape.dimensions().size(),
               error::kOutOfRange)
      << "dimension " << dimension << " is out of bounds for tensor of rank "
      << shape.dimensions().size();
  TT_RET_CHECK(shape.dimensions()[dimension] >= lower_bound &&
                   shape.dimensions()[dimension] <= upper_bound,
               error::kOutOfRange)
      << "trying to mark dimension " << dimension << " as dynamic with bounds ["
      << lower_bound << ", " << upper_bound << "], but the dimension size is "
      << shape.dimensions()[dimension];
  auto it_find = std::find_if(
      shape.dynamic_dimensions().begin(), shape.dynamic_dimensions().end(),
      [dimension](const BoundedDynamicDimension& dynamic_dimension) {
        return dynamic_dimension.dimension == dimension;
      });
  if (it_find != shape.dynamic_dimensions().end()) {
    it_find->lower_bound = lower_bound;
    it_find->upper_bound = upper_bound;
  } else {
    shape.dynamic_dimensions().push_back({.dimension = dimension,
                                          .lower_bound = lower_bound,
                                          .upper_bound = upper_bound});
  }
  return absl::OkStatus();
};

absl::Span<const BoundedDynamicDimension> DeviceBufferList::dynamic_dimensions(
    int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  return shapes_[index].dynamic_dimensions();
}

absl::StatusOr<DeviceBufferRef> DeviceBufferRef::Create(
    SharedDeviceBufferList device_buffer_list, int64_t index) {
  // Gracefully return an error on creation, but crash hard if the bounds check
  // is violated afterwards, as that would indicate this check was bypassed.
  TT_RET_CHECK(index >= 0 && index < device_buffer_list->size(),
               error::kOutOfRange)
      << "index " << index << " is out of bounds for DeviceBufferList of size "
      << device_buffer_list->size();
  return DeviceBufferRef(std::move(device_buffer_list), index);
}

size_t DeviceBufferRef::size_bytes() const {
  return device_buffer_list_->size_bytes(index_);
}

absl::StatusOr<size_t> DeviceBufferRef::pjrt_buffer_size() const {
  return device_buffer_list_->pjrt_buffer_size(index_);
}

DeviceBufferRefState DeviceBufferRef::state() const {
  return device_buffer_list_->state(index_);
}

[[nodiscard]] absl::Span<const int64_t> DeviceBufferRef::dimensions() const {
  return device_buffer_list_->dimensions(index_);
}

[[nodiscard]] int64_t DeviceBufferRef::num_elements() const {
  return device_buffer_list_->num_elements(index_);
}

[[nodiscard]] mlir::ElementType DeviceBufferRef::element_type() const {
  return device_buffer_list_->element_type(index_);
}

[[nodiscard]] const DeferredOp* absl_nullable DeviceBufferRef::deferred_op()
    const {
  return device_buffer_list_->deferred_op();
}

absl::Status DeviceBufferRef::Synchronize() const {
  return device_buffer_list_->Synchronize();
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull>
DeviceBufferRef::GetOrMaterializeBuffer() const {
  return device_buffer_list_->GetOrMaterializeBuffer(index_);
}

absl::Status DeviceBufferRef::MarkDynamic(int64_t dimension,
                                          int64_t lower_bound,
                                          int64_t upper_bound) const {
  return device_buffer_list()->MarkDynamic(index_, dimension, lower_bound,
                                           upper_bound);
}

absl::Span<const BoundedDynamicDimension> DeviceBufferRef::dynamic_dimensions()
    const {
  return device_buffer_list()->dynamic_dimensions(index_);
}

void DeviceBufferRef::IncrementNumChildOps() const {
  device_buffer_list_->IncrementNumChildOps();
}

}  // namespace torch_tpu
