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
#include <cstdint>
#include <cstring>
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
#include "ATen/core/ATen_fwd.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/util/accumulate.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

void Subgraph::Prune() {
  queue_.erase(std::remove_if(
                   queue_.begin(), queue_.end(),
                   [](std::weak_ptr<DeviceBufferList>& weak_node) {
                     std::shared_ptr<DeviceBufferList> node = weak_node.lock();
                     if (!node) {
                       return true;
                     }
                     const auto* deferred_op = node->deferred_op();
                     if (!deferred_op || deferred_op->num_child_ops() > 0) {
                       return true;
                     }
                     return false;
                   }),
               queue_.end());
}

void Subgraph::PruneAndReturnLeafNodes(
    std::vector<SharedDeviceBufferList>& leaf_nodes_out) {
  queue_.erase(
      std::remove_if(
          queue_.begin(), queue_.end(),
          [&leaf_nodes_out](std::weak_ptr<DeviceBufferList>& weak_node) {
            std::shared_ptr<DeviceBufferList> node = weak_node.lock();
            if (!node) {
              return true;
            }
            const auto* deferred_op = node->deferred_op();
            if (!deferred_op || deferred_op->num_child_ops() > 0) {
              return true;
            }
            leaf_nodes_out.push_back(std::move(node));
            return false;
          }),
      queue_.end());
}

void Subgraph::push(std::weak_ptr<DeviceBufferList> device_buffer) {
  absl::MutexLock lock(mu_);
  if (queue_.size() >= queue_.capacity()) {
    // Try to free up capacity by pruning down to just the live leaf nodes.
    Prune();
  }
  queue_.push_back(std::move(device_buffer));
}

std::shared_ptr<Subgraph> Subgraph::Find() {
  std::shared_ptr<Subgraph> root = shared_from_this();
  while (true) {
    std::shared_ptr<Subgraph> next_root;
    {
      absl::MutexLock lock(root->mu_);
      next_root = root->parent_;
    }
    if (!next_root) break;
    root = next_root;
  }

  // Path compression: update parents along the path to the root.
  std::shared_ptr<Subgraph> current = shared_from_this();
  while (current != root) {
    std::shared_ptr<Subgraph> next;
    {
      absl::MutexLock lock(current->mu_);
      next = current->parent_;
      if (next) {
        current->parent_ = root;
      }
    }
    if (!next || next == root) break;
    current = next;
  }

  return root;
}

void Subgraph::Merge(std::shared_ptr<Subgraph> s1,
                     std::shared_ptr<Subgraph> s2) {
  auto r1 = s1->Find();
  auto r2 = s2->Find();
  if (r1 == r2) return;

  absl::MutexLock lock1(r1->mu_);
  absl::MutexLock lock2(r2->mu_);

  // Prune r1's queue to avoid reallocation if possible.
  r1->Prune();

  // Push r2's live, leaf nodes onto r1's queue.
  for (auto& weak_node : r2->queue_) {
    if (auto node = weak_node.lock()) {
      const auto* deferred_op = node->deferred_op();
      if (deferred_op && deferred_op->num_child_ops() == 0) {
        r1->queue_.push_back(std::move(weak_node));
      }
    }
  }
  r2->queue_.clear();
  r2->parent_ = r1;
}

std::vector<SharedDeviceBufferList> Subgraph::GetLeafNodes() {
  absl::MutexLock lock(mu_);
  // Simultaneously prune the queue and write leaf nodes to the output.
  std::vector<SharedDeviceBufferList> leaf_nodes;
  PruneAndReturnLeafNodes(leaf_nodes);
  return leaf_nodes;
}

SubgraphRegistry& SubgraphRegistry::GetInstance() {
  static SubgraphRegistry* const registry = new SubgraphRegistry();
  return *registry;
}

absl_nonnull std::shared_ptr<Subgraph> SubgraphRegistry::MakeNewSubgraph() {
  absl::MutexLock lock(mu_);
  if (subgraphs_.size() >= subgraphs_.capacity()) {
    // Try to free up capacity by pruning down to just the live subgraphs.
    subgraphs_.erase(std::remove_if(subgraphs_.begin(), subgraphs_.end(),
                                    [](std::weak_ptr<Subgraph>& weak_subgraph) {
                                      return weak_subgraph.lock() == nullptr;
                                    }),
                     subgraphs_.end());
  }

  auto subgraph = std::make_shared<Subgraph>();
  subgraphs_.push_back(subgraph);  // intentional copy
  return subgraph;
}

absl_nonnull std::shared_ptr<Subgraph> SubgraphRegistry::MergeAll() {
  absl::MutexLock lock(mu_);
  // Find the first non-expired subgraph.
  std::shared_ptr<Subgraph> root;
  auto it = subgraphs_.begin();
  while (it != subgraphs_.end()) {
    root = it->lock();
    if (root) break;
    ++it;
  }
  if (it == subgraphs_.end()) {
    // All subgraphs are expired. Create a new one and register it as the only
    // subgraph.
    subgraphs_.clear();
    auto subgraph = std::make_shared<Subgraph>();
    subgraphs_.push_back(subgraph);  // intentional copy
    return subgraph;
  }

  // Merge all other subgraphs into the root subgraph.
  ++it;
  for (; it != subgraphs_.end(); ++it) {
    if (auto subgraph = it->lock()) {
      Subgraph::Merge(root, subgraph);
    }
  }

  // Make the root subgraph the only subgraph in the registry and return it.
  ABSL_CHECK(root);  // CRASH_OK=satisfying ClangTidy
  subgraphs_.clear();
  subgraphs_.push_back(root);  // intentional copy
  return root;
}

std::shared_ptr<Subgraph> Subgraph::Create() {
  return SubgraphRegistry::GetInstance().MakeNewSubgraph();
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

void DebugMonostate(std::ostream& os, absl::Span<const int64_t> dimensions) {
  bool is_zero_size = false;
  for (int64_t dim : dimensions) {
    if (dim == 0) {
      is_zero_size = true;
      break;
    }
  }
  if (is_zero_size) {
    os << "zero_size";
  } else {
    os << "placeholder";
  }
}

// Logs the data variant state with one of these formats:
//   materialized, pjrt_buffer addr 0x1234567890, on_device_shape: f32[8,16]
//   materialized, pjrt_buffer addr 0x1234567890, deleted
//   deferred, op_name: add
//   zero_size
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
    DebugMonostate(os, dimensions);
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
  // If a DeferredOp would produce only zero-sized tensors, we can avoid ever
  // materializing it.
  // If a DeferredOp produces a mix of zero-sized and non-zero-sized tensors, we
  // only need to execute it if the non-zero-sized tensors are required;
  // this will also produce some zero-sized PjRtBuffers that can be ignored.
  // In either case, we identify zero-sized tensors as kZeroSize (even if a
  // DeferredOp or PjRtBuffer happens to exist for it).
  for (int64_t dim : shapes_[index].dimensions()) {
    if (dim == 0) {
      return DeviceBufferRefState::kZeroSize;
    }
  }
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
  for (auto i = 0; i < size(); ++i) {
    if (state(i) == DeviceBufferRefState::kZeroSize ||
        state(i) == DeviceBufferRefState::kPlaceholder) {
      continue;
    }
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
    subgraph = SubgraphRegistry::GetInstance().MergeAll();
  }

  for (const auto& input : inputs) {
    if (input.state() == DeviceBufferRefState::kDeferred) {
      auto input_subgraph = input.device_buffer_list()->subgraph();
      if (!input_subgraph) continue;

      auto input_rep = input_subgraph->Find();
      if (!subgraph) {
        subgraph = input_rep;
      } else if (subgraph != input_rep) {
        Subgraph::Merge(subgraph, input_rep);
        subgraph = subgraph->Find();
      }
    }
  }

  if (!subgraph) {
    subgraph = Subgraph::Create();
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

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateEmpty(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_ASSIGN_OR_RETURN(auto byte_size,
                      ValidateTensorByteSize(dimensions, element_type));
  if (byte_size == 0) {
    return DeviceBufferList::CreateZeroSize(std::move(dimensions),
                                            element_type);
  }

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
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(dimensions), element_type));
  return DeviceBufferRef(std::move(device_buffer), 0);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::MakePlaceholder(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_ASSIGN_OR_RETURN(auto byte_size,
                      ValidateTensorByteSize(dimensions, element_type));
  if (byte_size == 0) {
    return DeviceBufferList::CreateZeroSize(std::move(dimensions),
                                            element_type);
  }
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
    TT_RET_CHECK(shape.dynamic_dimensions().empty(), error::kInvalidArgument)
        << "only one dynamic dimension is supported per tensor";
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

}  // namespace torch_tpu
