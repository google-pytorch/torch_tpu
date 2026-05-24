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
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/eager_mode.h"
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

namespace {

bool ShouldPruneNode(const std::shared_ptr<DeviceBufferList>& node) {
  // Ref count is 0, so the node should be pruned.
  if (!node) {
    return true;
  }
  const std::shared_ptr<DeferredOp> absl_nullable deferred_op =
      node->deferred_op();
  bool is_side_effecting_op =
      deferred_op && IsSideEffectingOp(deferred_op->op_name());
  // Side-effecting ops should never be pruned.
  if (is_side_effecting_op) {
    return false;
  }

  if (!deferred_op || node->num_child_ops() > 0) {
    return true;
  }
  return false;
}

}  // namespace

void Subgraph::Prune() {
  queue_.erase(std::remove_if(queue_.begin(), queue_.end(),
                              [](std::weak_ptr<DeviceBufferList>& weak_node) {
                                std::shared_ptr<DeviceBufferList> node =
                                    weak_node.lock();
                                return ShouldPruneNode(node);
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
            if (ShouldPruneNode(node)) {
              return true;
            }
            leaf_nodes_out.push_back(std::move(node));
            return false;
          }),
      queue_.end());
  // At this point we have to clear the unprunable side effects, otherwise we
  // maintain a circular dependency between the Subgraph and the
  // DeviceBufferList.
  unprunable_side_effects_.clear();
}

void Subgraph::push(std::weak_ptr<DeviceBufferList> device_buffer) {
  absl::MutexLock lock(mu_);
  if (queue_.size() >= queue_.capacity()) {
    // Try to free up capacity by pruning down to just the live leaf nodes.
    Prune();
  }
  queue_.push_back(std::move(device_buffer));
}

void Subgraph::AnchorSideEffect(
    std::shared_ptr<DeviceBufferList> device_buffer) {
  absl::MutexLock lock(mu_);
  unprunable_side_effects_.push_back(std::move(device_buffer));
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

  // Swap r1 and r2 based on their addresses to ensure consistent locking order
  // in case two threads try to merge the same two subgraphs in opposite orders.
  const bool swap = r1 > r2;
  if (swap) {
    std::swap(r1, r2);
  }
  absl::MutexLock lock1(r1->mu_);
  absl::MutexLock lock2(r2->mu_);

  // Use non swapped r1 and r2 for the rest of the function to maintain the
  // order of merging requested by the caller.
  if (swap) {
    std::swap(r1, r2);
  }

  // Prune r1's queue to avoid reallocation if possible.
  r1->Prune();

  // Push r2's live, leaf nodes onto r1's queue.
  for (auto& weak_node : r2->queue_) {
    if (auto node = weak_node.lock()) {
      if (node->state() == DeviceBufferRefState::kDeferred &&
          node->num_child_ops() == 0) {
        r1->queue_.push_back(std::move(weak_node));
      }
    }
  }
  r2->queue_.clear();
  r1->unprunable_side_effects_.insert(
      r1->unprunable_side_effects_.end(),
      std::make_move_iterator(r2->unprunable_side_effects_.begin()),
      std::make_move_iterator(r2->unprunable_side_effects_.end()));
  r2->unprunable_side_effects_.clear();
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

DeviceBufferRefState DeviceBufferList::Data::state() const {
  if (materialization_pending_) {
    return DeviceBufferRefState::kMaterialized;
  } else if (created_as_placeholder_) {
    return DeviceBufferRefState::kPlaceholder;
  } else {
    return DeviceBufferRefState::kDeferred;
  }
}

absl_nullable std::shared_ptr<DeferredOp> DeviceBufferList::Data::deferred_op()
    const {
  if (created_as_placeholder_ || materialization_pending_) {
    // DeviceBufferList::Data that is created in the placeholder state never had
    // a DeferredOp. DeviceBufferList::Data that is pending materialization may
    // have had a DeferredOp, but if it did, it has been consumed.
    return nullptr;
  }
  absl::MutexLock lock(deferred_op_mutex_);
  return deferred_op_;
}

void DeviceBufferList::Data::SetMaterializationPending() {
  // Immediately mark the DeviceBufferList::Data as pending materialization, and
  // check if this was the first time this was called.
  const bool already_pending = materialization_pending_.exchange(true);

  // If the DeviceBufferList::Data was already pending materialization, or was
  // created as a placeholder, then we're not responsible for clearing the
  // DeferredOp and don't need to acquire the mutex.
  if (already_pending || created_as_placeholder_) return;

  absl::MutexLock lock(deferred_op_mutex_);
  deferred_op_.reset();
}

absl::Status DeviceBufferList::Data::SetMaterializationError(
    absl::Status status) {
  TT_RET_CHECK(!status.ok(), error::kInvalidArgument)
      << "can only set a materialization error with a non-OK status. Got: "
      << status;

  SetMaterializationPending();

  const bool already_started = materialization_started_.exchange(true);
  TT_RET_CHECK(!already_started, error::kFailedPrecondition)
      << "attempted to set materialization error after materialization was "
         "already started";

  materialization_status_ = status;
  materialization_promise_.Set(status);

  return absl::OkStatus();
}

absl::Status DeviceBufferList::Data::SetMaterializationStarted(
    std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers) {
  SetMaterializationPending();

  const bool already_started = materialization_started_.exchange(true);
  TT_RET_CHECK(!already_started, error::kFailedPrecondition)
      << "attempted to set materialized buffers after materialization was "
         "already started";

  buffers_ = std::move(buffers);
  materialization_promise_.Set(absl::OkStatus());

  return absl::OkStatus();
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull>
DeviceBufferList::Data::operator[](int64_t index) const {
  if (!materialization_future_.IsKnownReady()) {
    TT_RETURN_IF_ERROR(materialization_future_.Await());
  }

  if (!materialization_status_.ok()) {
    return materialization_status_;
  }

  TT_RET_CHECK(index >= 0 && index < buffers_.size(), error::kInvalidArgument)
      << "index " << index << " is out of bounds for buffers of size "
      << buffers_.size();
  return buffers_[index].get();
}

std::ostream& DeviceBufferList::Data::PrintDebug(std::ostream& os) const {
  // The order of these checks is important.
  // If we get a DeferredOp, then we know that it's in the deferred state.
  // Otherwise, it could be a placeholder or materialized.
  if (const auto maybe_deferred_op = deferred_op();
      maybe_deferred_op != nullptr) {
    return os << "deferred, op_name: " << maybe_deferred_op->op_name();
  }
  if (!materialization_pending_) {
    // If it's not deferred and not pending materialization, it must be a
    // placeholder.
    return os << "placeholder";
  }
  if (!materialization_future_.IsReady()) {
    // If it hasn't finished materialization, then it's pending.
    return os << "materialized, pending";
  }
  if (!materialization_status_.ok()) {
    // If materialization finished with an error, then report the error.
    return os << "materialized, error: " << materialization_status_;
  }

  // Materialization finished without an error. Safe to access buffers_.
  os << "materialized, ready";
  xla::PjRtBuffer* maybe_pjrt_buffer = buffers_[0].get();
  if (maybe_pjrt_buffer == nullptr) {
    return os << ", null";
  }
  const xla::PjRtBuffer* pjrt_buffer = maybe_pjrt_buffer;
  if (pjrt_buffer->IsDeleted()) {
    return os << ", deleted";
  }
  return os << ", on_device_shape: "
            << pjrt_buffer->on_device_shape().ToString(true);
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
  TT_ASSIGN_OR_RETURN(xla::PjRtBuffer* const pjrt_buffer, AwaitBuffer(index));

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

std::string DeviceBufferList::DebugString() const {
  std::ostringstream os;
  os << "DeviceBufferList:"
     << "\n\tAddress: " << this << "\n\tNum buffers: " << size() << "\n\t";
  data_.PrintDebug(os);
  return os.str();
}

std::ostream& DeviceBufferList::DebugData(std::ostream& os) const {
  return data_.PrintDebug(os);
}

std::string DeviceBufferRef::DebugString() const {
  std::ostringstream os;
  os << "DeviceBufferRef:"
     << "\n\tAddress of ref: " << this
     << "\n\tAddress of DeviceBufferList: " << device_buffer_list_.get()
     << "\n\tIndex: " << index_
     << "\n\tShape and type: " << ToString(element_type())
     << ToString(dimensions()) << "\n\tData state: ";
  device_buffer_list_->DebugData(os);
  return os.str();
}

DeviceBufferRefState DeviceBufferList::state() const { return data_.state(); }

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
    TT_ASSIGN_OR_RETURN(auto* buffer, AwaitBuffer(i));
    auto future = buffer->GetReadyFuture();
    TT_RETURN_IF_ERROR(future.Await());
  }
  return absl::OkStatus();
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull> DeviceBufferList::AwaitBuffer(
    int64_t index) const {
  return data_[index];
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

absl::StatusOr<std::vector<DeviceBufferRef>> DeviceBufferList::CreateDeferred(
    OpName op_name, MlirOpBuilder op_builder,
    std::vector<DeviceBufferRef> inputs, OpParamCacheKeys op_param_cache_keys,
    std::vector<Shape> output_shapes, OpSplitMode split_mode,
    Indices donated_indices, bool skip_subgraph) {
  // Validate that the output shapes are valid.
  for (const auto& output_shape : output_shapes) {
    TT_RETURN_IF_ERROR(ValidateTensorByteSize(output_shape.dimensions(),
                                              output_shape.dtype()));
  }
  int64_t num_outputs = output_shapes.size();

  std::shared_ptr<Subgraph> subgraph = nullptr;

  if (!skip_subgraph) {
    if (IsDistributedOp(op_name)) {
      // Each distributed op acts as a barrier; all prior operations (connected
      // or not) that were created before it must be considered part of the same
      // graph so that proper ordering is maintained.
      // Otherwise, two independent collective operations could be isolated in
      // disconnected subgraphs, and different rank processes could have
      // different orderings of these subgraphs, leading to a deadlock.
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
  }

  // Create the DeferredOp.
  auto op = std::make_unique<DeferredOp>(
      op_name, std::move(op_builder), std::move(inputs),
      std::move(op_param_cache_keys), output_shapes, subgraph, split_mode,
      std::move(donated_indices));

  // Wrap the DeferredOp in a DeviceBufferList.
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(op), std::move(output_shapes), subgraph));

  if (subgraph) {
    subgraph->push(std::weak_ptr<DeviceBufferList>(device_buffer));
    if (IsSideEffectingOp(op_name) &&
        GetEagerMode() != EagerMode::kInternalDeferAll) {
      subgraph->AnchorSideEffect(device_buffer);
    }
  }

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
    mlir::ElementType element_type, bool skip_subgraph) {
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
  TT_ASSIGN_OR_RETURN(auto results,
                      DeviceBufferList::CreateDeferred(
                          op_name, std::move(op_builder), /*inputs=*/{},
                          std::move(op_param_cache_keys),
                          std::move(output_shapes), OpSplitMode::kNone,
                          /*donated_indices=*/{}, skip_subgraph));
  TT_RET_CHECK(results.size() == 1, error::kInternal)
      << "CreateConstant should return exactly one output";
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateEmpty(
    Dimensions dimensions, mlir::ElementType element_type, bool skip_subgraph) {
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
  TT_ASSIGN_OR_RETURN(auto results,
                      DeviceBufferList::CreateDeferred(
                          OpName::kEmpty, std::move(op_builder), /*inputs=*/{},
                          OpParamCacheKeys::Empty(), {std::move(output_shape)},
                          OpSplitMode::kNone,
                          /*donated_indices=*/{}, skip_subgraph));
  ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
  return std::move(results[0]);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreateZeroSize(
    Dimensions dimensions, mlir::ElementType element_type, bool skip_subgraph) {
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
  return CreateConstant({}, std::move(dimensions), element_type, skip_subgraph);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::MakePlaceholder(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(dimensions), element_type));
  return DeviceBufferRef(std::move(device_buffer), 0);
}

void DeviceBufferList::SetMaterializationPending() {
  ABSL_VLOG(1)
      << "[SetMaterializationPending] Setting to pending materialization";
  data_.SetMaterializationPending();
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

absl::Status DeviceBufferList::VerifyMaterialization(
    absl::Span<const absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers)
    const {
  TT_RET_CHECK(shapes_.size() == buffers.size(), error::kInvalidArgument)
      << "unexpected number of buffers; expected: " << shapes_.size()
      << " but got: " << buffers.size();
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
  return absl::OkStatus();
}

absl::Status DeviceBufferList::SetAsMaterialized(
    std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers) {
  if (auto status = VerifyMaterialization(buffers); !status.ok()) {
    TT_RETURN_IF_ERROR(data_.SetMaterializationError(status));
    return status;
  }
  ABSL_VLOG(1) << "[SetAsMaterialized] Setting as materialized";
  return data_.SetMaterializationStarted(std::move(buffers));
}

void DeviceBufferList::SetAsError(absl::Status error) {
  auto set_error_status = data_.SetMaterializationError(error);
  if (!set_error_status.ok()) {
    ABSL_LOG(ERROR) << "[SetAsError] Failed to set materialization error: "
                    << set_error_status;
  }
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
  return device_buffer_list_->state();
}

bool DeviceBufferRef::IsMaterialized() const {
  return state() == DeviceBufferRefState::kMaterialized;
}

[[nodiscard]] const Shape& DeviceBufferRef::shape() const {
  return device_buffer_list_->shapes()[index_];
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

[[nodiscard]] absl_nullable std::shared_ptr<DeferredOp>
DeviceBufferRef::deferred_op() const {
  return device_buffer_list_->deferred_op();
}

absl::Status DeviceBufferRef::Synchronize() const {
  return device_buffer_list_->Synchronize();
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull> DeviceBufferRef::AwaitBuffer()
    const {
  return device_buffer_list_->AwaitBuffer(index_);
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

template <>
inline void HashCombine<Shape>(std::size_t& h, const Shape& shape) {
  HashCombine(h, shape.dimensions().size());
  for (auto dim : shape.dimensions()) {
    HashCombine(h, dim);
  }
  HashCombine(h, static_cast<size_t>(shape.dtype()));
}

size_t DeferredOp::Hash() const {
  auto h = static_cast<size_t>(op_name_);

  HashCombine(h, inputs_.size());
  for (const auto& input : inputs_) {
    HashCombine(h, input.shape());
  }

  HashCombine(h, donated_indices_.size());
  for (auto index : donated_indices_) {
    HashCombine(h, index);
  }

  HashCombine(h, output_shapes_.size());
  for (const auto& output_shape : output_shapes_) {
    HashCombine(h, output_shape);
  }

  HashCombine(h, op_param_cache_keys_.size());
  for (const auto& [key, value] : op_param_cache_keys_) {
    HashCombine(h, value);
  }

  return h;
}

}  // namespace torch_tpu
