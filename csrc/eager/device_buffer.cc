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

#include "csrc/eager/device_buffer.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
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
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "c10/util/accumulate.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

DeviceBufferList::Data::Data(bool placeholder)
    : placeholder_(placeholder), materialization_pending_(!placeholder) {
  if (materialization_pending_) {
    auto [promise, future] = xla::MakePromise<void>();
    materialization_promise_ = std::move(promise);
    materialization_future_ = std::move(future);
  }
}

DeviceBufferList::Data::Data(
    absl_nonnull std::shared_ptr<DeferredOp> deferred_op)
    : constant_(IsConstantOp(deferred_op->op_name())),
      deferred_op_(std::move(deferred_op)) {
  auto [promise, future] = xla::MakePromise<void>();
  materialization_promise_ = std::move(promise);
  materialization_future_ = std::move(future);
}

DeviceBufferList::Data::Data(
    absl_nonnull std::unique_ptr<xla::PjRtBuffer> buffer)
    : materialization_pending_(true), materialization_started_(true) {
  auto client_or = PjrtBackend::GetInstance().GetSharedClient();
  ABSL_CHECK(client_or.ok()) << client_or.status();  // CRASH_OK
  std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers;
  buffers.push_back(std::move(buffer));
  materialized_buffers_ =
      MaterializedData(std::move(client_or).value(), std::move(buffers));
  auto [promise, future] = xla::MakePromise<void>();
  materialization_promise_ = std::move(promise);
  materialization_future_ = std::move(future);
  materialization_promise_.Set(absl::OkStatus());
}

absl_nullable std::shared_ptr<DeferredOp> DeviceBufferList::Data::deferred_op()
    const {
  if (placeholder_ || materialization_pending_) {
    // DeviceBufferList::Data that is created in the placeholder state never had
    // a DeferredOp. DeviceBufferList::Data that is pending materialization may
    // have had a DeferredOp, but if it did, it has been consumed.
    return nullptr;
  }
  absl::MutexLock lock(deferred_op_mutex_);
  return deferred_op_;
}

absl::Status DeviceBufferList::Data::SetAsPendingMaterialization() {
  TT_RET_CHECK(!placeholder_, error::kFailedPrecondition)
      << "placeholders cannot be materialized";
  // Immediately mark the DeviceBufferList::Data as pending materialization, and
  // check if this was the first time this was called.
  const bool already_pending = materialization_pending_.exchange(true);

  // If the DeviceBufferList::Data was already pending materialization, then
  // we're not responsible for clearing the DeferredOp and don't need to acquire
  // the mutex.
  if (!already_pending) {
    absl::MutexLock lock(deferred_op_mutex_);
    deferred_op_.reset();
  }
  return absl::OkStatus();
}

absl::Status DeviceBufferList::Data::SetMaterializationError(
    absl::Status status) {
  TT_RET_CHECK(!status.ok(), error::kInvalidArgument)
      << "can only set a materialization error with a non-OK status. Got: "
      << status;

  TT_RETURN_IF_ERROR(SetAsPendingMaterialization());

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
  TT_RETURN_IF_ERROR(SetAsPendingMaterialization());

  const bool already_started = materialization_started_.exchange(true);
  TT_RET_CHECK(!already_started, error::kFailedPrecondition)
      << "attempted to set materialized buffers after materialization was "
         "already started";

  TT_ASSIGN_OR_RETURN(auto client,
                      PjrtBackend::GetInstance().GetSharedClient());

  materialized_buffers_ =
      MaterializedData(std::move(client), std::move(buffers));
  materialization_promise_.Set(absl::OkStatus());

  return absl::OkStatus();
}

void DeviceBufferList::Data::ValidateMaterializedBuffers(int64_t index) const {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=internal check
      materialized_buffers_.has_value(), error::kFailedPrecondition)
      << "expected materialized buffers to be populated";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=internal check
      index >= 0 && index < materialized_buffers_->buffers().size(),
      error::kInvalidArgument)
      << "index " << index << " is out of bounds for buffers of size "
      << materialized_buffers_->buffers().size();
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull>
DeviceBufferList::Data::operator[](int64_t index) const {
  TT_RET_CHECK(!placeholder_, error::kFailedPrecondition)
      << "placeholders do not have buffers";

  if (!materialization_future_.IsKnownReady()) {
    TT_RETURN_IF_ERROR(materialization_future_.Await());
  }

  if (!materialization_status_.ok()) {
    return materialization_status_;
  }

  ValidateMaterializedBuffers(index);

  return materialized_buffers_->buffers()[index].get();
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

  // Materialization finished without an error. Safe to access
  // materialized_buffers_->buffers.
  os << "materialized, ready";
  xla::PjRtBuffer* maybe_pjrt_buffer =
      materialized_buffers_->buffers()[0].get();
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
    if (buffer->IsDeleted()) {
      continue;
    }
    auto future = buffer->GetReadyFuture();
    TT_RETURN_IF_ERROR(
        AdaptXlaError(future.Await(),
                      /* context= */ "failed to synchronize device buffer"));
  }
  return absl::OkStatus();
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull> DeviceBufferList::AwaitBuffer(
    int64_t index) const {
  return data_[index];
}

std::atomic_uint64_t DeviceBufferList::g_creation_index_ = 0;

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
    Indices donated_indices) {
  // Validate that the output shapes are valid.
  for (const auto& output_shape : output_shapes) {
    TT_RETURN_IF_ERROR(ValidateTensorByteSize(output_shape.dimensions(),
                                              output_shape.dtype()));
  }
  // Validate that donated input indices are within bounds.
  for (int64_t idx : donated_indices) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal invariant; input indices are
                   // verified at dispatch call sites.
        idx >= 0 && idx < static_cast<int64_t>(inputs.size()), error::kInternal)
        << "donated input index " << idx << " is out of bounds for op "
        << op_name << " with " << inputs.size() << " inputs.";
  }
  int64_t num_outputs = output_shapes.size();

  const auto eager_mode = GetEagerMode();
  const bool is_defer_never = IsDeferNeverMode(eager_mode);

  // Runtime validation for in-place buffer donation in DeferNever mode:
  //
  // In PjRt, buffer donation transfers physical ownership of the underlying
  // device memory buffer to the operation output, consuming and invalidating
  // the input PjRtBuffer.
  //
  // Although the ATen dispatch site requests donation when an op's output
  // aliases its input (e.g., `a.add_(1)`), the dispatch kernel only inspects
  // local arguments and cannot know if other references share the storage.
  // We therefore verify that a valid underlying buffer exists (`dbl !=
  // nullptr`) and has exclusive ownership (`dbl->live_data_ptrs() == 1`) as a
  // final safety check:
  //
  // 1. Storage Aliasing & Views:
  //    Consider the following Python example:
  //      a = torch.randn(1024, device="tpu")
  //      b = a             # b references the same storage
  //      # or:
  //      v = a.view_as(a)  # view sharing the same storage
  //      a.add_(1.0)       # in-place operation on a
  //
  //    If `a` were donated, PjRt would invalidate `a`'s input buffer to reuse
  //    it for the output. While `a`'s storage pointer is updated to the newly
  //    produced buffer, `b` (or `v`) would still point to the old, invalidated
  //    buffer. Subsequent accesses (e.g., `b.cpu()`) would cause
  //    use-after-free, memory corruption, or PjRt runtime crashes. Because
  //    `live_data_ptrs() >= 2` when multiple references or views exist,
  //    requiring `live_data_ptrs() == 1` prevents this.
  //
  // 2. Autograd Saved Tensors (`ctx.save_for_backward`):
  //    If an activation was saved for backward, its storage reference count
  //    is > 1. Skipping donation ensures in-place mutations do not corrupt
  //    the buffer needed for gradient computation.
  //
  // We only donate an input buffer if it exists (`dbl != nullptr`) and
  // satisfies the exclusivity check (`dbl->live_data_ptrs() == 1`), falling
  // back safely to normal allocation otherwise.
  //
  // Note: We bypass this check for custom kernels (Pallas), which manage buffer
  // donation explicitly through their compiler/runtime signatures, and for
  // non-DeferNever modes (where DAG buffer assignment is handled globally).
  if (is_defer_never && !donated_indices.empty() &&
      op_name != OpName::kCustomKernel) {
    Indices safe_donated_indices;
    safe_donated_indices.reserve(donated_indices.size());
    for (int64_t idx : donated_indices) {
      const auto& input_ref = inputs[idx];
      const auto& dbl = input_ref.device_buffer_list();
      if (dbl != nullptr && dbl->live_data_ptrs() == 1) {
        safe_donated_indices.push_back(idx);
      }
    }
    donated_indices = std::move(safe_donated_indices);
  }

  // Create the DeferredOp.
  auto op = std::make_unique<DeferredOp>(
      op_name, std::move(op_builder), std::move(inputs),
      std::move(op_param_cache_keys), output_shapes, split_mode,
      std::move(donated_indices));

  // Wrap the DeferredOp in a DeviceBufferList.
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(std::move(op), std::move(output_shapes)));

  // Construct one DeviceBufferRef for each output.
  std::vector<DeviceBufferRef> device_buffer_refs;
  device_buffer_refs.reserve(num_outputs);
  for (int64_t i = 0; i < num_outputs; ++i) {
    device_buffer_refs.push_back(DeviceBufferRef(device_buffer, i));
  }
  return device_buffer_refs;
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreatePlaceholder(
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(new DeviceBufferList(
      std::move(dimensions), element_type, /*placeholder=*/true));
  return DeviceBufferRef(std::move(device_buffer), 0);
}

absl::StatusOr<DeviceBufferRef> DeviceBufferList::CreatePending(
    const Shape& shape, c10::DeviceIndex device_index,
    c10::StreamId stream_id) {
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(shape.dimensions(), shape.dtype()));
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(
      new DeviceBufferList(shape, /*placeholder=*/false,
                           /*device_index_override=*/device_index,
                           /*stream_id_override=*/stream_id));
  return DeviceBufferRef(std::move(device_buffer), 0);
}

absl::Status DeviceBufferList::SetAsPendingMaterialization() {
  ABSL_VLOG(1)
      << "[SetAsPendingMaterialization] Setting to pending materialization";
  return data_.SetAsPendingMaterialization();
}

namespace {

// Checks that the requested `at_shape` that a deferred op expects is valid for
// the given `buffer_shape` that is on device.
//
// For static dimensions the buffer and op shape must match exactly, for bounded
// dimensions, only check upper bounds. We could get the real on-device shape
// from `buffers[i]->logical_dimensions()`, but this is a device syncing
// operation (blocking), so we want to avoid it.
//
// This function will crash if `at_shape` is not valid for the given
// `buffer_shape`.
void CheckBufferShape(const Shape& at_shape, const xla::Shape& buffer_shape) {
  // Ranks must match
  absl::Span<const int64_t> buffer_dims = buffer_shape.dimensions();
  ABSL_CHECK_EQ(at_shape.dimensions().size(), buffer_dims.size())  // CRASH_OK
      << "Expected the PjRtBuffer to have the same number of dimensions as "
         "this DeviceBufferList ("
      << ToString(at_shape.dimensions().size()) << "), got "
      << ToString(buffer_dims.size()) << ".";

  // Static dims must match and dynamic dims must be LTE the upper bound.
  for (int64_t d = 0; d < at_shape.dimensions().size(); ++d) {
    if (buffer_shape.is_dynamic_dimension(d)) {
      ABSL_CHECK_LE(at_shape.dimensions()[d], buffer_dims[d])  // CRASH_OK
          << "Expected the size of dimension " << d
          << " of the PjRtBuffer (marked as dynamic in this buffer) of shape "
          << ToString(buffer_dims)
          << " to be >= the size of the same dimension of this "
             "DeviceBufferList of shape "
          << ToString(at_shape.dimensions()) << ", got " << buffer_dims[d]
          << ".";
    } else {
      ABSL_CHECK_EQ(at_shape.dimensions()[d], buffer_dims[d])  // CRASH_OK
          << "Expected the size of dimension " << d
          << " of the PjRtBuffer of shape " << ToString(buffer_dims)
          << " to match the size of the same dimension of this "
             "DeviceBufferList of shape "
          << ToString(at_shape.dimensions()) << ", got " << buffer_dims[d]
          << ".";
    }
  }
}

}  // namespace

void DeviceBufferList::CheckMaterialization(
    absl::Span<const absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers)
    const {
  ABSL_CHECK_EQ(shapes_.size(), buffers.size())  // CRASH_OK
      << "Expected " << shapes_.size()
      << " materialized PjRtBuffers for this DeviceBufferList, got "
      << buffers.size() << ".";

  for (size_t i = 0; i < shapes_.size(); ++i) {
    ABSL_CHECK(!buffers[i]->IsDeleted())  // CRASH_OK
        << "Buffer " << i << " is deleted.";

    CheckBufferShape(shapes_[i], buffers[i]->on_device_shape());

    TT_ASSIGN_OR_CRASH(  // CRASH_OK
        mlir::ElementType actual_element_type,
        ConvertTo<mlir::ElementType>(buffers[i]->element_type()),
        _.SetOverride() << "Buffer " << i
                        << " has an unsupported element type: "
                        << xla::primitive_util::LowercasePrimitiveTypeName(
                               buffers[i]->element_type())
                        << ".");

    ABSL_CHECK_EQ(actual_element_type, shapes_[i].dtype())  // CRASH_OK
        << "Expected the PjRtBuffer element type to match the element type of "
           "this DeviceBufferList ("
        << ToString(shapes_[i].dtype()) << ") for buffer " << i << ", got "
        << ToString(actual_element_type) << ".";
  }
}

absl::Status DeviceBufferList::SetAsMaterialized(
    std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers) {
  CheckMaterialization(buffers);
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
  TT_RET_CHECK(index >= 0 && index < shapes_.size(), error::kPythonIndexError)
      << "index " << index << " is out of bounds for DeviceBufferList of size "
      << shapes_.size();
  Shape& shape = shapes_[index];
  TT_RET_CHECK(dimension >= 0 && dimension < shape.dimensions().size(),
               error::kPythonIndexError)
      << "dimension " << dimension << " is out of bounds for tensor of rank "
      << shape.dimensions().size();
  TT_RET_CHECK(shape.dimensions()[dimension] >= lower_bound &&
                   shape.dimensions()[dimension] <= upper_bound,
               error::kPythonIndexError)
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

bool DeviceBufferList::on_device_shape_is_dynamic(int64_t index) const {
  ABSL_CHECK(index >= 0 && index < shapes_.size());  // CRASH_OK
  return shapes_[index].on_device_shape_is_dynamic();
}

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
               error::kPythonIndexError)
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

[[nodiscard]] bool DeviceBufferRef::is_placeholder() const {
  return device_buffer_list_->is_placeholder();
}

[[nodiscard]] bool DeviceBufferRef::is_deferred() const {
  return device_buffer_list_->is_deferred();
}

[[nodiscard]] bool DeviceBufferRef::is_materializing() const {
  return device_buffer_list_->is_materializing();
}

[[nodiscard]] bool DeviceBufferRef::is_executing() const {
  return device_buffer_list_->is_executing();
}

[[nodiscard]] bool DeviceBufferRef::is_materialized() const {
  return device_buffer_list_->is_materialized();
}

[[nodiscard]] bool DeviceBufferRef::is_constant() const {
  return device_buffer_list_->is_constant();
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

xla::Future<void> DeviceBufferRef::GetMaterializationFuture() const {
  return device_buffer_list_->GetMaterializationFuture();
}

void DeviceBufferList::RecordChildOp(uint64_t child_index) const {
  num_child_ops_++;

  // Do the equivalent of std::atomic::fetch_max to set last_child_index_.
  // This is a C++ 26 feature, but we use C++ 20, so we do a manual spinlock.
  uint64_t old_index = last_child_index_.load();
  while (old_index < child_index &&
         !last_child_index_.compare_exchange_weak(old_index, child_index)) {
    // This will loop until either:
    //   - last_child_index_ < child_index with no concurrent updates, at which
    //     point it will update last_child_index_ to child_index, or
    //   - last_child_index_ >= child_index, at which point it will do nothing
    //     as the value has already been set to a higher value.
  }
}

xla::Future<void> DeviceBufferRef::GetReadyFuture() const {
  if (GetMaterializationFuture().IsKnownReady()) {
    auto buffer_or = AwaitBuffer();
    if (!buffer_or.ok()) {
      return xla::Future<void>(buffer_or.status());
    }
    auto ready_future = buffer_or.value()->GetReadyFuture();
    if (ready_future.IsKnownReady()) {
      return ready_future;
    }
    auto [promise, future] = xla::MakePromise<void>();
    ready_future.OnReady(
        [promise = std::move(promise),
         device_buffer_list = device_buffer_list_](
            absl::Status status) mutable { promise.Set(status); });
    return future;
  }

  auto [promise, future] = xla::MakePromise<void>();
  // Capture by value to extend lifetime.
  auto device_buffer_list = device_buffer_list_;
  auto index = index_;
  GetMaterializationFuture().OnReady([promise = std::move(promise),
                                      device_buffer_list =
                                          std::move(device_buffer_list),
                                      index](absl::Status status) mutable {
    if (!status.ok()) {
      promise.Set(status);
      return;
    }
    auto buffer_or = device_buffer_list->AwaitBuffer(index);
    if (!buffer_or.ok()) {
      promise.Set(buffer_or.status());
      return;
    }
    auto ready_future = buffer_or.value()->GetReadyFuture();
    ready_future.OnReady([promise = std::move(promise),
                          device_buffer_list = std::move(device_buffer_list)](
                             absl::Status status) mutable {
      promise.Set(std::move(status));
      device_buffer_list.reset();
    });
  });
  return future;
}

absl::Status DeviceBufferRef::MarkDynamic(int64_t dimension,
                                          int64_t lower_bound,
                                          int64_t upper_bound) const {
  return device_buffer_list()->MarkDynamic(index_, dimension, lower_bound,
                                           upper_bound);
}

bool DeviceBufferRef::on_device_shape_is_dynamic() const {
  return device_buffer_list()->on_device_shape_is_dynamic(index_);
}

absl::Span<const BoundedDynamicDimension> DeviceBufferRef::dynamic_dimensions()
    const {
  return device_buffer_list()->dynamic_dimensions(index_);
}

void DeviceBufferRef::RecordChildOp(uint64_t child_index) const {
  device_buffer_list_->RecordChildOp(child_index);
}

c10::DeviceIndex DeviceBufferRef::device_index() const {
  return device_buffer_list_->device_index();
}

c10::StreamId DeviceBufferRef::stream_id() const {
  return device_buffer_list_->stream_id();
}

FingerprintType Fingerprint(const Shape& shape) {
  return FingerprintCat(shape.dimensions(), static_cast<int>(shape.dtype()));
}

size_t DeferredOp::Hash() const {
  // The numerical value of the op_name_ enum is unstable as enumerators are
  // added to or removed from the enum type, and therefore Hash() is not stable
  // across TorchTPU library versions.
  FingerprintType h = static_cast<size_t>(op_name_);

  for (const auto& input : inputs_) {
    h = FingerprintCat(h, input.shape());
  }

  for (auto index : donated_indices_) {
    h = FingerprintCat(h, index);
  }

  for (const auto& output_shape : output_shapes_) {
    h = FingerprintCat(h, output_shape);
  }

  for (const auto& [key, value] : op_param_cache_keys_) {
    h = FingerprintCat(h, value);
  }

  return static_cast<size_t>(h);
}

}  // namespace torch_tpu
