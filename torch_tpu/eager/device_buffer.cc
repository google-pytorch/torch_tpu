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
#include "c10/util/accumulate.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/pjrt/pjrt_state.h"
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
    : empty_(IsEmptyOp(deferred_op->op_name())),
      rng_state_update_(IsRngStateUpdateOp(deferred_op->op_name())),
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

[[nodiscard]] bool DeviceBufferList::Data::depends_on_placeholder() const {
  if (placeholder_ || materialization_pending_) {
    return false;
  }
  absl::MutexLock lock(deferred_op_mutex_);
  if (!deferred_op_) {
    return false;
  }
  return deferred_op_->depends_on_placeholder();
}

absl::Status DeviceBufferList::Data::SetAsPendingMaterialization() {
  TT_RET_CHECK(!placeholder_, error::kFailedPrecondition)
      << "placeholders cannot be materialized";
  TT_RET_CHECK(!depends_on_placeholder(), error::kFailedPrecondition)
      << "compiled mode tensors which depend on a placeholder cannot be "
         "materialized";
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
  TT_RET_CHECK(!depends_on_placeholder(), error::kFailedPrecondition)
      << "compiled mode tensors which depend on a placeholder cannot be "
         "materialized and will never have PjRtBuffers";

  if (auto op = deferred_op(); op != nullptr) {
    ABSL_LOG(WARNING)
        << "Accessing the PjRtBuffer of a deferred DeviceBufferList (op: "
        << op->op_name()
        << "), i.e. a buffer whose materialization has not yet started. This "
           "thread will wait until the materialization process (which should "
           "be triggered by another thread) finishes. If this is unexpected, "
           "you should call MaybeMaterialize() on the corresponding "
           "buffer/tensor.";
  }

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
    auto future = buffer->GetReadyFuture();
    TT_RETURN_IF_ERROR(AdaptXlaError(future.Await()));
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
  int64_t num_outputs = output_shapes.size();

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
    Dimensions dimensions, mlir::ElementType element_type) {
  TT_RETURN_IF_ERROR(ValidateTensorByteSize(dimensions, element_type));
  // Can't use make_shared because the constructor is private.
  auto device_buffer = std::shared_ptr<DeviceBufferList>(new DeviceBufferList(
      std::move(dimensions), element_type, /*placeholder=*/false));
  return DeviceBufferRef(std::move(device_buffer), 0);
}

absl::Status DeviceBufferList::SetAsPendingMaterialization() {
  ABSL_VLOG(1)
      << "[SetAsPendingMaterialization] Setting to pending materialization";
  return data_.SetAsPendingMaterialization();
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

[[nodiscard]] bool DeviceBufferRef::depends_on_placeholder() const {
  return device_buffer_list_->depends_on_placeholder();
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

[[nodiscard]] bool DeviceBufferRef::is_empty() const {
  return device_buffer_list_->is_empty();
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
                                      device_buffer_list,
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
                          device_buffer_list](absl::Status status) mutable {
      promise.Set(status);
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

absl::Span<const BoundedDynamicDimension> DeviceBufferRef::dynamic_dimensions()
    const {
  return device_buffer_list()->dynamic_dimensions(index_);
}

void DeviceBufferRef::RecordChildOp(uint64_t child_index) const {
  device_buffer_list_->RecordChildOp(child_index);
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
