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

#include "torch_tpu/distributed/process_group_tpu.h"

#include <atomic>
#include <chrono>  // NOLINT - needed for PyTorch API
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/ivalue_inl.h"
#include "ATen/core/jit_type.h"
#include "ATen/ops/cat.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_like.h"
#include "ATen/ops/flatten.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/stack.h"
#include "ATen/ops/zeros.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Store.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch/headeronly/core/DeviceType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/distributed/allgather.h"
#include "torch_tpu/distributed/allreduce.h"
#include "torch_tpu/distributed/alltoall.h"
#include "torch_tpu/distributed/reduce_scatter.h"
#include "torch_tpu/distributed/types.h"
#include "torch_tpu/distributed/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"

ABSL_FLAG(bool, torch_tpu_internal_materialize_collective_tensors, true,
          "Split the execution graph before and after collectives.");

namespace torch_tpu {

namespace {

class TpuWork : public c10d::Work {
 public:
  explicit TpuWork(std::vector<at::Tensor> result_tensors, int rank = -1,
                   c10d::OpType op_type = c10d::OpType::UNKNOWN,
                   c10::intrusive_ptr<c10::ivalue::Future> future = nullptr)
      : Work(rank, op_type), results_(std::move(result_tensors)) {
    results_future_ = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));

    // If the future is null, mark the results future as completed. Otherwise,
    // add a callback to mark the results future as completed when the future
    // is completed.
    if (future) {
      future->addCallback([this](c10::ivalue::Future& future) {
        if (future.hasError()) {
          results_future_->setError(std::make_exception_ptr(TtError(
              TT_ERROR(error::kInternal) << future.tryRetrieveErrorMessage(),
              TT_SOURCE_LOCATION)));
        } else {
          results_future_->markCompleted(c10::IValue(results_));
        }
      });
    } else {
      results_future_->markCompleted(c10::IValue(results_));
    }
  }

  // This class is neither copyable nor movable.
  TpuWork(const TpuWork&) = delete;
  TpuWork& operator=(const TpuWork&) = delete;
  TpuWork(TpuWork&&) = delete;
  TpuWork& operator=(TpuWork&&) = delete;

  // NOTE: We implement TpuWork to match the PyTorch API surface.
  bool isCompleted() override { return results_future_->completed(); }

  bool isSuccess() const override {
    return results_future_->completed() && !results_future_->hasError();
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    results_future_->wait();
    TT_CHECK_THROW(!results_future_->hasError(), error::kInternal)
        << results_future_->tryRetrieveErrorMessage();
    return true;
  }

  std::vector<at::Tensor> result() override { return results_; }

  void abort() override {}

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    return results_future_;
  }

 private:
  std::vector<at::Tensor> results_;
  c10::intrusive_ptr<c10::ivalue::Future> results_future_;

  friend class ::torch_tpu::ProcessGroupTpu;  // To allow changing op type.
};

std::string DeviceBufferRefDebugString(const at::Tensor& input) {
  auto maybe_base_buffer_ref = GetBaseBuffer(input);
  if (!maybe_base_buffer_ref.ok()) {
    return std::string(maybe_base_buffer_ref.status().message());
  }
  return maybe_base_buffer_ref->DebugString();
}

absl::Status CheckTensorsUniformShape(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    return absl::OkStatus();
  }

  const auto& first_tensor_size = tensors[0].sizes();
  for (int i = 1; i < tensors.size(); ++i) {
    TT_RET_CHECK(tensors[i].sizes() == first_tensor_size,
                 error::kInvalidArgument)
        << "tensors in the list must have the same shape, got "
        << first_tensor_size << " at index 0 and " << tensors[i].sizes()
        << " at index " << i;
  }

  return absl::OkStatus();
}

enum class ReduceScatterShapeMode {
  kConcat,  // Input tensor is concatenated along dimension 0
  kStack,   // Input tensor is stacked along a new dimension (at position 0)
};

// Helper to validate the reduce-scatter input tensor shape, and determine if
// input tensor is concatenated or stacked (PyTorch API allows both).
absl::StatusOr<ReduceScatterShapeMode> CheckReduceScatterInputTensorShape(
    at::Tensor& output, at::Tensor& input, size_t process_group_size) {
  // Concat mode: same shape as output, but dim=0 is pg_size times larger:
  Dimensions concat_mode_dims;
  if (output.dim() == 0) {
    concat_mode_dims = {static_cast<int64_t>(process_group_size)};
  } else {
    concat_mode_dims = CopyIntVector(output.sizes());
    concat_mode_dims[0] *= process_group_size;
  }
  ABSL_VLOG(1) << "CheckReduceScatterInputTensorShape: concat_mode_dims: "
               << ToString(concat_mode_dims);

  // Stack mode: prepend new dimension of size pg_size, and remaining dims same.
  Dimensions stack_mode_dims(output.dim() + 1);
  stack_mode_dims[0] = process_group_size;
  for (int i = 0; i < output.dim(); ++i) {
    stack_mode_dims[i + 1] = output.size(i);
  }
  ABSL_VLOG(1) << "CheckReduceScatterInputTensorShape: stack_mode_dims: "
               << ToString(stack_mode_dims);

  Dimensions actual_input_dims{input.sizes().begin(), input.sizes().end()};
  ABSL_VLOG(1) << "CheckReduceScatterInputTensorShape: actual_input_dims: "
               << ToString(actual_input_dims);

  if (actual_input_dims == concat_mode_dims) {
    return ReduceScatterShapeMode::kConcat;
  } else if (actual_input_dims == stack_mode_dims) {
    return ReduceScatterShapeMode::kStack;
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "input tensor shape must be either " << ToString(concat_mode_dims)
           << " or " << ToString(stack_mode_dims) << ", but got "
           << ToString(actual_input_dims);
  }
}

absl::Status CheckSplitSizesForAllToAllSingle(
    const std::vector<int64_t>& split_sizes,  // INT_VEC_OK
    const at::Tensor& tensor, size_t group_size) {
  auto dim0 = tensor.sizes()[0];
  if (split_sizes.empty()) {
    TT_RET_CHECK(dim0 % group_size == 0, error::kInvalidArgument)
        << "tensor first dimension must be divisible by process group "
        << "size, got " << group_size << " for process group size" << " and "
        << dim0 << " for tensor shape " << tensor.sizes() << " dim 0";
  } else {
    TT_RET_CHECK(split_sizes.size() == group_size, error::kInvalidArgument)
        << "split sizes must have the same size as process group size, got "
        << group_size << " for process group size and " << split_sizes.size()
        << " for split sizes [" << split_sizes << "]";

    const int64_t split_sizes_sum = absl::c_accumulate(split_sizes, 0L);
    TT_RET_CHECK(split_sizes_sum == dim0, error::kInvalidArgument)
        << "split sizes sum must be equal to tensor first dimension, got "
        << split_sizes_sum << " for split sizes [" << split_sizes << "] and "
        << dim0 << " for tensor shape " << tensor.sizes() << " dim 0";
  }
  return absl::OkStatus();
}

bool IsEqualSplits(const std::vector<int64_t>& split_sizes) {  // INT_VEC_OK
  if (split_sizes.empty()) {
    return true;
  }
  const int64_t first_size = split_sizes[0];
  for (size_t i = 1; i < split_sizes.size(); ++i) {
    if (split_sizes[i] != first_size) {
      return false;
    }
  }
  return true;
}

// All input and output tensors must have the same shape, and the number of
// input tensors must be the same as the number of output tensors.
absl::Status CheckAllToAllShapeConsistency(
    const std::vector<at::Tensor>& output_tensors,             // INT_VEC_OK
    const std::vector<at::Tensor>& input_tensors) {            // INT_VEC_OK
  ABSL_CHECK_EQ(output_tensors.size(), input_tensors.size());  // CRASH_OK
  ABSL_CHECK_GT(input_tensors.size(), 0);                      // CRASH_OK

  auto first_tensor_sizes = input_tensors[0].sizes();
  for (int64_t i = 0; i < output_tensors.size(); ++i) {
    auto output_tensor_sizes = output_tensors[i].sizes();
    auto input_tensor_sizes = input_tensors[i].sizes();
    TT_RET_CHECK(first_tensor_sizes == input_tensor_sizes,
                 error::kInvalidArgument)
        << "all input tensors must be of same shape, got " << input_tensor_sizes
        << " at index " << i << " and " << first_tensor_sizes << " at index 0";
    TT_RET_CHECK(output_tensor_sizes == input_tensor_sizes,
                 error::kInvalidArgument)
        << "output and input tensors must have the same shape, got "
        << output_tensor_sizes << " for output and " << input_tensor_sizes
        << " for input at index " << i;
  }
  return absl::OkStatus();
}

// Returns the OpSplitMode to use for collectives.
OpSplitMode GetCollectiveSplitMode() {
  if (IsSpmdSafe()) {
    return OpSplitMode::kNone;
  }
  if (GetFlagOnce<bool,
                  &FLAGS_torch_tpu_internal_materialize_collective_tensors>()) {
    return OpSplitMode::kSplitBoth;
  }
  return OpSplitMode::kNone;
}

xla::CrossHostTransferKey GetCrossHostTransferKey(int64_t src_device_id,
                                                  int64_t dst_device_id,
                                                  int tag,
                                                  size_t tensor_index = 0) {
  uint64_t seed = 0;
  torch_tpu::HashCombine(seed, static_cast<uint64_t>(src_device_id));
  torch_tpu::HashCombine(seed, static_cast<uint64_t>(dst_device_id));
  torch_tpu::HashCombine(seed, static_cast<uint64_t>(tag));
  torch_tpu::HashCombine(seed, static_cast<uint64_t>(tensor_index));
  return xla::CrossHostTransferKey(static_cast<int64_t>(seed));
}

std::string GetCrossHostTransferDescriptorStoreKey(
    xla::CrossHostTransferKey key) {
  static constexpr std::string_view kKeyPrefix =
      "cross_host_transfer_descriptor:";
  return absl::StrCat(kKeyPrefix, key.value());
}

}  // namespace

absl::StatusOr<c10::intrusive_ptr<c10::ivalue::Future>>
ProcessGroupTpu::CrossHostSendBuffers(
    std::vector<xla::PjRtBuffer*>& buffers,
    absl::Span<const xla::CrossHostTransferKey> transfer_keys) {
  TT_RET_CHECK(transfer_keys.size() == buffers.size(), error::kInvalidArgument)
      << "expected transfer keys to be the same size as buffers, got "
      << transfer_keys.size() << " transfer keys and " << buffers.size()
      << " buffers";

  auto all_sent_future =
      c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

  if (buffers.empty()) {
    all_sent_future->markCompleted(c10::IValue());
    return all_sent_future;
  }

  // Map transfer keys to their corresponding store keys to retrieve the
  // serialized descriptors generated by the remote host.
  std::vector<std::string> descriptor_keys;
  descriptor_keys.reserve(transfer_keys.size());
  for (const auto& transfer_key : transfer_keys) {
    descriptor_keys.push_back(
        GetCrossHostTransferDescriptorStoreKey(transfer_key));
  }

  // Retrieve descriptors from the distributed store. This is a blocking call
  // that waits until the remote receiver has posted its descriptors.
  std::vector<std::vector<uint8_t>> descriptors;
  try {
    descriptors = store_->multiGet(descriptor_keys);
  } catch (const std::exception& e) {
    all_sent_future->setError(std::current_exception());
  }

  auto remaining_sends = std::make_shared<std::atomic<size_t>>(buffers.size());
  for (size_t i = 0; i < descriptors.size(); ++i) {
    const std::string descriptor(descriptors[i].begin(), descriptors[i].end());

    buffers[i]->CopyToRemoteDevice(
        std::move(descriptor),
        [all_sent_future, remaining_sends](absl::Status status,
                                           bool sends_were_enqueued) {
          // If the aggregate future has already been resolved (likely due to a
          // failure in a parallel callback), exit immediately.
          if (all_sent_future->completed()) {
            return;
          }

          if (!status.ok()) {
            all_sent_future->setError(std::make_exception_ptr(
                TtError(TT_ERROR(error::kInternal)
                            << "PjRtBuffer::CopyToRemoteDevice: " << status,
                        TT_SOURCE_LOCATION)));
          } else if (!sends_were_enqueued) {
            all_sent_future->setError(std::make_exception_ptr(TtError(
                TT_ERROR(error::kInternal) << "PjRtBuffer::CopyToRemoteDevice: "
                                              "sends were not enqueued",
                TT_SOURCE_LOCATION)));
          } else if (remaining_sends->fetch_sub(1) == 1) {
            // fetch_sub returns the value before subtraction. If it was 1,
            // this is the final transfer to complete successfully.
            all_sent_future->markCompleted(c10::IValue());
          }
        });
  }
  return all_sent_future;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
ProcessGroupTpu::CrossHostReceiveBuffers(
    absl::Span<const xla::Shape> shapes,
    absl::Span<const xla::CrossHostTransferKey> transfer_keys) {
  auto setup_future =
      c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

  auto notifier =
      [this, setup_future,
       transfer_keys = std::vector<xla::CrossHostTransferKey>(
           transfer_keys.begin(), transfer_keys.end())](
          absl::StatusOr<xla::PjRtCrossHostRecvState> recv_state) -> void {
    if (!recv_state.ok()) {
      setup_future->setError(std::make_exception_ptr(TtError(
          TT_ERROR(error::kInternal)
              << "xla::PjRtClient::MakeCrossHostReceiveBuffers: invalid "
                 "recv_state: "
              << recv_state.status(),
          TT_SOURCE_LOCATION)));
      return;
    }

    if (recv_state->descriptors.size() != transfer_keys.size()) {
      setup_future->setError(std::make_exception_ptr(TtError(
          TT_ERROR(error::kInternal)
              << "xla::PjRtClient::MakeCrossHostReceiveBuffers: expected "
                 "descriptors to be the same size as transfer keys, got "
              << recv_state->descriptors.size() << " descriptors and "
              << transfer_keys.size() << " keys",
          TT_SOURCE_LOCATION)));
      return;
    }

    // Collect the serialized descriptors and push them to the distributed
    // store. This act signals the sender that the receiver is ready.
    std::vector<std::string> descriptor_keys;
    std::vector<std::vector<uint8_t>> descriptors;
    descriptor_keys.reserve(transfer_keys.size());
    descriptors.reserve(transfer_keys.size());

    for (int i = 0; i < transfer_keys.size(); ++i) {
      descriptor_keys.push_back(
          GetCrossHostTransferDescriptorStoreKey(transfer_keys[i]));

      std::string_view descriptor =
          recv_state->descriptors[i].serialized_descriptors.front();
      descriptors.push_back(
          std::vector<uint8_t>(descriptor.begin(), descriptor.end()));
    }

    // Update the store to unblock the sender's multiGet call.
    try {
      this->store_->multiSet(descriptor_keys, descriptors);
      setup_future->markCompleted(c10::IValue());
    } catch (const std::exception& e) {
      setup_future->setError(std::current_exception());
    }
  };

  // Initiate the receive process on the PJRT client. The resulting buffers
  // will be populated once the remote sender initiates its transfer.
  TT_ASSIGN_OR_RETURN(
      auto buffers,
      PjrtBackend::GetInstance().GetClient()->MakeCrossHostReceiveBuffers(
          shapes, PjrtBackend::GetInstance().GetDevice(), std::move(notifier)));

  // Wait for the buffer to be ready to transfer and descriptors to be
  // posted to the store. This ensures the enqueueing order of send and receive
  // can be maintained from the caller's perspective.
  setup_future->waitAndThrow();
  return buffers;
}

ProcessGroupTpu::ProcessGroupTpu(c10::intrusive_ptr<c10d::Store> store,
                                 int rank, int group_size)
    : c10d::Backend(rank, group_size), store_(std::move(store)) {
  ABSL_VLOG(1) << "[ProcessGroupTpu] ctor: "
               << "rank: " << rank << ", group size: " << group_size;

  const auto* const pjrt_client = PjrtBackend::GetInstance().GetClient();
  TT_CHECK_THROW(pjrt_client != nullptr, error::kInternal)
      << "PjRtClient is not initialized.";

  // All devices in the slice:
  for (const auto* dev : pjrt_client->devices()) {
    ABSL_CHECK(dev != nullptr) << "Got a nullptr PjRtDevice.";  // CRASH_OK
    device_ids_.push_back(dev->global_device_id().value());
  }

  // Addressable device "assigned" to this process/rank:
  TT_CHECK_THROW(pjrt_client && pjrt_client->addressable_devices().size() == 1,
                 error::kInternal)
      << "expected exactly one TPU device per PyTorch process.";
  addressable_device_id_ =
      pjrt_client->addressable_devices()[0]->global_device_id().value();

  // We support creating "flat" process groups, as well as "2D+ mesh" process
  // (sub)groups. The (sub)group size must evenly divide the world size, i.e.
  // the total number of devices.
  TT_CHECK_THROW(device_ids_.size() % group_size == 0, error::kInternal)
      << "TPU slice size (" << device_ids_.size() << ") must be divisible by "
      << "process group size (" << group_size << ")";

  // Use the distributed KV-store (c10d::Store) to exchange rank-to-device-id
  // mappings. The end-result is that every PyTorch rank/process within the
  // same group sees the same device ids membership, and in the same order.
  store_->set(absl::StrFormat("rank_to_dev_id:%d", rank),
              std::to_string(addressable_device_id_));

  std::vector<std::string> peer_keys;
  peer_keys.reserve(group_size);
  for (int64_t peer_rank = 0; peer_rank < group_size; ++peer_rank) {
    peer_keys.push_back(absl::StrFormat("rank_to_dev_id:%d", peer_rank));
  }
  std::vector<std::vector<uint8_t>> peer_values = store_->multiGet(peer_keys);

  rank_to_device_id_.resize(group_size);
  for (int64_t peer_rank = 0; peer_rank < group_size; ++peer_rank) {
    const auto& peer_value = peer_values[peer_rank];
    const std::string value_str(peer_value.begin(), peer_value.end());
    int64_t peer_dev_id;
    TT_CHECK_THROW(absl::SimpleAtoi(value_str, &peer_dev_id), error::kInternal)
        << "cannot parse device id value: " << value_str;
    rank_to_device_id_[peer_rank] = peer_dev_id;
  }
  ABSL_CHECK_EQ(rank_to_device_id_[rank], addressable_device_id_)  // CRASH_OK
      << "Inconsistent rank-to-device-id mapping. This is a bug.";

  ABSL_VLOG(1) << "[ProcessGroupTpu] Rank-to-device-id mapping for "
               << "rank=" << rank << ", group_size=" << group_size << ": "
               << ToString(rank_to_device_id_);

  if (group_size == device_ids_.size()) {
    // This new group includes all the devices, so the subgroup list is trivial:
    subgroup_device_ids_.push_back(rank_to_device_id_);
  } else {
    // This new group is a proper subgroup. The `c10d::Store` allows us to
    // exchange membership information *within* this subgroup, which is
    // sufficient for PyTorch.
    // However, in case of XLA, we need every process to know about *all* the
    // subgroups, including ones it isn't a member of.
    subgroup_device_ids_ = GatherAllSubgroups();
  }

  ABSL_VLOG(1) << "[ProcessGroupTpu] rank: " << getRank()
               << ", group size: " << getSize()
               << ", device id: " << addressable_device_id_
               << ", group device ids: " << ToString(rank_to_device_id_)
               << ", all subgroups: " << ToString(subgroup_device_ids_);
}

// Sets the process group ID in the given param cache keys.
#define TT_SET_PROCESS_GROUP_ID(param_keys) \
  TT_THROW_IF_ERROR(param_keys.SetParam("pg_id", pg_id_))

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  TT_KERNEL(OpName::kDistributedAllReduce, param_keys, (tensors, opts), {
    TT_SET_PROCESS_GROUP_ID(param_keys);
    // TODO(vladbelous): Implement support for multiple input/output
    // tensors:
    TT_CHECK_THROW(tensors.size() == 1, error::kUnimplemented)
        << "does not yet support multiple tensors.";

    at::Tensor& tensor = tensors[0];

    TT_THROW_IF_ERROR(ValidateReductionOp(opts.reduceOp, tensor.scalar_type()));

    ABSL_VLOG(1) << "[ProcessGroupTpu::allreduce] Rank: " << getRank()
                 << ", DeviceBufferRef: " << DeviceBufferRefDebugString(tensor);

    auto& maybe_materialized_input_tensor = tensor;

    auto op_builder = [subgroups = subgroup_device_ids_,
                       reduce_op = opts.reduceOp](mlir::MlirOp input) {
      return BuildDistributedAllReduceShlo(input, reduce_op, subgroups);
    };

    TT_ASSIGN_OR_THROW(auto output_dtype,
                       ConvertTo<mlir::ElementType>(tensor.scalar_type()));

    TT_ASSIGN_OR_THROW(
        auto result_buffer,
        DispatchOp<1>(std::move(op_builder), maybe_materialized_input_tensor,
                      {.out_dtype = output_dtype,
                       .out_dims = tensor.sizes(),
                       .op_param_cache_keys = std::move(param_keys),
                       .split_mode = GetCollectiveSplitMode()}));
    // TODO: respect async.

    // All-reduce is inplace, assign output buffer ref to input tensors.
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffer), tensor));

    // NOTE: from distributed/c10d/ProcessGroupNCCL.cpp;l=3604 in
    // pytorch/pytorch if not async should return nullptr.
    if (opts.asyncOp) {
      return c10::make_intrusive<TpuWork>(tensors, getRank(),
                                          c10d::OpType::ALLREDUCE);
    }
    return nullptr;
  });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  TT_KERNEL(OpName::kDistributedBroadcast, _,
            (tensors, IgnoreInCacheKey(opts, "Legacy usage")), {
              auto src_rank = opts.rootRank;
              auto src_dev_id = rank_to_device_id_[src_rank];
              auto cur_dev_id = addressable_device_id_;

              // NOTE: NCCL implementation does the same check. Gloo
              // implementation does something different. In either case, the
              // python-side of the API doesn't actually directly expose
              // multi-tensor variant. Revisit this later.
              TT_CHECK_THROW(tensors.size() == 1, error::kInvalidArgument)
                  << "single tensor expected, but got multiple tensors.";
              auto& tensor = tensors[0];

              if (src_dev_id != cur_dev_id) {
                // NOTE: At the pybind layer (distributed/c10d/init.cpp), torch
                // effectively holds on to the TensorImpl of the original
                // input/output. See
                // https://github.com/pytorch/pytorch/issues/159686 for details.
                // This is why we need to copy_ here, and cannot simply
                // re-assign.
                tensor.copy_(at::zeros(tensor.sizes(), tensor.options()));
              } else {
                // Rank 0 (root rank): We must push the exact same sequence of
                // operations to the dispatcher to keep the thread-local
                // `detect_repeated_ops` heuristic synchronized across all
                // ranks. We execute the same ops on a discarded dummy tensor so
                // we don't corrupt the actual broadcast data.
                auto dummy = at::empty(tensor.sizes(), tensor.options());
                dummy.copy_(at::zeros(tensor.sizes(), tensor.options()));
              }
              ABSL_VLOG(1) << OpDebugString("broadcast") << "DeviceBufferRef: "
                           << DeviceBufferRefDebugString(tensor);

              // In the short-term, we implement broadcast in terms of
              // all-reduce, which is suboptimal. This is because
              // stablehlo.collective_broadcast is not implemented for TPUs. We
              // can re-visit this later and implement a more efficient version.
              c10d::AllreduceOptions allreduce_opts{c10d::ReduceOp::SUM,
                                                    opts.timeout, opts.asyncOp};
              auto work_ptr = allreduce(tensors, allreduce_opts);

              // TODO(b/495494333): Remove the need for this forced
              // materialization. It exists because dist.broadcast_object_list
              // dispatches broadcast() operations from the source rank that are
              // not materialized, so it hangs.
              TT_THROW_IF_ERROR(GetMaterialized(
                  tensors, MaterializationReason::kDistributedOp));

              if (work_ptr != nullptr) {
                dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get())->opType_ =
                    c10d::OpType::BROADCAST;
              }
              return work_ptr;
            });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::experimental_send(
    std::vector<at::Tensor>& tensors, int dst_rank, int tag) {
  std::vector<xla::PjRtBuffer*> pjrt_buffers;
  std::vector<xla::CrossHostTransferKey> transfer_keys;

  pjrt_buffers.reserve(tensors.size());
  transfer_keys.reserve(tensors.size());

  int64_t src_device_id = addressable_device_id_;
  int64_t dst_device_id = rank_to_device_id_[dst_rank];
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    // Extract the underlying hardware buffer from the tensor.
    TT_ASSIGN_OR_THROW(
        const DeviceBufferRef device_buffer,
        GetMaterialized(tensor, MaterializationReason::kDistributedOp));
    TT_ASSIGN_OR_THROW(xla::PjRtBuffer * pjrt_buffer,
                       device_buffer.AwaitBuffer());

    pjrt_buffers.push_back(pjrt_buffer);
    transfer_keys.push_back(
        GetCrossHostTransferKey(src_device_id, dst_device_id, tag, i));
  }

  // Initiate the cross-host send logic, which will block until
  // descriptors from the receiver are available in the store and then
  // begin the asynchronous transfer.
  TT_ASSIGN_OR_THROW(auto send_future,
                     CrossHostSendBuffers(pjrt_buffers, transfer_keys));

  return c10::make_intrusive<TpuWork>(tensors, getRank(), c10d::OpType::SEND,
                                      std::move(send_future));
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::experimental_recv(
    std::vector<at::Tensor>& tensors, int src_rank, int tag) {
  std::vector<xla::Shape> recv_shapes;
  std::vector<xla::CrossHostTransferKey> transfer_keys;
  transfer_keys.reserve(tensors.size());
  recv_shapes.reserve(tensors.size());

  int64_t src_device_id = rank_to_device_id_[src_rank];
  int64_t dst_device_id = addressable_device_id_;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    TT_ASSIGN_OR_THROW(auto xla_dtype,
                       ConvertTo<xla::PrimitiveType>(tensor.scalar_type()));
    xla::Shape xla_shape = xla::ShapeUtil::MakeShape(xla_dtype, tensor.sizes());
    recv_shapes.push_back(std::move(xla_shape));
    transfer_keys.push_back(
        GetCrossHostTransferKey(src_device_id, dst_device_id, tag, i));
  }

  // Initiate the receive process. This creates placeholder buffers and
  // posts their network descriptors to the distributed store for the
  // sender to find.
  TT_ASSIGN_OR_THROW(
      auto recv_buffers,
      CrossHostReceiveBuffers(recv_shapes, std::move(transfer_keys)));

  // Bind the incoming PJRT buffer back to the tensors.
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& tensor = tensors[i];
    std::unique_ptr<xla::PjRtBuffer> buffer = std::move(recv_buffers[i]);
    TT_ASSIGN_OR_THROW(auto device_buffer,
                       DeviceBufferList::CreateMaterialized(std::move(buffer)));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(device_buffer), tensor));
  }

  return c10::make_intrusive<TpuWork>(tensors, getRank(), c10d::OpType::RECV);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllGather, param_keys,
      (output_tensors, input_tensors, opts), {
        TT_SET_PROCESS_GROUP_ID(param_keys);
        const int64_t group_size = getSize();
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch only ever passes
                         // single-element input lists.
            input_tensors.size() == 1 && output_tensors.size() == 1,
            error::kUnimplemented)
            << "multiple input tensors not supported.";

        auto& input_tensor = input_tensors[0];
        std::vector<at::Tensor>& output_tensor_list = output_tensors[0];
        TT_CHECK_THROW(output_tensor_list.size() == group_size,
                       error::kInvalidArgument)
            << "output tensor list must have one tensor per process, got "
            << FormatCount(output_tensor_list.size(), "tensor", "tensors")
            << " and " << FormatCount(group_size, "process", "processes");

        // NOTE: PyTorch NCCL backend does support non-uniform shapes
        // (implemented as a sequence of broadcasts, one per tensor). We decided
        // not to support this yet. Revisit later if needed.
        TT_THROW_IF_ERROR(CheckTensorsUniformShape(output_tensor_list));

        TT_CHECK_THROW(input_tensor.sizes() == output_tensor_list[0].sizes(),
                       error::kInvalidArgument)
            << "input tensor shape " << input_tensor.sizes()
            << " must match output tensor shape "
            << output_tensor_list[0].sizes();

        ABSL_VLOG(1) << OpDebugString("allgather") << "DeviceBufferRef: "
                     << DeviceBufferRefDebugString(input_tensor);

        auto& maybe_materialized_input_tensor = input_tensor;

        auto op_builder =
            [device_ids = rank_to_device_id_, subgroups = subgroup_device_ids_](
                mlir::MlirOp input) -> absl::StatusOr<DynamicMlirOpResults> {
          return BuildDistributedAllGatherShlo(input, subgroups,
                                               AllGatherOutputMode::kList);
        };

        // The PyTorch layer ensures that all input and output tensors
        // have the same dtype.
        TT_ASSIGN_OR_THROW(auto output_dtype, ConvertTo<mlir::ElementType>(
                                                  input_tensor.scalar_type()));
        const std::vector<mlir::ElementType> output_dtypes(group_size,
                                                           output_dtype);
        // When all-gathering scalars, StableHLO will return a list of shape [1]
        // MlirOps/PjRtBuffers.
        int64_t scalar_size[1] = {1};
        std::vector<absl::Span<const int64_t>> output_dims;
        output_dims.reserve(group_size);
        for (const auto& output_tensor : output_tensor_list) {
          if (output_tensor.dim() == 0) {
            output_dims.push_back(scalar_size);
          } else {
            output_dims.push_back(output_tensor.sizes());
          }
        }

        TT_ASSIGN_OR_THROW(
            auto result_buffers,
            (DispatchOp<1, kDynamicSize>(
                std::move(op_builder), maybe_materialized_input_tensor,
                {.out_dtypes = output_dtypes,
                 .out_dims_list = output_dims,
                 .op_param_cache_keys = std::move(param_keys),
                 .split_mode = GetCollectiveSplitMode()})));

        // TODO: respect async.

        ABSL_CHECK_EQ(result_buffers.size(), group_size);  // CRASH_OK
        for (int64_t i = 0; i < group_size; ++i) {
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffers[i]),
                                                   output_tensor_list[i]));
        }

        if (opts.asyncOp) {
          return c10::make_intrusive<TpuWork>(output_tensor_list, getRank(),
                                              c10d::OpType::ALLGATHER);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::_allgather_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::AllgatherOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllGatherIntoTensor, param_keys,
      (output_tensor, input_tensor, opts), {
        TT_SET_PROCESS_GROUP_ID(param_keys);
        const int64_t world_size = getSize();
        auto input_sizes = input_tensor.sizes();
        auto output_sizes = output_tensor.sizes();
        AllGatherOutputMode output_mode;

        if (input_tensor.dim() == 0) {
          // Stacks scalars into a 1-dimensional tensor.
          // The MLIR Op builder wraps each scalar as a 1-dimensional tensor.
          // Concatenating these tensors achieves the desired output shape.
          output_mode = AllGatherOutputMode::kConcat;
          TT_CHECK_THROW(
              output_tensor.dim() == 1 && output_sizes[0] == world_size,
              error::kInvalidArgument)
              << "for scalar input, output tensor must be 1-dimensional "
                 "with size equal to world size. Got output shape "
              << output_sizes << ", world size " << world_size;
        } else if (output_tensor.dim() == input_tensor.dim() + 1) {
          // Stacks tensors.
          output_mode = AllGatherOutputMode::kStack;
          TT_CHECK_THROW(output_sizes[0] == world_size, error::kInvalidArgument)
              << "for stacking, output tensor size "
                 "at dimension 0 must be world size. Got output shape "
              << output_sizes << ", input shape " << input_sizes
              << ", world size " << world_size;
          for (int i = 0; i < input_sizes.size(); ++i) {
            TT_CHECK_THROW(output_sizes[i + 1] == input_sizes[i],
                           error::kInvalidArgument)
                << "for stacking, output tensor shape must match input tensor "
                   "shape along all other dimensions. Got output shape "
                << output_sizes << ", input shape " << input_sizes;
          }
        } else if (output_tensor.dim() == input_tensor.dim()) {
          // Concatenates tensors.
          output_mode = AllGatherOutputMode::kConcat;
          TT_CHECK_THROW(output_sizes[0] == input_sizes[0] * world_size,
                         error::kInvalidArgument)
              << "for concatenation, output tensor size at "
                 "dimension 0 must be world size * input tensor size at "
                 "dimension 0. Got output shape "
              << output_sizes << ", input shape " << input_sizes
              << ", world size " << world_size;
          for (int i = 1; i < input_sizes.size(); ++i) {
            TT_CHECK_THROW(output_sizes[i] == input_sizes[i],
                           error::kInvalidArgument)
                << "for concatenation, output "
                   "tensor shape must match input tensor shape along all other "
                   "dimensions. Got output shape "
                << output_sizes << ", input shape " << input_sizes;
          }
        } else {
          // Arbitrary value to make compiler happy.
          output_mode = AllGatherOutputMode::kConcat;
          TT_CHECK_THROW(false, error::kInvalidArgument)
              << "invalid output tensor shape. "
                 "Number of output dimensions must equal number of input "
                 "dimensions (concatenation) or input dimensions + 1 "
                 "(stacking)."
                 " Got output shape "
              << output_sizes << ", input shape " << input_sizes;
        }

        ABSL_VLOG(1) << OpDebugString("_allgather_base") << "DeviceBufferRef: "
                     << DeviceBufferRefDebugString(input_tensor);

        auto& maybe_materialized_input_tensor = input_tensor;

        auto op_builder =
            [device_ids = rank_to_device_id_, subgroups = subgroup_device_ids_,
             output_mode](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          TT_ASSIGN_OR_RETURN(auto results, BuildDistributedAllGatherShlo(
                                                input, subgroups, output_mode));
          // `results` must have a single element because this kernel only
          // invokes BuildDistributedAllGatherShlo with kConcat or kStack.
          ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
          return results[0];
        };

        TT_ASSIGN_OR_THROW(
            const auto output_dtype,
            ConvertTo<mlir::ElementType>(input_tensor.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<1>(std::move(op_builder),
                          maybe_materialized_input_tensor,
                          {.out_dtype = output_dtype,
                           .out_dims = output_tensor.sizes(),
                           .op_param_cache_keys = std::move(param_keys),
                           .split_mode = GetCollectiveSplitMode()}));

        // TODO: b/443120101 - defer if opts.asyncOp is true.

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), output_tensor));

        if (opts.asyncOp) {
          return c10::make_intrusive<TpuWork>(
              std::vector<at::Tensor>{output_tensor}, getRank(),
              c10d::OpType::_ALLGATHER_BASE);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const c10d::AllgatherOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllGatherIntoTensor, _, (outputs, inputs, opts), {
        // NOTE: there's no such op in `torch.distributed` module, so it
        // is called indirectly.
        TT_CHECK_THROW(inputs.size() == outputs.size(), error::kInvalidArgument)
            << "inputs and outputs must have the same size.";

        // Calls allgather once for each input. This is an acceptable
        // simplification for now as we are primarily focused on supporting
        // higher level distributed APIs such as DTensor, which only ever call
        // this method with a single input.
        for (int i = 0; i < inputs.size(); ++i) {
          _allgather_base(outputs[i], inputs[i], opts);
        }

        if (opts.asyncOp) {
          return c10::make_intrusive<TpuWork>(
              outputs, getRank(), c10d::OpType::ALLGATHER_COALESCED);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::gather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors, const c10d::GatherOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedGather, param_keys,
      (output_tensors, input_tensors, opts), {
        TT_SET_PROCESS_GROUP_ID(param_keys);
        const int64_t rank = getRank();
        const int64_t root_rank = opts.rootRank;
        TT_CHECK_THROW(input_tensors.size() == 1, error::kInvalidArgument)
            << "a single input tensor must be provided, got "
            << input_tensors.size();
        auto& input = input_tensors[0];

        if (rank == root_rank) {
          TT_CHECK_THROW(output_tensors.size() == 1, error::kInvalidArgument)
              << "there must be a single list of output tensors on the root "
                 "rank, got "
              << output_tensors.size();
          const auto& output_tensor_list = output_tensors[0];
          TT_CHECK_THROW(output_tensor_list.size() == getSize(),
                         error::kInvalidArgument)
              << "the number of output tensors on the root rank must be equal "
                 "to the group size, got "
              << output_tensor_list.size() << " tensors and " << getSize()
              << " processes";
          TT_THROW_IF_ERROR(CheckTensorsUniformShape(output_tensor_list))
                  .SetPrepend()
              << "output tensors on the root rank: ";
          TT_CHECK_THROW(input.sizes() == output_tensor_list[0].sizes(),
                         error::kInvalidArgument)
              << "input tensor shape must match output tensor shape";
        } else {
          TT_CHECK_THROW(output_tensors.empty(), error::kInvalidArgument)
              << "on non-root rank " << rank
              << " the list of output tensors must be empty";
        }

        std::vector<at::Tensor> allgather_outputs;
        if (rank == root_rank) {
          allgather_outputs = output_tensors[0];
        } else {
          allgather_outputs.reserve(getSize());
          for (int i = 0; i < getSize(); ++i) {
            allgather_outputs.push_back(at::empty_like(input));
          }
        }

        // Materialize the outputs before the collective to ensure symmetry.
        // This is needed because on the root rank, output_tensors[0] might have
        // deferred ops. On non-root ranks, the temporary tensors are fresh,
        // which can result in an asymmetry.
        TT_THROW_IF_ERROR(GetMaterialized(
            allgather_outputs, MaterializationReason::kDistributedOp));

        c10d::AllgatherOptions allgather_opts;
        allgather_opts.timeout = opts.timeout;
        allgather_opts.asyncOp = opts.asyncOp;

        std::vector<std::vector<at::Tensor>> allgather_output_tensors = {
            std::move(allgather_outputs)};
        auto work_ptr =
            allgather(allgather_output_tensors, input_tensors, allgather_opts);

        // TODO(b/495494333): Remove the need for this forced materialization.
        // On non-root ranks, the all-gather outputs are local temporaries and
        // never materialized if unused, e.g. if the root rank does not use all
        // the gathered outputs.
        // Therefore, we need to force materialization on at least all non-root
        // ranks. If we do not materialize all ranks synchronously, it can cause
        // a stutter where non-root ranks block early while waiting for the root
        // rank to reach a graph break and then need to catch up. Materializing
        // on all ranks synchronously is safer and keeps all devices moving
        // forward smoothly.
        TT_THROW_IF_ERROR(
            GetMaterialized(allgather_output_tensors[0],
                            MaterializationReason::kDistributedOp));

        if (work_ptr != nullptr) {
          dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get())->opType_ =
              c10d::OpType::GATHER;
        }

        return work_ptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const c10d::ScatterOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedScatter, _,
      (outputs, inputs, IgnoreInCacheKey(opts, "Legacy usage")), {
        const int64_t rank = getRank();
        const int64_t root_rank = opts.rootRank;
        TT_CHECK_THROW(outputs.size() == 1, error::kInvalidArgument)
            << "a single output tensor must be provided, got "
            << outputs.size();
        auto& output = outputs[0];
        bool is_scalar = output.dim() == 0;

        at::Tensor scatter_input;  // UNINITIALIZED_TENSOR_OK
        if (rank == root_rank) {
          TT_CHECK_THROW(inputs.size() == 1, error::kInvalidArgument)
              << "there must be a single list of input tensors on the root "
                 "rank, got "
              << inputs.size();
          const auto& input_tensors = inputs[0];
          TT_CHECK_THROW(input_tensors.size() == getSize(),
                         error::kInvalidArgument)
              << "the number of input tensors on the root rank must be equal to"
              << " the group size, got " << input_tensors.size()
              << " tensors and " << getSize() << " processes";

          TT_THROW_IF_ERROR(CheckTensorsUniformShape(input_tensors))
                  .SetPrepend()
              << "input tensors on the root rank: ";
          TT_CHECK_THROW(output.sizes() == input_tensors[0].sizes(),
                         error::kInvalidArgument)
              << "output tensor shape must match input tensor shape, got "
              << output.sizes() << " and " << input_tensors[0].sizes();
          if (is_scalar) {
            scatter_input = at::stack(input_tensors);
          } else {
            scatter_input = at::cat(input_tensors, 0);
          }
        } else {
          TT_CHECK_THROW(inputs.empty(), error::kInvalidArgument)
              << "on non-root rank " << rank
              << " the list of input tensors must be empty";

          if (is_scalar) {
            scatter_input = at::zeros({getSize()}, output.options());
          } else {
            auto scatter_input_dims = CopyIntVector(output.sizes());
            scatter_input_dims[0] *= getSize();
            scatter_input = at::zeros(scatter_input_dims, output.options());
          }
        }

        c10d::ReduceScatterOptions reduce_scatter_opts;
        reduce_scatter_opts.reduceOp = c10d::ReduceOp::SUM;
        reduce_scatter_opts.timeout = opts.timeout;
        reduce_scatter_opts.asyncOp = opts.asyncOp;

        c10::intrusive_ptr<c10d::Work> work_ptr =
            _reduce_scatter_base(output, scatter_input, reduce_scatter_opts);
        if (work_ptr != nullptr) {
          // If async updates the Work object to have the correct op type.
          dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get())->opType_ =
              c10d::OpType::SCATTER;
        }
        return work_ptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedReduceScatter, _,
      (output_tensors, input_tensors, IgnoreInCacheKey(opts, "Legacy usage")), {
        // NOTE: Python side API only exposes single-element reduce_scatter op.
        // Same validation is done in NCCL backend.
        TT_CHECK_THROW(input_tensors.size() == 1, error::kUnimplemented)
            << "multiple input tensor lists not supported.";
        TT_CHECK_THROW(output_tensors.size() == 1, error::kUnimplemented)
            << "multiple output tensor lists not supported.";

        std::vector<at::Tensor>& inputs = input_tensors[0];
        at::Tensor& output = output_tensors[0];

        TT_CHECK_THROW(inputs.size() == getSize(), error::kInvalidArgument)
            << "length of input tensors list must match "
            << "world size, got " << inputs.size() << " input tensors and "
            << getSize() << " processes.";

        // NOTE: NCCL does support this case, but is doing that by running a
        // sequence of world_size separate reduce calls (coalesced). Revisit
        // later if we want to support this case too. It would be very
        // problematic for error handling.
        TT_THROW_IF_ERROR(CheckTensorsUniformShape(inputs));

        TT_CHECK_THROW(output.sizes() == inputs[getRank()].sizes(),
                       error::kInvalidArgument)
            << "output tensor shape " << output.sizes()
            << " must match input tensor shape " << inputs[getRank()].sizes();

        // We have `_reduce_scatter_base`, which aligns much better with
        // what StableHLO expects for reduce-scatter, so we delegate to that.
        at::Tensor input = at::cat(inputs);
        auto work_ptr = _reduce_scatter_base(output, input, opts);
        if (work_ptr != nullptr) {
          dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get())->opType_ =
              c10d::OpType::REDUCE_SCATTER;
        }
        return work_ptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::_reduce_scatter_base(
    at::Tensor& output, at::Tensor& input,
    const c10d::ReduceScatterOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedReduceScatterTensor, param_keys,
      (output, input, opts), {
        TT_SET_PROCESS_GROUP_ID(param_keys);
        TT_THROW_IF_ERROR(
            ValidateReductionOp(opts.reduceOp, input.scalar_type()));

        TT_ASSIGN_OR_THROW(auto input_mode, CheckReduceScatterInputTensorShape(
                                                output, input, getSize()));

        // Reduces to the concat case as that's what StableHLO expects.
        at::Tensor input_concat;  // UNINITIALIZED_TENSOR_OK
        if (input_mode == ReduceScatterShapeMode::kConcat) {
          input_concat = input;
        } else {
          input_concat = at::flatten(input, /*start_dim=*/0, /*end_dim=*/1);
        }

        ABSL_VLOG(1) << OpDebugString("_reduce_scatter_base")
                     << "DeviceBufferRef: "
                     << DeviceBufferRefDebugString(input_concat);

        auto& maybe_materialized_input_concat_tensor = input_concat;

        auto op_builder = [reduce_op = opts.reduceOp,
                           subgroups =
                               subgroup_device_ids_](mlir::MlirOp input) {
          return BuildDistributedReduceScatterShlo(input, reduce_op, subgroups);
        };

        TT_ASSIGN_OR_THROW(auto out_dtype,
                           ConvertTo<mlir::ElementType>(output.scalar_type()));
        // Calculate the output shape by splitting on dim=0 (our scatter
        // dimension).
        Dimensions out_dims =
            CopyIntVector(maybe_materialized_input_concat_tensor.sizes());
        size_t group_size = subgroup_device_ids_[0].size();
        ABSL_CHECK_EQ(out_dims[0] % group_size, 0);  // CRASH_OK
        out_dims[0] /= group_size;

        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<1>(std::move(op_builder),
                          maybe_materialized_input_concat_tensor,
                          {.out_dtype = out_dtype,
                           .out_dims = std::move(out_dims),
                           .op_param_cache_keys = std::move(param_keys),
                           .split_mode = GetCollectiveSplitMode()}));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), output));

        if (opts.asyncOp) {
          return c10::make_intrusive<TpuWork>(
              std::vector<at::Tensor>{output}, getRank(),
              c10d::OpType::_REDUCE_SCATTER_BASE);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const c10d::ReduceScatterOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedReduceScatterTensorCoalesced, _,
      (outputs, inputs, IgnoreInCacheKey(opts, "Legacy usage")), {
        TT_CHECK_THROW(inputs.size() == outputs.size(), error::kInvalidArgument)
            << "inputs and outputs must have the same size, got "
            << inputs.size() << " inputs and " << outputs.size() << " outputs";

        // Calls reduce_scatter once for each input. This is an acceptable
        // simplification for now as we are primarily focused on supporting
        // higher level distributed APIs such as DTensor,
        // which only ever call this method with a single input.
        for (int i = 0; i < inputs.size(); ++i) {
          _reduce_scatter_base(outputs[i], inputs[i], opts);
        }

        if (opts.asyncOp) {
          return c10::make_intrusive<TpuWork>(
              outputs, getRank(), c10d::OpType::_REDUCE_SCATTER_BASE);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::alltoall_base(
    at::Tensor& output, at::Tensor& input,
    std::vector<int64_t>& output_split_sizes,  // INT_VEC_OK
    std::vector<int64_t>& input_split_sizes,   // INT_VEC_OK
    const c10d::AllToAllOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllToAllSingle, _,
      (output, input, IgnoreInCacheKey(output_split_sizes, "Legacy usage"),
       IgnoreInCacheKey(input_split_sizes, "Legacy usage"), opts),
      {
        const int64_t rank = getRank();
        const bool async = opts.asyncOp;
        const int64_t group_size = subgroup_device_ids_[0].size();

        // Check on input and output dtypes already done in the PyTorch
        // layer. Hence, we do not need to check for dtypes here.

        TT_THROW_IF_ERROR(CheckSplitSizesForAllToAllSingle(input_split_sizes,
                                                           input, group_size));
        TT_THROW_IF_ERROR(CheckSplitSizesForAllToAllSingle(output_split_sizes,
                                                           output, group_size));

        const bool equal_input_splits = IsEqualSplits(input_split_sizes);
        const bool equal_output_splits = IsEqualSplits(output_split_sizes);

        if (equal_input_splits && equal_output_splits) {
          TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                             AllToAllBaseEqualSplits(output, input));
          TT_THROW_IF_ERROR(
              AssignBufferToAtTensor(std::move(result_buf), output));
        } else {
          TT_ASSIGN_OR_THROW(
              DeviceBufferRef result_buf,
              AllToAllBaseUnevenSplits(output, input, output_split_sizes,
                                       input_split_sizes));
          TT_THROW_IF_ERROR(
              AssignBufferToAtTensor(std::move(result_buf), output));
        }

        if (async) {
          return c10::make_intrusive<TpuWork>(std::vector<at::Tensor>{output},
                                              rank,
                                              c10d::OpType::ALLTOALL_BASE);
        }
        return nullptr;
      });
}

absl::StatusOr<DeviceBufferRef> ProcessGroupTpu::AllToAllBaseEqualSplits(
    at::Tensor& output, at::Tensor& input) {
  auto& maybe_materialized_input_tensor = input;
  TT_RET_CHECK(output.sizes() == input.sizes(), error::kInvalidArgument)
      << "for equal splits, shape of input and output must be the same,"
      << " got " << input.sizes() << " and " << output.sizes();

  TT_ASSIGN_OR_RETURN(auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(pg_id_));

  auto op_builder = [device_groups = subgroup_device_ids_](mlir::MlirOp input) {
    return BuildDistributedAllToAllBaseShlo(input, device_groups);
  };

  TT_ASSIGN_OR_RETURN(auto output_dtype,
                      ConvertTo<mlir::ElementType>(output.scalar_type()));

  // Give the all_to_all split is on input tensor dimension 0,
  // the output tensor shape will be the same as the input tensor shape.
  return DispatchOp<1>(std::move(op_builder), maybe_materialized_input_tensor,
                       {.out_dtype = output_dtype,
                        .out_dims = output.sizes(),
                        .op_param_cache_keys = std::move(param_keys),
                        .split_mode = GetCollectiveSplitMode()});
}

// TODO(mkkhanna): Implement support for uneven splits.
// b/446982820
absl::StatusOr<DeviceBufferRef> ProcessGroupTpu::AllToAllBaseUnevenSplits(
    at::Tensor& output, at::Tensor& input,
    const std::vector<int64_t>& output_split_sizes,   // INT_VEC_OK
    const std::vector<int64_t>& input_split_sizes) {  // INT_VEC_OK
  return TT_ERROR(error::kUnimplemented) << "uneven splits is not implemented";
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors, const c10d::AllToAllOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllToAll, param_keys,
      (output_tensors, input_tensors, opts), {
        const int64_t rank = getRank();
        const bool async = opts.asyncOp;
        TT_THROW_IF_ERROR(param_keys.SetParam("rank", rank));
        TT_THROW_IF_ERROR(param_keys.SetParam("async", async));
        TT_SET_PROCESS_GROUP_ID(param_keys);
        TT_CHECK_THROW(output_tensors.size() == input_tensors.size(),
                       error::kInvalidArgument)
            << "output and input tensors must have the same number of tensors, "
               "got "
            << output_tensors.size() << " for output and "
            << input_tensors.size() << " for input";

        const int64_t group_size = getSize();

        TT_CHECK_THROW(input_tensors.size() == group_size,
                       error::kInvalidArgument)
            << "input tensors must have the same number of tensors as the "
               "process "
               "group size, got "
            << input_tensors.size() << " for input and " << group_size
            << " for process group size";

        TT_THROW_IF_ERROR(
            CheckAllToAllShapeConsistency(output_tensors, input_tensors));

        auto op_builder = [device_groups = subgroup_device_ids_](
                              absl::Span<mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder) {
          return BuildDistributedAllToAllShlo(inputs, device_groups);
        };

        TT_ASSIGN_OR_THROW(
            auto output_dtype,
            ConvertTo<mlir::ElementType>(output_tensors[0].scalar_type()));

        std::vector<mlir::ElementType> result_dtypes(output_tensors.size(),
                                                     output_dtype);

        std::vector<absl::Span<const int64_t>> result_dims_list;
        result_dims_list.reserve(output_tensors.size());
        for (const auto& output_tensor : output_tensors) {
          result_dims_list.push_back(output_tensor.sizes());
        }

        TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> result_buffers,
                           (DispatchOp<kDynamicSize, kDynamicSize>(
                               std::move(op_builder), input_tensors,
                               {.out_dtypes = result_dtypes,
                                .out_dims_list = result_dims_list,
                                .op_param_cache_keys = std::move(param_keys),
                                .split_mode = GetCollectiveSplitMode()})));

        for (int64_t i = 0; i < result_buffers.size(); ++i) {
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffers[i]),
                                                   output_tensors[i]));
        }

        if (async) {
          return c10::make_intrusive<TpuWork>(output_tensors, rank,
                                              c10d::OpType::ALLTOALL);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::barrier(
    const c10d::BarrierOptions& opts) {
  TT_KERNEL(OpName::kDistributedBarrier, _, (opts), {
    // Check for unsupported options.
    TT_CHECK_THROW(opts.device_ids.empty(), error::kUnimplemented)
        << "device_ids in barrier options is not supported.";
    TT_CHECK_THROW(opts.timeout == c10d::kUnsetTimeout, error::kUnimplemented)
        << "timeout in barrier options is not supported.";
    TT_CHECK_THROW(!opts.device.has_value(), error::kUnimplemented)
        << "device in barrier options is not supported.";

    // A barrier is implemented by performing an all-reduce operation on a dummy
    // tensor. Since all-reduce is a synchronizing collective, this ensures all
    // participating devices reach this point before any can proceed.
    at::Tensor dummy_tensor =
        at::ones({1}, at::device(GetPrivateUse1DeviceType()).dtype(at::kLong));
    std::vector<at::Tensor> tensors = {dummy_tensor};

    c10d::AllreduceOptions allreduce_opts;
    allreduce_opts.reduceOp = c10d::ReduceOp::SUM;
    allreduce_opts.asyncOp = opts.asyncOp;

    auto work_ptr = allreduce(tensors, allreduce_opts);

    // If the barrier is synchronous, we must block the host until the
    // operation completes on the device. We do this by moving the result
    // to the CPU, which forces a synchronization.
    if (!opts.asyncOp) {
      dummy_tensor.item();
    }

    if (work_ptr != nullptr) {
      dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get())->opType_ =
          c10d::OpType::BARRIER;
    }
    return work_ptr;
  });
}

DeviceGroupList ProcessGroupTpu::GatherAllSubgroups() {
  TT_KERNEL(OpName::kTorchTpuInternalGatherAllSubgroups, param_keys, (), {
    TT_SET_PROCESS_GROUP_ID(param_keys);
    TT_THROW_IF_ERROR(param_keys.SetParam("device_ids", device_ids_));
    int64_t world_size = device_ids_.size();  // All devices in the TPU slice
    int64_t group_size = getSize();  // Devices in this particular process group

    // We cannot use c10d::Store to exchange information between processes
    // in different subgroups, so instead we use StableHLO collectives over
    // the global TPU slice to do that.
    //
    // We do this by leveraging the existing DispatchOp machinery, hence
    // create the data as torch tensors, and use existing all-reduce
    // builder.
    //
    // NOTE: this tensor size is O(device_count^2). We can fix it later, if
    // still needed after b/445283852 (or if latter cannot be done).
    at::Tensor subgroups = at::zeros({world_size, group_size},
                                     at::device(at::kCPU).dtype(at::kLong));

    // This process knows the members of its own subgroup, thanks to
    // c10d::Store.
    ABSL_CHECK_EQ(group_size, rank_to_device_id_.size())  // CRASH_OK
        << "Inconsisent device subgroup. This is a bug.";
    for (int i = 0; i < group_size; ++i) {
      subgroups.index({addressable_device_id_, i}) = rank_to_device_id_[i];
    }
    subgroups = subgroups.to(at::device(GetPrivateUse1DeviceType()));

    // Thanks to PJRT, we know all the devices in the TPU slice, so we can
    // perform a collective, without having to know the "default" process
    // group. We just need to reshape device list as [1, N] to use as
    // subgroup list.
    auto world_group = DeviceGroupList(1, device_ids_);
    auto op_builder = [group = std::move(world_group)](mlir::MlirOp input) {
      return BuildDistributedAllReduceShlo(input, c10d::ReduceOp::SUM, group);
    };

    TT_ASSIGN_OR_THROW(auto output_dtype,
                       ConvertTo<mlir::ElementType>(subgroups.scalar_type()));

    TT_ASSIGN_OR_THROW(
        auto result_buffer,
        DispatchOp<1>(std::move(op_builder), subgroups,
                      {.out_dtype = output_dtype,
                       .out_dims = subgroups.sizes(),
                       .op_param_cache_keys = std::move(param_keys)}));
    TT_THROW_IF_ERROR(
        AssignBufferToAtTensor(std::move(result_buffer), subgroups));
    subgroups = subgroups.cpu();

    ABSL_VLOG(1) << "[ProcessGroupTpu::GatherAllSubgroups] "
                 << "rank: " << getRank() << ", size " << getSize()
                 << ", subgroups: " << subgroups;

    // Convert from at::Tensor to a non-pytorch type.
    auto result = DeviceGroupList(world_size, DeviceGroup(group_size));
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < group_size; ++j) {
        result[i][j] = subgroups.index({i, j}).item().toLong();
      }
    }

    // Sort (lexicographic order) and deduplicate the subgroups.
    std::set<DeviceGroup> subgroups_set(result.cbegin(), result.cend());
    return DeviceGroupList(subgroups_set.cbegin(), subgroups_set.cend());
  });
}

}  // namespace torch_tpu
