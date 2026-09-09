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

#include "csrc/distributed/process_group_tpu.h"

#include <atomic>
#include <chrono>  // NOLINT - needed for PyTorch API
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/macro_utils.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Store.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#if TT_TORCH_VERSION_GE(2, 14)
#include "torch/csrc/distributed/c10d/Window.hpp"
#endif
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/distributed/allgather.h"
#include "csrc/distributed/allreduce.h"
#include "csrc/distributed/alltoall.h"
#include "csrc/distributed/reduce_scatter.h"
#include "csrc/distributed/types.h"
#include "csrc/distributed/utils.h"
#include "csrc/distributed/window_tpu.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/internal/compile/torch_compile_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"

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

  bool wait(std::chrono::milliseconds timeout) override {
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

absl::Status ValidateTensorsUniformShape(
    const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    return absl::OkStatus();
  }

  const auto& first_tensor_size = tensors[0].sizes();
  for (int i = 1; i < tensors.size(); ++i) {
    TT_RET_CHECK(tensors[i].sizes() == first_tensor_size,
                 error::kInvalidArgument)
        << "expected all input tensors to have the same shape, got "
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
absl::StatusOr<ReduceScatterShapeMode> ValidateReduceScatterInputTensorShape(
    at::Tensor& output, at::Tensor& input, size_t process_group_size) {
  // Concat mode: same shape as output, but dim=0 is pg_size times larger:
  Dimensions concat_mode_dims;
  if (output.dim() == 0) {
    concat_mode_dims = {static_cast<int64_t>(process_group_size)};
  } else {
    concat_mode_dims = CopyIntVector(output.sizes());
    concat_mode_dims[0] *= process_group_size;
  }
  ABSL_VLOG(1) << "ValidateReduceScatterInputTensorShape: concat_mode_dims: "
               << ToString(concat_mode_dims);

  // Stack mode: prepend new dimension of size pg_size, and remaining dims same.
  Dimensions stack_mode_dims(output.dim() + 1);
  stack_mode_dims[0] = process_group_size;
  for (int i = 0; i < output.dim(); ++i) {
    stack_mode_dims[i + 1] = output.size(i);
  }
  ABSL_VLOG(1) << "ValidateReduceScatterInputTensorShape: stack_mode_dims: "
               << ToString(stack_mode_dims);

  Dimensions actual_input_dims{input.sizes().begin(), input.sizes().end()};
  ABSL_VLOG(1) << "ValidateReduceScatterInputTensorShape: actual_input_dims: "
               << ToString(actual_input_dims);

  if (actual_input_dims == concat_mode_dims) {
    return ReduceScatterShapeMode::kConcat;
  } else if (actual_input_dims == stack_mode_dims) {
    return ReduceScatterShapeMode::kStack;
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "expected the input tensor to be either of shape "
           << ToString(concat_mode_dims) << " or of shape "
           << ToString(stack_mode_dims) << ", got "
           << ToString(actual_input_dims);
  }
}

absl::Status ValidateSplitSizesForAllToAllSingle(c10::IntArrayRef split_sizes,
                                                 const at::Tensor& tensor,
                                                 size_t group_size) {
  auto dim0 = tensor.sizes()[0];
  if (split_sizes.empty()) {
    TT_RET_CHECK(dim0 % group_size == 0, error::kInvalidArgument)
        << "expected tensor first dimension to be divisible by process group "
        << "size, got " << dim0 << " is not divisible by " << group_size;
  } else {
    TT_RET_CHECK(split_sizes.size() == group_size, error::kInvalidArgument)
        << "expected split sizes to have the same size as process group size, "
        << "got " << split_sizes.size() << " for split sizes and " << group_size
        << " for process group size";

    for (int64_t size : split_sizes) {
      TT_RET_CHECK(size >= 0, error::kInvalidArgument)
          << "expected split sizes to be non-negative, got " << size;
    }

    const int64_t split_sizes_sum = absl::c_accumulate(split_sizes, 0L);
    TT_RET_CHECK(split_sizes_sum == dim0, error::kInvalidArgument)
        << "expected split sizes sum to be equal to tensor first dimension, "
        << "got " << split_sizes_sum << " for split sizes sum and " << dim0
        << " for tensor shape " << tensor.sizes() << " dim 0";
  }
  return absl::OkStatus();
}

bool IsEqualSplits(c10::IntArrayRef split_sizes) {
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

// Validates tensor list invariants for all-to-all: matching process group
// sizes, matching dtypes, matching dimension counts, and matching trailing
// feature dimensions (dim 1 onwards).
absl::Status ValidateAllToAllTensors(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<at::Tensor>& input_tensors) {
  ABSL_CHECK_EQ(output_tensors.size(), input_tensors.size());  // CRASH_OK
  ABSL_CHECK_GT(input_tensors.size(), 0);                      // CRASH_OK

  const at::ScalarType dtype = input_tensors[0].scalar_type();
  const int64_t dim = input_tensors[0].dim();
  const auto trailing_dims =
      dim > 0 ? input_tensors[0].sizes().slice(1) : c10::IntArrayRef();

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    TT_RET_CHECK(input_tensors[i].scalar_type() == dtype,
                 error::kInvalidArgument)
        << "expected all input tensors to have the same dtype, got "
        << input_tensors[i].scalar_type() << " and " << dtype;
    TT_RET_CHECK(output_tensors[i].scalar_type() == dtype,
                 error::kInvalidArgument)
        << "expected all output tensors to have the same dtype, got "
        << output_tensors[i].scalar_type() << " and " << dtype;

    TT_RET_CHECK(input_tensors[i].dim() == dim, error::kInvalidArgument)
        << "expected all input tensors to have the same number of dimensions, "
        << "got " << input_tensors[i].dim() << " and " << dim;
    TT_RET_CHECK(output_tensors[i].dim() == dim, error::kInvalidArgument)
        << "expected all output tensors to have the same number of "
        << "dimensions, got " << output_tensors[i].dim() << " and " << dim;

    if (dim > 0) {
      TT_RET_CHECK(input_tensors[i].sizes().slice(1) == trailing_dims,
                   error::kInvalidArgument)
          << "expected trailing dimensions of input tensors to match, got "
          << input_tensors[i].sizes() << " and " << input_tensors[0].sizes();
      TT_RET_CHECK(output_tensors[i].sizes().slice(1) == trailing_dims,
                   error::kInvalidArgument)
          << "expected trailing dimensions of output tensors to match, got "
          << output_tensors[i].sizes() << " and " << input_tensors[0].sizes();
    }
  }
  return absl::OkStatus();
}

// Returns true if all input and output tensors have identical shapes, enabling
// the uniform all-to-all lowering path without ragged collective slicing.
bool IsUniformAllToAll(const std::vector<at::Tensor>& output_tensors,
                       const std::vector<at::Tensor>& input_tensors) {
  const auto first_shape = input_tensors[0].sizes();
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    if (input_tensors[i].sizes() != first_shape ||
        output_tensors[i].sizes() != first_shape) {
      return false;
    }
  }
  return true;
}

// Returns the OpSplitMode to use for collectives.
OpSplitMode GetCollectiveSplitMode() {
  if (IsSpmdSafe()) {
    return OpSplitMode::kNone;
  }

  if (PyGetMaterializeCollectiveTensorsEnvVarOnce()) {
    return OpSplitMode::kSplitBoth;
  }
  return OpSplitMode::kNone;
}

// Offsets and slice sizes required for StableHLO ragged all-to-all custom
// calls.
struct RaggedAllToAllOffsets {
  std::vector<int32_t> input_offsets;
  std::vector<int32_t> send_sizes;
  std::vector<int32_t> output_offsets;
  std::vector<int32_t> recv_sizes;
};

// Synchronizes destination output offsets across all ranks via c10d::Store.
//
// In the StableHLO ragged all-to-all custom call, each sender must supply
// `output_offsets`, which defines the exact element offset on the destination
// rank where the sender's data slice will land upon arrival. Because
// destination offsets depend on the sum of slice sizes from all preceding ranks
// (src < sender) arriving at that peer, ranks must exchange their local output
// placement offsets so that rank `my_rank` can obtain `output_offsets[peer] =
// peer_offsets[my_rank]`.
//
// Synchronization is performed synchronously on the host CPU at dispatch time,
// before `DispatchOp` is called. Resolving `output_offsets` on the host allows:
//   1. Baking offsets into `param_keys` (`OpParamCacheKeys`) for PJRT/XLA
//      compilation cache lookups.
//   2. Embedding offsets as `mlir::stablehlo::Constant` ops in the MLIR op
//      builder.
//
// Note on Performance & Alternatives:
//   Using `c10d::Store` introduces a synchronous host-side barrier on CPU
//   threads across all nodes before kernel dispatch. An alternative design is
//   to eliminate `c10d::Store` and compute `output_offsets` directly on TPU by
//   emitting a 1-element-per-rank uniform `stablehlo::AllToAll` on
//   `local_output_offsets` inside the MLIR graph. That would allow the entire
//   offset exchange to run asynchronously over TPU Inter-Chip Interconnect
//   (ICI) at sub-microsecond hardware speed.
absl::StatusOr<std::vector<int32_t>> SynchronizeOutputOffsets(
    c10d::Store* store, int my_rank, uint64_t seq,
    absl::Span<const int32_t> local_output_offsets) {
  const size_t group_size = local_output_offsets.size();
  std::vector<int32_t> output_offsets_vec(group_size, 0);
  if (group_size <= 1) {
    return output_offsets_vec;
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Store is required for multi-rank
                 // distributed setup.
      store != nullptr, error::kInvalidArgument)
      << "c10d::Store is required for multi-rank uneven all-to-all";
  std::string key_prefix = absl::StrCat("alltoall_offsets:", seq, ":");
  std::string my_key = absl::StrCat(key_prefix, my_rank);

  // 1. Publish local output offsets to c10d::Store.
  std::vector<uint8_t> my_bytes(local_output_offsets.size() * sizeof(int32_t));
  std::memcpy(my_bytes.data(), local_output_offsets.data(), my_bytes.size());
  store->set(my_key, my_bytes);

  // 2. Fetch destination offsets from all peer ranks.
  std::vector<std::string> peer_keys;
  peer_keys.reserve(group_size);
  for (size_t peer = 0; peer < group_size; ++peer) {
    peer_keys.push_back(absl::StrCat(key_prefix, peer));
  }
  std::vector<std::vector<uint8_t>> peer_values = store->multiGet(peer_keys);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=multiGet returns group_size values or
                 // fails.
      peer_values.size() == group_size, error::kInternal)
      << "expected " << group_size << " peer values, got "
      << peer_values.size();

  // 3. Extract the destination offset on each peer rank where this rank's
  // outgoing slice will be placed.
  const size_t expected_payload_bytes = group_size * sizeof(int32_t);
  for (size_t peer = 0; peer < group_size; ++peer) {
    const auto& peer_val = peer_values[peer];
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Peers always exchange fixed size
                   // payload.
        peer_val.size() == expected_payload_bytes, error::kInternal)
        << "peer " << peer << " payload size mismatch: " << peer_val.size();
    const int32_t* peer_offsets =
        reinterpret_cast<const int32_t*>(peer_val.data());
    output_offsets_vec[peer] = peer_offsets[my_rank];
  }
  return output_offsets_vec;
}

// Computes local input offsets via prefix sums over send_sizes, computes local
// output slice offsets over recv_sizes, and coordinates with peer ranks via
// SynchronizeOutputOffsets to obtain remote destination output offsets required
// by the StableHLO ragged_all_to_all collective.
absl::StatusOr<RaggedAllToAllOffsets> ComputeRaggedAllToAllOffsets(
    c10d::Store* store, int rank, uint64_t seq,
    absl::Span<const int32_t> send_sizes,
    absl::Span<const int32_t> recv_sizes) {
  const size_t group_size = send_sizes.size();
  RaggedAllToAllOffsets offsets{
      .input_offsets = std::vector<int32_t>(group_size, 0),
      .send_sizes = std::vector<int32_t>(send_sizes.begin(), send_sizes.end()),
      .output_offsets = std::vector<int32_t>(group_size, 0),
      .recv_sizes = std::vector<int32_t>(recv_sizes.begin(), recv_sizes.end()),
  };
  std::vector<int32_t> local_output_offsets(group_size, 0);
  int32_t in_offset = 0;
  int32_t out_offset = 0;
  // Compute local send buffer slice offsets (input_offsets) and local receive
  // buffer placement offsets (local_output_offsets) via prefix sums.
  for (size_t i = 0; i < group_size; ++i) {
    offsets.input_offsets[i] = in_offset;
    in_offset += send_sizes[i];
    local_output_offsets[i] = out_offset;
    out_offset += recv_sizes[i];
  }
  // Synchronize with all peer ranks to determine where outgoing slices land
  // on remote destination buffers.
  TT_ASSIGN_OR_RETURN(
      offsets.output_offsets,
      SynchronizeOutputOffsets(store, rank, seq, local_output_offsets));
  return offsets;
}

// Populates kernel cache parameter keys for ragged all-to-all collectives.
absl::Status SetRaggedAllToAllParamKeys(OpParamCacheKeys& param_keys,
                                        const RaggedAllToAllOffsets& offsets) {
  TT_RETURN_IF_ERROR(
      param_keys.SetParam("input_offsets", offsets.input_offsets));
  TT_RETURN_IF_ERROR(param_keys.SetParam("send_sizes", offsets.send_sizes));
  TT_RETURN_IF_ERROR(
      param_keys.SetParam("output_offsets", offsets.output_offsets));
  TT_RETURN_IF_ERROR(param_keys.SetParam("recv_sizes", offsets.recv_sizes));
  return absl::OkStatus();
}

}  // namespace

xla::CrossHostTransferKey GetCrossHostTransferKey(int64_t src_device_id,
                                                  int64_t dst_device_id,
                                                  int tag,
                                                  size_t tensor_index) {
  const FingerprintType fp =
      FingerprintCat(src_device_id, dst_device_id, tag, tensor_index);
  return xla::CrossHostTransferKey(static_cast<int64_t>(fp));
}

namespace {

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
  auto* client = PjrtBackend::GetInstance().GetClient();
  TT_RET_CHECK(client != nullptr, error::kInternal)
      << "PjRtClient is not initialized.";
  TT_ASSIGN_OR_RETURN(
      auto buffers,
      client->MakeCrossHostReceiveBuffers(
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
  ABSL_CHECK(pjrt_client != nullptr);  // CRASH_OK

  // All devices in the slice:
  for (const auto* dev : pjrt_client->devices()) {
    ABSL_CHECK(dev != nullptr) << "Got a nullptr PjRtDevice.";  // CRASH_OK
    device_ids_.push_back(dev->global_device_id().value());
  }

  // Addressable device "assigned" to this process/rank:
  TT_CHECK_THROW(pjrt_client && pjrt_client->addressable_devices().size() == 1,
                 error::kInternal)
      << "expected exactly one TPU device per PyTorch process";
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
    std::vector<int64_t> rank_to_logical_device_id(group_size);  // INT_VEC_OK
    for (int i = 0; i < group_size; ++i) {
      rank_to_logical_device_id[i] = GetLogicalDeviceId(rank_to_device_id_[i]);
    }
    subgroup_device_ids_ =
        DeviceGroupList({std::move(rank_to_logical_device_id)});
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

// Sets the subgroups fingerprint in the given param cache keys.
#define TT_SET_SUBGROUPS_FINGERPRINT(param_keys) \
  TT_THROW_IF_ERROR(                             \
      param_keys.SetParam("subgroups_fingerprint", subgroup_device_ids_))

const DeviceGroupList& ProcessGroupTpu::GetSubgroupDeviceIds() const {
  return subgroup_device_ids_;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  TT_KERNEL(OpName::kDistributedAllReduce, param_keys, (tensors, opts), {
    TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
    // TODO(vladbelous): Implement support for multiple input/output
    // tensors:
    TT_CHECK_THROW(tensors.size() == 1, error::kPythonNotImplementedError)
        << "does not yet support multiple tensors";

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
            (tensors, IgnoreInCacheKey(opts, "Doesn't affect SHLO")), {
              auto src_rank = opts.rootRank;
              auto src_dev_id = rank_to_device_id_[src_rank];
              auto cur_dev_id = addressable_device_id_;

              // NOTE: NCCL implementation does the same check. Gloo
              // implementation does something different. In either case, the
              // python-side of the API doesn't actually directly expose
              // multi-tensor variant. Revisit this later.
              TT_CHECK_THROW(tensors.size() == 1, error::kInvalidArgument)
                  << "expected a single tensor, got " << tensors.size();
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
        MaterializeAndReturn(tensor, MaterializationReason::kDistributedOp));
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
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        const int64_t group_size = getSize();
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch only ever passes
                         // single-element input lists.
            input_tensors.size() == 1 && output_tensors.size() == 1,
            error::kPythonNotImplementedError)
            << "multiple input tensors not supported";

        auto& input_tensor = input_tensors[0];
        std::vector<at::Tensor>& output_tensor_list = output_tensors[0];
        TT_CHECK_THROW(output_tensor_list.size() == group_size,
                       error::kInvalidArgument)
            << "expected output tensor list to have one tensor per process, "
            << "got "
            << FormatCount(output_tensor_list.size(), "tensor", "tensors")
            << " and " << FormatCount(group_size, "process", "processes");

        // NOTE: PyTorch NCCL backend does support non-uniform shapes
        // (implemented as a sequence of broadcasts, one per tensor). We decided
        // not to support this yet. Revisit later if needed.
        TT_THROW_IF_ERROR(ValidateTensorsUniformShape(output_tensor_list));

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
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
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
              << "expected output for scalar inputs to be a 1D tensor with "
                 "size equal to the world size, got output shape "
              << output_sizes << " and world size " << world_size;
        } else if (output_tensor.dim() == input_tensor.dim() + 1) {
          // Stacks tensors.
          output_mode = AllGatherOutputMode::kStack;
          TT_CHECK_THROW(output_sizes[0] == world_size, error::kInvalidArgument)
              << "expected output for stacking to be a tensor with size equal "
                 "to the world size at dimension 0, got output shape "
              << output_sizes << ", input shape " << input_sizes
              << ", and world size " << world_size;
          for (int i = 0; i < input_sizes.size(); ++i) {
            TT_CHECK_THROW(output_sizes[i + 1] == input_sizes[i],
                           error::kInvalidArgument)
                << "expected output for stacking to match the input tensor "
                   "shape along all other dimensions, got output shape "
                << output_sizes << " and input shape " << input_sizes;
          }
        } else if (output_tensor.dim() == input_tensor.dim()) {
          // Concatenates tensors.
          output_mode = AllGatherOutputMode::kConcat;
          TT_CHECK_THROW(output_sizes[0] == input_sizes[0] * world_size,
                         error::kInvalidArgument)
              << "expected output for concatenation to be a tensor with size "
                 "equal to world size * input size at dimension 0, got output "
                 "shape "
              << output_sizes << ", input shape " << input_sizes
              << ", and world size " << world_size;
          for (int i = 1; i < input_sizes.size(); ++i) {
            TT_CHECK_THROW(output_sizes[i] == input_sizes[i],
                           error::kInvalidArgument)
                << "expected output for concatenation to match the input "
                   "tensor shape along all other dimensions, got output shape "
                << output_sizes << " and input shape " << input_sizes;
          }
        } else {
          // Arbitrary value to make compiler happy.
          output_mode = AllGatherOutputMode::kConcat;
          TT_CHECK_THROW(false, error::kInvalidArgument)
              << "expected number of output dimensions to be equal the number "
                 "of input dimensions (concatenation) or input dimensions + 1 "
                 "(stacking), got output shape "
              << output_sizes << " and input shape " << input_sizes;
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
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        const int64_t rank = getRank();
        const int64_t root_rank = opts.rootRank;
        TT_CHECK_THROW(input_tensors.size() == 1, error::kInvalidArgument)
            << "expected a single input tensor, got " << input_tensors.size();
        auto& input = input_tensors[0];

        if (rank == root_rank) {
          TT_CHECK_THROW(output_tensors.size() == 1, error::kInvalidArgument)
              << "expected a single list of output tensors on the root rank, "
                 "got "
              << output_tensors.size();
          const auto& output_tensor_list = output_tensors[0];
          TT_CHECK_THROW(output_tensor_list.size() == getSize(),
                         error::kInvalidArgument)
              << "expected the number of output tensors on the root rank to be "
                 "equal to the group size, got "
              << output_tensor_list.size() << " tensors and " << getSize()
              << " processes";
          TT_THROW_IF_ERROR(ValidateTensorsUniformShape(output_tensor_list))
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
            auto tensor = at::empty_like(input);
            // Materialize the outputs before the collective to ensure symmetry.
            // This is needed because on the root rank, output_tensors[0] might
            // have deferred ops. On non-root ranks, the temporary tensors are
            // fresh, which can result in an asymmetry.
            TT_ASSIGN_OR_THROW(auto empty_buffer, GetBaseBuffer(tensor));
            TT_THROW_IF_ERROR(Materialize(
                empty_buffer, MaterializationReason::kDistributedOp));

            allgather_outputs.push_back(std::move(tensor));
          }
        }

        c10d::AllgatherOptions allgather_opts;
        allgather_opts.timeout = opts.timeout;
        allgather_opts.asyncOp = opts.asyncOp;

        std::vector<std::vector<at::Tensor>> allgather_output_tensors = {
            std::move(allgather_outputs)};
        auto work_ptr =
            allgather(allgather_output_tensors, input_tensors, allgather_opts);

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
      (outputs, inputs, IgnoreInCacheKey(opts, "Doesn't affect SHLO")), {
        const int64_t rank = getRank();
        const int64_t root_rank = opts.rootRank;
        TT_CHECK_THROW(outputs.size() == 1, error::kInvalidArgument)
            << "expected a single output tensor, got " << outputs.size();
        auto& output = outputs[0];
        bool is_scalar = output.dim() == 0;

        at::Tensor scatter_input;  // UNINITIALIZED_TENSOR_OK
        if (rank == root_rank) {
          TT_CHECK_THROW(inputs.size() == 1, error::kInvalidArgument)
              << "expected a single list of input tensors on the root rank, "
                 "got "
              << inputs.size();
          const auto& input_tensors = inputs[0];
          TT_CHECK_THROW(input_tensors.size() == getSize(),
                         error::kInvalidArgument)
              << "expected the number of input tensors on the root rank to be "
                 "equal to the group size, got "
              << input_tensors.size() << " tensors and " << getSize()
              << " processes";

          TT_THROW_IF_ERROR(ValidateTensorsUniformShape(input_tensors))
                  .SetPrepend()
              << "input tensors on the root rank: ";
          TT_CHECK_THROW(output.sizes() == input_tensors[0].sizes(),
                         error::kInvalidArgument)
              << "expected output tensor shape to match input tensor shape, "
                 "got "
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
      (output_tensors, input_tensors,
       IgnoreInCacheKey(opts, "Delegates to _reduce_scatter_base")),
      {
        // NOTE: Python side API only exposes single-element reduce_scatter op.
        // Same validation is done in NCCL backend.
        TT_CHECK_THROW(input_tensors.size() == 1,
                       error::kPythonNotImplementedError)
            << "multiple input tensor lists not supported";
        TT_CHECK_THROW(output_tensors.size() == 1,
                       error::kPythonNotImplementedError)
            << "multiple output tensor lists not supported";

        std::vector<at::Tensor>& inputs = input_tensors[0];
        at::Tensor& output = output_tensors[0];

        TT_CHECK_THROW(inputs.size() == getSize(), error::kInvalidArgument)
            << "expected 1 input tensor for each process, got " << inputs.size()
            << " input tensors and " << getSize() << " processes";

        // NOTE: NCCL does support this case, but is doing that by running a
        // sequence of world_size separate reduce calls (coalesced). Revisit
        // later if we want to support this case too. It would be very
        // problematic for error handling.
        TT_THROW_IF_ERROR(ValidateTensorsUniformShape(inputs));

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
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        TT_THROW_IF_ERROR(
            ValidateReductionOp(opts.reduceOp, input.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto input_mode,
            ValidateReduceScatterInputTensorShape(output, input, getSize()));

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
      (outputs, inputs,
       IgnoreInCacheKey(opts, "Delegates to _reduce_scatter_base")),
      {
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
  TT_KERNEL(OpName::kDistributedAllToAllSingle, _,
            (output, input,
             IgnoreInCacheKey(output_split_sizes, "Doesn't affect SHLO"),
             IgnoreInCacheKey(input_split_sizes, "Doesn't affect SHLO"), opts),
            {
              const int64_t rank = getRank();
              const bool async = opts.asyncOp;
              const int64_t group_size = subgroup_device_ids_[0].size();

              // Check on input and output dtypes already done in the
              // PyTorch layer. Hence, we do not need to check for dtypes
              // here.

              TT_THROW_IF_ERROR(ValidateSplitSizesForAllToAllSingle(
                  input_split_sizes, input, group_size));
              TT_THROW_IF_ERROR(ValidateSplitSizesForAllToAllSingle(
                  output_split_sizes, output, group_size));

              const bool equal_input_splits = IsEqualSplits(input_split_sizes);
              const bool equal_output_splits =
                  IsEqualSplits(output_split_sizes);

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
                return c10::make_intrusive<TpuWork>(
                    std::vector<at::Tensor>{output}, rank,
                    c10d::OpType::ALLTOALL_BASE);
              }
              return nullptr;
            });
}

absl::StatusOr<DeviceBufferRef> ProcessGroupTpu::AllToAllBaseEqualSplits(
    at::Tensor& output, at::Tensor& input) {
  auto& maybe_materialized_input_tensor = input;
  TT_RET_CHECK(output.sizes() == input.sizes(), error::kInvalidArgument)
      << "expected shape of input and output to be the same for equal splits, "
      << "got " << input.sizes() << " and " << output.sizes();

  TT_ASSIGN_OR_RETURN(auto param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(subgroup_device_ids_));

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

absl::StatusOr<DeviceBufferRef> ProcessGroupTpu::AllToAllBaseUnevenSplits(
    at::Tensor& output, at::Tensor& input, c10::IntArrayRef output_split_sizes,
    c10::IntArrayRef input_split_sizes) {
  // 1. Validate tensor shapes and rank compatibility.
  // Both input and output must have identical dimension rank (e.g. both 2D),
  // and all trailing feature dimensions (dim 1 onwards) must match exactly.
  auto& maybe_materialized_input_tensor = input;
  TT_RET_CHECK(output.dim() == input.dim(), error::kInvalidArgument)
      << "expected input and output tensors to have the same number of "
      << "dimensions, got " << input.dim() << " and " << output.dim();
  TT_RET_CHECK(output.sizes().slice(1) == input.sizes().slice(1),
               error::kInvalidArgument)
      << "expected trailing dimensions of input and output to match, got "
      << input.sizes() << " and " << output.sizes();

  const size_t group_size = subgroup_device_ids_[0].size();

  // 2. Compute slice sizes for sending and receiving.
  const bool uniform_input = input_split_sizes.empty();
  const bool uniform_output = output_split_sizes.empty();
  const int32_t uniform_input_size =
      uniform_input ? static_cast<int32_t>(input.size(0) / group_size) : 0;
  const int32_t uniform_output_size =
      uniform_output ? static_cast<int32_t>(output.size(0) / group_size) : 0;

  std::vector<int32_t> send_sizes(group_size, 0);
  std::vector<int32_t> recv_sizes(group_size, 0);
  for (size_t i = 0; i < group_size; ++i) {
    send_sizes[i] = uniform_input ? uniform_input_size
                                  : static_cast<int32_t>(input_split_sizes[i]);
    recv_sizes[i] = uniform_output
                        ? uniform_output_size
                        : static_cast<int32_t>(output_split_sizes[i]);
  }

  // 3. Compute cumulative slice offsets and synchronize remote destination
  // output offsets across all ranks via c10d::Store.
  TT_ASSIGN_OR_RETURN(auto offsets,
                      ComputeRaggedAllToAllOffsets(store_.get(), getRank(),
                                                   alltoall_seq_.fetch_add(1),
                                                   send_sizes, recv_sizes));

  TT_ASSIGN_OR_RETURN(auto param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(subgroup_device_ids_));
  TT_RETURN_IF_ERROR(SetRaggedAllToAllParamKeys(param_keys, offsets));

  // 4. Build the MLIR StableHLO custom call with device replica group
  // attributes and MLIR constant metadata tensors.
  auto op_builder =
      [device_groups = subgroup_device_ids_,
       offsets = std::move(offsets)](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [input, output] = inputs;
        return BuildDistributedAllToAllBaseUnevenSplitsShlo(
            input, output, offsets.input_offsets, offsets.send_sizes,
            offsets.output_offsets, offsets.recv_sizes, device_groups);
      };

  TT_ASSIGN_OR_RETURN(auto output_dtype,
                      ConvertTo<mlir::ElementType>(output.scalar_type()));

  // 5. Dispatch the custom call with input and output buffers.
  return DispatchOp<2>(std::move(op_builder),
                       {maybe_materialized_input_tensor, output},
                       {.out_dtype = output_dtype,
                        .out_dims = output.sizes(),
                        .op_param_cache_keys = std::move(param_keys),
                        .split_mode = GetCollectiveSplitMode()});
}

// Dispatches the torch.distributed.all_to_all collective, supporting both
// uniform tensor shapes and non-uniform tensor shapes along dimension 0 for
// dynamic MoE token routing.
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
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        TT_CHECK_THROW(output_tensors.size() == input_tensors.size(),
                       error::kInvalidArgument)
            << "expected output and input tensors to have the same number of "
               "tensors, got "
            << output_tensors.size() << " for output and "
            << input_tensors.size() << " for input";

        const int64_t group_size = getSize();

        TT_CHECK_THROW(input_tensors.size() == group_size,
                       error::kInvalidArgument)
            << "expected input tensors to have the same number of tensors as "
               "the process group size, got "
            << input_tensors.size() << " for input and " << group_size
            << " for process group size";

        TT_THROW_IF_ERROR(
            ValidateAllToAllTensors(output_tensors, input_tensors));

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

        NAryMlirOpBuilder<kDynamicSize, kDynamicSize> op_builder;

        if (IsUniformAllToAll(output_tensors, input_tensors)) {
          op_builder = [device_groups = subgroup_device_ids_](
                           absl::Span<mlir::MlirOp> inputs,
                           mlir::MlirBuilder& builder) {
            return BuildDistributedAllToAllShlo(inputs, device_groups);
          };
        } else {
          // Non-uniform / uneven tensor list shapes across ranks (e.g. for MoE
          // dynamic token routing). All concatenations along dimension 0,
          // ragged all-to-all custom call execution, and slicing back into
          // individual destination tensors are executed directly in MLIR
          // inside the op builder.

          // 1. Calculate local slice sizes and compute synchronized offsets.
          std::vector<int32_t> send_sizes(group_size);
          std::vector<int32_t> recv_sizes(group_size);
          for (size_t i = 0; i < group_size; ++i) {
            send_sizes[i] = static_cast<int32_t>(input_tensors[i].size(0));
            recv_sizes[i] = static_cast<int32_t>(output_tensors[i].size(0));
          }

          // 2. Synchronize remote destination output offsets across all ranks
          // via c10d::Store.
          TT_ASSIGN_OR_THROW(
              auto offsets,
              ComputeRaggedAllToAllOffsets(store_.get(), static_cast<int>(rank),
                                           alltoall_seq_.fetch_add(1),
                                           send_sizes, recv_sizes));

          // 3. Populate kernel parameter cache keys for graph compilation.
          TT_THROW_IF_ERROR(SetRaggedAllToAllParamKeys(param_keys, offsets));

          // 4. Construct the MLIR op builder lowering to the StableHLO ragged
          // all-to-all custom call.
          op_builder = [device_groups = subgroup_device_ids_, output_dtype,
                        offsets = std::move(offsets)](
                           absl::Span<mlir::MlirOp> inputs,
                           mlir::MlirBuilder& builder) {
            return BuildDistributedAllToAllUnevenSplitsShlo(
                inputs, output_dtype, offsets.input_offsets, offsets.send_sizes,
                offsets.output_offsets, offsets.recv_sizes, device_groups);
          };
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
    //
    // A process addresses exactly one TPU, so the only device id it can name
    // is 0 (client->addressable_devices()). Accept that id as a no-op --
    // the barrier below always runs on this process's own device -- so that
    // device-agnostic callers such as
    // torch.distributed.barrier(device_ids=[torch.tpu.current_device()]) work
    // unchanged. Any other id names a device this process does not own.
    TT_CHECK_THROW(opts.device_ids.empty() ||
                       (opts.device_ids.size() == 1 && opts.device_ids[0] == 0),
                   error::kPythonNotImplementedError)
        << "device_ids in barrier options is not supported, except "
           "device_ids=[0] which names this process's own device";
    TT_CHECK_THROW(opts.timeout == c10d::kUnsetTimeout,
                   error::kPythonNotImplementedError)
        << "timeout in barrier options is not supported";
    TT_CHECK_THROW(!opts.device.has_value(), error::kPythonNotImplementedError)
        << "device in barrier options is not supported";

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
    // We do not add subgroup_device_ids_ to param_keys here because this
    // function computes the subgroup_device_ids_.
    // Instead, `world_group` is added to param_keys below.
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
        << "Inconsistent device subgroup. This is a bug.";
    int64_t relative_device_id = GetLogicalDeviceId(addressable_device_id_);
    for (int i = 0; i < group_size; ++i) {
      subgroups.index({relative_device_id, i}) =
          GetLogicalDeviceId(rank_to_device_id_[i]);
    }
    subgroups = subgroups.to(at::device(GetPrivateUse1DeviceType()));

    // Thanks to PJRT, we know all the devices in the TPU slice, so we can
    // perform a collective, without having to know the "default" process
    // group. We just need to reshape device list as [1, N] to use as
    // subgroup list.
    DeviceGroup logical_device_ids(world_size);
    std::iota(logical_device_ids.begin(), logical_device_ids.end(), 0);
    auto world_group = DeviceGroupList({std::move(logical_device_ids)});
    TT_THROW_IF_ERROR(
        param_keys.SetParam("subgroups_fingerprint", world_group));
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
    std::vector<DeviceGroup> result(world_size, DeviceGroup(group_size));
    for (int i = 0; i < world_size; ++i) {
      for (int j = 0; j < group_size; ++j) {
        result[i][j] = subgroups.index({i, j}).item().toLong();
      }
    }

    return DeviceGroupList(std::move(result));
  });
}

int64_t ProcessGroupTpu::GetLogicalDeviceId(int64_t physical_device_id) const {
  auto it = absl::c_find(device_ids_, physical_device_id);
  ABSL_CHECK(it != device_ids_.end())  // CRASH_OK
      << "Could not map physical ID " << physical_device_id
      << " to logical ID.";
  return it - device_ids_.begin();
}

#if TT_TORCH_VERSION_GE(2, 14)
c10::intrusive_ptr<c10d::Window> ProcessGroupTpu::new_window(
    const std::optional<at::Tensor>& tensor) {
  auto win = c10::make_intrusive<WindowTpu>(
      c10::intrusive_ptr<ProcessGroupTpu>::unsafe_reclaim_from_nonowning(this),
      rank_);
  if (tensor.has_value()) {
    win->tensor_register(*tensor, /*owning=*/true);
  }
  return win;
}
#endif

}  // namespace torch_tpu
