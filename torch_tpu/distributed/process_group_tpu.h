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

#ifndef TORCH_TPU_DISTRIBUTED_PROCESS_GROUP_TPU_H_
#define TORCH_TPU_DISTRIBUTED_PROCESS_GROUP_TPU_H_

// A c10d::Backend implementation for TPUs.
// All the PyTorch collective operations for tensors on TPU device will be
// delegated to methods of this class.
//
// NOTE: We currently implement only the "non-functional" collectives.
// PyTorch has a default implementation for functional collectives via
// "non-functional" ones, but as some point we may want to specialize.

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue_inl.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Store.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/distributed/types.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_names.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"

namespace torch_tpu {

class ProcessGroupTpu : public c10d::Backend {
 public:
  // The standard pattern for PyTorch distributed backend initialization is
  // to use a distributed key-value store to exchange various information
  // required to set up a particular communication backend.
  //
  // Rank and size (also known as "process group size", or "world size")
  // are concepts defined in PyTorch, see
  // https://docs.pytorch.org/docs/stable/distributed.html
  //
  // Note that PyTorch rank is not necessarily the same as TPU global device id.
  ProcessGroupTpu(c10::intrusive_ptr<c10d::Store> store, int rank,
                  int group_size);

  // This class is neither copyable nor movable.
  ProcessGroupTpu(const ProcessGroupTpu&) = delete;
  ProcessGroupTpu& operator=(const ProcessGroupTpu&) = delete;
  ProcessGroupTpu(ProcessGroupTpu&&) = delete;
  ProcessGroupTpu& operator=(ProcessGroupTpu&&) = delete;

  ~ProcessGroupTpu() override = default;

  const std::string getBackendName() const override { return "tpu_dist"; }

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& output_tensor, at::Tensor& input_tensor,
      const c10d::AllgatherOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
      const c10d::AllgatherOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts) override;

  // Scatters a list of tensors from the root process to other processes
  // in the group such that each process gets exactly one tensor.
  // This function is invoked by torch.distributed.scatter.
  //
  // Args:
  //    outputs: A one-element list containing a tensor to store the result in.
  //    inputs: On the root rank process this is a single-element list that
  //      contains the list of tensors to scatter. On other processes it is
  //      an empty list.
  //    opts: Options: the root rank, timeout (currently ignored), and whether
  //      or not to return the result asynchronously.
  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const c10d::ScatterOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ReduceScatterOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output, at::Tensor& input,
      const c10d::ReduceScatterOptions& opts) override;

  // Coalesces multiple independent reduce_scatter operations.
  // This is not used directly by any torch.distributed collective op.
  // It is invoked by the default implementation of the functional version
  // of reduce_scatter. It could also be invoked via the coalescing manager.
  //
  // Args:
  //    outputs: A list of output tensors in which to store the result.
  //    inputs: A list of input tensors to be reduced and scattered.
  //    opts: Options: reduction type, timeout (currently ignored), and
  //      whether or not to return the result asynchronously.
  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
      const c10d::ReduceScatterOptions& opts) override;

  // Splits the input tensor and then scatters the split list to all processes
  // in the group. Later the received tensors are concatenated from all the
  // processes in the group and returned as a single output tensor.
  //
  //  PyTorch API: torch.distributed.all_to_all_single
  //
  //  Args:
  //     output: Gathered concatenated output tensor.
  //     input: Input tensor to scatter.
  //     output_split_sizes: Output split sizes for dim 0. if empty, dim 0
  //         of output tensor must divide equally by world_size.
  //     input_split_sizes: Input split sizes for dim 0. if empty, dim 0
  //         of input tensor must divide equally by world_size.
  //     opts: Collective options.
  //
  //
  //  Example 1: Equal splits
  //  World size = 4
  //  input
  //         tensor([0, 1, 2, 3])     # Rank 0
  //         tensor([4, 5, 6, 7])     # Rank 1
  //         tensor([8, 9, 10, 11])   # Rank 2
  //         tensor([12, 13, 14, 15]) # Rank 3
  //  input_splits (empty)
  //  output_splits (empty)
  //  output
  //         tensor([0, 4, 8, 12])    # Rank 0
  //         tensor([1, 5, 9, 13])    # Rank 1
  //         tensor([2, 6, 10, 14])   # Rank 2
  //         tensor([3, 7, 11, 15])   # Rank 3
  //
  //  Example 2: Uneven splits. (Currently not supported)
  //  World size = 4
  //  input
  //         tensor([0, 1, 2, 3, 4, 5])                       # Rank 0
  //         tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])     # Rank 1
  //         tensor([20, 21, 22, 23, 24])                     # Rank 2
  //         tensor([30, 31, 32, 33, 34, 35, 36])             # Rank 3
  //  input_splits
  //         [2, 2, 1, 1]                                     # Rank 0
  //         [3, 2, 2, 2]                                     # Rank 1
  //         [2, 1, 1, 1]                                     # Rank 2
  //         [2, 2, 2, 1]                                     # Rank 3
  //  output_splits
  //         [2, 3, 2, 2]                                     # Rank 0
  //         [2, 2, 1, 2]                                     # Rank 1
  //         [1, 2, 1, 2]                                     # Rank 2
  //         [1, 2, 1, 1]                                     # Rank 3
  //  output
  //         tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])     # Rank 0
  //         tensor([ 2,  3, 13, 14, 22, 32, 33])             # Rank 1
  //         tensor([ 4, 15, 16, 23, 34, 35])                 # Rank 2
  //         tensor([ 5, 17, 18, 24, 36])                     # Rank 3
  //
  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output, at::Tensor& input,
      std::vector<int64_t>& output_split_sizes,  // INT_VEC_OK
      std::vector<int64_t>& input_split_sizes,   // INT_VEC_OK
      const c10d::AllToAllOptions& opts) override;

  // Scatters list of input tensors to all processes in a group and returns
  // gathered list of tensors in output list.
  //
  //   PyTorch API: torch.distributed.all_to_all
  //
  //   Args:
  //       output_tensors: List of tensors to be gathered one per rank
  //       input_tensors: List of tensors to scatter one per rank
  //
  //   Example: Uniform shape of input and output tensors.
  //     input
  //       [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
  //       [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
  //       [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
  //       [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
  //     output
  //       [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
  //       [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
  //       [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
  //       [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3
  //
  //   Note: The current implementation requires all input and output tensors
  //   to be of uniform shape. Non-uniform shape is not supported.
  //   TODO(mkkhanna): Support tensors of non-uniform shape.
  //   b/447429739
  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts) override;

  // Experimental P2P send. Mirrors the semantics of c10d::Backend::send,
  // but is kept isolated from the public torch.distributed API to safely
  // prototype TPU-specific behaviors.
  c10::intrusive_ptr<c10d::Work> experimental_send(
      std::vector<at::Tensor>& tensors, int dst_rank, int tag);

  // Experimental P2P recv. Mirrors the semantics of c10d::Backend::recv,
  // but is kept isolated from the public torch.distributed API to safely
  // prototype TPU-specific behaviors.
  c10::intrusive_ptr<c10d::Work> experimental_recv(
      std::vector<at::Tensor>& tensors, int src_rank, int tag);

 private:
  // Differently from PyTorch distributed APIs, XLA requires that all processes
  // supply *all* the device id subgroups (in replica_groups attribute), even
  // the subgroups that this process isn't a member of. This helper function
  // uses the established TPU connectivity within the slice to directly exchange
  // this information. This is called only once during ProcessGroup constructor.
  DeviceGroupList GatherAllSubgroups();

  // Helper to reduce logging boilerplate. Must use only from implementation
  // methods of collective ops, so we know everything here is initialized.
  std::string OpDebugString(std::string_view method) {
    return absl::StrFormat(
        "[ProcessGroupTpu::%s] rank=%d, device=%d, group_devices=%s "
        "(world_size=%d): ",
        method, getRank(), addressable_device_id_, ToString(rank_to_device_id_),
        device_ids_.size());
  }

  std::string OpDebugString(OpName method) {
    std::stringstream ss;
    ss << method;
    return OpDebugString(ss.str());
  }

  // Helper functions to dispatch the all_to_all_single collective.
  absl::StatusOr<DeviceBufferRef> AllToAllBaseEqualSplits(at::Tensor& output,
                                                          at::Tensor& input);

  absl::StatusOr<DeviceBufferRef> AllToAllBaseUnevenSplits(
      at::Tensor& output, at::Tensor& input,
      const std::vector<int64_t>& output_split_sizes,  // INT_VEC_OK
      const std::vector<int64_t>& input_split_sizes);  // INT_VEC_OK

 private:
  // Extracts receive descriptors from store_ and sends buffers to a
  // remote device.
  absl::StatusOr<c10::intrusive_ptr<c10::ivalue::Future>> CrossHostSendBuffers(
      std::vector<xla::PjRtBuffer*>& buffers,
      absl::Span<const xla::CrossHostTransferKey> transfer_keys);

  // Populates store_ with receive descriptors and places buffers
  // from a cross-host send onto device.
  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
  CrossHostReceiveBuffers(
      absl::Span<const xla::Shape> shapes,
      absl::Span<const xla::CrossHostTransferKey> transfer_keys);

  c10::intrusive_ptr<c10d::Store> store_;

  std::vector<int64_t> device_ids_;  // All the TPUs in the slice. INT_VEC_OK
  int64_t addressable_device_id_;    // The TPU 'assigned' to this process.

  // The unique (local) identifier of this ProcessGroup.
  //
  // This should be included in the op param cache keys, to distinguish
  // different process groups that may have the same op arguments.
  const int64_t pg_id_ = this->getID();

  // List of TPU device ids in this process group (which could be a subgroup),
  // ordered by the corresponding PyTorch rank value.
  DeviceGroup rank_to_device_id_;

  // All the TPU device ids, split into subgroups (including the trivial world
  // group). This includes subgroups this process isn't a member of, but which
  // are required to be known by XLA. This effectively becomes our
  // replica_groups attribute in StableHLO collectives.
  DeviceGroupList subgroup_device_ids_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_DISTRIBUTED_PROCESS_GROUP_TPU_H_
