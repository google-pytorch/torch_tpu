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

#include "csrc/ops/experimental/collectives/collectives_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue_inl.h"
#include "ATen/ops/empty.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/DimVector.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/const_tensor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/distributed/allgather.h"
#include "csrc/distributed/process_group_tpu.h"
#include "csrc/distributed/reduce_scatter.h"
#include "csrc/distributed/types.h"
#include "csrc/distributed/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/GroupRegistry.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// The validated collective dimension and the resulting output shape.
struct CollectiveShape {
  // The collective dimension, normalized to [0, input.dim()).
  int64_t dim = 0;
  // Symbolic if the input is a FakeTensor with symbolic sizes.
  c10::SymDimVector output_shape;
};

// Takes ConstTensor so that the validation doesn't specialize the symbolic
// shapes of the FakeTensors that torch.compile traces with.
absl::StatusOr<CollectiveShape> ValidateArgs(const ConstTensor& input,
                                             int64_t dim, int64_t group_size) {
  TT_ASSIGN_OR_RETURN(const int64_t rank, input.dim());
  TT_RET_CHECK(rank > 0, error::kInvalidArgument)
      << "expected input to have at least 1 dimension, got 0";
  TT_ASSIGN_OR_RETURN(const int64_t wrapped_dim, SafeWrapDim(dim, rank));
  TT_RET_CHECK(group_size > 0, error::kInvalidArgument)
      << "expected group_size to be > 0, got " << group_size;
  TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef sizes, input.sym_sizes());
  return CollectiveShape{
      .dim = wrapped_dim,
      .output_shape = c10::SymDimVector(sizes.begin(), sizes.end())};
}

absl::StatusOr<CollectiveShape> AllGatherShape(const at::Tensor& input,
                                               int64_t dim,
                                               int64_t group_size) {
  TT_ASSIGN_OR_RETURN(CollectiveShape shape,
                      ValidateArgs(input, dim, group_size));
  shape.output_shape[shape.dim] *= group_size;
  return shape;
}

absl::StatusOr<CollectiveShape> ReduceScatterShape(const at::Tensor& input,
                                                   int64_t dim,
                                                   int64_t group_size) {
  TT_ASSIGN_OR_RETURN(CollectiveShape shape,
                      ValidateArgs(input, dim, group_size));
  c10::SymInt& size = shape.output_shape[shape.dim];
  // For a symbolic size, this adds a divisibility guard but doesn't specialize.
  TT_RET_CHECK(size % group_size == 0, error::kInvalidArgument)
      << "expected the size of dim " << shape.dim
      << " to be divisible by group_size " << group_size << ", got " << size;
  size /= group_size;
  return shape;
}

absl::StatusOr<c10d::ReduceOp::RedOpType> ParseReduceOp(
    std::string_view reduce_op, c10::ScalarType scalar_type) {
  std::optional<c10d::ReduceOp::RedOpType> parsed;
  std::vector<std::string> supported;
  // PREMUL_SUM and later ops can't be constructed from a bare RedOpType.
  for (int i = 0; i < c10d::ReduceOp::PREMUL_SUM; ++i) {
    const auto op = static_cast<c10d::ReduceOp::RedOpType>(i);
    if (supported.emplace_back(ToString(op)) == reduce_op) {
      parsed = op;
    }
  }
  TT_RET_CHECK(parsed.has_value(), error::kInvalidArgument)
      << "expected reduce_op to be one of " << absl::StrJoin(supported, ", ")
      << ", got " << reduce_op;
  TT_RETURN_IF_ERROR(ValidateReductionOp(*parsed, scalar_type));
  return *parsed;
}

// resolve_process_group() and getBackend() throw on an unknown or destroyed
// group or a group without a TPU backend, so they never return null. The TPU
// backend can still be a wrapper around ProcessGroupTpu, e.g. under
// TORCH_DISTRIBUTED_DEBUG=DETAIL or TorchComms.
absl::StatusOr<c10::intrusive_ptr<ProcessGroupTpu>> GetProcessGroupTpu(
    std::string_view group_name, int64_t group_size) {
  c10::intrusive_ptr<c10d::ProcessGroup> pg =
      c10d::resolve_process_group(std::string(group_name));
  ABSL_CHECK(pg != nullptr);  // CRASH_OK
  // The output shape comes from group_size, while the lowered collective's
  // shape comes from the process group.
  TT_RET_CHECK(group_size == pg->getSize(), error::kInvalidArgument)
      << "expected group_size to match process group size, got " << group_size
      << " vs " << pg->getSize();
  c10::intrusive_ptr<c10d::Backend> backend =
      pg->getBackend(c10::DeviceType::PrivateUse1);
  ABSL_CHECK(backend != nullptr);  // CRASH_OK
  auto process_group_tpu =
      c10::dynamic_intrusive_pointer_cast<ProcessGroupTpu>(std::move(backend));
  TT_RET_CHECK(process_group_tpu != nullptr, error::kInvalidArgument)
      << "expected the TPU backend of process group " << group_name
      << " to be an unwrapped ProcessGroupTpu, got a wrapped or non-TPU "
         "backend";
  return process_group_tpu;
}

template <typename CollectiveBuilder>
absl::StatusOr<DeviceBufferRef> DispatchCollectiveWithDim(
    const at::Tensor& input, int64_t group_size, Dimensions output_shape,
    std::string_view group_name, CollectiveBuilder collective_builder,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(auto process_group,
                      GetProcessGroupTpu(group_name, group_size));
  const DeviceGroupList& device_groups = process_group->GetReplicaGroups();
  TT_ASSIGN_OR_RETURN(auto output_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));

  auto op_builder = [collective_builder, device_groups](mlir::MlirOp input_op) {
    return collective_builder(input_op, device_groups);
  };

  return DispatchOp<1>(
      std::move(op_builder), input,
      {.out_dtype = output_dtype,
       .out_dims = std::move(output_shape),
       .op_param_cache_keys = std::move(param_keys),
       .split_mode = GetCollectiveSplitMode(),
       .device_assignment = process_group->device_assignment()});
}

}  // namespace

at::Tensor AllGatherDim(const at::Tensor& input, int64_t dim,
                        int64_t group_size, std::string_view group_name) {
  TT_KERNEL(
      OpName::kDistributedAllGatherDim, param_keys,
      (input, dim, group_size, group_name), {
        TT_ASSIGN_OR_THROW(auto shape, AllGatherShape(input, dim, group_size));
        auto builder = [dim = shape.dim](mlir::MlirOp input_op,
                                         const DeviceGroupList& device_groups)
            -> absl::StatusOr<mlir::MlirOp> {
          TT_ASSIGN_OR_RETURN(
              auto outputs,
              BuildDistributedAllGatherShlo(input_op, device_groups,
                                            AllGatherOutputMode::kConcat, dim));
          return outputs[0];
        };
        TT_ASSIGN_OR_THROW(Dimensions output_dims,
                           ConcreteSizeToDimensions(shape.output_shape));
        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchCollectiveWithDim(input, group_size, std::move(output_dims),
                                      group_name, std::move(builder),
                                      std::move(param_keys)));
        return MakeTensor(std::move(result));
      });
}

at::Tensor ReduceScatterDim(const at::Tensor& input, std::string_view reduce_op,
                            int64_t dim, int64_t group_size,
                            std::string_view group_name) {
  TT_KERNEL(OpName::kDistributedReduceScatterDim, param_keys,
            (input, reduce_op, dim, group_size, group_name), {
              TT_ASSIGN_OR_THROW(auto op,
                                 ParseReduceOp(reduce_op, input.scalar_type()));
              TT_ASSIGN_OR_THROW(auto shape,
                                 ReduceScatterShape(input, dim, group_size));
              auto builder = [op, dim = shape.dim](
                                 mlir::MlirOp input_op,
                                 const DeviceGroupList& device_groups) {
                return BuildDistributedReduceScatterShlo(input_op, op,
                                                         device_groups, dim);
              };
              TT_ASSIGN_OR_THROW(Dimensions output_dims,
                                 ConcreteSizeToDimensions(shape.output_shape));
              TT_ASSIGN_OR_THROW(
                  auto result,
                  DispatchCollectiveWithDim(
                      input, group_size, std::move(output_dims), group_name,
                      std::move(builder), std::move(param_keys)));
              return MakeTensor(std::move(result));
            });
}

at::Tensor MetaAllGatherDim(const at::Tensor& input, int64_t dim,
                            int64_t group_size,
                            std::string_view /*group_name*/) {
  TT_ASSIGN_OR_THROW(auto shape, AllGatherShape(input, dim, group_size));
  return at::empty_symint(shape.output_shape, input.options());
}

at::Tensor MetaReduceScatterDim(const at::Tensor& input,
                                std::string_view reduce_op, int64_t dim,
                                int64_t group_size,
                                std::string_view /*group_name*/) {
  TT_THROW_IF_ERROR(ParseReduceOp(reduce_op, input.scalar_type()).status());
  TT_ASSIGN_OR_THROW(auto shape, ReduceScatterShape(input, dim, group_size));
  return at::empty_symint(shape.output_shape, input.options());
}

}  // namespace torch_tpu
