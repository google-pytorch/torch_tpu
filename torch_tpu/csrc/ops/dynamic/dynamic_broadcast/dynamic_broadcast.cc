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

#include "torch_tpu/csrc/ops/dynamic/dynamic_broadcast/dynamic_broadcast.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"

namespace torch_tpu {

at::Tensor DynamicBroadcast(const at::Tensor& input, at::TensorList shape,
                            at::IntArrayRef broadcast_dims,
                            at::IntArrayRef static_shape,
                            c10::List<bool> is_dynamic) {
  TT_KERNEL(
      OpName::kDynamicBroadcast, param_keys,
      (input, shape, broadcast_dims, static_shape, is_dynamic), {
        const size_t rank = static_shape.size();
        TT_CHECK_THROW(shape.size() == rank, error::kInvalidArgument)
            << "expected shape list size to match static_shape size, got "
            << "shape list size " << shape.size() << " and static_shape size "
            << rank;
        TT_CHECK_THROW(is_dynamic.size() == rank, error::kInvalidArgument)
            << "expected is_dynamic size to match static_shape size, got "
            << "is_dynamic size " << is_dynamic.size()
            << " and static_shape size " << rank;

        for (size_t i = 0; i < shape.size(); ++i) {
          const auto& t = shape[i];
          TT_CHECK_THROW(t.dim() == 0, error::kInvalidArgument)
              << "expected shape tensor at index " << i
              << " to be a 0-D (scalar) tensor, got " << t.dim() << "-D tensor";
          TT_CHECK_THROW(t.scalar_type() == at::kInt, error::kInvalidArgument)
              << "expected shape to be a list of int32 tensors, got "
              << ToString(t.scalar_type()) << " tensor at index " << i;
        }

        const int64_t input_rank = input.dim();
        TT_CHECK_THROW(broadcast_dims.size() == input_rank,
                       error::kInvalidArgument)
            << "expected broadcast_dims size to match input rank, got "
            << "broadcast_dims size " << broadcast_dims.size()
            << " and input rank " << input_rank;

        for (size_t i = 0; i < broadcast_dims.size(); ++i) {
          const int64_t dim = broadcast_dims[i];
          TT_CHECK_THROW(dim >= 0 && dim < static_cast<int64_t>(rank),
                         error::kInvalidArgument)
              << "expected broadcast_dims to be in range [0, " << rank
              << "), got " << dim << " for broadcast dim at index " << i;
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        const auto out_dims = CopyIntVector(static_shape);
        const std::vector<bool> is_dynamic_vec(is_dynamic.begin(),
                                               is_dynamic.end());
        const auto bcast_dims = CopyIntVector(broadcast_dims);

        // Concatenate input tensor and shape descriptors
        std::vector<at::Tensor> all_inputs;
        all_inputs.reserve(1 + shape.size());
        all_inputs.push_back(input);
        for (const auto& t : shape) {
          all_inputs.push_back(t);
        }

        auto builder =
            [shape_len = shape.size(), out_dims, is_dynamic_vec, bcast_dims](
                absl::Span<mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          // 1. Broadcast to static shape (bounds)
          auto static_result_type = mlir::RankedTensorType::get(
              out_dims, GetTensorTypeOrDie(inputs[0]).getElementType());

          auto broadcast_op = mlir::stablehlo::BroadcastInDim(
              static_result_type, inputs[0], bcast_dims);

          // 2. Set dimension sizes for dynamic dimensions
          auto current_op = broadcast_op;
          for (size_t i = 0; i < out_dims.size(); ++i) {
            if (is_dynamic_vec[i]) {
              current_op = mlir::stablehlo::SetDimensionSize(current_op,
                                                             inputs[1 + i], i);
            }
          }
          return current_op;
        };

        TT_ASSIGN_OR_THROW(auto result_buf,
                           DispatchOp<kDynamicSize>(
                               std::move(builder), all_inputs,
                               {.out_dtype = mlir_dtype,
                                .out_dims = out_dims,
                                .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result_buf));
      });
}

}  // namespace torch_tpu
