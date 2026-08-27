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

#include "torch_tpu/csrc/ops/dynamic/dynamic_slice/dynamic_slice.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
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

at::Tensor DynamicSlice(const at::Tensor& input, at::TensorList start_indices,
                        at::IntArrayRef slice_sizes) {
  TT_KERNEL(
      OpName::kDynamicSlice, param_keys, (input, start_indices, slice_sizes), {
        const size_t input_dims = input.dim();
        TT_CHECK_THROW(start_indices.size() == input_dims,
                       error::kInvalidArgument)
            << "expected start_indices list size to match input number of "
               "dimensions, got start_indices size "
            << start_indices.size() << " and input number of dimensions "
            << input_dims;
        TT_CHECK_THROW(slice_sizes.size() == input_dims,
                       error::kInvalidArgument)
            << "expected slice_sizes size to match input number of dimensions, "
               "got slice_sizes size "
            << slice_sizes.size() << " and input number of dimensions "
            << input_dims;

        for (size_t i = 0; i < start_indices.size(); ++i) {
          const auto& t = start_indices[i];
          TT_CHECK_THROW(t.dim() == 0, error::kInvalidArgument)
              << "expected start_indices tensor at index " << i
              << " to be a 0-D (scalar) tensor, got " << t.dim() << "-D tensor";
          TT_CHECK_THROW(
              t.scalar_type() == at::kInt || t.scalar_type() == at::kLong,
              error::kInvalidArgument)
              << "expected start_indices to be a list of int32 or int64 "
                 "tensors, got "
              << ToString(t.scalar_type()) << " tensor at index " << i;
          TT_CHECK_THROW(t.scalar_type() == start_indices[0].scalar_type(),
                         error::kInvalidArgument)
              << "expected all start_indices to have the same dtype, got "
              << ToString(start_indices[0].scalar_type()) << " at index 0 but "
              << ToString(t.scalar_type()) << " at index " << i;
        }

        for (size_t i = 0; i < slice_sizes.size(); ++i) {
          TT_CHECK_THROW(slice_sizes[i] >= 0 && slice_sizes[i] <= input.size(i),
                         error::kInvalidArgument)
              << "expected slice_sizes at index " << i << " to be in range [0, "
              << input.size(i) << "], got " << slice_sizes[i];
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        const auto out_dims = CopyIntVector(slice_sizes);

        // Concatenate input tensor and start_indices
        std::vector<at::Tensor> all_inputs;
        all_inputs.reserve(1 + start_indices.size());
        all_inputs.push_back(input);
        for (const auto& t : start_indices) {
          all_inputs.push_back(t);
        }

        auto builder = [input_dims, out_dims](absl::Span<mlir::MlirOp> inputs,
                                              mlir::MlirBuilder& mlir_builder)
            -> absl::StatusOr<mlir::MlirOp> {
          std::vector<mlir::MlirOp> start_ops;
          start_ops.reserve(input_dims);
          for (size_t i = 0; i < input_dims; ++i) {
            start_ops.push_back(inputs[1 + i]);
          }

          return mlir::stablehlo::DynamicSlice(inputs[0], start_ops, out_dims);
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
