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

#include "csrc/ops/dynamic/dynamic_reshape/dynamic_reshape.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/cat/cat.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

at::Tensor DynamicReshape(const at::Tensor& input, at::TensorList shape,
                          at::IntArrayRef static_shape,
                          c10::List<bool> is_dynamic) {
  TT_KERNEL(
      OpName::kDynamicReshape, param_keys,
      (input, shape, static_shape, is_dynamic), {
        const size_t rank = static_shape.size();
        TT_CHECK_THROW(shape.size() == rank, error::kInvalidArgument)
            << "expected shape list size to match static_shape size, got shape "
               "list "
               "size "
            << shape.size() << " and static_shape size " << rank;
        TT_CHECK_THROW(is_dynamic.size() == rank, error::kInvalidArgument)
            << "expected is_dynamic size to match static_shape size, got "
               "is_dynamic "
               "size "
            << is_dynamic.size() << " and static_shape size " << rank;

        for (size_t i = 0; i < shape.size(); ++i) {
          const auto& t = shape[i];
          TT_CHECK_THROW(t.dim() == 0, error::kInvalidArgument)
              << "expected shape tensor at index " << i
              << " to be a 0-D (scalar) tensor, got " << t.dim() << "-D tensor";
          TT_CHECK_THROW(t.scalar_type() == at::kInt, error::kInvalidArgument)
              << "expected shape to be a list of int32 tensors, got "
              << ToString(t.scalar_type()) << " tensor at index " << i;
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        const auto out_dims = CopyIntVector(static_shape);
        const std::vector<bool> is_dynamic_vec(is_dynamic.begin(),
                                               is_dynamic.end());

        // Concatenate input tensor and shape descriptors
        std::vector<at::Tensor> all_inputs;
        all_inputs.reserve(1 + shape.size());
        all_inputs.push_back(input);
        for (const auto& t : shape) {
          all_inputs.push_back(t);
        }

        auto builder =
            [shape_len = shape.size(), out_dims, is_dynamic_vec](
                absl::Span<mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          // 1. Process and concatenate shape inputs to a single 1D tensor
          std::vector<mlir::MlirOp> shape_ops;
          shape_ops.reserve(shape_len);

          for (size_t i = 0; i < shape_len; ++i) {
            mlir::MlirOp dim_op = inputs[1 + i];
            auto dim_type = GetTensorTypeOrDie(dim_op);
            if (dim_type.getRank() == 0) {
              dim_op = mlir::stablehlo::Reshape(dim_op, {1});
            }
            shape_ops.push_back(dim_op);
          }

          TT_ASSIGN_OR_RETURN(mlir::MlirOp concatenated_shape,
                              BuildCatShlo(absl::MakeSpan(shape_ops), 0));

          // 2. Prepare MLIR dimensions and bounds
          Dimensions mlir_out_dims;
          mlir_out_dims.reserve(out_dims.size());
          Dimensions bounds;
          bounds.reserve(out_dims.size());

          for (size_t i = 0; i < out_dims.size(); ++i) {
            if (is_dynamic_vec[i]) {
              // Dynamic dimension: placeholder in shape, bounded by
              // static_shape
              mlir_out_dims.push_back(mlir::ShapedType::kDynamic);
              bounds.push_back(out_dims[i]);
            } else {
              // Static dimension: explicit size, no variable upper bound
              mlir_out_dims.push_back(out_dims[i]);
              bounds.push_back(mlir::ShapedType::kDynamic);
            }
          }

          // 3. Attach Bounds as Type Extensions Encoding
          auto bounds_attr = mlir::stablehlo::TypeExtensionsAttr::get(
              &builder.getContext(), bounds);

          auto result_type = mlir::RankedTensorType::get(
              mlir_out_dims, GetTensorTypeOrDie(inputs[0]).getElementType(),
              bounds_attr);

          // 4. Emit the StableHLO op
          return mlir::stablehlo::DynamicReshape(result_type, inputs[0],
                                                 concatenated_shape);
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
