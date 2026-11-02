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

#include "csrc/ops/dynamic/pad_real_to_static/pad_real_to_static.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

// Builds the StableHLO PadRealToStatic custom call.
//
// Expected contract of the "PadRealToStatic" custom call:
// - Call target: "PadRealToStatic"
// - Side effects: false (pure computation)
// - API version: API_VERSION_ORIGINAL
// - Operands:
//     - inputs[0]: The input tensor (bounded dynamic tensor or real tensor).
//     - inputs[1 + i] (for i in [0, rank - 1]): Scalar int32 tensors
//       specifying the runtime dynamic dimension size for each dimension i.
// - Result type:
//     - Statically shaped ranked tensor with dimensions equal to `out_dims`
//       (the static upper bound shape) and element type matching `inputs[0]`.
//       On TPU, the custom call pads the physical tensor representation up
//       to the static bounds.
mlir::stablehlo::CustomCallOp BuildPadRealToStaticCustomCall(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs,
    llvm::ArrayRef<int64_t> out_dims) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const auto static_result_type = mlir::RankedTensorType::get(
      out_dims, GetTensorTypeOrDie(inputs[0]).getElementType());

  const mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
      "call_target_name", op_builder.getStringAttr("PadRealToStatic"));
  const mlir::NamedAttribute has_side_effect_attr =
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(false));
  const auto api_version_attr = op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          &builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));

  std::vector<mlir::Value> operands;
  operands.reserve(inputs.size());
  for (mlir::MlirOp input : inputs) {
    operands.push_back(input.getValue());
  }

  return mlir::stablehlo::CustomCallOp::create(
      op_builder, builder.getLoc(),
      /*resultTypes=*/{static_result_type},
      /*operands=*/operands,
      {call_target_attr, has_side_effect_attr, api_version_attr});
}

}  // namespace

at::Tensor PadRealToStatic(const at::Tensor& input,
                           at::TensorList runtime_shape,
                           at::IntArrayRef static_shape,
                           c10::List<bool> is_dynamic) {
  TT_KERNEL(
      OpName::kPadRealToStatic, param_keys,
      (input, runtime_shape, static_shape, is_dynamic), {
        const size_t rank = static_shape.size();
        const int64_t input_rank = input.dim();
        TT_CHECK_THROW(input_rank == static_cast<int64_t>(rank),
                       error::kInvalidArgument)
            << "expected input rank to match static_shape size, got input rank "
            << input_rank << " and static_shape size " << rank;
        TT_CHECK_THROW(runtime_shape.size() == rank, error::kInvalidArgument)
            << "expected runtime_shape list size to match static_shape size, "
               "got runtime_shape list size "
            << runtime_shape.size() << " and static_shape size " << rank;
        TT_CHECK_THROW(is_dynamic.size() == rank, error::kInvalidArgument)
            << "expected is_dynamic size to match static_shape size, got "
               "is_dynamic size "
            << is_dynamic.size() << " and static_shape size " << rank;

        for (size_t i = 0; i < runtime_shape.size(); ++i) {
          const auto& t = runtime_shape[i];
          TT_CHECK_THROW(t.dim() == 0, error::kInvalidArgument)
              << "expected runtime_shape tensor at index " << i
              << " to be a 0-D (scalar) tensor, got " << t.dim() << "-D tensor";
          TT_CHECK_THROW(t.scalar_type() == at::kInt, error::kInvalidArgument)
              << "expected runtime_shape to be a list of int32 tensors, got "
              << ToString(t.scalar_type()) << " tensor at index " << i;
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        const auto out_dims = CopyIntVector(static_shape);
        const std::vector<bool> is_dynamic_vec(is_dynamic.begin(),
                                               is_dynamic.end());

        // Concatenate input tensor and runtime shape descriptors
        std::vector<at::Tensor> all_inputs;
        all_inputs.reserve(1 + runtime_shape.size());
        all_inputs.push_back(input);
        for (const auto& t : runtime_shape) {
          all_inputs.push_back(t);
        }

        auto builder = [out_dims, is_dynamic_vec](
                           absl::Span<mlir::MlirOp> inputs,
                           mlir::MlirBuilder& builder) {
          auto custom_call_op =
              BuildPadRealToStaticCustomCall(builder, inputs, out_dims);

          mlir::MlirOp current_op(builder, custom_call_op.getResult(0));

          // Set dimension sizes for dynamic dimensions
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
