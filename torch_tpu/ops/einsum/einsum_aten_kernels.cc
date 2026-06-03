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

#include "torch_tpu/ops/einsum/einsum_aten_kernels.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/RedispatchFunctions.h"
#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/op_name_stack.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

class ScopedSuspendOpName {
 public:
  ScopedSuspendOpName() {
    top_ = ::torch_tpu::internal::OpNameStack::MaybeTop();
    if (top_.has_value()) {
      ::torch_tpu::internal::OpNameStack::Pop();
    }
  }
  ~ScopedSuspendOpName() {
    if (top_.has_value()) {
      ::torch_tpu::internal::OpNameStack::Push(*top_);
    }
  }

 private:
  std::optional<OpName> top_;
};

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildEinsumShlo(std::string_view equation,
                                             absl::Span<mlir::MlirOp> inputs,
                                             mlir::Type output_type) {
  if (inputs.empty()) {
    return TT_ERROR(error::kInvalidArgument)
           << "einsum requires at least 1 operand";
  }

  mlir::MlirBuilder& builder = inputs[0].getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::Location loc = builder.getLoc();

  llvm::SmallVector<mlir::Value> mlir_inputs;
  for (const auto& input : inputs) {
    mlir_inputs.push_back(input.getValue());
  }

  mlir::StringAttr equation_attr = op_builder.getStringAttr(equation);

  mlir::Value result_val;
  if (mlir_inputs.size() == 1) {
    auto op = op_builder.create<mlir::stablehlo::UnaryEinsumOp>(
        loc, output_type, mlir_inputs[0], equation_attr);
    result_val = op.getResult();
  } else if (mlir_inputs.size() == 2) {
    auto op = op_builder.create<mlir::stablehlo::EinsumOp>(
        loc, output_type, mlir_inputs[0], mlir_inputs[1], equation_attr);
    result_val = op.getResult();
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "stablehlo.einsum supports 1 or 2 operands, got "
           << mlir_inputs.size();
  }

  return mlir::MlirOp(builder, result_val);
}

at::Tensor AtenEinsumMeta(std::string_view equation, at::TensorList tensors,
                          at::OptionalIntArrayRef path) {
  c10::DispatchKeySet composite_keyset =
      c10::DispatchKeySet(c10::DispatchKey::CompositeImplicitAutograd);
  return at::redispatch::einsum(composite_keyset, equation, tensors, path);
}

at::Tensor AtenEinsum(std::string_view equation, at::TensorList tensors,
                      at::OptionalIntArrayRef path) {
  TT_KERNEL(OpName::kEinsum, param_keys, (equation, tensors, path), {
    at::Tensor meta_out = [&]() {
      ScopedSuspendOpName suspend_guard;
      return AtenEinsumMeta(equation, tensors, path);
    }();
    Dimensions output_dims_vec(meta_out.sizes().begin(),
                               meta_out.sizes().end());
    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype_mlir,
                       ConvertTo<mlir::ElementType>(meta_out.scalar_type()));

    TT_ASSIGN_OR_THROW(auto updated_param_keys,
                       *OpParamCacheKeysBuilder(std::move(param_keys))
                            .SetParam("eq", std::string(equation)));

    auto op_builder =
        [equation = std::string(equation), output_dims_vec, output_dtype_mlir](
            absl::Span<mlir::MlirOp> inputs,
            mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
      mlir::Type output_type = mlir::makeTensorType(
          builder.getContext(), output_dims_vec, output_dtype_mlir);
      TT_ASSIGN_OR_RETURN(auto result,
                          BuildEinsumShlo(equation, inputs, output_type));
      return result;
    };

    std::vector<at::Tensor> inputs_vec(tensors.begin(), tensors.end());

    TT_ASSIGN_OR_THROW(
        auto result_buffer,
        (DispatchOp<kDynamicSize, 1>(
            std::move(op_builder), inputs_vec,
            {.out_dtype = output_dtype_mlir,
             .out_dims = output_dims_vec,
             .op_param_cache_keys = std::move(updated_param_keys)})));

    return MakeTensor(std::move(result_buffer));
  });
}

}  // namespace torch_tpu
