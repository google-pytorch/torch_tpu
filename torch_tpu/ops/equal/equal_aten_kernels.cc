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

#include "torch_tpu/ops/equal/equal_aten_kernels.h"

#include <numeric>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {
absl::StatusOr<mlir::MlirOp> BuildEqualShlo(mlir::MlirOp self_input,
                                            mlir::MlirOp other_input,
                                            mlir::ElementType promoted_dtype) {
  auto& builder = self_input.getBuilder();
  mlir::Type bool_type = builder.getOpBuilder().getI1Type();
  mlir::MlirOp init_val = MakeScalarConstant(builder, true, bool_type);

  // Convert inputs to the promoted dtype if needed
  if (promoted_dtype != GetElementTypeOrDie(self_input)) {
    self_input =
        mlir::stablehlo::ConvertElementType(self_input, promoted_dtype);
  }
  if (promoted_dtype != GetElementTypeOrDie(other_input)) {
    other_input =
        mlir::stablehlo::ConvertElementType(other_input, promoted_dtype);
  }

  // Compare inputs element-wise
  auto compare_op = mlir::stablehlo::Compare(
      self_input, other_input, mlir::stablehlo::ComparisonDirection::EQ);

  // Reduce the comparison result to a scalar
  const auto rank = GetTensorTypeOrDie(self_input).getRank();
  Dimensions dims(rank);
  std::iota(dims.begin(), dims.end(), 0);

  return mlir::stablehlo::Reduce(
      builder, {compare_op}, {init_val},
      [&](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AndOp>(
            bool_type, rb.getRegion(), rb.getOpBuilder());
      },
      /*dimensions_to_reduce=*/dims)[0];
}
}  // namespace

bool AtenEqual(const at::Tensor& self, const at::Tensor& other) {
  if (self.sizes() != other.sizes()) {
    return false;
  }

  bool result_scalar = false;

  TT_KERNEL(OpName::kEqual, param_keys, (self, other), {
    TT_ASSIGN_OR_THROW(
        const auto promoted_dtype,
        ConvertTo<mlir::ElementType>(at::result_type(self, other)));

    auto op_builder = [promoted_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      return BuildEqualShlo(inputs[0], inputs[1], promoted_dtype);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<2>(std::move(op_builder), {self, other},
                      {.out_dtype = mlir::ElementType::PRED,
                       .out_dims = {},
                       .op_param_cache_keys = std::move(param_keys)}));

    auto result = MakeTensor(std::move(result_buf));
    result_scalar = result.item<bool>();
  });

  return result_scalar;
}

}  // namespace torch_tpu
